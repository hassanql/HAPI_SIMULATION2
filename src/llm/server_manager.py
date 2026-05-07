"""vLLM server lifecycle (spec §4.4).

The orchestrator spawns vLLM HTTP servers as Python subprocesses, polls
`/health` until ready, runs the experiment loop, and tears them down on exit.
The user never runs `vllm serve` by hand.

Stage 0 ships the *interface* and a unit-testable subprocess wrapper. Live
launches against real vLLM are first exercised in Stage 3.
"""

from __future__ import annotations

import os
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _is_port_in_use(host: str, port: int, timeout: float = 1.0) -> bool:
    """True if something is listening on (host, port).

    Used as a preflight check before spawning a vLLM server, so we don't
    silently inherit an orphan process's port. See OPEN_QUESTIONS #46.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(timeout)
        try:
            return s.connect_ex((host, port)) == 0
        except OSError:
            return False


def _port_holder_hint(port: int) -> str:
    """Best-effort lookup of which PID is holding a port.

    Returns a hint string for inclusion in an error message. We try
    `lsof -ti :PORT` (commonly available) and fall back to a generic
    "use lsof or ss" instruction if it's not installed or returns
    nothing useful.
    """
    try:
        out = subprocess.run(
            ["lsof", "-ti", f":{port}"],
            capture_output=True, text=True, timeout=2.0,
        )
        if out.returncode == 0 and out.stdout.strip():
            pids = out.stdout.strip().split()
            return (
                f"PID(s) {' '.join(pids)} hold port {port}. "
                f"Run `kill {' '.join(pids)}` and verify with "
                f"`nvidia-smi --query-gpu=memory.used --format=csv` (should drop to baseline)."
            )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return (
        f"Run `lsof -i :{port}` (or `ss -lptn 'sport = :{port}'`) to find the holder, "
        f"then `kill <PID>`."
    )


def _resolve_vllm_binary() -> str:
    """Locate the ``vllm`` CLI for subprocess.Popen.

    We invoke run.py as ``.venv/bin/python run.py`` rather than activating the
    venv, which means the child subprocess inherits the unactivated PATH —
    ``.venv/bin/vllm`` is NOT discoverable via plain ``["vllm", ...]``.

    Resolution order:
        1. <python_dir>/vllm   — same venv as the running interpreter (the
           normal case for our deploy: pip install -e ".[prod]" installs
           the vllm console-script next to .venv/bin/python).
        2. shutil.which("vllm") — system PATH (covers `vllm` installed
           globally, or developers who happen to have activated the venv).

    If neither resolves, fall back to the bare string ``"vllm"`` and let
    Popen raise FileNotFoundError with the original "No such file or
    directory: 'vllm'" message — that's what surfaced this bug originally,
    and it's the right error if vllm genuinely isn't installed.
    """
    candidate = Path(sys.executable).parent / "vllm"
    if candidate.is_file() and os.access(candidate, os.X_OK):
        return str(candidate)
    found = shutil.which("vllm")
    if found:
        return found
    return "vllm"


@dataclass
class VLLMServerSpec:
    """All the information needed to launch one vLLM server."""

    role: str
    model_id: str
    port: int
    gpu_mem_util: float
    max_model_len: int
    extra_args: list[str]
    log_path: Path
    health_timeout_seconds: float = 300.0
    poll_interval_seconds: float = 2.0


class VLLMServer(AbstractContextManager["VLLMServer"]):
    """Context manager that owns a vllm subprocess, its port, and its health probe.

    Usage (Stage 3+):

        with VLLMServer(spec) as server:
            url = server.base_url()
            # ... use the OpenAI-compatible endpoint at `url` ...
        # subprocess is SIGTERMed on context exit, SIGKILLed if it ignores SIGTERM.
    """

    def __init__(self, spec: VLLMServerSpec) -> None:
        self.spec = spec
        self.process: subprocess.Popen[bytes] | None = None
        self._log_handle = None
        self._revision_sha: str | None = None

    def __enter__(self) -> "VLLMServer":
        # Preflight: refuse to spawn if the target port is already taken.
        # Without this, an orphan from a previous crashed run can answer the
        # /health probe and the orchestrator will silently route inference
        # to it for a while before the orphan dies. See OPEN_QUESTIONS #46.
        if _is_port_in_use("localhost", self.spec.port):
            raise RuntimeError(
                f"Port {self.spec.port} is already in use — refusing to "
                f"spawn '{self.spec.role}' vLLM server because /health "
                f"checks would be answered by the existing process, not "
                f"the one we're trying to start. {_port_holder_hint(self.spec.port)}"
            )
        self._spawn()
        try:
            self._wait_for_health()
        except BaseException:
            # Spawn happened but health check failed — terminate the orphan
            # subprocess so it doesn't keep the GPU pinned. Without this the
            # caller can't add the server to a pool (it raised before
            # add()'s self._servers update) and the pool's __exit__ won't
            # find anything to tear down. See OPEN_QUESTIONS #45.
            self._terminate()
            raise
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._terminate()

    # -- Public ----------------------------------------------------------

    def base_url(self) -> str:
        return f"http://localhost:{self.spec.port}/v1"

    def model_revision_sha(self) -> str:
        """Resolve the HF Hub commit SHA for the model.

        Stage 0 returns a placeholder; Stage 3 uses `huggingface_hub.HfApi().model_info()`.
        """
        if self._revision_sha is not None:
            return self._revision_sha
        try:
            from huggingface_hub import HfApi  # type: ignore[import-untyped]

            info = HfApi().model_info(self.spec.model_id)
            self._revision_sha = getattr(info, "sha", None) or "unknown"
        except Exception:
            self._revision_sha = "unknown"
        return self._revision_sha

    # -- Internals -------------------------------------------------------

    def _spawn(self) -> None:
        self.spec.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_handle = self.spec.log_path.open("ab")
        cmd = self._build_cmd()
        # Prepend the running interpreter's bin/ to PATH so vllm's own
        # subprocesses (e.g. its tokenizer download helper) can find any
        # other venv-installed CLIs they might shell out to.
        env = os.environ.copy()
        venv_bin = str(Path(sys.executable).parent)
        env["PATH"] = venv_bin + os.pathsep + env.get("PATH", "")
        self.process = subprocess.Popen(
            cmd,
            stdout=self._log_handle,
            stderr=subprocess.STDOUT,
            preexec_fn=os.setsid if os.name == "posix" else None,
            env=env,
        )

    def _build_cmd(self) -> list[str]:
        """Compose the vllm-serve command line."""
        return [
            _resolve_vllm_binary(),
            "serve",
            self.spec.model_id,
            "--port",
            str(self.spec.port),
            "--gpu-memory-utilization",
            str(self.spec.gpu_mem_util),
            "--max-model-len",
            str(self.spec.max_model_len),
            *self.spec.extra_args,
        ]

    def _wait_for_health(self) -> None:
        url = f"http://localhost:{self.spec.port}/health"
        deadline = time.monotonic() + self.spec.health_timeout_seconds
        while time.monotonic() < deadline:
            if self.process is not None and self.process.poll() is not None:
                rc = self.process.returncode
                raise RuntimeError(
                    f"vLLM server for {self.spec.role!r} exited before becoming healthy "
                    f"(rc={rc}). See log: {self.spec.log_path}"
                )
            try:
                with urllib.request.urlopen(url, timeout=2.0) as resp:
                    if 200 <= resp.status < 300:
                        return
            except (urllib.error.URLError, ConnectionError, TimeoutError):
                pass
            time.sleep(self.spec.poll_interval_seconds)
        raise TimeoutError(
            f"vLLM server for {self.spec.role!r} did not become healthy within "
            f"{self.spec.health_timeout_seconds:.0f}s. Log: {self.spec.log_path}"
        )

    def _terminate(self) -> None:
        if self.process is None:
            return
        if self.process.poll() is None:
            try:
                if os.name == "posix":
                    os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
                else:
                    self.process.terminate()
            except (ProcessLookupError, PermissionError):
                pass
            try:
                self.process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                try:
                    if os.name == "posix":
                        os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
                    else:
                        self.process.kill()
                except (ProcessLookupError, PermissionError):
                    pass
                self.process.wait(timeout=5)
        if self._log_handle is not None:
            self._log_handle.close()
            self._log_handle = None


class ServerPool(AbstractContextManager["ServerPool"]):
    """Multi-server lifecycle. Tears down in reverse order on exit.

    Enforces the spec §4.4 rule that the sum of `gpu_mem_util` across
    concurrent servers stays under `gpu_mem_budget` (default 0.90).
    """

    def __init__(self, gpu_mem_budget: float = 0.90) -> None:
        self.gpu_mem_budget = gpu_mem_budget
        self._servers: dict[str, VLLMServer] = {}
        self._entered = False

    def __enter__(self) -> "ServerPool":
        self._entered = True
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        # Reverse-order teardown.
        for role in reversed(list(self._servers)):
            try:
                self._servers[role].__exit__(exc_type, exc, tb)
            except Exception:  # pragma: no cover - best-effort cleanup
                pass
        self._servers.clear()

    def add(self, server: VLLMServer) -> None:
        if not self._entered:
            raise RuntimeError("ServerPool.add() called outside the context.")
        spec = server.spec
        used = sum(s.spec.gpu_mem_util for s in self._servers.values())
        if used + spec.gpu_mem_util > self.gpu_mem_budget + 1e-9:
            raise RuntimeError(
                f"Adding server {spec.role!r} (gpu_mem_util={spec.gpu_mem_util}) "
                f"would exceed gpu_mem_budget={self.gpu_mem_budget} "
                f"(already used: {used})."
            )
        if spec.role in self._servers:
            raise RuntimeError(f"Server with role {spec.role!r} already in pool.")
        # Enter the child context — spawns and health-checks.
        server.__enter__()
        self._servers[spec.role] = server

    def get_url(self, role: str) -> str:
        if role not in self._servers:
            raise KeyError(f"No server with role {role!r}. Known: {sorted(self._servers)}")
        return self._servers[role].base_url()

    def get(self, role: str) -> VLLMServer:
        return self._servers[role]

    def base_urls(self) -> dict[str, str]:
        return {role: s.base_url() for role, s in self._servers.items()}


def aggregate_gpu_mem(specs: list[VLLMServerSpec]) -> float:
    """Helper for preflight checks: total declared GPU memory utilisation."""
    return sum(s.gpu_mem_util for s in specs)
