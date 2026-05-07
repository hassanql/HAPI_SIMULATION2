"""Tests for `src.llm.server_manager` — VLLMServer lifecycle helpers.

These tests don't actually launch vLLM; they exercise the preflight checks
and helpers that protect us from the orphan-server class of failures
documented in OPEN_QUESTIONS #45 / #46.
"""

from __future__ import annotations

import socket
from pathlib import Path

import pytest

from src.llm.server_manager import (
    VLLMServer,
    VLLMServerSpec,
    _is_port_in_use,
)


def _spec(port: int, tmp_path: Path) -> VLLMServerSpec:
    return VLLMServerSpec(
        role="patient",
        model_id="not-a-real-model",
        port=port,
        gpu_mem_util=0.4,
        max_model_len=1024,
        extra_args=[],
        log_path=tmp_path / "vllm_test.log",
        # Tight to keep tests snappy if the preflight ever silently passes.
        health_timeout_seconds=2.0,
        poll_interval_seconds=0.1,
    )


def test_is_port_in_use_returns_false_for_free_port() -> None:
    # Pick an ephemeral port the OS guarantees is free, close it, then
    # immediately probe — there's a tiny race window but socket reuse
    # makes false-positives essentially never happen here.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    # Socket closed → port should now read free.
    assert _is_port_in_use("127.0.0.1", port, timeout=0.3) is False


def test_is_port_in_use_returns_true_when_listening() -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        s.listen(1)
        port = s.getsockname()[1]
        assert _is_port_in_use("127.0.0.1", port, timeout=0.3) is True


def test_vllmserver_enter_refuses_when_port_already_in_use(tmp_path: Path) -> None:
    """Preflight in VLLMServer.__enter__ must fail BEFORE spawning vllm.

    Without this guard, an orphan vLLM from a previous crashed run can
    answer the new server's /health probe, and the orchestrator silently
    routes inference to the orphan until it dies. See OPEN_QUESTIONS #46.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as occupied:
        occupied.bind(("127.0.0.1", 0))
        occupied.listen(1)
        port = occupied.getsockname()[1]

        server = VLLMServer(_spec(port=port, tmp_path=tmp_path))
        with pytest.raises(RuntimeError, match=f"Port {port} is already in use"):
            server.__enter__()
        # And the server should NOT have spawned a subprocess.
        assert server.process is None
