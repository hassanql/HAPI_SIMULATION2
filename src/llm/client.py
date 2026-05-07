"""Unified LLM client (spec §4.5, §11).

Wraps three execution paths behind one interface:

  - `mock`    — `MockBackend`, deterministic canned responses (Stage 0+).
  - `server`  — `openai.OpenAI` pointed at a local vLLM HTTP endpoint.
  - `library` — `vllm.LLM` instantiated in-process (Stage 6 onwards).

Every call is disk-cached. Cache key is the SHA-256 of a JSON document carrying
(model_revision, prompt, sampling_params, seed) per spec §11. A repeat call with
identical inputs is a cache hit and returns instantly.

Cache location is per-stage: `results/<stage>/llm_cache/` (OPEN_QUESTIONS.md #13).
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Callable

from pydantic import BaseModel, Field

from src.llm.mock_backend import MockBackend


# ---------------------------------------------------------------------------
# Sampling and request/response schemas
# ---------------------------------------------------------------------------


class SamplingParams(BaseModel):
    temperature: float = 0.0
    top_p: float = 1.0
    seed: int = 0
    max_tokens: int = 512

    model_config = {"frozen": True}


class LLMRequest(BaseModel):
    role: str = Field(..., description="'agent' / 'patient' / 'judge' / 'covert_attacker'.")
    prompt: str = Field(..., description="User-side prompt (the per-turn task).")
    system_prompt: str | None = Field(
        default=None,
        description=(
            "Optional system-side prompt (persona / static context). When set, "
            "library and server backends emit messages=[{role:'system', ...}, "
            "{role:'user', ...}]; the mock backend concatenates system+user "
            "for substring-based family classification."
        ),
    )
    sampling: SamplingParams = SamplingParams()
    schema_name: str | None = Field(
        default=None,
        description="Hint to the mock backend about which prompt family this is.",
    )


class LLMResponse(BaseModel):
    text: str
    cache_hit: bool
    latency_seconds: float
    model_revision: str
    role: str


# ---------------------------------------------------------------------------
# Disk cache
# ---------------------------------------------------------------------------


def _cache_key(
    model_revision: str,
    role: str,
    prompt: str,
    sampling: SamplingParams,
    system_prompt: str | None = None,
) -> str:
    payload = json.dumps(
        {
            "model_revision": model_revision,
            "role": role,
            "system_prompt": system_prompt or "",
            "prompt": prompt,
            "sampling": sampling.model_dump(),
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class LLMClient:
    """Façade over the three execution paths.

    Stage 0 instantiates with `backend="mock"` and a `MockBackend` instance.
    Stage 3+ will pass `backend="server"` and a `base_url` for the vLLM endpoint.
    """

    def __init__(
        self,
        backend: str,
        cache_dir: Path | None,
        *,
        mock_backend: MockBackend | None = None,
        library_backend: "Any | None" = None,
        base_urls: dict[str, str] | None = None,
        model_revisions: dict[str, str] | None = None,
        on_call: Callable[[LLMResponse], None] | None = None,
    ) -> None:
        if backend not in {"mock", "server", "library"}:
            raise ValueError(f"Unknown backend: {backend!r}")
        if backend == "mock" and mock_backend is None:
            raise ValueError("backend='mock' requires a MockBackend instance.")
        if backend == "library" and library_backend is None:
            raise ValueError("backend='library' requires a LibraryBackend instance.")
        self.backend = backend
        self.mock_backend = mock_backend
        self.library_backend = library_backend
        self.base_urls = base_urls or {}
        self.model_revisions = model_revisions or {}
        self.on_call = on_call

        # Lazy diskcache import: avoids paying the import cost in tests that
        # don't actually call the LLM client (some tests use MockBackend directly).
        self._cache = None
        self._cache_dir = cache_dir
        if cache_dir is not None:
            cache_dir.mkdir(parents=True, exist_ok=True)
            from diskcache import Cache  # type: ignore[import-untyped]

            self._cache = Cache(str(cache_dir))

    # -- Public API ------------------------------------------------------

    def generate(self, request: LLMRequest) -> LLMResponse:
        revision = self.model_revisions.get(request.role, f"{self.backend}-unknown")
        key = _cache_key(
            revision,
            request.role,
            request.prompt,
            request.sampling,
            system_prompt=request.system_prompt,
        )

        # Cache lookup.
        if self._cache is not None:
            cached = self._cache.get(key)
            if cached is not None:
                resp = LLMResponse(
                    text=cached["text"],
                    cache_hit=True,
                    latency_seconds=0.0,
                    model_revision=revision,
                    role=request.role,
                )
                if self.on_call:
                    self.on_call(resp)
                return resp

        # Miss → call backend.
        t0 = time.perf_counter()
        text = self._call_backend(request, revision)
        elapsed = time.perf_counter() - t0
        resp = LLMResponse(
            text=text,
            cache_hit=False,
            latency_seconds=elapsed,
            model_revision=revision,
            role=request.role,
        )

        if self._cache is not None:
            self._cache.set(key, {"text": text})

        if self.on_call:
            self.on_call(resp)
        return resp

    # -- Public API: batch ---------------------------------------------

    def generate_batch(self, requests: list[LLMRequest]) -> list[LLMResponse]:
        """Issue N requests, returning N responses in input order.

        Cache hits are filled immediately (no backend round-trip). Cache
        misses are forwarded to the backend's batch path:
          - LibraryBackend: vLLM continuous batching across the misses
          - MockBackend: sequential loop (mock is fast enough)
          - Server mode: sequential calls (each goes to its own endpoint)

        Each response is cached individually after the batch returns. This
        preserves cache semantics: a re-run with identical inputs short-
        circuits without any backend call.
        """
        if not requests:
            return []
        revisions = [
            self.model_revisions.get(r.role, f"{self.backend}-unknown")
            for r in requests
        ]
        keys = [
            _cache_key(
                rev,
                req.role,
                req.prompt,
                req.sampling,
                system_prompt=req.system_prompt,
            )
            for rev, req in zip(revisions, requests)
        ]

        responses: list[LLMResponse | None] = [None] * len(requests)
        miss_indices: list[int] = []

        # Pass 1: cache lookups. Hits get filled now.
        for i, (req, rev, key) in enumerate(zip(requests, revisions, keys)):
            if self._cache is not None:
                cached = self._cache.get(key)
                if cached is not None:
                    resp = LLMResponse(
                        text=cached["text"],
                        cache_hit=True,
                        latency_seconds=0.0,
                        model_revision=rev,
                        role=req.role,
                    )
                    responses[i] = resp
                    if self.on_call:
                        self.on_call(resp)
                    continue
            miss_indices.append(i)

        if not miss_indices:
            return responses  # type: ignore[return-value]

        # Pass 2: batch the misses through the backend.
        miss_requests = [requests[i] for i in miss_indices]
        miss_revisions = [revisions[i] for i in miss_indices]

        t0 = time.perf_counter()
        miss_texts = self._call_backend_batch(miss_requests, miss_revisions)
        elapsed = time.perf_counter() - t0
        per_call_latency = elapsed / max(1, len(miss_requests))

        # Pass 3: build responses, cache writes, callback.
        for j, idx in enumerate(miss_indices):
            req = requests[idx]
            rev = revisions[idx]
            text = miss_texts[j] if j < len(miss_texts) else ""
            resp = LLMResponse(
                text=text,
                cache_hit=False,
                latency_seconds=per_call_latency,
                model_revision=rev,
                role=req.role,
            )
            responses[idx] = resp
            if self._cache is not None:
                self._cache.set(keys[idx], {"text": text})
            if self.on_call:
                self.on_call(resp)

        return responses  # type: ignore[return-value]

    def _call_backend_batch(
        self, requests: list[LLMRequest], revisions: list[str]
    ) -> list[str]:
        if self.backend == "mock":
            assert self.mock_backend is not None
            items = [
                dict(
                    role=r.role,
                    prompt=r.prompt,
                    system_prompt=r.system_prompt,
                    schema_name=r.schema_name,
                    sampling=r.sampling.model_dump(),
                )
                for r in requests
            ]
            return self.mock_backend.respond_batch(items)
        if self.backend == "library":
            if self.library_backend is None:
                raise RuntimeError(
                    "LLMClient(backend='library') requires a LibraryBackend instance."
                )
            # All requests in a single vLLM batch. Sampling is taken from the
            # first request — within a batch we expect identical sampling
            # (the EIG predict + likelihoods calls all use the same).
            sampling = requests[0].sampling
            return self.library_backend.generate_batch(
                prompts=[r.prompt for r in requests],
                system_prompts=[r.system_prompt for r in requests],
                max_tokens=sampling.max_tokens,
                sampling=sampling,
            )
        if self.backend == "server":
            # OpenAI-compatible server mode: vLLM's HTTP API serves one
            # request at a time per connection; concurrent requests would
            # need an async client. For now, fall back to sequential — Stage
            # 3+ can revisit if it becomes a bottleneck.
            return [
                self._call_openai_server(r, rev)
                for r, rev in zip(requests, revisions)
            ]
        raise RuntimeError(f"Unhandled backend: {self.backend}")

    def close(self) -> None:
        if self._cache is not None:
            self._cache.close()
            self._cache = None

    # Context-manager sugar.
    def __enter__(self) -> "LLMClient":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    # -- Backends --------------------------------------------------------

    def _call_backend(self, request: LLMRequest, revision: str) -> str:
        if self.backend == "mock":
            assert self.mock_backend is not None
            return self.mock_backend.respond(
                role=request.role,
                prompt=request.prompt,
                system_prompt=request.system_prompt,
                schema_name=request.schema_name,
                sampling=request.sampling.model_dump(),
            )

        if self.backend == "server":
            return self._call_openai_server(request, revision)

        if self.backend == "library":
            return self._call_library(request, revision)

        raise RuntimeError(f"Unhandled backend: {self.backend}")

    def _call_openai_server(self, request: LLMRequest, revision: str) -> str:
        # Lazy import — avoids needing openai in tests that only use the mock.
        from openai import OpenAI  # type: ignore[import-untyped]

        base_url = self.base_urls.get(request.role)
        if base_url is None:
            raise RuntimeError(
                f"No base_url configured for role {request.role!r}. "
                f"Known: {sorted(self.base_urls)}"
            )
        client = OpenAI(base_url=base_url, api_key="EMPTY")
        messages: list[dict[str, str]] = []
        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})
        messages.append({"role": "user", "content": request.prompt})
        result: Any = client.chat.completions.create(
            model=revision,
            messages=messages,
            temperature=request.sampling.temperature,
            top_p=request.sampling.top_p,
            seed=request.sampling.seed,
            max_tokens=request.sampling.max_tokens,
        )
        return result.choices[0].message.content or ""

    def _call_library(self, request: LLMRequest, revision: str) -> str:
        if self.library_backend is None:
            raise RuntimeError(
                "LLMClient with backend='library' requires a LibraryBackend instance."
            )
        return self.library_backend.generate(
            request.prompt,
            system_prompt=request.system_prompt,
            max_tokens=request.sampling.max_tokens,
            sampling=request.sampling,
        )
