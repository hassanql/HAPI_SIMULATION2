"""Gemini API backend for LLMClient.

Mirrors the LibraryBackend interface (generate / generate_batch) but routes
to Google's `google.genai` SDK instead of an in-process vLLM. Designed to
support per-role model identity (different model for agent vs patient vs
judge) and async-parallel batches with a concurrency cap to stay under the
API's rate limits.

Rate-limit handling:
  - Asyncio.Semaphore caps concurrent in-flight requests at `max_concurrent`.
  - 429 / RESOURCE_EXHAUSTED responses are retried with exponential backoff.
  - The client surfaces the model_revision (model identifier) so the
    LLMClient cache key isolates results per Gemini model version.

This backend is selected via `LLMClient(backend="api", api_backend=...)`;
existing mock / server / library backends are unaffected.
"""
from __future__ import annotations

import asyncio
import logging
import os
import random
import threading
from dataclasses import dataclass, field
from typing import Any

import httpx

logger = logging.getLogger(__name__)

# Default rate-limit and retry settings. Overridable via APIBackendConfig.
_DEFAULT_MAX_CONCURRENT = 16
_DEFAULT_RETRY_ATTEMPTS = 5
_DEFAULT_RETRY_BASE_SECONDS = 1.0


@dataclass
class APIBackendConfig:
    """Configuration for the API backend.

    `models` maps a role name (e.g. "agent", "patient", "judge",
    "covert_attacker") to a Gemini model identifier
    (e.g. "gemini-3-flash-preview"). The cache key uses these identifiers,
    so swapping the agent's model invalidates only agent-side cache entries.
    """
    models: dict[str, str]
    api_key: str | None = None  # falls back to GOOGLE_API_KEY env
    max_concurrent: int = _DEFAULT_MAX_CONCURRENT
    retry_attempts: int = _DEFAULT_RETRY_ATTEMPTS
    retry_base_seconds: float = _DEFAULT_RETRY_BASE_SECONDS
    # Per-call timeout. Gemini's SDK has no enforced default and we observed
    # individual calls hanging indefinitely on hard reasoning cases (1+
    # trajectory blocking the whole sequential pipeline for 18+ minutes
    # before we noticed). 90s is well above the 99th-percentile call time
    # we've seen on Flash with 8192 max_tokens; anything longer is treated
    # as a stuck call and retried.
    request_timeout_seconds: float = 90.0
    enable_logprobs: bool = False
    logprobs_top_k: int = 5
    safety_settings: list[dict[str, str]] | None = field(default=None)


class APIBackend:
    """Synchronous + async batched Gemini API client.

    Usage from LLMClient:

        backend = APIBackend(APIBackendConfig(models={
            "agent":   "gemini-3-flash-preview",
            "patient": "gemini-3-flash-preview",
            "judge":   "gemini-3.1-pro-preview",
        }))
        client = LLMClient(backend="api", api_backend=backend, ...)

    The backend never reads the cache — that's the LLMClient's job. The
    backend just routes a request to the API and returns the response text.
    """

    def __init__(self, config: APIBackendConfig) -> None:
        self.config = config
        self._api_key = config.api_key or os.environ.get("GOOGLE_API_KEY")
        if not self._api_key:
            raise RuntimeError(
                "APIBackend requires an API key. Pass APIBackendConfig(api_key=...) "
                "or set GOOGLE_API_KEY in the environment."
            )
        # Each worker thread gets its OWN genai.Client + asyncio loop.
        # The Client's httpx pool binds to the loop that first uses it; if
        # threads share a Client they race over the loop binding and
        # serialise on the GIL inside the SDK's request path. Per-thread
        # isolation keeps trajectory parallelism (Option B) clean.
        self._tls = threading.local()
        # Backward-compat: tests and older callers reference `self._client`
        # directly. Construct the main-thread client eagerly for them.
        self._client = self._make_client()
        # Async semaphore for in-batch concurrency cap. Lazy-initialised on
        # first use within an event loop (created per-thread via _get_loop).
        self._semaphore: asyncio.Semaphore | None = None

    def _make_client(self):
        """Construct a genai.Client with the configured timeout."""
        from google import genai  # local import keeps tests hermetic
        try:
            return genai.Client(
                api_key=self._api_key,
                http_options={
                    "timeout": int(self.config.request_timeout_seconds * 1000),
                },
            )
        except TypeError:
            # Older SDK versions or test fakes that don't accept http_options.
            return genai.Client(api_key=self._api_key)

    def _thread_client(self):
        """Return the calling thread's genai.Client, creating it on first
        access. Each thread gets its own Client + event loop; this lets
        the trajectory-level ThreadPoolExecutor in run_pilot_stage drive
        N concurrent trajectories without async/sync interleaving across
        threads corrupting the SDK's httpx pool binding."""
        client = getattr(self._tls, "client", None)
        if client is None:
            client = self._make_client()
            self._tls.client = client
        return client

    def _get_loop(self) -> asyncio.AbstractEventLoop:
        """Return the calling thread's event loop, creating it lazily.

        Per-thread isolation: each worker thread holds its own loop in
        thread-local storage. Multiple threads can therefore call
        `generate_batch` (which dispatches an asyncio.gather over the
        thread's loop) concurrently without contending on a shared loop.
        """
        loop = getattr(self._tls, "loop", None)
        if loop is None or loop.is_closed():
            loop = asyncio.new_event_loop()
            self._tls.loop = loop
        return loop

    def close(self) -> None:
        """Tear down the calling thread's event loop. Safe to call multiple
        times. Other threads' loops survive."""
        loop = getattr(self._tls, "loop", None)
        if loop is not None and not loop.is_closed():
            try:
                loop.close()
            except Exception:
                pass
        self._tls.loop = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Public API: model lookup
    # ------------------------------------------------------------------

    def model_for(self, role: str) -> str:
        """Return the Gemini model identifier configured for this role."""
        if role not in self.config.models:
            raise KeyError(
                f"No Gemini model configured for role {role!r}. "
                f"Known: {sorted(self.config.models)}"
            )
        return self.config.models[role]

    def model_revision(self, role: str) -> str:
        """Stable revision identifier for cache keys. Today this is just the
        model id (Google does not expose per-version SHAs); when they
        promote `-preview` to dated `-001` etc. the model id itself encodes
        the version, which is sufficient for cache invalidation."""
        return self.model_for(role)

    # ------------------------------------------------------------------
    # Public API: single sync call
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt: str,
        *,
        role: str,
        system_prompt: str | None = None,
        max_tokens: int = 512,
        temperature: float = 0.0,
        top_p: float = 1.0,
        seed: int = 0,
    ) -> str:
        """Synchronous single-call entry point. Used by LLMClient.generate().

        Uses the sync genai client (`client.models.generate_content`) rather
        than wrapping the async path in `asyncio.run`. The async path holds
        an httpx.AsyncClient whose connection pool is bound to whichever
        event loop first touches it; under repeated `asyncio.run` calls that
        loop is recreated each time and the connection-pool callbacks fire
        on the closed loop, raising RuntimeError. The sync client has no
        such loop affinity.
        """
        # Tests inject a fake `aio.models.generate_content` but no sync
        # `models.generate_content`. To stay test-compatible, fall back to
        # the async path when the sync surface isn't available.
        sync_models = getattr(self._client, "models", None)
        if sync_models is None or not hasattr(sync_models, "generate_content"):
            return self._get_loop().run_until_complete(
                self._generate_async(
                    prompt,
                    role=role,
                    system_prompt=system_prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    seed=seed,
                )
            )
        # Tests register `_FakeSyncModels` with only `.list()` — detect this
        # and route to async fake instead.
        if not callable(getattr(sync_models, "generate_content", None)):
            return self._get_loop().run_until_complete(
                self._generate_async(
                    prompt,
                    role=role,
                    system_prompt=system_prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    seed=seed,
                )
            )
        return self._do_generate_sync(
            prompt, role, system_prompt, max_tokens, temperature, top_p, seed
        )

    def _do_generate_sync(
        self,
        prompt: str,
        role: str,
        system_prompt: str | None,
        max_tokens: int,
        temperature: float,
        top_p: float,
        seed: int,
    ) -> str:
        from google.genai import errors as genai_errors

        model_id = self.model_for(role)
        gen_config = self._build_gen_config(
            system_prompt, max_tokens, temperature, top_p, seed
        )
        client = self._thread_client()
        last_err: Exception | None = None
        for attempt in range(self.config.retry_attempts):
            try:
                resp = client.models.generate_content(
                    model=model_id, contents=prompt, config=gen_config
                )
                return resp.text or ""
            except genai_errors.ClientError as e:
                if getattr(e, "code", None) == 429:
                    last_err = e
                    delay = self._backoff_delay(attempt)
                    logger.warning(
                        "Gemini 429 (sync) attempt %d for role=%s; sleeping %.1fs",
                        attempt + 1, role, delay,
                    )
                    import time as _time
                    _time.sleep(delay)
                    continue
                raise
            except genai_errors.ServerError as e:
                last_err = e
                delay = self._backoff_delay(attempt)
                logger.warning(
                    "Gemini 5xx (sync) attempt %d for role=%s; sleeping %.1fs",
                    attempt + 1, role, delay,
                )
                import time as _time
                _time.sleep(delay)
                continue
            except (httpx.RemoteProtocolError, httpx.ReadError, httpx.ConnectError, httpx.WriteError, httpx.PoolTimeout) as e:
                # See the matching except in `_do_generate` for context.
                last_err = e
                delay = self._backoff_delay(attempt)
                logger.warning(
                    "Gemini network error (sync, %s) attempt %d role=%s; sleep %.1fs",
                    type(e).__name__, attempt + 1, role, delay,
                )
                import time as _time
                _time.sleep(delay)
                continue
        raise RuntimeError(
            f"Gemini API failed after {self.config.retry_attempts} attempts "
            f"for role={role!r}: {last_err!r}"
        )

    def _build_gen_config(
        self,
        system_prompt: str | None,
        max_tokens: int,
        temperature: float,
        top_p: float,
        seed: int,
    ) -> dict[str, Any]:
        # Gemini 3 Flash and Pro are reasoning models with built-in
        # chain-of-thought. `max_output_tokens` is the TOTAL budget — thinking
        # tokens come out of it before the user-visible response. With our
        # original 512 cap, thinking consumed ~488 tokens leaving ~20 for the
        # JSON output, which truncated every action proposal to ~40 chars and
        # made every trajectory degenerate (4-step runs that re-asked the same
        # opening question 3 times). 4096 is a working balance: Flash's
        # average call uses ~1500-3000 thinking tokens and ~200-1000 visible,
        # comfortably under 4096. The rare thinking-spill we see is recovered
        # by `_extract_first_json` (which scans past leading prose for the
        # JSON value) so we don't pay for a higher ceiling. At 8192 the
        # billed-output token volume roughly doubled with negligible quality
        # gain (~1pp τ-recovery), so 4096 is the better cost/quality point.
        effective_max = max(int(max_tokens), 4096)
        gen_config: dict[str, Any] = {
            "temperature": temperature,
            "top_p": top_p,
            "max_output_tokens": effective_max,
            "seed": seed,
        }
        if system_prompt:
            gen_config["system_instruction"] = system_prompt
        if self.config.safety_settings:
            gen_config["safety_settings"] = self.config.safety_settings
        if self.config.enable_logprobs:
            gen_config["response_logprobs"] = True
            gen_config["logprobs"] = self.config.logprobs_top_k
        return gen_config

    # ------------------------------------------------------------------
    # Public API: batched concurrent calls
    # ------------------------------------------------------------------

    def generate_batch(
        self,
        prompts: list[str],
        *,
        roles: list[str] | str,
        system_prompts: list[str | None] | None = None,
        max_tokens: int = 512,
        temperature: float = 0.0,
        top_p: float = 1.0,
        seed: int = 0,
    ) -> list[str]:
        """Concurrent batch of N prompts -> N responses in input order.

        Each request runs in its own asyncio task; the semaphore caps how
        many are in-flight simultaneously so we stay under the API's RPM
        limit. Cache hits should be served by the LLMClient before this
        is called; this method is for cache misses only.

        `roles` can be a list (per-prompt model selection) or a single
        string (all prompts use the same role's model).
        """
        n = len(prompts)
        if isinstance(roles, str):
            roles_list = [roles] * n
        else:
            if len(roles) != n:
                raise ValueError(
                    f"roles list length ({len(roles)}) != prompts length ({n})"
                )
            roles_list = list(roles)
        if system_prompts is None:
            system_prompts_list: list[str | None] = [None] * n
        else:
            if len(system_prompts) != n:
                raise ValueError(
                    f"system_prompts length ({len(system_prompts)}) != prompts length ({n})"
                )
            system_prompts_list = list(system_prompts)

        return self._get_loop().run_until_complete(
            self._generate_batch_async(
                prompts,
                roles_list,
                system_prompts_list,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                seed=seed,
            )
        )

    # ------------------------------------------------------------------
    # Internal: async machinery
    # ------------------------------------------------------------------

    async def _generate_batch_async(
        self,
        prompts: list[str],
        roles: list[str],
        system_prompts: list[str | None],
        *,
        max_tokens: int,
        temperature: float,
        top_p: float,
        seed: int,
    ) -> list[str]:
        # Lazy semaphore: created on the asyncio event loop running this batch.
        self._semaphore = asyncio.Semaphore(self.config.max_concurrent)
        tasks = [
            self._generate_async(
                p,
                role=r,
                system_prompt=sp,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                seed=seed,
            )
            for p, r, sp in zip(prompts, roles, system_prompts)
        ]
        return await asyncio.gather(*tasks)

    async def _generate_async(
        self,
        prompt: str,
        *,
        role: str,
        system_prompt: str | None,
        max_tokens: int,
        temperature: float,
        top_p: float,
        seed: int,
    ) -> str:
        """Single async call with retry+backoff. The semaphore is acquired
        only when the batch path drives this; for solo `generate()` calls
        we don't gate."""
        sem = self._semaphore
        if sem is None:
            return await self._do_generate(
                prompt, role, system_prompt, max_tokens, temperature, top_p, seed
            )
        async with sem:
            return await self._do_generate(
                prompt, role, system_prompt, max_tokens, temperature, top_p, seed
            )

    async def _do_generate(
        self,
        prompt: str,
        role: str,
        system_prompt: str | None,
        max_tokens: int,
        temperature: float,
        top_p: float,
        seed: int,
    ) -> str:
        from google import genai  # noqa: F401  -- imported for type check
        from google.genai import errors as genai_errors

        model_id = self.model_for(role)

        # Build generation config via the shared helper so the thinking-aware
        # max_output_tokens floor is applied to both sync and async paths.
        gen_config = self._build_gen_config(
            system_prompt, max_tokens, temperature, top_p, seed
        )

        client = self._thread_client()
        last_err: Exception | None = None
        for attempt in range(self.config.retry_attempts):
            try:
                resp = await asyncio.wait_for(
                    client.aio.models.generate_content(
                        model=model_id,
                        contents=prompt,
                        config=gen_config,
                    ),
                    timeout=self.config.request_timeout_seconds,
                )
                # Defensive: Gemini may return None text when the response
                # is filtered by safety. Surface as empty string for the
                # caller to handle (logging happens at LLMClient layer).
                text = resp.text or ""
                return text
            except asyncio.TimeoutError:
                # The genai SDK has no enforced default timeout; we observed
                # individual calls hanging indefinitely on hard reasoning
                # cases. Treat a timeout as a transient failure and retry.
                last_err = TimeoutError(
                    f"Gemini call exceeded {self.config.request_timeout_seconds}s"
                )
                delay = self._backoff_delay(attempt)
                logger.warning(
                    "Gemini request_timeout on attempt %d for role=%s; "
                    "sleeping %.1fs before retry",
                    attempt + 1, role, delay,
                )
                await asyncio.sleep(delay)
                continue
            except genai_errors.ClientError as e:
                # 4xx errors are usually permanent (bad prompt, too long)
                # except 429 (rate limit). Retry only on 429.
                if getattr(e, "code", None) == 429:
                    last_err = e
                    delay = self._backoff_delay(attempt)
                    logger.warning(
                        "Gemini 429 rate-limit on attempt %d for role=%s; "
                        "sleeping %.1fs",
                        attempt + 1, role, delay,
                    )
                    await asyncio.sleep(delay)
                    continue
                raise
            except genai_errors.ServerError as e:
                # 5xx — always retry with backoff.
                last_err = e
                delay = self._backoff_delay(attempt)
                logger.warning(
                    "Gemini 5xx on attempt %d for role=%s; sleeping %.1fs",
                    attempt + 1, role, delay,
                )
                await asyncio.sleep(delay)
                continue
            except (httpx.RemoteProtocolError, httpx.ReadError, httpx.ConnectError, httpx.WriteError, httpx.PoolTimeout) as e:
                # Network-level transient failures — the genai SDK does not
                # wrap these as ServerError, so without explicit handling they
                # bubble all the way up and crash the trajectory. Saw this on
                # 2026-05-08 at trajectory ~541/600 of the headline run:
                # `Server disconnected without sending a response` killed the
                # whole pilot. Retry is the right move; these errors mean the
                # upstream dropped us, not that the request was malformed.
                last_err = e
                delay = self._backoff_delay(attempt)
                logger.warning(
                    "Gemini network error (%s) on attempt %d for role=%s; "
                    "sleeping %.1fs",
                    type(e).__name__, attempt + 1, role, delay,
                )
                await asyncio.sleep(delay)
                continue

        # Out of retries.
        raise RuntimeError(
            f"Gemini API failed after {self.config.retry_attempts} attempts "
            f"for role={role!r}: {last_err!r}"
        )

    def _backoff_delay(self, attempt: int) -> float:
        """Exponential backoff with full jitter."""
        base = self.config.retry_base_seconds * (2 ** attempt)
        return random.uniform(0.0, base)


def build_api_backend_from_config(
    model_cfg: dict[str, Any],
    profile: str | None = None,
) -> APIBackend:
    """Construct an APIBackend from a parsed `models_api.yaml`-style dict.

    `model_cfg["models"]` is profile-keyed:
        models:
          flash_dev:    {agent: ..., patient: ...}
          pro_headline: {agent: ..., patient: ...}

    `profile` selects one. If None, uses `model_cfg["default_profile"]`.

    Other top-level keys (`max_concurrent`, `retry_attempts`, etc.) become
    APIBackendConfig fields. The API key is read from the env var named in
    `api_key_env` (default GOOGLE_API_KEY) — this path matches the
    `set -a; source .env; set +a` convention used for the Gemini setup.
    """
    profile = profile or model_cfg.get("default_profile")
    if profile is None:
        raise RuntimeError(
            "API model config has no `default_profile` and none was supplied."
        )
    models_block = model_cfg.get("models") or {}
    if profile not in models_block:
        raise RuntimeError(
            f"Profile {profile!r} not in API config; "
            f"known profiles: {sorted(models_block)}"
        )
    role_models: dict[str, str] = dict(models_block[profile])
    if not role_models:
        raise RuntimeError(f"API profile {profile!r} has no role->model mappings.")

    api_key_env = model_cfg.get("api_key_env", "GOOGLE_API_KEY")
    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise RuntimeError(
            f"API backend requires {api_key_env} env var. "
            f"Run `set -a; source .env; set +a` first, or export it."
        )

    cfg = APIBackendConfig(
        models=role_models,
        api_key=api_key,
        max_concurrent=int(model_cfg.get("max_concurrent", _DEFAULT_MAX_CONCURRENT)),
        retry_attempts=int(model_cfg.get("retry_attempts", _DEFAULT_RETRY_ATTEMPTS)),
        retry_base_seconds=float(
            model_cfg.get("retry_base_seconds", _DEFAULT_RETRY_BASE_SECONDS)
        ),
        enable_logprobs=bool(model_cfg.get("enable_logprobs", False)),
        logprobs_top_k=int(model_cfg.get("logprobs_top_k", 5)),
    )
    return APIBackend(cfg)
