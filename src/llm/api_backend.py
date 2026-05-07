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
from dataclasses import dataclass, field
from typing import Any

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
        from google import genai  # local import keeps tests hermetic

        self.config = config
        api_key = config.api_key or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError(
                "APIBackend requires an API key. Pass APIBackendConfig(api_key=...) "
                "or set GOOGLE_API_KEY in the environment."
            )
        self._client = genai.Client(api_key=api_key)
        # The async client is the same object — google.genai exposes
        # `client.aio.models.generate_content` for the async path.
        self._semaphore: asyncio.Semaphore | None = None  # lazy

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
        """Synchronous single-call entry point. Used by LLMClient.generate()."""
        return asyncio.run(
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

        return asyncio.run(
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

        # Build generation config. Gemini's API expects a different format
        # than OpenAI; system instruction is a top-level field.
        gen_config: dict[str, Any] = {
            "temperature": temperature,
            "top_p": top_p,
            "max_output_tokens": max_tokens,
            "seed": seed,
        }
        if system_prompt:
            gen_config["system_instruction"] = system_prompt
        if self.config.safety_settings:
            gen_config["safety_settings"] = self.config.safety_settings
        if self.config.enable_logprobs:
            gen_config["response_logprobs"] = True
            gen_config["logprobs"] = self.config.logprobs_top_k

        last_err: Exception | None = None
        for attempt in range(self.config.retry_attempts):
            try:
                resp = await self._client.aio.models.generate_content(
                    model=model_id,
                    contents=prompt,
                    config=gen_config,
                )
                # Defensive: Gemini may return None text when the response
                # is filtered by safety. Surface as empty string for the
                # caller to handle (logging happens at LLMClient layer).
                text = resp.text or ""
                return text
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

        # Out of retries.
        raise RuntimeError(
            f"Gemini API failed after {self.config.retry_attempts} attempts "
            f"for role={role!r}: {last_err!r}"
        )

    def _backoff_delay(self, attempt: int) -> float:
        """Exponential backoff with full jitter."""
        base = self.config.retry_base_seconds * (2 ** attempt)
        return random.uniform(0.0, base)
