"""In-process vLLM (`vllm.LLM`) wrapper for single-model phases.

Used in Stage 2 (dev-mode agent + patient share one Qwen3-4B-Instruct-2507
instance) and in any later stage that needs a single model in-process. The
constraint (spec §4): only one `vllm.LLM` may be alive in a Python process
at a time.

The wrapper uses vLLM's `LLM.chat()` API which auto-applies the model's
chat template via the tokenizer. System and user prompts are passed as
proper roles in the message list — the dev-mode agent and patient differ
ONLY by system prompt, both routed through this single backend.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from typing import Any

from src.llm.client import SamplingParams


@dataclass
class LibraryBackendConfig:
    model_id: str
    gpu_memory_utilization: float = 0.80
    max_model_len: int = 4096
    max_num_seqs: int = 4
    enforce_eager: bool = True
    enable_prefix_caching: bool = True
    seed: int = 0
    dtype: str = "bfloat16"
    trust_remote_code: bool = True
    download_dir: str | None = None     # override HF_HOME if set


class LibraryBackend:
    """Thin wrapper around `vllm.LLM`. Lazy-loaded — `load()` is the heavy
    operation (model weights → GPU). Idempotent: a second `load()` is a no-op."""

    def __init__(self, config: LibraryBackendConfig) -> None:
        self.config = config
        self._llm: Any | None = None
        self._sampling_default: Any | None = None
        self._revision: str | None = None

    # ---- vLLM probe ----------------------------------------------------

    @staticmethod
    def vllm_available() -> bool:
        """Cheap import-check used by preflight; does not actually load vLLM."""
        return importlib.util.find_spec("vllm") is not None

    # ---- lifecycle -----------------------------------------------------

    def load(self) -> None:
        if self._llm is not None:
            return
        # Lazy import so tests on macOS without vLLM installed still pass.
        from vllm import LLM, SamplingParams as VLLMSampling  # type: ignore[import-untyped]

        kwargs: dict[str, Any] = {
            "model": self.config.model_id,
            "gpu_memory_utilization": self.config.gpu_memory_utilization,
            "max_model_len": self.config.max_model_len,
            "max_num_seqs": self.config.max_num_seqs,
            "enforce_eager": self.config.enforce_eager,
            # Automatic block-level KV-cache reuse for repeated prefixes.
            # Within a turn, all K EIG-predict calls share the system prompt
            # + chief_complaint + vignette + trajectory_summary prefix; vLLM
            # caches the prefill of that prefix and only computes the
            # candidate-specific suffix per call. Compatible with both eager
            # and CUDA-graphs modes.
            "enable_prefix_caching": self.config.enable_prefix_caching,
            "seed": self.config.seed,
            "dtype": self.config.dtype,
            "trust_remote_code": self.config.trust_remote_code,
        }
        if self.config.download_dir:
            kwargs["download_dir"] = self.config.download_dir

        self._llm = LLM(**kwargs)
        self._sampling_default = VLLMSampling(
            temperature=0.0, top_p=1.0, seed=0, max_tokens=512
        )
        # Capture the resolved checkpoint commit SHA, when available.
        try:
            from huggingface_hub import HfApi  # type: ignore[import-untyped]

            info = HfApi().model_info(self.config.model_id)
            self._revision = getattr(info, "sha", None) or "unknown"
        except Exception:
            self._revision = "unknown"

    def revision(self) -> str:
        return self._revision or "unloaded"

    # ---- generation ----------------------------------------------------

    def generate(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        max_tokens: int = 512,
        sampling: SamplingParams | None = None,
    ) -> str:
        if self._llm is None:
            self.load()
        from vllm import SamplingParams as VLLMSampling  # type: ignore[import-untyped]

        sp = VLLMSampling(
            temperature=sampling.temperature if sampling else 0.0,
            top_p=sampling.top_p if sampling else 1.0,
            seed=sampling.seed if sampling else 0,
            max_tokens=max_tokens,
        )

        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # vLLM's .chat() auto-applies the model's chat template via the
        # tokenizer. For Qwen3-4B-Instruct-2507, this emits the correct
        # <|im_start|>...<|im_end|> wrapping without us hand-rolling it.
        outputs = self._llm.chat(messages=messages, sampling_params=sp, use_tqdm=False)
        if not outputs:
            return ""
        completion = outputs[0]
        if not completion.outputs:
            return ""
        return completion.outputs[0].text or ""

    # ---- batched generation -------------------------------------------

    def generate_batch(
        self,
        prompts: list[str],
        *,
        system_prompts: list[str | None] | None = None,
        max_tokens: int = 512,
        sampling: SamplingParams | None = None,
    ) -> list[str]:
        """Submit N prompts as a single vLLM batch — outputs preserved in
        input order. With `max_num_seqs >= N`, vLLM runs all N sequences
        through continuous batching, overlapping their decode phases.

        Empty input → empty output. System prompts may be `None` per request;
        only those with a non-None system prompt get a system message in their
        chat thread."""
        if not prompts:
            return []
        if self._llm is None:
            self.load()
        from vllm import SamplingParams as VLLMSampling  # type: ignore[import-untyped]

        if system_prompts is None:
            system_prompts = [None] * len(prompts)
        if len(system_prompts) != len(prompts):
            raise ValueError(
                f"system_prompts length ({len(system_prompts)}) must match "
                f"prompts length ({len(prompts)})."
            )

        sp = VLLMSampling(
            temperature=sampling.temperature if sampling else 0.0,
            top_p=sampling.top_p if sampling else 1.0,
            seed=sampling.seed if sampling else 0,
            max_tokens=max_tokens,
        )

        messages_list: list[list[dict[str, str]]] = []
        for prompt, sys_p in zip(prompts, system_prompts):
            msgs: list[dict[str, str]] = []
            if sys_p:
                msgs.append({"role": "system", "content": sys_p})
            msgs.append({"role": "user", "content": prompt})
            messages_list.append(msgs)

        # vLLM accepts a list-of-conversations and batches via continuous
        # batching. Outputs are returned in the same order as inputs.
        outputs = self._llm.chat(
            messages=messages_list, sampling_params=sp, use_tqdm=False
        )
        result: list[str] = []
        for o in outputs:
            if o.outputs:
                result.append(o.outputs[0].text or "")
            else:
                result.append("")
        return result

    def close(self) -> None:
        # vLLM has no explicit close — drop our reference so GC reclaims the
        # GPU memory at the next collection.
        self._llm = None
        self._sampling_default = None

    def __enter__(self) -> "LibraryBackend":
        self.load()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
