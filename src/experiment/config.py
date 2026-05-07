"""Stage / experiment configuration models."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class SamplingConfig(BaseModel):
    temperature: float = 0.0
    top_p: float = 1.0
    seed: int = 0
    max_tokens: int = 512


class CacheConfig(BaseModel):
    enabled: bool = True
    subpath: str = "llm_cache"


class StageConfig(BaseModel):
    """Resolved per-stage config. Permissive `extra` fields preserved for
    stage-specific keys not modelled here."""

    stage: str
    backend: str = "mock"
    output_subdir: str
    runtime_budget_hours: float = 24.0
    sampling: SamplingConfig = SamplingConfig()
    cache: CacheConfig = CacheConfig()
    extra: dict[str, Any] = Field(default_factory=dict)
    config_path: Path | None = None
    base_path: Path | None = None

    model_config = {"arbitrary_types_allowed": True}


class ExperimentConfig(BaseModel):
    """Higher-level config used in Stage 5+. Stage 0 doesn't exercise this."""

    name: str
    stages: list[str]
    seeds: list[int] = Field(default_factory=lambda: [0])


# ---------------------------------------------------------------------------
# YAML loading with simple `defaults` resolution. Hydra is overkill for Stage 0.
# ---------------------------------------------------------------------------


def _deep_merge(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    out = dict(a)
    for k, v in b.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_stage_config(path: Path, configs_root: Path | None = None) -> StageConfig:
    """Load a stage YAML and merge `base.yaml` + any `defaults:` references.

    Permissive: every field not in StageConfig is parked under `.extra`.
    """
    path = Path(path)
    if configs_root is None:
        configs_root = path.parent
    base_path = configs_root / "base.yaml"
    base = _read_yaml(base_path) if base_path.exists() else {}
    raw = _read_yaml(path)
    merged = _deep_merge(base, raw)
    # Drop the `defaults:` sentinel — we don't run a Hydra resolver.
    merged.pop("defaults", None)

    known_fields = set(StageConfig.model_fields)
    base_kwargs: dict[str, Any] = {}
    extra: dict[str, Any] = {}
    for k, v in merged.items():
        if k in known_fields:
            base_kwargs[k] = v
        else:
            extra[k] = v

    base_kwargs.setdefault("stage", path.stem.replace("stage_", ""))
    base_kwargs.setdefault("output_subdir", base_kwargs["stage"])
    base_kwargs["extra"] = extra
    base_kwargs["config_path"] = path
    base_kwargs["base_path"] = base_path
    return StageConfig.model_validate(base_kwargs)


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML at {path} did not parse to a mapping.")
    return data
