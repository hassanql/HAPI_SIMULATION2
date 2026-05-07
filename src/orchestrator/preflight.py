"""Pre-stage environment checks.

Stage 0 only needs disk-space + config-readable checks. Stage 3+ adds CUDA
visibility, HF token, model-cache directory.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path

from src.experiment.config import StageConfig


@dataclass
class PreflightResult:
    ok: bool
    issues: list[str]


def preflight_check(
    config: StageConfig, *, results_root: Path, min_disk_gb: float = 1.0
) -> PreflightResult:
    issues: list[str] = []

    # 1. Disk space.
    try:
        free_gb = shutil.disk_usage(str(results_root.parent if results_root.parent.exists() else "."))[2] / (1024**3)
        if free_gb < min_disk_gb:
            issues.append(
                f"Free disk < {min_disk_gb:.1f} GB at {results_root}: {free_gb:.2f} GB"
            )
    except Exception as e:  # pragma: no cover - best-effort
        issues.append(f"Could not stat disk usage: {e}")

    # 2. Output dir is writable (or creatable).
    try:
        results_root.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        issues.append(f"Cannot create results dir {results_root}: {e}")

    # 3. Backend-specific checks.
    if config.backend == "server":
        if not os.environ.get("HF_TOKEN") and not os.environ.get("HUGGING_FACE_HUB_TOKEN"):
            issues.append(
                "HF_TOKEN is not set. Stages that download gated models will fail. "
                "Set it via .env or shell."
            )
        if not os.environ.get("CUDA_VISIBLE_DEVICES"):
            issues.append("CUDA_VISIBLE_DEVICES is not set; vLLM may not find a GPU.")

    # 4. Mock backend always OK.
    return PreflightResult(ok=not issues, issues=issues)
