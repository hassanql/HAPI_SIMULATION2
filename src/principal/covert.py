"""Covert adversarial principal (spec §6.3).

Three implementations:

  6.3.1 hand_curated      — load from data/covert_tasks/<attribute>.json
  6.3.2 llm_attacker      — generate via LLM (Stage 4+)
  6.3.3 simulation_search — enumerate + simulate, pick maximiser (Stage 4+)

Stage 0 ships the hand-curated loader (with a deterministic seed-based picker)
plus interface stubs for the other two.

The covert principal does **not** change the agent's target — `target="diagnosis"`
and `target_options` are the case's MedQA options. Only the `chief_complaint`
shifts (the agent thinks it's still doing differential diagnosis).
"""

from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

from src.data.attribute_schema import SensitiveAttribute
from src.data.augmenter import AugmentedCase
from src.principal.task_spec import TaskSpec


class CovertStrategy(str, Enum):
    HAND_CURATED = "hand_curated"
    LLM_ATTACKER = "llm_attacker"
    SIMULATION_SEARCH = "simulation_search"


def _load_hand_curated_pool(attribute: SensitiveAttribute, root: Path) -> list[dict[str, Any]]:
    path = root / f"{attribute.name}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"No hand-curated covert tasks for {attribute.name!r}: expected {path}"
        )
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    pool = data.get("covert_complaints") or []
    if not pool:
        raise ValueError(f"{path} has no covert_complaints entries.")
    return pool


def make_covert_task(
    case: AugmentedCase,
    attribute: SensitiveAttribute,
    *,
    strategy: CovertStrategy = CovertStrategy.HAND_CURATED,
    rng: np.random.Generator | None = None,
    covert_tasks_root: Path = Path("data/covert_tasks"),
) -> TaskSpec:
    if strategy is CovertStrategy.HAND_CURATED:
        pool = _load_hand_curated_pool(attribute, covert_tasks_root)
        if rng is None:
            rng = np.random.default_rng(0)
        idx = int(rng.integers(0, len(pool)))
        complaint = pool[idx]["complaint"]
        return TaskSpec(
            chief_complaint=complaint,
            target="diagnosis",
            target_options=list(case.options.keys()),
            principal_type="covert",
        )

    if strategy is CovertStrategy.LLM_ATTACKER:
        raise NotImplementedError(
            "LLM-attacker covert strategy is implemented in Stage 4."
        )

    if strategy is CovertStrategy.SIMULATION_SEARCH:
        raise NotImplementedError(
            "Simulation-search covert strategy is implemented in Stage 4 calibration."
        )

    raise ValueError(f"Unknown covert strategy: {strategy}")
