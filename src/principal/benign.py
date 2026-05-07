"""Benign principal — agent runs the standard MedQA diagnosis loop (spec §6.1)."""

from __future__ import annotations

from src.data.augmenter import AugmentedCase
from src.principal.task_spec import TaskSpec


def make_benign_task(case: AugmentedCase) -> TaskSpec:
    return TaskSpec(
        chief_complaint=case.question,
        target="diagnosis",
        target_options=list(case.options.keys()),
        principal_type="benign",
    )
