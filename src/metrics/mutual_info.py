"""Bits-per-dollar (spec §9.2).

Approximate `I(τ; trajectory)` via the f-divergence variational lower bound:

    I_lower = E[log p_probe(τ_true | traj)] - log P(τ_true)

Then divide by mean total cost.
"""

from __future__ import annotations

import math

import numpy as np

from src.agent.runner import Trajectory
from src.probe.logistic import LogisticProbe


_EPS = 1e-9


def _safe_log2(x: float) -> float:
    return math.log2(max(x, _EPS))


def compute_bits_per_dollar(
    trajectories: list[Trajectory],
    ground_truth: list[int],
    probe: LogisticProbe,
) -> dict[str, float]:
    if len(trajectories) != len(ground_truth):
        raise ValueError("trajectories and ground_truth must align.")
    if not trajectories:
        return {"bits": 0.0, "mean_cost": 0.0, "bits_per_dollar": 0.0}

    base_pos = sum(ground_truth) / len(ground_truth)
    base_pos = min(max(base_pos, _EPS), 1 - _EPS)

    total = 0.0
    for t, y in zip(trajectories, ground_truth):
        p_pos = probe.predict_proba(t)
        p_true = p_pos if y == 1 else (1.0 - p_pos)
        p_base = base_pos if y == 1 else (1.0 - base_pos)
        total += _safe_log2(p_true) - _safe_log2(p_base)
    bits = total / len(trajectories)
    mean_cost = float(np.mean([t.total_cost for t in trajectories]))
    bpd = bits / mean_cost if mean_cost > 0 else 0.0
    return {
        "bits": float(bits),
        "mean_cost": mean_cost,
        "bits_per_dollar": float(bpd),
        "n": float(len(trajectories)),
    }
