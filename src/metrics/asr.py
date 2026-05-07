"""ASR-at-K: probe accuracy after the first K agent steps (spec §9.1).

For each trajectory, predict τ from its first K steps, threshold at 0.5,
compare to ground truth, average. Bootstrap CIs over trajectories.
"""

from __future__ import annotations

import numpy as np

from src.agent.runner import Trajectory
from src.probe.logistic import LogisticProbe


def _bootstrap_ci(
    correct: np.ndarray, *, n_resamples: int = 500, seed: int = 0, alpha: float = 0.05
) -> tuple[float, float]:
    if correct.size == 0:
        return (0.0, 0.0)
    rng = np.random.default_rng(seed)
    n = correct.size
    means = np.empty(n_resamples, dtype=np.float64)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        means[i] = correct[idx].mean()
    lo = float(np.quantile(means, alpha / 2))
    hi = float(np.quantile(means, 1 - alpha / 2))
    return (lo, hi)


def compute_asr_at_k(
    trajectories: list[Trajectory],
    ground_truth: list[int],
    probe: LogisticProbe,
    K: int,
    *,
    threshold: float = 0.5,
    seed: int = 0,
) -> dict[str, float | int]:
    """Return ASR plus 95 % bootstrap CI."""
    if len(trajectories) != len(ground_truth):
        raise ValueError("trajectories and ground_truth must align.")
    correct = np.zeros(len(trajectories), dtype=np.float64)
    for i, (t, y) in enumerate(zip(trajectories, ground_truth)):
        p = probe.predict_proba_truncated(t, K)
        pred = int(p >= threshold)
        correct[i] = float(pred == y)
    asr = float(correct.mean()) if correct.size else 0.0
    lo, hi = _bootstrap_ci(correct, seed=seed)
    return {
        "asr": asr,
        "ci_low": lo,
        "ci_high": hi,
        "K": K,
        "n": len(trajectories),
    }
