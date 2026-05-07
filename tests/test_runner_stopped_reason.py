"""Regression tests for `_stopped_reason` labeling.

The labeler must mirror `PolicyConfig.epsilon_entropy`, otherwise raising
the policy's stop threshold (e.g. 0.3 → 0.6) silently mislabels stopped
trajectories as "loop_exit" instead of "entropy_threshold". See
OPEN_QUESTIONS #47.
"""

from __future__ import annotations

import math

from src.agent.belief import Belief
from src.agent.runner import _stopped_reason


def _belief_with_target_entropy(options: list[str], target_h: float) -> Belief:
    """Construct a 4-option belief with one option at probability `p_top`
    and the rest uniform, calibrated so total Shannon entropy ≈ target_h."""
    # For 4 options with top p, others (1-p)/3 uniform:
    # H = -p ln p - 3 * ((1-p)/3) ln((1-p)/3) = -p ln p - (1-p) ln((1-p)/3)
    # We sweep p to find the closest entropy.
    p_top, best = 0.5, 1e9
    for cand in [0.55 + 0.005 * i for i in range(90)]:
        rest = (1 - cand) / 3
        if rest <= 0:
            continue
        h = -cand * math.log(cand) - 3 * rest * math.log(rest)
        if abs(h - target_h) < best:
            best = abs(h - target_h)
            p_top = cand
    rest = (1 - p_top) / 3
    return Belief(options, prior={options[0]: p_top, **{o: rest for o in options[1:]}})


def test_entropy_threshold_uses_explicit_epsilon() -> None:
    """Entropy 0.5 nats: with epsilon_entropy=0.6 → entropy_threshold;
    with epsilon_entropy=0.3 → loop_exit."""
    options = ["A", "B", "C", "D"]
    b = _belief_with_target_entropy(options, target_h=0.5)
    # Sanity check: belief entropy is approximately 0.5 nats.
    assert 0.3 < b.entropy() < 0.6

    raised = _stopped_reason(b, cost=10.0, n_steps=5, max_q=15, cost_cap=500.0,
                             epsilon_entropy=0.6)
    legacy = _stopped_reason(b, cost=10.0, n_steps=5, max_q=15, cost_cap=500.0,
                             epsilon_entropy=0.3)
    assert raised == "entropy_threshold"
    assert legacy == "loop_exit"


def test_max_queries_takes_priority() -> None:
    options = ["A", "B", "C", "D"]
    b = _belief_with_target_entropy(options, target_h=0.1)  # very confident
    # Even at low entropy, hitting max_q labels max_queries.
    assert _stopped_reason(b, cost=10.0, n_steps=15, max_q=15, cost_cap=500.0,
                           epsilon_entropy=0.3) == "max_queries"


def test_cost_budget_takes_priority_over_entropy() -> None:
    options = ["A", "B", "C", "D"]
    b = _belief_with_target_entropy(options, target_h=0.1)
    assert _stopped_reason(b, cost=600.0, n_steps=5, max_q=15, cost_cap=500.0,
                           epsilon_entropy=0.3) == "cost_budget"


def test_default_epsilon_is_legacy_03() -> None:
    """Calling without epsilon_entropy should fall back to 0.3 for back-compat."""
    options = ["A", "B", "C", "D"]
    b = _belief_with_target_entropy(options, target_h=0.5)
    assert _stopped_reason(b, cost=10.0, n_steps=5, max_q=15,
                           cost_cap=500.0) == "loop_exit"
    b_low = _belief_with_target_entropy(options, target_h=0.2)
    assert _stopped_reason(b_low, cost=10.0, n_steps=5, max_q=15,
                           cost_cap=500.0) == "entropy_threshold"
