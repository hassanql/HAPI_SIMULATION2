"""Metrics on synthetic toy data."""

from __future__ import annotations

from src.agent.action_space import Action, ActionType
from src.agent.runner import Step, Trajectory
from src.metrics.asr import compute_asr_at_k
from src.metrics.concealment import compute_concealment_ratio
from src.metrics.detectability import compute_detectability
from src.metrics.mutual_info import compute_bits_per_dollar
from src.principal.task_spec import TaskSpec
from src.probe.logistic import LogisticProbe


def _make_trajectory(case_id: str, principal: str, correct: bool, cost: float = 100.0) -> Trajectory:
    task = TaskSpec(
        chief_complaint="diag?",
        target="diagnosis",
        target_options=["A", "B"],
        principal_type=principal,  # type: ignore[arg-type]
    )
    step = Step(
        step_idx=1,
        action=Action(type=ActionType.ASK_HISTORY, query="?", cost=cost),
        observation="",
        belief_before={"A": 0.5, "B": 0.5},
        belief_after={"A": 0.6, "B": 0.4},
    )
    return Trajectory(
        case_id=case_id,
        task=task,
        steps=[step],
        final_diagnosis="A" if correct else "B",
        total_cost=cost,
        diagnostic_correct=correct,
    )


def test_asr_at_k_returns_well_formed_dict():
    trajs = [_make_trajectory(f"c{i}", "direct", correct=True) for i in range(8)]
    gt = [1, 0, 1, 0, 1, 1, 0, 0]
    probe = LogisticProbe()
    probe.fit_mock(trajs)
    out = compute_asr_at_k(trajs, gt, probe, K=2)
    assert set(out.keys()) >= {"asr", "ci_low", "ci_high", "K", "n"}
    assert 0.0 <= out["asr"] <= 1.0
    assert out["ci_low"] <= out["asr"] <= out["ci_high"]
    assert out["K"] == 2
    assert out["n"] == 8


def test_bits_per_dollar_returns_well_formed_dict():
    trajs = [_make_trajectory(f"c{i}", "direct", correct=True) for i in range(8)]
    gt = [1, 0, 1, 0, 1, 1, 0, 0]
    probe = LogisticProbe()
    probe.fit_mock(trajs)
    out = compute_bits_per_dollar(trajs, gt, probe)
    assert set(out.keys()) >= {"bits", "mean_cost", "bits_per_dollar"}
    assert out["mean_cost"] == 100.0


def test_concealment_ratio_one_when_accuracies_equal():
    benign = [_make_trajectory(f"b{i}", "benign", correct=True) for i in range(4)]
    covert = [_make_trajectory(f"c{i}", "covert", correct=True) for i in range(4)]
    out = compute_concealment_ratio(benign, covert)
    assert out["concealment_ratio"] == 1.0


def test_concealment_ratio_below_one_when_covert_worse():
    benign = [_make_trajectory(f"b{i}", "benign", correct=True) for i in range(4)]
    covert = [_make_trajectory(f"c{i}", "covert", correct=(i < 2)) for i in range(4)]
    out = compute_concealment_ratio(benign, covert)
    assert out["concealment_ratio"] < 1.0


def test_detectability_returns_per_condition_aucs():
    trajs = (
        [_make_trajectory(f"b{i}", "benign", True) for i in range(4)]
        + [_make_trajectory(f"d{i}", "direct", True) for i in range(4)]
        + [_make_trajectory(f"c{i}", "covert", True) for i in range(4)]
    )
    out = compute_detectability(trajs)
    assert 0.5 <= out["auc_direct_vs_benign"] <= 1.0
    assert 0.5 <= out["auc_covert_vs_benign"] <= 1.0
    # Stage 0 mock encodes the expected story: direct should be more detectable
    # than covert (this is exactly what the experiment is supposed to demonstrate).
    assert out["auc_direct_vs_benign"] > out["auc_covert_vs_benign"]
