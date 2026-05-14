"""Probe stubs."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.agent.action_space import Action, ActionType
from src.agent.runner import Step, Trajectory
from src.principal.task_spec import TaskSpec
from src.probe.llm_judge import LLMJudgeProbe, _parse_probability, serialize_trajectory
from src.probe.logistic import LogisticProbe, extract_features


def _toy_trajectory(case_id: str = "case-x", principal: str = "benign") -> Trajectory:
    task = TaskSpec(
        chief_complaint="What is the most likely diagnosis?",
        target="diagnosis",
        target_options=["A", "B"],
        principal_type=principal,  # type: ignore[arg-type]
    )
    step = Step(
        step_idx=1,
        action=Action(type=ActionType.ASK_HISTORY, query="How long?", cost=1.0),
        observation="Two weeks.",
        belief_before={"A": 0.5, "B": 0.5},
        belief_after={"A": 0.7, "B": 0.3},
    )
    return Trajectory(
        case_id=case_id,
        task=task,
        steps=[step],
        final_diagnosis="A",
        total_cost=1.0,
        diagnostic_correct=True,
    )


def test_extract_features_keys():
    t = _toy_trajectory()
    feats = extract_features(t)
    expected = {
        "n_steps", "total_cost", "n_ask_history", "n_ask_exam",
        "n_order_test", "mean_eig", "max_eig",
    }
    assert expected.issubset(feats.keys())


def test_logistic_probe_mock_predict_in_range():
    probe = LogisticProbe()
    probe.fit_mock([])
    for principal in ("benign", "direct", "covert"):
        t = _toy_trajectory(principal=principal)
        p = probe.predict_proba(t)
        assert 0.0 <= p <= 1.0
        assert 0.0 <= probe.predict_proba_truncated(t, 1) <= 1.0


def test_logistic_probe_save_load_roundtrip(tmp_path: Path):
    probe = LogisticProbe()
    probe.fit_mock([])
    p1 = probe.predict_proba(_toy_trajectory())
    path = tmp_path / "probe.pkl"
    probe.save(path)
    probe2 = LogisticProbe.load(path)
    p2 = probe2.predict_proba(_toy_trajectory())
    assert p1 == p2


def test_real_logistic_fit_validates_inputs():
    """Stage 6 implements `fit()`. The unimplemented marker has been
    replaced by input-validation checks (empty list / mismatched lengths /
    missing label classes). See `src/probe/logistic.py::LogisticProbe.fit`.
    """
    probe = LogisticProbe()
    with pytest.raises(ValueError, match="at least one trajectory"):
        probe.fit([], [])
    with pytest.raises(ValueError, match="trajectory count"):
        probe.fit([_toy_trajectory()], [])
    # All-same-label train set should be rejected (LogisticRegression
    # requires both classes).
    with pytest.raises(ValueError, match="both label classes"):
        probe.fit([_toy_trajectory("c1"), _toy_trajectory("c2")], [1, 1])


def test_real_logistic_fit_predict_smoke():
    """End-to-end smoke: fit on a tiny dataset with both classes, get a
    probability back from predict_proba. Covers the TF-IDF + numerical
    feature concat path."""
    probe = LogisticProbe()
    trajectories = [
        _toy_trajectory("c1"),
        _toy_trajectory("c2"),
        _toy_trajectory("c3"),
        _toy_trajectory("c4"),
    ]
    probe.fit(trajectories, [1, 1, 0, 0])
    p = probe.predict_proba(trajectories[0])
    assert 0.0 <= p <= 1.0
    # Batch path matches solo path.
    bp = probe.predict_proba_batch(trajectories)
    assert len(bp) == 4
    for v in bp:
        assert 0.0 <= v <= 1.0


def test_judge_probe_returns_in_range(llm_client, prompts_dir, hiv_attribute):
    template = (prompts_dir / "llm_judge_probe.txt").read_text()
    judge = LLMJudgeProbe(client=llm_client, prompt_template=template)
    p = judge.predict_proba(_toy_trajectory(), hiv_attribute)
    assert 0.0 <= p <= 1.0


def test_parse_probability_handles_garbage():
    assert _parse_probability("not a number") == 0.5
    assert _parse_probability("0.42") == 0.42
    assert _parse_probability("1.5") == 1.0
    assert _parse_probability("-0.1") == 0.0


def test_serialise_trajectory_format():
    t = _toy_trajectory()
    out = serialize_trajectory(t)
    assert "[Step 1]" in out
    assert "Diagnosis: A" in out
    assert "Total cost: $1.00" in out
