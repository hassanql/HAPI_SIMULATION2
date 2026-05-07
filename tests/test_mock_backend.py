"""Mock backend determinism + family classification."""

from __future__ import annotations

import json

from src.llm.mock_backend import MockBackend


def test_classify_action_proposal():
    p = "Propose 3 candidate next actions. Format as JSON array."
    assert MockBackend.classify(p) == "action_proposal"


def test_classify_belief_update():
    p = "score the likelihood ... 0-10 scale ..."
    assert MockBackend.classify(p) == "belief_update"


def test_classify_patient_response():
    p = "You are simulating a patient.\n...Clinician's query: ..."
    assert MockBackend.classify(p) == "patient_response"


def test_classify_judge_probe():
    p = "auditing a diagnostic conversation ... P(positive) =\n"
    assert MockBackend.classify(p) == "judge_probe"


def test_classify_covert_complaint():
    p = "designing a clinical chief complaint ..."
    assert MockBackend.classify(p) == "covert_complaint"


def test_action_proposal_returns_valid_json_array():
    backend = MockBackend()
    out = backend.respond(
        role="agent",
        prompt="Propose 3 candidate next actions. Format as JSON array.",
    )
    parsed = json.loads(out)
    assert isinstance(parsed, list)
    assert len(parsed) == 3
    for entry in parsed:
        assert "type" in entry and "query" in entry


def test_belief_update_returns_valid_json_dict():
    backend = MockBackend()
    prompt = (
        "score the likelihood\n0-10 scale\nOptions:\n"
        "  A: option A\n  B: option B\n  C: option C\n  D: option D\n"
    )
    out = backend.respond(role="agent", prompt=prompt)
    parsed = json.loads(out)
    assert set(parsed.keys()) == {"A", "B", "C", "D"}
    for v in parsed.values():
        assert 0 <= float(v) <= 10


def test_judge_probe_returns_number_in_range():
    backend = MockBackend()
    out = backend.respond(
        role="judge",
        prompt="auditing a diagnostic conversation\n... P(positive) =\n",
    )
    val = float(out)
    assert 0.0 <= val <= 1.0


def test_responses_are_deterministic():
    backend1 = MockBackend()
    backend2 = MockBackend()
    prompt = "Propose 3 candidate next actions. Format as JSON array."
    assert backend1.respond(role="agent", prompt=prompt) == backend2.respond(role="agent", prompt=prompt)


def test_revision_changes_only_when_file_changes():
    """Two MockBackend instances built from the same source file should report
    the same revision SHA. (Stage 0 cache invalidation invariant —
    OPEN_QUESTIONS.md #10.)"""
    a, b = MockBackend(), MockBackend()
    assert a.revision == b.revision
    assert a.revision.startswith("mock-")
