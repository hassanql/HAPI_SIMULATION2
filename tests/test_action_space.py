"""Regression tests for `cost_of_action` in src.agent.action_space.

The original implementation used unanchored substring matching, so the
short key "ct" matched any query containing those two adjacent letters
(e.g. "galactocerebrosidase", "c-reactive", "electrophoresis"). All four
showed up in real R3 trajectories charged 6× too much. The fix uses
word-boundary matching (re.escape + \\b...\\b). See OPEN_QUESTIONS #49.
"""

from __future__ import annotations

import pytest

from src.agent.action_space import (
    ActionType,
    cost_of_action,
)


# --- Real CT scans should still cost 300 ---------------------------------


@pytest.mark.parametrize("query", [
    "Non-contrast head CT",
    "CT scan",
    "CT angiography of head and neck",
    "head CT with contrast",
    "Non-contrast CT of the head",
    "Computed tomography is not enough; we want CT",  # capitals
])
def test_real_ct_scans_get_300(query: str) -> None:
    assert cost_of_action(ActionType.ORDER_TEST, query) == 300.0


# --- False positives that must NOT cost 300 ------------------------------


@pytest.mark.parametrize("query,expected", [
    ("Enzyme assay for galactocerebrosidase activity", 50.0),
    ("Blood galactose-1-phosphate level", 50.0),
    ("Blood cultures and c-reactive protein (CRP)", 50.0),
    ("Hemoglobin electrophoresis", 50.0),
    ("Lung function test", 50.0),
    ("Active surveillance protocol", 50.0),
    ("Doctor's note review", 50.0),
])
def test_false_positives_no_longer_match_ct(query: str, expected: float) -> None:
    assert cost_of_action(ActionType.ORDER_TEST, query) == expected


# --- Other override keys still work --------------------------------------


@pytest.mark.parametrize("query,expected", [
    ("CBC", 15.0),
    ("Complete blood count (CBC)", 15.0),
    ("BMP", 15.0),
    ("LFT", 20.0),
    ("Urinalysis", 10.0),
    ("MRI brain with gadolinium", 800.0),
    ("Chest x-ray", 50.0),
    ("Chest xray", 50.0),
    ("Lumbar puncture", 250.0),
    ("Endoscopy with biopsy", 1200.0),  # endoscopy=1200 > biopsy=500
    ("Skin biopsy", 500.0),
    ("HIV test", 30.0),
    ("Hepatitis panel", 60.0),
    ("RPR for syphilis screening", 25.0),
    ("Beta-hCG", 20.0),
    ("Ultrasound abdomen", 150.0),
])
def test_other_overrides_unaffected(query: str, expected: float) -> None:
    assert cost_of_action(ActionType.ORDER_TEST, query) == expected


# --- Default cost when no override matches -------------------------------


@pytest.mark.parametrize("query", [
    "Throat swab",
    "Stool culture",
    "Blood gas analysis",
    "Vitamin D level",
])
def test_default_test_cost_when_no_override(query: str) -> None:
    assert cost_of_action(ActionType.ORDER_TEST, query) == 50.0


# --- Non-ORDER_TEST actions get the type's flat cost ---------------------


def test_ask_history_cost_unchanged() -> None:
    assert cost_of_action(ActionType.ASK_HISTORY, "Anything containing CT mentions") == 1.0


def test_ask_exam_cost_unchanged() -> None:
    assert cost_of_action(ActionType.ASK_EXAM, "Activity level assessment") == 5.0


def test_diagnose_cost_zero() -> None:
    assert cost_of_action(ActionType.DIAGNOSE, "A") == 0.0
