"""Tests for src/data/travel_persona.py — persona generation determinism
and τ-conditioning."""
from __future__ import annotations

from src.data.travel_data import TravelCase
from src.data.travel_persona import (
    ATTRIBUTE_TABLES,
    all_dimensions_for_attribute,
    generate_persona,
    generate_persona_pair,
)


def _mk_case(idx: int = 0) -> TravelCase:
    return TravelCase(
        case_id=f"tp_test_{idx:03d}",
        idx=idx,
        org="Boston", dest="Phoenix",
        days=5, date=["2024-04-01"] * 5,
        people_number=1, local_constraint={"house rule": None, "cuisine": None,
                                            "room type": None, "transportation": None},
        budget=4500,
        query="Plan a 5-day trip from Boston to Phoenix for $4500 in April 2024.",
        level="medium",
        reference_sections={"Restaurants in Phoenix": "(stub)"},
    )


def test_persona_is_deterministic_for_same_seed():
    case = _mk_case()
    p1 = generate_persona(case, 1, "pregnancy", base_seed=42)
    p2 = generate_persona(case, 1, "pregnancy", base_seed=42)
    assert p1.stated_prefs == p2.stated_prefs
    assert p1.seed == p2.seed


def test_persona_differs_across_tau_in_expected_direction():
    """Aggregate over many cases: τ=1 should produce markers more often
    than τ=0 (because each marker has p1 > p0 by construction)."""
    n_runs = 50
    pregnancy_marker_counts_tau1 = 0
    pregnancy_marker_counts_tau0 = 0
    for i in range(n_runs):
        case = _mk_case(idx=i)
        p1 = generate_persona(case, 1, "pregnancy", base_seed=i)
        p0 = generate_persona(case, 0, "pregnancy", base_seed=i)
        # Count markers (target values) per persona
        for dim, (target, _p1, _p0) in ATTRIBUTE_TABLES["pregnancy"].items():
            if p1.stated_prefs[dim] == target:
                pregnancy_marker_counts_tau1 += 1
            if p0.stated_prefs[dim] == target:
                pregnancy_marker_counts_tau0 += 1
    # τ=1 should produce markers more often than τ=0
    assert pregnancy_marker_counts_tau1 > pregnancy_marker_counts_tau0, (
        f"τ=1 markers: {pregnancy_marker_counts_tau1}, τ=0 markers: {pregnancy_marker_counts_tau0}"
    )


def test_all_three_attributes_supported():
    case = _mk_case()
    for attr in ("pregnancy", "religious_observance", "financial_distress"):
        p = generate_persona(case, 1, attr, base_seed=0)
        assert p.tau_attr == attr
        # All declared dimensions should be present in stated_prefs
        for dim in all_dimensions_for_attribute(attr):
            assert dim in p.stated_prefs, f"missing dim {dim} for {attr}"


def test_generate_persona_pair_returns_both_tau():
    case = _mk_case()
    p0, p1 = generate_persona_pair(case, "pregnancy", base_seed=7)
    assert p0.tau == 0
    assert p1.tau == 1
    assert p0.case_id == p1.case_id == case.case_id


def test_unknown_attribute_raises():
    case = _mk_case()
    try:
        generate_persona(case, 1, "made_up_attribute", base_seed=0)
        assert False, "should have raised ValueError"
    except ValueError:
        pass
