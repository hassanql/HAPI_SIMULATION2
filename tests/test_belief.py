"""Belief: entropy, normalisation, Bayes update."""

from __future__ import annotations

import math

import pytest

from src.agent.belief import Belief, kl_divergence


def test_uniform_initialisation_sums_to_one():
    b = Belief(["A", "B", "C", "D"])
    assert math.isclose(sum(b.to_dict().values()), 1.0, abs_tol=1e-9)


def test_uniform_entropy_equals_log_n():
    b = Belief(["A", "B", "C", "D"])
    assert math.isclose(b.entropy(), math.log(4), abs_tol=1e-9)


def test_concentrated_belief_has_low_entropy():
    b = Belief(["A", "B"], prior={"A": 0.99, "B": 0.01})
    assert b.entropy() < 0.1


def test_bayes_update_keeps_normalised():
    b = Belief(["A", "B", "C"])
    b.update("obs", {"A": 5.0, "B": 1.0, "C": 1.0})
    s = sum(b.to_dict().values())
    assert math.isclose(s, 1.0, abs_tol=1e-9)
    # MAP should now be A.
    assert b.map_estimate() == "A"


def test_zero_likelihood_does_not_break_normalisation():
    """Pathological case: every likelihood is zero. Belief falls back to uniform."""
    b = Belief(["A", "B"])
    b.update("obs", {"A": 0.0, "B": 0.0})
    assert math.isclose(sum(b.to_dict().values()), 1.0, abs_tol=1e-9)


def test_negative_likelihood_rejected():
    b = Belief(["A", "B"])
    with pytest.raises(ValueError):
        b.update("obs", {"A": 1.0, "B": -0.5})


def test_likelihood_keys_must_match_options():
    b = Belief(["A", "B"])
    with pytest.raises(ValueError):
        b.update("obs", {"A": 1.0, "C": 1.0})


def test_kl_divergence_zero_for_identical_distributions():
    p = {"A": 0.3, "B": 0.7}
    assert kl_divergence(p, p) == pytest.approx(0.0, abs=1e-9)


def test_kl_divergence_positive_for_different_distributions():
    p = {"A": 0.9, "B": 0.1}
    q = {"A": 0.5, "B": 0.5}
    assert kl_divergence(p, q) > 0.0


def test_belief_copy_is_independent():
    b = Belief(["A", "B"])
    b2 = b.copy()
    b2.update("obs", {"A": 10.0, "B": 1.0})
    # Original should be unchanged.
    assert math.isclose(b["A"], 0.5, abs_tol=1e-9)
