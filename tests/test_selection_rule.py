"""Unit tests for `_apply_selection_rule` — the EIG-policy decision tree.

Three branches:
  1. max(EIG) < epsilon_stop          -> DIAGNOSE
  2. max(EIG) > eig_raw_floor         -> argmax raw EIG (NEW: ignores cost)
  3. otherwise                        -> argmax EIG / cost (legacy)

Default eig_raw_floor=inf skips branch 2 entirely (back-compat).
See OPEN_QUESTIONS #48.
"""

from __future__ import annotations

from src.agent.action_space import Action, ActionType
from src.agent.belief import Belief
from src.agent.policy import _apply_selection_rule


def _key(a: Action) -> str:
    return f"{a.type.value}|{a.query}"


def _candidates() -> list[Action]:
    """A typical 3-candidate proposal: cheap history vs cheap exam vs expensive test."""
    return [
        Action(type=ActionType.ASK_HISTORY, query="any cough?", cost=1.0),
        Action(type=ActionType.ASK_EXAM, query="auscultate lungs", cost=5.0),
        Action(type=ActionType.ORDER_TEST, query="chest x-ray", cost=50.0),
    ]


def test_diagnose_when_all_eigs_below_eps_stop() -> None:
    cands = _candidates()
    eig = {_key(c): 0.01 for c in cands}  # all below 0.05
    belief = Belief(["A", "B", "C", "D"], prior={"A": 0.4, "B": 0.3, "C": 0.2, "D": 0.1})
    out = _apply_selection_rule(
        candidates=cands, eig_estimates=eig, cand_key=_key,
        belief=belief, epsilon_stop=0.05, eig_raw_floor=float("inf"),
    )
    assert out.type is ActionType.DIAGNOSE
    assert out.query == "A"  # MAP estimate


def test_legacy_eig_over_cost_when_floor_inf() -> None:
    """With floor=inf, branch 2 is skipped → cost-aware always wins."""
    cands = _candidates()
    eig = {
        _key(cands[0]): 0.10,    # ASK_HISTORY  → 0.10/1 = 0.100
        _key(cands[1]): 0.15,    # ASK_EXAM     → 0.15/5 = 0.030
        _key(cands[2]): 0.27,    # ORDER_TEST   → 0.27/50 = 0.0054
    }
    belief = Belief(["A", "B", "C", "D"])
    out = _apply_selection_rule(
        candidates=cands, eig_estimates=eig, cand_key=_key,
        belief=belief, epsilon_stop=0.05, eig_raw_floor=float("inf"),
    )
    assert out.type is ActionType.ASK_HISTORY  # cheap wins on EIG/cost


def test_hybrid_picks_test_when_floor_engages() -> None:
    """Same EIGs but with floor=0.15 → branch 2 triggers; ORDER_TEST (raw EIG max) wins."""
    cands = _candidates()
    eig = {
        _key(cands[0]): 0.10,
        _key(cands[1]): 0.15,
        _key(cands[2]): 0.27,    # max raw EIG
    }
    belief = Belief(["A", "B", "C", "D"])
    out = _apply_selection_rule(
        candidates=cands, eig_estimates=eig, cand_key=_key,
        belief=belief, epsilon_stop=0.05, eig_raw_floor=0.15,
    )
    assert out.type is ActionType.ORDER_TEST


def test_hybrid_falls_back_to_cost_aware_when_no_candidate_clears_floor() -> None:
    """All EIGs below 0.15 floor → fall back to cost-aware. Cheap action wins."""
    cands = _candidates()
    eig = {
        _key(cands[0]): 0.08,
        _key(cands[1]): 0.10,
        _key(cands[2]): 0.13,    # below 0.15 floor
    }
    belief = Belief(["A", "B", "C", "D"])
    out = _apply_selection_rule(
        candidates=cands, eig_estimates=eig, cand_key=_key,
        belief=belief, epsilon_stop=0.05, eig_raw_floor=0.15,
    )
    assert out.type is ActionType.ASK_HISTORY  # 0.08/1 = 0.08, beats 0.10/5 and 0.13/50


def test_eps_stop_takes_priority_over_floor() -> None:
    """Even with high floor + high single EIG, if max(EIG) < epsilon_stop → DIAGNOSE."""
    cands = _candidates()
    eig = {_key(c): 0.03 for c in cands}  # all below epsilon_stop
    belief = Belief(["A", "B", "C", "D"], prior={"B": 0.7, "A": 0.1, "C": 0.1, "D": 0.1})
    out = _apply_selection_rule(
        candidates=cands, eig_estimates=eig, cand_key=_key,
        belief=belief, epsilon_stop=0.05, eig_raw_floor=0.15,
    )
    assert out.type is ActionType.DIAGNOSE
    assert out.query == "B"


def test_empty_eig_estimates_diagnoses_with_map() -> None:
    """Edge case: empty estimates dict → DIAGNOSE with MAP."""
    belief = Belief(["A", "B", "C", "D"], prior={"C": 0.6, "A": 0.2, "B": 0.1, "D": 0.1})
    out = _apply_selection_rule(
        candidates=[], eig_estimates={}, cand_key=_key,
        belief=belief, epsilon_stop=0.05, eig_raw_floor=0.15,
    )
    assert out.type is ActionType.DIAGNOSE
    assert out.query == "C"


def test_floor_at_exact_max_does_not_trigger() -> None:
    """`>` comparison: max(EIG) exactly at floor → fall through to cost-aware."""
    cands = _candidates()
    eig = {
        _key(cands[0]): 0.10,
        _key(cands[1]): 0.12,
        _key(cands[2]): 0.15,    # exactly at floor; should NOT trigger raw-EIG path
    }
    belief = Belief(["A", "B", "C", "D"])
    out = _apply_selection_rule(
        candidates=cands, eig_estimates=eig, cand_key=_key,
        belief=belief, epsilon_stop=0.05, eig_raw_floor=0.15,
    )
    # ASK_HISTORY: 0.10/1 = 0.100 ; ASK_EXAM: 0.12/5 = 0.024 ; ORDER_TEST: 0.15/50 = 0.003
    assert out.type is ActionType.ASK_HISTORY


# --- min_steps_before_diagnose (OPEN_QUESTIONS #50) -----------------------


def test_min_steps_blocks_early_diagnose_falls_through() -> None:
    """All EIGs < epsilon_stop but n_steps_so_far < min_steps_before_diagnose
    → don't DIAGNOSE; pick best available action instead."""
    cands = _candidates()
    eig = {_key(c): 0.03 for c in cands}  # below epsilon_stop=0.05
    belief = Belief(["A", "B", "C", "D"], prior={"A": 0.4, "B": 0.3, "C": 0.2, "D": 0.1})
    out = _apply_selection_rule(
        candidates=cands, eig_estimates=eig, cand_key=_key,
        belief=belief, epsilon_stop=0.05, eig_raw_floor=float("inf"),
        n_steps_so_far=1, min_steps_before_diagnose=3,
    )
    # Should fall through to argmax(EIG/cost). All EIGs equal at 0.03,
    # so cheapest wins: ASK_HISTORY (0.03/1 = 0.03).
    assert out.type is ActionType.ASK_HISTORY


def test_min_steps_allows_diagnose_once_threshold_reached() -> None:
    """Same scenario but n_steps_so_far >= min_steps_before_diagnose → DIAGNOSE."""
    cands = _candidates()
    eig = {_key(c): 0.03 for c in cands}
    belief = Belief(["A", "B", "C", "D"], prior={"B": 0.7, "A": 0.1, "C": 0.1, "D": 0.1})
    out = _apply_selection_rule(
        candidates=cands, eig_estimates=eig, cand_key=_key,
        belief=belief, epsilon_stop=0.05, eig_raw_floor=float("inf"),
        n_steps_so_far=3, min_steps_before_diagnose=3,
    )
    assert out.type is ActionType.DIAGNOSE
    assert out.query == "B"


def test_min_steps_default_zero_does_not_block() -> None:
    """Default min_steps_before_diagnose=0 reproduces legacy behavior."""
    cands = _candidates()
    eig = {_key(c): 0.03 for c in cands}
    belief = Belief(["A", "B", "C", "D"], prior={"C": 0.5, "A": 0.2, "B": 0.2, "D": 0.1})
    out = _apply_selection_rule(
        candidates=cands, eig_estimates=eig, cand_key=_key,
        belief=belief, epsilon_stop=0.05, eig_raw_floor=float("inf"),
        n_steps_so_far=0,  # min_steps_before_diagnose default 0
    )
    assert out.type is ActionType.DIAGNOSE
    assert out.query == "C"


# --- Budget-aware R3 (OPEN_QUESTIONS #51) ---------------------------------


def test_budget_aware_skips_unaffordable_high_eig_action() -> None:
    """High-EIG ORDER_TEST exceeds remaining budget → fall back to cheaper."""
    cands = _candidates()
    eig = {
        _key(cands[0]): 0.20,    # ASK_HISTORY  cost 1
        _key(cands[1]): 0.18,    # ASK_EXAM     cost 5
        _key(cands[2]): 0.30,    # ORDER_TEST   cost 50 — too expensive
    }
    belief = Belief(["A", "B", "C", "D"])
    # Remaining budget = 100 - 60 = 40, can't afford the $50 ORDER_TEST.
    out = _apply_selection_rule(
        candidates=cands, eig_estimates=eig, cand_key=_key,
        belief=belief, epsilon_stop=0.05, eig_raw_floor=0.15,
        cost_so_far=60.0, cost_budget=100.0,
    )
    # Filtered set: [ASK_HISTORY (EIG 0.20, cost 1), ASK_EXAM (EIG 0.18, cost 5)].
    # Both are above floor 0.15. Branch 2 (raw EIG argmax) picks ASK_HISTORY.
    assert out.type is ActionType.ASK_HISTORY


def test_budget_aware_picks_cheapest_when_all_unaffordable() -> None:
    """No candidate fits remaining budget → DIAGNOSE fallback."""
    cands = _candidates()
    eig = {_key(c): 0.20 for c in cands}
    belief = Belief(["A", "B", "C", "D"], prior={"D": 0.5, "A": 0.2, "B": 0.2, "C": 0.1})
    # Budget 0.5 left. Cheapest action is ASK_HISTORY at 1.0 — doesn't fit.
    out = _apply_selection_rule(
        candidates=cands, eig_estimates=eig, cand_key=_key,
        belief=belief, epsilon_stop=0.05, eig_raw_floor=0.15,
        cost_so_far=499.5, cost_budget=500.0,
    )
    assert out.type is ActionType.DIAGNOSE
    assert out.query == "D"


def test_budget_aware_default_inf_does_not_filter() -> None:
    """Default cost_budget=inf → all candidates feasible (legacy behavior)."""
    cands = _candidates()
    eig = {
        _key(cands[0]): 0.10,
        _key(cands[1]): 0.12,
        _key(cands[2]): 0.30,
    }
    belief = Belief(["A", "B", "C", "D"])
    out = _apply_selection_rule(
        candidates=cands, eig_estimates=eig, cand_key=_key,
        belief=belief, epsilon_stop=0.05, eig_raw_floor=0.15,
        cost_so_far=0.0,  # cost_budget default inf
    )
    # Without budget filter and floor=0.15: ORDER_TEST has raw EIG 0.30 > floor → picked.
    assert out.type is ActionType.ORDER_TEST
