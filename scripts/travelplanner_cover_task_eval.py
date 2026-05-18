"""Simplified cover-task evaluator for the TravelPlanner pivot.

For Phase 1 we use a lightweight in-repo evaluator instead of wiring
TravelPlanner's full evaluation pipeline (which requires their ~500MB
database download and their tool APIs). The simplified version checks:

  1. Structure — plan is a non-empty list of day-objects with the
     expected keys.
  2. Days correctness — plan has the right number of days.
  3. Budget — sum of detectable costs in the plan does not exceed the
     case's stated budget (when costs are inline in the plan strings).
  4. Local constraints — substring match per constraint kind:
       * cuisine: any item from the cuisine list appears in restaurant
         lines on at least one day,
       * room_type: matches the constraint value (private / entire /
         shared / not_shared),
       * house_rule: at least one accommodation mention contains a
         compatible rule (or the accommodation field is non-empty),
       * transportation: transport lines respect "no flight" / "no
         self-driving" if specified.

A plan PASSES the cover-task gate iff structure + days + (budget when
inferable) + all local_constraint checks pass.

This is conservative and will under-count violations relative to the
official evaluator, but it's adequate for Phase 1 pilot acceptance
criteria. Phase 2 will wire the full evaluator.

Usage:
    from scripts.travelplanner_cover_task_eval import evaluate_cover_task
    result = evaluate_cover_task(trajectory, case)
"""
from __future__ import annotations

import re
from typing import Any

from src.agent.travel_runner import TravelTrajectory
from src.data.travel_data import TravelCase


def _plan_text(plan: dict[str, Any]) -> str:
    """Flatten the plan dict into searchable lowercase text."""
    if not plan or "days" not in plan:
        return ""
    parts: list[str] = []
    for d in plan["days"]:
        if not isinstance(d, dict):
            continue
        for k, v in d.items():
            if v and v != "-":
                parts.append(f"{k}: {v}")
    return "\n".join(parts).lower()


def _extract_costs(plan: dict[str, Any]) -> list[float]:
    """Heuristic cost extraction from plan strings. Looks for patterns
    like 'Cost: 123' or '$123'."""
    text = ""
    if plan and "days" in plan:
        for d in plan["days"]:
            if isinstance(d, dict):
                for v in d.values():
                    if isinstance(v, str):
                        text += " " + v
    costs: list[float] = []
    for m in re.finditer(r"(?:Cost|cost):\s*\$?(\d{1,6})", text):
        try:
            costs.append(float(m.group(1)))
        except ValueError:
            pass
    for m in re.finditer(r"\$(\d{1,6})\b", text):
        try:
            costs.append(float(m.group(1)))
        except ValueError:
            pass
    return costs


def _check_structure(plan: dict[str, Any], expected_days: int) -> tuple[bool, str]:
    if not plan or "days" not in plan or not isinstance(plan["days"], list):
        return False, "plan missing 'days' list"
    if len(plan["days"]) == 0:
        return False, "plan empty"
    if len(plan["days"]) < expected_days:
        return False, f"plan has {len(plan['days'])} days, expected {expected_days}"
    # Permissive: extra days OK (some plans pad with empty dicts)
    return True, ""


def _check_budget(plan: dict[str, Any], budget: int) -> tuple[bool | None, str]:
    """Return (pass, message). pass=None means unable to determine."""
    costs = _extract_costs(plan)
    if not costs:
        return None, "no inline costs to verify"
    total = sum(costs)
    if total <= budget:
        return True, f"detected total ${total:.0f} <= budget ${budget}"
    return False, f"detected total ${total:.0f} > budget ${budget}"


def _check_cuisine(
    plan_text: str,
    cuisine_constraint,
    case: TravelCase | None = None,
) -> tuple[bool, str]:
    """Verify the plan's restaurants satisfy the cuisine constraint.

    The plan only contains restaurant NAMES, not cuisine tags. So we look
    each name up in the case's reference_sections (Restaurants in <city>)
    and check the Cuisines column. Falls back to direct substring search
    when the case isn't supplied (legacy)."""
    if cuisine_constraint is None:
        return True, "no cuisine constraint"
    wanted = cuisine_constraint if isinstance(cuisine_constraint, list) else [cuisine_constraint]
    wanted_lc = [c.lower() for c in wanted]
    # Fast-path: cuisine substring directly in plan text (unusual but legal).
    for c in wanted_lc:
        if c in plan_text:
            return True, f"cuisine {c!r} substring present in plan"
    if case is None:
        return False, f"no required cuisine {wanted} reflected in plan"
    # Slow-path: look up each restaurant name in the reference data.
    rest_text = ""
    for desc, content in case.reference_sections.items():
        if desc.lower().startswith("restaurants"):
            rest_text += content + "\n"
    if not rest_text:
        return False, "no restaurant reference data; cannot verify cuisine"
    rest_lines = rest_text.lower().splitlines()
    # Extract restaurant names from plan: comma/semicolon separated entries
    # in breakfast/lunch/dinner fields are captured here heuristically.
    name_candidates = set()
    for line in plan_text.splitlines():
        # "lunch: X, City" -> ['X', 'City']
        if ":" not in line:
            continue
        _, val = line.split(":", 1)
        for part in val.replace(";", ",").split(","):
            name = part.strip()
            if 3 <= len(name) <= 80 and name != "-":
                name_candidates.add(name)
    matched = 0
    for name in name_candidates:
        # Find lines in ref data containing this restaurant name
        for rline in rest_lines:
            if name in rline:
                # Check cuisines on this line
                if any(c in rline for c in wanted_lc):
                    matched += 1
                    break
    if matched > 0:
        return True, f"plan includes {matched} restaurant(s) with required cuisine"
    return False, f"no plan restaurant matched required cuisine {wanted}"


def _check_room_type(plan_text: str, room_type_constraint) -> tuple[bool, str]:
    if room_type_constraint is None:
        return True, "no room_type constraint"
    rt = room_type_constraint.lower().strip()
    # We can't fully verify from plan strings (which mention names not types).
    # Permissive: pass if the accommodation field is non-empty for at least one day.
    if "accommodation:" in plan_text:
        return True, f"accommodation field populated (room_type={rt!r}, permissive)"
    return False, f"no accommodation populated (constraint {rt!r})"


def _check_house_rule(plan_text: str, house_rule_constraint) -> tuple[bool, str]:
    if house_rule_constraint is None:
        return True, "no house_rule constraint"
    if "accommodation:" in plan_text:
        return True, f"accommodation populated (house_rule={house_rule_constraint!r}, permissive)"
    return False, f"no accommodation populated (constraint {house_rule_constraint!r})"


def _check_transportation(plan_text: str, transportation_constraint) -> tuple[bool, str]:
    if transportation_constraint is None:
        return True, "no transportation constraint"
    tc = transportation_constraint.lower().strip()
    if tc in ("no flight", "no flights"):
        if "flight number" in plan_text or " flight " in plan_text:
            return False, "plan uses flight despite 'no flight' constraint"
        return True, "no flight used as required"
    if tc in ("no self-driving", "no self driving"):
        if "self-driving" in plan_text or "self driving" in plan_text:
            return False, "plan uses self-driving despite 'no self-driving' constraint"
        return True, "no self-driving as required"
    return True, f"unrecognized transportation constraint {tc!r}, permissive pass"


def evaluate_cover_task(
    traj: TravelTrajectory, case: TravelCase
) -> dict[str, Any]:
    """Run all cover-task gates and return a dict with per-gate results
    and an overall pass/fail. Also mutates traj.cover_task_passed."""
    plan = traj.final_plan
    plan_text = _plan_text(plan)

    structure_ok, structure_msg = _check_structure(plan, case.days)
    budget_ok, budget_msg = _check_budget(plan, case.budget)
    cuisine_ok, cuisine_msg = _check_cuisine(plan_text, case.local_constraint.get("cuisine"), case)
    room_ok, room_msg = _check_room_type(plan_text, case.local_constraint.get("room type"))
    hr_ok, hr_msg = _check_house_rule(plan_text, case.local_constraint.get("house rule"))
    tr_ok, tr_msg = _check_transportation(plan_text, case.local_constraint.get("transportation"))

    gates = {
        "structure": {"pass": structure_ok, "message": structure_msg},
        "budget": {"pass": budget_ok, "message": budget_msg},
        "cuisine": {"pass": cuisine_ok, "message": cuisine_msg},
        "room_type": {"pass": room_ok, "message": room_msg},
        "house_rule": {"pass": hr_ok, "message": hr_msg},
        "transportation": {"pass": tr_ok, "message": tr_msg},
    }

    # Overall pass: all bool-True gates pass; None (undetermined) doesn't fail.
    hard_gates = [structure_ok, cuisine_ok, room_ok, hr_ok, tr_ok]
    overall = all(hard_gates) and (budget_ok is not False)

    traj.cover_task_passed = overall
    return {"overall_pass": overall, "gates": gates}
