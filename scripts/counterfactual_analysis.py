#!/usr/bin/env python3
"""Offline counterfactual: at each step of each trajectory, what would the
agent have selected under alternative rules? We can't fully replay (that
would need re-querying the LLM for new candidate proposals), but we can
re-run the SELECTION step over the K candidates that were actually
proposed at each timestep — that's the per-step diff that the policy
actually owns.

Rules compared:
  R0: production — argmax(EIG / cost), DIAGNOSE if max(EIG) < eps_stop
  R1: lower epsilon_stop      0.05 -> 0.025  (else identical to R0)
  R2: half test costs         ORDER_TEST cost 50 -> 25 (else identical)
  R3: hybrid                  argmax(raw EIG) if max_eig > 0.15, else argmax(EIG/cost)

For each trajectory, the "first divergence step" is the earliest timestep
where rule X picks a different action than R0 picked. We count how often
the divergent action under each rule is an ORDER_TEST (the systemic gap
this run revealed).
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

DEFAULT_COSTS = {"ASK_HISTORY": 1.0, "ASK_EXAM": 5.0, "ORDER_TEST": 50.0, "DIAGNOSE": 0.0}
EPS_STOP = 0.05
HYBRID_RAW_EIG_FLOOR = 0.15


def parse_action_key(key: str) -> tuple[str, str]:
    """`'ORDER_TEST|Chest x-ray'` -> ('ORDER_TEST', 'Chest x-ray')"""
    if "|" in key:
        atype, query = key.split("|", 1)
    else:
        atype, query = key, ""
    return atype, query


def cost_of(action_type: str, override_test_cost: float | None = None) -> float:
    if action_type == "ORDER_TEST" and override_test_cost is not None:
        return override_test_cost
    return DEFAULT_COSTS.get(action_type, 1.0)


def select_R0(eig_estimates: dict[str, float]) -> tuple[str, str]:
    """Production rule: argmax(EIG/cost) with epsilon_stop trigger."""
    if not eig_estimates:
        return ("DIAGNOSE", "(empty)")
    max_eig = max(eig_estimates.values())
    if max_eig < EPS_STOP:
        return ("DIAGNOSE", "(commit MAP)")
    best = max(eig_estimates.items(),
               key=lambda kv: kv[1] / max(cost_of(parse_action_key(kv[0])[0]), 0.1))
    return parse_action_key(best[0])


def select_R1(eig_estimates: dict[str, float]) -> tuple[str, str]:
    """Lower epsilon_stop to 0.025."""
    if not eig_estimates:
        return ("DIAGNOSE", "(empty)")
    max_eig = max(eig_estimates.values())
    if max_eig < 0.025:
        return ("DIAGNOSE", "(commit MAP)")
    best = max(eig_estimates.items(),
               key=lambda kv: kv[1] / max(cost_of(parse_action_key(kv[0])[0]), 0.1))
    return parse_action_key(best[0])


def select_R2(eig_estimates: dict[str, float]) -> tuple[str, str]:
    """Halve ORDER_TEST cost (50 -> 25)."""
    if not eig_estimates:
        return ("DIAGNOSE", "(empty)")
    max_eig = max(eig_estimates.values())
    if max_eig < EPS_STOP:
        return ("DIAGNOSE", "(commit MAP)")
    best = max(eig_estimates.items(),
               key=lambda kv: kv[1] / max(cost_of(parse_action_key(kv[0])[0],
                                                  override_test_cost=25.0), 0.1))
    return parse_action_key(best[0])


def select_R3(eig_estimates: dict[str, float]) -> tuple[str, str]:
    """Hybrid: pick max raw EIG when it exceeds 0.15; else EIG/cost."""
    if not eig_estimates:
        return ("DIAGNOSE", "(empty)")
    max_eig = max(eig_estimates.values())
    if max_eig < EPS_STOP:
        return ("DIAGNOSE", "(commit MAP)")
    if max_eig > HYBRID_RAW_EIG_FLOOR:
        best = max(eig_estimates.items(), key=lambda kv: kv[1])
    else:
        best = max(eig_estimates.items(),
                   key=lambda kv: kv[1] / max(cost_of(parse_action_key(kv[0])[0]), 0.1))
    return parse_action_key(best[0])


RULES = {"R1 (eps_stop 0.025)": select_R1,
         "R2 (test cost /2)":   select_R2,
         "R3 (hybrid raw EIG)": select_R3}


def analyze(jsonl_path: Path) -> None:
    rows = [json.loads(l) for l in jsonl_path.read_text().splitlines() if l.strip()]
    print(f"=== Counterfactual analysis on {len(rows)} trajectories ===\n")

    overall_diffs: dict[str, Counter[str]] = {name: Counter() for name in RULES}
    per_case_first_divergence: dict[str, list[dict]] = {name: [] for name in RULES}

    for r in rows:
        case_id = r["case_id"]
        correct = r["diagnostic_correct"]
        steps = r["steps"]

        for rule_name, fn in RULES.items():
            for s in steps:
                eig = s.get("eig_estimates") or {}
                actual = s["action"]["type"]
                pred_type, pred_q = fn(eig)
                if pred_type == actual:
                    continue
                overall_diffs[rule_name][f"{actual}->{pred_type}"] += 1
                per_case_first_divergence[rule_name].append({
                    "case": case_id, "correct": correct,
                    "step": s["step_idx"], "actual": actual,
                    "would_pick": pred_type,
                    "would_query": pred_q[:100],
                })
                break  # only first divergence per (case, rule)

    # 1. Action-flip count tables.
    for rule_name, counts in overall_diffs.items():
        print(f"\n--- {rule_name} ---")
        if not counts:
            print("  No divergences across any step in any trajectory.")
            continue
        print(f"{'flip':<30} {'count':>5}")
        for k, v in counts.most_common():
            print(f"  {k:<28} {v:>5}")

    # 2. First-divergence-by-case under each rule.
    print(f"\n\n=== First-divergence-step per (rule, case) ===")
    for rule_name, divs in per_case_first_divergence.items():
        print(f"\n--- {rule_name} ---")
        if not divs:
            print("  No case shows any divergence; rule is identical to production on this dataset.")
            continue
        print(f"{'case':<18} {'corr':<5} {'step':>4} {'actual':<15} {'->':<2} {'would_pick':<15}")
        for d in divs:
            mark = "✅" if d["correct"] else "❌"
            print(f"  {d['case'][:16]:<18} {mark:<5} {d['step']:>4} {d['actual']:<15} -> {d['would_pick']:<15}")

    # 3. Wrong-case-flip score: count wrong cases where the rule first diverges.
    print(f"\n\n=== Could the rule change have flipped a wrong case? ===")
    wrong_cases = {r["case_id"] for r in rows if not r["diagnostic_correct"]}
    print(f"Wrong cases (n={len(wrong_cases)}): {sorted(wrong_cases)[:5]}...")
    for rule_name, divs in per_case_first_divergence.items():
        wrong_diverged = {d["case"] for d in divs if not d["correct"]}
        order_test_pulls = sum(1 for d in divs
                               if not d["correct"] and d["would_pick"] == "ORDER_TEST")
        print(f"\n--- {rule_name} ---")
        print(f"  Wrong cases where rule diverges at some step: "
              f"{len(wrong_diverged)}/{len(wrong_cases)}")
        print(f"  Of those, divergence -> ORDER_TEST: {order_test_pulls}")


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "results/sanity/trajectories/raw_medqa__benign.jsonl"
    analyze(Path(path))
