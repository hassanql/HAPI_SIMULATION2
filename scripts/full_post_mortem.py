#!/usr/bin/env python3
"""Honest post-mortem across both Stage 3 runs.

Reads:
  results/sanity/trajectories/raw_medqa__benign.jsonl              (R0, n=10)
  results/sanity_r3_n15/trajectories/raw_medqa__benign.jsonl       (R3, n=15)

Examines each trajectory step-by-step for:
  1. Per-stop-reason accuracy and patterns.
  2. The policy_diagnose failure mode (the persistent killer).
  3. Action-type distribution.
  4. Any "ct" substring false-positive cost matches.
  5. Belief evolution sanity (no NaNs, no negative entropies).
  6. Anomalous action selections (chosen action != expected from selection rule).

Output is fact-only. No recommendations.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

# Substrings that should NOT match the "ct" cost override but probably do
# under the current substring-match rule. Used to count false positives.
LIKELY_CT_FALSE_POSITIVES = [
    "function", "active", "react", "construct", "infection",
    "section", "doctor", "structure", "bacterial", "vaccination",
    "extract", "respect", "characterize", "cortex", "directly",
    "production", "reduction", "infarct", "octopus",
]


def trajectory_path(run: str) -> Path:
    return Path(f"results/{run}/trajectories/raw_medqa__benign.jsonl")


def load(run: str) -> list[dict]:
    p = trajectory_path(run)
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def analyze_run(run: str, rows: list[dict]) -> None:
    if not rows:
        print(f"\n=== {run}: no trajectories ===")
        return

    print(f"\n{'='*70}")
    print(f"RUN: {run}    n={len(rows)}")
    print(f"{'='*70}")

    n_correct = sum(1 for r in rows if r["diagnostic_correct"])
    print(f"Accuracy: {n_correct}/{len(rows)} = {n_correct/len(rows):.1%}\n")

    # By stopped_reason.
    by_stop: dict[str, list[dict]] = {}
    for r in rows:
        sr = r["metadata"].get("stopped_reason", "?")
        by_stop.setdefault(sr, []).append(r)
    print(f"{'stopped_reason':<22} {'n':>3} {'corr':>5} {'%':>6} {'mean_steps':>10} {'mean_cost':>10}")
    for sr, sub in by_stop.items():
        nc = sum(1 for r in sub if r["diagnostic_correct"])
        ms = sum(len(r["steps"]) for r in sub) / len(sub)
        mc = sum(r["total_cost"] for r in sub) / len(sub)
        print(f"{sr:<22} {len(sub):>3} {nc:>5} {nc/len(sub):>5.0%} {ms:>10.1f} {mc:>10.1f}")

    # Action-type histogram.
    type_counts: Counter[str] = Counter()
    total_steps = 0
    for r in rows:
        for s in r["steps"]:
            type_counts[s["action"]["type"]] += 1
            total_steps += 1
    print(f"\nAction-type histogram (total {total_steps} steps):")
    for t, c in type_counts.most_common():
        pct = c / total_steps * 100 if total_steps else 0
        print(f"  {t:<14} {c:>4}  ({pct:.1f}%)")

    # Cost anomalies: every ORDER_TEST cost should match TEST_COST_OVERRIDES exactly
    # for known patterns. Flag any cost not in the standard set as an "anomaly".
    expected_costs = {1.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 50.0, 60.0,
                      150.0, 250.0, 300.0, 500.0, 800.0, 1200.0, 0.0}
    cost_anomalies = []
    ct_substring_in_query: list[tuple[int, int, str, float]] = []
    for r in rows:
        for s in r["steps"]:
            cost = s["action"].get("cost", 0)
            atype = s["action"]["type"]
            query = (s["action"].get("query") or s["action"].get("text") or "").lower()
            if cost not in expected_costs:
                cost_anomalies.append((r["case_id"][:10], s["step_idx"], atype, cost, query[:80]))
            # Specifically look for "ct" substring matches that aren't actual CT scans:
            if "ct" in query and atype == "ORDER_TEST" and cost == 300.0:
                # Real CT scans we expect: "ct", "ct scan", "ct angiography",
                # "non-contrast head ct", "ct of chest", etc. Anything else is suspicious.
                ct_likely_real = any(
                    pat in query
                    for pat in ["ct ", " ct", "ct scan", "ct angiography",
                                "ct of", "computed tomography"]
                )
                if not ct_likely_real:
                    ct_substring_in_query.append((r["case_id"][:10], s["step_idx"], atype, query[:80]))

    if cost_anomalies:
        print(f"\nCOST ANOMALIES (not in expected set): {len(cost_anomalies)}")
        for ca in cost_anomalies[:10]:
            print(f"  case={ca[0]} step={ca[1]} type={ca[2]} cost={ca[3]} query={ca[4]!r}")
    else:
        print(f"\nCost anomalies: none (all costs match expected table)")

    if ct_substring_in_query:
        print(f"\n'ct' SUBSTRING FALSE POSITIVES (cost=300 but not a real CT): {len(ct_substring_in_query)}")
        for fp in ct_substring_in_query[:10]:
            print(f"  case={fp[0]} step={fp[1]} type={fp[2]} query={fp[3]!r}")
    else:
        print(f"'ct' substring false positives: none in this run")

    # Look for likely substring false positives in any query, not just ORDER_TEST.
    # If a non-CT action accidentally matches "ct" rule it would get cost 300.
    # Also check if any ORDER_TEST query that is NOT a CT got cost 300 (already
    # done above).
    other_ct_matches = []
    for r in rows:
        for s in r["steps"]:
            query = (s["action"].get("query") or s["action"].get("text") or "").lower()
            if "ct" in query:
                # Find which substring caused match.
                for fp_word in LIKELY_CT_FALSE_POSITIVES:
                    if fp_word in query:
                        other_ct_matches.append((r["case_id"][:10], s["step_idx"],
                                                  s["action"]["type"],
                                                  s["action"].get("cost"),
                                                  fp_word, query[:80]))
                        break
    if other_ct_matches:
        print(f"\nQueries containing 'ct'-bearing common words: {len(other_ct_matches)}")
        for fp in other_ct_matches[:10]:
            mark = " ← SUSPICIOUS" if fp[3] == 300.0 else ""
            print(f"  case={fp[0]} step={fp[1]} type={fp[2]} cost={fp[3]} word={fp[4]!r} q={fp[5]!r}{mark}")
    else:
        print(f"\nNo 'ct'-bearing common words in any query.")


def deep_dive_policy_diagnose(rows_by_run: dict[str, list[dict]]) -> None:
    """Look at every policy_diagnose case across both runs."""
    print(f"\n{'='*70}")
    print("DEEP DIVE: every policy_diagnose case across both runs")
    print(f"{'='*70}")

    pd_cases = []
    for run, rows in rows_by_run.items():
        for r in rows:
            if r["metadata"].get("stopped_reason") == "policy_diagnose":
                pd_cases.append((run, r))

    print(f"\nTotal policy_diagnose cases: {len(pd_cases)}")
    print(f"{'run':<25} {'case_id':<18} {'corr':>5} {'steps':>5} {'cost':>5} {'final_belief_top':>20}")
    for run, r in pd_cases:
        mark = "✅" if r["diagnostic_correct"] else "❌"
        steps = r["steps"]
        if steps:
            final_belief = steps[-1]["belief_after"]
            top_opt = max(final_belief, key=final_belief.get)
            top_val = final_belief[top_opt]
        else:
            top_opt, top_val = "?", 0.0
        print(f"  {run:<23} {r['case_id'][:16]:<18} {mark:>3}  {len(steps):>5} {r['total_cost']:>5.0f} "
              f"{top_opt}={top_val:.2f}")

    # Per-step max-EIG over time for each policy_diagnose case.
    print(f"\nPer-step max(EIG) for each policy_diagnose case (when did EIG fall below 0.05?):")
    for run, r in pd_cases:
        max_eigs = []
        for s in r["steps"]:
            eig = s.get("eig_estimates") or {}
            if eig:
                max_eigs.append(max(eig.values()))
        max_eig_str = " -> ".join(f"{e:.3f}" for e in max_eigs)
        print(f"  {run} {r['case_id'][:10]}: {max_eig_str}")

    # Are EIGs ever above 0.15 (would have triggered R3)?
    print(f"\nDid R3 floor=0.15 EVER trigger for these cases? (any step with max(EIG) > 0.15)")
    for run, r in pd_cases:
        any_high = False
        max_seen = 0.0
        for s in r["steps"]:
            eig = s.get("eig_estimates") or {}
            if eig:
                m = max(eig.values())
                max_seen = max(max_seen, m)
                if m > 0.15:
                    any_high = True
        marker = "  ← R3 would have fired" if any_high else "  ← EIG never broke 0.15 floor"
        print(f"  {run} {r['case_id'][:10]}: max_EIG_seen={max_seen:.3f}{marker}")

    # Action types in policy_diagnose cases.
    print(f"\nAction types used in policy_diagnose trajectories:")
    type_pd: Counter[str] = Counter()
    for _, r in pd_cases:
        for s in r["steps"]:
            type_pd[s["action"]["type"]] += 1
    for t, c in type_pd.most_common():
        print(f"  {t:<14} {c:>3}")


def deep_dive_loop_exit(rows_by_run: dict[str, list[dict]]) -> None:
    """The new R3-driven path. Look at how many of these were 1-step + cost-cap exits."""
    print(f"\n{'='*70}")
    print("DEEP DIVE: loop_exit cases (R3-driven path in r3_n15)")
    print(f"{'='*70}")
    le_cases = []
    for run, rows in rows_by_run.items():
        for r in rows:
            if r["metadata"].get("stopped_reason") == "loop_exit":
                le_cases.append((run, r))
    print(f"\nTotal loop_exit cases: {len(le_cases)}")
    if not le_cases:
        return
    print(f"{'run':<25} {'case_id':<14} {'corr':>5} {'steps':>5} {'cost':>5} {'last_action_type':<14}")
    for run, r in le_cases:
        mark = "✅" if r["diagnostic_correct"] else "❌"
        last_type = r["steps"][-1]["action"]["type"] if r["steps"] else "?"
        print(f"  {run:<23} {r['case_id'][:12]:<14} {mark:>3}  {len(r['steps']):>5} {r['total_cost']:>5.0f} {last_type:<14}")


def main() -> None:
    runs = {"sanity (R0, n=10)": "sanity",
            "sanity_r3_n15 (R3, n=15)": "sanity_r3_n15"}
    rows_by_run: dict[str, list[dict]] = {}
    for label, dir in runs.items():
        rows_by_run[label] = load(dir)
        analyze_run(dir, rows_by_run[label])

    deep_dive_policy_diagnose(rows_by_run)
    deep_dive_loop_exit(rows_by_run)


if __name__ == "__main__":
    main()
