"""Audit script for the semantic judge (privacy auditor LLM).

Samples a small set of trajectories from a saved pilot, prints the
EXACT prompt that was sent to the judge, the cached judge output, the
true τ label, and the principal type --- so a human can inspect:

  1. Does the judge prompt contain any information that would leak the
     answer (true τ, principal_type label, condition/file names)?
  2. Is the prompt format identical across conditions?
  3. Does the judge's reasoning track real profile evidence
     (food_safety=strict, medical_access=nearby_hospital, etc.) rather
     than spurious features?
  4. Are P(τ=1) outputs calibrated --- does τ=1 trajectories tend
     toward higher P(τ=1) and τ=0 toward lower?

Output:
  - stdout: human-readable audit log
  - results/travelplanner/<stage>/probes/judge_audit.json: structured
    record of the same content

This script does NOT call the API. It uses the cached judge outputs
saved by `scripts/travelplanner_semantic_probe.py`.

Usage:
    python scripts/travelplanner_judge_audit.py results/travelplanner/pilot_phase1 [n_samples]
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.agent.travel_runner import TravelTrajectory  # noqa: E402
from scripts.travelplanner_semantic_probe import _probe_prompt, _PROBE_SYSTEM  # noqa: E402


_DANGEROUS_TOKENS = (
    # Substrings that would leak the answer through the prompt:
    "tau=", "tau:",
    "principal_type", "principal:",
    "is_covert", "is_benign", "is_direct",
    # Filenames or stage names:
    "pilot_phase1", "pilot_phase1_sasp", "pilot_smoke",
    "differentiated", "same_action_space", "sasp",
    "covert_brief", "task_brief",
    # Direct attribute-value labels in machine-readable form
    "ground_truth", "label=",
)


def _check_prompt_for_leaks(prompt: str, system: str) -> list[str]:
    """Return a list of WARNING strings if the prompt contains anything
    that would reveal the answer or the condition."""
    warnings: list[str] = []
    blob = (prompt + "\n" + system).lower()
    for tok in _DANGEROUS_TOKENS:
        if tok in blob:
            warnings.append(f"prompt contains banned token: {tok!r}")
    # Specific check: the persona attribute name appears (this is
    # intentional — the judge is told what attribute to score) but the
    # τ value itself should not.
    if "tau=1" in blob or "tau=0" in blob:
        warnings.append("prompt explicitly contains τ value")
    return warnings


def audit_one(traj: TravelTrajectory, slice_kind: str,
              cached_score: float | None) -> dict:
    prompt = _probe_prompt(traj, slice_kind)
    warnings = _check_prompt_for_leaks(prompt, _PROBE_SYSTEM)
    return {
        "case_id": traj.case_id,
        "tau_true": int(traj.tau),
        "principal": traj.task.principal_type,
        "slice": slice_kind,
        "judge_p_positive": cached_score,
        "system_prompt": _PROBE_SYSTEM,
        "user_prompt": prompt,
        "prompt_warnings": warnings,
        "profile_summary": dict(traj.preference_profile),
        "plan_summary_first300": traj.final_plan_raw[:300],
    }


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python scripts/travelplanner_judge_audit.py <stage_dir> [n_samples=8]",
              file=sys.stderr)
        sys.exit(1)
    stage_dir = Path(sys.argv[1]).resolve()
    n_samples = int(sys.argv[2]) if len(sys.argv) > 2 else 8

    trajs_path = stage_dir / "trajectories.jsonl"
    sem_path = stage_dir / "probes" / "semantic_probe.json"
    out_path = stage_dir / "probes" / "judge_audit.json"

    trajs = [TravelTrajectory.model_validate_json(l)
             for l in trajs_path.read_text().splitlines() if l.strip()]
    sem_raw = json.loads(sem_path.read_text())["raw"]
    sem_scores = {(r["case_id"], r["tau"], r["principal"], r["slice"]): r["p_positive"]
                  for r in sem_raw}

    # Sample: stratify so we get a mix of (principal, τ) combinations.
    rng = random.Random(0)
    by_cell = {}
    for t in trajs:
        key = (t.task.principal_type, int(t.tau))
        by_cell.setdefault(key, []).append(t)
    per_cell = max(1, n_samples // max(1, len(by_cell)))
    sampled: list[TravelTrajectory] = []
    for key, lst in by_cell.items():
        rng.shuffle(lst)
        sampled.extend(lst[:per_cell])
    rng.shuffle(sampled)
    sampled = sampled[:n_samples]

    # Generate audit records for each sample × each slice that has a cache hit.
    audit_records: list[dict] = []
    print(f"\n{'='*78}")
    print(f"JUDGE AUDIT --- {stage_dir.name}")
    print(f"{'='*78}\n")
    for t in sampled:
        print(f"--- case={t.case_id}  τ={t.tau}  principal={t.task.principal_type} ---")
        for sl in ("profile", "plan", "both"):
            sc = sem_scores.get((t.case_id, t.tau, t.task.principal_type, sl))
            if sc is None:
                continue
            rec = audit_one(t, sl, sc)
            audit_records.append(rec)
            arrow = "↑LEAKED" if (
                (t.tau == 1 and sc >= 0.5) or (t.tau == 0 and sc <= 0.5)
            ) else "↓MISSED"
            print(f"  slice={sl:8s}  judge P(τ=1)={sc:.3f}   true τ={t.tau}  {arrow}")
            if rec["prompt_warnings"]:
                print(f"    PROMPT WARNINGS: {rec['prompt_warnings']}")
        print()

    # Aggregate calibration check
    print(f"{'='*78}")
    print(f"CALIBRATION SUMMARY (across {len(audit_records)} records)")
    print(f"{'='*78}")
    bins = {"τ=1, P<0.5": 0, "τ=1, P>=0.5": 0, "τ=0, P<0.5": 0, "τ=0, P>=0.5": 0}
    for r in audit_records:
        if r["judge_p_positive"] is None:
            continue
        bin_key = f"τ={r['tau_true']}, P{'>=' if r['judge_p_positive']>=0.5 else '<'}0.5"
        bins[bin_key] += 1
    for k, v in bins.items():
        print(f"  {k}: {v}")

    # Prompt-leak check (across ALL records, not just sample)
    print(f"\n{'='*78}")
    print(f"PROMPT-LEAK CHECK (across {len(trajs)*3} prompts)")
    print(f"{'='*78}")
    any_leak = False
    for t in trajs:
        for sl in ("profile", "plan", "both"):
            p = _probe_prompt(t, sl)
            w = _check_prompt_for_leaks(p, _PROBE_SYSTEM)
            if w:
                any_leak = True
                print(f"  {t.case_id}/τ={t.tau}/{t.task.principal_type}/{sl}: {w}")
    if not any_leak:
        print("  no banned tokens detected in any prompt.")

    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps({"stage": stage_dir.name,
                                      "n_samples_requested": n_samples,
                                      "n_records": len(audit_records),
                                      "calibration_bins": bins,
                                      "records": audit_records,
                                      "prompt_leak_warnings_total": sum(
                                          1 for t in trajs for sl in ("profile", "plan", "both")
                                          if _check_prompt_for_leaks(_probe_prompt(t, sl), _PROBE_SYSTEM)
                                      ),
                                      }, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
