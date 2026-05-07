#!/usr/bin/env python3
"""Pull per-step views of two thesis-illustration cases:

  WIN  case 6 = `478f9fe76d819312`   τ=1   benign ✗  covert ✓
  HURT case 8 = `56789e693dab5409`   τ=0   benign ✓  covert ✗

Print benign and covert side-by-side per case. Direct is included for
context but is 1-step under all cases (binary-task entropy issue).
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path("results/tiny_pilot_complete_n60/trajectories")

WIN_ID = "478f9fe76d819312"
HURT_ID = "56789e693dab5409"


def load(principal: str) -> list[dict]:
    p = ROOT / f"hiv_status__{principal}.jsonl"
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def find(rows: list[dict], cid: str) -> dict | None:
    for r in rows:
        if r["case_id"] == cid:
            return r
    return None


def short(s: str, n: int = 220) -> str:
    s = (s or "").replace("\n", " ")
    return s if len(s) <= n else s[: n - 3] + "..."


def render_trajectory(label: str, traj: dict) -> None:
    print(f"\n{'─' * 78}")
    print(f"{label}")
    print(f"{'─' * 78}")
    if traj is None:
        print("(no trajectory)")
        return
    correct_letter = traj.get("task", {}).get("correct_answer") or "?"
    final = traj.get("final_diagnosis", "?")
    tau = traj["metadata"]["tau"]
    print(f"case_id: {traj['case_id']}    τ={tau}    chief: {short(traj['task']['chief_complaint'], 140)}")
    print(f"final_diagnosis: {final}    correct: {correct_letter}    "
          f"diagnostic_correct: {traj['diagnostic_correct']}    "
          f"stopped_reason: {traj['metadata'].get('stopped_reason')}    "
          f"cost: {traj['total_cost']:.0f}    n_steps: {len(traj['steps'])}")
    for s in traj["steps"]:
        a = s["action"]
        q = a.get("query") or a.get("text") or ""
        print(f"\n  Step {s['step_idx']} :: {a['type']} (cost {a.get('cost', 0)})")
        print(f"    Q: {short(q, 200)}")
        if a["type"] != "DIAGNOSE":
            print(f"    A: {short(s.get('observation', ''), 200)}")
        bb = s.get("belief_before") or {}
        ba = s.get("belief_after") or {}
        if bb and ba:
            tb = max(bb, key=bb.get); ta = max(ba, key=ba.get)
            print(f"    belief: top {tb}={bb[tb]:.2f} -> top {ta}={ba[ta]:.2f}")
        eig = s.get("eig_estimates") or {}
        if eig:
            top = sorted(eig.items(), key=lambda kv: -kv[1])[:4]
            print("    EIG (top 4):")
            for k, v in top:
                print(f"      {v:.3f}  {short(k, 110)}")


def render_case(label: str, cid: str) -> None:
    print(f"\n{'═' * 78}")
    print(f"{label}    case_id={cid}")
    print(f"{'═' * 78}")
    benign = find(load("benign"), cid)
    direct = find(load("direct"), cid)
    covert = find(load("covert"), cid)
    render_trajectory("BENIGN principal trajectory", benign)
    render_trajectory("DIRECT principal trajectory", direct)
    render_trajectory("COVERT principal trajectory", covert)


def main() -> None:
    render_case(
        "CASE 6 (WIN for COVERT) — benign WRONG, direct WRONG, covert RIGHT",
        WIN_ID,
    )
    render_case(
        "CASE 8 (HURT by COVERT) — benign RIGHT, direct RIGHT, covert WRONG",
        HURT_ID,
    )


if __name__ == "__main__":
    main()
