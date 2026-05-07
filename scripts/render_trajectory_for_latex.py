#!/usr/bin/env python3
"""Render trajectory excerpts from the n=60 tiny_pilot dataset as LaTeX-safe
verbatim blocks. Used by progress_report.tex case studies.

Usage:
    python3 scripts/render_trajectory_for_latex.py [case_id] [principal] [step_filter]

  case_id      : 478f9fe76d819312 (case 6) or 56789e693dab5409 (case 8)
  principal    : benign | direct | covert
  step_filter  : "all" or comma-separated step indices like "1,5,10,15"
"""
from __future__ import annotations

import json
import sys
import textwrap
from pathlib import Path

ROOT = Path("results/tiny_pilot_complete_n60/trajectories")


def load(principal: str) -> list[dict]:
    p = ROOT / f"hiv_status__{principal}.jsonl"
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def find(rows: list[dict], cid: str) -> dict | None:
    for r in rows:
        if r["case_id"] == cid:
            return r
    return None


def short(s: str, n: int = 100) -> str:
    s = (s or "").replace("\n", " ").strip()
    return s if len(s) <= n else s[: n - 3] + "..."


def render(case_id: str, principal: str, step_filter: str = "all") -> str:
    traj = find(load(principal), case_id)
    if traj is None:
        return f"(no trajectory for {case_id} under {principal})\n"
    out: list[str] = []
    chief = traj["task"]["chief_complaint"]
    final = traj["final_diagnosis"]
    tau = traj["metadata"]["tau"]
    sr = traj["metadata"].get("stopped_reason")
    correct = traj["diagnostic_correct"]
    out.append(
        f"Case {case_id} | principal={principal} | tau={tau} | "
        f"final={final} | stopped={sr} | correct={correct} | "
        f"cost=${traj['total_cost']:.0f} | n_steps={len(traj['steps'])}"
    )
    out.append("")
    out.append("Chief complaint:")
    for line in textwrap.wrap(chief, width=80):
        out.append(f"  {line}")
    out.append("")
    if step_filter == "all":
        steps = traj["steps"]
    else:
        keep = set(int(x) for x in step_filter.split(","))
        steps = [s for s in traj["steps"] if s["step_idx"] in keep]
    for s in steps:
        a = s["action"]
        q = a.get("query") or a.get("text") or ""
        out.append(f"STEP {s['step_idx']}")
        out.append(f"  Action : {a['type']} (cost ${a.get('cost', 0):.0f})")
        for line in textwrap.wrap(q, width=72, initial_indent="  Q      : ",
                                  subsequent_indent="           "):
            out.append(line)
        if a["type"] != "DIAGNOSE":
            obs = short(s.get("observation", ""), 100)
            for line in textwrap.wrap(obs, width=72, initial_indent="  Patient: ",
                                      subsequent_indent="           "):
                out.append(line)
        bb = s.get("belief_before") or {}
        ba = s.get("belief_after") or {}
        if bb and ba:
            tb = max(bb, key=bb.get); ta = max(ba, key=ba.get)
            out.append(f"  Belief : top {tb}={bb[tb]:.2f} -> top {ta}={ba[ta]:.2f}")
        eig = s.get("eig_estimates") or {}
        if eig:
            top_eig = max(eig.items(), key=lambda kv: kv[1])
            ek, ev = top_eig
            for line in textwrap.wrap(
                f"top EIG candidate: {ev:.3f}  {ek}",
                width=72, initial_indent="  EIG    : ",
                subsequent_indent="           ",
            ):
                out.append(line)
        out.append("")
    out.append(f"Final diagnosis: {final}  (correct={correct}, tau={tau})")
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        # Default: print the three blocks needed by progress_report.tex
        print("=" * 78)
        print("CASE 6 covert (full)")
        print("=" * 78)
        print(render("478f9fe76d819312", "covert", "all"))
        print("=" * 78)
        print("CASE 6 benign (full, for comparison)")
        print("=" * 78)
        print(render("478f9fe76d819312", "benign", "all"))
        print("=" * 78)
        print("CASE 8 covert (steps 1, 5, 10, 15)")
        print("=" * 78)
        print(render("56789e693dab5409", "covert", "1,5,10,15"))
    elif len(args) >= 2:
        sf = args[2] if len(args) >= 3 else "all"
        print(render(args[0], args[1], sf))
