#!/usr/bin/env python3
"""Inspect trajectories produced by a stage run.

Usage (from repo root):
    python3 scripts/inspect_trajectories.py
    python3 scripts/inspect_trajectories.py summary
    python3 scripts/inspect_trajectories.py latest
    python3 scripts/inspect_trajectories.py case 3
    python3 scripts/inspect_trajectories.py state

    --stage sanity       (default)
    --principal benign   (default)
    --attribute raw_medqa  (default)

Paths and filenames match `StageStorage.trajectory_jsonl_path` so any
stage that writes trajectories with the canonical layout is inspectable.
"""
from __future__ import annotations

import argparse
import json
import sys
import textwrap
from pathlib import Path


def _trajectory_path(stage: str, attribute: str, principal: str) -> Path:
    return Path("results") / stage / "trajectories" / f"{attribute}__{principal}.jsonl"


def _load(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def _short(s: str, n: int = 240) -> str:
    s = (s or "").replace("\n", " ")
    return textwrap.shorten(s, n, placeholder="...")


def _print_step(s: dict) -> None:
    a = s["action"]
    q = a.get("text") or a.get("query") or ""
    print(f"--- Step {s['step_idx']} :: {a['type']} (cost {a.get('cost', 0)}) ---")
    print(f"  Q: {_short(q)}")
    print(f"  A: {_short(s.get('observation', ''))}")
    bb = s.get("belief_before", {})
    ba = s.get("belief_after", {})
    if bb and ba:
        tb = max(bb, key=bb.get)
        ta = max(ba, key=ba.get)
        print(f"  belief: top {tb}={bb[tb]:.2f} -> top {ta}={ba[ta]:.2f}")
    eig = s.get("eig_estimates") or {}
    if eig:
        top_eig = sorted(eig.items(), key=lambda kv: -kv[1])[:4]
        print(f"  EIG: {[(k, round(v, 3)) for k, v in top_eig]}")
    print()


def cmd_summary(rows: list[dict]) -> None:
    if not rows:
        print("No trajectories yet.")
        return
    n = len(rows)
    c = sum(1 for r in rows if r["diagnostic_correct"])
    print(f"=== {c}/{n} = {c/n:.0%} correct ===")
    by_reason: dict[str, list[dict]] = {}
    for r in rows:
        sr = r["metadata"].get("stopped_reason", "?")
        by_reason.setdefault(sr, []).append(r)
    print()
    print(f"{'reason':<22} {'n':>3} {'corr':>5} {'%':>6} {'mean_steps':>11} {'mean_cost':>10}")
    print("-" * 60)
    for sr, sub in by_reason.items():
        nc = sum(1 for r in sub if r["diagnostic_correct"])
        ms = sum(len(r["steps"]) for r in sub) / len(sub)
        mc = sum(r["total_cost"] for r in sub) / len(sub)
        print(f"{sr:<22} {len(sub):>3} {nc:>5} {nc/len(sub):>5.0%} {ms:>11.1f} {mc:>10.1f}")
    print()
    print(f"{'#':>3} {'':<3} {'steps':>5} {'cost':>5} {'stopped':<22} case_id")
    for i, r in enumerate(rows, start=1):
        mark = "✅" if r["diagnostic_correct"] else "❌"
        sr = r["metadata"].get("stopped_reason", "?")
        print(f"{i:>3} {mark:<3} {len(r['steps']):>5} {r['total_cost']:>5.0f} {sr:<22} {r['case_id'][:16]}")


def cmd_latest(rows: list[dict]) -> None:
    if not rows:
        print("No trajectories yet.")
        return
    cmd_case(rows, len(rows))


def cmd_case(rows: list[dict], idx: int) -> None:
    if not rows:
        print("No trajectories yet.")
        return
    if idx < 1 or idx > len(rows):
        print(f"Trajectory index {idx} out of range (have {len(rows)}).")
        return
    t = rows[idx - 1]
    correct_letter = t.get("task", {}).get("correct_answer", "?")
    print(f"=== Trajectory #{idx}: case {t['case_id']} ===")
    print(f"final_diagnosis:    {t.get('final_diagnosis', '?')}   (correct: {correct_letter})")
    print(f"diagnostic_correct: {t['diagnostic_correct']}")
    print(f"stopped_reason:     {t['metadata'].get('stopped_reason')}")
    print(f"total_cost:         {t['total_cost']}")
    print(f"n_steps:            {len(t['steps'])}")
    print()
    for s in t["steps"]:
        _print_step(s)


def cmd_state(stage: str, attribute: str, principal: str) -> None:
    """Print where the run is right now (for when nothing prints)."""
    import subprocess

    traj_path = _trajectory_path(stage, attribute, principal)
    log_path = Path("results") / stage / "log.jsonl"
    print(f"=== Trajectory JSONL: {traj_path} ===")
    if traj_path.exists():
        size = traj_path.stat().st_size
        n_lines = len(traj_path.read_text().splitlines())
        print(f"  exists: {size} bytes, {n_lines} trajectories")
    else:
        print("  does not exist yet")
    print()
    print(f"=== Latest 5 events from {log_path} ===")
    if log_path.exists():
        lines = log_path.read_text().splitlines()
        for l in lines[-5:]:
            print(f"  {l}")
    else:
        print("  log.jsonl does not exist")
    print()
    print("=== Processes ===")
    try:
        out = subprocess.run(
            ["ps", "-ef"], capture_output=True, text=True, check=True
        ).stdout
        for l in out.splitlines():
            if "run.py" in l or "vllm serve" in l:
                if "grep" not in l:
                    print(f"  {l}")
    except Exception as e:
        print(f"  ps failed: {e}")
    print()
    print("=== GPU ===")
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.free,utilization.gpu",
             "--format=csv"],
            capture_output=True, text=True, check=True,
        ).stdout
        print("  " + out.replace("\n", "\n  ").rstrip())
    except Exception as e:
        print(f"  nvidia-smi failed: {e}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("subcmd", nargs="?", default="summary",
                   choices=["summary", "latest", "case", "state"])
    p.add_argument("idx", nargs="?", type=int, default=None,
                   help="case number for `case` subcommand (1-indexed)")
    p.add_argument("--stage", default="sanity")
    p.add_argument("--attribute", default="raw_medqa")
    p.add_argument("--principal", default="benign")
    p.add_argument("--path", default=None,
                   help="Override the trajectory JSONL path "
                        "(useful for archived files like .run2_partial.jsonl).")
    args = p.parse_args()

    if args.subcmd == "state":
        cmd_state(args.stage, args.attribute, args.principal)
        return

    path = Path(args.path) if args.path else _trajectory_path(
        args.stage, args.attribute, args.principal
    )
    rows = _load(path)

    if args.subcmd == "summary":
        cmd_summary(rows)
    elif args.subcmd == "latest":
        cmd_latest(rows)
    elif args.subcmd == "case":
        if args.idx is None:
            print("Provide a case index, e.g. `case 3`")
            sys.exit(2)
        cmd_case(rows, args.idx)


if __name__ == "__main__":
    main()
