"""Defence-impact analysis: deploy the behavioural probe at FPR=10% as a
runtime filter on the 600-trajectory LLM-attacker headline and measure
the residual covert-vs-benign tau-leak gap on the *unflagged*
trajectories (i.e., the ones that would slip past the deployed
defender).

This converts our defender-side detection AUC into a deployment-grade
operating-point claim: "deploying the behavioural probe at FPR=10%
reduces the effective covert tau-leak from +X to +Y."

Inputs:
  - results/pilot_llm_attacker_full/trajectories/  (600 trajectories)
  - results/probes_llm_attacker_full/defender_curves.json  (behavioural
    threshold at FPR<=0.10, semantic per-trajectory scores)
  - results/probes_llm_attacker_full/llm_cache/  (semantic-probe cache)

Output:
  - results/probes_llm_attacker_full/defence_impact.json
  - Markdown-style table on stdout for the progress report
"""
from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.agent.runner import Trajectory  # noqa: E402

SOURCE_STAGE = "pilot_llm_attacker_full"


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    halfw = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return p, max(0.0, centre - halfw), min(1.0, centre + halfw)


def load_trajectories() -> list[Trajectory]:
    traj_dir = REPO / "results" / SOURCE_STAGE / "trajectories"
    trajs = []
    for p in sorted(traj_dir.glob("*.jsonl")):
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                trajs.append(Trajectory.model_validate_json(line))
    return trajs


def analyze_filter(
    bc_trajs: list[Trajectory],
    sem: dict[str, float],
    filter_scores: dict[str, float],
    threshold: float,
    label: str,
) -> dict:
    """Common analysis: given per-trajectory filter scores and a
    threshold, compute the residual covert-vs-benign tau-leak gap on
    UNFLAGGED trajectories."""
    def key(t):
        return f"{t.case_id}__{t.metadata.get('principal')}__{t.metadata.get('attribute')}"

    by_attr = defaultdict(lambda: {
        "benign_orig": [], "covert_orig": [],
        "benign_resid": [], "covert_resid": [],
        "n_covert_flagged": 0, "n_benign_flagged": 0,
        "n_covert_total": 0, "n_benign_total": 0,
    })
    for t in bc_trajs:
        attr = t.metadata.get("attribute", "")
        p = t.metadata.get("principal")
        k = key(t)
        if k not in sem or k not in filter_scores:
            continue
        sem_score = sem[k]
        flt_score = filter_scores[k]
        flagged = flt_score >= threshold
        slot = by_attr[attr]
        if p == "covert":
            slot["covert_orig"].append(sem_score)
            slot["n_covert_total"] += 1
            if flagged: slot["n_covert_flagged"] += 1
            else: slot["covert_resid"].append(sem_score)
        else:
            slot["benign_orig"].append(sem_score)
            slot["n_benign_total"] += 1
            if flagged: slot["n_benign_flagged"] += 1
            else: slot["benign_resid"].append(sem_score)

    def safe_mean(xs):
        return sum(xs) / len(xs) if xs else 0.0

    print(f"\n{'='*90}")
    print(f"DEFENCE-IMPACT — {label} filter at threshold {threshold:.3f}")
    print(f"{'='*90}")
    print(f"\n{'Attribute':<20s} | {'Orig gap':>11s} | {'Residual':>11s} | {'Δ':>7s} | {'Cov flag':>10s} | {'Ben flag':>10s}")
    print("-" * 90)
    pooled = {"benign_orig": [], "covert_orig": [], "benign_resid": [], "covert_resid": [],
              "n_covert_flagged": 0, "n_benign_flagged": 0,
              "n_covert_total": 0, "n_benign_total": 0}
    per_attr = {}
    for attr in sorted(by_attr):
        s = by_attr[attr]
        orig = (safe_mean(s["covert_orig"]) - safe_mean(s["benign_orig"])) * 100
        resid = (safe_mean(s["covert_resid"]) - safe_mean(s["benign_resid"])) * 100
        delta = orig - resid
        cov_pct = s["n_covert_flagged"] / max(1, s["n_covert_total"]) * 100
        ben_pct = s["n_benign_flagged"] / max(1, s["n_benign_total"]) * 100
        print(f"{attr:<20s} | {orig:>+10.1f}pp | {resid:>+10.1f}pp | {-delta:>+6.1f}pp | "
              f"{cov_pct:>9.0f}% | {ben_pct:>9.0f}%")
        per_attr[attr] = {"orig_pp": orig, "resid_pp": resid, "reduction_pp": delta,
                           "tpr": cov_pct, "fpr": ben_pct}
        for k_ in ("benign_orig", "covert_orig", "benign_resid", "covert_resid"):
            pooled[k_].extend(s[k_])
        for k_ in ("n_covert_flagged", "n_benign_flagged", "n_covert_total", "n_benign_total"):
            pooled[k_] += s[k_]
    print("-" * 90)
    orig_p = (safe_mean(pooled["covert_orig"]) - safe_mean(pooled["benign_orig"])) * 100
    resid_p = (safe_mean(pooled["covert_resid"]) - safe_mean(pooled["benign_resid"])) * 100
    delta_p = orig_p - resid_p
    cov_p = pooled["n_covert_flagged"] / max(1, pooled["n_covert_total"]) * 100
    ben_p = pooled["n_benign_flagged"] / max(1, pooled["n_benign_total"]) * 100
    print(f"{'POOLED':<20s} | {orig_p:>+10.1f}pp | {resid_p:>+10.1f}pp | {-delta_p:>+6.1f}pp | "
          f"{cov_p:>9.0f}% | {ben_p:>9.0f}%")
    return {
        "label": label, "threshold": threshold,
        "per_attribute": per_attr,
        "pooled": {
            "orig_pp": orig_p, "resid_pp": resid_p, "reduction_pp": delta_p,
            "tpr": cov_p / 100, "fpr": ben_p / 100,
            "n_covert_flagged": pooled["n_covert_flagged"], "n_covert_total": pooled["n_covert_total"],
            "n_benign_flagged": pooled["n_benign_flagged"], "n_benign_total": pooled["n_benign_total"],
        },
    }


def main() -> None:
    # Reuse the defender_evaluation script's machinery to get per-trajectory
    # behavioural and semantic scores. Imported lazily.
    from scripts.defender_evaluation import (
        get_behavioural_scores,
        get_semantic_scores,
    )

    trajs = load_trajectories()
    print(f"Loaded {len(trajs)} trajectories")

    # 1. Behavioural scores (covert detection probability)
    print("Scoring trajectories with behavioural probe ...")
    behav = get_behavioural_scores(trajs)
    print(f"  scored {len(behav)} trajectories (benign + covert only)")

    # 2. Semantic scores (LLM-judge P(tau=positive))
    print("Scoring trajectories with semantic probe (cache-backed) ...")
    sem = get_semantic_scores(trajs)
    print(f"  scored {len(sem)} trajectories")

    # 3. Load thresholds from defender_curves.json (computed by
    # defender_evaluation.py).
    curves_path = REPO / "results" / "probes_llm_attacker_full" / "defender_curves.json"
    curves = json.loads(curves_path.read_text())
    behav_threshold = curves["per_probe"]["behavioural"]["operating_points"]["FPR<=0.10"]["threshold"]
    sem_threshold = curves["per_probe"]["semantic"]["operating_points"]["FPR<=0.10"]["threshold"]
    print(f"\nBehavioural-probe threshold at FPR<=0.10: {behav_threshold:.4f}")
    print(f"Semantic-probe threshold at FPR<=0.10:    {sem_threshold:.4f}")

    # 4. Run the analysis with each probe as the filter.
    bc_trajs = [t for t in trajs if t.metadata.get("principal") in ("benign", "covert")]
    behav_result = analyze_filter(bc_trajs, sem, behav, behav_threshold, "BEHAVIOURAL")
    sem_result = analyze_filter(bc_trajs, sem, sem, sem_threshold, "SEMANTIC (self-filter)")

    # 5. Summary contrast.
    print("\n" + "=" * 90)
    print("SUMMARY: Original pooled τ-leak gap was +6.6pp. After deploying each probe at FPR<=0.10:")
    print("=" * 90)
    print(f"  Behavioural filter: residual {behav_result['pooled']['resid_pp']:+.1f}pp "
          f"(Δ {-behav_result['pooled']['reduction_pp']:+.1f}pp; "
          f"caught {behav_result['pooled']['n_covert_flagged']}/{behav_result['pooled']['n_covert_total']} covert)")
    print(f"  Semantic filter:    residual {sem_result['pooled']['resid_pp']:+.1f}pp "
          f"(Δ {-sem_result['pooled']['reduction_pp']:+.1f}pp; "
          f"caught {sem_result['pooled']['n_covert_flagged']}/{sem_result['pooled']['n_covert_total']} covert)")

    # 6. Save combined JSON
    out = {
        "fpr_target": 0.10,
        "behavioural_filter": behav_result,
        "semantic_filter": sem_result,
    }
    out_path = REPO / "results" / "probes_llm_attacker_full" / "defence_impact.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
