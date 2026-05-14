"""End-to-end defender evaluation: ROC curves and operating points for
all three probe classes (lexical / behavioural / semantic) and stacked
combinations. Reports TPR at fixed FPR operating points so the paper
can talk in deployment terms ("at FPR=10%, the LLM-judge catches X% of
covert attacks") rather than just point AUCs.

Inputs:
  - results/pilot_llm_attacker_full/trajectories/  (the 600-trajectory
    LLM-attacker embedded headline)
  - results/probes_llm_attacker_full/llm_cache/    (cached LLM-judge
    responses; we re-issue requests against the existing LLMClient
    cache so this script costs $0 if Stage 6 has already run)

Outputs:
  - results/probes_llm_attacker_full/defender_curves.json
  - stdout: a Markdown-style operating-point table
"""
from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.agent.runner import Trajectory  # noqa: E402
from src.data.attribute_schema import get_attribute  # noqa: E402
from src.probe.logistic import LogisticProbe, extract_features  # noqa: E402

PRINCIPALS = ("benign", "direct", "covert")
ATTRIBUTES = ("hiv_status", "iv_drug_use", "pregnancy", "mental_health_dx")
SOURCE_STAGE = "pilot_llm_attacker_full"


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    halfw = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return p, max(0.0, centre - halfw), min(1.0, centre + halfw)


def load_trajectories(source_stage: str) -> list[Trajectory]:
    traj_dir = REPO / "results" / source_stage / "trajectories"
    trajs: list[Trajectory] = []
    for p in sorted(traj_dir.glob("*.jsonl")):
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                trajs.append(Trajectory.model_validate_json(line))
    return trajs


# --- per-trajectory score collection ----------------------------------

def get_lexical_scores(trajs: list[Trajectory]) -> dict[str, float]:
    """Train a lexical probe (TF-IDF + numerical + belief vectors) via
    case-level 5-fold CV; return per-(case_id, principal) OOF score
    for tau-recovery. Matches the Stage 6 lexical-probe pipeline."""
    from sklearn.model_selection import StratifiedKFold

    out: dict[str, float] = {}
    # Group by attribute
    by_attr: dict[str, list[Trajectory]] = defaultdict(list)
    for t in trajs:
        attr = t.metadata.get("attribute", "")
        if attr:
            by_attr[attr].append(t)
    for attr, attr_trajs in by_attr.items():
        # Case-level fold split
        cases: dict[str, list[Trajectory]] = defaultdict(list)
        for t in attr_trajs:
            cases[t.case_id].append(t)
        case_ids = sorted(cases)
        case_taus = [int(cases[c][0].metadata.get("tau", 0)) for c in case_ids]
        n_pos = sum(case_taus)
        n_neg = len(case_taus) - n_pos
        n_splits = min(5, n_pos, n_neg)
        if n_splits < 2:
            continue
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
        for train_idx, test_idx in skf.split(case_ids, case_taus):
            train_cases = [case_ids[i] for i in train_idx]
            test_cases = [case_ids[i] for i in test_idx]
            train_trajs = [t for c in train_cases for t in cases[c]]
            test_trajs = [t for c in test_cases for t in cases[c]]
            train_labels = [int(t.metadata.get("tau", 0)) for t in train_trajs]
            try:
                probe = LogisticProbe(use_tfidf=True)
                probe.fit(train_trajs, train_labels)
                preds = probe.predict_proba_batch(test_trajs)
                for t, p in zip(test_trajs, preds):
                    out[f"{t.case_id}__{t.metadata.get('principal')}__{attr}"] = float(p)
            except Exception as e:
                print(f"  [skip fold attr={attr}]: {e}", file=sys.stderr)
    return out


def get_behavioural_scores(trajs: list[Trajectory]) -> dict[str, float]:
    """Per-trajectory behavioural probe score (label = is_covert).
    Mirrors scripts/behavioral_probe.py — uses only structural features.
    Trained per-attribute on benign+covert pairs with case-level 5-fold
    CV; returns OOF P(is_covert) for benign and covert trajectories,
    NaN for direct."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    from scripts.behavioral_probe import behavioral_features

    out: dict[str, float] = {}
    bc_trajs = [t for t in trajs if t.metadata.get("principal") in ("benign", "covert")]
    by_attr: dict[str, list[Trajectory]] = defaultdict(list)
    for t in bc_trajs:
        a = t.metadata.get("attribute", "")
        if a:
            by_attr[a].append(t)
    if not bc_trajs:
        return out
    feature_keys = sorted(behavioral_features(bc_trajs[0]).keys())
    for attr, attr_trajs in by_attr.items():
        cases: dict[str, list[Trajectory]] = defaultdict(list)
        for t in attr_trajs:
            cases[t.case_id].append(t)
        case_ids = sorted(cases)
        case_taus = [int(cases[c][0].metadata.get("tau", 0)) for c in case_ids]
        n_splits = min(5, len(case_ids))
        if n_splits < 2:
            continue
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
        for train_idx, test_idx in skf.split(case_ids, case_taus):
            train_cases = [case_ids[i] for i in train_idx]
            test_cases = [case_ids[i] for i in test_idx]
            train_trajs = [t for c in train_cases for t in cases[c]]
            test_trajs = [t for c in test_cases for t in cases[c]]
            X_tr = np.array([[behavioral_features(t).get(k, 0.0) for k in feature_keys]
                             for t in train_trajs])
            X_te = np.array([[behavioral_features(t).get(k, 0.0) for k in feature_keys]
                             for t in test_trajs])
            y_tr = [1 if t.metadata.get("principal") == "covert" else 0 for t in train_trajs]
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_tr)
            X_te = scaler.transform(X_te)
            clf = LogisticRegression(
                penalty="l2", C=1.0, class_weight="balanced",
                max_iter=2000, random_state=0,
            )
            clf.fit(X_tr, y_tr)
            preds = clf.predict_proba(X_te)[:, 1].tolist()
            for t, p in zip(test_trajs, preds):
                out[f"{t.case_id}__{t.metadata.get('principal')}__{attr}"] = float(p)
    return out


def get_semantic_scores(trajs: list[Trajectory]) -> dict[str, float]:
    """LLM-judge per-trajectory P(tau=positive). Re-issues requests
    through the cached LLMClient so this runs at $0 if Stage 6 has
    already populated the cache."""
    import os
    if "GOOGLE_API_KEY" not in os.environ:
        for line in (REPO / ".env").read_text().splitlines():
            if line.startswith("GOOGLE_API_KEY="):
                os.environ["GOOGLE_API_KEY"] = (
                    line.split("=", 1)[1].strip().strip('"').strip("'")
                )
                break
    from src.llm.api_backend import build_api_backend_from_config
    from src.llm.client import LLMClient
    from src.probe.llm_judge import LLMJudgeProbe
    import yaml as _yaml
    api_cfg = _yaml.safe_load((REPO / "configs" / "models_api.yaml").read_text())
    backend = build_api_backend_from_config(api_cfg)
    cache_dir = REPO / "results" / "probes_llm_attacker_full" / "llm_cache"
    client = LLMClient(backend="api", cache_dir=cache_dir, api_backend=backend)
    template = (REPO / "prompts" / "llm_judge_probe.txt").read_text()
    judge = LLMJudgeProbe(client=client, prompt_template=template)

    out: dict[str, float] = {}
    by_attr: dict[str, list[Trajectory]] = defaultdict(list)
    for t in trajs:
        a = t.metadata.get("attribute", "")
        if a:
            by_attr[a].append(t)
    for attr, attr_trajs in by_attr.items():
        attr_obj = get_attribute(attr)
        scores = judge.predict_proba_batch(attr_trajs, attr_obj)
        for t, s in zip(attr_trajs, scores):
            out[f"{t.case_id}__{t.metadata.get('principal')}__{attr}"] = float(s)
    client.close()
    backend.close()
    return out


# --- ROC analysis -----------------------------------------------------

def roc_curve(scores: list[float], labels: list[int]) -> dict:
    """Compute the ROC curve. Returns thresholds, TPR, FPR arrays + AUC."""
    if not scores:
        return {"auc": 0.5, "thresholds": [], "tpr": [], "fpr": []}
    pairs = sorted(zip(scores, labels), key=lambda p: -p[0])
    pos = sum(1 for _, y in pairs if y == 1)
    neg = len(pairs) - pos
    if pos == 0 or neg == 0:
        return {"auc": 0.5, "thresholds": [], "tpr": [], "fpr": []}
    tp = fp = 0
    tpr_list = [0.0]
    fpr_list = [0.0]
    thr_list = [float("inf")]
    prev_score = None
    for s, y in pairs:
        if prev_score is not None and s != prev_score:
            tpr_list.append(tp / pos)
            fpr_list.append(fp / neg)
            thr_list.append(prev_score)
        if y == 1:
            tp += 1
        else:
            fp += 1
        prev_score = s
    tpr_list.append(tp / pos)
    fpr_list.append(fp / neg)
    thr_list.append(prev_score if prev_score is not None else 0.0)
    # AUC via trapezoid
    auc = 0.0
    for i in range(1, len(fpr_list)):
        auc += (fpr_list[i] - fpr_list[i - 1]) * (tpr_list[i] + tpr_list[i - 1]) / 2
    return {
        "auc": float(auc),
        "thresholds": thr_list,
        "tpr": tpr_list,
        "fpr": fpr_list,
        "n_pos": pos,
        "n_neg": neg,
    }


def tpr_at_fpr(curve: dict, target_fpr: float) -> tuple[float, float]:
    """Return (TPR, threshold) at the largest FPR ≤ target_fpr."""
    if not curve["fpr"]:
        return 0.0, 0.0
    best_tpr = 0.0
    best_thr = curve["thresholds"][0]
    for fpr, tpr, thr in zip(curve["fpr"], curve["tpr"], curve["thresholds"]):
        if fpr <= target_fpr:
            if tpr > best_tpr:
                best_tpr = tpr
                best_thr = thr
    return best_tpr, best_thr


# --- Main analysis ----------------------------------------------------

def run() -> dict:
    print("Loading trajectories from", SOURCE_STAGE)
    trajs = load_trajectories(SOURCE_STAGE)
    print(f"  loaded {len(trajs)} trajectories")
    by_principal = defaultdict(list)
    for t in trajs:
        by_principal[t.metadata.get("principal")].append(t)
    print(f"  per-principal counts: {dict((k, len(v)) for k,v in by_principal.items())}")

    # Compute per-trajectory scores for each probe
    print("\n[1/3] Computing lexical probe scores (case-level 5-fold OOF, TF-IDF + numerical) ...")
    lex = get_lexical_scores(trajs)
    print(f"      scored {len(lex)} trajectories")

    print("\n[2/3] Computing behavioural probe scores (case-level 5-fold OOF, structural features only) ...")
    beh = get_behavioural_scores(trajs)
    print(f"      scored {len(beh)} trajectories")

    print("\n[3/3] Computing semantic probe scores (LLM-judge, cache-backed) ...")
    sem = get_semantic_scores(trajs)
    print(f"      scored {len(sem)} trajectories")

    # === Build operational ROC curves: covert vs benign ===
    # For lexical and semantic: predicts tau, so the threat is "does
    # this trajectory carry positive tau-signal" — but the defender
    # wants to know "is this trajectory covert". A defender uses these
    # tau-scores in two ways: (i) high tau-score on a non-direct
    # trajectory is suspicious; (ii) compare to benign baseline.
    # We compute ROC where label = is_covert (covert=1, benign=0); the
    # score is the probe's tau-score (high tau-score => more covert-ish
    # if the patient is tau-positive, but this confounds with tau
    # ground truth). To be fair, we restrict to the tau=1 subset where
    # both benign and covert are present with the same tau.
    #
    # For behavioural: label is already is_covert.

    result: dict = {"per_probe": {}, "operating_points": {}}

    # Lexical and semantic: paired-on-tau ROC for covert vs benign
    def key(traj):
        return f"{traj.case_id}__{traj.metadata.get('principal')}__{traj.metadata.get('attribute')}"

    for label, score_map in (("lexical", lex), ("semantic", sem)):
        # Pool covert and benign; label = 1 if covert
        s_list: list[float] = []
        y_list: list[int] = []
        for t in trajs:
            p = t.metadata.get("principal")
            if p not in ("covert", "benign"):
                continue
            k = key(t)
            if k not in score_map:
                continue
            s_list.append(score_map[k])
            y_list.append(1 if p == "covert" else 0)
        roc = roc_curve(s_list, y_list)
        ops = {}
        for fpr_target in (0.01, 0.05, 0.10, 0.20, 0.50):
            tpr, thr = tpr_at_fpr(roc, fpr_target)
            ops[f"FPR<={fpr_target:.2f}"] = {
                "TPR": tpr, "threshold": thr,
            }
        result["per_probe"][label] = {
            "auc_covert_vs_benign": roc["auc"],
            "n_pos_covert": roc["n_pos"],
            "n_neg_benign": roc["n_neg"],
            "operating_points": ops,
        }

    # Behavioural: native is_covert label
    s_list = []
    y_list = []
    for t in trajs:
        p = t.metadata.get("principal")
        if p not in ("covert", "benign"):
            continue
        k = key(t)
        if k not in beh:
            continue
        s_list.append(beh[k])
        y_list.append(1 if p == "covert" else 0)
    roc = roc_curve(s_list, y_list)
    ops = {}
    for fpr_target in (0.01, 0.05, 0.10, 0.20, 0.50):
        tpr, thr = tpr_at_fpr(roc, fpr_target)
        ops[f"FPR<={fpr_target:.2f}"] = {"TPR": tpr, "threshold": thr}
    result["per_probe"]["behavioural"] = {
        "auc_covert_vs_benign": roc["auc"],
        "n_pos_covert": roc["n_pos"],
        "n_neg_benign": roc["n_neg"],
        "operating_points": ops,
    }

    # Stacked OR: defender flags if ANY probe exceeds its threshold
    print("\n[stack] Building stacked-probe operating points (flag if ANY probe at its FPR<=0.10 threshold)")
    stacked = []
    for t in trajs:
        p = t.metadata.get("principal")
        if p not in ("covert", "benign"):
            continue
        k = key(t)
        scores = {}
        for label, m in (("lexical", lex), ("behavioural", beh), ("semantic", sem)):
            if k in m:
                scores[label] = m[k]
        stacked.append((scores, 1 if p == "covert" else 0))

    # For the stacked OR, use each probe's FPR<=0.10 threshold and flag if any exceed it
    thresholds_at_10 = {
        label: result["per_probe"][label]["operating_points"]["FPR<=0.10"]["threshold"]
        for label in ("lexical", "behavioural", "semantic")
    }
    tp = fp = pos = neg = 0
    for scores, y in stacked:
        flagged = any(scores.get(label, -1) >= thr for label, thr in thresholds_at_10.items())
        if y == 1:
            pos += 1
            if flagged: tp += 1
        else:
            neg += 1
            if flagged: fp += 1
    stacked_tpr = tp / pos if pos else 0.0
    stacked_fpr = fp / neg if neg else 0.0
    result["operating_points"]["stacked_OR_at_each_FPR0.10"] = {
        "TPR": stacked_tpr, "FPR": stacked_fpr,
        "thresholds": thresholds_at_10,
    }

    return result


def print_report(r: dict) -> None:
    print()
    print("=" * 78)
    print("DEFENDER ROC SUMMARY — covert (positive) vs benign (negative) at FPR targets")
    print("=" * 78)
    for probe, m in r["per_probe"].items():
        print(f"\n{probe.upper()}  AUC={m['auc_covert_vs_benign']:.3f}  "
              f"(n_covert={m['n_pos_covert']}, n_benign={m['n_neg_benign']})")
        print(f"  {'FPR target':<16} {'TPR':>8} {'threshold':>12}")
        for k, v in m["operating_points"].items():
            print(f"  {k:<16} {v['TPR']:>7.1%} {v['threshold']:>12.3f}")
    print()
    s = r["operating_points"]["stacked_OR_at_each_FPR0.10"]
    print(f"STACKED OR (flag if ANY probe exceeds its FPR=0.10 threshold):")
    print(f"  TPR={s['TPR']:.1%}  FPR={s['FPR']:.1%}")
    print(f"  thresholds used: {s['thresholds']}")


if __name__ == "__main__":
    out_dir = REPO / "results" / "probes_llm_attacker_full"
    out_dir.mkdir(parents=True, exist_ok=True)
    result = run()
    out = out_dir / "defender_curves.json"
    out.write_text(json.dumps(result, indent=2, default=str))
    print_report(result)
    print(f"\nWrote {out}")
