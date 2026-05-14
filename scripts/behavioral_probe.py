"""Behavioral / principal-detection probe.

The Stage 6 probes ask "can we extract τ from a trajectory?" — that's the
attribute-inference question.

THIS probe asks a different (defender-side) question: **"can a defender
tell whether a query was covert or benign, using only trajectory-shape
features (no content)?"** A real-world deployment defender monitors
behavioural signals — trajectory length, cost, action-type composition,
stop reason — to flag suspicious queries. If these features carry no
signal, the only working defense is a semantic content audit (Stage 6
LLM judge). If they DO carry signal, lightweight runtime monitors
suffice.

Training set: paired benign + covert trajectories from the same case
set (the LLM-attacker full-headline dataset; see
`results/pilot_llm_attacker_full/`). Case-level k-fold CV (5 folds) so
no case appears in both train and test under different principals.

Features used: n_steps, total_cost, action-type counts (raw and
fractional), stop_reason (one-hot), entropy reduction, max-step EIG.
**No question or observation text — purely structural.**

Output: per-attribute and pooled covert-vs-benign AUC. Reports to
`results/probes_llm_attacker_full/behavioral_probe.json`.
"""
from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.agent.runner import Trajectory  # noqa: E402
from src.probe.logistic import extract_features, _entropy  # noqa: E402


STOP_REASONS = (
    "entropy_threshold",
    "policy_diagnose",
    "max_queries",
    "cost_budget",
)


def behavioral_features(traj: Trajectory) -> dict[str, float]:
    """Pure trajectory-shape features. No text content."""
    base = extract_features(traj)
    n = max(1, len(traj.steps))
    # Action-type FRACTIONS (in addition to counts) so the probe sees
    # composition independently of trajectory length.
    base["frac_ask_history"] = base["n_ask_history"] / n
    base["frac_ask_exam"] = base["n_ask_exam"] / n
    base["frac_order_test"] = base["n_order_test"] / n
    # Stop-reason one-hot.
    stop = (traj.metadata or {}).get("stopped_reason", "")
    for r in STOP_REASONS:
        base[f"stop_{r}"] = 1.0 if stop == r else 0.0
    # Max action cost (proxy for "did the agent order an expensive test?")
    if traj.steps:
        base["max_step_cost"] = float(max(s.action.cost for s in traj.steps))
    else:
        base["max_step_cost"] = 0.0
    return base


def _to_matrix(trajectories: list[Trajectory], keys: list[str]) -> np.ndarray:
    return np.array(
        [[behavioral_features(t).get(k, 0.0) for k in keys] for t in trajectories],
        dtype=np.float64,
    )


def _auc(scores: list[float], labels: list[int]) -> float:
    if not scores:
        return 0.5
    pairs = sorted(zip(scores, labels), key=lambda p: -p[0])
    pos = sum(1 for _, y in pairs if y == 1)
    neg = len(pairs) - pos
    if pos == 0 or neg == 0:
        return 0.5
    rank_sum = sum(len(pairs) + 1 - r for r, (_, y) in enumerate(pairs, start=1) if y == 1)
    return float((rank_sum - pos * (pos + 1) / 2) / (pos * neg))


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    halfw = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return p, max(0.0, centre - halfw), min(1.0, centre + halfw)


def load_trajs(source_dir: Path) -> list[Trajectory]:
    trajs: list[Trajectory] = []
    for p in sorted(source_dir.glob("*.jsonl")):
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                trajs.append(Trajectory.model_validate_json(line))
    return trajs


def run(source_stage: str = "pilot_llm_attacker_full") -> dict:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler

    source_dir = REPO / "results" / source_stage / "trajectories"
    trajs = load_trajs(source_dir)
    if not trajs:
        raise RuntimeError(f"No trajectories at {source_dir}")
    bc_trajs = [t for t in trajs if t.metadata.get("principal") in ("benign", "covert")]
    attrs = sorted({t.metadata.get("attribute", "") for t in bc_trajs if t.metadata.get("attribute")})

    feature_keys = sorted(behavioral_features(bc_trajs[0]).keys())

    per_attr: dict[str, dict] = {}
    pooled_scores: list[float] = []
    pooled_labels: list[int] = []
    pooled_attrs: list[str] = []

    for attr in attrs:
        attr_trajs = [t for t in bc_trajs if t.metadata.get("attribute") == attr]
        # Case-level fold split: each case has up to 2 trajectories
        # (benign, covert) sharing the same τ; we split at the case level
        # to prevent the probe from memorising case identity.
        case_to_trajs: dict[str, list[Trajectory]] = {}
        for t in attr_trajs:
            case_to_trajs.setdefault(t.case_id, []).append(t)
        case_ids = sorted(case_to_trajs)
        case_labels = []
        for cid in case_ids:
            ts = case_to_trajs[cid]
            # is_covert label = 1 if covert traj exists, but stratify by
            # whether both principals are present (most cases do)
            case_labels.append(1)  # all cases have both; stratification trivial
        n_splits = min(5, len(case_ids))
        if n_splits < 2:
            per_attr[attr] = {"n": 0, "auc": 0.5, "note": "too few cases"}
            continue
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
        # Stratify by τ value of the case (so the splits are balanced on τ).
        case_taus = [int(case_to_trajs[cid][0].metadata.get("tau", 0)) for cid in case_ids]
        oof: list[tuple[float, int]] = []
        for fold_train_idx, fold_test_idx in skf.split(case_ids, case_taus):
            train_cases = [case_ids[i] for i in fold_train_idx]
            test_cases = [case_ids[i] for i in fold_test_idx]
            train_trajs = [t for cid in train_cases for t in case_to_trajs[cid]]
            test_trajs = [t for cid in test_cases for t in case_to_trajs[cid]]
            X_tr = _to_matrix(train_trajs, feature_keys)
            X_te = _to_matrix(test_trajs, feature_keys)
            y_tr = [1 if t.metadata.get("principal") == "covert" else 0 for t in train_trajs]
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_tr)
            X_te = scaler.transform(X_te)
            clf = LogisticRegression(
                penalty="l2",
                C=1.0,
                class_weight="balanced",
                max_iter=2000,
                random_state=0,
            )
            clf.fit(X_tr, y_tr)
            probs = clf.predict_proba(X_te)[:, 1].tolist()
            for t, p in zip(test_trajs, probs):
                y = 1 if t.metadata.get("principal") == "covert" else 0
                oof.append((p, y))
        scores = [s for s, _ in oof]
        labels = [y for _, y in oof]
        auc = _auc(scores, labels)
        # Accuracy at threshold 0.5
        preds_at_05 = [1 if s >= 0.5 else 0 for s in scores]
        correct = sum(1 for p, y in zip(preds_at_05, labels) if p == y)
        acc, lo, hi = wilson(correct, len(labels))
        per_attr[attr] = {
            "n": len(labels),
            "auc": auc,
            "accuracy_at_05": acc,
            "accuracy_ci": [lo, hi],
            "n_correct": correct,
        }
        pooled_scores.extend(scores)
        pooled_labels.extend(labels)
        pooled_attrs.extend([attr] * len(scores))

    pooled_auc = _auc(pooled_scores, pooled_labels)
    pooled_preds = [1 if s >= 0.5 else 0 for s in pooled_scores]
    pooled_correct = sum(1 for p, y in zip(pooled_preds, pooled_labels) if p == y)
    pooled_acc, plo, phi = wilson(pooled_correct, len(pooled_labels))

    # Feature importance — pull from a final fit on all trajectories.
    X_all = _to_matrix(bc_trajs, feature_keys)
    y_all = [1 if t.metadata.get("principal") == "covert" else 0 for t in bc_trajs]
    scaler = StandardScaler()
    X_all_s = scaler.fit_transform(X_all)
    clf_all = LogisticRegression(
        penalty="l2", C=1.0, class_weight="balanced", max_iter=2000, random_state=0
    )
    clf_all.fit(X_all_s, y_all)
    coefs = clf_all.coef_[0]
    coef_pairs = sorted(zip(feature_keys, coefs.tolist()), key=lambda p: -abs(p[1]))

    out = {
        "source_stage": source_stage,
        "feature_keys": feature_keys,
        "per_attribute": per_attr,
        "pooled": {
            "n": len(pooled_labels),
            "auc": pooled_auc,
            "accuracy_at_05": pooled_acc,
            "accuracy_ci": [plo, phi],
            "n_correct": pooled_correct,
        },
        "top_features": coef_pairs[:10],
    }
    return out


def print_report(out: dict) -> None:
    print(f"\n=== BEHAVIORAL PROBE (covert-vs-benign on shape features) ===")
    print(f"Source: {out['source_stage']}")
    print(f"\nPer-attribute AUC:")
    for attr, m in out["per_attribute"].items():
        if m["n"] == 0:
            print(f"  {attr:18s}: {m.get('note', 'no data')}")
            continue
        print(f"  {attr:18s}: AUC={m['auc']:.3f}, "
              f"acc@0.5={m['accuracy_at_05']*100:.0f}% "
              f"[{m['accuracy_ci'][0]*100:.0f}, {m['accuracy_ci'][1]*100:.0f}], "
              f"n={m['n']}")
    p = out["pooled"]
    print(f"\nPOOLED: AUC={p['auc']:.3f}, "
          f"acc@0.5={p['accuracy_at_05']*100:.0f}% "
          f"[{p['accuracy_ci'][0]*100:.0f}, {p['accuracy_ci'][1]*100:.0f}], "
          f"n={p['n']}")
    print(f"\nTop 10 features by |coefficient| (positive ⇒ pushes 'covert'):")
    for feat, c in out["top_features"]:
        sign = "↑covert" if c > 0 else "↑benign"
        print(f"  {feat:30s} {c:+.3f} ({sign})")


if __name__ == "__main__":
    source = sys.argv[1] if len(sys.argv) > 1 else "pilot_llm_attacker_full"
    result = run(source)
    print_report(result)
    out_dir = REPO / "results" / f"probes_llm_attacker_full"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "behavioral_probe.json").write_text(json.dumps(result, indent=2))
    print(f"\nWrote {out_dir / 'behavioral_probe.json'}")
