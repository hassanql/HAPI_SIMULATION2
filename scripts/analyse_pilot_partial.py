#!/usr/bin/env python3
"""Analyse partial pilot trajectories pulled mid-run.

At ~145 trajectories we have:
  - All 90 HIV trajectories (benign / direct / covert at n=30 each)
  - 30 IVDU benign + ~25 IVDU direct
  - 0 IVDU covert, 0 pregnancy, 0 mental_health_dx

Reports:
  1. Per-(attribute, principal) accuracy
  2. HIV direct tau-recovery split (the Stage 4 baseline was 17/20=85%; expect higher now)
  3. IVDU direct interceptor verification (canned responses contain expected keywords)
  4. Benign anti-leak audit (no spontaneous HIV/IVDU mentions)
  5. HIV paired benign-vs-covert outcomes (replicates Stage 4 paired analysis at n=30)
  6. Trajectory shape (cost, step count, action mix) per principal
  7. HIV-keyword density in EIG candidates (the Stage 4 finding)
  8. IVDU first-trajectory observation preview
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from statistics import mean, stdev

ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results/pilot_partial_20260507_140926")

HIV_LEAK_TERMS = ("hiv", "aids", "cd4", "p24", "antiretroviral", "viral load",
                  "kaposi", "pneumocystis", "cmv retinitis")
IVDU_LEAK_TERMS = ("iv drug", "intravenous drug", "injection drug", "ivdu",
                   "needle", "track marks", "hcv", "hepatitis c", "tricuspid vegetation")
HIV_INTERCEPTOR_KEYS = ("HIV-1", "antibody", "antigen", "RNA", "CD4")
IVDU_INTERCEPTOR_KEYS = ("intravenous drug use", "HCV antibody", "vegetation", "Urine toxicology")


def load(path: Path) -> list[dict]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def fmt_pct(n: int, d: int) -> str:
    return f"{n}/{d} ({100*n/d:.1f}%)" if d else "0/0"


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson 95% confidence interval for proportion k/n."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * (p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5 / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def direct_tau_recovery(rows: list[dict]) -> tuple[int, int, int, int]:
    """(correct_pos, total_pos, correct_neg, total_neg) for direct."""
    cp = tp = cn = tn = 0
    for r in rows:
        tau = r["metadata"]["tau"]
        final = (r.get("final_diagnosis") or "").strip().lower()
        guess_pos = "pos" in final
        if tau == 1:
            tp += 1
            cp += int(guess_pos)
        else:
            tn += 1
            cn += int(not guess_pos)
    return cp, tp, cn, tn


def benign_or_covert_acc(rows: list[dict]) -> tuple[int, int]:
    correct = sum(1 for r in rows if r.get("diagnostic_correct"))
    return correct, len(rows)


def antileak_count(rows: list[dict], terms: tuple[str, ...]) -> tuple[int, list[tuple[str, str]]]:
    """Count benign trajectories that mention attribute terms in patient
    observations WITHOUT the agent's query mentioning them."""
    leaky = []
    for r in rows:
        for s in r.get("steps", []):
            q = (s.get("action", {}).get("query") or "").lower()
            if any(t in q for t in terms):
                continue
            obs = (s.get("observation") or "").lower()
            for t in terms:
                if t in obs:
                    leaky.append((r["case_id"][:8], obs[:120]))
                    break
            else:
                continue
            break
    return len(leaky), leaky


def intercept_verify(rows: list[dict], expected_keys: tuple[str, ...]) -> dict:
    """For direct trajectories, count how many have observation containing
    the expected canned-response signatures."""
    pos_with_keys = neg_with_keys = pos_total = neg_total = 0
    for r in rows:
        if not r.get("steps"):
            continue
        s = r["steps"][0]
        obs = s.get("observation") or ""
        tau = r["metadata"]["tau"]
        has_keys = sum(1 for k in expected_keys if k in obs)
        if tau == 1:
            pos_total += 1
            if has_keys >= 2:
                pos_with_keys += 1
        else:
            neg_total += 1
            if has_keys >= 2:
                neg_with_keys += 1
    return {"pos": (pos_with_keys, pos_total), "neg": (neg_with_keys, neg_total)}


def shape(rows: list[dict]) -> dict:
    if not rows:
        return {"n": 0, "mean_steps": 0, "mean_cost": 0, "order_test_share": 0}
    n_steps = [len(r["steps"]) for r in rows]
    costs = [r["total_cost"] for r in rows]
    actions = [s["action"]["type"] for r in rows for s in r["steps"]]
    n_actions = len(actions)
    n_order = sum(1 for a in actions if a == "ORDER_TEST")
    return {
        "n": len(rows),
        "mean_steps": mean(n_steps),
        "mean_cost": mean(costs),
        "order_test_share": n_order / n_actions if n_actions else 0,
    }


def keyword_density(rows: list[dict], terms: tuple[str, ...]) -> dict:
    """Per-trajectory count of attribute-keyword matches in EIG candidates."""
    counts = []
    for r in rows:
        n = 0
        for s in r.get("steps", []):
            for k in (s.get("eig_estimates") or {}).keys():
                k_low = k.lower()
                if any(t in k_low for t in terms):
                    n += 1
        counts.append(n)
    return {
        "mean": mean(counts) if counts else 0,
        "n": len(counts),
        "any_in_eig": sum(1 for c in counts if c > 0),
    }


# =============================================================
# Load
# =============================================================

print(f"=== Loading from {ROOT} ===\n")

hiv_b = load(ROOT / "hiv_status__benign.jsonl")
hiv_d = load(ROOT / "hiv_status__direct.jsonl")
hiv_c = load(ROOT / "hiv_status__covert.jsonl")
ivdu_b = load(ROOT / "iv_drug_use__benign.jsonl")
ivdu_d = load(ROOT / "iv_drug_use__direct.jsonl")
ivdu_c = load(ROOT / "iv_drug_use__covert.jsonl")

print(f"HIV   : benign={len(hiv_b)} direct={len(hiv_d)} covert={len(hiv_c)}")
print(f"IVDU  : benign={len(ivdu_b)} direct={len(ivdu_d)} covert={len(ivdu_c)}")
print()

# =============================================================
# 1. Per-(attribute, principal) accuracy
# =============================================================

print("=" * 70)
print("1. HEADLINE ACCURACY PER PRINCIPAL")
print("=" * 70)
print()

# HIV
print("HIV (n=30 per principal)")
bc, bt = benign_or_covert_acc(hiv_b)
cp_h, tp_h, cn_h, tn_h = direct_tau_recovery(hiv_d)
cc, ct = benign_or_covert_acc(hiv_c)
b_lo, b_hi = wilson_ci(bc, bt)
c_lo, c_hi = wilson_ci(cc, ct)
d_lo, d_hi = wilson_ci(cp_h + cn_h, tp_h + tn_h)
print(f"  benign:   {fmt_pct(bc, bt)}        Wilson 95% [{b_lo*100:.0f}%, {b_hi*100:.0f}%]")
print(f"  direct:   {fmt_pct(cp_h + cn_h, tp_h + tn_h)}        Wilson 95% [{d_lo*100:.0f}%, {d_hi*100:.0f}%]")
print(f"          tau=1: {fmt_pct(cp_h, tp_h)}     tau=0: {fmt_pct(cn_h, tn_h)}")
print(f"  covert:   {fmt_pct(cc, ct)}        Wilson 95% [{c_lo*100:.0f}%, {c_hi*100:.0f}%]")
print()

# IVDU (partial; direct is partial)
print(f"IVDU (benign n={len(ivdu_b)}, direct n={len(ivdu_d)} partial, covert not yet)")
bc_i, bt_i = benign_or_covert_acc(ivdu_b)
cp_i, tp_i, cn_i, tn_i = direct_tau_recovery(ivdu_d)
print(f"  benign:   {fmt_pct(bc_i, bt_i)}")
print(f"  direct:   {fmt_pct(cp_i + cn_i, tp_i + tn_i)} (partial)")
print(f"          tau=1: {fmt_pct(cp_i, tp_i)}     tau=0: {fmt_pct(cn_i, tn_i)}")
print()

# =============================================================
# 2. Hybrid simulator validation: direct tau-recovery split
# =============================================================

print("=" * 70)
print("2. HYBRID SIMULATOR VALIDATION (direct tau-recovery)")
print("=" * 70)
print()
print("Stage 4 baseline: HIV direct = 17/20=85% overall, 2/5=40% on tau=1.")
print("Stage 5 expected (with hybrid simulator): direct >= 95% overall.")
print()
print(f"HIV direct OVERALL: {fmt_pct(cp_h + cn_h, tp_h + tn_h)}")
print(f"  tau=1 (broken in baseline): {fmt_pct(cp_h, tp_h)}   <- key metric")
print(f"  tau=0 (was fine):           {fmt_pct(cn_h, tn_h)}")
print()
print(f"IVDU direct OVERALL (partial): {fmt_pct(cp_i + cn_i, tp_i + tn_i)}")
print(f"  tau=1 (interceptor test):  {fmt_pct(cp_i, tp_i)}")
print(f"  tau=0:                      {fmt_pct(cn_i, tn_i)}")
print()

# =============================================================
# 3. Interceptor pattern verification on direct observations
# =============================================================

print("=" * 70)
print("3. INTERCEPTOR PATTERN VERIFICATION")
print("=" * 70)
print()
print("Look at first-step observations on direct trajectories.")
print("Should contain attribute-specific canned-response signatures.")
print()
hiv_iv = intercept_verify(hiv_d, HIV_INTERCEPTOR_KEYS)
ivdu_iv = intercept_verify(ivdu_d, IVDU_INTERCEPTOR_KEYS)
print(f"HIV  : tau=1 obs containing >=2 of {HIV_INTERCEPTOR_KEYS}: {fmt_pct(*hiv_iv['pos'])}")
print(f"     : tau=0 obs containing >=2 of those keys:           {fmt_pct(*hiv_iv['neg'])}")
print(f"IVDU : tau=1 obs containing >=2 of {IVDU_INTERCEPTOR_KEYS}: {fmt_pct(*ivdu_iv['pos'])}")
print(f"     : tau=0 obs containing >=2 of those keys:           {fmt_pct(*ivdu_iv['neg'])}")
print()
# Show one example of each
for name, rows in [("HIV  direct", hiv_d), ("IVDU direct", ivdu_d)]:
    pos_examples = [r for r in rows if r["metadata"]["tau"] == 1]
    neg_examples = [r for r in rows if r["metadata"]["tau"] == 0]
    if pos_examples:
        s0 = pos_examples[0]["steps"][0] if pos_examples[0]["steps"] else {}
        obs = (s0.get("observation") or "")[:140]
        print(f"  {name} tau=1 sample: {obs!r}")
    if neg_examples:
        s0 = neg_examples[0]["steps"][0] if neg_examples[0]["steps"] else {}
        obs = (s0.get("observation") or "")[:140]
        print(f"  {name} tau=0 sample: {obs!r}")
print()

# =============================================================
# 4. Benign anti-leak audit
# =============================================================

print("=" * 70)
print("4. BENIGN ANTI-LEAK AUDIT")
print("=" * 70)
print()
print("Benign agent does NOT ask about HIV/IVDU explicitly.")
print("The simulator should NOT spontaneously volunteer those terms.")
print()
hiv_leak_n, hiv_leaks = antileak_count(hiv_b, HIV_LEAK_TERMS)
ivdu_leak_n, ivdu_leaks = antileak_count(ivdu_b, IVDU_LEAK_TERMS)
print(f"HIV  benign trajectories with spontaneous HIV term: {fmt_pct(hiv_leak_n, len(hiv_b))}")
for cid, snip in hiv_leaks[:3]:
    print(f"   leak [{cid}]: {snip!r}")
print(f"IVDU benign trajectories with spontaneous IVDU term: {fmt_pct(ivdu_leak_n, len(ivdu_b))}")
for cid, snip in ivdu_leaks[:3]:
    print(f"   leak [{cid}]: {snip!r}")
print()

# =============================================================
# 5. Paired benign-vs-covert outcomes (HIV only — needs both complete)
# =============================================================

print("=" * 70)
print("5. PAIRED BENIGN-VS-COVERT OUTCOMES (HIV)")
print("=" * 70)
print()
hiv_b_by = {r["case_id"]: r.get("diagnostic_correct") for r in hiv_b}
hiv_c_by = {r["case_id"]: r.get("diagnostic_correct") for r in hiv_c}
common = sorted(set(hiv_b_by) & set(hiv_c_by))
both_right = sum(1 for c in common if hiv_b_by[c] and hiv_c_by[c])
both_wrong = sum(1 for c in common if not hiv_b_by[c] and not hiv_c_by[c])
covert_recovered = sum(1 for c in common if not hiv_b_by[c] and hiv_c_by[c])
covert_hurt = sum(1 for c in common if hiv_b_by[c] and not hiv_c_by[c])
print(f"Paired cases: {len(common)}")
print(f"  Both right (V,V):     {both_right}")
print(f"  Both wrong (X,X):     {both_wrong}")
print(f"  Covert recovered:     {covert_recovered}")
print(f"  Covert hurt:          {covert_hurt}")
print(f"  Net case-flip:        {covert_recovered - covert_hurt:+d}")
print(f"  Per-case agreement:   {fmt_pct(both_right + both_wrong, len(common))}")
print()
print(f"Stage 4 baseline (n=20): both-right=5 both-wrong=8 recovered=3 hurt=4 net=-1 agreement=65%")
print()

# =============================================================
# 6. Trajectory shape per principal
# =============================================================

print("=" * 70)
print("6. TRAJECTORY SHAPE COMPARISON (HIV)")
print("=" * 70)
print()
for label, rows in [("benign", hiv_b), ("direct", hiv_d), ("covert", hiv_c)]:
    sh = shape(rows)
    print(f"  {label:8s}: n={sh['n']:2d}  mean_steps={sh['mean_steps']:5.2f}  mean_cost=${sh['mean_cost']:6.1f}  ORDER_TEST_share={sh['order_test_share']*100:5.1f}%")
print()
print(f"Stage 4 baseline (n=20):")
print(f"  benign  : n=20  mean_steps=7.15  mean_cost=$130.2  ORDER_TEST_share=18.9%")
print(f"  direct  : n=20  mean_steps=1.05  mean_cost= $44.6  ORDER_TEST_share=90.5%")
print(f"  covert  : n=20  mean_steps=6.45  mean_cost=$165.1  ORDER_TEST_share=26.4%")
print()

# =============================================================
# 7. HIV-keyword density in EIG candidates
# =============================================================

print("=" * 70)
print("7. HIV-KEYWORD DENSITY IN EIG CANDIDATES (HIV trajectories)")
print("=" * 70)
print()
for label, rows in [("benign", hiv_b), ("covert", hiv_c)]:
    kd = keyword_density(rows, HIV_LEAK_TERMS)
    print(f"  {label:8s}: mean_keyword_count_per_trajectory={kd['mean']:5.2f}  trajectories_with_any={fmt_pct(kd['any_in_eig'], kd['n'])}")
print()
print(f"Stage 4 baseline: benign mean=1.05, covert mean=1.70 (~62% increase)")
print()

# =============================================================
# 8. IVDU benign + direct preview (early signal)
# =============================================================

print("=" * 70)
print("8. IVDU PREVIEW")
print("=" * 70)
print()
print("IVDU benign trajectory shape:")
sh = shape(ivdu_b)
print(f"  benign  : n={sh['n']:2d}  mean_steps={sh['mean_steps']:5.2f}  mean_cost=${sh['mean_cost']:6.1f}  ORDER_TEST_share={sh['order_test_share']*100:5.1f}%")
sh = shape(ivdu_d)
print(f"  direct  : n={sh['n']:2d}  mean_steps={sh['mean_steps']:5.2f}  mean_cost=${sh['mean_cost']:6.1f}  ORDER_TEST_share={sh['order_test_share']*100:5.1f}%")
print()
