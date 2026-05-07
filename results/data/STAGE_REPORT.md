# STAGE_REPORT — `data`

> Filled in by `python run.py --stage data` at `2026-05-05T21:30:25`.
> Source-of-truth template (with `<placeholder>` annotations) lives at the repo root in `STAGE_REPORT_template.md`. This file is the machine-fillable copy.

---

## Stage

- **Name**: `data` (one of `bootstrap`, `data`, `agent_dev`, `sanity`, `tiny_pilot`, `pilot`, `probes`, `defences`, `full`)
- **Config file**: `/Users/salih/Desktop/HAPI_SIMULATION2/configs/stage_data.yaml`
- **Model config**: `n/a`
- **Run started**: `2026-05-05T19:30:18Z`
- **Run ended**: `2026-05-05T19:30:25Z`
- **Wall-clock duration**: `00:00:07`

## Status

- **Overall**: ✅ PASS
- **One-line summary**: Loaded 5304 filtered MedQA cases; augmented 4 attribute(s) × 1 correlation_strength(s). All §12.1 acceptance criteria pass.

## Acceptance criteria

Each criterion is from spec §12 or §13.1. Every applicable criterion is listed explicitly.

| # | Criterion (with spec ref) | Target | Measured | Status |
|---|---|---|---|---|
| 1 | MedQA download + filter (§5.1) | 4000 ≤ n ≤ 6000 cases | 5304 | ✅ |
| 2 | ≥ min_cases_per_attribute augmented per (attribute, cs) (§12.1) | ≥ 200 | 5304 | ✅ |
| 3 | τ class balance (each ≥ floor) (§5.3) | both n_pos and n_neg ≥ 80 | n_pos=2617, n_neg=2687 | ✅ |
| 4 | Correct-answer / visible⊆full / unique-IDs invariants (§5.3) | all hold across attributes | ok | ✅ |
| 5 | No structural failures introduced by stripping (vs original) ((this review)) | 0 new malformed sentences | ok | ✅ |
| 6 | LLM-judge structural integrity ≥ 88% yes (sample=50) ((this review)) | min(yes_rate) ≥ 0.88 | 0.900 | ✅ |
| 7 | hiv_status/oral thrush correlation gap ≈ cs (§12.1) | |gap - 0.30| ≤ 0.05 | gap=+0.285 (err=0.015) | ✅ |
| 8 | hiv_status/lymphadenopathy correlation gap ≈ cs (§12.1) | |gap - 0.30| ≤ 0.05 | gap=+0.295 (err=0.005) | ✅ |
| 9 | hiv_status/weight loss correlation gap ≈ cs (§12.1) | |gap - 0.30| ≤ 0.05 | gap=+0.299 (err=0.001) | ✅ |
| 10 | hiv_status/night sweats correlation gap ≈ cs (§12.1) | |gap - 0.30| ≤ 0.05 | gap=+0.289 (err=0.011) | ✅ |
| 11 | hiv_status/low cd4 correlation gap ≈ cs (§12.1) | |gap - 0.30| ≤ 0.05 | gap=+0.303 (err=0.003) | ✅ |
| 12 | iv_drug_use/track marks correlation gap ≈ cs (§12.1) | |gap - 0.30| ≤ 0.05 | gap=+0.285 (err=0.015) | ✅ |
| 13 | iv_drug_use/abscess correlation gap ≈ cs (§12.1) | |gap - 0.30| ≤ 0.05 | gap=+0.303 (err=0.003) | ✅ |
| 14 | iv_drug_use/hep c serology correlation gap ≈ cs (§12.1) | |gap - 0.30| ≤ 0.05 | gap=+0.311 (err=0.011) | ✅ |
| 15 | iv_drug_use/endocarditis correlation gap ≈ cs (§12.1) | |gap - 0.30| ≤ 0.05 | gap=+0.289 (err=0.011) | ✅ |
| 16 | pregnancy/amenorrhea correlation gap ≈ cs (§12.1) | |gap - 0.30| ≤ 0.05 | gap=+0.288 (err=0.012) | ✅ |
| 17 | pregnancy/morning nausea correlation gap ≈ cs (§12.1) | |gap - 0.30| ≤ 0.05 | gap=+0.304 (err=0.004) | ✅ |
| 18 | pregnancy/breast tenderness correlation gap ≈ cs (§12.1) | |gap - 0.30| ≤ 0.05 | gap=+0.313 (err=0.013) | ✅ |
| 19 | pregnancy/positive beta-hcg correlation gap ≈ cs (§12.1) | |gap - 0.30| ≤ 0.05 | gap=+0.287 (err=0.013) | ✅ |
| 20 | mental_health_dx/sleep disturbance correlation gap ≈ cs (§12.1) | |gap - 0.30| ≤ 0.05 | gap=+0.285 (err=0.015) | ✅ |
| 21 | mental_health_dx/anhedonia correlation gap ≈ cs (§12.1) | |gap - 0.30| ≤ 0.05 | gap=+0.307 (err=0.007) | ✅ |
| 22 | mental_health_dx/suicidal ideation correlation gap ≈ cs (§12.1) | |gap - 0.30| ≤ 0.05 | gap=+0.310 (err=0.010) | ✅ |
| 23 | mental_health_dx/prior psychiatric medication correlation gap ≈ cs (§12.1) | |gap - 0.30| ≤ 0.05 | gap=+0.284 (err=0.016) | ✅ |

If any row is ❌, see the **Blockers** section below.

## What ran

- **Command**: `python run.py --stage data`
- **Cases processed**: 5304/5304
- **GPU-hours consumed**: `0.0000`
- **Cache hit rate**: `0.0%` (0/0 calls)
- **Models used** (with HF revision SHAs):
  - Agent: `n/a (no LLM in Stage 1)`
  - Patient: `n/a (no LLM in Stage 1)`
  - Judge: `n/a (no LLM in Stage 1)`
- **vLLM version**: `n/a (mock backend)`
- **PyTorch version**: `n/a (mock backend)`
- **CUDA driver**: `n/a (mock backend)`
- **GPU**: `n/a (mock backend)`
- **Peak VRAM observed**: `n/a (mock backend)` (from `nvidia-smi` polling, when applicable)

## Key numbers

The 3–5 numbers that matter for this stage.

| Metric | Value |
|---|---|
| `n_filtered_cases` | `5304` |
| `n_augmented[hiv_status@cs=0.3]` | `5304` |
| `empirical_base_rate[hiv_status@cs=0.3]` | 0.4934 |
| `mean_corr_gap[hiv_status@cs=0.3]` | 0.2939 |
| `n_augmented[iv_drug_use@cs=0.3]` | `5304` |
| `empirical_base_rate[iv_drug_use@cs=0.3]` | 0.4885 |
| `mean_corr_gap[iv_drug_use@cs=0.3]` | 0.2969 |
| `n_augmented[pregnancy@cs=0.3]` | `5304` |
| `empirical_base_rate[pregnancy@cs=0.3]` | 0.4881 |
| `mean_corr_gap[pregnancy@cs=0.3]` | 0.2980 |
| `n_augmented[mental_health_dx@cs=0.3]` | `5304` |
| `empirical_base_rate[mental_health_dx@cs=0.3]` | 0.4942 |
| `mean_corr_gap[mental_health_dx@cs=0.3]` | 0.2964 |

## Artifacts

Paths to files the user should look at, in priority order.

1. `/Users/salih/Desktop/HAPI_SIMULATION2/data/augmented/_validation.json` — augmenter validation summary (per-finding rates + acceptance)
2. `/Users/salih/Desktop/HAPI_SIMULATION2/data/augmented/hiv_status_0.3.jsonl` — augmented cases for hiv_status@0.3
3. `/Users/salih/Desktop/HAPI_SIMULATION2/data/augmented/iv_drug_use_0.3.jsonl` — augmented cases for iv_drug_use@0.3
4. `/Users/salih/Desktop/HAPI_SIMULATION2/data/augmented/pregnancy_0.3.jsonl` — augmented cases for pregnancy@0.3
5. `/Users/salih/Desktop/HAPI_SIMULATION2/data/augmented/mental_health_dx_0.3.jsonl` — augmented cases for mental_health_dx@0.3
6. `/Users/salih/Desktop/HAPI_SIMULATION2/data/raw/medqa_filtered_v2.jsonl` — filtered MedQA cache (re-runs skip the HF download)
7. `/Users/salih/Desktop/HAPI_SIMULATION2/results/data/metrics/augmenter_validation.json` — duplicate of validation summary inside the stage's results dir

## Surprises and decisions

Anything not dictated by the spec that affects how the results should be read. New entries here also land in `OPEN_QUESTIONS.md`.

- Spec §5.1 prescribed `bigbio/med_qa` with config `med_qa_en_source`, but `datasets >= 4.x` rejects script-based datasets. Loaded from `GBaker/MedQA-USMLE-4-options` instead — same MedQA-USMLE source content. See OPEN_QUESTIONS.md #18.
- Per-finding acceptance is the **correlation gap** `P(present|τ=1) - P(present|τ=0)`, which equals cs algebraically regardless of natural marker prevalence. The looser literal reading of §12.1 ('rate ≈ cs / rate ≈ 1-cs') would fail on real MedQA because HIV markers are rare in untouched cases. See OPEN_QUESTIONS.md #19.
- Filtered MedQA cached at `data/raw/medqa_filtered_<LOADER_VERSION>.jsonl`; subsequent stage re-runs skip the network download.

## Recommended next stage

- **Default**: `stage 2 (`python run.py --stage agent_dev`)`
- **Recommendation**: stage 2 — first time the cost-aware EIG agent loop runs end-to-end on the dev model.
- **Reason**: Augmented JSONL is the input to every later stage; gating Stage 2 on a passing Stage 1 keeps downstream signal interpretable.

## Blockers

None.

## Reproducibility checklist

Confirmed before declaring the stage complete:

- [x] Filtered MedQA cached on disk (`data/raw/medqa_filtered.jsonl`)
- [x] `environment.json` written (environment.json)
- [x] Random seed logged — numpy default_rng seed=0
- [x] MedQA source + split recorded in validation summary — GBaker/MedQA-USMLE-4-options, split=train
- [x] Augmented JSONL filenames carry the (attribute, cs) tuple — hiv_status_0.3.jsonl, iv_drug_use_0.3.jsonl, pregnancy_0.3.jsonl, mental_health_dx_0.3.jsonl
- [x] Stage report path noted for later commit — path: /Users/salih/Desktop/HAPI_SIMULATION2/results/data/STAGE_REPORT.md

## Files modified or created in this stage

```
(no files tracked — repo not under version control yet)
```

## Notes for the user

Stage 1 augmented 5304 HIV cases at correlation_strength 0.3; per-finding correlation gap converges within ±0.05 of cs as expected. τ=1 marker-count distribution: mean=1.54 markers/case, buckets={'0': 420, '1': 909, '2': 831, '3': 378, '4': 75, '5': 4} Other attributes (IVDU, pregnancy, mental_health) come online in Stage 5 — only HIV is augmented here per spec §13.1. The augmenter validation file at `data/augmented/_validation.json` is the single artifact to read first.

---

*Auto-generated by `src/orchestrator/stage_report.py`. The next run overwrites it. The canonical human-readable template (with `<placeholder>` annotations describing each field) is at `STAGE_REPORT_template.md` in the repo root.*
