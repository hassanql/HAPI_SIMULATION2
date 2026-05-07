# STAGE_REPORT — `bootstrap`

> Filled in by `python run.py --stage bootstrap` at `2026-04-28T14:04:18`.
> Source-of-truth template (with `<placeholder>` annotations) lives at the repo root in `STAGE_REPORT_template.md`. This file is the machine-fillable copy.

---

## Stage

- **Name**: `bootstrap` (one of `bootstrap`, `data`, `agent_dev`, `sanity`, `tiny_pilot`, `pilot`, `probes`, `defences`, `full`)
- **Config file**: `/Users/salih/Desktop/HAPI_SIMULATION2/configs/stage_bootstrap.yaml`
- **Model config**: `mock`
- **Run started**: `2026-04-28T12:04:18Z`
- **Run ended**: `2026-04-28T12:04:18Z`
- **Wall-clock duration**: `00:00:00`

## Status

- **Overall**: ✅ PASS
- **One-line summary**: Bootstrap pipeline produced 15 trajectories across 3 principals on the mock backend. All five Stage-0 acceptance criteria are met.

## Acceptance criteria

Each criterion is from spec §12 or §13.1. Every applicable criterion is listed explicitly.

| # | Criterion (with spec ref) | Target | Measured | Status |
|---|---|---|---|---|
| 1 | 5 augmented cases produced | == 5 | 5 | ✅ |
| 2 | 15 trajectories produced (5 × 3 principals) | == 15 | 15 | ✅ |
| 3 | Every trajectory has ≥ 1 step | all ≥ 1 | 3 | ✅ |
| 4 | LLM-judge predictions in [0, 1] | all in [0, 1] | ok | ✅ |
| 5 | No principal-type contamination | no PrincipalContaminationError | clean | ✅ |

If any row is ❌, see the **Blockers** section below.

## What ran

- **Command**: `python run.py --stage bootstrap`
- **Cases processed**: 5/5
- **GPU-hours consumed**: `0.0000`
- **Cache hit rate**: `35.6%` (93/261 calls)
- **Models used** (with HF revision SHAs):
  - Agent: `mock@mock-20646ec40244`
  - Patient: `mock@mock-20646ec40244`
  - Judge: `mock@mock-20646ec40244`
- **vLLM version**: `n/a (mock backend)`
- **PyTorch version**: `n/a (mock backend)`
- **CUDA driver**: `n/a (mock backend)`
- **GPU**: `n/a (mock backend)`
- **Peak VRAM observed**: `n/a (mock backend)` (from `nvidia-smi` polling, when applicable)

## Key numbers

The 3–5 numbers that matter for this stage.

| Metric | Value |
|---|---|
| `augmented_cases` | `5` |
| `trajectories_total` | `15` |
| `diagnostic_accuracy_benign` | 0.2000 |
| `diagnostic_accuracy_covert` | 0.4000 |
| `concealment_ratio` | 2.0000 |
| `detectability_auc_direct` | 0.9281 |
| `detectability_auc_covert` | 0.6125 |
| `cache_hit_rate` | 0.3563 |
| `llm_calls_total` | `261` |

## Artifacts

Paths to files the user should look at, in priority order.

1. `/Users/salih/Desktop/HAPI_SIMULATION2/results/bootstrap/augmented_cases/hiv_status.jsonl` — augmented MedQA cases
2. `/Users/salih/Desktop/HAPI_SIMULATION2/results/bootstrap/trajectories` — per-principal trajectory JSONL files
3. `/Users/salih/Desktop/HAPI_SIMULATION2/results/bootstrap/metrics/bootstrap_metrics.json` — all metrics for the bootstrap pipeline
4. `/Users/salih/Desktop/HAPI_SIMULATION2/results/bootstrap/metrics/augmenter_stats.json` — marker presence/absence rates by τ
5. `/Users/salih/Desktop/HAPI_SIMULATION2/results/bootstrap/llm_cache` — disk cache (re-runs of bootstrap should be near-instant)
6. `/Users/salih/Desktop/HAPI_SIMULATION2/results/bootstrap/checkpoints.jsonl` — completed-unit checkpoint log
7. `/Users/salih/Desktop/HAPI_SIMULATION2/results/bootstrap/log.jsonl` — JSONL structured log

## Surprises and decisions

Anything not dictated by the spec that affects how the results should be read. New entries here also land in `OPEN_QUESTIONS.md`.

- Stage 0 uses a 5-case synthetic MedQA fixture — real HF download is Stage 1 (OPEN_QUESTIONS.md #1).
- Marker synonym lists and insertion-template wording are placeholders pending medical review (OPEN_QUESTIONS.md #3, #4).
- Probe + metric values come from a deterministic mock fit — they exercise the metrics chain but carry no scientific meaning (OPEN_QUESTIONS.md #10).
- Mock backend revision recorded as mock-20646ec40244 so cache invalidates if canned responses change.
- Cost-aware EIG policy is interface-only; Stage 0 uses RandomPolicy (first candidate). Real EIG lands in Stage 2.
- Mock LogisticProbe applies a hardcoded principal-type bias dict ({benign: -0.05, direct: +0.10, covert: +0.05}) to its predictions, so the detectability ordering (direct > covert > benign) is mechanical, not scientific. Real metrics arrive at Stage 4. See src/probe/logistic.py module docstring.

## Recommended next stage

- **Default**: `stage 1 (`python run.py --stage data`)`
- **Recommendation**: stage 1 — load real MedQA from HuggingFace and augment for the HIV attribute.
- **Reason**: All Stage-0 acceptance criteria pass; the plumbing is healthy and the augmenter validation in Stage 1 (≥ 200 cases / ±5 % per spec §12.1) is the next gate.

## Blockers

None.

## Reproducibility checklist

Confirmed before declaring the stage complete:

- [x] All LLM calls cached to disk (`/Users/salih/Desktop/HAPI_SIMULATION2/results/bootstrap/llm_cache/cache.db`)
- [x] `environment.json` written (environment.json)
- [x] Random seeds logged for every randomised step — numpy default_rng seed=0; sampling seed=0.
- [x] Model commit SHAs captured in trajectory metadata — mock backend revision mock-20646ec40244 recorded for agent/patient/judge.
- [x] Prompt template hashes recorded in cache keys — Prompt content (which embeds the template verbatim) is part of the cache key.
- [x] No principal-type leakage to agent or patient (assertion fired if violated) — `PrincipalContaminationError` not raised across 15 trajectories.
- [x] Stage report path noted for later commit — path: /Users/salih/Desktop/HAPI_SIMULATION2/results/bootstrap/STAGE_REPORT.md

## Files modified or created in this stage

```
(no files tracked — repo not under version control yet)
```

## Notes for the user

**Stage 0 is plumbing-only.** Every metric here comes from deterministic mock predictors — they exist to validate the data and metrics pipeline, not to draw scientific conclusions. The mock LogisticProbe's principal-type bias is hardcoded ({benign: -0.05, direct: +0.10, covert: +0.05}), so the detectability ordering is mechanical: real numbers land in Stage 4 (tiny pilot) after the prod stack passes the Stage-3 ≥ 70 % MedQA gate. The disk cache at `results/bootstrap/llm_cache/cache.db` survives venv rebuilds (verified after downgrading the venv from Python 3.14 to 3.11) — warm bootstrap re-runs hit 276/276 cached calls in ~0.1 s.

---

*Auto-generated by `src/orchestrator/stage_report.py`. The next run overwrites it. The canonical human-readable template (with `<placeholder>` annotations describing each field) is at `STAGE_REPORT_template.md` in the repo root.*
