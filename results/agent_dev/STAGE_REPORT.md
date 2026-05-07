# STAGE_REPORT — `agent_dev`

> Filled in by `python run.py --stage agent_dev` at `2026-04-28T14:39:42`.
> Source-of-truth template (with `<placeholder>` annotations) lives at the repo root in `STAGE_REPORT_template.md`. This file is the machine-fillable copy.

---

## Stage

- **Name**: `agent_dev` (one of `bootstrap`, `data`, `agent_dev`, `sanity`, `tiny_pilot`, `pilot`, `probes`, `defences`, `full`)
- **Config file**: `/home/hassanh_aims_ac_za/HAPI_SIMULATION2/configs/stage_agent_dev.yaml`
- **Model config**: `models_dev`
- **Run started**: `2026-04-28T13:33:55Z`
- **Run ended**: `2026-04-28T14:39:42Z`
- **Wall-clock duration**: `01:05:46`

## Status

- **Overall**: ❌ FAIL
- **One-line summary**: Stage 2 FAIL — 2 blocker(s); see Blockers.

## Acceptance criteria

Each criterion is from spec §12 or §13.1. Every applicable criterion is listed explicitly.

| # | Criterion (with spec ref) | Target | Measured | Status |
|---|---|---|---|---|
| 1 | 20 trajectories produced (§13.1) | == 20 | 20 | ✅ |
| 2 | Mean diagnostic accuracy ≥ 0.50 (dev floor) (§13.1) | ≥ 0.50 | 0.400 (8/20) | ❌ |
| 3 | Mean belief entropy decreases over turns (cohort-level) (§12.2) | non-increasing on average | curve=['1.386', '1.323', '1.240', '1.153', '1.079', '1.031', '0.956', '0.962', '0.952', '0.883', '0.887'] | ❌ |
| 4 | Mean queries per trajectory in [3, 15] (§13.1) | 3 ≤ mean ≤ 15 | 7.60 | ✅ |
| 5 | Action mix sensible (not all-DIAGNOSE / all-ORDER_TEST on turn 1) (§13.1) | ASK_* present; first-action diversity | first_diagnose=0/20, first_test=0/20, totals={'ASK_HISTORY': 115, 'ASK_EXAM': 21, 'DIAGNOSE': 11, 'ORDER_TEST': 5} | ✅ |
| 6 | No principal-type contamination (Appendix C) | no PrincipalContaminationError | clean | ✅ |

If any row is ❌, see the **Blockers** section below.

## What ran

- **Command**: `python run.py --stage agent_dev`
- **Cases processed**: 20/20
- **GPU-hours consumed**: `1.0841`
- **Cache hit rate**: `2.0%` (21/1042 calls)
- **Models used** (with HF revision SHAs):
  - Agent: `Qwen/Qwen3-4B-Instruct-2507@cdbee75f17c01a7cc42f958dc650907174af0554`
  - Patient: `Qwen/Qwen3-4B-Instruct-2507@cdbee75f17c01a7cc42f958dc650907174af0554 (shared)`
  - Judge: `n/a (no judge phase in Stage 2)`
- **vLLM version**: `0.20.0`
- **PyTorch version**: `2.11.0+cu130`
- **CUDA driver**: `595.58.03`
- **GPU**: `Tesla T4`
- **Peak VRAM observed**: `n/a (not polled in Stage 2)` (from `nvidia-smi` polling, when applicable)

## Key numbers

The 3–5 numbers that matter for this stage.

| Metric | Value |
|---|---|
| `n_trajectories` | `20` |
| `diagnostic_accuracy` | 0.4000 |
| `mean_queries_per_trajectory` | 7.6000 |
| `mean_total_cost` | 21.7500 |
| `n_early_diagnose` | `11` |
| `type_counts` | `{"ASK_HISTORY": 115, "ASK_EXAM": 21, "DIAGNOSE": 11, "ORDER_TEST": 5}` |
| `cache_hit_rate` | 0.0202 |
| `llm_calls_total` | `1042` |

## Artifacts

Paths to files the user should look at, in priority order.

1. `/home/hassanh_aims_ac_za/HAPI_SIMULATION2/results/agent_dev/trajectories/hiv_status__benign.jsonl` — 20 benign-principal trajectories (Pydantic-validated)
2. `/home/hassanh_aims_ac_za/HAPI_SIMULATION2/results/agent_dev/llm_cache` — disk cache (warm re-runs are near-instant)
3. `/home/hassanh_aims_ac_za/HAPI_SIMULATION2/results/agent_dev/checkpoints.jsonl` — completed-unit checkpoint log
4. `/home/hassanh_aims_ac_za/HAPI_SIMULATION2/results/agent_dev/log.jsonl` — JSONL structured log

## Surprises and decisions

Anything not dictated by the spec that affects how the results should be read. New entries here also land in `OPEN_QUESTIONS.md`.

- Stage 2 uses vLLM library mode with `Qwen/Qwen3-4B-Instruct-2507` for both agent and patient (one in-process LLM, two distinct system prompts) per spec §4.0 dev configuration.
- EIG implementation combines spec §7.3 step 2 (predict K_resp responses) and step 3 (per-response likelihood scoring) into ONE LLM call per candidate via the `agent_eig_predict.txt` prompt. Information content is identical; saves ~4× LLM calls per turn.
- ε_stop fires inside `EIGPolicy.select`: if max EIG across candidates falls below `epsilon_stop`, the policy returns a `DIAGNOSE` action with `belief.map_estimate()` instead of executing a low-information query.
- Bootstrap (Stage 0) and Stage 1 keep using the mock backend; library mode is selected per stage via `config.backend`.

## Recommended next stage

- **Default**: `stage 3 (`python run.py --stage sanity`)`
- **Recommendation**: Re-run after fixing the failing acceptance criterion.
- **Reason**: Stage 2 must pass before the prod-stack run.

## Blockers

- Mean diagnostic accuracy ≥ 0.50 (dev floor) (target ≥ 0.50, measured 0.400 (8/20))
- Mean belief entropy decreases over turns (cohort-level) (target non-increasing on average, measured curve=['1.386', '1.323', '1.240', '1.153', '1.079', '1.031', '0.956', '0.962', '0.952', '0.883', '0.887'])

## Reproducibility checklist

Confirmed before declaring the stage complete:

- [x] All LLM calls cached (`/home/hassanh_aims_ac_za/HAPI_SIMULATION2/results/agent_dev/llm_cache/cache.db`)
- [x] `environment.json` written (environment.json)
- [x] Random seeds logged — numpy default_rng seed=0; sampling seed=0.
- [x] Model commit SHAs in trajectory metadata — agent=Qwen/Qwen3-4B-Instruct-2507@cdbee75f17c0; patient shares the model.
- [x] Prompt templates referenced in cache key — System+user prompts both contribute to the SHA cache key.
- [x] No principal-type leakage to agent or patient — Asserted by `assemble_agent_chief_complaint` / `assemble_patient_system_prompt`.
- [x] Stage report path noted for later commit — path: /home/hassanh_aims_ac_za/HAPI_SIMULATION2/results/agent_dev/STAGE_REPORT.md

## Files modified or created in this stage

```
(no files tracked — repo not under version control yet)
```

## Notes for the user

Stage 2 ran 20 benign-principal trajectories against Qwen/Qwen3-4B-Instruct-2507 in vLLM library mode. Mean diagnostic accuracy: 0.400; mean queries: 7.60; entropy curve over turns: ['1.386', '1.323', '1.240', '1.153', '1.079', '1.031', '0.956', '0.962', '0.952', '0.883', '0.887']. Trajectories are at `/home/hassanh_aims_ac_za/HAPI_SIMULATION2/results/agent_dev/trajectories`; sample one with `head -1 hiv_status__benign.jsonl | python -m json.tool`. Cache hit rate 2.0% — warm re-runs of identical configs should be near-instant.

---

*Auto-generated by `src/orchestrator/stage_report.py`. The next run overwrites it. The canonical human-readable template (with `<placeholder>` annotations describing each field) is at `STAGE_REPORT_template.md` in the repo root.*
