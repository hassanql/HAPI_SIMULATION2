# STAGE_REPORT — `sanity`

> Filled in by `python run.py --stage sanity` at `2026-04-29T14:43:31`.
> Source-of-truth template (with `<placeholder>` annotations) lives at the repo root in `STAGE_REPORT_template.md`. This file is the machine-fillable copy.

---

## Stage

- **Name**: `sanity` (one of `bootstrap`, `data`, `agent_dev`, `sanity`, `tiny_pilot`, `pilot`, `probes`, `defences`, `full`)
- **Config file**: `/home/hassanh_aims_ac_za/HAPI_SIMULATION2/configs/stage_sanity.yaml`
- **Model config**: `models_prod`
- **Run started**: `2026-04-29T13:14:54Z`
- **Run ended**: `2026-04-29T14:43:28Z`
- **Wall-clock duration**: `01:28:34`

## Status

- **Overall**: ✅ PASS
- **One-line summary**: Prod-stack sanity benchmark: 10 raw-MedQA trajectories, accuracy 0.700 (95 % CI [0.700, 0.700]), mean 6.9 queries/case. All Stage-3 acceptance criteria pass — load-bearing 70 % gate met.

## Acceptance criteria

Each criterion is from spec §12 or §13.1. Every applicable criterion is listed explicitly.

| # | Criterion (with spec ref) | Target | Measured | Status |
|---|---|---|---|---|
| 1 | 10 trajectories produced (§13.1) | == 10 | 10 | ✅ |
| 2 | Mean diagnostic accuracy ≥ 0.70 (HARD GATE) (§12.2) | ≥ 0.70 | 0.700 (7/10); 95 % bootstrap CI [0.700, 0.700] | ✅ |
| 3 | Mean queries per trajectory in [3, 15] (§13.1) | 3 ≤ mean ≤ 15 | 6.90 | ✅ |
| 4 | No principal-type contamination (Appendix C) | no PrincipalContaminationError | clean | ✅ |

If any row is ❌, see the **Blockers** section below.

## What ran

- **Command**: `python run.py --stage sanity`
- **Cases processed**: 10/10
- **GPU-hours consumed**: `1.4239`
- **Cache hit rate**: `14.6%` (70/479 calls)
- **Models used** (with HF revision SHAs):
  - Agent: `Qwen/Qwen3-30B-A3B-Instruct-2507-FP8@5a5a776300a41aaa681dd7ff0106608ef2bc90db`
  - Patient: `google/gemma-4-26B-A4B-it@4c55b528bdc40b4e79ed7fd4e2f8e46fa5aaed5a`
  - Judge: `n/a (not loaded — Stage 6 owns the judge)`
- **vLLM version**: `0.20.0`
- **PyTorch version**: `2.11.0+cu130`
- **CUDA driver**: `580.126.09`
- **GPU**: `NVIDIA A100-SXM4-80GB`
- **Peak VRAM observed**: `n/a (not polled in Stage 3)` (from `nvidia-smi` polling, when applicable)

## Key numbers

The 3–5 numbers that matter for this stage.

| Metric | Value |
|---|---|
| `n_trajectories` | `10` |
| `diagnostic_accuracy` | 0.7000 |
| `accuracy_ci_low` | 0.7000 |
| `accuracy_ci_high` | 0.7000 |
| `mean_queries_per_trajectory` | 6.9000 |
| `mean_total_cost` | 19.7000 |
| `type_counts` | `{"ASK_HISTORY": 57, "ASK_EXAM": 8, "ORDER_TEST": 2, "DIAGNOSE": 2}` |
| `cache_hit_rate` | 0.1461 |
| `llm_calls_total` | `479` |

## Artifacts

Paths to files the user should look at, in priority order.

1. `/home/hassanh_aims_ac_za/HAPI_SIMULATION2/results/sanity/trajectories/raw_medqa__benign.jsonl` — 10 raw-MedQA benign-principal trajectories
2. `/home/hassanh_aims_ac_za/HAPI_SIMULATION2/results/sanity/llm_cache` — disk cache (warm re-runs are near-instant)
3. `/home/hassanh_aims_ac_za/HAPI_SIMULATION2/results/sanity/checkpoints.jsonl` — completed-unit checkpoint log
4. `/home/hassanh_aims_ac_za/HAPI_SIMULATION2/results/sanity/log.jsonl` — JSONL structured log
5. `/home/hassanh_aims_ac_za/HAPI_SIMULATION2/results/sanity/vllm_agent.log` — vLLM stdout/stderr for the agent server
6. `/home/hassanh_aims_ac_za/HAPI_SIMULATION2/results/sanity/vllm_patient.log` — vLLM stdout/stderr for the patient server

## Surprises and decisions

Anything not dictated by the spec that affects how the results should be read. New entries here also land in `OPEN_QUESTIONS.md`.

- Stage 3 uses vLLM **server mode** with two concurrent HTTP endpoints — agent on port 8001, patient on port 8002 — both managed by `ServerPool`. Library mode (Stage 2) is single-model in-process; server mode is required when agent and patient are different models.
- Models: agent=`Qwen/Qwen3-30B-A3B-Instruct-2507-FP8` (spec asked for `Qwen/Qwen3.6-35B-A3B-FP8`), patient=`google/gemma-4-26B-A4B-it` (spec asked for `google/gemma-4-26B-A4B-it`). Both swapped to confirmed-real HF repos per OPEN_QUESTIONS.md #31 lesson.
- Test data is **raw MedQA** (no Stage-1 augmentation, no marker stripping) — Stage 3 measures the agent's clean MedQA capability. Stage 4+ uses augmented cases for attack measurement.
- Hard gate: accuracy ≥ 0.70. If this fails, Stages 4-8 are blocked because no attack measurement is meaningful when the agent can't reach basic-competence diagnostic accuracy.

## Recommended next stage

- **Default**: `stage 4 (`python run.py --stage tiny_pilot`)`
- **Recommendation**: stage 4 — first stage that actually exercises the privacy attack (covert principal).
- **Reason**: Stage 3 verified prod-stack baseline diagnostic capability; Stage 4 is the first time the covert principal runs and the privacy attack is measured.

## Blockers

None.

## Reproducibility checklist

Confirmed before declaring the stage complete:

- [x] All LLM calls cached (`/home/hassanh_aims_ac_za/HAPI_SIMULATION2/results/sanity/llm_cache/cache.db`)
- [x] `environment.json` written (environment.json)
- [x] Random seeds logged — numpy default_rng seed=0; sampling seed=0.
- [x] Model commit SHAs in trajectory metadata — agent=Qwen/Qwen3-30B-A3B-Instruct-2507-FP8@5a5a776300a4; patient=google/gemma-4-26B-A4B-it@4c55b528bdc4.
- [x] Prompt templates referenced in cache key — System+user prompts both contribute to the SHA cache key.
- [x] No principal-type leakage to agent or patient — Asserted by `assemble_agent_chief_complaint` / `assemble_patient_system_prompt`.
- [x] Stage report path noted for later commit — path: /home/hassanh_aims_ac_za/HAPI_SIMULATION2/results/sanity/STAGE_REPORT.md

## Files modified or created in this stage

```
(no files tracked — repo not under version control yet)
```

## Notes for the user

Stage 3 ran 10 raw-MedQA trajectories on the prod stack. Mean diagnostic accuracy: 0.700 (95 % CI [0.700, 0.700]); mean queries: 6.90. Trajectories are at `/home/hassanh_aims_ac_za/HAPI_SIMULATION2/results/sanity/trajectories`. vLLM server logs at `/home/hassanh_aims_ac_za/HAPI_SIMULATION2/results/sanity/vllm_*.log`. Cache hit rate 14.6% — warm re-runs of identical configs should be near-instant.

---

*Auto-generated by `src/orchestrator/stage_report.py`. The next run overwrites it. The canonical human-readable template (with `<placeholder>` annotations describing each field) is at `STAGE_REPORT_template.md` in the repo root.*
