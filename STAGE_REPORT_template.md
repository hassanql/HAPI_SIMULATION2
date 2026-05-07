# STAGE_REPORT — `<stage_name>`

> Filled in by the stage handler at the end of every stage. Copy this template to `results/<stage>/STAGE_REPORT.md` and replace each `<placeholder>` with the actual value. Do not leave placeholders unfilled — if a value isn't applicable, write `n/a` and explain why.

---

## Stage

- **Name**: `<stage_name>` (e.g., `bootstrap`, `data`, `agent_dev`, `sanity`, `tiny_pilot`, `pilot`, `probes`, `defences`, `full`)
- **Config file**: `configs/stage_<name>.yaml`
- **Model config**: `<dev | prod | mock | none>`
- **Run started**: `<ISO 8601 timestamp>`
- **Run ended**: `<ISO 8601 timestamp>`
- **Wall-clock duration**: `<HH:MM:SS>`

## Status

- **Overall**: ✅ PASS / ❌ FAIL / ⚠️ PARTIAL
- **One-line summary**: `<one sentence — what happened, what passed, what's blocking>`

## Acceptance criteria

Each criterion is from Section 12 or Section 13.1 of the spec. List every applicable criterion explicitly — do not summarise.

| # | Criterion (with spec ref) | Target | Measured | Status |
|---|---|---|---|---|
| 1 | `<e.g. "Marker-presence rate within ±5% of correlation strength (§12.1)">` | `<e.g. 0.80 ± 0.05>` | `<e.g. 0.78>` | ✅ / ❌ |
| 2 | ... | | | |

If any criterion is ❌, explain in the **Blockers** section below.

## What ran

- **Command**: `<exact command, e.g. "python run.py --stage tiny_pilot">`
- **Cases processed**: `<n>` (of `<total>` in config)
- **GPU-hours consumed**: `<float>`
- **Cache hit rate**: `<percentage>` (cold runs should be 0%, warm should be > 95%)
- **Models used**: list with HF revision SHAs
  - Agent: `<repo_id>@<sha>`
  - Patient: `<repo_id>@<sha>`
  - Judge: `<repo_id>@<sha>` (n/a unless this is the probes stage or later)
- **vLLM version**: `<version>`
- **PyTorch version**: `<version>`
- **CUDA driver**: `<version>`
- **GPU**: `<model, e.g. NVIDIA A100-SXM4-80GB>`
- **Peak VRAM observed**: `<GB>` (from `nvidia-smi` polling, written to `vram_log.jsonl`)

## Key numbers

The 3–5 numbers that matter for this stage. Pick the ones a reviewer would want to see first. Examples by stage:

- **bootstrap**: number of mock trajectories produced; tests passed/total
- **data**: cases retained after filtering; marker-presence rate per τ value; base rate sampled
- **agent_dev**: MedQA accuracy on dev model; mean steps per case; mean entropy reduction across turns
- **sanity**: MedQA accuracy on prod model with bootstrap 95% CI; mean trajectory length; mean cost per trajectory
- **tiny_pilot**: ASR-at-K=15 for benign / direct / covert; concealment ratio; bits-per-dollar
- **pilot**: ASR-at-K curves per attribute; concealment ratios; detectability AUC for direct vs covert
- **probes**: logistic probe AUC per attribute; LLM-judge probe AUC per attribute; agreement between probes
- **defences**: defence Pareto frontier — leakage reduction vs utility loss for each defence

| Metric | Value | Notes |
|---|---|---|
| `<metric>` | `<value>` | `<one-line context>` |

## Artifacts

Paths to files the user should look at, in priority order. Be specific — point to the most informative artifact first.

1. `<path>` — `<one-line description, e.g. "leakage-vs-budget plot, three lines per principal">`
2. `<path>` — ...
3. ...

## Surprises and decisions

Anything you encountered that wasn't dictated by the spec, in chronological order. If you made a decision that the spec was silent on, log it here AND append it to `OPEN_QUESTIONS.md` so the user can review it across stages.

- `<surprise or decision>`: `<one-paragraph explanation, including the option you chose and why>`
- `<surprise or decision>`: ...

If nothing of note happened, write "Nothing notable."

## Recommended next stage

- **Default**: `<next stage name, e.g. "stage_2_agent_dev">`
- **Recommendation**: `<usually the default; explain if recommending otherwise>`
- **Reason**: `<one sentence>`

## Blockers

If the stage failed, list every blocker. Each blocker should be specific enough that the user can act on it.

- `<blocker>`: `<what's wrong, what would fix it, whether you tried something already>`

If the stage passed, write "None."

## Reproducibility checklist

Confirm each item before declaring the stage complete:

- [ ] All LLM calls cached to disk (cache directory: `<path>`)
- [ ] `environment.json` written to results directory with full version stack
- [ ] Random seeds logged for every randomised step (numpy, Python random, attribute sampler)
- [ ] Model commit SHAs captured in trajectory metadata, not just model names
- [ ] Prompt template hashes recorded in cache keys
- [ ] No principal-type leakage to agent or patient (assertion fired if violated)
- [ ] Stage report committed to git alongside the artifacts (or path noted for later commit)

## Files modified or created in this stage

```
<paste the output of `git status --short` here, or list manually>
```

## Notes for the user

Free-form section. Mention anything that would be useful for the user to know before approving the next stage that doesn't fit elsewhere — e.g. unexpected token usage, weird LLM behaviour, parts of the spec that were ambiguous, suggestions for refining future stages.
