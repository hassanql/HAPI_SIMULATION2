# Kickoff prompt for Claude Code (staged delivery, single-command execution)

Paste the prose below into Claude Code's first message, with `SIMULATION_SPEC.md` and `STAGE_REPORT_template.md` available in the working directory.

---

I'm building a research simulation for my MSc thesis on AI safety, supervised by Umang Bhatt, targeting AAAI submission at the end of summer 2026. The full technical specification is in `SIMULATION_SPEC.md` — read it end to end before writing any code, especially **Section 13 (the nine-stage implementation plan)** which defines exactly how this work is structured.

## How this delivery works

This is a **staged delivery, not a one-shot delivery.** Section 13 of the spec defines nine stages. You will implement them one at a time. Between stages, I review what you produced and approve advancing to the next stage. You do not advance on your own.

This is a deliberate choice: the previous attempt at one-shot delivery took too long and made it impossible to catch problems early. A research simulation has too many subtle correctness issues — a wrong belief update or a leaky principal will produce code that runs cleanly but generates meaningless trajectories. Staged delivery lets us catch those at the cheapest possible point.

**Two design constraints that shape everything**:

1. **Single-command execution.** Every stage must be runnable as `python run.py --stage <name>` and nothing else. The orchestrator (described in spec Sections 3.1 and 4.4) manages vLLM servers, health checks, and shutdown internally as Python subprocesses. I should never need to open multiple terminals or run `vllm serve` by hand. `tmux` is purely for keeping SSH sessions alive on GCP, not for managing the experiment itself.

2. **Dev/prod model split.** Section 4.0 of the spec defines two model configurations: dev (Qwen3-4B for both agent and patient, runs on any GPU) for plumbing-and-logic stages, and prod (Qwen3.6-35B-A3B + Gemma 4 26B + Phi-4) for scientific stages on the A100 80 GB. The split is essential because most bugs surface with the small models, and we shouldn't burn A100 time finding them. The orchestrator picks the model config based on the stage; you don't hardcode model IDs in code, you read them from `configs/models_dev.yaml` or `configs/models_prod.yaml`.

## Deployment context

The codebase will be deployed to a Google Cloud Platform VM (`a2-ultragpu-1g`, single A100 80 GB) and the experiment will run there. I will not be debugging interactively — when a stage runs on GCP, I see only the stage report and the artifacts. So: every stage must be self-validating, and every error message must be actionable without you present.

A100 instances are roughly USD 3.67/hour. A 6-hour pilot is ~$22; a 30-hour full run is ~$110. Be respectful of this — defensive coding and good preflight checks pay for themselves immediately. The dev-mode stages let me develop and test without spinning up the A100 at all, which is the whole point of the dev/prod split.

## Required deliverables (the whole project, but built in stages)

When the project is complete (after all approved stages), the repository must contain:

1. **`run.py`** at the repo root — the single entry point. `python run.py --stage <name>` is the only command needed for any stage. Also supports `--list`, `--resume`, `--dry-run`.

2. **The full codebase** matching the layout in Section 3 of the spec, including `src/orchestrator/`, `src/llm/server_manager.py`, and `src/llm/mock_backend.py`.

3. **`README.md`** — quickstart for someone who has never seen the project. Brief description, link to the spec, the nine `python run.py --stage ...` commands in order.

4. **`README_GCP.md`** — step-by-step GCP deployment guide, written for someone who knows GCP basics but not this project. Must cover instance creation (`a2-ultragpu-1g` in a zone with A100 80 GB availability), Deep Learning VM image with CUDA 12.4+, ≥ 500 GB persistent SSD for model cache, HuggingFace token setup, the exact `gcloud compute instances create` command, `tmux` usage to survive SSH disconnections, `gcloud compute scp` to pull results back, and a cost-estimate table for each stage.

5. **`scripts/setup_gcp.sh`** — idempotent setup script for a fresh Deep Learning VM. Installs pinned dependencies, downloads model weights, leaves the environment ready for `python run.py --stage bootstrap` to succeed.

6. **`templates/STAGE_REPORT_template.md`** — already provided; copy it into the repo. The orchestrator copies this into each stage's results directory at stage start and the stage handler fills it in at stage end.

7. **`OPEN_QUESTIONS.md`** at the repo root — running log of every non-trivial decision you made that the spec didn't dictate, with one-line rationale per entry. New entries appended at each stage. I review this periodically.

8. **`pyproject.toml`** with every dependency pinned to an exact version (not `>=`). Lock file (`uv.lock` or `requirements.txt` from `pip freeze`) committed too.

9. **`tests/`** — pytest suite covering every component listed in Section 12 of the spec. Tests must run without any GPU using the mock LLM backend.

10. **`configs/`** — one config per stage as specified in Section 3 of the spec, plus `models_dev.yaml` and `models_prod.yaml` for the model split.

## Working mode (per stage)

For each stage:

1. **Read the spec section that defines the stage** (Section 13.1 has the per-stage details). Re-read Sections 3, 4, and 11 every time — these describe the orchestrator, model split, and reproducibility requirements that apply to every stage.

2. **Implement only what's needed for this stage.** Do not pre-build modules for later stages. If Stage 2 needs a probe and the probe doesn't exist yet, that's expected — Stage 6 builds it. Stages are vertical slices of the full pipeline; later stages add capabilities, they don't refactor earlier ones.

3. **Make decisions and document them.** When the spec is silent (exact prompt wording, threshold values, edge-case behaviour), choose a sensible default and add an entry to `OPEN_QUESTIONS.md`. Do not block on questions.

4. **Build defensively.** GPU OOM, model download failures, HuggingFace rate limits, malformed LLM outputs (the LLM will sometimes ignore your JSON schema), timeouts, disk-full conditions, SIGTERM handlers that flush caches and checkpoints. Resumability is mandatory from Stage 5 onwards: `--resume` must pick up at the case after the last completed one, not at case 1.

5. **Fill in `STAGE_REPORT.md`** at the end of the stage using `templates/STAGE_REPORT_template.md`. Every section. Every placeholder replaced. The status field must be honest — if an acceptance criterion fails, status is ❌ FAIL or ⚠️ PARTIAL, not ✅.

6. **Stop.** Do not start the next stage. Wait for me to review the report and explicitly approve advancing. I will reply with either "approved, proceed to stage X" or specific feedback to address before re-running.

## Critical rules from the spec

These are the load-bearing constraints. They apply to every stage. Don't let any of them slip:

- **Sanity benchmark (Section 12.2).** The agent must reach ≥ 70% accuracy on raw MedQA under the benign principal in Stage 3 before any attack experiment is meaningful. This is the load-bearing acceptance criterion. If Stage 3 fails this gate, all later stages are blocked until it's fixed.

- **Caching (Section 11).** Disk-cache every LLM call from Stage 0 onwards, keyed by `(model_revision, prompt_template_hash, prompt_content, sampling_params, seed)`. The cache must survive crashes and stage restarts. Without this, every re-run on GCP burns hours of GPU time.

- **Determinism.** `temperature=0.0`, `top_p=1.0`, `seed=0`, `--enforce-eager` on every vLLM server. Pin the exact vLLM version (≥ 0.19.0 mandatory for Qwen3.6-35B-A3B). Record the model commit SHA in trajectory metadata, never just the model name.

- **No principal contamination (Appendix C).** The patient simulator and the agent must never see `TaskSpec.principal_type`. Add an assertion in the prompt-assembly code that fails loudly if this leaks. Test this in Stage 0 with the mock backend.

- **Single-command execution.** `python run.py --stage <name>` is the only command. The orchestrator manages vLLM servers internally. I should never run `vllm serve` myself, never open a second terminal, never edit a port number.

- **Dev/prod split.** Stages 0–2 use dev models. Stages 3–8 use prod models. The mapping is in Section 4.0 of the spec. Do not hardcode model IDs in source files — read them from the model config YAML.

- **Out of scope (Appendix D).** Do not implement memory extraction, multi-human attacks, behavioural orchestration, or human-study tooling. If you find yourself reaching for one of these, stop and re-read the spec.

## Stage 0 specifically (your first task)

Start with Stage 0 (bootstrap). Its purpose is to prove the architecture works end-to-end without any GPU. Specifically Stage 0 must deliver:

- The full repo skeleton matching Section 3 of the spec.
- `run.py` at the repo root with the `--stage`, `--list`, `--resume`, `--dry-run` flags wired up. Listing should show all nine stages and their current status (not-run / completed / failed).
- The orchestrator (`src/orchestrator/stages.py`) with a stub registered for every stage. Only the bootstrap stage needs to actually do something; the rest can raise `NotImplementedError("Stage X not yet implemented")` cleanly.
- The LLM client abstraction (`src/llm/client.py`) with the disk cache wired in.
- A mock LLM backend (`src/llm/mock_backend.py`) that returns deterministic canned responses for the prompts the bootstrap stage will use. The mock should be good enough to walk a fake case through data prep → all three principals → fake agent loop → fake probe → fake metrics. Its purpose is to test the *plumbing*, not to produce meaningful results.
- The data preparation logic for the HIV attribute (Section 5 of the spec), tested with the augmenter unit tests from Section 12.1.
- Stage 0's config at `configs/stage_bootstrap.yaml` and stage handler.
- All unit tests passing (covering augmenter, belief, mock backend, orchestrator).
- `STAGE_REPORT.md` filled in at `results/bootstrap/STAGE_REPORT.md`.
- `OPEN_QUESTIONS.md` at the repo root with whatever you decided that the spec was silent on.
- `README.md` and the start of `README_GCP.md` (the GCP guide can be filled out incrementally — Stage 0 just needs the structure).
- `pyproject.toml` with dependencies pinned.

What Stage 0 does NOT need:

- Any actual model weights downloaded.
- Any vLLM serving (the `server_manager.py` module needs to exist and have its interface defined, but doesn't need to be tested live until Stage 3).
- Any GPU.
- The covert principal logic implemented (it's stubbed; the real implementation comes in Stage 4).
- The actual EIG policy (random action selection is fine for the mock; cost-aware EIG comes in Stage 2).
- The real probes, real metrics, or any figures.

## Decision policy

- **Spec is explicit** → follow it, even if you disagree. Note disagreement in `OPEN_QUESTIONS.md`.
- **Spec is silent and choice is small** (default values, log formats, temp paths) → pick something reasonable, document it.
- **Spec is silent and choice is large** (architectural decisions, library substitutions, deviation from metric definitions) → pick the option closest to the spec's intent, document it, proceed.
- **Spec contradicts itself** → document, choose the more conservative interpretation (i.e., the one that makes the experimental result harder to claim positive), proceed.

Never delay shipping a stage on an open question. Document and continue.

## Quality bar

This will be reviewed by Umang and may be evaluated for paper publication. The code does not need to be production polished but it must be:

- **Type-hinted** throughout, especially at module boundaries.
- **Pydantic-validated** for every data structure that crosses an LLM call.
- **Logged structurally** (JSONL) so I can grep failed runs.
- **Documented** at the function level for non-obvious logic (EIG estimation, belief update, covert task generation).
- **Tested** at the unit level using the mock backend, with no GPU dependencies in the unit test suite.

## What success looks like (per stage)

After Stage 0 (and analogously for every later stage):

1. I can clone the repo to my laptop, run `pip install -e .`, run `python run.py --list`, see the nine stages and their status, then run `python run.py --stage bootstrap` and see ✅ PASS in under 5 minutes without a GPU.
2. I can read `STAGE_REPORT.md` and understand exactly what was built, what was tested, what passed, what was deferred.
3. I can read `OPEN_QUESTIONS.md` and either approve or override your decisions.
4. I have enough confidence in the foundation that approving Stage 1 feels low-risk.

## First action — your response to this kickoff

Do not start coding immediately. First respond with:

1. **A two-paragraph summary** of what you understood the project to be, so I can catch misreadings.
2. **What Stage 0 specifically will deliver**, broken down by file. For each file you'll create in Stage 0, one line describing its purpose and an estimate of its size in lines.
3. **Top 5–10 spec ambiguities you noticed during the read** that affect Stage 0, with your planned resolution for each (these become the seed for `OPEN_QUESTIONS.md`).
4. **Estimated runtime** for you to deliver Stage 0 (so I can decide whether it fits in this session or needs to be split).
5. **Anything you need from me before starting** — HuggingFace token, GCP project ID, etc. Most likely the answer is "nothing for Stage 0, since it doesn't touch real models", but ask once.

Wait for my approval before writing code. Once I approve, build Stage 0 and only Stage 0. End your final response for Stage 0 with the filled-in `STAGE_REPORT.md` content and a clear "Stage 0 complete, awaiting approval to proceed to Stage 1."
