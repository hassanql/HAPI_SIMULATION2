# A2 Simulation: Adversarial-Principal Attribute Extraction

Research simulation for the MSc thesis "Task Laundering: Covert Attribute Extraction in Human-Querying Diagnostic Agents." Empirically demonstrates that an adversarial principal can recover sensitive attributes from a patient by commissioning a benign-looking clinical task whose Bayes-optimal sensing policy happens to maximise information gain about the attribute. The agent is honest, the patient answers truthfully, every individual query is clinically appropriate — and yet the attribute leaks.

The full technical specification lives in [`SIMULATION_SPEC.md`](SIMULATION_SPEC.md). The kickoff brief and staged-delivery contract are in [`CLAUDE_CODE_KICKOFF.md`](CLAUDE_CODE_KICKOFF.md). Decisions made during implementation are logged in [`OPEN_QUESTIONS.md`](OPEN_QUESTIONS.md).

## Quickstart

This repo runs on **Python 3.11+** (validated on 3.14.3). All Stage 0 work runs without a GPU.

```bash
# 1. Create venv and install
python3.14 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# 2. List the nine stages and their status
python run.py --list

# 3. Run the bootstrap (no GPU, < 5 min)
python run.py --stage bootstrap

# 4. Read the report
cat results/bootstrap/STAGE_REPORT.md
```

## The nine stages

Each stage runs as `python run.py --stage <name>`. The orchestrator manages vLLM subprocesses internally — never run `vllm serve` by hand, never open a second terminal.

| # | Stage | Command | Model config | Time | Gate |
|---|---|---|---|---|---|
| 0 | bootstrap | `python run.py --stage bootstrap` | mock | < 5 min | Tests pass; mock trajectories well-formed |
| 1 | data | `python run.py --stage data` | none | < 5 min | Marker presence rates within ±5 % of correlation strength |
| 2 | agent_dev | `python run.py --stage agent_dev` | dev | 20–40 min | Dev-model agent ≥ 50 % MedQA accuracy |
| 3 | sanity | `python run.py --stage sanity` | prod | 1–2 h | **≥ 70 % raw-MedQA accuracy on prod stack (hard gate)** |
| 4 | tiny_pilot | `python run.py --stage tiny_pilot` | prod | 1–2 h | Covert ASR materially > benign incidental |
| 5 | pilot | `python run.py --stage pilot` | prod | 6–10 h | All 1800 trajectories produced |
| 6 | probes | `python run.py --stage probes` | prod (judge) | 30–60 min | Logistic probe AUC ≥ 0.8 on direct |
| 7 | defences | `python run.py --stage defences` | prod | 4–8 h | Workflow audit beats agent-side defences |
| 8 | full | `python run.py --stage full` | prod | 30–50 h | Optional |

## Other entry-point flags

```bash
python run.py --list                       # show stages + current status
python run.py --stage pilot --resume       # resume after interruption
python run.py --stage pilot --dry-run      # estimate runtime, exit
```

## Layout

See spec §3. Briefly:

```
configs/         per-stage YAML configs + dev/prod model configs
prompts/         LLM prompt templates (load_prompt helper)
src/data/        MedQA loader, attribute schema, augmenter
src/agent/       action space, belief, EIG policy, runner
src/patient/     patient simulator
src/principal/   benign / direct / covert task generators
src/probe/       logistic + LLM-judge probes
src/metrics/     ASR-at-K, bits-per-$, concealment, detectability
src/llm/         unified client, mock backend, vLLM server manager, cost tracker
src/orchestrator/ stage registry, STAGE_REPORT writer, checkpoints, preflight
tests/           pytest suite (no GPU; mock backend only)
templates/       STAGE_REPORT_template.md
results/         one subdirectory per stage; gitignored
```

## Deployment

Stage 0 runs on a laptop. From Stage 3 onwards the experiment runs on GCP (`a2-ultragpu-1g`, single A100 80 GB). See [`README_GCP.md`](README_GCP.md) for the deployment guide.

## Contributing

This is a staged delivery: each stage is gated by a `STAGE_REPORT.md` and human approval. Implementations should respect the spec's load-bearing constraints (single-command execution, dev/prod split, deterministic generation, principal contamination prohibition, disk caching). When the spec is silent on a detail, document the decision in `OPEN_QUESTIONS.md`.
