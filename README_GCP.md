# GCP Deployment Guide

This guide covers deploying the A2 Simulation to GCP for **Stage 2** (small-GPU dress rehearsal on a T4) and **Stages 3–8** (production stack on an A100 80 GB).

- **Stages 0–1** run on a laptop. No GPU.
- **Stage 2** is the first stage that needs a GPU. **Required**: GCP `n1-standard-4` (4 vCPU / 15 GB RAM) with one **NVIDIA T4 16 GB**. *Not recommended — required.* `n1-standard-1` (1 vCPU / 3.6 GB RAM) — the GCP web-console default — OOM-kills the HuggingFace download at 22–33 % completion because vLLM stages the safetensors in CPU RAM before transferring to GPU. See OPEN_QUESTIONS.md #36. Total cost for the dress rehearsal is small (≈ $0.50–1.50).
- **Stages 3–8** run on `a2-ultragpu-1g` with one **NVIDIA A100 80 GB**.

---

## 1 · Prerequisites

- GCP account with billing enabled and the **Compute Engine API** turned on in your project.
- A **HuggingFace token** for gated checkpoints (Qwen3-4B is currently public; Qwen3.6 / Gemma 4 / Phi-4 may be gated for some accounts — get one regardless).
- Familiarity with `gcloud`, `tmux`, and `ssh`.

---

## 2 · Stage 2 — small-GPU dress rehearsal (T4 16 GB)

### 2.1 Memory math for Qwen3-4B-Instruct-2507 on T4 16 GB (bf16)

| Component | Size |
|---|---:|
| Weights (4B params × 2 bytes, bf16) | **~ 9.0 GB** |
| KV cache (`max_model_len=4096`, `max_num_seqs=4` → 16,384 token slots; 32 layers × 8 KV heads × 96 head_dim × 2 bytes × 2 KV per layer ≈ 96 KB / token) | **~ 1.5 GB** |
| Activations + vLLM framework + scheduler | **~ 2.0 GB** |
| CUDA driver + cuBLAS + cuDNN reserved | **~ 1.0 GB** |
| **Total** | **~ 13.5 GB** |
| T4 capacity | 16.0 GB |
| **Headroom** | **~ 2.5 GB** ✓ |

Settings the project commits to (`configs/models_dev.yaml`):

```yaml
gpu_memory_utilization: 0.80     # 12.8 GB allocated → ~700 MB observed headroom
max_model_len: 4096
max_num_seqs: 4
dtype: bfloat16
enforce_eager: false             # Stage 2 dev only — CUDA graphs ON for ~40% speedup;
                                 # prod (Stage 3+) flips back to True per spec §11.
                                 # See OPEN_QUESTIONS.md #30.
enable_prefix_caching: true      # block-level KV-cache reuse across calls sharing
                                 # the system prompt + vignette + trajectory prefix.
                                 # See OPEN_QUESTIONS.md #30.
```

We start at `0.80` (not `0.85`) because at `0.85 = 13.6 GB` allocated, the
measured ~ 13.5 GB usage leaves only 100 MB of slack — narrow enough to OOM
under transient fragmentation. `0.80` matches what failure-mode 4.4 below
already prescribes as the OOM fix. See OPEN_QUESTIONS.md #28.

`enforce_eager=False` + `enable_prefix_caching=True` + within-turn EIG
batching (in `LLMClient.generate_batch`) combine for an estimated 3–4×
speedup on T4 vs a literal-spec sequential loop. See OPEN_QUESTIONS.md #30
for the full rationale.

These fit comfortably on T4. **Plan B is not needed.** If you hit OOM despite this:
1. Drop `gpu_memory_utilization` further to `0.75` (12 GB allocated, but vLLM uses less of the visible GPU memory).
2. Drop `max_model_len` to 2048 (loses 0.7 GB of KV).
3. Drop `max_num_seqs` to 2 (loses 0.7 GB of KV).
4. Switch to **L4 24 GB** (`g2-standard-4`) — drop-in replacement, no config change required, ~ 1.5× the cost of T4.

`max_model_len=8192` does **not** fit on T4 (KV cache balloons to ~3 GB → only ~1 GB headroom, will OOM under fragmentation). Stay at 4096 on T4; the longest prompt-trajectory pair in Stage 2 is ~1700 tokens, so 4096 is comfortable.

### 2.2 Choose a zone (T4 availability — verify before launching)

T4 supply is regional. Per GCP capacity reports through Q1 2026, these zones have reliable T4 capacity:

- `us-central1-a`, `us-central1-b`, `us-central1-f`
- `us-east1-c`, `us-east1-d`
- `europe-west4-b`, `europe-west4-c`
- `asia-northeast1-a`, `asia-east1-c`

Verify before `gcloud compute instances create`:

```bash
gcloud compute accelerator-types list --filter="name:nvidia-tesla-t4 AND zone~us-central1" --format="value(zone,maximumCardsPerInstance)"
```

If the chosen zone returns `RESOURCE_NOT_AVAILABLE` at create-time, switch to the next zone and retry.

### 2.3 Create the instance

```bash
gcloud compute instances create a2-sim-dev \
    --zone=us-central1-a \
    --machine-type=n1-standard-4 \
    --accelerator=type=nvidia-tesla-t4,count=1 \
    --image-family=pytorch-latest-gpu \
    --image-project=deeplearning-platform-release \
    --boot-disk-size=200GB \
    --boot-disk-type=pd-ssd \
    --metadata="install-nvidia-driver=True" \
    --maintenance-policy=TERMINATE \
    --scopes=cloud-platform
```

Notes:
- `pytorch-latest-gpu` ships CUDA 12.4 + PyTorch 2.4 + drivers — meets the spec's vLLM ≥ 0.19.0 requirement.
- **`n1-standard-4` is the floor, not a recommendation.** `n1-standard-1` OOM-kills the HF download at ~ 30 % because vLLM stages safetensors in CPU RAM (needs ~ 6 GB free) before GPU transfer. If you launched a smaller VM, resize: `gcloud compute instances stop $VM && gcloud compute instances set-machine-type $VM --machine-type=n1-standard-4 && gcloud compute instances start $VM`. The boot disk and any partial model cache survive the resize.
- **`setup_gcp.sh` installs `nvidia-cuda-toolkit` (≈ 1 GB).** vLLM 0.20+ JIT-compiles some kernels at first call and needs `nvcc` at runtime — the driver-only DLVM image doesn't include it. The setup script auto-detects and installs if missing.
- 200 GB SSD: model weights (~ 9 GB Qwen3-4B-Instruct-2507) + diskcache + venv + system + CUDA toolkit. Adequate for Stage 2.
- `TERMINATE`: required for accelerator-attached VMs (preemption can't migrate GPUs).
- `cloud-platform` scope: lets the VM use any GCP API the user has IAM for. Trim to `compute,storage-rw` if you prefer least-privilege.

### 2.4 Post-SSH command sequence (Stage 2)

```bash
# 1. SSH into the VM. Attach tmux IMMEDIATELY so SSH disconnects don't kill the run.
gcloud compute ssh a2-sim-dev --zone=us-central1-a
tmux new -s a2sim

# 2. Inside the tmux session: clone the repo. (setup_gcp.sh does NOT clone.)
git clone <YOUR_REPO_URL> HAPI_SIMULATION2 && cd HAPI_SIMULATION2

# 3. Set up environment.
cp .env.example .env
nano .env                           # paste HF_TOKEN, save
export $(grep -v '^#' .env | xargs)

# 4. Idempotent setup — creates venv, installs deps, prefetches Qwen3-4B (~ 5 min cold).
bash scripts/setup_gcp.sh --stage agent_dev

# 5. Pre-flight: verify the unit tests still pass on this machine. (~ 5 sec)
.venv/bin/python -m pytest -q

# 6. Stage 0 plumbing test (~ 5 sec, no GPU).
.venv/bin/python run.py --stage bootstrap

# 7. Stage 1 data prep — downloads MedQA (~ 12 sec cold; cache lives at data/raw/medqa_filtered.jsonl).
.venv/bin/python run.py --stage data

# 8. Stage 2 — the actual dress rehearsal. Expected wall-clock: 1–3 hours cold on T4.
#    On warm re-run (cache hit): minutes.
.venv/bin/python run.py --stage agent_dev
```

If you want to monitor without staying SSH'd: `tmux detach` (`Ctrl-b d`), reconnect later with `gcloud compute ssh a2-sim-dev` then `tmux attach -t a2sim`.

### 2.5 Success indicators

After Stage 2 finishes, all of these should be true:

| Check | How |
|---|---|
| `STAGE_REPORT.md` status is ✅ PASS | `head -20 results/agent_dev/STAGE_REPORT.md` |
| 20 trajectory rows present | `wc -l results/agent_dev/trajectories/hiv_status__benign.jsonl` → 20 |
| Mean diagnostic accuracy ≥ 0.50 | listed in the report's "Key numbers" table as `diagnostic_accuracy` |
| Mean queries / trajectory in [3, 15] | listed as `mean_queries_per_trajectory` |
| Principal-contamination guard never fired | acceptance row #6 ✅ |
| GPU never OOM'd | `tail results/agent_dev/log.jsonl`; no `cuda out of memory` events |

If any of these is red, see §2.7 below before re-running.

### 2.6 Pull results back to laptop

```bash
gcloud compute scp --recurse \
    a2-sim-dev:~/HAPI_SIMULATION2/results/agent_dev \
    ./results/ \
    --zone=us-central1-a
```

The single most important file to read locally:
```bash
cat results/agent_dev/STAGE_REPORT.md
```

### 2.7 Stop the instance when you're done

```bash
gcloud compute instances stop a2-sim-dev --zone=us-central1-a
```

T4 on `n1-standard-4` is **~ $0.40 / hour** on-demand (US zones, 2026 pricing). Boot disk + IP retention while stopped is ~ $0.04 / day. Stop the instance whenever you're not actively running a stage.

---

## 3 · Stages 3+ — A100 80 GB production stack

This section covers the prod-stack VM. Filled in incrementally as later stages run on real hardware.

### 3.1 Quota: A100 80 GB availability

A100 80 GB instances are scarce. Before creating the VM:

```bash
gcloud compute accelerator-types list --filter="name~nvidia-a100-80gb"
```

Pick a zone that lists `nvidia-a100-80gb` (commonly `us-central1-a/b/c`, `europe-west4-a`, `asia-northeast1-a`).

If your project has a 0 quota for `NVIDIA_A100_80GB_GPUS`, request a quota increase via `IAM & Admin → Quotas`. Approval typically takes 1–2 business days.

### 3.2 Create the prod-stack instance

```bash
gcloud compute instances create a2-sim-prod \
    --zone=us-central1-a \
    --machine-type=a2-ultragpu-1g \
    --accelerator=type=nvidia-a100-80gb,count=1 \
    --image-family=pytorch-latest-gpu \
    --image-project=deeplearning-platform-release \
    --boot-disk-size=500GB \
    --boot-disk-type=pd-ssd \
    --metadata="install-nvidia-driver=True" \
    --maintenance-policy=TERMINATE
```

A100 80 GB on-demand: **~ $3.67 / hour** (us-central1, 2026 pricing). Spot is roughly 1/3 of on-demand if your run can survive preemption; cache survives preemption, so spot is reasonable for stages with checkpointing (Stage 5+).

### 3.3 Cost estimates per stage

| Stage | Time | Hardware | On-demand | Spot |
|---|---|---|---:|---:|
| 0 bootstrap | < 5 min (laptop) | — | $0 | — |
| 1 data | < 5 min (laptop) | — | $0 | — |
| **2 agent_dev** | **1–3 h** | **T4 16 GB** | **~ $0.40–1.20** | — |
| 3 sanity | 1–2 h | A100 80 GB | $4–7 | — |
| 4 tiny_pilot | 1–2 h | A100 80 GB | $4–7 | — |
| 5 pilot | 6–10 h cold (then < 1 h warm with cache) | A100 80 GB | $22–37 | $7–13 |
| 6 probes | 30–60 min | A100 80 GB | $2–4 | — |
| 7 defences | 4–8 h | A100 80 GB | $15–30 | $5–10 |
| 8 full | 30–50 h | A100 80 GB | $110–185 | $40–65 |

Cache hits are free GPU-time-wise — re-running an identical pilot config consumes minutes.

---

## 4 · Things that could go wrong on GCP — top 5 failure modes for Stage 2

### 4.1 T4 unavailable in the chosen zone

**Symptom:** `gcloud compute instances create` exits with `ZONE_RESOURCE_POOL_EXHAUSTED` or `RESOURCE_NOT_AVAILABLE`.

**Diagnosis:** GCP doesn't have free T4 capacity in that zone right now. T4 supply oscillates by hour.

**Fix:**
1. Try the next zone in §2.2 (`us-east1-c` is a common backup for `us-central1`).
2. Try a different region entirely (`europe-west4-b`).
3. As last resort, switch to L4 (`--accelerator=type=nvidia-l4,count=1 --machine-type=g2-standard-4`). L4 is 24 GB so the same `models_dev.yaml` config still fits with even more headroom; cost is ~ 1.5× T4.

### 4.2 HuggingFace gated-model access denied

**Symptom:** During `setup_gcp.sh` or at vLLM load: `HfHubHTTPError: 401 Client Error` or `403 Client Error: Forbidden`.

**Diagnosis:** `HF_TOKEN` is missing, expired, or doesn't have access to the requested model.

**Fix:**
1. `cat ~/HAPI_SIMULATION2/.env` — confirm `HF_TOKEN=hf_...` is set and exported.
2. `huggingface-cli whoami` — confirm the token authenticates.
3. Visit `https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507-2507` in a browser logged in as the same account; click "Access repository" if a gate prompt appears. (Qwen3-4B-Instruct-2507 is ungated as of 2026-04, but check.)
4. Re-run `bash scripts/setup_gcp.sh --stage agent_dev` after fixing.

### 4.3 vLLM library import fails

**Symptom:** During `python run.py --stage agent_dev`: `ImportError: cannot import name 'LLM' from 'vllm'` or `RuntimeError: CUDA error: no kernel image is available for execution on the device`.

**Diagnosis:** CUDA driver vs vLLM/PyTorch mismatch. `pytorch-latest-gpu` images ship the right combination, but kernel updates can drift.

**Fix:**
1. `nvidia-smi` — confirm driver is ≥ 535 (CUDA 12.4 compatible). If older: `sudo /opt/deeplearning/install-driver.sh && sudo reboot`.
2. `.venv/bin/python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"` — should print `True` and `12.4` (or higher).
3. `.venv/bin/python -c "import vllm; print(vllm.__version__)"` — should be ≥ 0.19.0.
4. If torch reports CPU-only: `pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/cu124`.

### 4.4 OOM at model load

**Symptom:** During Stage 2 startup: `torch.OutOfMemoryError: CUDA out of memory. Tried to allocate X GB`.

**Diagnosis:** Either another process is holding the GPU, or our memory math was too tight for this specific T4.

**Fix:**
1. `nvidia-smi` — confirm no other process is using the GPU. If yes, kill it.
2. The committed default is already `gpu_memory_utilization: 0.80` (~ 700 MB headroom). If OOM persists, drop it to `0.75`. Retry.
3. If still OOM: drop `max_model_len: 4096` to `2048`. Retry.
4. If still OOM: stop the T4 instance and recreate as L4 (`g2-standard-4`); the same config will fit with ~ 9 GB headroom.

### 4.5 Stage 2 trajectories show 0 % accuracy

**Symptom:** STAGE_REPORT.md shows accuracy ≈ 0.0; the report fails the "≥ 0.50" gate. The pipeline ran without errors.

**Diagnosis:** Almost always a **prompt-template** issue, not a model issue. Qwen3-4B-Instruct-2507 is competent enough on USMLE-style MCQ that 0 % accuracy means the agent isn't even producing parseable letter-grade outputs.

**Where to look (in priority order):**

| File | What to check |
|---|---|
| `prompts/agent_action_select.txt` | Does it explicitly request a JSON array of objects with `type`/`query`/`rationale`? Does the model see the option keys (A/B/C/D)? |
| `prompts/agent_belief_update.txt` | Does the JSON object spec match the option keys exactly? |
| `prompts/agent_eig_predict.txt` | Are `{n_predictions}`, `{action_type}`, `{action_query}`, `{options_block}` all rendering correctly? |
| `prompts/agent_system.txt` | Does the system prompt instruct "output only JSON, no commentary"? Qwen models often add prose before/after JSON unless told not to. |
| `results/agent_dev/log.jsonl` | Search for `trajectory_done` events — what is `final_diagnosis`? Is it always one of `A/B/C/D`, or is it free-form text? |
| `results/agent_dev/trajectories/hiv_status__benign.jsonl` | Inspect `steps[*].observation` — do patient responses make sense, or are they "I am an AI..." disclaimers? If the latter, the **patient system prompt** is fighting the chat template. |

**Most-common fixes:**
1. The agent's `final_diagnosis` is "A. Pneumonia" (with prose) instead of `A`. Tighten the option block parsing in `src.agent.policy._parse_action_candidates` or post-process in `src.agent.runner` to extract just the letter.
2. The patient simulator is responding as the AI rather than the patient. Strengthen `prompts/patient_system.txt` ("**You ARE the patient.** Speak in first person. Do not break character.").
3. The eig_predict outputs are unparseable JSON. Inspect a few via `.venv/bin/python -c "from diskcache import Cache; c = Cache('results/agent_dev/llm_cache'); print(list(c)[:3])"` and re-tighten the prompt.

A clean re-run after a prompt fix is fast: the disk cache invalidates only the changed prompts, so most of the previous run's calls reuse cache hits.

---

## 5 · Tearing down

When all stages have completed and you've pulled `results/` back to your laptop:

```bash
gcloud compute instances delete a2-sim-dev --zone=us-central1-a       # Stage 2 VM
gcloud compute instances delete a2-sim-prod --zone=us-central1-a      # Stage 3+ VM
```

Pull `results/` back first; the boot disk goes with the VM.
