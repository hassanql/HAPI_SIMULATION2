#!/usr/bin/env bash
# Idempotent setup script for the GCP small-GPU dress rehearsal (Stage 2,
# T4) and the A100 production runs (Stage 3+).
#
# Each step prints either "==> ACTION" (work happened) or "==> Skipping X
# (already present)" so a second invocation is observably a no-op for the
# parts that have already completed.
#
# This script does NOT:
#   - clone the repo (you do that before running it)
#   - touch data/augmented/ or results/ under any circumstances
#   - re-download model weights that are already in the HF cache
#   - re-create the venv if it already has the right Python version
#
# Run as: bash scripts/setup_gcp.sh [--stage agent_dev|sanity|...]

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

# --------------------------------------------------------------------------
# Argument parsing — `--stage X` controls which model weights to prefetch.
# --------------------------------------------------------------------------

STAGE="${STAGE:-agent_dev}"
while [[ $# -gt 0 ]]; do
    case $1 in
        --stage) STAGE="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 2 ;;
    esac
done

echo "==> A2 Simulation GCP setup (target stage: $STAGE)"
echo "==> Repo root: $REPO_ROOT"

# --------------------------------------------------------------------------
# 1. Hardware sanity check.
# --------------------------------------------------------------------------

if command -v nvidia-smi >/dev/null 2>&1; then
    GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
    GPU_MEM="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)"
    echo "==> GPU detected: $GPU_NAME ($GPU_MEM MiB)"
else
    echo "WARNING: nvidia-smi not found. Stage 2 / Stage 3+ will fail without a GPU."
fi

# --------------------------------------------------------------------------
# 1.5. CUDA developer toolkit (nvcc). vLLM 0.20+ defers some kernel
#      compilation to first-call JIT, which needs nvcc at runtime, not just
#      the runtime libraries. The Deep Learning VM ships with the runtime
#      driver but not always the developer toolkit. See OPEN_QUESTIONS.md #35.
# --------------------------------------------------------------------------

if command -v nvcc >/dev/null 2>&1 || [ -x "/usr/local/cuda/bin/nvcc" ]; then
    echo "==> Skipping CUDA toolkit install (nvcc already present)"
else
    echo "==> nvcc not found — installing nvidia-cuda-toolkit (~ 1 GB, ~ 3 min)"
    sudo apt-get update -qq
    sudo apt-get install -y -qq nvidia-cuda-toolkit
fi

# --------------------------------------------------------------------------
# 1.6. Host C compiler (gcc).
#
# Triton — used by torch._inductor inside vLLM's CUDA-graph compile path
# — shells out to a host C compiler at first inference to build the kernel
# launcher .so files. Symptom when it's missing:
#
#     torch._inductor.exc.InductorError: RuntimeError:
#     Failed to find C compiler. Please specify via CC environment variable.
#
# The GCP Deep Learning VM image ships /usr/local/cuda/bin/nvcc but no host
# gcc, so the nvcc check above passes on a fresh DLVM and we'd otherwise
# hit the failure on the first vLLM request. Library-mode stages (Stage 2
# `agent_dev`) don't take the inductor path so they don't surface the bug.
# --------------------------------------------------------------------------

if command -v gcc >/dev/null 2>&1; then
    echo "==> Skipping gcc install (gcc already present at $(command -v gcc))"
else
    echo "==> gcc not found — installing build-essential (~ 200 MB, ~ 1 min)"
    sudo apt-get update -qq
    sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq build-essential
fi

# Triton's `compile_module_from_src` builds a small CUDA driver shim at
# runtime that #includes <Python.h>. Ubuntu 24.04 splits the headers into
# python3.X-dev (separate apt package). Without it, gcc fails:
#     fatal error: Python.h: No such file or directory
# We install both python3-dev (meta-package) and the version-specific one
# matching the interpreter we just selected.
if ! [ -f /usr/include/python3.12/Python.h ] && ! [ -f /usr/include/python3.11/Python.h ]; then
    echo "==> Python headers missing — installing python3-dev / python${PY#python}-dev"
    sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
        python3-dev "python${PY#python}-dev"
fi

# --------------------------------------------------------------------------
# 2. Python interpreter — Python 3.11.x required (per OPEN_QUESTIONS.md #11).
# --------------------------------------------------------------------------

PY=""
for cand in python3.11 python3.12; do
    if command -v "$cand" >/dev/null 2>&1; then
        PY="$cand"; break
    fi
done
if [ -z "$PY" ]; then
    echo "ERROR: Python 3.11 (preferred) or 3.12 is required. Install via your distro or pyenv." >&2
    exit 1
fi
echo "==> System Python: $PY ($("$PY" --version))"

# Ubuntu 24.04 ships /usr/bin/python3.12 but splits the `venv` module into a
# separate apt package (`python3.12-venv`). Without it, `python3.12 -m venv`
# fails with "ensurepip is not available", which is confusing if you only
# look at the venv command. Probe for the module up front and install the
# matching apt package on demand. If the deploy_to_gcp.sh wrapper already
# installed venv, this is a fast no-op.
if ! "$PY" -c "import venv, ensurepip" >/dev/null 2>&1; then
    PY_VER="${PY#python}"             # python3.12 -> 3.12
    VENV_PKG="python${PY_VER}-venv"
    echo "==> $PY missing venv/ensurepip — installing $VENV_PKG"
    sudo apt-get update -qq
    sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq "$VENV_PKG"
    if ! "$PY" -c "import venv, ensurepip" >/dev/null 2>&1; then
        echo "ERROR: $VENV_PKG installed but $PY still cannot import venv/ensurepip." >&2
        exit 1
    fi
fi

# --------------------------------------------------------------------------
# 3. Virtualenv. Skip recreation if the existing one is on Python 3.11.x.
# --------------------------------------------------------------------------

VENV_PYTHON_OK=0
if [ -x ".venv/bin/python" ]; then
    EXISTING_VER="$(.venv/bin/python --version 2>&1 | awk '{print $2}')"
    case "$EXISTING_VER" in
        3.11.*|3.12.*) VENV_PYTHON_OK=1 ;;
    esac
    if [ "$VENV_PYTHON_OK" = "1" ]; then
        echo "==> Skipping venv creation (already present at .venv, Python $EXISTING_VER)"
    else
        echo "==> Existing .venv has Python $EXISTING_VER (want 3.11.x or 3.12.x); recreating"
        rm -rf .venv
    fi
fi
if [ "$VENV_PYTHON_OK" = "0" ]; then
    echo "==> Creating venv with $PY"
    "$PY" -m venv .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate

# --------------------------------------------------------------------------
# 4. Pip dependencies. Idempotent: pip skips already-satisfied packages by
#    default; we just announce whether any work happened.
# --------------------------------------------------------------------------

A2_INSTALLED="$(pip show a2-sim 2>/dev/null | awk '/^Version:/ {print $2}' || true)"
if [ -n "$A2_INSTALLED" ]; then
    echo "==> a2-sim already installed (version $A2_INSTALLED); pip install will be a no-op for satisfied deps"
fi

echo "==> Upgrading pip / setuptools / wheel (no-op if already current)"
pip install --upgrade --quiet pip setuptools wheel

# Determine which extras to install based on the stage.
EXTRAS="dev"
case "$STAGE" in
    agent_dev|sanity|tiny_pilot|pilot|probes|defences|full)
        EXTRAS="dev,prod"
        ;;
esac
echo "==> Installing project with extras [$EXTRAS]"
pip install --quiet -e ".[$EXTRAS]"

# Surface any unmet / incompatible transitive requirements right after the
# install. Without this, conflicts only show up at import time deep inside a
# stage run, where the traceback rarely names the offender. We don't abort —
# pyproject's [prod] extras pull torch/vllm/transformers, which periodically
# disagree on tokenizers / numpy / protobuf versions, and the conflicts are
# usually informational rather than blocking. The deploy_to_gcp.sh wrapper
# does abort on conflict; that is the policy gate for fresh installs.
echo "==> pip check (informational; deploy_to_gcp.sh treats conflicts as fatal):"
pip check || echo "    (above conflicts may be benign; see deploy_to_gcp.sh §6)"

# --------------------------------------------------------------------------
# 5. .env / HF_TOKEN.
# --------------------------------------------------------------------------

if [ -f ".env" ]; then
    echo "==> Sourcing .env (already present)"
    # shellcheck disable=SC1091
    set -a; source .env; set +a
elif [ -f ".env.example" ]; then
    echo "WARNING: .env not found. Copy .env.example to .env and set HF_TOKEN before running stages that touch gated models."
fi
if [ -z "${HF_TOKEN:-}" ]; then
    echo "WARNING: HF_TOKEN is empty. Public models still work; gated models will 403."
fi

# --------------------------------------------------------------------------
# 6. Pre-fetch model weights for the target stage (idempotent — HF caches).
# --------------------------------------------------------------------------

prefetch_model() {
    local model_id="$1"
    echo "==> Prefetching $model_id (HuggingFace cache hits are free)"
    python - <<PY
import os, sys
from huggingface_hub import snapshot_download
try:
    p = snapshot_download(
        repo_id="$model_id",
        token=os.environ.get("HF_TOKEN") or None,
    )
    print(f"   -> cached at {p}")
except Exception as e:
    print(f"WARNING: snapshot_download for $model_id failed: {type(e).__name__}: {e}", file=sys.stderr)
    sys.exit(0)   # don't abort setup on a single download failure
PY
}

case "$STAGE" in
    bootstrap|data)
        echo "==> Skipping model weight prefetch (Stage $STAGE does not need a model)"
        ;;
    agent_dev)
        prefetch_model "Qwen/Qwen3-4B-Instruct-2507"
        ;;
    sanity|tiny_pilot|pilot|defences|full)
        # Matches `configs/models_prod.yaml`. Spec asked for
        # Qwen3.6-35B-A3B-FP8 (Qwen3.6 family doesn't exist) → using
        # Qwen3-30B-A3B-Instruct-2507-FP8 instead. Patient is the real
        # spec model (gemma-4-26B-A4B-it).
        prefetch_model "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8"
        prefetch_model "google/gemma-4-26B-A4B-it"
        ;;
    probes)
        prefetch_model "microsoft/Phi-4"
        ;;
    *)
        echo "WARNING: Unknown stage '$STAGE'; skipping model prefetch"
        ;;
esac

# --------------------------------------------------------------------------
# 7. Sanity check: artifacts under data/augmented/ and results/ are NOT
#    touched by this script. Print their state so the user can verify.
# --------------------------------------------------------------------------

echo "==> Checking that this script did NOT touch experiment artifacts:"
if [ -d "data/augmented" ]; then
    nfiles=$(find data/augmented -maxdepth 1 -type f | wc -l | tr -d ' ')
    echo "   data/augmented/: $nfiles files (this script never writes here)"
fi
if [ -d "results" ]; then
    nstages=$(find results -maxdepth 1 -type d -mindepth 1 | wc -l | tr -d ' ')
    echo "   results/: $nstages stage directories (this script never writes here)"
fi

echo "==> Setup complete."
echo "==> Next:"
echo "    python run.py --list"
echo "    python run.py --stage $STAGE"
