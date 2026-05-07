#!/usr/bin/env bash
# Local-side deploy: sync this repo to the GCP VM and install all packages.
#
# Run from the repo root:
#     bash scripts/deploy_to_gcp.sh
#
# Three phases, each phase is idempotent and safe to re-run:
#   1. rsync source files over SSH (no --delete; excludes match .gitignore)
#   2. ssh into the VM and run scripts/setup_gcp.sh (creates venv, pip install)
#   3. ssh into the VM and run `pip check` to confirm no dependency conflicts
#
# Defaults match the user's current A100 instance (Stage 3+):
#     instance: instance-20260428-154747   (A100 80 GB, Ubuntu 24.04)
#     zone:     us-central1-c
#     project:  hassanh-project
#     stage:    sanity                     (Stage 3 — first prod-stack run)
#
# The previous T4 instance (instance-20260427-145156, us-central1-b) was used
# for Stage 2 (agent_dev) and is retained in OPEN_QUESTIONS.md for context.
# Override with --instance / --zone / --stage if you spin up a new VM.
#
# Override per-run:
#     bash scripts/deploy_to_gcp.sh \
#         --instance my-vm --zone us-central1-a --project my-proj \
#         --stage agent_dev
#
# Notes:
#   - This script never deletes remote files. The VM's .env, results/, and
#     data/augmented/ are preserved across re-runs.
#   - It never runs any pipeline stage (run.py). Install only — you start
#     stages by hand on the VM after this finishes.
#   - The remote-side install is delegated to scripts/setup_gcp.sh, which
#     this script just invokes; do not duplicate logic between the two.

set -euo pipefail

# --------------------------------------------------------------------------
# Defaults — match the user's existing VM.
# --------------------------------------------------------------------------

INSTANCE="${INSTANCE:-instance-20260428-154747}"
ZONE="${ZONE:-us-central1-c}"
PROJECT="${PROJECT:-hassanh-project}"
STAGE="${STAGE:-sanity}"
REMOTE_DIR="${REMOTE_DIR:-~/HAPI_SIMULATION2}"
DO_SYNC=1
DO_INSTALL=1
USE_DELETE=0
ASSUME_YES=0

while [[ $# -gt 0 ]]; do
    case $1 in
        --instance)    INSTANCE="$2"; shift 2 ;;
        --zone)        ZONE="$2"; shift 2 ;;
        --project)     PROJECT="$2"; shift 2 ;;
        --stage)       STAGE="$2"; shift 2 ;;
        --remote-dir)  REMOTE_DIR="$2"; shift 2 ;;
        --no-sync)     DO_SYNC=0; shift ;;
        --no-install)  DO_INSTALL=0; shift ;;
        --delete)      USE_DELETE=1; shift ;;
        --yes|-y)      ASSUME_YES=1; shift ;;
        -h|--help)
            sed -n '2,30p' "$0"
            exit 0
            ;;
        *) echo "Unknown arg: $1" >&2; exit 2 ;;
    esac
done

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

echo "==> Repo:     $REPO_ROOT"
echo "==> Target:   $INSTANCE  (zone $ZONE, project $PROJECT)"
echo "==> Stage:    $STAGE   (controls which extras setup_gcp.sh installs)"
echo "==> Remote:   $REMOTE_DIR"

# --------------------------------------------------------------------------
# 1. Pre-flight: tools, repo root, gcloud auth.
# --------------------------------------------------------------------------

for cmd in gcloud rsync ssh; do
    if ! command -v "$cmd" >/dev/null 2>&1; then
        echo "ERROR: '$cmd' is not installed locally." >&2
        exit 1
    fi
done

if [ ! -f "pyproject.toml" ] || [ ! -f "scripts/setup_gcp.sh" ]; then
    echo "ERROR: $(pwd) does not look like the project root (no pyproject.toml or scripts/setup_gcp.sh)." >&2
    exit 1
fi

if ! gcloud auth list --filter="status:ACTIVE" --format="value(account)" 2>/dev/null | grep -q .; then
    echo "ERROR: no active gcloud account. Run 'gcloud auth login' first." >&2
    exit 1
fi

# --------------------------------------------------------------------------
# 2. Verify the VM exists; offer to start it if stopped.
# --------------------------------------------------------------------------

VM_STATUS="$(gcloud compute instances describe "$INSTANCE" \
    --zone="$ZONE" --project="$PROJECT" \
    --format="value(status)" 2>/dev/null || true)"

if [ -z "$VM_STATUS" ]; then
    echo "ERROR: instance '$INSTANCE' not found in zone '$ZONE' of project '$PROJECT'." >&2
    exit 1
fi
echo "==> VM status: $VM_STATUS"

if [ "$VM_STATUS" != "RUNNING" ]; then
    if [ "$ASSUME_YES" = "1" ]; then
        START_VM="y"
    else
        read -r -p "    Instance is $VM_STATUS. Start it now? [y/N] " START_VM
    fi
    case "$START_VM" in
        y|Y|yes|YES)
            gcloud compute instances start "$INSTANCE" --zone="$ZONE" --project="$PROJECT"
            # SSH daemon needs a few seconds after RUNNING is reported.
            sleep 10
            ;;
        *)
            echo "    Aborting; start the instance and re-run." >&2
            exit 1
            ;;
    esac
fi

# --------------------------------------------------------------------------
# 3. Use `gcloud compute ssh` as the transport for everything.
#
#    Plain `ssh` (after `gcloud compute config-ssh`) is faster but flaky on
#    first run: it needs the host key in known_hosts, the SSH key loaded in
#    ssh-agent, and a route that doesn't require IAP. Going through
#    `gcloud compute ssh` instead lets gcloud handle host-key TOFU, key
#    push to project metadata / OS Login, and IAP tunneling automatically.
#
#    For rsync (which can't call `gcloud compute ssh` directly because the
#    arg layout doesn't match what rsync expects from an SSH binary), we
#    drop a tiny wrapper script and pass it via `rsync -e`.
# --------------------------------------------------------------------------

# All cleanups in one trap; rsync excludes file is added later.
CLEANUP_FILES=()
cleanup() { [ ${#CLEANUP_FILES[@]} -eq 0 ] || rm -f "${CLEANUP_FILES[@]}"; }
trap cleanup EXIT

SSH_WRAPPER="$(mktemp -t a2sim_ssh_wrapper.XXXXXX)"
CLEANUP_FILES+=("$SSH_WRAPPER")
# Rsync invokes the wrapper as: WRAPPER [user@]host remote-cmd...
# We drop user@ if present, and forward the rest as the gcloud --command.
cat >"$SSH_WRAPPER" <<EOF
#!/usr/bin/env bash
set -e
HOST="\${1#*@}"; shift
exec gcloud compute ssh --zone="$ZONE" --project="$PROJECT" "\$HOST" -- "\$@"
EOF
chmod +x "$SSH_WRAPPER"

# Helper: run a single command on the VM via gcloud SSH.
remote_ssh() {
    gcloud compute ssh "$INSTANCE" --zone="$ZONE" --project="$PROJECT" --command="$1"
}

# Smoke test. The first SSH to a brand-new VM may need to push a key to
# project metadata and accept a host key — gcloud handles both, but if
# something is wrong (auth, IAP policy, firewall) we want to fail here
# with a clear message rather than mid-rsync.
echo "==> Smoke-testing 'gcloud compute ssh' to $INSTANCE (may take ~10s on first run)"
if ! remote_ssh "echo ssh-ok && uname -srm" ; then
    echo "ERROR: gcloud compute ssh failed. Try the same command interactively to see the prompt:" >&2
    echo "       gcloud compute ssh --zone $ZONE --project $PROJECT $INSTANCE --command='true'" >&2
    exit 1
fi

# --------------------------------------------------------------------------
# 4. Sync source files. Excludes mirror .gitignore so we never push the
#    local venv (mac binaries, won't run on Linux), pyc caches, results
#    that the VM may have generated, or the local .env.
# --------------------------------------------------------------------------

if [ "$DO_SYNC" = "1" ]; then
    EXCLUDES_FILE="$(mktemp -t a2sim_rsync_excludes.XXXXXX)"
    CLEANUP_FILES+=("$EXCLUDES_FILE")
    cat >"$EXCLUDES_FILE" <<'EOF'
# Virtualenvs (must NOT cross OS boundaries)
.venv/
venv/
env/

# Python build / cache artifacts
__pycache__/
*.pyc
*.pyo
*.egg-info/
*.egg
build/
dist/
.pytest_cache/
.mypy_cache/
.ruff_cache/

# Editor / OS junk
.DS_Store
.idea/
.vscode/
*.swp
*.swo

# Repo metadata not needed on the VM
.git/

# Local secrets — preserve whatever .env the VM already has
.env

# Experiment artifacts: regenerated on the VM, VM's copies are authoritative
results/
data/raw/medqa_*.jsonl
data/augmented/*.jsonl
data/covert_tasks/*_generated.json
vllm_*.log
EOF

    DELETE_FLAG=""
    if [ "$USE_DELETE" = "1" ]; then
        # Bare --delete preserves files that match an exclude (e.g. .env,
        # results/, data/augmented/). We never want --delete-excluded here.
        echo "==> --delete is ON: remote files not present locally will be removed (excluded paths preserved)"
        DELETE_FLAG="--delete"
    fi

    # Ensure the remote root exists. Rsync would create it, but doing it
    # explicitly gives a clearer error if the home dir is unwritable.
    remote_ssh "mkdir -p $REMOTE_DIR"

    echo "==> Rsync $REPO_ROOT/  -->  $INSTANCE:$REMOTE_DIR/"
    # -a archive, -v verbose (file list), -z compress, -h human-readable bytes,
    # --progress per-file progress. We avoid --info=progress2: the
    # macOS-bundled openrsync / rsync 2.6.9 doesn't have it.
    # Trailing slash on source matters: copy contents, not the parent dir.
    # Transport is the gcloud SSH wrapper from step 3 above.
    rsync -avzh --progress \
        --exclude-from="$EXCLUDES_FILE" \
        $DELETE_FLAG \
        -e "$SSH_WRAPPER" \
        "$REPO_ROOT/" \
        "$INSTANCE:$REMOTE_DIR/"
    echo "==> Sync complete."
else
    echo "==> Skipping sync (--no-sync)"
fi

# --------------------------------------------------------------------------
# 5. Install packages on the VM via the existing idempotent setup script.
#    setup_gcp.sh handles: venv creation, pip install -e .[dev,prod],
#    HF_TOKEN check, and (per stage) model weight prefetch. We do not
#    re-implement any of that here.
# --------------------------------------------------------------------------

if [ "$DO_INSTALL" = "1" ]; then
    # ----------------------------------------------------------------------
    # 5a. Ensure python3.11 + venv exists on the VM.
    #
    # setup_gcp.sh requires python3.11 (or 3.12) on PATH — the project's
    # pyproject.toml pins `requires-python = ">=3.11"`. Two install paths,
    # tried in order:
    #
    #   1. apt-get install python3.11 python3.11-venv python3.11-dev
    #      Works on Ubuntu 22.04 / 24.04 LTS and any distro where 3.11 is in
    #      the archives. Cheap and standard, but fails on very fresh distros
    #      (Ubuntu 26.04 "resolute" has no python3.11 in main and the
    #      deadsnakes PPA hasn't published for it yet).
    #
    #   2. uv (https://astral.sh/uv) downloads standalone CPython builds
    #      from the python-build-standalone project. Distro- and PPA-
    #      independent; the only fallback that's guaranteed to work on any
    #      Linux. We symlink the resulting binary to /usr/local/bin/python3.11
    #      so setup_gcp.sh's existing `command -v python3.11` finds it.
    #
    # We do this here, not inside setup_gcp.sh, so setup_gcp.sh stays
    # usable on machines without sudo or curl.
    # ----------------------------------------------------------------------
    echo "==> Ensuring python3.11 + venv module are present on $INSTANCE"
    remote_ssh 'set -e

ensure_via_apt() {
    # Stale/broken third-party PPAs (e.g. deadsnakes for an Ubuntu codename
    # it does not ship for) make apt-get update exit non-zero. We tolerate
    # that and only care whether the install step itself succeeds.
    {
        sudo apt-get update -qq 2>/dev/null || true
        sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
            python3.11 python3.11-venv python3.11-dev 2>/dev/null
    } >/dev/null
    python3.11 -c "import venv" >/dev/null 2>&1
}

ensure_via_uv() {
    if ! command -v uv >/dev/null 2>&1 && [ ! -x "$HOME/.local/bin/uv" ]; then
        echo "    Installing uv (~5s)"
        curl -LsSf https://astral.sh/uv/install.sh -o /tmp/uv-install.sh
        sh /tmp/uv-install.sh >/dev/null
        rm -f /tmp/uv-install.sh
    fi
    export PATH="$HOME/.local/bin:$PATH"
    echo "    Installing CPython 3.11 via uv (~20s on first run, cached after)"
    uv python install 3.11
    UV_PY="$(uv python find 3.11)"
    if [ -z "$UV_PY" ] || [ ! -x "$UV_PY" ]; then
        echo "ERROR: uv install did not produce a 3.11 binary." >&2
        return 1
    fi
    # Make it discoverable as `python3.11` for setup_gcp.sh.
    sudo ln -sf "$UV_PY" /usr/local/bin/python3.11
    python3.11 -c "import venv" >/dev/null 2>&1
}

if python3.11 -c "import venv" >/dev/null 2>&1; then
    echo "    Already present: $(python3.11 --version)"
elif python3.12 -c "import venv" >/dev/null 2>&1; then
    echo "    Already present: $(python3.12 --version)"
elif ensure_via_apt; then
    echo "    Installed via apt: $(python3.11 --version)"
elif ensure_via_uv; then
    echo "    Installed via uv: $(python3.11 --version)"
else
    echo "ERROR: could not install python3.11 via apt or uv." >&2
    echo "       Inspect the VM: cat /etc/os-release; ls /etc/apt/sources.list.d" >&2
    exit 1
fi'

    echo "==> Running scripts/setup_gcp.sh --stage $STAGE on $INSTANCE"
    # Stream output live so the user can watch pip / HF download progress.
    remote_ssh "cd $REMOTE_DIR && bash scripts/setup_gcp.sh --stage $STAGE"

    # ----------------------------------------------------------------------
    # 6. Conflict check. `pip check` reports any dependency with an unmet
    #    or incompatible requirement. Exits non-zero on conflict.
    # ----------------------------------------------------------------------
    echo "==> Running 'pip check' on the remote venv"
    if remote_ssh "cd $REMOTE_DIR && .venv/bin/pip check"; then
        echo "==> pip check: no dependency conflicts."
    else
        echo "WARNING: pip check reported conflicts. Inspect the output above." >&2
        echo "         Common cause: pyproject's [prod] extras pulled a torch /" >&2
        echo "         vllm / transformers combo that disagrees on a transitive." >&2
        echo "         Fix by pinning the offender in pyproject.toml and re-running." >&2
        exit 1
    fi

    # Remove a stale profile.py if a previous deploy left one behind. The
    # repo used to ship a top-level profile.py which shadows the stdlib
    # `profile` module (transitively imported by torch._dynamo via
    # cProfile), so `import torch` from the repo root would crash with
    # "module 'profile' has no attribute 'run'". The file has been renamed
    # to run_profile.py upstream; rsync without --delete won't remove the
    # old copy on its own, so we do it explicitly here.
    remote_ssh "rm -f $REMOTE_DIR/profile.py"

    # Quick smoke import of the heavy prod-stack libs so failures surface
    # here, not later when you launch a stage. Skipped for stages that
    # don't install [prod]. We still pass -P (don't prepend cwd to
    # sys.path) as belt-and-suspenders against any future name collision.
    case "$STAGE" in
        agent_dev|sanity|tiny_pilot|pilot|probes|defences|full)
            echo "==> Smoke-importing torch / vllm / transformers"
            remote_ssh "cd $REMOTE_DIR && .venv/bin/python -P -c 'import torch, vllm, transformers; \
                print(\"torch       \", torch.__version__, \"cuda=\", torch.cuda.is_available()); \
                print(\"vllm        \", vllm.__version__); \
                print(\"transformers\", transformers.__version__)'"
            ;;
    esac
else
    echo "==> Skipping install (--no-install)"
fi

# --------------------------------------------------------------------------
# Summary.
# --------------------------------------------------------------------------

cat <<EOF

==> Deploy complete.

Next steps (run on the VM, do NOT auto-run from here):

    gcloud compute ssh --zone $ZONE --project $PROJECT $INSTANCE
    tmux new -s a2sim                     # so SSH disconnects don't kill the run
    cd ${REMOTE_DIR/#\~/\$HOME}
    .venv/bin/python run.py --list        # see available stages
    .venv/bin/python run.py --stage $STAGE

To re-deploy after local edits, just run this script again — rsync will
ship only the changed files and setup_gcp.sh will skip already-installed
packages.
EOF
