#!/usr/bin/env bash
# VM-side: run a stage to completion, then auto-shutdown the VM.
#
# Run this in tmux on the GCP VM:
#     tmux new -s pilot
#     bash scripts/run_and_shutdown.sh
#
# When the stage exits (success OR failure OR exception), the VM is
# scheduled to shut down after a grace period. Disk and results are
# preserved across shutdown — the user pulls them later by starting the
# VM, scp'ing, and stopping again.
#
# Cancel a pending shutdown (within the grace window):
#     sudo shutdown -c
#
# Override defaults with env vars:
#     STAGE=tiny_pilot         (default: pilot)
#     SHUTDOWN_GRACE=2         (minutes before halt; default 2)
#     NO_SHUTDOWN=1            (skip shutdown entirely; for local testing)

set -uo pipefail

STAGE="${STAGE:-pilot}"
SHUTDOWN_GRACE="${SHUTDOWN_GRACE:-2}"
NO_SHUTDOWN="${NO_SHUTDOWN:-0}"

cd "$(dirname "$0")/.."

# ---------------------------------------------------------------------------
# Safety guard: refuse to run on anything that isn't a GCP VM. Without this
# check the script has shut down a developer's laptop after a 0.3-second
# python-failure when invoked from the wrong machine. We detect a GCP VM
# by checking the metadata server (which only exists on GCP) and the
# Linux kernel.
# ---------------------------------------------------------------------------

is_gcp_vm() {
    [[ "$(uname -s)" == "Linux" ]] || return 1
    # GCP metadata server: requires the special header. 1-second timeout.
    curl --fail --silent --max-time 1 \
        -H "Metadata-Flavor: Google" \
        "http://metadata.google.internal/computeMetadata/v1/instance/id" \
        > /dev/null 2>&1
}

if [[ "$NO_SHUTDOWN" != "1" ]] && ! is_gcp_vm; then
    cat <<'EOF' >&2
ERROR: scripts/run_and_shutdown.sh is for GCP VMs only.

This host does not look like a GCP VM (uname is non-Linux, or the GCP
metadata server is unreachable). The script schedules `sudo shutdown -h`
on exit -- running it elsewhere has shut down developer machines.

If you meant to run the pipeline locally without auto-shutdown:
    NO_SHUTDOWN=1 bash scripts/run_and_shutdown.sh

If you meant to run it on the VM, SSH in first:
    gcloud compute ssh <instance> --zone <zone> --project <project>
    cd ~/HAPI_SIMULATION2
    tmux new -s pilot
    bash scripts/run_and_shutdown.sh
EOF
    exit 2
fi

ts() { date '+%Y-%m-%d %H:%M:%S'; }

LOGFILE="results/${STAGE}_run_$(date +%Y%m%d_%H%M%S).log"
mkdir -p results

echo "[$(ts)] launching stage '$STAGE'"
echo "[$(ts)] log:                 $LOGFILE"
echo "[$(ts)] auto-shutdown grace: ${SHUTDOWN_GRACE} min after exit"
echo "[$(ts)] NO_SHUTDOWN flag:    $NO_SHUTDOWN  (set to 1 to skip)"
echo

# Activate the venv if present (idempotent — no error if already active).
if [[ -f .venv/bin/activate ]]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
fi

# Schedule shutdown on ANY exit path: success, failure, or signal.
# `trap ... EXIT` fires on normal completion, set -e abort, and Ctrl-C
# (because Ctrl-C triggers SIGINT which the script propagates to its exit
# handlers). The user has the grace window to `sudo shutdown -c` if they
# want to abort the shutdown.
on_exit() {
    local rc=$?
    echo
    echo "[$(ts)] stage '$STAGE' exited with code $rc"
    if [[ "$NO_SHUTDOWN" == "1" ]]; then
        echo "[$(ts)] NO_SHUTDOWN=1 — skipping auto-shutdown"
        echo "[$(ts)] log saved to: $LOGFILE"
        return
    fi
    echo "[$(ts)] scheduling VM shutdown in ${SHUTDOWN_GRACE} minute(s)"
    echo "[$(ts)] cancel with: sudo shutdown -c"
    sudo shutdown -h "+${SHUTDOWN_GRACE}" \
        "Auto-shutdown after stage '$STAGE' (rc=$rc). Cancel with 'sudo shutdown -c'." \
        2>&1 | tee -a "$LOGFILE"
}
trap on_exit EXIT

# Run the pipeline. The exit code propagates to `on_exit` via the trap.
python run.py --stage "$STAGE" 2>&1 | tee "$LOGFILE"
