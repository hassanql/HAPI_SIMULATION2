#!/usr/bin/env bash
# Mac-side helper: start a stopped VM briefly, pull stage results, stop again.
#
# Use after `scripts/run_and_shutdown.sh` has finished on the VM and the
# VM is in TERMINATED state (auto-shutdown completed).
#
# Run from the Mac:
#     bash scripts/pull_results_from_stopped_vm.sh
#
# Override defaults with env vars:
#     INSTANCE=instance-...     ZONE=us-central1-c    PROJECT=hassanh-project
#     STAGE=pilot               (default: pilot)
#     LOCAL_DEST=results/...    (default: results/<stage>_post_<timestamp>/)

set -uo pipefail

INSTANCE="${INSTANCE:-instance-20260428-154747}"
ZONE="${ZONE:-us-central1-c}"
PROJECT="${PROJECT:-hassanh-project}"
STAGE="${STAGE:-pilot}"
LOCAL_DEST="${LOCAL_DEST:-results/${STAGE}_post_$(date +%Y%m%d_%H%M%S)}"
REMOTE_RESULTS="${REMOTE_RESULTS:-~/HAPI_SIMULATION2/results/${STAGE}}"

cd "$(dirname "$0")/.."

ts() { date +%H:%M:%S; }

# 1. Check current VM state.
echo "[$(ts)] checking $INSTANCE state..."
state=$(gcloud compute instances describe "$INSTANCE" --zone="$ZONE" \
    --project="$PROJECT" --format='value(status)' 2>/dev/null)
echo "[$(ts)] current state: $state"

started_us=0
if [[ "$state" == "TERMINATED" || "$state" == "STOPPING" ]]; then
    echo "[$(ts)] starting VM..."
    gcloud compute instances start "$INSTANCE" --zone="$ZONE" --project="$PROJECT"
    started_us=1
    # Give SSH a moment to become reachable.
    echo "[$(ts)] waiting 30s for SSH to come up..."
    sleep 30
elif [[ "$state" == "RUNNING" ]]; then
    echo "[$(ts)] VM already running — skipping start"
else
    echo "[$(ts)] unexpected state '$state' — bailing out"
    exit 1
fi

# 2. Pull results.
echo "[$(ts)] pulling $REMOTE_RESULTS to $LOCAL_DEST"
if gcloud compute scp --recurse --zone="$ZONE" --project="$PROJECT" \
        "${INSTANCE}:${REMOTE_RESULTS}" \
        "$LOCAL_DEST"
then
    echo "[$(ts)] results pulled successfully"
    PULL_OK=1
else
    echo "[$(ts)] scp FAILED"
    PULL_OK=0
fi

# 3. Stop VM again only if WE started it (don't stop a VM the user is using).
if [[ $started_us -eq 1 ]]; then
    if [[ $PULL_OK -eq 1 ]]; then
        echo "[$(ts)] stopping VM (we started it)..."
        gcloud compute instances stop "$INSTANCE" --zone="$ZONE" --project="$PROJECT"
        echo "[$(ts)] VM stopped. Done."
    else
        echo "[$(ts)] leaving VM running for investigation (we started it but pull failed)"
        exit 1
    fi
else
    echo "[$(ts)] leaving VM in its original state (RUNNING — user has it)"
fi
