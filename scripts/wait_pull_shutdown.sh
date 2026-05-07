#!/usr/bin/env bash
# Run on the Mac (NOT the VM). Polls the GCP VM until a given stage's
# python process exits, pulls the stage's results back, then stops the VM.
#
# Usage:
#     bash scripts/wait_pull_shutdown.sh                    # tiny_pilot (default)
#     STAGE=pilot bash scripts/wait_pull_shutdown.sh        # Stage 5
#     STAGE=sanity bash scripts/wait_pull_shutdown.sh       # Stage 3
#
# Stays awake via caffeinate -i (caller's responsibility). Cancel with Ctrl-C.

set -uo pipefail

INSTANCE="${INSTANCE:-instance-20260428-154747}"
ZONE="${ZONE:-us-central1-c}"
PROJECT="${PROJECT:-hassanh-project}"
STAGE="${STAGE:-tiny_pilot}"
LOCAL_DEST="${LOCAL_DEST:-results/${STAGE}_post_coherence_$(date +%Y%m%d_%H%M%S)}"
REMOTE_RESULTS="${REMOTE_RESULTS:-~/HAPI_SIMULATION2/results/${STAGE}}"
POLL_INTERVAL="${POLL_INTERVAL:-300}"

cd "$(dirname "$0")/.."

ts() { date +%H:%M:%S; }

echo "[$(ts)] Polling $INSTANCE every ${POLL_INTERVAL}s for stage='$STAGE' completion..."
# Match only python processes whose argv contains 'run.py' AND '--stage <STAGE>'.
# We pgrep -af python first to filter to python only — that excludes the
# wrapper bash that gcloud spawns to evaluate the command (which would
# otherwise self-match because its argv contains the pattern).
PGREP='pgrep -af python | grep -E "run\\.py.*--stage '"$STAGE"'( |$)" >/dev/null'
while gcloud compute ssh "$INSTANCE" --zone="$ZONE" --project="$PROJECT" \
        --command="$PGREP" 2>/dev/null
do
    echo "[$(ts)] still running..."
    sleep "$POLL_INTERVAL"
done

echo "[$(ts)] run finished — pulling $REMOTE_RESULTS to $LOCAL_DEST"
if gcloud compute scp --recurse --zone="$ZONE" --project="$PROJECT" \
        "${INSTANCE}:${REMOTE_RESULTS}" \
        "$LOCAL_DEST"
then
    echo "[$(ts)] results pulled — stopping VM"
    gcloud compute instances stop "$INSTANCE" --zone="$ZONE" --project="$PROJECT"
    echo "[$(ts)] VM stopped. Done."
else
    echo "[$(ts)] scp FAILED — leaving VM running for investigation"
    exit 1
fi
