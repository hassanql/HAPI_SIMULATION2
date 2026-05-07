#!/usr/bin/env bash
# Cleanup helper: kill any leftover vLLM subprocesses and verify GPU is free.
#
# Use this between stage runs if a previous run crashed and may have left
# orphan vLLM servers behind (a known failure mode — see OPEN_QUESTIONS
# #45 #46). Safe to run when nothing is leftover; it just confirms the
# clean state and exits 0.
#
# Run as: bash scripts/cleanup_vllm.sh
#
# Exits 0 on success, 1 if processes survived after both SIGTERM and
# SIGKILL (something is wrong; investigate before re-launching).

set -uo pipefail

echo "==> Looking for leftover vLLM subprocesses..."

# Pattern matches both `vllm serve ...` and the engine-core / API-server
# children that vLLM spawns. The grep -v guards against matching this
# script itself or the pipe.
PIDS="$(ps -eo pid,cmd | grep -E "vllm serve|EngineCore|APIServer pid=" | grep -v grep | grep -v cleanup_vllm | awk '{print $1}' | sort -u | tr '\n' ' ')"
PIDS="$(echo "$PIDS" | xargs)"   # trim whitespace

if [ -z "$PIDS" ]; then
    echo "==> No vLLM processes found."
else
    echo "==> Found PIDs: $PIDS"
    echo "==> Sending SIGTERM..."
    # `kill` may fail if a child died in the interval — that's fine.
    kill $PIDS 2>/dev/null || true

    # Give vLLM up to 15 s to flush its NCCL state and exit cleanly.
    for _ in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15; do
        REMAINING="$(ps -p $PIDS -o pid= 2>/dev/null | xargs)"
        if [ -z "$REMAINING" ]; then
            echo "==> All processes exited cleanly."
            break
        fi
        sleep 1
    done

    REMAINING="$(ps -p $PIDS -o pid= 2>/dev/null | xargs)"
    if [ -n "$REMAINING" ]; then
        echo "==> $REMAINING did not respond to SIGTERM after 15 s. Sending SIGKILL..."
        kill -KILL $REMAINING 2>/dev/null || true
        sleep 2
        REMAINING="$(ps -p $PIDS -o pid= 2>/dev/null | xargs)"
        if [ -n "$REMAINING" ]; then
            echo "ERROR: PIDs still alive after SIGKILL: $REMAINING. Investigate manually." >&2
            exit 1
        fi
    fi
fi

echo
echo "==> Verifying ports 8001 / 8002 / 8003 are free:"
for port in 8001 8002 8003; do
    if lsof -ti ":$port" >/dev/null 2>&1; then
        HOLDER="$(lsof -ti ":$port" | tr '\n' ' ')"
        echo "    port $port: STILL HELD by PID(s) $HOLDER" >&2
        exit 1
    else
        echo "    port $port: free"
    fi
done

echo
if command -v nvidia-smi >/dev/null 2>&1; then
    echo "==> GPU memory after cleanup:"
    nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu --format=csv
fi

echo
echo "==> Clean. Safe to launch a stage."
