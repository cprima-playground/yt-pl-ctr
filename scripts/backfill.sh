#!/bin/bash
# Backfill script - runs fetch and process for a specified duration
# Usage: ./scripts/backfill.sh [hours]

HOURS=${1:-6}
CONFIG="configs/channels.yaml"
LIMIT=50
DELAY=2
FETCH_PAUSE=60  # Pause between fetch batches (seconds)

PROC_PID=""

cleanup() {
    echo ""
    echo "=== Stopping backfill ==="
    if [ -n "$PROC_PID" ]; then
        kill -9 $PROC_PID 2>/dev/null || true
        pkill -9 -P $PROC_PID 2>/dev/null || true
    fi
    pkill -9 -f "yt-pl-ctr process" 2>/dev/null || true
    pkill -9 -f "yt-pl-ctr fetch" 2>/dev/null || true
    echo "Stopped"
    exit 0
}
trap cleanup EXIT INT TERM QUIT

END_TIME=$(($(date +%s) + HOURS * 3600))

echo "=== Backfill started at $(date) ==="
echo "Duration: ${HOURS} hours"
echo "Config: $CONFIG"
echo "Batch size: $LIMIT, Delay: ${DELAY}s"
echo "To stop: Ctrl+C or kill $$"
echo ""

# Start processor in background
echo "Starting processor..."
uv run yt-pl-ctr process -c "$CONFIG" --watch &
PROC_PID=$!
echo "Processor PID: $PROC_PID"

sleep 3

# Fetch loop
BATCH=1
while [ $(date +%s) -lt $END_TIME ]; do
    echo ""
    echo "=== Batch $BATCH at $(date) ==="

    uv run yt-pl-ctr fetch -c "$CONFIG" --limit $LIMIT --delay $DELAY || true

    uv run yt-pl-ctr queue-status

    REMAINING=$((END_TIME - $(date +%s)))
    if [ $REMAINING -gt $FETCH_PAUSE ]; then
        echo "Sleeping ${FETCH_PAUSE}s..."
        sleep $FETCH_PAUSE
    else
        break
    fi

    BATCH=$((BATCH + 1))
done

echo "Done. Waiting for processor..."
sleep 10
