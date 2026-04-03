#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home/chen034/workspace/stwm"
QUEUE_ROOT="${1:-$REPO_ROOT/outputs/queue/stwm_protocol_v2}"

for q in d0_eval d1_train; do
  mkdir -p "$QUEUE_ROOT/$q/pending" "$QUEUE_ROOT/$q/running" "$QUEUE_ROOT/$q/done" "$QUEUE_ROOT/$q/failed" "$QUEUE_ROOT/$q/logs" "$QUEUE_ROOT/$q/status" "$QUEUE_ROOT/$q/pids"
  touch "$QUEUE_ROOT/$q/queue_events.log"
done
mkdir -p "$QUEUE_ROOT/leases"

echo "queue_root=$QUEUE_ROOT"
echo "d0_queue=$QUEUE_ROOT/d0_eval"
echo "d1_queue=$QUEUE_ROOT/d1_train"
