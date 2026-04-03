#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home/chen034/workspace/stwm"
QUEUE_ROOT="${1:-$REPO_ROOT/outputs/queue/stwm_protocol_v2}"

D0_QUEUE="$QUEUE_ROOT/d0_eval"
D1_QUEUE="$QUEUE_ROOT/d1_train"

D0_SESSION="${STWM_PROTOCOL_V2_D0_SESSION:-stwm_protocol_v2_d0_eval_worker}"
D1_SESSION="${STWM_PROTOCOL_V2_D1_SESSION:-stwm_protocol_v2_d1_train_worker}"

D0_LOG="${STWM_PROTOCOL_V2_D0_WORKER_LOG:-$REPO_ROOT/logs/${D0_SESSION}.log}"
D1_LOG="${STWM_PROTOCOL_V2_D1_WORKER_LOG:-$REPO_ROOT/logs/${D1_SESSION}.log}"

bash "$REPO_ROOT/scripts/setup_stwm_protocol_v2_queue.sh" "$QUEUE_ROOT"

start_if_missing() {
  local session="$1"
  local queue_dir="$2"
  local class_type="$3"
  local log_file="$4"
  local idle_sleep="$5"

  if tmux has-session -t "$session" 2>/dev/null; then
    echo "[worker] existing session: $session"
  else
    bash "$REPO_ROOT/scripts/start_protocol_v2_queue_tmux.sh" \
      --session "$session" \
      --queue-dir "$queue_dir" \
      --class-type "$class_type" \
      --idle-sleep "$idle_sleep" \
      --workdir "$REPO_ROOT" \
      --log-file "$log_file"
  fi

  pane_pid="$(tmux list-panes -t "$session" -F '#{pane_pid}' | head -n 1)"
  echo "$pane_pid" > "$queue_dir/worker.pid"
  echo "$session" > "$queue_dir/worker.session"
  echo "$log_file" > "$queue_dir/worker.logpath"
}

start_if_missing "$D0_SESSION" "$D0_QUEUE" "A" "$D0_LOG" "10"
start_if_missing "$D1_SESSION" "$D1_QUEUE" "B" "$D1_LOG" "15"

echo "queue_root=$QUEUE_ROOT"
echo "d0_session=$D0_SESSION"
echo "d1_session=$D1_SESSION"
echo "d0_queue=$D0_QUEUE"
echo "d1_queue=$D1_QUEUE"
