#!/usr/bin/env bash
set -euo pipefail

WORK_ROOT="${STWM_ROOT:-/home/chen034/workspace/stwm}"
LOG_PATH="${TRACEWM_STAGE2_FULLSCALE_WAVE2_LOG:-$WORK_ROOT/logs/tracewm_stage2_fullscale_wave2_20260409.log}"
SESSION_NAME="tracewm_stage2_fullscale_wave2_20260409"

if [[ -n "${TRACEWM_PYTHON_BIN:-}" ]]; then
  PYTHON_BIN="${TRACEWM_PYTHON_BIN}"
elif [[ -x "/home/chen034/miniconda3/envs/stwm/bin/python" ]]; then
  PYTHON_BIN="/home/chen034/miniconda3/envs/stwm/bin/python"
else
  PYTHON_BIN="python3"
fi

mkdir -p "$WORK_ROOT/logs" "$WORK_ROOT/reports" "$WORK_ROOT/docs"
export PYTHONPATH="$WORK_ROOT/code:${PYTHONPATH:-}"

exec > >(tee "$LOG_PATH") 2>&1

echo "[stage2-fullscale-wave2] start: $(date '+%Y-%m-%d %H:%M:%S %z')"
echo "[stage2-fullscale-wave2] python=$PYTHON_BIN"
echo "[stage2-fullscale-wave2] session=$SESSION_NAME"

"$PYTHON_BIN" "$WORK_ROOT/code/stwm/tools/run_tracewm_stage2_fullscale_wave2_20260409.py" \
  --mode orchestrate \
  --work-root "$WORK_ROOT" \
  --python-bin "$PYTHON_BIN" \
  --tmux-session "$SESSION_NAME"

echo "[stage2-fullscale-wave2] done: $(date '+%Y-%m-%d %H:%M:%S %z')"
