#!/usr/bin/env bash
set -euo pipefail

ROOT="${STWM_ROOT:-/raid/chen034/workspace/stwm}"
SESSION="stwm_ostf_v30_density_pooling_pilot_20260508"
LOG="$ROOT/logs/stwm_ostf_v30_density_pooling_pilot_20260508.log"
RUN="$ROOT/scripts/run_ostf_v30_density_pooling_pilot_20260508.sh"
mkdir -p "$(dirname "$LOG")" "$ROOT/reports"
if tmux has-session -t "$SESSION" >/dev/null 2>&1; then
  echo "$SESSION already running"
  exit 0
fi
tmux new-session -d -s "$SESSION" "cd '$ROOT' && bash '$RUN' > '$LOG' 2>&1"
cat > "$ROOT/reports/stwm_ostf_v30_density_pooling_pilot_launch_manifest_20260508.json" <<JSON
{"session":"$SESSION","log":"${LOG#$ROOT/}","command":"bash ${RUN#$ROOT/}","launched_at_local":"$(date -Iseconds)","M":[512,1024],"H":[32,64,96],"pooling_modes":["mean_reused","moments","induced_attention","hybrid_moments_attention"],"seeds":[42,123],"steps":4000}
JSON
cat > "$ROOT/reports/stwm_ostf_v30_density_pooling_pilot_status_20260508.json" <<JSON
{"session":"$SESSION","log":"${LOG#$ROOT/}","status":"launched","updated_at_local":"$(date -Iseconds)"}
JSON
echo "session=$SESSION"
echo "log=$LOG"
