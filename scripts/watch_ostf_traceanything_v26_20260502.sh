#!/usr/bin/env bash
set -euo pipefail

ROOT="${STWM_ROOT:-/raid/chen034/workspace/stwm}"
RUN_SCRIPT="$ROOT/scripts/run_ostf_traceanything_v26_20260502.sh"
RUN_DIR="$ROOT/reports/stwm_ostf_v26_runs"

required=(
  "$RUN_DIR/v26_traceanything_m128_h32_seed42.json"
  "$RUN_DIR/v26_traceanything_m128_h32_wo_semantic_memory_seed42.json"
  "$RUN_DIR/v26_traceanything_m128_h32_wo_dense_points_seed42.json"
  "$RUN_DIR/v26_traceanything_m128_h32_single_mode_seed42.json"
  "$RUN_DIR/v26_traceanything_m128_h32_wo_physics_prior_seed42.json"
  "$RUN_DIR/v26_traceanything_m512_h32_seed42.json"
  "$RUN_DIR/v26_traceanything_m128_h64_seed42.json"
)

while true; do
  missing=0
  missing_list=()
  for f in "${required[@]}"; do
    if [[ ! -f "$f" ]]; then
      missing=1
      missing_list+=("$(basename "$f")")
    fi
  done
  if [[ "$missing" -ne 0 ]]; then
    printf '[V26][watch] waiting missing=%s\n' "${missing_list[*]}"
  fi
  if [[ "$missing" -eq 0 ]]; then
    printf '[V26][watch] all run reports ready, launching eval+visualize\n'
    bash "$RUN_SCRIPT" eval
    bash "$RUN_SCRIPT" visualize
    printf '[V26][watch] eval+visualize completed\n'
    break
  fi
  sleep 60
done
