#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/common.sh"

OUT_ROOT="${1:-$STWM_ROOT/outputs/training/stwm_v4_2_1b_confirmation_staged}"
PHASE_NAME="${STWM_V4_2_1B_STAGE_NAME:-phase_seed42}"
SEEDS_CSV="${STWM_V4_2_1B_STAGE_SEEDS:-42}"
RUNS_CSV="${STWM_V4_2_1B_STAGE_RUNS:-full_v4_2}"
BUILD_FIGURES="${STWM_V4_2_1B_STAGE_BUILD_FIGURES:-0}"
FIGURE_MIN_CONSISTENT_SEEDS="${STWM_V4_2_1B_STAGE_MIN_CONSISTENT_SEEDS:-1}"

BASE_ROOT="$OUT_ROOT/base"
STATE_ROOT="$OUT_ROOT/state"
STATUS_DIR="$OUT_ROOT/status"
STATUS_FILE="$STATUS_DIR/${PHASE_NAME}.status"

BASE_DECOUPLING_JSON="${STWM_V4_2_1B_BASE_DECOUPLING_JSON:-$STWM_ROOT/reports/stwm_v4_2_1b_query_decoupling_multiseed.json}"
STATE_DECOUPLING_JSON="${STWM_V4_2_1B_IDENTIFIABILITY_DECOUPLING_JSON:-$STWM_ROOT/reports/stwm_v4_2_1b_state_identifiability_decoupling_v1.json}"

mkdir -p "$OUT_ROOT" "$STATUS_DIR"

start_ts="$(timestamp)"
echo "[1b-confirmation-staged] phase=${PHASE_NAME} start=${start_ts} seeds=${SEEDS_CSV} runs=${RUNS_CSV}"

echo "[1b-confirmation-staged] step=base"
STWM_V4_2_1B_MINIVAL_SEEDS="$SEEDS_CSV" \
STWM_V4_2_1B_MINIVAL_RUNS="$RUNS_CSV" \
STWM_V4_2_1B_MINIVAL_SKIP_EXISTING=1 \
  bash "$SCRIPT_DIR/run_stwm_v4_2_1b_minival_multiseed.sh" "$BASE_ROOT"

echo "[1b-confirmation-staged] step=state_identifiability"
STWM_V4_2_1B_IDENTIFIABILITY_SEEDS="$SEEDS_CSV" \
STWM_V4_2_1B_IDENTIFIABILITY_RUNS="$RUNS_CSV" \
STWM_V4_2_1B_IDENTIFIABILITY_SKIP_EXISTING=1 \
STWM_V4_2_1B_IDENTIFIABILITY_BUILD_FIGURES="$BUILD_FIGURES" \
STWM_V4_2_1B_IDENTIFIABILITY_FIGURE_MIN_CONSISTENT_SEEDS="$FIGURE_MIN_CONSISTENT_SEEDS" \
STWM_V4_2_1B_BASE_RUNS_ROOT="$BASE_ROOT" \
  bash "$SCRIPT_DIR/run_stwm_v4_2_1b_state_identifiability.sh" "$STATE_ROOT"

end_ts="$(timestamp)"

cat > "$STATUS_FILE" <<EOF
phase=${PHASE_NAME}
start_ts=${start_ts}
end_ts=${end_ts}
seeds=${SEEDS_CSV}
runs=${RUNS_CSV}
base_root=${BASE_ROOT}
state_root=${STATE_ROOT}
base_summary=${BASE_ROOT}/comparison_multiseed.json
state_summary=${STATE_ROOT}/comparison_state_identifiability.json
base_decoupling=${BASE_DECOUPLING_JSON}
state_decoupling=${STATE_DECOUPLING_JSON}
EOF

echo "[1b-confirmation-staged] phase done: ${PHASE_NAME}"
echo "  status_file: $STATUS_FILE"
echo "  base_summary: $BASE_ROOT/comparison_multiseed.json"
echo "  state_summary: $STATE_ROOT/comparison_state_identifiability.json"
