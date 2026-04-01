#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/common.sh"

OUT_ROOT="${1:-$STWM_ROOT/outputs/training/stwm_v4_2_1b_real_confirmation}"
SEEDS_CSV="${2:-42,123}"
VIS_SEEDS_CSV="${3:-$SEEDS_CSV}"

POLL_SECONDS="${STWM_V4_2_REAL_1B_FINALIZE_POLL_SECONDS:-60}"
TIMEOUT_SECONDS="${STWM_V4_2_REAL_1B_FINALIZE_TIMEOUT_SECONDS:-0}"

if ! [[ "$POLL_SECONDS" =~ ^[0-9]+$ ]] || (( POLL_SECONDS <= 0 )); then
  echo "invalid POLL_SECONDS: $POLL_SECONDS" >&2
  exit 1
fi

if ! [[ "$TIMEOUT_SECONDS" =~ ^[0-9]+$ ]]; then
  echo "invalid TIMEOUT_SECONDS: $TIMEOUT_SECONDS" >&2
  exit 1
fi

status_dir="$OUT_ROOT/status"
mkdir -p "$status_dir"

start_ts="$(timestamp)"
start_epoch="$(date +%s)"

echo "[1b-real-watch] start=${start_ts} out_root=${OUT_ROOT} seeds=${SEEDS_CSV}"

IFS=',' read -r -a seed_list <<< "$SEEDS_CSV"

while true; do
  missing=0
  for seed in "${seed_list[@]}"; do
    status_file="$status_dir/seed_${seed}.status"
    if [[ ! -f "$status_file" ]]; then
      missing=1
      echo "[1b-real-watch] waiting for ${status_file}"
    fi
  done

  if (( missing == 0 )); then
    break
  fi

  now_epoch="$(date +%s)"
  elapsed="$(( now_epoch - start_epoch ))"
  if (( TIMEOUT_SECONDS > 0 && elapsed >= TIMEOUT_SECONDS )); then
    echo "[1b-real-watch] timeout after ${elapsed}s" >&2
    exit 3
  fi
  sleep "$POLL_SECONDS"
done

echo "[1b-real-watch] mandatory seeds completed, finalize summaries"
STWM_V4_2_REAL_1B_SEEDS="$SEEDS_CSV" \
  bash "$SCRIPT_DIR/finalize_stwm_v4_2_1b_real_confirmation.sh" "$OUT_ROOT"

echo "[1b-real-watch] build real 1B visual assets"
STWM_V4_2_REAL_1B_VIS_SEEDS="$VIS_SEEDS_CSV" \
  bash "$SCRIPT_DIR/build_stwm_v4_2_real_1b_visualization.sh" "$OUT_ROOT"

end_ts="$(timestamp)"
final_status="$status_dir/finalize_${SEEDS_CSV//,/_}.status"
cat > "$final_status" <<EOF
start_ts=${start_ts}
end_ts=${end_ts}
out_root=${OUT_ROOT}
seeds=${SEEDS_CSV}
vis_seeds=${VIS_SEEDS_CSV}
base_summary=${OUT_ROOT}/base/comparison_multiseed.json
state_summary=${OUT_ROOT}/state/comparison_state_identifiability.json
compare_json=${STWM_ROOT}/reports/stwm_v4_2_220m_vs_1b_real_confirmation.json
demo_manifest=${STWM_ROOT}/outputs/visualizations/stwm_v4_2_real_1b_demo/demo_manifest.json
EOF

echo "[1b-real-watch] all done"
echo "  final_status: $final_status"
