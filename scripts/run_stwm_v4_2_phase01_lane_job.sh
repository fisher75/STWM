#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/common.sh"

usage() {
  cat <<'USAGE'
Usage:
  run_stwm_v4_2_phase01_lane_job.sh \
    --scale {220m|1b} \
    --seed <int> \
    --run-name <name> \
    --lane-dir <path> \
    --train-root <path> \
    --warmup-steps <int> \
    --checkpoint-interval <int>
USAGE
}

SCALE=""
SEED=""
RUN_NAME=""
LANE_DIR=""
TRAIN_ROOT=""
WARMUP_STEPS=""
CHECKPOINT_INTERVAL=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --scale)
      SCALE="$2"
      shift 2
      ;;
    --seed)
      SEED="$2"
      shift 2
      ;;
    --run-name)
      RUN_NAME="$2"
      shift 2
      ;;
    --lane-dir)
      LANE_DIR="$2"
      shift 2
      ;;
    --train-root)
      TRAIN_ROOT="$2"
      shift 2
      ;;
    --warmup-steps)
      WARMUP_STEPS="$2"
      shift 2
      ;;
    --checkpoint-interval)
      CHECKPOINT_INTERVAL="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ -z "$SCALE" || -z "$SEED" || -z "$RUN_NAME" || -z "$LANE_DIR" || -z "$TRAIN_ROOT" || -z "$WARMUP_STEPS" || -z "$CHECKPOINT_INTERVAL" ]]; then
  usage
  exit 1
fi

mkdir -p "$LANE_DIR" "$TRAIN_ROOT"

assigned="${STWM_ASSIGNED_GPUS:-unknown}"
gpu="${assigned%%,*}"

meta_file="$LANE_DIR/resource_probe_meta.txt"
gpu_csv="$LANE_DIR/gpu_usage_trace.csv"
disk_csv="$LANE_DIR/disk_trace.csv"

echo "start_ts=$(date +%F\ %T)" > "$meta_file"
echo "assigned_gpus=$assigned" >> "$meta_file"
echo "primary_gpu=$gpu" >> "$meta_file"
echo "host=$(hostname)" >> "$meta_file"
echo "pid=$$" >> "$meta_file"
echo "scale=$SCALE" >> "$meta_file"
echo "seed=$SEED" >> "$meta_file"
echo "run_name=$RUN_NAME" >> "$meta_file"
echo "warmup_steps=$WARMUP_STEPS" >> "$meta_file"

echo "ts,index,memory_used_mib,memory_total_mib,utilization_gpu" > "$gpu_csv"
echo "ts,disk_free_gb" > "$disk_csv"

(
  while true; do
    ts="$(date +%F\ %T)"
    nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits \
      | awk -F, -v g="$gpu" -v t="$ts" '{idx=$1; gsub(/^ +| +$/,"",idx); if (idx==g) {m=$2; mt=$3; u=$4; gsub(/^ +| +$/,"",m); gsub(/^ +| +$/,"",mt); gsub(/^ +| +$/,"",u); print t "," idx "," m "," mt "," u}}' >> "$gpu_csv" || true

    free_gb="$(df -BG "$TRAIN_ROOT" | awk 'NR==2 {gsub("G","",$4); print $4}')"
    echo "$ts,$free_gb" >> "$disk_csv"

    sleep 1
  done
) &
mon_pid=$!

status=0
STWM_V4_2_REAL_FORCE_STEPS="$WARMUP_STEPS" \
STWM_V4_2_REAL_RUNS="$RUN_NAME" \
STWM_V4_2_CHECKPOINT_INTERVAL="$CHECKPOINT_INTERVAL" \
STWM_V4_2_MILESTONE_INTERVAL=0 \
  bash "$SCRIPT_DIR/run_stwm_v4_2_real_train_seed.sh" --scale "$SCALE" --seed "$SEED" "$TRAIN_ROOT" || status=$?

kill "$mon_pid" 2>/dev/null || true
wait "$mon_pid" 2>/dev/null || true

src_dir="$TRAIN_ROOT/seed_${SEED}/${RUN_NAME}"
if [[ -d "$src_dir" ]]; then
  cp -f "$src_dir/train_log.jsonl" "$LANE_DIR/train_log.jsonl" 2>/dev/null || true
  cp -f "$src_dir/mini_val_summary.json" "$LANE_DIR/mini_val_summary.json" 2>/dev/null || true
  mkdir -p "$LANE_DIR/checkpoints"
  cp -f "$src_dir/checkpoints/latest.pt" "$LANE_DIR/checkpoints/latest.pt" 2>/dev/null || true
  cp -f "$src_dir/checkpoints/best.pt" "$LANE_DIR/checkpoints/best.pt" 2>/dev/null || true
fi

echo "end_ts=$(date +%F\ %T)" >> "$meta_file"
echo "exit_code=$status" >> "$meta_file"
echo "train_root=$TRAIN_ROOT" >> "$meta_file"
echo "run_dir=$src_dir" >> "$meta_file"

exit "$status"
