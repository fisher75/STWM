#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/common.sh"

# Keep BLAS/OpenMP thread pools pinned to 1 for predictable CPU scheduling
# under multi-tenant training loads.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

usage() {
  cat <<'USAGE'
Usage:
  run_stwm_v4_2_real_train_seed.sh --scale {220m|1b} --seed <int> [out_root]

Description:
  Run one seed of strict real training matrix on true train split manifest.

Defaults:
  - Training set: VSPW train + VIPSeg train manifest
  - Budget: target_epochs=3, min_steps=5000, max_steps=8000
  - Checkpoint retention: latest + best (+ sparse milestones if enabled)

Environment knobs:
  STWM_V4_2_REAL_MANIFEST=<path>
  STWM_V4_2_REAL_RUNS=<csv>
  STWM_V4_2_REAL_FORCE_STEPS=<int>             # when >0, run exact steps for warmup/audit
  STWM_V4_2_REAL_TARGET_EPOCHS=3
  STWM_V4_2_REAL_MIN_STEPS=5000
  STWM_V4_2_REAL_MAX_STEPS=8000
  STWM_V4_2_CHECKPOINT_INTERVAL=300
  STWM_V4_2_MILESTONE_INTERVAL=0
USAGE
}

SCALE=""
SEED=""
OUT_ROOT=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --scale)
      [[ $# -ge 2 ]] || { echo "Missing value for --scale" >&2; exit 1; }
      SCALE="$2"
      shift 2
      ;;
    --seed)
      [[ $# -ge 2 ]] || { echo "Missing value for --seed" >&2; exit 1; }
      SEED="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      if [[ -z "$OUT_ROOT" ]]; then
        OUT_ROOT="$1"
        shift
      else
        echo "Unknown argument: $1" >&2
        usage
        exit 1
      fi
      ;;
  esac
done

if [[ -z "$SCALE" || -z "$SEED" ]]; then
  usage
  exit 1
fi

if ! [[ "$SEED" =~ ^[0-9]+$ ]]; then
  echo "seed must be integer, got: $SEED" >&2
  exit 1
fi

MANIFEST="${STWM_V4_2_REAL_MANIFEST:-$STWM_ROOT/manifests/realsplits/stwm_v4_2_vspw_vipseg_train_v1.json}"
if [[ ! -f "$MANIFEST" ]]; then
  echo "real train manifest not found: $MANIFEST" >&2
  exit 2
fi

TARGET_EPOCHS="${STWM_V4_2_REAL_TARGET_EPOCHS:-3}"
MIN_STEPS="${STWM_V4_2_REAL_MIN_STEPS:-5000}"
MAX_STEPS="${STWM_V4_2_REAL_MAX_STEPS:-8000}"
FORCE_STEPS="${STWM_V4_2_REAL_FORCE_STEPS:-0}"
SAMPLE_LIMIT="${STWM_V4_2_REAL_SAMPLE_LIMIT:-0}"
CHECKPOINT_INTERVAL="${STWM_V4_2_CHECKPOINT_INTERVAL:-300}"
MILESTONE_INTERVAL="${STWM_V4_2_MILESTONE_INTERVAL:-0}"

case "$SCALE" in
  220m)
    MODEL_PRESET="prototype_220m_v4_2"
    PRESET_FILE="$STWM_ROOT/code/stwm/configs/model_presets_v4_2.json"
    MICRO_BATCH=2
    GRAD_ACCUM=8
    NUM_WORKERS=12
    RUNS_DEFAULT="full_v4_2,wo_semantics_v4_2,wo_object_bias_v4_2"
    OUT_ROOT="${OUT_ROOT:-$STWM_ROOT/outputs/training/stwm_v4_2_real_220m}"
    ;;
  1b)
    MODEL_PRESET="prototype_1b_v4_2"
    PRESET_FILE="$STWM_ROOT/code/stwm/configs/model_presets_v4_2_1b.json"
    MICRO_BATCH=1
    GRAD_ACCUM=16
    NUM_WORKERS=14
    RUNS_DEFAULT="full_v4_2_1b,wo_semantics_v4_2_1b,wo_object_bias_v4_2_1b"
    OUT_ROOT="${OUT_ROOT:-$STWM_ROOT/outputs/training/stwm_v4_2_real_1b}"
    ;;
  *)
    echo "--scale must be 220m or 1b, got: $SCALE" >&2
    exit 1
    ;;
esac

RUNS_CSV="${STWM_V4_2_REAL_RUNS:-$RUNS_DEFAULT}"
IFS=',' read -r -a RUN_LIST <<< "$RUNS_CSV"

RETENTION_TEXT="latest_every_${CHECKPOINT_INTERVAL}+best"
if [[ "${MILESTONE_INTERVAL}" =~ ^[0-9]+$ ]] && (( MILESTONE_INTERVAL > 0 )); then
  RETENTION_TEXT="${RETENTION_TEXT}+milestone_every_${MILESTONE_INTERVAL}"
fi
echo "[stwm-v4.2-real] checkpoint_policy=${RETENTION_TEXT}"
echo "[stwm-v4.2-real] checkpoint_interval=${CHECKPOINT_INTERVAL} milestone_interval=${MILESTONE_INTERVAL}"
echo "[stwm-v4.2-real] dataloader_policy num_workers=${NUM_WORKERS} prefetch_factor=2 persistent_workers=true pin_memory=true"
echo "[stwm-v4.2-real] thread_policy OMP_NUM_THREADS=${OMP_NUM_THREADS} MKL_NUM_THREADS=${MKL_NUM_THREADS} OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS}"

if (( FORCE_STEPS > 0 )); then
  BUDGET_ARGS=(
    --steps "$FORCE_STEPS"
    --target-epochs 0
    --min-optimizer-steps 0
    --max-optimizer-steps 0
  )
else
  BUDGET_ARGS=(
    --steps 0
    --target-epochs "$TARGET_EPOCHS"
    --min-optimizer-steps "$MIN_STEPS"
    --max-optimizer-steps "$MAX_STEPS"
  )
fi

run_case() {
  local run_name="$1"
  shift

  local out_dir="$OUT_ROOT/seed_${SEED}/${run_name}"
  local log_file="$STWM_ROOT/logs/stwm_v4_2_real_${SCALE}_seed${SEED}_${run_name}.log"

  echo "[stwm-v4.2-real] start scale=${SCALE} seed=${SEED} run=${run_name}"

  PYTHONPATH="$STWM_ROOT/code:${PYTHONPATH:-}" \
    /home/chen034/miniconda3/bin/conda run --no-capture-output -n stwm \
    python "$STWM_ROOT/code/stwm/trainers/train_stwm_v4_2_real.py" \
      --data-root "$STWM_ROOT/data/external" \
      --manifest "$MANIFEST" \
      --output-dir "$out_dir" \
      --run-name "$run_name" \
      --model-preset "$MODEL_PRESET" \
      --preset-file "$PRESET_FILE" \
      --seed "$SEED" \
      --sample-limit "$SAMPLE_LIMIT" \
      --use-teacher-priors \
      --summary-name mini_val_summary.json \
      --log-name train_log.jsonl \
      --save-checkpoint \
      --checkpoint-interval "$CHECKPOINT_INTERVAL" \
      --milestone-interval "$MILESTONE_INTERVAL" \
      --checkpoint-dir-name checkpoints \
      --auto-resume \
      --micro-batch-per-gpu "$MICRO_BATCH" \
      --grad-accum "$GRAD_ACCUM" \
      --num-workers "$NUM_WORKERS" \
      --prefetch-factor 2 \
      --persistent-workers \
      --pin-memory \
      --bf16 \
      --activation-checkpointing \
      "${BUDGET_ARGS[@]}" \
      "$@" \
      >"$log_file" 2>&1

  echo "[stwm-v4.2-real] done scale=${SCALE} seed=${SEED} run=${run_name}"
}

for run_name in "${RUN_LIST[@]}"; do
  run_trimmed="$(echo "$run_name" | xargs)"
  case "$run_trimmed" in
    full_v4_2|full_v4_2_1b)
      run_case "$run_trimmed"
      ;;
    wo_semantics_v4_2|wo_semantics_v4_2_1b)
      run_case "$run_trimmed" --disable-semantics
      ;;
    wo_object_bias_v4_2|wo_object_bias_v4_2_1b)
      run_case "$run_trimmed" --neutralize-object-bias
      ;;
    *)
      echo "unsupported run: $run_trimmed" >&2
      exit 3
      ;;
  esac
done

echo "[stwm-v4.2-real] all runs done scale=${SCALE} seed=${SEED} out_root=${OUT_ROOT}"