#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/home/chen034/miniconda3/envs/stwm/bin/python}"
WORK_ROOT="${WORK_ROOT:-/home/chen034/workspace/stwm}"
CONTRACT_PATH="${CONTRACT_PATH:-/home/chen034/workspace/data/_manifests/stage1_v2_trace_cache_contract_20260408.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/chen034/workspace/stwm/outputs/profiler/stage1_v2_nsys_20260408}"
STATUS_JSON="${STATUS_JSON:-/home/chen034/workspace/stwm/reports/stage1_v2_nsys_profile_20260408.json}"
SELECTED_GPU="${SELECTED_GPU:-}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --python-bin)
      PYTHON_BIN="$2"
      shift 2
      ;;
    --work-root)
      WORK_ROOT="$2"
      shift 2
      ;;
    --contract-path)
      CONTRACT_PATH="$2"
      shift 2
      ;;
    --output-root)
      OUTPUT_ROOT="$2"
      shift 2
      ;;
    --status-json)
      STATUS_JSON="$2"
      shift 2
      ;;
    --selected-gpu)
      SELECTED_GPU="$2"
      shift 2
      ;;
    *)
      echo "[stage1-v2-nsys] unknown_arg=$1"
      shift
      ;;
  esac
done

mkdir -p "$(dirname "$STATUS_JSON")" "$OUTPUT_ROOT" "$WORK_ROOT/reports" "$WORK_ROOT/docs" "$WORK_ROOT/outputs/training"

if ! command -v nsys >/dev/null 2>&1; then
  cat > "$STATUS_JSON" <<JSON
{
  "status": "unavailable",
  "reason": "nsys_not_found",
  "output_root": "$OUTPUT_ROOT"
}
JSON
  echo "[stage1-v2-nsys] unavailable"
  exit 0
fi

if [[ -n "$SELECTED_GPU" ]]; then
  export CUDA_VISIBLE_DEVICES="$SELECTED_GPU"
fi

TRAINER="$WORK_ROOT/code/stwm/tracewm_v2/trainers/train_tracewm_stage1_v2.py"
OUT_PREFIX="$OUTPUT_ROOT/stage1_v2_perf_debug_small"
TRAIN_SUMMARY="$WORK_ROOT/reports/stage1_v2_perf_nsys_debug_small_summary_20260408.json"
TRAIN_DOC="$WORK_ROOT/docs/STAGE1_V2_PERF_NSYS_DEBUG_SMALL_20260408.md"
TRAIN_TIMING="$WORK_ROOT/reports/stage1_v2_perf_nsys_debug_small_timing_20260408.json"

set +e
nsys profile \
  --force-overwrite=true \
  --sample=none \
  --trace=cuda,osrt,nvtx \
  -o "$OUT_PREFIX" \
  "$PYTHON_BIN" "$TRAINER" \
  --contract-path "$CONTRACT_PATH" \
  --dataset-names pointodyssey kubric \
  --train-split train \
  --obs-len 8 \
  --fut-len 8 \
  --max-tokens 64 \
  --max-samples-per-dataset 64 \
  --model-preset debug_small \
  --epochs 1 \
  --steps-per-epoch 10 \
  --batch-size 2 \
  --num-workers 0 \
  --enable-visibility \
  --enable-residual \
  --enable-velocity \
  --ablation-tag stage1_v2_perf_nsys_debug_small \
  --output-dir "$WORK_ROOT/outputs/training/stage1_v2_perf_nsys_debug_small_20260408" \
  --summary-json "$TRAIN_SUMMARY" \
  --results-md "$TRAIN_DOC" \
  --perf-step-timing-json "$TRAIN_TIMING"
NSYS_RC=$?
set -e

if [[ $NSYS_RC -eq 0 ]]; then
  cat > "$STATUS_JSON" <<JSON
{
  "status": "available",
  "nsys_return_code": 0,
  "output_root": "$OUTPUT_ROOT",
  "output_prefix": "$OUT_PREFIX",
  "nsys_report": "$OUT_PREFIX.nsys-rep"
}
JSON
  echo "[stage1-v2-nsys] success_report=$OUT_PREFIX.nsys-rep"
else
  cat > "$STATUS_JSON" <<JSON
{
  "status": "available_but_failed",
  "nsys_return_code": $NSYS_RC,
  "output_root": "$OUTPUT_ROOT",
  "output_prefix": "$OUT_PREFIX"
}
JSON
  echo "[stage1-v2-nsys] failed_rc=$NSYS_RC"
fi

exit 0
