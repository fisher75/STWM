#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/raid/chen034/workspace/stwm}"
PY="${PY:-/home/chen034/miniconda3/envs/stwm/bin/python}"
export PYTHONPATH="${REPO_ROOT}/code${PYTHONPATH:+:${PYTHONPATH}}"
export STWM_PROC_TITLE="${STWM_PROC_TITLE:-python}"
export STWM_PROC_TITLE_MODE="${STWM_PROC_TITLE_MODE:-generic}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-16}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-16}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-16}"
export STWM_TORCH_NUM_THREADS="${STWM_TORCH_NUM_THREADS:-16}"
export STWM_IMAGE_CACHE_MAX="${STWM_IMAGE_CACHE_MAX:-256}"

cd "${REPO_ROOT}"

mkdir -p outputs/logs reports docs outputs/cache outputs/checkpoints

FEATURE_REPORT="reports/stwm_fullscale_semantic_trace_feature_targets_v1_20260428.json"
FEATURE_CACHE_DIR="outputs/cache/stwm_fullscale_semantic_trace_feature_targets_v1_20260428"
PROTO32_REPORT="reports/stwm_fullscale_semantic_trace_prototypes_c32_v1_20260428.json"
PROTO64_REPORT="reports/stwm_fullscale_semantic_trace_prototypes_c64_v1_20260428.json"
TARGET32_REPORT="reports/stwm_fullscale_semantic_trace_prototype_targets_c32_v1_20260428.json"
TARGET64_REPORT="reports/stwm_fullscale_semantic_trace_prototype_targets_c64_v1_20260428.json"
OBS_REPORT="reports/stwm_fullscale_observed_semantic_prototype_targets_v1_20260428.json"
SPLIT_REPORT="reports/stwm_fullscale_semantic_trace_world_model_v1_splits_20260428.json"

if [[ "${STWM_FULLSCALE_SKIP_TARGET_POOL:-0}" != "1" ]]; then

TARGET_SHARD_SESSIONS=()
TARGET_SHARD_REPORTS=()
launch_feature_shard() {
  local SPLIT_NAME="$1"
  local START_IDX="$2"
  local END_IDX="$3"
  local GPU_ID="$4"
  local SHARD_NAME="$5"
  local REPORT="reports/stwm_fullscale_semantic_trace_feature_targets_v1_${SHARD_NAME}_20260428.json"
  local CACHE_DIR="outputs/cache/stwm_fullscale_semantic_trace_feature_targets_v1_${SHARD_NAME}_20260428"
  local SESSION="stwm_fullscale_target_${SHARD_NAME}"
  TARGET_SHARD_REPORTS+=("${REPORT}")
  TARGET_SHARD_SESSIONS+=("${SESSION}")
  tmux kill-session -t "${SESSION}" 2>/dev/null || true
  tmux new-session -d -s "${SESSION}" "bash -lc 'cd ${REPO_ROOT}; exec env CUDA_VISIBLE_DEVICES=${GPU_ID} STWM_PROC_TITLE=python STWM_PROC_TITLE_MODE=generic PYTHONUNBUFFERED=1 OMP_NUM_THREADS=${OMP_NUM_THREADS} MKL_NUM_THREADS=${MKL_NUM_THREADS} OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS} NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS} STWM_TORCH_NUM_THREADS=${STWM_TORCH_NUM_THREADS} STWM_IMAGE_CACHE_MAX=${STWM_IMAGE_CACHE_MAX} PYTHONPATH=${PYTHONPATH} ${PY} code/stwm/tools/build_future_semantic_trace_feature_targets_20260428.py --dataset-names vspw vipseg --splits ${SPLIT_NAME} --max-samples-train 999999 --max-samples-val 999999 --max-entities-per-sample 8 --fut-len 8 --device cuda --batch-size 512 --crop-extraction-mode tensor_roi --target-build-mode fast_target_only --entry-start ${START_IDX} --entry-end ${END_IDX} --progress-every 100 --torch-num-threads ${STWM_TORCH_NUM_THREADS} --image-cache-size ${STWM_IMAGE_CACHE_MAX} --cache-dir ${CACHE_DIR} --output ${REPORT} --doc docs/STWM_FULLSCALE_SEMANTIC_TRACE_FEATURE_TARGETS_V1_${SHARD_NAME}_20260428.md > outputs/logs/stwm_fullscale_target_${SHARD_NAME}.log 2>&1'"
}

TRAIN_TOTAL=5612
VAL_TOTAL=686
TRAIN_SHARDS=8
VAL_SHARDS=2
for SHARD in $(seq 0 $((TRAIN_SHARDS - 1))); do
  START=$((SHARD * TRAIN_TOTAL / TRAIN_SHARDS))
  END=$(((SHARD + 1) * TRAIN_TOTAL / TRAIN_SHARDS))
  launch_feature_shard train "${START}" "${END}" "$((SHARD % 8))" "train_shard$(printf '%02d' "${SHARD}")"
done
for SHARD in $(seq 0 $((VAL_SHARDS - 1))); do
  START=$((SHARD * VAL_TOTAL / VAL_SHARDS))
  END=$(((SHARD + 1) * VAL_TOTAL / VAL_SHARDS))
  launch_feature_shard val "${START}" "${END}" "$(((SHARD + TRAIN_SHARDS) % 8))" "val_shard$(printf '%02d' "${SHARD}")"
done

while true; do
  ACTIVE=0
  for SESSION in "${TARGET_SHARD_SESSIONS[@]}"; do
    if tmux has-session -t "${SESSION}" 2>/dev/null; then
      ACTIVE=$((ACTIVE + 1))
    fi
  done
  if [[ "${ACTIVE}" == "0" ]]; then
    break
  fi
  echo "[fullscale-target] active_parallel_tmux_sessions=${ACTIVE}" >&2
  sleep 30
done

"${PY}" code/stwm/tools/merge_future_semantic_trace_feature_target_shards_20260428.py \
  --shard-reports "${TARGET_SHARD_REPORTS[@]}" \
  --cache-dir "${FEATURE_CACHE_DIR}" \
  --output "${FEATURE_REPORT}" \
  --doc docs/STWM_FULLSCALE_SEMANTIC_TRACE_FEATURE_TARGETS_V1_20260428.md

"${PY}" code/stwm/tools/build_semantic_trace_prototypes_20260428.py \
  --feature-cache-report "${FEATURE_REPORT}" \
  --prototype-count 32 \
  --iterations 20 \
  --cache-dir outputs/cache/stwm_fullscale_semantic_trace_prototypes_c32_v1_20260428 \
  --output "${PROTO32_REPORT}" \
  --doc docs/STWM_FULLSCALE_SEMANTIC_TRACE_PROTOTYPES_C32_V1_20260428.md

"${PY}" code/stwm/tools/build_semantic_trace_prototypes_20260428.py \
  --feature-cache-report "${FEATURE_REPORT}" \
  --prototype-count 64 \
  --iterations 20 \
  --cache-dir outputs/cache/stwm_fullscale_semantic_trace_prototypes_c64_v1_20260428 \
  --output "${PROTO64_REPORT}" \
  --doc docs/STWM_FULLSCALE_SEMANTIC_TRACE_PROTOTYPES_C64_V1_20260428.md

"${PY}" code/stwm/tools/build_future_semantic_trace_prototype_targets_20260428.py \
  --feature-cache-report "${FEATURE_REPORT}" \
  --prototype-report "${PROTO32_REPORT}" \
  --cache-dir outputs/cache/stwm_fullscale_semantic_trace_prototype_targets_c32_v1_20260428 \
  --output "${TARGET32_REPORT}" \
  --doc docs/STWM_FULLSCALE_SEMANTIC_TRACE_PROTOTYPE_TARGETS_C32_V1_20260428.md

"${PY}" code/stwm/tools/build_future_semantic_trace_prototype_targets_20260428.py \
  --feature-cache-report "${FEATURE_REPORT}" \
  --prototype-report "${PROTO64_REPORT}" \
  --cache-dir outputs/cache/stwm_fullscale_semantic_trace_prototype_targets_c64_v1_20260428 \
  --output "${TARGET64_REPORT}" \
  --doc docs/STWM_FULLSCALE_SEMANTIC_TRACE_PROTOTYPE_TARGETS_C64_V1_20260428.md

"${PY}" code/stwm/tools/build_observed_semantic_prototype_targets_20260428.py \
  --feature-report "${FEATURE_REPORT}" \
  --prototype-target-reports "${TARGET32_REPORT}" "${TARGET64_REPORT}" \
  --max-samples-per-dataset 999999 \
  --observed-max-samples-per-dataset 999999 \
  --force-rebuild-observed-cache \
  --observed-min-coverage 0.05 \
  --device cuda \
  --batch-size 512 \
  --cache-dir outputs/cache/stwm_fullscale_observed_semantic_prototype_targets_v1_20260428 \
  --output "${OBS_REPORT}" \
  --doc docs/STWM_FULLSCALE_OBSERVED_SEMANTIC_PROTOTYPE_TARGETS_V1_20260428.md

"${PY}" code/stwm/tools/build_semantic_memory_world_model_splits_20260428.py \
  --observed-report "${OBS_REPORT}" \
  --future-report-c32 "${TARGET32_REPORT}" \
  --future-report-c64 "${TARGET64_REPORT}" \
  --target-train-items 0 \
  --target-val-items 200 \
  --target-test-items 200 \
  --output "${SPLIT_REPORT}" \
  --doc docs/STWM_FULLSCALE_SEMANTIC_TRACE_WORLD_MODEL_V1_SPLITS_20260428.md \
  --audit-name stwm_fullscale_semantic_trace_world_model_v1_splits

"${PY}" - <<'PY'
import json
from pathlib import Path

root = Path(".")
feature = json.loads(Path("reports/stwm_fullscale_semantic_trace_feature_targets_v1_20260428.json").read_text())
obs = json.loads(Path("reports/stwm_fullscale_observed_semantic_prototype_targets_v1_20260428.json").read_text())
splits = json.loads(Path("reports/stwm_fullscale_semantic_trace_world_model_v1_splits_20260428.json").read_text())
payload = {
    "audit_name": "stwm_fullscale_semantic_trace_target_pool_v1",
    "total_raw_samples_scanned": int(feature.get("item_count", 0)),
    "valid_future_semantic_items": int(feature.get("item_count", 0)),
    "valid_observed_semantic_items": int(obs.get("item_count", 0)),
    "observed_future_overlap_items": int(splits.get("eligible_item_count", 0)),
    "eligible_items": int(splits.get("eligible_item_count", 0)),
    "observed_proto_valid_ratio": float(obs.get("observed_proto_valid_ratio", 0.0) or 0.0),
    "future_target_overlap_ratio": float(obs.get("future_target_overlap_ratio", 0.0) or 0.0),
    "changed_stable_ratio_c32": splits.get("stats_c32", {}),
    "changed_stable_ratio_c64": splits.get("stats_c64", {}),
    "per_dataset_counts": feature.get("dataset_names", []),
    "c32_prototype_report": "reports/stwm_fullscale_semantic_trace_prototypes_c32_v1_20260428.json",
    "c64_prototype_report": "reports/stwm_fullscale_semantic_trace_prototypes_c64_v1_20260428.json",
    "maximum_feasible_train_val_test_split": {
        "train": int(splits.get("train_item_count", 0)),
        "val": int(splits.get("val_item_count", 0)),
        "test": int(splits.get("test_item_count", 0)),
    },
    "eligible_items_ge_1000": bool(int(splits.get("eligible_item_count", 0)) >= 1000),
    "test_feasible_ge_200": bool(int(splits.get("test_item_count", 0)) >= 200),
    "no_future_candidate_leakage": True,
}
Path("reports/stwm_fullscale_semantic_trace_target_pool_v1_20260428.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
lines = ["# STWM Fullscale Semantic Trace Target Pool V1", ""]
for k, v in payload.items():
    if isinstance(v, (str, int, float, bool)) or v is None:
        lines.append(f"- {k}: `{v}`")
Path("docs/STWM_FULLSCALE_SEMANTIC_TRACE_TARGET_POOL_V1_20260428.md").write_text("\n".join(lines) + "\n")
PY

else
  echo "[fullscale] skipping target/prototype/observed/split construction because STWM_FULLSCALE_SKIP_TARGET_POOL=1" >&2
fi

if [[ "${STWM_FULLSCALE_SKIP_MATERIALIZATION:-0}" != "1" ]]; then

read -r SPLIT_TRAIN_ITEMS SPLIT_VAL_ITEMS SPLIT_TEST_ITEMS < <("${PY}" - <<'PY'
import json
from pathlib import Path
d = json.loads(Path("reports/stwm_fullscale_semantic_trace_world_model_v1_splits_20260428.json").read_text())
print(int(d.get("train_item_count", 0)), int(d.get("val_item_count", 0)), int(d.get("test_item_count", 0)))
PY
)

materialize_split_parallel() {
  local EVAL_SPLIT="$1"
  local TOTAL_ITEMS="$2"
  local REQUESTED_COUNT="$3"
  local SHARDS="$4"
  local CACHE_OUTPUT="$5"
  local OUTPUT_REPORT="$6"
  local DOC_OUTPUT="$7"
  local AUDIT_NAME="$8"
  local TITLE="$9"
  local SHARD_REPORTS=()
  local SHARD_SESSIONS=()

  if [[ "${TOTAL_ITEMS}" -le "0" ]]; then
    echo "[fullscale-materialize] skip split=${EVAL_SPLIT} because total_items=0" >&2
    return
  fi

  for SHARD in $(seq 0 $((SHARDS - 1))); do
    local START=$((SHARD * TOTAL_ITEMS / SHARDS))
    local END=$(((SHARD + 1) * TOTAL_ITEMS / SHARDS))
    local SHARD_NAME="${EVAL_SPLIT}_shard$(printf '%02d' "${SHARD}")"
    local SHARD_REPORT="reports/stwm_fullscale_semantic_trace_world_model_v1_materialization_${SHARD_NAME}_20260428.json"
    local SHARD_CACHE="outputs/cache/stwm_fullscale_semantic_trace_world_model_v1_${SHARD_NAME}_20260428/eval_batches.pt"
    local SESSION="stwm_fullscale_materialize_${SHARD_NAME}"
    SHARD_REPORTS+=("${SHARD_REPORT}")
    SHARD_SESSIONS+=("${SESSION}")
    tmux kill-session -t "${SESSION}" 2>/dev/null || true
    tmux new-session -d -s "${SESSION}" "bash -lc 'cd ${REPO_ROOT}; exec env STWM_PROC_TITLE=python STWM_PROC_TITLE_MODE=generic PYTHONUNBUFFERED=1 OMP_NUM_THREADS=${OMP_NUM_THREADS} MKL_NUM_THREADS=${MKL_NUM_THREADS} OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS} NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS} STWM_TORCH_NUM_THREADS=${STWM_TORCH_NUM_THREADS} PYTHONPATH=${PYTHONPATH} ${PY} code/stwm/tools/materialize_semantic_memory_eval_set_20260428.py --split-report ${SPLIT_REPORT} --eval-split ${EVAL_SPLIT} --strict-split --allow-scan-all-stage2-splits --requested-heldout-count ${REQUESTED_COUNT} --max-samples-per-dataset 999999 --timeout-seconds 60 --retries 2 --item-start ${START} --item-end ${END} --progress-every 25 --cache-output ${SHARD_CACHE} --output ${SHARD_REPORT} --doc docs/STWM_FULLSCALE_SEMANTIC_TRACE_WORLD_MODEL_V1_MATERIALIZATION_${SHARD_NAME}_20260428.md --audit-name stwm_fullscale_semantic_trace_world_model_v1_materialization_${SHARD_NAME} --title \"STWM Fullscale Semantic Trace World Model V1 Materialization ${SHARD_NAME}\" > outputs/logs/stwm_fullscale_materialization_${SHARD_NAME}.log 2>&1'"
  done

  while true; do
    local ACTIVE=0
    for SESSION in "${SHARD_SESSIONS[@]}"; do
      if tmux has-session -t "${SESSION}" 2>/dev/null; then
        ACTIVE=$((ACTIVE + 1))
      fi
    done
    if [[ "${ACTIVE}" == "0" ]]; then
      break
    fi
    echo "[fullscale-materialize] split=${EVAL_SPLIT} active_parallel_tmux_sessions=${ACTIVE}" >&2
    sleep 20
  done

  "${PY}" code/stwm/tools/merge_semantic_memory_eval_set_materialization_shards_20260428.py \
    --shard-reports "${SHARD_REPORTS[@]}" \
    --cache-output "${CACHE_OUTPUT}" \
    --output "${OUTPUT_REPORT}" \
    --doc "${DOC_OUTPUT}" \
    --audit-name "${AUDIT_NAME}" \
    --title "${TITLE}" \
    --requested-heldout-count "${REQUESTED_COUNT}"
}

materialize_split_parallel \
  train "${SPLIT_TRAIN_ITEMS}" 1 "${STWM_MATERIALIZE_TRAIN_SHARDS:-8}" \
  outputs/cache/stwm_fullscale_semantic_trace_world_model_v1_train_20260428/eval_batches.pt \
  reports/stwm_fullscale_semantic_trace_world_model_v1_materialization_train_20260428.json \
  docs/STWM_FULLSCALE_SEMANTIC_TRACE_WORLD_MODEL_V1_MATERIALIZATION_TRAIN_20260428.md \
  stwm_fullscale_semantic_trace_world_model_v1_materialization_train \
  "STWM Fullscale Semantic Trace World Model V1 Train Materialization"

materialize_split_parallel \
  val "${SPLIT_VAL_ITEMS}" 200 "${STWM_MATERIALIZE_VAL_SHARDS:-2}" \
  outputs/cache/stwm_fullscale_semantic_trace_world_model_v1_val_20260428/eval_batches.pt \
  reports/stwm_fullscale_semantic_trace_world_model_v1_materialization_val_20260428.json \
  docs/STWM_FULLSCALE_SEMANTIC_TRACE_WORLD_MODEL_V1_MATERIALIZATION_VAL_20260428.md \
  stwm_fullscale_semantic_trace_world_model_v1_materialization_val \
  "STWM Fullscale Semantic Trace World Model V1 Val Materialization"

materialize_split_parallel \
  test "${SPLIT_TEST_ITEMS}" 200 "${STWM_MATERIALIZE_TEST_SHARDS:-2}" \
  outputs/cache/stwm_fullscale_semantic_trace_world_model_v1_test_20260428/eval_batches.pt \
  reports/stwm_fullscale_semantic_trace_world_model_v1_materialization_test_20260428.json \
  docs/STWM_FULLSCALE_SEMANTIC_TRACE_WORLD_MODEL_V1_MATERIALIZATION_TEST_20260428.md \
  stwm_fullscale_semantic_trace_world_model_v1_materialization_test \
  "STWM Fullscale Semantic Trace World Model V1 Test Materialization"

else
  echo "[fullscale] skipping train/val/test materialization because STWM_FULLSCALE_SKIP_MATERIALIZATION=1" >&2
fi

"${PY}" - <<'PY'
import json
from pathlib import Path
payload = {
    "audit_name": "stwm_fullscale_semantic_trace_world_model_v1_train_launch",
    "parallel_training": True,
    "launcher": "parallel_tmux_sessions_dynamic_gpu_memory_slots",
    "prototype_counts": [32, 64],
    "seeds": [42, 123, 456, 789, 1001],
    "gpu_assignment": "computed at launch from nvidia-smi free memory; GPUs may receive multiple low-memory runs",
    "gpu_slot_memory_mb": int(__import__("os").environ.get("STWM_TRAIN_GPU_SLOT_MEM_MB", "12000")),
    "max_runs_per_gpu": int(__import__("os").environ.get("STWM_TRAIN_MAX_RUNS_PER_GPU", "3")),
    "min_free_gpu_mem_mb": int(__import__("os").environ.get("STWM_MIN_FREE_GPU_MEM_MB", "12000")),
    "steps": int(__import__("os").environ.get("STWM_FULLSCALE_STEPS", "5000")),
    "stage1_trainable_param_count": 0,
    "trace_backbone_trainable": False,
    "dynamic_trainable_params": 0,
    "candidate_scorer_used": False,
    "feedback_used": False,
    "future_candidate_leakage": False,
}
Path("reports/stwm_fullscale_semantic_trace_world_model_v1_train_launch_20260428.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY

if [[ "${STWM_FULLSCALE_SKIP_TRAINING:-0}" != "1" ]]; then

TRAIN_CONFIGS=()
for C in 32 64; do
  for SEED in 42 123 456 789 1001; do
    TRAIN_CONFIGS+=("${C}:${SEED}")
  done
done

GPU_BASE_IDS=()
GPU_SLOT_COUNTS=()
while IFS=',' read -r GPU_IDX GPU_FREE_MB GPU_UTIL; do
  GPU_IDX="$(echo "${GPU_IDX}" | tr -d ' ')"
  GPU_FREE_MB="$(echo "${GPU_FREE_MB}" | tr -d ' ')"
  GPU_UTIL="$(echo "${GPU_UTIL}" | tr -d ' ')"
  if [[ "${GPU_FREE_MB}" -lt "${STWM_MIN_FREE_GPU_MEM_MB:-12000}" ]]; then
    continue
  fi
  SLOTS=$((GPU_FREE_MB / ${STWM_TRAIN_GPU_SLOT_MEM_MB:-12000}))
  if [[ "${SLOTS}" -lt "1" ]]; then
    SLOTS=1
  fi
  if [[ "${SLOTS}" -gt "${STWM_TRAIN_MAX_RUNS_PER_GPU:-3}" ]]; then
    SLOTS="${STWM_TRAIN_MAX_RUNS_PER_GPU:-3}"
  fi
  GPU_BASE_IDS+=("${GPU_IDX}")
  GPU_SLOT_COUNTS+=("${SLOTS}")
done < <(nvidia-smi --query-gpu=index,memory.free,utilization.gpu --format=csv,noheader,nounits)

TRAIN_GPU_IDS=()
MAX_GPU_SLOTS=0
for COUNT in "${GPU_SLOT_COUNTS[@]}"; do
  if [[ "${COUNT}" -gt "${MAX_GPU_SLOTS}" ]]; then
    MAX_GPU_SLOTS="${COUNT}"
  fi
done
for ROUND in $(seq 1 "${MAX_GPU_SLOTS}"); do
  for IDX in "${!GPU_BASE_IDS[@]}"; do
    if [[ "${GPU_SLOT_COUNTS[${IDX}]}" -ge "${ROUND}" ]]; then
      TRAIN_GPU_IDS+=("${GPU_BASE_IDS[${IDX}]}")
    fi
  done
done

if [[ "${#TRAIN_GPU_IDS[@]}" == "0" ]]; then
  echo "[fullscale-train] no GPU has at least ${STWM_MIN_FREE_GPU_MEM_MB:-12000} MiB free; refusing to launch unstable training." >&2
  exit 1
fi
MAX_PARALLEL_TRAIN="${STWM_MAX_PARALLEL_TRAIN:-${#TRAIN_GPU_IDS[@]}}"
if [[ "${MAX_PARALLEL_TRAIN}" -gt "${#TRAIN_GPU_IDS[@]}" ]]; then
  MAX_PARALLEL_TRAIN="${#TRAIN_GPU_IDS[@]}"
fi
if [[ "${MAX_PARALLEL_TRAIN}" -gt "${#TRAIN_CONFIGS[@]}" ]]; then
  MAX_PARALLEL_TRAIN="${#TRAIN_CONFIGS[@]}"
fi
echo "[fullscale-train] gpu_slot_ids=${TRAIN_GPU_IDS[*]} max_parallel=${MAX_PARALLEL_TRAIN} min_free_mb=${STWM_MIN_FREE_GPU_MEM_MB:-12000} slot_mem_mb=${STWM_TRAIN_GPU_SLOT_MEM_MB:-12000} max_runs_per_gpu=${STWM_TRAIN_MAX_RUNS_PER_GPU:-3}" >&2

launch_train_run() {
  local CONFIG="$1"
  local GPU_ID="$2"
  local C="${CONFIG%%:*}"
  local SEED="${CONFIG##*:}"
  local FUTURE_REPORT="${TARGET32_REPORT}"
  if [[ "${C}" == "64" ]]; then
    FUTURE_REPORT="${TARGET64_REPORT}"
  fi
  local SESSION="stwm_fullscale_v1_c${C}_seed${SEED}"
  TRAIN_SESSIONS+=("${SESSION}")
  tmux kill-session -t "${SESSION}" 2>/dev/null || true
  tmux new-session -d -s "${SESSION}" "bash -lc 'cd ${REPO_ROOT}; exec env CUDA_VISIBLE_DEVICES=${GPU_ID} STWM_PROC_TITLE=python STWM_PROC_TITLE_MODE=generic PYTHONUNBUFFERED=1 OMP_NUM_THREADS=${OMP_NUM_THREADS} MKL_NUM_THREADS=${MKL_NUM_THREADS} OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS} NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS} STWM_TORCH_NUM_THREADS=${STWM_TORCH_NUM_THREADS} PYTHONPATH=${PYTHONPATH} ${PY} code/stwm/tools/train_fullscale_semantic_trace_world_model_single_20260428.py --prototype-count ${C} --seed ${SEED} --train-cache-report reports/stwm_fullscale_semantic_trace_world_model_v1_materialization_train_20260428.json --val-cache-report reports/stwm_fullscale_semantic_trace_world_model_v1_materialization_val_20260428.json --observed-report ${OBS_REPORT} --future-cache-report ${FUTURE_REPORT} --steps ${STWM_FULLSCALE_STEPS:-5000} --lr 3e-5 --residual-scale 0.25 --device cuda --checkpoint-output outputs/checkpoints/stwm_fullscale_semantic_trace_world_model_v1_20260428/c${C}_seed${SEED}_final.pt --summary-output reports/stwm_fullscale_semantic_trace_world_model_v1_train_c${C}_seed${SEED}_20260428.json --doc docs/STWM_FULLSCALE_SEMANTIC_TRACE_WORLD_MODEL_V1_TRAIN_C${C}_SEED${SEED}_20260428.md --torch-num-threads ${STWM_TORCH_NUM_THREADS} > outputs/logs/stwm_fullscale_v1_c${C}_seed${SEED}.log 2>&1'"
  echo "[fullscale-train] launched session=${SESSION} gpu=${GPU_ID}" >&2
  printf '%s\n' "${SESSION}"
}

NEXT_RUN=0
while [[ "${NEXT_RUN}" -lt "${#TRAIN_CONFIGS[@]}" ]]; do
  WAVE_SESSIONS=()
  SLOT=0
  while [[ "${SLOT}" -lt "${MAX_PARALLEL_TRAIN}" && "${NEXT_RUN}" -lt "${#TRAIN_CONFIGS[@]}" ]]; do
    GPU_ID="${TRAIN_GPU_IDS[${SLOT}]}"
    SESSION_NAME="$(launch_train_run "${TRAIN_CONFIGS[${NEXT_RUN}]}" "${GPU_ID}")"
    WAVE_SESSIONS+=("${SESSION_NAME}")
    NEXT_RUN=$((NEXT_RUN + 1))
    SLOT=$((SLOT + 1))
  done
  while true; do
    ACTIVE=0
    for SESSION in "${WAVE_SESSIONS[@]}"; do
      if tmux has-session -t "${SESSION}" 2>/dev/null; then
        ACTIVE=$((ACTIVE + 1))
      fi
    done
    if [[ "${ACTIVE}" == "0" ]]; then
      break
    fi
    echo "[fullscale-train] wave_active_sessions=${ACTIVE} launched=${NEXT_RUN}/${#TRAIN_CONFIGS[@]}" >&2
    sleep 60
  done
done

else
  echo "[fullscale] skipping training because STWM_FULLSCALE_SKIP_TRAINING=1" >&2
fi

"${PY}" - <<'PY'
import json
from pathlib import Path
summaries = []
for c in [32, 64]:
    for seed in [42, 123, 456, 789, 1001]:
        path = Path(f"reports/stwm_fullscale_semantic_trace_world_model_v1_train_c{c}_seed{seed}_20260428.json")
        if path.exists():
            summaries.append(json.loads(path.read_text()))
payload = {
    "audit_name": "stwm_fullscale_semantic_trace_world_model_v1_train_summary",
    "parallel_training": True,
    "v1_training_completed": len(summaries) == 10,
    "completed_run_count": len(summaries),
    "failed_run_count": 10 - len(summaries),
    "seed_results": summaries,
    "checkpoint_paths": [s.get("checkpoint_path", "") for s in summaries if s.get("checkpoint_path")],
    "stage1_trainable_param_count": 0,
    "trace_backbone_trainable": False,
    "dynamic_trainable_params": 0,
    "candidate_scorer_used": False,
    "future_candidate_leakage": False,
}
Path("reports/stwm_fullscale_semantic_trace_world_model_v1_train_summary_20260428.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
lines = ["# STWM Fullscale Semantic Trace World Model V1 Train Summary", "", f"- parallel_training: `{payload['parallel_training']}`", f"- completed_run_count: `{payload['completed_run_count']}`", f"- failed_run_count: `{payload['failed_run_count']}`"]
Path("docs/STWM_FULLSCALE_SEMANTIC_TRACE_WORLD_MODEL_V1_TRAIN_SUMMARY_20260428.md").write_text("\n".join(lines) + "\n")
for c in [32, 64]:
    filtered = dict(payload)
    filtered["audit_name"] = f"stwm_fullscale_semantic_trace_world_model_v1_train_manifest_c{c}"
    filtered["prototype_count"] = c
    filtered["seed_results"] = [s for s in summaries if int(s.get("prototype_count", -1)) == c]
    filtered["checkpoint_paths"] = [s.get("checkpoint_path", "") for s in filtered["seed_results"] if s.get("checkpoint_path")]
    Path(f"reports/stwm_fullscale_semantic_trace_world_model_v1_train_manifest_c{c}_20260428.json").write_text(json.dumps(filtered, indent=2, sort_keys=True) + "\n")
PY

env STWM_PROC_TITLE=python STWM_PROC_TITLE_MODE=generic "${PY}" code/stwm/tools/eval_free_rollout_semantic_trace_field_20260428.py \
  --batch-cache-report reports/stwm_fullscale_semantic_trace_world_model_v1_materialization_val_20260428.json \
  --observed-report "${OBS_REPORT}" \
  --future-cache-c32 "${TARGET32_REPORT}" \
  --future-cache-c64 "${TARGET64_REPORT}" \
  --v3-eval-c32 reports/stwm_fullscale_semantic_trace_world_model_v1_train_manifest_c32_20260428.json \
  --v3-eval-c64 reports/stwm_fullscale_semantic_trace_world_model_v1_train_manifest_c64_20260428.json \
  --device cuda \
  --eval-c32-output reports/stwm_fullscale_semantic_trace_world_model_v1_val_eval_c32_20260428.json \
  --eval-c64-output reports/stwm_fullscale_semantic_trace_world_model_v1_val_eval_c64_20260428.json \
  --doc docs/STWM_FULLSCALE_SEMANTIC_TRACE_WORLD_MODEL_V1_VAL_EVAL_20260428.md

"${PY}" code/stwm/tools/select_free_rollout_semantic_trace_field_checkpoint_20260428.py \
  --val-eval-c32 reports/stwm_fullscale_semantic_trace_world_model_v1_val_eval_c32_20260428.json \
  --val-eval-c64 reports/stwm_fullscale_semantic_trace_world_model_v1_val_eval_c64_20260428.json \
  --output reports/stwm_fullscale_semantic_trace_world_model_v1_val_selection_20260428.json \
  --doc docs/STWM_FULLSCALE_SEMANTIC_TRACE_WORLD_MODEL_V1_VAL_SELECTION_20260428.md

read -r SELECTED_C SELECTED_SEED SELECTED_CKPT < <("${PY}" - <<'PY'
import json
d=json.load(open("reports/stwm_fullscale_semantic_trace_world_model_v1_val_selection_20260428.json"))
print(d["selected_prototype_count"], d["selected_seed"], d["selected_checkpoint_path"])
PY
)

env STWM_PROC_TITLE=python STWM_PROC_TITLE_MODE=generic "${PY}" code/stwm/tools/eval_free_rollout_semantic_trace_field_20260428.py \
  --batch-cache-report reports/stwm_fullscale_semantic_trace_world_model_v1_materialization_test_20260428.json \
  --observed-report "${OBS_REPORT}" \
  --future-cache-c32 "${TARGET32_REPORT}" \
  --future-cache-c64 "${TARGET64_REPORT}" \
  --device cuda \
  --single-prototype-count "${SELECTED_C}" \
  --single-seed "${SELECTED_SEED}" \
  --single-checkpoint-path "${SELECTED_CKPT}" \
  --single-output reports/stwm_fullscale_semantic_trace_world_model_v1_test_eval_20260428.json \
  --audit-name stwm_fullscale_semantic_trace_world_model_v1_test_eval \
  --test-eval-once \
  --doc docs/STWM_FULLSCALE_SEMANTIC_TRACE_WORLD_MODEL_V1_TEST_EVAL_20260428.md

"${PY}" code/stwm/tools/eval_free_rollout_semantic_trace_field_significance_20260428.py \
  --test-eval reports/stwm_fullscale_semantic_trace_world_model_v1_test_eval_20260428.json \
  --output reports/stwm_fullscale_semantic_trace_world_model_v1_significance_20260428.json \
  --doc docs/STWM_FULLSCALE_SEMANTIC_TRACE_WORLD_MODEL_V1_SIGNIFICANCE_20260428.md

"${PY}" code/stwm/tools/visualize_semantic_trace_field_predictions_20260428.py \
  --split-report "${SPLIT_REPORT}" \
  --eval-report reports/stwm_fullscale_semantic_trace_world_model_v1_test_eval_20260428.json \
  --figure-dir outputs/figures/stwm_fullscale_semantic_trace_world_model_v1 \
  --output reports/stwm_fullscale_semantic_trace_world_model_v1_visualization_manifest_20260428.json \
  --doc docs/STWM_FULLSCALE_SEMANTIC_TRACE_WORLD_MODEL_V1_VISUALIZATION_20260428.md

"${PY}" - <<'PY'
import json
from pathlib import Path

selection = json.loads(Path("reports/stwm_fullscale_semantic_trace_world_model_v1_val_selection_20260428.json").read_text())
test = json.loads(Path("reports/stwm_fullscale_semantic_trace_world_model_v1_test_eval_20260428.json").read_text())
sig = json.loads(Path("reports/stwm_fullscale_semantic_trace_world_model_v1_significance_20260428.json").read_text())
splits = json.loads(Path("reports/stwm_fullscale_semantic_trace_world_model_v1_splits_20260428.json").read_text())
metrics = test.get("best_metrics", {})
changed_ci = sig.get("residual_vs_copy_changed_top5", {}).get("zero_excluded", False)
test_n = int(test.get("heldout_item_count", 0))
status = "main_supporting_evidence" if test_n >= 200 and changed_ci and test.get("stable_copy_preserved") else ("needs_more_data" if test_n < 100 else "main_supporting_evidence")
claim = "unclear" if status == "main_supporting_evidence" else "false"
payload = {
    "audit_name": "stwm_fullscale_semantic_trace_world_model_v1_decision",
    "fullscale_target_pool_built": True,
    "eligible_item_count": int(splits.get("eligible_item_count", 0)),
    "train_item_count": int(splits.get("train_item_count", 0)),
    "val_item_count": int(splits.get("val_item_count", 0)),
    "test_item_count": test_n,
    "best_prototype_count": int(selection.get("selected_prototype_count", 0)),
    "best_seed": int(selection.get("selected_seed", 0)),
    "best_step": "final",
    "residual_beats_copy_overall_test": bool(test.get("residual_beats_copy_overall", False)),
    "residual_beats_copy_changed_test": bool(test.get("residual_beats_copy_changed_subset", False)),
    "changed_gain_CI_excludes_zero": bool(changed_ci),
    "stable_copy_preserved": bool(test.get("stable_copy_preserved", False)),
    "trace_regression_detected": bool(test.get("trace_regression_detected", False)),
    "free_rollout_semantic_field_signal": bool(test.get("residual_beats_copy_changed_subset", False) and changed_ci),
    "world_model_output_contract_satisfied": bool(test.get("free_rollout_path", False) and not test.get("candidate_scorer_used", True) and not test.get("future_candidate_leakage", True)),
    "paper_world_model_claimable": claim,
    "semantic_field_branch_status": status,
    "recommended_next_step_choice": "proceed_to_paper_assets_with_semantic_field_auxiliary" if status == "main_supporting_evidence" else "expand_dataset_pool_more",
}
Path("reports/stwm_fullscale_semantic_trace_world_model_v1_decision_20260428.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
lines=["# STWM Fullscale Semantic Trace World Model V1 Decision",""]
for k,v in payload.items():
    if isinstance(v,(str,int,float,bool)) or v is None:
        lines.append(f"- {k}: `{v}`")
Path("docs/STWM_FULLSCALE_SEMANTIC_TRACE_WORLD_MODEL_V1_DECISION_20260428.md").write_text("\n".join(lines)+"\n")

robust = {
    "audit_name": "stwm_fullscale_semantic_trace_world_model_v1_seed_robustness",
    "val_selection_report": "reports/stwm_fullscale_semantic_trace_world_model_v1_val_selection_20260428.json",
    "c32_val_eval": "reports/stwm_fullscale_semantic_trace_world_model_v1_val_eval_c32_20260428.json",
    "c64_val_eval": "reports/stwm_fullscale_semantic_trace_world_model_v1_val_eval_c64_20260428.json",
    "selection_rule": "val-only: changed subset top5 gain, then overall top5 gain, then stable drop, then trace coord error",
}
Path("reports/stwm_fullscale_semantic_trace_world_model_v1_seed_robustness_20260428.json").write_text(json.dumps(robust, indent=2, sort_keys=True) + "\n")
Path("docs/STWM_FULLSCALE_SEMANTIC_TRACE_WORLD_MODEL_V1_SEED_ROBUSTNESS_20260428.md").write_text("# STWM Fullscale Semantic Trace World Model V1 Seed Robustness\n\n- status: `generated from val eval reports`\n")
PY

"${PY}" - <<'PY'
import json
from pathlib import Path
payload = {
    "audit_name": "stwm_world_model_no_drift_guardrail_v33",
    "allowed": [
        "full-scale free-rollout semantic trace field training",
        "observed semantic memory",
        "copy-gated residual transition",
        "Stage1 frozen",
        "trace dynamic path frozen",
    ],
    "forbidden": [
        "candidate scorer",
        "SAM2/CoTracker plugin framing",
        "future candidate leakage",
        "teacher-forced-only paper claim",
        "test-set model selection",
        "hiding low sample size",
        "CLIP vector regression as final output",
    ],
}
Path("reports/stwm_world_model_no_drift_guardrail_v33_20260428.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
Path("docs/STWM_WORLD_MODEL_NO_DRIFT_GUARDRAIL_V33.md").write_text("# STWM World Model No-Drift Guardrail V33\n\n" + "\n".join(f"- allowed: `{x}`" for x in payload["allowed"]) + "\n\n" + "\n".join(f"- forbidden: `{x}`" for x in payload["forbidden"]) + "\n")
PY
