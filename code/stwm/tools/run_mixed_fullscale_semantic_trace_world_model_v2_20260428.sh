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

cd "${REPO_ROOT}"
mkdir -p outputs/logs reports docs outputs/cache outputs/checkpoints

SPLIT_REPORT="reports/stwm_mixed_semantic_trace_world_model_v2_splits_20260428.json"
OBS_REPORT="reports/stwm_mixed_observed_semantic_prototype_targets_v2_20260428.json"
TARGET32_REPORT="reports/stwm_fullscale_semantic_trace_prototype_targets_c32_v1_20260428.json"
TARGET64_REPORT="reports/stwm_fullscale_semantic_trace_prototype_targets_c64_v1_20260428.json"
CKPT_DIR="outputs/checkpoints/stwm_mixed_fullscale_v2_20260428"

"${PY}" code/stwm/tools/build_mixed_fullscale_v2_final_assets_20260428.py --mode prepare

read -r SPLIT_TRAIN_ITEMS SPLIT_VAL_ITEMS SPLIT_TEST_ITEMS < <("${PY}" - <<'PY'
import json
from pathlib import Path
d = json.loads(Path("reports/stwm_mixed_semantic_trace_world_model_v2_splits_20260428.json").read_text())
print(int(d.get("train_item_count", 0)), int(d.get("val_item_count", 0)), int(d.get("test_item_count", 0)))
PY
)

materialize_split_parallel() {
  local SPLIT_REPORT_PATH="$1"
  local EVAL_SPLIT="$2"
  local TOTAL_ITEMS="$3"
  local REQUESTED_COUNT="$4"
  local SHARDS="$5"
  local CACHE_OUTPUT="$6"
  local OUTPUT_REPORT="$7"
  local DOC_OUTPUT="$8"
  local AUDIT_NAME="$9"
  local TITLE="${10}"
  local LOG_PREFIX="${11}"
  local SHARD_REPORTS=()
  local SHARD_SESSIONS=()

  if [[ "${TOTAL_ITEMS}" -le "0" ]]; then
    echo "[mixed-v2-materialize] skip split=${EVAL_SPLIT} because total_items=0" >&2
    return
  fi

  for SHARD in $(seq 0 $((SHARDS - 1))); do
    local START=$((SHARD * TOTAL_ITEMS / SHARDS))
    local END=$(((SHARD + 1) * TOTAL_ITEMS / SHARDS))
    local SHARD_NAME="${LOG_PREFIX}_${EVAL_SPLIT}_shard$(printf '%02d' "${SHARD}")"
    local SHARD_REPORT="reports/${SHARD_NAME}_20260428.json"
    local SHARD_CACHE="outputs/cache/${SHARD_NAME}_20260428/eval_batches.pt"
    local SESSION="${SHARD_NAME}"
    SHARD_REPORTS+=("${SHARD_REPORT}")
    SHARD_SESSIONS+=("${SESSION}")
    tmux kill-session -t "${SESSION}" 2>/dev/null || true
    tmux new-session -d -s "${SESSION}" "bash -lc 'cd ${REPO_ROOT}; exec env STWM_PROC_TITLE=python STWM_PROC_TITLE_MODE=generic PYTHONUNBUFFERED=1 OMP_NUM_THREADS=${OMP_NUM_THREADS} MKL_NUM_THREADS=${MKL_NUM_THREADS} OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS} NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS} STWM_TORCH_NUM_THREADS=${STWM_TORCH_NUM_THREADS} PYTHONPATH=${PYTHONPATH} ${PY} code/stwm/tools/materialize_semantic_memory_eval_set_20260428.py --split-report ${SPLIT_REPORT_PATH} --eval-split ${EVAL_SPLIT} --strict-split --allow-scan-all-stage2-splits --requested-heldout-count ${REQUESTED_COUNT} --max-samples-per-dataset 999999 --timeout-seconds 60 --retries 2 --item-start ${START} --item-end ${END} --progress-every 50 --cache-output ${SHARD_CACHE} --output ${SHARD_REPORT} --doc docs/${SHARD_NAME}.md --audit-name ${SHARD_NAME} --title \"${TITLE} ${SHARD_NAME}\" > outputs/logs/${SHARD_NAME}.log 2>&1'"
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
    echo "[mixed-v2-materialize] split=${EVAL_SPLIT} active_parallel_tmux_sessions=${ACTIVE}" >&2
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

if [[ "${STWM_MIXED_V2_SKIP_MATERIALIZATION:-0}" != "1" ]]; then
  materialize_split_parallel \
    "${SPLIT_REPORT}" train "${SPLIT_TRAIN_ITEMS}" 1 "${STWM_MIXED_V2_MATERIALIZE_TRAIN_SHARDS:-8}" \
    outputs/cache/stwm_mixed_fullscale_v2_train_20260428/eval_batches.pt \
    reports/stwm_mixed_fullscale_v2_materialization_train_20260428.json \
    docs/STWM_MIXED_FULLSCALE_V2_MATERIALIZATION_TRAIN_20260428.md \
    stwm_mixed_fullscale_v2_materialization_train \
    "STWM Mixed Fullscale V2 Train Materialization" \
    stwm_mixed_fullscale_v2_trainmat

  materialize_split_parallel \
    "${SPLIT_REPORT}" val "${SPLIT_VAL_ITEMS}" "${SPLIT_VAL_ITEMS}" "${STWM_MIXED_V2_MATERIALIZE_VAL_SHARDS:-4}" \
    outputs/cache/stwm_mixed_fullscale_v2_val_20260428/eval_batches.pt \
    reports/stwm_mixed_fullscale_v2_materialization_val_20260428.json \
    docs/STWM_MIXED_FULLSCALE_V2_MATERIALIZATION_VAL_20260428.md \
    stwm_mixed_fullscale_v2_materialization_val \
    "STWM Mixed Fullscale V2 Val Materialization" \
    stwm_mixed_fullscale_v2_valmat

  materialize_split_parallel \
    "${SPLIT_REPORT}" test "${SPLIT_TEST_ITEMS}" "${SPLIT_TEST_ITEMS}" "${STWM_MIXED_V2_MATERIALIZE_TEST_SHARDS:-4}" \
    outputs/cache/stwm_mixed_fullscale_v2_test_20260428/eval_batches.pt \
    reports/stwm_mixed_fullscale_v2_materialization_test_20260428.json \
    docs/STWM_MIXED_FULLSCALE_V2_MATERIALIZATION_TEST_20260428.md \
    stwm_mixed_fullscale_v2_materialization_test \
    "STWM Mixed Fullscale V2 Test Materialization" \
    stwm_mixed_fullscale_v2_mixedtestmat

  materialize_split_parallel \
    reports/stwm_mixed_fullscale_v2_splits_vspw_test_20260428.json test 229 229 2 \
    outputs/cache/stwm_mixed_fullscale_v2_vspw_test_20260428/eval_batches.pt \
    reports/stwm_mixed_fullscale_v2_materialization_vspw_test_20260428.json \
    docs/STWM_MIXED_FULLSCALE_V2_MATERIALIZATION_VSPW_TEST_20260428.md \
    stwm_mixed_fullscale_v2_materialization_vspw_test \
    "STWM Mixed Fullscale V2 VSPW Test Materialization" \
    stwm_mixed_fullscale_v2_vspwtestmat

  materialize_split_parallel \
    reports/stwm_mixed_fullscale_v2_splits_vipseg_test_20260428.json test 418 418 4 \
    outputs/cache/stwm_mixed_fullscale_v2_vipseg_test_20260428/eval_batches.pt \
    reports/stwm_mixed_fullscale_v2_materialization_vipseg_test_20260428.json \
    docs/STWM_MIXED_FULLSCALE_V2_MATERIALIZATION_VIPSEG_TEST_20260428.md \
    stwm_mixed_fullscale_v2_materialization_vipseg_test \
    "STWM Mixed Fullscale V2 VIPSeg Test Materialization" \
    stwm_mixed_fullscale_v2_vipsegtestmat
fi

"${PY}" - <<'PY'
import json, os, subprocess
from pathlib import Path
payload = {
    "audit_name": "stwm_mixed_fullscale_v2_train_launch",
    "parallel_training": True,
    "launcher": "parallel_tmux_sessions_dynamic_gpu_memory_slots",
    "prototype_counts": [32, 64],
    "seeds": [42, 123, 456, 789, 1001],
    "steps": int(os.environ.get("STWM_MIXED_V2_STEPS", "5000")),
    "stage1_trainable_param_count": 0,
    "trace_backbone_trainable": False,
    "dynamic_trainable_params": 0,
    "candidate_scorer_used": False,
    "feedback_used": False,
    "future_candidate_leakage": False,
    "gpu_assignment_policy": {
        "min_free_gpu_mem_mb": int(os.environ.get("STWM_MIXED_V2_MIN_FREE_GPU_MEM_MB", "16000")),
        "slot_mem_mb": int(os.environ.get("STWM_MIXED_V2_TRAIN_GPU_SLOT_MEM_MB", "20000")),
        "max_runs_per_gpu": int(os.environ.get("STWM_MIXED_V2_TRAIN_MAX_RUNS_PER_GPU", "2")),
    },
}
Path("reports/stwm_mixed_fullscale_v2_train_launch_20260428.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
Path("docs/STWM_MIXED_FULLSCALE_V2_TRAIN_SUMMARY_20260428.md").write_text("# STWM Mixed Fullscale V2 Train Summary\n\n- launch_created: `true`\n", encoding="utf-8")
PY

if [[ "${STWM_MIXED_V2_SKIP_TRAINING:-0}" != "1" ]]; then
  TRAIN_CONFIGS=()
  for C in 32 64; do
    for SEED in 42 123 456 789 1001; do
      SUMMARY_PATH="reports/stwm_mixed_fullscale_v2_train_c${C}_seed${SEED}_20260428.json"
      CKPT_PATH="${CKPT_DIR}/c${C}_seed${SEED}_final.pt"
      if [[ -s "${SUMMARY_PATH}" && -s "${CKPT_PATH}" ]]; then
        echo "[mixed-v2-train] skip completed c${C}_seed${SEED}" >&2
        continue
      fi
      TRAIN_CONFIGS+=("${C}:${SEED}")
    done
  done

  if [[ "${#TRAIN_CONFIGS[@]}" == "0" ]]; then
    echo "[mixed-v2-train] all C/seed runs already completed" >&2
  fi

  GPU_BASE_IDS=()
  GPU_SLOT_COUNTS=()
  while IFS=',' read -r GPU_IDX GPU_FREE_MB GPU_UTIL; do
    GPU_IDX="$(echo "${GPU_IDX}" | tr -d ' ')"
    GPU_FREE_MB="$(echo "${GPU_FREE_MB}" | tr -d ' ')"
    if [[ "${GPU_FREE_MB}" -lt "${STWM_MIXED_V2_MIN_FREE_GPU_MEM_MB:-16000}" ]]; then
      continue
    fi
    SLOTS=$((GPU_FREE_MB / ${STWM_MIXED_V2_TRAIN_GPU_SLOT_MEM_MB:-20000}))
    if [[ "${SLOTS}" -lt "1" ]]; then
      SLOTS=1
    fi
    if [[ "${SLOTS}" -gt "${STWM_MIXED_V2_TRAIN_MAX_RUNS_PER_GPU:-2}" ]]; then
      SLOTS="${STWM_MIXED_V2_TRAIN_MAX_RUNS_PER_GPU:-2}"
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
    echo "[mixed-v2-train] no GPU satisfies free-memory threshold" >&2
    exit 1
  fi
  MAX_PARALLEL_TRAIN="${STWM_MIXED_V2_MAX_PARALLEL_TRAIN:-6}"
  if [[ "${MAX_PARALLEL_TRAIN}" -gt "${#TRAIN_GPU_IDS[@]}" ]]; then
    MAX_PARALLEL_TRAIN="${#TRAIN_GPU_IDS[@]}"
  fi
  if [[ "${MAX_PARALLEL_TRAIN}" -gt "${#TRAIN_CONFIGS[@]}" ]]; then
    MAX_PARALLEL_TRAIN="${#TRAIN_CONFIGS[@]}"
  fi
  echo "[mixed-v2-train] gpu_slot_ids=${TRAIN_GPU_IDS[*]} max_parallel=${MAX_PARALLEL_TRAIN}" >&2

  launch_train_run() {
    local CONFIG="$1"
    local GPU_ID="$2"
    local C="${CONFIG%%:*}"
    local SEED="${CONFIG##*:}"
    local FUTURE_REPORT="${TARGET32_REPORT}"
    if [[ "${C}" == "64" ]]; then
      FUTURE_REPORT="${TARGET64_REPORT}"
    fi
    local SESSION="stwm_mixed_fullscale_v2_c${C}_seed${SEED}"
    tmux kill-session -t "${SESSION}" 2>/dev/null || true
    tmux new-session -d -s "${SESSION}" "bash -lc 'cd ${REPO_ROOT}; exec env CUDA_VISIBLE_DEVICES=${GPU_ID} STWM_PROC_TITLE=python STWM_PROC_TITLE_MODE=generic PYTHONUNBUFFERED=1 OMP_NUM_THREADS=${OMP_NUM_THREADS} MKL_NUM_THREADS=${MKL_NUM_THREADS} OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS} NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS} STWM_TORCH_NUM_THREADS=${STWM_TORCH_NUM_THREADS} PYTHONPATH=${PYTHONPATH} ${PY} code/stwm/tools/train_fullscale_semantic_trace_world_model_single_20260428.py --prototype-count ${C} --seed ${SEED} --train-cache-report reports/stwm_mixed_fullscale_v2_materialization_train_20260428.json --val-cache-report reports/stwm_mixed_fullscale_v2_materialization_val_20260428.json --observed-report ${OBS_REPORT} --future-cache-report ${FUTURE_REPORT} --steps ${STWM_MIXED_V2_STEPS:-5000} --lr 3e-5 --residual-scale 0.25 --device cuda --checkpoint-output ${CKPT_DIR}/c${C}_seed${SEED}_final.pt --summary-output reports/stwm_mixed_fullscale_v2_train_c${C}_seed${SEED}_20260428.json --doc docs/STWM_MIXED_FULLSCALE_V2_TRAIN_C${C}_SEED${SEED}_20260428.md --torch-num-threads ${STWM_TORCH_NUM_THREADS} > outputs/logs/stwm_mixed_fullscale_v2_c${C}_seed${SEED}.log 2>&1'"
    echo "${SESSION}"
  }

  NEXT_RUN=0
  while [[ "${NEXT_RUN}" -lt "${#TRAIN_CONFIGS[@]}" ]]; do
    WAVE_SESSIONS=()
    SLOT=0
    while [[ "${SLOT}" -lt "${MAX_PARALLEL_TRAIN}" && "${NEXT_RUN}" -lt "${#TRAIN_CONFIGS[@]}" ]]; do
      GPU_ID="${TRAIN_GPU_IDS[${SLOT}]}"
      SESSION_NAME="$(launch_train_run "${TRAIN_CONFIGS[${NEXT_RUN}]}" "${GPU_ID}")"
      WAVE_SESSIONS+=("${SESSION_NAME}")
      echo "[mixed-v2-train] launched ${SESSION_NAME} gpu=${GPU_ID}" >&2
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
      echo "[mixed-v2-train] wave_active_sessions=${ACTIVE} launched=${NEXT_RUN}/${#TRAIN_CONFIGS[@]}" >&2
      sleep 60
    done
  done
fi

"${PY}" - <<'PY'
import json
from pathlib import Path
summaries = []
failed = []
for c in [32, 64]:
    for seed in [42, 123, 456, 789, 1001]:
        path = Path(f"reports/stwm_mixed_fullscale_v2_train_c{c}_seed{seed}_20260428.json")
        if path.exists():
            summaries.append(json.loads(path.read_text()))
        else:
            failed.append({"prototype_count": c, "seed": seed, "reason": "summary_missing"})
payload = {
    "audit_name": "stwm_mixed_fullscale_v2_train_summary",
    "parallel_training": True,
    "mixed_training_completed": len(summaries) == 10,
    "completed_run_count": len(summaries),
    "failed_run_count": len(failed),
    "failed_runs": failed,
    "seed_results": summaries,
    "checkpoint_paths": [s.get("checkpoint_path", "") for s in summaries if s.get("checkpoint_path")],
    "stage1_trainable_param_count": 0,
    "trace_backbone_trainable": False,
    "dynamic_trainable_params": 0,
    "candidate_scorer_used": False,
    "future_candidate_leakage": False,
}
Path("reports/stwm_mixed_fullscale_v2_train_summary_20260428.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
for c in [32, 64]:
    filtered = dict(payload)
    filtered["audit_name"] = f"stwm_mixed_fullscale_v2_train_manifest_c{c}"
    filtered["prototype_count"] = c
    filtered["seed_results"] = [s for s in summaries if int(s.get("prototype_count", -1)) == c]
    filtered["checkpoint_paths"] = [s.get("checkpoint_path", "") for s in filtered["seed_results"] if s.get("checkpoint_path")]
    Path(f"reports/stwm_mixed_fullscale_v2_train_manifest_c{c}_20260428.json").write_text(json.dumps(filtered, indent=2, sort_keys=True) + "\n")
lines = ["# STWM Mixed Fullscale V2 Train Summary", "", f"- mixed_training_completed: `{payload['mixed_training_completed']}`", f"- completed_run_count: `{payload['completed_run_count']}`", f"- failed_run_count: `{payload['failed_run_count']}`"]
Path("docs/STWM_MIXED_FULLSCALE_V2_TRAIN_SUMMARY_20260428.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
PY

env STWM_PROC_TITLE=python STWM_PROC_TITLE_MODE=generic "${PY}" code/stwm/tools/eval_free_rollout_semantic_trace_field_20260428.py \
  --batch-cache-report reports/stwm_mixed_fullscale_v2_materialization_val_20260428.json \
  --observed-report "${OBS_REPORT}" \
  --future-cache-c32 "${TARGET32_REPORT}" \
  --future-cache-c64 "${TARGET64_REPORT}" \
  --v3-eval-c32 reports/stwm_mixed_fullscale_v2_train_manifest_c32_20260428.json \
  --v3-eval-c64 reports/stwm_mixed_fullscale_v2_train_manifest_c64_20260428.json \
  --device cuda \
  --eval-c32-output reports/stwm_mixed_fullscale_v2_val_eval_c32_20260428.json \
  --eval-c64-output reports/stwm_mixed_fullscale_v2_val_eval_c64_20260428.json \
  --doc docs/STWM_MIXED_FULLSCALE_V2_VAL_EVAL_20260428.md

"${PY}" code/stwm/tools/select_free_rollout_semantic_trace_field_checkpoint_20260428.py \
  --val-eval-c32 reports/stwm_mixed_fullscale_v2_val_eval_c32_20260428.json \
  --val-eval-c64 reports/stwm_mixed_fullscale_v2_val_eval_c64_20260428.json \
  --output reports/stwm_mixed_fullscale_v2_val_selection_20260428.json \
  --doc docs/STWM_MIXED_FULLSCALE_V2_VAL_SELECTION_20260428.md

read -r SELECTED_C SELECTED_SEED SELECTED_CKPT < <("${PY}" - <<'PY'
import json
d=json.load(open("reports/stwm_mixed_fullscale_v2_val_selection_20260428.json"))
print(d["selected_prototype_count"], d["selected_seed"], d["selected_checkpoint_path"])
PY
)

eval_selected() {
  local BATCH_REPORT="$1"
  local OUTPUT="$2"
  local AUDIT="$3"
  local DOC="$4"
  env STWM_PROC_TITLE=python STWM_PROC_TITLE_MODE=generic "${PY}" code/stwm/tools/eval_free_rollout_semantic_trace_field_20260428.py \
    --batch-cache-report "${BATCH_REPORT}" \
    --observed-report "${OBS_REPORT}" \
    --future-cache-c32 "${TARGET32_REPORT}" \
    --future-cache-c64 "${TARGET64_REPORT}" \
    --device cuda \
    --single-prototype-count "${SELECTED_C}" \
    --single-seed "${SELECTED_SEED}" \
    --single-checkpoint-path "${SELECTED_CKPT}" \
    --single-output "${OUTPUT}" \
    --audit-name "${AUDIT}" \
    --test-eval-once \
    --doc "${DOC}"
}

eval_selected reports/stwm_mixed_fullscale_v2_materialization_test_20260428.json \
  reports/stwm_mixed_fullscale_v2_mixed_test_eval_20260428.json \
  stwm_mixed_fullscale_v2_mixed_test_eval \
  docs/STWM_MIXED_FULLSCALE_V2_MIXED_TEST_EVAL_20260428.md

eval_selected reports/stwm_mixed_fullscale_v2_materialization_vspw_test_20260428.json \
  reports/stwm_mixed_fullscale_v2_vspw_test_eval_20260428.json \
  stwm_mixed_fullscale_v2_vspw_test_eval \
  docs/STWM_MIXED_FULLSCALE_V2_VSPW_TEST_EVAL_20260428.md

eval_selected reports/stwm_mixed_fullscale_v2_materialization_vipseg_test_20260428.json \
  reports/stwm_mixed_fullscale_v2_vipseg_test_eval_20260428.json \
  stwm_mixed_fullscale_v2_vipseg_test_eval \
  docs/STWM_MIXED_FULLSCALE_V2_VIPSEG_TEST_EVAL_20260428.md

"${PY}" code/stwm/tools/visualize_semantic_trace_field_predictions_20260428.py \
  --split-report "${SPLIT_REPORT}" \
  --eval-report reports/stwm_mixed_fullscale_v2_mixed_test_eval_20260428.json \
  --figure-dir outputs/figures/stwm_mixed_fullscale_v2 \
  --output reports/stwm_mixed_fullscale_v2_visualization_manifest_20260428.json \
  --doc docs/STWM_MIXED_FULLSCALE_V2_VISUALIZATION_20260428.md \
  --audit-name stwm_mixed_fullscale_v2_visualization_manifest \
  --title "STWM Mixed Fullscale V2 Visualization"

"${PY}" code/stwm/tools/build_mixed_fullscale_v2_final_assets_20260428.py --mode final
