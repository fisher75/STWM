#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/raid/chen034/workspace/stwm}"
LOG="${LOG:-${REPO_ROOT}/outputs/logs/stwm_mixed_v2_after_condland.log}"
CHECK_INTERVAL_SECONDS="${CHECK_INTERVAL_SECONDS:-120}"
QUIET_CHECKS_REQUIRED="${QUIET_CHECKS_REQUIRED:-3}"

mkdir -p "$(dirname "${LOG}")"

active_condland_training_count() {
  ps -eo args= \
    | grep -E "/home/chen034/miniconda3/envs/condland/bin/python .*/CELLVQ_final/ablation_study/code/(ablation_study_driver.py|run_stage1_chain_variant.py)" \
    | grep -v grep \
    | wc -l \
    | tr -d " "
}

{
  echo "[resume-guard] started $(date)"
  quiet_checks=0
  while true; do
    active="$(active_condland_training_count)"
    echo "[resume-guard] $(date) active_condland_training=${active} quiet=${quiet_checks}/${QUIET_CHECKS_REQUIRED}"
    if [[ "${active}" == "0" ]]; then
      quiet_checks=$((quiet_checks + 1))
      if [[ "${quiet_checks}" -ge "${QUIET_CHECKS_REQUIRED}" ]]; then
        break
      fi
    else
      quiet_checks=0
    fi
    sleep "${CHECK_INTERVAL_SECONDS}"
  done

  echo "[resume-guard] condland quiet window satisfied $(date); launching STWM mixed fullscale V2"
  cd "${REPO_ROOT}"
  export REPO_ROOT
  export PY="${PY:-/home/chen034/miniconda3/envs/stwm/bin/python}"
  export PYTHONPATH="${REPO_ROOT}/code"
  export STWM_PROC_TITLE="${STWM_PROC_TITLE:-python}"
  export STWM_PROC_TITLE_MODE="${STWM_PROC_TITLE_MODE:-generic}"
  export STWM_MIXED_V2_SKIP_MATERIALIZATION="${STWM_MIXED_V2_SKIP_MATERIALIZATION:-1}"
  export STWM_MIXED_V2_MAX_PARALLEL_TRAIN="${STWM_MIXED_V2_MAX_PARALLEL_TRAIN:-8}"
  export STWM_MIXED_V2_TRAIN_MAX_RUNS_PER_GPU="${STWM_MIXED_V2_TRAIN_MAX_RUNS_PER_GPU:-2}"
  export STWM_MIXED_V2_TRAIN_GPU_SLOT_MEM_MB="${STWM_MIXED_V2_TRAIN_GPU_SLOT_MEM_MB:-22000}"
  export STWM_MIXED_V2_MIN_FREE_GPU_MEM_MB="${STWM_MIXED_V2_MIN_FREE_GPU_MEM_MB:-16000}"
  export OMP_NUM_THREADS="${OMP_NUM_THREADS:-12}"
  export MKL_NUM_THREADS="${MKL_NUM_THREADS:-12}"
  export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-12}"
  export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-12}"
  export STWM_TORCH_NUM_THREADS="${STWM_TORCH_NUM_THREADS:-12}"

  bash code/stwm/tools/run_mixed_fullscale_semantic_trace_world_model_v2_20260428.sh
  rc=$?
  echo "[resume-guard] STWM mixed fullscale V2 finished rc=${rc} $(date)"
  exit "${rc}"
} >> "${LOG}" 2>&1
