#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="${REPO_ROOT:-/raid/chen034/workspace/stwm}"
PY="${PY:-/home/chen034/miniconda3/envs/stwm/bin/python}"
MAX_PARALLEL="${MAX_PARALLEL:-8}"
STEPS="${STEPS:-5000}"
export PYTHONPATH="${REPO_ROOT}/code${PYTHONPATH:+:${PYTHONPATH}}"
export STWM_PROC_TITLE=python
export STWM_PROC_TITLE_MODE=generic
export PYTHONUNBUFFERED=1
cd "${REPO_ROOT}"
mkdir -p outputs/logs/stwm_fstf_scaling_v12_20260502 outputs/checkpoints/stwm_fstf_scaling_v12_20260502 reports docs

gpu_for_slot() {
  local slot="$1"
  nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits |
    awk -F', ' '{free=$3-$2; score=free-$4*256; print $1,score}' |
    sort -k2,2nr | awk -v s="${slot}" 'NR==((s%8)+1){print $1}'
}

future_for_k() {
  case "$1" in
    8) echo "reports/stwm_fullscale_semantic_trace_prototype_targets_c32_v1_20260428.json" ;;
    16) echo "reports/stwm_fstf_trace_density_k16_prototype_targets_c32_v12_20260502.json" ;;
    32) echo "reports/stwm_fstf_trace_density_k32_prototype_targets_c32_v12_20260502.json" ;;
  esac
}

observed_for_k() {
  case "$1" in
    8) echo "reports/stwm_mixed_observed_semantic_prototype_targets_v11_20260502.json" ;;
    16) echo "reports/stwm_fstf_trace_density_k16_observed_targets_v12_20260502.json" ;;
    32) echo "reports/stwm_fstf_trace_density_k32_observed_targets_v12_20260502.json" ;;
  esac
}

batch_report_for_k() {
  local K="$1" SPLIT="$2"
  if [[ "${K}" == "8" ]]; then
    echo "reports/stwm_mixed_fullscale_v2_materialization_${SPLIT}_20260428.json"
  else
    echo "reports/stwm_fstf_trace_density_k${K}_batch_${SPLIT}_v12_20260502.json"
  fi
}

launch_one() {
  local K="$1" SEED="$2" SLOT="$3"
  local FUTURE OBS TRAIN_BATCH VAL_BATCH TEST_BATCH
  FUTURE="$(future_for_k "${K}")"
  OBS="$(observed_for_k "${K}")"
  TRAIN_BATCH="$(batch_report_for_k "${K}" train)"
  VAL_BATCH="$(batch_report_for_k "${K}" val)"
  TEST_BATCH="$(batch_report_for_k "${K}" test)"
  if [[ ! -s "${FUTURE}" || ! -s "${OBS}" || ! -s "${TRAIN_BATCH}" || ! -s "${VAL_BATCH}" || ! -s "${TEST_BATCH}" ]]; then
    echo "[v12-density] skip K=${K} seed=${SEED}; missing cache report" >&2
    return
  fi
  local GPU_ID
  GPU_ID="$(gpu_for_slot "${SLOT}")"
  local NAME="density_k${K}_seed${SEED}"
  local SESSION="stwm_fstf_scaling_v12_${NAME}"
  local CKPT="outputs/checkpoints/stwm_fstf_scaling_v12_20260502/${NAME}/checkpoint.pt"
  local TRAIN="reports/stwm_fstf_scaling_v12_${NAME}_train_20260502.json"
  local EVAL="reports/stwm_fstf_scaling_v12_${NAME}_eval_20260502.json"
  local LOG="outputs/logs/stwm_fstf_scaling_v12_20260502/${NAME}.log"
  if [[ -s "${EVAL}" && -s "${CKPT}" ]]; then
    echo "[v12-density] already complete ${NAME}" >&2
    return
  fi
  tmux kill-session -t "${SESSION}" 2>/dev/null || true
  tmux new-session -d -s "${SESSION}" "bash -lc 'cd ${REPO_ROOT}; echo START \$(date -Is) session=${SESSION} cuda=${GPU_ID}; export CUDA_VISIBLE_DEVICES=${GPU_ID} STWM_PROC_TITLE=python STWM_PROC_TITLE_MODE=generic PYTHONUNBUFFERED=1 PYTHONPATH=${PYTHONPATH}; ${PY} code/stwm/tools/train_stwm_fstf_scaling_v11_20260502.py --scaling-axis density --scaling-value K${K} --prototype-count 32 --seed ${SEED} --train-cache-report ${TRAIN_BATCH} --val-cache-report ${VAL_BATCH} --future-cache-report ${FUTURE} --observed-report ${OBS} --steps ${STEPS} --checkpoint-output ${CKPT} --summary-output ${TRAIN} --device cuda && ${PY} code/stwm/tools/eval_stwm_fstf_scaling_v11_20260502.py --scaling-axis density --scaling-value K${K} --prototype-count 32 --checkpoint ${CKPT} --test-cache-report ${TEST_BATCH} --future-cache-report ${FUTURE} --observed-report ${OBS} --output ${EVAL} --device cuda; echo END \$(date -Is)' > ${LOG} 2>&1"
  echo "${SESSION}"
}

for K in 8 16 32; do
  if [[ "${K}" != "8" ]]; then
    while [[ ! -s "reports/stwm_fstf_trace_density_cache_k${K}_v12_20260502.json" ]]; do
      echo "[v12-density] waiting for K${K} cache" >&2
      sleep 60
    done
  fi
  slot=0
  for SEED in 42 123 456; do
    while [[ "$(tmux list-sessions 2>/dev/null | grep -c '^stwm_fstf_scaling_v12_density_')" -ge "${MAX_PARALLEL}" ]]; do sleep 20; done
    launch_one "${K}" "${SEED}" "${slot}" || true
    slot=$((slot+1))
  done
done
while tmux list-sessions 2>/dev/null | grep -q '^stwm_fstf_scaling_v12_density_'; do sleep 30; done
