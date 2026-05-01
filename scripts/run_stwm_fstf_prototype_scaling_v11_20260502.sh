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
mkdir -p outputs/logs/stwm_fstf_scaling_v11_20260502 outputs/checkpoints/stwm_fstf_scaling_v11_20260502 reports docs
OBS_REPORT="${OBS_REPORT:-reports/stwm_mixed_observed_semantic_prototype_targets_v11_20260502.json}"
if [[ ! -s "${OBS_REPORT}" ]]; then
  OBS_REPORT="reports/stwm_mixed_observed_semantic_prototype_targets_v2_20260428.json"
fi

future_report_for_c() {
  case "$1" in
    16) echo "reports/stwm_fstf_future_semantic_trace_prototype_targets_c16_v11_20260502.json" ;;
    32) echo "reports/stwm_fullscale_semantic_trace_prototype_targets_c32_v1_20260428.json" ;;
    64) echo "reports/stwm_fullscale_semantic_trace_prototype_targets_c64_v1_20260428.json" ;;
    128) echo "reports/stwm_fstf_future_semantic_trace_prototype_targets_c128_v11_20260502.json" ;;
    *) return 1 ;;
  esac
}

gpu_for_slot() {
  local slot="$1"
  nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits |
    awk -F', ' '{free=$3-$2; score=free-$4*256; print $1,score}' |
    sort -k2,2nr | awk -v s="${slot}" 'NR==((s%8)+1){print $1}'
}

launch_one() {
  local C="$1" SEED="$2" SLOT="$3"
  local FUTURE_REPORT
  FUTURE_REPORT="$(future_report_for_c "${C}")"
  if [[ ! -s "${FUTURE_REPORT}" ]]; then
    echo "[v11-prototype] skip C=${C} seed=${SEED}; missing ${FUTURE_REPORT}" >&2
    return
  fi
  if [[ "${OBS_REPORT}" == *"v2_20260428.json" && "${C}" != "32" && "${C}" != "64" ]]; then
    echo "[v11-prototype] skip C=${C} seed=${SEED}; observed-v11 not ready and observed-v2 only contains C32/C64" >&2
    return
  fi
  local GPU_ID
  GPU_ID="$(gpu_for_slot "${SLOT}")"
  local NAME="prototype_c${C}_seed${SEED}"
  local SESSION="stwm_fstf_scaling_v11_${NAME}"
  local CKPT="outputs/checkpoints/stwm_fstf_scaling_v11_20260502/${NAME}/checkpoint.pt"
  local TRAIN="reports/stwm_fstf_scaling_v11_${NAME}_train_20260502.json"
  local EVAL="reports/stwm_fstf_scaling_v11_${NAME}_eval_20260502.json"
  local LOG="outputs/logs/stwm_fstf_scaling_v11_20260502/${NAME}.log"
  if [[ -s "${EVAL}" && -s "${CKPT}" ]]; then
    echo "[v11-prototype] already complete ${NAME}" >&2
    return
  fi
  tmux kill-session -t "${SESSION}" 2>/dev/null || true
  tmux new-session -d -s "${SESSION}" "bash -lc 'cd ${REPO_ROOT}; echo START \$(date -Is) session=${SESSION} cuda=${GPU_ID} observed=${OBS_REPORT}; export CUDA_VISIBLE_DEVICES=${GPU_ID}; export STWM_PROC_TITLE=python STWM_PROC_TITLE_MODE=generic PYTHONUNBUFFERED=1 PYTHONPATH=${PYTHONPATH}; ${PY} code/stwm/tools/train_stwm_fstf_scaling_v11_20260502.py --scaling-axis prototype --scaling-value C${C} --prototype-count ${C} --seed ${SEED} --future-cache-report ${FUTURE_REPORT} --observed-report ${OBS_REPORT} --steps ${STEPS} --checkpoint-output ${CKPT} --summary-output ${TRAIN} --device cuda && ${PY} code/stwm/tools/eval_stwm_fstf_scaling_v11_20260502.py --scaling-axis prototype --scaling-value C${C} --prototype-count ${C} --checkpoint ${CKPT} --future-cache-report ${FUTURE_REPORT} --observed-report ${OBS_REPORT} --output ${EVAL} --device cuda; echo END \$(date -Is)' > ${LOG} 2>&1"
  echo "${SESSION}"
}

slot=0
for C in 16 32 64 128; do
  for SEED in 42 123 456; do
    while [[ "$(tmux list-sessions 2>/dev/null | grep -c '^stwm_fstf_scaling_v11_prototype_')" -ge "${MAX_PARALLEL}" ]]; do sleep 15; done
    launch_one "${C}" "${SEED}" "${slot}" || true
    slot=$((slot+1))
  done
done

while tmux list-sessions 2>/dev/null | grep -q '^stwm_fstf_scaling_v11_prototype_'; do
  echo "[v11-prototype] active=$(tmux list-sessions 2>/dev/null | grep -c '^stwm_fstf_scaling_v11_prototype_')" >&2
  sleep 30
done

"${PY}" code/stwm/tools/aggregate_stwm_fstf_scaling_v11_20260502.py
