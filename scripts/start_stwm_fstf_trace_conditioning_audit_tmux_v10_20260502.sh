#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="${REPO_ROOT:-/raid/chen034/workspace/stwm}"
SESSION="${SESSION:-stwm_fstf_trace_conditioning_audit_v10}"
LOG="${LOG:-${REPO_ROOT}/logs/stwm_fstf_trace_conditioning_audit_v10_20260502.log}"
mkdir -p "$(dirname "${LOG}")"
cd "${REPO_ROOT}"
GPU="${CUDA_VISIBLE_DEVICES:-}"
if [[ -z "${GPU}" ]]; then
  GPU="$(nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits | awk -F, 'BEGIN{best=-1; gpu=0} {free=$3-$2; if (free>best) {best=free; gpu=$1}} END{gsub(/ /,"",gpu); print gpu}')"
fi
tmux kill-session -t "${SESSION}" 2>/dev/null || true
tmux new-session -d -s "${SESSION}" "CUDA_VISIBLE_DEVICES=${GPU} STWM_PROC_TITLE=python STWM_PROC_TITLE_MODE=generic bash scripts/run_stwm_fstf_trace_conditioning_audit_v10_20260502.sh > '${LOG}' 2>&1"
tmux set-environment -t "${SESSION}" CUDA_VISIBLE_DEVICES "${GPU}"
echo "[trace-v10] launched session=${SESSION} gpu=${GPU} log=${LOG}"
