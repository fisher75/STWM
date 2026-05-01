#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/raid/chen034/workspace/stwm}"
PY="${PY:-/home/chen034/miniconda3/envs/stwm/bin/python}"
export PYTHONPATH="${REPO_ROOT}/code${PYTHONPATH:+:${PYTHONPATH}}"
export STWM_PROC_TITLE="${STWM_PROC_TITLE:-python}"
export STWM_PROC_TITLE_MODE="${STWM_PROC_TITLE_MODE:-generic}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-16}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-16}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-16}"
export STWM_TORCH_NUM_THREADS="${STWM_TORCH_NUM_THREADS:-16}"
cd "${REPO_ROOT}"

RUN_TAG="fstf_strong_copyaware_baselines_v8_20260501"
CKPT_ROOT="outputs/checkpoints/${RUN_TAG}"
LOG_ROOT="logs/${RUN_TAG}"
STATUS_ROOT="outputs/run_status/${RUN_TAG}"
mkdir -p "${CKPT_ROOT}" "${LOG_ROOT}" "${STATUS_ROOT}" reports docs

BASELINES=(copy_residual_mlp copy_residual_transformer copy_gated_residual_no_trace copy_gated_residual_trace_only copy_gated_residual_plain_trace_semantic)
SEEDS=(42 123 456 789 1001)
STEPS="${STEPS:-5000}"
MAX_PARALLEL="${MAX_PARALLEL:-25}"
MAX_RUNS_PER_GPU="${MAX_RUNS_PER_GPU:-6}"
GPU_MEM_RESERVE_MB="${GPU_MEM_RESERVE_MB:-8000}"
POLL_SECONDS="${POLL_SECONDS:-60}"

active_sessions() {
  tmux ls 2>/dev/null | awk -F: '/^fstf_v8_/ {print $1}' | wc -l | tr -d ' '
}

gpu_active_count() {
  local gpu="$1"
  tmux ls 2>/dev/null | awk -F: '/^fstf_v8_/ {print $1}' | while read -r sess; do
    tmux show-environment -t "$sess" CUDA_VISIBLE_DEVICES 2>/dev/null | sed 's/^CUDA_VISIBLE_DEVICES=//'
  done | awk -v g="${gpu}" '$1==g {n++} END {print n+0}'
}

select_gpu() {
  local best_gpu="" best_score=-999999
  while IFS=, read -r gpu used total; do
    gpu="$(echo "$gpu" | xargs)"
    used="$(echo "$used" | xargs)"
    total="$(echo "$total" | xargs)"
    [[ -z "${gpu}" ]] && continue
    local active free score
    active="$(gpu_active_count "${gpu}")"
    free=$(( total - used ))
    (( free > GPU_MEM_RESERVE_MB )) || continue
    (( active < MAX_RUNS_PER_GPU )) || continue
    score=$(( free - active * 20000 ))
    if (( score > best_score )); then
      best_score="${score}"
      best_gpu="${gpu}"
    fi
  done < <(nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits)
  [[ -n "${best_gpu}" ]] && { echo "${best_gpu}"; return 0; }
  return 1
}

launch_one() {
  local baseline="$1" seed="$2"
  local out_dir="${CKPT_ROOT}/${baseline}/${seed}"
  local ckpt="${out_dir}/checkpoint.pt"
  local train_summary="${out_dir}/train_summary.json"
  local eval_summary="${out_dir}/eval_test.json"
  local log="${LOG_ROOT}/${baseline}_seed${seed}.log"
  local status="${STATUS_ROOT}/${baseline}_seed${seed}.status.json"
  local session="fstf_v8_${baseline}_s${seed}"
  local run_script="${LOG_ROOT}/${baseline}_seed${seed}.run.sh"
  if [[ -f "${ckpt}" && -f "${train_summary}" && -f "${eval_summary}" && -s "${log}" ]]; then
    echo "[fstf-v8] reuse completed ${baseline} seed=${seed}" >&2
    return
  fi
  if tmux has-session -t "${session}" 2>/dev/null; then
    echo "[fstf-v8] keep running ${session}" >&2
    return
  fi
  while (( "$(active_sessions)" >= MAX_PARALLEL )); do sleep "${POLL_SECONDS}"; done
  local gpu
  until gpu="$(select_gpu)"; do
    echo "[fstf-v8] waiting for GPU slot" >&2
    sleep "${POLL_SECONDS}"
  done
  mkdir -p "${out_dir}"
  cat > "${run_script}" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "${REPO_ROOT}"
export CUDA_VISIBLE_DEVICES="${gpu}"
export STWM_PROC_TITLE=python
export STWM_PROC_TITLE_MODE=generic
export PYTHONUNBUFFERED=1
export PYTHONPATH="${PYTHONPATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS}"
export STWM_TORCH_NUM_THREADS="${STWM_TORCH_NUM_THREADS}"
python - <<PY
import json,time
from pathlib import Path
Path("${status}").write_text(json.dumps({"status":"running","baseline":"${baseline}","seed":${seed},"tmux_session":"${session}","cuda_visible_devices":"${gpu}","start_timestamp":time.time(),"launched_command":"${PY} train_fstf_strong_copyaware_baseline_v8 --baseline ${baseline} --seed ${seed} --steps ${STEPS}"}, indent=2)+"\\n")
PY
"${PY}" code/stwm/tools/train_fstf_strong_copyaware_baseline_v8_20260501.py \
  --baseline "${baseline}" --prototype-count 32 --seed "${seed}" \
  --train-cache-report reports/stwm_mixed_fullscale_v2_materialization_train_20260428.json \
  --val-cache-report reports/stwm_mixed_fullscale_v2_materialization_val_20260428.json \
  --observed-report reports/stwm_mixed_observed_semantic_prototype_targets_v2_20260428.json \
  --future-cache-report reports/stwm_fullscale_semantic_trace_prototype_targets_c32_v1_20260428.json \
  --steps "${STEPS}" --lr 3e-4 --d-model 256 --layers 3 --heads 4 --residual-scale 0.25 --device cuda \
  --checkpoint-output "${ckpt}" --summary-output "${train_summary}" --progress-every 100
"${PY}" code/stwm/tools/eval_fstf_strong_copyaware_baseline_v8_20260501.py \
  --baseline "${baseline}" --checkpoint "${ckpt}" \
  --test-cache-report reports/stwm_mixed_fullscale_v2_materialization_test_20260428.json \
  --observed-report reports/stwm_mixed_observed_semantic_prototype_targets_v2_20260428.json \
  --future-cache-report reports/stwm_fullscale_semantic_trace_prototype_targets_c32_v1_20260428.json \
  --device cuda --output "${eval_summary}"
python - <<PY
import json,time
from pathlib import Path
p=Path("${status}"); d=json.loads(p.read_text()); log=Path("${log}"); ck=Path("${ckpt}")
d.update({"status":"completed","end_timestamp":time.time(),"checkpoint_path":str(ck),"checkpoint_exists":ck.exists(),"checkpoint_mtime":ck.stat().st_mtime if ck.exists() else 0,"train_summary_path":"${train_summary}","eval_summary_path":"${eval_summary}","log_path":str(log),"log_size_bytes":log.stat().st_size if log.exists() else 0})
p.write_text(json.dumps(d, indent=2, sort_keys=True)+"\\n")
PY
EOF
  chmod +x "${run_script}"
  tmux kill-session -t "${session}" 2>/dev/null || true
  tmux new-session -d -s "${session}" "bash -lc '${run_script}' > '${log}' 2>&1"
  tmux set-environment -t "${session}" CUDA_VISIBLE_DEVICES "${gpu}"
  echo "[fstf-v8] launched ${session} gpu=${gpu} log=${log}" >&2
}

for baseline in "${BASELINES[@]}"; do
  for seed in "${SEEDS[@]}"; do
    launch_one "${baseline}" "${seed}"
  done
done

while true; do
  active="$(active_sessions)"
  echo "[fstf-v8] active_sessions=${active}" >&2
  [[ "${active}" == "0" ]] && break
  sleep "${POLL_SECONDS}"
done

echo "[fstf-v8] training jobs finished" >&2

mkdir -p "${CKPT_ROOT}/copy_semantic_memory_baseline" "${CKPT_ROOT}/oracle_change_gate_upper_bound"
"${PY}" code/stwm/tools/eval_fstf_strong_copyaware_baseline_v8_20260501.py \
  --baseline copy_semantic_memory_baseline \
  --test-cache-report reports/stwm_mixed_fullscale_v2_materialization_test_20260428.json \
  --observed-report reports/stwm_mixed_observed_semantic_prototype_targets_v2_20260428.json \
  --future-cache-report reports/stwm_fullscale_semantic_trace_prototype_targets_c32_v1_20260428.json \
  --device cuda --output "${CKPT_ROOT}/copy_semantic_memory_baseline/eval_test.json"
"${PY}" code/stwm/tools/eval_fstf_strong_copyaware_baseline_v8_20260501.py \
  --baseline oracle_change_gate_upper_bound \
  --test-cache-report reports/stwm_mixed_fullscale_v2_materialization_test_20260428.json \
  --observed-report reports/stwm_mixed_observed_semantic_prototype_targets_v2_20260428.json \
  --future-cache-report reports/stwm_fullscale_semantic_trace_prototype_targets_c32_v1_20260428.json \
  --device cuda --output "${CKPT_ROOT}/oracle_change_gate_upper_bound/eval_test.json"
"${PY}" code/stwm/tools/aggregate_fstf_strong_copyaware_baseline_v8_20260501.py
"${PY}" code/stwm/tools/audit_fstf_v8_live_artifacts_20260501.py
echo "[fstf-v8] aggregate and live artifact audit finished" >&2
