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

RUN_TAG="fstf_same_output_baselines_v7_20260501"
CKPT_ROOT="outputs/checkpoints/${RUN_TAG}"
LOG_ROOT="logs/${RUN_TAG}"
STATUS_ROOT="outputs/run_status/${RUN_TAG}"
MANIFEST="reports/stwm_fstf_same_output_baseline_run_manifest_v7_20260501.json"
SUITE_JSON="reports/stwm_fstf_same_output_baseline_suite_v7_20260501.json"
BOOT_JSON="reports/stwm_fstf_same_output_baseline_bootstrap_v7_20260501.json"
DOC="docs/STWM_FSTF_SAME_OUTPUT_BASELINE_SUITE_V7_20260501.md"
mkdir -p "${CKPT_ROOT}" "${LOG_ROOT}" "${STATUS_ROOT}" reports docs

BASELINES=(
  trace_only_ar_transformer
  semantic_only_memory_transition
  trace_semantic_transformer
  slotformer_like_trace_unit_dynamics
  dino_wm_like_latent_dynamics_proxy
)
SEEDS=(42 123 456)
PROTOTYPE_COUNT="${PROTOTYPE_COUNT:-32}"
STEPS="${STEPS:-5000}"
LR="${LR:-3e-4}"
D_MODEL="${D_MODEL:-192}"
LAYERS="${LAYERS:-2}"
HEADS="${HEADS:-4}"
MAX_PARALLEL="${MAX_PARALLEL:-15}"
POLL_SECONDS="${POLL_SECONDS:-60}"
GPU_UTIL_MAX="${GPU_UTIL_MAX:-90}"
GPU_MEM_USED_MAX_MB="${GPU_MEM_USED_MAX_MB:-150000}"
GPU_UTIL_MAX_FOR_DOUBLE="${GPU_UTIL_MAX_FOR_DOUBLE:-75}"
GPU_MEM_USED_MAX_MB_FOR_DOUBLE="${GPU_MEM_USED_MAX_MB_FOR_DOUBLE:-120000}"
GPU_UTIL_MAX_FOR_TRIPLE="${GPU_UTIL_MAX_FOR_TRIPLE:-45}"
GPU_MEM_USED_MAX_MB_FOR_TRIPLE="${GPU_MEM_USED_MAX_MB_FOR_TRIPLE:-80000}"
MAX_RUNS_PER_GPU="${MAX_RUNS_PER_GPU:-5}"
GPU_MEM_RESERVE_MB="${GPU_MEM_RESERVE_MB:-8000}"
FORCE_RERUN="${FORCE_RERUN:-0}"

TRAIN_REPORT="${TRAIN_REPORT:-reports/stwm_mixed_fullscale_v2_materialization_train_20260428.json}"
VAL_REPORT="${VAL_REPORT:-reports/stwm_mixed_fullscale_v2_materialization_val_20260428.json}"
TEST_REPORT="${TEST_REPORT:-reports/stwm_mixed_fullscale_v2_materialization_test_20260428.json}"
OBS_REPORT="${OBS_REPORT:-reports/stwm_mixed_observed_semantic_prototype_targets_v2_20260428.json}"
FUTURE_REPORT="${FUTURE_REPORT:-reports/stwm_fullscale_semantic_trace_prototype_targets_c32_v1_20260428.json}"

active_training_sessions() {
  tmux ls 2>/dev/null | awk -F: '/^fstf_v7_/ {print $1}' | wc -l | tr -d ' '
}

gpu_active_count() {
  local gpu="$1"
  tmux ls 2>/dev/null | awk -F: '/^fstf_v7_/ {print $1}' | while read -r sess; do
    tmux show-environment -t "$sess" CUDA_VISIBLE_DEVICES 2>/dev/null | sed 's/^CUDA_VISIBLE_DEVICES=//'
  done | awk -v g="${gpu}" '$1==g {n++} END {print n+0}'
}

select_gpu() {
  local stats
  stats="$(nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits)"
  local best_gpu="" best_score=-999999
  while IFS=, read -r gpu mem total util; do
    gpu="$(echo "$gpu" | xargs)"
    mem="$(echo "$mem" | xargs)"
    total="$(echo "$total" | xargs)"
    util="$(echo "$util" | xargs)"
    [[ -z "${gpu}" ]] && continue
    local active cap
    active="$(gpu_active_count "${gpu}")"
    cap="${MAX_RUNS_PER_GPU}"
    local free=$(( total - mem ))
    if (( free <= GPU_MEM_RESERVE_MB )); then
      continue
    fi
    if (( active < cap )); then
      local score=$(( free - active * 20000 ))
      if (( score > best_score )); then
        best_score="${score}"
        best_gpu="${gpu}"
      fi
    fi
  done <<< "${stats}"
  if [[ -n "${best_gpu}" ]]; then
    echo "${best_gpu}"
    return 0
  fi
  return 1
}

write_manifest() {
  "${PY}" - <<PY
import json, time
from pathlib import Path
from stwm.tools.train_fstf_same_output_baseline_v7_20260501 import baseline_officiality_metadata
root=Path("${REPO_ROOT}")
statuses=[]
for p in sorted((root/"${STATUS_ROOT}").glob("*.status.json")):
    try:
        statuses.append(json.loads(p.read_text()))
    except Exception as exc:
        statuses.append({"path":str(p),"status":"unreadable","error":str(exc)})
payload={
  "audit_name":"stwm_fstf_same_output_baseline_run_manifest_v7",
  "run_tag":"${RUN_TAG}",
  "generated_at_unix":time.time(),
  "baselines":${BASELINES_JSON},
  "baseline_officiality": {name: baseline_officiality_metadata(name) for name in ${BASELINES_JSON}},
  "external_boundary_official_baselines": [
    {
      "baseline_name": "SAM2 official external",
      "baseline_family": "external_mask_tracking_boundary",
      "evidence_level": "official_external",
      "output_contract_matched": False,
      "uses_future_candidate_measurement": True,
      "allowed_table_placement": "external_boundary_table",
      "forbidden_claim": "Do not present as same-output STWM-FSTF world-model baseline."
    },
    {
      "baseline_name": "CoTracker official external",
      "baseline_family": "external_point_tracking_boundary",
      "evidence_level": "official_external",
      "output_contract_matched": False,
      "uses_future_candidate_measurement": True,
      "allowed_table_placement": "external_boundary_table",
      "forbidden_claim": "Do not present as same-output STWM-FSTF world-model baseline."
    },
    {
      "baseline_name": "Cutie official external",
      "baseline_family": "external_video_object_segmentation_boundary",
      "evidence_level": "official_external",
      "output_contract_matched": False,
      "uses_future_candidate_measurement": True,
      "allowed_table_placement": "external_boundary_table",
      "forbidden_claim": "Do not present as same-output STWM-FSTF world-model baseline."
    }
  ],
  "seeds":[42,123,456],
  "prototype_count":${PROTOTYPE_COUNT},
  "steps":${STEPS},
  "train_report":"${TRAIN_REPORT}",
  "val_report":"${VAL_REPORT}",
  "test_report":"${TEST_REPORT}",
  "gpu_jobs_launched": bool(statuses),
  "gpu_job_evidence": statuses,
}
out=root/"${MANIFEST}"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(payload, indent=2, sort_keys=True)+"\n")
PY
}

launch_one() {
  local baseline="$1"
  local seed="$2"
  local out_dir="${CKPT_ROOT}/${baseline}/${seed}"
  local ckpt="${out_dir}/checkpoint.pt"
  local train_summary="${out_dir}/train_summary.json"
  local eval_summary="${out_dir}/eval_test.json"
  local status="${STATUS_ROOT}/${baseline}_seed${seed}.status.json"
  local log="${LOG_ROOT}/${baseline}_seed${seed}.log"
  local run_script="${LOG_ROOT}/${baseline}_seed${seed}.run.sh"
  local session="fstf_v7_${baseline}_s${seed}"
  if tmux has-session -t "${session}" 2>/dev/null; then
    echo "[fstf-v7] keep running session=${session}" >&2
    return
  fi
  if [[ "${FORCE_RERUN}" != "1" && -f "${ckpt}" && -f "${train_summary}" && -f "${eval_summary}" && -s "${log}" ]]; then
    echo "[fstf-v7] reuse completed baseline=${baseline} seed=${seed}" >&2
    return
  fi
  while (( "$(active_training_sessions)" >= MAX_PARALLEL )); do
    echo "[fstf-v7] waiting active=$(active_training_sessions) max=${MAX_PARALLEL}" >&2
    sleep "${POLL_SECONDS}"
  done
  local gpu=""
  until gpu="$(select_gpu)"; do
    echo "[fstf-v7] waiting for GPU slot" >&2
    sleep "${POLL_SECONDS}"
  done
  mkdir -p "${out_dir}"
  tmux kill-session -t "${session}" 2>/dev/null || true
  cat > "${run_script}" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "${REPO_ROOT}"
export CUDA_VISIBLE_DEVICES="${gpu}"
export STWM_PROC_TITLE="python"
export STWM_PROC_TITLE_MODE="generic"
export PYTHONUNBUFFERED=1
export PYTHONPATH="${PYTHONPATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS}"
export STWM_TORCH_NUM_THREADS="${STWM_TORCH_NUM_THREADS}"
python - <<PY
import json, time
from pathlib import Path
Path("${status}").write_text(json.dumps({
  "status":"running",
  "baseline":"${baseline}",
  "seed":${seed},
  "tmux_session":"${session}",
  "cuda_visible_devices":"${gpu}",
  "start_timestamp":time.time(),
  "launched_command":"${PY} code/stwm/tools/train_fstf_same_output_baseline_v7_20260501.py --baseline ${baseline} --seed ${seed} --steps ${STEPS}"
}, indent=2)+"\\n")
PY
mark_failed() {
  local rc="\$1"
  if [[ "\${rc}" == "0" ]]; then
    return 0
  fi
  python - <<PY
import json, time
from pathlib import Path
status_path=Path("${status}")
payload={}
if status_path.exists():
    try:
        payload=json.loads(status_path.read_text())
    except Exception as exc:
        payload={"status_read_error": str(exc)}
payload.update({
  "status":"failed",
  "exit_code":int("\${rc}"),
  "end_timestamp":time.time(),
  "checkpoint_path":"${ckpt}",
  "checkpoint_exists":Path("${ckpt}").exists(),
  "train_summary_path":"${train_summary}",
  "train_summary_exists":Path("${train_summary}").exists(),
  "eval_summary_path":"${eval_summary}",
  "eval_summary_exists":Path("${eval_summary}").exists(),
  "log_path":"${log}",
  "log_size_bytes":Path("${log}").stat().st_size if Path("${log}").exists() else 0
})
status_path.write_text(json.dumps(payload, indent=2, sort_keys=True)+"\\n")
PY
}
trap 'rc=\$?; mark_failed "\$rc"' EXIT
"${PY}" code/stwm/tools/train_fstf_same_output_baseline_v7_20260501.py \
  --baseline "${baseline}" \
  --prototype-count "${PROTOTYPE_COUNT}" \
  --seed "${seed}" \
  --train-cache-report "${TRAIN_REPORT}" \
  --val-cache-report "${VAL_REPORT}" \
  --observed-report "${OBS_REPORT}" \
  --future-cache-report "${FUTURE_REPORT}" \
  --steps "${STEPS}" \
  --lr "${LR}" \
  --d-model "${D_MODEL}" \
  --layers "${LAYERS}" \
  --heads "${HEADS}" \
  --device cuda \
  --checkpoint-output "${ckpt}" \
  --summary-output "${train_summary}" \
  --progress-every 100
"${PY}" code/stwm/tools/eval_fstf_same_output_baseline_v7_20260501.py \
  --checkpoint "${ckpt}" \
  --test-cache-report "${TEST_REPORT}" \
  --observed-report "${OBS_REPORT}" \
  --future-cache-report "${FUTURE_REPORT}" \
  --device cuda \
  --output "${eval_summary}"
python - <<PY
import json, time
from pathlib import Path
status_path=Path("${status}")
payload=json.loads(status_path.read_text())
payload.update({
  "status":"completed",
  "end_timestamp":time.time(),
  "checkpoint_path":"${ckpt}",
  "checkpoint_mtime":Path("${ckpt}").stat().st_mtime if Path("${ckpt}").exists() else 0,
  "train_summary_path":"${train_summary}",
  "eval_summary_path":"${eval_summary}",
  "log_path":"${log}",
  "log_size_bytes":Path("${log}").stat().st_size if Path("${log}").exists() else 0
})
status_path.write_text(json.dumps(payload, indent=2, sort_keys=True)+"\\n")
PY
EOF
  chmod +x "${run_script}"
  tmux new-session -d -s "${session}" "bash -lc '${run_script}' > '${log}' 2>&1"
  tmux set-environment -t "${session}" CUDA_VISIBLE_DEVICES "${gpu}"
  echo "[fstf-v7] launched session=${session} gpu=${gpu} log=${log}" >&2
}

BASELINES_JSON="$(printf '%s\n' "${BASELINES[@]}" | "${PY}" -c 'import json,sys; print(json.dumps([x.strip() for x in sys.stdin if x.strip()]))')"
write_manifest

for baseline in "${BASELINES[@]}"; do
  for seed in "${SEEDS[@]}"; do
    launch_one "${baseline}" "${seed}"
    write_manifest
  done
done

while true; do
  active="$(active_training_sessions)"
  write_manifest
  echo "[fstf-v7] active_sessions=${active}" >&2
  if [[ "${active}" == "0" ]]; then
    break
  fi
  sleep "${POLL_SECONDS}"
done

"${PY}" code/stwm/tools/eval_fstf_same_output_baseline_v7_20260501.py \
  --aggregate-suite \
  --suite-dir "${CKPT_ROOT}" \
  --log-dir "${LOG_ROOT}" \
  --manifest "${MANIFEST}" \
  --output "${SUITE_JSON}" \
  --bootstrap-output "${BOOT_JSON}" \
  --doc "${DOC}" \
  --gpu-jobs-launched true
write_manifest
echo "[fstf-v7] all baseline jobs finished; suite=${SUITE_JSON}" >&2
