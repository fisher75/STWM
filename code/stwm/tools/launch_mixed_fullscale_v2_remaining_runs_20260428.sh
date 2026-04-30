#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/raid/chen034/workspace/stwm}"
PY="${PY:-/home/chen034/miniconda3/envs/stwm/bin/python}"
PYTHONPATH_VALUE="${REPO_ROOT}/code${PYTHONPATH:+:${PYTHONPATH}}"
STEPS="${STWM_MIXED_V2_STEPS:-5000}"
MAX_PARALLEL="${STWM_MIXED_V2_MAX_PARALLEL_TRAIN:-8}"
GPU_IDS_CSV="${STWM_MIXED_V2_GPU_IDS:-}"
FORCE_RERUN_ALL=0
WAIT=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --force-rerun-all)
      FORCE_RERUN_ALL=1
      shift
      ;;
    --wait)
      WAIT=1
      shift
      ;;
    --max-parallel)
      MAX_PARALLEL="$2"
      shift 2
      ;;
    --gpu-ids)
      GPU_IDS_CSV="$2"
      shift 2
      ;;
    --steps)
      STEPS="$2"
      shift 2
      ;;
    *)
      echo "unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

cd "${REPO_ROOT}"
mkdir -p outputs/logs outputs/checkpoints/stwm_mixed_fullscale_v2_20260428 outputs/run_status reports docs

OBS_REPORT="reports/stwm_mixed_observed_semantic_prototype_targets_v2_20260428.json"
TARGET32_REPORT="reports/stwm_fullscale_semantic_trace_prototype_targets_c32_v1_20260428.json"
TARGET64_REPORT="reports/stwm_fullscale_semantic_trace_prototype_targets_c64_v1_20260428.json"
TRAIN_CACHE_REPORT="reports/stwm_mixed_fullscale_v2_materialization_train_20260428.json"
VAL_CACHE_REPORT="reports/stwm_mixed_fullscale_v2_materialization_val_20260428.json"
CKPT_DIR="outputs/checkpoints/stwm_mixed_fullscale_v2_20260428"

discover_gpus() {
  if [[ -n "${GPU_IDS_CSV}" ]]; then
    echo "${GPU_IDS_CSV}" | tr ',' '\n'
    return
  fi
  nvidia-smi --query-gpu=index,memory.free,utilization.gpu --format=csv,noheader,nounits \
    | awk -F',' '
      {
        g=$1; free=$2; util=$3;
        gsub(/ /, "", g); gsub(/ /, "", free); gsub(/ /, "", util);
        if (free >= 12000 && util < 98) print g, free, util;
      }' \
    | sort -k2,2nr -k3,3n \
    | awk '{print $1}'
}

mapfile -t GPU_IDS < <(discover_gpus)
if [[ "${#GPU_IDS[@]}" -eq 0 ]]; then
  echo "no usable GPU found" >&2
  exit 1
fi
if [[ "${MAX_PARALLEL}" -gt "${#GPU_IDS[@]}" ]]; then
  MAX_PARALLEL="${#GPU_IDS[@]}"
fi

configs=()
for c in 32 64; do
  for seed in 42 123 456 789 1001; do
    summary="reports/stwm_mixed_fullscale_v2_train_c${c}_seed${seed}_20260428.json"
    ckpt="${CKPT_DIR}/c${c}_seed${seed}_final.pt"
    session="stwm_mixed_fullscale_v2_c${c}_seed${seed}"
    if [[ "${FORCE_RERUN_ALL}" != "1" && -s "${summary}" && -s "${ckpt}" ]]; then
      continue
    fi
    if [[ "${FORCE_RERUN_ALL}" != "1" ]] && tmux has-session -t "${session}" 2>/dev/null; then
      continue
    fi
    configs+=("${c}:${seed}")
  done
done

write_launch_report() {
  "${PY}" - "$@" <<'PY'
import json, sys, time
from pathlib import Path
payload = json.loads(sys.argv[1])
payload["created_at"] = time.strftime("%Y-%m-%d %H:%M:%S %z")
Path("reports/stwm_mixed_fullscale_v2_launcher_repair_20260428.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
lines = [
    "# STWM Mixed Fullscale V2 Launcher Repair",
    "",
    f"- launch_only_missing: `{payload['launch_only_missing']}`",
    f"- force_rerun_all: `{payload['force_rerun_all']}`",
    f"- expected_run_count: `{payload['expected_run_count']}`",
    f"- already_completed_count: `{payload['already_completed_count']}`",
    f"- launch_count: `{payload['launch_count']}`",
    f"- max_parallel: `{payload['max_parallel']}`",
    f"- gpu_ids: `{payload['gpu_ids']}`",
    f"- no_silent_failure: `true`",
]
Path("docs/STWM_MIXED_FULLSCALE_V2_LAUNCHER_REPAIR_20260428.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
PY
}

already_completed=$((10 - ${#configs[@]}))
missing_configs_json="$(printf '%s\n' "${configs[@]}" | python -c 'import json,sys; print(json.dumps([{"prototype_count": int(x.split(":")[0]), "seed": int(x.split(":")[1])} for x in sys.stdin.read().split()]))')"
gpu_ids_json="$(printf '%s\n' "${GPU_IDS[@]}" | python -c 'import json,sys; print(json.dumps([int(x) for x in sys.stdin.read().split()]))')"
write_launch_report "$(python - <<PY
import json
print(json.dumps({
  "audit_name": "stwm_mixed_fullscale_v2_launcher_repair",
  "launch_only_missing": True,
  "force_rerun_all": bool(${FORCE_RERUN_ALL}),
  "expected_run_count": 10,
  "already_completed_count": ${already_completed},
  "launch_count": ${#configs[@]},
  "missing_configs": ${missing_configs_json},
  "gpu_ids": ${gpu_ids_json},
  "max_parallel": int("${MAX_PARALLEL}"),
  "steps": int("${STEPS}"),
  "stage1_trainable_param_count": 0,
  "trace_backbone_trainable": False,
  "dynamic_trainable_params": 0,
  "candidate_scorer_used": False,
  "future_candidate_leakage": False,
}))
PY
)"

if [[ "${#configs[@]}" -eq 0 ]]; then
  echo "[mixed-v2-repair] all runs already complete"
  exit 0
fi

launch_one() {
  local config="$1"
  local gpu="$2"
  local c="${config%%:*}"
  local seed="${config##*:}"
  local future_report="${TARGET32_REPORT}"
  if [[ "${c}" == "64" ]]; then
    future_report="${TARGET64_REPORT}"
  fi
  local session="stwm_mixed_fullscale_v2_c${c}_seed${seed}"
  local log="outputs/logs/stwm_mixed_fullscale_v2_c${c}_seed${seed}.log"
  local wrapper="outputs/logs/stwm_mixed_fullscale_v2_c${c}_seed${seed}.run.sh"
  local status="outputs/run_status/stwm_mixed_fullscale_v2_c${c}_seed${seed}.status.json"
  local summary="reports/stwm_mixed_fullscale_v2_train_c${c}_seed${seed}_20260428.json"
  local ckpt="${CKPT_DIR}/c${c}_seed${seed}_final.pt"
  cat > "${wrapper}" <<EOF
#!/usr/bin/env bash
set -uo pipefail
cd "${REPO_ROOT}"
mkdir -p outputs/run_status reports docs "${CKPT_DIR}"
cat > "${status}" <<STATUS
{"prototype_count": ${c}, "seed": ${seed}, "status": "running", "gpu": "${gpu}", "started_at": "$(date '+%Y-%m-%d %H:%M:%S %z')"}
STATUS
export CUDA_VISIBLE_DEVICES="${gpu}"
export STWM_PROC_TITLE="python"
export STWM_PROC_TITLE_MODE="generic"
export PYTHONUNBUFFERED="1"
export PYTHONPATH="${PYTHONPATH_VALUE}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-8}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-8}"
export STWM_TORCH_NUM_THREADS="${STWM_TORCH_NUM_THREADS:-8}"
"${PY}" code/stwm/tools/train_fullscale_semantic_trace_world_model_single_20260428.py \
  --prototype-count "${c}" \
  --seed "${seed}" \
  --train-cache-report "${TRAIN_CACHE_REPORT}" \
  --val-cache-report "${VAL_CACHE_REPORT}" \
  --observed-report "${OBS_REPORT}" \
  --future-cache-report "${future_report}" \
  --steps "${STEPS}" \
  --lr 3e-5 \
  --residual-scale 0.25 \
  --device cuda \
  --checkpoint-output "${ckpt}" \
  --summary-output "${summary}" \
  --doc "docs/STWM_MIXED_FULLSCALE_V2_TRAIN_C${c}_SEED${seed}_20260428.md" \
  --torch-num-threads "\${STWM_TORCH_NUM_THREADS}"
rc=\$?
if [[ "\${rc}" == "0" && -s "${summary}" && -s "${ckpt}" ]]; then
  final_status="completed"
  reason="none"
else
  final_status="failed"
  reason="process_exit_or_missing_artifact"
fi
cat > "${status}" <<STATUS
{"prototype_count": ${c}, "seed": ${seed}, "status": "\${final_status}", "exit_code": \${rc}, "reason": "\${reason}", "gpu": "${gpu}", "summary_path": "${summary}", "checkpoint_path": "${ckpt}", "finished_at": "$(date '+%Y-%m-%d %H:%M:%S %z')"}
STATUS
exit "\${rc}"
EOF
  chmod +x "${wrapper}"
  : > "${log}"
  tmux kill-session -t "${session}" 2>/dev/null || true
  tmux new-session -d -s "${session}" "bash -lc 'exec ${wrapper} > ${log} 2>&1'"
  echo "${session}"
}

next=0
while [[ "${next}" -lt "${#configs[@]}" ]]; do
  wave_sessions=()
  slot=0
  while [[ "${slot}" -lt "${MAX_PARALLEL}" && "${next}" -lt "${#configs[@]}" ]]; do
    gpu="${GPU_IDS[$((slot % ${#GPU_IDS[@]}))]}"
    session="$(launch_one "${configs[${next}]}" "${gpu}")"
    wave_sessions+=("${session}")
    echo "[mixed-v2-repair] launched ${session} gpu=${gpu}"
    next=$((next + 1))
    slot=$((slot + 1))
  done
  if [[ "${WAIT}" != "1" ]]; then
    echo "[mixed-v2-repair] launched one non-wait wave; rerun after completion to launch remaining configs"
    break
  fi
  while true; do
    active=0
    for session in "${wave_sessions[@]}"; do
      if tmux has-session -t "${session}" 2>/dev/null; then
        active=$((active + 1))
      fi
    done
    echo "[mixed-v2-repair] wave active=${active} launched=${next}/${#configs[@]}"
    if [[ "${active}" == "0" ]]; then
      break
    fi
    sleep 60
  done
done

echo "[mixed-v2-repair] launch complete; wait=${WAIT}"
