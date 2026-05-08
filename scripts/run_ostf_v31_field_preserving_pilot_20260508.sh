#!/usr/bin/env bash
set -euo pipefail

ROOT="/raid/chen034/workspace/stwm"
cd "${ROOT}"
PY="${STWM_PYTHON:-/home/chen034/miniconda3/envs/stwm/bin/python}"
export PYTHONPATH="$ROOT/code:${PYTHONPATH:-}"

LOG_DIR="logs/stwm_ostf_v31_field_preserving_pilot_20260508"
mkdir -p "${LOG_DIR}" reports/stwm_ostf_v31_field_preserving_runs

MANIFEST="reports/stwm_ostf_v31_field_preserving_pilot_launch_manifest_20260508.json"
STATUS="reports/stwm_ostf_v31_field_preserving_pilot_status_20260508.json"

gpu_list() {
  nvidia-smi --query-gpu=index,memory.free,memory.used --format=csv,noheader,nounits 2>/dev/null \
    | awk -F, '{gsub(/ /,"",$1); gsub(/ /,"",$2); gsub(/ /,"",$3); if (($2+0)>=40000 && ($3+0)<=170000) print "0 "$2" "$1; else if (($2+0)>=25000 && ($3+0)<=170000) print "1 "$2" "$1}' \
    | sort -k1,1n -k2,2nr | awk '{print $3}'
}
mapfile -t GPUS < <(gpu_list || true)
if [[ "${#GPUS[@]}" -eq 0 ]]; then GPUS=("0"); fi
RUN_INDEX=0

python - <<'PY'
import json, pathlib, time
path = pathlib.Path("reports/stwm_ostf_v31_field_preserving_pilot_launch_manifest_20260508.json")
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps({"generated_at_unix": time.time(), "session": "stwm_ostf_v31_field_preserving_pilot_20260508", "runs": []}, indent=2) + "\n")
PY

append_manifest() {
  local name="$1"; local gpu="$2"; local log="$3"; local cmd="$4"
  python - "$name" "$gpu" "$log" "$cmd" <<'PY'
import json, pathlib, sys
path = pathlib.Path("reports/stwm_ostf_v31_field_preserving_pilot_launch_manifest_20260508.json")
payload = json.loads(path.read_text())
payload["runs"].append({"experiment_name": sys.argv[1], "gpu": sys.argv[2], "log_path": sys.argv[3], "command": sys.argv[4]})
path.write_text(json.dumps(payload, indent=2) + "\n")
PY
}

run_one() {
  local name="$1"; shift
  local report="reports/stwm_ostf_v31_field_preserving_runs/${name}.json"
  if [[ -s "${report}" ]] && jq -e '.completed == true' "${report}" >/dev/null 2>&1; then
    echo "[skip] ${name} already completed"
    return 0
  fi
  local gpu="${GPUS[$((RUN_INDEX % ${#GPUS[@]}))]}"
  RUN_INDEX=$((RUN_INDEX + 1))
  local cmd="CUDA_VISIBLE_DEVICES=${gpu} '$PY' code/stwm/tools/train_ostf_field_preserving_v31_20260508.py --experiment-name ${name} $* --amp"
  append_manifest "${name}" "${gpu}" "${LOG_DIR}/${name}.log" "${cmd}"
  echo "[launch] ${name} gpu=${gpu}"
  bash -lc "${cmd}" > "${LOG_DIR}/${name}.log" 2>&1 &
}

run_one v31_field_m128_h32_seed42 --horizon 32 --m-points 128 --seed 42 --steps 4000 --eval-interval 1000 --batch-size 64 --hidden-dim 192 --field-layers 2 --temporal-layers 2 --heads 6 --num-workers 4
run_one v31_field_m128_h64_seed42 --horizon 64 --m-points 128 --seed 42 --steps 4000 --eval-interval 1000 --batch-size 48 --hidden-dim 192 --field-layers 2 --temporal-layers 2 --heads 6 --num-workers 4
run_one v31_field_m128_h96_seed42 --horizon 96 --m-points 128 --seed 42 --steps 4000 --eval-interval 1000 --batch-size 40 --hidden-dim 192 --field-layers 2 --temporal-layers 2 --heads 6 --num-workers 4
run_one v31_field_m512_h32_seed42 --horizon 32 --m-points 512 --seed 42 --steps 4000 --eval-interval 1000 --batch-size 24 --hidden-dim 192 --field-layers 2 --temporal-layers 2 --heads 6 --num-workers 4
run_one v31_field_m512_h64_seed42 --horizon 64 --m-points 512 --seed 42 --steps 4000 --eval-interval 1000 --batch-size 16 --hidden-dim 192 --field-layers 2 --temporal-layers 2 --heads 6 --num-workers 4
run_one v31_field_m512_h96_seed42 --horizon 96 --m-points 512 --seed 42 --steps 4000 --eval-interval 1000 --batch-size 12 --hidden-dim 192 --field-layers 2 --temporal-layers 2 --heads 6 --num-workers 4

python - <<'PY'
import json, pathlib, time
path = pathlib.Path("reports/stwm_ostf_v31_field_preserving_pilot_status_20260508.json")
path.write_text(json.dumps({"generated_at_unix": time.time(), "status": "launched", "completed": False}, indent=2) + "\n")
PY

wait
"$PY" code/stwm/tools/aggregate_ostf_field_preserving_v31_pilot_20260508.py
python - <<'PY'
import json, pathlib, time
path = pathlib.Path("reports/stwm_ostf_v31_field_preserving_pilot_status_20260508.json")
payload = {"generated_at_unix": time.time(), "status": "completed", "completed": True}
path.write_text(json.dumps(payload, indent=2) + "\n")
PY
