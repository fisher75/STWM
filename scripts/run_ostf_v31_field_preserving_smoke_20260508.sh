#!/usr/bin/env bash
set -euo pipefail

ROOT="/raid/chen034/workspace/stwm"
cd "${ROOT}"
PY="${STWM_PYTHON:-/home/chen034/miniconda3/envs/stwm/bin/python}"
export PYTHONPATH="$ROOT/code:${PYTHONPATH:-}"

LOG_DIR="logs/stwm_ostf_v31_field_preserving_smoke_20260508"
mkdir -p "${LOG_DIR}" reports/stwm_ostf_v31_field_preserving_runs

pick_gpu() {
  python - <<'PY'
import subprocess
try:
    out = subprocess.check_output([
        "nvidia-smi",
        "--query-gpu=index,memory.free,memory.used",
        "--format=csv,noheader,nounits",
    ], text=True)
    rows = []
    for line in out.strip().splitlines():
        idx, free, used = [int(x.strip()) for x in line.split(",")[:3]]
        rows.append((free, -used, idx))
    rows.sort(reverse=True)
    print(rows[0][2] if rows else 0)
except Exception:
    print(0)
PY
}

run_one() {
  local name="$1"; shift
  local report="reports/stwm_ostf_v31_field_preserving_runs/${name}.json"
  if [[ -s "${report}" ]] && jq -e '.completed == true' "${report}" >/dev/null 2>&1; then
    echo "[skip] ${name} already completed"
    return 0
  fi
  local gpu
  gpu="$(pick_gpu)"
  echo "[launch] ${name} gpu=${gpu}"
  CUDA_VISIBLE_DEVICES="${gpu}" "$PY" code/stwm/tools/train_ostf_field_preserving_v31_20260508.py \
    --experiment-name "${name}" "$@" --smoke --amp \
    > "${LOG_DIR}/${name}.log" 2>&1 &
}

run_one v31_field_m128_h32_seed42_smoke --horizon 32 --m-points 128 --seed 42 --steps 800 --eval-interval 400 --batch-size 64 --hidden-dim 192 --field-layers 2 --temporal-layers 2 --heads 6 --num-workers 4
run_one v31_field_m128_h64_seed42_smoke --horizon 64 --m-points 128 --seed 42 --steps 800 --eval-interval 400 --batch-size 48 --hidden-dim 192 --field-layers 2 --temporal-layers 2 --heads 6 --num-workers 4
run_one v31_field_m128_h96_seed42_smoke --horizon 96 --m-points 128 --seed 42 --steps 800 --eval-interval 400 --batch-size 40 --hidden-dim 192 --field-layers 2 --temporal-layers 2 --heads 6 --num-workers 4
run_one v31_field_m512_h32_seed42_smoke --horizon 32 --m-points 512 --seed 42 --steps 800 --eval-interval 400 --batch-size 24 --hidden-dim 192 --field-layers 2 --temporal-layers 2 --heads 6 --num-workers 4
run_one v31_field_m512_h64_seed42_smoke --horizon 64 --m-points 512 --seed 42 --steps 800 --eval-interval 400 --batch-size 16 --hidden-dim 192 --field-layers 2 --temporal-layers 2 --heads 6 --num-workers 4

wait
"$PY" code/stwm/tools/aggregate_ostf_field_preserving_v31_smoke_20260508.py
