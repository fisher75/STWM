#!/usr/bin/env bash
set -euo pipefail

ROOT="${STWM_ROOT:-/raid/chen034/workspace/stwm}"
PY="${STWM_PYTHON:-/home/chen034/miniconda3/envs/stwm/bin/python}"
export PYTHONPATH="$ROOT/code:${PYTHONPATH:-}"
cd "$ROOT"

pick_gpu() {
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo ""
    return 0
  fi
  nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
    | awk -F, '{gsub(/ /,"",$1); gsub(/ /,"",$2); if ($2>=25000) print $2" "$1}' \
    | sort -nr | head -1 | awk '{print $2}'
}

GPU="${CUDA_VISIBLE_DEVICES:-$(pick_gpu)}"
if [[ -n "${GPU}" ]]; then
  export CUDA_VISIBLE_DEVICES="$GPU"
fi

echo "[V30 smoke] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-cpu}"
"$PY" code/stwm/tools/audit_ostf_v30_training_readiness_20260508.py
if [[ ! -s reports/stwm_ostf_v30_external_gt_prior_suite_20260508.json ]]; then
  "$PY" code/stwm/tools/eval_ostf_v30_external_gt_prior_suite_20260508.py
else
  echo "[V30 smoke] reuse reports/stwm_ostf_v30_external_gt_prior_suite_20260508.json"
fi

STEPS="${V30_SMOKE_STEPS:-1000}"
"$PY" code/stwm/tools/train_ostf_external_gt_v30_20260508.py \
  --experiment-name v30_extgt_m128_h32_seed42_smoke --horizon 32 --m-points 128 --seed 42 \
  --steps "$STEPS" --batch-size "${V30_SMOKE_BATCH:-8}" --eval-interval 500 --amp --smoke
"$PY" code/stwm/tools/train_ostf_external_gt_v30_20260508.py \
  --experiment-name v30_extgt_m128_h64_seed42_smoke --horizon 64 --m-points 128 --seed 42 \
  --steps "$STEPS" --batch-size "${V30_SMOKE_BATCH:-8}" --eval-interval 500 --amp --smoke
"$PY" code/stwm/tools/write_ostf_v30_smoke_summary_20260508.py
