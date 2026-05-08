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

echo "[V30 round1] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-cpu}"
"$PY" code/stwm/tools/audit_ostf_v30_training_readiness_20260508.py
if [[ ! -s reports/stwm_ostf_v30_external_gt_prior_suite_20260508.json ]]; then
  "$PY" code/stwm/tools/eval_ostf_v30_external_gt_prior_suite_20260508.py
else
  echo "[V30 round1] reuse reports/stwm_ostf_v30_external_gt_prior_suite_20260508.json"
fi

run_if_missing() {
  local name="$1"
  shift
  if [[ -s "reports/stwm_ostf_v30_external_gt_runs/${name}.json" ]]; then
    echo "[V30 round1] skip completed ${name}"
  else
    "$PY" code/stwm/tools/train_ostf_external_gt_v30_20260508.py --experiment-name "$name" "$@"
  fi
}

STEPS="${V30_ROUND1_STEPS:-4000}"
BATCH="${V30_ROUND1_BATCH:-8}"
run_if_missing v30_extgt_m128_h32_seed42 \
  --horizon 32 --m-points 128 --seed 42 \
  --steps "$STEPS" --batch-size "$BATCH" --eval-interval 1000 --amp
run_if_missing v30_extgt_m128_h64_seed42 \
  --horizon 64 --m-points 128 --seed 42 \
  --steps "$STEPS" --batch-size "$BATCH" --eval-interval 1000 --amp
run_if_missing v30_extgt_m128_h32_wo_semantic_seed42 \
  --horizon 32 --m-points 128 --seed 42 \
  --steps "$STEPS" --batch-size "$BATCH" --eval-interval 1000 --amp --wo-semantic
run_if_missing v30_extgt_m128_h64_wo_semantic_seed42 \
  --horizon 64 --m-points 128 --seed 42 \
  --steps "$STEPS" --batch-size "$BATCH" --eval-interval 1000 --amp --wo-semantic

if [[ "${V30_RUN_OPTIONAL_M512:-1}" == "1" ]]; then
  M512_STEPS="${V30_ROUND1_M512_STEPS:-2500}"
  run_if_missing v30_extgt_m512_h32_seed42 \
    --horizon 32 --m-points 512 --seed 42 \
    --steps "$M512_STEPS" --batch-size "${V30_ROUND1_M512_BATCH:-2}" --eval-interval 500 --amp --hidden-dim 240
  run_if_missing v30_extgt_m512_h64_seed42 \
    --horizon 64 --m-points 512 --seed 42 \
    --steps "$M512_STEPS" --batch-size "${V30_ROUND1_M512_BATCH:-2}" --eval-interval 500 --amp --hidden-dim 240
fi

"$PY" code/stwm/tools/aggregate_ostf_external_gt_v30_round1_20260508.py --prefix v30_extgt_ --suffix round1
