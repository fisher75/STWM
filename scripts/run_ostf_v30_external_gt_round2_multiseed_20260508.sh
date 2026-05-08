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
if [[ -n "$GPU" ]]; then
  export CUDA_VISIBLE_DEVICES="$GPU"
fi

echo "[V30 round2] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-cpu}"
"$PY" code/stwm/tools/audit_ostf_v30_round1_claim_boundary_20260508.py
"$PY" code/stwm/tools/audit_ostf_v30_round2_prelaunch_code_20260508.py

run_if_missing() {
  local name="$1"
  shift
  if "$PY" - "$name" <<'PY'
import json, sys
from pathlib import Path
root=Path('/raid/chen034/workspace/stwm')
path=root/'reports/stwm_ostf_v30_external_gt_runs'/f'{sys.argv[1]}.json'
ok=False
if path.exists() and path.stat().st_size>0:
    try:
        p=json.loads(path.read_text())
        ok=bool(p.get('completed') and p.get('test_item_rows') and p.get('checkpoint_path'))
    except Exception:
        ok=False
raise SystemExit(0 if ok else 1)
PY
  then
    echo "[V30 round2] skip completed $name"
  else
    "$PY" code/stwm/tools/train_ostf_external_gt_v30_20260508.py --experiment-name "$name" "$@"
  fi
}

SEEDS=(42 123 456 789 2026)
STEPS="${V30_ROUND2_STEPS:-4000}"
BATCH="${V30_ROUND2_BATCH:-8}"
for seed in "${SEEDS[@]}"; do
  run_if_missing "v30_extgt_m128_h32_seed${seed}" \
    --horizon 32 --m-points 128 --seed "$seed" \
    --steps "$STEPS" --batch-size "$BATCH" --eval-interval 1000 --amp
  run_if_missing "v30_extgt_m128_h64_seed${seed}" \
    --horizon 64 --m-points 128 --seed "$seed" \
    --steps "$STEPS" --batch-size "$BATCH" --eval-interval 1000 --amp
done

"$PY" code/stwm/tools/aggregate_ostf_external_gt_v30_round2_multiseed_20260508.py
