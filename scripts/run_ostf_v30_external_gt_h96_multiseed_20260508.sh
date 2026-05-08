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

echo "[V30 H96 multiseed] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-cpu}"
"$PY" - <<'PY'
import json
from pathlib import Path
root=Path('/raid/chen034/workspace/stwm')
decision=json.loads((root/'reports/stwm_ostf_v30_external_gt_h96_pilot_decision_20260508.json').read_text())
if decision.get('next_step_choice') != 'run_v30_h96_5seed':
    raise SystemExit('H96 pilot did not authorize 5seed multiseed')
PY

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
    echo "[V30 H96 multiseed] skip completed $name"
  else
    "$PY" code/stwm/tools/train_ostf_external_gt_v30_20260508.py --experiment-name "$name" "$@"
  fi
}

STEPS="${V30_H96_STEPS:-4000}"
BATCH="${V30_H96_BATCH:-4}"
for seed in 456 789 2026; do
  run_if_missing "v30_extgt_m128_h96_seed${seed}" --horizon 96 --m-points 128 --seed "$seed" --steps "$STEPS" --batch-size "$BATCH" --eval-interval 1000 --amp
done

"$PY" code/stwm/tools/aggregate_ostf_external_gt_v30_h96_multiseed_20260508.py
cat > "$ROOT/reports/stwm_ostf_v30_external_gt_h96_multiseed_status_20260508.json" <<JSON
{
  "session": "stwm_ostf_v30_extgt_h96_multiseed_20260508",
  "log": "logs/stwm_ostf_v30_extgt_h96_multiseed_20260508.log",
  "status": "completed",
  "updated_at_local": "$(date -Iseconds)",
  "decision_path": "reports/stwm_ostf_v30_external_gt_h96_multiseed_decision_20260508.json"
}
JSON
