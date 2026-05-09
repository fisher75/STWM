#!/usr/bin/env bash
set -euo pipefail

ROOT="/raid/chen034/workspace/stwm"
cd "$ROOT"
PY="${STWM_PYTHON:-/home/chen034/miniconda3/envs/stwm/bin/python}"

LOG_DIR="logs/stwm_ostf_v32_recurrent_field_pilot_20260509"
RUN_DIR="reports/stwm_ostf_v32_recurrent_field_runs"
MANIFEST="reports/stwm_ostf_v32_recurrent_field_pilot_launch_manifest_20260509.json"
STATUS="reports/stwm_ostf_v32_recurrent_field_pilot_status_20260509.json"
mkdir -p "$LOG_DIR" "$RUN_DIR" reports docs

mapfile -t GPUS < <(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits 2>/dev/null | awk -F, '$2+0 >= 25000 {print $1 "," $2+0}' | sort -t, -k2,2nr | awk -F, '{print $1}')
if [[ ${#GPUS[@]} -eq 0 ]]; then
  echo "No GPU with >=25GB free memory found." >&2
  exit 2
fi

declare -a SPECS=(
  "v32_rf_m128_h32_seed42 128 32 full 64"
  "v32_rf_m128_h64_seed42 128 64 full 48"
  "v32_rf_m128_h96_seed42 128 96 full 32"
  "v32_rf_m512_h32_seed42 512 32 full 10"
  "v32_rf_m512_h64_seed42 512 64 full 8"
  "v32_rf_m512_h96_seed42 512 96 full 6"
)

export GPUS_CSV="$(IFS=,; echo "${GPUS[*]}")"
python - <<'PY'
import json, os, subprocess, time
from pathlib import Path
root = Path("/raid/chen034/workspace/stwm")
specs = [
  ("v32_rf_m128_h32_seed42",128,32,"full",64),
  ("v32_rf_m128_h64_seed42",128,64,"full",48),
  ("v32_rf_m128_h96_seed42",128,96,"full",32),
  ("v32_rf_m512_h32_seed42",512,32,"full",10),
  ("v32_rf_m512_h64_seed42",512,64,"full",8),
  ("v32_rf_m512_h96_seed42",512,96,"full",6),
]
gpus = [g for g in os.environ.get("GPUS_CSV", "").split(",") if g]
rows = []
for i, (name, m, h, mode, batch) in enumerate(specs):
    gpu = gpus[i % len(gpus)]
    log = f"logs/stwm_ostf_v32_recurrent_field_pilot_20260509/{name}.log"
    cmd = [
        os.environ.get("STWM_PYTHON", "/home/chen034/miniconda3/envs/stwm/bin/python"),"code/stwm/tools/train_ostf_recurrent_field_v32_20260509.py",
        "--experiment-name",name,"--m-points",str(m),"--horizon",str(h),"--seed","42",
        "--steps","4000","--eval-interval","1000","--batch-size",str(batch),"--hidden-dim","192",
        "--field-layers","2","--field-attention-mode",mode,"--heads","6","--learned-modes","4",
        "--num-workers","2","--amp"
    ]
    rows.append({"experiment_name": name, "M": m, "H": h, "gpu": gpu, "log_path": log, "report_path": f"reports/stwm_ostf_v32_recurrent_field_runs/{name}.json", "command": " ".join(cmd)})
payload = {"generated_at_epoch": time.time(), "session": "stwm_ostf_v32_recurrent_field_pilot_20260509", "gpu_policy": "prefer highest free memory, require >=25GB", "jobs": rows}
(root/"reports/stwm_ostf_v32_recurrent_field_pilot_launch_manifest_20260509.json").write_text(json.dumps(payload, indent=2, sort_keys=True)+"\n")
PY

pids=()
for i in "${!SPECS[@]}"; do
  read -r NAME M H MODE BATCH <<<"${SPECS[$i]}"
  GPU="${GPUS[$((i % ${#GPUS[@]}))]}"
  REPORT="$RUN_DIR/${NAME}.json"
  LOG="$LOG_DIR/${NAME}.log"
  if python - "$REPORT" <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
raise SystemExit(0 if p.exists() and json.loads(p.read_text()).get("completed") else 1)
PY
  then
    echo "skip existing complete $NAME" | tee "$LOG"
    continue
  fi
  (
    set +e
    CUDA_VISIBLE_DEVICES="$GPU" "$PY" code/stwm/tools/train_ostf_recurrent_field_v32_20260509.py \
      --experiment-name "$NAME" --m-points "$M" --horizon "$H" --seed 42 \
      --steps 4000 --eval-interval 1000 --batch-size "$BATCH" --hidden-dim 192 \
      --field-layers 2 --field-attention-mode "$MODE" --heads 6 --learned-modes 4 \
      --num-workers 2 --amp >"$LOG" 2>&1
    rc=$?
    if [[ $rc -ne 0 && "$M" -ge 512 ]]; then
      echo "primary pilot failed rc=$rc; retrying $NAME with induced attention and smaller batch" >>"$LOG"
      CUDA_VISIBLE_DEVICES="$GPU" "$PY" code/stwm/tools/train_ostf_recurrent_field_v32_20260509.py \
        --experiment-name "$NAME" --m-points "$M" --horizon "$H" --seed 42 \
        --steps 4000 --eval-interval 1000 --batch-size 4 --hidden-dim 192 \
        --field-layers 2 --field-attention-mode induced --induced-tokens 32 --heads 6 --learned-modes 4 \
        --num-workers 2 --amp >>"$LOG" 2>&1
      rc=$?
    fi
    exit $rc
  ) &
  pids+=("$!")
done

rc=0
for pid in "${pids[@]:-}"; do
  if ! wait "$pid"; then
    rc=1
  fi
done

"$PY" code/stwm/tools/aggregate_ostf_recurrent_field_v32_pilot_20260509.py || rc=1
python - <<'PY'
import json, time
from pathlib import Path
root = Path("/raid/chen034/workspace/stwm")
summary_path = root/"reports/stwm_ostf_v32_recurrent_field_pilot_summary_20260509.json"
decision_path = root/"reports/stwm_ostf_v32_recurrent_field_pilot_decision_20260509.json"
summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
decision = json.loads(decision_path.read_text()) if decision_path.exists() else {}
payload = {"generated_at_epoch": time.time(), "terminal": True, "summary_path": str(summary_path.relative_to(root)), "decision_path": str(decision_path.relative_to(root)), "completed_run_count": summary.get("completed_run_count"), "missing_runs": summary.get("missing_runs"), "recommended_next_step": decision.get("recommended_next_step")}
(root/"reports/stwm_ostf_v32_recurrent_field_pilot_status_20260509.json").write_text(json.dumps(payload, indent=2, sort_keys=True)+"\n")
PY
exit "$rc"
