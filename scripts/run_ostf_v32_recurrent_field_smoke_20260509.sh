#!/usr/bin/env bash
set -euo pipefail

ROOT="/raid/chen034/workspace/stwm"
cd "$ROOT"
PY="${STWM_PYTHON:-/home/chen034/miniconda3/envs/stwm/bin/python}"

LOG_DIR="logs/stwm_ostf_v32_recurrent_field_smoke_20260509"
RUN_DIR="reports/stwm_ostf_v32_recurrent_field_runs"
MANIFEST="reports/stwm_ostf_v32_recurrent_field_smoke_launch_manifest_20260509.json"
STATUS="reports/stwm_ostf_v32_recurrent_field_smoke_status_20260509.json"
mkdir -p "$LOG_DIR" "$RUN_DIR" reports docs

mapfile -t GPUS < <(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits 2>/dev/null | awk -F, '$2+0 >= 25000 {print $1 "," $2+0}' | sort -t, -k2,2nr | awk -F, '{print $1}')
if [[ ${#GPUS[@]} -eq 0 ]]; then
  echo "No GPU with >=25GB free memory found." >&2
  exit 2
fi

declare -a SPECS=(
  "v32_rf_m128_h32_seed42_smoke 128 32 full 64"
  "v32_rf_m128_h64_seed42_smoke 128 64 full 48"
  "v32_rf_m128_h96_seed42_smoke 128 96 full 32"
  "v32_rf_m512_h32_seed42_smoke 512 32 full 10"
  "v32_rf_m512_h64_seed42_smoke 512 64 full 8"
)

python - <<'PY'
import json, os, subprocess, time
from pathlib import Path
root = Path("/raid/chen034/workspace/stwm")
specs = [
  ("v32_rf_m128_h32_seed42_smoke",128,32,"full",64),
  ("v32_rf_m128_h64_seed42_smoke",128,64,"full",48),
  ("v32_rf_m128_h96_seed42_smoke",128,96,"full",32),
  ("v32_rf_m512_h32_seed42_smoke",512,32,"full",10),
  ("v32_rf_m512_h64_seed42_smoke",512,64,"full",8),
]
gpus = os.environ.get("GPUS_CSV", "").split(",") if os.environ.get("GPUS_CSV") else []
if not gpus:
    gpus = [line.split(",")[0].strip() for line in subprocess.run(
        ["nvidia-smi","--query-gpu=index,memory.free","--format=csv,noheader,nounits"],
        text=True, stdout=subprocess.PIPE, check=False
    ).stdout.splitlines() if line.strip() and float(line.split(",")[1]) >= 25000]
rows = []
for i, (name, m, h, mode, batch) in enumerate(specs):
    gpu = gpus[i % len(gpus)]
    log = f"logs/stwm_ostf_v32_recurrent_field_smoke_20260509/{name}.log"
    cmd = [
        os.environ.get("STWM_PYTHON", "/home/chen034/miniconda3/envs/stwm/bin/python"),"code/stwm/tools/train_ostf_recurrent_field_v32_20260509.py",
        "--experiment-name",name,"--m-points",str(m),"--horizon",str(h),"--seed","42",
        "--steps","800","--eval-interval","400","--batch-size",str(batch),"--hidden-dim","192",
        "--field-layers","2","--field-attention-mode",mode,"--heads","6","--learned-modes","4",
        "--max-train-items","256","--max-eval-items","128","--num-workers","2","--amp","--smoke"
    ]
    rows.append({"experiment_name": name, "M": m, "H": h, "gpu": gpu, "log_path": log, "report_path": f"reports/stwm_ostf_v32_recurrent_field_runs/{name}.json", "command": " ".join(cmd)})
payload = {"generated_at_epoch": time.time(), "session": "direct_or_tmux", "gpu_policy": "prefer highest free memory, require >=25GB", "jobs": rows}
(root/"reports/stwm_ostf_v32_recurrent_field_smoke_launch_manifest_20260509.json").write_text(json.dumps(payload, indent=2, sort_keys=True)+"\n")
PY

export GPUS_CSV="$(IFS=,; echo "${GPUS[*]}")"
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
      --steps 800 --eval-interval 400 --batch-size "$BATCH" --hidden-dim 192 \
      --field-layers 2 --field-attention-mode "$MODE" --heads 6 --learned-modes 4 \
      --max-train-items 256 --max-eval-items 128 --num-workers 2 --amp --smoke >"$LOG" 2>&1
    rc=$?
    if [[ $rc -ne 0 && "$M" -ge 512 ]]; then
      echo "primary smoke failed rc=$rc; retrying $NAME with induced attention and smaller batch" >>"$LOG"
      CUDA_VISIBLE_DEVICES="$GPU" "$PY" code/stwm/tools/train_ostf_recurrent_field_v32_20260509.py \
        --experiment-name "$NAME" --m-points "$M" --horizon "$H" --seed 42 \
        --steps 800 --eval-interval 400 --batch-size 4 --hidden-dim 192 \
        --field-layers 2 --field-attention-mode induced --induced-tokens 32 --heads 6 --learned-modes 4 \
        --max-train-items 256 --max-eval-items 128 --num-workers 2 --amp --smoke >>"$LOG" 2>&1
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

"$PY" code/stwm/tools/write_ostf_v32_smoke_summary_20260509.py || rc=1
python - <<'PY'
import json, time
from pathlib import Path
root = Path("/raid/chen034/workspace/stwm")
summary_path = root/"reports/stwm_ostf_v32_recurrent_field_smoke_summary_20260509.json"
summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
payload = {"generated_at_epoch": time.time(), "terminal": True, "return_code_proxy": 0 if summary.get("smoke_passed") else 1, "summary_path": str(summary_path.relative_to(root)), "smoke_passed": summary.get("smoke_passed"), "completed_run_count": summary.get("completed_run_count"), "missing_runs": summary.get("missing_runs"), "failed_runs": summary.get("failed_runs")}
(root/"reports/stwm_ostf_v32_recurrent_field_smoke_status_20260509.json").write_text(json.dumps(payload, indent=2, sort_keys=True)+"\n")
PY
exit "$rc"
