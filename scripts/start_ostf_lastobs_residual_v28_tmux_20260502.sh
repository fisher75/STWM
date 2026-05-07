#!/usr/bin/env bash
set -euo pipefail

ROOT="${STWM_ROOT:-/raid/chen034/workspace/stwm}"
RUN_SCRIPT="$ROOT/scripts/run_ostf_lastobs_residual_v28_20260502.sh"
LOG_DIR="$ROOT/logs/stwm_ostf_v28"
MANIFEST="$ROOT/reports/stwm_ostf_v28_launch_manifest_20260502.json"
mkdir -p "$LOG_DIR"
MODE="${1:-full}"
LAUNCH_ROWS=()

pick_gpus() {
  python - <<'PY'
import subprocess
raw = subprocess.check_output(
    [
        "nvidia-smi",
        "--query-gpu=index,memory.used,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits",
    ],
    text=True,
)
rows = []
for line in raw.strip().splitlines():
    idx, used, free, util = [x.strip() for x in line.split(",")]
    idx_i = int(idx)
    used_i = int(used)
    free_i = int(free)
    util_i = int(util)
    if free_i < 25 * 1024:
        continue
    pref = 0 if free_i >= 40 * 1024 else 1
    rows.append((pref, -free_i, util_i, idx_i, free_i, used_i))
rows.sort()
print(" ".join(str(r[3]) for r in rows))
PY
}

mapfile -t GPUS < <(pick_gpus | tr ' ' '\n')
if [[ "${#GPUS[@]}" -eq 0 ]]; then
  echo "no_eligible_gpus_found" >&2
  exit 1
fi

gpu_at() {
  local idx="$1"
  echo "${GPUS[$(( idx % ${#GPUS[@]} ))]}"
}

launch() {
  local session="$1"
  local gpu="$2"
  local log_name="$3"
  shift 3
  local cmd="$*"
  local report_path=""
  local horizon=""
  if [[ "$cmd" == *"--experiment-name "* ]]; then
    local exp_name
    exp_name="$(awk '{
      for (i = 1; i <= NF; i++) {
        if ($i == "--experiment-name") {
          print $(i + 1)
          exit
        }
      }
    }' <<< "$cmd")"
    report_path="$ROOT/reports/stwm_ostf_v28_runs/${exp_name}.json"
  fi
  if [[ "$cmd" == *"--horizon "* ]]; then
    horizon="$(awk '{
      for (i = 1; i <= NF; i++) {
        if ($i == "--horizon") {
          print $(i + 1)
          exit
        }
      }
    }' <<< "$cmd")"
  fi
  tmux kill-session -t "$session" >/dev/null 2>&1 || true
  tmux new-session -d -s "$session" \
    "cd '$ROOT' && export CUDA_VISIBLE_DEVICES='$gpu' PYTHONPATH='$ROOT/code:\${PYTHONPATH:-}' STWM_PROC_TITLE=python STWM_PROC_TITLE_MODE=generic && bash '$RUN_SCRIPT' $cmd |& tee '$LOG_DIR/$log_name'"
  LAUNCH_ROWS+=("{\"session_name\":\"$session\",\"horizon\":\"$horizon\",\"gpu_id\":\"$gpu\",\"log_path\":\"$LOG_DIR/$log_name\",\"report_path\":\"$report_path\",\"command\":$(python - <<'PY' "$cmd"
import json, sys
print(json.dumps(sys.argv[1]))
PY
)}")
  echo "[launch] session=$session gpu=$gpu log=$LOG_DIR/$log_name cmd=$cmd"
}

if [[ "$MODE" == "full" || "$MODE" == "stage1" ]]; then
  launch "stwm_v28_verify" "$(gpu_at 0)" "verify_cache.log" "verify-cache"
  launch "stwm_v28_h32_m128" "$(gpu_at 1)" "v28_h32_m128.log" \
    "train --experiment-name v28_lastobs_m128_h32_seed42 --model-kind v28_lastobs_m128_h32 --horizon 32 --seed 42 --steps 30000 --batch-size 2 --device cuda --eval-every 1500"
  launch "stwm_v28_h32_m128_nodense" "$(gpu_at 2)" "v28_h32_m128_wo_dense.log" \
    "train --experiment-name v28_lastobs_m128_h32_wo_dense_points_seed42 --model-kind v28_lastobs_m128_h32_wo_dense_points --horizon 32 --seed 42 --steps 20000 --batch-size 2 --device cuda --eval-every 1500"
  launch "stwm_v28_h32_m128_nosem" "$(gpu_at 3)" "v28_h32_m128_wo_semantic.log" \
    "train --experiment-name v28_lastobs_m128_h32_wo_semantic_memory_seed42 --model-kind v28_lastobs_m128_h32_wo_semantic_memory --horizon 32 --seed 42 --steps 20000 --batch-size 2 --device cuda --eval-every 1500"
  launch "stwm_v28_h32_m128_nores" "$(gpu_at 4)" "v28_h32_m128_wo_residual.log" \
    "train --experiment-name v28_lastobs_m128_h32_wo_residual_modes_seed42 --model-kind v28_lastobs_m128_h32_wo_residual_modes --horizon 32 --seed 42 --steps 20000 --batch-size 2 --device cuda --eval-every 1500"
  launch "stwm_v28_h32_m128_prioronly" "$(gpu_at 5)" "v28_h32_m128_prior_only.log" \
    "train --experiment-name v28_lastobs_m128_h32_prior_only_seed42 --model-kind v28_lastobs_m128_h32_prior_only --horizon 32 --seed 42 --steps 10000 --batch-size 2 --device cuda --eval-every 1000"
fi

if [[ "$MODE" == "full" || "$MODE" == "stage2" ]]; then
  launch "stwm_v28_h32_m512" "$(gpu_at 6)" "v28_h32_m512.log" \
    "train --experiment-name v28_lastobs_m512_h32_seed42 --model-kind v28_lastobs_m512_h32 --horizon 32 --seed 42 --steps 30000 --batch-size 1 --device cuda --eval-every 1000"
  launch "stwm_v28_h64_m128" "$(gpu_at 7)" "v28_h64_m128.log" \
    "train --experiment-name v28_lastobs_m128_h64_seed42 --model-kind v28_lastobs_m128_h64 --horizon 64 --seed 42 --steps 30000 --batch-size 1 --device cuda --eval-every 1000"
fi

if [[ "$MODE" != "full" && "$MODE" != "stage1" && "$MODE" != "stage2" ]]; then
  echo "unknown_mode=$MODE expected full|stage1|stage2" >&2
  exit 2
fi

python - "$MANIFEST" "${LAUNCH_ROWS[@]}" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

path = Path(sys.argv[1])
rows = [json.loads(x) for x in sys.argv[2:]]
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps({"generated_at_utc": datetime.now(timezone.utc).isoformat(), "launches": rows}, indent=2) + "\n")
PY
