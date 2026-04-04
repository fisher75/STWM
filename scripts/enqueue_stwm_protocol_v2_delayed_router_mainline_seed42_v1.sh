#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home/chen034/workspace/stwm"
QUEUE_ROOT="${STWM_PROTOCOL_V2_QUEUE_ROOT:-$REPO_ROOT/outputs/queue/stwm_protocol_v2_frontend_default_v1}"
QUEUE_DIR="$QUEUE_ROOT/d1_train"

TRAIN_SCRIPT="$REPO_ROOT/code/stwm/trainers/train_stwm_v4_2_real.py"
TRAIN_MANIFEST="${STWM_D1_TRAIN_MANIFEST:-$REPO_ROOT/manifests/protocol_v2/train_v2.json}"
PROTOCOL_MAIN_MANIFEST="${STWM_D1_PROTOCOL_MAIN_MANIFEST:-$REPO_ROOT/manifests/protocol_v2/protocol_val_main_v1.json}"
PROTOCOL_EVENTFUL_MANIFEST="${STWM_D1_PROTOCOL_EVENTFUL_MANIFEST:-$REPO_ROOT/manifests/protocol_v2/protocol_val_eventful_v1.json}"
MODEL_PRESET="${STWM_D1_MODEL_PRESET:-prototype_220m_v4_2}"
PRESET_FILE="${STWM_D1_PRESET_FILE:-$REPO_ROOT/code/stwm/configs/model_presets_v4_2.json}"
DATA_ROOT="${STWM_D1_DATA_ROOT:-$REPO_ROOT/data/external}"

DATA_MODE="${STWM_D1_DATA_MODE:-frontend_cache}"
FRONTEND_CACHE_DIR="${STWM_D1_FRONTEND_CACHE_DIR:-$REPO_ROOT/data/cache/frontend_cache_protocol_v2_full_v1}"
FRONTEND_CACHE_INDEX="${STWM_D1_FRONTEND_CACHE_INDEX:-$FRONTEND_CACHE_DIR/index.json}"
FRONTEND_CACHE_MAX_SHARDS_IN_MEMORY="${STWM_D1_FRONTEND_CACHE_MAX_SHARDS_IN_MEMORY:-8}"

SEED="${STWM_D1_SEED:-42}"
STEPS="${STWM_D1_STEPS:-2000}"
SAMPLE_LIMIT="${STWM_D1_SAMPLE_LIMIT:-0}"

LSEM_10="${STWM_D1_LSEM_10:-0.5}"
PROTOCOL_EVAL_INTERVAL="${STWM_D1_PROTOCOL_EVAL_INTERVAL:-500}"
CHECKPOINT_INTERVAL="${STWM_D1_CHECKPOINT_INTERVAL:-500}"
OBJECT_BIAS_GATE_THRESHOLD="${STWM_D1_OBJECT_BIAS_GATE_THRESHOLD:-0.5}"
PREFERRED_GPU_ALL="${STWM_D1_PREFERRED_GPU_ALL:-}"

OUT_ROOT="${STWM_DELAYED_ROUTER_OUT_ROOT:-$REPO_ROOT/outputs/training/stwm_v4_2_220m_protocol_frozen_frontend_delayed_router_mainline_seed42_v1}"
REPORT_TSV="${STWM_DELAYED_ROUTER_SUBMIT_TSV:-$REPO_ROOT/reports/stwm_delayed_router_mainline_submit_v1.tsv}"
REPORT_MD="${STWM_DELAYED_ROUTER_SUBMIT_MD:-$REPO_ROOT/docs/STWM_DELAYED_ROUTER_MAINLINE_SUBMIT_V1.md}"

mkdir -p "$OUT_ROOT"
mkdir -p "$(dirname "$REPORT_TSV")"
mkdir -p "$(dirname "$REPORT_MD")"

echo -e "run_name\tunique_change_point\tjob_id\tstatus_file\tmain_log\toutput_dir" > "$REPORT_TSV"

submit_one() {
  local run_name="$1"
  local object_bias_alpha="$2"
  local object_bias_delay_steps="$3"
  local object_bias_gated="$4"
  local unique_change_point="$5"
  local preferred_gpu="${6:-}"

  local out_dir="$OUT_ROOT/seed_${SEED}/${run_name}"
  mkdir -p "$out_dir"

  local notes="Delayed-Router mainline seed42 diagnostic v1 | data_mode=${DATA_MODE}"
  local resume_hint="Resume with same output_dir and --auto-resume; compare official-rule metrics against seed123 baseline lineage"

  local cmd=(
    env "PYTHONPATH=$REPO_ROOT/code:${PYTHONPATH:-}"
    conda run --no-capture-output -n stwm
    python "$TRAIN_SCRIPT"
    --data-root "$DATA_ROOT"
    --manifest "$TRAIN_MANIFEST"
    --output-dir "$out_dir"
    --run-name "$run_name"
    --seed "$SEED"
    --steps "$STEPS"
    --target-epochs 0
    --min-optimizer-steps 0
    --max-optimizer-steps 0
    --sample-limit "$SAMPLE_LIMIT"
    --model-preset "$MODEL_PRESET"
    --preset-file "$PRESET_FILE"
    --use-teacher-priors
    --save-checkpoint
    --checkpoint-dir-name checkpoints
    --checkpoint-interval "$CHECKPOINT_INTERVAL"
    --milestone-interval 0
    --auto-resume
    --micro-batch-per-gpu 2
    --grad-accum 8
    --num-workers 12
    --prefetch-factor 2
    --persistent-workers
    --pin-memory
    --bf16
    --activation-checkpointing
    --lambda-traj 1.0
    --lambda-vis 0.25
    --lambda-sem "$LSEM_10"
    --lambda-reid 0.25
    --lambda-query 0.25
    --lambda-reconnect 0.1
    --gradient-audit-interval 0
    --protocol-eval-interval "$PROTOCOL_EVAL_INTERVAL"
    --protocol-eval-manifest "$PROTOCOL_MAIN_MANIFEST"
    --protocol-eval-dataset all
    --protocol-eval-max-clips 0
    --protocol-eval-seed "$SEED"
    --protocol-eval-obs-steps 8
    --protocol-eval-pred-steps 8
    --protocol-eval-run-name protocol_val_main
    --protocol-diagnostics-manifest "$PROTOCOL_EVENTFUL_MANIFEST"
    --protocol-diagnostics-dataset all
    --protocol-diagnostics-max-clips 0
    --protocol-diagnostics-run-name protocol_val_eventful
    --protocol-version v2_4_detached_frozen
    --protocol-best-checkpoint-name best_protocol_main.pt
    --protocol-best-selection-name best_protocol_main_selection.json
  )

  if [[ "$DATA_MODE" == "frontend_cache" ]]; then
    cmd+=(
      --data-mode frontend_cache
      --frontend-cache-dir "$FRONTEND_CACHE_DIR"
      --frontend-cache-index "$FRONTEND_CACHE_INDEX"
      --frontend-cache-max-shards-in-memory "$FRONTEND_CACHE_MAX_SHARDS_IN_MEMORY"
    )
  elif [[ "$DATA_MODE" == "raw" ]]; then
    cmd+=(--data-mode raw)
  else
    echo "unsupported STWM_D1_DATA_MODE=$DATA_MODE (expected: frontend_cache|raw)" >&2
    exit 2
  fi

  if [[ -n "$object_bias_alpha" ]]; then
    cmd+=(--object-bias-alpha "$object_bias_alpha")
  fi
  if [[ -n "$object_bias_delay_steps" ]]; then
    cmd+=(--object-bias-delay-steps "$object_bias_delay_steps")
  fi
  if [[ "$object_bias_gated" == "1" ]]; then
    cmd+=(--object-bias-gated --object-bias-gate-threshold "$OBJECT_BIAS_GATE_THRESHOLD")
  fi

  local submit_args=(
    --queue-dir "$QUEUE_DIR"
    --job-name "$run_name"
    --class-type B
    --workdir "$REPO_ROOT"
    --notes "$notes"
    --resume-hint "$resume_hint"
  )
  if [[ -n "$preferred_gpu" ]]; then
    submit_args+=(--preferred-gpu "$preferred_gpu")
  fi

  local submit_output
  submit_output="$(bash "$REPO_ROOT/scripts/protocol_v2_queue_submit.sh" "${submit_args[@]}" -- "${cmd[@]}")"
  echo "$submit_output"

  local job_id status_file main_log
  job_id="$(echo "$submit_output" | sed -n 's/^  job_id:[[:space:]]*//p' | tail -n 1)"
  status_file="$(echo "$submit_output" | sed -n 's/^  status_file:[[:space:]]*//p' | tail -n 1)"
  main_log="$(echo "$submit_output" | sed -n 's/^  main_log:[[:space:]]*//p' | tail -n 1)"

  echo -e "${run_name}\t${unique_change_point}\t${job_id}\t${status_file}\t${main_log}\t${out_dir}" >> "$REPORT_TSV"
}

submit_one "delayed_only_seed42_challenge_v1" "1.0" "200" "0" "Enable delay only: --object-bias-delay-steps 200" "$PREFERRED_GPU_ALL"
submit_one "two_path_residual_seed42_challenge_v1" "0.50" "0" "1" "Residual two-path proxy: --object-bias-alpha 0.50 + gated(th=0.5)" "$PREFERRED_GPU_ALL"
submit_one "delayed_residual_router_seed42_challenge_v1" "0.50" "200" "1" "Combined: delay 200 + residual alpha 0.50 + gated(th=0.5)" "$PREFERRED_GPU_ALL"

python - "$REPORT_TSV" "$REPORT_MD" <<'PY'
from pathlib import Path
import csv
import time
import sys

tsv = Path(sys.argv[1])
md = Path(sys.argv[2])
rows = []
with tsv.open('r', newline='') as f:
    rows = list(csv.DictReader(f, delimiter='\t'))

lines = []
lines.append('# STWM Delayed Router Mainline Submit V1')
lines.append('')
lines.append(f'Generated: {time.strftime("%Y-%m-%d %H:%M:%S")}')
lines.append('')
lines.append('| run_name | unique_change_point | job_id | status_file | main_log | output_dir |')
lines.append('|---|---|---|---|---|---|')
for r in rows:
    lines.append(
        '| '
        f"{r.get('run_name','')} | {r.get('unique_change_point','')} | {r.get('job_id','')} | "
        f"{r.get('status_file','')} | {r.get('main_log','')} | {r.get('output_dir','')} |"
    )
lines.append('')

md.write_text('\n'.join(lines) + '\n')
PY

echo "[delayed-router-seed42-enqueue] queue_dir=$QUEUE_DIR"
echo "[delayed-router-seed42-enqueue] submit_tsv=$REPORT_TSV"
echo "[delayed-router-seed42-enqueue] submit_doc=$REPORT_MD"
