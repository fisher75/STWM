#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home/chen034/workspace/stwm"
QUEUE_ROOT="${STWM_PROTOCOL_V2_QUEUE_ROOT:-$REPO_ROOT/outputs/queue/stwm_protocol_v2_frontend_default_v1}"
QUEUE_DIR="$QUEUE_ROOT/d1_train"

TRAIN_SCRIPT="$REPO_ROOT/code/stwm/trainers/train_stwm_v4_2_real.py"
TRAIN_MANIFEST="${STWM_D1_TRAIN_MANIFEST:-$REPO_ROOT/manifests/protocol_v2/train_v2.json}"
PROTOCOL_MAIN_MANIFEST="${STWM_D1_PROTOCOL_MAIN_MANIFEST:-$REPO_ROOT/manifests/protocol_v2/protocol_val_main_v1.json}"
PROTOCOL_EVENTFUL_MANIFEST="${STWM_D1_PROTOCOL_EVENTFUL_MANIFEST:-$REPO_ROOT/manifests/protocol_v2/protocol_val_eventful_v1.json}"
SEMANTIC_HARD_MANIFEST="${STWM_SEMTEACHER_HARD_MANIFEST:-$REPO_ROOT/manifests/protocol_v2/semantic_hard_seed42_v1.json}"
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

SEMTEACHER_DISTILL_WEIGHT="${STWM_SEMTEACHER_DISTILL_WEIGHT:-0.20}"
SEMTEACHER_ASSOC_TEMP="${STWM_SEMTEACHER_ASSOC_TEMP:-0.25}"
SEMTEACHER_CONF_THRESH="${STWM_SEMTEACHER_CONF_THRESH:-0.55}"
SEMTEACHER_RERANK_TOPK="${STWM_SEMTEACHER_RERANK_TOPK:-3}"

OUT_ROOT="${STWM_SEMTEACHER_OUT_ROOT:-$REPO_ROOT/outputs/training/stwm_v4_2_220m_protocol_frozen_frontend_semteacher_mainline_seed42_v1}"
REPORT_TSV="${STWM_SEMTEACHER_SUBMIT_TSV:-$REPO_ROOT/reports/stwm_semteacher_mainline_seed42_submit_v1.tsv}"
REPORT_MD="${STWM_SEMTEACHER_SUBMIT_MD:-$REPO_ROOT/docs/STWM_SEMTEACHER_MAINLINE_SEED42_SUBMIT_V1.md}"

mkdir -p "$OUT_ROOT"
mkdir -p "$(dirname "$REPORT_TSV")"
mkdir -p "$(dirname "$REPORT_MD")"

echo -e "run_name\tunique_change_point\tjob_id\tstatus_file\tmain_log\toutput_dir" > "$REPORT_TSV"

submit_one() {
  local run_name="$1"
  local unique_change_point="$2"
  local mode="$3"

  local out_dir="$OUT_ROOT/seed_${SEED}/${run_name}"
  mkdir -p "$out_dir"

  local notes="Semteacher seed42 mainline v1 | data_mode=${DATA_MODE}"
  local resume_hint="Resume with same output_dir and --auto-resume; compare by official rule on full+semantic-hard"

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
    --semantic-adapter-mode teacher_v2
    --semteacher-strict
    --semteacher-use-teacher-targets
    --semteacher-require-cache-fields
    --semteacher-capability-report "$out_dir/semteacher_capability_gap.json"
    --semteacher-hard-manifest "$SEMANTIC_HARD_MANIFEST"
    --neutralize-object-bias
    --qtsa-disable-semantic-transition
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

  case "$mode" in
    control)
      ;;
    distill)
      cmd+=(
        --semteacher-distill-enable
        --semteacher-distill-weight "$SEMTEACHER_DISTILL_WEIGHT"
        --semteacher-association-temperature "$SEMTEACHER_ASSOC_TEMP"
      )
      ;;
    distill_rerank)
      cmd+=(
        --semteacher-distill-enable
        --semteacher-distill-weight "$SEMTEACHER_DISTILL_WEIGHT"
        --semteacher-association-temperature "$SEMTEACHER_ASSOC_TEMP"
        --semteacher-confidence-rerank-enable
        --semteacher-confidence-threshold "$SEMTEACHER_CONF_THRESH"
        --semteacher-rerank-topk "$SEMTEACHER_RERANK_TOPK"
      )
      ;;
    *)
      echo "unknown mode: $mode" >&2
      exit 2
      ;;
  esac

  local submit_output
  submit_output="$(bash "$REPO_ROOT/scripts/protocol_v2_queue_submit.sh" \
    --queue-dir "$QUEUE_DIR" \
    --job-name "$run_name" \
    --class-type B \
    --workdir "$REPO_ROOT" \
    --notes "$notes" \
    --resume-hint "$resume_hint" \
    -- "${cmd[@]}")"
  echo "$submit_output"

  local job_id status_file main_log
  job_id="$(echo "$submit_output" | sed -n 's/^  job_id:[[:space:]]*//p' | tail -n 1)"
  status_file="$(echo "$submit_output" | sed -n 's/^  status_file:[[:space:]]*//p' | tail -n 1)"
  main_log="$(echo "$submit_output" | sed -n 's/^  main_log:[[:space:]]*//p' | tail -n 1)"

  echo -e "${run_name}\t${unique_change_point}\t${job_id}\t${status_file}\t${main_log}\t${out_dir}" >> "$REPORT_TSV"
}

submit_one "trace_baseline_seed42_semteacher_control_v1" "teacher-grounded semantics control; no distill, no rerank; no transition rewrite" "control"
submit_one "semteacher_distill_seed42_challenge_v1" "teacher-grounded query-readout distill only; no transition rewrite" "distill"
submit_one "semteacher_distill_confidence_rerank_seed42_challenge_v1" "teacher-grounded distill + confidence-gated semantic-hard rerank" "distill_rerank"

python - "$REPORT_TSV" "$REPORT_MD" <<'PY'
from pathlib import Path
import csv
import time
import sys

tsv = Path(sys.argv[1])
md = Path(sys.argv[2])
with tsv.open('r', newline='') as f:
    rows = list(csv.DictReader(f, delimiter='\t'))

lines = []
lines.append('# STWM Semteacher Mainline Seed42 Submit V1')
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

echo "[semteacher-seed42-enqueue] queue_dir=$QUEUE_DIR"
echo "[semteacher-seed42-enqueue] submit_tsv=$REPORT_TSV"
echo "[semteacher-seed42-enqueue] submit_doc=$REPORT_MD"
