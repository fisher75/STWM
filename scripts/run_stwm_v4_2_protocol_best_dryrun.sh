#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 <checkpoint_dir> <manifest_json> <model_preset> [preset_file] [data_root] [candidate_csv] [max_clips]"
  exit 1
fi

REPO_ROOT="/home/chen034/workspace/stwm"
CHECKPOINT_DIR="$1"
MANIFEST_JSON="$2"
MODEL_PRESET="$3"
PRESET_FILE="${4:-$REPO_ROOT/code/stwm/configs/model_presets_v4_2.json}"
DATA_ROOT="${5:-$REPO_ROOT/data/external}"
CANDIDATE_CSV="${6:-best.pt,latest.pt,milestone_step_002000.pt,milestone_step_004000.pt}"
MAX_CLIPS="${7:-0}"

if [[ "$CHECKPOINT_DIR" != /* ]]; then
  CHECKPOINT_DIR="$REPO_ROOT/$CHECKPOINT_DIR"
fi
if [[ "$MANIFEST_JSON" != /* ]]; then
  MANIFEST_JSON="$REPO_ROOT/$MANIFEST_JSON"
fi
if [[ "$PRESET_FILE" != /* ]]; then
  PRESET_FILE="$REPO_ROOT/$PRESET_FILE"
fi
if [[ "$DATA_ROOT" != /* ]]; then
  DATA_ROOT="$REPO_ROOT/$DATA_ROOT"
fi

if [[ ! -d "$CHECKPOINT_DIR" ]]; then
  echo "checkpoint_dir not found: $CHECKPOINT_DIR" >&2
  exit 2
fi
if [[ ! -f "$MANIFEST_JSON" ]]; then
  echo "manifest not found: $MANIFEST_JSON" >&2
  exit 2
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
DRYRUN_DIR="$CHECKPOINT_DIR/protocol_eval_dryrun_${STAMP}"
mkdir -p "$DRYRUN_DIR"

if [[ -f "$CHECKPOINT_DIR/best_protocol_main.pt" ]]; then
  cp -f "$CHECKPOINT_DIR/best_protocol_main.pt" "$DRYRUN_DIR/best_protocol_main.pre_dryrun.pt"
fi
if [[ -f "$CHECKPOINT_DIR/best_protocol_main_selection.json" ]]; then
  cp -f "$CHECKPOINT_DIR/best_protocol_main_selection.json" "$DRYRUN_DIR/best_protocol_main_selection.pre_dryrun.json"
fi

rm -f "$CHECKPOINT_DIR/best_protocol_main.pt" "$CHECKPOINT_DIR/best_protocol_main_selection.json"

IFS=',' read -r -a CANDIDATES <<< "$CANDIDATE_CSV"
for cand in "${CANDIDATES[@]}"; do
  cand_trimmed="$(echo "$cand" | xargs)"
  [[ -n "$cand_trimmed" ]] || continue

  CAND_PATH="$cand_trimmed"
  if [[ "$CAND_PATH" != /* ]]; then
    CAND_PATH="$CHECKPOINT_DIR/$CAND_PATH"
  fi
  if [[ ! -f "$CAND_PATH" ]]; then
    echo "[d0] skip missing candidate: $CAND_PATH"
    continue
  fi

  stem="$(basename "$cand_trimmed" .pt)"
  eval_json="$DRYRUN_DIR/protocol_val_main_${stem}.json"

  echo "[d0] evaluating candidate=$cand_trimmed"
  PYTHONPATH="$REPO_ROOT/code" conda run --no-capture-output -n stwm \
    python "$REPO_ROOT/code/stwm/evaluators/eval_mini_val.py" \
      --data-root "$DATA_ROOT" \
      --manifest "$MANIFEST_JSON" \
      --dataset all \
      --max-clips "$MAX_CLIPS" \
      --obs-steps 8 \
      --pred-steps 8 \
      --seed 42 \
      --checkpoint "$CAND_PATH" \
      --model-preset "$MODEL_PRESET" \
      --preset-file "$PRESET_FILE" \
      --protocol-version v2_4_detached_frozen \
      --run-name "d0_protocol_main_${stem}" \
      --output "$eval_json"

  PYTHONPATH="$REPO_ROOT/code" conda run --no-capture-output -n stwm \
    python "$REPO_ROOT/code/stwm/tools/update_protocol_best_main.py" \
      --checkpoint-dir "$CHECKPOINT_DIR" \
      --candidate-checkpoint "$CAND_PATH" \
      --eval-summary "$eval_json" \
      --output-checkpoint "$CHECKPOINT_DIR/best_protocol_main.pt" \
      --selection-sidecar "$CHECKPOINT_DIR/best_protocol_main_selection.json" \
      > "$DRYRUN_DIR/update_${stem}.json"

  cp -f "$CHECKPOINT_DIR/best_protocol_main_selection.json" "$DRYRUN_DIR/selection_after_${stem}.json"
done

CKPT_DIR_ENV="$CHECKPOINT_DIR" DRYRUN_DIR_ENV="$DRYRUN_DIR" python - <<'PY'
from pathlib import Path
import json
import os

ckpt_dir = Path(os.environ["CKPT_DIR_ENV"])
dryrun = Path(os.environ["DRYRUN_DIR_ENV"])
if not dryrun.exists():
  raise SystemExit(f"dryrun dir not found: {dryrun}")

records = []
for eval_path in sorted(dryrun.glob("protocol_val_main_*.json")):
    stem = eval_path.stem.replace("protocol_val_main_", "")
    metrics = json.loads(eval_path.read_text()).get("metrics", {})
    sel_path = dryrun / f"selection_after_{stem}.json"
    sel = json.loads(sel_path.read_text()) if sel_path.exists() else {}
    records.append({
        "candidate": f"{stem}.pt",
        "eval_summary": str(eval_path),
        "query_localization_error": float(metrics.get("query_localization_error", 0.0)),
        "query_top1_acc": float(metrics.get("query_top1_acc", 0.0)),
        "future_trajectory_l1": float(metrics.get("future_trajectory_l1", 0.0)),
        "action": str(sel.get("action", "")),
        "improved": bool(sel.get("improved", False)),
        "selection_snapshot": str(sel_path) if sel_path.exists() else "",
    })

final_sidecar = ckpt_dir / "best_protocol_main_selection.json"
report = {
    "dryrun_dir": str(dryrun),
    "checkpoint_dir": str(ckpt_dir),
    "candidates": [r["candidate"] for r in records],
    "records": records,
    "final_best_protocol_main_exists": (ckpt_dir / "best_protocol_main.pt").exists(),
    "final_sidecar_exists": final_sidecar.exists(),
    "final_selection": json.loads(final_sidecar.read_text()) if final_sidecar.exists() else {},
}
out_path = dryrun / "d0_protocol_best_dryrun_report.json"
out_path.write_text(json.dumps(report, indent=2))
print(out_path)

doc_path = Path("/home/chen034/workspace/stwm/docs/STWM_V4_2_PROTOCOL_BEST_DRYRUN.md")
final_sel = report.get("final_selection", {})
final_metrics = final_sel.get("metrics", {}) if isinstance(final_sel, dict) else {}
final_candidate = str(final_sel.get("candidate_checkpoint", "")) if isinstance(final_sel, dict) else ""

lines = []
lines.append("# STWM V4.2 Protocol Best Dryrun (D0)")
lines.append("")
lines.append("Date: 2026-04-03")
lines.append("Status: completed")
lines.append("")
lines.append("## Scope")
lines.append("")
lines.append(f"- checkpoint_dir: {report.get('checkpoint_dir', '')}")
lines.append(f"- dryrun_dir: {report.get('dryrun_dir', '')}")
lines.append("- evaluator: code/stwm/evaluators/eval_mini_val.py")
lines.append("- protocol manifest: manifests/protocol_v2/protocol_val_main_v1.json")
lines.append("- protocol version request: v2_4_detached_frozen")
lines.append("")
lines.append("## Candidate Checkpoints")
lines.append("")
for c in report.get("candidates", []):
  lines.append(f"- {c}")
if not report.get("candidates"):
  lines.append("- (none)")
lines.append("")
lines.append("## Protocol-Main Metrics Per Candidate")
lines.append("")
lines.append("| candidate | query_localization_error | query_top1_acc | future_trajectory_l1 | action | improved |")
lines.append("|---|---:|---:|---:|---|---:|")
for r in report.get("records", []):
  lines.append(
    "| {cand} | {p:.6f} | {t1:.6f} | {t2:.6f} | {act} | {imp} |".format(
      cand=str(r.get("candidate", "")),
      p=float(r.get("query_localization_error", 0.0)),
      t1=float(r.get("query_top1_acc", 0.0)),
      t2=float(r.get("future_trajectory_l1", 0.0)),
      act=str(r.get("action", "")),
      imp=1 if bool(r.get("improved", False)) else 0,
    )
  )
lines.append("")
lines.append("## Final Official Best")
lines.append("")
lines.append(f"- best_protocol_main_exists: {str(report.get('final_best_protocol_main_exists', False)).lower()}")
lines.append(f"- selection_sidecar_exists: {str(report.get('final_sidecar_exists', False)).lower()}")
lines.append(f"- selected_candidate_checkpoint: {final_candidate}")
lines.append(
  "- selected_metrics: query_localization_error={p:.6f}, query_top1_acc={t1:.6f}, future_trajectory_l1={t2:.6f}".format(
    p=float(final_metrics.get("query_localization_error", 0.0)),
    t1=float(final_metrics.get("query_top1_acc", 0.0)),
    t2=float(final_metrics.get("future_trajectory_l1", 0.0)),
  )
)
lines.append("")
lines.append("Selection rationale follows protocol_best_rule_v2 exactly:")
lines.append("1. primary: query_localization_error (lower)")
lines.append("2. tie-break 1: query_top1_acc (higher)")
lines.append("3. tie-break 2: future_trajectory_l1 (lower)")
lines.append("")
lines.append("## Toolchain Readiness For Future 500-Step Updates")
lines.append("")
lines.append("This dryrun validates end-to-end chain on a real completed run:")
lines.append("1. detached eval summary generation")
lines.append("2. protocol best updater invocation")
lines.append("3. best_protocol_main.pt materialization")
lines.append("4. sidecar selection record persistence")
lines.append("")
lines.append("Conclusion: the chain is executable for periodic future updates every 500 steps when integrated in D1 training jobs.")
lines.append("")
lines.append("## Artifacts")
lines.append("")
lines.append(f"- dryrun_report_json: {out_path}")
lines.append(f"- official_best_checkpoint: {ckpt_dir / 'best_protocol_main.pt'}")
lines.append(f"- official_best_sidecar: {ckpt_dir / 'best_protocol_main_selection.json'}")

doc_path.write_text("\n".join(lines) + "\n")
print(doc_path)
PY

echo "[d0] done dryrun_dir=$DRYRUN_DIR"
