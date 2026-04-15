#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/chen034/workspace/stwm
LOG=/home/chen034/workspace/stwm/logs/stage2_calibration_only_finalization_pack_20260414.log
mkdir -p "$(dirname "$LOG")"
source /home/chen034/miniconda3/etc/profile.d/conda.sh
conda activate stwm
export PYTHONPATH="$ROOT/code:${PYTHONPATH:-}"
export STWM_PROC_TITLE="${STWM_PROC_TITLE:-python}"
export STWM_PROC_TITLE_MODE="${STWM_PROC_TITLE_MODE:-generic}"
{
  echo "[$(date -Iseconds)] finalization_pack_monitor_start"
  python - <<'PY2'
import subprocess
from stwm.tools import run_tracewm_stage2_calibration_only_wave2_20260414 as m
args = m.parse_args()
summary = m.wait_for_completion(args)
diagnosis = m.diagnose(args)
ablation = m._json_or_empty(args.ablation_pack_report)
if bool(summary.get('all_runs_terminal', False)) and int(summary.get('failed_count', 0)) == 0 and bool(ablation.get('all_runs_terminal', False)) and int(ablation.get('failed_count', 0)) == 0:
    m._run_aux_external_probe_batch(args, summary)
    qual_cmd = [
        str(args.python_bin),
        str(m.WORK_ROOT / 'code/stwm/tools/run_tracewm_stage1_stage2_qualitative_pack_v5_20260414.py'),
        '--wave2-summary-report', str(args.summary_report),
        '--wave2-diagnosis-report', str(args.diagnosis_report),
        '--ablation-pack-report', str(args.ablation_pack_report),
        '--final-pack-diagnosis-report', str(args.final_pack_diagnosis_report),
    ]
    subprocess.run(qual_cmd, cwd=str(args.work_root), check=False)
summary = m.summarize(args)
diagnosis = m.diagnose(args)
print({'summary_status': summary.get('stage2_calibration_only_wave2_status'), 'diagnosis_status': diagnosis.get('status'), 'next_step_choice': m._json_or_empty(args.final_pack_diagnosis_report).get('next_step_choice')})
PY2
  echo "[$(date -Iseconds)] finalization_pack_monitor_end"
} >> "$LOG" 2>&1
