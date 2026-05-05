#!/usr/bin/env bash
set -euo pipefail

ROOT=/raid/chen034/workspace/stwm
MANIFEST="$ROOT/reports/stwm_traceanything_hardbench_launch_manifest_v25_20260502.json"
WATCH_PATH="$ROOT/reports/stwm_traceanything_hardbench_watcher_v25_20260502.json"
POLL_SECONDS="${POLL_SECONDS:-30}"
WAIT_MODE="${WAIT_MODE:-1}"

while true; do
  /home/chen034/miniconda3/envs/stwm/bin/python - "$MANIFEST" "$WATCH_PATH" <<'PY'
import json, subprocess, sys, time
from pathlib import Path

manifest_path = Path(sys.argv[1])
watch_path = Path(sys.argv[2])
root = Path('/raid/chen034/workspace/stwm')
payload = json.loads(manifest_path.read_text())
rows = []
all_terminal = True

def tmux_exists(session: str) -> bool:
    return subprocess.call(["tmux", "has-session", "-t", session], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) == 0

for row in payload.get("launch_manifest", []):
    report = root / row["report_path"]
    log = root / row["log_path"]
    session = row["session_name"]
    if tmux_exists(session):
        status = "running"
        all_terminal = False
        reason = None
    elif report.exists():
        data = json.loads(report.read_text())
        status = data.get("shard_terminal_status", "completed")
        reason = None
        if status == "skipped_with_reason":
            reason = data.get("failed_clip_reasons") or data.get("selection_stats", {}).get("candidate_blocker_counts")
        elif status == "failed":
            reason = data.get("failed_clip_reasons") or "failed_without_report_reason"
    else:
        status = "failed"
        reason = "tmux_ended_but_report_missing"
    rows.append(
        {
            **row,
            "terminal_status": status,
            "reason": reason,
            "log_exists": log.exists(),
            "report_exists": report.exists(),
        }
    )

summary = {
    "watcher_name": "stwm_traceanything_hardbench_watcher_v25",
    "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "all_terminal": all_terminal,
    "completed_count": sum(r["terminal_status"] == "completed" for r in rows),
    "failed_count": sum(r["terminal_status"] == "failed" for r in rows),
    "skipped_count": sum(r["terminal_status"] == "skipped_with_reason" for r in rows),
    "running_count": sum(r["terminal_status"] == "running" for r in rows),
    "rows": rows,
}
watch_path.write_text(json.dumps(summary, indent=2) + "\n")
print(str(summary["all_terminal"]).lower())
PY
  done_flag="$(tail -n 1 "$WATCH_PATH" 2>/dev/null | tr -d '\n\r ' || true)"
  all_terminal="$(/home/chen034/miniconda3/envs/stwm/bin/python - <<'PY'
import json
from pathlib import Path
p=Path('/raid/chen034/workspace/stwm/reports/stwm_traceanything_hardbench_watcher_v25_20260502.json')
print('true' if json.loads(p.read_text()).get('all_terminal') else 'false')
PY
)"
  if [[ "$WAIT_MODE" != "1" || "$all_terminal" == "true" ]]; then
    break
  fi
  sleep "$POLL_SECONDS"
done

echo "$WATCH_PATH"
