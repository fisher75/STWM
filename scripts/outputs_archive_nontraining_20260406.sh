#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home/chen034/workspace/stwm"
ARCHIVE_ROOT="$REPO_ROOT/archives"
LIST_ROOT="$REPO_ROOT/reports/archive_lists_20260406"
LOG_FILE="$REPO_ROOT/reports/outputs_archive_nontraining_20260406.run.log"
SUMMARY_JSON="$REPO_ROOT/reports/outputs_archive_nontraining_report_20260406.json"

mkdir -p "$ARCHIVE_ROOT"
mkdir -p "$REPO_ROOT/reports"

NONTRAIN_GROUPS=(eval visualizations queue audits benchmarks smoke_tests background_jobs)

echo "[$(date '+%F %T')] nontraining archive start" | tee "$LOG_FILE"

tmp_json="$REPO_ROOT/reports/.outputs_archive_nontraining_entries_20260406.jsonl"
: > "$tmp_json"

for g in "${NONTRAIN_GROUPS[@]}"; do
  list_file="$LIST_ROOT/${g}_paths.txt"
  archive_file="$ARCHIVE_ROOT/outputs_${g}_archive_20260406.tar.zst"
  sha_file="${archive_file}.sha256"
  manifest_file="$ARCHIVE_ROOT/outputs_${g}_archive_20260406.contents_manifest.txt"

  if [[ ! -s "$list_file" ]]; then
    echo "[$(date '+%F %T')] SKIP $g empty list" | tee -a "$LOG_FILE"
    continue
  fi

  pre_size_bytes=$(awk '{print $0}' "$list_file" | while read -r p; do
    if [[ -e "$REPO_ROOT/$p" ]]; then
      du -sb "$REPO_ROOT/$p" | awk '{print $1}'
    fi
  done | awk '{s+=$1} END {print s+0}')

  echo "[$(date '+%F %T')] GROUP $g pre_size_bytes=$pre_size_bytes" | tee -a "$LOG_FILE"

  if [[ ! -s "$archive_file" ]]; then
    (
      cd "$REPO_ROOT"
      tar -cvf - -T "$list_file" 2> "$manifest_file" \
        | zstd -T0 -1 \
        | tee "$archive_file" \
        | sha256sum \
        | awk -v f="$archive_file" '{print $1 "  " f}' > "$sha_file"
    )
  else
    [[ -s "$sha_file" ]] || sha256sum "$archive_file" > "$sha_file"
    [[ -s "$manifest_file" ]] || (
      cd "$REPO_ROOT"
      while read -r p; do
        [[ -z "$p" ]] && continue
        [[ -e "$p" ]] && find "$p" -print
      done < "$list_file" > "$manifest_file"
    )
  fi

  if tar --use-compress-program="zstd -d -T0" -tf "$archive_file" | head -n 5 >/dev/null; then
    list_check="ok"
  else
    list_check="failed"
  fi

  if [[ -s "$sha_file" ]] && awk 'NF>=2 {ok=1} END{exit ok?0:1}' "$sha_file"; then
    sha_check="ok"
  else
    sha_check="failed"
  fi

  archive_size_bytes=$(stat -c %s "$archive_file")
  ratio=$(python3 - <<PY
pre=$pre_size_bytes
arc=$archive_size_bytes
print(f"{(arc/pre if pre else 0):.6f}")
PY
)

  echo "[$(date '+%F %T')] DONE $g archive_size_bytes=$archive_size_bytes ratio=$ratio sha_check=$sha_check list_check=$list_check" | tee -a "$LOG_FILE"

  python3 - <<PY >> "$tmp_json"
import json
print(json.dumps({
  "group": "$g",
  "list_file": "$list_file",
  "archive_file": "$archive_file",
  "sha256_file": "$sha_file",
  "manifest_file": "$manifest_file",
  "pre_size_bytes": int($pre_size_bytes),
  "archive_size_bytes": int($archive_size_bytes),
  "compression_ratio": float($ratio),
  "sha_check": "$sha_check",
  "list_check": "$list_check",
}, ensure_ascii=False))
PY
done

python3 - <<PY
import json
from datetime import datetime
tmp = "$tmp_json"
out = "$SUMMARY_JSON"
entries = []
with open(tmp, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            entries.append(json.loads(line))
payload = {
    "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "archive_root": "$ARCHIVE_ROOT",
    "entries": entries,
}
with open(out, "w", encoding="utf-8") as f:
    json.dump(payload, f, indent=2, ensure_ascii=False)
print(out)
PY

echo "[$(date '+%F %T')] nontraining archive finished" | tee -a "$LOG_FILE"
