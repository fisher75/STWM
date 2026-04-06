#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home/chen034/workspace/stwm"
ARCHIVE_ROOT="$REPO_ROOT/archives"
LIST_ROOT="$REPO_ROOT/reports/archive_lists_20260406"
LOG_FILE="$REPO_ROOT/reports/outputs_archive_20260406.run.log"
SUMMARY_JSON="$REPO_ROOT/reports/outputs_archive_report_20260406.json"

mkdir -p "$ARCHIVE_ROOT"
mkdir -p "$REPO_ROOT/reports"

ARCHIVE_GROUPS=(training eval visualizations queue audits benchmarks smoke_tests background_jobs)

echo "[$(date '+%F %T')] archive start" | tee "$LOG_FILE"

json_entries=""

for g in "${ARCHIVE_GROUPS[@]}"; do
  list_file="$LIST_ROOT/${g}_paths.txt"
  archive_file="$ARCHIVE_ROOT/outputs_${g}_archive_20260406.tar.zst"
  sha_file="${archive_file}.sha256"
  manifest_file="$ARCHIVE_ROOT/outputs_${g}_archive_20260406.contents_manifest.txt"

  if [[ ! -s "$list_file" ]]; then
    echo "[$(date '+%F %T')] SKIP $g (empty list)" | tee -a "$LOG_FILE"
    continue
  fi

  pre_size_bytes=$(awk '{print $0}' "$list_file" | while read -r p; do
    if [[ -e "$REPO_ROOT/$p" ]]; then
      du -sb "$REPO_ROOT/$p" | awk '{print $1}'
    fi
  done | awk '{s+=$1} END {print s+0}')

  echo "[$(date '+%F %T')] PACK $g pre_size_bytes=$pre_size_bytes" | tee -a "$LOG_FILE"

  (
    cd "$REPO_ROOT"
    tar --use-compress-program="zstd -T0 -1" -cf "$archive_file" -T "$list_file"
  )

  sha256sum "$archive_file" > "$sha_file"
  tar --use-compress-program="zstd -d -T0" -tf "$archive_file" > "$manifest_file"
  sha256sum -c "$sha_file" >> "$LOG_FILE" 2>&1

  archive_size_bytes=$(stat -c %s "$archive_file")
  ratio=$(python3 - <<PY
pre=$pre_size_bytes
arc=$archive_size_bytes
print(f"{(arc/pre if pre else 0):.6f}")
PY
)

  echo "[$(date '+%F %T')] DONE $g archive_size_bytes=$archive_size_bytes ratio=$ratio" | tee -a "$LOG_FILE"

  json_entries+=$(cat <<JSON
{"group":"$g","list_file":"$list_file","archive_file":"$archive_file","sha256_file":"$sha_file","manifest_file":"$manifest_file","pre_size_bytes":$pre_size_bytes,"archive_size_bytes":$archive_size_bytes,"compression_ratio":$ratio},
JSON
)
done

json_entries="[${json_entries%,}]"

cat > "$SUMMARY_JSON" <<JSON
{
  "generated_at": "$(date '+%F %T')",
  "archive_root": "$ARCHIVE_ROOT",
  "entries": $json_entries
}
JSON

echo "[$(date '+%F %T')] archive finished" | tee -a "$LOG_FILE"
echo "$SUMMARY_JSON"
