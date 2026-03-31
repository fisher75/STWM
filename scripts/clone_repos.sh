#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/common.sh"

ensure_dir "$STWM_ROOT/third_party" "$STWM_ROOT/manifests"
REPOS_CSV="$STWM_ROOT/manifests/repos.csv"
init_csv "$REPOS_CSV" "name,url,path,commit_hash,cloned_at,status"

REPOS=(
  "TraceAnything|https://github.com/ByteDance-Seed/TraceAnything.git"
  "OV2VSS|https://github.com/AVC2-UESTC/OV2VSS.git"
  "sam2|https://github.com/facebookresearch/sam2.git"
  "Tracking-Anything-with-DEVA|https://github.com/hkchengrex/Tracking-Anything-with-DEVA.git"
  "Cutie|https://github.com/hkchengrex/Cutie.git"
  "XMem|https://github.com/hkchengrex/XMem.git"
  "YOLO-World|https://github.com/AILab-CVC/YOLO-World.git"
)

for spec in "${REPOS[@]}"; do
  IFS="|" read -r name url <<< "$spec"
  dst="$STWM_ROOT/third_party/$name"
  status="ok"

  if [[ -d "$dst/.git" ]]; then
    echo "Repo exists, recording current state without modifying: $name"
  else
    retry 3 git clone --depth 1 "$url" "$dst" || status="clone_failed"
  fi

  commit_hash="-"
  if [[ -d "$dst/.git" ]]; then
    commit_hash="$(git -C "$dst" rev-parse HEAD)"
  fi

  printf "%s,%s,%s,%s,%s,%s\n" \
    "$name" "$url" "$dst" "$commit_hash" "$(timestamp)" "$status" >> "$REPOS_CSV"
done

echo "Repository manifest updated at $REPOS_CSV"
