#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/common.sh"

ensure_dir "$STWM_ROOT/data/raw" "$STWM_ROOT/data/external" "$STWM_ROOT/manifests"
DATASETS_CSV="$STWM_ROOT/manifests/datasets.csv"
init_csv "$DATASETS_CSV" "name,status,source,target_archive,target_extract,size_bytes,updated_at,notes"

record_dataset() {
  local name="$1"
  local status="$2"
  local source="$3"
  local archive="$4"
  local extract="$5"
  local notes="$6"
  local size_bytes="0"
  if [[ -e "$archive" ]]; then
    size_bytes="$(stat -c %s "$archive" 2>/dev/null || du -sb "$archive" | awk '{print $1}')"
  elif [[ -e "$extract" ]]; then
    size_bytes="$(du -sb "$extract" | awk '{print $1}')"
  fi
  printf "%s,%s,%s,%s,%s,%s,%s,%s\n" \
    "$name" "$status" "$source" "$archive" "$extract" "$size_bytes" "$(timestamp)" "$notes" >> "$DATASETS_CSV"
}

extract_archive() {
  local archive="$1"
  local dst="$2"
  ensure_dir "$dst"
  local file_type=""
  file_type="$(file -b "$archive" 2>/dev/null || true)"

  if [[ "$file_type" == *"tar archive"* ]]; then
    tar -xf "$archive" -C "$dst"
    return 0
  fi

  case "$archive" in
    *.zip)
      unzip -o "$archive" -d "$dst"
      ;;
    *.tar.gz|*.tgz)
      tar -xzf "$archive" -C "$dst"
      ;;
    *.tar)
      tar -xf "$archive" -C "$dst"
      ;;
    *)
      echo "Skipping extraction for unsupported archive type: $archive"
      return 0
      ;;
  esac
}

download_archive() {
  local url="$1"
  local out_path="$2"
  ensure_dir "$(dirname "$out_path")"
  if [[ "$url" == *"drive.google.com"* ]]; then
    if command -v gdown >/dev/null 2>&1; then
      retry 3 gdown --fuzzy --continue "$url" -O "$out_path"
      return 0
    fi
    if python -m gdown --help >/dev/null 2>&1; then
      retry 3 python -m gdown --fuzzy --continue "$url" -O "$out_path"
      return 0
    fi
    if conda env list 2>/dev/null | awk '{print $1}' | grep -Fxq stwm; then
      retry 3 conda run -n stwm python -m gdown --fuzzy --continue "$url" -O "$out_path"
      return 0
    fi
    echo "gdown is required for Google Drive downloads but is not currently available."
    return 1
  fi

  retry 3 aria2c -c -x "$STWM_DOWNLOAD_CONNECTIONS" -s "$STWM_DOWNLOAD_SPLITS" -d "$(dirname "$out_path")" -o "$(basename "$out_path")" "$url"
}

archive_and_extract() {
  local name="$1"
  local url="$2"
  local archive="$3"
  local extract="$4"
  local notes="$5"
  if [[ -z "$url" ]]; then
    note_blocker "$name download URL missing" "$notes"
    record_dataset "$name" "blocked" "-" "$archive" "$extract" "$notes"
    return 0
  fi
  if download_archive "$url" "$archive"; then
    extract_archive "$archive" "$extract" || true
    record_dataset "$name" "downloaded" "$url" "$archive" "$extract" "$notes"
  else
    note_blocker "$name download failed" "URL: $url"$'\n'"Archive: $archive"
    record_dataset "$name" "failed" "$url" "$archive" "$extract" "$notes"
  fi
}

download_vspw() {
  archive_and_extract \
    "VSPW" \
    "${VSPW_URL:-https://drive.google.com/file/d/14yHWsGneoa1pVdULFk7cah3t-THl7yEz/view?usp=sharing}" \
    "$STWM_ROOT/data/raw/vspw/VSPW_data.tar" \
    "$STWM_ROOT/data/external/vspw" \
    "Official VSPW full dataset archive"
}

download_visor() {
  archive_and_extract \
    "VISOR" \
    "${VISOR_URL:-https://data.bris.ac.uk/datasets/tar/2v6cgv1x04ol22qp9rm9x2j6a7.zip}" \
    "$STWM_ROOT/data/raw/visor/visor_complete.zip" \
    "$STWM_ROOT/data/external/visor" \
    "Official VISOR complete archive"
}

download_vipseg() {
  archive_and_extract \
    "VIPSeg" \
    "${VIPSEG_URL:-https://drive.google.com/file/d/1B13QUiE82xf7N6nVHclb4ErN-Zuai-sZ/view?usp=sharing}" \
    "$STWM_ROOT/data/raw/vipseg/vipseg_archive.zip" \
    "$STWM_ROOT/data/external/vipseg" \
    "Official VIPSeg dataset archive"
}

download_burst() {
  archive_and_extract \
    "BURST annotations" \
    "${BURST_ANN_URL:-https://omnomnom.vision.rwth-aachen.de/data/BURST/annotations.zip}" \
    "$STWM_ROOT/data/raw/burst/annotations.zip" \
    "$STWM_ROOT/data/external/burst/annotations" \
    "Official BURST annotations archive"

  archive_and_extract \
    "TAO train images" \
    "${TAO_TRAIN_URL:-https://motchallenge.net/data/1-TAO_TRAIN.zip}" \
    "$STWM_ROOT/data/raw/burst/1-TAO_TRAIN.zip" \
    "$STWM_ROOT/data/external/burst/images/train" \
    "Official TAO train archive"

  archive_and_extract \
    "TAO val images" \
    "${TAO_VAL_URL:-https://motchallenge.net/data/2-TAO_VAL.zip}" \
    "$STWM_ROOT/data/raw/burst/2-TAO_VAL.zip" \
    "$STWM_ROOT/data/external/burst/images/val" \
    "Official TAO val archive"

  archive_and_extract \
    "TAO test images" \
    "${TAO_TEST_URL:-https://motchallenge.net/data/3-TAO_TEST.zip}" \
    "$STWM_ROOT/data/raw/burst/3-TAO_TEST.zip" \
    "$STWM_ROOT/data/external/burst/images/test" \
    "Official TAO test archive"
}

download_burst_test() {
  archive_and_extract \
    "TAO test images" \
    "${TAO_TEST_URL:-https://motchallenge.net/data/3-TAO_TEST.zip}" \
    "$STWM_ROOT/data/raw/burst/3-TAO_TEST.zip" \
    "$STWM_ROOT/data/external/burst/images/test" \
    "Official TAO test archive"
}

TARGET="${1:---all}"

case "$TARGET" in
  --all|all)
    download_vspw
    download_visor
    download_vipseg
    download_burst
    ;;
  vspw)
    download_vspw
    ;;
  visor)
    download_visor
    ;;
  vipseg)
    download_vipseg
    ;;
  burst|tao)
    download_burst
    ;;
  burst-test|burst_test|tao-test|tao_test)
    download_burst_test
    ;;
  *)
    echo "Unknown target: $TARGET"
    exit 1
    ;;
esac
