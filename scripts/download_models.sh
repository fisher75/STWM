#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/common.sh"

ensure_dir "$STWM_ROOT/models/checkpoints" "$STWM_ROOT/manifests"
MODELS_CSV="$STWM_ROOT/manifests/models.csv"
init_csv "$MODELS_CSV" "name,status,source_url,target_path,size_bytes,updated_at,notes"

record_model() {
  local name="$1"
  local status="$2"
  local url="$3"
  local target="$4"
  local notes="$5"
  local size_bytes="0"
  if [[ -e "$target" ]]; then
    size_bytes="$(stat -c %s "$target" 2>/dev/null || du -sb "$target" | awk '{print $1}')"
  fi
  printf "%s,%s,%s,%s,%s,%s,%s\n" \
    "$name" "$status" "$url" "$target" "$size_bytes" "$(timestamp)" "$notes" >> "$MODELS_CSV"
}

download_file() {
  local url="$1"
  local out_path="$2"
  ensure_dir "$(dirname "$out_path")"
  if [[ "$url" == *"drive.google.com"* ]]; then
    retry 3 gdown --fuzzy --continue "$url" -O "$out_path"
  else
    retry 3 aria2c -c -x "$STWM_DOWNLOAD_CONNECTIONS" -s "$STWM_DOWNLOAD_SPLITS" -d "$(dirname "$out_path")" -o "$(basename "$out_path")" "$url"
  fi
}

direct_or_block() {
  local name="$1"
  local url="$2"
  local out_path="$3"
  local notes="$4"
  if [[ -z "$url" ]]; then
    note_blocker "$name checkpoint URL missing" "$notes"
    record_model "$name" "blocked" "-" "$out_path" "$notes"
    return 0
  fi
  if download_file "$url" "$out_path"; then
    record_model "$name" "downloaded" "$url" "$out_path" "$notes"
  else
    note_blocker "$name checkpoint download failed" "URL: $url"$'\n'"Target: $out_path"
    record_model "$name" "failed" "$url" "$out_path" "$notes"
  fi
}

download_traceanything() {
  local dst_dir="$STWM_ROOT/models/checkpoints/traceanything"
  local url="${TRACEANYTHING_CKPT_URL:-https://huggingface.co/depth-anything/trace-anything/resolve/main/trace_anything.pt?download=true}"
  local out_path="$dst_dir/traceanything_pretrained.pt"
  direct_or_block \
    "TraceAnything pretrained" \
    "$url" \
    "$out_path" \
    "Official TraceAnything checkpoint"
}

download_sam2() {
  local dst_dir="$STWM_ROOT/models/checkpoints/sam2"
  local large_url="${SAM2_LARGE_URL:-https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt}"
  local basep_url="${SAM2_BASE_PLUS_URL:-https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt}"
  direct_or_block "SAM2.1 hiera large" "$large_url" "$dst_dir/sam2.1_hiera_large.pt" "Official SAM2.1 checkpoint"
  direct_or_block "SAM2.1 hiera base+" "$basep_url" "$dst_dir/sam2.1_hiera_base_plus.pt" "Official SAM2.1 checkpoint"
}

download_deva() {
  local dst_dir="$STWM_ROOT/models/checkpoints/deva"
  direct_or_block "DEVA propagation" \
    "${DEVA_CKPT_URL:-https://github.com/hkchengrex/Tracking-Anything-with-DEVA/releases/download/v1.0/DEVA-propagation.pth}" \
    "$dst_dir/DEVA-propagation.pth" \
    "Official DEVA weight"
  direct_or_block "DEVA groundingdino swint" \
    "${DEVA_GDINO_URL:-https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth}" \
    "$dst_dir/groundingdino_swint_ogc.pth" \
    "Official GroundingDINO dependency for DEVA"
  direct_or_block "DEVA SAM ViT-H" \
    "${DEVA_SAM_URL:-https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth}" \
    "$dst_dir/sam_vit_h_4b8939.pth" \
    "Official SAM dependency for DEVA"
  direct_or_block "DEVA SAM-HQ ViT-H" \
    "${DEVA_SAM_HQ_H_URL:-https://huggingface.co/lkeab/hq-sam/resolve/main/sam_hq_vit_h.pth?download=true}" \
    "$dst_dir/sam_hq_vit_h.pth" \
    "Official SAM-HQ dependency for DEVA"
  direct_or_block "DEVA SAM-HQ ViT-Tiny" \
    "${DEVA_SAM_HQ_TINY_URL:-https://huggingface.co/lkeab/hq-sam/resolve/main/sam_hq_vit_tiny.pth?download=true}" \
    "$dst_dir/sam_hq_vit_tiny.pth" \
    "Official SAM-HQ dependency for DEVA"
  direct_or_block "DEVA MobileSAM" \
    "${DEVA_MOBILE_SAM_URL:-https://github.com/hkchengrex/Tracking-Anything-with-DEVA/releases/download/v1.0/mobile_sam.pt}" \
    "$dst_dir/mobile_sam.pt" \
    "Official MobileSAM dependency for DEVA"
  direct_or_block "DEVA GroundingDINO config" \
    "${DEVA_GDINO_CFG_URL:-https://github.com/hkchengrex/Tracking-Anything-with-DEVA/releases/download/v1.0/GroundingDINO_SwinT_OGC.py}" \
    "$dst_dir/GroundingDINO_SwinT_OGC.py" \
    "Official GroundingDINO config for DEVA"
}

download_cutie() {
  local dst_dir="$STWM_ROOT/models/checkpoints/cutie"
  direct_or_block \
    "Cutie pretrained" \
    "${CUTIE_CKPT_URL:-https://github.com/hkchengrex/Cutie/releases/download/v1.0/cutie-base-mega.pth}" \
    "$dst_dir/cutie-base-mega.pth" \
    "Official Cutie checkpoint"
  direct_or_block \
    "Cutie interactive checkpoint" \
    "${CUTIE_INTERACTIVE_URL:-https://github.com/hkchengrex/Cutie/releases/download/v1.0/coco_lvis_h18_itermask.pth}" \
    "$dst_dir/coco_lvis_h18_itermask.pth" \
    "Official interactive checkpoint packaged by Cutie"
}

download_yolo_world() {
  local dst_dir="$STWM_ROOT/models/checkpoints/yolo_world"
  direct_or_block \
    "YOLO-World small" \
    "${YOLO_WORLD_S_URL:-https://huggingface.co/wondervictor/YOLO-World-V2.1/resolve/main/s_stage2-4466ab94.pth}" \
    "$dst_dir/yolo_world_s_stage2.pth" \
    "Official YOLO-World V2.1 small checkpoint"
  direct_or_block \
    "YOLO-World medium" \
    "${YOLO_WORLD_M_URL:-https://huggingface.co/wondervictor/YOLO-World-V2.1/resolve/main/m_stage2-9987dcb1.pth}" \
    "$dst_dir/yolo_world_m_stage2.pth" \
    "Official YOLO-World V2.1 medium checkpoint"
  direct_or_block \
    "YOLO-World large" \
    "${YOLO_WORLD_L_URL:-https://huggingface.co/wondervictor/YOLO-World-V2.1/resolve/main/l_stage2-b3e3dc3f.pth}" \
    "$dst_dir/yolo_world_l_stage2.pth" \
    "Official YOLO-World V2.1 large checkpoint"
}

TARGET="${1:---all}"

case "$TARGET" in
  --all|all)
    download_traceanything
    download_sam2
    download_deva
    download_cutie
    download_yolo_world
    ;;
  traceanything)
    download_traceanything
    ;;
  sam2)
    download_sam2
    ;;
  deva)
    download_deva
    ;;
  cutie)
    download_cutie
    ;;
  yolo-world|yolo_world)
    download_yolo_world
    ;;
  *)
    echo "Unknown target: $TARGET"
    exit 1
    ;;
esac
