#!/usr/bin/env python3
from __future__ import annotations

from argparse import ArgumentParser
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple
import json
import math

import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils


def _repo_root() -> Path:
    for candidate in [
        Path("/raid/chen034/workspace/stwm"),
        Path("/home/chen034/workspace/stwm"),
    ]:
        if candidate.exists():
            return candidate
    raise RuntimeError("unable to resolve STWM repo root")


ROOT = _repo_root()
OBS_LEN = 8
FUT_LEN = 8
TOTAL_STEPS = OBS_LEN + FUT_LEN
QUERY_STEP = 0
FUTURE_STEP = TOTAL_STEPS - 1
PANEL_TARGETS = {
    "occlusion_reappearance": 18,
    "crossing_ambiguity": 18,
    "small_object": 18,
    "appearance_change": 18,
    "long_gap_persistence": 18,
}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"json not found: {p}")
    payload = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected dict json payload: {p}")
    return payload


def write_json(path: str | Path, payload: Dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def write_md(path: str | Path, lines: List[str]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _norm_name(name: str) -> str:
    return str(name).strip().upper()


def _read_split_ids(path: str | Path) -> List[str]:
    out: List[str] = []
    for raw in Path(path).read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if line and not line.startswith("._"):
            out.append(line)
    return out


def _list_visible_files(root: Path, suffixes: Iterable[str]) -> List[Path]:
    allowed = {str(x).lower() for x in suffixes}
    if not root.exists():
        return []
    return sorted(
        [
            p
            for p in root.iterdir()
            if p.is_file() and not p.name.startswith("._") and p.suffix.lower() in allowed
        ]
    )


def _temporal_indices(frame_count: int, total_steps: int = TOTAL_STEPS) -> List[int]:
    if frame_count <= 0:
        return [0 for _ in range(total_steps)]
    idx = np.linspace(0, frame_count - 1, num=total_steps, dtype=np.int64)
    return [int(x) for x in idx.tolist()]


def _vipseg_mask(path: Path) -> np.ndarray:
    arr = np.asarray(Image.open(path))
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr.astype(np.int32)


def _vipseg_semantic_id(label_id: int) -> int:
    if int(label_id) < 125:
        return int(label_id) - 1
    return int(label_id) // 100 - 1


def _vipseg_is_candidate_instance(label_id: int) -> bool:
    return int(label_id) >= 125


def _burst_rle_to_mask(rle_counts: str, height: int, width: int) -> np.ndarray:
    decoded = mask_utils.decode({"size": [int(height), int(width)], "counts": rle_counts.encode("utf-8")})
    if decoded.ndim == 3:
        decoded = decoded[..., 0]
    return decoded.astype(bool)


def _mask_box(mask: np.ndarray) -> List[float]:
    ys, xs = np.where(mask)
    return [float(xs.min()), float(ys.min()), float(xs.max()) + 1.0, float(ys.max()) + 1.0]


def _mask_centroid(mask: np.ndarray) -> Tuple[float, float]:
    ys, xs = np.where(mask)
    return float(xs.mean()), float(ys.mean())


def _mask_area_ratio(mask: np.ndarray) -> float:
    return float(mask.mean())


def _masked_rgb_mean(frame_path: Path, mask: np.ndarray) -> List[float]:
    rgb = np.asarray(Image.open(frame_path).convert("RGB"), dtype=np.float32) / 255.0
    pix = rgb[mask]
    if pix.size == 0:
        return [0.0, 0.0, 0.0]
    return [float(x) for x in np.mean(pix, axis=0).tolist()]


def _max_missing_span(presence: List[bool]) -> int:
    spans: List[int] = []
    current = 0
    started = False
    ever_after = False
    for idx, flag in enumerate(presence):
        if flag:
            if current > 0 and started:
                spans.append(current)
            started = True
            current = 0
            ever_after = idx < len(presence) - 1
        elif started and ever_after:
            current += 1
    return max(spans) if spans else 0


def _same_category_min_distance(
    target_center: Tuple[float, float],
    distractor_masks: List[np.ndarray],
    width: int,
    height: int,
) -> float:
    if not distractor_masks:
        return 1e9
    diag = max(math.sqrt(float(width * width + height * height)), 1.0)
    best = 1e9
    tx, ty = target_center
    for mask in distractor_masks:
        if not np.any(mask):
            continue
        cx, cy = _mask_centroid(mask)
        best = min(best, math.sqrt((cx - tx) ** 2 + (cy - ty) ** 2) / diag)
    return float(best)


def _presence_signature(present: List[bool]) -> str:
    return "".join("1" if x else "0" for x in present)


def _difficulty_score(
    *,
    gap: int,
    ambiguity_distance: float,
    small_object: bool,
    appearance_change: bool,
    long_gap: bool,
) -> float:
    ambiguity_strength = 0.0 if ambiguity_distance >= 1e8 else max(0.0, 1.0 - float(ambiguity_distance) / 0.20)
    return float(
        0.30 * min(float(gap) / 4.0, 1.0)
        + 0.30 * ambiguity_strength
        + 0.15 * float(bool(small_object))
        + 0.15 * float(bool(appearance_change))
        + 0.10 * float(bool(long_gap))
    )


def _vipseg_candidates_for_clip(clip_id: str, frame_paths: List[Path], mask_paths: List[Path]) -> List[Dict[str, Any]]:
    if len(frame_paths) < TOTAL_STEPS or len(mask_paths) < TOTAL_STEPS:
        return []
    indices = _temporal_indices(len(frame_paths), TOTAL_STEPS)
    sel_frames = [frame_paths[i] for i in indices]
    sel_masks = [mask_paths[i] for i in indices]
    query_mask = _vipseg_mask(sel_masks[QUERY_STEP])
    future_mask = _vipseg_mask(sel_masks[FUTURE_STEP])
    height, width = int(query_mask.shape[0]), int(query_mask.shape[1])

    query_ids = [int(x) for x in np.unique(query_mask).tolist() if _vipseg_is_candidate_instance(int(x))]
    future_ids = {int(x) for x in np.unique(future_mask).tolist() if _vipseg_is_candidate_instance(int(x))}
    candidates: List[Dict[str, Any]] = []
    for target_id in query_ids:
        if int(target_id) not in future_ids:
            continue
        present: List[bool] = []
        area_ratios: List[float] = []
        last_present_mask = None
        for mask_path in sel_masks:
            arr = _vipseg_mask(mask_path)
            current = arr == int(target_id)
            is_present = bool(np.any(current))
            present.append(is_present)
            if is_present:
                area_ratios.append(_mask_area_ratio(current))
                last_present_mask = current
        if not present[QUERY_STEP] or not present[FUTURE_STEP]:
            continue
        if last_present_mask is None:
            continue
        qmask = query_mask == int(target_id)
        fmask = future_mask == int(target_id)
        if not np.any(qmask) or not np.any(fmask):
            continue
        qcenter = _mask_centroid(qmask)
        fcenter = _mask_centroid(fmask)
        qbox = _mask_box(qmask)
        semantic_id = int(_vipseg_semantic_id(target_id))
        future_same_cat_masks: List[np.ndarray] = []
        for cand in future_ids:
            if int(cand) == int(target_id):
                continue
            if int(_vipseg_semantic_id(cand)) != semantic_id:
                continue
            cmask = future_mask == int(cand)
            if np.any(cmask):
                future_same_cat_masks.append(cmask)
        ambiguity_dist = _same_category_min_distance(fcenter, future_same_cat_masks, width=width, height=height)
        gap = _max_missing_span(present)
        small_object = bool(area_ratios and float(sum(area_ratios) / len(area_ratios)) < 0.02)
        qrgb = _masked_rgb_mean(sel_frames[QUERY_STEP], qmask)
        frgb = _masked_rgb_mean(sel_frames[FUTURE_STEP], fmask)
        rgb_shift = float(np.linalg.norm(np.asarray(qrgb) - np.asarray(frgb)))
        qarea = _mask_area_ratio(qmask)
        farea = _mask_area_ratio(fmask)
        appearance_change = bool(rgb_shift >= 0.18 or abs(farea - qarea) >= 0.03)
        subset_tags: List[str] = []
        if gap >= 2:
            subset_tags.append("occlusion_reappearance")
        if ambiguity_dist < 0.15:
            subset_tags.append("crossing_ambiguity")
        if small_object:
            subset_tags.append("small_object")
        if appearance_change:
            subset_tags.append("appearance_change")
        score = _difficulty_score(
            gap=gap,
            ambiguity_distance=ambiguity_dist,
            small_object=small_object,
            appearance_change=appearance_change,
            long_gap=False,
        )
        candidates.append(
            {
                "protocol_item_id": f"vipseg::{clip_id}::{target_id}",
                "dataset": "VIPSeg",
                "clip_id": str(clip_id),
                "target_id": int(target_id),
                "category_id": int(semantic_id),
                "query_step": int(QUERY_STEP),
                "future_step": int(FUTURE_STEP),
                "selected_frame_indices": indices,
                "selected_frame_paths": [str(p) for p in sel_frames],
                "selected_mask_paths": [str(p) for p in sel_masks],
                "image_size": {"width": int(width), "height": int(height)},
                "query_box_xyxy": [float(x) for x in qbox],
                "subset_tags": subset_tags,
                "stats": {
                    "presence_signature": _presence_signature(present),
                    "max_missing_span": int(gap),
                    "same_category_distractor_count_future": int(len(future_same_cat_masks)),
                    "same_category_min_distance_future": float(ambiguity_dist if ambiguity_dist < 1e8 else 1.0),
                    "mean_area_ratio": float(sum(area_ratios) / max(len(area_ratios), 1)),
                    "query_area_ratio": float(qarea),
                    "future_area_ratio": float(farea),
                    "query_rgb_mean": qrgb,
                    "future_rgb_mean": frgb,
                    "appearance_rgb_shift_l2": float(rgb_shift),
                    "difficulty_score": float(score),
                },
                "annotation_source": "VIPSeg original panomasks with instance continuity from label ids >= 125",
            }
        )
    return candidates


def _burst_candidates(
    seq: Dict[str, Any],
    annotation_file: Path,
    frames_root: Path,
) -> List[Dict[str, Any]]:
    dataset_name = str(seq.get("dataset", ""))
    seq_name = str(seq.get("seq_name", ""))
    annotated_paths = [str(x) for x in seq.get("annotated_image_paths", []) if str(x)]
    segmentations = seq.get("segmentations", []) if isinstance(seq.get("segmentations", []), list) else []
    track_category_ids = seq.get("track_category_ids", {}) if isinstance(seq.get("track_category_ids", {}), dict) else {}
    if len(annotated_paths) < TOTAL_STEPS or len(segmentations) < TOTAL_STEPS:
        return []
    seq_root = frames_root / dataset_name / seq_name
    if not seq_root.exists():
        return []
    indices = _temporal_indices(len(annotated_paths), TOTAL_STEPS)
    sel_frame_names = [annotated_paths[i] for i in indices]
    sel_frames = [seq_root / name for name in sel_frame_names]
    if any(not p.exists() for p in sel_frames):
        return []
    sel_segs = [segmentations[i] if isinstance(segmentations[i], dict) else {} for i in indices]
    height = int(seq.get("height", 0) or 0)
    width = int(seq.get("width", 0) or 0)
    if height <= 0 or width <= 0:
        return []
    query_seg = sel_segs[QUERY_STEP]
    future_seg = sel_segs[FUTURE_STEP]
    query_ids = {str(k) for k, v in query_seg.items() if isinstance(v, dict) and str(v.get("rle", ""))}
    future_ids = {str(k) for k, v in future_seg.items() if isinstance(v, dict) and str(v.get("rle", ""))}
    candidates: List[Dict[str, Any]] = []
    for target_id in sorted(query_ids & future_ids):
        present = [str(target_id) in step and isinstance(step.get(str(target_id), {}), dict) for step in sel_segs]
        if not present[QUERY_STEP] or not present[FUTURE_STEP]:
            continue
        q_rle = str(query_seg[str(target_id)].get("rle", ""))
        f_rle = str(future_seg[str(target_id)].get("rle", ""))
        if not q_rle or not f_rle:
            continue
        qmask = _burst_rle_to_mask(q_rle, height=height, width=width)
        fmask = _burst_rle_to_mask(f_rle, height=height, width=width)
        if not np.any(qmask) or not np.any(fmask):
            continue
        qcenter = _mask_centroid(qmask)
        fcenter = _mask_centroid(fmask)
        qbox = _mask_box(qmask)
        semantic_id = int(track_category_ids.get(str(target_id), -1))
        same_cat_masks: List[np.ndarray] = []
        for cand_id, payload in future_seg.items():
            if str(cand_id) == str(target_id) or not isinstance(payload, dict):
                continue
            if int(track_category_ids.get(str(cand_id), -9999)) != semantic_id:
                continue
            rle = str(payload.get("rle", ""))
            if not rle:
                continue
            cmask = _burst_rle_to_mask(rle, height=height, width=width)
            if np.any(cmask):
                same_cat_masks.append(cmask)
        ambiguity_dist = _same_category_min_distance(fcenter, same_cat_masks, width=width, height=height)
        gap = _max_missing_span(present)
        qarea = _mask_area_ratio(qmask)
        farea = _mask_area_ratio(fmask)
        qrgb = _masked_rgb_mean(sel_frames[QUERY_STEP], qmask)
        frgb = _masked_rgb_mean(sel_frames[FUTURE_STEP], fmask)
        rgb_shift = float(np.linalg.norm(np.asarray(qrgb) - np.asarray(frgb)))
        small_object = bool(((qarea + farea) * 0.5) < 0.02)
        appearance_change = bool(rgb_shift >= 0.18 or abs(farea - qarea) >= 0.03)
        subset_tags: List[str] = []
        if gap >= 2:
            subset_tags.append("occlusion_reappearance")
        if gap >= 4:
            subset_tags.append("long_gap_persistence")
        if ambiguity_dist < 0.15:
            subset_tags.append("crossing_ambiguity")
        if small_object:
            subset_tags.append("small_object")
        if appearance_change:
            subset_tags.append("appearance_change")
        score = _difficulty_score(
            gap=gap,
            ambiguity_distance=ambiguity_dist,
            small_object=small_object,
            appearance_change=appearance_change,
            long_gap=gap >= 4,
        )
        candidates.append(
            {
                "protocol_item_id": f"burst::{dataset_name}::{seq_name}::{target_id}",
                "dataset": "BURST",
                "clip_id": f"{dataset_name}/{seq_name}",
                "target_id": str(target_id),
                "category_id": int(semantic_id),
                "query_step": int(QUERY_STEP),
                "future_step": int(FUTURE_STEP),
                "selected_frame_indices": indices,
                "selected_frame_paths": [str(p) for p in sel_frames],
                "selected_frame_names": sel_frame_names,
                "burst_annotation_file": str(annotation_file),
                "burst_dataset_name": str(dataset_name),
                "burst_seq_name": str(seq_name),
                "image_size": {"width": int(width), "height": int(height)},
                "query_box_xyxy": [float(x) for x in qbox],
                "subset_tags": subset_tags,
                "stats": {
                    "presence_signature": _presence_signature(present),
                    "max_missing_span": int(gap),
                    "same_category_distractor_count_future": int(len(same_cat_masks)),
                    "same_category_min_distance_future": float(ambiguity_dist if ambiguity_dist < 1e8 else 1.0),
                    "mean_area_ratio": float((qarea + farea) * 0.5),
                    "query_area_ratio": float(qarea),
                    "future_area_ratio": float(farea),
                    "query_rgb_mean": qrgb,
                    "future_rgb_mean": frgb,
                    "appearance_rgb_shift_l2": float(rgb_shift),
                    "difficulty_score": float(score),
                },
                "annotation_source": f"BURST {annotation_file.name} with track continuity from track ids",
            }
        )
    return candidates


def _select_candidates(all_candidates: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, List[str]]]:
    ranked = sorted(
        all_candidates,
        key=lambda item: (
            -float(((item.get("stats") or {}).get("difficulty_score", 0.0))),
            str(item.get("protocol_item_id", "")),
        ),
    )
    panel_members: Dict[str, List[str]] = {key: [] for key in PANEL_TARGETS}
    selected: List[Dict[str, Any]] = []
    selected_ids = set()
    for item in ranked:
        tags = [str(x) for x in item.get("subset_tags", []) if str(x) in PANEL_TARGETS]
        if not tags:
            continue
        useful = [tag for tag in tags if len(panel_members[tag]) < int(PANEL_TARGETS[tag])]
        if not useful:
            continue
        item_id = str(item.get("protocol_item_id", ""))
        if item_id not in selected_ids:
            selected.append(item)
            selected_ids.add(item_id)
        for tag in useful:
            panel_members[tag].append(item_id)
        if all(len(panel_members[tag]) >= int(PANEL_TARGETS[tag]) for tag in PANEL_TARGETS):
            break
    return selected, panel_members


def parse_args() -> Any:
    parser = ArgumentParser(description="Build real Stage2 state-identifiability / future grounding protocol")
    parser.add_argument("--contract-json", default=str(ROOT / "reports/stage2_bootstrap_data_contract_20260408.json"))
    parser.add_argument("--output-json", default=str(ROOT / "reports/stage2_state_identifiability_protocol_20260415.json"))
    parser.add_argument("--output-md", default=str(ROOT / "docs/STAGE2_STATE_IDENTIFIABILITY_PROTOCOL_20260415.md"))
    parser.add_argument("--vipseg-max-clips", type=int, default=220)
    parser.add_argument("--burst-max-seqs", type=int, default=220)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    contract = read_json(args.contract_json)
    ds_map = {
        _norm_name(str(rec.get("dataset_name", ""))): rec
        for rec in contract.get("datasets", [])
        if isinstance(rec, dict)
    }
    vipseg = ds_map["VIPSEG"]
    burst = ds_map["BURST"]

    vipseg_split = Path(vipseg["split_mapping"]["val"]["split_file"])
    vipseg_frame_root = Path(vipseg["split_mapping"]["val"]["frame_root"])
    vipseg_mask_root = Path(vipseg["split_mapping"]["val"]["mask_root"])
    vipseg_ids = _read_split_ids(vipseg_split)[: max(int(args.vipseg_max_clips), 1)]

    all_candidates: List[Dict[str, Any]] = []
    vipseg_candidate_count = 0
    for clip_id in vipseg_ids:
        frame_paths = _list_visible_files(vipseg_frame_root / clip_id, [".jpg", ".jpeg", ".png"])
        mask_paths = _list_visible_files(vipseg_mask_root / clip_id, [".png"])
        if len(frame_paths) < TOTAL_STEPS or len(mask_paths) < TOTAL_STEPS:
            continue
        candidates = _vipseg_candidates_for_clip(clip_id=clip_id, frame_paths=frame_paths, mask_paths=mask_paths)
        vipseg_candidate_count += len(candidates)
        all_candidates.extend(candidates)

    burst_cfg = burst["split_mapping"]["val"]
    burst_annotation_file = Path(burst_cfg["annotation_file"])
    burst_frames_root = Path(burst_cfg["frames_root"])
    burst_payload = read_json(burst_annotation_file)
    burst_sequences = burst_payload.get("sequences", []) if isinstance(burst_payload.get("sequences", []), list) else []
    burst_candidate_count = 0
    for seq in burst_sequences[: max(int(args.burst_max_seqs), 1)]:
        if not isinstance(seq, dict):
            continue
        candidates = _burst_candidates(seq=seq, annotation_file=burst_annotation_file, frames_root=burst_frames_root)
        burst_candidate_count += len(candidates)
        all_candidates.extend(candidates)

    selected_items, panel_members = _select_candidates(all_candidates)
    selected_ids = {str(item.get("protocol_item_id", "")) for item in selected_items}
    panel_counts = {
        "full_identifiability_panel": int(len(selected_items)),
        "occlusion_reappearance": int(len(panel_members["occlusion_reappearance"])),
        "crossing_ambiguity": int(len(panel_members["crossing_ambiguity"])),
        "small_object": int(len(panel_members["small_object"])),
        "appearance_change": int(len(panel_members["appearance_change"])),
        "long_gap_persistence": int(len(panel_members["long_gap_persistence"])),
    }
    payload = {
        "generated_at_utc": now_iso(),
        "protocol_name": "Stage2 real state-identifiability / future grounding protocol",
        "protocol_version": "20260415",
        "task_definition": {
            "query": "given a historical object / mask query, recover the same target from future state / future candidate masks",
            "query_frame_index_convention": int(QUERY_STEP),
            "future_target_frame_index_convention": int(FUTURE_STEP),
            "uses_real_instance_identity": True,
            "uses_real_future_masks": True,
            "rollout_error_proxy_only": False,
        },
        "dataset_policy": {
            "VIPSeg": "primary real instance-identifiability panel",
            "BURST": "primary long-gap persistence / continuity panel",
            "VSPW": "not part of main protocol; can remain supplemental scene/stuff-only context",
        },
        "panel_targets": {k: int(v) for k, v in PANEL_TARGETS.items()},
        "panel_counts": panel_counts,
        "scan_stats": {
            "vipseg_val_clip_count_scanned": int(len(vipseg_ids)),
            "vipseg_candidate_count": int(vipseg_candidate_count),
            "burst_val_sequence_count_scanned": int(min(len(burst_sequences), int(args.burst_max_seqs))),
            "burst_candidate_count": int(burst_candidate_count),
            "selected_protocol_item_count": int(len(selected_items)),
        },
        "panel_members": panel_members,
        "selected_protocol_item_ids": sorted(selected_ids),
        "items": selected_items,
    }
    write_json(args.output_json, payload)
    write_md(
        args.output_md,
        [
            "# Stage2 State-Identifiability Protocol 20260415",
            "",
            "- task: real future grounding with true instance continuity and future masks",
            "- primary datasets: VIPSeg, BURST",
            "- VSPW: supplemental only; excluded from main identifiability protocol",
            f"- full_identifiability_panel: {panel_counts['full_identifiability_panel']}",
            f"- occlusion_reappearance: {panel_counts['occlusion_reappearance']}",
            f"- crossing_ambiguity: {panel_counts['crossing_ambiguity']}",
            f"- small_object: {panel_counts['small_object']}",
            f"- appearance_change: {panel_counts['appearance_change']}",
            f"- long_gap_persistence: {panel_counts['long_gap_persistence']}",
            "",
            "## Notes",
            "",
            "- Query frame uses the same semantic-frame convention as the current Stage2 mainline.",
            "- Future recovery uses true instance identity / future mask continuity rather than rollout-L2 proxy scores.",
        ],
    )
    print(json.dumps(payload, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
