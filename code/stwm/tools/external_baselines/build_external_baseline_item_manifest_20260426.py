from __future__ import annotations

import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from stwm.tools.external_baselines.common_io import DOCS, REPORTS, ROOT, sha256_json, write_json, write_markdown  # noqa: E402


KEYWORDS = [
    "frame_path",
    "frame_paths",
    "image_path",
    "video_path",
    "mask_path",
    "mask_rle",
    "bbox",
    "candidate",
    "candidate_masks",
    "future_frame",
    "target_mask",
    "target_box",
    "predecode",
    "cache",
    "protocol_item_id",
    "clip_id",
    "video_id",
]

PROTOCOL_SOURCES = [
    ROOT / "reports" / "stage2_state_identifiability_protocol_v3_20260416.json",
    ROOT / "reports" / "stage2_protocol_v3_extended_evalset_20260420.json",
]

EVAL_SOURCES = [
    ROOT / "reports" / "stwm_belief_final_eval_20260424.json",
    ROOT / "reports" / "stwm_belief_true_ood_eval_20260424.json",
]

MASK_OUTPUT_ROOT = ROOT / "baselines" / "outputs" / "manifest_payload_masks"


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def safe_rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except Exception:
        return str(path)


def discover_source_files() -> dict[str, Any]:
    roots = [
        ROOT / "reports",
        ROOT / "docs",
        ROOT / "data",
        ROOT / "outputs",
        ROOT / "cache",
        ROOT / "code" / "stwm" / "tools",
        ROOT / "code" / "stwm" / "tracewm_v2_stage2",
        ROOT / "manifests",
    ]
    suffixes = {".json", ".jsonl", ".md", ".py", ".npz", ".yaml", ".yml", ".txt"}
    entries = []
    for root in roots:
        if not root.exists():
            entries.append(
                {
                    "source_file": str(root),
                    "exists": False,
                    "fields_found": [],
                    "contains_frame_paths": False,
                    "contains_observed_prompt": False,
                    "contains_future_candidates": False,
                    "contains_masks_or_boxes": False,
                    "usable_for_manifest": False,
                    "exact_error": "root_missing",
                }
            )
            continue
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in suffixes:
                continue
            if "/baselines/repos/" in str(path):
                continue
            text = ""
            fields = set()
            exact_error = None
            try:
                if path.suffix.lower() == ".npz":
                    with np.load(path, allow_pickle=True) as z:
                        fields.update(z.files)
                        text = " ".join(z.files)
                else:
                    size = path.stat().st_size
                    if size > 150_000_000:
                        exact_error = f"file_too_large_for_full_text_scan:{size}"
                        text = path.read_text(errors="ignore")[:2_000_000]
                    else:
                        text = path.read_text(errors="ignore")
                    if path.suffix.lower() in {".json", ".jsonl"}:
                        fields.update(re.findall(r'"([^"]+)"\s*:', text[:5_000_000]))
            except Exception as exc:
                exact_error = f"{type(exc).__name__}:{exc}"
            hits = sorted({kw for kw in KEYWORDS if kw in text or kw in fields})
            contains_frame = any(x in hits for x in ["frame_path", "frame_paths", "image_path", "video_path"])
            contains_prompt = any(x in hits for x in ["target_mask", "target_box", "bbox", "mask_path", "frame_paths"])
            contains_candidates = any(x in hits for x in ["candidate", "candidate_masks", "future_frame"])
            contains_masks = any(x in hits for x in ["mask_path", "mask_rle", "bbox", "target_box"])
            usable = contains_frame and contains_prompt and contains_candidates and contains_masks
            if hits or usable or path.name in {
                "stage2_state_identifiability_protocol_v3_20260416.json",
                "stage2_protocol_v3_extended_evalset_20260420.json",
            }:
                entries.append(
                    {
                        "source_file": safe_rel(path),
                        "exists": True,
                        "fields_found": hits[:80],
                        "contains_frame_paths": contains_frame,
                        "contains_observed_prompt": contains_prompt,
                        "contains_future_candidates": contains_candidates,
                        "contains_masks_or_boxes": contains_masks,
                        "usable_for_manifest": usable,
                        "size_bytes": path.stat().st_size,
                        "exact_error": exact_error,
                    }
                )
    summary = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "keywords": KEYWORDS,
        "source_count": len(entries),
        "usable_source_count": sum(1 for x in entries if x["usable_for_manifest"]),
        "sources": sorted(entries, key=lambda x: (not x["usable_for_manifest"], x["source_file"])),
    }
    return summary


def data_root_audit(discovery: dict[str, Any]) -> dict[str, Any]:
    absolute_paths = set()
    for source in discovery.get("sources", []):
        p = ROOT / source["source_file"]
        if p.exists() and p.suffix.lower() in {".json", ".jsonl", ".md", ".py", ".txt", ".yaml", ".yml"}:
            try:
                text = p.read_text(errors="ignore")[:8_000_000]
            except Exception:
                continue
            absolute_paths.update(re.findall(r"/(?:home|raid)/chen034/[A-Za-z0-9_./\\-]+", text))
    candidates = {
        ROOT,
        Path("/raid/chen034/workspace/stwm"),
        Path("/home/chen034/workspace/data"),
        Path("/raid/chen034/workspace/data"),
    }
    for raw in list(absolute_paths)[:5000]:
        path = Path(raw)
        for parent in [path, *path.parents[:6]]:
            if parent.name in {"stwm", "data", "vipseg", "burst", "VIPSeg", "images", "annotations", "processed", "cache"}:
                candidates.add(parent)
    existing = sorted(str(p) for p in candidates if p.exists())
    missing = sorted(str(p) for p in candidates if not p.exists())
    frame_dirs = []
    mask_dirs = []
    cache_dirs = []
    for base in [ROOT / "data", Path("/home/chen034/workspace/data"), Path("/raid/chen034/workspace/data")]:
        if not base.exists():
            continue
        for d in base.rglob("*"):
            if not d.is_dir():
                continue
            name = d.name.lower()
            if name in {"imgs", "frames"} or "frames" in str(d).lower():
                if any(d.glob("*.jpg")) or any(d.glob("*.png")) or any(d.rglob("*.jpg")):
                    frame_dirs.append(str(d))
            if "mask" in name or "panomask" in str(d).lower() or "annotation" in str(d).lower():
                if any(d.glob("*.png")) or any(d.glob("*.json")) or any(d.rglob("*.png")):
                    mask_dirs.append(str(d))
            if "cache" in str(d).lower() or "processed" in str(d).lower():
                cache_dirs.append(str(d))
    return {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "candidate_data_roots": sorted(str(p) for p in candidates),
        "existing_roots": existing,
        "frame_dir_candidates": sorted(frame_dirs)[:200],
        "mask_dir_candidates": sorted(mask_dirs)[:200],
        "cache_dir_candidates": sorted(cache_dirs)[:200],
        "missing_roots": missing,
        "exact_blocking_reason_if_none_found": None if existing and frame_dirs else "no_existing_data_roots_or_frame_dirs_found",
    }


def manifest_schema() -> dict[str, Any]:
    return {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "schema_name": "stwm_external_baseline_item_manifest_20260426",
        "item_fields": {
            "item_id": "string",
            "protocol_item_id": "string",
            "source_protocol": "string or list[string]",
            "dataset": "string",
            "video_id": "string",
            "clip_id": "string",
            "frame_paths": "list[string]",
            "observed_frame_indices": "list[int]",
            "observed_prompt_frame_index": "int",
            "future_frame_index": "int",
            "observed_target": {
                "target_id": "string",
                "bbox": "[x1,y1,x2,y2] or null",
                "mask_path": "string or null",
                "mask_rle": "COCO RLE dict or null",
                "point_prompt": "[x,y] or null",
            },
            "future_candidates": [
                {
                    "candidate_id": "string",
                    "bbox": "[x1,y1,x2,y2] or null",
                    "mask_path": "string or null",
                    "mask_rle": "COCO RLE dict or null",
                }
            ],
            "gt_candidate_id": "string",
            "subset_tags": "dict[str,bool]",
            "leakage_policy": {
                "gt_candidate_id_used_only_for_eval": True,
                "future_masks_used_only_for_candidate_matching": True,
                "observed_prompt_does_not_use_future_gt": True,
            },
            "readiness": {
                "cutie_ready": "bool",
                "sam2_ready": "bool",
                "cotracker_ready": "bool",
                "missing_reasons": "list[string]",
            },
        },
        "null_policy": "If a field cannot be filled, it remains null and a missing_reason is recorded; fake paths are forbidden.",
    }


def _rle_from_bool(mask: np.ndarray) -> dict[str, Any]:
    arr = np.asfortranarray(mask.astype(np.uint8))
    rle = mask_utils.encode(arr)
    counts = rle["counts"]
    if isinstance(counts, bytes):
        counts = counts.decode("ascii")
    return {"size": [int(mask.shape[0]), int(mask.shape[1])], "counts": counts}


def _decode_rle(rle: str | dict[str, Any], height: int, width: int) -> np.ndarray:
    if isinstance(rle, str):
        rle_obj = {"size": [int(height), int(width)], "counts": rle.encode("ascii")}
    else:
        counts = rle["counts"].encode("ascii") if isinstance(rle.get("counts"), str) else rle["counts"]
        rle_obj = {"size": rle.get("size", [int(height), int(width)]), "counts": counts}
    return mask_utils.decode(rle_obj).astype(bool)


def _bbox_from_mask(mask: np.ndarray) -> list[float] | None:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]


def _point_from_bbox(bbox: list[float] | None) -> list[float] | None:
    if not bbox:
        return None
    return [float((bbox[0] + bbox[2]) / 2.0), float((bbox[1] + bbox[3]) / 2.0)]


def load_protocol_items() -> dict[str, dict[str, Any]]:
    items: dict[str, dict[str, Any]] = {}
    source_names: dict[str, list[str]] = defaultdict(list)
    for path in PROTOCOL_SOURCES:
        if not path.exists():
            continue
        data = load_json(path)
        for item in data.get("items", []):
            pid = str(item.get("protocol_item_id"))
            if not pid:
                continue
            current = items.get(pid)
            if current is None or path.name.endswith("extended_evalset_20260420.json"):
                items[pid] = dict(item)
            source_names[pid].append(safe_rel(path))
    for pid, item in items.items():
        item["_source_protocols"] = sorted(set(source_names[pid]))
    return items


def load_requested_item_ids() -> dict[str, set[str]]:
    requested: dict[str, set[str]] = defaultdict(set)
    for path in EVAL_SOURCES:
        if not path.exists():
            continue
        data = load_json(path)
        if path.name == "stwm_belief_final_eval_20260424.json":
            for panel_name, panel in data.get("panels", {}).items():
                for row in panel.get("per_item_results", []):
                    if row.get("method_name", "").startswith("TUSB-v3.1::official"):
                        requested[panel_name].add(str(row["protocol_item_id"]))
                    tags = set(row.get("subset_tags") or [])
                    if row.get("method_name", "").startswith("TUSB-v3.1::official") and ({"occlusion_reappearance", "long_gap_persistence"} & tags):
                        requested["reacquisition_v2"].add(str(row["protocol_item_id"]))
        elif path.name == "stwm_belief_true_ood_eval_20260424.json":
            for split_name, split in data.get("splits", {}).items():
                panel = split.get("panel", {})
                for row in panel.get("per_item_results", []):
                    if row.get("method_name", "").startswith("TUSB-v3.1::official"):
                        requested[split_name].add(str(row["protocol_item_id"]))
                        tags = set(row.get("subset_tags") or [])
                        if {"occlusion_reappearance", "long_gap_persistence"} & tags:
                            requested["reacquisition_v2"].add(str(row["protocol_item_id"]))
    return requested


def _load_burst_sequence_cache(protocol_items: dict[str, dict[str, Any]]) -> dict[tuple[str, str, str], dict[str, Any]]:
    needed_files = sorted({item.get("burst_annotation_file") for item in protocol_items.values() if item.get("dataset") == "BURST" and item.get("burst_annotation_file")})
    cache: dict[tuple[str, str, str], dict[str, Any]] = {}
    loaded: dict[str, Any] = {}
    for file in needed_files:
        if file not in loaded:
            loaded[file] = load_json(Path(file))
        for seq in loaded[file].get("sequences", []):
            cache[(file, str(seq.get("dataset")), str(seq.get("seq_name")))] = seq
    return cache


def _materialize_vipseg_item(item: dict[str, Any]) -> tuple[dict[str, Any] | None, list[str]]:
    missing: list[str] = []
    frame_paths = [str(x) for x in item.get("selected_frame_paths", [])]
    mask_paths = [str(x) for x in item.get("selected_mask_paths", [])]
    query_step = int(item.get("query_step", 0))
    future_step = int(item.get("future_step", len(frame_paths) - 1))
    target_id = str(item.get("target_id"))
    if not frame_paths:
        missing.append("missing_selected_frame_paths")
    if not mask_paths:
        missing.append("missing_selected_mask_paths")
    if future_step >= len(frame_paths):
        missing.append("future_step_out_of_range")
    if query_step >= len(frame_paths):
        missing.append("query_step_out_of_range")
    if missing:
        return None, missing
    q_mask_path = Path(mask_paths[query_step])
    f_mask_path = Path(mask_paths[future_step])
    if not q_mask_path.exists():
        missing.append("query_mask_path_missing")
    if not f_mask_path.exists():
        missing.append("future_mask_path_missing")
    if missing:
        return None, missing
    q_arr = np.array(Image.open(q_mask_path))
    f_arr = np.array(Image.open(f_mask_path))
    tid = int(item.get("target_id"))
    q_mask = q_arr == tid
    f_labels = [int(x) for x in np.unique(f_arr) if int(x) >= 125]
    candidates = []
    for cid in f_labels:
        cm = f_arr == cid
        bbox = _bbox_from_mask(cm)
        if bbox is None:
            continue
        candidates.append(
            {
                "candidate_id": str(cid),
                "bbox": bbox,
                "mask_path": str(f_mask_path),
                "mask_rle": _rle_from_bool(cm),
                "mask_source": "vipseg_panoptic_mask_label",
            }
        )
    observed_bbox = _bbox_from_mask(q_mask) or item.get("query_box_xyxy")
    observed_rle = _rle_from_bool(q_mask) if q_mask.any() else None
    if str(tid) not in {c["candidate_id"] for c in candidates}:
        missing.append("gt_candidate_not_present_in_future_mask")
    payload = {
        "observed_target": {
            "target_id": str(tid),
            "bbox": observed_bbox,
            "mask_path": str(q_mask_path),
            "mask_rle": observed_rle,
            "point_prompt": _point_from_bbox(observed_bbox),
            "mask_source": "vipseg_panoptic_mask_label",
        },
        "future_candidates": candidates,
    }
    return payload, missing


def _materialize_burst_item(item: dict[str, Any], burst_cache: dict[tuple[str, str, str], dict[str, Any]]) -> tuple[dict[str, Any] | None, list[str]]:
    missing: list[str] = []
    frame_paths = [str(x) for x in item.get("selected_frame_paths", [])]
    frame_names = [str(x) for x in item.get("selected_frame_names", [])]
    query_step = int(item.get("query_step", 0))
    future_step = int(item.get("future_step", len(frame_paths) - 1))
    if not frame_paths or not frame_names:
        missing.append("missing_selected_frame_paths_or_names")
    if query_step >= len(frame_names) or future_step >= len(frame_names):
        missing.append("query_or_future_step_out_of_range")
    seq = burst_cache.get((str(item.get("burst_annotation_file")), str(item.get("burst_dataset_name")), str(item.get("burst_seq_name"))))
    if seq is None:
        missing.append("burst_sequence_not_found")
    if missing:
        return None, missing
    annotated = seq.get("annotated_image_paths", [])
    try:
        q_ann = annotated.index(frame_names[query_step])
        f_ann = annotated.index(frame_names[future_step])
    except ValueError:
        return None, ["selected_frame_not_in_burst_annotated_image_paths"]
    height, width = int(seq.get("height")), int(seq.get("width"))
    target_id = str(item.get("target_id"))
    q_seg = seq.get("segmentations", [])[q_ann]
    f_seg = seq.get("segmentations", [])[f_ann]
    observed_rle = None
    observed_bbox = item.get("query_box_xyxy")
    if target_id in q_seg:
        q_mask = _decode_rle(q_seg[target_id]["rle"], height, width)
        observed_rle = _rle_from_bool(q_mask)
        observed_bbox = _bbox_from_mask(q_mask) or observed_bbox
    candidates = []
    for cid, entry in sorted(f_seg.items(), key=lambda kv: str(kv[0])):
        cm = _decode_rle(entry["rle"], height, width)
        bbox = _bbox_from_mask(cm)
        if bbox is None:
            continue
        candidates.append(
            {
                "candidate_id": str(cid),
                "bbox": bbox,
                "mask_path": None,
                "mask_rle": {"size": [height, width], "counts": entry["rle"]},
                "mask_source": "burst_annotation_rle",
            }
        )
    if target_id not in {c["candidate_id"] for c in candidates}:
        missing.append("gt_candidate_not_present_in_future_segmentation")
    payload = {
        "observed_target": {
            "target_id": target_id,
            "bbox": observed_bbox,
            "mask_path": None,
            "mask_rle": observed_rle,
            "point_prompt": _point_from_bbox(observed_bbox),
            "mask_source": "burst_annotation_rle" if observed_rle else "protocol_query_box",
        },
        "future_candidates": candidates,
    }
    return payload, missing


def item_readiness(manifest_item: dict[str, Any], missing: list[str]) -> dict[str, Any]:
    frame_ok = bool(manifest_item.get("frame_paths")) and all(Path(p).exists() for p in manifest_item.get("frame_paths", []))
    observed = manifest_item.get("observed_target") or {}
    candidates = manifest_item.get("future_candidates") or []
    observed_prompt = bool(observed.get("mask_path") or observed.get("mask_rle") or observed.get("bbox") or observed.get("point_prompt"))
    observed_box_or_mask = bool(observed.get("mask_path") or observed.get("mask_rle") or observed.get("bbox"))
    candidates_ok = bool(candidates) and all(c.get("mask_path") or c.get("mask_rle") or c.get("bbox") for c in candidates)
    future_ok = manifest_item.get("future_frame_index") is not None
    gt_ok = manifest_item.get("gt_candidate_id") is not None and any(str(c["candidate_id"]) == str(manifest_item.get("gt_candidate_id")) for c in candidates)
    missing_reasons = list(missing)
    for ok, reason in [
        (frame_ok, "frame_paths_missing_or_nonexistent"),
        (observed_prompt, "observed_prompt_missing"),
        (candidates_ok, "future_candidates_missing"),
        (future_ok, "future_frame_index_missing"),
        (gt_ok, "gt_candidate_missing_from_future_candidates"),
    ]:
        if not ok and reason not in missing_reasons:
            missing_reasons.append(reason)
    return {
        "cutie_ready": bool(frame_ok and observed_box_or_mask and candidates_ok and future_ok and gt_ok),
        "sam2_ready": bool(frame_ok and observed_prompt and candidates_ok and future_ok and gt_ok),
        "cotracker_ready": bool(frame_ok and observed_box_or_mask and candidates_ok and future_ok and gt_ok),
        "missing_reasons": sorted(set(missing_reasons)),
    }


def build_manifest() -> dict[str, Any]:
    protocol_items = load_protocol_items()
    requested = load_requested_item_ids()
    requested_union = sorted(set().union(*requested.values())) if requested else sorted(protocol_items.keys())
    burst_cache = _load_burst_sequence_cache({pid: protocol_items[pid] for pid in requested_union if pid in protocol_items})
    materialized = []
    skipped = []
    per_source_counts = Counter()
    per_subset_counts = Counter()
    for pid in requested_union:
        item = protocol_items.get(pid)
        if item is None:
            skipped.append({"protocol_item_id": pid, "missing_reasons": ["protocol_item_not_found_in_v3_or_extended_sources"]})
            continue
        dataset = str(item.get("dataset"))
        if dataset == "VIPSeg":
            payload, missing = _materialize_vipseg_item(item)
        elif dataset == "BURST":
            payload, missing = _materialize_burst_item(item, burst_cache)
        else:
            payload, missing = None, [f"unsupported_dataset:{dataset}"]
        if payload is None:
            skipped.append({"protocol_item_id": pid, "missing_reasons": missing})
            continue
        frame_paths = [str(x) for x in item.get("selected_frame_paths", [])]
        query_step = int(item.get("query_step", 0))
        future_step = int(item.get("future_step", len(frame_paths) - 1))
        source_protocols = sorted([name for name, ids in requested.items() if pid in ids])
        subset_dict = {str(t): True for t in item.get("subset_tags", [])}
        manifest_item = {
            "item_id": pid,
            "protocol_item_id": pid,
            "source_protocol": source_protocols or item.get("_source_protocols", []),
            "dataset": dataset,
            "video_id": item.get("clip_id"),
            "clip_id": item.get("clip_id"),
            "frame_paths": frame_paths,
            "observed_frame_indices": [query_step],
            "observed_prompt_frame_index": query_step,
            "future_frame_index": future_step,
            "observed_target": payload["observed_target"],
            "future_candidates": payload["future_candidates"],
            "gt_candidate_id": str(item.get("target_id")),
            "subset_tags": subset_dict,
            "leakage_policy": {
                "gt_candidate_id_used_only_for_eval": True,
                "future_masks_used_only_for_candidate_matching": True,
                "observed_prompt_does_not_use_future_gt": True,
            },
            "source_protocol_files": item.get("_source_protocols", []),
            "image_size": item.get("image_size"),
        }
        manifest_item["readiness"] = item_readiness(manifest_item, missing)
        materialized.append(manifest_item)
        for src in manifest_item["source_protocol"]:
            per_source_counts[src] += 1
        for tag in subset_dict:
            per_subset_counts[tag] += 1
    skipped_reasons = Counter()
    for row in skipped:
        skipped_reasons.update(row["missing_reasons"])
    report = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "manifest_schema": "reports/stwm_external_baseline_manifest_schema_20260426.json",
        "source_protocol_files": [safe_rel(p) for p in PROTOCOL_SOURCES],
        "source_eval_files": [safe_rel(p) for p in EVAL_SOURCES],
        "total_items_considered": len(requested_union),
        "materialized_items": len(materialized),
        "cutie_ready_items": sum(1 for x in materialized if x["readiness"]["cutie_ready"]),
        "sam2_ready_items": sum(1 for x in materialized if x["readiness"]["sam2_ready"]),
        "cotracker_ready_items": sum(1 for x in materialized if x["readiness"]["cotracker_ready"]),
        "skipped_items": len(skipped),
        "skipped_reason_counts": dict(skipped_reasons),
        "per_source_protocol_counts": dict(per_source_counts),
        "per_subset_counts": dict(per_subset_counts),
        "per_dataset_counts": dict(Counter(x["dataset"] for x in materialized)),
        "items": materialized,
        "skipped_examples": skipped[:100],
        "per_item_payload_hash": sha256_json(materialized),
    }
    return report


def write_discovery_docs(discovery: dict[str, Any]) -> None:
    lines = [
        f"- source_count: `{discovery['source_count']}`",
        f"- usable_source_count: `{discovery['usable_source_count']}`",
        "",
        "| source_file | frame_paths | observed_prompt | future_candidates | masks_or_boxes | usable |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for src in discovery["sources"][:80]:
        lines.append(
            f"| `{src['source_file']}` | `{src['contains_frame_paths']}` | `{src['contains_observed_prompt']}` | `{src['contains_future_candidates']}` | `{src['contains_masks_or_boxes']}` | `{src['usable_for_manifest']}` |"
        )
    write_markdown(DOCS / "STWM_EXTERNAL_BASELINE_PAYLOAD_SOURCE_DISCOVERY_20260426.md", "STWM External Baseline Payload Source Discovery 20260426", lines)


def write_data_root_docs(audit: dict[str, Any]) -> None:
    lines = [
        f"- existing_roots: `{len(audit['existing_roots'])}`",
        f"- frame_dir_candidates: `{len(audit['frame_dir_candidates'])}`",
        f"- mask_dir_candidates: `{len(audit['mask_dir_candidates'])}`",
        f"- exact_blocking_reason_if_none_found: `{audit['exact_blocking_reason_if_none_found']}`",
        "",
        "## Existing Roots",
    ]
    lines.extend(f"- `{x}`" for x in audit["existing_roots"][:80])
    lines.append("")
    lines.append("## Frame Dir Candidates")
    lines.extend(f"- `{x}`" for x in audit["frame_dir_candidates"][:80])
    lines.append("")
    lines.append("## Mask Dir Candidates")
    lines.extend(f"- `{x}`" for x in audit["mask_dir_candidates"][:80])
    write_markdown(DOCS / "STWM_EXTERNAL_BASELINE_DATA_ROOT_AUDIT_20260426.md", "STWM External Baseline Data Root Audit 20260426", lines)


def write_schema_docs(schema: dict[str, Any]) -> None:
    lines = [
        f"- schema_name: `{schema['schema_name']}`",
        f"- null_policy: {schema['null_policy']}",
        "",
        "The JSON schema records raw frames, observed target prompt, future candidate masks/boxes, GT candidate id for evaluation only, and leakage policy for every item.",
    ]
    write_markdown(DOCS / "STWM_EXTERNAL_BASELINE_MANIFEST_SCHEMA_20260426.md", "STWM External Baseline Manifest Schema 20260426", lines)


def write_manifest_docs(manifest: dict[str, Any]) -> None:
    lines = [
        f"- total_items_considered: `{manifest['total_items_considered']}`",
        f"- materialized_items: `{manifest['materialized_items']}`",
        f"- cutie_ready_items: `{manifest['cutie_ready_items']}`",
        f"- sam2_ready_items: `{manifest['sam2_ready_items']}`",
        f"- cotracker_ready_items: `{manifest['cotracker_ready_items']}`",
        f"- skipped_items: `{manifest['skipped_items']}`",
        "",
        "## Per Source Protocol Counts",
    ]
    lines.extend(f"- `{k}`: {v}" for k, v in sorted(manifest["per_source_protocol_counts"].items()))
    lines.append("")
    lines.append("## Skipped Reason Counts")
    lines.extend(f"- `{k}`: {v}" for k, v in sorted(manifest["skipped_reason_counts"].items(), key=lambda kv: -kv[1])[:20])
    write_markdown(DOCS / "STWM_EXTERNAL_BASELINE_ITEM_MANIFEST_20260426.md", "STWM External Baseline Item Manifest 20260426", lines)


def main() -> None:
    discovery = discover_source_files()
    write_json(REPORTS / "stwm_external_baseline_payload_source_discovery_20260426.json", discovery)
    write_discovery_docs(discovery)

    roots = data_root_audit(discovery)
    write_json(REPORTS / "stwm_external_baseline_data_root_audit_20260426.json", roots)
    write_data_root_docs(roots)

    schema = manifest_schema()
    write_json(REPORTS / "stwm_external_baseline_manifest_schema_20260426.json", schema)
    write_schema_docs(schema)

    manifest = build_manifest()
    write_json(REPORTS / "stwm_external_baseline_item_manifest_20260426.json", manifest)
    write_manifest_docs(manifest)


if __name__ == "__main__":
    main()

