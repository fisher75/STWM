#!/usr/bin/env python3
from __future__ import annotations

from argparse import ArgumentParser
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List
import json

import numpy as np


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_args() -> Any:
    p = ArgumentParser(description="Build Stage2 bootstrap data contract from existing Stage1 resources")
    p.add_argument("--stage1-contract-path", default="/home/chen034/workspace/data/_manifests/stage1_v2_trace_cache_contract_20260408.json")
    p.add_argument("--report-json", default="/home/chen034/workspace/stwm/reports/stage2_bootstrap_data_contract_20260408.json")
    p.add_argument("--report-md", default="/home/chen034/workspace/stwm/docs/STAGE2_BOOTSTRAP_DATA_CONTRACT_20260408.md")
    p.add_argument("--stage1-backbone-checkpoint", default="/home/chen034/workspace/stwm/outputs/checkpoints/stage1_v2_longtrain_220m_mainline_20260408/best.pt")
    p.add_argument("--max-samples-per-dataset", type=int, default=6)
    return p.parse_args()


def _read_json(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"json not found: {p}")
    payload = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"json payload must be object: {p}")
    return payload


def _mask_candidates(frame_path: Path) -> List[Path]:
    stem = frame_path.stem
    parent = frame_path.parent
    return [
        parent / f"mask_{stem.split('_')[-1]}.png",
        parent / f"{stem.replace('rgb_', 'mask_')}.png",
        parent / f"{stem.replace('rgba_', 'mask_')}.png",
        parent.parent / "masks" / f"{stem}.png",
        parent.parent / "masks" / f"mask_{stem}.png",
    ]


def _inspect_entry(cache_path: Path, source_ref: str) -> Dict[str, Any]:
    payload = np.load(cache_path, allow_pickle=True)
    keys = sorted(payload.files)
    frame_paths = [str(x) for x in np.asarray(payload["frame_paths"]).tolist()] if "frame_paths" in payload.files else []

    has_region_inputs = bool(
        "tracks_2d" in payload.files
        and "valid" in payload.files
        and len(frame_paths) > 0
    )

    mask_ready = False
    if frame_paths:
        fp = Path(frame_paths[0])
        for cand in _mask_candidates(fp):
            if cand.exists():
                mask_ready = True
                break

    src = Path(source_ref)
    if (not mask_ready) and src.exists() and src.suffix.lower() == ".json":
        try:
            obj = json.loads(src.read_text(encoding="utf-8"))
        except Exception:
            obj = {}
        if isinstance(obj, dict):
            for k in ["mask_paths", "segmentation_paths", "instance_mask_paths"]:
                if isinstance(obj.get(k), list) and obj.get(k):
                    mask_ready = True
                    break

    return {
        "cache_path": str(cache_path),
        "source_ref": str(source_ref),
        "cache_keys": keys,
        "frame_paths_available": bool(len(frame_paths) > 0),
        "region_crop_ready": bool(has_region_inputs),
        "mask_crop_ready": bool(mask_ready),
    }


def build_contract(args: Any) -> Dict[str, Any]:
    stage1 = _read_json(args.stage1_contract_path)
    datasets = stage1.get("datasets", []) if isinstance(stage1.get("datasets", []), list) else []

    dataset_reports: List[Dict[str, Any]] = []
    global_region_ready = True

    for ds in datasets:
        if not isinstance(ds, dict):
            continue
        if not bool(ds.get("enabled", False)):
            continue

        dataset_name = str(ds.get("dataset_name", ""))
        index_path = Path(str(ds.get("index_path", "")))
        if not index_path.exists():
            dataset_reports.append(
                {
                    "dataset_name": dataset_name,
                    "status": "missing_index",
                    "index_path": str(index_path),
                    "region_crop_ready": False,
                    "mask_crop_ready": False,
                    "inspected_samples": [],
                }
            )
            global_region_ready = False
            continue

        index_payload = _read_json(index_path)
        entries = index_payload.get("entries", []) if isinstance(index_payload.get("entries", []), list) else []

        inspected = []
        mask_ready_any = False
        region_ready_all = True

        for rec in entries[: max(int(args.max_samples_per_dataset), 1)]:
            if not isinstance(rec, dict):
                continue
            cache_path = Path(str(rec.get("cache_path", "")))
            source_ref = str(rec.get("source_ref", ""))
            if not cache_path.exists():
                region_ready_all = False
                continue
            info = _inspect_entry(cache_path=cache_path, source_ref=source_ref)
            inspected.append(info)
            mask_ready_any = bool(mask_ready_any or info["mask_crop_ready"])
            region_ready_all = bool(region_ready_all and info["region_crop_ready"])

        if not inspected:
            region_ready_all = False

        global_region_ready = bool(global_region_ready and region_ready_all)

        dataset_reports.append(
            {
                "dataset_name": dataset_name,
                "status": "ready_for_bootstrap" if region_ready_all else "insufficient_for_bootstrap",
                "index_path": str(index_path),
                "split_stats": ds.get("split_stats", {}),
                "track_source": str(ds.get("track_source", "")),
                "region_crop_ready": bool(region_ready_all),
                "mask_crop_ready": bool(mask_ready_any),
                "inspected_sample_count": int(len(inspected)),
                "inspected_samples": inspected,
            }
        )

    stage1_ckpt = Path(str(args.stage1_backbone_checkpoint))

    return {
        "generated_at_utc": now_iso(),
        "objective": "Stage2 bootstrap data interface mapping on top of frozen Stage1 resources",
        "stage1_contract_path": str(args.stage1_contract_path),
        "stage1_backbone_checkpoint": {
            "path": str(stage1_ckpt),
            "exists": bool(stage1_ckpt.exists()),
        },
        "semantic_source_definition": {
            "mainline": "object_region_or_mask_crop_visual_state",
            "disallow_fake_hash_label": True,
            "disallow_clip_teacher_as_mainline": True,
        },
        "stage2_input_mapping": {
            "frozen_stage1_trace_tokens": "from Stage1-v2 trace cache state tokens",
            "semantic_tokens": "from frame-path visual region crops with optional mask crop when available",
        },
        "datasets": dataset_reports,
        "bootstrap_interface_ready": bool(global_region_ready and stage1_ckpt.exists()),
        "notes": [
            "No new data expansion in this round.",
            "Mask crop is used when mask resources are present; otherwise region crop fallback is used.",
        ],
    }


def write_markdown(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Stage2 Bootstrap Data Contract",
        "",
        f"- generated_at_utc: {payload.get('generated_at_utc', '')}",
        f"- bootstrap_interface_ready: {payload.get('bootstrap_interface_ready', False)}",
        f"- stage1_backbone_checkpoint_exists: {((payload.get('stage1_backbone_checkpoint', {}) if isinstance(payload.get('stage1_backbone_checkpoint', {}), dict) else {}).get('exists', False))}",
        "",
        "## Semantic Source",
        "- mainline: object_region_or_mask_crop_visual_state",
        "- fake hash label: disallowed",
        "- CLIP teacher distillation as mainline: disallowed",
        "",
        "## Dataset Mapping",
        "| dataset | region_crop_ready | mask_crop_ready | inspected_sample_count | status |",
        "|---|---|---|---:|---|",
    ]

    for ds in payload.get("datasets", []) if isinstance(payload.get("datasets", []), list) else []:
        if not isinstance(ds, dict):
            continue
        lines.append(
            f"| {ds.get('dataset_name', '')} | {bool(ds.get('region_crop_ready', False))} | {bool(ds.get('mask_crop_ready', False))} | {int(ds.get('inspected_sample_count', 0))} | {ds.get('status', '')} |"
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    payload = build_contract(args)

    report_json = Path(args.report_json)
    report_md = Path(args.report_md)
    report_json.parent.mkdir(parents=True, exist_ok=True)

    report_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(report_md, payload)

    print(f"[stage2-contract] report_json={report_json}")
    print(f"[stage2-contract] report_md={report_md}")
    print(f"[stage2-contract] bootstrap_interface_ready={payload.get('bootstrap_interface_ready', False)}")


if __name__ == "__main__":
    main()
