#!/usr/bin/env python3
from __future__ import annotations

from argparse import ArgumentParser
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List
import json


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_args() -> Any:
    p = ArgumentParser(description="Build Stage2 bootstrap contract from latest Stage2 dataset audit")
    p.add_argument(
        "--stage2-audit-json",
        default="/home/chen034/workspace/data/_manifests/stage2_dataset_audit_20260408.json",
    )
    p.add_argument(
        "--stage1-backbone-checkpoint",
        default="/home/chen034/workspace/stwm/outputs/checkpoints/stage1_v2_longtrain_220m_mainline_20260408/best.pt",
    )
    p.add_argument(
        "--report-json",
        default="/home/chen034/workspace/stwm/reports/stage2_bootstrap_data_contract_20260408.json",
    )
    p.add_argument(
        "--report-md",
        default="/home/chen034/workspace/stwm/docs/STAGE2_BOOTSTRAP_DATA_CONTRACT_20260408.md",
    )
    return p.parse_args()


def _read_json(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"json not found: {p}")
    payload = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"json payload must be dict: {p}")
    return payload


def _row_map(audit_payload: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    rows = audit_payload.get("rows", [])
    out: Dict[str, Dict[str, Any]] = {}
    if not isinstance(rows, list):
        return out
    for row in rows:
        if not isinstance(row, dict):
            continue
        name = str(row.get("dataset_name", "")).strip().upper()
        if not name:
            continue
        out[name] = row
    return out


def _dataset_record(name: str, role: str, row: Dict[str, Any]) -> Dict[str, Any]:
    name_upper = str(name).strip().upper()
    display_name = str(name).strip()
    local_path = str(row.get("local_path", ""))
    status = str(row.get("status", "unknown"))

    if name_upper == "VSPW":
        split_mapping = {
            "train": {
                "split_file": f"{local_path}/train.txt",
                "frame_root": f"{local_path}/data",
                "frame_subdir": "origin",
                "mask_root": f"{local_path}/data",
                "mask_subdir": "mask",
            },
            "val": {
                "split_file": f"{local_path}/val.txt",
                "frame_root": f"{local_path}/data",
                "frame_subdir": "origin",
                "mask_root": f"{local_path}/data",
                "mask_subdir": "mask",
            },
            "test": {
                "split_file": f"{local_path}/test.txt",
                "frame_root": f"{local_path}/data",
                "frame_subdir": "origin",
                "mask_root": f"{local_path}/data",
                "mask_subdir": "mask",
            },
        }
        annotation_source = "VSPW mask PNG under data/<clip_id>/mask (train/val), test may not ship dense masks"
        used_train = True
        used_eval = True
    elif name_upper == "VIPSEG":
        split_mapping = {
            "train": {
                "split_file": f"{local_path}/train.txt",
                "frame_root": f"{local_path}/imgs",
                "frame_subdir": "",
                "mask_root": f"{local_path}/panomasks",
                "mask_subdir": "",
            },
            "val": {
                "split_file": f"{local_path}/val.txt",
                "frame_root": f"{local_path}/imgs",
                "frame_subdir": "",
                "mask_root": f"{local_path}/panomasks",
                "mask_subdir": "",
            },
            "test": {
                "split_file": f"{local_path}/test.txt",
                "frame_root": f"{local_path}/imgs",
                "frame_subdir": "",
                "mask_root": f"{local_path}/panomasks",
                "mask_subdir": "",
            },
        }
        annotation_source = "VIPSeg panomasks under panomasks/<clip_id> with split files train/val/test"
        used_train = True
        used_eval = True
    elif name_upper == "BURST":
        split_mapping = {
            "train": {
                "frames_root": f"{local_path}/images/train/frames/train",
                "annotation_file": f"{local_path}/annotations/train/train.json",
            },
            "val": {
                "frames_root": f"{local_path}/images/val/frames/val",
                "annotation_file": f"{local_path}/annotations/val/all_classes.json",
            },
            "test": {
                "frames_root": f"{local_path}/images/test/frames/test",
                "annotation_file": f"{local_path}/annotations/test/all_classes.json",
            },
        }
        annotation_source = "BURST annotation JSON (train/train.json, val/test all_classes.json)"
        used_train = False
        used_eval = True
    else:
        split_mapping = {}
        annotation_source = ""
        used_train = False
        used_eval = False

    return {
        "dataset_name": display_name,
        "role_in_stage2": role,
        "status_from_audit": status,
        "local_path": local_path,
        "split_mapping": split_mapping,
        "annotation_source": annotation_source,
        "expected_semantic_fields": [
            "semantic_frame_path",
            "semantic_region_box_xyxy",
            "semantic_embedding",
            "semantic_source_mode",
            "mask_crop_used",
        ],
        "used_in_bootstrap_train": bool(used_train),
        "used_in_bootstrap_eval": bool(used_eval),
        "not_in_current_bootstrap": False,
        "reason": "",
    }


def build_contract(args: Any) -> Dict[str, Any]:
    audit = _read_json(args.stage2_audit_json)
    rows = _row_map(audit)

    final_decision = str(audit.get("final_decision", "unknown"))
    stage1_ckpt = Path(str(args.stage1_backbone_checkpoint))

    vspw = _dataset_record("VSPW", "Stage2 core data", rows.get("VSPW", {}))
    vipseg = _dataset_record("VIPSeg", "Stage2 core data", rows.get("VIPSEG", {}))
    burst = _dataset_record("BURST", "Stage2 open-world extension", rows.get("BURST", {}))

    tao_status = str(rows.get("TAO", {}).get("status", "unknown"))
    visor_status = str(rows.get("VISOR", {}).get("status", "unknown"))

    excluded = [
        {
            "dataset_name": "TAO",
            "role_in_stage2": "optional large-scale extension",
            "status_from_audit": tao_status,
            "not_in_current_bootstrap": True,
            "reason": tao_status,
        },
        {
            "dataset_name": "VISOR",
            "role_in_stage2": "transfer / egocentric extension",
            "status_from_audit": visor_status,
            "not_in_current_bootstrap": True,
            "reason": visor_status,
        },
    ]

    core_ready = (
        str(vspw.get("status_from_audit", "")) == "complete"
        and str(vipseg.get("status_from_audit", "")) == "complete"
    )

    return {
        "generated_at_utc": now_iso(),
        "round": "stage2_bootstrap_20260408",
        "objective": "Bootstrap-ready Stage2 with frozen Stage1 220m backbone and small smoke only",
        "source_audit_json": str(args.stage2_audit_json),
        "stage1_backbone_checkpoint": {
            "path": str(stage1_ckpt),
            "exists": bool(stage1_ckpt.exists()),
            "frozen_in_this_round": True,
        },
        "audit_snapshot": {
            "VSPW": str(rows.get("VSPW", {}).get("status", "unknown")),
            "VIPSeg": str(rows.get("VIPSEG", {}).get("status", "unknown")),
            "BURST": str(rows.get("BURST", {}).get("status", "unknown")),
            "TAO": tao_status,
            "VISOR": visor_status,
            "final_decision": final_decision,
        },
        "stage2_bootstrap_binding": {
            "core": ["VSPW", "VIPSeg"],
            "optional_extension": ["BURST"],
        },
        "datasets": [vspw, vipseg, burst],
        "excluded_datasets": excluded,
        "semantic_source_policy": {
            "mainline": "object_region_or_mask_crop_visual_state",
            "disallow_fake_hash_label": True,
            "disallow_clip_teacher_as_mainline": True,
        },
        "bootstrap_interface_ready": bool(core_ready and stage1_ckpt.exists()),
        "notes": [
            "TAO and VISOR are explicitly marked not_in_current_bootstrap in this round.",
            "BURST is optional extension and can be used as bootstrap eval extension.",
            "No full Stage2 longtrain is allowed in this round.",
        ],
    }


def write_markdown(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: List[str] = []
    lines.append("# Stage2 Bootstrap Data Contract")
    lines.append("")
    lines.append(f"- generated_at_utc: {payload.get('generated_at_utc', '')}")
    lines.append(f"- bootstrap_interface_ready: {payload.get('bootstrap_interface_ready', False)}")
    lines.append(f"- source_audit_json: {payload.get('source_audit_json', '')}")
    lines.append("")
    lines.append("## Binding")
    binding = payload.get("stage2_bootstrap_binding", {})
    lines.append(f"- core: {binding.get('core', [])}")
    lines.append(f"- optional_extension: {binding.get('optional_extension', [])}")
    lines.append("")
    lines.append("## Included Datasets")
    lines.append("| dataset | role | status_from_audit | used_in_bootstrap_train | used_in_bootstrap_eval | local_path |")
    lines.append("|---|---|---|---|---|---|")
    for ds in payload.get("datasets", []):
        if not isinstance(ds, dict):
            continue
        lines.append(
            f"| {ds.get('dataset_name', '')} | {ds.get('role_in_stage2', '')} | {ds.get('status_from_audit', '')} | "
            f"{bool(ds.get('used_in_bootstrap_train', False))} | {bool(ds.get('used_in_bootstrap_eval', False))} | {ds.get('local_path', '')} |"
        )
    lines.append("")
    lines.append("## Excluded Datasets")
    lines.append("| dataset | not_in_current_bootstrap | reason |")
    lines.append("|---|---|---|")
    for ds in payload.get("excluded_datasets", []):
        if not isinstance(ds, dict):
            continue
        lines.append(
            f"| {ds.get('dataset_name', '')} | {bool(ds.get('not_in_current_bootstrap', False))} | {ds.get('reason', '')} |"
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    payload = build_contract(args)

    report_json = Path(str(args.report_json))
    report_md = Path(str(args.report_md))
    report_json.parent.mkdir(parents=True, exist_ok=True)

    report_json.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    write_markdown(report_md, payload)

    print(f"[stage2-contract] report_json={report_json}")
    print(f"[stage2-contract] report_md={report_md}")
    print(f"[stage2-contract] bootstrap_interface_ready={payload.get('bootstrap_interface_ready', False)}")


if __name__ == "__main__":
    main()
