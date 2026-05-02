#!/usr/bin/env python3
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _dump(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_doc(path: Path, payload: dict[str, Any]) -> None:
    lines = ["# STWM Object-Dense Trace Teacher Audit V15", ""]
    for name, row in payload.get("teachers", {}).items():
        lines.append(f"## {name}")
        for key in ["teacher_available", "checkpoint_path", "expected_point_count", "gpu_memory_estimate", "speed_estimate", "license_status", "exact_blocker"]:
            lines.append(f"- {key}: `{row.get(key)}`")
        lines.append("")
    lines.append("## Decision")
    for key, value in payload.get("decision", {}).items():
        lines.append(f"- {key}: `{value}`")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _exists_any(paths: list[str]) -> str:
    for raw in paths:
        p = Path(raw)
        if p.exists():
            return str(p)
    return ""


def main() -> int:
    traceanything = _exists_any(["third_party/TraceAnything"])
    cotracker_repo = _exists_any(["baselines/repos/co-tracker/cotracker", "third_party/co-tracker"])
    cotracker_ckpt = _exists_any(list(str(p) for p in Path("baselines/checkpoints/cotracker").glob("**/*") if p.is_file())[:20])
    tapir = _exists_any(["third_party/tapir", "baselines/repos/tapir"])
    pointodyssey_data = _exists_any(["data/external/pointodyssey", "data/raw/pointodyssey"])
    stage1_cache = _exists_any(["data/processed/stage2_tusb_v3_predecode_cache_20260418"])
    teachers: dict[str, dict[str, Any]] = {
        "TraceAnything": {
            "teacher_available": bool(traceanything),
            "repo_or_path": traceanything,
            "checkpoint_path": "",
            "expected_point_count": "interactive/sparse-to-dense depending adapter; no audited local checkpoint in this repo",
            "gpu_memory_estimate": "unknown; must audit before full use",
            "speed_estimate": "unknown",
            "license_status": "third_party source present; official checkpoint/license not audited for OSTF target generation",
            "exact_blocker": "" if traceanything else "third_party/TraceAnything not present",
        },
        "CoTracker_or_CoTracker3": {
            "teacher_available": bool(cotracker_repo and cotracker_ckpt),
            "repo_or_path": cotracker_repo,
            "checkpoint_path": cotracker_ckpt,
            "expected_point_count": "128-2048 query points feasible per object after adapter work",
            "gpu_memory_estimate": "likely B200-feasible for batched clips; exact memory not profiled in V15",
            "speed_estimate": "adapter exists for external baseline; OSTF mask-grid target generation not yet wired",
            "license_status": "external baseline used previously; OSTF teacher use needs separate protocol note",
            "exact_blocker": "" if (cotracker_repo and cotracker_ckpt) else "repo or checkpoint missing",
        },
        "TAPIR": {
            "teacher_available": bool(tapir),
            "repo_or_path": tapir,
            "checkpoint_path": "",
            "expected_point_count": "not available locally",
            "gpu_memory_estimate": "not audited",
            "speed_estimate": "not audited",
            "license_status": "not audited",
            "exact_blocker": "" if tapir else "TAPIR repo/checkpoint not present locally",
        },
        "PointOdyssey_GT": {
            "teacher_available": bool(pointodyssey_data),
            "repo_or_path": pointodyssey_data,
            "checkpoint_path": "GT trajectories if dataset exists",
            "expected_point_count": "dense GT possible only if local PointOdyssey data exists",
            "gpu_memory_estimate": "CPU data read",
            "speed_estimate": "not applicable",
            "license_status": "dataset not found locally" if not pointodyssey_data else "local data present; license still should be cited",
            "exact_blocker": "" if pointodyssey_data else "PointOdyssey data not present under data/external or data/raw",
        },
        "internal_stage1_or_stage2_trace_cache": {
            "teacher_available": bool(stage1_cache),
            "repo_or_path": stage1_cache,
            "checkpoint_path": "predecoded entity boxes/masks, not object-internal physical point tracks",
            "expected_point_count": "M1/M128/M512 pseudo point targets can be derived from masks/bboxes",
            "gpu_memory_estimate": "CPU materialization; no teacher GPU required",
            "speed_estimate": "minutes for mixed split depending M",
            "license_status": "derived from VSPW/VIPSeg local masks",
            "exact_blocker": "not a true physical point tracker; must label as mask_bbox_relative_pseudo_track",
        },
    }
    decision = {
        "physical_point_teacher_ready_for_full_OSTF": bool(cotracker_repo and cotracker_ckpt),
        "true_GT_dense_trajectory_dataset_available": bool(pointodyssey_data),
        "phase1_cache_source": "mask_bbox_relative_pseudo_track from internal Stage2 predecode masks/boxes",
        "can_claim_physical_dense_trace_GT": False,
        "can_build_object_internal_point_supervision_now": bool(stage1_cache),
        "required_next_teacher_step": "wire CoTracker/TraceAnything teacher if physical long-range point supervision is required",
    }
    payload = {
        "audit_name": "stwm_object_dense_trace_teacher_audit_v15",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "teachers": teachers,
        "decision": decision,
    }
    _dump(Path("reports/stwm_object_dense_trace_teacher_audit_v15_20260502.json"), payload)
    _write_doc(Path("docs/STWM_OBJECT_DENSE_TRACE_TEACHER_AUDIT_V15_20260502.md"), payload)
    print("reports/stwm_object_dense_trace_teacher_audit_v15_20260502.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
