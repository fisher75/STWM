#!/usr/bin/env python3
from __future__ import annotations

import json
import py_compile
from pathlib import Path
from typing import Any

from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, write_doc
from stwm.tools.ostf_v30_external_gt_schema_20260508 import utc_now

REPORT = ROOT / "reports/stwm_ostf_v33_latest_repo_truth_refresh_20260509.json"
DOC = ROOT / "docs/STWM_OSTF_V33_LATEST_REPO_TRUTH_REFRESH_20260509.md"

V33_CODE = [
    "code/stwm/tools/audit_ostf_v33_dense_field_target_coverage_20260509.py",
    "code/stwm/tools/audit_ostf_v33_latest_repo_truth_20260509.py",
    "code/stwm/tools/audit_ostf_v33_pointodyssey_semantic_identity_sources_20260509.py",
    "code/stwm/tools/audit_ostf_v33_semantic_identity_code_contract_20260509.py",
    "code/stwm/tools/audit_ostf_v33_trajectory_backbone_freeze_20260509.py",
    "code/stwm/tools/build_ostf_v33_pointodyssey_identity_targets_20260509.py",
    "code/stwm/tools/eval_ostf_v33_semantic_identity_field_20260509.py",
    "code/stwm/tools/ostf_v33_semantic_identity_schema_20260509.py",
    "code/stwm/tools/preflight_ostf_v33_visual_teacher_semantic_prototypes_20260509.py",
    "code/stwm/tools/train_ostf_v33_semantic_identity_head_20260509.py",
    "code/stwm/tools/write_ostf_v33_semantic_identity_target_construction_decision_20260509.py",
    "code/stwm/modules/ostf_semantic_identity_heads_v33.py",
]

V33_REPORTS = [
    "reports/stwm_ostf_v33_latest_repo_truth_audit_20260509.json",
    "reports/stwm_ostf_v33_trajectory_backbone_freeze_audit_20260509.json",
    "reports/stwm_ostf_v33_pointodyssey_semantic_identity_source_audit_20260509.json",
    "reports/stwm_ostf_v33_pointodyssey_identity_target_build_20260509.json",
    "reports/stwm_ostf_v33_visual_teacher_preflight_20260509.json",
    "reports/stwm_ostf_v33_semantic_identity_code_contract_audit_20260509.json",
    "reports/stwm_ostf_v33_dense_field_target_coverage_20260509.json",
    "reports/stwm_ostf_v33_semantic_identity_smoke_summary_20260509.json",
    "reports/stwm_ostf_v33_semantic_identity_smoke_decision_20260509.json",
    "reports/stwm_ostf_v33_semantic_identity_target_construction_decision_20260509.json",
]


def compile_status(rel: str) -> dict[str, Any]:
    path = ROOT / rel
    if not path.exists():
        return {"exists": False, "py_compile_ok": False, "exact_error": "missing"}
    try:
        py_compile.compile(str(path), doraise=True)
        return {"exists": True, "py_compile_ok": True, "exact_error": None}
    except Exception as exc:
        return {"exists": True, "py_compile_ok": False, "exact_error": f"{type(exc).__name__}: {exc}"}


def read_text(rel: str) -> str:
    path = ROOT / rel
    return path.read_text(encoding="utf-8", errors="ignore") if path.exists() else ""


def load_json(rel: str) -> dict[str, Any]:
    path = ROOT / rel
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"_json_load_error": f"{type(exc).__name__}: {exc}"}


def main() -> int:
    code = {rel: compile_status(rel) for rel in V33_CODE}
    reports = {}
    missing_artifacts = []
    for rel in V33_REPORTS:
        path = ROOT / rel
        reports[rel] = {"exists": path.exists(), "size_bytes": path.stat().st_size if path.exists() else 0}
        if not path.exists():
            missing_artifacts.append(rel)
    eval_text = read_text("code/stwm/tools/eval_ostf_v33_semantic_identity_field_20260509.py")
    train_text = read_text("code/stwm/tools/train_ostf_v33_semantic_identity_head_20260509.py")
    builder_text = read_text("code/stwm/tools/build_ostf_v33_pointodyssey_identity_targets_20260509.py")
    build = load_json("reports/stwm_ostf_v33_pointodyssey_identity_target_build_20260509.json")
    smoke = load_json("reports/stwm_ostf_v33_semantic_identity_smoke_summary_20260509.json")
    sidecar_root = ROOT / "outputs/cache/stwm_ostf_v33_semantic_identity_targets/pointodyssey"
    sidecar_count = sum(1 for _ in sidecar_root.glob("*/*.npz")) if sidecar_root.exists() else 0
    smoke_ckpt = smoke.get("checkpoint_path")
    smoke_ckpt_exists = bool(smoke_ckpt and (ROOT / str(smoke_ckpt)).exists())
    eval_stub_detected = "print(summary.relative_to(ROOT)" in eval_text and "model.load_state_dict" not in eval_text
    v30_checkpoint_not_consumed = "torch.load(args.v30_checkpoint" not in train_text
    last_frame_assignment_risk = "obs_instance[:, -1]" in builder_text
    partial = {
        "target_builder_real": "fut_instance_available_mask" in builder_text and "point_to_instance_assignment_method" in builder_text,
        "visual_teacher_preflight_real": bool(load_json("reports/stwm_ostf_v33_visual_teacher_preflight_20260509.json")),
        "code_contract_real": bool(load_json("reports/stwm_ostf_v33_semantic_identity_code_contract_audit_20260509.json")),
        "dense_coverage_real": bool(load_json("reports/stwm_ostf_v33_dense_field_target_coverage_20260509.json")),
        "smoke_real": bool(smoke),
    }
    payload = {
        "generated_at_utc": utc_now(),
        "code": code,
        "reports": reports,
        "all_v33_code_py_compile_ok": all(v.get("py_compile_ok") for v in code.values()),
        "reports_existing_count": sum(int(v["exists"]) for v in reports.values()),
        "reports_expected_count": len(reports),
        "code_skeleton_vs_real_run": partial,
        "eval_stub_detected": eval_stub_detected,
        "v30_checkpoint_not_consumed": v30_checkpoint_not_consumed,
        "last_frame_assignment_risk": last_frame_assignment_risk,
        "sidecar_target_cache_exists": sidecar_count > 0,
        "sidecar_target_cache_count": sidecar_count,
        "smoke_checkpoint_exists": smoke_ckpt_exists,
        "smoke_level": smoke.get("smoke_level", "not_run") if smoke else "not_run",
        "integrated_v30_backbone_used": bool(smoke.get("integrated_v30_backbone_used", False)) if smoke else False,
        "v30_checkpoint_consumed_in_smoke": bool(smoke.get("v30_checkpoint_consumed_in_smoke", False)) if smoke else False,
        "identity_build_total_samples": build.get("total_samples_processed"),
        "exact_missing_artifacts": missing_artifacts,
    }
    dump_json(REPORT, payload)
    write_doc(
        DOC,
        "STWM OSTF V33 Latest Repo Truth Refresh",
        payload,
        [
            "all_v33_code_py_compile_ok",
            "reports_existing_count",
            "reports_expected_count",
            "eval_stub_detected",
            "v30_checkpoint_not_consumed",
            "last_frame_assignment_risk",
            "sidecar_target_cache_count",
            "smoke_checkpoint_exists",
            "smoke_level",
            "exact_missing_artifacts",
        ],
    )
    print(REPORT.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
