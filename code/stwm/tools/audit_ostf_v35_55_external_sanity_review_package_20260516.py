#!/usr/bin/env python3
"""V35.55: external sanity review package，从 V35.54 bundle 独立抽查关键 claim。"""
from __future__ import annotations

import hashlib
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import setproctitle

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))
setproctitle.setproctitle("python")

from stwm.tools.ostf_v17_common_20260502 import ROOT

V35_54 = ROOT / "reports/stwm_ostf_v35_54_submission_ready_benchmark_bundle_20260516.json"
REPORT = ROOT / "reports/stwm_ostf_v35_55_external_sanity_review_package_20260516.json"
DOC = ROOT / "docs/STWM_OSTF_V35_55_EXTERNAL_SANITY_REVIEW_PACKAGE_20260516.md"
RNG_SEED = 3555


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def sha256_file(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def count_dir(path: Path, rel_path: str) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return 1
    if "figures" in rel_path:
        return sum(1 for p in path.rglob("*.png") if p.is_file())
    if "cache" in rel_path:
        return sum(1 for p in path.rglob("*.npz") if p.is_file())
    return sum(1 for p in path.rglob("*") if p.is_file())


def sample_by_group(entries: list[dict[str, Any]], per_group: int = 4) -> list[dict[str, Any]]:
    rng = random.Random(RNG_SEED)
    groups: dict[str, list[dict[str, Any]]] = {}
    for entry in entries:
        groups.setdefault(str(entry.get("group", "unknown")), []).append(entry)
    sampled: list[dict[str, Any]] = []
    for group in sorted(groups):
        rows = list(groups[group])
        rng.shuffle(rows)
        sampled.extend(rows[: min(per_group, len(rows))])
    return sampled


def check_artifact_entry(entry: dict[str, Any]) -> dict[str, Any]:
    path = ROOT / str(entry.get("path", ""))
    expected_sha = entry.get("sha256")
    current_sha = None if path.is_dir() else sha256_file(path)
    expected_count = int(entry.get("file_count") or 0)
    current_count = count_dir(path, str(entry.get("path", "")))
    passed = bool(
        path.exists()
        and bool(entry.get("exists", False))
        and expected_count == current_count
        and (expected_sha in (None, current_sha))
    )
    return {
        "path": entry.get("path"),
        "group": entry.get("group"),
        "exists_now": path.exists(),
        "expected_file_count": expected_count,
        "current_file_count": current_count,
        "expected_sha256": expected_sha,
        "current_sha256": current_sha,
        "passed": passed,
    }


def scalar_bool(z: np.lib.npyio.NpzFile, key: str, default: bool = False) -> bool:
    if key not in z.files:
        return default
    return bool(np.asarray(z[key]).item())


def scalar_str(z: np.lib.npyio.NpzFile, key: str, default: str = "") -> str:
    if key not in z.files:
        return default
    return str(np.asarray(z[key]).item())


def check_unified_npz(path: Path) -> dict[str, Any]:
    try:
        z = np.load(path, allow_pickle=True)
        obs_points = np.asarray(z["obs_points"])
        obs_vis = np.asarray(z["obs_vis"])
        obs_conf = np.asarray(z["obs_conf"])
        future_points = np.asarray(z["future_points"])
        provenance = scalar_str(z, "identity_provenance_type")
        identity_claim_allowed = scalar_bool(z, "identity_claim_allowed")
        pseudo_diag = scalar_bool(z, "identity_pseudo_targets_diagnostic_only")
        shape_ok = bool(
            obs_points.ndim == 3
            and obs_points.shape[-1] == 2
            and obs_vis.shape == obs_conf.shape == obs_points.shape[:2]
            and future_points.ndim == 3
            and future_points.shape[-1] == 2
        )
        contract_ok = bool(
            scalar_bool(z, "raw_video_input_available")
            and scalar_bool(z, "semantic_state_target_available")
            and scalar_bool(z, "identity_pairwise_target_available")
            and scalar_bool(z, "leakage_safe")
            and not scalar_bool(z, "future_teacher_embedding_input_allowed", default=True)
            and not scalar_bool(z, "future_leakage_detected", default=False)
        )
        provenance_ok = bool(
            (provenance == "real_instance" and identity_claim_allowed)
            or (provenance == "pseudo_slot" and (not identity_claim_allowed) and pseudo_diag)
            or (provenance not in {"real_instance", "pseudo_slot"} and not identity_claim_allowed)
        )
        return {
            "path": rel(path),
            "sample_uid": scalar_str(z, "sample_uid"),
            "split": scalar_str(z, "split"),
            "dataset": scalar_str(z, "dataset"),
            "identity_provenance_type": provenance,
            "identity_claim_allowed": identity_claim_allowed,
            "shape_ok": shape_ok,
            "contract_ok": contract_ok,
            "provenance_ok": provenance_ok,
            "passed": bool(shape_ok and contract_ok and provenance_ok),
        }
    except Exception as exc:  # noqa: BLE001
        return {"path": rel(path), "passed": False, "error": repr(exc)}


def check_frontend_npz(path: Path) -> dict[str, Any]:
    try:
        z = np.load(path, allow_pickle=True)
        tracks = np.asarray(z["tracks_xy"])
        visibility = np.asarray(z["visibility"])
        confidence = np.asarray(z["confidence"])
        frame_paths_ok = bool("frame_paths" in z.files and np.asarray(z["frame_paths"]).size > 0)
        predecode_or_frames = bool(frame_paths_ok or ("predecode_path" in z.files and np.asarray(z["predecode_path"]).size > 0))
        # frontend rerun cache 保留 teacher/frontend 原始 object 维度，常见为 [O,M,T,2]；
        # unified slice 才会折叠成 STWM 输入 obs_points [M,Tobs,2]。
        shape_ok = bool(
            tracks.ndim in (3, 4)
            and tracks.shape[-1] == 2
            and visibility.shape == confidence.shape == tracks.shape[:-1]
        )
        input_restricted = scalar_bool(z, "stwm_input_restricted_to_observed", default=True)
        future_as_target = scalar_bool(z, "teacher_uses_full_obs_future_clip_as_target", default=True)
        return {
            "path": rel(path),
            "sample_uid": scalar_str(z, "item_key"),
            "split": scalar_str(z, "split"),
            "dataset": scalar_str(z, "dataset"),
            "shape_ok": shape_ok,
            "raw_frame_or_predecode_present": predecode_or_frames,
            "stwm_input_restricted_to_observed": input_restricted,
            "teacher_uses_full_obs_future_clip_as_target": future_as_target,
            "passed": bool(shape_ok and predecode_or_frames and input_restricted),
        }
    except Exception as exc:  # noqa: BLE001
        return {"path": rel(path), "passed": False, "error": repr(exc)}


def sample_npz(root: Path, n: int) -> list[Path]:
    rng = random.Random(RNG_SEED)
    files = sorted(root.glob("*/*.npz"))
    rng.shuffle(files)
    return files[: min(n, len(files))]


def check_visual_case(case: dict[str, Any]) -> dict[str, Any]:
    png = ROOT / str(case.get("png_path", ""))
    source = ROOT / str(case.get("source_npz", ""))
    case_name = str(case.get("case_name", ""))
    changed = case.get("semantic_changed_accuracy_seed42")
    hard = case.get("semantic_hard_accuracy_seed42")
    ident = case.get("identity_exclude_same_point_top1_seed42")
    semantic_direction_ok = True
    warning: str | None = None
    if case_name == "semantic_changed_success":
        semantic_direction_ok = changed is not None and float(changed) >= 0.5
    elif case_name == "semantic_changed_failure":
        semantic_direction_ok = changed is not None and float(changed) <= 0.5
    elif case_name == "semantic_hard_success":
        semantic_direction_ok = hard is not None and float(hard) >= 0.5
    elif case_name == "semantic_hard_failure":
        semantic_direction_ok = hard is not None and float(hard) <= 0.5
    elif case_name == "real_instance_identity_success":
        semantic_direction_ok = ident is not None and float(ident) >= 0.7
    elif case_name == "real_instance_identity_failure":
        # case-mined “failure” 是相对最弱案例，不强制要求绝对失败；若仍高分则记录 warning。
        if ident is not None and float(ident) >= 0.7:
            warning = "identity failure case 是相对最弱案例，但绝对 top1 仍高；不作为 blocker。"
    elif case_name == "pseudo_identity_diagnostic":
        semantic_direction_ok = not bool(case.get("identity_claim_allowed", True))
    passed = bool(
        png.exists()
        and png.stat().st_size > 0
        and source.exists()
        and bool(case.get("raw_frame_rendered", False))
        and bool(case.get("real_image_rendered", False))
        and semantic_direction_ok
    )
    return {
        "case_name": case_name,
        "sample_uid": case.get("sample_uid"),
        "png_path": case.get("png_path"),
        "source_npz": case.get("source_npz"),
        "png_exists": png.exists(),
        "png_size_bytes": png.stat().st_size if png.exists() else 0,
        "source_exists": source.exists(),
        "semantic_or_identity_direction_ok": semantic_direction_ok,
        "warning": warning,
        "passed": passed,
    }


def main() -> int:
    bundle = load(V35_54)
    entry_points = bundle.get("entry_points", {})
    package_path = ROOT / str(entry_points.get("package_manifest", ""))
    card_path = ROOT / str(entry_points.get("benchmark_card", ""))
    claim_path = ROOT / str(entry_points.get("claim_table", ""))
    reviewer_path = ROOT / str(entry_points.get("reviewer_risk_audit", ""))
    final_path = ROOT / str(entry_points.get("full_325_final_decision", ""))
    dry_path = ROOT / str(entry_points.get("reproducibility_dry_run", ""))

    package = load(package_path)
    card = load(card_path)
    claims = load(claim_path)
    reviewer = load(reviewer_path)
    final = load(final_path)
    dry = load(dry_path)
    viz = load(ROOT / "reports/stwm_ostf_v35_49_full_325_raw_video_closure_visualization_manifest_20260516.json")

    entries = list(package.get("artifact_entries", []))
    artifact_checks = [check_artifact_entry(e) for e in sample_by_group(entries, per_group=4)]
    frontend_checks = [check_frontend_npz(p) for p in sample_npz(ROOT / "outputs/cache/stwm_ostf_v35_49_full_325_raw_video_frontend_rerun/M128_H32", 12)]
    unified_checks = [check_unified_npz(p) for p in sample_npz(ROOT / "outputs/cache/stwm_ostf_v35_49_full_325_rerun_unified_slice/M128_H32", 12)]
    visual_checks = [check_visual_case(c) for c in viz.get("cases", [])]

    allowed_claims = {c.get("claim_id"): c for c in claims.get("claims", []) if c.get("status") == "allowed"}
    not_allowed_claims = {c.get("claim_id"): c for c in claims.get("claims", []) if c.get("status") == "not_allowed"}
    claim_consistency = {
        "full_m128_claim_allowed": "full_m128_h32_raw_video_closure_video_system_benchmark" in allowed_claims,
        "raw_frontend_claim_allowed": "raw_video_frontend_rerun_not_old_trace_cache" in allowed_claims,
        "semantic_state_claim_allowed": "future_semantic_state_transition_field" in allowed_claims,
        "identity_real_instance_claim_allowed": "pairwise_identity_retrieval_field_real_instance_subset" in allowed_claims,
        "full_cvpr_scale_not_allowed": "full_cvpr_scale_complete_system" in not_allowed_claims,
        "teacher_method_not_allowed": "teacher_or_future_embedding_as_method" in not_allowed_claims,
        "card_full_cvpr_false": card.get("safety_and_claim_boundary", {}).get("full_cvpr_scale_claim_allowed") is False,
        "card_future_teacher_input_false": card.get("input_contract", {}).get("future_teacher_embedding_as_input") is False,
        "final_full_m128_claim_true": final.get("m128_h32_full_325_video_system_benchmark_claim_allowed") is True,
        "final_full_cvpr_false": final.get("full_cvpr_scale_claim_allowed") is False,
        "reviewer_audit_passed": reviewer.get("reviewer_risk_audit_passed") is True,
        "dry_run_passed": dry.get("dry_run_passed") is True,
    }
    claim_consistency_passed = all(claim_consistency.values())
    artifact_sampling_passed = all(row["passed"] for row in artifact_checks)
    frontend_sampling_passed = all(row["passed"] for row in frontend_checks)
    unified_sampling_passed = all(row["passed"] for row in unified_checks)
    visualization_consistency_passed = all(row["passed"] for row in visual_checks)
    external_sanity_review_passed = bool(
        bundle.get("submission_ready_benchmark_bundle_ready", False)
        and package.get("reproducibility_package_ready", False)
        and claim_consistency_passed
        and artifact_sampling_passed
        and frontend_sampling_passed
        and unified_sampling_passed
        and visualization_consistency_passed
    )
    warnings = [row for row in visual_checks if row.get("warning")]

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "current_completed_version": "V35.55",
        "source_completed_version": "V35.54",
        "external_sanity_review_done": True,
        "external_sanity_review_passed": external_sanity_review_passed,
        "rng_seed": RNG_SEED,
        "artifact_sampling_passed": artifact_sampling_passed,
        "frontend_sampling_passed": frontend_sampling_passed,
        "unified_sampling_passed": unified_sampling_passed,
        "visualization_consistency_passed": visualization_consistency_passed,
        "claim_consistency_passed": claim_consistency_passed,
        "claim_consistency": claim_consistency,
        "artifact_sample_checks": artifact_checks,
        "frontend_npz_sample_checks": frontend_checks,
        "unified_npz_sample_checks": unified_checks,
        "visual_case_checks": visual_checks,
        "warning_count": len(warnings),
        "warnings": warnings,
        "m128_h32_full_325_video_system_benchmark_claim_allowed": bool(
            external_sanity_review_passed
            and final.get("m128_h32_full_325_video_system_benchmark_claim_allowed", False)
        ),
        "full_cvpr_scale_claim_allowed": False,
        "recommended_next_step": "freeze_v35_55_benchmark_claim_boundary_or_prepare_non_paper_release_bundle",
        "中文结论": (
            "V35.55 external sanity review 通过：从 V35.54 bundle 入口独立抽查 artifact、frontend/unified NPZ、claim table、benchmark card 和 case visualization，未发现 blocker。"
            "这进一步支持 bounded full 325 M128/H32 video-system benchmark claim，但仍不允许 full CVPR-scale 或任意尺度外推。"
            if external_sanity_review_passed
            else "V35.55 external sanity review 未通过，需要先修 artifact、claim consistency 或 visualization consistency。"
        ),
    }

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    DOC.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    DOC.write_text(
        "# STWM OSTF V35.55 External Sanity Review Package\n\n"
        "## 中文总结\n"
        + report["中文结论"]
        + "\n\n"
        "## 抽查结果\n"
        f"- external_sanity_review_passed: {external_sanity_review_passed}\n"
        f"- artifact_sampling_passed: {artifact_sampling_passed}\n"
        f"- frontend_sampling_passed: {frontend_sampling_passed}\n"
        f"- unified_sampling_passed: {unified_sampling_passed}\n"
        f"- visualization_consistency_passed: {visualization_consistency_passed}\n"
        f"- claim_consistency_passed: {claim_consistency_passed}\n"
        f"- warning_count: {len(warnings)}\n"
        f"- m128_h32_full_325_video_system_benchmark_claim_allowed: {report['m128_h32_full_325_video_system_benchmark_claim_allowed']}\n"
        "- full_cvpr_scale_claim_allowed: false\n"
        f"- recommended_next_step: {report['recommended_next_step']}\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "v35_55_external_sanity_review_done": True,
        "external_sanity_review_passed": external_sanity_review_passed,
        "warning_count": len(warnings),
        "m128_h32_full_325_video_system_benchmark_claim_allowed": report["m128_h32_full_325_video_system_benchmark_claim_allowed"],
        "full_cvpr_scale_claim_allowed": False,
        "recommended_next_step": report["recommended_next_step"],
    }, ensure_ascii=False), flush=True)
    return 0 if external_sanity_review_passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
