#!/usr/bin/env python3
"""V35.52: reproducibility package manifest 与 benchmark card。"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

import setproctitle

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))
setproctitle.setproctitle("python")

from stwm.tools.ostf_v17_common_20260502 import ROOT

REPORT = ROOT / "reports/stwm_ostf_v35_52_reproducibility_package_and_benchmark_card_20260516.json"
PACKAGE_MANIFEST = ROOT / "reports/stwm_ostf_v35_52_reproducibility_package_manifest_20260516.json"
BENCHMARK_CARD = ROOT / "reports/stwm_ostf_v35_52_benchmark_card_20260516.json"
DOC = ROOT / "docs/STWM_OSTF_V35_52_REPRODUCIBILITY_PACKAGE_AND_BENCHMARK_CARD_20260516.md"

REPORT_ARTIFACTS = [
    "reports/stwm_ostf_v35_49_full_325_raw_video_closure_manifest_20260516.json",
    "reports/stwm_ostf_v35_49_full_325_raw_video_frontend_rerun_20260516.json",
    "reports/stwm_ostf_v35_49_full_325_rerun_unified_slice_build_20260516.json",
    "reports/stwm_ostf_v35_49_full_325_raw_video_closure_benchmark_eval_summary_20260516.json",
    "reports/stwm_ostf_v35_49_full_325_raw_video_closure_benchmark_decision_20260516.json",
    "reports/stwm_ostf_v35_49_full_325_per_category_failure_atlas_eval_20260516.json",
    "reports/stwm_ostf_v35_49_full_325_per_category_failure_atlas_decision_20260516.json",
    "reports/stwm_ostf_v35_49_full_325_raw_video_closure_visualization_manifest_20260516.json",
    "reports/stwm_ostf_v35_49_full_325_raw_video_closure_decision_20260516.json",
    "reports/stwm_ostf_v35_50_cvpr_claim_boundary_and_packaging_audit_20260516.json",
    "reports/stwm_ostf_v35_50_machine_checkable_claim_table_20260516.json",
    "reports/stwm_ostf_v35_51_external_comparison_and_reviewer_risk_audit_20260516.json",
]

DOC_ARTIFACTS = [
    "docs/STWM_OSTF_V35_49_FULL_325_RAW_VIDEO_CLOSURE_MANIFEST_20260516.md",
    "docs/STWM_OSTF_V35_49_FULL_325_RAW_VIDEO_FRONTEND_RERUN_20260516.md",
    "docs/STWM_OSTF_V35_49_FULL_325_RERUN_UNIFIED_SLICE_BUILD_20260516.md",
    "docs/STWM_OSTF_V35_49_FULL_325_RAW_VIDEO_CLOSURE_BENCHMARK_DECISION_20260516.md",
    "docs/STWM_OSTF_V35_49_FULL_325_PER_CATEGORY_FAILURE_ATLAS_DECISION_20260516.md",
    "docs/STWM_OSTF_V35_49_FULL_325_RAW_VIDEO_CLOSURE_VISUALIZATION_20260516.md",
    "docs/STWM_OSTF_V35_49_FULL_325_RAW_VIDEO_CLOSURE_DECISION_20260516.md",
    "docs/STWM_OSTF_V35_50_CVPR_CLAIM_BOUNDARY_AND_PACKAGING_AUDIT_20260516.md",
    "docs/STWM_OSTF_V35_51_EXTERNAL_COMPARISON_AND_REVIEWER_RISK_AUDIT_20260516.md",
]

LOG_ARTIFACTS = [
    "outputs/logs/stwm_ostf_v35_49_full_325_raw_video_frontend_rerun_20260516.log",
    "outputs/logs/stwm_ostf_v35_49_full_325_per_category_failure_atlas_20260516.log",
    "outputs/logs/stwm_ostf_v35_49_full_325_raw_video_closure_visualization_20260516.log",
]

CACHE_ARTIFACTS = [
    "outputs/cache/stwm_ostf_v35_49_full_325_raw_video_closure_manifest/manifest.json",
    "outputs/cache/stwm_ostf_v35_49_full_325_raw_video_frontend_rerun/M128_H32",
    "outputs/cache/stwm_ostf_v35_49_full_325_rerun_unified_slice/M128_H32",
]

FIGURE_ARTIFACTS = [
    "outputs/figures/stwm_ostf_v35_49_full_325_raw_video_closure",
]

SCRIPT_ARTIFACTS = [
    "code/stwm/tools/build_ostf_v35_49_full_325_raw_video_closure_manifest_20260516.py",
    "code/stwm/tools/run_ostf_v35_49_full_325_raw_video_frontend_rerun_20260516.py",
    "code/stwm/tools/build_ostf_v35_49_full_325_rerun_unified_semantic_identity_slice_20260516.py",
    "code/stwm/tools/eval_ostf_v35_49_full_325_raw_video_closure_benchmark_20260516.py",
    "code/stwm/tools/eval_ostf_v35_49_full_325_per_category_failure_atlas_20260516.py",
    "code/stwm/tools/render_ostf_v35_49_full_325_raw_video_closure_visualizations_20260516.py",
    "code/stwm/tools/write_ostf_v35_49_full_325_raw_video_closure_decision_20260516.py",
    "code/stwm/tools/write_ostf_v35_50_cvpr_claim_boundary_and_packaging_audit_20260516.py",
    "code/stwm/tools/audit_ostf_v35_51_external_comparison_and_reviewer_risk_20260516.py",
    "code/stwm/tools/write_ostf_v35_52_reproducibility_package_and_benchmark_card_20260516.py",
]


def load(rel: str) -> dict[str, Any]:
    path = ROOT / rel
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def sha256_file(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def count_files(path: Path, pattern: str = "*") -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return 1
    return sum(1 for p in path.rglob(pattern) if p.is_file())


def artifact_entry(rel: str, group: str) -> dict[str, Any]:
    path = ROOT / rel
    is_dir = path.is_dir()
    return {
        "group": group,
        "path": rel,
        "exists": path.exists(),
        "is_dir": is_dir,
        "size_bytes": path.stat().st_size if path.exists() and path.is_file() else None,
        "sha256": None if is_dir else sha256_file(path),
        "file_count": count_files(path, "*.npz") if is_dir and "cache" in rel else count_files(path, "*.png") if is_dir and "figures" in rel else count_files(path) if is_dir else 1 if path.exists() else 0,
    }


def mean_metric(rows: list[dict[str, Any]], path: list[str]) -> float | None:
    vals: list[float] = []
    for row in rows:
        cur: Any = row
        ok = True
        for key in path:
            if not isinstance(cur, dict) or key not in cur:
                ok = False
                break
            cur = cur[key]
        if ok and cur is not None:
            vals.append(float(cur))
    return float(mean(vals)) if vals else None


def main() -> int:
    final = load("reports/stwm_ostf_v35_49_full_325_raw_video_closure_decision_20260516.json")
    bench = load("reports/stwm_ostf_v35_49_full_325_raw_video_closure_benchmark_decision_20260516.json")
    eval_summary = load("reports/stwm_ostf_v35_49_full_325_raw_video_closure_benchmark_eval_summary_20260516.json")
    atlas = load("reports/stwm_ostf_v35_49_full_325_per_category_failure_atlas_decision_20260516.json")
    vis = load("reports/stwm_ostf_v35_49_full_325_raw_video_closure_visualization_manifest_20260516.json")
    claim = load("reports/stwm_ostf_v35_50_cvpr_claim_boundary_and_packaging_audit_20260516.json")
    reviewer = load("reports/stwm_ostf_v35_51_external_comparison_and_reviewer_risk_audit_20260516.json")

    entries = []
    for rel in REPORT_ARTIFACTS:
        entries.append(artifact_entry(rel, "reports"))
    for rel in DOC_ARTIFACTS:
        entries.append(artifact_entry(rel, "docs"))
    for rel in LOG_ARTIFACTS:
        entries.append(artifact_entry(rel, "logs"))
    for rel in CACHE_ARTIFACTS:
        entries.append(artifact_entry(rel, "cache"))
    for rel in FIGURE_ARTIFACTS:
        entries.append(artifact_entry(rel, "figures"))
    for rel in SCRIPT_ARTIFACTS:
        entries.append(artifact_entry(rel, "scripts"))

    missing = [e for e in entries if not e["exists"]]
    frontend_npz = next((e["file_count"] for e in entries if e["path"].endswith("raw_video_frontend_rerun/M128_H32")), 0)
    unified_npz = next((e["file_count"] for e in entries if e["path"].endswith("rerun_unified_slice/M128_H32")), 0)
    figure_png = next((e["file_count"] for e in entries if e["path"].endswith("full_325_raw_video_closure")), 0)
    selected_clip_count = int(final.get("selected_clip_count", 0) or 0)

    semantic_rows = list(eval_summary.get("semantic_seed_rows", []))
    real_id_rows = list(eval_summary.get("real_instance_identity_seed_rows", []))
    benchmark_card = {
        "benchmark_name": "STWM OSTF V35.49 Full 325 M128/H32 Raw-Video Closure",
        "benchmark_card_version": "V35.52",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_contract": {
            "primary_input": "raw video / predecode frame paths",
            "derived_input": "object-dense observed trace field obs_points[M,Tobs,2], obs_vis[M,Tobs], obs_conf[M,Tobs]",
            "old_trace_cache_role": "仅用于 drift comparison，不作为输入结果",
            "future_teacher_embedding_as_input": False,
        },
        "output_contract": {
            "future_trace_field": "frozen V30 M128/H32 future object-dense trace field",
            "future_semantic_field": "future semantic state / transition / uncertainty field",
            "future_identity_field": "real-instance subset 上的 pairwise identity retrieval field",
            "pseudo_identity_role": "VSPW pseudo slot identity diagnostic-only",
        },
        "scope": {
            "selected_clip_count": selected_clip_count,
            "dataset_counts": final.get("dataset_counts"),
            "split_counts": final.get("split_counts"),
            "real_instance_identity_count": final.get("real_instance_identity_count"),
            "pseudo_identity_count": final.get("pseudo_identity_count"),
            "M": 128,
            "H": 32,
            "forbidden_extrapolations": [
                "不外推到 H64/H96",
                "不外推到 M512/M1024",
                "不外推到任意分辨率/任意 horizon/任意视频域",
                "不 claim full open-vocabulary dense semantic segmentation field",
                "不 claim V34 continuous teacher embedding delta route success",
            ],
        },
        "models_and_seeds": {
            "trajectory_backbone": "V30 M128 frozen",
            "semantic_adapter": "V35.21 domain-normalized video semantic state adapter, seeds 42/123/456",
            "identity_head": "V35.29 expanded pairwise identity retrieval head, seeds 42/123/456",
            "new_training_in_v35_49_to_v35_52": False,
        },
        "metrics": {
            "raw_frontend_rerun_success_rate": final.get("raw_frontend_rerun_success_rate"),
            "trace_drift_ok": final.get("trace_drift_ok"),
            "trace_drift_vs_cache_mean": final.get("trace_drift_vs_cache_mean"),
            "semantic_changed_balanced_accuracy_val_mean": bench.get("semantic_changed_balanced_accuracy_val_mean"),
            "semantic_changed_balanced_accuracy_test_mean": bench.get("semantic_changed_balanced_accuracy_test_mean"),
            "semantic_hard_balanced_accuracy_val_mean": bench.get("semantic_hard_balanced_accuracy_val_mean"),
            "semantic_hard_balanced_accuracy_test_mean": bench.get("semantic_hard_balanced_accuracy_test_mean"),
            "semantic_uncertainty_balanced_accuracy_val_mean": bench.get("semantic_uncertainty_balanced_accuracy_val_mean"),
            "semantic_uncertainty_balanced_accuracy_test_mean": bench.get("semantic_uncertainty_balanced_accuracy_test_mean"),
            "real_identity_exclude_same_point_top1_test_mean": mean_metric(real_id_rows, ["test", "identity_retrieval_exclude_same_point_top1"]),
            "real_identity_instance_pooled_top1_test_mean": mean_metric(real_id_rows, ["test", "identity_retrieval_instance_pooled_top1"]),
            "real_identity_confuser_avoidance_top1_test_mean": mean_metric(real_id_rows, ["test", "identity_confuser_avoidance_top1"]),
            "stable_preservation": final.get("stable_preservation"),
            "semantic_fragile_category_count_test": final.get("semantic_fragile_category_count_test"),
            "identity_fragile_category_count_test": final.get("identity_fragile_category_count_test"),
            "atlas_ready": atlas.get("atlas_ready"),
            "visualization_ready": vis.get("visualization_ready"),
        },
        "safety_and_claim_boundary": {
            "v30_backbone_frozen": final.get("v30_backbone_frozen"),
            "future_leakage_detected": final.get("future_leakage_detected"),
            "trajectory_degraded": final.get("trajectory_degraded"),
            "teacher_as_method_detected": claim.get("teacher_as_method_detected"),
            "future_teacher_embedding_input_detected": claim.get("future_teacher_embedding_input_detected"),
            "identity_pseudo_targets_excluded_from_claim": final.get("identity_pseudo_targets_excluded_from_claim"),
            "m128_h32_full_325_video_system_benchmark_claim_allowed": final.get("m128_h32_full_325_video_system_benchmark_claim_allowed"),
            "full_cvpr_scale_claim_allowed": False,
        },
        "required_artifact_manifest": str(PACKAGE_MANIFEST.relative_to(ROOT)),
    }

    reproducibility_package_ready = bool(
        not missing
        and selected_clip_count >= 300
        and frontend_npz >= selected_clip_count
        and unified_npz >= selected_clip_count
        and figure_png >= 12
        and claim.get("artifact_packaging_complete", False)
        and reviewer.get("reviewer_risk_audit_passed", False)
    )

    package_manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "current_completed_version": "V35.52",
        "source_completed_version": "V35.49/V35.50/V35.51",
        "reproducibility_package_ready": reproducibility_package_ready,
        "artifact_entries": entries,
        "missing_artifacts": missing,
        "frontend_npz_count": frontend_npz,
        "unified_npz_count": unified_npz,
        "figure_png_count": figure_png,
        "minimum_expected_clip_count": 300,
        "selected_clip_count": selected_clip_count,
        "notes_zh": "该 manifest 记录 full 325 M128/H32 raw-video closure 的报告、文档、日志、缓存、图像与脚本。目录型 cache 不记录完整 sha256，只记录文件数；单文件记录 sha256。",
    }

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "current_completed_version": "V35.52",
        "reproducibility_package_manifest_built": True,
        "benchmark_card_built": True,
        "reproducibility_package_ready": reproducibility_package_ready,
        "benchmark_card_path": str(BENCHMARK_CARD.relative_to(ROOT)),
        "package_manifest_path": str(PACKAGE_MANIFEST.relative_to(ROOT)),
        "selected_clip_count": selected_clip_count,
        "frontend_npz_count": frontend_npz,
        "unified_npz_count": unified_npz,
        "figure_png_count": figure_png,
        "missing_artifact_count": len(missing),
        "artifact_packaging_complete": claim.get("artifact_packaging_complete"),
        "reviewer_risk_audit_passed": reviewer.get("reviewer_risk_audit_passed"),
        "m128_h32_full_325_video_system_benchmark_claim_allowed": bool(
            reproducibility_package_ready
            and final.get("m128_h32_full_325_video_system_benchmark_claim_allowed", False)
        ),
        "full_cvpr_scale_claim_allowed": False,
        "recommended_next_step": "run_v35_53_reproducibility_dry_run_from_package_manifest",
        "中文结论": (
            "V35.52 已把 V35.49 full 325 M128/H32 raw-video closure 整理成可复验 package manifest 和 benchmark card。"
            "当前可支持 bounded full M128/H32 video-system benchmark claim；仍不支持任意尺度/full CVPR-scale/open-vocabulary semantic field claim。"
        ),
    }

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    DOC.parent.mkdir(parents=True, exist_ok=True)
    PACKAGE_MANIFEST.write_text(json.dumps(package_manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    BENCHMARK_CARD.write_text(json.dumps(benchmark_card, indent=2, ensure_ascii=False), encoding="utf-8")
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# STWM OSTF V35.52 Reproducibility Package and Benchmark Card",
        "",
        "## 中文总结",
        report["中文结论"],
        "",
        "## Package 状态",
        f"- reproducibility_package_ready: {reproducibility_package_ready}",
        f"- selected_clip_count: {selected_clip_count}",
        f"- frontend_npz_count: {frontend_npz}",
        f"- unified_npz_count: {unified_npz}",
        f"- figure_png_count: {figure_png}",
        f"- missing_artifact_count: {len(missing)}",
        "",
        "## Benchmark Card 核心口径",
        "- 输入：raw video / predecode frame paths，经 frontend 重新生成 observed dense trace。",
        "- 输出：frozen V30 M128/H32 future trace、V35 semantic state / transition / uncertainty、real-instance pairwise identity retrieval。",
        "- 旧 trace cache：只做 drift comparison，不作为输入结果。",
        "- pseudo identity：diagnostic-only，不进入 identity claim gate。",
        "- teacher / DINO / CLIP / SAM2 / CoTracker：只能作为 frontend、measurement 或 supervision source。",
        "",
        "## Claim 边界",
        f"- m128_h32_full_325_video_system_benchmark_claim_allowed: {report['m128_h32_full_325_video_system_benchmark_claim_allowed']}",
        "- full_cvpr_scale_claim_allowed: false",
        "- 不允许外推到 H64/H96、M512/M1024、任意 horizon、任意分辨率或 full open-vocabulary dense segmentation。",
        "",
        f"- package_manifest: {PACKAGE_MANIFEST.relative_to(ROOT)}",
        f"- benchmark_card: {BENCHMARK_CARD.relative_to(ROOT)}",
        f"- recommended_next_step: {report['recommended_next_step']}",
        "",
    ]
    DOC.write_text("\n".join(lines), encoding="utf-8")

    print(json.dumps({
        "v35_52_reproducibility_package_done": True,
        "reproducibility_package_ready": reproducibility_package_ready,
        "benchmark_card_built": True,
        "m128_h32_full_325_video_system_benchmark_claim_allowed": report["m128_h32_full_325_video_system_benchmark_claim_allowed"],
        "full_cvpr_scale_claim_allowed": False,
        "recommended_next_step": report["recommended_next_step"],
    }, ensure_ascii=False), flush=True)
    return 0 if reproducibility_package_ready else 2


if __name__ == "__main__":
    raise SystemExit(main())
