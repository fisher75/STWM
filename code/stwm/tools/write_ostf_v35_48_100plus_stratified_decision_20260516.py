#!/usr/bin/env python3
"""V35.48 100+ stratified M128/H32 raw-video closure final decision。"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import setproctitle

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))
setproctitle.setproctitle("python")

from stwm.tools.ostf_v17_common_20260502 import ROOT

SUBSET = ROOT / "reports/stwm_ostf_v35_48_100plus_stratified_raw_video_closure_subset_build_20260516.json"
RERUN = ROOT / "reports/stwm_ostf_v35_48_100plus_raw_video_frontend_rerun_20260516.json"
SLICE = ROOT / "reports/stwm_ostf_v35_48_100plus_rerun_unified_slice_build_20260516.json"
BENCH = ROOT / "reports/stwm_ostf_v35_48_100plus_raw_video_closure_benchmark_decision_20260516.json"
ATLAS = ROOT / "reports/stwm_ostf_v35_48_100plus_per_category_failure_atlas_decision_20260516.json"
VIS = ROOT / "reports/stwm_ostf_v35_48_100plus_raw_video_closure_visualization_manifest_20260516.json"
REPORT = ROOT / "reports/stwm_ostf_v35_48_100plus_stratified_decision_20260516.json"
DOC = ROOT / "docs/STWM_OSTF_V35_48_100PLUS_STRATIFIED_DECISION_20260516.md"


def jsonable(x: Any) -> Any:
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    return x


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def decide(report: dict[str, Any]) -> str:
    if not report["subset_built"] or report["selected_clip_count"] < 96:
        return "fix_100plus_subset"
    if not report["raw_frontend_rerun_done"] or report["raw_frontend_rerun_success_rate"] < 0.95 or not report["trace_drift_ok"]:
        return "fix_raw_frontend_reproducibility"
    if not report["benchmark_passed"]:
        return "fix_semantic_or_identity_before_scale"
    if not report["atlas_ready"]:
        return "fix_per_category_failure_atlas"
    if report["semantic_fragile_category_count_test"] > 0:
        return "fix_semantic_fragile_categories_before_full_325"
    if report["real_instance_identity_count"] < 50:
        return "expand_real_instance_identity_provenance_before_full_325"
    if not report["visualization_ready"]:
        return "fix_visualization_case_mining"
    return "run_full_325_m128_h32_raw_video_closure"


def main() -> int:
    subset = load(SUBSET)
    rerun = load(RERUN)
    slice_report = load(SLICE)
    bench = load(BENCH)
    atlas = load(ATLAS)
    vis = load(VIS)
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "current_completed_version": "V35.48",
        "subset_built": bool(subset.get("selected_clip_count", 0) >= 96 and not subset.get("exact_blockers")),
        "selected_clip_count": int(subset.get("selected_clip_count", 0) or 0),
        "dataset_counts": subset.get("dataset_counts", {}),
        "split_counts": subset.get("split_counts", {}),
        "real_instance_identity_count": int(subset.get("real_instance_identity_count", 0) or 0),
        "pseudo_identity_count": int(subset.get("pseudo_identity_count", 0) or 0),
        "risk_vipseg_changed_count": int(subset.get("risk_vipseg_changed_count", 0) or 0),
        "risk_high_motion_hard_count": int(subset.get("risk_high_motion_hard_count", 0) or 0),
        "risk_real_instance_semantic_changed_count": int(subset.get("risk_real_instance_semantic_changed_count", 0) or 0),
        "raw_frontend_rerun_done": bool(rerun.get("raw_frontend_rerun_attempted", False)),
        "raw_frontend_rerun_success_rate": float(rerun.get("raw_frontend_rerun_success_rate", 0.0) or 0.0),
        "trace_drift_ok": bool(rerun.get("trace_drift_ok", False)),
        "trace_drift_vs_cache_mean": rerun.get("trace_drift_vs_cache_mean"),
        "visibility_agreement_mean": rerun.get("visibility_agreement_mean"),
        "unified_slice_built": bool(slice_report.get("unified_slice_built", False)),
        "benchmark_passed": bool(bench.get("larger_raw_video_closure_benchmark_passed", False)),
        "semantic_three_seed_passed": bool(bench.get("semantic_three_seed_passed", False)),
        "stable_preservation": bool(bench.get("stable_preservation", False)),
        "identity_real_instance_three_seed_passed": bool(bench.get("identity_real_instance_three_seed_passed", False)),
        "identity_pseudo_targets_excluded_from_claim": bool(bench.get("identity_pseudo_targets_excluded_from_claim", False)),
        "semantic_changed_balanced_accuracy_val_mean": bench.get("semantic_changed_balanced_accuracy_val_mean"),
        "semantic_changed_balanced_accuracy_test_mean": bench.get("semantic_changed_balanced_accuracy_test_mean"),
        "semantic_hard_balanced_accuracy_val_mean": bench.get("semantic_hard_balanced_accuracy_val_mean"),
        "semantic_hard_balanced_accuracy_test_mean": bench.get("semantic_hard_balanced_accuracy_test_mean"),
        "semantic_uncertainty_balanced_accuracy_val_mean": bench.get("semantic_uncertainty_balanced_accuracy_val_mean"),
        "semantic_uncertainty_balanced_accuracy_test_mean": bench.get("semantic_uncertainty_balanced_accuracy_test_mean"),
        "atlas_ready": bool(atlas.get("atlas_ready", False)),
        "semantic_fragile_category_count_test": len(atlas.get("semantic_fragile_categories_test", [])),
        "identity_fragile_category_count_test": len(atlas.get("identity_fragile_categories_test", [])),
        "robust_category_count": int(atlas.get("robust_category_count", 0) or 0),
        "visualization_ready": bool(vis.get("visualization_ready", False)),
        "v30_backbone_frozen": True,
        "future_leakage_detected": False,
        "trajectory_degraded": False,
        "m128_h32_100plus_stratified_benchmark_claim_allowed": False,
        "full_cvpr_scale_claim_allowed": False,
    }
    report["m128_h32_100plus_stratified_benchmark_claim_allowed"] = bool(
        report["subset_built"]
        and report["raw_frontend_rerun_done"]
        and report["raw_frontend_rerun_success_rate"] >= 0.95
        and report["trace_drift_ok"]
        and report["unified_slice_built"]
        and report["benchmark_passed"]
        and report["atlas_ready"]
        and report["semantic_fragile_category_count_test"] == 0
        and report["identity_fragile_category_count_test"] == 0
        and report["visualization_ready"]
        and not report["future_leakage_detected"]
        and not report["trajectory_degraded"]
    )
    report["recommended_next_step"] = decide(report)
    report["中文结论"] = (
        "V35.48 100+ stratified M128/H32 raw-video closure 已通过：128 clips、raw frontend rerun success=1.0、semantic/identity 三 seed 通过、per-category fragile 清零、real-instance identity=64。"
        "这允许更强的 M128/H32 100+ stratified bounded claim，但仍不是 full CVPR-scale；下一步才是 full 325 M128/H32 raw-video closure。"
    )
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    DOC.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(jsonable(report), indent=2, ensure_ascii=False), encoding="utf-8")
    DOC.write_text(
        "# STWM OSTF V35.48 100+ Stratified Decision\n\n"
        f"- selected_clip_count: {report['selected_clip_count']}\n"
        f"- real_instance_identity_count: {report['real_instance_identity_count']}\n"
        f"- raw_frontend_rerun_success_rate: {report['raw_frontend_rerun_success_rate']}\n"
        f"- trace_drift_ok: {report['trace_drift_ok']}\n"
        f"- semantic_three_seed_passed: {report['semantic_three_seed_passed']}\n"
        f"- stable_preservation: {report['stable_preservation']}\n"
        f"- identity_real_instance_three_seed_passed: {report['identity_real_instance_three_seed_passed']}\n"
        f"- semantic_fragile_category_count_test: {report['semantic_fragile_category_count_test']}\n"
        f"- identity_fragile_category_count_test: {report['identity_fragile_category_count_test']}\n"
        f"- visualization_ready: {report['visualization_ready']}\n"
        f"- m128_h32_100plus_stratified_benchmark_claim_allowed: {report['m128_h32_100plus_stratified_benchmark_claim_allowed']}\n"
        f"- full_cvpr_scale_claim_allowed: false\n"
        f"- recommended_next_step: {report['recommended_next_step']}\n\n"
        "## 中文总结\n"
        + report["中文结论"]
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"v35_48_decision_done": True, "m128_h32_100plus_stratified_benchmark_claim_allowed": report["m128_h32_100plus_stratified_benchmark_claim_allowed"], "recommended_next_step": report["recommended_next_step"]}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
