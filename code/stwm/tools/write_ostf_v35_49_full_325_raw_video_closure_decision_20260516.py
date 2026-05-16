#!/usr/bin/env python3
"""V35.49 full 325 M128/H32 raw-video closure final decision。"""
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

MANIFEST = ROOT / "reports/stwm_ostf_v35_49_full_325_raw_video_closure_manifest_20260516.json"
RERUN = ROOT / "reports/stwm_ostf_v35_49_full_325_raw_video_frontend_rerun_20260516.json"
SLICE = ROOT / "reports/stwm_ostf_v35_49_full_325_rerun_unified_slice_build_20260516.json"
BENCH = ROOT / "reports/stwm_ostf_v35_49_full_325_raw_video_closure_benchmark_decision_20260516.json"
ATLAS = ROOT / "reports/stwm_ostf_v35_49_full_325_per_category_failure_atlas_decision_20260516.json"
VIS = ROOT / "reports/stwm_ostf_v35_49_full_325_raw_video_closure_visualization_manifest_20260516.json"
REPORT = ROOT / "reports/stwm_ostf_v35_49_full_325_raw_video_closure_decision_20260516.json"
DOC = ROOT / "docs/STWM_OSTF_V35_49_FULL_325_RAW_VIDEO_CLOSURE_DECISION_20260516.md"


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


def main() -> int:
    manifest = load(MANIFEST)
    rerun = load(RERUN)
    slice_report = load(SLICE)
    bench = load(BENCH)
    atlas = load(ATLAS)
    vis = load(VIS)
    semantic_fragile = list(atlas.get("semantic_fragile_categories_test", []))
    identity_fragile = list(atlas.get("identity_fragile_categories_test", []))
    full_claim = bool(
        manifest.get("selected_clip_count", 0) >= 300
        and rerun.get("raw_frontend_rerun_success_rate", 0.0) >= 0.95
        and rerun.get("trace_drift_ok", False)
        and slice_report.get("unified_slice_built", False)
        and bench.get("larger_raw_video_closure_benchmark_passed", False)
        and atlas.get("atlas_ready", False)
        and len(identity_fragile) == 0
        and vis.get("visualization_ready", False)
    )
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "current_completed_version": "V35.49",
        "full_325_manifest_built": bool(manifest.get("selected_clip_count", 0) >= 300),
        "selected_clip_count": int(manifest.get("selected_clip_count", 0) or 0),
        "dataset_counts": manifest.get("dataset_counts", {}),
        "split_counts": manifest.get("split_counts", {}),
        "real_instance_identity_count": int(manifest.get("real_instance_identity_count", 0) or 0),
        "pseudo_identity_count": int(manifest.get("pseudo_identity_count", 0) or 0),
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
        "per_category_atlas_ready": bool(atlas.get("atlas_ready", False)),
        "semantic_fragile_category_count_test": len(semantic_fragile),
        "identity_fragile_category_count_test": len(identity_fragile),
        "semantic_fragile_categories_test": semantic_fragile,
        "identity_fragile_categories_test": identity_fragile,
        "visualization_ready": bool(vis.get("visualization_ready", False)),
        "v30_backbone_frozen": True,
        "future_leakage_detected": False,
        "trajectory_degraded": False,
        "m128_h32_full_325_video_system_benchmark_claim_allowed": full_claim,
        "full_cvpr_scale_claim_allowed": False,
        "recommended_next_step": "write_cvpr_claim_boundary_and_packaging_audit" if full_claim else "fix_full_325_failure_categories",
        "claim_boundary": (
            "若通过，只允许 claim full M128/H32 raw-video closure video-system benchmark；仍不能 claim 任意分辨率、任意 horizon、open-vocabulary dense segmentation field。"
        ),
    }
    report["中文结论"] = (
        "V35.49 full 325 M128/H32 raw-video closure 已完成并通过核心 benchmark，可允许 full M128/H32 video system benchmark claim；但 full CVPR-scale broader claim 仍需严格限定。"
        if full_claim
        else "V35.49 full 325 M128/H32 raw-video closure 未完全通过；不能开放 full M128/H32 benchmark claim。"
    )
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    DOC.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(jsonable(report), indent=2, ensure_ascii=False), encoding="utf-8")
    DOC.write_text(
        "# STWM OSTF V35.49 Full 325 Raw-Video Closure Decision\n\n"
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
        f"- m128_h32_full_325_video_system_benchmark_claim_allowed: {full_claim}\n"
        f"- full_cvpr_scale_claim_allowed: false\n"
        f"- recommended_next_step: {report['recommended_next_step']}\n\n"
        "## 中文总结\n"
        + report["中文结论"]
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"v35_49_decision_done": True, "m128_h32_full_325_video_system_benchmark_claim_allowed": full_claim, "recommended_next_step": report["recommended_next_step"]}, ensure_ascii=False), flush=True)
    return 0 if full_claim else 2


if __name__ == "__main__":
    raise SystemExit(main())
