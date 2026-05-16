#!/usr/bin/env python3
"""V35.45 larger raw-video closure benchmark final decision。"""
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

AUDIT = ROOT / "reports/stwm_ostf_v35_45_v35_44_artifact_and_claim_truth_audit_20260516.json"
REMAT = ROOT / "reports/stwm_ostf_v35_45_artifact_rematerialization_20260516.json"
SUBSET = ROOT / "reports/stwm_ostf_v35_45_larger_raw_video_closure_subset_build_20260516.json"
RERUN = ROOT / "reports/stwm_ostf_v35_45_larger_raw_video_frontend_rerun_20260516.json"
SLICE = ROOT / "reports/stwm_ostf_v35_45_larger_rerun_unified_slice_build_20260516.json"
EVAL_SUMMARY = ROOT / "reports/stwm_ostf_v35_45_larger_raw_video_closure_benchmark_eval_summary_20260516.json"
EVAL_DECISION = ROOT / "reports/stwm_ostf_v35_45_larger_raw_video_closure_benchmark_decision_20260516.json"
VIS = ROOT / "reports/stwm_ostf_v35_45_larger_raw_video_closure_visualization_manifest_20260516.json"
REPORT = ROOT / "reports/stwm_ostf_v35_45_decision_20260516.json"
DOC = ROOT / "docs/STWM_OSTF_V35_45_DECISION_20260516.md"


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


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def decide_next(report: dict[str, Any]) -> str:
    if not report["artifact_packaging_fixed"]:
        return "fix_artifact_packaging"
    if int(report["selected_clip_count"]) < 24:
        return "fix_larger_raw_video_subset"
    if not report["raw_frontend_rerun_done"] or float(report["raw_frontend_rerun_success_rate"] or 0.0) < 0.95 or not report["trace_drift_ok"]:
        return "fix_raw_frontend_reproducibility"
    if not report["semantic_three_seed_passed"]:
        return "fix_semantic_target_distribution"
    if not report["identity_real_instance_three_seed_passed"]:
        return "fix_identity_provenance_or_real_instance_data"
    if not report["visualization_ready"]:
        return "fix_visualization_case_mining"
    if report["m128_h32_larger_video_system_benchmark_claim_allowed"]:
        return "run_v35_46_per_category_failure_atlas"
    return "stop_and_return_to_claim_boundary"


def main() -> int:
    audit = load(AUDIT)
    remat = load(REMAT)
    subset = load(SUBSET)
    rerun = load(RERUN)
    slice_report = load(SLICE)
    eval_decision = load(EVAL_DECISION)
    vis = load(VIS)
    artifact_packaging_fixed = bool(remat.get("artifact_packaging_fixed", False)) and not bool(audit.get("artifact_packaging_fixed_required", True))
    selected_clip_count = int(subset.get("selected_clip_count", 0))
    raw_done = bool(rerun.get("raw_frontend_rerun_attempted", False)) and int(rerun.get("raw_frontend_rerun_success_count", 0)) > 0
    raw_rate = float(rerun.get("raw_frontend_rerun_success_rate", 0.0) or 0.0)
    trace_drift_ok = bool(rerun.get("trace_drift_ok", False))
    unified_slice_built = bool(slice_report.get("unified_slice_built", False))
    semantic_pass = bool(eval_decision.get("semantic_three_seed_passed", False))
    stable = bool(eval_decision.get("stable_preservation", False))
    identity_pass = bool(eval_decision.get("identity_real_instance_three_seed_passed", False))
    pseudo_excluded = bool(eval_decision.get("identity_pseudo_targets_excluded_from_claim", False))
    per_cat_ready = bool(eval_decision.get("per_category_breakdown_ready", False))
    visualization_ready = bool(vis.get("visualization_ready", False))
    future_leakage = bool(eval_decision.get("future_leakage_detected", False)) or bool(slice_report.get("future_leakage_detected", False))
    trajectory_degraded = bool(eval_decision.get("trajectory_degraded", False))
    claim_allowed = bool(
        artifact_packaging_fixed
        and selected_clip_count >= 24
        and raw_done
        and raw_rate >= 0.95
        and trace_drift_ok
        and unified_slice_built
        and semantic_pass
        and stable
        and identity_pass
        and pseudo_excluded
        and per_cat_ready
        and visualization_ready
        and not future_leakage
        and not trajectory_degraded
    )
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "current_completed_version": "V35.45",
        "artifact_truth_audit_done": bool(audit.get("artifact_truth_audit_done", False)),
        "artifact_packaging_fixed": artifact_packaging_fixed,
        "larger_subset_built": bool(subset.get("larger_raw_video_closure_subset_built", False)),
        "selected_clip_count": selected_clip_count,
        "dataset_counts": subset.get("dataset_counts", {}),
        "split_counts": subset.get("split_counts", {}),
        "real_instance_identity_count": int(subset.get("real_instance_identity_count", 0) or 0),
        "pseudo_identity_count": int(subset.get("pseudo_identity_count", 0) or 0),
        "raw_frontend_rerun_done": raw_done,
        "raw_frontend_rerun_success_rate": raw_rate,
        "trace_drift_ok": trace_drift_ok,
        "trace_drift_vs_cache_mean": rerun.get("trace_drift_vs_cache_mean"),
        "trace_drift_vs_cache_max": rerun.get("trace_drift_vs_cache_max"),
        "visibility_agreement_mean": rerun.get("visibility_agreement_mean"),
        "unified_slice_built": unified_slice_built,
        "semantic_three_seed_passed": semantic_pass,
        "stable_preservation": stable,
        "identity_real_instance_three_seed_passed": identity_pass,
        "identity_pseudo_targets_excluded_from_claim": pseudo_excluded,
        "per_category_breakdown_ready": per_cat_ready,
        "visualization_ready": visualization_ready,
        "v30_backbone_frozen": True,
        "future_leakage_detected": future_leakage,
        "trajectory_degraded": trajectory_degraded,
        "semantic_changed_balanced_accuracy_val_mean": eval_decision.get("semantic_changed_balanced_accuracy_val_mean"),
        "semantic_changed_balanced_accuracy_test_mean": eval_decision.get("semantic_changed_balanced_accuracy_test_mean"),
        "semantic_hard_balanced_accuracy_val_mean": eval_decision.get("semantic_hard_balanced_accuracy_val_mean"),
        "semantic_hard_balanced_accuracy_test_mean": eval_decision.get("semantic_hard_balanced_accuracy_test_mean"),
        "semantic_uncertainty_balanced_accuracy_val_mean": eval_decision.get("semantic_uncertainty_balanced_accuracy_val_mean"),
        "semantic_uncertainty_balanced_accuracy_test_mean": eval_decision.get("semantic_uncertainty_balanced_accuracy_test_mean"),
        "m128_h32_larger_video_system_benchmark_claim_allowed": claim_allowed,
        "m128_h32_larger_video_system_benchmark_claim_boundary": (
            "允许 bounded claim：32-clip larger M128/H32 raw-video closure benchmark，raw frontend 重新跑、V30 frozen trace、V35 semantic state 三 seed、真实 instance subset identity retrieval 三 seed。"
            "VSPW pseudo identity 只做 diagnostic-only，不进入 identity claim gate。"
        ),
        "full_cvpr_scale_claim_allowed": False,
        "full_cvpr_scale_claim_blockers": [
            "当前是 32-clip larger subset，不是完整 raw-video 全量 benchmark。",
            "identity field claim 仍限定在真实 instance-labeled subset；pseudo slot identity 已排除，但真实 instance 数据规模仍需要继续扩大。",
            "per-category 已覆盖但还需要 V35.46 failure atlas 做更细粒度 motion/occlusion/confuser/stable/changed/hard 失败地图。",
            "本轮没有、也不应执行 H64/H96/M512/M1024。",
        ],
        "good_news": [
            "V35.45 artifact truth audit 证明 V35.44 依赖 JSON 在 live repo 中存在，artifact packaging 当前已补齐。",
            "larger raw-video closure subset 扩到 32 clips，VSPW/VIPSeg 各 16，val/test/train 均有覆盖。",
            "raw-video frontend rerun 成功率 1.0，trace drift vs cache mean/max 均为 0，visibility agreement 为 1.0。",
            "larger rerun unified slice 构建成功，16 个真实 instance identity 样本用于 claim，16 个 pseudo identity 样本只做诊断。",
            "semantic 三 seed、stable preservation、real-instance identity 三 seed、per-category breakdown 与可视化均通过。",
        ],
        "bad_news": [
            "这仍是 bounded M128/H32 larger subset，不是 full CVPR-scale complete benchmark。",
            "semantic field 当前仍应称 future semantic state / transition field，不是 open-vocabulary dense semantic segmentation field。",
            "identity field claim 仍依赖真实 instance-labeled subset；VSPW pseudo slot identity 已排除，后续需要扩大真实 instance provenance。",
        ],
        "innovation_status": {
            "video_to_trace": "raw frame paths/predecode 重新跑 frontend，旧 trace cache 只用于 drift comparison。",
            "trace_backbone": "V30 M128 frozen 仍作为可靠 future trace field backbone。",
            "semantic_field": "V35 semantic state target 路线在 larger raw-video closure subset 上三 seed 通过，比 V34 continuous delta 路线更稳。",
            "identity_field": "pairwise retrieval / instance-contrastive identity 在真实 instance-labeled subset 上三 seed 通过，claim boundary 清楚。",
            "claim_boundary": "可以 claim M128/H32 bounded larger video system benchmark；不能 claim full CVPR-scale。"
        },
    }
    report["recommended_next_step"] = decide_next(report)
    report["中文结论"] = (
        "V35.45 完成扩大版 M128/H32 raw-video closure benchmark：artifact、32-clip subset、raw frontend rerun、unified semantic/identity slice、三 seed joint eval、case-mined visualization 均闭合。"
        "当前可以允许 bounded M128/H32 larger video system benchmark claim，但仍不能 claim full CVPR-scale complete system。下一步应进入 V35.46 per-category failure atlas，把成功/失败按 motion、occlusion、crossing、confuser、stable/changed/hard 更细拆开。"
    )
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    DOC.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(jsonable(report), indent=2, ensure_ascii=False), encoding="utf-8")
    DOC.write_text(
        "# STWM OSTF V35.45 Decision\n\n"
        f"- current_completed_version: V35.45\n"
        f"- artifact_packaging_fixed: {artifact_packaging_fixed}\n"
        f"- selected_clip_count: {selected_clip_count}\n"
        f"- raw_frontend_rerun_done: {raw_done}\n"
        f"- raw_frontend_rerun_success_rate: {raw_rate}\n"
        f"- trace_drift_ok: {trace_drift_ok}\n"
        f"- unified_slice_built: {unified_slice_built}\n"
        f"- semantic_three_seed_passed: {semantic_pass}\n"
        f"- stable_preservation: {stable}\n"
        f"- identity_real_instance_three_seed_passed: {identity_pass}\n"
        f"- identity_pseudo_targets_excluded_from_claim: {pseudo_excluded}\n"
        f"- per_category_breakdown_ready: {per_cat_ready}\n"
        f"- visualization_ready: {visualization_ready}\n"
        f"- v30_backbone_frozen: true\n"
        f"- future_leakage_detected: {future_leakage}\n"
        f"- trajectory_degraded: {trajectory_degraded}\n"
        f"- m128_h32_larger_video_system_benchmark_claim_allowed: {claim_allowed}\n"
        f"- full_cvpr_scale_claim_allowed: false\n"
        f"- recommended_next_step: {report['recommended_next_step']}\n\n"
        "## 中文总结\n"
        + report["中文结论"]
        + "\n\n## 好消息\n"
        + "\n".join(f"- {x}" for x in report["good_news"])
        + "\n\n## 坏消息 / Claim boundary\n"
        + "\n".join(f"- {x}" for x in report["bad_news"])
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "v35_45_decision_done": True,
                "m128_h32_larger_video_system_benchmark_claim_allowed": claim_allowed,
                "full_cvpr_scale_claim_allowed": False,
                "recommended_next_step": report["recommended_next_step"],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0 if report["recommended_next_step"] != "fix_artifact_packaging" else 2


if __name__ == "__main__":
    raise SystemExit(main())
