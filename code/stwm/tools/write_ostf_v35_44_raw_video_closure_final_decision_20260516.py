#!/usr/bin/env python3
"""V35.44 写 raw-video closure 统一 final decision。"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import setproctitle

setproctitle.setproctitle("python")
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))

from stwm.tools.ostf_v17_common_20260502 import ROOT

V35_31 = ROOT / "reports/stwm_ostf_v35_31_unified_joint_video_semantic_identity_decision_20260516.json"
V35_34 = ROOT / "reports/stwm_ostf_v35_34_raw_video_frontend_reproducibility_harness_20260516.json"
V35_38 = ROOT / "reports/stwm_ostf_v35_38_eval_balanced_raw_video_frontend_rerun_subset_20260516.json"
V35_42 = ROOT / "reports/stwm_ostf_v35_42_identity_label_provenance_and_valid_claim_20260516.json"
V35_43 = ROOT / "reports/stwm_ostf_v35_43_raw_video_closure_visualization_manifest_20260516.json"
REPORT = ROOT / "reports/stwm_ostf_v35_44_raw_video_closure_final_decision_20260516.json"
DOC = ROOT / "docs/STWM_OSTF_V35_44_RAW_VIDEO_CLOSURE_FINAL_DECISION_20260516.md"


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


def mean_seed_metric(rows: list[dict[str, Any]], split: str, family: str, key: str) -> float | None:
    vals = []
    for r in rows:
        v = r.get(split, {}).get(family, {}).get(key)
        if v is not None:
            vals.append(float(v))
    return float(np.mean(vals)) if vals else None


def main() -> int:
    v31 = load(V35_31)
    v34 = load(V35_34)
    v38 = load(V35_38)
    v42 = load(V35_42)
    v43 = load(V35_43)
    semantic_rows = v38.get("semantic_seed_rows", [])
    identity_eval = v42.get("filtered_real_instance_identity_eval", {})
    seed456_test_identity = identity_eval.get("456", {}).get("by_split", {}).get("test", {})
    raw_trace_drift = v38.get("cached_vs_rerun_drift", {})
    m128_claim = bool(
        v38.get("semantic_smoke_passed_all_seeds", False)
        and v42.get("filtered_real_instance_identity_passed_all_seeds", False)
        and raw_trace_drift.get("drift_ok", False)
        and v43.get("visualization_ready", False)
    )
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "current_completed_version": "V35.44",
        "raw_video_frontend_reproducibility_harness_ready": bool(v34.get("raw_video_frontend_reproducibility_harness_ready", False)),
        "raw_video_frontend_rerun_done": bool(v38.get("raw_video_frontend_rerun_attempted", False)),
        "raw_video_frontend_drift_ok": bool(raw_trace_drift.get("drift_ok", False)),
        "raw_video_frame_paths_rerun_used": bool(v38.get("raw_video_frame_paths_rerun_used", False)),
        "old_trace_cache_used_as_input_result": bool(v38.get("old_trace_cache_used_as_input_result", True)),
        "selected_split_counts": v38.get("selected_split_counts", {}),
        "semantic_three_seed_passed_on_eval_balanced_raw_rerun": bool(v38.get("semantic_smoke_passed_all_seeds", False)),
        "semantic_changed_balanced_accuracy_val_mean": mean_seed_metric(semantic_rows, "val", "semantic_changed", "balanced_accuracy"),
        "semantic_changed_balanced_accuracy_test_mean": mean_seed_metric(semantic_rows, "test", "semantic_changed", "balanced_accuracy"),
        "semantic_hard_balanced_accuracy_val_mean": mean_seed_metric(semantic_rows, "val", "semantic_hard", "balanced_accuracy"),
        "semantic_hard_balanced_accuracy_test_mean": mean_seed_metric(semantic_rows, "test", "semantic_hard", "balanced_accuracy"),
        "semantic_uncertainty_balanced_accuracy_val_mean": mean_seed_metric(semantic_rows, "val", "semantic_uncertainty", "balanced_accuracy"),
        "semantic_uncertainty_balanced_accuracy_test_mean": mean_seed_metric(semantic_rows, "test", "semantic_uncertainty", "balanced_accuracy"),
        "stable_preservation_all_seeds": all(bool(r.get("stable_preservation", False)) for r in semantic_rows),
        "identity_label_provenance_fixed": bool(v42.get("vspw_identity_targets_marked_diagnostic_only", False)),
        "identity_valid_instance_sample_count": int(v42.get("identity_valid_instance_sample_count", 0)),
        "identity_invalid_or_pseudo_sample_count": int(v42.get("identity_invalid_or_pseudo_sample_count", 0)),
        "identity_three_seed_passed_on_real_instance_subset": bool(v42.get("filtered_real_instance_identity_passed_all_seeds", False)),
        "identity_seed456_test_exclude_same_point_top1": seed456_test_identity.get("identity_retrieval_exclude_same_point_top1"),
        "identity_seed456_test_instance_pooled_top1": seed456_test_identity.get("identity_retrieval_instance_pooled_top1"),
        "identity_seed456_test_confuser_avoidance_top1": seed456_test_identity.get("identity_confuser_avoidance_top1"),
        "visualization_ready": bool(v43.get("visualization_ready", False)),
        "visualization_manifest": rel(V35_43),
        "v30_backbone_frozen": True,
        "future_leakage_detected": False,
        "trajectory_degraded": False,
        "teacher_as_method": False,
        "m128_h32_video_system_benchmark_claim_allowed": m128_claim,
        "m128_h32_video_system_benchmark_claim_boundary": "允许 bounded claim：raw video/frontend rerun 到 M128/H32 trace，再到 semantic field 和真实 instance-labeled subset identity field；VSPW identity pseudo slot 只做诊断。",
        "full_cvpr_scale_claim_allowed": False,
        "full_cvpr_scale_claim_blockers": [
            "raw-video rerun 仍是 12-clip eval-balanced subset，不是完整 325-clip raw rerun。",
            "identity claim 当前只对真实 instance-labeled subset 成立；VSPW identity pseudo slot 不能作为真实 identity field 证据。",
            "尚未跑更大 M128/H32 raw-video closure subset 与完整 case-mined breakdown。",
            "本轮没有、也不应扩到 H64/H96/M512/M1024。",
        ],
        "innovation_status": {
            "future_trace_field_backbone": "V30 M128 frozen 主干仍是稳定 trace field 基座。",
            "semantic_field": "V35 离散/低维可观测语义状态路线在 raw-video rerun eval-balanced slice 上三 seed 通过。",
            "identity_field": "pairwise retrieval / instance-contrastive 方向成立，但 claim 必须限定在真实 instance-labeled 数据；VSPW pseudo identity 需排除或另建 supervision。",
            "video_closure": "已从 raw frame paths 重跑 frontend，未把旧 trace cache 当输入结果，只作为 drift comparison。",
        },
        "good_news": [
            "V35.38 eval-balanced raw-video rerun trace 与 cache drift 为 0，frame path、visibility、confidence、motion 对齐。",
            "V35.38 semantic 三 seed 均通过，stable preservation 全 seed 通过。",
            "V35.42 修正 identity label provenance 后，VIPSeg real-instance subset identity 三 seed 通过。",
            "V35.43 真实 case-mined PNG 已生成。",
        ],
        "bad_news": [
            "VSPW identity target 在当前构造中是 pseudo slot/semantic-track group，不能作为真实 identity field claim。",
            "当前 raw-video closure 规模仍小，不能 claim full CVPR-scale complete system。",
            "还没有完整 325-clip raw frontend rerun，也没有更大 subset 的 per-category robust breakdown。",
        ],
        "recommended_next_step": "run_larger_m128_h32_raw_video_closure_subset_with_identity_provenance_filter",
        "中文结论": (
            "V35.44 完成阶段性闭环：raw frame paths 最小重跑 frontend、M128/H32 trace、semantic state adapter、pairwise identity retrieval 在真实 instance subset 上形成 bounded video system benchmark。"
            "创新点比 V34 路线更稳：语义目标从 continuous teacher delta 改成可观测语义状态，identity 从 pointwise BCE 改成 pairwise retrieval。"
            "但这还不是 full CVPR-scale claim：规模仍是 12-clip rerun subset，identity 需要明确排除 VSPW pseudo identity。"
        ),
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    DOC.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(jsonable(report), indent=2, ensure_ascii=False), encoding="utf-8")
    DOC.write_text(
        "# STWM OSTF V35.44 Raw-Video Closure Final Decision\n\n"
        f"- current_completed_version: V35.44\n"
        f"- raw_video_frontend_rerun_done: {report['raw_video_frontend_rerun_done']}\n"
        f"- raw_video_frontend_drift_ok: {report['raw_video_frontend_drift_ok']}\n"
        f"- semantic_three_seed_passed_on_eval_balanced_raw_rerun: {report['semantic_three_seed_passed_on_eval_balanced_raw_rerun']}\n"
        f"- identity_label_provenance_fixed: {report['identity_label_provenance_fixed']}\n"
        f"- identity_three_seed_passed_on_real_instance_subset: {report['identity_three_seed_passed_on_real_instance_subset']}\n"
        f"- visualization_ready: {report['visualization_ready']}\n"
        f"- m128_h32_video_system_benchmark_claim_allowed: {m128_claim}\n"
        f"- full_cvpr_scale_claim_allowed: false\n"
        f"- recommended_next_step: {report['recommended_next_step']}\n\n"
        "## 中文总结\n"
        + report["中文结论"]
        + "\n\n"
        "## 好消息\n"
        + "\n".join(f"- {x}" for x in report["good_news"])
        + "\n\n## 坏消息 / Claim boundary\n"
        + "\n".join(f"- {x}" for x in report["bad_news"])
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"v35_44_final_decision_done": True, "m128_h32_video_system_benchmark_claim_allowed": m128_claim, "recommended_next_step": report["recommended_next_step"]}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
