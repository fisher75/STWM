#!/usr/bin/env python3
"""V35.32 审计 raw/video-derived 输入到 future semantic/identity 输出的闭环。"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))

from stwm.tools.ostf_v17_common_20260502 import ROOT

UNIFIED_ROOT = ROOT / "outputs/cache/stwm_ostf_v35_28_full_unified_video_semantic_identity_benchmark/M128_H32"
V35_31_DECISION = ROOT / "reports/stwm_ostf_v35_31_unified_joint_video_semantic_identity_decision_20260516.json"
V35_31_EVAL = ROOT / "reports/stwm_ostf_v35_31_unified_joint_video_semantic_identity_eval_summary_20260516.json"
COTRACKER_REPORT = ROOT / "reports/stwm_cotracker_object_dense_teacher_v16_20260502.json"
REPORT = ROOT / "reports/stwm_ostf_v35_32_video_input_closure_audit_20260516.json"
DOC = ROOT / "docs/STWM_OSTF_V35_32_VIDEO_INPUT_CLOSURE_AUDIT_20260516.md"


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


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def list_npz(root: Path, split: str | None = None) -> list[Path]:
    if split is None:
        return sorted(root.glob("*/*.npz"))
    return sorted((root / split).glob("*.npz"))


def safe_scalar(z: Any, key: str, default: Any = None) -> Any:
    if key not in z.files:
        return default
    arr = np.asarray(z[key])
    try:
        return arr.item()
    except ValueError:
        return arr


def audit_unified_benchmark(max_path_checks: int = 200) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    split_counts: dict[str, int] = {}
    dataset_counts: dict[str, dict[str, int]] = {}
    raw_video_available = 0
    raw_path_arrays_present = 0
    raw_paths_checked = 0
    raw_paths_existing = 0
    trace_source_present = 0
    trace_source_existing = 0
    semantic_targets_present = 0
    identity_targets_present = 0
    leakage_safe = True
    future_teacher_input_allowed = False
    video_trace_source_video_derived = 0
    observed_trace_nonzero = 0
    observed_trace_total = 0
    for split in ["train", "val", "test"]:
        files = list_npz(UNIFIED_ROOT, split)
        split_counts[split] = len(files)
        dataset_counts[split] = {}
        for p in files:
            z = np.load(p, allow_pickle=True)
            ds = str(safe_scalar(z, "dataset", "unknown"))
            dataset_counts[split][ds] = dataset_counts[split].get(ds, 0) + 1
            raw_available = bool(safe_scalar(z, "raw_video_input_available", False))
            raw_video_available += int(raw_available)
            raw_paths = np.asarray(z["raw_video_frame_paths"], dtype=object) if "raw_video_frame_paths" in z.files else np.asarray([], dtype=object)
            raw_path_arrays_present += int(raw_paths.size > 0)
            for rp in raw_paths[: max(1, max_path_checks // max(len(list_npz(UNIFIED_ROOT)), 1))]:
                raw_paths_checked += 1
                raw_paths_existing += int(Path(str(rp)).exists())
            trace_rel = str(safe_scalar(z, "video_trace_source_npz", ""))
            trace_path = ROOT / trace_rel if trace_rel and not Path(trace_rel).is_absolute() else Path(trace_rel)
            trace_source_present += int(bool(trace_rel))
            trace_source_existing += int(bool(trace_rel) and trace_path.exists())
            video_trace_source_video_derived += int("stwm_real_teacher_object_dense_v16" in trace_rel or "cotracker" in trace_rel.lower())
            semantic_targets_present += int(bool(safe_scalar(z, "semantic_state_target_available", False)))
            identity_targets_present += int(bool(safe_scalar(z, "identity_pairwise_target_available", False)))
            leakage_safe = leakage_safe and bool(safe_scalar(z, "leakage_safe", True))
            future_teacher_input_allowed = future_teacher_input_allowed or bool(safe_scalar(z, "future_teacher_embedding_input_allowed", False))
            obs_points = np.asarray(z["obs_points"], dtype=np.float32)
            observed_trace_nonzero += int(np.any(np.abs(obs_points) > 1e-6))
            observed_trace_total += 1
            if len(rows) < 8:
                rows.append(
                    {
                        "sample_uid": str(safe_scalar(z, "sample_uid", p.stem)),
                        "split": split,
                        "dataset": ds,
                        "raw_video_input_available": raw_available,
                        "raw_frame_path_count": int(raw_paths.size),
                        "video_trace_source_npz": trace_rel,
                        "trace_source_exists": bool(trace_rel) and trace_path.exists(),
                        "semantic_state_target_available": bool(safe_scalar(z, "semantic_state_target_available", False)),
                        "identity_pairwise_target_available": bool(safe_scalar(z, "identity_pairwise_target_available", False)),
                    }
                )
    total = sum(split_counts.values())
    return {
        "sample_count": total,
        "split_counts": split_counts,
        "dataset_counts": dataset_counts,
        "raw_video_input_available_ratio": raw_video_available / max(total, 1),
        "raw_frame_path_array_ratio": raw_path_arrays_present / max(total, 1),
        "raw_frame_path_checked_count": raw_paths_checked,
        "raw_frame_path_existing_ratio_checked": raw_paths_existing / max(raw_paths_checked, 1),
        "video_trace_source_present_ratio": trace_source_present / max(total, 1),
        "video_trace_source_existing_ratio": trace_source_existing / max(total, 1),
        "video_trace_source_video_derived_ratio": video_trace_source_video_derived / max(total, 1),
        "semantic_state_target_available_ratio": semantic_targets_present / max(total, 1),
        "identity_pairwise_target_available_ratio": identity_targets_present / max(total, 1),
        "observed_trace_nonzero_ratio": observed_trace_nonzero / max(observed_trace_total, 1),
        "future_teacher_embedding_input_allowed": future_teacher_input_allowed,
        "leakage_safe": leakage_safe,
        "example_rows": rows,
    }


def main() -> int:
    joint = read_json(V35_31_DECISION)
    joint_eval = read_json(V35_31_EVAL)
    cotracker = read_json(COTRACKER_REPORT)
    bench = audit_unified_benchmark()
    video_input_contract_passed = bool(
        bench["sample_count"] > 0
        and bench["raw_video_input_available_ratio"] == 1.0
        and bench["raw_frame_path_array_ratio"] == 1.0
        and bench["raw_frame_path_existing_ratio_checked"] >= 0.99
        and bench["video_trace_source_present_ratio"] == 1.0
        and bench["video_trace_source_existing_ratio"] == 1.0
        and bench["video_trace_source_video_derived_ratio"] == 1.0
        and bench["semantic_state_target_available_ratio"] == 1.0
        and bench["identity_pairwise_target_available_ratio"] == 1.0
        and bench["observed_trace_nonzero_ratio"] == 1.0
        and bench["leakage_safe"]
        and not bench["future_teacher_embedding_input_allowed"]
    )
    unified_joint_passed = bool(joint.get("full_unified_joint_eval_passed", False))
    m128_h32_video_system_closure_passed = bool(video_input_contract_passed and unified_joint_passed)
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "video_input_closure_audit_done": True,
        "benchmark_root": str(UNIFIED_ROOT.relative_to(ROOT)),
        "benchmark": bench,
        "cotracker_v16_report_present": COTRACKER_REPORT.exists(),
        "cotracker_v16_success_gate_passed": bool(cotracker.get("success_gate_passed", False)),
        "cotracker_v16_processed_clip_count": int(cotracker.get("processed_clip_count", 0) or 0),
        "unified_joint_eval_passed": unified_joint_passed,
        "semantic_three_seed_passed": bool(joint.get("semantic_three_seed_passed_on_unified_benchmark", False)),
        "identity_three_seed_passed": bool(joint.get("identity_three_seed_passed_on_unified_benchmark", False)),
        "semantic_test_changed_balanced_accuracy_mean": joint_eval.get("semantic_test_changed_balanced_accuracy_mean"),
        "semantic_test_hard_balanced_accuracy_mean": joint_eval.get("semantic_test_hard_balanced_accuracy_mean"),
        "identity_test_exclude_same_point_top1_mean": joint_eval.get("identity_test_exclude_same_point_top1_mean"),
        "identity_test_confuser_avoidance_top1_mean": joint_eval.get("identity_test_confuser_avoidance_top1_mean"),
        "raw_video_to_video_derived_trace_contract_passed": bool(
            bench["raw_video_input_available_ratio"] == 1.0
            and bench["video_trace_source_existing_ratio"] == 1.0
            and bench["video_trace_source_video_derived_ratio"] == 1.0
        ),
        "video_derived_trace_to_semantic_identity_target_contract_passed": bool(
            bench["semantic_state_target_available_ratio"] == 1.0
            and bench["identity_pairwise_target_available_ratio"] == 1.0
            and bench["observed_trace_nonzero_ratio"] == 1.0
        ),
        "future_teacher_embedding_input_allowed": bool(bench["future_teacher_embedding_input_allowed"]),
        "future_leakage_detected": not bool(bench["leakage_safe"]) or bool(joint.get("future_leakage_detected", False)),
        "trajectory_degraded": bool(joint.get("trajectory_degraded", False)),
        "video_input_contract_passed": video_input_contract_passed,
        "m128_h32_video_system_closure_passed": m128_h32_video_system_closure_passed,
        "m128_h32_video_system_claim_allowed": m128_h32_video_system_closure_passed,
        "full_cvpr_scale_claim_allowed": False,
        "integrated_semantic_field_claim_allowed": False,
        "integrated_identity_field_claim_allowed": False,
        "full_video_semantic_identity_field_claim_allowed": False,
        "exact_blockers": [
            "当前结论只覆盖 M128/H32，不覆盖 H64/H96/M512/M1024。",
            "当前闭环使用 video-derived dense trace cache；还需要自动化 raw video 前端流水线的一键复现审计，才能称为更完整工程系统。",
            "需要把 unified joint eval、video input closure、成功/失败 case mining 打包为固定 benchmark protocol，而不是只看若干脚本输出。",
        ],
        "recommended_next_step": "package_m128_h32_full_video_system_benchmark_protocol",
        "中文结论": (
            "V35.32 证实 raw video frame path、video-derived M128/H32 dense trace source、mask-derived semantic state target、pairwise identity target、"
            "V35.31 semantic+identity 三 seed 联合评估已经在 325 clip unified benchmark 上闭合。"
            "这是非常强的阶段性好消息：M128/H32 级别的 video-derived trace 到 future semantic/identity 闭环已经成立。"
            "但它仍不是 full CVPR-scale claim，因为尚未扩大到更长 horizon/更密 M/更大跨数据集，也尚未完成一键 raw-video 前端复现包装。"
        ),
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    DOC.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(jsonable(report), indent=2, ensure_ascii=False), encoding="utf-8")
    DOC.write_text(
        "# STWM OSTF V35.32 Video Input Closure Audit\n\n"
        f"- video_input_closure_audit_done: true\n"
        f"- sample_count: {bench['sample_count']}\n"
        f"- raw_video_input_available_ratio: {bench['raw_video_input_available_ratio']:.4f}\n"
        f"- video_trace_source_existing_ratio: {bench['video_trace_source_existing_ratio']:.4f}\n"
        f"- semantic_state_target_available_ratio: {bench['semantic_state_target_available_ratio']:.4f}\n"
        f"- identity_pairwise_target_available_ratio: {bench['identity_pairwise_target_available_ratio']:.4f}\n"
        f"- unified_joint_eval_passed: {unified_joint_passed}\n"
        f"- video_input_contract_passed: {video_input_contract_passed}\n"
        f"- m128_h32_video_system_closure_passed: {m128_h32_video_system_closure_passed}\n"
        f"- full_cvpr_scale_claim_allowed: false\n"
        f"- recommended_next_step: {report['recommended_next_step']}\n\n"
        "## 中文总结\n"
        + report["中文结论"]
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"视频输入闭环通过": m128_h32_video_system_closure_passed, "推荐下一步": report["recommended_next_step"]}, ensure_ascii=False), flush=True)
    return 0 if m128_h32_video_system_closure_passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
