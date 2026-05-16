#!/usr/bin/env python3
"""审计 V35.8 到 raw/video-derived 输入闭环的真实状态。"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))

from stwm.tools.ostf_v17_common_20260502 import ROOT

V35_REPLICATION = ROOT / "reports/stwm_ostf_v35_8_identity_retrieval_replication_decision_20260515.json"
MEASUREMENT_BANK = ROOT / "outputs/cache/stwm_ostf_v34_9_trace_preserving_semantic_measurement_bank/pointodyssey"
V35_TARGETS = ROOT / "outputs/cache/stwm_ostf_v35_1_fixed_semantic_state_targets/pointodyssey"
COTRACKER_V16 = ROOT / "reports/stwm_cotracker_object_dense_teacher_v16_20260502.json"
FSTF_VIDEO_AUDIT = ROOT / "reports/stwm_fstf_video_input_pipeline_audit_v9_20260501.json"
OUT = ROOT / "reports/stwm_ostf_v35_9_video_input_closure_audit_20260515.json"
DOC = ROOT / "docs/STWM_OSTF_V35_9_VIDEO_INPUT_CLOSURE_AUDIT_20260515.md"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def sample_npz(root: Path, split: str = "val") -> dict[str, Any]:
    paths = sorted((root / split).glob("*.npz"))
    if not paths:
        return {"exists": False}
    z = np.load(paths[0], allow_pickle=True)
    return {
        "exists": True,
        "path": str(paths[0].relative_to(ROOT)),
        "files": list(z.files),
        "trace_source_npz": str(z["trace_source_npz"]) if "trace_source_npz" in z.files else None,
        "old_measurement_bank": str(z["old_measurement_bank"]) if "old_measurement_bank" in z.files else None,
        "obs_points_zero_ratio": float((np.asarray(z["obs_points"]) == 0).mean()) if "obs_points" in z.files else None,
        "obs_vis_mean": float(np.asarray(z["obs_vis"]).mean()) if "obs_vis" in z.files else None,
        "obs_conf_mean": float(np.asarray(z["obs_conf"]).mean()) if "obs_conf" in z.files else None,
        "has_raw_frame_paths": any("frame_path" in k or "raw_frame" in k for k in z.files),
        "future_teacher_embeddings_input_allowed": bool(z["future_teacher_embeddings_input_allowed"]) if "future_teacher_embeddings_input_allowed" in z.files else None,
        "leakage_safe": bool(z["leakage_safe"]) if "leakage_safe" in z.files else None,
    }


def main() -> None:
    v35 = read_json(V35_REPLICATION)
    cotracker = read_json(COTRACKER_V16)
    fstf = read_json(FSTF_VIDEO_AUDIT)
    bank_sample = sample_npz(MEASUREMENT_BANK)
    target_sample = sample_npz(V35_TARGETS)
    trace_source = str(bank_sample.get("trace_source_npz") or "")
    current_trace_source_external_gt = "stwm_ostf_v30_external_gt" in trace_source
    current_trace_source_video_derived = any(x in trace_source.lower() for x in ["cotracker", "traceanything", "video"])
    v16_per_combo = cotracker.get("per_combo", {})
    video_trace_frontend_available = bool(cotracker.get("success_gate_passed", False) and cotracker.get("real_teacher_tracks_exist", False))
    m128_h32_video_trace_cache_available = bool((ROOT / "outputs/cache/stwm_real_teacher_object_dense_v16/M128_H32").exists())
    m128_h16_video_trace_cache_available = bool(v16_per_combo.get("M128_H16", {}).get("success_gate_passed", False))
    m128_h8_video_trace_cache_available = bool(v16_per_combo.get("M128_H8", {}).get("success_gate_passed", False))
    trace_state_contract_passed = bool(
        bank_sample.get("exists")
        and bank_sample.get("obs_points_zero_ratio") is not None
        and float(bank_sample["obs_points_zero_ratio"]) < 0.01
        and bank_sample.get("future_teacher_embeddings_input_allowed") is False
        and bank_sample.get("leakage_safe") is True
    )
    raw_video_input_closed_for_v35 = bool(current_trace_source_video_derived and bank_sample.get("has_raw_frame_paths") and m128_h32_video_trace_cache_available)
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "v35_8_identity_semantic_replicated": bool(v35.get("semantic_state_head_passed_all") and v35.get("identity_retrieval_passed_all")),
        "v35_8_integrated_identity_field_claim_allowed_current_contract": bool(v35.get("integrated_identity_field_claim_allowed")),
        "v35_8_integrated_semantic_field_claim_allowed": bool(v35.get("integrated_semantic_field_claim_allowed")),
        "current_v35_input_contract": "PointOdyssey external-GT observed trace + observed semantic measurement bank",
        "current_trace_source_external_gt": current_trace_source_external_gt,
        "current_trace_source_video_derived": current_trace_source_video_derived,
        "trace_state_contract_passed": trace_state_contract_passed,
        "measurement_bank_sample": bank_sample,
        "v35_target_sample": target_sample,
        "video_derived_trace_frontend_available": video_trace_frontend_available,
        "cotracker_v16_success_gate_passed": bool(cotracker.get("success_gate_passed", False)),
        "cotracker_v16_processed_clip_count": int(cotracker.get("processed_clip_count", 0) or 0),
        "cotracker_v16_point_count": int(cotracker.get("point_count", 0) or 0),
        "m128_h8_video_trace_cache_available": m128_h8_video_trace_cache_available,
        "m128_h16_video_trace_cache_available": m128_h16_video_trace_cache_available,
        "m128_h32_video_trace_cache_available": m128_h32_video_trace_cache_available,
        "old_fstf_video_input_claim_allowed": bool(fstf.get("video_input_claim_allowed", False)),
        "raw_video_input_closed_for_v35": raw_video_input_closed_for_v35,
        "future_leakage_detected": bool(v35.get("future_leakage_detected", False)) or bool(fstf.get("future_leakage_detected", False)),
        "trajectory_degraded": bool(v35.get("trajectory_degraded", False)),
        "full_video_semantic_identity_field_claim_allowed": False,
        "exact_blockers": [
            "V35.8 当前训练/评估 trace_source_npz 指向 PointOdyssey external-GT cache，不是 video-derived trace cache。",
            "V35.8 当前 measurement bank 样本没有 raw frame path 字段，不能逐样本追溯 raw video closure。",
            "已存在 CoTracker video-derived object-dense trace frontend，但可用报告主要是 M128_H8/H16，不是 V35 当前 M128_H32 cache。",
            "unit_memory/assignment load-bearing 没有在 V35.8 三 seed 全过，不能把 unit memory 作为稳健创新主证据。",
        ],
        "recommended_next_step": "build_v35_video_derived_m128_h32_trace_measurement_cache",
        "secondary_next_step": "fix_unit_assignment_load_bearing",
        "中文结论": (
            "V35.8 已经在 external-GT trace + observed semantic measurement 合同下完成 identity/semantic state 三 seed 复现，"
            "但 V35.8 还没有完成 raw/video-derived 输入闭环。下一步应构建 V35 专用 M128/H32 video-derived trace + semantic measurement cache，"
            "再用同一 V35.8 checkpoint/eval 协议做 video-input closure smoke；同时另行处理 unit/assignment load-bearing 未复现问题。"
        ),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    DOC.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    DOC.write_text(
        "# STWM OSTF V35.9 Video Input Closure Audit\n\n"
        f"- v35_8_identity_semantic_replicated: {report['v35_8_identity_semantic_replicated']}\n"
        f"- current_trace_source_external_gt: {current_trace_source_external_gt}\n"
        f"- current_trace_source_video_derived: {current_trace_source_video_derived}\n"
        f"- trace_state_contract_passed: {trace_state_contract_passed}\n"
        f"- video_derived_trace_frontend_available: {video_trace_frontend_available}\n"
        f"- m128_h32_video_trace_cache_available: {m128_h32_video_trace_cache_available}\n"
        f"- raw_video_input_closed_for_v35: {raw_video_input_closed_for_v35}\n"
        f"- full_video_semantic_identity_field_claim_allowed: false\n"
        f"- recommended_next_step: {report['recommended_next_step']}\n"
        f"- secondary_next_step: {report['secondary_next_step']}\n\n"
        "## 中文总结\n"
        + report["中文结论"]
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"raw_video_input_closed_for_v35": raw_video_input_closed_for_v35, "recommended_next_step": report["recommended_next_step"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
