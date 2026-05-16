#!/usr/bin/env python3
"""V36 final decision: causal past-only M128/H32 trace rollout closure。"""
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

from stwm.tools.ostf_v17_common_20260502 import ROOT  # noqa: E402

AUDIT = ROOT / "reports/stwm_ostf_v36_v35_49_causal_trace_contract_audit_20260516.json"
INPUT_BUILD = ROOT / "reports/stwm_ostf_v36_past_only_observed_trace_input_build_20260516.json"
ROLLOUT = ROOT / "reports/stwm_ostf_v36_v30_past_only_future_trace_rollout_20260516.json"
SLICE_BUILD = ROOT / "reports/stwm_ostf_v36_causal_unified_semantic_identity_slice_build_20260516.json"
BENCH_DECISION = ROOT / "reports/stwm_ostf_v36_causal_past_only_video_world_model_benchmark_decision_20260516.json"
VIS = ROOT / "reports/stwm_ostf_v36_causal_past_only_world_model_visualization_manifest_20260516.json"
REPORT = ROOT / "reports/stwm_ostf_v36_decision_20260516.json"
DOC = ROOT / "docs/STWM_OSTF_V36_DECISION_20260516.md"


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


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


def main() -> int:
    audit = load(AUDIT)
    inp = load(INPUT_BUILD)
    rollout = load(ROLLOUT)
    slice_build = load(SLICE_BUILD)
    bench = load(BENCH_DECISION)
    vis = load(VIS)
    teacher_upper_bound = bool(audit.get("v35_49_is_teacher_trace_upper_bound", False))
    causal_pass = bool(bench.get("causal_benchmark_passed", False))
    v30_beats = bool(rollout.get("v30_beats_strongest_prior", False))
    semantic_pass = bool(bench.get("semantic_three_seed_passed", False))
    identity_pass = bool(bench.get("identity_real_instance_three_seed_passed", False))
    stable = bool(bench.get("stable_preservation", False))
    leakage = bool(bench.get("future_leakage_detected", False))
    degraded = bool(bench.get("trajectory_degraded", False))
    visualization_ready = bool(vis.get("visualization_ready", False))
    teacher_gap = bench.get("v35_49_teacher_trace_upper_bound_gap", {})
    if causal_pass:
        recommended = "run_v36_seed123_replication"
    elif not bool(inp.get("obs_only_input_built", False)):
        recommended = "fix_past_only_trace_rollout"
    elif not bool(rollout.get("future_trace_predicted_from_past_only", False)):
        recommended = "fix_past_only_trace_rollout"
    elif not v30_beats:
        recommended = "fix_v30_vs_strongest_prior"
    elif not bool(slice_build.get("causal_unified_slice_built", False)):
        recommended = "fix_causal_semantic_identity_slice"
    elif not semantic_pass:
        recommended = "fix_semantic_state_on_predicted_trace"
    elif not identity_pass:
        recommended = "fix_identity_retrieval_on_predicted_trace"
    else:
        recommended = "stop_and_return_to_claim_boundary"

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "v35_49_causal_trace_contract_audit_done": bool(audit.get("v35_49_causal_trace_contract_audit_done", False)),
        "v35_49_is_teacher_trace_upper_bound": teacher_upper_bound,
        "past_only_observed_trace_input_built": bool(inp.get("obs_only_input_built", False)),
        "v30_past_only_future_trace_rollout_done": bool(rollout.get("v30_past_only_future_trace_rollout_done", False)),
        "future_trace_predicted_from_past_only": bool(rollout.get("future_trace_predicted_from_past_only", False)),
        "v30_beats_strongest_prior": v30_beats,
        "causal_unified_slice_built": bool(slice_build.get("causal_unified_slice_built", False)),
        "causal_benchmark_ran": bool(bench.get("causal_past_only_video_world_model_benchmark_done", False)),
        "causal_benchmark_passed": causal_pass,
        "teacher_trace_upper_bound_gap": teacher_gap,
        "semantic_three_seed_passed": semantic_pass,
        "stable_preservation": stable,
        "identity_real_instance_three_seed_passed": identity_pass,
        "identity_pseudo_targets_excluded_from_claim": True,
        "visualization_ready": visualization_ready,
        "v30_backbone_frozen": bool(rollout.get("v30_backbone_frozen", False)),
        "future_leakage_detected": leakage,
        "trajectory_degraded": degraded,
        "m128_h32_causal_video_world_model_claim_allowed": causal_pass,
        "m128_h32_teacher_trace_upper_bound_claim_allowed": teacher_upper_bound,
        "full_cvpr_scale_claim_allowed": False,
        "recommended_next_step": recommended,
        "paths": {
            "audit": rel(AUDIT),
            "past_only_input": rel(INPUT_BUILD),
            "rollout": rel(ROLLOUT),
            "causal_slice": rel(SLICE_BUILD),
            "benchmark_decision": rel(BENCH_DECISION),
            "visualization": rel(VIS),
        },
        "中文结论": (
            "V36 已把 V35.49 的 teacher-trace upper-bound 边界修正为真正 past-only V30 rollout 闭环，并且 causal benchmark 通过。"
            if causal_pass
            else "V36 已完成因果 contract 审计与 causal pipeline 重建，但当前 causal benchmark 未过；不能 claim M128/H32 causal video world model。"
        ),
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    DOC.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(jsonable(report), indent=2, ensure_ascii=False), encoding="utf-8")
    DOC.write_text(
        "# STWM OSTF V36 Decision\n\n"
        f"- v35_49_is_teacher_trace_upper_bound: {teacher_upper_bound}\n"
        f"- past_only_observed_trace_input_built: {report['past_only_observed_trace_input_built']}\n"
        f"- future_trace_predicted_from_past_only: {report['future_trace_predicted_from_past_only']}\n"
        f"- v30_beats_strongest_prior: {v30_beats}\n"
        f"- causal_unified_slice_built: {report['causal_unified_slice_built']}\n"
        f"- causal_benchmark_passed: {causal_pass}\n"
        f"- semantic_three_seed_passed: {semantic_pass}\n"
        f"- stable_preservation: {stable}\n"
        f"- identity_real_instance_three_seed_passed: {identity_pass}\n"
        f"- visualization_ready: {visualization_ready}\n"
        f"- m128_h32_causal_video_world_model_claim_allowed: {causal_pass}\n"
        f"- m128_h32_teacher_trace_upper_bound_claim_allowed: {teacher_upper_bound}\n"
        "- full_cvpr_scale_claim_allowed: false\n"
        f"- recommended_next_step: {recommended}\n\n"
        "## 中文总结\n"
        + report["中文结论"]
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"V36最终决策完成": True, "causal_claim_allowed": causal_pass, "下一步": recommended}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
