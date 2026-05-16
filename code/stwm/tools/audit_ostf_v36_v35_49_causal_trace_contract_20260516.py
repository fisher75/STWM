#!/usr/bin/env python3
"""V36: 审计 V35.49 是否满足因果 past-only trace contract。"""
from __future__ import annotations

import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import setproctitle

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))
setproctitle.setproctitle("python")

from stwm.tools.ostf_v17_common_20260502 import ROOT  # noqa: E402

REPORT = ROOT / "reports/stwm_ostf_v36_v35_49_causal_trace_contract_audit_20260516.json"
DOC = ROOT / "docs/STWM_OSTF_V36_V35_49_CAUSAL_TRACE_CONTRACT_AUDIT_20260516.md"

FILES = {
    "cotracker_frontend": ROOT / "code/stwm/tools/run_cotracker_object_dense_teacher_v15c_20260502.py",
    "v35_45_frontend_wrapper": ROOT / "code/stwm/tools/run_ostf_v35_45_larger_raw_video_frontend_rerun_20260516.py",
    "v35_49_frontend_wrapper": ROOT / "code/stwm/tools/run_ostf_v35_49_full_325_raw_video_frontend_rerun_20260516.py",
    "v35_45_slice_builder": ROOT / "code/stwm/tools/build_ostf_v35_45_larger_rerun_unified_semantic_identity_slice_20260516.py",
    "v35_49_slice_builder": ROOT / "code/stwm/tools/build_ostf_v35_49_full_325_rerun_unified_semantic_identity_slice_20260516.py",
    "v35_45_smoke_common": ROOT / "code/stwm/tools/run_ostf_v35_35_raw_video_frontend_rerun_smoke_20260516.py",
    "v35_49_eval": ROOT / "code/stwm/tools/eval_ostf_v35_49_full_325_raw_video_closure_benchmark_20260516.py",
}


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


def grep_locations(path: Path, patterns: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        rows.append({"path": rel(path), "line": None, "text": "文件缺失"})
        return rows
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    compiled = [re.compile(p) for p in patterns]
    for i, line in enumerate(lines, start=1):
        if any(p.search(line) for p in compiled):
            rows.append({"path": rel(path), "line": i, "text": line.strip()})
    return rows


def file_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore") if path.exists() else ""


def scalar(z: np.lib.npyio.NpzFile, key: str, default: Any = None) -> Any:
    if key not in z.files:
        return default
    arr = np.asarray(z[key])
    return arr.item() if arr.shape == () else arr


def inspect_one_slice() -> dict[str, Any]:
    root = ROOT / "outputs/cache/stwm_ostf_v35_49_full_325_rerun_unified_slice/M128_H32"
    paths = sorted(root.glob("*/*.npz"))
    if not paths:
        return {"slice_present": False}
    p = paths[0]
    z = np.load(p, allow_pickle=True)
    return {
        "slice_present": True,
        "sample_path": rel(p),
        "has_future_points": "future_points" in z.files,
        "has_future_vis": "future_vis" in z.files,
        "has_future_conf": "future_conf" in z.files,
        "identity_feature_dim": int(np.asarray(z["identity_identity_input_features"]).shape[-1]) if "identity_identity_input_features" in z.files else None,
        "future_points_shape": list(np.asarray(z["future_points"]).shape) if "future_points" in z.files else None,
        "video_trace_source_npz": str(scalar(z, "video_trace_source_npz", "")),
    }


def main() -> int:
    frontend_text = file_text(FILES["cotracker_frontend"])
    smoke_text = file_text(FILES["v35_45_smoke_common"])
    semantic_text = file_text(ROOT / "code/stwm/tools/eval_ostf_v35_14_mask_video_semantic_state_predictability_20260515.py")
    identity_text = smoke_text

    frontend_reads_future_frames = bool("obs_len + args.horizon" in frontend_text or "args.obs_len + args.horizon" in frontend_text)
    cotracker_offline_sees_future_frames = bool("offline=True" in smoke_text or "offline=True" in frontend_text)
    future_points_from_full_clip_frontend = bool(
        'payload["future_points"] = tracks[:, :, obs_len' in smoke_text
        or "tracks[:, :, obs_len : obs_len + horizon]" in smoke_text
    )
    semantic_input_contains_future_trace = bool(
        "future_points = np.asarray" in semantic_text
        and "future_vis = np.asarray" in semantic_text
        and "fut_disp" in semantic_text
    )
    identity_input_contains_future_trace = bool(
        "def trace_features_from_payload" in identity_text
        and "future_points" in identity_text
        and "fut_disp" in identity_text
        and "identity_identity_input_features" in identity_text
    )
    v35_49_is_causal = bool(
        not frontend_reads_future_frames
        and not cotracker_offline_sees_future_frames
        and not future_points_from_full_clip_frontend
        and not semantic_input_contains_future_trace
        and not identity_input_contains_future_trace
    )
    upper_bound = not v35_49_is_causal

    locations = {
        "frontend_full_clip": grep_locations(
            FILES["cotracker_frontend"],
            [r"obs_len\s*\+\s*args\.horizon", r"query_frame", r"teacher_uses_full_obs_future_clip_as_target", r"CoTrackerPredictor"],
        ),
        "rerun_wrapper": grep_locations(FILES["v35_49_frontend_wrapper"], [r"base\.MANIFEST", r"base\.OUT_ROOT"]),
        "slice_future_trace": grep_locations(
            FILES["v35_45_smoke_common"],
            [r'payload\["future_points"\]', r'trace_features_from_payload', r'identity_identity_input_features', r'future_crossing_pair'],
        ),
        "semantic_future_features": grep_locations(
            ROOT / "code/stwm/tools/eval_ostf_v35_14_mask_video_semantic_state_predictability_20260515.py",
            [r"future_points", r"future_vis", r"future_conf", r"fut_disp", r"fut_step"],
        ),
    }
    slice_probe = inspect_one_slice()
    decision_path = ROOT / "reports/stwm_ostf_v35_49_full_325_raw_video_closure_decision_20260516.json"
    eval_path = ROOT / "reports/stwm_ostf_v35_49_full_325_raw_video_closure_benchmark_eval_summary_20260516.json"
    v35_decision = json.loads(decision_path.read_text(encoding="utf-8")) if decision_path.exists() else {}

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "v35_49_causal_trace_contract_audit_done": True,
        "frontend_reads_future_frames": frontend_reads_future_frames,
        "cotracker_offline_sees_future_frames": cotracker_offline_sees_future_frames,
        "future_points_from_full_clip_frontend": future_points_from_full_clip_frontend,
        "semantic_input_contains_future_trace": semantic_input_contains_future_trace,
        "identity_input_contains_future_trace": identity_input_contains_future_trace,
        "v35_49_is_causal_past_only_world_model": v35_49_is_causal,
        "v35_49_is_teacher_trace_upper_bound": upper_bound,
        "claim_boundary_requires_rename": upper_bound,
        "v35_49_previous_claim_field": {
            "m128_h32_full_325_video_system_benchmark_claim_allowed": v35_decision.get("m128_h32_full_325_video_system_benchmark_claim_allowed"),
            "full_cvpr_scale_claim_allowed": v35_decision.get("full_cvpr_scale_claim_allowed"),
        },
        "suggested_renamed_claim": {
            "m128_h32_full_325_teacher_trace_closure_claim_allowed": bool(v35_decision.get("m128_h32_full_325_video_system_benchmark_claim_allowed", False) and upper_bound),
            "m128_h32_causal_video_world_model_claim_allowed": False,
        },
        "source_reports": {
            "v35_49_decision": rel(decision_path),
            "v35_49_eval_summary": rel(eval_path),
        },
        "slice_probe": slice_probe,
        "exact_code_locations": locations,
        "recommended_fix": "build_v36_past_only_observed_trace_then_run_frozen_v30_rollout",
        "中文总结": (
            "V35.49 的 raw-video rerun 是 full-clip CoTracker teacher trace closure：frontend 读取 obs+future 帧，"
            "future_points 来自 full-clip tracks，并且 semantic/identity 输入特征会使用 future trace 字段。"
            "因此 V35.49 只能标注为 teacher-trace upper-bound，不是严格因果 past-only world model。"
        ),
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    DOC.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(jsonable(report), indent=2, ensure_ascii=False), encoding="utf-8")
    DOC.write_text(
        "# STWM OSTF V36 / V35.49 因果 Trace Contract 审计\n\n"
        f"- frontend_reads_future_frames: {frontend_reads_future_frames}\n"
        f"- cotracker_offline_sees_future_frames: {cotracker_offline_sees_future_frames}\n"
        f"- future_points_from_full_clip_frontend: {future_points_from_full_clip_frontend}\n"
        f"- semantic_input_contains_future_trace: {semantic_input_contains_future_trace}\n"
        f"- identity_input_contains_future_trace: {identity_input_contains_future_trace}\n"
        f"- v35_49_is_causal_past_only_world_model: {v35_49_is_causal}\n"
        f"- v35_49_is_teacher_trace_upper_bound: {upper_bound}\n"
        f"- claim_boundary_requires_rename: {upper_bound}\n"
        f"- recommended_fix: {report['recommended_fix']}\n\n"
        "## 中文总结\n"
        + report["中文总结"]
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"V36审计完成": True, "V35_49是teacher_trace_upper_bound": upper_bound, "下一步": report["recommended_fix"]}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
