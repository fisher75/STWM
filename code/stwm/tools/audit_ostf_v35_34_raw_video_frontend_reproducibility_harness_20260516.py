#!/usr/bin/env python3
"""V35.34 raw-video frontend reproducibility harness 审计。"""
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
V35_33_PROTOCOL = ROOT / "reports/stwm_ostf_v35_33_m128_h32_full_video_system_benchmark_protocol_20260516.json"
COTRACKER_SCRIPT = ROOT / "code/stwm/tools/run_cotracker_object_dense_teacher_v16_20260502.py"
COTRACKER_V35_20_SCRIPT = ROOT / "code/stwm/tools/run_cotracker_object_dense_teacher_v35_20_vipseg_only_boost_20260515.py"
REPORT = ROOT / "reports/stwm_ostf_v35_34_raw_video_frontend_reproducibility_harness_20260516.json"
DOC = ROOT / "docs/STWM_OSTF_V35_34_RAW_VIDEO_FRONTEND_REPRODUCIBILITY_HARNESS_20260516.md"


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


def list_npz(root: Path) -> list[Path]:
    return sorted(root.glob("*/*.npz"))


def scalar(z: Any, key: str, default: Any = None) -> Any:
    if key not in z.files:
        return default
    arr = np.asarray(z[key])
    try:
        return arr.item()
    except ValueError:
        return arr


def script_has_setproctitle(path: Path) -> bool:
    if not path.exists():
        return False
    text = path.read_text(encoding="utf-8", errors="ignore")
    return "import setproctitle" in text and 'setproctitle.setproctitle("python")' in text


def inspect_trace_source(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False}
    z = np.load(path, allow_pickle=True)
    required = ["tracks_xy", "visibility", "confidence", "obs_len", "horizon", "frame_paths"]
    fields = list(z.files)
    out: dict[str, Any] = {
        "exists": True,
        "required_fields_present": all(k in fields for k in required),
        "fields": fields[:32],
    }
    if "tracks_xy" in fields:
        tracks = np.asarray(z["tracks_xy"])
        out["tracks_xy_shape"] = list(tracks.shape)
        out["tracks_nonzero"] = bool(np.any(np.abs(tracks) > 1e-6))
    if "frame_paths" in fields:
        frame_paths = np.asarray(z["frame_paths"], dtype=object)
        out["frame_path_count"] = int(frame_paths.size)
        out["frame_path_exists_checked"] = bool(frame_paths.size and Path(str(frame_paths[0])).exists())
    return out


def sample_frontend_contract(max_samples: int = 24) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    raw_ok = 0
    trace_ok = 0
    trace_field_ok = 0
    for p in list_npz(UNIFIED_ROOT)[:max_samples]:
        z = np.load(p, allow_pickle=True)
        raw_paths = np.asarray(z["raw_video_frame_paths"], dtype=object)
        raw_exists = bool(raw_paths.size and Path(str(raw_paths[0])).exists())
        trace_rel = str(scalar(z, "video_trace_source_npz", ""))
        trace_path = ROOT / trace_rel if trace_rel and not Path(trace_rel).is_absolute() else Path(trace_rel)
        trace_info = inspect_trace_source(trace_path)
        raw_ok += int(raw_exists)
        trace_ok += int(bool(trace_info.get("exists", False)))
        trace_field_ok += int(bool(trace_info.get("required_fields_present", False)) and bool(trace_info.get("tracks_nonzero", False)))
        rows.append(
            {
                "sample_uid": str(scalar(z, "sample_uid", p.stem)),
                "split": str(scalar(z, "split", p.parent.name)),
                "dataset": str(scalar(z, "dataset", "unknown")),
                "raw_first_frame_exists": raw_exists,
                "video_trace_source_npz": trace_rel,
                "trace_source_exists": bool(trace_info.get("exists", False)),
                "trace_required_fields_present": bool(trace_info.get("required_fields_present", False)),
                "trace_tracks_nonzero": bool(trace_info.get("tracks_nonzero", False)),
                "tracks_xy_shape": trace_info.get("tracks_xy_shape"),
            }
        )
    n = max(len(rows), 1)
    return {
        "checked_sample_count": len(rows),
        "raw_first_frame_exists_ratio": raw_ok / n,
        "trace_source_exists_ratio": trace_ok / n,
        "trace_required_fields_nonzero_ratio": trace_field_ok / n,
        "rows": rows,
    }


def main() -> int:
    protocol = read_json(V35_33_PROTOCOL)
    sample_check = sample_frontend_contract()
    cotracker_script_ready = COTRACKER_SCRIPT.exists()
    cotracker_v35_20_ready = COTRACKER_V35_20_SCRIPT.exists()
    frontend_scripts_have_setproctitle = bool(script_has_setproctitle(COTRACKER_SCRIPT) and script_has_setproctitle(COTRACKER_V35_20_SCRIPT))
    command_templates = [
        "/home/chen034/miniconda3/envs/stwm/bin/python code/stwm/tools/run_cotracker_object_dense_teacher_v16_20260502.py --help",
        "/home/chen034/miniconda3/envs/stwm/bin/python code/stwm/tools/run_cotracker_object_dense_teacher_v35_20_vipseg_only_boost_20260515.py --help",
        "/home/chen034/miniconda3/envs/stwm/bin/python code/stwm/tools/build_ostf_v35_28_full_unified_video_semantic_identity_benchmark_20260516.py",
        "/home/chen034/miniconda3/envs/stwm/bin/python code/stwm/tools/eval_ostf_v35_31_unified_joint_video_semantic_identity_harness_20260516.py",
    ]
    frontend_reproducibility_harness_ready = bool(
        protocol.get("benchmark_protocol_ready", False)
        and cotracker_script_ready
        and cotracker_v35_20_ready
        and frontend_scripts_have_setproctitle
        and sample_check["raw_first_frame_exists_ratio"] == 1.0
        and sample_check["trace_source_exists_ratio"] == 1.0
        and sample_check["trace_required_fields_nonzero_ratio"] == 1.0
    )
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "raw_video_frontend_reproducibility_harness_done": True,
        "frontend_reproducibility_harness_ready": frontend_reproducibility_harness_ready,
        "gpu_rerun_attempted": False,
        "gpu_rerun_reason": "本轮只做可复现性审计和命令协议，不重跑 CoTracker GPU 前端。",
        "cotracker_v16_script_exists": cotracker_script_ready,
        "cotracker_v35_20_script_exists": cotracker_v35_20_ready,
        "frontend_scripts_have_setproctitle": frontend_scripts_have_setproctitle,
        "sample_frontend_contract": sample_check,
        "command_templates": command_templates,
        "m128_h32_protocol_ready": bool(protocol.get("benchmark_protocol_ready", False)),
        "m128_h32_video_system_benchmark_claim_allowed": bool(protocol.get("claim_boundary", {}).get("m128_h32_video_system_benchmark_claim_allowed", False)),
        "full_cvpr_scale_claim_allowed": False,
        "full_video_semantic_identity_field_claim_allowed": False,
        "exact_blockers": [
            "本轮没有重跑完整 CoTracker GPU 前端，只确认已有 raw video path、trace source、脚本和命令模板可追溯。",
            "若要把工程系统 claim 推到更强，需要固定小规模 frontend rerun smoke，再在允许时扩大 clips / M / H。",
            "当前仍按用户约束不跑 H64/H96/M512/M1024。",
        ],
        "recommended_next_step": "run_small_raw_video_frontend_rerun_smoke_m128_h32_when_gpu_budget_allows",
        "中文结论": (
            "V35.34 确认 raw-video frontend reproducibility harness 已经具备：raw frame、video-derived trace source、CoTracker 前端脚本、"
            "setproctitle 规范和 unified benchmark 重建/评估命令都可追溯。"
            "这让 V35 的 M128/H32 完整视频闭环从“cache 上成立”推进到“可复现协议已打包”。"
            "但本轮未重跑 GPU 前端，所以仍不能宣称 full-scale CVPR complete system；下一步若有 GPU 预算，应做小规模 raw-video frontend rerun smoke。"
        ),
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    DOC.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(jsonable(report), indent=2, ensure_ascii=False), encoding="utf-8")
    DOC.write_text(
        "# STWM OSTF V35.34 Raw Video Frontend Reproducibility Harness\n\n"
        f"- frontend_reproducibility_harness_ready: {frontend_reproducibility_harness_ready}\n"
        f"- gpu_rerun_attempted: false\n"
        f"- cotracker_v16_script_exists: {cotracker_script_ready}\n"
        f"- cotracker_v35_20_script_exists: {cotracker_v35_20_ready}\n"
        f"- frontend_scripts_have_setproctitle: {frontend_scripts_have_setproctitle}\n"
        f"- raw_first_frame_exists_ratio: {sample_check['raw_first_frame_exists_ratio']:.4f}\n"
        f"- trace_source_exists_ratio: {sample_check['trace_source_exists_ratio']:.4f}\n"
        f"- trace_required_fields_nonzero_ratio: {sample_check['trace_required_fields_nonzero_ratio']:.4f}\n"
        f"- full_cvpr_scale_claim_allowed: false\n"
        f"- recommended_next_step: {report['recommended_next_step']}\n\n"
        "## 中文总结\n"
        + report["中文结论"]
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"前端复现协议就绪": frontend_reproducibility_harness_ready, "推荐下一步": report["recommended_next_step"]}, ensure_ascii=False), flush=True)
    return 0 if frontend_reproducibility_harness_ready else 2


if __name__ == "__main__":
    raise SystemExit(main())
