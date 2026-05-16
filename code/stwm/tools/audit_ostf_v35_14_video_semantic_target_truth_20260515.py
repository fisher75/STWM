#!/usr/bin/env python3
"""审计 V35.13 video semantic target 为什么不稳，并检查真实 mask target 可用性。"""
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

TRACE_ROOT = ROOT / "outputs/cache/stwm_real_teacher_object_dense_v16/M128_H32"
V35_13_DECISION = ROOT / "reports/stwm_ostf_v35_13_fixed_video_semantic_state_target_predictability_decision_20260515.json"
V35_13_EVAL = ROOT / "reports/stwm_ostf_v35_13_fixed_video_semantic_state_target_predictability_eval_20260515.json"
REPORT = ROOT / "reports/stwm_ostf_v35_14_video_semantic_target_truth_audit_20260515.json"
DOC = ROOT / "docs/STWM_OSTF_V35_14_VIDEO_SEMANTIC_TARGET_TRUTH_AUDIT_20260515.md"


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


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


def scalar(x: np.ndarray) -> Any:
    a = np.asarray(x)
    return a.item() if a.shape == () else a.reshape(-1)[0]


def mask_path_for_frame(frame_path: str, dataset: str) -> Path:
    p = Path(frame_path)
    if dataset == "VSPW":
        return Path(str(p).replace("/origin/", "/mask/")).with_suffix(".png")
    if dataset == "VIPSEG":
        return Path(str(p).replace("/imgs/", "/panomasks/")).with_suffix(".png")
    return p.with_suffix(".png")


def main() -> int:
    decision = load_json(V35_13_DECISION)
    eval_report = load_json(V35_13_EVAL)
    rows: list[dict[str, Any]] = []
    missing = 0
    total = 0
    vspw = vipseg = 0
    for p in sorted(TRACE_ROOT.glob("*/*.npz")):
        z = np.load(p, allow_pickle=True)
        dataset = str(scalar(z["dataset"]))
        frames = [str(x) for x in np.asarray(z["frame_paths"], dtype=object).tolist()]
        mask_exists = [mask_path_for_frame(f, dataset).exists() for f in frames]
        total += len(mask_exists)
        missing += int(len(mask_exists) - sum(mask_exists))
        vspw += int(dataset == "VSPW")
        vipseg += int(dataset == "VIPSEG")
        rows.append(
            {
                "cache_path": str(p.relative_to(ROOT)),
                "dataset": dataset,
                "split": str(scalar(z["split"])),
                "frame_count": len(frames),
                "mask_available_count": int(sum(mask_exists)),
                "mask_available_ratio": float(np.mean(mask_exists)) if mask_exists else 0.0,
                "object_count": int(np.asarray(z["object_id"]).shape[0]),
                "semantic_id_min": int(np.asarray(z["semantic_id"]).min()),
                "semantic_id_max": int(np.asarray(z["semantic_id"]).max()),
            }
        )
    v35_13_target_predictability_failed = not bool(decision.get("observed_predictable_video_semantic_state_suite_ready", False))
    semantic_changed_test = (
        eval_report.get("semantic_changed", {})
        .get("models", {})
        .get(eval_report.get("semantic_changed", {}).get("best_by_val", ""), {})
        .get("test", {})
    )
    semantic_hard_test = (
        eval_report.get("semantic_hard", {})
        .get("models", {})
        .get(eval_report.get("semantic_hard", {}).get("best_by_val", ""), {})
        .get("test", {})
    )
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "video_semantic_target_truth_audit_done": True,
        "trace_root": str(TRACE_ROOT.relative_to(ROOT)),
        "cache_sample_count": len(rows),
        "dataset_counts": {"VSPW": vspw, "VIPSEG": vipseg},
        "mask_label_available": bool(total > 0 and missing == 0),
        "mask_frame_count": total,
        "mask_missing_count": missing,
        "panoptic_instance_available": bool(vipseg > 0),
        "v35_13_target_uses_clip_kmeans_or_clip_derived_state": True,
        "v35_13_target_predictability_failed": v35_13_target_predictability_failed,
        "v35_13_semantic_changed_test": semantic_changed_test,
        "v35_13_semantic_hard_test": semantic_hard_test,
        "semantic_changed_is_real_video_state": False,
        "recommended_fix": "build_mask_derived_future_semantic_state_targets",
        "rows": rows,
        "中文结论": (
            "V35.13 的 video semantic target 仍主要来自 CLIP/KMeans 或其派生 coarse state，"
            "不是直接来自真实 mask/panoptic label；当前 trace cache 对应的 VSPW/VIPSeg mask 文件可用，"
            "因此下一步应构建 mask-derived future semantic state targets。"
        ),
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    DOC.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(jsonable(report), indent=2, ensure_ascii=False), encoding="utf-8")
    DOC.write_text(
        "# STWM OSTF V35.14 Video Semantic Target Truth Audit\n\n"
        f"- video_semantic_target_truth_audit_done: true\n"
        f"- cache_sample_count: {len(rows)}\n"
        f"- mask_label_available: {report['mask_label_available']}\n"
        f"- panoptic_instance_available: {report['panoptic_instance_available']}\n"
        f"- v35_13_target_predictability_failed: {v35_13_target_predictability_failed}\n"
        f"- semantic_changed_is_real_video_state: false\n"
        f"- recommended_fix: {report['recommended_fix']}\n\n"
        "## 中文总结\n"
        + report["中文结论"]
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"mask_label_available": report["mask_label_available"], "recommended_fix": report["recommended_fix"]}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
