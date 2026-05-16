#!/usr/bin/env python3
"""V35.49 构建 full 325 M128/H32 raw-video closure manifest。"""
from __future__ import annotations

import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import setproctitle

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))
setproctitle.setproctitle("python")

from stwm.tools.build_ostf_v35_45_larger_raw_video_closure_subset_20260516 import (  # noqa: E402
    UNIFIED_ROOT,
    list_npz,
    sample_row,
    trace_motion_from_source,
)
from stwm.tools.build_ostf_v35_48_100plus_stratified_raw_video_closure_subset_20260516 import risk_tags  # noqa: E402
from stwm.tools.ostf_v17_common_20260502 import ROOT  # noqa: E402

OUT_ROOT = ROOT / "outputs/cache/stwm_ostf_v35_49_full_325_raw_video_closure_manifest"
MANIFEST = OUT_ROOT / "manifest.json"
REPORT = ROOT / "reports/stwm_ostf_v35_49_full_325_raw_video_closure_manifest_20260516.json"
DOC = ROOT / "docs/STWM_OSTF_V35_49_FULL_325_RAW_VIDEO_CLOSURE_MANIFEST_20260516.md"


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


def retarget(row: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    split = out["split"]
    uid = out["sample_uid"]
    out["expected_rerun_trace_path"] = f"outputs/cache/stwm_ostf_v35_49_full_325_raw_video_frontend_rerun/M128_H32/{split}/{uid}.npz"
    out["selection_reason"] = "full_325_all_available_raw_video_closure_candidate"
    return out


def main() -> int:
    paths = []
    for p in list_npz(UNIFIED_ROOT):
        z = np.load(p, allow_pickle=True)
        if int(np.asarray(z["point_id"]).size) <= 1280:
            paths.append(p)
    motions = [trace_motion_from_source(p) for p in paths]
    median = float(np.median(motions)) if motions else 0.0
    rows = [retarget(sample_row(p, median)) for p in paths]
    rows = [r for r in rows if r["predecode_available"] and r["raw_frame_paths"]]
    rows = sorted(rows, key=lambda r: (r["split"], r["dataset"], r["sample_uid"]))
    blockers: list[str] = []
    if len(rows) < 300:
        blockers.append(f"full raw-video closure candidate 数不足：{len(rows)} < 300")
    counts = {
        "dataset_counts": dict(Counter(r["dataset"] for r in rows)),
        "split_counts": dict(Counter(r["split"] for r in rows)),
        "real_instance_identity_count": int(sum(r["identity_claim_allowed"] for r in rows)),
        "pseudo_identity_count": int(sum(r["identity_provenance_type"] == "pseudo_slot" for r in rows)),
        "semantic_changed_counts": int(sum("semantic_changed" in r["category_tags"] for r in rows)),
        "semantic_hard_counts": int(sum("semantic_hard" in r["category_tags"] for r in rows)),
        "stable_counts": int(sum("stable_heavy" in r["category_tags"] for r in rows)),
        "occlusion_count": int(sum("occlusion" in r["category_tags"] for r in rows)),
        "crossing_count": int(sum("crossing" in r["category_tags"] for r in rows)),
        "confuser_count": int(sum("identity_confuser" in r["category_tags"] for r in rows)),
        "high_motion_count": int(sum("high_motion" in r["category_tags"] for r in rows)),
        "risk_vipseg_changed_count": int(sum("risk_vipseg_changed" in risk_tags(r) for r in rows)),
        "risk_high_motion_hard_count": int(sum("risk_high_motion_hard" in risk_tags(r) for r in rows)),
        "risk_real_instance_semantic_changed_count": int(sum("risk_real_instance_semantic_changed" in risk_tags(r) for r in rows)),
    }
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "version": "V35.49",
        "m": 128,
        "horizon": 32,
        "selected_clip_count": len(rows),
        "target_scope": "full_available_325_m128_h32_raw_video_closure",
        "old_trace_cache_used_as_input_result": False,
        "old_trace_cache_used_for_comparison_only": True,
        "samples": rows,
        "exact_blockers": blockers,
        **counts,
        "中文结论": f"V35.49 full manifest 已构建：{len(rows)} clips；VSPW pseudo identity 保持 diagnostic-only，VIPSeg real-instance 可进入 identity claim gate。",
    }
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    DOC.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(jsonable(manifest), indent=2, ensure_ascii=False), encoding="utf-8")
    REPORT.write_text(json.dumps(jsonable(manifest), indent=2, ensure_ascii=False), encoding="utf-8")
    DOC.write_text(
        "# STWM OSTF V35.49 Full 325 Raw-Video Closure Manifest\n\n"
        f"- selected_clip_count: {len(rows)}\n"
        f"- dataset_counts: {counts['dataset_counts']}\n"
        f"- split_counts: {counts['split_counts']}\n"
        f"- real_instance_identity_count: {counts['real_instance_identity_count']}\n"
        f"- pseudo_identity_count: {counts['pseudo_identity_count']}\n"
        f"- exact_blockers: {blockers}\n\n"
        "## 中文总结\n"
        + manifest["中文结论"]
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"full_325_manifest_built": not blockers, "selected_clip_count": len(rows)}, ensure_ascii=False), flush=True)
    return 0 if not blockers else 2


if __name__ == "__main__":
    raise SystemExit(main())
