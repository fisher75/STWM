#!/usr/bin/env python3
"""V35.49 full 325 raw-video closure case-mined 可视化。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import setproctitle

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))
setproctitle.setproctitle("python")

from stwm.tools.ostf_v17_common_20260502 import ROOT  # noqa: E402
from stwm.tools import render_ostf_v35_45_larger_raw_video_closure_visualizations_20260516 as base  # noqa: E402

base.SLICE_ROOT = ROOT / "outputs/cache/stwm_ostf_v35_49_full_325_rerun_unified_slice/M128_H32"
base.SUBSET_MANIFEST = ROOT / "outputs/cache/stwm_ostf_v35_49_full_325_raw_video_closure_manifest/manifest.json"
base.EVAL_REPORT = ROOT / "reports/stwm_ostf_v35_49_full_325_raw_video_closure_benchmark_eval_summary_20260516.json"
base.EVAL_DECISION = ROOT / "reports/stwm_ostf_v35_49_full_325_raw_video_closure_benchmark_decision_20260516.json"
base.FIG_ROOT = ROOT / "outputs/figures/stwm_ostf_v35_49_full_325_raw_video_closure"
base.REPORT = ROOT / "reports/stwm_ostf_v35_49_full_325_raw_video_closure_visualization_manifest_20260516.json"
base.DOC = ROOT / "docs/STWM_OSTF_V35_49_FULL_325_RAW_VIDEO_CLOSURE_VISUALIZATION_20260516.md"
base.LOG = ROOT / "outputs/logs/stwm_ostf_v35_49_full_325_raw_video_closure_visualization_20260516.log"


def main() -> int:
    code = base.main()
    if base.REPORT.exists():
        data = json.loads(base.REPORT.read_text(encoding="utf-8"))
        data["current_completed_version"] = "V35.49"
        data["source_scale"] = "full_325_m128_h32_raw_video_closure"
        data["中文结论"] = data.get("中文结论", "").replace("V35.45", "V35.49")
        base.REPORT.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
