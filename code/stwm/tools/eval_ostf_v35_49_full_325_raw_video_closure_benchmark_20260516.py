#!/usr/bin/env python3
"""V35.49 full 325 raw-video closure benchmark eval，不训练新 head。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import setproctitle

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))
setproctitle.setproctitle("python")

from stwm.tools.ostf_v17_common_20260502 import ROOT  # noqa: E402
from stwm.tools import eval_ostf_v35_45_larger_raw_video_closure_benchmark_20260516 as base  # noqa: E402

base.SLICE_ROOT = ROOT / "outputs/cache/stwm_ostf_v35_49_full_325_rerun_unified_slice/M128_H32"
base.MANIFEST = ROOT / "outputs/cache/stwm_ostf_v35_49_full_325_raw_video_closure_manifest/manifest.json"
base.RERUN_REPORT = ROOT / "reports/stwm_ostf_v35_49_full_325_raw_video_frontend_rerun_20260516.json"
base.EVAL_REPORT = ROOT / "reports/stwm_ostf_v35_49_full_325_raw_video_closure_benchmark_eval_summary_20260516.json"
base.DECISION_REPORT = ROOT / "reports/stwm_ostf_v35_49_full_325_raw_video_closure_benchmark_decision_20260516.json"
base.DOC = ROOT / "docs/STWM_OSTF_V35_49_FULL_325_RAW_VIDEO_CLOSURE_BENCHMARK_DECISION_20260516.md"


def main() -> int:
    code = base.main()
    for path in [base.EVAL_REPORT, base.DECISION_REPORT]:
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            data["current_completed_version"] = "V35.49"
            data["source_scale"] = "full_325_m128_h32_raw_video_closure"
            if path == base.DECISION_REPORT:
                data["中文结论"] = data.get("中文结论", "").replace("V35.45 larger", "V35.49 full 325")
            path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    base.DOC.write_text(
        "# STWM OSTF V35.49 Full 325 Raw-Video Closure Benchmark Decision\n\n"
        f"- eval_summary: {base.EVAL_REPORT.relative_to(ROOT)}\n"
        f"- decision_report: {base.DECISION_REPORT.relative_to(ROOT)}\n"
        "- train_new_model: false\n"
        "- run_h64_h96: false\n"
        "- run_m512_m1024: false\n\n"
        "## 中文总结\n"
        "V35.49 使用 V35.21 semantic adapter 三 seed 与 V35.29 identity retrieval head 三 seed，对 full 325 rerun unified slice 做 joint eval；本阶段不训练新模型。\n",
        encoding="utf-8",
    )
    return code


if __name__ == "__main__":
    raise SystemExit(main())
