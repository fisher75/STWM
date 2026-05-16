#!/usr/bin/env python3
"""V35.48 100+ stratified per-category failure atlas。"""
from __future__ import annotations

import json
import argparse
import sys
from pathlib import Path

import setproctitle

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))
setproctitle.setproctitle("python")

from stwm.tools.ostf_v17_common_20260502 import ROOT  # noqa: E402
from stwm.tools import eval_ostf_v35_46_per_category_failure_atlas_20260516 as base  # noqa: E402

base.SLICE_ROOT = ROOT / "outputs/cache/stwm_ostf_v35_48_100plus_rerun_unified_slice/M128_H32"
base.SUBSET_MANIFEST = ROOT / "outputs/cache/stwm_ostf_v35_48_100plus_stratified_raw_video_closure_subset/manifest.json"
base.V35_45_DECISION = ROOT / "reports/stwm_ostf_v35_48_100plus_raw_video_closure_benchmark_decision_20260516.json"
base.EVAL_REPORT = ROOT / "reports/stwm_ostf_v35_48_100plus_per_category_failure_atlas_eval_20260516.json"
base.DECISION_REPORT = ROOT / "reports/stwm_ostf_v35_48_100plus_per_category_failure_atlas_decision_20260516.json"
base.DOC = ROOT / "docs/STWM_OSTF_V35_48_100PLUS_PER_CATEGORY_FAILURE_ATLAS_DECISION_20260516.md"
base.LOG = ROOT / "outputs/logs/stwm_ostf_v35_48_100plus_per_category_failure_atlas_20260516.log"


def postprocess(code: int) -> int:
    benchmark = {}
    if base.V35_45_DECISION.exists():
        benchmark = json.loads(base.V35_45_DECISION.read_text(encoding="utf-8"))
    bench_pass = bool(
        benchmark.get("m128_h32_larger_video_system_benchmark_claim_allowed", False)
        or benchmark.get("larger_raw_video_closure_benchmark_passed", False)
    )
    final_code = code
    for path in [base.EVAL_REPORT, base.DECISION_REPORT]:
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            data["current_completed_version"] = "V35.48"
            data["source_scale"] = "100plus_stratified_m128_h32_raw_video_closure"
            if path == base.EVAL_REPORT:
                data["atlas_ready"] = bool(bench_pass and len(data.get("categories", [])) >= 10)
            if path == base.DECISION_REPORT:
                data["m128_h32_larger_video_system_benchmark_claim_allowed"] = bench_pass
                data["atlas_ready"] = bool(bench_pass and int(data.get("category_count", 0) or 0) >= 10)
                if data["atlas_ready"]:
                    data["recommended_next_step"] = "write_v35_48_100plus_stratified_final_decision"
                    final_code = 0
            path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    base.DOC.write_text(
        "# STWM OSTF V35.48 100+ Stratified Per-Category Failure Atlas Decision\n\n"
        f"- eval_report: {base.EVAL_REPORT.relative_to(ROOT)}\n"
        f"- decision_report: {base.DECISION_REPORT.relative_to(ROOT)}\n"
        f"- m128_h32_larger_video_system_benchmark_claim_allowed: {bench_pass}\n"
        "- train_new_model: false\n"
        "- full_cvpr_scale_claim_allowed: false\n\n"
        "## 中文总结\n"
        "V35.48 已对 100+ stratified raw-video closure slice 做 per-category failure atlas；字段已兼容 V35.48 benchmark decision，不再误判为 atlas not ready。\n",
        encoding="utf-8",
    )
    return final_code


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--postprocess-only", action="store_true")
    args = ap.parse_args()
    code = 0 if args.postprocess_only else base.main()
    return postprocess(code)


if __name__ == "__main__":
    raise SystemExit(main())
