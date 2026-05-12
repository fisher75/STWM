#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import setproctitle

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))

setproctitle.setproctitle("python")

from stwm.tools import render_ostf_v34_8_causal_assignment_residual_visualizations_20260512 as r348
from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, write_doc


TRAIN_SUMMARY = ROOT / "reports/stwm_ostf_v34_9_trace_fixed_oracle_residual_probe_train_summary_20260512.json"
REPORT = ROOT / "reports/stwm_ostf_v34_9_trace_fixed_residual_visualization_manifest_20260512.json"
DOC = ROOT / "docs/STWM_OSTF_V34_9_TRACE_FIXED_RESIDUAL_VISUALIZATION_20260512.md"
OUT_DIR = ROOT / "outputs/figures/stwm_ostf_v34_9_trace_fixed_residual"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()
    r348.TRAIN_SUMMARY = TRAIN_SUMMARY
    r348.REPORT = REPORT
    r348.DOC = DOC
    r348.OUT_DIR = OUT_DIR
    old_argv = sys.argv
    sys.argv = [old_argv[0], "--batch-size", str(args.batch_size), "--num-workers", str(args.num_workers)] + (["--cpu"] if args.cpu else [])
    try:
        r348.main()
    finally:
        sys.argv = old_argv
    if REPORT.exists():
        payload = json.loads(REPORT.read_text(encoding="utf-8"))
        payload["中文结论"] = "V34.9 可视化从 trace-fixed causal residual eval batch 中挖掘 case；包含 trace contract 修复后 residual overlay。"
        dump_json(REPORT, payload)
        write_doc(DOC, "V34.9 trace-fixed residual 可视化中文报告", payload, ["中文结论", "real_images_rendered", "case_mining_used", "placeholder_only", "png_count", "visualization_ready", "examples"])
    print(f"已写出 V34.9 可视化 manifest: {REPORT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
