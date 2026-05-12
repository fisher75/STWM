#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import setproctitle

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))

setproctitle.setproctitle("python")

from stwm.tools import train_ostf_v34_8_causal_assignment_oracle_residual_probe_20260512 as v348
from stwm.tools.ostf_v17_common_20260502 import ROOT, write_doc


SUMMARY = ROOT / "reports/stwm_ostf_v34_9_trace_fixed_oracle_residual_probe_train_summary_20260512.json"
DOC = ROOT / "docs/STWM_OSTF_V34_9_TRACE_FIXED_ORACLE_RESIDUAL_PROBE_TRAIN_SUMMARY_20260512.md"
CKPT_DIR = ROOT / "outputs/checkpoints/stwm_ostf_v34_9_trace_fixed_oracle_residual_probe_h32_m128"
TARGET_REPORT = ROOT / "reports/stwm_ostf_v34_9_causal_assignment_residual_target_build_20260512.json"
MEAS_ROOT = ROOT / "outputs/cache/stwm_ostf_v34_9_trace_preserving_semantic_measurement_bank/pointodyssey"
TARGET_ROOT = ROOT / "outputs/cache/stwm_ostf_v34_9_causal_assignment_residual_targets/pointodyssey"


def main() -> int:
    args = v348.parse_args()
    args.semantic_measurement_bank_root = str(MEAS_ROOT)
    args.causal_assignment_residual_target_root = str(TARGET_ROOT)
    v348.SUMMARY = SUMMARY
    v348.DOC = DOC
    v348.CKPT_DIR = CKPT_DIR
    v348.TARGET_REPORT = TARGET_REPORT
    payload = v348.train_one(args)
    if SUMMARY.exists():
        data = json.loads(SUMMARY.read_text(encoding="utf-8"))
        data["中文结论"] = "V34.9 trace-fixed causal assignment oracle residual probe 已完成训练；measurement bank 使用真实 trace state；未训练 learned gate。"
        SUMMARY.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        write_doc(DOC, "V34.9 trace-fixed oracle residual probe 训练中文摘要", data, ["中文结论", "oracle_residual_probe_ran", "fresh_training_completed", "checkpoint_path", "train_sample_count", "v30_backbone_frozen", "pointwise_base_frozen", "future_leakage_detected", "train_loss_decreased"])
    print(f"已写出 V34.9 trace-fixed 训练摘要: {SUMMARY.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
