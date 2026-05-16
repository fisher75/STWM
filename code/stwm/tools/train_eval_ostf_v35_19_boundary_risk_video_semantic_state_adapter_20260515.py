#!/usr/bin/env python3
"""V35.19 boundary-risk target 通过后 video semantic state adapter 训练入口。"""
from __future__ import annotations

import sys
from pathlib import Path

import setproctitle

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))
setproctitle.setproctitle("python")

from stwm.tools import train_eval_ostf_v35_14_video_semantic_state_adapter_20260515 as base
from stwm.tools.ostf_v17_common_20260502 import ROOT

base.TARGET_ROOT = ROOT / "outputs/cache/stwm_ostf_v35_19_boundary_risk_video_semantic_state_targets/M128_H32"
base.CKPT_DIR = ROOT / "outputs/checkpoints/stwm_ostf_v35_19_boundary_risk_video_semantic_state_adapter_h32_m128"
base.TRAIN_REPORT = ROOT / "reports/stwm_ostf_v35_19_boundary_risk_video_semantic_state_adapter_train_summary_20260515.json"
base.EVAL_REPORT = ROOT / "reports/stwm_ostf_v35_19_boundary_risk_video_semantic_state_adapter_eval_summary_20260515.json"
base.DECISION_REPORT = ROOT / "reports/stwm_ostf_v35_19_boundary_risk_video_semantic_state_adapter_decision_20260515.json"
base.DOC = ROOT / "docs/STWM_OSTF_V35_19_BOUNDARY_RISK_VIDEO_SEMANTIC_STATE_ADAPTER_DECISION_20260515.md"


if __name__ == "__main__":
    raise SystemExit(base.main())
