#!/usr/bin/env python3
"""V35.15 扩展版 mask-derived video semantic state predictability 审计入口。"""
from __future__ import annotations

import sys
from pathlib import Path

import setproctitle

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))
setproctitle.setproctitle("python")

from stwm.tools import eval_ostf_v35_14_mask_video_semantic_state_predictability_20260515 as base
from stwm.tools.ostf_v17_common_20260502 import ROOT

base.TARGET_ROOT = ROOT / "outputs/cache/stwm_ostf_v35_15_expanded_mask_derived_video_semantic_state_targets/M128_H32"
base.EVAL_REPORT = ROOT / "reports/stwm_ostf_v35_15_expanded_mask_video_semantic_state_predictability_eval_20260515.json"
base.DECISION_REPORT = ROOT / "reports/stwm_ostf_v35_15_expanded_mask_video_semantic_state_predictability_decision_20260515.json"
base.DOC = ROOT / "docs/STWM_OSTF_V35_15_EXPANDED_MASK_VIDEO_SEMANTIC_STATE_PREDICTABILITY_DECISION_20260515.md"


if __name__ == "__main__":
    raise SystemExit(base.main())
