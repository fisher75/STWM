#!/usr/bin/env python3
"""V35.15 扩展版 mask-derived video semantic state targets 构建入口。"""
from __future__ import annotations

import sys
from pathlib import Path

import setproctitle

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))
setproctitle.setproctitle("python")

from stwm.tools import build_ostf_v35_14_mask_derived_video_semantic_state_targets_20260515 as base
from stwm.tools.ostf_v17_common_20260502 import ROOT

base.OUT_ROOT = ROOT / "outputs/cache/stwm_ostf_v35_15_expanded_mask_derived_video_semantic_state_targets/M128_H32"
base.REPORT = ROOT / "reports/stwm_ostf_v35_15_expanded_mask_derived_video_semantic_state_target_build_20260515.json"
base.DOC = ROOT / "docs/STWM_OSTF_V35_15_EXPANDED_MASK_DERIVED_VIDEO_SEMANTIC_STATE_TARGET_BUILD_20260515.md"


if __name__ == "__main__":
    raise SystemExit(base.main())
