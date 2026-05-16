#!/usr/bin/env python3
"""V35.17 cross-dataset video identity pairwise target 构建入口。"""
from __future__ import annotations

import sys
from pathlib import Path

import setproctitle

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))
setproctitle.setproctitle("python")

from stwm.tools import build_ostf_v35_16_video_identity_pairwise_retrieval_targets_20260515 as base
from stwm.tools.ostf_v17_common_20260502 import ROOT

base.TARGET_ROOT = ROOT / "outputs/cache/stwm_ostf_v35_17_cross_dataset_mask_derived_video_semantic_state_targets/M128_H32"
base.OUT_ROOT = ROOT / "outputs/cache/stwm_ostf_v35_17_cross_dataset_video_identity_pairwise_retrieval_targets/M128_H32"
base.REPORT = ROOT / "reports/stwm_ostf_v35_17_cross_dataset_video_identity_pairwise_target_build_20260515.json"
base.DOC = ROOT / "docs/STWM_OSTF_V35_17_CROSS_DATASET_VIDEO_IDENTITY_PAIRWISE_TARGET_BUILD_20260515.md"


if __name__ == "__main__":
    raise SystemExit(base.main())
