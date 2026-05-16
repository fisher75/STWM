#!/usr/bin/env python3
"""V35.27 将 video identity pairwise targets 扩展到 V35.24 的 325 clips。"""
from __future__ import annotations

import sys
from pathlib import Path

import setproctitle

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))
setproctitle.setproctitle("python")

from stwm.tools import build_ostf_v35_16_video_identity_pairwise_retrieval_targets_20260515 as base
from stwm.tools.ostf_v17_common_20260502 import ROOT

base.TARGET_ROOT = ROOT / "outputs/cache/stwm_ostf_v35_24_balanced_cross_dataset_changed_targets/M128_H32"
base.OUT_ROOT = ROOT / "outputs/cache/stwm_ostf_v35_27_expanded_video_identity_pairwise_retrieval_targets/M128_H32"
base.REPORT = ROOT / "reports/stwm_ostf_v35_27_expanded_video_identity_pairwise_target_build_20260516.json"
base.DOC = ROOT / "docs/STWM_OSTF_V35_27_EXPANDED_VIDEO_IDENTITY_PAIRWISE_TARGET_BUILD_20260516.md"


if __name__ == "__main__":
    raise SystemExit(base.main())
