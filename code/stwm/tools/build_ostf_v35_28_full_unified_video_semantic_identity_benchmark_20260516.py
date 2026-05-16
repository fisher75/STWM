#!/usr/bin/env python3
"""V35.28 使用 V35.27 expanded identity targets 重建 full unified benchmark。"""
from __future__ import annotations

import sys
from pathlib import Path

import setproctitle

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))
setproctitle.setproctitle("python")

from stwm.tools import build_ostf_v35_26_unified_video_semantic_identity_benchmark_20260516 as base
from stwm.tools.ostf_v17_common_20260502 import ROOT

base.ID_ROOT = ROOT / "outputs/cache/stwm_ostf_v35_27_expanded_video_identity_pairwise_retrieval_targets/M128_H32"
base.OUT_ROOT = ROOT / "outputs/cache/stwm_ostf_v35_28_full_unified_video_semantic_identity_benchmark/M128_H32"
base.REPORT = ROOT / "reports/stwm_ostf_v35_28_full_unified_video_semantic_identity_benchmark_build_20260516.json"
base.DOC = ROOT / "docs/STWM_OSTF_V35_28_FULL_UNIFIED_VIDEO_SEMANTIC_IDENTITY_BENCHMARK_BUILD_20260516.md"


if __name__ == "__main__":
    raise SystemExit(base.main())
