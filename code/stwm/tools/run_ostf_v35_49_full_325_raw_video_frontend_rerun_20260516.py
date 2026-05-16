#!/usr/bin/env python3
"""V35.49 full 325 raw-video frontend rerun，M128/H32。"""
from __future__ import annotations

import sys
from pathlib import Path

import setproctitle

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))
setproctitle.setproctitle("python")

from stwm.tools.ostf_v17_common_20260502 import ROOT  # noqa: E402
from stwm.tools import run_ostf_v35_45_larger_raw_video_frontend_rerun_20260516 as base  # noqa: E402

base.MANIFEST = ROOT / "outputs/cache/stwm_ostf_v35_49_full_325_raw_video_closure_manifest/manifest.json"
base.OUT_ROOT = ROOT / "outputs/cache/stwm_ostf_v35_49_full_325_raw_video_frontend_rerun/M128_H32"
base.REPORT = ROOT / "reports/stwm_ostf_v35_49_full_325_raw_video_frontend_rerun_20260516.json"
base.DOC = ROOT / "docs/STWM_OSTF_V35_49_FULL_325_RAW_VIDEO_FRONTEND_RERUN_20260516.md"
base.LOG = ROOT / "outputs/logs/stwm_ostf_v35_49_full_325_raw_video_frontend_rerun_20260516.log"


if __name__ == "__main__":
    raise SystemExit(base.main())
