#!/usr/bin/env python3
"""V35.49 full 325 rerun unified semantic/identity slice。"""
from __future__ import annotations

import sys
from pathlib import Path

import setproctitle

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))
setproctitle.setproctitle("python")

from stwm.tools.ostf_v17_common_20260502 import ROOT  # noqa: E402
from stwm.tools import build_ostf_v35_45_larger_rerun_unified_semantic_identity_slice_20260516 as base  # noqa: E402

base.MANIFEST = ROOT / "outputs/cache/stwm_ostf_v35_49_full_325_raw_video_closure_manifest/manifest.json"
base.RERUN_TRACE_ROOT = ROOT / "outputs/cache/stwm_ostf_v35_49_full_325_raw_video_frontend_rerun/M128_H32"
base.OUT_ROOT = ROOT / "outputs/cache/stwm_ostf_v35_49_full_325_rerun_unified_slice/M128_H32"
base.REPORT = ROOT / "reports/stwm_ostf_v35_49_full_325_rerun_unified_slice_build_20260516.json"
base.DOC = ROOT / "docs/STWM_OSTF_V35_49_FULL_325_RERUN_UNIFIED_SLICE_BUILD_20260516.md"


if __name__ == "__main__":
    raise SystemExit(base.main())
