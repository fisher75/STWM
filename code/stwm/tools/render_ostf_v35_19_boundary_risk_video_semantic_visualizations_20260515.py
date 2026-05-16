#!/usr/bin/env python3
"""V35.19 boundary-risk video semantic target 可视化入口。"""
from __future__ import annotations

import sys
from pathlib import Path

import setproctitle

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))
setproctitle.setproctitle("python")

from stwm.tools import render_ostf_v35_18_vipseg_to_vspw_domain_shift_visualizations_20260515 as base
from stwm.tools.ostf_v17_common_20260502 import ROOT

base.TARGET_ROOT = ROOT / "outputs/cache/stwm_ostf_v35_19_boundary_risk_video_semantic_state_targets/M128_H32"
base.OUT_DIR = ROOT / "reports/visualizations/v35_19_boundary_risk_video_semantic"
base.MANIFEST = ROOT / "reports/stwm_ostf_v35_19_boundary_risk_video_semantic_visualization_manifest_20260515.json"
base.DOC = ROOT / "docs/STWM_OSTF_V35_19_BOUNDARY_RISK_VIDEO_SEMANTIC_VISUALIZATION_20260515.md"


if __name__ == "__main__":
    raise SystemExit(base.main())
