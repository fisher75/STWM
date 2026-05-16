#!/usr/bin/env python3
"""V35.29 在 325-clip expanded video identity targets 上训练/评估 pairwise retrieval head。"""
from __future__ import annotations

import sys
from pathlib import Path

import setproctitle

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))
setproctitle.setproctitle("python")

from stwm.tools import train_eval_ostf_v35_16_video_identity_pairwise_retrieval_head_20260515 as base
from stwm.tools.ostf_v17_common_20260502 import ROOT

base.TARGET_ROOT = ROOT / "outputs/cache/stwm_ostf_v35_27_expanded_video_identity_pairwise_retrieval_targets/M128_H32"
base.CKPT_DIR = ROOT / "outputs/checkpoints/stwm_ostf_v35_29_expanded_video_identity_pairwise_retrieval_h32_m128"
base.TRAIN_REPORT = ROOT / "reports/stwm_ostf_v35_29_expanded_video_identity_pairwise_retrieval_train_summary_20260516.json"
base.EVAL_REPORT = ROOT / "reports/stwm_ostf_v35_29_expanded_video_identity_pairwise_retrieval_eval_summary_20260516.json"
base.DECISION_REPORT = ROOT / "reports/stwm_ostf_v35_29_expanded_video_identity_pairwise_retrieval_decision_20260516.json"
base.DOC = ROOT / "docs/STWM_OSTF_V35_29_EXPANDED_VIDEO_IDENTITY_PAIRWISE_RETRIEVAL_DECISION_20260516.md"
base.EXPERIMENT_ID = "V35.29"
base.EXPERIMENT_LABEL = "V35.29 expanded video identity pairwise retrieval"
base.CKPT_PREFIX = "v35_29_expanded_video_identity_pairwise_retrieval"
base.NEXT_STEP_ON_PASS = "aggregate_v35_29_expanded_identity_three_seed_replication"
base.NEXT_STEP_ON_FAIL = "fix_expanded_video_identity_pairwise_retrieval_head"


if __name__ == "__main__":
    raise SystemExit(base.main())
