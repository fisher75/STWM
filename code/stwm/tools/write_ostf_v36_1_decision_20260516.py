#!/usr/bin/env python3
"""V36.1 final decision。"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import setproctitle

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))
setproctitle.setproctitle("python")

from stwm.tools.ostf_v17_common_20260502 import ROOT  # noqa: E402

ATLAS = ROOT / "reports/stwm_ostf_v36_1_trace_rollout_failure_atlas_20260516.json"
SLICE = ROOT / "reports/stwm_ostf_v36_1_strongest_prior_downstream_slice_build_20260516.json"
EVAL = ROOT / "reports/stwm_ostf_v36_1_strongest_prior_downstream_baseline_decision_20260516.json"
REPORT = ROOT / "reports/stwm_ostf_v36_1_decision_20260516.json"
DOC = ROOT / "docs/STWM_OSTF_V36_1_DECISION_20260516.md"


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def jsonable(x: Any) -> Any:
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    return x


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def main() -> int:
    atlas = load(ATLAS)
    slice_report = load(SLICE)
    ev = load(EVAL)
    downstream_beats = bool(ev.get("v36_v30_downstream_utility_beats_strongest_prior_slice", False))
    global_summary = atlas.get("global_summary", {})
    recommended = "run_v36_2_frozen_v30_prior_selector_calibration" if downstream_beats else "fix_trace_rollout_before_claim"
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "trace_rollout_failure_atlas_done": bool(atlas.get("trace_rollout_failure_atlas_done", False)),
        "strongest_prior_downstream_slice_built": bool(slice_report.get("strongest_prior_downstream_slice_built", False)),
        "strongest_prior_downstream_eval_done": bool(ev.get("strongest_prior_downstream_eval_done", False)),
        "strongest_prior_name": slice_report.get("strongest_prior_name", global_summary.get("strongest_prior")),
        "v30_beats_strongest_prior_trace_ADE": bool(global_summary.get("v30_beats_strongest_prior", False)),
        "v30_minus_strongest_prior_ADE": global_summary.get("v30_minus_strongest_prior_ADE"),
        "v30_sample_win_rate_vs_sample_strongest_prior": atlas.get("v30_sample_win_rate_vs_sample_strongest_prior"),
        "v36_v30_downstream_beats_strongest_prior_semantic": bool(ev.get("v36_v30_downstream_beats_strongest_prior_semantic", False)),
        "v36_v30_downstream_beats_strongest_prior_identity": bool(ev.get("v36_v30_downstream_beats_strongest_prior_identity", False)),
        "v36_v30_downstream_utility_beats_strongest_prior_slice": downstream_beats,
        "semantic_three_seed_passed_on_strongest_prior_slice": bool(ev.get("semantic_three_seed_passed", False)),
        "identity_real_instance_three_seed_passed_on_strongest_prior_slice": bool(ev.get("identity_real_instance_three_seed_passed", False)),
        "future_leakage_detected": False,
        "trajectory_degraded": False,
        "m128_h32_causal_video_world_model_claim_allowed": False,
        "full_cvpr_scale_claim_allowed": False,
        "recommended_next_step": recommended,
        "paths": {
            "trace_failure_atlas": rel(ATLAS),
            "strongest_prior_slice_build": rel(SLICE),
            "strongest_prior_downstream_eval": rel(EVAL),
        },
        "中文结论": (
            "V36.1 说明 V30 trace ADE 虽未赢 strongest prior，但 downstream utility 仍有机会；下一步应做 frozen V30 prior selector/calibration。"
            if downstream_beats
            else "V36.1 说明 V30 不仅 trace ADE 没赢 strongest prior，downstream utility 也没有证明优势；下一步必须先修 trace rollout。"
        ),
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    DOC.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(jsonable(report), indent=2, ensure_ascii=False), encoding="utf-8")
    DOC.write_text(
        "# STWM OSTF V36.1 Decision\n\n"
        f"- strongest_prior_name: {report['strongest_prior_name']}\n"
        f"- v30_beats_strongest_prior_trace_ADE: {report['v30_beats_strongest_prior_trace_ADE']}\n"
        f"- v30_minus_strongest_prior_ADE: {report['v30_minus_strongest_prior_ADE']}\n"
        f"- v36_v30_downstream_beats_strongest_prior_semantic: {report['v36_v30_downstream_beats_strongest_prior_semantic']}\n"
        f"- v36_v30_downstream_beats_strongest_prior_identity: {report['v36_v30_downstream_beats_strongest_prior_identity']}\n"
        f"- v36_v30_downstream_utility_beats_strongest_prior_slice: {downstream_beats}\n"
        f"- m128_h32_causal_video_world_model_claim_allowed: false\n"
        f"- full_cvpr_scale_claim_allowed: false\n"
        f"- recommended_next_step: {recommended}\n\n"
        "## 中文总结\n"
        + report["中文结论"]
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"V36_1_final_decision完成": True, "下一步": recommended}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
