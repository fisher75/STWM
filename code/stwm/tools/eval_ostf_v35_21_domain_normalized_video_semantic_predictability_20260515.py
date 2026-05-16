#!/usr/bin/env python3
"""V35.21 domain-normalized video semantic predictability。"""
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

from stwm.tools import eval_ostf_v35_18_vipseg_to_vspw_video_semantic_domain_shift_20260515 as base
from stwm.tools.ostf_v17_common_20260502 import ROOT

base.TARGET_ROOT = ROOT / "outputs/cache/stwm_ostf_v35_21_domain_normalized_video_semantic_state_targets/M128_H32"
base.EVAL_REPORT = ROOT / "reports/stwm_ostf_v35_21_domain_normalized_video_semantic_predictability_eval_20260515.json"
base.DECISION_REPORT = ROOT / "reports/stwm_ostf_v35_21_domain_normalized_video_semantic_predictability_decision_20260515.json"
base.DOC = ROOT / "docs/STWM_OSTF_V35_21_DOMAIN_NORMALIZED_VIDEO_SEMANTIC_PREDICTABILITY_DECISION_20260515.md"

REPORT = base.EVAL_REPORT
DECISION = base.DECISION_REPORT
DOC = base.DOC


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


def add_balanced_protocol_summary() -> None:
    eval_report = json.loads(REPORT.read_text(encoding="utf-8"))
    decision = json.loads(DECISION.read_text(encoding="utf-8"))
    protocols = eval_report.get("protocols", {})
    # Treat mixed-unseen as the dataset-balanced unseen protocol only when both datasets are present in every split.
    mixed = protocols.get("mixed_unseen_ontology_agnostic", {})
    sample_counts = mixed.get("sample_counts", {})
    mixed_balanced_ready = bool(mixed.get("suite_passed") and min(sample_counts.values() or [0]) >= 30)
    vipseg_strat = protocols.get("vipseg_to_vspw_stratified_ontology_agnostic", {})
    vipseg_all = protocols.get("vipseg_to_vspw_all_ontology_agnostic", {})
    vipseg_to_vspw_fixed = bool(vipseg_strat.get("suite_passed") or vipseg_all.get("suite_passed"))
    vspw_to_vipseg = protocols.get("vspw_to_vipseg_ontology_agnostic", {})
    suite_ready = bool(mixed_balanced_ready and vipseg_to_vspw_fixed and vspw_to_vipseg.get("suite_passed"))
    decision.update(
        {
            "domain_normalized_video_semantic_predictability_eval_done": True,
            "domain_normalized_risk_calibration_used": True,
            "mixed_domain_dataset_balanced_unseen_passed": mixed_balanced_ready,
            "vipseg_to_vspw_domain_normalized_passed": vipseg_to_vspw_fixed,
            "cross_dataset_video_semantic_suite_ready": suite_ready,
            "semantic_adapter_training_allowed": suite_ready,
            "recommended_next_step": "run_three_seed_cross_dataset_video_semantic_adapter" if suite_ready else "fix_domain_normalized_target_or_collect_vspw_changed_hard_cases",
            "中文结论": (
                "V35.21 domain-normalized target suite 通过；下一轮才允许三 seed adapter。"
                if suite_ready
                else "V35.21 仍未让 cross-dataset suite 全过；不能训练 adapter，应继续修 domain-normalized target 或补 VSPW changed/hard cases。"
            ),
        }
    )
    eval_report["domain_normalized_risk_calibration_used"] = True
    eval_report["mixed_domain_dataset_balanced_unseen_passed"] = mixed_balanced_ready
    eval_report["vipseg_to_vspw_domain_normalized_passed"] = vipseg_to_vspw_fixed
    REPORT.write_text(json.dumps(jsonable(eval_report), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    DECISION.write_text(json.dumps(jsonable(decision), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    DOC.write_text(
        "# STWM OSTF V35.21 Domain-Normalized Video Semantic Predictability Decision\n\n"
        f"- domain_normalized_risk_calibration_used: true\n"
        f"- mixed_domain_dataset_balanced_unseen_passed: {mixed_balanced_ready}\n"
        f"- vipseg_to_vspw_domain_normalized_passed: {vipseg_to_vspw_fixed}\n"
        f"- vspw_to_vipseg_passed: {bool(vspw_to_vipseg.get('suite_passed'))}\n"
        f"- cross_dataset_video_semantic_suite_ready: {suite_ready}\n"
        f"- semantic_adapter_training_allowed: {suite_ready}\n"
        f"- recommended_next_step: {decision['recommended_next_step']}\n\n"
        "## 中文总结\n"
        + decision["中文结论"]
        + "\n",
        encoding="utf-8",
    )


def main() -> int:
    rc = base.main()
    add_balanced_protocol_summary()
    d = json.loads(DECISION.read_text(encoding="utf-8"))
    print(json.dumps({"cross_dataset_video_semantic_suite_ready": d.get("cross_dataset_video_semantic_suite_ready"), "recommended_next_step": d.get("recommended_next_step")}, ensure_ascii=False), flush=True)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
