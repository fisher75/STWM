#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))

from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, write_doc
from stwm.tools.ostf_v30_external_gt_schema_20260508 import utc_now


MAIN = ROOT / "reports/stwm_ostf_v33_6_identity_contrastive_eval_summary_20260509.json"
CONTROL = ROOT / "reports/stwm_ostf_v33_6_old_local_instance_control_eval_summary_20260509.json"
REPORT = ROOT / "reports/stwm_ostf_v33_6_identity_label_ablation_summary_20260509.json"
DOC = ROOT / "docs/STWM_OSTF_V33_6_IDENTITY_LABEL_ABLATION_20260509.md"


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def mean(payload: dict[str, Any], metric: str, split: str) -> float | None:
    val = payload.get("metrics", {}).get(metric, {}).get(split, {}).get("mean")
    return float(val) if val is not None else None


def delta(main: dict[str, Any], control: dict[str, Any], metric: str, split: str) -> float | None:
    a = mean(main, metric, split)
    b = mean(control, metric, split)
    return None if a is None or b is None else float(a - b)


def main() -> int:
    main_payload = load(MAIN)
    control_payload = load(CONTROL)
    payload = {
        "generated_at_utc": utc_now(),
        "main_eval_summary_path": str(MAIN.relative_to(ROOT)),
        "control_eval_summary_path": str(CONTROL.relative_to(ROOT)),
        "main_exists": bool(main_payload),
        "control_exists": bool(control_payload),
        "global_label_model_vs_old_label_control": {
            "hard_identity_ROC_AUC_delta_val": delta(main_payload, control_payload, "hard_identity_ROC_AUC", "val"),
            "hard_identity_ROC_AUC_delta_test": delta(main_payload, control_payload, "hard_identity_ROC_AUC", "test"),
            "hard_identity_balanced_accuracy_delta_val": delta(main_payload, control_payload, "hard_identity_balanced_accuracy", "val"),
            "hard_identity_balanced_accuracy_delta_test": delta(main_payload, control_payload, "hard_identity_balanced_accuracy", "test"),
            "strict_retrieval_delta_val": delta(main_payload, control_payload, "identity_retrieval_exclude_same_point_top1", "val"),
            "strict_retrieval_delta_test": delta(main_payload, control_payload, "identity_retrieval_exclude_same_point_top1", "test"),
            "same_frame_retrieval_delta_val": delta(main_payload, control_payload, "identity_retrieval_same_frame_top1", "val"),
            "same_frame_retrieval_delta_test": delta(main_payload, control_payload, "identity_retrieval_same_frame_top1", "test"),
        },
    }
    dump_json(REPORT, payload)
    write_doc(
        DOC,
        "STWM OSTF V33.6 Identity Label Ablation",
        payload,
        ["main_exists", "control_exists", "global_label_model_vs_old_label_control"],
    )
    print(REPORT.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
