#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))

from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, write_doc
from stwm.tools.ostf_v30_external_gt_schema_20260508 import utc_now


TRAIN_TOOL = ROOT / "code/stwm/tools/train_ostf_v34_4_oracle_residual_probe_20260511.py"
EVAL_TOOL = ROOT / "code/stwm/tools/eval_ostf_v34_4_oracle_residual_probe_20260511.py"
MODEL = ROOT / "code/stwm/modules/ostf_v34_3_pointwise_unit_residual_world_model.py"
TRAIN = ROOT / "reports/stwm_ostf_v34_4_oracle_residual_probe_train_summary_20260511.json"
EVAL = ROOT / "reports/stwm_ostf_v34_4_oracle_residual_probe_eval_summary_20260511.json"
ORACLE = ROOT / "reports/stwm_ostf_v34_4_oracle_residual_probe_decision_20260511.json"
DECISION = ROOT / "reports/stwm_ostf_v34_4_decision_20260511.json"
TARGETS = ROOT / "reports/stwm_ostf_v34_4_residual_utility_target_build_20260511.json"
OUT = ROOT / "reports/stwm_ostf_v34_5_v34_4_residual_objective_audit_20260511.json"
DOC = ROOT / "docs/STWM_OSTF_V34_5_V34_4_RESIDUAL_OBJECTIVE_AUDIT_20260511.md"


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def text(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def main() -> int:
    train_code = text(TRAIN_TOOL)
    model_code = text(MODEL)
    oracle = load(ORACLE)
    decision = load(DECISION)
    targets = load(TARGETS)
    ratios = targets.get("residual_semantic_positive_ratio_by_split", {})
    residual_supervised_as_standalone = "residual_direct = cosine_loss(out[\"unit_semantic_residual\"], batch[\"fut_teacher_embedding\"]" in train_code
    final_is_pointwise_plus_residual = "pointwise_semantic_belief\" + sem_gate" in model_code or "pointwise_semantic_belief\"] + sem_gate" in model_code
    delta_missing = "delta_target" not in train_code and "projection" not in train_code
    aligned = not (residual_supervised_as_standalone and final_is_pointwise_plus_residual)
    gain = oracle.get("residual_utility_subset_gain") or {}
    oracle_fail_borderline = bool(
        oracle.get("oracle_residual_probe_passed") is False
        and gain.get("val") is not None
        and gain.get("test") is not None
        and 0.003 <= float(gain["val"]) < 0.005
        and float(gain["test"]) >= 0.005
    )
    broad = any(float(v) > 0.25 for v in ratios.values()) if ratios else False
    payload = {
        "generated_at_utc": utc_now(),
        "checked_files": [str(p.relative_to(ROOT)) for p in [TRAIN_TOOL, EVAL_TOOL, MODEL, TRAIN, EVAL, ORACLE, DECISION, TARGETS]],
        "unit_semantic_residual_supervised_as_target_embedding": residual_supervised_as_standalone,
        "final_semantic_is_pointwise_plus_residual": final_is_pointwise_plus_residual,
        "residual_direct_loss_aligned_with_final_composition": aligned,
        "force_gate_one_hurts_because_residual_content_harmful": bool(decision.get("force_gate_one_hurts", False)),
        "oracle_residual_failed_due_val_gain_slightly_below_threshold": oracle_fail_borderline,
        "residual_utility_target_positive_ratio_too_broad": broad,
        "semantic_hard_signal_failed_despite_changed_signal_positive": bool(
            not any((oracle.get("semantic_hard_signal") or {}).values())
            and any((oracle.get("changed_semantic_signal") or {}).values())
        ),
        "identity_auc_is_oracle_only_not_final_claim": True,
        "residual_supervised_as_standalone_target": residual_supervised_as_standalone,
        "delta_residual_objective_missing": delta_missing,
        "force_gate_one_hurts_due_residual_content": bool(decision.get("force_gate_one_hurts", False)),
        "oracle_fail_is_borderline": oracle_fail_borderline,
        "residual_positive_ratio_too_broad": broad,
        "identity_auc_oracle_only": True,
        "recommended_fix": "Train unit_semantic_residual as an orthogonal delta correction over frozen pointwise_semantic_belief, and narrow residual utility positives with split-specific confidence/error quantiles.",
    }
    dump_json(OUT, payload)
    write_doc(
        DOC,
        "STWM OSTF V34.5 V34.4 Residual Objective Audit",
        payload,
        [
            "residual_supervised_as_standalone_target",
            "final_semantic_is_pointwise_plus_residual",
            "residual_direct_loss_aligned_with_final_composition",
            "delta_residual_objective_missing",
            "force_gate_one_hurts_due_residual_content",
            "oracle_fail_is_borderline",
            "residual_positive_ratio_too_broad",
            "semantic_hard_signal_failed_despite_changed_signal_positive",
            "identity_auc_oracle_only",
            "recommended_fix",
        ],
    )
    print(OUT.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
