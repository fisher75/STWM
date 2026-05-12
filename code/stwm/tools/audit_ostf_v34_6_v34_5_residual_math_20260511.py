#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))

from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, write_doc
from stwm.tools.ostf_v30_external_gt_schema_20260508 import utc_now


TRAIN = ROOT / "code/stwm/tools/train_ostf_v34_5_delta_residual_probe_20260511.py"
EVAL = ROOT / "code/stwm/tools/eval_ostf_v34_5_delta_residual_probe_20260511.py"
ABL = ROOT / "code/stwm/tools/eval_ostf_v34_5_residual_content_ablation_20260511.py"
MODEL = ROOT / "code/stwm/modules/ostf_v34_3_pointwise_unit_residual_world_model.py"
DECISION = ROOT / "reports/stwm_ostf_v34_5_delta_residual_probe_decision_20260511.json"
EVAL_SUMMARY = ROOT / "reports/stwm_ostf_v34_5_delta_residual_probe_eval_summary_20260511.json"
FINAL = ROOT / "reports/stwm_ostf_v34_5_decision_20260511.json"
STANDALONE = ROOT / "reports/stwm_ostf_v34_4_oracle_residual_probe_decision_20260511.json"
TARGET_DOC = ROOT / "docs/STWM_OSTF_V34_5_STRICT_RESIDUAL_UTILITY_TARGET_BUILD_20260511.md"
TARGET_JSON = ROOT / "reports/stwm_ostf_v34_5_strict_residual_utility_target_build_20260511.json"
OUT = ROOT / "reports/stwm_ostf_v34_6_v34_5_residual_math_audit_20260511.json"
DOC = ROOT / "docs/STWM_OSTF_V34_6_V34_5_RESIDUAL_MATH_AUDIT_20260511.md"


def text(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def main() -> int:
    train = text(TRAIN)
    model = text(MODEL)
    ablation = text(ABL)
    current_delta_orth = "target_n - alpha * base_n" in train
    true_vector_missing = "target_n - base_n" not in train and "target - base" not in train
    anti_base_allowed = "orthogonality_weight" not in train or "orthogonality_weight\", type=float, default=0.0" in train
    orth_misaligned = "residual_orthogonality" in train and current_delta_orth
    v34_4_init_not_used = "v34_4_oracle" not in train and "stwm_ostf_v34_4_oracle_residual_probe" not in train
    ablation_not_real = any(s in ablation for s in ["not_run_separately_for_delta_probe", "mostly compares", "available_in_v34_3_failure_audit"])
    residual_comp = (
        "normalize(base + gate * residual)"
        if "\"pointwise_semantic_belief\"] + sem_gate" in model
        or "base[\"pointwise_semantic_belief\"] + sem_gate" in model
        else "unknown"
    )
    target_json_missing_but_doc_exists = bool(TARGET_DOC.exists() and not TARGET_JSON.exists())
    payload = {
        "generated_at_utc": utc_now(),
        "checked_files": [str(p.relative_to(ROOT)) for p in [TRAIN, EVAL, ABL, MODEL, DECISION, EVAL_SUMMARY, FINAL, STANDALONE, TARGET_DOC]],
        "residual_composition": residual_comp,
        "current_delta_target_type": "orthogonal_projection_only" if current_delta_orth else "other",
        "current_delta_can_subtract_wrong_base_component": False if current_delta_orth else True,
        "orthogonality_regularization_prevents_anti_base_correction": orth_misaligned,
        "v34_5_starts_from_pointwise_init": "load_pointwise_into_residual" in train,
        "v34_5_starts_from_v34_4_residual_init": not v34_4_init_not_used,
        "residual_content_ablation_actually_ran_random_no_memory_shuffled": not ablation_not_real,
        "strict_residual_target_json_missing_but_md_exists": target_json_missing_but_doc_exists,
        "exact_code_locations": {
            "composition": "code/stwm/modules/ostf_v34_3_pointwise_unit_residual_world_model.py: final_sem = normalize(pointwise_semantic_belief + gate * unit_semantic_residual)",
            "orthogonal_delta": "code/stwm/tools/train_ostf_v34_5_delta_residual_probe_20260511.py: delta = target_n - alpha * base_n",
            "orthogonality": "code/stwm/tools/train_ostf_v34_5_delta_residual_probe_20260511.py: residual_orthogonality",
            "ablation_truth": "code/stwm/tools/eval_ostf_v34_5_residual_content_ablation_20260511.py",
        },
        "recommended_residual_parameterizations": [
            "standalone_target_residual",
            "orthogonal_delta_residual",
            "true_vector_delta_residual",
            "scaled_tangent_delta_residual",
            "mixture_residual",
        ],
        "current_delta_is_orthogonal_only": current_delta_orth,
        "true_vector_delta_missing": true_vector_missing,
        "anti_base_correction_allowed": anti_base_allowed,
        "orthogonality_regularization_misaligned": orth_misaligned,
        "v34_4_residual_init_not_used": v34_4_init_not_used,
        "residual_content_ablation_not_real": ablation_not_real,
        "recommended_fix": "Run a residual parameterization/capacity sweep, including true_vector_delta and v34_4 residual initialization, before any learned gate training.",
    }
    dump_json(OUT, payload)
    write_doc(
        DOC,
        "STWM OSTF V34.6 V34.5 Residual Math Audit",
        payload,
        [
            "residual_composition",
            "current_delta_target_type",
            "current_delta_is_orthogonal_only",
            "true_vector_delta_missing",
            "anti_base_correction_allowed",
            "orthogonality_regularization_misaligned",
            "v34_4_residual_init_not_used",
            "residual_content_ablation_not_real",
            "strict_residual_target_json_missing_but_md_exists",
            "recommended_residual_parameterizations",
            "recommended_fix",
        ],
    )
    print(OUT.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
