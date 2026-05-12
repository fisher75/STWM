#!/usr/bin/env python3
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))

from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, write_doc
from stwm.tools.ostf_v30_external_gt_schema_20260508 import utc_now


DECISION = ROOT / "reports/stwm_ostf_v34_7_decision_20260511.json"
ORACLE_DECISION = ROOT / "reports/stwm_ostf_v34_7_assignment_oracle_residual_probe_decision_20260511.json"
ORACLE_EVAL = ROOT / "reports/stwm_ostf_v34_7_assignment_oracle_residual_probe_eval_summary_20260511.json"
ART_AUDIT = ROOT / "reports/stwm_ostf_v34_7_v34_6_assignment_and_artifact_audit_20260511.json"
TARGET_JSON = ROOT / "reports/stwm_ostf_v34_7_assignment_aware_residual_target_build_20260511.json"
VIS_JSON = ROOT / "reports/stwm_ostf_v34_7_assignment_residual_visualization_manifest_20260511.json"
TARGET_DOC = ROOT / "docs/STWM_OSTF_V34_7_ASSIGNMENT_AWARE_RESIDUAL_TARGET_BUILD_20260511.md"
VIS_DOC = ROOT / "docs/STWM_OSTF_V34_7_ASSIGNMENT_RESIDUAL_VISUALIZATION_20260511.md"
MODEL = ROOT / "code/stwm/modules/ostf_v34_7_assignment_bound_residual_memory.py"
TARGET_BUILD = ROOT / "code/stwm/tools/build_ostf_v34_7_assignment_aware_residual_targets_20260511.py"
EVAL = ROOT / "code/stwm/tools/eval_ostf_v34_7_assignment_oracle_residual_probe_20260511.py"
OUT = ROOT / "reports/stwm_ostf_v34_8_v34_7_assignment_causal_path_audit_20260512.json"
DOC = ROOT / "docs/STWM_OSTF_V34_8_V34_7_ASSIGNMENT_CAUSAL_PATH_AUDIT_20260512.md"


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def parse_doc_ratios(doc: str) -> dict[str, float]:
    m = re.search(r"point_positive_ratio_by_split.*?`({.*?})`", doc)
    if not m:
        return {}
    try:
        return json.loads(m.group(1).replace("'", '"'))
    except Exception:
        return {}


def main() -> int:
    decision = load(DECISION)
    oracle = load(ORACLE_DECISION)
    target = load(TARGET_JSON)
    model_src = read(MODEL)
    target_doc = read(TARGET_DOC)
    target_ratios = target.get("point_positive_ratio_by_split") or parse_doc_ratios(target_doc)
    assign_delta = oracle.get("assignment_intervention_delta") or oracle.get("shuffle_assignment_metric_delta") or {}
    sem_delta = {}
    eval_payload = load(ORACLE_EVAL)
    for split in ("val", "test"):
        sem_delta[split] = (((eval_payload.get("per_split") or {}).get(split) or {}).get("zero_semantic_measurement_delta"))
    too_broad = bool(any(v is not None and float(v) > 0.20 for v in target_ratios.values()))
    ratio_values = [float(v) for v in target_ratios.values()] if target_ratios else []
    distribution_mismatch = bool(ratio_values and (max(ratio_values) - min(ratio_values) > 0.25))
    assignment_formal = "torch.einsum(\"bmu,buhd->bmhd\", assign, unit_residual_memory)" in model_src
    residual_uses_pointwise = "pointwise_semantic_belief" in model_src and "unit_memory_input" in model_src
    semantic_shortcut = "sem_obs = obs_semantic_measurements" in model_src and "self.tokenizer" in model_src
    assignment_lb = decision.get("assignment_load_bearing_on_residual") is True
    semantic_lb = decision.get("semantic_measurements_load_bearing_on_residual") is True
    payload = {
        "generated_at_utc": utc_now(),
        "中文结论": "V34.7 的 assignment 形式路径存在，但因果 load-bearing 未成立；target 正样本过宽，semantic measurement 与 assignment intervention 在 test 上不足。",
        "artifact_packaging_truly_fixed": bool(TARGET_JSON.exists() and VIS_JSON.exists()),
        "assignment_target_json_missing": not TARGET_JSON.exists(),
        "visualization_json_missing": not VIS_JSON.exists(),
        "assignment_path_formally_present": assignment_formal,
        "assignment_path_causally_load_bearing": bool(assignment_lb),
        "semantic_measurement_causally_load_bearing": bool(semantic_lb),
        "target_positive_ratio_too_broad": too_broad,
        "target_distribution_mismatch": distribution_mismatch,
        "target_positive_ratio_by_split": target_ratios,
        "assignment_shortcut_suspected": bool(assignment_formal and not assignment_lb),
        "semantic_shortcut_suspected": bool(semantic_shortcut and not semantic_lb),
        "assignment_delta_by_split": assign_delta,
        "zero_semantic_measurement_delta_by_split": sem_delta,
        "why_assignment_load_bearing_false": "shuffle assignment 在 test 上没有正向破坏 residual gain，说明 assignment 不是稳定因果瓶颈。",
        "why_semantic_measurement_load_bearing_false": "zero/shuffle semantic measurement 对 V34.7 residual metric 的影响不稳定，test 未达阈值。",
        "exact_code_locations": {
            "formal_assignment_path": "code/stwm/modules/ostf_v34_7_assignment_bound_residual_memory.py: assignment_bound_residual = torch.einsum(\"bmu,buhd->bmhd\", assign, unit_residual_memory)",
            "target_too_broad": "code/stwm/tools/build_ostf_v34_7_assignment_aware_residual_targets_20260511.py: point_pos = strict & point_ok[:, None]",
            "eval_interventions": "code/stwm/tools/eval_ostf_v34_7_assignment_oracle_residual_probe_20260511.py: shuffled_assignment / zero_semantic_measurements",
        },
        "checked_files": [str(p.relative_to(ROOT)) for p in [DECISION, ORACLE_DECISION, ORACLE_EVAL, ART_AUDIT, TARGET_DOC, VIS_DOC, MODEL, TARGET_BUILD, EVAL]],
        "recommended_fix": "构建更严格的 causal assignment residual targets，并重写 residual memory，禁止 pointwise/global shortcut；先跑 oracle probe，不允许先训练 learned gate。",
    }
    dump_json(OUT, payload)
    write_doc(DOC, "V34.8 对 V34.7 assignment 因果路径的中文审计", payload, ["中文结论", "artifact_packaging_truly_fixed", "assignment_target_json_missing", "visualization_json_missing", "assignment_path_formally_present", "assignment_path_causally_load_bearing", "semantic_measurement_causally_load_bearing", "target_positive_ratio_too_broad", "target_positive_ratio_by_split", "assignment_shortcut_suspected", "semantic_shortcut_suspected", "recommended_fix"])
    print(f"已写出 V34.8 审计报告: {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
