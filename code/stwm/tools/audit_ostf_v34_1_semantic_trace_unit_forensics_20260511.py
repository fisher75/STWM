#!/usr/bin/env python3
from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))

from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, write_doc
from stwm.tools.ostf_v30_external_gt_schema_20260508 import utc_now


MODEL = ROOT / "code/stwm/modules/ostf_v34_semantic_trace_units.py"
TRAIN = ROOT / "code/stwm/tools/train_ostf_v34_semantic_trace_units_20260510.py"
EVAL = ROOT / "code/stwm/tools/eval_ostf_v34_semantic_trace_units_20260510.py"
RENDER = ROOT / "code/stwm/tools/render_ostf_v34_semantic_trace_unit_diagnostics_20260510.py"
DECISION = ROOT / "reports/stwm_ostf_v34_semantic_trace_unit_decision_20260510.json"
EVAL_SUMMARY = ROOT / "reports/stwm_ostf_v34_semantic_trace_units_eval_summary_20260510.json"
TRAIN_SUMMARY = ROOT / "reports/stwm_ostf_v34_semantic_trace_units_train_summary_20260510.json"
MEAS_DOC = ROOT / "docs/STWM_OSTF_V34_SEMANTIC_MEASUREMENT_BANK_20260510.md"
BASELINE_DOC = ROOT / "docs/STWM_OSTF_V34_AGAINST_TEACHER_PROTOTYPE_BASELINES_20260510.md"
VIS_DOC = ROOT / "docs/STWM_OSTF_V34_SEMANTIC_TRACE_UNIT_VISUALIZATION_20260510.md"
OUT = ROOT / "reports/stwm_ostf_v34_1_semantic_trace_unit_forensics_20260511.json"
DOC = ROOT / "docs/STWM_OSTF_V34_1_SEMANTIC_TRACE_UNIT_FORENSICS_20260511.md"


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def forward_body(src: str, class_name: str) -> str:
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return ""
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for child in node.body:
                if isinstance(child, ast.FunctionDef) and child.name == "forward":
                    return ast.get_source_segment(src, child) or ""
    return ""


def count_after_assignment(text: str, token: str, marker: str = "state = self.factorized_state") -> int:
    idx = text.find(marker)
    tail = text[idx:] if idx >= 0 else text
    return len(re.findall(rf"\b{re.escape(token)}\b", tail))


def main() -> int:
    model = read(MODEL)
    train = read(TRAIN)
    eval_src = read(EVAL)
    render = read(RENDER)
    fwd = forward_body(model, "SemanticTraceUnitsWorldModelV34")
    decision = load(DECISION)
    eval_summary = load(EVAL_SUMMARY)
    train_summary = load(TRAIN_SUMMARY)

    z_dyn_refs = count_after_assignment(fwd, "z_dyn")
    z_sem_refs = count_after_assignment(fwd, "z_sem")
    identity_key_refs = count_after_assignment(fwd, "identity_key")
    unit_conf_refs = count_after_assignment(fwd, "unit_confidence")
    assign_refs = len(re.findall(r"\bassign\b", fwd))

    temporal_markers = ["GRU", "LSTM", "for h", "for step", "time_embed", "horizon_embed", "rollout_steps"]
    rollout_cls = re.search(r"class SemanticTraceUnitRollout\(.*?class ", model, flags=re.S)
    rollout_src = rollout_cls.group(0) if rollout_cls else model

    consistency = (decision.get("semantic_belief_consistency") or eval_summary.get("decision", {}).get("semantic_belief_consistency") or {})
    high_consistency = any(isinstance(v, (int, float)) and v > 0.995 for v in consistency.values()) if isinstance(consistency, dict) else False
    sem_fail = not any((decision.get("changed_semantic_signal") or {}).values()) if isinstance(decision.get("changed_semantic_signal"), dict) else True

    meas_doc = read(MEAS_DOC)
    cov = {}
    m = re.search(r"measurement_coverage_by_split: `([^`]+)`", meas_doc)
    if m:
        cov = m.group(1)
    meas_report = load(ROOT / "reports/stwm_ostf_v34_semantic_measurement_bank_20260510.json")
    if meas_report.get("measurement_coverage_by_split"):
        cov = meas_report.get("measurement_coverage_by_split")

    payload: dict[str, Any] = {
        "generated_at_utc": utc_now(),
        "checked_files": [str(p.relative_to(ROOT)) for p in [MODEL, TRAIN, EVAL, RENDER, DECISION, EVAL_SUMMARY, TRAIN_SUMMARY, MEAS_DOC, BASELINE_DOC, VIS_DOC]],
        "z_dyn_used_in_prediction": bool(z_dyn_refs > 1),
        "z_sem_used_in_prediction": bool(z_sem_refs > 1),
        "identity_key_used_in_loss_or_prediction": bool(identity_key_refs > 1 or "identity_key" in train or "identity_key" in eval_src),
        "unit_confidence_used": bool(unit_conf_refs > 1 or "unit_confidence" in train or "unit_confidence" in eval_src),
        "unit_assignment_used": bool(assign_refs >= 3),
        "semantic_trace_unit_rollout_is_temporal": bool(any(t in rollout_src for t in temporal_markers)),
        "strict_identity_retrieval_implemented": bool("identity_retrieval_exclude_same_point" in eval_src and "None" not in re.sub(r"\s+", "", re.search(r"identity_retrieval_exclude_same_point.*", eval_src).group(0) if re.search(r"identity_retrieval_exclude_same_point.*", eval_src) else "")),
        "visualization_case_mining_real": bool("case_mining_used" in render and ("argmax" in render or "argsort" in render or "failure" in render.lower()) and "fixed" not in render.lower()),
        "semantic_belief_consistency_collapse_suspected": bool(high_consistency and sem_fail),
        "teacher_measurement_bank_coverage": cov,
        "v34_metrics": decision,
        "whether_v34_is_true_TUSB_like_semantic_trace_units": bool(False),
        "mechanism_classification": "shallow_measurement_conditioned_head",
        "exact_risks": [
            "z_dyn is produced by FactorizedTraceSemanticState but does not materially enter current V34 prediction path.",
            "identity_key is returned but not used by V34 loss or identity prediction.",
            "unit_confidence is returned but not used for uncertainty, gating, or loss.",
            "SemanticTraceUnitRollout is a per-field MLP with no recurrent/unit-level temporal state update.",
            "Strict identity retrieval is absent in V34 eval.",
            "High semantic_belief_consistency with failed stable/changed/hard semantic signals suggests smoothing/collapse rather than semantic success.",
        ],
    }
    dump_json(OUT, payload)
    write_doc(
        DOC,
        "STWM OSTF V34.1 Semantic Trace Unit Forensics",
        payload,
        [
            "z_dyn_used_in_prediction",
            "z_sem_used_in_prediction",
            "identity_key_used_in_loss_or_prediction",
            "unit_confidence_used",
            "unit_assignment_used",
            "semantic_trace_unit_rollout_is_temporal",
            "strict_identity_retrieval_implemented",
            "visualization_case_mining_real",
            "semantic_belief_consistency_collapse_suspected",
            "teacher_measurement_bank_coverage",
            "mechanism_classification",
        ],
    )
    print(OUT.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
