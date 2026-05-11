#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))

from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, write_doc
from stwm.tools.ostf_v30_external_gt_schema_20260508 import utc_now


MODEL = ROOT / "code/stwm/modules/ostf_v34_3_pointwise_unit_residual_world_model.py"
TRAIN_TOOL = ROOT / "code/stwm/tools/train_ostf_v34_3_pointwise_unit_residual_20260511.py"
EVAL_TOOL = ROOT / "code/stwm/tools/eval_ostf_v34_3_pointwise_unit_residual_20260511.py"
TRAIN = ROOT / "reports/stwm_ostf_v34_3_pointwise_unit_residual_train_summary_20260511.json"
EVAL = ROOT / "reports/stwm_ostf_v34_3_pointwise_unit_residual_eval_summary_20260511.json"
EVAL_DECISION = ROOT / "reports/stwm_ostf_v34_3_pointwise_unit_residual_eval_decision_20260511.json"
FINAL = ROOT / "reports/stwm_ostf_v34_3_decision_20260511.json"
POINTWISE = ROOT / "reports/stwm_ostf_v34_2_pointwise_no_unit_eval_summary_20260511.json"
DUAL = ROOT / "reports/stwm_ostf_v34_2_dual_source_semantic_trace_units_eval_summary_20260511.json"
OUT = ROOT / "reports/stwm_ostf_v34_4_v34_3_residual_failure_audit_20260511.json"
DOC = ROOT / "docs/STWM_OSTF_V34_4_V34_3_RESIDUAL_FAILURE_AUDIT_20260511.md"


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def code_contains(path: Path, needle: str) -> bool:
    return path.exists() and needle in path.read_text(encoding="utf-8")


def mean_gate(final: dict[str, Any], key: str) -> float | None:
    value = final.get(key)
    if not isinstance(value, dict):
        return None
    vals = [v for v in value.values() if v is not None]
    return float(sum(vals) / len(vals)) if vals else None


def main() -> int:
    train = load(TRAIN)
    ev = load(EVAL)
    evd = load(EVAL_DECISION)
    final = load(FINAL)
    interventions = ev.get("interventions", {})
    force_one = {}
    force_zero = {}
    force_one_hurts = False
    for split in ("val", "test"):
        f1 = interventions.get(split, {}).get("force_gate_one", {})
        f0 = interventions.get(split, {}).get("force_gate_zero", {})
        force_one[split] = f1
        force_zero[split] = f0
        met = f1.get("metric_delta_vs_normal", {})
        if (met.get("teacher_top5") is not None and met["teacher_top5"] < -0.005) or (met.get("semantic_hard_model_cosine") is not None and met["semantic_hard_model_cosine"] < -0.005):
            force_one_hurts = True
    stable_gate = mean_gate(final, "semantic_residual_gate_mean_stable")
    changed_gate = mean_gate(final, "semantic_residual_gate_mean_changed")
    hard_gate = mean_gate(final, "semantic_residual_gate_mean_hard")
    full_gate = None
    per = ev.get("per_split", {})
    full_vals = [per.get(s, {}).get("semantic_residual_gate_mean_full") for s in ("val", "test")]
    full_vals = [v for v in full_vals if v is not None]
    if full_vals:
        full_gate = float(sum(full_vals) / len(full_vals))
    gate_collapsed = bool(max([v for v in [stable_gate, changed_gate, hard_gate, full_gate] if v is not None] or [1.0]) < 0.01)
    gate_order_ok = bool((changed_gate is not None and stable_gate is not None and changed_gate > stable_gate) or (hard_gate is not None and stable_gate is not None and hard_gate > stable_gate))
    direct_residual_supervision_missing = not code_contains(TRAIN_TOOL, "residual_semantic_utility_mask") and not code_contains(TRAIN_TOOL, "residual_gate_target")
    changed_reward_weak = code_contains(TRAIN_TOOL, "0.2 * changed_gate_reward") or code_contains(TRAIN_TOOL, "0.2 * hard_gate_reward")
    residual_content_not_proven = bool(not final.get("residual_improves_over_pointwise_on_hard", False) or force_one_hurts)
    pointwise_dominates = bool(final.get("pointwise_baseline_dominates", True))
    recommended = "build_residual_utility_targets_and_train_oracle_residual_probe"
    if not direct_residual_supervision_missing and not residual_content_not_proven:
        recommended = "train_supervised_residual_gate"
    payload = {
        "generated_at_utc": utc_now(),
        "checked_files": [str(p.relative_to(ROOT)) for p in [MODEL, TRAIN_TOOL, EVAL_TOOL, TRAIN, EVAL, EVAL_DECISION, FINAL, POINTWISE, DUAL]],
        "units_load_bearing_v34_3_final": bool(final.get("units_load_bearing", False)),
        "semantic_measurements_load_bearing": bool(final.get("semantic_measurements_load_bearing", False)),
        "pointwise_baseline_dominates": pointwise_dominates,
        "residual_improves_over_pointwise_on_hard": bool(final.get("residual_improves_over_pointwise_on_hard", False)),
        "residual_does_not_degrade_stable": bool(final.get("residual_does_not_degrade_stable", False)),
        "semantic_residual_gate_mean_full": full_gate,
        "semantic_residual_gate_mean_stable": final.get("semantic_residual_gate_mean_stable"),
        "semantic_residual_gate_mean_changed": final.get("semantic_residual_gate_mean_changed"),
        "semantic_residual_gate_mean_hard": final.get("semantic_residual_gate_mean_hard"),
        "semantic_gate_order_ok": gate_order_ok,
        "force_gate_one_metric_delta": force_one,
        "force_gate_zero_metric_delta": force_zero,
        "force_gate_one_hurts": force_one_hurts,
        "residual_content_harmful_when_gate_opens": force_one_hurts,
        "gate_collapse_rational_given_harmful_residual_content": bool(gate_collapsed and force_one_hurts),
        "changed_gate_reward_too_weak": changed_reward_weak,
        "hard_gate_reward_too_weak": changed_reward_weak,
        "residual_branch_has_direct_residual_target_supervision": not direct_residual_supervision_missing,
        "exact_code_locations": {
            "final_residual_mixture": "code/stwm/modules/ostf_v34_3_pointwise_unit_residual_world_model.py: final_sem = normalize(pointwise + gate * residual)",
            "weak_gate_reward": "code/stwm/tools/train_ostf_v34_3_pointwise_unit_residual_20260511.py: 0.2 * changed_gate_reward / hard_gate_reward",
            "missing_residual_target": "train script has no residual_semantic_utility_mask or residual_gate_target loader",
        },
        "gate_collapsed": gate_collapsed,
        "gate_order_wrong": not gate_order_ok,
        "residual_content_not_proven": residual_content_not_proven,
        "pointwise_base_dominates": pointwise_dominates,
        "direct_residual_supervision_missing": direct_residual_supervision_missing,
        "recommended_fix": recommended,
    }
    dump_json(OUT, payload)
    write_doc(
        DOC,
        "STWM OSTF V34.4 V34.3 Residual Failure Audit",
        payload,
        [
            "units_load_bearing_v34_3_final",
            "semantic_measurements_load_bearing",
            "pointwise_baseline_dominates",
            "residual_improves_over_pointwise_on_hard",
            "residual_does_not_degrade_stable",
            "semantic_residual_gate_mean_full",
            "semantic_residual_gate_mean_stable",
            "semantic_residual_gate_mean_changed",
            "semantic_residual_gate_mean_hard",
            "semantic_gate_order_ok",
            "force_gate_one_hurts",
            "gate_collapsed",
            "gate_order_wrong",
            "residual_content_not_proven",
            "direct_residual_supervision_missing",
            "recommended_fix",
        ],
    )
    print(OUT.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
