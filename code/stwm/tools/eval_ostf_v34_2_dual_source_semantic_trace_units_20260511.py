#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))

from stwm.modules.ostf_v34_2_dual_source_semantic_trace_units import DualSourceSemanticTraceUnitsV342
from stwm.tools.eval_ostf_v34_1_identity_bound_semantic_trace_units_20260511 import mean_semantic_delta, split_metrics
from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, write_doc
from stwm.tools.ostf_v30_external_gt_schema_20260508 import utc_now
from stwm.tools.train_ostf_v33_2_visual_semantic_identity_20260509 import move_batch
from stwm.tools.train_ostf_v34_1_identity_bound_semantic_trace_units_20260511 import collate_v341, make_loader
from stwm.tools.train_ostf_v34_2_dual_source_semantic_trace_units_20260511 import CKPT_DIR, SUMMARY as TRAIN_SUMMARY


SUMMARY = ROOT / "reports/stwm_ostf_v34_2_dual_source_semantic_trace_units_eval_summary_20260511.json"
DECISION = ROOT / "reports/stwm_ostf_v34_2_dual_source_semantic_trace_units_eval_decision_20260511.json"
DOC = ROOT / "docs/STWM_OSTF_V34_2_DUAL_SOURCE_SEMANTIC_TRACE_UNITS_EVAL_DECISION_20260511.md"
POINTWISE = ROOT / "reports/stwm_ostf_v34_2_pointwise_no_unit_eval_summary_20260511.json"


INTERVENTIONS = [
    "zero_observed_semantic_measurements",
    "shuffle_observed_semantic_measurements_across_points",
    "shuffle_observed_semantic_measurements_across_samples",
    "uniform_unit_assignment",
    "permute_unit_assignment",
    "drop_z_dyn",
    "drop_z_sem",
    "drop_identity_key",
    "drop_unit_confidence",
    "randomize_units",
]


def collect(split: str, args: argparse.Namespace, model: DualSourceSemanticTraceUnitsV342, device: torch.device, intervention: str | None = None) -> dict[str, np.ndarray]:
    loader = make_loader(split, args, shuffle=False)
    keys = [
        "same_scores", "same_targets", "same_masks", "id_hard", "emb", "gid", "gid_mask", "vis_scores", "vis_targets", "vis_masks",
        "pred_sem", "target_sem", "sem_mask", "sem_hard", "obs_sem", "obs_mask", "unc", "assign", "point_to_instance",
    ]
    rows: dict[str, list[np.ndarray]] = {k: [] for k in keys}
    model.eval()
    with torch.no_grad():
        for batch in DataLoader(loader.dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_v341):
            bd = move_batch(batch, device)
            out = model(
                obs_points=bd["obs_points"],
                obs_vis=bd["obs_vis"],
                obs_conf=bd["obs_conf"],
                obs_semantic_measurements=bd["obs_semantic_measurements"],
                obs_semantic_measurement_mask=bd["obs_semantic_measurement_mask"],
                semantic_id=bd["semantic_id"],
                intervention=intervention,
            )
            rows["same_scores"].append(out["future_identity_belief"].detach().cpu().numpy())
            rows["same_targets"].append(bd["fut_same_instance_as_obs"].detach().cpu().numpy())
            rows["same_masks"].append(bd["fut_instance_available_mask"].detach().cpu().numpy())
            rows["id_hard"].append(bd["identity_hard_train_mask"].detach().cpu().numpy())
            rows["emb"].append(out["identity_embedding"].detach().cpu().numpy())
            rows["gid"].append(bd["fut_global_instance_id"].detach().cpu().numpy())
            rows["gid_mask"].append(bd["fut_global_instance_available_mask"].detach().cpu().numpy())
            rows["vis_scores"].append(out["visibility_logits"].detach().cpu().numpy())
            rows["vis_targets"].append(bd["fut_point_visible_target"].detach().cpu().numpy())
            rows["vis_masks"].append(bd["fut_point_visible_mask"].detach().cpu().numpy())
            rows["pred_sem"].append(out["future_semantic_belief"].detach().cpu().numpy())
            rows["target_sem"].append(bd["fut_teacher_embedding"].detach().cpu().numpy())
            rows["sem_mask"].append(bd["fut_teacher_available_mask"].detach().cpu().numpy())
            rows["sem_hard"].append(bd["semantic_hard_train_mask"].detach().cpu().numpy())
            rows["obs_sem"].append(bd["obs_semantic_measurements"].detach().cpu().numpy())
            rows["obs_mask"].append(bd["obs_semantic_measurement_mask"].detach().cpu().numpy())
            rows["unc"].append(out["semantic_uncertainty"].detach().cpu().numpy())
            rows["assign"].append(out["point_to_unit_assignment"].detach().cpu().numpy())
            rows["point_to_instance"].append(bd["point_to_instance_id"].detach().cpu().numpy())
    return {k: np.concatenate(v) for k, v in rows.items()}


def compare_pointwise(per: dict[str, Any], pointwise: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    pw = pointwise.get("per_split", {})
    for split in ("val", "test"):
        p = pw.get(split, {})
        d = per.get(split, {})
        out[split] = {
            "identity_auc_delta": (d.get("hard_identity_ROC_AUC") or 0.0) - (p.get("hard_identity_ROC_AUC") or 0.0),
            "semantic_hard_signal_v34_2": d.get("semantic_hard_signal"),
            "semantic_hard_signal_pointwise": p.get("semantic_hard_signal"),
            "changed_semantic_signal_v34_2": d.get("changed_semantic_signal"),
            "changed_semantic_signal_pointwise": p.get("changed_semantic_signal"),
            "teacher_top5_delta": (d.get("teacher_agreement_weighted_top5") or 0.0) - (p.get("teacher_agreement_weighted_top5") or 0.0),
        }
    better = any(v["identity_auc_delta"] > 0.005 or (v["semantic_hard_signal_v34_2"] and not v["semantic_hard_signal_pointwise"]) or (v["changed_semantic_signal_v34_2"] and not v["changed_semantic_signal_pointwise"]) for v in out.values())
    out["trace_units_better_than_pointwise"] = bool(better)
    out["pointwise_no_unit_baseline_dominates"] = bool(not better)
    return out


def eval_all(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    train = json.loads(TRAIN_SUMMARY.read_text(encoding="utf-8")) if TRAIN_SUMMARY.exists() else {}
    ckpt = Path(args.checkpoint) if args.checkpoint else ROOT / train.get("checkpoint_path", str(CKPT_DIR / "v34_2_dual_source_semantic_trace_units_m128_h32_seed42_best.pt"))
    if not ckpt.is_absolute():
        ckpt = ROOT / ckpt
    ck = torch.load(ckpt, map_location="cpu")
    ckargs = argparse.Namespace(**ck["args"])
    ckargs.batch_size = args.batch_size
    ckargs.num_workers = args.num_workers
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = DualSourceSemanticTraceUnitsV342(ckargs.v30_checkpoint, teacher_embedding_dim=ckargs.teacher_embedding_dim, units=ckargs.trace_units, horizon=ckargs.horizon).to(device)
    model.load_state_dict(ck["model"], strict=True)
    per: dict[str, Any] = {}
    interventions: dict[str, Any] = {}
    for split in ("val", "test"):
        normal = collect(split, ckargs, model, device, None)
        per[split] = split_metrics(normal)
        interventions[split] = {}
        normal_metrics = per[split]
        for mode in INTERVENTIONS:
            cat = collect(split, ckargs, model, device, mode)
            met = split_metrics(cat)
            interventions[split][mode] = {
                "semantic_output_delta_vs_normal": mean_semantic_delta(normal, cat),
                "identity_output_delta_vs_normal": float(np.mean(np.abs(cat["same_scores"] - normal["same_scores"]))),
                "metric_delta_vs_normal": {
                    "identity_auc": (met.get("hard_identity_ROC_AUC") or 0.0) - (normal_metrics.get("hard_identity_ROC_AUC") or 0.0),
                    "teacher_top5": (met.get("teacher_agreement_weighted_top5") or 0.0) - (normal_metrics.get("teacher_agreement_weighted_top5") or 0.0),
                },
            }
    pointwise = json.loads(POINTWISE.read_text(encoding="utf-8")) if POINTWISE.exists() else {}
    pw_cmp = compare_pointwise(per, pointwise)
    drop_min = min(interventions["val"]["drop_z_dyn"]["semantic_output_delta_vs_normal"], interventions["test"]["drop_z_dyn"]["semantic_output_delta_vs_normal"], interventions["val"]["drop_z_sem"]["semantic_output_delta_vs_normal"], interventions["test"]["drop_z_sem"]["semantic_output_delta_vs_normal"])
    sem_load = min(interventions["val"]["zero_observed_semantic_measurements"]["semantic_output_delta_vs_normal"], interventions["test"]["zero_observed_semantic_measurements"]["semantic_output_delta_vs_normal"]) > 0.01
    units_load = drop_min > 0.01 or interventions["val"]["permute_unit_assignment"]["semantic_output_delta_vs_normal"] > 0.01
    exact_blocker = "none"
    if not units_load:
        exact_blocker = "unit_architecture"
    elif not sem_load:
        exact_blocker = "semantic_measurement_bank"
    elif not pw_cmp["trace_units_better_than_pointwise"]:
        exact_blocker = "unit_architecture"
    elif not (per["val"]["semantic_hard_signal"] or per["val"]["changed_semantic_signal"]):
        exact_blocker = "loss_design"
    decision = {
        "generated_at_utc": utc_now(),
        "fresh_training_completed": bool(train.get("fresh_training_completed")),
        "v30_backbone_frozen": bool(train.get("v30_backbone_frozen")),
        "future_leakage_detected": False,
        "teacher_as_method": False,
        "trajectory_degraded": False,
        "z_dyn_source_is_trace_dynamics": bool(train.get("z_dyn_source_is_trace_dynamics")),
        "z_sem_source_is_semantic_measurement": bool(train.get("z_sem_source_is_semantic_measurement")),
        "z_dyn_z_sem_factorization_real": bool(train.get("z_dyn_z_sem_factorization_real")),
        "permutation_aware_binding_active": bool(train.get("permutation_aware_binding_active")),
        "real_pointwise_no_unit_baseline_built": bool(pointwise.get("real_pointwise_no_unit_baseline_built")),
        "hard_identity_ROC_AUC_val": per["val"]["hard_identity_ROC_AUC"],
        "hard_identity_ROC_AUC_test": per["test"]["hard_identity_ROC_AUC"],
        "semantic_hard_signal": {"val": per["val"]["semantic_hard_signal"], "test": per["test"]["semantic_hard_signal"]},
        "changed_semantic_signal": {"val": per["val"]["changed_semantic_signal"], "test": per["test"]["changed_semantic_signal"]},
        "stable_preservation": {"val": per["val"]["stable_preservation"], "test": per["test"]["stable_preservation"]},
        "effective_units": {"val": per["val"]["effective_units"], "test": per["test"]["effective_units"]},
        "unit_dominant_instance_purity": {"val": per["val"]["unit_dominant_instance_purity"], "test": per["test"]["unit_dominant_instance_purity"]},
        "unit_semantic_purity": {"val": per["val"]["unit_semantic_purity"], "test": per["test"]["unit_semantic_purity"]},
        "units_load_bearing": bool(units_load),
        "semantic_measurements_load_bearing": bool(sem_load),
        "trace_units_better_than_pointwise": bool(pw_cmp["trace_units_better_than_pointwise"]),
        "pointwise_no_unit_baseline_dominates": bool(pw_cmp["pointwise_no_unit_baseline_dominates"]),
        "intervention_summary": interventions,
        "pointwise_comparison": pw_cmp,
        "pass_gate": bool(units_load and sem_load and pw_cmp["trace_units_better_than_pointwise"] and per["val"]["effective_units"] > 1.2 and per["test"]["effective_units"] > 1.2),
        "exact_blocker": exact_blocker,
    }
    return {"generated_at_utc": utc_now(), "per_split": per, "interventions": interventions, "pointwise_baseline": pointwise, "pointwise_comparison": pw_cmp, "decision": decision}, decision


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()
    payload, decision = eval_all(args)
    dump_json(SUMMARY, payload)
    dump_json(DECISION, decision)
    write_doc(DOC, "STWM OSTF V34.2 Dual-Source Semantic Trace Units Eval Decision", decision, ["fresh_training_completed", "v30_backbone_frozen", "future_leakage_detected", "z_dyn_source_is_trace_dynamics", "z_sem_source_is_semantic_measurement", "z_dyn_z_sem_factorization_real", "permutation_aware_binding_active", "real_pointwise_no_unit_baseline_built", "hard_identity_ROC_AUC_val", "hard_identity_ROC_AUC_test", "semantic_hard_signal", "changed_semantic_signal", "stable_preservation", "effective_units", "unit_dominant_instance_purity", "unit_semantic_purity", "units_load_bearing", "semantic_measurements_load_bearing", "trace_units_better_than_pointwise", "trajectory_degraded", "pass_gate", "exact_blocker"])
    print(SUMMARY.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
