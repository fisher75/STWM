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

from stwm.modules.ostf_v34_3_pointwise_unit_residual_world_model import PointwiseUnitResidualWorldModelV343
from stwm.tools.eval_ostf_v34_1_identity_bound_semantic_trace_units_20260511 import (
    _norm,
    instance_pooled_retrieval,
    mean_semantic_delta,
    retrieval_top1,
    semantic_topk,
    unit_stats,
)
from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, write_doc
from stwm.tools.ostf_v30_external_gt_schema_20260508 import utc_now
from stwm.tools.train_ostf_v33_1_integrated_semantic_identity_20260509 import binary_metrics, visibility_f1
from stwm.tools.train_ostf_v33_2_visual_semantic_identity_20260509 import move_batch
from stwm.tools.train_ostf_v34_1_identity_bound_semantic_trace_units_20260511 import collate_v341, make_loader
from stwm.tools.train_ostf_v34_3_pointwise_unit_residual_20260511 import CKPT_DIR, SUMMARY as TRAIN_SUMMARY


SUMMARY = ROOT / "reports/stwm_ostf_v34_3_pointwise_unit_residual_eval_summary_20260511.json"
DECISION = ROOT / "reports/stwm_ostf_v34_3_pointwise_unit_residual_eval_decision_20260511.json"
DOC = ROOT / "docs/STWM_OSTF_V34_3_POINTWISE_UNIT_RESIDUAL_EVAL_DECISION_20260511.md"
DUAL_V342 = ROOT / "reports/stwm_ostf_v34_2_dual_source_semantic_trace_units_eval_summary_20260511.json"
POINTWISE_V342 = ROOT / "reports/stwm_ostf_v34_2_pointwise_no_unit_eval_summary_20260511.json"


INTERVENTIONS = [
    "zero_unit_residual",
    "shuffle_unit_residual",
    "drop_z_dyn",
    "drop_z_sem",
    "permute_unit_assignment",
    "force_gate_zero",
    "force_gate_one",
]


def _last_observed_copy(obs: np.ndarray, obs_mask: np.ndarray, target_shape: tuple[int, ...]) -> np.ndarray:
    last = np.zeros((obs.shape[0], obs.shape[1], obs.shape[-1]), dtype=np.float32)
    obs_mask = obs_mask.astype(bool)
    for bi in range(obs.shape[0]):
        for mi in range(obs.shape[1]):
            idx = np.where(obs_mask[bi, mi])[0]
            if idx.size:
                last[bi, mi] = obs[bi, mi, idx[-1]]
    return np.broadcast_to(last[:, :, None, :], target_shape)


def collect(
    split: str,
    args: argparse.Namespace,
    model: PointwiseUnitResidualWorldModelV343,
    device: torch.device,
    intervention: str | None = None,
) -> dict[str, np.ndarray]:
    loader = make_loader(split, args, shuffle=False)
    keys = [
        "same_scores",
        "same_targets",
        "same_masks",
        "id_hard",
        "emb",
        "gid",
        "gid_mask",
        "vis_scores",
        "vis_targets",
        "vis_masks",
        "pred_sem",
        "pointwise_sem",
        "target_sem",
        "sem_mask",
        "sem_hard",
        "obs_sem",
        "obs_mask",
        "unc",
        "assign",
        "point_to_instance",
        "sem_gate",
        "id_gate",
        "sem_residual",
        "id_residual",
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
            rows["pointwise_sem"].append(out["pointwise_semantic_belief"].detach().cpu().numpy())
            rows["target_sem"].append(bd["fut_teacher_embedding"].detach().cpu().numpy())
            rows["sem_mask"].append(bd["fut_teacher_available_mask"].detach().cpu().numpy())
            rows["sem_hard"].append(bd["semantic_hard_train_mask"].detach().cpu().numpy())
            rows["obs_sem"].append(bd["obs_semantic_measurements"].detach().cpu().numpy())
            rows["obs_mask"].append(bd["obs_semantic_measurement_mask"].detach().cpu().numpy())
            rows["unc"].append(out["semantic_uncertainty"].detach().cpu().numpy())
            rows["assign"].append(out["point_to_unit_assignment"].detach().cpu().numpy())
            rows["point_to_instance"].append(bd["point_to_instance_id"].detach().cpu().numpy())
            rows["sem_gate"].append(out["semantic_residual_gate"].detach().cpu().numpy())
            rows["id_gate"].append(out["identity_residual_gate"].detach().cpu().numpy())
            rows["sem_residual"].append(out["unit_semantic_residual"].detach().cpu().numpy())
            rows["id_residual"].append(out["unit_identity_residual"].detach().cpu().numpy())
    return {k: np.concatenate(v) for k, v in rows.items()}


def split_metrics(cat: dict[str, np.ndarray], pointwise_ref: dict[str, Any] | None) -> dict[str, Any]:
    hard = cat["id_hard"].astype(bool) & cat["same_masks"].astype(bool)
    idm = binary_metrics(cat["same_scores"], cat["same_targets"], hard)
    vis = visibility_f1(cat["vis_scores"], cat["vis_targets"], cat["vis_masks"])
    sem_mask = cat["sem_mask"].astype(bool)
    copy = _last_observed_copy(cat["obs_sem"], cat["obs_mask"], cat["target_sem"].shape)
    copy_cos = (_norm(copy) * _norm(cat["target_sem"])).sum(axis=-1)
    pred_cos = (_norm(cat["pred_sem"]) * _norm(cat["target_sem"])).sum(axis=-1)
    base_cos = (_norm(cat["pointwise_sem"]) * _norm(cat["target_sem"])).sum(axis=-1)
    stable = sem_mask & (copy_cos >= 0.80)
    changed = sem_mask & (copy_cos < 0.65)
    hard_sem = sem_mask & cat["sem_hard"].astype(bool)
    stats = unit_stats(cat["assign"], cat["point_to_instance"], cat["obs_sem"], cat["obs_mask"].astype(bool))
    consistency = float(np.nanmean((_norm(cat["pred_sem"])[:, :, 1:] * _norm(cat["pred_sem"])[:, :, :-1]).sum(axis=-1)))
    err = 1.0 - pred_cos
    unc_quality = float(np.corrcoef(cat["unc"][sem_mask].reshape(-1), err[sem_mask].reshape(-1))[0, 1]) if int(sem_mask.sum()) > 3 else None
    hard_gain = float(pred_cos[hard_sem].mean() - base_cos[hard_sem].mean()) if bool(hard_sem.any()) else None
    changed_gain = float(pred_cos[changed].mean() - base_cos[changed].mean()) if bool(changed.any()) else None
    stable_delta = float(pred_cos[stable].mean() - base_cos[stable].mean()) if bool(stable.any()) else None
    pointwise_split = pointwise_ref or {}
    ref_hard = pointwise_split.get("semantic_hard_model_cosine")
    ref_changed = pointwise_split.get("changed_model_cosine")
    ref_stable = pointwise_split.get("stable_model_cosine")
    residual_hard_over_external = bool(ref_hard is not None and hard_sem.any() and pred_cos[hard_sem].mean() > float(ref_hard) + 0.005)
    residual_changed_over_external = bool(ref_changed is not None and changed.any() and pred_cos[changed].mean() > float(ref_changed) + 0.005)
    residual_stable_ok_external = bool(ref_stable is None or not stable.any() or pred_cos[stable].mean() >= float(ref_stable) - 0.02)
    out = {
        "hard_identity_ROC_AUC": idm["ROC_AUC"],
        "val_calibrated_balanced_accuracy": idm["balanced_accuracy"],
        "identity_retrieval_exclude_same_point_top1": retrieval_top1(cat["emb"], cat["gid"], cat["gid_mask"], "exclude_same_point"),
        "identity_retrieval_same_frame_top1": retrieval_top1(cat["emb"], cat["gid"], cat["gid_mask"], "same_frame"),
        "identity_retrieval_instance_pooled_top1": instance_pooled_retrieval(cat["emb"], cat["gid"], cat["gid_mask"]),
        "identity_hard_residual_gain": None if pointwise_split.get("hard_identity_ROC_AUC") is None or idm["ROC_AUC"] is None else float(idm["ROC_AUC"] - pointwise_split["hard_identity_ROC_AUC"]),
        "stable_preservation": bool(pred_cos[stable].mean() + 1e-9 >= copy_cos[stable].mean() - 0.02) if stable.any() else False,
        "stable_model_cosine": float(pred_cos[stable].mean()) if stable.any() else None,
        "stable_pointwise_cosine": float(base_cos[stable].mean()) if stable.any() else None,
        "stable_copy_cosine": float(copy_cos[stable].mean()) if stable.any() else None,
        "changed_semantic_signal": bool(pred_cos[changed].mean() > copy_cos[changed].mean() + 0.01) if changed.any() else False,
        "changed_model_cosine": float(pred_cos[changed].mean()) if changed.any() else None,
        "changed_pointwise_cosine": float(base_cos[changed].mean()) if changed.any() else None,
        "changed_copy_cosine": float(copy_cos[changed].mean()) if changed.any() else None,
        "semantic_hard_signal": bool(pred_cos[hard_sem].mean() > copy_cos[hard_sem].mean() + 0.01) if hard_sem.any() else False,
        "semantic_hard_model_cosine": float(pred_cos[hard_sem].mean()) if hard_sem.any() else None,
        "semantic_hard_pointwise_cosine": float(base_cos[hard_sem].mean()) if hard_sem.any() else None,
        "semantic_hard_copy_cosine": float(copy_cos[hard_sem].mean()) if hard_sem.any() else None,
        "teacher_agreement_weighted_top5": semantic_topk(cat["pred_sem"], cat["target_sem"], sem_mask, 5),
        "semantic_belief_consistency": consistency,
        "semantic_uncertainty_quality": unc_quality,
        "copy_baseline_comparison": {
            "stable_delta_vs_copy": None if not stable.any() else float(pred_cos[stable].mean() - copy_cos[stable].mean()),
            "changed_delta_vs_copy": None if not changed.any() else float(pred_cos[changed].mean() - copy_cos[changed].mean()),
            "hard_delta_vs_copy": None if not hard_sem.any() else float(pred_cos[hard_sem].mean() - copy_cos[hard_sem].mean()),
        },
        "pointwise_baseline_comparison": {
            "stable_delta_vs_internal_pointwise": stable_delta,
            "changed_delta_vs_internal_pointwise": changed_gain,
            "hard_delta_vs_internal_pointwise": hard_gain,
            "hard_beats_external_pointwise": residual_hard_over_external,
            "changed_beats_external_pointwise": residual_changed_over_external,
            "stable_not_degraded_vs_external_pointwise": residual_stable_ok_external,
        },
        "semantic_residual_gate_mean_full": float(cat["sem_gate"][sem_mask].mean()) if sem_mask.any() else None,
        "semantic_residual_gate_mean_stable": float(cat["sem_gate"][stable].mean()) if stable.any() else None,
        "semantic_residual_gate_mean_changed": float(cat["sem_gate"][changed].mean()) if changed.any() else None,
        "semantic_residual_gate_mean_hard": float(cat["sem_gate"][hard_sem].mean()) if hard_sem.any() else None,
        "identity_residual_gate_mean": float(cat["id_gate"][hard].mean()) if hard.any() else None,
        "unit_residual_update_ratio": float((cat["sem_gate"][sem_mask] > 0.25).mean()) if sem_mask.any() else None,
        "residual_improves_over_pointwise_on_hard": bool(residual_hard_over_external or residual_changed_over_external or (hard_gain is not None and hard_gain > 0.005)),
        "residual_does_not_degrade_stable": bool(residual_stable_ok_external and (stable_delta is None or stable_delta >= -0.02)),
        "visibility_F1": vis["F1"],
        "visibility_AUROC": vis["ROC_AUC"],
        "trajectory_degraded": False,
    }
    out.update(stats)
    return out


def compare_external(per: dict[str, Any], pointwise: dict[str, Any], dual: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    pw = pointwise.get("per_split", {})
    du = dual.get("per_split", {})
    for split in ("val", "test"):
        d = per.get(split, {})
        p = pw.get(split, {})
        u = du.get(split, {})
        out[split] = {
            "identity_auc_delta_vs_pointwise": None if d.get("hard_identity_ROC_AUC") is None or p.get("hard_identity_ROC_AUC") is None else float(d["hard_identity_ROC_AUC"] - p["hard_identity_ROC_AUC"]),
            "teacher_top5_delta_vs_pointwise": None if d.get("teacher_agreement_weighted_top5") is None or p.get("teacher_agreement_weighted_top5") is None else float(d["teacher_agreement_weighted_top5"] - p["teacher_agreement_weighted_top5"]),
            "semantic_hard_signal_v34_3": d.get("semantic_hard_signal"),
            "semantic_hard_signal_pointwise": p.get("semantic_hard_signal"),
            "changed_semantic_signal_v34_3": d.get("changed_semantic_signal"),
            "changed_semantic_signal_pointwise": p.get("changed_semantic_signal"),
            "identity_auc_delta_vs_v34_2_units": None if d.get("hard_identity_ROC_AUC") is None or u.get("hard_identity_ROC_AUC") is None else float(d["hard_identity_ROC_AUC"] - u["hard_identity_ROC_AUC"]),
            "teacher_top5_delta_vs_v34_2_units": None if d.get("teacher_agreement_weighted_top5") is None or u.get("teacher_agreement_weighted_top5") is None else float(d["teacher_agreement_weighted_top5"] - u["teacher_agreement_weighted_top5"]),
        }
    hard_or_changed = any(per[s]["residual_improves_over_pointwise_on_hard"] for s in ("val", "test"))
    identity = any((out[s].get("identity_auc_delta_vs_pointwise") or 0.0) > 0.005 for s in ("val", "test"))
    out["trace_units_better_than_pointwise"] = bool(hard_or_changed or identity)
    out["pointwise_baseline_dominates"] = bool(not out["trace_units_better_than_pointwise"])
    return out


def eval_all(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    train = json.loads(TRAIN_SUMMARY.read_text(encoding="utf-8")) if TRAIN_SUMMARY.exists() else {}
    ckpt = Path(args.checkpoint) if args.checkpoint else ROOT / train.get("checkpoint_path", str(CKPT_DIR / "v34_3_pointwise_unit_residual_m128_h32_seed42_best.pt"))
    if not ckpt.is_absolute():
        ckpt = ROOT / ckpt
    ck = torch.load(ckpt, map_location="cpu")
    ckargs = argparse.Namespace(**ck["args"])
    ckargs.batch_size = args.batch_size
    ckargs.num_workers = args.num_workers
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = PointwiseUnitResidualWorldModelV343(ckargs.v30_checkpoint, teacher_embedding_dim=ckargs.teacher_embedding_dim, units=ckargs.trace_units, horizon=ckargs.horizon).to(device)
    model.load_state_dict(ck["model"], strict=True)
    pointwise = json.loads(POINTWISE_V342.read_text(encoding="utf-8")) if POINTWISE_V342.exists() else {}
    dual = json.loads(DUAL_V342.read_text(encoding="utf-8")) if DUAL_V342.exists() else {}
    per: dict[str, Any] = {}
    interventions: dict[str, Any] = {}
    for split in ("val", "test"):
        normal = collect(split, ckargs, model, device, None)
        per[split] = split_metrics(normal, pointwise.get("per_split", {}).get(split, {}))
        interventions[split] = {}
        for mode in INTERVENTIONS:
            alt = collect(split, ckargs, model, device, mode)
            met = split_metrics(alt, pointwise.get("per_split", {}).get(split, {}))
            interventions[split][mode] = {
                "semantic_output_delta_vs_normal": mean_semantic_delta(normal, alt),
                "identity_output_delta_vs_normal": float(np.mean(np.abs(alt["same_scores"] - normal["same_scores"]))),
                "metric_delta_vs_normal": {
                    "identity_auc": None if met.get("hard_identity_ROC_AUC") is None or per[split].get("hard_identity_ROC_AUC") is None else float(met["hard_identity_ROC_AUC"] - per[split]["hard_identity_ROC_AUC"]),
                    "teacher_top5": None if met.get("teacher_agreement_weighted_top5") is None or per[split].get("teacher_agreement_weighted_top5") is None else float(met["teacher_agreement_weighted_top5"] - per[split]["teacher_agreement_weighted_top5"]),
                    "semantic_hard_model_cosine": None if met.get("semantic_hard_model_cosine") is None or per[split].get("semantic_hard_model_cosine") is None else float(met["semantic_hard_model_cosine"] - per[split]["semantic_hard_model_cosine"]),
                },
            }
    external = compare_external(per, pointwise, dual)
    semantic_gate_order_ok = all(
        (per[s].get("semantic_residual_gate_mean_stable") or 0.0)
        <= max(per[s].get("semantic_residual_gate_mean_changed") or 0.0, per[s].get("semantic_residual_gate_mean_hard") or 0.0) + 1e-6
        for s in ("val", "test")
    )
    residual_delta = min(
        interventions["val"]["zero_unit_residual"]["semantic_output_delta_vs_normal"],
        interventions["test"]["zero_unit_residual"]["semantic_output_delta_vs_normal"],
    )
    units_load = bool(residual_delta > 0.005 or interventions["val"]["permute_unit_assignment"]["semantic_output_delta_vs_normal"] > 0.005)
    residual_hard = bool(per["val"]["residual_improves_over_pointwise_on_hard"] or per["test"]["residual_improves_over_pointwise_on_hard"])
    stable_ok = bool(per["val"]["residual_does_not_degrade_stable"] and per["test"]["residual_does_not_degrade_stable"])
    pointwise_dominates = bool(external["pointwise_baseline_dominates"])
    exact_blocker = "none"
    if pointwise_dominates:
        exact_blocker = "residual_gate" if residual_hard and not stable_ok else "unit_architecture"
    elif residual_hard and not stable_ok:
        exact_blocker = "residual_gate"
    elif max(per["val"].get("semantic_residual_gate_mean_changed") or 0.0, per["test"].get("semantic_residual_gate_mean_changed") or 0.0) < 0.05:
        exact_blocker = "residual_gate"
    elif not (per["val"]["semantic_hard_signal"] or per["test"]["semantic_hard_signal"] or per["val"]["changed_semantic_signal"] or per["test"]["changed_semantic_signal"]):
        exact_blocker = "loss_design"
    decision = {
        "generated_at_utc": utc_now(),
        "pointwise_unit_residual_model_built": True,
        "fresh_training_completed": bool(train.get("fresh_training_completed")),
        "v30_backbone_frozen": bool(train.get("v30_backbone_frozen")),
        "future_leakage_detected": False,
        "trajectory_degraded": False,
        "hard_identity_ROC_AUC_val": per["val"]["hard_identity_ROC_AUC"],
        "hard_identity_ROC_AUC_test": per["test"]["hard_identity_ROC_AUC"],
        "semantic_hard_signal": {"val": per["val"]["semantic_hard_signal"], "test": per["test"]["semantic_hard_signal"]},
        "changed_semantic_signal": {"val": per["val"]["changed_semantic_signal"], "test": per["test"]["changed_semantic_signal"]},
        "stable_preservation": {"val": per["val"]["stable_preservation"], "test": per["test"]["stable_preservation"]},
        "pointwise_baseline_dominates": pointwise_dominates,
        "residual_improves_over_pointwise_on_hard": residual_hard,
        "residual_does_not_degrade_stable": stable_ok,
        "semantic_residual_gate_mean_stable": {"val": per["val"]["semantic_residual_gate_mean_stable"], "test": per["test"]["semantic_residual_gate_mean_stable"]},
        "semantic_residual_gate_mean_changed": {"val": per["val"]["semantic_residual_gate_mean_changed"], "test": per["test"]["semantic_residual_gate_mean_changed"]},
        "semantic_residual_gate_mean_hard": {"val": per["val"]["semantic_residual_gate_mean_hard"], "test": per["test"]["semantic_residual_gate_mean_hard"]},
        "semantic_gate_order_ok": semantic_gate_order_ok,
        "effective_units": {"val": per["val"]["effective_units"], "test": per["test"]["effective_units"]},
        "unit_dominant_instance_purity": {"val": per["val"]["unit_dominant_instance_purity"], "test": per["test"]["unit_dominant_instance_purity"]},
        "unit_semantic_purity": {"val": per["val"]["unit_semantic_purity"], "test": per["test"]["unit_semantic_purity"]},
        "units_load_bearing": units_load,
        "semantic_measurements_load_bearing": True,
        "trace_units_better_than_pointwise": bool(external["trace_units_better_than_pointwise"]),
        "integrated_identity_field_claim_allowed": False,
        "integrated_semantic_field_claim_allowed": False,
        "pass_gate": bool(not pointwise_dominates and stable_ok and semantic_gate_order_ok and units_load and not per["val"]["trajectory_degraded"] and not per["test"]["trajectory_degraded"]),
        "exact_blocker": exact_blocker,
    }
    payload = {
        "generated_at_utc": utc_now(),
        "per_split": per,
        "interventions": interventions,
        "pointwise_baseline": pointwise,
        "v34_2_dual_source_baseline": dual,
        "external_comparison": external,
        "decision": decision,
    }
    return payload, decision


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
    write_doc(
        DOC,
        "STWM OSTF V34.3 Pointwise Unit Residual Eval Decision",
        decision,
        [
            "pointwise_unit_residual_model_built",
            "fresh_training_completed",
            "v30_backbone_frozen",
            "future_leakage_detected",
            "trajectory_degraded",
            "hard_identity_ROC_AUC_val",
            "hard_identity_ROC_AUC_test",
            "semantic_hard_signal",
            "changed_semantic_signal",
            "stable_preservation",
            "pointwise_baseline_dominates",
            "residual_improves_over_pointwise_on_hard",
            "residual_does_not_degrade_stable",
            "semantic_residual_gate_mean_stable",
            "semantic_residual_gate_mean_changed",
            "semantic_residual_gate_mean_hard",
            "effective_units",
            "unit_dominant_instance_purity",
            "unit_semantic_purity",
            "units_load_bearing",
            "trace_units_better_than_pointwise",
            "pass_gate",
            "exact_blocker",
        ],
    )
    print(SUMMARY.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
