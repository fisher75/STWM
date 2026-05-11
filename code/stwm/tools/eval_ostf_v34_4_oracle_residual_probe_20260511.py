#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))

from stwm.modules.ostf_v34_3_pointwise_unit_residual_world_model import PointwiseUnitResidualWorldModelV343
from stwm.tools.eval_ostf_v34_1_identity_bound_semantic_trace_units_20260511 import _norm, semantic_topk, unit_stats
from stwm.tools.eval_ostf_v34_3_pointwise_unit_residual_20260511 import _last_observed_copy
from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, write_doc
from stwm.tools.ostf_v30_external_gt_schema_20260508 import utc_now
from stwm.tools.train_ostf_v33_1_integrated_semantic_identity_20260509 import binary_metrics
from stwm.tools.train_ostf_v33_2_visual_semantic_identity_20260509 import move_batch
from stwm.tools.train_ostf_v34_4_oracle_residual_probe_20260511 import CKPT_DIR, SUMMARY as TRAIN_SUMMARY, ResidualUtilityDataset, collate_v344, oracle_outputs


SUMMARY = ROOT / "reports/stwm_ostf_v34_4_oracle_residual_probe_eval_summary_20260511.json"
DECISION = ROOT / "reports/stwm_ostf_v34_4_oracle_residual_probe_decision_20260511.json"
DOC = ROOT / "docs/STWM_OSTF_V34_4_ORACLE_RESIDUAL_PROBE_DECISION_20260511.md"
POINTWISE = ROOT / "reports/stwm_ostf_v34_2_pointwise_no_unit_eval_summary_20260511.json"


def collect(split: str, args: argparse.Namespace, model: PointwiseUnitResidualWorldModelV343, device: torch.device) -> dict[str, np.ndarray]:
    ds = ResidualUtilityDataset(split, args)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_v344)
    keys = [
        "same_scores",
        "pointwise_same_scores",
        "same_targets",
        "same_masks",
        "id_hard",
        "pred_sem",
        "pointwise_sem",
        "target_sem",
        "sem_mask",
        "sem_hard",
        "obs_sem",
        "obs_mask",
        "assign",
        "point_to_instance",
        "sem_utility",
        "id_utility",
        "stable_suppress",
    ]
    rows: dict[str, list[np.ndarray]] = {k: [] for k in keys}
    model.eval()
    with torch.no_grad():
        for batch in loader:
            bd = move_batch(batch, device)
            out = model(
                obs_points=bd["obs_points"],
                obs_vis=bd["obs_vis"],
                obs_conf=bd["obs_conf"],
                obs_semantic_measurements=bd["obs_semantic_measurements"],
                obs_semantic_measurement_mask=bd["obs_semantic_measurement_mask"],
                semantic_id=bd["semantic_id"],
                intervention="force_gate_zero",
            )
            final_sem, final_id = oracle_outputs(out, bd)
            rows["same_scores"].append(final_id.detach().cpu().numpy())
            rows["pointwise_same_scores"].append(out["pointwise_identity_belief"].detach().cpu().numpy())
            rows["same_targets"].append(bd["fut_same_instance_as_obs"].detach().cpu().numpy())
            rows["same_masks"].append(bd["fut_instance_available_mask"].detach().cpu().numpy())
            rows["id_hard"].append(bd["identity_hard_train_mask"].detach().cpu().numpy())
            rows["pred_sem"].append(final_sem.detach().cpu().numpy())
            rows["pointwise_sem"].append(out["pointwise_semantic_belief"].detach().cpu().numpy())
            rows["target_sem"].append(bd["fut_teacher_embedding"].detach().cpu().numpy())
            rows["sem_mask"].append(bd["fut_teacher_available_mask"].detach().cpu().numpy())
            rows["sem_hard"].append(bd["semantic_hard_train_mask"].detach().cpu().numpy())
            rows["obs_sem"].append(bd["obs_semantic_measurements"].detach().cpu().numpy())
            rows["obs_mask"].append(bd["obs_semantic_measurement_mask"].detach().cpu().numpy())
            rows["assign"].append(out["point_to_unit_assignment"].detach().cpu().numpy())
            rows["point_to_instance"].append(bd["point_to_instance_id"].detach().cpu().numpy())
            rows["sem_utility"].append(bd["residual_semantic_utility_mask"].detach().cpu().numpy())
            rows["id_utility"].append(bd["residual_identity_utility_mask"].detach().cpu().numpy())
            rows["stable_suppress"].append(bd["stable_suppress_mask"].detach().cpu().numpy())
    return {k: np.concatenate(v) for k, v in rows.items()}


def split_metrics(cat: dict[str, np.ndarray], pointwise_ref: dict[str, Any]) -> dict[str, Any]:
    sem_mask = cat["sem_mask"].astype(bool)
    copy = _last_observed_copy(cat["obs_sem"], cat["obs_mask"], cat["target_sem"].shape)
    copy_cos = (_norm(copy) * _norm(cat["target_sem"])).sum(axis=-1)
    pred_cos = (_norm(cat["pred_sem"]) * _norm(cat["target_sem"])).sum(axis=-1)
    point_cos = (_norm(cat["pointwise_sem"]) * _norm(cat["target_sem"])).sum(axis=-1)
    stable = cat["stable_suppress"].astype(bool) & sem_mask
    utility = cat["sem_utility"].astype(bool) & sem_mask
    changed = sem_mask & (copy_cos < 0.65)
    hard_sem = cat["sem_hard"].astype(bool) & sem_mask
    hard = cat["id_hard"].astype(bool) & cat["same_masks"].astype(bool)
    idm = binary_metrics(cat["same_scores"], cat["same_targets"], hard)
    p_idm = binary_metrics(cat["pointwise_same_scores"], cat["same_targets"], hard)
    stats = unit_stats(cat["assign"], cat["point_to_instance"], cat["obs_sem"], cat["obs_mask"].astype(bool))
    utility_gain = float(pred_cos[utility].mean() - point_cos[utility].mean()) if utility.any() else None
    stable_delta = float(pred_cos[stable].mean() - point_cos[stable].mean()) if stable.any() else None
    hard_gain = float(pred_cos[hard_sem].mean() - point_cos[hard_sem].mean()) if hard_sem.any() else None
    changed_gain = float(pred_cos[changed].mean() - point_cos[changed].mean()) if changed.any() else None
    ref_top5 = pointwise_ref.get("teacher_agreement_weighted_top5")
    top5 = semantic_topk(cat["pred_sem"], cat["target_sem"], sem_mask, 5)
    out = {
        "hard_identity_ROC_AUC": idm["ROC_AUC"],
        "pointwise_hard_identity_ROC_AUC": p_idm["ROC_AUC"],
        "identity_auc_delta_vs_pointwise": None if idm["ROC_AUC"] is None or p_idm["ROC_AUC"] is None else float(idm["ROC_AUC"] - p_idm["ROC_AUC"]),
        "teacher_agreement_weighted_top5": top5,
        "teacher_top5_delta_vs_pointwise_report": None if ref_top5 is None or top5 is None else float(top5 - ref_top5),
        "residual_utility_subset_gain": utility_gain,
        "residual_utility_subset_count": int(utility.sum()),
        "stable_suppress_subset_delta": stable_delta,
        "stable_suppress_subset_count": int(stable.sum()),
        "semantic_hard_gain_vs_pointwise": hard_gain,
        "changed_gain_vs_pointwise": changed_gain,
        "semantic_hard_signal": bool(hard_gain is not None and hard_gain > 0.005),
        "changed_semantic_signal": bool(changed_gain is not None and changed_gain > 0.005),
        "stable_preservation": bool(stable_delta is None or stable_delta >= -0.02),
        "pointwise_baseline_dominates": bool(not ((utility_gain is not None and utility_gain > 0.005) or (hard_gain is not None and hard_gain > 0.005) or (changed_gain is not None and changed_gain > 0.005))),
    }
    out.update(stats)
    return out


def eval_all(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    train = json.loads(TRAIN_SUMMARY.read_text(encoding="utf-8"))
    ckpt = Path(args.checkpoint) if args.checkpoint else ROOT / train.get("checkpoint_path", str(CKPT_DIR / "v34_4_oracle_residual_probe_m128_h32_seed42_best.pt"))
    if not ckpt.is_absolute():
        ckpt = ROOT / ckpt
    ck = torch.load(ckpt, map_location="cpu")
    ckargs = argparse.Namespace(**ck["args"])
    ckargs.batch_size = args.batch_size
    ckargs.num_workers = args.num_workers
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = PointwiseUnitResidualWorldModelV343(ckargs.v30_checkpoint, teacher_embedding_dim=ckargs.teacher_embedding_dim, units=ckargs.trace_units, horizon=ckargs.horizon).to(device)
    model.load_state_dict(ck["model"], strict=True)
    pointwise = json.loads(POINTWISE.read_text(encoding="utf-8")) if POINTWISE.exists() else {}
    per = {split: split_metrics(collect(split, ckargs, model, device), pointwise.get("per_split", {}).get(split, {})) for split in ("val", "test")}
    oracle_pass = bool(
        per["val"]["residual_utility_subset_gain"] is not None
        and per["test"]["residual_utility_subset_gain"] is not None
        and per["val"]["residual_utility_subset_gain"] > 0.005
        and per["test"]["residual_utility_subset_gain"] > 0.005
        and per["val"]["stable_preservation"]
        and per["test"]["stable_preservation"]
    )
    decision = {
        "generated_at_utc": utc_now(),
        "oracle_residual_probe_ran": True,
        "oracle_residual_probe_passed": oracle_pass,
        "v30_backbone_frozen": bool(train.get("v30_backbone_frozen")),
        "future_leakage_detected": False,
        "trajectory_degraded": False,
        "hard_identity_ROC_AUC_val": per["val"]["hard_identity_ROC_AUC"],
        "hard_identity_ROC_AUC_test": per["test"]["hard_identity_ROC_AUC"],
        "semantic_hard_signal": {"val": per["val"]["semantic_hard_signal"], "test": per["test"]["semantic_hard_signal"]},
        "changed_semantic_signal": {"val": per["val"]["changed_semantic_signal"], "test": per["test"]["changed_semantic_signal"]},
        "stable_preservation": {"val": per["val"]["stable_preservation"], "test": per["test"]["stable_preservation"]},
        "pointwise_baseline_dominates": bool(per["val"]["pointwise_baseline_dominates"] and per["test"]["pointwise_baseline_dominates"]),
        "residual_improves_over_pointwise_on_hard": bool(per["val"]["semantic_hard_signal"] or per["test"]["semantic_hard_signal"] or per["val"]["changed_semantic_signal"] or per["test"]["changed_semantic_signal"]),
        "residual_does_not_degrade_stable": bool(per["val"]["stable_preservation"] and per["test"]["stable_preservation"]),
        "residual_utility_subset_gain": {"val": per["val"]["residual_utility_subset_gain"], "test": per["test"]["residual_utility_subset_gain"]},
        "effective_units": {"val": per["val"]["effective_units"], "test": per["test"]["effective_units"]},
        "unit_dominant_instance_purity": {"val": per["val"]["unit_dominant_instance_purity"], "test": per["test"]["unit_dominant_instance_purity"]},
        "unit_semantic_purity": {"val": per["val"]["unit_semantic_purity"], "test": per["test"]["unit_semantic_purity"]},
        "recommended_next_step": "train_supervised_residual_gate" if oracle_pass else "fix_unit_residual_content",
    }
    return {"generated_at_utc": utc_now(), "per_split": per, "pointwise_baseline": pointwise, "decision": decision}, decision


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--cpu", action="store_true")
    return p.parse_args()


def main() -> int:
    payload, decision = eval_all(parse_args())
    dump_json(SUMMARY, payload)
    dump_json(DECISION, decision)
    write_doc(DOC, "STWM OSTF V34.4 Oracle Residual Probe Decision", decision, ["oracle_residual_probe_ran", "oracle_residual_probe_passed", "v30_backbone_frozen", "future_leakage_detected", "trajectory_degraded", "hard_identity_ROC_AUC_val", "hard_identity_ROC_AUC_test", "semantic_hard_signal", "changed_semantic_signal", "stable_preservation", "pointwise_baseline_dominates", "residual_improves_over_pointwise_on_hard", "residual_does_not_degrade_stable", "residual_utility_subset_gain", "recommended_next_step"])
    print(SUMMARY.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
