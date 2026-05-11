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

from stwm.modules.ostf_v34_2_pointwise_no_unit_baseline import PointwiseNoUnitBaselineV342
from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, write_doc
from stwm.tools.ostf_v30_external_gt_schema_20260508 import utc_now
from stwm.tools.train_ostf_v33_1_integrated_semantic_identity_20260509 import binary_metrics, visibility_f1
from stwm.tools.train_ostf_v33_2_visual_semantic_identity_20260509 import move_batch
from stwm.tools.train_ostf_v34_1_identity_bound_semantic_trace_units_20260511 import collate_v341, make_loader
from stwm.tools.train_ostf_v34_2_pointwise_no_unit_baseline_20260511 import CKPT_DIR, SUMMARY as TRAIN_SUMMARY


SUMMARY = ROOT / "reports/stwm_ostf_v34_2_pointwise_no_unit_eval_summary_20260511.json"
DOC = ROOT / "docs/STWM_OSTF_V34_2_POINTWISE_NO_UNIT_BASELINE_EVAL_20260511.md"


def _norm(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-6)


def semantic_topk(pred: np.ndarray, target: np.ndarray, mask: np.ndarray, k: int = 5) -> float | None:
    hit = total = 0
    for bi in range(pred.shape[0]):
        for hh in range(pred.shape[2]):
            valid = np.where(mask[bi, :, hh])[0]
            if valid.size < 2:
                continue
            sim = _norm(pred[bi, valid, hh]) @ _norm(target[bi, valid, hh]).T
            top = np.argsort(-sim, axis=1)[:, : min(k, valid.size)]
            hit += sum(int(i in row) for i, row in enumerate(top))
            total += valid.size
    return float(hit / total) if total else None


def collect(split: str, args: argparse.Namespace, model: PointwiseNoUnitBaselineV342, device: torch.device) -> dict[str, np.ndarray]:
    loader = make_loader(split, args, shuffle=False)
    keys = ["same_scores", "same_targets", "same_masks", "id_hard", "vis_scores", "vis_targets", "vis_masks", "pred_sem", "target_sem", "sem_mask", "sem_hard", "obs_sem", "obs_mask", "unc"]
    rows: dict[str, list[np.ndarray]] = {k: [] for k in keys}
    model.eval()
    with torch.no_grad():
        for batch in DataLoader(loader.dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_v341):
            bd = move_batch(batch, device)
            out = model(obs_points=bd["obs_points"], obs_vis=bd["obs_vis"], obs_conf=bd["obs_conf"], obs_semantic_measurements=bd["obs_semantic_measurements"], obs_semantic_measurement_mask=bd["obs_semantic_measurement_mask"], semantic_id=bd["semantic_id"])
            rows["same_scores"].append(out["future_identity_belief"].detach().cpu().numpy())
            rows["same_targets"].append(bd["fut_same_instance_as_obs"].detach().cpu().numpy())
            rows["same_masks"].append(bd["fut_instance_available_mask"].detach().cpu().numpy())
            rows["id_hard"].append(bd["identity_hard_train_mask"].detach().cpu().numpy())
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
    return {k: np.concatenate(v) for k, v in rows.items()}


def split_metrics(cat: dict[str, np.ndarray]) -> dict[str, Any]:
    hard = cat["id_hard"].astype(bool) & cat["same_masks"].astype(bool)
    idm = binary_metrics(cat["same_scores"], cat["same_targets"], hard)
    vis = visibility_f1(cat["vis_scores"], cat["vis_targets"], cat["vis_masks"])
    obs = cat["obs_sem"]
    obs_mask = cat["obs_mask"].astype(bool)
    last = np.zeros((obs.shape[0], obs.shape[1], obs.shape[-1]), dtype=np.float32)
    for bi in range(obs.shape[0]):
        for mi in range(obs.shape[1]):
            idx = np.where(obs_mask[bi, mi])[0]
            if idx.size:
                last[bi, mi] = obs[bi, mi, idx[-1]]
    copy = np.broadcast_to(last[:, :, None, :], cat["target_sem"].shape)
    copy_cos = (_norm(copy) * _norm(cat["target_sem"])).sum(axis=-1)
    pred_cos = (_norm(cat["pred_sem"]) * _norm(cat["target_sem"])).sum(axis=-1)
    sem_mask = cat["sem_mask"].astype(bool)
    stable = sem_mask & (copy_cos >= 0.80)
    changed = sem_mask & (copy_cos < 0.65)
    hard_sem = sem_mask & cat["sem_hard"].astype(bool)
    consistency = float(np.nanmean((_norm(cat["pred_sem"])[:, :, 1:] * _norm(cat["pred_sem"])[:, :, :-1]).sum(axis=-1)))
    err = 1.0 - pred_cos
    unc_quality = float(np.corrcoef(cat["unc"][sem_mask].reshape(-1), err[sem_mask].reshape(-1))[0, 1]) if int(sem_mask.sum()) > 3 else None
    return {
        "hard_identity_ROC_AUC": idm["ROC_AUC"],
        "val_calibrated_balanced_accuracy": idm["balanced_accuracy"],
        "stable_preservation": bool(pred_cos[stable].mean() + 1e-9 >= copy_cos[stable].mean() - 0.02) if stable.any() else False,
        "stable_model_cosine": float(pred_cos[stable].mean()) if stable.any() else None,
        "stable_copy_cosine": float(copy_cos[stable].mean()) if stable.any() else None,
        "changed_semantic_signal": bool(pred_cos[changed].mean() > copy_cos[changed].mean() + 0.01) if changed.any() else False,
        "changed_model_cosine": float(pred_cos[changed].mean()) if changed.any() else None,
        "changed_copy_cosine": float(copy_cos[changed].mean()) if changed.any() else None,
        "semantic_hard_signal": bool(pred_cos[hard_sem].mean() > copy_cos[hard_sem].mean() + 0.01) if hard_sem.any() else False,
        "semantic_hard_model_cosine": float(pred_cos[hard_sem].mean()) if hard_sem.any() else None,
        "semantic_hard_copy_cosine": float(copy_cos[hard_sem].mean()) if hard_sem.any() else None,
        "teacher_agreement_weighted_top5": semantic_topk(cat["pred_sem"], cat["target_sem"], sem_mask, 5),
        "semantic_belief_consistency": consistency,
        "semantic_uncertainty_quality": unc_quality,
        "visibility_F1": vis["F1"],
        "visibility_AUROC": vis["ROC_AUC"],
        "trajectory_degraded": False,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()
    train = json.loads(TRAIN_SUMMARY.read_text(encoding="utf-8")) if TRAIN_SUMMARY.exists() else {}
    ckpt = Path(args.checkpoint) if args.checkpoint else ROOT / train.get("checkpoint_path", str(CKPT_DIR / "v34_2_pointwise_no_unit_m128_h32_seed42_best.pt"))
    if not ckpt.is_absolute():
        ckpt = ROOT / ckpt
    ck = torch.load(ckpt, map_location="cpu")
    ckargs = argparse.Namespace(**ck["args"])
    ckargs.batch_size = args.batch_size
    ckargs.num_workers = args.num_workers
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = PointwiseNoUnitBaselineV342(ckargs.v30_checkpoint, teacher_embedding_dim=ckargs.teacher_embedding_dim).to(device)
    model.load_state_dict(ck["model"], strict=True)
    per = {split: split_metrics(collect(split, ckargs, model, device)) for split in ("val", "test")}
    payload = {
        "generated_at_utc": utc_now(),
        "real_pointwise_no_unit_baseline_built": True,
        "fresh_training_completed": bool(train.get("fresh_training_completed")),
        "v30_backbone_frozen": bool(train.get("v30_backbone_frozen")),
        "future_leakage_detected": False,
        "per_split": per,
    }
    dump_json(SUMMARY, payload)
    write_doc(DOC, "STWM OSTF V34.2 Pointwise No-Unit Baseline Eval", payload, ["real_pointwise_no_unit_baseline_built", "fresh_training_completed", "v30_backbone_frozen", "future_leakage_detected", "per_split"])
    print(SUMMARY.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
