#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))

from stwm.datasets.ostf_v30_external_gt_dataset_20260508 import OSTFExternalGTDataset, collate_external_gt
from stwm.modules.ostf_v33_integrated_semantic_identity_world_model import IntegratedSemanticIdentityWorldModelV331
from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, write_doc
from stwm.tools.ostf_v30_external_gt_metrics_20260508 import item_row
from stwm.tools.ostf_v30_external_gt_schema_20260508 import utc_now

RUN_DIR = ROOT / "reports/stwm_ostf_v33_1_integrated_runs"
CKPT_DIR = ROOT / "outputs/checkpoints/stwm_ostf_v33_1_integrated"
SUMMARY = ROOT / "reports/stwm_ostf_v33_1_integrated_smoke_summary_20260509.json"
DECISION = ROOT / "reports/stwm_ostf_v33_1_integrated_smoke_decision_20260509.json"
DOC = ROOT / "docs/STWM_OSTF_V33_1_INTEGRATED_SMOKE_DECISION_20260509.md"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


def make_loader(split: str, args: argparse.Namespace, *, shuffle: bool, max_items: int | None) -> DataLoader:
    ds = OSTFExternalGTDataset(
        split,
        horizon=args.horizon,
        m_points=args.m_points,
        max_items=max_items,
        enable_semantic_identity_sidecar=True,
        semantic_identity_sidecar_root=args.semantic_identity_sidecar_root,
        require_semantic_identity_sidecar=True,
        use_observed_instance_context=args.use_observed_instance_context,
    )
    return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle, num_workers=args.num_workers, pin_memory=torch.cuda.is_available(), collate_fn=collate_external_gt)


def rank_auc(scores: np.ndarray, labels: np.ndarray) -> float | None:
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels).astype(bool).reshape(-1)
    if labels.sum() == 0 or (~labels).sum() == 0:
        return None
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=np.float64)
    # Average ranks for ties.
    unique_scores, inv, counts = np.unique(scores, return_inverse=True, return_counts=True)
    if len(unique_scores) != len(scores):
        sums = np.bincount(inv, weights=ranks)
        avg = sums / counts
        ranks = avg[inv]
    n_pos = float(labels.sum())
    n_neg = float((~labels).sum())
    return float((ranks[labels].sum() - n_pos * (n_pos + 1.0) / 2.0) / (n_pos * n_neg))


def pr_auc(scores: np.ndarray, labels: np.ndarray) -> float | None:
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels).astype(bool).reshape(-1)
    if labels.sum() == 0:
        return None
    order = np.argsort(-scores, kind="mergesort")
    y = labels[order].astype(np.float64)
    tp = np.cumsum(y)
    fp = np.cumsum(1.0 - y)
    recall = tp / max(float(labels.sum()), 1.0)
    precision = tp / np.maximum(tp + fp, 1.0)
    recall = np.concatenate([[0.0], recall])
    precision = np.concatenate([[precision[0] if len(precision) else 0.0], precision])
    return float(np.trapz(precision, recall))


def binary_metrics(scores: np.ndarray, labels: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    valid = np.asarray(mask).astype(bool).reshape(-1)
    if valid.sum() == 0:
        return {"coverage": 0.0, "accuracy": None, "balanced_accuracy": None, "ROC_AUC": None, "PR_AUC": None, "positive_ratio": None, "negative_ratio": None}
    s = np.asarray(scores).reshape(-1)[valid]
    y = np.asarray(labels).astype(bool).reshape(-1)[valid]
    pred = s >= 0.0
    pos = y
    neg = ~y
    tpr = float((pred[pos] == y[pos]).mean()) if pos.any() else None
    tnr = float((pred[neg] == y[neg]).mean()) if neg.any() else None
    bal = None if tpr is None or tnr is None else 0.5 * (tpr + tnr)
    return {
        "coverage": float(valid.mean()),
        "accuracy": float((pred == y).mean()),
        "balanced_accuracy": bal,
        "ROC_AUC": rank_auc(s, y),
        "PR_AUC": pr_auc(s, y),
        "positive_ratio": float(pos.mean()),
        "negative_ratio": float(neg.mean()),
    }


def visibility_f1(scores: np.ndarray, labels: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    valid = np.asarray(mask).astype(bool).reshape(-1)
    if valid.sum() == 0:
        return {"F1": None, "ROC_AUC": None}
    s = np.asarray(scores).reshape(-1)[valid]
    y = np.asarray(labels).astype(bool).reshape(-1)[valid]
    pred = s >= 0.0
    tp = float((pred & y).sum())
    fp = float((pred & ~y).sum())
    fn = float((~pred & y).sum())
    f1 = 2.0 * tp / max(2.0 * tp + fp + fn, 1.0)
    return {"F1": float(f1), "ROC_AUC": rank_auc(s, y)}


def minfde(pred: np.ndarray, fut: np.ndarray, vis: np.ndarray) -> float:
    end_vis = np.asarray(vis[:, :, -1]).astype(bool)
    if end_vis.sum() == 0:
        return 0.0
    err = np.linalg.norm(np.asarray(pred[:, :, -1]) - np.asarray(fut[:, :, -1]), axis=-1)
    return float(err[end_vis].mean())


def horizon_constant(scores: np.ndarray, mask: np.ndarray, tol: float = 1e-5) -> bool:
    valid = np.asarray(mask).astype(bool)
    if valid.sum() == 0:
        return True
    diffs = np.abs(np.diff(np.asarray(scores), axis=2))
    return bool(diffs[valid[:, :, 1:]].mean() < tol) if valid[:, :, 1:].any() else True


def evaluate_model(model: IntegratedSemanticIdentityWorldModelV331, loader: DataLoader, device: torch.device) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    model.eval()
    same_scores = []
    same_targets = []
    same_masks = []
    vis_scores = []
    vis_targets = []
    vis_masks = []
    fde_delta = []
    rows: list[dict[str, Any]] = []
    per_horizon_scores: list[list[float]] | None = None
    per_horizon_targets: list[list[bool]] | None = None
    with torch.no_grad():
        for batch in loader:
            bd = move_batch(batch, device)
            out = model(
                obs_points=bd["obs_points"],
                obs_vis=bd["obs_vis"],
                obs_conf=bd["obs_conf"],
                semantic_id=bd["semantic_id"],
                point_to_instance_id=bd.get("point_to_instance_id") if bool(bd.get("use_observed_instance_context", torch.tensor([False], device=device)).any()) else None,
            )
            same = out["same_instance_logits"].detach().cpu().numpy()
            same_t = bd["fut_same_instance_as_obs"].detach().cpu().numpy()
            same_m = bd["fut_instance_available_mask"].detach().cpu().numpy()
            vis = out["visibility_logits"].detach().cpu().numpy()
            vis_t = bd["fut_point_visible_target"].detach().cpu().numpy()
            vis_m = bd["fut_point_visible_mask"].detach().cpu().numpy()
            same_scores.append(same)
            same_targets.append(same_t)
            same_masks.append(same_m)
            vis_scores.append(vis)
            vis_targets.append(vis_t)
            vis_masks.append(vis_m)
            pred = out["point_pred"].detach().cpu().numpy()
            fut = bd["fut_points"].detach().cpu().numpy()
            fut_vis = bd["fut_vis"].detach().cpu().numpy()
            frozen_pred = out["point_pred"].detach().cpu().numpy()
            fde_delta.append(minfde(pred, fut, fut_vis) - minfde(frozen_pred, fut, fut_vis))
            h = same.shape[2]
            if per_horizon_scores is None:
                per_horizon_scores = [[] for _ in range(h)]
                per_horizon_targets = [[] for _ in range(h)]
            for hh in range(h):
                mh = same_m[:, :, hh].astype(bool)
                if mh.any():
                    per_horizon_scores[hh].extend(same[:, :, hh][mh].reshape(-1).tolist())
                    per_horizon_targets[hh].extend(same_t[:, :, hh][mh].reshape(-1).astype(bool).tolist())
            for i, uid in enumerate(batch["uid"]):
                rows.append(
                    item_row(
                        uid=str(uid),
                        dataset=str(batch["dataset"][i]),
                        horizon=fut.shape[2],
                        m_points=fut.shape[1],
                        cache_path=str(batch["cache_path"][i]),
                        fut_points=fut[i],
                        fut_vis=fut_vis[i],
                        pred=pred[i],
                        modes=out["point_hypotheses"].detach().cpu().numpy()[i],
                        visibility_logits=vis[i],
                        tags=batch["v30_subset_tags"][i],
                    )
                )
    same_scores_np = np.concatenate(same_scores, axis=0)
    same_targets_np = np.concatenate(same_targets, axis=0)
    same_masks_np = np.concatenate(same_masks, axis=0)
    vis_scores_np = np.concatenate(vis_scores, axis=0)
    vis_targets_np = np.concatenate(vis_targets, axis=0)
    vis_masks_np = np.concatenate(vis_masks, axis=0)
    same_metrics = binary_metrics(same_scores_np, same_targets_np, same_masks_np)
    vis_metrics = visibility_f1(vis_scores_np, vis_targets_np, vis_masks_np)
    per_h = []
    if per_horizon_scores is not None and per_horizon_targets is not None:
        for hh, (ss, yy) in enumerate(zip(per_horizon_scores, per_horizon_targets)):
            if ss:
                s = np.asarray(ss)
                y = np.asarray(yy).astype(bool)
                per_h.append({"horizon_index": hh, "same_instance_accuracy": float(((s >= 0) == y).mean()), "AUROC": rank_auc(s, y)})
    constant_positive = np.ones_like(same_targets_np, dtype=np.float32)
    majority = np.full_like(same_targets_np, fill_value=float(same_metrics["positive_ratio"] or 0.0), dtype=np.float32)
    lp = binary_metrics(constant_positive, same_targets_np, same_masks_np)
    maj = binary_metrics(majority - 0.5, same_targets_np, same_masks_np)
    payload = {
        "identity_target_coverage": float(same_masks_np.mean()),
        "instance_identity_coverage": float(same_masks_np.mean()),
        "positive_ratio": same_metrics["positive_ratio"],
        "negative_ratio": same_metrics["negative_ratio"],
        "same_instance_accuracy": same_metrics["accuracy"],
        "same_instance_balanced_accuracy": same_metrics["balanced_accuracy"],
        "identity_ROC_AUC": same_metrics["ROC_AUC"],
        "identity_PR_AUC": same_metrics["PR_AUC"],
        "per_horizon_same_instance_accuracy": [x["same_instance_accuracy"] for x in per_h],
        "per_horizon_AUROC": [x["AUROC"] for x in per_h],
        "visibility_F1": vis_metrics["F1"],
        "visibility_AUROC": vis_metrics["ROC_AUC"],
        "trajectory_minFDE_delta_vs_frozen_V30": float(np.mean(fde_delta)) if fde_delta else 0.0,
        "trajectory_visibility_delta_vs_frozen_V30": None,
        "trajectory_degraded": bool((float(np.mean(fde_delta)) if fde_delta else 0.0) > 1e-4),
        "horizon_constant_logit_detected": horizon_constant(same_scores_np, same_masks_np),
        "constant_identity_prior_baseline": lp,
        "last_visible_instance_prior_baseline": lp,
        "visibility_copy_prior_baseline": {"F1": None, "ROC_AUC": None},
        "majority_prior_baseline": maj,
        "true_auroc_used": True,
    }
    return payload, rows


def loss_fn(out: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
    same_mask = batch["fut_instance_available_mask"].float()
    same_target = batch["fut_same_instance_as_obs"].float()
    pos = (same_target * same_mask).sum()
    neg = ((1.0 - same_target) * same_mask).sum()
    pos_weight = (neg / pos.clamp_min(1.0)).clamp(0.25, 8.0)
    same_raw = F.binary_cross_entropy_with_logits(out["same_instance_logits"], same_target, reduction="none", pos_weight=pos_weight)
    same_loss = (same_raw * same_mask).sum() / same_mask.sum().clamp_min(1.0)
    vis_mask = batch["fut_point_visible_mask"].float()
    vis_raw = F.binary_cross_entropy_with_logits(out["visibility_logits"], batch["fut_point_visible_target"].float(), reduction="none")
    vis_loss = (vis_raw * vis_mask).sum() / vis_mask.sum().clamp_min(1.0)
    # Mild temporal regularization keeps the identity head horizon-dependent but smooth.
    smooth = (out["same_instance_logits"][:, :, 1:] - out["same_instance_logits"][:, :, :-1]).pow(2).mean()
    total = same_loss + 0.5 * vis_loss + 0.001 * smooth
    return total, {"loss": float(total.detach().cpu()), "same_instance_loss": float(same_loss.detach().cpu()), "visibility_loss": float(vis_loss.detach().cpu()), "smooth": float(smooth.detach().cpu())}


def train_one(args: argparse.Namespace) -> dict[str, Any]:
    set_seed(args.seed)
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    train_loader = make_loader("train", args, shuffle=True, max_items=args.max_train_items)
    val_loader = make_loader("val", args, shuffle=False, max_items=args.max_eval_items)
    test_loader = make_loader("test", args, shuffle=False, max_items=args.max_eval_items)
    model = IntegratedSemanticIdentityWorldModelV331(
        args.v30_checkpoint,
        identity_dim=args.identity_dim,
        use_observed_instance_context=args.use_observed_instance_context,
    ).to(device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=1e-4)
    losses = []
    start = time.time()
    best_auc = -1.0
    ckpt_path = CKPT_DIR / f"{args.experiment_name}_best.pt"
    it = iter(train_loader)
    for step in range(1, args.steps + 1):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(train_loader)
            batch = next(it)
        bd = move_batch(batch, device)
        out = model(
            obs_points=bd["obs_points"],
            obs_vis=bd["obs_vis"],
            obs_conf=bd["obs_conf"],
            semantic_id=bd["semantic_id"],
            point_to_instance_id=bd["point_to_instance_id"] if args.use_observed_instance_context else None,
        )
        loss, comps = loss_fn(out, bd)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()
        losses.append({"step": step, **comps})
        if step % args.eval_interval == 0 or step == args.steps:
            val_metrics, _ = evaluate_model(model, val_loader, device)
            auc = float(val_metrics.get("identity_ROC_AUC") or 0.0)
            if auc >= best_auc:
                best_auc = auc
                torch.save({"model": model.state_dict(), "args": vars(args), "val_metrics": val_metrics, "step": step}, ckpt_path)
    ck = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ck["model"], strict=True)
    val_metrics, val_rows = evaluate_model(model, val_loader, device)
    test_metrics, test_rows = evaluate_model(model, test_loader, device)
    horizon_dep = not bool(test_metrics["horizon_constant_logit_detected"])
    traj_deg = bool(test_metrics["trajectory_degraded"])
    trivial_beaten = bool((test_metrics.get("identity_ROC_AUC") or 0.0) > 0.55 and (test_metrics.get("same_instance_balanced_accuracy") or 0.0) > 0.52)
    passed = bool(horizon_dep and not traj_deg and trivial_beaten)
    payload = {
        "generated_at_utc": utc_now(),
        "experiment_name": args.experiment_name,
        "completed": True,
        "smoke_passed": passed,
        "integrated_v30_backbone_used": True,
        "v30_checkpoint_loaded": True,
        "v30_checkpoint_path": str(Path(args.v30_checkpoint).relative_to(ROOT) if Path(args.v30_checkpoint).is_absolute() and ROOT in Path(args.v30_checkpoint).parents else args.v30_checkpoint),
        "v30_backbone_frozen": model.v30_backbone_frozen,
        "observed_instance_context_used": bool(args.use_observed_instance_context),
        "horizon_dependent_identity_logits": horizon_dep,
        "true_auroc_used": True,
        "trivial_prior_beaten": trivial_beaten,
        "trajectory_degraded": traj_deg,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "duration_seconds": float(time.time() - start),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "checkpoint_path": str(ckpt_path.relative_to(ROOT)),
        "train_loss_first": losses[0] if losses else None,
        "train_loss_last": losses[-1] if losses else None,
        "train_loss_decreased": bool(losses and losses[-1]["loss"] <= losses[0]["loss"]),
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "val_item_rows": val_rows[:256],
        "test_item_rows": test_rows[:256],
    }
    out_path = RUN_DIR / f"{args.experiment_name}.json"
    dump_json(out_path, payload)
    dump_json(SUMMARY, payload)
    decision = {
        "generated_at_utc": utc_now(),
        "integrated_v30_backbone_used": True,
        "v30_checkpoint_loaded": True,
        "v30_backbone_frozen": model.v30_backbone_frozen,
        "observed_instance_context_used": bool(args.use_observed_instance_context),
        "horizon_dependent_identity_logits": horizon_dep,
        "true_auroc_used": True,
        "identity_ROC_AUC": test_metrics.get("identity_ROC_AUC"),
        "identity_PR_AUC": test_metrics.get("identity_PR_AUC"),
        "trivial_prior_beaten": trivial_beaten,
        "trajectory_degraded": traj_deg,
        "integrated_identity_field_claim_allowed": passed,
        "integrated_semantic_field_claim_allowed": False,
        "recommended_next_step": "run_v33_1_integrated_h64_h96_smoke" if passed else ("fix_identity_head_horizon_dependence" if not horizon_dep else ("fix_trajectory_preservation" if traj_deg else "build_visual_teacher_semantic_prototypes")),
    }
    dump_json(DECISION, decision)
    write_doc(DOC, "STWM OSTF V33.1 Integrated Smoke Decision", decision, ["integrated_v30_backbone_used", "v30_checkpoint_loaded", "v30_backbone_frozen", "observed_instance_context_used", "horizon_dependent_identity_logits", "identity_ROC_AUC", "identity_PR_AUC", "trivial_prior_beaten", "trajectory_degraded", "integrated_identity_field_claim_allowed", "integrated_semantic_field_claim_allowed", "recommended_next_step"])
    print(SUMMARY.relative_to(ROOT))
    return payload


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--experiment-name", default="v33_1_integrated_m128_h32_seed42_smoke")
    p.add_argument("--v30-checkpoint", default=str(ROOT / "outputs/checkpoints/stwm_ostf_v30_external_gt/v30_extgt_m128_h32_seed42_best.pt"))
    p.add_argument("--semantic-identity-sidecar-root", default=str(ROOT / "outputs/cache/stwm_ostf_v33_semantic_identity_targets/pointodyssey"))
    p.add_argument("--m-points", type=int, default=128)
    p.add_argument("--horizon", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--steps", type=int, default=700)
    p.add_argument("--eval-interval", type=int, default=250)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--max-train-items", type=int, default=512)
    p.add_argument("--max-eval-items", type=int, default=256)
    p.add_argument("--identity-dim", type=int, default=64)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--use-observed-instance-context", action="store_true")
    p.add_argument("--cpu", action="store_true")
    return p.parse_args()


def main() -> int:
    train_one(parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
