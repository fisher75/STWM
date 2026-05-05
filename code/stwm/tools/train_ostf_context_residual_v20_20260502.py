#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import GradScaler, autocast

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))

from stwm.modules.ostf_context_residual_world_model_v20 import OSTFContextResidualConfig, OSTFContextResidualWorldModel
from stwm.tools.ostf_v17_common_20260502 import ROOT, batch_from_samples, dump_json, set_seed
from stwm.tools.ostf_v18_common_20260502 import metrics_by_dataset
from stwm.tools.ostf_v20_common_20260502 import (
    build_context_cache_for_combo,
    evaluate_subset_metrics,
    hard_subset_flags,
    key_tuple,
    load_combo_rows,
    load_context_cache,
    sample_key,
)


def _apply_process_title() -> None:
    try:
        import setproctitle  # type: ignore

        setproctitle.setproctitle(str(os.environ.get("STWM_PROC_TITLE", "python")))
    except Exception:
        pass


def _combo_for_model(kind: str, horizon: int) -> str:
    if "m128" in kind:
        return f"M128_H{horizon}"
    return f"M512_H{horizon}"


def _build_model(kind: str, horizon: int) -> OSTFContextResidualWorldModel:
    cfg = OSTFContextResidualConfig(
        horizon=horizon,
        hidden_dim=384,
        point_dim=192,
        num_layers=4,
        num_heads=8,
        refinement_layers=2,
        num_hypotheses=1 if "single_hypothesis" in kind else 3,
        use_context="wo_context" not in kind,
        use_dense_points="wo_dense_points" not in kind,
        use_multi_hypothesis="single_hypothesis" not in kind,
        use_semantic_memory=True,
        use_affine_prior=True,
        use_cv_prior=True,
    )
    return OSTFContextResidualWorldModel(cfg)


def _sample_weights(records: list[dict[str, Any]]) -> list[float]:
    scores = np.asarray([r["hardness_score"] for r in records], dtype=np.float32)
    mu = float(scores.mean()) if scores.size else 0.0
    std = float(scores.std() + 1e-6) if scores.size else 1.0
    out = []
    for r in records:
        z = (float(r["hardness_score"]) - mu) / std
        w = 1.0 + max(0.0, 0.50 * z) + 0.40 * float(r["occlusion_ratio"]) + 0.25 * float(r["reappearance_flag"])
        out.append(float(min(3.0, max(1.0, w))))
    return out


def _build_eval_flags(samples: list[Any], ctx_map: dict[tuple[str, int], dict[str, Any]]) -> dict[str, np.ndarray]:
    records = []
    for s in samples:
        c = ctx_map[sample_key(s)]
        records.append(
            {
                "cv_point_l1_proxy": c["cv_point_l1_proxy"],
                "curvature_proxy": c["curvature_proxy"],
                "occlusion_ratio": c["occlusion_ratio"],
                "interaction_proxy": c["interaction_proxy"],
            }
        )
    return hard_subset_flags(records)


def _context_batch(batch_rows: list[Any], ctx_map: dict[tuple[str, int], dict[str, Any]], device: torch.device) -> dict[str, torch.Tensor]:
    crop = []
    box = []
    neigh = []
    glob = []
    hard = []
    occ = []
    for s in batch_rows:
        c = ctx_map[sample_key(s)]
        crop.append(c["crop_feat"])
        box.append(c["box_feat"])
        neigh.append(c["neighbor_feat"])
        glob.append(c["global_feat"])
        hard.append(c["hardness_score"])
        occ.append(c["occlusion_ratio"])
    return {
        "crop_feat": torch.tensor(np.stack(crop), device=device, dtype=torch.float32),
        "box_feat": torch.tensor(np.stack(box), device=device, dtype=torch.float32),
        "neighbor_feat": torch.tensor(np.stack(neigh), device=device, dtype=torch.float32),
        "global_feat": torch.tensor(np.stack(glob), device=device, dtype=torch.float32),
        "hardness_score": torch.tensor(hard, device=device, dtype=torch.float32),
        "occlusion_ratio": torch.tensor(occ, device=device, dtype=torch.float32),
    }


def _min_hypothesis_selection(point_hyp: torch.Tensor, gt_points: torch.Tensor, valid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # point_hyp: [B,M,H,K,2], gt_points: [B,M,H,2], valid: [B,M,H]
    diff = F.smooth_l1_loss(point_hyp, gt_points[:, :, :, None, :].expand_as(point_hyp), reduction="none").sum(dim=-1)
    mask = valid[:, :, :, None].float()
    denom = mask.sum(dim=(1, 2)).clamp_min(1.0)  # [B,K] after broadcast sum
    loss_per_h = (diff * mask).sum(dim=(1, 2)) / denom
    best_idx = loss_per_h.argmin(dim=-1)
    gather_idx = best_idx[:, None, None, None, None].expand(-1, point_hyp.shape[1], point_hyp.shape[2], 1, point_hyp.shape[4])
    selected = point_hyp.gather(3, gather_idx).squeeze(3)
    return selected, best_idx


def _weighted_mean_per_sample(loss_tensor: torch.Tensor, sample_weight: torch.Tensor) -> torch.Tensor:
    return (loss_tensor * sample_weight).sum() / sample_weight.sum().clamp_min(1e-6)


def _losses(
    out: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    batch_ctx: dict[str, torch.Tensor],
    *,
    use_hard_weighting: bool,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    fut_points = batch["fut_points"]
    fut_vis = batch["fut_vis"]
    point_hyp = out["point_hypotheses"]
    selected, best_idx = _min_hypothesis_selection(point_hyp, fut_points, fut_vis)

    sample_weight = torch.ones(fut_points.shape[0], device=fut_points.device, dtype=torch.float32)
    if use_hard_weighting:
        hard = batch_ctx["hardness_score"]
        occ = batch_ctx["occlusion_ratio"]
        hard_z = (hard - hard.mean()) / (hard.std() + 1e-6)
        sample_weight = 1.0 + torch.clamp(0.60 * hard_z, min=0.0) + 0.40 * occ
        sample_weight = sample_weight.clamp(1.0, 3.0)

    diff = F.smooth_l1_loss(selected, fut_points, reduction="none").sum(dim=-1)
    mask = fut_vis.float()
    point_loss_per_sample = (diff * mask).sum(dim=(1, 2)) / mask.sum(dim=(1, 2)).clamp_min(1.0)

    end_diff = F.smooth_l1_loss(selected[:, :, -1], fut_points[:, :, -1], reduction="none").sum(dim=-1)
    end_mask = fut_vis[:, :, -1].float()
    endpoint_per_sample = (end_diff * end_mask).sum(dim=1) / end_mask.sum(dim=1).clamp_min(1.0)

    pred_anchor = out["anchor_pred"]
    anchor_diff = F.smooth_l1_loss(pred_anchor, batch["anchor_fut"], reduction="none").sum(dim=-1)
    anchor_per_sample = anchor_diff.mean(dim=1)

    pred_vis = out["visibility_logits"]
    vis_bce = F.binary_cross_entropy_with_logits(pred_vis, fut_vis.float(), reduction="none")
    vis_per_sample = vis_bce.mean(dim=(1, 2))

    # extent / covariance loss
    def _extent_center(points: torch.Tensor, valid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        big = torch.full_like(points[..., 0], 1e6)
        x = points[..., 0]
        y = points[..., 1]
        x_min = torch.where(valid, x, big).amin(dim=1)
        y_min = torch.where(valid, y, big).amin(dim=1)
        x_max = torch.where(valid, x, -big).amax(dim=1)
        y_max = torch.where(valid, y, -big).amax(dim=1)
        extent = torch.stack([x_max - x_min, y_max - y_min], dim=-1)
        center = torch.stack([(x_max + x_min) * 0.5, (y_max + y_min) * 0.5], dim=-1)
        return torch.nan_to_num(extent), torch.nan_to_num(center)

    pred_extent, pred_center = _extent_center(selected, fut_vis)
    gt_extent, gt_center = _extent_center(fut_points, fut_vis)
    frame_mask = fut_vis.any(dim=1).float()
    extent_l = F.smooth_l1_loss(pred_extent, gt_extent, reduction="none").sum(dim=-1)
    center_l = F.smooth_l1_loss(pred_center, gt_center, reduction="none").sum(dim=-1)
    extent_per_sample = (extent_l * frame_mask).sum(dim=1) / frame_mask.sum(dim=1).clamp_min(1.0) + 0.5 * (
        (center_l * frame_mask).sum(dim=1) / frame_mask.sum(dim=1).clamp_min(1.0)
    )

    # semantic
    proto_logits = out["semantic_logits"].reshape(-1, out["semantic_logits"].shape[-1])
    proto_target = batch["proto_target"][:, None].expand(-1, out["semantic_logits"].shape[1]).reshape(-1)
    sem_ce = F.cross_entropy(proto_logits, proto_target, reduction="none").reshape(batch["proto_target"].shape[0], -1).mean(dim=1)

    # hypothesis classifier
    hyp_logits = out["hypothesis_logits"]
    if hyp_logits.shape[-1] > 1:
        hyp_loss = F.cross_entropy(hyp_logits, best_idx, reduction="none")
    else:
        hyp_loss = hyp_logits.new_zeros(hyp_logits.shape[0])

    # regularizers
    accel = selected[:, :, 2:] - 2.0 * selected[:, :, 1:-1] + selected[:, :, :-2] if selected.shape[2] >= 3 else selected.new_zeros(())
    smooth_per_sample = accel.abs().mean(dim=(1, 2, 3)) if selected.shape[2] >= 3 else selected.new_zeros(selected.shape[0])
    residual_mag = out["delta"].abs().mean(dim=(1, 2, 3, 4))

    total = (
        _weighted_mean_per_sample(point_loss_per_sample, sample_weight)
        + 0.75 * _weighted_mean_per_sample(endpoint_per_sample, sample_weight)
        + 0.40 * _weighted_mean_per_sample(anchor_per_sample, sample_weight)
        + 0.30 * _weighted_mean_per_sample(extent_per_sample, sample_weight)
        + 0.20 * _weighted_mean_per_sample(vis_per_sample, sample_weight)
        + 0.05 * _weighted_mean_per_sample(sem_ce, sample_weight)
        + 0.05 * _weighted_mean_per_sample(hyp_loss, sample_weight)
        + 0.03 * _weighted_mean_per_sample(smooth_per_sample, sample_weight)
        + 0.005 * _weighted_mean_per_sample(residual_mag, sample_weight)
    )
    comps = {
        "point": _weighted_mean_per_sample(point_loss_per_sample, sample_weight),
        "endpoint": _weighted_mean_per_sample(endpoint_per_sample, sample_weight),
        "anchor": _weighted_mean_per_sample(anchor_per_sample, sample_weight),
        "extent": _weighted_mean_per_sample(extent_per_sample, sample_weight),
        "visibility": _weighted_mean_per_sample(vis_per_sample, sample_weight),
        "semantic": _weighted_mean_per_sample(sem_ce, sample_weight),
        "hypothesis_ce": _weighted_mean_per_sample(hyp_loss, sample_weight),
        "smooth": _weighted_mean_per_sample(smooth_per_sample, sample_weight),
        "residual_reg": _weighted_mean_per_sample(residual_mag, sample_weight),
        "hard_weight_mean": sample_weight.mean(),
    }
    return total, comps


def _evaluate_model(
    model: OSTFContextResidualWorldModel,
    samples: list[Any],
    ctx_map: dict[tuple[str, int], dict[str, Any]],
    batch_size: int,
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    model.eval()
    point_preds = []
    vis_logits = []
    sem_logits = []
    item_scores = []
    with torch.no_grad():
        for start in range(0, len(samples), batch_size):
            batch_rows = samples[start : start + batch_size]
            batch = batch_from_samples(batch_rows, device)
            batch_ctx = _context_batch(batch_rows, ctx_map, device)
            out = model(
                obs_points=batch["obs_points"],
                obs_vis=batch["obs_vis"],
                rel_xy=batch["rel_xy"],
                anchor_obs=batch["anchor_obs"],
                anchor_obs_vel=batch["anchor_obs_vel"],
                semantic_feat=batch["semantic_feat"],
                crop_feat=batch_ctx["crop_feat"],
                box_feat=batch_ctx["box_feat"],
                neighbor_feat=batch_ctx["neighbor_feat"],
                global_feat=batch_ctx["global_feat"],
            )
            point_preds.append(out["point_pred"].detach().cpu().numpy())
            vis_logits.append(out["visibility_logits"].detach().cpu().numpy())
            sem_logits.append(out["semantic_logits"].detach().cpu().numpy())
    pred_points = np.concatenate(point_preds, axis=0)
    pred_vis_logits = np.concatenate(vis_logits, axis=0)
    pred_sem_logits = np.concatenate(sem_logits, axis=0)
    all_metrics = evaluate_subset_metrics(samples, pred_points, pred_vis_logits, pred_sem_logits)
    by_dataset = metrics_by_dataset(samples, pred_points, pred_vis_logits, pred_sem_logits)
    flags = _build_eval_flags(samples, ctx_map)
    subsets = {
        "cv_hard_top20": evaluate_subset_metrics(samples, pred_points, pred_vis_logits, pred_sem_logits, flags["top20_cv_hard"]),
        "nonlinear": evaluate_subset_metrics(samples, pred_points, pred_vis_logits, pred_sem_logits, flags["nonlinear_hard"]),
        "occlusion": evaluate_subset_metrics(samples, pred_points, pred_vis_logits, pred_sem_logits, flags["occlusion_hard"]),
        "interaction": evaluate_subset_metrics(samples, pred_points, pred_vis_logits, pred_sem_logits, flags["interaction_hard"]),
    }
    # item scores
    pred_vis = pred_vis_logits > 0.0
    for i, s in enumerate(samples):
        err = np.abs(pred_points[i] - s.fut_points).sum(axis=-1) * 1000.0
        valid = s.fut_vis
        px = float(err[valid].mean()) if np.any(valid) else 0.0
        endpoint = float(err[:, -1][s.fut_vis[:, -1]].mean()) if np.any(s.fut_vis[:, -1]) else px
        vis_f1 = None
        pv = pred_vis[i]
        tp = int(np.logical_and(pv, s.fut_vis).sum())
        fp = int(np.logical_and(pv, np.logical_not(s.fut_vis)).sum())
        fn = int(np.logical_and(np.logical_not(pv), s.fut_vis).sum())
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        vis_f1 = float(2 * prec * rec / max(prec + rec, 1e-8))
        top5 = np.argsort(pred_sem_logits[i], axis=-1)[..., -5:]
        sem_top5 = float((top5 == s.proto_target).any(axis=-1).mean())
        row = {
            "item_key": s.item_key,
            "dataset": s.dataset,
            "object_index": s.object_index,
            "object_id": s.object_id,
            "point_l1_px": px,
            "endpoint_error_px": endpoint,
            "visibility_f1": vis_f1,
            "extent_iou": all_metrics["object_extent_iou"] if len(samples) == 1 else None,
            "semantic_top5": sem_top5,
            "cv_hard20": bool(flags["top20_cv_hard"][i]),
            "nonlinear_hard": bool(flags["nonlinear_hard"][i]),
            "occlusion_hard": bool(flags["occlusion_hard"][i]),
            "interaction_hard": bool(flags["interaction_hard"][i]),
        }
        # item extent IoU
        vals = []
        for t in range(s.h):
            mask = valid[:, t]
            if not np.any(mask):
                continue
            pred = pred_points[i, mask, t]
            gt = s.fut_points[mask, t]
            px0, py0 = pred.min(axis=0)
            px1, py1 = pred.max(axis=0)
            gx0, gy0 = gt.min(axis=0)
            gx1, gy1 = gt.max(axis=0)
            ix0, iy0 = max(px0, gx0), max(py0, gy0)
            ix1, iy1 = min(px1, gx1), min(py1, gy1)
            inter = max(ix1 - ix0, 0.0) * max(iy1 - iy0, 0.0)
            pa = max(px1 - px0, 0.0) * max(py1 - py0, 0.0)
            ga = max(gx1 - gx0, 0.0) * max(gy1 - gy0, 0.0)
            union = pa + ga - inter
            vals.append(float(inter / union) if union > 0 else 0.0)
        row["extent_iou"] = float(np.mean(vals)) if vals else 0.0
        item_scores.append(row)
    return all_metrics, by_dataset, subsets, item_scores


def _val_score(all_metrics: dict[str, Any], hard_metrics: dict[str, Any]) -> float:
    vis_h = 0.0 if hard_metrics.get("visibility_F1") is None else float(hard_metrics["visibility_F1"])
    vis_all = 0.0 if all_metrics.get("visibility_F1") is None else float(all_metrics["visibility_F1"])
    return (
        -1.2 * float(hard_metrics["point_L1_px"])
        - 0.8 * float(hard_metrics["endpoint_error_px"])
        + 18.0 * float(hard_metrics["PCK_16px"])
        + 10.0 * float(hard_metrics["PCK_32px"])
        + 140.0 * float(hard_metrics["object_extent_iou"])
        + 4.0 * vis_h
        - 0.4 * float(all_metrics["point_L1_px"])
        - 0.2 * float(all_metrics["endpoint_error_px"])
        + 70.0 * float(all_metrics["object_extent_iou"])
        + 6.0 * float(all_metrics["PCK_16px"])
        + 2.0 * vis_all
    )


def _load_init_checkpoint(model: torch.nn.Module, path: str) -> list[str]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    src = ckpt.get("model_state_dict", ckpt)
    dst = model.state_dict()
    loaded = []
    for key, value in src.items():
        if key in dst and dst[key].shape == value.shape:
            dst[key] = value
            loaded.append(key)
    model.load_state_dict(dst)
    return loaded


def main() -> int:
    _apply_process_title()
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument(
        "--model-kind",
        required=True,
        choices=[
            "v20_context_residual_m128",
            "v20_context_residual_m512",
            "v20_context_residual_m512_wo_context",
            "v20_context_residual_m512_wo_dense_points",
            "v20_context_residual_m512_wo_hard_weighting",
            "v20_context_residual_m512_single_hypothesis",
        ],
    )
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=12000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--init-from-checkpoint", default=None)
    parser.add_argument("--context-cache-path", default=None)
    args = parser.parse_args()

    started = datetime.now(timezone.utc)
    wall_start = time.time()
    set_seed(args.seed)
    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    combo = _combo_for_model(args.model_kind, args.horizon)
    rows, proto_centers = load_combo_rows(combo, seed=args.seed)

    context_cache_path = Path(args.context_cache_path) if args.context_cache_path else ROOT / "outputs/cache/stwm_ostf_context_features_v20" / f"{combo}_context_features.npz"
    if not context_cache_path.exists():
        bundle = build_context_cache_for_combo(combo, seed=args.seed)
        context_cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(context_cache_path, **{})
        # overwrite with proper writer from common
        from stwm.tools.ostf_v20_common_20260502 import save_context_cache

        save_context_cache(bundle, context_cache_path)
    ctx_map = load_context_cache(context_cache_path)
    train_records = [ctx_map[sample_key(s)] for s in rows["train"]]
    train_weights = _sample_weights(train_records)
    train_pairs = list(zip(rows["train"], train_weights))

    model = _build_model(args.model_kind, args.horizon).to(device)
    param_count = int(sum(p.numel() for p in model.parameters()))
    opt = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-4)
    scaler = GradScaler(device="cuda", enabled=device.type == "cuda")
    loaded_keys = []
    if args.init_from_checkpoint:
        loaded_keys = _load_init_checkpoint(model, args.init_from_checkpoint)

    print(
        f"[V20][start] exp={args.experiment_name} kind={args.model_kind} combo={combo} h={args.horizon} seed={args.seed} steps={args.steps} batch={args.batch_size} device={device}",
        flush=True,
    )
    print(f"[V20][data] train={len(rows['train'])} val={len(rows['val'])} test={len(rows['test'])}", flush=True)
    print(f"[V20][model] params={param_count} init_loaded={len(loaded_keys)} context_cache={context_cache_path}", flush=True)

    use_hard_weighting = "wo_hard_weighting" not in args.model_kind
    best_state = None
    best_val_score = -1e18
    best_step = 0
    loss_history = []
    ckpt_dir = ROOT / "outputs/checkpoints/stwm_ostf_v20"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    final_rel = f"outputs/checkpoints/stwm_ostf_v20/{args.experiment_name}_final.pt"
    best_rel = f"outputs/checkpoints/stwm_ostf_v20/{args.experiment_name}_best.pt"
    for step in range(1, args.steps + 1):
        if use_hard_weighting:
            batch_rows = [s for s, _ in random.choices(train_pairs, weights=[w for _, w in train_pairs], k=min(args.batch_size, len(train_pairs)))]
        else:
            batch_rows = random.sample(rows["train"], k=min(args.batch_size, len(rows["train"])))
        batch = batch_from_samples(batch_rows, device)
        batch_ctx = _context_batch(batch_rows, ctx_map, device)
        model.train()
        with autocast(device_type="cuda", enabled=device.type == "cuda"):
            out = model(
                obs_points=batch["obs_points"],
                obs_vis=batch["obs_vis"],
                rel_xy=batch["rel_xy"],
                anchor_obs=batch["anchor_obs"],
                anchor_obs_vel=batch["anchor_obs_vel"],
                semantic_feat=batch["semantic_feat"],
                crop_feat=batch_ctx["crop_feat"],
                box_feat=batch_ctx["box_feat"],
                neighbor_feat=batch_ctx["neighbor_feat"],
                global_feat=batch_ctx["global_feat"],
            )
            total, comps = _losses(out, batch, batch_ctx, use_hard_weighting=use_hard_weighting)
        opt.zero_grad(set_to_none=True)
        scaler.scale(total).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        if step == 1 or step % 500 == 0 or step == args.steps:
            row = {"step": step, "total": float(total.detach().cpu())}
            row.update({k: float(v.detach().cpu()) for k, v in comps.items()})
            row["residual_scale"] = float(out["residual_scale"].mean().detach().cpu())
            loss_history.append(row)
            print("[V20][train] " + " ".join(f"{k}={v:.6f}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()), flush=True)
        if step % args.eval_every == 0 or step == args.steps:
            val_all, _, val_sub, _ = _evaluate_model(model, rows["val"], ctx_map, args.batch_size, device)
            score = _val_score(val_all, val_sub["cv_hard_top20"])
            print(
                f"[V20][val] step={step} all_point={val_all['point_L1_px']:.4f} hard_point={val_sub['cv_hard_top20']['point_L1_px']:.4f} "
                f"hard_endpoint={val_sub['cv_hard_top20']['endpoint_error_px']:.4f} hard_pck16={val_sub['cv_hard_top20']['PCK_16px']:.4f} hard_extent={val_sub['cv_hard_top20']['object_extent_iou']:.4f} score={score:.4f}",
                flush=True,
            )
            if score > best_val_score:
                best_val_score = score
                best_step = step
                best_state = {
                    "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                    "config": vars(args),
                    "proto_centers": proto_centers,
                    "context_cache_path": str(context_cache_path),
                }
                torch.save(best_state, ROOT / best_rel)

    assert best_state is not None
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": vars(args),
            "proto_centers": proto_centers,
            "context_cache_path": str(context_cache_path),
        },
        ROOT / final_rel,
    )
    torch.save(best_state, ROOT / best_rel)
    model.load_state_dict(best_state["model_state_dict"])

    train_all, train_ds, train_sub, _ = _evaluate_model(model, rows["train"], ctx_map, args.batch_size, device)
    val_all, val_ds, val_sub, _ = _evaluate_model(model, rows["val"], ctx_map, args.batch_size, device)
    test_all, test_ds, test_sub, item_scores = _evaluate_model(model, rows["test"], ctx_map, args.batch_size, device)
    loss_history.append({"best_step": best_step, "best_val_score": best_val_score})
    report = {
        "audit_name": "stwm_ostf_v20_run",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "started_at_utc": started.isoformat(),
        "duration_sec": float(time.time() - wall_start),
        "experiment_name": args.experiment_name,
        "model_kind": args.model_kind,
        "horizon": args.horizon,
        "seed": args.seed,
        "source_combo": combo,
        "steps": args.steps,
        "parameter_count": param_count,
        "checkpoint_path": final_rel,
        "best_checkpoint_path": best_rel,
        "best_val_score": float(best_val_score),
        "context_cache_path": str(context_cache_path.relative_to(ROOT)),
        "loaded_init_key_count": len(loaded_keys),
        "use_hard_weighting": use_hard_weighting,
        "loss_history": loss_history,
        "train_metrics": train_all,
        "val_metrics": val_all,
        "test_metrics": test_all,
        "train_metrics_by_dataset": train_ds,
        "val_metrics_by_dataset": val_ds,
        "test_metrics_by_dataset": test_ds,
        "train_subset_metrics": train_sub,
        "val_subset_metrics": val_sub,
        "test_subset_metrics": test_sub,
        "item_scores": item_scores,
        "metric_note": "V20 uses observed-only context features with CV/affine prior and hard-subset-aware residual training. Semantic metrics remain secondary because OSTF target semantics are mostly static.",
    }
    out_path = ROOT / f"reports/stwm_ostf_v20_runs/{args.experiment_name}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dump_json(out_path, report)
    print(out_path.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
