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

from stwm.modules.ostf_refinement_world_model_v19 import OSTFRefinementConfig, OSTFRefinementWorldModel
from stwm.tools.ostf_v18_common_20260502 import (
    ROOT,
    batch_from_samples,
    build_v18_rows,
    dump_json,
    eval_metrics_extended,
    item_scores_from_predictions,
    metrics_by_dataset,
    semantic_logits_from_observed_memory,
    set_seed,
)


def _apply_process_title() -> None:
    try:
        import setproctitle  # type: ignore

        setproctitle.setproctitle(str(os.environ.get("STWM_PROC_TITLE", "python")))
    except Exception:
        pass


def _combo_for_experiment(kind: str, horizon: int) -> str:
    if "m128" in kind.lower():
        return f"M128_H{horizon}"
    return f"M512_H{horizon}"


def _iter_batches(samples: list[Any], batch_size: int, *, shuffle: bool, seed: int):
    idx = list(range(len(samples)))
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(idx)
    for start in range(0, len(idx), batch_size):
        yield [samples[i] for i in idx[start : start + batch_size]]


def _build_model(kind: str, horizon: int) -> OSTFRefinementWorldModel:
    cfg = OSTFRefinementConfig(
        horizon=horizon,
        hidden_dim=384,
        point_dim=192,
        num_layers=4,
        num_heads=8,
        refinement_layers=2,
        use_semantic_memory="wo_semantic_memory" not in kind,
        use_dense_points="wo_dense_points" not in kind,
        use_refinement_transformer="wo_refinement_transformer" not in kind,
        use_learnable_residual_scale="wo_learnable_residual_scale" not in kind,
        use_affine_prior=True,
        use_cv_prior=True,
    )
    return OSTFRefinementWorldModel(cfg)


def _weighted_point_loss(
    pred_points: torch.Tensor,
    gt_points: torch.Tensor,
    valid: torch.Tensor,
    rel_xy: torch.Tensor,
) -> torch.Tensor:
    if not torch.any(valid):
        return pred_points.new_tensor(0.0)
    diff = F.smooth_l1_loss(pred_points, gt_points, reduction="none").sum(dim=-1)
    motion = torch.norm(gt_points[:, :, -1] - gt_points[:, :, 0], dim=-1, keepdim=True)
    motion = motion / motion.mean().clamp_min(1e-6)
    boundary = ((rel_xy - 0.5).abs().amax(dim=-1, keepdim=True) > 0.30).float()
    weights = 1.0 + 0.75 * motion + 0.35 * boundary
    w = weights.expand_as(diff)
    diff = diff * w
    return diff[valid].mean()


def _endpoint_loss(pred_points: torch.Tensor, gt_points: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    last_valid = valid[:, :, -1]
    if not torch.any(last_valid):
        return pred_points.new_tensor(0.0)
    return F.smooth_l1_loss(pred_points[:, :, -1][last_valid], gt_points[:, :, -1][last_valid])


def _centroid_loss(pred_points: torch.Tensor, gt_anchor: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    weight = valid.float()
    denom = weight.sum(dim=1).clamp_min(1.0)
    centroid = (pred_points * weight[..., None]).sum(dim=1) / denom[..., None]
    return F.smooth_l1_loss(centroid, gt_anchor)


def _extent_center_from_points(points: torch.Tensor, valid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
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


def _extent_loss(pred_points: torch.Tensor, gt_points: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    pred_extent, pred_center = _extent_center_from_points(pred_points, valid)
    gt_extent, gt_center = _extent_center_from_points(gt_points, valid)
    frame_mask = valid.any(dim=1)
    if not torch.any(frame_mask):
        return pred_points.new_tensor(0.0)
    return F.smooth_l1_loss(pred_extent[frame_mask], gt_extent[frame_mask]) + 0.5 * F.smooth_l1_loss(
        pred_center[frame_mask],
        gt_center[frame_mask],
    )


def _temporal_smoothness(pred_points: torch.Tensor) -> torch.Tensor:
    if pred_points.shape[2] < 3:
        return pred_points.new_tensor(0.0)
    accel = pred_points[:, :, 2:] - 2.0 * pred_points[:, :, 1:-1] + pred_points[:, :, :-2]
    return accel.abs().mean()


def _val_score(metrics: dict[str, Any]) -> float:
    vis = 0.0 if metrics.get("visibility_F1") is None else float(metrics["visibility_F1"])
    return (
        -float(metrics["point_L1_px"])
        - 0.5 * float(metrics["endpoint_error_px"])
        + 12.0 * float(metrics["PCK_16px"])
        + 6.0 * float(metrics["PCK_32px"])
        + 120.0 * float(metrics["object_extent_iou"])
        + 4.0 * vis
    )


def _evaluate_model(
    model: OSTFRefinementWorldModel,
    samples: list[Any],
    batch_size: int,
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    model.eval()
    point_preds = []
    vis_logits = []
    sem_logits = []
    with torch.no_grad():
        for batch_rows in _iter_batches(samples, batch_size, shuffle=False, seed=0):
            batch = batch_from_samples(batch_rows, device)
            out = model(
                obs_points=batch["obs_points"],
                obs_vis=batch["obs_vis"],
                rel_xy=batch["rel_xy"],
                anchor_obs=batch["anchor_obs"],
                anchor_obs_vel=batch["anchor_obs_vel"],
                semantic_feat=batch["semantic_feat"],
            )
            point_preds.append(out["point_pred"].detach().cpu().numpy())
            vis_logits.append(out["visibility_logits"].detach().cpu().numpy())
            sem_logits.append(out["semantic_logits"].detach().cpu().numpy())
    pred_points = np.concatenate(point_preds, axis=0)
    pred_vis_logits = np.concatenate(vis_logits, axis=0)
    pred_sem_logits = np.concatenate(sem_logits, axis=0)
    metrics = eval_metrics_extended(
        pred_points=pred_points,
        pred_vis_logits=pred_vis_logits,
        pred_proto_logits=pred_sem_logits,
        gt_points=np.stack([s.fut_points for s in samples]),
        gt_vis=np.stack([s.fut_vis for s in samples]),
        gt_anchor=np.stack([s.anchor_fut for s in samples]),
        proto_target=np.asarray([s.proto_target for s in samples], dtype=np.int64),
    )
    by_dataset = metrics_by_dataset(samples, pred_points, pred_vis_logits, pred_sem_logits)
    item_scores = item_scores_from_predictions(samples, pred_points, pred_vis_logits, pred_sem_logits)
    return metrics, by_dataset, item_scores


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
            "v19_refinement_m128",
            "v19_refinement_m512",
            "v19_refinement_m512_wo_refinement_transformer",
            "v19_refinement_m512_wo_learnable_residual_scale",
            "v19_refinement_m512_wo_dense_points",
        ],
    )
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=12000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--init-from-checkpoint", default=None)
    args = parser.parse_args()

    started = datetime.now(timezone.utc)
    wall_start = time.time()
    set_seed(args.seed)
    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    combo = _combo_for_experiment(args.model_kind, args.horizon)
    rows, proto_centers = build_v18_rows(combo, seed=args.seed)
    model = _build_model(args.model_kind, args.horizon).to(device)
    param_count = int(sum(p.numel() for p in model.parameters()))
    opt = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-4)
    scaler = GradScaler(device="cuda", enabled=device.type == "cuda")
    loaded_keys = []
    if args.init_from_checkpoint:
        loaded_keys = _load_init_checkpoint(model, args.init_from_checkpoint)

    print(
        f"[V19][start] exp={args.experiment_name} kind={args.model_kind} combo={combo} h={args.horizon} seed={args.seed} steps={args.steps} batch={args.batch_size} device={device}",
        flush=True,
    )
    print(f"[V19][data] train={len(rows['train'])} val={len(rows['val'])} test={len(rows['test'])}", flush=True)
    print(f"[V19][model] params={param_count} init_loaded={len(loaded_keys)}", flush=True)

    best_state = None
    best_val_score = -1e18
    best_step = 0
    loss_history = []
    ckpt_dir = ROOT / "outputs/checkpoints/stwm_ostf_v19"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    for step in range(1, args.steps + 1):
        batch_rows = random.sample(rows["train"], k=min(args.batch_size, len(rows["train"])))
        batch = batch_from_samples(batch_rows, device)
        model.train()
        with autocast(device_type="cuda", enabled=device.type == "cuda"):
            out = model(
                obs_points=batch["obs_points"],
                obs_vis=batch["obs_vis"],
                rel_xy=batch["rel_xy"],
                anchor_obs=batch["anchor_obs"],
                anchor_obs_vel=batch["anchor_obs_vel"],
                semantic_feat=batch["semantic_feat"],
            )
            point_loss = _weighted_point_loss(out["point_pred"], batch["fut_points"], batch["fut_vis"], batch["rel_xy"])
            endpoint_loss = _endpoint_loss(out["point_pred"], batch["fut_points"], batch["fut_vis"])
            centroid_loss = _centroid_loss(out["point_pred"], batch["anchor_fut"], batch["fut_vis"])
            extent_loss = _extent_loss(out["point_pred"], batch["fut_points"], batch["fut_vis"])
            vis_loss = F.binary_cross_entropy_with_logits(out["visibility_logits"], batch["fut_vis"].float())
            proto_logits = out["semantic_logits"].reshape(-1, out["semantic_logits"].shape[-1])
            proto_target = batch["proto_target"][:, None].expand(-1, out["semantic_logits"].shape[1]).reshape(-1)
            sem_loss = F.cross_entropy(proto_logits, proto_target)
            smooth_loss = _temporal_smoothness(out["point_pred"])
            residual_reg = out["delta"].abs().mean()
            residual_target = batch["fut_points"] - out["point_prior"]
            residual_supervision = F.smooth_l1_loss(out["delta"][batch["fut_vis"]], residual_target[batch["fut_vis"]]) if torch.any(batch["fut_vis"]) else out["delta"].new_tensor(0.0)
            total = (
                point_loss
                + 0.75 * endpoint_loss
                + 0.40 * centroid_loss
                + 0.35 * extent_loss
                + 0.20 * vis_loss
                + 0.05 * sem_loss
                + 0.05 * smooth_loss
                + 0.20 * residual_supervision
                + 0.005 * residual_reg
            )
        opt.zero_grad(set_to_none=True)
        scaler.scale(total).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        if step == 1 or step % 500 == 0 or step == args.steps:
            row = {
                "step": step,
                "total": float(total.detach().cpu()),
                "point": float(point_loss.detach().cpu()),
                "endpoint": float(endpoint_loss.detach().cpu()),
                "centroid": float(centroid_loss.detach().cpu()),
                "extent": float(extent_loss.detach().cpu()),
                "visibility": float(vis_loss.detach().cpu()),
                "semantic": float(sem_loss.detach().cpu()),
                "smooth": float(smooth_loss.detach().cpu()),
                "residual_supervision": float(residual_supervision.detach().cpu()),
                "residual_reg": float(residual_reg.detach().cpu()),
                "residual_scale_mean": float(out["residual_scale"].mean().detach().cpu()),
            }
            loss_history.append(row)
            print(
                "[V19][train] " + " ".join(f"{k}={v:.6f}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()),
                flush=True,
            )
        if step % args.eval_every == 0 or step == args.steps:
            val_metrics_tmp, _, _ = _evaluate_model(model, rows["val"], args.batch_size, device)
            score = _val_score(val_metrics_tmp)
            print(
                f"[V19][val] step={step} point={val_metrics_tmp['point_L1_px']:.4f} endpoint={val_metrics_tmp['endpoint_error_px']:.4f} "
                f"extent={val_metrics_tmp['object_extent_iou']:.4f} pck16={val_metrics_tmp['PCK_16px']:.4f} vis={val_metrics_tmp['visibility_F1']:.4f} score={score:.4f}",
                flush=True,
            )
            if score > best_val_score:
                best_val_score = score
                best_step = step
                best_state = {
                    "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                    "config": vars(args),
                    "proto_centers": proto_centers,
                }
    assert best_state is not None
    final_rel = f"outputs/checkpoints/stwm_ostf_v19/{args.experiment_name}_final.pt"
    best_rel = f"outputs/checkpoints/stwm_ostf_v19/{args.experiment_name}_best.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": vars(args),
            "proto_centers": proto_centers,
        },
        ROOT / final_rel,
    )
    torch.save(best_state, ROOT / best_rel)
    model.load_state_dict(best_state["model_state_dict"])

    train_metrics, train_by_ds, _ = _evaluate_model(model, rows["train"], args.batch_size, device)
    val_metrics, val_by_ds, _ = _evaluate_model(model, rows["val"], args.batch_size, device)
    test_metrics, test_by_ds, item_scores = _evaluate_model(model, rows["test"], args.batch_size, device)
    loss_history.append({"best_step": best_step, "best_val_score": best_val_score})

    fair_semantic_logits = semantic_logits_from_observed_memory(rows["test"], proto_centers=proto_centers, proto_count=32)
    fair_semantic_top5 = float(
        (np.argsort(fair_semantic_logits, axis=-1)[..., -5:] == np.asarray([s.proto_target for s in rows["test"]], dtype=np.int64)[:, None, None]).any(axis=-1).mean()
    )
    report = {
        "audit_name": "stwm_ostf_v19_run",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "started_at_utc": started.isoformat(),
        "duration_sec": float(time.time() - wall_start),
        "experiment_name": args.experiment_name,
        "model_kind": args.model_kind,
        "horizon": args.horizon,
        "seed": args.seed,
        "effective_M": 128 if "m128" in args.model_kind else 512,
        "source_combo": combo,
        "steps": args.steps,
        "parameter_count": param_count,
        "checkpoint_path": final_rel,
        "best_checkpoint_path": best_rel,
        "best_val_score": float(best_val_score),
        "loaded_init_key_count": len(loaded_keys),
        "loss_history": loss_history,
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "train_metrics_by_dataset": train_by_ds,
        "val_metrics_by_dataset": val_by_ds,
        "test_metrics_by_dataset": test_by_ds,
        "item_scores": item_scores,
        "metric_note": "V19 uses fair semantic eval for analytic baselines; learned semantic logits are evaluated directly. visibility-masked point losses, centroid, extent, and residual-scale logging are enabled.",
        "fair_semantic_reference_top5_from_observed_memory": fair_semantic_top5,
    }
    out_path = ROOT / f"reports/stwm_ostf_v19_runs/{args.experiment_name}.json"
    dump_json(out_path, report)
    print(out_path.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
