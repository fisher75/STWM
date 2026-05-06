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

from stwm.modules.ostf_traceanything_world_model_v26 import OSTFTraceAnythingConfig, OSTFTraceAnythingWorldModelV26
from stwm.tools.ostf_traceanything_common_v26_20260502 import (
    ROOT,
    batch_from_samples_v26,
    build_v26_rows,
    set_seed,
)
from stwm.tools.ostf_traceanything_metrics_v26_20260502 import (
    aggregate_item_rows_v26,
    hypothesis_diversity_valid_v26,
    multimodal_item_scores_v26,
)


def _apply_process_title() -> None:
    try:
        import setproctitle  # type: ignore

        setproctitle.setproctitle(str(os.environ.get("STWM_PROC_TITLE", "python")))
    except Exception:
        pass


def _combo_for_model(kind: str, horizon: int) -> str:
    return f"M128_H{horizon}" if "m128" in kind else f"M512_H{horizon}"


def _build_model(kind: str, horizon: int) -> OSTFTraceAnythingWorldModelV26:
    num_hyp = 1 if "single_mode" in kind else 6
    hidden_dim = 384 if "m128" in kind else 448
    point_dim = 224 if "m128" in kind else 256
    cfg = OSTFTraceAnythingConfig(
        horizon=horizon,
        hidden_dim=hidden_dim,
        point_dim=point_dim,
        num_layers=4 if horizon <= 32 else 5,
        num_heads=8,
        refinement_layers=2 if "m128" in kind else 3,
        num_hypotheses=num_hyp,
        use_context=True,
        use_dense_points="wo_dense_points" not in kind,
        use_semantic_memory="wo_semantic_memory" not in kind,
        use_physics_prior="wo_physics_prior" not in kind,
        use_multimodal="single_mode" not in kind,
        predict_variance=False,
    )
    return OSTFTraceAnythingWorldModelV26(cfg)


def _mode_selection(
    point_hyp: torch.Tensor,
    gt_points: torch.Tensor,
    valid: torch.Tensor,
    conf: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    point_hyp = torch.nan_to_num(point_hyp, nan=0.0, posinf=1.25, neginf=-0.25)
    diff = F.smooth_l1_loss(point_hyp, gt_points[:, :, :, None, :].expand_as(point_hyp), reduction="none").sum(dim=-1)
    weight = valid[:, :, :, None].float() * (0.5 + conf[:, :, :, None].float())
    denom = weight.sum(dim=(1, 2)).clamp_min(1.0)
    ade = (diff * weight).sum(dim=(1, 2)) / denom
    end_weight = valid[:, :, -1, None].float() * (0.5 + conf[:, :, -1, None].float())
    end = (diff[:, :, -1] * end_weight).sum(dim=1) / end_weight.sum(dim=1).clamp_min(1.0)
    cost = ade + 0.75 * end
    best_idx = cost.argmin(dim=-1)
    gather_idx = best_idx[:, None, None, None, None].expand(-1, point_hyp.shape[1], point_hyp.shape[2], 1, point_hyp.shape[4])
    selected = point_hyp.gather(3, gather_idx).squeeze(3)
    return selected, best_idx, ade, end


def _weighted_mean_per_sample(loss_tensor: torch.Tensor, sample_weight: torch.Tensor) -> torch.Tensor:
    loss_tensor = torch.nan_to_num(loss_tensor, nan=0.0, posinf=1e3, neginf=-1e3)
    return (loss_tensor * sample_weight).sum() / sample_weight.sum().clamp_min(1e-6)


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


def _losses(out: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    fut_points = batch["fut_points"]
    fut_vis = batch["fut_vis"]
    fut_conf = batch["fut_conf"]
    point_hyp = torch.nan_to_num(out["point_hypotheses"], nan=0.0, posinf=1.25, neginf=-0.25)
    point_pred = torch.nan_to_num(out["point_pred"], nan=0.0, posinf=1.25, neginf=-0.25)
    top1_pred = torch.nan_to_num(out["top1_point_pred"], nan=0.0, posinf=1.25, neginf=-0.25)
    selected, best_idx, ade, end = _mode_selection(point_hyp, fut_points, fut_vis, fut_conf)

    sample_weight = 1.0 + 0.45 * batch["hardness_score"] + 0.35 * batch["occlusion_ratio"] + 0.20 * batch["reappearance_flag"]
    sample_weight = sample_weight.clamp(1.0, 4.0)
    weight = fut_vis.float() * (0.5 + fut_conf.float())
    diff_sel = F.smooth_l1_loss(selected, fut_points, reduction="none").sum(dim=-1)
    point_loss_per_sample = (diff_sel * weight).sum(dim=(1, 2)) / weight.sum(dim=(1, 2)).clamp_min(1.0)

    end_sel = F.smooth_l1_loss(selected[:, :, -1], fut_points[:, :, -1], reduction="none").sum(dim=-1)
    end_weight = fut_vis[:, :, -1].float() * (0.5 + fut_conf[:, :, -1].float())
    endpoint_per_sample = (end_sel * end_weight).sum(dim=1) / end_weight.sum(dim=1).clamp_min(1.0)

    diff_det = F.smooth_l1_loss(top1_pred, fut_points, reduction="none").sum(dim=-1)
    det_point_per_sample = (diff_det * weight).sum(dim=(1, 2)) / weight.sum(dim=(1, 2)).clamp_min(1.0)
    det_end = F.smooth_l1_loss(top1_pred[:, :, -1], fut_points[:, :, -1], reduction="none").sum(dim=-1)
    det_endpoint_per_sample = (det_end * end_weight).sum(dim=1) / end_weight.sum(dim=1).clamp_min(1.0)

    pred_anchor = torch.nan_to_num(out["anchor_pred"], nan=0.0, posinf=1.25, neginf=-0.25)
    anchor_diff = F.smooth_l1_loss(pred_anchor, batch["anchor_fut"], reduction="none").sum(dim=-1)
    anchor_per_sample = anchor_diff.mean(dim=1)

    pred_extent, pred_center = _extent_center(selected, fut_vis)
    gt_extent, gt_center = _extent_center(fut_points, fut_vis)
    frame_mask = fut_vis.any(dim=1).float()
    extent_l = F.smooth_l1_loss(pred_extent, gt_extent, reduction="none").sum(dim=-1)
    center_l = F.smooth_l1_loss(pred_center, gt_center, reduction="none").sum(dim=-1)
    extent_per_sample = (extent_l * frame_mask).sum(dim=1) / frame_mask.sum(dim=1).clamp_min(1.0) + 0.5 * (
        (center_l * frame_mask).sum(dim=1) / frame_mask.sum(dim=1).clamp_min(1.0)
    )

    pred_vis = torch.nan_to_num(out["visibility_logits"], nan=0.0, posinf=8.0, neginf=-8.0)
    vis_bce = F.binary_cross_entropy_with_logits(pred_vis, fut_vis.float(), reduction="none")
    vis_per_sample = vis_bce.mean(dim=(1, 2))

    proto_logits = torch.nan_to_num(out["semantic_logits"], nan=0.0, posinf=8.0, neginf=-8.0).reshape(-1, out["semantic_logits"].shape[-1])
    proto_target = batch["proto_target"][:, None].expand(-1, out["semantic_logits"].shape[1]).reshape(-1)
    sem_ce = F.cross_entropy(proto_logits, proto_target, reduction="none").reshape(batch["proto_target"].shape[0], -1).mean(dim=1)

    hyp_logits = torch.nan_to_num(out["hypothesis_logits"], nan=0.0, posinf=8.0, neginf=-8.0)
    if hyp_logits.shape[-1] > 1:
        hyp_loss = F.cross_entropy(hyp_logits, best_idx, reduction="none")
    else:
        hyp_loss = hyp_logits.new_zeros(hyp_logits.shape[0])

    if selected.shape[2] >= 3:
        accel = selected[:, :, 2:] - 2.0 * selected[:, :, 1:-1] + selected[:, :, :-2]
        smooth_per_sample = accel.abs().mean(dim=(1, 2, 3))
    else:
        smooth_per_sample = selected.new_zeros(selected.shape[0])

    residual_mag = out["delta"].abs().mean(dim=(1, 2, 3, 4))
    physics_mix = out["physics_mix_alpha"].mean(dim=1)
    miss_proxy = F.relu(endpoint_per_sample * 1000.0 - 32.0) / 32.0

    if point_hyp.shape[3] > 1:
        learned = point_hyp[:, :, :, 1:] if point_hyp.shape[3] > 1 else point_hyp
        pairwise = []
        for a in range(learned.shape[3]):
            for b in range(a + 1, learned.shape[3]):
                d = torch.norm(learned[:, :, :, a] - learned[:, :, :, b], dim=-1)
                pairwise.append((d * fut_vis.float()).sum(dim=(1, 2)) / fut_vis.float().sum(dim=(1, 2)).clamp_min(1.0))
        if pairwise:
            pairwise_dist = torch.stack(pairwise, dim=1).mean(dim=1)
            diversity_per_sample = F.relu(0.010 - pairwise_dist)
        else:
            diversity_per_sample = selected.new_zeros(selected.shape[0])
    else:
        diversity_per_sample = selected.new_zeros(selected.shape[0])

    nll_per_sample = selected.new_zeros(selected.shape[0])
    if out.get("mode_logvar") is not None:
        logvar = out["mode_logvar"]
        gather_idx = best_idx[:, None, None].expand(-1, logvar.shape[1], 1)
        sel_logvar = logvar.gather(2, gather_idx).squeeze(2)
        sel_logvar = sel_logvar[:, None, :].expand(-1, fut_points.shape[1], -1)
        sqerr = ((selected - fut_points) ** 2).sum(dim=-1)
        nll = 0.5 * (torch.exp(-sel_logvar) * sqerr + sel_logvar)
        nll_per_sample = (nll * weight).sum(dim=(1, 2)) / weight.sum(dim=(1, 2)).clamp_min(1.0)

    total = (
        1.0 * _weighted_mean_per_sample(point_loss_per_sample, sample_weight)
        + 0.70 * _weighted_mean_per_sample(endpoint_per_sample, sample_weight)
        + 0.35 * _weighted_mean_per_sample(anchor_per_sample, sample_weight)
        + 0.35 * _weighted_mean_per_sample(extent_per_sample, sample_weight)
        + 0.20 * _weighted_mean_per_sample(vis_per_sample, sample_weight)
        + 0.10 * _weighted_mean_per_sample(sem_ce, sample_weight)
        + 0.12 * _weighted_mean_per_sample(hyp_loss, sample_weight)
        + 0.05 * _weighted_mean_per_sample(smooth_per_sample, sample_weight)
        + 0.05 * _weighted_mean_per_sample(diversity_per_sample, sample_weight)
        + 0.03 * _weighted_mean_per_sample(residual_mag, sample_weight)
        + 0.08 * _weighted_mean_per_sample(miss_proxy, sample_weight)
        + 0.05 * _weighted_mean_per_sample(nll_per_sample, sample_weight)
    )
    total = torch.nan_to_num(total, nan=1e4, posinf=1e4, neginf=1e4)
    comps = {
        "point": _weighted_mean_per_sample(point_loss_per_sample, sample_weight),
        "endpoint": _weighted_mean_per_sample(endpoint_per_sample, sample_weight),
        "det_point": _weighted_mean_per_sample(det_point_per_sample, sample_weight),
        "det_endpoint": _weighted_mean_per_sample(det_endpoint_per_sample, sample_weight),
        "anchor": _weighted_mean_per_sample(anchor_per_sample, sample_weight),
        "extent": _weighted_mean_per_sample(extent_per_sample, sample_weight),
        "visibility": _weighted_mean_per_sample(vis_per_sample, sample_weight),
        "semantic": _weighted_mean_per_sample(sem_ce, sample_weight),
        "hypothesis_ce": _weighted_mean_per_sample(hyp_loss, sample_weight),
        "diversity": _weighted_mean_per_sample(diversity_per_sample, sample_weight),
        "smooth": _weighted_mean_per_sample(smooth_per_sample, sample_weight),
        "residual_reg": _weighted_mean_per_sample(residual_mag, sample_weight),
        "miss_proxy": _weighted_mean_per_sample(miss_proxy, sample_weight),
        "nll": _weighted_mean_per_sample(nll_per_sample, sample_weight),
        "hard_weight_mean": sample_weight.mean(),
        "selected_ade": ade.mean(),
        "selected_fde": end.mean(),
        "physics_mix_alpha": physics_mix.mean(),
    }
    return total, comps


def _subset_aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "top20_cv_hard": aggregate_item_rows_v26(rows, subset_key="top20_cv_hard"),
        "top30_cv_hard": aggregate_item_rows_v26(rows, subset_key="top30_cv_hard"),
        "occlusion": aggregate_item_rows_v26(rows, subset_key="occlusion_hard"),
        "nonlinear": aggregate_item_rows_v26(rows, subset_key="nonlinear_hard"),
        "interaction": aggregate_item_rows_v26(rows, subset_key="interaction_hard"),
    }


def _evaluate_model(
    model: OSTFTraceAnythingWorldModelV26,
    samples: list[Any],
    batch_size: int,
    device: torch.device,
    cv_mode_index: int,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], list[dict[str, Any]], bool]:
    model.eval()
    point_modes = []
    mode_logits = []
    top1_preds = []
    weighted_preds = []
    vis_logits = []
    sem_logits = []
    logvars = []
    with torch.no_grad():
        for start in range(0, len(samples), batch_size):
            batch_rows = samples[start : start + batch_size]
            batch = batch_from_samples_v26(batch_rows, device)
            out = model(
                obs_points=batch["obs_points"],
                obs_vis=batch["obs_vis"],
                obs_conf=batch["obs_conf"],
                rel_xy=batch["rel_xy"],
                anchor_obs=batch["anchor_obs"],
                anchor_obs_vel=batch["anchor_obs_vel"],
                semantic_feat=batch["semantic_feat"],
                semantic_id=batch["semantic_id"],
                box_feat=batch["box_feat"],
                neighbor_feat=batch["neighbor_feat"],
                global_feat=batch["global_feat"],
                tusb_token=batch["tusb_token"],
            )
            point_modes.append(out["point_hypotheses"].detach().cpu().numpy())
            mode_logits.append(out["hypothesis_logits"].detach().cpu().numpy())
            top1_preds.append(out["top1_point_pred"].detach().cpu().numpy())
            weighted_preds.append(out["point_pred"].detach().cpu().numpy())
            vis_logits.append(out["visibility_logits"].detach().cpu().numpy())
            sem_logits.append(out["semantic_logits"].detach().cpu().numpy())
            if out.get("mode_logvar") is not None:
                logvars.append(out["mode_logvar"].detach().cpu().numpy())
    point_modes_np = np.concatenate(point_modes, axis=0)
    mode_logits_np = np.concatenate(mode_logits, axis=0)
    top1_np = np.concatenate(top1_preds, axis=0)
    weighted_np = np.concatenate(weighted_preds, axis=0)
    vis_logits_np = np.concatenate(vis_logits, axis=0)
    sem_logits_np = np.concatenate(sem_logits, axis=0)
    logvar_np = np.concatenate(logvars, axis=0) if logvars else None
    rows = multimodal_item_scores_v26(
        samples,
        point_modes=point_modes_np,
        mode_logits=mode_logits_np,
        top1_point_pred=top1_np,
        weighted_point_pred=weighted_np,
        pred_vis_logits=vis_logits_np,
        pred_proto_logits=sem_logits_np,
        pred_logvar=logvar_np,
        cv_mode_index=cv_mode_index,
    )
    all_metrics = aggregate_item_rows_v26(rows)
    by_dataset = {ds: aggregate_item_rows_v26(rows, dataset=ds) for ds in sorted({s.dataset for s in samples})}
    subsets = _subset_aggregate(rows)
    return all_metrics, by_dataset, subsets, rows, hypothesis_diversity_valid_v26(rows)


def _val_score(all_metrics: dict[str, Any], hard_metrics: dict[str, Any]) -> float:
    def _sf(x: Any, bad: float) -> float:
        try:
            v = float(x)
        except Exception:
            return bad
        return v if np.isfinite(v) else bad

    return (
        -1.25 * _sf(hard_metrics.get("minFDE_K_px"), 1e6)
        - 0.60 * _sf(hard_metrics.get("minADE_K_px"), 1e6)
        + 18.0 * _sf(hard_metrics.get("BestOfK_PCK_16px"), 0.0)
        + 10.0 * _sf(hard_metrics.get("BestOfK_PCK_32px"), 0.0)
        - 10.0 * _sf(hard_metrics.get("MissRate_32px"), 1.0)
        + 8.0 * _sf(hard_metrics.get("top1_visibility_F1"), 0.0)
        + 5.0 * _sf(all_metrics.get("semantic_top5"), 0.0)
        - 0.10 * _sf(all_metrics.get("top1_endpoint_error_px"), 1e6)
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
    model.load_state_dict(dst, strict=False)
    return loaded


def main() -> int:
    _apply_process_title()
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument(
        "--model-kind",
        required=True,
        choices=[
            "v26_traceanything_m128_h32",
            "v26_traceanything_m128_h32_wo_semantic_memory",
            "v26_traceanything_m128_h32_wo_dense_points",
            "v26_traceanything_m128_h32_single_mode",
            "v26_traceanything_m128_h32_wo_physics_prior",
            "v26_traceanything_m512_h32",
            "v26_traceanything_m128_h64",
        ],
    )
    parser.add_argument("--horizon", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=30000)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--eval-every", type=int, default=1500)
    parser.add_argument("--init-from-checkpoint", default=None)
    args = parser.parse_args()

    started = datetime.now(timezone.utc)
    wall_start = time.time()
    set_seed(args.seed)
    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    combo = _combo_for_model(args.model_kind, args.horizon)
    rows, proto_centers = build_v26_rows(combo, seed=args.seed)
    if not rows["train"] or not rows["val"] or not rows["test"]:
        raise SystemExit(f"No usable TraceAnything V25 samples found for combo={combo}")
    model = _build_model(args.model_kind, args.horizon).to(device)
    param_count = int(sum(p.numel() for p in model.parameters()))
    lr = 2e-4 if "m128" in args.model_kind else 1.5e-4
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=1e-4)
    scaler = GradScaler(device="cuda", enabled=device.type == "cuda")
    loaded_keys = []
    if args.init_from_checkpoint:
        loaded_keys = _load_init_checkpoint(model, args.init_from_checkpoint)

    train_weights = [float(max(1.0, s.hardness_score)) for s in rows["train"]]
    cv_mode_index = 0 if "single_mode" not in args.model_kind else -1

    print(
        f"[V26][start] exp={args.experiment_name} kind={args.model_kind} combo={combo} h={args.horizon} seed={args.seed} steps={args.steps} batch={args.batch_size} device={device}",
        flush=True,
    )
    print(f"[V26][data] train={len(rows['train'])} val={len(rows['val'])} test={len(rows['test'])}", flush=True)
    print(f"[V26][model] params={param_count} init_loaded={len(loaded_keys)}", flush=True)

    best_state = None
    best_val_score = -1e18
    best_step = 0
    loss_history = []
    ckpt_dir = ROOT / "outputs/checkpoints/stwm_ostf_v26"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    final_rel = f"outputs/checkpoints/stwm_ostf_v26/{args.experiment_name}_final.pt"
    best_rel = f"outputs/checkpoints/stwm_ostf_v26/{args.experiment_name}_best.pt"

    for step in range(1, args.steps + 1):
        batch_rows = random.choices(rows["train"], weights=train_weights, k=min(args.batch_size, len(rows["train"])))
        batch = batch_from_samples_v26(batch_rows, device)
        model.train()
        with autocast(device_type="cuda", enabled=device.type == "cuda"):
            out = model(
                obs_points=batch["obs_points"],
                obs_vis=batch["obs_vis"],
                obs_conf=batch["obs_conf"],
                rel_xy=batch["rel_xy"],
                anchor_obs=batch["anchor_obs"],
                anchor_obs_vel=batch["anchor_obs_vel"],
                semantic_feat=batch["semantic_feat"],
                semantic_id=batch["semantic_id"],
                box_feat=batch["box_feat"],
                neighbor_feat=batch["neighbor_feat"],
                global_feat=batch["global_feat"],
                tusb_token=batch["tusb_token"],
            )
            total, comps = _losses(out, batch)
        if not torch.isfinite(total):
            print(f"[V26][warn] step={step} nonfinite_total_skip_batch", flush=True)
            opt.zero_grad(set_to_none=True)
            continue
        opt.zero_grad(set_to_none=True)
        scaler.scale(total).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.75 if "m512" in args.model_kind else 1.0)
        scaler.step(opt)
        scaler.update()

        if step == 1 or step % 500 == 0 or step == args.steps:
            safe_hyp_logits = torch.nan_to_num(
                out["hypothesis_logits"].detach().float(),
                nan=0.0,
                posinf=25.0,
                neginf=-25.0,
            )
            row = {"step": step, "total": float(total.detach().cpu())}
            row.update({k: float(v.detach().cpu()) for k, v in comps.items()})
            row["residual_scale"] = float(out["residual_scale"].mean().detach().cpu())
            row["hyp_entropy"] = float(torch.distributions.Categorical(logits=safe_hyp_logits).entropy().mean().detach().cpu())
            loss_history.append(row)
            print("[V26][train] " + " ".join(f"{k}={v:.6f}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()), flush=True)

        if step % args.eval_every == 0 or step == args.steps:
            val_all, _, val_sub, _, val_div = _evaluate_model(model, rows["val"], args.batch_size, device, cv_mode_index)
            score = _val_score(val_all, val_sub["top20_cv_hard"])
            print(
                f"[V26][val] step={step} top1_point={val_all['top1_point_L1_px']:.4f} minfde={val_all['minFDE_K_px']:.4f} "
                f"hard_minfde={val_sub['top20_cv_hard'].get('minFDE_K_px')} hard_miss32={val_sub['top20_cv_hard'].get('MissRate_32px')} "
                f"hard_bestpck16={val_sub['top20_cv_hard'].get('BestOfK_PCK_16px')} div={int(val_div)} score={score:.4f}",
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
                torch.save(best_state, ROOT / best_rel)

    assert best_state is not None
    torch.save({"model_state_dict": model.state_dict(), "config": vars(args), "proto_centers": proto_centers}, ROOT / final_rel)
    torch.save(best_state, ROOT / best_rel)
    model.load_state_dict(best_state["model_state_dict"])

    train_all, train_ds, train_sub, train_rows, train_div = _evaluate_model(model, rows["train"], args.batch_size, device, cv_mode_index)
    val_all, val_ds, val_sub, val_rows, val_div = _evaluate_model(model, rows["val"], args.batch_size, device, cv_mode_index)
    test_all, test_ds, test_sub, test_rows, test_div = _evaluate_model(model, rows["test"], args.batch_size, device, cv_mode_index)
    loss_history.append({"best_step": best_step, "best_val_score": best_val_score})

    report = {
        "audit_name": "stwm_ostf_v26_run",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "started_at_utc": started.isoformat(),
        "duration_sec": float(time.time() - wall_start),
        "experiment_name": args.experiment_name,
        "model_kind": args.model_kind,
        "source_combo": combo,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "device": str(device),
        "parameter_count": param_count,
        "best_step": best_step,
        "best_val_score": float(best_val_score),
        "best_checkpoint_path": best_rel,
        "final_checkpoint_path": final_rel,
        "loaded_init_keys_count": len(loaded_keys),
        "teacher_source": "traceanything_official_trajectory_field",
        "model_input_observed_only": True,
        "train_metrics": train_all,
        "train_subset_metrics": train_sub,
        "train_metrics_by_dataset": train_ds,
        "train_diversity_valid": train_div,
        "val_metrics": val_all,
        "val_subset_metrics": val_sub,
        "val_metrics_by_dataset": val_ds,
        "val_diversity_valid": val_div,
        "test_metrics": test_all,
        "test_subset_metrics": test_sub,
        "test_metrics_by_dataset": test_ds,
        "test_diversity_valid": test_div,
        "loss_history_tail": loss_history[-20:],
        "item_scores": test_rows,
    }
    out_path = ROOT / "reports/stwm_ostf_v26_runs" / f"{args.experiment_name}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(__import__("json").dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"[V26][done] report={out_path} best={best_rel} final={final_rel}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
