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

from stwm.modules.ostf_calibrated_multimodal_world_model_v22 import (
    OSTFCalibratedMultimodalConfig,
    OSTFCalibratedMultimodalWorldModel,
)
from stwm.tools.materialize_ostf_point_selection_v22_20260502 import (
    derive_rows_with_point_selection,
    point_metadata,
)
from stwm.tools.ostf_multimodal_metrics_v22 import aggregate_rows_v22, calibration_summary, multimodal_item_scores_v22
from stwm.tools.ostf_v17_common_20260502 import ROOT, OSTFObjectSample, batch_from_samples, dump_json, set_seed
from stwm.tools.ostf_v20_common_20260502 import (
    build_context_cache_for_combo,
    hard_subset_flags,
    load_context_cache,
    load_combo_rows,
    sample_key,
    save_context_cache,
)


def _apply_process_title() -> None:
    try:
        import setproctitle  # type: ignore

        setproctitle.setproctitle(str(os.environ.get("STWM_PROC_TITLE", "python")))
    except Exception:
        pass


def combo_for_model(kind: str, horizon: int) -> str:
    if "m256" in kind or "m512" in kind:
        return f"M512_H{horizon}"
    return f"M128_H{horizon}"


def point_recipe_for_model(kind: str) -> tuple[int | None, str]:
    if "m512" in kind:
        return 512, "boundary_interior"
    if "m256" in kind:
        return 256, "boundary_interior"
    if "m128" in kind:
        return None, "native"
    raise ValueError(f"Unsupported model kind: {kind}")


def prepare_rows_for_model(kind: str, horizon: int, seed: int) -> tuple[dict[str, list[OSTFObjectSample]], np.ndarray, str, int | None, str]:
    combo = combo_for_model(kind, horizon)
    rows, proto_centers = load_combo_rows(combo, seed=seed)
    target_m, strategy = point_recipe_for_model(kind)
    if target_m is not None and target_m < rows["train"][0].m:
        rows = derive_rows_with_point_selection(rows, target_m=target_m, strategy=strategy, seed=seed)
    return rows, proto_centers, combo, target_m, strategy


def build_model(kind: str, horizon: int) -> OSTFCalibratedMultimodalWorldModel:
    num_hyp = 1 if "single_mode" in kind else 8
    cfg = OSTFCalibratedMultimodalConfig(
        horizon=horizon,
        hidden_dim=384,
        point_dim=224,
        num_layers=4,
        num_heads=8,
        refinement_layers=2,
        num_hypotheses=num_hyp,
        use_context="wo_context" not in kind,
        use_dense_points="wo_dense_points" not in kind,
        use_semantic_memory="wo_semantic_memory" not in kind,
        use_cv_mode=("single_mode" not in kind),
        use_affine_prior=True,
        use_cv_prior=True,
    )
    return OSTFCalibratedMultimodalWorldModel(cfg)


def _sample_weights(records: list[dict[str, Any]]) -> list[float]:
    scores = np.asarray([r["hardness_score"] for r in records], dtype=np.float32)
    mu = float(scores.mean()) if scores.size else 0.0
    std = float(scores.std() + 1e-6) if scores.size else 1.0
    out = []
    for r in records:
        z = (float(r["hardness_score"]) - mu) / std
        w = 1.0 + max(0.0, 0.70 * z) + 0.40 * float(r["occlusion_ratio"]) + 0.25 * float(r["reappearance_flag"])
        out.append(float(min(4.0, max(1.0, w))))
    return out


def context_batch(batch_rows: list[OSTFObjectSample], ctx_map: dict[tuple[str, int], dict[str, Any]], device: torch.device) -> dict[str, torch.Tensor]:
    crop = []
    box = []
    neigh = []
    glob = []
    hard = []
    occ = []
    reap = []
    for s in batch_rows:
        c = ctx_map[sample_key(s)]
        crop.append(c["crop_feat"])
        box.append(c["box_feat"])
        neigh.append(c["neighbor_feat"])
        glob.append(c["global_feat"])
        hard.append(c["hardness_score"])
        occ.append(c["occlusion_ratio"])
        reap.append(c["reappearance_flag"])
    return {
        "crop_feat": torch.tensor(np.stack(crop), device=device, dtype=torch.float32),
        "box_feat": torch.tensor(np.stack(box), device=device, dtype=torch.float32),
        "neighbor_feat": torch.tensor(np.stack(neigh), device=device, dtype=torch.float32),
        "global_feat": torch.tensor(np.stack(glob), device=device, dtype=torch.float32),
        "hardness_score": torch.tensor(hard, device=device, dtype=torch.float32),
        "occlusion_ratio": torch.tensor(occ, device=device, dtype=torch.float32),
        "reappearance_flag": torch.tensor(reap, device=device, dtype=torch.float32),
    }


def batch_from_samples_v22(samples: list[OSTFObjectSample], device: torch.device) -> dict[str, torch.Tensor]:
    batch = batch_from_samples(samples, device)
    batch["point_meta"] = torch.tensor(np.stack([point_metadata(s) for s in samples]), device=device, dtype=torch.float32)
    return batch


def subset_flags(samples: list[OSTFObjectSample], ctx_map: dict[tuple[str, int], dict[str, Any]]) -> dict[str, np.ndarray]:
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


def _mode_costs(point_hyp: torch.Tensor, gt_points: torch.Tensor, valid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    diff = F.smooth_l1_loss(point_hyp, gt_points[:, :, :, None, :].expand_as(point_hyp), reduction="none").sum(dim=-1)
    mask = valid[:, :, :, None].float()
    denom = mask.sum(dim=(1, 2)).clamp_min(1.0)
    ade = (diff * mask).sum(dim=(1, 2)) / denom
    end_mask = valid[:, :, -1, None].float()
    end = (diff[:, :, -1] * end_mask).sum(dim=1) / end_mask.sum(dim=1).clamp_min(1.0)
    return ade, end


def _select_mode(point_hyp: torch.Tensor, best_idx: torch.Tensor) -> torch.Tensor:
    gather_idx = best_idx[:, None, None, None, None].expand(-1, point_hyp.shape[1], point_hyp.shape[2], 1, point_hyp.shape[4])
    return point_hyp.gather(3, gather_idx).squeeze(3)


def _weighted_mean_per_sample(loss_tensor: torch.Tensor, sample_weight: torch.Tensor) -> torch.Tensor:
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


def _mixture_nll(
    point_hyp: torch.Tensor,
    gt_points: torch.Tensor,
    valid: torch.Tensor,
    mode_logits: torch.Tensor,
    logvar_modes: torch.Tensor | None,
) -> torch.Tensor:
    if logvar_modes is None:
        return gt_points.new_tensor(0.0)
    err2 = ((point_hyp - gt_points[:, :, :, None, :]) ** 2).sum(dim=-1)
    logvar = logvar_modes[:, None, :, :].expand_as(err2)
    mask = valid[:, :, :, None].float()
    denom = mask.sum(dim=(1, 2)).clamp_min(1.0)
    log_comp = -0.5 * (((err2 / torch.exp(logvar)) + logvar) * mask).sum(dim=(1, 2)) / denom[:, None]
    log_prob = F.log_softmax(mode_logits, dim=-1) + log_comp
    return -torch.logsumexp(log_prob, dim=-1).mean()


def _losses(
    out: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    batch_ctx: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    fut_points = batch["fut_points"]
    fut_vis = batch["fut_vis"]
    point_hyp = torch.nan_to_num(out["point_hypotheses"], nan=0.0, posinf=1.25, neginf=-0.25)
    point_pred = torch.nan_to_num(out["point_pred"], nan=0.0, posinf=1.25, neginf=-0.25)
    top1_pred = torch.nan_to_num(out["top1_point_pred"], nan=0.0, posinf=1.25, neginf=-0.25)
    mode_logits = torch.nan_to_num(out["hypothesis_logits"], nan=0.0, posinf=8.0, neginf=-8.0)
    logvar_modes = torch.nan_to_num(out["mode_logvar"], nan=0.0, posinf=2.0, neginf=-6.0)

    ade, end = _mode_costs(point_hyp, fut_points, fut_vis)
    cost = ade + 0.75 * end
    best_idx = cost.argmin(dim=-1)
    soft_target = torch.softmax(-cost / 4.0, dim=-1)
    selected = _select_mode(point_hyp, best_idx)

    hard = batch_ctx["hardness_score"]
    occ = batch_ctx["occlusion_ratio"]
    reap = batch_ctx["reappearance_flag"]
    hard_z = (hard - hard.mean()) / (hard.std() + 1e-6)
    sample_weight = (1.0 + torch.clamp(0.70 * hard_z, min=0.0) + 0.35 * occ + 0.20 * reap).clamp(1.0, 4.0)

    mask = fut_vis.float()
    diff_sel = F.smooth_l1_loss(selected, fut_points, reduction="none").sum(dim=-1)
    point_loss_per_sample = (diff_sel * mask).sum(dim=(1, 2)) / mask.sum(dim=(1, 2)).clamp_min(1.0)

    diff_top1 = F.smooth_l1_loss(top1_pred, fut_points, reduction="none").sum(dim=-1)
    top1_point_per_sample = (diff_top1 * mask).sum(dim=(1, 2)) / mask.sum(dim=(1, 2)).clamp_min(1.0)

    diff_det = F.smooth_l1_loss(point_pred, fut_points, reduction="none").sum(dim=-1)
    det_point_per_sample = (diff_det * mask).sum(dim=(1, 2)) / mask.sum(dim=(1, 2)).clamp_min(1.0)

    end_mask = fut_vis[:, :, -1].float()
    end_sel = F.smooth_l1_loss(selected[:, :, -1], fut_points[:, :, -1], reduction="none").sum(dim=-1)
    endpoint_per_sample = (end_sel * end_mask).sum(dim=1) / end_mask.sum(dim=1).clamp_min(1.0)
    end_top1 = F.smooth_l1_loss(top1_pred[:, :, -1], fut_points[:, :, -1], reduction="none").sum(dim=-1)
    top1_endpoint_per_sample = (end_top1 * end_mask).sum(dim=1) / end_mask.sum(dim=1).clamp_min(1.0)

    anchor_pred = torch.nan_to_num(out["anchor_pred"], nan=0.0, posinf=1.25, neginf=-0.25)
    anchor_diff = F.smooth_l1_loss(anchor_pred, batch["anchor_fut"], reduction="none").sum(dim=-1)
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

    if mode_logits.shape[-1] > 1:
        hard_ce = F.cross_entropy(mode_logits, best_idx, reduction="none")
        log_probs = F.log_softmax(mode_logits, dim=-1)
        soft_ce = -(soft_target * log_probs).sum(dim=-1)
    else:
        hard_ce = mode_logits.new_zeros(mode_logits.shape[0])
        soft_ce = mode_logits.new_zeros(mode_logits.shape[0])

    if selected.shape[2] >= 3:
        accel = selected[:, :, 2:] - 2.0 * selected[:, :, 1:-1] + selected[:, :, :-2]
        smooth_per_sample = accel.abs().mean(dim=(1, 2, 3))
    else:
        smooth_per_sample = selected.new_zeros(selected.shape[0])
    residual_mag = out["delta"].abs().mean(dim=(1, 2, 3, 4))

    if point_hyp.shape[3] > 1:
        learned = point_hyp[:, :, :, 1:] if point_hyp.shape[3] > 1 else point_hyp
        pairwise = []
        for a in range(learned.shape[3]):
            for b in range(a + 1, learned.shape[3]):
                d = torch.norm(learned[:, :, :, a] - learned[:, :, :, b], dim=-1)
                pairwise.append((d * fut_vis.float()).sum(dim=(1, 2)) / fut_vis.float().sum(dim=(1, 2)).clamp_min(1.0))
        if pairwise:
            pairwise_dist = torch.stack(pairwise, dim=1).mean(dim=1)
            diversity_per_sample = F.relu(0.012 - pairwise_dist)
        else:
            diversity_per_sample = selected.new_zeros(selected.shape[0])
    else:
        diversity_per_sample = selected.new_zeros(selected.shape[0])

    mix_nll = _mixture_nll(point_hyp, fut_points, fut_vis, mode_logits, logvar_modes)
    total = (
        _weighted_mean_per_sample(point_loss_per_sample, sample_weight)
        + 0.85 * _weighted_mean_per_sample(endpoint_per_sample, sample_weight)
        + 0.55 * _weighted_mean_per_sample(top1_point_per_sample, sample_weight)
        + 0.45 * _weighted_mean_per_sample(top1_endpoint_per_sample, sample_weight)
        + 0.18 * _weighted_mean_per_sample(det_point_per_sample, sample_weight)
        + 0.30 * _weighted_mean_per_sample(anchor_per_sample, sample_weight)
        + 0.22 * _weighted_mean_per_sample(extent_per_sample, sample_weight)
        + 0.20 * _weighted_mean_per_sample(vis_per_sample, sample_weight)
        + 0.05 * _weighted_mean_per_sample(sem_ce, sample_weight)
        + 0.12 * _weighted_mean_per_sample(hard_ce, sample_weight)
        + 0.06 * _weighted_mean_per_sample(soft_ce, sample_weight)
        + 0.08 * mix_nll
        + 0.03 * _weighted_mean_per_sample(diversity_per_sample, sample_weight)
        + 0.02 * _weighted_mean_per_sample(smooth_per_sample, sample_weight)
        + 0.005 * _weighted_mean_per_sample(residual_mag, sample_weight)
    )
    comps = {
        "point": _weighted_mean_per_sample(point_loss_per_sample, sample_weight),
        "endpoint": _weighted_mean_per_sample(endpoint_per_sample, sample_weight),
        "top1_point": _weighted_mean_per_sample(top1_point_per_sample, sample_weight),
        "top1_endpoint": _weighted_mean_per_sample(top1_endpoint_per_sample, sample_weight),
        "det_point": _weighted_mean_per_sample(det_point_per_sample, sample_weight),
        "anchor": _weighted_mean_per_sample(anchor_per_sample, sample_weight),
        "extent": _weighted_mean_per_sample(extent_per_sample, sample_weight),
        "visibility": _weighted_mean_per_sample(vis_per_sample, sample_weight),
        "semantic": _weighted_mean_per_sample(sem_ce, sample_weight),
        "mode_ce": _weighted_mean_per_sample(hard_ce, sample_weight),
        "mode_soft_ce": _weighted_mean_per_sample(soft_ce, sample_weight),
        "mix_nll": mix_nll,
        "diversity": _weighted_mean_per_sample(diversity_per_sample, sample_weight),
        "smooth": _weighted_mean_per_sample(smooth_per_sample, sample_weight),
        "residual_reg": _weighted_mean_per_sample(residual_mag, sample_weight),
        "hard_weight_mean": sample_weight.mean(),
        "oracle_ade": ade.mean(),
        "oracle_fde": end.mean(),
        "top1_match_acc": (mode_logits.argmax(dim=-1) == best_idx).float().mean(),
    }
    return total, comps


def evaluate_model(
    model: OSTFCalibratedMultimodalWorldModel,
    samples: list[OSTFObjectSample],
    ctx_map: dict[tuple[str, int], dict[str, Any]],
    batch_size: int,
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    model.eval()
    point_modes = []
    mode_logits = []
    point_preds = []
    top1_preds = []
    vis_logits = []
    sem_logits = []
    logvar_modes = []
    with torch.no_grad():
        for start in range(0, len(samples), batch_size):
            batch_rows = samples[start : start + batch_size]
            batch = batch_from_samples_v22(batch_rows, device)
            batch_ctx = context_batch(batch_rows, ctx_map, device)
            out = model(
                obs_points=batch["obs_points"],
                obs_vis=batch["obs_vis"],
                rel_xy=batch["rel_xy"],
                point_meta=batch["point_meta"],
                anchor_obs=batch["anchor_obs"],
                anchor_obs_vel=batch["anchor_obs_vel"],
                semantic_feat=batch["semantic_feat"],
                crop_feat=batch_ctx["crop_feat"],
                box_feat=batch_ctx["box_feat"],
                neighbor_feat=batch_ctx["neighbor_feat"],
                global_feat=batch_ctx["global_feat"],
            )
            point_modes.append(out["point_hypotheses"].detach().cpu().numpy())
            mode_logits.append(out["hypothesis_logits"].detach().cpu().numpy())
            point_preds.append(out["point_pred"].detach().cpu().numpy())
            top1_preds.append(out["top1_point_pred"].detach().cpu().numpy())
            vis_logits.append(out["visibility_logits"].detach().cpu().numpy())
            sem_logits.append(out["semantic_logits"].detach().cpu().numpy())
            logvar_modes.append(out["mode_logvar"].detach().cpu().numpy())
    point_modes_np = np.concatenate(point_modes, axis=0)
    mode_logits_np = np.concatenate(mode_logits, axis=0)
    point_pred_np = np.concatenate(point_preds, axis=0)
    top1_pred_np = np.concatenate(top1_preds, axis=0)
    vis_logits_np = np.concatenate(vis_logits, axis=0)
    sem_logits_np = np.concatenate(sem_logits, axis=0)
    logvar_np = np.concatenate(logvar_modes, axis=0)
    flags = subset_flags(samples, ctx_map)
    rows_mm = multimodal_item_scores_v22(
        samples,
        point_modes=point_modes_np,
        mode_logits=mode_logits_np,
        point_pred=point_pred_np,
        top1_pred=top1_pred_np,
        pred_vis_logits=vis_logits_np,
        pred_proto_logits=sem_logits_np,
        logvar_modes=logvar_np,
        subset_flags=flags,
        cv_mode_index=0 if model.cfg.use_cv_mode else -1,
    )
    all_metrics = aggregate_rows_v22(rows_mm)
    by_dataset = {ds: aggregate_rows_v22(rows_mm, dataset=ds) for ds in sorted({s.dataset for s in samples})}
    subsets = {
        "cv_hard_top20": aggregate_rows_v22(rows_mm, subset_key="top20_cv_hard"),
        "occlusion": aggregate_rows_v22(rows_mm, subset_key="occlusion_hard"),
        "nonlinear": aggregate_rows_v22(rows_mm, subset_key="nonlinear_hard"),
        "interaction": aggregate_rows_v22(rows_mm, subset_key="interaction_hard"),
    }
    calib = calibration_summary(rows_mm)
    return all_metrics, by_dataset, subsets, rows_mm, calib


def _val_score(all_metrics: dict[str, Any], hard_metrics: dict[str, Any], calib: dict[str, Any]) -> float:
    return (
        -1.10 * float(hard_metrics["top1_endpoint_error_px"])
        - 0.55 * float(hard_metrics["top1_point_l1_px"])
        - 0.25 * float(all_metrics["weighted_point_l1_px"])
        - 0.15 * float(all_metrics["weighted_endpoint_error_px"])
        - 0.30 * float(hard_metrics["expected_FDE_px"])
        + 18.0 * float(hard_metrics["top1_PCK_16px"])
        + 10.0 * float(hard_metrics["top1_PCK_32px"])
        + 8.0 * float(hard_metrics["BestOfK_PCK_16px"])
        + 4.0 * float(hard_metrics["BestOfK_PCK_32px"])
        - 8.0 * float(hard_metrics["top1_MissRate_32px"])
        - 4.0 * float(hard_metrics["MissRate_32px"])
        + 80.0 * float(all_metrics["weighted_object_extent_iou"])
        + 60.0 * float(hard_metrics["top1_object_extent_iou"])
        + 8.0 * float(calib["top1_mode_accuracy"])
        - 3.0 * float(calib["ece_top1_mode"])
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
            "v22_calibrated_m32",
            "v22_calibrated_m128",
            "v22_calibrated_m256",
            "v22_calibrated_m512",
            "v22_calibrated_m32_wo_context",
            "v22_calibrated_m128_wo_context",
            "v22_calibrated_m256_wo_context",
            "v22_calibrated_m512_wo_context",
            "v22_calibrated_m32_wo_dense_points",
            "v22_calibrated_m128_wo_dense_points",
            "v22_calibrated_m256_wo_dense_points",
            "v22_calibrated_m512_wo_dense_points",
            "v22_calibrated_m32_wo_semantic_memory",
            "v22_calibrated_m128_wo_semantic_memory",
            "v22_calibrated_m256_wo_semantic_memory",
            "v22_calibrated_m512_wo_semantic_memory",
            "v22_calibrated_m32_single_mode",
            "v22_calibrated_m128_single_mode",
            "v22_calibrated_m256_single_mode",
            "v22_calibrated_m512_single_mode",
        ],
    )
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=30000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--eval-every", type=int, default=1500)
    parser.add_argument("--init-from-checkpoint", default=None)
    parser.add_argument("--context-cache-path", default=None)
    parser.add_argument("--lr", type=float, default=None)
    args = parser.parse_args()

    started = datetime.now(timezone.utc)
    wall_start = time.time()
    set_seed(args.seed)
    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")

    rows, proto_centers, combo, target_m, point_strategy = prepare_rows_for_model(args.model_kind, args.horizon, args.seed)
    context_cache_path = Path(args.context_cache_path) if args.context_cache_path else ROOT / "outputs/cache/stwm_ostf_context_features_v20" / f"{combo}_context_features.npz"
    if not context_cache_path.exists():
        bundle = build_context_cache_for_combo(combo, seed=args.seed)
        context_cache_path.parent.mkdir(parents=True, exist_ok=True)
        save_context_cache(bundle, context_cache_path)
    ctx_map = load_context_cache(context_cache_path)
    train_records = [ctx_map[sample_key(s)] for s in rows["train"]]
    train_weights = _sample_weights(train_records)
    train_pairs = list(zip(rows["train"], train_weights))

    model = build_model(args.model_kind, args.horizon).to(device)
    param_count = int(sum(p.numel() for p in model.parameters()))
    lr = args.lr if args.lr is not None else (5e-4 if "m128" in args.model_kind else 4e-4)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scaler = GradScaler(device="cuda", enabled=device.type == "cuda")
    loaded_keys = []
    if args.init_from_checkpoint:
        loaded_keys = _load_init_checkpoint(model, args.init_from_checkpoint)

    print(
        f"[V22][start] exp={args.experiment_name} kind={args.model_kind} combo={combo} target_m={target_m or 'native'} "
        f"strategy={point_strategy} h={args.horizon} seed={args.seed} steps={args.steps} batch={args.batch_size} device={device}",
        flush=True,
    )
    print(f"[V22][data] train={len(rows['train'])} val={len(rows['val'])} test={len(rows['test'])}", flush=True)
    print(f"[V22][model] params={param_count} init_loaded={len(loaded_keys)} context_cache={context_cache_path}", flush=True)

    best_state = None
    best_val_score = -1e18
    best_step = 0
    loss_history: list[dict[str, Any]] = []
    ckpt_dir = ROOT / "outputs/checkpoints/stwm_ostf_v22"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    final_rel = f"outputs/checkpoints/stwm_ostf_v22/{args.experiment_name}_final.pt"
    best_rel = f"outputs/checkpoints/stwm_ostf_v22/{args.experiment_name}_best.pt"

    for step in range(1, args.steps + 1):
        batch_rows = [s for s, _ in random.choices(train_pairs, weights=[w for _, w in train_pairs], k=min(args.batch_size, len(train_pairs)))]
        batch = batch_from_samples_v22(batch_rows, device)
        batch_ctx = context_batch(batch_rows, ctx_map, device)
        model.train()
        with autocast(device_type="cuda", enabled=device.type == "cuda"):
            out = model(
                obs_points=batch["obs_points"],
                obs_vis=batch["obs_vis"],
                rel_xy=batch["rel_xy"],
                point_meta=batch["point_meta"],
                anchor_obs=batch["anchor_obs"],
                anchor_obs_vel=batch["anchor_obs_vel"],
                semantic_feat=batch["semantic_feat"],
                crop_feat=batch_ctx["crop_feat"],
                box_feat=batch_ctx["box_feat"],
                neighbor_feat=batch_ctx["neighbor_feat"],
                global_feat=batch_ctx["global_feat"],
            )
            total, comps = _losses(out, batch, batch_ctx)
        if not torch.isfinite(total):
            print(f"[V22][warn] step={step} nonfinite_total_skip_batch", flush=True)
            opt.zero_grad(set_to_none=True)
            continue
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
            row["hyp_entropy"] = float(torch.distributions.Categorical(logits=out["hypothesis_logits"]).entropy().mean().detach().cpu())
            row["temperature"] = float(out["hypothesis_temperature"].mean().detach().cpu())
            loss_history.append(row)
            print("[V22][train] " + " ".join(f"{k}={v:.6f}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()), flush=True)

        if step % args.eval_every == 0 or step == args.steps:
            val_all, _, val_sub, _, val_calib = evaluate_model(model, rows["val"], ctx_map, args.batch_size, device)
            score = _val_score(val_all, val_sub["cv_hard_top20"], val_calib)
            print(
                f"[V22][val] step={step} top1_fde={val_all['top1_endpoint_error_px']:.4f} top1_p16={val_all['top1_PCK_16px']:.4f} "
                f"minfde={val_all['minFDE_K_px']:.4f} hard_top1_fde={val_sub['cv_hard_top20']['top1_endpoint_error_px']:.4f} "
                f"hard_bestofk_fde={val_sub['cv_hard_top20']['minFDE_K_px']:.4f} hard_miss32={val_sub['cv_hard_top20']['top1_MissRate_32px']:.4f} "
                f"calib_acc={val_calib['top1_mode_accuracy']:.4f} ece={val_calib['ece_top1_mode']:.4f} score={score:.4f}",
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
                    "target_m": target_m,
                    "point_strategy": point_strategy,
                }
                torch.save(best_state, ROOT / best_rel)

    assert best_state is not None
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": vars(args),
            "proto_centers": proto_centers,
            "context_cache_path": str(context_cache_path),
            "target_m": target_m,
            "point_strategy": point_strategy,
        },
        ROOT / final_rel,
    )
    torch.save(best_state, ROOT / best_rel)
    model.load_state_dict(best_state["model_state_dict"])

    train_all, train_ds, train_sub, train_rows, train_calib = evaluate_model(model, rows["train"], ctx_map, args.batch_size, device)
    val_all, val_ds, val_sub, val_rows, val_calib = evaluate_model(model, rows["val"], ctx_map, args.batch_size, device)
    test_all, test_ds, test_sub, test_rows, test_calib = evaluate_model(model, rows["test"], ctx_map, args.batch_size, device)
    loss_history.append({"best_step": best_step, "best_val_score": best_val_score})

    report = {
        "audit_name": "stwm_ostf_v22_run",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "started_at_utc": started.isoformat(),
        "duration_sec": float(time.time() - wall_start),
        "experiment_name": args.experiment_name,
        "model_kind": args.model_kind,
        "horizon": args.horizon,
        "seed": args.seed,
        "source_combo": combo,
        "target_m": target_m or rows["train"][0].m,
        "point_selection_strategy": point_strategy,
        "steps": args.steps,
        "parameter_count": param_count,
        "checkpoint_path": final_rel,
        "best_checkpoint_path": best_rel,
        "context_cache_path": str(context_cache_path.relative_to(ROOT) if context_cache_path.is_absolute() else context_cache_path),
        "init_from_checkpoint": args.init_from_checkpoint,
        "loaded_init_key_count": len(loaded_keys),
        "best_step": best_step,
        "best_val_score": best_val_score,
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
        "train_calibration": train_calib,
        "val_calibration": val_calib,
        "test_calibration": test_calib,
        "item_scores": test_rows,
        "metric_note": "V22 uses corrected semantic inputs only from observed semantic memory/context. Multimodal trajectory metrics are primary evidence.",
    }
    out_path = ROOT / f"reports/stwm_ostf_v22_runs/{args.experiment_name}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dump_json(out_path, report)
    print(out_path.relative_to(ROOT), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
