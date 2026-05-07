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

from stwm.modules.ostf_lastobs_residual_world_model_v28 import (
    OSTFLastObservedResidualConfigV28,
    OSTFLastObservedResidualWorldModelV28,
)
from stwm.tools.ostf_lastobs_v28_common_20260502 import (
    ROOT,
    add_v28_flags_to_item_rows,
    batch_from_samples_v26,
    build_v28_rows,
    v28_subset_aggregate,
)
from stwm.tools.ostf_traceanything_metrics_v26_20260502 import (
    aggregate_item_rows_v26,
    hypothesis_diversity_valid_v26,
    multimodal_item_scores_v26,
)
from stwm.tools.ostf_traceanything_common_v26_20260502 import set_seed
from stwm.tools.train_ostf_traceanything_v26_20260502 import (
    _apply_process_title,
    _extent_center,
    _load_init_checkpoint,
    _losses as _v26_losses,
    _mode_selection,
    _weighted_mean_per_sample,
)


def _combo_for_model(kind: str, horizon: int) -> str:
    return f"M128_H{horizon}" if "m128" in kind else f"M512_H{horizon}"


def _build_model(kind: str, horizon: int, damped_gamma: float) -> OSTFLastObservedResidualWorldModelV28:
    hidden_dim = 384 if "m128" in kind else 448
    point_dim = 224 if "m128" in kind else 256
    cfg = OSTFLastObservedResidualConfigV28(
        horizon=horizon,
        hidden_dim=hidden_dim,
        point_dim=point_dim,
        num_layers=4 if horizon <= 32 else 5,
        num_heads=8,
        refinement_layers=2 if "m128" in kind else 3,
        num_hypotheses=1 if "prior_only" in kind else 6,
        damped_gamma=float(damped_gamma),
        use_dense_points="wo_dense_points" not in kind,
        use_semantic_memory="wo_semantic_memory" not in kind,
        use_context=True,
        use_residual_modes=("wo_residual_modes" not in kind and "prior_only" not in kind),
        use_damped_prior="prior_only" not in kind,
        use_cv_prior="prior_only" not in kind,
        predict_variance=False,
    )
    return OSTFLastObservedResidualWorldModelV28(cfg)


def _losses_v28(out: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    total, comps = _v26_losses(out, batch)
    point_hyp = torch.nan_to_num(out["point_hypotheses"], nan=0.0, posinf=1.25, neginf=-0.25)
    num_prior = int(float(out.get("num_prior_modes", point_hyp.new_ones(1)).detach().flatten()[0].cpu()))
    sample_weight = 1.0 + 0.70 * batch["hardness_score"] + 0.35 * batch["occlusion_ratio"] + 0.25 * batch["reappearance_flag"]
    sample_weight = sample_weight.clamp(1.0, 5.0)
    fut_points = batch["fut_points"]
    fut_vis = batch["fut_vis"]
    fut_conf = batch["fut_conf"]
    weight = fut_vis.float() * (0.5 + fut_conf.float())
    learned_aux = point_hyp.new_zeros(())
    learned_endpoint_aux = point_hyp.new_zeros(())
    if point_hyp.shape[3] > num_prior:
        learned = point_hyp[:, :, :, num_prior:]
        gt = fut_points[:, :, :, None, :].expand_as(learned)
        diff = F.smooth_l1_loss(learned, gt, reduction="none").sum(dim=-1)
        denom = weight.sum(dim=(1, 2)).clamp_min(1.0)
        learned_cost = (diff * weight[:, :, :, None]).sum(dim=(1, 2)) / denom[:, None]
        learned_min = learned_cost.min(dim=-1).values
        end_diff = F.smooth_l1_loss(learned[:, :, -1], fut_points[:, :, -1, None, :].expand_as(learned[:, :, -1]), reduction="none").sum(dim=-1)
        end_w = fut_vis[:, :, -1].float() * (0.5 + fut_conf[:, :, -1].float())
        learned_end_cost = (end_diff * end_w[:, :, None]).sum(dim=1) / end_w.sum(dim=1).clamp_min(1.0)[:, None]
        learned_end_min = learned_end_cost.min(dim=-1).values
        learned_aux = _weighted_mean_per_sample(learned_min, sample_weight)
        learned_endpoint_aux = _weighted_mean_per_sample(learned_end_min, sample_weight)
        total = total + 0.65 * learned_aux + 0.55 * learned_endpoint_aux
    comps["learned_residual_aux"] = learned_aux
    comps["learned_residual_endpoint_aux"] = learned_endpoint_aux
    return total, comps


def _subset_aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return v28_subset_aggregate(rows)


def _evaluate_model(
    model: OSTFLastObservedResidualWorldModelV28,
    samples: list[Any],
    batch_size: int,
    device: torch.device,
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
    rows = multimodal_item_scores_v26(
        samples,
        point_modes=np.concatenate(point_modes, axis=0),
        mode_logits=np.concatenate(mode_logits, axis=0),
        top1_point_pred=np.concatenate(top1_preds, axis=0),
        weighted_point_pred=np.concatenate(weighted_preds, axis=0),
        pred_vis_logits=np.concatenate(vis_logits, axis=0),
        pred_proto_logits=np.concatenate(sem_logits, axis=0),
        pred_logvar=np.concatenate(logvars, axis=0) if logvars else None,
        cv_mode_index=2,
    )
    rows = add_v28_flags_to_item_rows(rows, samples)
    all_metrics = aggregate_item_rows_v26(rows)
    by_dataset = {ds: aggregate_item_rows_v26(rows, dataset=ds) for ds in sorted({s.dataset for s in samples})}
    subsets = _subset_aggregate(rows)
    return all_metrics, by_dataset, subsets, rows, hypothesis_diversity_valid_v26(rows)


def _score(metrics: dict[str, Any], subset: dict[str, Any]) -> float:
    def f(x: Any, bad: float) -> float:
        try:
            v = float(x)
        except Exception:
            return bad
        return v if np.isfinite(v) else bad

    return (
        -1.65 * f(subset.get("minFDE_K_px"), 1e6)
        -0.70 * f(subset.get("minADE_K_px"), 1e6)
        -0.22 * f(metrics.get("minFDE_K_px"), 1e6)
        +22.0 * f(subset.get("BestOfK_PCK_32px"), 0.0)
        +12.0 * f(subset.get("BestOfK_PCK_16px"), 0.0)
        -14.0 * f(subset.get("MissRate_32px"), 1.0)
        +7.0 * f(metrics.get("semantic_top5"), 0.0)
        +5.0 * f(metrics.get("top1_visibility_F1"), 0.0)
    )


def main() -> int:
    _apply_process_title()
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument(
        "--model-kind",
        required=True,
        choices=[
            "v28_lastobs_m128_h32",
            "v28_lastobs_m128_h32_wo_dense_points",
            "v28_lastobs_m128_h32_wo_semantic_memory",
            "v28_lastobs_m128_h32_wo_residual_modes",
            "v28_lastobs_m128_h32_prior_only",
            "v28_lastobs_m512_h32",
            "v28_lastobs_m128_h64",
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
    rows, proto_centers, damped_gamma, subset_summary = build_v28_rows(combo, seed=args.seed)
    if not rows["train"] or not rows["val"] or not rows["test"]:
        raise SystemExit(f"No usable TraceAnything V25 samples found for combo={combo}")
    model = _build_model(args.model_kind, args.horizon, damped_gamma).to(device)
    param_count = int(sum(p.numel() for p in model.parameters()))
    lr = 1.8e-4 if "m128" in args.model_kind else 1.3e-4
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=1e-4)
    scaler = GradScaler(device="cuda", enabled=device.type == "cuda")
    loaded_keys = []
    if args.init_from_checkpoint:
        loaded_keys = _load_init_checkpoint(model, args.init_from_checkpoint)

    train_weights = [float(max(1.0, s.hardness_score)) for s in rows["train"]]
    print(
        f"[V28][start] exp={args.experiment_name} kind={args.model_kind} combo={combo} h={args.horizon} "
        f"seed={args.seed} steps={args.steps} batch={args.batch_size} device={device} damped_gamma={damped_gamma}",
        flush=True,
    )
    print(f"[V28][data] train={len(rows['train'])} val={len(rows['val'])} test={len(rows['test'])}", flush=True)
    print(f"[V28][model] params={param_count} init_loaded={len(loaded_keys)}", flush=True)

    best_state = None
    best_val_score = -1e18
    best_step = 0
    loss_history = []
    ckpt_dir = ROOT / "outputs/checkpoints/stwm_ostf_v28"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    final_rel = f"outputs/checkpoints/stwm_ostf_v28/{args.experiment_name}_final.pt"
    best_rel = f"outputs/checkpoints/stwm_ostf_v28/{args.experiment_name}_best.pt"

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
            total, comps = _losses_v28(out, batch)
        if not torch.isfinite(total):
            print(f"[V28][warn] step={step} nonfinite_total_skip_batch", flush=True)
            opt.zero_grad(set_to_none=True)
            continue
        opt.zero_grad(set_to_none=True)
        scaler.scale(total).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.75 if "m512" in args.model_kind else 1.0)
        scaler.step(opt)
        scaler.update()

        if step == 1 or step % 500 == 0 or step == args.steps:
            safe_logits = torch.nan_to_num(out["hypothesis_logits"].detach().float(), nan=0.0, posinf=25.0, neginf=-25.0)
            row = {"step": step, "total": float(total.detach().cpu())}
            row.update({k: float(v.detach().cpu()) for k, v in comps.items()})
            row["residual_scale"] = float(out["residual_scale"].mean().detach().cpu())
            row["hyp_entropy"] = float(torch.distributions.Categorical(logits=safe_logits).entropy().mean().detach().cpu())
            loss_history.append(row)
            print("[V28][train] " + " ".join(f"{k}={v:.6f}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()), flush=True)

        if step % args.eval_every == 0 or step == args.steps:
            val_all, _, val_sub, _, val_div = _evaluate_model(model, rows["val"], args.batch_size, device)
            hard = val_sub.get("last_observed_hard_top20", {})
            score = _score(val_all, hard)
            print(
                f"[V28][val] step={step} top1_point={val_all.get('top1_point_L1_px'):.4f} minfde={val_all.get('minFDE_K_px'):.4f} "
                f"last_hard_minfde={hard.get('minFDE_K_px')} last_hard_miss32={hard.get('MissRate_32px')} div={int(val_div)} score={score:.4f}",
                flush=True,
            )
            if score > best_val_score:
                best_val_score = score
                best_step = step
                best_state = {
                    "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                    "config": vars(args),
                    "damped_gamma": float(damped_gamma),
                    "proto_centers": proto_centers,
                }
                torch.save(best_state, ROOT / best_rel)

    assert best_state is not None
    torch.save({"model_state_dict": model.state_dict(), "config": vars(args), "damped_gamma": float(damped_gamma), "proto_centers": proto_centers}, ROOT / final_rel)
    torch.save(best_state, ROOT / best_rel)
    model.load_state_dict(best_state["model_state_dict"])

    train_all, train_ds, train_sub, train_rows, train_div = _evaluate_model(model, rows["train"], args.batch_size, device)
    val_all, val_ds, val_sub, val_rows, val_div = _evaluate_model(model, rows["val"], args.batch_size, device)
    test_all, test_ds, test_sub, test_rows, test_div = _evaluate_model(model, rows["test"], args.batch_size, device)
    loss_history.append({"best_step": best_step, "best_val_score": best_val_score})

    report = {
        "audit_name": "stwm_ostf_v28_run",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "started_at_utc": started.isoformat(),
        "duration_sec": float(time.time() - wall_start),
        "experiment_name": args.experiment_name,
        "model_kind": args.model_kind,
        "source_combo": combo,
        "damped_gamma": float(damped_gamma),
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
        "strongest_prior_used": "last_observed_copy",
        "subset_summary": subset_summary,
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
    out_path = ROOT / "reports/stwm_ostf_v28_runs" / f"{args.experiment_name}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(__import__("json").dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"[V28][done] report={out_path} best={best_rel} final={final_rel}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
