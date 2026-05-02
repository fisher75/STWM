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

from stwm.modules.ostf_physics_residual_world_model_v18 import OSTFPhysicsResidualConfig, OSTFPhysicsResidualWorldModel
from stwm.tools.ostf_v18_common_20260502 import (
    ROOT,
    analytic_affine_motion_predict,
    analytic_constant_velocity_predict,
    batch_from_samples,
    build_v18_rows,
    dump_json,
    eval_metrics_extended,
    item_scores_from_predictions,
    loss_endpoint,
    loss_extent,
    loss_point_valid,
    loss_temporal_smoothness,
    metrics_by_dataset,
    set_seed,
)


def _apply_process_title() -> None:
    try:
        import setproctitle  # type: ignore

        setproctitle.setproctitle(str(os.environ.get("STWM_PROC_TITLE", "python")))
    except Exception:
        pass


def _combo_for_experiment(kind: str, horizon: int) -> str:
    if "m128" in kind:
        return f"M128_H{horizon}"
    return f"M512_H{horizon}"


def _iter_batches(samples: list[Any], batch_size: int, *, shuffle: bool, seed: int):
    idx = list(range(len(samples)))
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(idx)
    for start in range(0, len(idx), batch_size):
        yield [samples[i] for i in idx[start : start + batch_size]]


def _build_model(kind: str, horizon: int) -> OSTFPhysicsResidualWorldModel:
    cfg = OSTFPhysicsResidualConfig(
        horizon=horizon,
        hidden_dim=256,
        point_dim=128,
        num_layers=4,
        num_heads=8,
        use_semantic_memory=kind not in {"v18_wo_semantic_memory"},
        use_dense_points=kind not in {"v18_wo_dense_points"},
        use_residual_decoder=kind not in {"v18_wo_residual_decoder", "affine_motion_prior_only"},
        use_affine_prior=kind not in {"v18_wo_affine_prior", "dct_residual_prior_only"},
        use_cv_prior=kind not in {"v18_wo_cv_prior"},
    )
    return OSTFPhysicsResidualWorldModel(cfg)


def _evaluate_learned(model: OSTFPhysicsResidualWorldModel, samples: list[Any], batch_size: int, device: torch.device) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
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


def _evaluate_analytic(kind: str, samples: list[Any]) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    if kind == "constant_velocity_copy":
        pred_points, pred_vis_logits, pred_sem_logits = analytic_constant_velocity_predict(samples)
    elif kind == "affine_motion_prior_only":
        pred_points, pred_vis_logits, pred_sem_logits = analytic_affine_motion_predict(samples)
    else:
        raise KeyError(kind)
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


def _val_score(metrics: dict[str, Any]) -> float:
    return -float(metrics["point_L1_px"]) + 100.0 * float(metrics["object_extent_iou"]) + 10.0 * float(metrics["PCK_16px"])


def main() -> int:
    _apply_process_title()
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument(
        "--model-kind",
        required=True,
        choices=[
            "constant_velocity_copy",
            "affine_motion_prior_only",
            "dct_residual_prior_only",
            "v18_physics_residual_m128",
            "v18_physics_residual_m512",
            "v18_wo_semantic_memory",
            "v18_wo_dense_points",
            "v18_wo_residual_decoder",
            "v18_wo_affine_prior",
            "v18_wo_cv_prior",
        ],
    )
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--eval-every", type=int, default=1000)
    args = parser.parse_args()

    started = datetime.now(timezone.utc)
    wall_start = time.time()
    set_seed(args.seed)
    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    combo = _combo_for_experiment(args.model_kind, args.horizon)
    rows, proto_centers = build_v18_rows(combo, seed=args.seed)
    print(
        f"[V18][start] exp={args.experiment_name} kind={args.model_kind} combo={combo} h={args.horizon} seed={args.seed} steps={args.steps} batch={args.batch_size} device={device}",
        flush=True,
    )
    print(
        f"[V18][data] train={len(rows['train'])} val={len(rows['val'])} test={len(rows['test'])}",
        flush=True,
    )

    if args.model_kind in {"constant_velocity_copy", "affine_motion_prior_only"}:
        train_metrics, train_by_ds, _ = _evaluate_analytic(args.model_kind, rows["train"])
        val_metrics, val_by_ds, _ = _evaluate_analytic(args.model_kind, rows["val"])
        test_metrics, test_by_ds, item_scores = _evaluate_analytic(args.model_kind, rows["test"])
        ckpt_rel = None
        best_ckpt_rel = None
        loss_history: list[dict[str, Any]] = []
        best_val_score = _val_score(val_metrics)
        param_count = 0
    else:
        model = _build_model(args.model_kind, args.horizon).to(device)
        param_count = int(sum(p.numel() for p in model.parameters()))
        print(f"[V18][model] params={param_count}", flush=True)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scaler = GradScaler(device="cuda", enabled=device.type == "cuda")
        best_state = None
        best_val_score = -1e18
        best_step = 0
        loss_history = []
        ckpt_dir = ROOT / "outputs/checkpoints/stwm_ostf_v18"
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
                point_loss = loss_point_valid(out["point_pred"], batch["fut_points"], batch["fut_vis"])
                endpoint_loss = loss_endpoint(out["point_pred"], batch["fut_points"], batch["fut_vis"])
                anchor_loss = F.smooth_l1_loss(out["anchor_pred"], batch["anchor_fut"])
                extent_loss = loss_extent(out["point_pred"], batch["fut_points"], batch["fut_vis"])
                vis_loss = F.binary_cross_entropy_with_logits(out["visibility_logits"], batch["fut_vis"].float())
                proto_logits = out["semantic_logits"].reshape(-1, out["semantic_logits"].shape[-1])
                proto_target = batch["proto_target"][:, None].expand(-1, out["semantic_logits"].shape[1]).reshape(-1)
                sem_loss = F.cross_entropy(proto_logits, proto_target)
                smooth_loss = loss_temporal_smoothness(out["point_pred"])
                residual_reg = out["residual"].abs().mean()
                total = (
                    point_loss
                    + 0.5 * endpoint_loss
                    + 0.4 * anchor_loss
                    + 0.25 * extent_loss
                    + 0.2 * vis_loss
                    + 0.15 * sem_loss
                    + 0.05 * smooth_loss
                    + 0.02 * residual_reg
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
                    "anchor": float(anchor_loss.detach().cpu()),
                    "extent": float(extent_loss.detach().cpu()),
                    "visibility": float(vis_loss.detach().cpu()),
                    "semantic": float(sem_loss.detach().cpu()),
                    "smooth": float(smooth_loss.detach().cpu()),
                    "residual_reg": float(residual_reg.detach().cpu()),
                }
                loss_history.append(row)
                print(
                    "[V18][train] "
                    + " ".join(f"{k}={v:.6f}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()),
                    flush=True,
                )
            if step % args.eval_every == 0 or step == args.steps:
                val_metrics_tmp, _, _ = _evaluate_learned(model, rows["val"], args.batch_size, device)
                score = _val_score(val_metrics_tmp)
                print(
                    f"[V18][val] step={step} point={val_metrics_tmp['point_L1_px']:.4f} "
                    f"extent={val_metrics_tmp['object_extent_iou']:.4f} pck16={val_metrics_tmp['PCK_16px']:.4f} score={score:.4f}",
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
        final_rel = f"outputs/checkpoints/stwm_ostf_v18/{args.experiment_name}_final.pt"
        best_rel = f"outputs/checkpoints/stwm_ostf_v18/{args.experiment_name}_best.pt"
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
        ckpt_rel = final_rel
        best_ckpt_rel = best_rel
        train_metrics, train_by_ds, _ = _evaluate_learned(model, rows["train"], args.batch_size, device)
        val_metrics, val_by_ds, _ = _evaluate_learned(model, rows["val"], args.batch_size, device)
        test_metrics, test_by_ds, item_scores = _evaluate_learned(model, rows["test"], args.batch_size, device)
        loss_history.append({"best_step": best_step, "best_val_score": best_val_score})

    report = {
        "audit_name": "stwm_ostf_v18_run",
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
        "checkpoint_path": ckpt_rel,
        "best_checkpoint_path": best_ckpt_rel,
        "best_val_score": float(best_val_score),
        "loss_history": loss_history,
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "train_metrics_by_dataset": train_by_ds,
        "val_metrics_by_dataset": val_by_ds,
        "test_metrics_by_dataset": test_by_ds,
        "item_scores": item_scores,
        "metric_note": "V18 reports point metrics, endpoint, extent IoU, visibility F1, and semantic top-k separately. Dense M512 point metrics are not treated as directly identical to sparse M1 anchor metrics.",
    }
    out_path = ROOT / "reports/stwm_ostf_v18_runs" / f"{args.experiment_name}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dump_json(out_path, report)
    print(out_path.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
