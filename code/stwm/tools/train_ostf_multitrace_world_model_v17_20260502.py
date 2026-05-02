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
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))

from stwm.modules.ostf_multitrace_world_model_v17 import (
    OSTFMultiTraceWorldModel,
    OSTFMultiTraceWorldModelConfig,
    PointMemoryTransformer,
)
from stwm.tools.ostf_v17_common_20260502 import (
    ROOT,
    assign_semantic_prototypes,
    batch_from_samples,
    collapse_to_m1,
    dump_json,
    eval_metrics_from_predictions,
    iter_batches,
    kmeans_semantic_prototypes,
    load_v16_samples,
    set_seed,
)


def _apply_process_title() -> None:
    try:
        import setproctitle  # type: ignore

        setproctitle.setproctitle(str(os.environ.get("STWM_PROC_TITLE", "python")))
    except Exception:
        pass


def _build_rows(model_kind: str, horizon: int) -> tuple[dict[str, list[Any]], int]:
    if model_kind == "m1_anchor_stwm":
        rows = collapse_to_m1(load_v16_samples(f"M512_H{horizon}"))
        return rows, 1
    if model_kind == "m1_anchor_stwm_m128":
        rows = collapse_to_m1(load_v16_samples(f"M128_H{horizon}"))
        return rows, 1
    if model_kind in {"ostf_multitrace_m128"}:
        return load_v16_samples(f"M128_H{horizon}"), 128
    return load_v16_samples(f"M512_H{horizon}"), 512


def _source_combo(model_kind: str, horizon: int) -> str:
    if model_kind in {"m1_anchor_stwm_m128", "ostf_multitrace_m128"}:
        return f"M128_H{horizon}"
    return f"M512_H{horizon}"


def _build_model(model_kind: str, horizon: int, proto_count: int) -> torch.nn.Module:
    if model_kind == "point_transformer_dense":
        return PointMemoryTransformer(
            OSTFMultiTraceWorldModelConfig(
                obs_len=8,
                horizon=horizon,
                hidden_dim=256,
                point_dim=128,
                num_layers=2,
                num_heads=4,
                prototype_count=proto_count,
                use_semantic_memory=False,
                use_dense_point_input=True,
                use_point_residual_decoder=True,
                use_semantic_unit_compression=False,
            )
        )
    cfg = OSTFMultiTraceWorldModelConfig(
        obs_len=8,
        horizon=horizon,
        hidden_dim=256,
        point_dim=128,
        num_layers=4,
        num_heads=8,
        prototype_count=proto_count,
        use_semantic_memory=model_kind not in {"ostf_m512_wo_semantic_memory"},
        use_dense_point_input=model_kind not in {"ostf_m512_wo_dense_point_input"},
        use_point_residual_decoder=model_kind not in {"ostf_m512_wo_point_residual_decoder", "m1_anchor_stwm"},
        use_semantic_unit_compression=model_kind not in {"point_transformer_dense"},
    )
    return OSTFMultiTraceWorldModel(cfg)


def _constant_velocity_eval(rows: list[Any]) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    point_preds = []
    vis_logits = []
    sem_logits = []
    gt_points = []
    gt_vis = []
    gt_anchor = []
    proto_target = []
    item_scores = []
    for s in rows:
        vel = np.zeros_like(s.anchor_obs)
        vel[1:] = s.anchor_obs[1:] - s.anchor_obs[:-1]
        last = s.anchor_obs[-1]
        last_vel = vel[-1]
        anchor_pred = np.stack([last + last_vel * float(t + 1) for t in range(s.h)], axis=0)
        pred_points = s.obs_points[:, -1:, :].repeat(s.h, axis=1)
        pred_points = pred_points + np.stack([last_vel * float(t + 1) for t in range(s.h)], axis=0)[None]
        point_preds.append(pred_points)
        vis_logits.append(np.where(np.repeat(s.obs_vis[:, -1:], s.h, axis=1), 5.0, -5.0))
        sem_logits.append(np.zeros((s.h, 32), dtype=np.float32))
        gt_points.append(s.fut_points)
        gt_vis.append(s.fut_vis)
        gt_anchor.append(s.anchor_fut)
        proto_target.append(s.proto_target)
        err = np.abs(pred_points - s.fut_points).sum(axis=-1) * 1000.0
        item_scores.append(
            {
                "item_key": s.item_key,
                "object_index": s.object_index,
                "object_id": s.object_id,
                "point_l1_px": float(err[s.fut_vis].mean()) if np.any(s.fut_vis) else 0.0,
                "anchor_l1_px": float((np.abs(pred_points.mean(axis=0) - s.anchor_fut).sum(axis=-1) * 1000.0).mean()),
            }
        )
    metrics = eval_metrics_from_predictions(
        pred_points=np.stack(point_preds),
        pred_vis_logits=np.stack(vis_logits),
        pred_proto_logits=np.stack(sem_logits),
        gt_points=np.stack(gt_points),
        gt_vis=np.stack(gt_vis),
        gt_anchor=np.stack(gt_anchor),
        proto_target=np.asarray(proto_target),
    )
    return metrics, {"item_scores": np.asarray(item_scores, dtype=object)}


def _evaluate(model: torch.nn.Module, rows: list[Any], batch_size: int, device: torch.device) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    model.eval()
    point_preds = []
    vis_logits = []
    sem_logits = []
    gt_points = []
    gt_vis = []
    gt_anchor = []
    proto_target = []
    item_scores: list[dict[str, Any]] = []
    with torch.no_grad():
        for batch_rows in iter_batches(rows, batch_size, shuffle=False, seed=0):
            batch = batch_from_samples(batch_rows, device)
            out = model(
                obs_points=batch["obs_points"],
                obs_vis=batch["obs_vis"],
                rel_xy=batch["rel_xy"],
                anchor_obs=batch["anchor_obs"],
                anchor_obs_vel=batch["anchor_obs_vel"],
                semantic_feat=batch["semantic_feat"],
            )
            pp = out["point_pred"].detach().cpu().numpy()
            pv = out["visibility_logits"].detach().cpu().numpy()
            ps = out["semantic_logits"].detach().cpu().numpy()
            fg = batch["fut_points"].cpu().numpy()
            fv = batch["fut_vis"].cpu().numpy()
            ga = batch["anchor_fut"].cpu().numpy()
            pt = batch["proto_target"].cpu().numpy()
            point_preds.append(pp)
            vis_logits.append(pv)
            sem_logits.append(ps)
            gt_points.append(fg)
            gt_vis.append(fv)
            gt_anchor.append(ga)
            proto_target.append(pt)
            err = np.abs(pp - fg).sum(axis=-1) * 1000.0
            for row, pp_i, e in zip(batch_rows, pp, err):
                item_scores.append(
                    {
                        "item_key": row.item_key,
                        "object_index": row.object_index,
                        "object_id": row.object_id,
                        "dataset": row.dataset,
                        "point_l1_px": float(e[row.fut_vis].mean()) if np.any(row.fut_vis) else 0.0,
                        "anchor_l1_px": float((np.abs(pp_i.mean(axis=0) - row.anchor_fut).sum(axis=-1) * 1000.0).mean()),
                    }
                )
    metrics = eval_metrics_from_predictions(
        pred_points=np.concatenate(point_preds, axis=0),
        pred_vis_logits=np.concatenate(vis_logits, axis=0),
        pred_proto_logits=np.concatenate(sem_logits, axis=0),
        gt_points=np.concatenate(gt_points, axis=0),
        gt_vis=np.concatenate(gt_vis, axis=0),
        gt_anchor=np.concatenate(gt_anchor, axis=0),
        proto_target=np.concatenate(proto_target, axis=0),
    )
    return metrics, item_scores


def main() -> int:
    _apply_process_title()
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument(
        "--model-kind",
        required=True,
        choices=[
            "m1_anchor_stwm",
            "m1_anchor_stwm_m128",
            "constant_velocity_copy",
            "point_transformer_dense",
            "ostf_multitrace_m128",
            "ostf_multitrace_m512",
            "ostf_m512_wo_semantic_memory",
            "ostf_m512_wo_dense_point_input",
            "ostf_m512_wo_point_residual_decoder",
        ],
    )
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    started_at = datetime.now(timezone.utc)
    wall_start = time.time()
    set_seed(args.seed)
    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    print(
        f"[V17][start] experiment={args.experiment_name} model={args.model_kind} "
        f"h={args.horizon} seed={args.seed} steps={args.steps} batch={args.batch_size} device={device}",
        flush=True,
    )
    rows, effective_m = _build_rows(args.model_kind, args.horizon)
    print(
        f"[V17][data] train={len(rows['train'])} val={len(rows['val'])} test={len(rows['test'])} effective_M={effective_m}",
        flush=True,
    )
    centers = kmeans_semantic_prototypes(rows["train"], k=32, iters=25, seed=args.seed)
    assign_semantic_prototypes(rows, centers)
    if args.model_kind == "constant_velocity_copy":
        train_metrics, _ = _constant_velocity_eval(rows["train"])
        val_metrics, _ = _constant_velocity_eval(rows["val"])
        test_metrics, item_scores = _constant_velocity_eval(rows["test"])
        ckpt_path = None
        loss_curve = []
    else:
        model = _build_model(args.model_kind, args.horizon, proto_count=32).to(device)
        parameter_count = int(sum(p.numel() for p in model.parameters()))
        trainable_parameter_count = int(sum(p.numel() for p in model.parameters() if p.requires_grad))
        print(
            f"[V17][model] params={parameter_count} trainable={trainable_parameter_count}",
            flush=True,
        )
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        loss_curve = []
        train_rows = rows["train"]
        for step in range(args.steps):
            batch_rows = random.sample(train_rows, k=min(args.batch_size, len(train_rows)))
            batch = batch_from_samples(batch_rows, device)
            out = model(
                obs_points=batch["obs_points"],
                obs_vis=batch["obs_vis"],
                rel_xy=batch["rel_xy"],
                anchor_obs=batch["anchor_obs"],
                anchor_obs_vel=batch["anchor_obs_vel"],
                semantic_feat=batch["semantic_feat"],
            )
            point_loss = F.smooth_l1_loss(out["point_pred"][batch["fut_vis"]], batch["fut_points"][batch["fut_vis"]])
            anchor_loss = F.smooth_l1_loss(out["anchor_pred"], batch["anchor_fut"])
            vis_loss = F.binary_cross_entropy_with_logits(out["visibility_logits"], batch["fut_vis"].float())
            proto_logits = out["semantic_logits"].reshape(-1, out["semantic_logits"].shape[-1])
            proto_target = batch["proto_target"][:, None].expand(-1, out["semantic_logits"].shape[1]).reshape(-1)
            sem_loss = F.cross_entropy(proto_logits, proto_target)
            loss = point_loss + 0.5 * anchor_loss + 0.2 * vis_loss + 0.1 * sem_loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if step % 100 == 0 or step == args.steps - 1:
                loss_value = float(loss.detach().cpu())
                loss_curve.append(loss_value)
                print(
                    f"[V17][train] experiment={args.experiment_name} step={step + 1}/{args.steps} loss={loss_value:.6f} "
                    f"point={float(point_loss.detach().cpu()):.6f} anchor={float(anchor_loss.detach().cpu()):.6f} "
                    f"vis={float(vis_loss.detach().cpu()):.6f} sem={float(sem_loss.detach().cpu()):.6f}",
                    flush=True,
                )
        ckpt_dir = ROOT / "outputs/checkpoints/stwm_ostf_v17"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = ckpt_dir / f"{args.experiment_name}.pt"
        torch.save({"model_state_dict": model.state_dict(), "config": vars(args), "proto_centers": centers}, ckpt_path)
        train_metrics, _ = _evaluate(model, rows["train"], args.batch_size, device)
        val_metrics, _ = _evaluate(model, rows["val"], args.batch_size, device)
        test_metrics, item_scores = _evaluate(model, rows["test"], args.batch_size, device)
    ended_at = datetime.now(timezone.utc)
    duration_sec = float(time.time() - wall_start)
    report = {
        "audit_name": "stwm_ostf_v17_run",
        "generated_at_utc": ended_at.isoformat(),
        "started_at_utc": started_at.isoformat(),
        "ended_at_utc": ended_at.isoformat(),
        "duration_sec": duration_sec,
        "experiment_name": args.experiment_name,
        "model_kind": args.model_kind,
        "effective_M": effective_m,
        "horizon": args.horizon,
        "seed": args.seed,
        "steps": args.steps,
        "device": str(device),
        "source_combo": _source_combo(args.model_kind, args.horizon),
        "checkpoint_path": str(ckpt_path.relative_to(ROOT)) if ckpt_path else None,
        "prototype_count": 32,
        "loss_curve": loss_curve,
        "parameter_count": locals().get("parameter_count", 0),
        "trainable_parameter_count": locals().get("trainable_parameter_count", 0),
        "split_object_counts": {k: len(v) for k, v in rows.items()},
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "item_scores": item_scores,
        "metric_note": "Anchor/centroid and dense point metrics are reported separately; M1 anchor is not treated as directly comparable to M512 dense point L1 as the sole criterion.",
    }
    out_path = ROOT / "reports/stwm_ostf_v17_runs" / f"{args.experiment_name}.json"
    dump_json(out_path, report)
    print(
        f"[V17][done] experiment={args.experiment_name} duration_sec={duration_sec:.2f} "
        f"checkpoint={str(ckpt_path.relative_to(ROOT)) if ckpt_path else 'none'} report={out_path.relative_to(ROOT)}",
        flush=True,
    )
    print(out_path.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
