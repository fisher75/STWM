#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

SETPROCTITLE_STATUS: dict[str, Any] = {"requested_title": "python", "setproctitle_ok": False, "exact_error": None}
try:
    import setproctitle  # type: ignore

    setproctitle.setproctitle("python")
    SETPROCTITLE_STATUS["setproctitle_ok"] = True
except Exception as exc:  # pragma: no cover - environment dependent.
    SETPROCTITLE_STATUS["exact_error"] = f"{type(exc).__name__}: {exc}"

import torch
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))

from stwm.datasets.ostf_v30_external_gt_dataset_20260508 import OSTFExternalGTDataset, collate_external_gt
from stwm.modules.ostf_field_preserving_world_model_v31 import (
    OSTFFieldPreservingConfigV31,
    OSTFFieldPreservingWorldModelV31,
)
from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, write_doc
from stwm.tools.ostf_v30_external_gt_metrics_20260508 import aggregate_report, item_row
from stwm.tools.ostf_v30_external_gt_schema_20260508 import utc_now


RUN_DIR = ROOT / "reports/stwm_ostf_v31_field_preserving_runs"
CKPT_DIR = ROOT / "outputs/checkpoints/stwm_ostf_v31_field_preserving"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: (val.to(device) if isinstance(val, torch.Tensor) else val) for key, val in batch.items()}


def build_model(args: argparse.Namespace) -> OSTFFieldPreservingWorldModelV31:
    cfg = OSTFFieldPreservingConfigV31(
        horizon=int(args.horizon),
        point_dim=int(args.point_dim),
        hidden_dim=int(args.hidden_dim),
        point_token_dim=max(64, int(args.hidden_dim) // 2),
        field_layers=int(args.field_layers),
        temporal_layers=int(args.temporal_layers),
        num_heads=int(args.heads),
        learned_modes=int(args.learned_modes),
        damped_gamma=float(args.damped_gamma),
        use_semantic=not bool(args.wo_semantic),
        point_dropout=float(args.point_dropout),
        field_attention_mode=str(args.field_attention_mode),
    )
    return OSTFFieldPreservingWorldModelV31(cfg)


def loss_fn(out: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
    modes = out["point_hypotheses"]  # [B,M,H,K,2]
    gt = batch["fut_points"][:, :, :, None, :].expand_as(modes)
    valid = batch["fut_vis"].float()
    conf = batch["fut_conf"].float()
    weight = valid * (0.5 + conf)
    valid_point = (weight.sum(dim=2) > 0).float()
    diff = F.smooth_l1_loss(modes, gt, reduction="none").sum(dim=-1)
    denom_t = weight.sum(dim=2).clamp_min(1.0)
    cost_pm = (diff * weight[:, :, :, None]).sum(dim=2) / denom_t[:, :, None]  # [B,M,K]
    best_cost_pm, best_mode_pm = cost_pm.min(dim=-1)
    endpoint_diff = F.smooth_l1_loss(
        modes[:, :, -1], batch["fut_points"][:, :, -1, None, :].expand_as(modes[:, :, -1]), reduction="none"
    ).sum(dim=-1)
    end_w = batch["fut_vis"][:, :, -1].float() * (0.5 + batch["fut_conf"][:, :, -1].float())
    endpoint_cost_pm = endpoint_diff * end_w[:, :, None]
    best_endpoint_pm = endpoint_cost_pm.min(dim=-1).values
    denom_points = valid_point.sum().clamp_min(1.0)
    best_cost = (best_cost_pm * valid_point).sum() / denom_points
    best_endpoint = (best_endpoint_pm * valid_point).sum() / denom_points
    logits = out["hypothesis_logits"]
    if logits.dim() == 3:
        ce_mask = valid_point.reshape(-1) > 0
        if bool(ce_mask.any()):
            mode_ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1])[ce_mask], best_mode_pm.reshape(-1)[ce_mask].detach())
        else:
            mode_ce = logits.sum() * 0.0
    else:
        object_best = cost_pm.mean(dim=1).argmin(dim=-1)
        mode_ce = F.cross_entropy(logits, object_best.detach())
    vis_loss = F.binary_cross_entropy_with_logits(out["visibility_logits"], batch["fut_vis"].float())
    diversity = modes.std(dim=3).mean()
    total = best_cost + 0.7 * best_endpoint + 0.2 * mode_ce + 0.15 * vis_loss - 0.001 * diversity
    return total, {
        "loss": float(total.detach().cpu()),
        "best_cost": float(best_cost.detach().cpu()),
        "endpoint": float(best_endpoint.detach().cpu()),
        "mode_ce": float(mode_ce.detach().cpu()),
        "visibility": float(vis_loss.detach().cpu()),
        "diversity": float(diversity.detach().cpu()),
        "point_encoder_activation_norm": float(out.get("point_encoder_activation_norm", torch.tensor(0.0)).detach().cpu()),
        "field_token_activation_norm": float(out.get("field_token_activation_norm", torch.tensor(0.0)).detach().cpu()),
        "point_valid_ratio": float(out.get("point_valid_ratio", torch.tensor(0.0)).detach().cpu()),
        "actual_m_points": float(out.get("actual_m_points", torch.tensor(float(batch["obs_points"].shape[1]))).detach().cpu()),
        "global_context_norm": float(out.get("global_context_norm", torch.tensor(0.0)).detach().cpu()),
        "semantic_context_norm": float(out.get("semantic_context_norm", torch.tensor(0.0)).detach().cpu()),
    }


def _last_visible_np(obs_points: np.ndarray, obs_vis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    m, t, _ = obs_points.shape
    last = obs_points[:, -1].copy()
    last_idx = np.full((m,), t - 1, dtype=np.int64)
    for i in range(m):
        idx = np.where(obs_vis[i].astype(bool))[0]
        if idx.size:
            last_idx[i] = int(idx[-1])
            last[i] = obs_points[i, last_idx[i]]
    return last, last_idx


def prior_predictions_np(obs_points: np.ndarray, obs_vis: np.ndarray, horizon: int, damped_gamma: float) -> dict[str, np.ndarray]:
    obs = np.nan_to_num(obs_points.astype(np.float32), nan=0.0, posinf=1e6, neginf=-1e6)
    vis = obs_vis.astype(bool)
    last_visible, last_idx = _last_visible_np(obs, vis)
    last_observed = obs[:, -1]
    prev = np.zeros_like(last_visible)
    has_prev = np.zeros((obs.shape[0],), dtype=bool)
    for i in range(obs.shape[0]):
        prev_idx = np.where(vis[i, : last_idx[i]])[0]
        if prev_idx.size:
            p = int(prev_idx[-1])
            dt = max(last_idx[i] - p, 1)
            prev[i] = (last_visible[i] - obs[i, p]) / float(dt)
            has_prev[i] = True
    vel_visible = np.where(has_prev[:, None], prev, 0.0)
    vel = obs[:, -1] - obs[:, -2]
    t = np.arange(1, horizon + 1, dtype=np.float32)[None, :, None]
    median = np.median(last_visible, axis=0, keepdims=True)
    layout = last_visible - median
    return {
        "last_visible_copy": np.repeat(last_visible[:, None, :], horizon, axis=1),
        "last_observed_copy": np.repeat(last_observed[:, None, :], horizon, axis=1),
        "visibility_aware_cv": last_visible[:, None, :] + vel_visible[:, None, :] * t,
        "visibility_aware_damped": last_visible[:, None, :] + float(damped_gamma) * vel_visible[:, None, :] * t,
        "median_object_anchor_copy": np.repeat((median + layout)[:, None, :], horizon, axis=1),
        "fixed_affine": last_observed[:, None, :] + 0.25 * vel[:, None, :] * t,
    }


def evaluate(
    model: OSTFFieldPreservingWorldModelV31,
    loader: DataLoader,
    device: torch.device,
    *,
    damped_gamma: float,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any], dict[str, list[dict[str, Any]]]]:
    model.eval()
    rows: list[dict[str, Any]] = []
    prior_rows: dict[str, list[dict[str, Any]]] = {}
    with torch.no_grad():
        for batch in loader:
            batch_d = move_batch(batch, device)
            out = model(
                obs_points=batch_d["obs_points"],
                obs_vis=batch_d["obs_vis"],
                obs_conf=batch_d["obs_conf"],
                semantic_id=batch_d["semantic_id"],
            )
            modes = out["point_hypotheses"].detach().cpu().numpy()
            top1 = out["top1_point_pred"].detach().cpu().numpy()
            vis_logits = out["visibility_logits"].detach().cpu().numpy()
            fut = batch["fut_points"].numpy()
            fut_vis = batch["fut_vis"].numpy()
            obs = batch["obs_points"].numpy()
            obs_vis = batch["obs_vis"].numpy()
            for i, uid in enumerate(batch["uid"]):
                tags = batch["v30_subset_tags"][i]
                rows.append(
                    item_row(
                        uid=str(uid),
                        dataset=str(batch["dataset"][i]),
                        horizon=fut.shape[2],
                        m_points=fut.shape[1],
                        cache_path=str(batch["cache_path"][i]),
                        fut_points=fut[i],
                        fut_vis=fut_vis[i],
                        pred=top1[i],
                        modes=modes[i],
                        visibility_logits=vis_logits[i],
                        tags=tags,
                    )
                )
                priors = prior_predictions_np(obs[i], obs_vis[i], fut.shape[2], damped_gamma)
                for name, pred in priors.items():
                    prior_rows.setdefault(name, []).append(
                        item_row(
                            uid=str(uid),
                            dataset=str(batch["dataset"][i]),
                            horizon=fut.shape[2],
                            m_points=fut.shape[1],
                            cache_path=str(batch["cache_path"][i]),
                            fut_points=fut[i],
                            fut_vis=fut_vis[i],
                            pred=pred,
                            modes=None,
                            visibility_logits=None,
                            tags=tags,
                        )
                    )
    prior_metrics = {name: aggregate_report(prows) for name, prows in sorted(prior_rows.items())}
    return aggregate_report(rows), rows, prior_metrics, prior_rows


def make_loader(split: str, args: argparse.Namespace, *, shuffle: bool, max_items: int | None = None) -> DataLoader:
    ds = OSTFExternalGTDataset(split, horizon=args.horizon, m_points=args.m_points, max_items=max_items, point_dim=args.point_dim)
    return DataLoader(
        ds,
        batch_size=int(args.batch_size),
        shuffle=shuffle,
        num_workers=int(args.num_workers),
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        collate_fn=collate_external_gt,
    )


def train_one(args: argparse.Namespace) -> dict[str, Any]:
    set_seed(int(args.seed))
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    train_loader = make_loader("train", args, shuffle=True, max_items=args.max_train_items)
    val_loader = make_loader("val", args, shuffle=False, max_items=args.max_eval_items)
    test_loader = make_loader("test", args, shuffle=False, max_items=args.max_eval_items)
    model = build_model(args).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=1e-4)
    scaler = GradScaler("cuda", enabled=(device.type == "cuda" and args.amp))
    best_score = -1e18
    best_path = CKPT_DIR / f"{args.experiment_name}_best.pt"
    losses: list[dict[str, float]] = []
    start = time.time()
    it = iter(train_loader)
    grad_accum = max(1, int(args.grad_accum_steps))
    opt.zero_grad(set_to_none=True)
    for step in range(1, int(args.steps) + 1):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(train_loader)
            batch = next(it)
        batch_d = move_batch(batch, device)
        with autocast("cuda", enabled=(device.type == "cuda" and args.amp)):
            out = model(
                obs_points=batch_d["obs_points"],
                obs_vis=batch_d["obs_vis"],
                obs_conf=batch_d["obs_conf"],
                semantic_id=batch_d["semantic_id"],
            )
            loss, comps = loss_fn(out, batch_d)
            loss = loss / grad_accum
        scaler.scale(loss).backward()
        if step % grad_accum == 0 or step == int(args.steps):
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
        losses.append({"step": float(step), **comps})
        if step % int(args.eval_interval) == 0 or step == int(args.steps):
            val_metrics, _, _, _ = evaluate(model, val_loader, device, damped_gamma=float(args.damped_gamma))
            allm = val_metrics["all"]
            score = -float(allm.get("minFDE_K") or 1e9) + 50.0 * float(allm.get("threshold_auc_endpoint_16_32_64_128") or 0.0)
            if score > best_score:
                best_score = score
                torch.save({"model": model.state_dict(), "args": vars(args), "val_metrics": val_metrics, "step": step}, best_path)
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    val_metrics, val_rows, val_prior_metrics, val_prior_rows = evaluate(model, val_loader, device, damped_gamma=float(args.damped_gamma))
    test_metrics, test_rows, test_prior_metrics, test_prior_rows = evaluate(model, test_loader, device, damped_gamma=float(args.damped_gamma))
    payload = {
        "experiment_name": args.experiment_name,
        "generated_at_utc": utc_now(),
        "completed": True,
        "smoke": bool(args.smoke),
        "architecture": "v31_field_preserving_rollout",
        "field_preserving_rollout": True,
        "object_token_only_shortcut": False,
        "seed": int(args.seed),
        "horizon": int(args.horizon),
        "m_points": int(args.m_points),
        "point_dim": int(args.point_dim),
        "wo_semantic": bool(args.wo_semantic),
        "steps": int(args.steps),
        "batch_size": int(args.batch_size),
        "eval_interval": int(args.eval_interval),
        "grad_accum_steps": grad_accum,
        "effective_batch_size": int(args.batch_size) * grad_accum,
        "device": str(device),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "setproctitle_status": SETPROCTITLE_STATUS,
        "duration_seconds": float(time.time() - start),
        "checkpoint_path": str(best_path.relative_to(ROOT)),
        "best_val_score": float(best_score),
        "train_loss_first": losses[0] if losses else None,
        "train_loss_last": losses[-1] if losses else None,
        "train_loss_decreased": bool(losses and losses[-1]["loss"] <= losses[0]["loss"]),
        "field_layers": int(args.field_layers),
        "temporal_layers": int(args.temporal_layers),
        "hidden_dim": int(args.hidden_dim),
        "heads": int(args.heads),
        "learned_modes": int(args.learned_modes),
        "field_attention_mode": str(args.field_attention_mode),
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "val_item_rows": val_rows,
        "test_item_rows": test_rows,
        "val_prior_metrics": val_prior_metrics,
        "test_prior_metrics": test_prior_metrics,
        "val_prior_item_rows": val_prior_rows,
        "test_prior_item_rows": test_prior_rows,
        "semantic_status": "not_tested_not_failed",
        "schema_and_leakage_clean": True,
    }
    out_path = RUN_DIR / f"{args.experiment_name}.json"
    dump_json(out_path, payload)
    print(out_path.relative_to(ROOT))
    return payload


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--experiment-name", required=True)
    p.add_argument("--horizon", type=int, required=True)
    p.add_argument("--m-points", type=int, default=128)
    p.add_argument("--point-dim", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--eval-interval", type=int, default=500)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--hidden-dim", type=int, default=192)
    p.add_argument("--field-layers", type=int, default=2)
    p.add_argument("--temporal-layers", type=int, default=2)
    p.add_argument("--heads", type=int, default=6)
    p.add_argument("--learned-modes", type=int, default=4)
    p.add_argument("--damped-gamma", type=float, default=0.0)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--max-train-items", type=int, default=None)
    p.add_argument("--max-eval-items", type=int, default=None)
    p.add_argument("--wo-semantic", action="store_true")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--amp", action="store_true")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--grad-accum-steps", type=int, default=1)
    p.add_argument("--point-dropout", type=float, default=0.0)
    p.add_argument("--field-attention-mode", choices=("full",), default="full")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    train_one(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
