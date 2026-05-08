#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

SETPROCTITLE_STATUS: dict[str, Any] = {
    "requested_title": "python",
    "setproctitle_ok": False,
    "exact_error": None,
}
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
from stwm.modules.ostf_external_gt_world_model_v30 import OSTFExternalGTConfigV30, OSTFExternalGTWorldModelV30
from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, write_doc
from stwm.tools.ostf_v30_external_gt_metrics_20260508 import aggregate_report, item_row, paired_bootstrap
from stwm.tools.ostf_v30_external_gt_schema_20260508 import utc_now


RUN_DIR = ROOT / "reports/stwm_ostf_v30_external_gt_runs"
CKPT_DIR = ROOT / "outputs/checkpoints/stwm_ostf_v30_external_gt"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    out = {}
    for key, val in batch.items():
        out[key] = val.to(device) if isinstance(val, torch.Tensor) else val
    return out


def build_model(args: argparse.Namespace) -> OSTFExternalGTWorldModelV30:
    hidden = int(args.hidden_dim)
    cfg = OSTFExternalGTConfigV30(
        horizon=int(args.horizon),
        point_dim=int(args.point_dim),
        hidden_dim=hidden,
        point_token_dim=max(64, hidden // 2),
        num_layers=int(args.layers),
        num_heads=int(args.heads),
        learned_modes=int(args.learned_modes),
        damped_gamma=float(args.damped_gamma),
        use_semantic=not bool(args.wo_semantic),
    )
    return OSTFExternalGTWorldModelV30(cfg)


def loss_fn(out: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
    modes = out["point_hypotheses"]
    gt = batch["fut_points"][:, :, :, None, :].expand_as(modes)
    valid = batch["fut_vis"].float()
    conf = batch["fut_conf"].float()
    weight = valid * (0.5 + conf)
    diff = F.smooth_l1_loss(modes, gt, reduction="none").sum(dim=-1)
    denom = weight.sum(dim=(1, 2)).clamp_min(1.0)
    cost = (diff * weight[:, :, :, None]).sum(dim=(1, 2)) / denom[:, None]
    best_cost, best_mode = cost.min(dim=-1)
    endpoint_diff = F.smooth_l1_loss(modes[:, :, -1], batch["fut_points"][:, :, -1, None, :].expand_as(modes[:, :, -1]), reduction="none").sum(dim=-1)
    end_w = (batch["fut_vis"][:, :, -1].float() * (0.5 + batch["fut_conf"][:, :, -1].float()))
    endpoint_cost = (endpoint_diff * end_w[:, :, None]).sum(dim=1) / end_w.sum(dim=1).clamp_min(1.0)[:, None]
    best_endpoint = endpoint_cost.min(dim=-1).values
    mode_ce = F.cross_entropy(out["hypothesis_logits"], best_mode.detach())
    vis_loss = F.binary_cross_entropy_with_logits(out["visibility_logits"], batch["fut_vis"].float())
    diversity = modes.std(dim=3).mean()
    total = best_cost.mean() + 0.7 * best_endpoint.mean() + 0.2 * mode_ce + 0.15 * vis_loss - 0.001 * diversity
    return total, {
        "loss": float(total.detach().cpu()),
        "best_cost": float(best_cost.mean().detach().cpu()),
        "endpoint": float(best_endpoint.mean().detach().cpu()),
        "mode_ce": float(mode_ce.detach().cpu()),
        "visibility": float(vis_loss.detach().cpu()),
        "diversity": float(diversity.detach().cpu()),
    }


def evaluate(model: OSTFExternalGTWorldModelV30, loader: DataLoader, device: torch.device) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    model.eval()
    rows: list[dict[str, Any]] = []
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
                        pred=top1[i],
                        modes=modes[i],
                        visibility_logits=vis_logits[i],
                        tags=batch["v30_subset_tags"][i],
                    )
                )
    return aggregate_report(rows), rows


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
    losses = []
    start = time.time()
    it = iter(train_loader)
    for step in range(1, int(args.steps) + 1):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(train_loader)
            batch = next(it)
        batch_d = move_batch(batch, device)
        opt.zero_grad(set_to_none=True)
        with autocast("cuda", enabled=(device.type == "cuda" and args.amp)):
            out = model(
                obs_points=batch_d["obs_points"],
                obs_vis=batch_d["obs_vis"],
                obs_conf=batch_d["obs_conf"],
                semantic_id=batch_d["semantic_id"],
            )
            loss, comps = loss_fn(out, batch_d)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        losses.append({"step": step, **comps})
        if step % int(args.eval_interval) == 0 or step == int(args.steps):
            val_metrics, _ = evaluate(model, val_loader, device)
            allm = val_metrics["all"]
            score = -float(allm.get("minFDE_K") or 1e9) + 50.0 * float(allm.get("threshold_auc_endpoint_16_32_64_128") or 0.0)
            if score > best_score:
                best_score = score
                torch.save({"model": model.state_dict(), "args": vars(args), "val_metrics": val_metrics, "step": step}, best_path)
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    val_metrics, val_rows = evaluate(model, val_loader, device)
    test_metrics, test_rows = evaluate(model, test_loader, device)
    payload = {
        "experiment_name": args.experiment_name,
        "generated_at_utc": utc_now(),
        "completed": True,
        "smoke": bool(args.smoke),
        "seed": int(args.seed),
        "horizon": int(args.horizon),
        "m_points": int(args.m_points),
        "point_dim": int(args.point_dim),
        "wo_semantic": bool(args.wo_semantic),
        "steps": int(args.steps),
        "device": str(device),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "setproctitle_status": SETPROCTITLE_STATUS,
        "duration_seconds": float(time.time() - start),
        "checkpoint_path": str(best_path.relative_to(ROOT)),
        "best_val_score": float(best_score),
        "train_loss_first": losses[0] if losses else None,
        "train_loss_last": losses[-1] if losses else None,
        "train_loss_decreased": bool(losses and losses[-1]["loss"] <= losses[0]["loss"]),
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "val_item_rows": val_rows,
        "test_item_rows": test_rows,
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
    p.add_argument("--layers", type=int, default=3)
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
    return p.parse_args()


def main() -> int:
    args = parse_args()
    train_one(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
