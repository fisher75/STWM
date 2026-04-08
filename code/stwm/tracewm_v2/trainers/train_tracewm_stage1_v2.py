#!/usr/bin/env python3
from __future__ import annotations

from argparse import ArgumentParser
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List
import json
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from stwm.tracewm_v2.constants import STATE_DIM
from stwm.tracewm_v2.datasets.stage1_v2_unified import Stage1V2UnifiedDataset, stage1_v2_collate_fn
from stwm.tracewm_v2.losses.structured_trace_loss import StructuredTraceLoss, StructuredTraceLossConfig
from stwm.tracewm_v2.models.causal_trace_transformer import (
    TraceCausalTransformer,
    build_tracewm_v2_config,
    estimate_parameter_count,
)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args() -> Any:
    parser = ArgumentParser(description="Train Stage1 v2 trace model with multi-token causal transformer")
    parser.add_argument("--contract-path", default="/home/chen034/workspace/data/_manifests/stage1_v2_trace_cache_contract_20260408.json")
    parser.add_argument("--dataset-names", nargs="*", default=["pointodyssey", "kubric"])
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--obs-len", type=int, default=8)
    parser.add_argument("--fut-len", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--max-samples-per-dataset", type=int, default=256)

    parser.add_argument("--model-preset", default="debug_small")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--steps-per-epoch", type=int, default=20)
    parser.add_argument("--clip-grad-norm", type=float, default=1.0)

    parser.add_argument("--coord-weight", type=float, default=1.0)
    parser.add_argument("--visibility-weight", type=float, default=0.5)
    parser.add_argument("--residual-weight", type=float, default=0.25)
    parser.add_argument("--velocity-weight", type=float, default=0.25)
    parser.add_argument("--endpoint-weight", type=float, default=0.1)
    parser.add_argument("--enable-visibility", action="store_true")
    parser.add_argument("--enable-residual", action="store_true")
    parser.add_argument("--enable-velocity", action="store_true")
    parser.add_argument("--enable-endpoint", action="store_true")

    parser.add_argument("--output-dir", default="/home/chen034/workspace/stwm/outputs/training/tracewm_stage1_v2")
    parser.add_argument("--summary-json", default="/home/chen034/workspace/stwm/reports/tracewm_stage1_v2_train_summary_20260408.json")
    parser.add_argument("--results-md", default="/home/chen034/workspace/stwm/docs/TRACEWM_STAGE1_V2_TRAIN_SUMMARY_20260408.md")
    parser.add_argument("--ablation-tag", default="mainline")
    parser.add_argument("--seed", type=int, default=20260408)
    return parser.parse_args()


def _to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, torch.Tensor]:
    return {
        "obs_state": batch["obs_state"].to(device),
        "fut_state": batch["fut_state"].to(device),
        "obs_valid": batch["obs_valid"].to(device),
        "fut_valid": batch["fut_valid"].to(device),
        "token_mask": batch["token_mask"].to(device),
    }


def _future_pred(pred: Dict[str, torch.Tensor], obs_len: int) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {
        "coord": pred["coord"][:, obs_len:],
        "vis_logit": pred["vis_logit"][:, obs_len:],
        "residual": pred["residual"][:, obs_len:],
        "velocity": pred["velocity"][:, obs_len:],
    }
    if "endpoint" in pred:
        out["endpoint"] = pred["endpoint"]
    return out


def train_one_epoch(
    model: TraceCausalTransformer,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: StructuredTraceLoss,
    device: torch.device,
    obs_len: int,
    clip_grad_norm: float,
    steps_per_epoch: int,
) -> Dict[str, float]:
    model.train()

    running = {
        "total_loss": 0.0,
        "coord_loss": 0.0,
        "visibility_loss": 0.0,
        "residual_loss": 0.0,
        "velocity_loss": 0.0,
        "endpoint_loss": 0.0,
    }
    steps = 0

    for batch in loader:
        data = _to_device(batch, device)
        full_state = torch.cat([data["obs_state"], data["fut_state"]], dim=1)
        full_valid = torch.cat([data["obs_valid"], data["fut_valid"]], dim=1)

        shifted = torch.zeros_like(full_state)
        shifted[:, 0] = full_state[:, 0]
        shifted[:, 1:] = full_state[:, :-1]

        pred = model(shifted, token_mask=data["token_mask"])
        losses = criterion(
            pred=_future_pred(pred, obs_len=obs_len),
            target_state=full_state[:, obs_len:],
            valid_mask=full_valid[:, obs_len:],
            token_mask=data["token_mask"],
        )

        total = losses["total_loss"]
        optimizer.zero_grad(set_to_none=True)
        total.backward()
        if clip_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(clip_grad_norm))
        optimizer.step()

        for k in running:
            running[k] += float(losses[k].detach().item())

        steps += 1
        if steps >= steps_per_epoch:
            break

    denom = float(max(steps, 1))
    return {k: float(v / denom) for k, v in running.items()}


def main() -> None:
    args = parse_args()
    set_seed(int(args.seed))

    output_dir = Path(args.output_dir)
    summary_json = Path(args.summary_json)
    results_md = Path(args.results_md)

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    results_md.parent.mkdir(parents=True, exist_ok=True)

    dataset = Stage1V2UnifiedDataset(
        dataset_names=[str(x) for x in args.dataset_names],
        split=str(args.train_split),
        contract_path=args.contract_path,
        obs_len=int(args.obs_len),
        fut_len=int(args.fut_len),
        max_tokens=int(args.max_tokens),
        max_samples_per_dataset=int(args.max_samples_per_dataset),
    )

    loader = DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=int(args.num_workers),
        collate_fn=stage1_v2_collate_fn,
    )

    cfg = build_tracewm_v2_config(str(args.model_preset))
    if cfg.state_dim != STATE_DIM:
        raise ValueError(f"config.state_dim={cfg.state_dim} does not match STATE_DIM={STATE_DIM}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TraceCausalTransformer(cfg).to(device)

    loss_cfg = StructuredTraceLossConfig(
        coord_weight=float(args.coord_weight),
        visibility_weight=float(args.visibility_weight),
        residual_weight=float(args.residual_weight),
        velocity_weight=float(args.velocity_weight),
        endpoint_weight=float(args.endpoint_weight),
        enable_visibility=bool(args.enable_visibility),
        enable_residual=bool(args.enable_residual),
        enable_velocity=bool(args.enable_velocity),
        enable_endpoint=bool(args.enable_endpoint),
    )
    criterion = StructuredTraceLoss(loss_cfg)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )

    history: List[Dict[str, float]] = []
    for epoch in range(int(args.epochs)):
        stats = train_one_epoch(
            model=model,
            loader=loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            obs_len=int(args.obs_len),
            clip_grad_norm=float(args.clip_grad_norm),
            steps_per_epoch=int(args.steps_per_epoch),
        )
        stats["epoch"] = float(epoch + 1)
        history.append(stats)
        print(
            f"[stage1-v2-train] epoch={epoch + 1} total={stats['total_loss']:.6f} "
            f"coord={stats['coord_loss']:.6f} vis={stats['visibility_loss']:.6f}"
        )

    final_metrics = history[-1] if history else {}
    total_params = int(sum(p.numel() for p in model.parameters()))
    estimated_params = int(estimate_parameter_count(cfg))

    summary = {
        "generated_at_utc": now_iso(),
        "ablation_tag": str(args.ablation_tag),
        "device": str(device),
        "args": vars(args),
        "model": {
            "preset": str(args.model_preset),
            "config": cfg.__dict__,
            "parameter_count": total_params,
            "estimated_parameter_count": estimated_params,
            "target_220m_range_pass": bool(200_000_000 <= estimated_params <= 240_000_000),
        },
        "dataset": {
            "size": int(len(dataset)),
            "split": str(args.train_split),
            "dataset_names": [str(x) for x in args.dataset_names],
        },
        "loss_config": loss_cfg.__dict__,
        "history": history,
        "final_metrics": final_metrics,
        "p1_shape_contract": {
            "state_shape": "[B,T,K,D]",
            "state_dim": STATE_DIM,
        },
    }

    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# TRACEWM Stage1 v2 Train Summary",
        "",
        f"- generated_at_utc: {summary['generated_at_utc']}",
        f"- ablation_tag: {summary['ablation_tag']}",
        f"- dataset_size: {summary['dataset']['size']}",
        f"- model_preset: {summary['model']['preset']}",
        f"- parameter_count: {summary['model']['parameter_count']}",
        f"- estimated_parameter_count: {summary['model']['estimated_parameter_count']}",
        f"- target_220m_range_pass: {summary['model']['target_220m_range_pass']}",
        "",
        "## Final Metrics",
        "",
        "| metric | value |",
        "|---|---:|",
    ]

    for key in ["total_loss", "coord_loss", "visibility_loss", "residual_loss", "velocity_loss", "endpoint_loss"]:
        lines.append(f"| {key} | {float(final_metrics.get(key, 0.0)):.6f} |")

    results_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    ckpt = output_dir / f"model_{args.ablation_tag}.pt"
    torch.save({"model": model.state_dict(), "config": cfg.__dict__}, ckpt)

    print(f"[stage1-v2-train] summary={summary_json}")
    print(f"[stage1-v2-train] results_md={results_md}")
    print(f"[stage1-v2-train] checkpoint={ckpt}")


if __name__ == "__main__":
    main()
