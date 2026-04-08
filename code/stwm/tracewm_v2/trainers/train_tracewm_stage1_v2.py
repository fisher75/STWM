#!/usr/bin/env python3
from __future__ import annotations

from argparse import ArgumentParser
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List
import json
import random
import time

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
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--pin-memory", dest="pin_memory", action="store_true")
    parser.add_argument("--no-pin-memory", dest="pin_memory", action="store_false")
    parser.add_argument("--persistent-workers", dest="persistent_workers", action="store_true")
    parser.add_argument("--no-persistent-workers", dest="persistent_workers", action="store_false")
    parser.add_argument("--prefetch-factor", type=int, default=4)
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
    parser.add_argument("--perf-step-timing-json", default="/home/chen034/workspace/stwm/reports/stage1_v2_perf_step_timing_20260408.json")
    parser.add_argument("--ablation-tag", default="mainline")
    parser.add_argument("--seed", type=int, default=20260408)
    parser.set_defaults(pin_memory=True, persistent_workers=True)
    return parser.parse_args()


def _to_device(batch: Dict[str, Any], device: torch.device, non_blocking: bool = False) -> Dict[str, torch.Tensor]:
    return {
        "obs_state": batch["obs_state"].to(device, non_blocking=non_blocking),
        "fut_state": batch["fut_state"].to(device, non_blocking=non_blocking),
        "obs_valid": batch["obs_valid"].to(device, non_blocking=non_blocking),
        "fut_valid": batch["fut_valid"].to(device, non_blocking=non_blocking),
        "token_mask": batch["token_mask"].to(device, non_blocking=non_blocking),
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


def _timing_stats(records: List[Dict[str, float]], key: str) -> Dict[str, float]:
    values = [float(x.get(key, 0.0)) for x in records]
    if not values:
        return {
            "mean": 0.0,
            "median": 0.0,
            "p95": 0.0,
        }
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p95": float(np.percentile(arr, 95.0)),
    }


def train_one_epoch(
    model: TraceCausalTransformer,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: StructuredTraceLoss,
    device: torch.device,
    obs_len: int,
    clip_grad_norm: float,
    steps_per_epoch: int,
    pin_memory: bool,
) -> Dict[str, Any]:
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
    step_timing: List[Dict[str, float]] = []

    data_iter = iter(loader)
    while steps < steps_per_epoch:
        wait_start = time.perf_counter()
        try:
            batch = next(data_iter)
        except StopIteration:
            break
        batch_wait_time = float(time.perf_counter() - wait_start)

        step_start = time.perf_counter()

        non_blocking = bool(pin_memory and device.type == "cuda")
        h2d_start = time.perf_counter()
        data = _to_device(batch, device, non_blocking=non_blocking)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        h2d_time = float(time.perf_counter() - h2d_start)

        full_state = torch.cat([data["obs_state"], data["fut_state"]], dim=1)
        full_valid = torch.cat([data["obs_valid"], data["fut_valid"]], dim=1)

        shifted = torch.zeros_like(full_state)
        shifted[:, 0] = full_state[:, 0]
        shifted[:, 1:] = full_state[:, :-1]

        forward_start = time.perf_counter()
        pred = model(shifted, token_mask=data["token_mask"])
        losses = criterion(
            pred=_future_pred(pred, obs_len=obs_len),
            target_state=full_state[:, obs_len:],
            valid_mask=full_valid[:, obs_len:],
            token_mask=data["token_mask"],
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        forward_time = float(time.perf_counter() - forward_start)

        total = losses["total_loss"]
        backward_start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        total.backward()
        if clip_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(clip_grad_norm))
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        backward_time = float(time.perf_counter() - backward_start)

        optimizer_start = time.perf_counter()
        optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        optimizer_time = float(time.perf_counter() - optimizer_start)

        step_time = float(time.perf_counter() - step_start)
        samples_per_sec = float(data["obs_state"].shape[0] / max(step_time, 1e-8))

        for k in running:
            running[k] += float(losses[k].detach().item())

        step_timing.append(
            {
                "batch_wait_time": batch_wait_time,
                "h2d_time": h2d_time,
                "forward_time": forward_time,
                "backward_time": backward_time,
                "optimizer_time": optimizer_time,
                "step_time": step_time,
                "samples_per_sec": samples_per_sec,
            }
        )

        steps += 1

    denom = float(max(steps, 1))
    return {
        "losses": {k: float(v / denom) for k, v in running.items()},
        "step_timing": step_timing,
        "steps": int(steps),
    }


def main() -> None:
    args = parse_args()
    set_seed(int(args.seed))

    output_dir = Path(args.output_dir)
    summary_json = Path(args.summary_json)
    results_md = Path(args.results_md)
    perf_step_timing_json = Path(args.perf_step_timing_json)

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    results_md.parent.mkdir(parents=True, exist_ok=True)
    perf_step_timing_json.parent.mkdir(parents=True, exist_ok=True)

    dataset = Stage1V2UnifiedDataset(
        dataset_names=[str(x) for x in args.dataset_names],
        split=str(args.train_split),
        contract_path=args.contract_path,
        obs_len=int(args.obs_len),
        fut_len=int(args.fut_len),
        max_tokens=int(args.max_tokens),
        max_samples_per_dataset=int(args.max_samples_per_dataset),
    )

    loader_kwargs: Dict[str, Any] = {
        "dataset": dataset,
        "batch_size": int(args.batch_size),
        "shuffle": True,
        "num_workers": int(args.num_workers),
        "pin_memory": bool(args.pin_memory),
        "collate_fn": stage1_v2_collate_fn,
    }
    if int(args.num_workers) > 0:
        loader_kwargs["persistent_workers"] = bool(args.persistent_workers)
        loader_kwargs["prefetch_factor"] = int(args.prefetch_factor)

    loader = DataLoader(**loader_kwargs)

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
    all_step_timing: List[Dict[str, float]] = []
    for epoch in range(int(args.epochs)):
        epoch_out = train_one_epoch(
            model=model,
            loader=loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            obs_len=int(args.obs_len),
            clip_grad_norm=float(args.clip_grad_norm),
            steps_per_epoch=int(args.steps_per_epoch),
            pin_memory=bool(args.pin_memory),
        )
        stats = dict(epoch_out["losses"])
        stats["epoch"] = float(epoch + 1)
        history.append(stats)
        all_step_timing.extend(list(epoch_out["step_timing"]))
        print(
            f"[stage1-v2-train] epoch={epoch + 1} total={stats['total_loss']:.6f} "
            f"coord={stats['coord_loss']:.6f} vis={stats['visibility_loss']:.6f}"
        )

    final_metrics = history[-1] if history else {}
    total_params = int(sum(p.numel() for p in model.parameters()))
    estimated_params = int(estimate_parameter_count(cfg))

    timing_stats = {
        "batch_wait_time": _timing_stats(all_step_timing, "batch_wait_time"),
        "h2d_time": _timing_stats(all_step_timing, "h2d_time"),
        "forward_time": _timing_stats(all_step_timing, "forward_time"),
        "backward_time": _timing_stats(all_step_timing, "backward_time"),
        "optimizer_time": _timing_stats(all_step_timing, "optimizer_time"),
        "step_time": _timing_stats(all_step_timing, "step_time"),
        "samples_per_sec": _timing_stats(all_step_timing, "samples_per_sec"),
    }

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
        "timing_stats": timing_stats,
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

    perf_payload = {
        "generated_at_utc": now_iso(),
        "ablation_tag": str(args.ablation_tag),
        "model_preset": str(args.model_preset),
        "device": str(device),
        "num_steps": int(len(all_step_timing)),
        "timing_stats": timing_stats,
        "step_records": all_step_timing,
    }
    perf_step_timing_json.write_text(json.dumps(perf_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    ckpt = output_dir / f"model_{args.ablation_tag}.pt"
    torch.save({"model": model.state_dict(), "config": cfg.__dict__}, ckpt)

    print(f"[stage1-v2-train] summary={summary_json}")
    print(f"[stage1-v2-train] results_md={results_md}")
    print(f"[stage1-v2-train] perf_step_timing={perf_step_timing_json}")
    print(f"[stage1-v2-train] checkpoint={ckpt}")


if __name__ == "__main__":
    main()
