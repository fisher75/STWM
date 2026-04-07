#!/usr/bin/env python3
from __future__ import annotations

from argparse import ArgumentParser
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List
import itertools
import json
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from stwm.tracewm.datasets.stage1_unified import Stage1UnifiedDataset, stage1_collate_fn


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _batch_item(batch: Dict[str, Any], key: str, idx: int) -> Any:
    v = batch.get(key)
    if isinstance(v, list):
        return v[idx]
    if isinstance(v, torch.Tensor):
        return v[idx]
    return None


def _reduce_tracks(tracks: Any) -> torch.Tensor | None:
    if tracks is None:
        return None
    if not isinstance(tracks, torch.Tensor):
        return None

    t = tracks.detach().to(torch.float32)
    if t.ndim == 3:
        # [T, N, C] -> [T, C]
        return t.mean(dim=1)
    if t.ndim == 2:
        # [T, C]
        return t
    return None


def build_state_batch(batch: Dict[str, Any], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    bs = int(batch.get("batch_size", 0))
    obs_states: List[torch.Tensor] = []
    fut_states: List[torch.Tensor] = []

    for i in range(bs):
        obs2d = _reduce_tracks(_batch_item(batch, "obs_tracks_2d", i))
        fut2d = _reduce_tracks(_batch_item(batch, "fut_tracks_2d", i))
        obs3d = _reduce_tracks(_batch_item(batch, "obs_tracks_3d", i))
        fut3d = _reduce_tracks(_batch_item(batch, "fut_tracks_3d", i))

        if obs2d is None and obs3d is None:
            obs_len = int(_batch_item(batch, "obs_valid", i).shape[0]) if isinstance(_batch_item(batch, "obs_valid", i), torch.Tensor) else 8
            obs_state = torch.zeros((obs_len, 5), dtype=torch.float32)
        else:
            obs_len = obs2d.shape[0] if obs2d is not None else obs3d.shape[0]
            obs_state = torch.zeros((obs_len, 5), dtype=torch.float32)
            if obs2d is not None:
                obs_state[:, 0:2] = obs2d[:, 0:2]
            if obs3d is not None:
                obs_state[:, 2:5] = obs3d[:, 0:3]

        if fut2d is None and fut3d is None:
            fut_len = int(_batch_item(batch, "fut_valid", i).shape[0]) if isinstance(_batch_item(batch, "fut_valid", i), torch.Tensor) else 8
            fut_state = torch.zeros((fut_len, 5), dtype=torch.float32)
        else:
            fut_len = fut2d.shape[0] if fut2d is not None else fut3d.shape[0]
            fut_state = torch.zeros((fut_len, 5), dtype=torch.float32)
            if fut2d is not None:
                fut_state[:, 0:2] = fut2d[:, 0:2]
            if fut3d is not None:
                fut_state[:, 2:5] = fut3d[:, 0:3]

        obs_states.append(obs_state)
        fut_states.append(fut_state)

    obs_batch = torch.stack(obs_states, dim=0).to(device)
    fut_batch = torch.stack(fut_states, dim=0).to(device)
    return obs_batch, fut_batch


class TinyTraceModel(nn.Module):
    def __init__(self, state_dim: int = 5, hidden_dim: int = 96) -> None:
        super().__init__()
        self.obs_encoder = nn.GRU(input_size=state_dim, hidden_size=hidden_dim, batch_first=True)
        self.roll_cell = nn.GRUCell(input_size=state_dim, hidden_size=hidden_dim)
        self.head = nn.Linear(hidden_dim, state_dim)

    def rollout_teacher_forced(self, obs: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        _, h = self.obs_encoder(obs)
        h_t = h[-1]
        fut_len = target.shape[1]
        preds: List[torch.Tensor] = []

        for t in range(fut_len):
            if t == 0:
                x_t = obs[:, -1, :]
            else:
                x_t = target[:, t - 1, :]
            h_t = self.roll_cell(x_t, h_t)
            p_t = self.head(h_t)
            preds.append(p_t)

        return torch.stack(preds, dim=1)

    def rollout_free(self, obs: torch.Tensor, fut_len: int) -> torch.Tensor:
        _, h = self.obs_encoder(obs)
        h_t = h[-1]
        prev = obs[:, -1, :]
        preds: List[torch.Tensor] = []

        for _ in range(fut_len):
            h_t = self.roll_cell(prev, h_t)
            p_t = self.head(h_t)
            preds.append(p_t)
            prev = p_t

        return torch.stack(preds, dim=1)


def evaluate(model: TinyTraceModel, loader: DataLoader, device: torch.device, max_batches: int = 4) -> Dict[str, float]:
    model.eval()
    mse = nn.MSELoss(reduction="mean")

    loss_teacher_vals: List[float] = []
    loss_free_vals: List[float] = []

    with torch.no_grad():
        for bi, batch in enumerate(loader):
            if bi >= max_batches:
                break
            obs, fut = build_state_batch(batch, device)
            pred_t = model.rollout_teacher_forced(obs, fut)
            pred_f = model.rollout_free(obs, fut.shape[1])

            loss_teacher_vals.append(float(mse(pred_t, fut).item()))
            loss_free_vals.append(float(mse(pred_f, fut).item()))

    def _mean(x: List[float]) -> float:
        return float(sum(x) / max(len(x), 1))

    return {
        "val_teacher_mse": _mean(loss_teacher_vals),
        "val_free_mse": _mean(loss_free_vals),
        "val_batches": len(loss_teacher_vals),
    }


def build_parser() -> ArgumentParser:
    p = ArgumentParser(description="Stage 1 trace-only tiny-train")
    p.add_argument("--data-root", default="/home/chen034/workspace/data")
    p.add_argument("--minisplit-path", default="/home/chen034/workspace/data/_manifests/stage1_minisplits_20260408.json")
    p.add_argument("--output-dir", default="/home/chen034/workspace/stwm/outputs/training/tracewm_stage1_tiny_20260408")
    p.add_argument("--summary-json", default="/home/chen034/workspace/stwm/reports/tracewm_stage1_tiny_summary_20260408.json")
    p.add_argument("--results-md", default="/home/chen034/workspace/stwm/docs/TRACEWM_STAGE1_TINY_RESULTS_20260408.md")
    p.add_argument("--seed", type=int, default=20260408)
    p.add_argument("--steps", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--device", default="auto")
    return p


def main() -> int:
    args = build_parser().parse_args()
    set_seed(args.seed)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    train_dataset = Stage1UnifiedDataset(
        dataset_names=["pointodyssey", "kubric"],
        split="train_mini",
        data_root=args.data_root,
        minisplit_path=args.minisplit_path,
        obs_len=8,
        fut_len=8,
    )
    val_dataset = Stage1UnifiedDataset(
        dataset_names=["pointodyssey", "kubric"],
        split="val_mini",
        data_root=args.data_root,
        minisplit_path=args.minisplit_path,
        obs_len=8,
        fut_len=8,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=stage1_collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=stage1_collate_fn,
    )

    model = TinyTraceModel(state_dim=5, hidden_dim=96).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    mse = nn.MSELoss(reduction="mean")

    teacher_losses: List[float] = []
    free_losses: List[float] = []
    total_losses: List[float] = []

    model.train()
    train_iter = itertools.cycle(train_loader)
    for step in range(args.steps):
        batch = next(train_iter)
        obs, fut = build_state_batch(batch, device)

        pred_teacher = model.rollout_teacher_forced(obs, fut)
        pred_free = model.rollout_free(obs, fut.shape[1])

        loss_teacher = mse(pred_teacher, fut)
        loss_free = mse(pred_free, fut)
        loss_total = loss_teacher + 0.5 * loss_free

        opt.zero_grad(set_to_none=True)
        loss_total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()

        teacher_losses.append(float(loss_teacher.item()))
        free_losses.append(float(loss_free.item()))
        total_losses.append(float(loss_total.item()))

    val_metrics = evaluate(model, val_loader, device=device, max_batches=6)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt = output_dir / "tiny_trace_model.pt"
    torch.save({"model": model.state_dict(), "args": vars(args)}, ckpt)

    summary = {
        "generated_at_utc": now_iso(),
        "task": "trace_only_future_trace_state_generation",
        "train_datasets": ["pointodyssey", "kubric"],
        "teacher_forced_supported": True,
        "free_rollout_supported": True,
        "device": str(device),
        "seed": args.seed,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "train_teacher_mse_last": teacher_losses[-1] if teacher_losses else None,
        "train_free_mse_last": free_losses[-1] if free_losses else None,
        "train_total_loss_last": total_losses[-1] if total_losses else None,
        "train_teacher_mse_mean": float(sum(teacher_losses) / max(len(teacher_losses), 1)),
        "train_free_mse_mean": float(sum(free_losses) / max(len(free_losses), 1)),
        "train_total_loss_mean": float(sum(total_losses) / max(len(total_losses), 1)),
        "val_metrics": val_metrics,
        "checkpoint_path": str(ckpt),
    }

    summary_path = Path(args.summary_json)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    md_lines = [
        "# TraceWM Stage 1 Tiny Results (2026-04-08)",
        "",
        f"- generated_at_utc: {summary['generated_at_utc']}",
        f"- task: {summary['task']}",
        f"- device: {summary['device']}",
        f"- train_datasets: {summary['train_datasets']}",
        f"- teacher_forced_supported: {summary['teacher_forced_supported']}",
        f"- free_rollout_supported: {summary['free_rollout_supported']}",
        f"- steps: {summary['steps']}",
        f"- batch_size: {summary['batch_size']}",
        f"- train_samples: {summary['train_samples']}",
        f"- val_samples: {summary['val_samples']}",
        "",
        "## Metrics",
        "",
        f"- train_teacher_mse_mean: {summary['train_teacher_mse_mean']:.6f}",
        f"- train_free_mse_mean: {summary['train_free_mse_mean']:.6f}",
        f"- train_total_loss_mean: {summary['train_total_loss_mean']:.6f}",
        f"- val_teacher_mse: {val_metrics['val_teacher_mse']:.6f}",
        f"- val_free_mse: {val_metrics['val_free_mse']:.6f}",
        "",
        f"- checkpoint_path: {summary['checkpoint_path']}",
        f"- summary_json: {summary_path}",
    ]

    md_path = Path(args.results_md)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print(f"[tiny-train] wrote: {summary_path}")
    print(f"[tiny-train] wrote: {md_path}")
    print(f"[tiny-train] checkpoint: {ckpt}")
    print("[tiny-train] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
