#!/usr/bin/env python3
from __future__ import annotations

from argparse import ArgumentParser
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple
import itertools
import json
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from stwm.tracewm.datasets.stage1_tapvid import Stage1TapVidDataset
from stwm.tracewm.datasets.stage1_tapvid3d import Stage1TapVid3DDataset
from stwm.tracewm.datasets.stage1_unified import Stage1UnifiedDataset, load_stage1_minisplits, stage1_collate_fn


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
        return t.mean(dim=1)
    if t.ndim == 2:
        return t
    return None


def build_state_batch(batch: Dict[str, Any], device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    bs = int(batch.get("batch_size", 0))
    obs_states: List[torch.Tensor] = []
    fut_states: List[torch.Tensor] = []

    for i in range(bs):
        obs2d = _reduce_tracks(_batch_item(batch, "obs_tracks_2d", i))
        fut2d = _reduce_tracks(_batch_item(batch, "fut_tracks_2d", i))
        obs3d = _reduce_tracks(_batch_item(batch, "obs_tracks_3d", i))
        fut3d = _reduce_tracks(_batch_item(batch, "fut_tracks_3d", i))

        if obs2d is None and obs3d is None:
            obs_len = 8
            obs_valid = _batch_item(batch, "obs_valid", i)
            if isinstance(obs_valid, torch.Tensor):
                obs_len = int(obs_valid.shape[0])
            obs_state = torch.zeros((obs_len, 5), dtype=torch.float32)
        else:
            obs_len = obs2d.shape[0] if obs2d is not None else obs3d.shape[0]
            obs_state = torch.zeros((obs_len, 5), dtype=torch.float32)
            if obs2d is not None:
                obs_state[:, 0:2] = obs2d[:, 0:2]
            if obs3d is not None:
                obs_state[:, 2:5] = obs3d[:, 0:3]

        if fut2d is None and fut3d is None:
            fut_len = 8
            fut_valid = _batch_item(batch, "fut_valid", i)
            if isinstance(fut_valid, torch.Tensor):
                fut_len = int(fut_valid.shape[0])
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


DATASET_ID_MAP = {
    "pointodyssey": 0,
    "kubric": 1,
    "tapvid": 2,
    "tapvid3d": 3,
}


def dataset_ids_from_batch(batch: Dict[str, Any], device: torch.device) -> torch.Tensor:
    ds = batch.get("dataset", [])
    if isinstance(ds, list):
        ids = [DATASET_ID_MAP.get(str(x), 7) for x in ds]
        return torch.tensor(ids, dtype=torch.long, device=device)
    bs = int(batch.get("batch_size", 0))
    return torch.zeros((bs,), dtype=torch.long, device=device)


class BaseTraceModel(nn.Module):
    def __init__(self, state_dim: int = 5, hidden_dim: int = 128) -> None:
        super().__init__()
        self.obs_encoder = nn.GRU(input_size=state_dim, hidden_size=hidden_dim, batch_first=True)
        self.roll_cell = nn.GRUCell(input_size=state_dim, hidden_size=hidden_dim)
        self.head = nn.Linear(hidden_dim, state_dim)

    def rollout_teacher_forced(
        self,
        obs: torch.Tensor,
        target: torch.Tensor,
        dataset_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del dataset_ids
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
            preds.append(self.head(h_t))

        return torch.stack(preds, dim=1)

    def rollout_free(
        self,
        obs: torch.Tensor,
        fut_len: int,
        dataset_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del dataset_ids
        _, h = self.obs_encoder(obs)
        h_t = h[-1]
        prev = obs[:, -1, :]
        preds: List[torch.Tensor] = []

        for _ in range(fut_len):
            h_t = self.roll_cell(prev, h_t)
            pred = self.head(h_t)
            preds.append(pred)
            prev = pred

        return torch.stack(preds, dim=1)


class SourceConditionedTraceModel(nn.Module):
    def __init__(self, state_dim: int = 5, hidden_dim: int = 128, source_emb_dim: int = 8, num_sources: int = 8) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.source_embedding = nn.Embedding(num_embeddings=num_sources, embedding_dim=source_emb_dim)
        self.obs_encoder = nn.GRU(input_size=state_dim + source_emb_dim, hidden_size=hidden_dim, batch_first=True)
        self.roll_cell = nn.GRUCell(input_size=state_dim + source_emb_dim, hidden_size=hidden_dim)
        self.head = nn.Linear(hidden_dim, state_dim)

    def _expand_obs_with_source(self, obs: torch.Tensor, dataset_ids: torch.Tensor) -> torch.Tensor:
        emb = self.source_embedding(dataset_ids)
        emb_seq = emb.unsqueeze(1).expand(-1, obs.shape[1], -1)
        return torch.cat([obs, emb_seq], dim=-1)

    def rollout_teacher_forced(
        self,
        obs: torch.Tensor,
        target: torch.Tensor,
        dataset_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if dataset_ids is None:
            raise ValueError("dataset_ids required for source_conditioned mode")

        obs_cond = self._expand_obs_with_source(obs, dataset_ids)
        _, h = self.obs_encoder(obs_cond)
        h_t = h[-1]

        emb = self.source_embedding(dataset_ids)
        fut_len = target.shape[1]
        preds: List[torch.Tensor] = []

        for t in range(fut_len):
            if t == 0:
                x_state = obs[:, -1, :]
            else:
                x_state = target[:, t - 1, :]
            x_t = torch.cat([x_state, emb], dim=-1)
            h_t = self.roll_cell(x_t, h_t)
            preds.append(self.head(h_t))

        return torch.stack(preds, dim=1)

    def rollout_free(
        self,
        obs: torch.Tensor,
        fut_len: int,
        dataset_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if dataset_ids is None:
            raise ValueError("dataset_ids required for source_conditioned mode")

        obs_cond = self._expand_obs_with_source(obs, dataset_ids)
        _, h = self.obs_encoder(obs_cond)
        h_t = h[-1]

        emb = self.source_embedding(dataset_ids)
        prev = obs[:, -1, :]
        preds: List[torch.Tensor] = []

        for _ in range(fut_len):
            x_t = torch.cat([prev, emb], dim=-1)
            h_t = self.roll_cell(x_t, h_t)
            pred = self.head(h_t)
            preds.append(pred)
            prev = pred

        return torch.stack(preds, dim=1)


def dataset_mix_info(ds: Stage1UnifiedDataset) -> Dict[str, int]:
    counter: Counter[str] = Counter()
    for idx in ds.index_map:
        counter[idx.dataset_name] += 1
    return {k: int(v) for k, v in sorted(counter.items())}


def compute_train_losses(
    pred_teacher: torch.Tensor,
    pred_free: torch.Tensor,
    target: torch.Tensor,
    dataset_names: List[str],
    fix_mode: str,
    free_loss_weight: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
    mse = nn.MSELoss(reduction="mean")

    if fix_mode != "joint_loss_normalized":
        loss_t = mse(pred_teacher, target)
        loss_f = mse(pred_free, target)
        loss_total = loss_t + free_loss_weight * loss_f
        aux = {
            "loss_mode": "standard_global_mse",
        }
        return loss_t, loss_f, loss_total, aux

    err_t = ((pred_teacher - target) ** 2).mean(dim=(1, 2))
    err_f = ((pred_free - target) ** 2).mean(dim=(1, 2))

    group_vals_t: List[torch.Tensor] = []
    group_vals_f: List[torch.Tensor] = []
    group_counts: Dict[str, int] = {}

    uniq = sorted(set(dataset_names))
    for d in uniq:
        idx = [i for i, x in enumerate(dataset_names) if x == d]
        if not idx:
            continue
        idxt = torch.tensor(idx, dtype=torch.long, device=err_t.device)
        group_vals_t.append(err_t.index_select(0, idxt).mean())
        group_vals_f.append(err_f.index_select(0, idxt).mean())
        group_counts[d] = len(idx)

    if not group_vals_t:
        loss_t = err_t.mean()
        loss_f = err_f.mean()
    else:
        loss_t = torch.stack(group_vals_t).mean()
        loss_f = torch.stack(group_vals_f).mean()

    loss_total = loss_t + free_loss_weight * loss_f
    aux = {
        "loss_mode": "dataset_normalized",
        "dataset_group_counts": group_counts,
    }
    return loss_t, loss_f, loss_total, aux


def evaluate_loader(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    free_loss_weight: float,
    max_batches: int = 0,
) -> Dict[str, float]:
    model.eval()
    mse = nn.MSELoss(reduction="mean")

    teacher_vals: List[float] = []
    free_vals: List[float] = []
    total_vals: List[float] = []

    with torch.no_grad():
        for bi, batch in enumerate(loader):
            if max_batches > 0 and bi >= max_batches:
                break
            obs, fut = build_state_batch(batch, device)
            dataset_ids = dataset_ids_from_batch(batch, device)

            pred_t = model.rollout_teacher_forced(obs, fut, dataset_ids=dataset_ids)
            pred_f = model.rollout_free(obs, fut.shape[1], dataset_ids=dataset_ids)

            loss_t = mse(pred_t, fut)
            loss_f = mse(pred_f, fut)
            loss_total = loss_t + free_loss_weight * loss_f

            teacher_vals.append(float(loss_t.item()))
            free_vals.append(float(loss_f.item()))
            total_vals.append(float(loss_total.item()))

    def _mean(x: List[float]) -> float:
        return float(sum(x) / max(len(x), 1))

    return {
        "val_teacher_forced_loss": _mean(teacher_vals),
        "val_free_rollout_loss": _mean(free_vals),
        "val_total_loss": _mean(total_vals),
        "val_batches": len(teacher_vals),
    }


def _records_for(minisplits: Dict[str, Any], dataset: str, split: str) -> List[Dict[str, Any]]:
    datasets = minisplits.get("datasets", {}) if isinstance(minisplits, dict) else {}
    ds = datasets.get(dataset, {}) if isinstance(datasets, dict) else {}
    out = ds.get(split, []) if isinstance(ds, dict) else []
    return [x for x in out if isinstance(x, dict)]


def _eval_dataset_state_space(
    model: nn.Module,
    dataset: Dataset,
    state_dims: Tuple[int, int],
    device: torch.device,
    max_samples: int,
) -> Dict[str, float]:
    mse = nn.MSELoss(reduction="mean")
    d0, d1 = state_dims

    teacher_mse_vals: List[float] = []
    free_mse_vals: List[float] = []
    teacher_endpoint_vals: List[float] = []
    free_endpoint_vals: List[float] = []

    model.eval()
    with torch.no_grad():
        n = min(len(dataset), max_samples)
        for i in range(n):
            sample = dataset[i]
            batch = stage1_collate_fn([sample])
            obs, fut = build_state_batch(batch, device)
            dataset_ids = dataset_ids_from_batch(batch, device)

            pred_t = model.rollout_teacher_forced(obs, fut, dataset_ids=dataset_ids)
            pred_f = model.rollout_free(obs, fut.shape[1], dataset_ids=dataset_ids)

            target = fut[..., d0:d1]
            p_teacher = pred_t[..., d0:d1]
            p_free = pred_f[..., d0:d1]

            teacher_mse_vals.append(float(mse(p_teacher, target).item()))
            free_mse_vals.append(float(mse(p_free, target).item()))

            teacher_endpoint = torch.linalg.norm(p_teacher[:, -1, :] - target[:, -1, :], dim=-1).mean().item()
            free_endpoint = torch.linalg.norm(p_free[:, -1, :] - target[:, -1, :], dim=-1).mean().item()
            teacher_endpoint_vals.append(float(teacher_endpoint))
            free_endpoint_vals.append(float(free_endpoint))

    def _mean(x: List[float]) -> float:
        return float(sum(x) / max(len(x), 1))

    return {
        "samples": min(len(dataset), max_samples),
        "teacher_forced_mse": _mean(teacher_mse_vals),
        "free_rollout_mse": _mean(free_mse_vals),
        "teacher_forced_endpoint_l2": _mean(teacher_endpoint_vals),
        "free_rollout_endpoint_l2": _mean(free_endpoint_vals),
    }


def evaluate_external(
    model: nn.Module,
    splits_path: str | Path,
    data_root: str | Path,
    device: torch.device,
    max_tapvid_samples: int,
    max_tapvid3d_samples: int,
) -> Dict[str, Any]:
    minisplits = load_stage1_minisplits(splits_path)

    tapvid_ds = Stage1TapVidDataset(
        split="eval_mini",
        minisplit_records=_records_for(minisplits, "tapvid", "eval_mini"),
    )
    tapvid3d_ds = Stage1TapVid3DDataset(
        data_root=data_root,
        split="eval_mini",
        minisplit_records=_records_for(minisplits, "tapvid3d", "eval_mini"),
    )

    tapvid_metrics = _eval_dataset_state_space(
        model=model,
        dataset=tapvid_ds,
        state_dims=(0, 2),
        device=device,
        max_samples=max_tapvid_samples,
    )
    tapvid3d_metrics = _eval_dataset_state_space(
        model=model,
        dataset=tapvid3d_ds,
        state_dims=(2, 5),
        device=device,
        max_samples=max_tapvid3d_samples,
    )

    tapvid_metrics["main_eval_ready"] = tapvid_metrics["samples"] > 0
    tapvid3d_metrics["limited_eval_ready"] = tapvid3d_metrics["samples"] > 0
    tapvid3d_metrics["full_eval_ready"] = False

    return {
        "tapvid": tapvid_metrics,
        "tapvid3d": tapvid3d_metrics,
    }


def evaluate_joint_source_breakdown(
    model: nn.Module,
    data_root: str | Path,
    splits_path: str | Path,
    device: torch.device,
    batch_size: int,
    free_loss_weight: float,
) -> Dict[str, Dict[str, float]]:
    point_val = Stage1UnifiedDataset(
        dataset_names=["pointodyssey"],
        split="val_iter1_pointodyssey",
        data_root=data_root,
        minisplit_path=splits_path,
        obs_len=8,
        fut_len=8,
    )
    kubric_val = Stage1UnifiedDataset(
        dataset_names=["kubric"],
        split="val_iter1_kubric",
        data_root=data_root,
        minisplit_path=splits_path,
        obs_len=8,
        fut_len=8,
    )

    point_loader = DataLoader(
        point_val,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=stage1_collate_fn,
    )
    kubric_loader = DataLoader(
        kubric_val,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=stage1_collate_fn,
    )

    return {
        "pointodyssey_val": evaluate_loader(model, point_loader, device, free_loss_weight=free_loss_weight, max_batches=0),
        "kubric_val": evaluate_loader(model, kubric_loader, device, free_loss_weight=free_loss_weight, max_batches=0),
    }


def build_parser() -> ArgumentParser:
    p = ArgumentParser(description="TraceWM Stage1 model-fix trainer")
    p.add_argument(
        "--fix-mode",
        required=True,
        choices=[
            "joint_balanced_sampler",
            "joint_loss_normalized",
            "joint_source_conditioned",
        ],
    )
    p.add_argument("--run-name", required=True)
    p.add_argument("--data-root", default="/home/chen034/workspace/data")
    p.add_argument("--splits-path", default="/home/chen034/workspace/data/_manifests/stage1_iter1_splits_20260408.json")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--summary-json", required=True)
    p.add_argument("--seed", type=int, default=20260408)
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--steps-per-epoch", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--free-loss-weight", type=float, default=0.5)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--source-emb-dim", type=int, default=8)
    p.add_argument("--eval-max-batches", type=int, default=0)
    p.add_argument("--eval-max-tapvid-samples", type=int, default=6)
    p.add_argument("--eval-max-tapvid3d-samples", type=int, default=12)
    p.add_argument("--resume", default="")
    p.add_argument("--device", default="auto")
    p.set_defaults(auto_resume=True)
    p.add_argument("--no-auto-resume", dest="auto_resume", action="store_false")
    return p


def _checkpoint_payload(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    global_step: int,
    history: List[Dict[str, Any]],
    best_val_total_loss: float,
    args: Any,
) -> Dict[str, Any]:
    return {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "history": history,
        "best_val_total_loss": best_val_total_loss,
        "args": vars(args),
    }


def main() -> int:
    args = build_parser().parse_args()
    set_seed(args.seed)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    train_dataset = Stage1UnifiedDataset(
        dataset_names=["pointodyssey", "kubric"],
        split="train_iter1_joint",
        data_root=args.data_root,
        minisplit_path=args.splits_path,
        obs_len=8,
        fut_len=8,
    )
    val_dataset = Stage1UnifiedDataset(
        dataset_names=["pointodyssey", "kubric"],
        split="val_iter1_joint",
        data_root=args.data_root,
        minisplit_path=args.splits_path,
        obs_len=8,
        fut_len=8,
    )

    sampler_info: Dict[str, Any] = {
        "mode": "default_shuffle",
    }

    if args.fix_mode == "joint_balanced_sampler":
        mix = dataset_mix_info(train_dataset)
        count_map = {k: max(int(v), 1) for k, v in mix.items()}
        weights: List[float] = []
        for idx in train_dataset.index_map:
            weights.append(1.0 / float(count_map.get(idx.dataset_name, 1)))
        sampler = WeightedRandomSampler(
            weights=torch.tensor(weights, dtype=torch.double),
            num_samples=len(train_dataset),
            replacement=True,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            sampler=sampler,
            num_workers=0,
            collate_fn=stage1_collate_fn,
        )
        sampler_info = {
            "mode": "weighted_balanced_sampler",
            "source_counts": mix,
            "num_samples": len(train_dataset),
        }
    else:
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

    if args.fix_mode == "joint_source_conditioned":
        model: nn.Module = SourceConditionedTraceModel(
            state_dim=5,
            hidden_dim=args.hidden_dim,
            source_emb_dim=args.source_emb_dim,
        ).to(device)
        model_info = {
            "type": "source_conditioned",
            "source_emb_dim": args.source_emb_dim,
        }
    else:
        model = BaseTraceModel(state_dim=5, hidden_dim=args.hidden_dim).to(device)
        model_info = {
            "type": "base_trace_model",
        }

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    start_epoch = 0
    global_step = 0
    history: List[Dict[str, Any]] = []
    best_val_total_loss = float("inf")

    latest_ckpt = ckpt_dir / "latest.pt"
    resume_path = Path(args.resume) if args.resume.strip() else None
    if (resume_path is not None and resume_path.exists()):
        payload = torch.load(resume_path, map_location="cpu")
        model.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        start_epoch = int(payload.get("epoch", -1)) + 1
        global_step = int(payload.get("global_step", 0))
        history = list(payload.get("history", []))
        best_val_total_loss = float(payload.get("best_val_total_loss", float("inf")))
        print(f"[fix] resumed from --resume: {resume_path}")
    elif args.auto_resume and latest_ckpt.exists():
        payload = torch.load(latest_ckpt, map_location="cpu")
        model.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        start_epoch = int(payload.get("epoch", -1)) + 1
        global_step = int(payload.get("global_step", 0))
        history = list(payload.get("history", []))
        best_val_total_loss = float(payload.get("best_val_total_loss", float("inf")))
        print(f"[fix] resumed from latest: {latest_ckpt}")

    train_iter = itertools.cycle(train_loader)

    for epoch in range(start_epoch, args.epochs):
        model.train()
        train_teacher_vals: List[float] = []
        train_free_vals: List[float] = []
        train_total_vals: List[float] = []
        aux_last: Dict[str, Any] = {}

        for _ in range(args.steps_per_epoch):
            batch = next(train_iter)
            obs, fut = build_state_batch(batch, device)
            dataset_ids = dataset_ids_from_batch(batch, device)
            dataset_names = [str(x) for x in batch.get("dataset", [])] if isinstance(batch.get("dataset"), list) else []

            pred_teacher = model.rollout_teacher_forced(obs, fut, dataset_ids=dataset_ids)
            pred_free = model.rollout_free(obs, fut.shape[1], dataset_ids=dataset_ids)

            loss_teacher, loss_free, loss_total, aux = compute_train_losses(
                pred_teacher=pred_teacher,
                pred_free=pred_free,
                target=fut,
                dataset_names=dataset_names,
                fix_mode=args.fix_mode,
                free_loss_weight=args.free_loss_weight,
            )

            optimizer.zero_grad(set_to_none=True)
            loss_total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_teacher_vals.append(float(loss_teacher.item()))
            train_free_vals.append(float(loss_free.item()))
            train_total_vals.append(float(loss_total.item()))
            aux_last = aux
            global_step += 1

        train_metrics = {
            "train_teacher_forced_loss": float(sum(train_teacher_vals) / max(len(train_teacher_vals), 1)),
            "train_free_rollout_loss": float(sum(train_free_vals) / max(len(train_free_vals), 1)),
            "train_total_loss": float(sum(train_total_vals) / max(len(train_total_vals), 1)),
            "steps_this_epoch": args.steps_per_epoch,
            "train_loss_aux": aux_last,
        }

        val_metrics = evaluate_loader(
            model=model,
            loader=val_loader,
            device=device,
            free_loss_weight=args.free_loss_weight,
            max_batches=args.eval_max_batches,
        )
        external_metrics = evaluate_external(
            model=model,
            splits_path=args.splits_path,
            data_root=args.data_root,
            device=device,
            max_tapvid_samples=args.eval_max_tapvid_samples,
            max_tapvid3d_samples=args.eval_max_tapvid3d_samples,
        )
        val_source_breakdown = evaluate_joint_source_breakdown(
            model=model,
            data_root=args.data_root,
            splits_path=args.splits_path,
            device=device,
            batch_size=args.batch_size,
            free_loss_weight=args.free_loss_weight,
        )

        epoch_metrics: Dict[str, Any] = {
            "epoch": epoch,
            "global_step": global_step,
            **train_metrics,
            **val_metrics,
            "tapvid": external_metrics["tapvid"],
            "tapvid3d": external_metrics["tapvid3d"],
            "val_source_breakdown": val_source_breakdown,
        }
        history.append(epoch_metrics)

        payload = _checkpoint_payload(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            global_step=global_step,
            history=history,
            best_val_total_loss=best_val_total_loss,
            args=args,
        )

        epoch_ckpt = ckpt_dir / f"epoch_{epoch:03d}.pt"
        torch.save(payload, epoch_ckpt)
        torch.save(payload, latest_ckpt)

        current_val_total = float(val_metrics["val_total_loss"])
        if current_val_total < best_val_total_loss:
            best_val_total_loss = current_val_total
            payload_best = _checkpoint_payload(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                global_step=global_step,
                history=history,
                best_val_total_loss=best_val_total_loss,
                args=args,
            )
            torch.save(payload_best, ckpt_dir / "best.pt")

        print(
            f"[fix][{args.run_name}] epoch={epoch} step={global_step} "
            f"train_total={train_metrics['train_total_loss']:.6f} "
            f"val_total={val_metrics['val_total_loss']:.6f} "
            f"tapvid_free_ep={external_metrics['tapvid']['free_rollout_endpoint_l2']:.6f} "
            f"tapvid3d_free_ep={external_metrics['tapvid3d']['free_rollout_endpoint_l2']:.6f}"
        )

    final_metrics = history[-1] if history else {}

    fix_flags = {
        "balanced_sampler": args.fix_mode == "joint_balanced_sampler",
        "loss_normalized": args.fix_mode == "joint_loss_normalized",
        "source_conditioned": args.fix_mode == "joint_source_conditioned",
    }

    summary = {
        "generated_at_utc": now_iso(),
        "task": "trace_only_future_trace_state_generation",
        "round": "stage1_model_fix",
        "run_name": args.run_name,
        "fix_mode": args.fix_mode,
        "fix_flags": fix_flags,
        "train_datasets": ["pointodyssey", "kubric"],
        "train_split": "train_iter1_joint",
        "val_split": "val_iter1_joint",
        "teacher_forced_supported": True,
        "free_rollout_supported": True,
        "device": str(device),
        "seed": args.seed,
        "epochs": args.epochs,
        "steps_per_epoch": args.steps_per_epoch,
        "total_steps": int(global_step),
        "batch_size": args.batch_size,
        "model_info": model_info,
        "sampler_info": sampler_info,
        "loss_family": {
            "teacher_forced_loss": "mse",
            "free_rollout_loss": "mse",
            "total_loss": f"teacher_forced + {args.free_loss_weight} * free_rollout",
            "free_loss_weight": args.free_loss_weight,
            "loss_mode": "dataset_normalized" if args.fix_mode == "joint_loss_normalized" else "standard_global_mse",
        },
        "dataset_mix_info": {
            "train": dataset_mix_info(train_dataset),
            "val": dataset_mix_info(val_dataset),
        },
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "epoch_history": history,
        "final_metrics": final_metrics,
        "best_val_total_loss": best_val_total_loss,
        "checkpoint_dir": str(ckpt_dir),
        "checkpoint_best": str(ckpt_dir / "best.pt"),
        "checkpoint_latest": str(latest_ckpt),
        "summary_json": str(Path(args.summary_json)),
        "splits_path": str(Path(args.splits_path)),
    }

    summary_path = Path(args.summary_json)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[fix][{args.run_name}] wrote summary: {summary_path}")
    print(f"[fix][{args.run_name}] checkpoints: {ckpt_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
