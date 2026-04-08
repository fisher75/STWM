#!/usr/bin/env python3
from __future__ import annotations

from argparse import ArgumentParser
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
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


SOURCE_NAMES = ["pointodyssey", "kubric"]


def dataset_ids_from_batch(batch: Dict[str, Any], device: torch.device) -> torch.Tensor:
    ds = batch.get("dataset", [])
    if isinstance(ds, list):
        ids = [DATASET_ID_MAP.get(str(x), 7) for x in ds]
        return torch.tensor(ids, dtype=torch.long, device=device)
    bs = int(batch.get("batch_size", 0))
    return torch.zeros((bs,), dtype=torch.long, device=device)


def empty_route_stats() -> Dict[str, int]:
    return {
        "calls": 0,
        "total_samples": 0,
        "point_private_hits": 0,
        "kubric_private_hits": 0,
        "other_hits": 0,
    }


def _copy_route_stats(stats: Dict[str, int]) -> Dict[str, int]:
    return {
        "calls": int(stats.get("calls", 0)),
        "total_samples": int(stats.get("total_samples", 0)),
        "point_private_hits": int(stats.get("point_private_hits", 0)),
        "kubric_private_hits": int(stats.get("kubric_private_hits", 0)),
        "other_hits": int(stats.get("other_hits", 0)),
    }


class BaseTraceModel(nn.Module):
    def __init__(self, state_dim: int = 5, hidden_dim: int = 128) -> None:
        super().__init__()
        self.obs_encoder = nn.GRU(input_size=state_dim, hidden_size=hidden_dim, batch_first=True)
        self.roll_cell = nn.GRUCell(input_size=state_dim, hidden_size=hidden_dim)
        self.head = nn.Linear(hidden_dim, state_dim)

    def shared_parameters_list(self) -> List[nn.Parameter]:
        return list(self.parameters())

    def private_parameters_list(self) -> List[nn.Parameter]:
        return []

    def rollout_teacher_forced(
        self,
        obs: torch.Tensor,
        target: torch.Tensor,
        dataset_ids: Optional[torch.Tensor] = None,
        route_stats: Optional[Dict[str, int]] = None,
    ) -> torch.Tensor:
        del dataset_ids, route_stats
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
        dataset_ids: Optional[torch.Tensor] = None,
        route_stats: Optional[Dict[str, int]] = None,
    ) -> torch.Tensor:
        del dataset_ids, route_stats
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


class SharedPrivateTraceModel(nn.Module):
    def __init__(self, state_dim: int = 5, hidden_dim: int = 128, adapter_dim: int = 16) -> None:
        super().__init__()
        self.obs_encoder = nn.GRU(input_size=state_dim, hidden_size=hidden_dim, batch_first=True)
        self.roll_cell = nn.GRUCell(input_size=state_dim, hidden_size=hidden_dim)
        self.head = nn.Linear(hidden_dim, state_dim)

        self.private_point = nn.Sequential(
            nn.Linear(hidden_dim, adapter_dim),
            nn.ReLU(),
            nn.Linear(adapter_dim, hidden_dim),
        )
        self.private_kubric = nn.Sequential(
            nn.Linear(hidden_dim, adapter_dim),
            nn.ReLU(),
            nn.Linear(adapter_dim, hidden_dim),
        )

    def shared_parameters_list(self) -> List[nn.Parameter]:
        params: List[nn.Parameter] = []
        params.extend(list(self.obs_encoder.parameters()))
        params.extend(list(self.roll_cell.parameters()))
        params.extend(list(self.head.parameters()))
        return params

    def private_parameters_list(self) -> List[nn.Parameter]:
        params: List[nn.Parameter] = []
        params.extend(list(self.private_point.parameters()))
        params.extend(list(self.private_kubric.parameters()))
        return params

    def private_point_parameters_list(self) -> List[nn.Parameter]:
        return list(self.private_point.parameters())

    def private_kubric_parameters_list(self) -> List[nn.Parameter]:
        return list(self.private_kubric.parameters())

    def _apply_private(
        self,
        h_t: torch.Tensor,
        dataset_ids: Optional[torch.Tensor],
        route_stats: Optional[Dict[str, int]] = None,
    ) -> torch.Tensor:
        if dataset_ids is None:
            return h_t

        if route_stats is not None:
            route_stats["calls"] = int(route_stats.get("calls", 0)) + 1
            route_stats["total_samples"] = int(route_stats.get("total_samples", 0)) + int(dataset_ids.numel())

        point_mask = dataset_ids == DATASET_ID_MAP["pointodyssey"]
        kubric_mask = dataset_ids == DATASET_ID_MAP["kubric"]
        other_mask = ~(point_mask | kubric_mask)

        if route_stats is not None:
            route_stats["point_private_hits"] = int(route_stats.get("point_private_hits", 0)) + int(point_mask.sum().item())
            route_stats["kubric_private_hits"] = int(route_stats.get("kubric_private_hits", 0)) + int(kubric_mask.sum().item())
            route_stats["other_hits"] = int(route_stats.get("other_hits", 0)) + int(other_mask.sum().item())

        delta = torch.zeros_like(h_t)
        if bool(point_mask.any()):
            delta[point_mask] = self.private_point(h_t[point_mask])
        if bool(kubric_mask.any()):
            delta[kubric_mask] = self.private_kubric(h_t[kubric_mask])

        return h_t + delta

    def rollout_teacher_forced(
        self,
        obs: torch.Tensor,
        target: torch.Tensor,
        dataset_ids: Optional[torch.Tensor] = None,
        route_stats: Optional[Dict[str, int]] = None,
    ) -> torch.Tensor:
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
            h_t = self._apply_private(h_t, dataset_ids=dataset_ids, route_stats=route_stats)
            preds.append(self.head(h_t))

        return torch.stack(preds, dim=1)

    def rollout_free(
        self,
        obs: torch.Tensor,
        fut_len: int,
        dataset_ids: Optional[torch.Tensor] = None,
        route_stats: Optional[Dict[str, int]] = None,
    ) -> torch.Tensor:
        _, h = self.obs_encoder(obs)
        h_t = h[-1]
        prev = obs[:, -1, :]
        preds: List[torch.Tensor] = []

        for _ in range(fut_len):
            h_t = self.roll_cell(prev, h_t)
            h_t = self._apply_private(h_t, dataset_ids=dataset_ids, route_stats=route_stats)
            pred = self.head(h_t)
            preds.append(pred)
            prev = pred

        return torch.stack(preds, dim=1)


def _param_count(params: Sequence[nn.Parameter]) -> int:
    return int(sum(p.numel() for p in params))


def dataset_mix_info(ds: Stage1UnifiedDataset) -> Dict[str, int]:
    counter: Counter[str] = Counter()
    for idx in ds.index_map:
        counter[idx.dataset_name] += 1
    return {k: int(v) for k, v in sorted(counter.items())}


def build_balanced_joint_loader(train_dataset: Stage1UnifiedDataset, batch_size: int) -> Tuple[DataLoader, Dict[str, Any]]:
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
    loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=0,
        collate_fn=stage1_collate_fn,
    )
    info = {
        "mode": "weighted_balanced_sampler",
        "source_counts": mix,
        "num_samples": len(train_dataset),
    }
    return loader, info


def _records_for(minisplits: Dict[str, Any], dataset: str, split: str) -> List[Dict[str, Any]]:
    datasets = minisplits.get("datasets", {}) if isinstance(minisplits, dict) else {}
    ds = datasets.get(dataset, {}) if isinstance(datasets, dict) else {}
    out = ds.get(split, []) if isinstance(ds, dict) else []
    return [x for x in out if isinstance(x, dict)]


def compute_source_task_losses(
    pred_teacher: torch.Tensor,
    pred_free: torch.Tensor,
    target: torch.Tensor,
    dataset_names: List[str],
    free_loss_weight: float,
) -> Dict[str, Any]:
    err_t = ((pred_teacher - target) ** 2).mean(dim=(1, 2))
    err_f = ((pred_free - target) ** 2).mean(dim=(1, 2))

    zero_with_grad = pred_teacher.sum() * 0.0
    source_out: Dict[str, Dict[str, Any]] = {}

    norm_t_list: List[torch.Tensor] = []
    norm_f_list: List[torch.Tensor] = []
    source_counts: Dict[str, int] = {}

    for source in SOURCE_NAMES:
        idx = [i for i, x in enumerate(dataset_names) if x == source]
        source_counts[source] = len(idx)
        if idx:
            idxt = torch.tensor(idx, dtype=torch.long, device=err_t.device)
            loss_t = err_t.index_select(0, idxt).mean()
            loss_f = err_f.index_select(0, idxt).mean()
            total = loss_t + free_loss_weight * loss_f
            norm_t_list.append(loss_t)
            norm_f_list.append(loss_f)
            source_out[source] = {
                "present": True,
                "count": len(idx),
                "teacher_loss": loss_t,
                "free_loss": loss_f,
                "total_loss": total,
                "teacher_loss_value": float(loss_t.detach().item()),
                "free_loss_value": float(loss_f.detach().item()),
                "total_loss_value": float(total.detach().item()),
            }
        else:
            source_out[source] = {
                "present": False,
                "count": 0,
                "teacher_loss": zero_with_grad,
                "free_loss": zero_with_grad,
                "total_loss": zero_with_grad,
                "teacher_loss_value": 0.0,
                "free_loss_value": 0.0,
                "total_loss_value": 0.0,
            }

    if norm_t_list:
        normalized_teacher = torch.stack(norm_t_list).mean()
        normalized_free = torch.stack(norm_f_list).mean()
    else:
        normalized_teacher = err_t.mean()
        normalized_free = err_f.mean()

    normalized_total = normalized_teacher + free_loss_weight * normalized_free

    return {
        "sources": source_out,
        "source_counts": source_counts,
        "normalized_teacher_loss": normalized_teacher,
        "normalized_free_loss": normalized_free,
        "normalized_total_loss": normalized_total,
    }


def _grad_norm_sq(grads: Sequence[Optional[torch.Tensor]], device: torch.device) -> torch.Tensor:
    out = torch.zeros([], device=device, dtype=torch.float32)
    for g in grads:
        if g is None:
            continue
        out = out + torch.sum(g.detach().to(torch.float32) ** 2)
    return out


def _grad_dot(g1: Sequence[Optional[torch.Tensor]], g2: Sequence[Optional[torch.Tensor]]) -> float:
    dot = 0.0
    for a, b in zip(g1, g2):
        if a is None or b is None:
            continue
        dot += float(torch.sum(a.detach() * b.detach()).item())
    return dot


def _project_grad(
    ga: Sequence[Optional[torch.Tensor]],
    gb: Sequence[Optional[torch.Tensor]],
    coeff: float,
) -> List[Optional[torch.Tensor]]:
    out: List[Optional[torch.Tensor]] = []
    for a, b in zip(ga, gb):
        if a is None:
            out.append(None)
            continue
        if b is None:
            out.append(a)
            continue
        out.append(a - coeff * b)
    return out


def _merge_grads(
    g1: Sequence[Optional[torch.Tensor]],
    g2: Sequence[Optional[torch.Tensor]],
) -> List[Optional[torch.Tensor]]:
    out: List[Optional[torch.Tensor]] = []
    for a, b in zip(g1, g2):
        if a is None and b is None:
            out.append(None)
        elif a is None:
            out.append(0.5 * b)
        elif b is None:
            out.append(0.5 * a)
        else:
            out.append(0.5 * (a + b))
    return out


def _assign_grads(params: Sequence[nn.Parameter], grads: Sequence[Optional[torch.Tensor]]) -> None:
    for p, g in zip(params, grads):
        if g is None:
            p.grad = None
        else:
            p.grad = g.detach().clone()


def pcgrad_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    shared_params: Sequence[nn.Parameter],
    private_params: Sequence[nn.Parameter],
    loss_point: torch.Tensor,
    loss_kubric: torch.Tensor,
    clip_grad_norm: float,
) -> Dict[str, Any]:
    del model
    eps = 1e-12

    optimizer.zero_grad(set_to_none=True)

    g_point = torch.autograd.grad(loss_point, shared_params, retain_graph=True, allow_unused=True)
    g_kubric = torch.autograd.grad(loss_kubric, shared_params, retain_graph=True, allow_unused=True)

    dot = _grad_dot(g_point, g_kubric)
    norm_point_sq = float(_grad_norm_sq(g_point, shared_params[0].device).item()) if shared_params else 0.0
    norm_kubric_sq = float(_grad_norm_sq(g_kubric, shared_params[0].device).item()) if shared_params else 0.0

    conflict = dot < 0.0
    applied = False

    g_point_proj = list(g_point)
    g_kubric_proj = list(g_kubric)

    if conflict and norm_point_sq > eps and norm_kubric_sq > eps:
        coeff_point = dot / (norm_kubric_sq + eps)
        coeff_kubric = dot / (norm_point_sq + eps)
        g_point_proj = _project_grad(g_point, g_kubric, coeff_point)
        g_kubric_proj = _project_grad(g_kubric, g_point, coeff_kubric)
        applied = True

    merged_shared = _merge_grads(g_point_proj, g_kubric_proj)
    _assign_grads(shared_params, merged_shared)

    if private_params:
        combined_private_loss = loss_point + loss_kubric
        g_private = torch.autograd.grad(combined_private_loss, private_params, retain_graph=False, allow_unused=True)
        _assign_grads(private_params, g_private)

    all_params = list(shared_params) + list(private_params)
    if all_params:
        torch.nn.utils.clip_grad_norm_(all_params, max_norm=clip_grad_norm)
    optimizer.step()

    merged_norm_sq = float(_grad_norm_sq(merged_shared, shared_params[0].device).item()) if shared_params else 0.0

    return {
        "pcgrad_applied": bool(applied),
        "pcgrad_conflict": bool(conflict),
        "pcgrad_dot": float(dot),
        "pcgrad_point_norm": float(np.sqrt(max(norm_point_sq, 0.0))),
        "pcgrad_kubric_norm": float(np.sqrt(max(norm_kubric_sq, 0.0))),
        "pcgrad_merged_shared_norm": float(np.sqrt(max(merged_norm_sq, 0.0))),
    }


def gradnorm_step(
    optimizer: torch.optim.Optimizer,
    shared_params: Sequence[nn.Parameter],
    loss_point: torch.Tensor,
    loss_kubric: torch.Tensor,
    state: Dict[str, Any],
    alpha: float,
    weight_lr: float,
    clip_grad_norm: float,
) -> Dict[str, Any]:
    eps = 1e-12
    weights: torch.Tensor = state["weights"]

    if state.get("initial_losses") is None:
        init_losses = torch.stack([loss_point.detach(), loss_kubric.detach()])
        state["initial_losses"] = torch.clamp(init_losses, min=1e-8)

    initial_losses: torch.Tensor = state["initial_losses"]
    weights_before = weights.detach().clone()

    g_point = torch.autograd.grad(weights[0] * loss_point, shared_params, retain_graph=True, allow_unused=True)
    g_kubric = torch.autograd.grad(weights[1] * loss_kubric, shared_params, retain_graph=True, allow_unused=True)

    norm_point = torch.sqrt(_grad_norm_sq(g_point, shared_params[0].device) + eps)
    norm_kubric = torch.sqrt(_grad_norm_sq(g_kubric, shared_params[0].device) + eps)

    loss_ratios = torch.stack([loss_point.detach(), loss_kubric.detach()]) / torch.clamp(initial_losses, min=1e-8)
    inv_rate = loss_ratios / torch.clamp(loss_ratios.mean(), min=1e-8)

    norm_avg = 0.5 * (norm_point.detach() + norm_kubric.detach())
    targets = norm_avg * torch.pow(inv_rate, alpha)

    weighted_total = weights[0] * loss_point + weights[1] * loss_kubric

    optimizer.zero_grad(set_to_none=True)
    weighted_total.backward()
    torch.nn.utils.clip_grad_norm_(list(shared_params), max_norm=clip_grad_norm)
    optimizer.step()

    with torch.no_grad():
        current_norms = torch.stack([norm_point.detach(), norm_kubric.detach()])
        ratio = torch.pow(torch.clamp(targets, min=1e-8) / torch.clamp(current_norms, min=1e-8), weight_lr)
        new_weights = weights * ratio
        new_weights = torch.clamp(new_weights, min=1e-3, max=1e3)
        new_weights = 2.0 * new_weights / torch.clamp(new_weights.sum(), min=1e-8)
        state["weights"] = new_weights

    return {
        "gradnorm_weights_before": [float(weights_before[0].item()), float(weights_before[1].item())],
        "gradnorm_weights_after": [float(state["weights"][0].item()), float(state["weights"][1].item())],
        "gradnorm_source_norm_current": [float(norm_point.item()), float(norm_kubric.item())],
        "gradnorm_source_norm_target": [float(targets[0].item()), float(targets[1].item())],
        "gradnorm_alpha": float(alpha),
        "gradnorm_weight_lr": float(weight_lr),
    }


def evaluate_loader(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    free_loss_weight: float,
    max_batches: int = 0,
    route_stats: Optional[Dict[str, int]] = None,
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

            pred_t = model.rollout_teacher_forced(obs, fut, dataset_ids=dataset_ids, route_stats=route_stats)
            pred_f = model.rollout_free(obs, fut.shape[1], dataset_ids=dataset_ids, route_stats=route_stats)

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


def _eval_dataset_state_space(
    model: nn.Module,
    dataset: Dataset,
    state_dims: Tuple[int, int],
    device: torch.device,
    max_samples: int,
    route_stats: Optional[Dict[str, int]] = None,
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

            pred_t = model.rollout_teacher_forced(obs, fut, dataset_ids=dataset_ids, route_stats=route_stats)
            pred_f = model.rollout_free(obs, fut.shape[1], dataset_ids=dataset_ids, route_stats=route_stats)

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
    route_stats: Optional[Dict[str, int]] = None,
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
        route_stats=route_stats,
    )
    tapvid3d_metrics = _eval_dataset_state_space(
        model=model,
        dataset=tapvid3d_ds,
        state_dims=(2, 5),
        device=device,
        max_samples=max_tapvid3d_samples,
        route_stats=route_stats,
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
    route_stats: Optional[Dict[str, int]] = None,
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
        "pointodyssey_val": evaluate_loader(
            model,
            point_loader,
            device,
            free_loss_weight=free_loss_weight,
            max_batches=0,
            route_stats=route_stats,
        ),
        "kubric_val": evaluate_loader(
            model,
            kubric_loader,
            device,
            free_loss_weight=free_loss_weight,
            max_batches=0,
            route_stats=route_stats,
        ),
    }


def build_parser() -> ArgumentParser:
    p = ArgumentParser(description="TraceWM Stage1 final joint rescue trainer")
    p.add_argument(
        "--train-mode",
        required=True,
        choices=[
            "pcgrad",
            "gradnorm",
            "shared_private",
            "shared_private_plus_pcgrad",
            "shared_private_plus_gradnorm",
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
    p.add_argument("--private-adapter-dim", type=int, default=16)
    p.add_argument("--gradnorm-alpha", type=float, default=1.5)
    p.add_argument("--gradnorm-weight-lr", type=float, default=0.025)
    p.add_argument("--clip-grad-norm", type=float, default=1.0)
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
    gradnorm_state: Optional[Dict[str, Any]],
    args: Any,
) -> Dict[str, Any]:
    return {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "history": history,
        "best_val_total_loss": best_val_total_loss,
        "gradnorm_state": gradnorm_state,
        "args": vars(args),
    }


def _mean(xs: List[float]) -> float:
    return float(sum(xs) / max(len(xs), 1))


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

    train_loader, sampler_info = build_balanced_joint_loader(train_dataset, batch_size=args.batch_size)
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=stage1_collate_fn,
    )

    use_pcgrad = args.train_mode in {"pcgrad", "shared_private_plus_pcgrad"}
    use_gradnorm = args.train_mode in {"gradnorm", "shared_private_plus_gradnorm"}
    use_shared_private = args.train_mode in {
        "shared_private",
        "shared_private_plus_pcgrad",
        "shared_private_plus_gradnorm",
    }

    if use_shared_private:
        model: nn.Module = SharedPrivateTraceModel(
            state_dim=5,
            hidden_dim=args.hidden_dim,
            adapter_dim=args.private_adapter_dim,
        ).to(device)
    else:
        model = BaseTraceModel(state_dim=5, hidden_dim=args.hidden_dim).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if hasattr(model, "shared_parameters_list"):
        shared_params = list(model.shared_parameters_list())
    else:
        shared_params = list(model.parameters())

    if hasattr(model, "private_parameters_list"):
        private_params = list(model.private_parameters_list())
    else:
        private_params = []

    total_param_count = _param_count(list(model.parameters()))
    shared_param_count = _param_count(shared_params)
    private_param_count = _param_count(private_params)

    private_point_param_count = 0
    private_kubric_param_count = 0
    if use_shared_private and hasattr(model, "private_point_parameters_list") and hasattr(model, "private_kubric_parameters_list"):
        private_point_param_count = _param_count(list(model.private_point_parameters_list()))
        private_kubric_param_count = _param_count(list(model.private_kubric_parameters_list()))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    start_epoch = 0
    global_step = 0
    history: List[Dict[str, Any]] = []
    best_val_total_loss = float("inf")

    gradnorm_state: Optional[Dict[str, Any]] = None
    if use_gradnorm:
        gradnorm_state = {
            "weights": torch.tensor([1.0, 1.0], dtype=torch.float32, device=device),
            "initial_losses": None,
        }

    latest_ckpt = ckpt_dir / "latest.pt"
    resume_path = Path(args.resume) if args.resume.strip() else None
    if resume_path is not None and resume_path.exists():
        payload = torch.load(resume_path, map_location="cpu")
        model.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        start_epoch = int(payload.get("epoch", -1)) + 1
        global_step = int(payload.get("global_step", 0))
        history = list(payload.get("history", []))
        best_val_total_loss = float(payload.get("best_val_total_loss", float("inf")))
        if use_gradnorm:
            restored = payload.get("gradnorm_state")
            if isinstance(restored, dict) and "weights" in restored:
                gradnorm_state = {
                    "weights": restored["weights"].to(device=device, dtype=torch.float32),
                    "initial_losses": restored.get("initial_losses"),
                }
                if isinstance(gradnorm_state["initial_losses"], torch.Tensor):
                    gradnorm_state["initial_losses"] = gradnorm_state["initial_losses"].to(device=device, dtype=torch.float32)
        print(f"[final_rescue] resumed from --resume: {resume_path}")
    elif args.auto_resume and latest_ckpt.exists():
        payload = torch.load(latest_ckpt, map_location="cpu")
        model.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        start_epoch = int(payload.get("epoch", -1)) + 1
        global_step = int(payload.get("global_step", 0))
        history = list(payload.get("history", []))
        best_val_total_loss = float(payload.get("best_val_total_loss", float("inf")))
        if use_gradnorm:
            restored = payload.get("gradnorm_state")
            if isinstance(restored, dict) and "weights" in restored:
                gradnorm_state = {
                    "weights": restored["weights"].to(device=device, dtype=torch.float32),
                    "initial_losses": restored.get("initial_losses"),
                }
                if isinstance(gradnorm_state["initial_losses"], torch.Tensor):
                    gradnorm_state["initial_losses"] = gradnorm_state["initial_losses"].to(device=device, dtype=torch.float32)
        print(f"[final_rescue] resumed from latest: {latest_ckpt}")

    train_iter = itertools.cycle(train_loader)

    for epoch in range(start_epoch, args.epochs):
        model.train()

        train_teacher_vals: List[float] = []
        train_free_vals: List[float] = []
        train_total_vals: List[float] = []
        train_point_vals: List[float] = []
        train_kubric_vals: List[float] = []
        source_counts_last: Dict[str, int] = {"pointodyssey": 0, "kubric": 0}

        pcgrad_applied_vals: List[float] = []
        pcgrad_conflict_vals: List[float] = []
        pcgrad_dot_vals: List[float] = []
        pcgrad_norm_point_vals: List[float] = []
        pcgrad_norm_kubric_vals: List[float] = []

        gradnorm_w_point_vals: List[float] = []
        gradnorm_w_kubric_vals: List[float] = []
        gradnorm_n_point_vals: List[float] = []
        gradnorm_n_kubric_vals: List[float] = []
        gradnorm_t_point_vals: List[float] = []
        gradnorm_t_kubric_vals: List[float] = []

        route_stats_train = empty_route_stats()

        for _ in range(args.steps_per_epoch):
            batch = next(train_iter)
            obs, fut = build_state_batch(batch, device)
            dataset_ids = dataset_ids_from_batch(batch, device)
            dataset_names = [str(x) for x in batch.get("dataset", [])] if isinstance(batch.get("dataset"), list) else []

            pred_teacher = model.rollout_teacher_forced(obs, fut, dataset_ids=dataset_ids, route_stats=route_stats_train)
            pred_free = model.rollout_free(obs, fut.shape[1], dataset_ids=dataset_ids, route_stats=route_stats_train)

            loss_pack = compute_source_task_losses(
                pred_teacher=pred_teacher,
                pred_free=pred_free,
                target=fut,
                dataset_names=dataset_names,
                free_loss_weight=args.free_loss_weight,
            )

            loss_point = loss_pack["sources"]["pointodyssey"]["total_loss"]
            loss_kubric = loss_pack["sources"]["kubric"]["total_loss"]
            loss_total = loss_pack["normalized_total_loss"]
            loss_teacher = loss_pack["normalized_teacher_loss"]
            loss_free = loss_pack["normalized_free_loss"]
            source_counts_last = dict(loss_pack["source_counts"])

            if use_pcgrad:
                step_stats = pcgrad_step(
                    model=model,
                    optimizer=optimizer,
                    shared_params=shared_params,
                    private_params=private_params,
                    loss_point=loss_point,
                    loss_kubric=loss_kubric,
                    clip_grad_norm=args.clip_grad_norm,
                )
                pcgrad_applied_vals.append(1.0 if step_stats["pcgrad_applied"] else 0.0)
                pcgrad_conflict_vals.append(1.0 if step_stats["pcgrad_conflict"] else 0.0)
                pcgrad_dot_vals.append(float(step_stats["pcgrad_dot"]))
                pcgrad_norm_point_vals.append(float(step_stats["pcgrad_point_norm"]))
                pcgrad_norm_kubric_vals.append(float(step_stats["pcgrad_kubric_norm"]))
            elif use_gradnorm:
                if gradnorm_state is None:
                    raise RuntimeError("gradnorm_state must be initialized when gradnorm is enabled")
                step_stats = gradnorm_step(
                    optimizer=optimizer,
                    shared_params=shared_params,
                    loss_point=loss_point,
                    loss_kubric=loss_kubric,
                    state=gradnorm_state,
                    alpha=args.gradnorm_alpha,
                    weight_lr=args.gradnorm_weight_lr,
                    clip_grad_norm=args.clip_grad_norm,
                )
                gradnorm_w_point_vals.append(float(step_stats["gradnorm_weights_before"][0]))
                gradnorm_w_kubric_vals.append(float(step_stats["gradnorm_weights_before"][1]))
                gradnorm_n_point_vals.append(float(step_stats["gradnorm_source_norm_current"][0]))
                gradnorm_n_kubric_vals.append(float(step_stats["gradnorm_source_norm_current"][1]))
                gradnorm_t_point_vals.append(float(step_stats["gradnorm_source_norm_target"][0]))
                gradnorm_t_kubric_vals.append(float(step_stats["gradnorm_source_norm_target"][1]))
            else:
                optimizer.zero_grad(set_to_none=True)
                loss_total.backward()
                torch.nn.utils.clip_grad_norm_(list(model.parameters()), max_norm=args.clip_grad_norm)
                optimizer.step()

            train_teacher_vals.append(float(loss_teacher.detach().item()))
            train_free_vals.append(float(loss_free.detach().item()))
            train_total_vals.append(float(loss_total.detach().item()))
            train_point_vals.append(float(loss_pack["sources"]["pointodyssey"]["total_loss_value"]))
            train_kubric_vals.append(float(loss_pack["sources"]["kubric"]["total_loss_value"]))

            global_step += 1

        route_stats_eval = empty_route_stats()
        val_metrics = evaluate_loader(
            model=model,
            loader=val_loader,
            device=device,
            free_loss_weight=args.free_loss_weight,
            max_batches=args.eval_max_batches,
            route_stats=route_stats_eval,
        )
        external_metrics = evaluate_external(
            model=model,
            splits_path=args.splits_path,
            data_root=args.data_root,
            device=device,
            max_tapvid_samples=args.eval_max_tapvid_samples,
            max_tapvid3d_samples=args.eval_max_tapvid3d_samples,
            route_stats=route_stats_eval,
        )
        val_source_breakdown = evaluate_joint_source_breakdown(
            model=model,
            data_root=args.data_root,
            splits_path=args.splits_path,
            device=device,
            batch_size=args.batch_size,
            free_loss_weight=args.free_loss_weight,
            route_stats=route_stats_eval,
        )

        train_loss_aux: Dict[str, Any] = {
            "loss_mode": "dataset_normalized",
            "source_group_counts_last": source_counts_last,
            "source_losses_epoch_mean": {
                "pointodyssey_total_loss": _mean(train_point_vals),
                "kubric_total_loss": _mean(train_kubric_vals),
            },
            "pcgrad_applied": bool(use_pcgrad),
            "gradnorm_applied": bool(use_gradnorm),
        }

        if use_pcgrad:
            train_loss_aux["pcgrad_stats"] = {
                "conflict_fraction": _mean(pcgrad_conflict_vals),
                "applied_fraction": _mean(pcgrad_applied_vals),
                "dot_mean": _mean(pcgrad_dot_vals),
                "dot_min": float(min(pcgrad_dot_vals) if pcgrad_dot_vals else 0.0),
                "point_norm_mean": _mean(pcgrad_norm_point_vals),
                "kubric_norm_mean": _mean(pcgrad_norm_kubric_vals),
            }
        if use_gradnorm:
            train_loss_aux["gradnorm_stats"] = {
                "source_weight_mean": {
                    "pointodyssey": _mean(gradnorm_w_point_vals),
                    "kubric": _mean(gradnorm_w_kubric_vals),
                },
                "source_norm_current_mean": {
                    "pointodyssey": _mean(gradnorm_n_point_vals),
                    "kubric": _mean(gradnorm_n_kubric_vals),
                },
                "source_norm_target_mean": {
                    "pointodyssey": _mean(gradnorm_t_point_vals),
                    "kubric": _mean(gradnorm_t_kubric_vals),
                },
            }

        routing_checks = {
            "shared_private_enabled": use_shared_private,
            "train_point_private_used": bool(route_stats_train.get("point_private_hits", 0) > 0) if use_shared_private else None,
            "train_kubric_private_used": bool(route_stats_train.get("kubric_private_hits", 0) > 0) if use_shared_private else None,
            "eval_point_private_used": bool(route_stats_eval.get("point_private_hits", 0) > 0) if use_shared_private else None,
            "eval_kubric_private_used": bool(route_stats_eval.get("kubric_private_hits", 0) > 0) if use_shared_private else None,
        }

        epoch_metrics: Dict[str, Any] = {
            "epoch": epoch,
            "global_step": global_step,
            "train_teacher_forced_loss": _mean(train_teacher_vals),
            "train_free_rollout_loss": _mean(train_free_vals),
            "train_total_loss": _mean(train_total_vals),
            "steps_this_epoch": args.steps_per_epoch,
            "train_loss_aux": train_loss_aux,
            **val_metrics,
            "tapvid": external_metrics["tapvid"],
            "tapvid3d": external_metrics["tapvid3d"],
            "val_source_breakdown": val_source_breakdown,
            "routing_train_epoch": _copy_route_stats(route_stats_train),
            "routing_eval_epoch": _copy_route_stats(route_stats_eval),
            "routing_checks": routing_checks,
        }
        history.append(epoch_metrics)

        payload = _checkpoint_payload(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            global_step=global_step,
            history=history,
            best_val_total_loss=best_val_total_loss,
            gradnorm_state=gradnorm_state,
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
                gradnorm_state=gradnorm_state,
                args=args,
            )
            torch.save(payload_best, ckpt_dir / "best.pt")

        print(
            f"[final_rescue][{args.run_name}] epoch={epoch} step={global_step} "
            f"train_total={epoch_metrics['train_total_loss']:.6f} "
            f"val_total={val_metrics['val_total_loss']:.6f} "
            f"point_loss={epoch_metrics['train_loss_aux']['source_losses_epoch_mean']['pointodyssey_total_loss']:.6f} "
            f"kubric_loss={epoch_metrics['train_loss_aux']['source_losses_epoch_mean']['kubric_total_loss']:.6f} "
            f"tapvid_free_ep={external_metrics['tapvid']['free_rollout_endpoint_l2']:.6f} "
            f"tapvid3d_free_ep={external_metrics['tapvid3d']['free_rollout_endpoint_l2']:.6f}"
        )

    final_metrics = history[-1] if history else {}

    summary = {
        "generated_at_utc": now_iso(),
        "task": "trace_only_future_trace_state_generation",
        "round": "stage1_final_joint_rescue",
        "run_name": args.run_name,
        "train_mode": args.train_mode,
        "method_flags": {
            "pcgrad": use_pcgrad,
            "gradnorm": use_gradnorm,
            "shared_private_adapters": use_shared_private,
            "source_conditioning": False,
            "warmup": False,
        },
        "base_recipe_lock": {
            "balanced_sampler": True,
            "loss_normalized": True,
            "same_stage1_data_contract": True,
            "same_eval_protocol": True,
        },
        "forbidden_components": {
            "source_conditioning": False,
            "warmup": False,
            "stage2_semantics": False,
            "wan": False,
            "motioncrafter_vae": False,
            "dynamic_replica": False,
            "new_data": False,
            "video_reconstruction": False,
        },
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
        "model_info": {
            "type": "shared_private_trace_model" if use_shared_private else "base_trace_model",
            "state_dim": 5,
            "hidden_dim": args.hidden_dim,
            "private_adapter_dim": args.private_adapter_dim if use_shared_private else 0,
            "total_parameter_count": total_param_count,
            "shared_parameter_count": shared_param_count,
            "private_parameter_count": private_param_count,
            "private_point_parameter_count": private_point_param_count,
            "private_kubric_parameter_count": private_kubric_param_count,
        },
        "sampler_info": sampler_info,
        "loss_family": {
            "teacher_forced_loss": "mse",
            "free_rollout_loss": "mse",
            "total_loss": f"teacher_forced + {args.free_loss_weight} * free_rollout",
            "free_loss_weight": args.free_loss_weight,
            "loss_mode": "dataset_normalized",
            "source_tasks": ["pointodyssey", "kubric"],
        },
        "gradnorm_config": {
            "enabled": use_gradnorm,
            "alpha": args.gradnorm_alpha,
            "weight_lr": args.gradnorm_weight_lr,
        },
        "pcgrad_config": {
            "enabled": use_pcgrad,
            "projection_scope": "shared_parameters_only",
            "private_projection": "disabled",
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

    print(f"[final_rescue][{args.run_name}] wrote summary: {summary_path}")
    print(f"[final_rescue][{args.run_name}] checkpoints: {ckpt_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
