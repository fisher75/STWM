#!/usr/bin/env python3
from __future__ import annotations

from argparse import ArgumentParser
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple
import json
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from stwm.tracewm_v2.constants import STATE_DIM
from stwm.tracewm_v2.datasets.stage1_v2_unified import Stage1V2UnifiedDataset, stage1_v2_collate_fn
from stwm.tracewm_v2.losses.structured_trace_loss import StructuredTraceLoss, StructuredTraceLossConfig
from stwm.tracewm_v2.models.causal_trace_transformer import (
    TraceCausalTransformer,
    build_tracewm_v2_config,
    estimate_parameter_count,
)
from stwm.tracewm_v2.models.gru_trace_baseline import GRUTraceBaseline, GRUTraceBaselineConfig


DATE_TAG = "20260408"

TAP_STATUS_AVAILABLE = "available_and_run"
TAP_STATUS_NOT_IMPLEMENTED = "not_implemented_yet"
TAP_STATUS_DATA_NOT_READY = "data_not_ready"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _read_json(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def parse_args() -> Any:
    parser = ArgumentParser(description="Stage1-v2 scientific rigor-fix revalidation round")
    parser.add_argument("--contract-path", default="/home/chen034/workspace/data/_manifests/stage1_v2_trace_cache_contract_20260408.json")
    parser.add_argument("--recommended-runtime-json", default="/home/chen034/workspace/stwm/reports/stage1_v2_recommended_runtime_20260408.json")
    parser.add_argument("--stage1-minisplit-path", default="/home/chen034/workspace/data/_manifests/stage1_minisplits_20260408.json")
    parser.add_argument("--data-root", default="/home/chen034/workspace/data")

    parser.add_argument("--state-report-json", default="/home/chen034/workspace/stwm/reports/stage1_v2_ablation_state_20260408.json")
    parser.add_argument("--state-report-md", default="/home/chen034/workspace/stwm/docs/STAGE1_V2_ABLATION_STATE_20260408.md")
    parser.add_argument("--backbone-report-json", default="/home/chen034/workspace/stwm/reports/stage1_v2_ablation_backbone_20260408.json")
    parser.add_argument("--backbone-report-md", default="/home/chen034/workspace/stwm/docs/STAGE1_V2_ABLATION_BACKBONE_20260408.md")
    parser.add_argument("--losses-report-json", default="/home/chen034/workspace/stwm/reports/stage1_v2_ablation_losses_20260408.json")
    parser.add_argument("--losses-report-md", default="/home/chen034/workspace/stwm/docs/STAGE1_V2_ABLATION_LOSSES_20260408.md")

    parser.add_argument("--mainline-replay-json", default="/home/chen034/workspace/stwm/reports/stage1_v2_mainline_replay_20260408.json")
    parser.add_argument("--mainline-replay-md", default="/home/chen034/workspace/stwm/docs/STAGE1_V2_MAINLINE_REPLAY_20260408.md")

    parser.add_argument("--final-report-json", default="/home/chen034/workspace/stwm/reports/stage1_v2_final_comparison_20260408.json")
    parser.add_argument("--final-report-md", default="/home/chen034/workspace/stwm/docs/TRACEWM_STAGE1_V2_RESULTS_20260408.md")
    parser.add_argument("--run-details-json", default="/home/chen034/workspace/stwm/reports/stage1_v2_scientific_revalidation_runs_20260408.json")

    parser.add_argument("--dataset-names", nargs="*", default=["pointodyssey", "kubric"])
    parser.add_argument("--obs-len", type=int, default=8)
    parser.add_argument("--fut-len", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--max-samples-per-dataset-train", type=int, default=128)
    parser.add_argument("--max-samples-per-dataset-val", type=int, default=64)

    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--steps-per-variant", type=int, default=8)
    parser.add_argument("--eval-steps", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--clip-grad-norm", type=float, default=1.0)

    parser.add_argument("--mainline-replay-steps", type=int, default=12)
    parser.add_argument("--mainline-replay-eval-steps", type=int, default=8)

    parser.add_argument("--eval-max-tapvid-samples", type=int, default=6)
    parser.add_argument("--eval-max-tapvid3d-samples", type=int, default=12)

    parser.add_argument("--gru-hidden-dim", type=int, default=384)
    parser.add_argument("--gru-num-layers", type=int, default=2)

    parser.add_argument("--state-min-win-delta", type=float, default=0.005)
    parser.add_argument("--backbone-min-win-delta", type=float, default=0.005)

    parser.add_argument("--seed", type=int, default=20260408)
    return parser.parse_args()


@dataclass
class RuntimeConfig:
    num_workers: int
    pin_memory: bool
    persistent_workers: bool
    prefetch_factor: int
    selected_gpu_id: int
    required_mem_gb: float
    safety_margin_gb: float
    single_gpu_only: bool


@dataclass
class VariantSpec:
    run_name: str
    group: str
    state_variant: str
    backbone_variant: str
    model_preset: str
    loss_variant: str
    enable_visibility: bool
    enable_residual: bool
    enable_velocity: bool
    enable_endpoint: bool
    notes: str


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return float(default)
        return float(x)
    except Exception:
        return float(default)


def _load_runtime_config(path: str | Path) -> RuntimeConfig:
    raw = _read_json(path)
    selected_policy = raw.get("selected_gpu_policy", {}) if isinstance(raw.get("selected_gpu_policy", {}), dict) else {}
    return RuntimeConfig(
        num_workers=int(raw.get("recommended_num_workers", 8) or 8),
        pin_memory=bool(raw.get("recommended_pin_memory", True)),
        persistent_workers=bool(raw.get("recommended_persistent_workers", True)),
        prefetch_factor=int(raw.get("recommended_prefetch_factor", 4) or 4),
        selected_gpu_id=int(selected_policy.get("selected_gpu_id", -1)),
        required_mem_gb=float(raw.get("required_mem_gb", 40.0) or 40.0),
        safety_margin_gb=float(raw.get("safety_margin_gb", 8.0) or 8.0),
        single_gpu_only=bool(raw.get("single_gpu_only", True)),
    )


def _to_device(batch: Dict[str, Any], device: torch.device, non_blocking: bool) -> Dict[str, torch.Tensor]:
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


def _build_loss_config(spec: VariantSpec) -> StructuredTraceLossConfig:
    return StructuredTraceLossConfig(
        coord_weight=1.0,
        visibility_weight=0.5,
        residual_weight=0.25,
        velocity_weight=0.25,
        endpoint_weight=0.1,
        enable_visibility=bool(spec.enable_visibility),
        enable_residual=bool(spec.enable_residual),
        enable_velocity=bool(spec.enable_velocity),
        enable_endpoint=bool(spec.enable_endpoint),
    )


def _build_model(spec: VariantSpec, max_tokens: int, gru_hidden_dim: int, gru_num_layers: int) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    if spec.backbone_variant == "stage1_v2_backbone_multitoken_gru" or spec.backbone_variant.endswith("_gru"):
        state_mode = "legacy_mean5d" if spec.state_variant == "legacy_mean5d" else "multitoken"
        cfg = GRUTraceBaselineConfig(
            state_mode=state_mode,
            max_tokens=int(max_tokens),
            state_dim=STATE_DIM,
            hidden_dim=int(gru_hidden_dim),
            num_layers=int(gru_num_layers),
            dropout=0.1,
            use_endpoint_head=True,
        )
        model = GRUTraceBaseline(cfg)
        params = int(model.parameter_count())
        meta = {
            "family": "gru",
            "preset": "gru_baseline",
            "config": cfg.__dict__,
            "parameter_count": params,
            "estimated_parameter_count": params,
            "target_220m_range_pass": bool(200_000_000 <= params <= 240_000_000),
        }
        return model, meta

    preset = "prototype_220m" if "prototype220m" in spec.backbone_variant or "prototype_220m" in spec.backbone_variant else "debug_small"
    cfg = build_tracewm_v2_config(preset)
    cfg.use_endpoint_head = True
    model = TraceCausalTransformer(cfg)
    params = int(sum(p.numel() for p in model.parameters()))
    estimated = int(estimate_parameter_count(cfg))
    meta = {
        "family": "transformer",
        "preset": preset,
        "config": cfg.__dict__,
        "parameter_count": params,
        "estimated_parameter_count": estimated,
        "target_220m_range_pass": bool(200_000_000 <= estimated <= 240_000_000),
    }
    return model, meta


def _masked_sum(value: torch.Tensor, mask: torch.Tensor) -> Tuple[float, float]:
    masked = value * mask
    return float(masked.sum().detach().cpu().item()), float(mask.sum().detach().cpu().item())


def _compose_state_from_pred(pred: Dict[str, torch.Tensor], index_t: int, state_dim: int) -> torch.Tensor:
    coord_t = pred["coord"][:, index_t]
    bsz, k_len, _ = coord_t.shape
    out = torch.zeros((bsz, k_len, state_dim), dtype=coord_t.dtype, device=coord_t.device)
    out[..., 0:2] = coord_t

    if "z" in pred:
        out[..., 2:3] = pred["z"][:, index_t]
    if "vis_logit" in pred:
        out[..., 3:4] = torch.sigmoid(pred["vis_logit"][:, index_t])
    if "velocity" in pred:
        out[..., 4:6] = pred["velocity"][:, index_t]
    if "residual" in pred:
        out[..., 6:8] = pred["residual"][:, index_t]
    return out


def _free_rollout_coords(
    model: torch.nn.Module,
    obs_state: torch.Tensor,
    token_mask: torch.Tensor,
    fut_len: int,
) -> torch.Tensor:
    bsz, obs_len, k_len, d = obs_state.shape
    total_len = obs_len + int(fut_len)

    generated = torch.zeros((bsz, total_len, k_len, d), dtype=obs_state.dtype, device=obs_state.device)
    generated[:, :obs_len] = obs_state

    for step in range(int(fut_len)):
        t_abs = obs_len + step
        prefix_len = t_abs + 1
        shifted = torch.zeros((bsz, prefix_len, k_len, d), dtype=obs_state.dtype, device=obs_state.device)
        shifted[:, 0] = generated[:, 0]
        if prefix_len > 1:
            shifted[:, 1:] = generated[:, :t_abs]
        pred = model(shifted, token_mask=token_mask)
        generated[:, t_abs] = _compose_state_from_pred(pred=pred, index_t=t_abs, state_dim=d)

    return generated[:, obs_len:, :, 0:2]


def _as_numpy(x: Any, dtype: Any | None = None) -> np.ndarray | None:
    if x is None:
        return None
    if isinstance(x, np.ndarray):
        arr = x
    elif isinstance(x, torch.Tensor):
        arr = x.detach().cpu().numpy()
    else:
        arr = np.asarray(x)
    if dtype is not None:
        arr = arr.astype(dtype, copy=False)
    return arr


def _select_tokens(valid: np.ndarray, max_tokens: int) -> np.ndarray:
    score = valid.sum(axis=0)
    idx = np.argsort(-score)
    if int(max_tokens) > 0:
        idx = idx[: int(max_tokens)]
    return idx


def _build_state_features(
    tracks_2d: np.ndarray,
    tracks_3d: np.ndarray,
    valid: np.ndarray,
    visibility: np.ndarray,
) -> np.ndarray:
    t_len, k_len, _ = tracks_2d.shape
    state = np.zeros((t_len, k_len, STATE_DIM), dtype=np.float32)

    coord = tracks_2d.astype(np.float32, copy=False)
    z = tracks_3d[..., 2].astype(np.float32, copy=False)
    vis = visibility.astype(np.float32, copy=False)

    vel = np.zeros_like(coord, dtype=np.float32)
    vel[1:] = coord[1:] - coord[:-1]
    residual = coord - coord[0:1]

    state[..., 0:2] = coord
    state[..., 2] = z
    state[..., 3] = vis
    state[..., 4:6] = vel
    state[..., 6:8] = residual

    mask = valid.astype(bool)
    state[..., 0:3] = np.where(mask[..., None], state[..., 0:3], 0.0)
    state[..., 4:8] = np.where(mask[..., None], state[..., 4:8], 0.0)
    state[..., 3] = np.where(mask, state[..., 3], 0.0)
    return state


def _extract_external_state_sample(
    sample: Dict[str, Any],
    obs_len: int,
    fut_len: int,
    max_tokens: int,
) -> Dict[str, np.ndarray] | None:
    obs_2d = _as_numpy(sample.get("obs_tracks_2d"), np.float32)
    fut_2d = _as_numpy(sample.get("fut_tracks_2d"), np.float32)
    obs_3d = _as_numpy(sample.get("obs_tracks_3d"), np.float32)
    fut_3d = _as_numpy(sample.get("fut_tracks_3d"), np.float32)

    if obs_2d is None or fut_2d is None:
        if obs_3d is None or fut_3d is None:
            return None
        obs_2d = obs_3d[..., 0:2]
        fut_2d = fut_3d[..., 0:2]

    if obs_3d is None or fut_3d is None:
        if obs_2d is None or fut_2d is None:
            return None
        obs_3d = np.zeros((obs_2d.shape[0], obs_2d.shape[1], 3), dtype=np.float32)
        fut_3d = np.zeros((fut_2d.shape[0], fut_2d.shape[1], 3), dtype=np.float32)
        obs_3d[..., 0:2] = obs_2d
        fut_3d[..., 0:2] = fut_2d

    if obs_2d.ndim != 3 or fut_2d.ndim != 3:
        return None
    if obs_3d.ndim != 3 or fut_3d.ndim != 3:
        return None

    tracks_2d = np.concatenate([obs_2d, fut_2d], axis=0).astype(np.float32, copy=False)
    tracks_3d = np.concatenate([obs_3d[..., 0:3], fut_3d[..., 0:3]], axis=0).astype(np.float32, copy=False)

    total_len = int(obs_len + fut_len)
    if tracks_2d.shape[0] < total_len:
        return None

    tracks_2d = tracks_2d[:total_len]
    tracks_3d = tracks_3d[:total_len]

    visibility = _as_numpy(sample.get("visibility"), np.bool_)
    if visibility is None:
        visibility = np.ones((total_len, tracks_2d.shape[1]), dtype=np.bool_)
    else:
        if visibility.ndim == 3 and visibility.shape[-1] == 1:
            visibility = visibility[..., 0]
        if visibility.ndim != 2:
            return None
        if visibility.shape[0] < total_len:
            pad_t = total_len - visibility.shape[0]
            visibility = np.concatenate([visibility, np.repeat(visibility[-1:], pad_t, axis=0)], axis=0)
        visibility = visibility[:total_len]
        if visibility.shape[1] != tracks_2d.shape[1]:
            k = min(visibility.shape[1], tracks_2d.shape[1])
            visibility = visibility[:, :k]
            tracks_2d = tracks_2d[:, :k]
            tracks_3d = tracks_3d[:, :k]

    obs_valid = _as_numpy(sample.get("obs_valid"), np.bool_)
    fut_valid = _as_numpy(sample.get("fut_valid"), np.bool_)

    if obs_valid is None:
        obs_valid_1d = np.ones((obs_len,), dtype=np.bool_)
    elif obs_valid.ndim == 1:
        obs_valid_1d = obs_valid.astype(np.bool_, copy=False)
    else:
        obs_valid_1d = obs_valid.astype(np.bool_, copy=False).any(axis=1)

    if fut_valid is None:
        fut_valid_1d = np.ones((fut_len,), dtype=np.bool_)
    elif fut_valid.ndim == 1:
        fut_valid_1d = fut_valid.astype(np.bool_, copy=False)
    else:
        fut_valid_1d = fut_valid.astype(np.bool_, copy=False).any(axis=1)

    if obs_valid_1d.shape[0] < obs_len:
        obs_valid_1d = np.concatenate([obs_valid_1d, np.ones((obs_len - obs_valid_1d.shape[0],), dtype=np.bool_)], axis=0)
    if fut_valid_1d.shape[0] < fut_len:
        fut_valid_1d = np.concatenate([fut_valid_1d, np.ones((fut_len - fut_valid_1d.shape[0],), dtype=np.bool_)], axis=0)

    obs_valid_1d = obs_valid_1d[:obs_len]
    fut_valid_1d = fut_valid_1d[:fut_len]
    frame_valid = np.concatenate([obs_valid_1d, fut_valid_1d], axis=0)

    valid = (visibility.astype(np.bool_, copy=False) & frame_valid[:, None])

    if valid.shape[1] == 0:
        return None

    select = _select_tokens(valid=valid, max_tokens=max_tokens)
    tracks_2d = tracks_2d[:, select]
    tracks_3d = tracks_3d[:, select]
    visibility = visibility[:, select]
    valid = valid[:, select]

    state = _build_state_features(
        tracks_2d=tracks_2d,
        tracks_3d=tracks_3d,
        valid=valid,
        visibility=visibility,
    )

    return {
        "obs_state": state[:obs_len].astype(np.float32, copy=False),
        "fut_state": state[obs_len:total_len].astype(np.float32, copy=False),
        "obs_valid": valid[:obs_len].astype(np.bool_, copy=False),
        "fut_valid": valid[obs_len:total_len].astype(np.bool_, copy=False),
        "token_mask": np.ones((state.shape[1],), dtype=np.bool_),
    }


def _records_for_minisplit(minisplits: Dict[str, Any], dataset: str, split: str) -> List[Dict[str, Any]]:
    datasets = minisplits.get("datasets", {}) if isinstance(minisplits, dict) else {}
    ds = datasets.get(dataset, {}) if isinstance(datasets, dict) else {}
    recs = ds.get(split, []) if isinstance(ds, dict) else []
    if not isinstance(recs, list):
        return []
    return [x for x in recs if isinstance(x, dict)]


def _tap_status_payload(
    *,
    status: str,
    reason: str,
    records_declared: int,
    records_with_existing_files: int,
    samples_evaluated: int,
    teacher_forced_coord_loss: float | None,
    free_rollout_coord_mean_l2: float | None,
    free_rollout_endpoint_l2: float | None,
) -> Dict[str, Any]:
    return {
        "supported": bool(status == TAP_STATUS_AVAILABLE),
        "status": str(status),
        "status_reason": str(reason),
        "records_declared": int(records_declared),
        "records_with_existing_files": int(records_with_existing_files),
        "samples_evaluated": int(samples_evaluated),
        "teacher_forced_coord_loss": teacher_forced_coord_loss,
        "free_rollout_coord_mean_l2": free_rollout_coord_mean_l2,
        "free_rollout_endpoint_l2": free_rollout_endpoint_l2,
    }


def _eval_external_dataset_state_space(
    model: torch.nn.Module,
    dataset: Dataset,
    device: torch.device,
    obs_len: int,
    fut_len: int,
    max_samples: int,
    max_tokens: int,
) -> Dict[str, Any]:
    tf_coord_sum = 0.0
    tf_coord_count = 0.0
    rollout_l2_sum = 0.0
    rollout_l2_count = 0.0
    endpoint_l2_sum = 0.0
    endpoint_l2_count = 0.0

    n = min(int(len(dataset)), int(max_samples))
    samples_ok = 0

    model.eval()
    with torch.no_grad():
        for i in range(n):
            sample = dataset[i]
            parsed = _extract_external_state_sample(
                sample=sample,
                obs_len=int(obs_len),
                fut_len=int(fut_len),
                max_tokens=int(max_tokens),
            )
            if parsed is None:
                continue

            obs_state = torch.from_numpy(parsed["obs_state"]).unsqueeze(0).to(device=device, dtype=torch.float32)
            fut_state = torch.from_numpy(parsed["fut_state"]).unsqueeze(0).to(device=device, dtype=torch.float32)
            obs_valid = torch.from_numpy(parsed["obs_valid"]).unsqueeze(0).to(device=device, dtype=torch.bool)
            fut_valid = torch.from_numpy(parsed["fut_valid"]).unsqueeze(0).to(device=device, dtype=torch.bool)
            token_mask = torch.from_numpy(parsed["token_mask"]).unsqueeze(0).to(device=device, dtype=torch.bool)

            full_state = torch.cat([obs_state, fut_state], dim=1)
            full_valid = torch.cat([obs_valid, fut_valid], dim=1)

            shifted = torch.zeros_like(full_state)
            shifted[:, 0] = full_state[:, 0]
            shifted[:, 1:] = full_state[:, :-1]

            pred = model(shifted, token_mask=token_mask)
            pred_fut = _future_pred(pred, obs_len=int(obs_len))

            fut_target = full_state[:, int(obs_len):]
            fut_valid_mask = full_valid[:, int(obs_len):]
            tf_mask = (fut_valid_mask.bool() & token_mask[:, None, :].bool()).float()

            tf_coord_err = ((pred_fut["coord"] - fut_target[..., 0:2]) ** 2).sum(dim=-1)
            add_sum, add_cnt = _masked_sum(tf_coord_err, tf_mask)
            tf_coord_sum += add_sum
            tf_coord_count += add_cnt

            rollout_coord = _free_rollout_coords(
                model=model,
                obs_state=obs_state,
                token_mask=token_mask,
                fut_len=int(fut_len),
            )
            target_coord = fut_target[..., 0:2]
            l2 = torch.sqrt(((rollout_coord - target_coord) ** 2).sum(dim=-1) + 1e-8)

            roll_sum, roll_cnt = _masked_sum(l2, tf_mask)
            rollout_l2_sum += roll_sum
            rollout_l2_count += roll_cnt

            end_l2 = l2[:, -1]
            end_mask = tf_mask[:, -1]
            end_sum, end_cnt = _masked_sum(end_l2, end_mask)
            endpoint_l2_sum += end_sum
            endpoint_l2_count += end_cnt

            samples_ok += 1

    if samples_ok <= 0:
        return {
            "samples": 0,
            "teacher_forced_coord_loss": None,
            "free_rollout_coord_mean_l2": None,
            "free_rollout_endpoint_l2": None,
        }

    return {
        "samples": int(samples_ok),
        "teacher_forced_coord_loss": float(tf_coord_sum / max(tf_coord_count, 1.0)),
        "free_rollout_coord_mean_l2": float(rollout_l2_sum / max(rollout_l2_count, 1.0)),
        "free_rollout_endpoint_l2": float(endpoint_l2_sum / max(endpoint_l2_count, 1.0)),
    }


def _evaluate_tap_metrics_for_model(model: torch.nn.Module, device: torch.device, args: Any) -> Dict[str, Any]:
    try:
        from stwm.tracewm.datasets.stage1_tapvid import Stage1TapVidDataset
        from stwm.tracewm.datasets.stage1_tapvid3d import Stage1TapVid3DDataset
        from stwm.tracewm.datasets.stage1_unified import load_stage1_minisplits
    except Exception as exc:
        reason = f"dataset/evaluator import failed: {type(exc).__name__}: {exc}"
        return {
            "tapvid_eval": _tap_status_payload(
                status=TAP_STATUS_NOT_IMPLEMENTED,
                reason=reason,
                records_declared=0,
                records_with_existing_files=0,
                samples_evaluated=0,
                teacher_forced_coord_loss=None,
                free_rollout_coord_mean_l2=None,
                free_rollout_endpoint_l2=None,
            ),
            "tapvid3d_limited_eval": _tap_status_payload(
                status=TAP_STATUS_NOT_IMPLEMENTED,
                reason=reason,
                records_declared=0,
                records_with_existing_files=0,
                samples_evaluated=0,
                teacher_forced_coord_loss=None,
                free_rollout_coord_mean_l2=None,
                free_rollout_endpoint_l2=None,
            ),
        }

    minisplit_path = Path(str(args.stage1_minisplit_path))
    if not minisplit_path.exists():
        reason = f"minisplit file missing: {minisplit_path}"
        return {
            "tapvid_eval": _tap_status_payload(
                status=TAP_STATUS_DATA_NOT_READY,
                reason=reason,
                records_declared=0,
                records_with_existing_files=0,
                samples_evaluated=0,
                teacher_forced_coord_loss=None,
                free_rollout_coord_mean_l2=None,
                free_rollout_endpoint_l2=None,
            ),
            "tapvid3d_limited_eval": _tap_status_payload(
                status=TAP_STATUS_DATA_NOT_READY,
                reason=reason,
                records_declared=0,
                records_with_existing_files=0,
                samples_evaluated=0,
                teacher_forced_coord_loss=None,
                free_rollout_coord_mean_l2=None,
                free_rollout_endpoint_l2=None,
            ),
        }

    minisplits = load_stage1_minisplits(minisplit_path)

    def _filter_existing(records: List[Dict[str, Any]], key: str) -> Tuple[List[Dict[str, Any]], int]:
        kept: List[Dict[str, Any]] = []
        for rec in records:
            p = str(rec.get(key, "")).strip()
            if p and Path(p).exists():
                kept.append(rec)
        return kept, len(kept)

    def _eval_one_tapvid() -> Dict[str, Any]:
        records = _records_for_minisplit(minisplits, "tapvid", "eval_mini")
        declared = len(records)
        if declared <= 0:
            return _tap_status_payload(
                status=TAP_STATUS_DATA_NOT_READY,
                reason="tapvid eval_mini records missing in minisplit",
                records_declared=declared,
                records_with_existing_files=0,
                samples_evaluated=0,
                teacher_forced_coord_loss=None,
                free_rollout_coord_mean_l2=None,
                free_rollout_endpoint_l2=None,
            )

        existing_records, existing_count = _filter_existing(records, "cache_npz")
        if existing_count <= 0:
            return _tap_status_payload(
                status=TAP_STATUS_DATA_NOT_READY,
                reason="tapvid cache_npz files not found for eval_mini records",
                records_declared=declared,
                records_with_existing_files=existing_count,
                samples_evaluated=0,
                teacher_forced_coord_loss=None,
                free_rollout_coord_mean_l2=None,
                free_rollout_endpoint_l2=None,
            )

        try:
            ds = Stage1TapVidDataset(
                split="eval_mini",
                minisplit_records=existing_records,
                obs_len=int(args.obs_len),
                fut_len=int(args.fut_len),
            )
            metrics = _eval_external_dataset_state_space(
                model=model,
                dataset=ds,
                device=device,
                obs_len=int(args.obs_len),
                fut_len=int(args.fut_len),
                max_samples=int(args.eval_max_tapvid_samples),
                max_tokens=int(args.max_tokens),
            )
            if int(metrics.get("samples", 0)) <= 0:
                return _tap_status_payload(
                    status=TAP_STATUS_DATA_NOT_READY,
                    reason="tapvid dataset built but no valid samples converted to v2 state space",
                    records_declared=declared,
                    records_with_existing_files=existing_count,
                    samples_evaluated=0,
                    teacher_forced_coord_loss=None,
                    free_rollout_coord_mean_l2=None,
                    free_rollout_endpoint_l2=None,
                )
            return _tap_status_payload(
                status=TAP_STATUS_AVAILABLE,
                reason="tapvid evaluator path available and executed",
                records_declared=declared,
                records_with_existing_files=existing_count,
                samples_evaluated=int(metrics["samples"]),
                teacher_forced_coord_loss=_safe_float(metrics.get("teacher_forced_coord_loss"), None),
                free_rollout_coord_mean_l2=_safe_float(metrics.get("free_rollout_coord_mean_l2"), None),
                free_rollout_endpoint_l2=_safe_float(metrics.get("free_rollout_endpoint_l2"), None),
            )
        except NotImplementedError as exc:
            return _tap_status_payload(
                status=TAP_STATUS_NOT_IMPLEMENTED,
                reason=f"tapvid evaluator hook not implemented: {exc}",
                records_declared=declared,
                records_with_existing_files=existing_count,
                samples_evaluated=0,
                teacher_forced_coord_loss=None,
                free_rollout_coord_mean_l2=None,
                free_rollout_endpoint_l2=None,
            )
        except (FileNotFoundError, KeyError, ValueError) as exc:
            return _tap_status_payload(
                status=TAP_STATUS_DATA_NOT_READY,
                reason=f"tapvid data path invalid for evaluator: {type(exc).__name__}: {exc}",
                records_declared=declared,
                records_with_existing_files=existing_count,
                samples_evaluated=0,
                teacher_forced_coord_loss=None,
                free_rollout_coord_mean_l2=None,
                free_rollout_endpoint_l2=None,
            )
        except Exception as exc:
            return _tap_status_payload(
                status=TAP_STATUS_NOT_IMPLEMENTED,
                reason=f"tapvid evaluator runtime mismatch: {type(exc).__name__}: {exc}",
                records_declared=declared,
                records_with_existing_files=existing_count,
                samples_evaluated=0,
                teacher_forced_coord_loss=None,
                free_rollout_coord_mean_l2=None,
                free_rollout_endpoint_l2=None,
            )

    def _eval_one_tapvid3d() -> Dict[str, Any]:
        records = _records_for_minisplit(minisplits, "tapvid3d", "eval_mini")
        declared = len(records)
        if declared <= 0:
            return _tap_status_payload(
                status=TAP_STATUS_DATA_NOT_READY,
                reason="tapvid3d eval_mini records missing in minisplit",
                records_declared=declared,
                records_with_existing_files=0,
                samples_evaluated=0,
                teacher_forced_coord_loss=None,
                free_rollout_coord_mean_l2=None,
                free_rollout_endpoint_l2=None,
            )

        existing_records, existing_count = _filter_existing(records, "npz_path")
        if existing_count <= 0:
            return _tap_status_payload(
                status=TAP_STATUS_DATA_NOT_READY,
                reason="tapvid3d npz_path files not found for eval_mini records",
                records_declared=declared,
                records_with_existing_files=existing_count,
                samples_evaluated=0,
                teacher_forced_coord_loss=None,
                free_rollout_coord_mean_l2=None,
                free_rollout_endpoint_l2=None,
            )

        try:
            ds = Stage1TapVid3DDataset(
                data_root=str(args.data_root),
                split="eval_mini",
                minisplit_records=existing_records,
                obs_len=int(args.obs_len),
                fut_len=int(args.fut_len),
            )
            metrics = _eval_external_dataset_state_space(
                model=model,
                dataset=ds,
                device=device,
                obs_len=int(args.obs_len),
                fut_len=int(args.fut_len),
                max_samples=int(args.eval_max_tapvid3d_samples),
                max_tokens=int(args.max_tokens),
            )
            if int(metrics.get("samples", 0)) <= 0:
                return _tap_status_payload(
                    status=TAP_STATUS_DATA_NOT_READY,
                    reason="tapvid3d dataset built but no valid samples converted to v2 state space",
                    records_declared=declared,
                    records_with_existing_files=existing_count,
                    samples_evaluated=0,
                    teacher_forced_coord_loss=None,
                    free_rollout_coord_mean_l2=None,
                    free_rollout_endpoint_l2=None,
                )
            return _tap_status_payload(
                status=TAP_STATUS_AVAILABLE,
                reason="tapvid3d limited evaluator path available and executed",
                records_declared=declared,
                records_with_existing_files=existing_count,
                samples_evaluated=int(metrics["samples"]),
                teacher_forced_coord_loss=_safe_float(metrics.get("teacher_forced_coord_loss"), None),
                free_rollout_coord_mean_l2=_safe_float(metrics.get("free_rollout_coord_mean_l2"), None),
                free_rollout_endpoint_l2=_safe_float(metrics.get("free_rollout_endpoint_l2"), None),
            )
        except NotImplementedError as exc:
            return _tap_status_payload(
                status=TAP_STATUS_NOT_IMPLEMENTED,
                reason=f"tapvid3d evaluator hook not implemented: {exc}",
                records_declared=declared,
                records_with_existing_files=existing_count,
                samples_evaluated=0,
                teacher_forced_coord_loss=None,
                free_rollout_coord_mean_l2=None,
                free_rollout_endpoint_l2=None,
            )
        except (FileNotFoundError, KeyError, ValueError) as exc:
            return _tap_status_payload(
                status=TAP_STATUS_DATA_NOT_READY,
                reason=f"tapvid3d data path invalid for evaluator: {type(exc).__name__}: {exc}",
                records_declared=declared,
                records_with_existing_files=existing_count,
                samples_evaluated=0,
                teacher_forced_coord_loss=None,
                free_rollout_coord_mean_l2=None,
                free_rollout_endpoint_l2=None,
            )
        except Exception as exc:
            return _tap_status_payload(
                status=TAP_STATUS_NOT_IMPLEMENTED,
                reason=f"tapvid3d evaluator runtime mismatch: {type(exc).__name__}: {exc}",
                records_declared=declared,
                records_with_existing_files=existing_count,
                samples_evaluated=0,
                teacher_forced_coord_loss=None,
                free_rollout_coord_mean_l2=None,
                free_rollout_endpoint_l2=None,
            )

    return {
        "tapvid_eval": _eval_one_tapvid(),
        "tapvid3d_limited_eval": _eval_one_tapvid3d(),
    }


def _evaluate_model(
    model: torch.nn.Module,
    criterion: StructuredTraceLoss,
    loader: DataLoader,
    device: torch.device,
    obs_len: int,
    fut_len: int,
    pin_memory: bool,
    eval_steps: int,
    args: Any,
) -> Dict[str, Any]:
    model.eval()

    tf_coord_sum = 0.0
    tf_coord_count = 0.0
    rollout_l2_sum = 0.0
    rollout_l2_count = 0.0
    endpoint_l2_sum = 0.0
    endpoint_l2_count = 0.0

    with torch.no_grad():
        for step_idx, batch in enumerate(loader):
            if step_idx >= int(eval_steps):
                break

            non_blocking = bool(pin_memory and device.type == "cuda")
            data = _to_device(batch, device=device, non_blocking=non_blocking)
            full_state = torch.cat([data["obs_state"], data["fut_state"]], dim=1)
            full_valid = torch.cat([data["obs_valid"], data["fut_valid"]], dim=1)

            shifted = torch.zeros_like(full_state)
            shifted[:, 0] = full_state[:, 0]
            shifted[:, 1:] = full_state[:, :-1]
            pred = model(shifted, token_mask=data["token_mask"])

            pred_fut = _future_pred(pred, obs_len=int(obs_len))
            fut_target = full_state[:, int(obs_len):]
            fut_valid = full_valid[:, int(obs_len):]
            token_mask = data["token_mask"]

            _ = criterion(
                pred=pred_fut,
                target_state=fut_target,
                valid_mask=fut_valid,
                token_mask=token_mask,
            )

            tf_mask = (fut_valid.bool() & token_mask[:, None, :].bool()).float()
            tf_coord_err = ((pred_fut["coord"] - fut_target[..., 0:2]) ** 2).sum(dim=-1)
            add_sum, add_cnt = _masked_sum(tf_coord_err, tf_mask)
            tf_coord_sum += add_sum
            tf_coord_count += add_cnt

            rollout_coord = _free_rollout_coords(
                model=model,
                obs_state=data["obs_state"],
                token_mask=token_mask,
                fut_len=int(fut_len),
            )
            target_coord = fut_target[..., 0:2]
            l2 = torch.sqrt(((rollout_coord - target_coord) ** 2).sum(dim=-1) + 1e-8)

            roll_sum, roll_cnt = _masked_sum(l2, tf_mask)
            rollout_l2_sum += roll_sum
            rollout_l2_count += roll_cnt

            end_l2 = l2[:, -1]
            end_mask = tf_mask[:, -1]
            end_sum, end_cnt = _masked_sum(end_l2, end_mask)
            endpoint_l2_sum += end_sum
            endpoint_l2_count += end_cnt

    teacher_forced_coord_loss = float(tf_coord_sum / max(tf_coord_count, 1.0))
    free_rollout_coord_mean_l2 = float(rollout_l2_sum / max(rollout_l2_count, 1.0))
    free_rollout_endpoint_l2 = float(endpoint_l2_sum / max(endpoint_l2_count, 1.0))

    tap_eval = _evaluate_tap_metrics_for_model(model=model, device=device, args=args)

    return {
        "teacher_forced_coord_loss": teacher_forced_coord_loss,
        "free_rollout_coord_mean_l2": free_rollout_coord_mean_l2,
        "free_rollout_endpoint_l2": free_rollout_endpoint_l2,
        "tapvid_eval": tap_eval["tapvid_eval"],
        "tapvid3d_limited_eval": tap_eval["tapvid3d_limited_eval"],
    }


def _train_variant(
    spec: VariantSpec,
    train_dataset: Stage1V2UnifiedDataset,
    val_dataset: Stage1V2UnifiedDataset,
    runtime: RuntimeConfig,
    args: Any,
    seed: int,
    steps_override: int | None = None,
    eval_steps_override: int | None = None,
) -> Dict[str, Any]:
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, model_meta = _build_model(
        spec=spec,
        max_tokens=int(args.max_tokens),
        gru_hidden_dim=int(args.gru_hidden_dim),
        gru_num_layers=int(args.gru_num_layers),
    )
    model = model.to(device)

    criterion = StructuredTraceLoss(_build_loss_config(spec))
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))

    train_loader_kwargs: Dict[str, Any] = {
        "dataset": train_dataset,
        "batch_size": int(args.batch_size),
        "shuffle": True,
        "num_workers": int(runtime.num_workers),
        "pin_memory": bool(runtime.pin_memory),
        "collate_fn": stage1_v2_collate_fn,
    }
    if int(runtime.num_workers) > 0:
        train_loader_kwargs["persistent_workers"] = bool(runtime.persistent_workers)
        train_loader_kwargs["prefetch_factor"] = int(runtime.prefetch_factor)

    train_loader = DataLoader(**train_loader_kwargs)

    val_loader = DataLoader(
        val_dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=0,
        collate_fn=stage1_v2_collate_fn,
        pin_memory=bool(runtime.pin_memory),
    )

    target_steps = int(steps_override) if steps_override is not None else int(args.steps_per_variant)
    target_eval_steps = int(eval_steps_override) if eval_steps_override is not None else int(args.eval_steps)

    history: List[Dict[str, float]] = []
    total_steps = 0
    for _epoch in range(int(args.epochs)):
        model.train()
        running = {
            "total_loss": 0.0,
            "coord_loss": 0.0,
            "visibility_loss": 0.0,
            "residual_loss": 0.0,
            "velocity_loss": 0.0,
            "endpoint_loss": 0.0,
        }

        data_iter = iter(train_loader)
        step_count = 0
        while step_count < int(target_steps):
            try:
                batch = next(data_iter)
            except StopIteration:
                break

            non_blocking = bool(runtime.pin_memory and device.type == "cuda")
            data = _to_device(batch, device=device, non_blocking=non_blocking)
            full_state = torch.cat([data["obs_state"], data["fut_state"]], dim=1)
            full_valid = torch.cat([data["obs_valid"], data["fut_valid"]], dim=1)

            shifted = torch.zeros_like(full_state)
            shifted[:, 0] = full_state[:, 0]
            shifted[:, 1:] = full_state[:, :-1]

            pred = model(shifted, token_mask=data["token_mask"])
            losses = criterion(
                pred=_future_pred(pred, obs_len=int(args.obs_len)),
                target_state=full_state[:, int(args.obs_len):],
                valid_mask=full_valid[:, int(args.obs_len):],
                token_mask=data["token_mask"],
            )

            optimizer.zero_grad(set_to_none=True)
            losses["total_loss"].backward()
            if float(args.clip_grad_norm) > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(args.clip_grad_norm))
            optimizer.step()

            for key in running:
                running[key] += float(losses[key].detach().cpu().item())

            step_count += 1
            total_steps += 1

        denom = float(max(step_count, 1))
        history.append({k: float(v / denom) for k, v in running.items()})

    final_metrics = history[-1] if history else {
        "total_loss": 0.0,
        "coord_loss": 0.0,
        "visibility_loss": 0.0,
        "residual_loss": 0.0,
        "velocity_loss": 0.0,
        "endpoint_loss": 0.0,
    }

    evaluation = _evaluate_model(
        model=model,
        criterion=criterion,
        loader=val_loader,
        device=device,
        obs_len=int(args.obs_len),
        fut_len=int(args.fut_len),
        pin_memory=bool(runtime.pin_memory),
        eval_steps=int(target_eval_steps),
        args=args,
    )

    return {
        "run_name": spec.run_name,
        "group": spec.group,
        "status": "ok",
        "state_variant": spec.state_variant,
        "backbone_variant": spec.backbone_variant,
        "loss_variant": spec.loss_variant,
        "model": model_meta,
        "runtime": {
            "num_workers": int(runtime.num_workers),
            "pin_memory": bool(runtime.pin_memory),
            "persistent_workers": bool(runtime.persistent_workers),
            "prefetch_factor": int(runtime.prefetch_factor),
            "single_gpu_only": bool(runtime.single_gpu_only),
        },
        "train_final_metrics": final_metrics,
        "evaluation": evaluation,
        "train_steps": int(total_steps),
        "notes": spec.notes,
    }


def _tertiary_external_metric(ev: Dict[str, Any]) -> float:
    vals: List[float] = []
    for key in ["tapvid_eval", "tapvid3d_limited_eval"]:
        item = ev.get(key, {}) if isinstance(ev.get(key, {}), dict) else {}
        if str(item.get("status", "")) != TAP_STATUS_AVAILABLE:
            continue
        v = item.get("free_rollout_endpoint_l2", None)
        if v is None:
            continue
        vals.append(float(v))
    if not vals:
        return 1e9
    return float(min(vals))


def _metric_tuple(run: Dict[str, Any]) -> Tuple[float, float, float, str]:
    ev = run.get("evaluation", {}) if isinstance(run.get("evaluation", {}), dict) else {}
    endpoint = float(ev.get("free_rollout_endpoint_l2", 1e9) or 1e9)
    mean_l2 = float(ev.get("free_rollout_coord_mean_l2", 1e9) or 1e9)
    tertiary = _tertiary_external_metric(ev)
    return endpoint, mean_l2, tertiary, str(run.get("run_name", ""))


def _pick_best_run(runs: List[Dict[str, Any]]) -> Dict[str, Any] | None:
    valid = [r for r in runs if str(r.get("status", "")) == "ok"]
    if not valid:
        return None
    return min(valid, key=_metric_tuple)


def _winner_reason(run: Dict[str, Any], best_run: Dict[str, Any] | None) -> str:
    if best_run is None:
        return "no winner"

    if str(run.get("run_name", "")) == str(best_run.get("run_name", "")):
        return "winner: best primary, then secondary, then tertiary"

    p_run, s_run, t_run, _ = _metric_tuple(run)
    p_best, s_best, t_best, _ = _metric_tuple(best_run)

    if abs(p_run - p_best) > 1e-12:
        return f"loses on primary by +{(p_run - p_best):.6f}"
    if abs(s_run - s_best) > 1e-12:
        return f"primary tied, loses on secondary by +{(s_run - s_best):.6f}"
    return f"primary+secondary tied, loses on tertiary by +{(t_run - t_best):.6f}"


def _group_winner_info(runs: List[Dict[str, Any]], best_run: Dict[str, Any] | None, min_delta: float) -> Dict[str, Any]:
    valid = [r for r in runs if str(r.get("status", "")) == "ok"]
    if best_run is None or not valid:
        return {
            "has_winner": False,
            "clear_winner": False,
            "winner": "none",
            "primary_margin_vs_runner_up": None,
            "min_required_margin": float(min_delta),
            "runner_up": "none",
        }

    ranked = sorted(valid, key=_metric_tuple)
    winner = ranked[0]
    runner_up = ranked[1] if len(ranked) > 1 else None
    if runner_up is None:
        return {
            "has_winner": True,
            "clear_winner": False,
            "winner": str(winner.get("run_name", "none")),
            "primary_margin_vs_runner_up": None,
            "min_required_margin": float(min_delta),
            "runner_up": "none",
        }

    winner_primary = _metric_tuple(winner)[0]
    runner_primary = _metric_tuple(runner_up)[0]
    margin = float(runner_primary - winner_primary)
    clear = bool(margin >= float(min_delta))

    return {
        "has_winner": True,
        "clear_winner": clear,
        "winner": str(winner.get("run_name", "none")),
        "primary_margin_vs_runner_up": margin,
        "min_required_margin": float(min_delta),
        "runner_up": str(runner_up.get("run_name", "none")),
    }


def _write_group_report(
    *,
    report_json: Path,
    report_md: Path,
    group_name: str,
    objective: str,
    runs: List[Dict[str, Any]],
    best_run: Dict[str, Any] | None,
    include_total_loss_reference: bool,
    min_required_margin: float,
) -> Dict[str, Any]:
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_md.parent.mkdir(parents=True, exist_ok=True)

    winner_info = _group_winner_info(runs=runs, best_run=best_run, min_delta=float(min_required_margin))

    payload = {
        "generated_at_utc": now_iso(),
        "group": group_name,
        "objective": objective,
        "selection_policy": {
            "primary": "free_rollout_endpoint_l2",
            "secondary": "free_rollout_coord_mean_l2",
            "tertiary": "tapvid_eval.free_rollout_endpoint_l2_or_tapvid3d_limited_eval.free_rollout_endpoint_l2",
            "total_loss_usage": "reference_only" if include_total_loss_reference else "not_used",
        },
        "runs": runs,
        "best_variant": str(best_run.get("run_name", "none")) if best_run else "none",
        "winner_analysis": winner_info,
    }
    report_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        f"# Stage1-v2 {group_name.title()} Ablation",
        "",
        f"- generated_at_utc: {payload['generated_at_utc']}",
        f"- objective: {objective}",
        f"- best_variant: {payload['best_variant']}",
        "- selection_policy: primary=free_rollout_endpoint_l2, secondary=free_rollout_coord_mean_l2, tertiary=tapvid_eval_or_tapvid3d_limited_eval",
        f"- clear_winner: {winner_info['clear_winner']} (margin={winner_info['primary_margin_vs_runner_up']}, required>={winner_info['min_required_margin']})",
    ]
    if include_total_loss_reference:
        lines.append("- total_loss is reference only and not used as main selection key")

    lines.extend(
        [
            "",
            "| variant | status | primary_endpoint_l2 | secondary_mean_l2 | tertiary_external_l2 | total_loss_ref | tapvid_status | tapvid3d_status | winner_reason |",
            "|---|---|---:|---:|---:|---:|---|---|---|",
        ]
    )

    for run in runs:
        ev = run.get("evaluation", {}) if isinstance(run.get("evaluation", {}), dict) else {}
        primary = _safe_float(ev.get("free_rollout_endpoint_l2"), 1e9)
        secondary = _safe_float(ev.get("free_rollout_coord_mean_l2"), 1e9)
        tertiary = _tertiary_external_metric(ev)
        tapvid_status = str((ev.get("tapvid_eval", {}) if isinstance(ev.get("tapvid_eval", {}), dict) else {}).get("status", "unknown"))
        tap3d_status = str((ev.get("tapvid3d_limited_eval", {}) if isinstance(ev.get("tapvid3d_limited_eval", {}), dict) else {}).get("status", "unknown"))
        total_loss_ref = float((run.get("train_final_metrics", {}) if isinstance(run.get("train_final_metrics", {}), dict) else {}).get("total_loss", 0.0) or 0.0)
        reason = _winner_reason(run, best_run)
        lines.append(
            f"| {run.get('run_name', '-')} | {run.get('status', 'fail')} | {primary:.6f} | {secondary:.6f} | {tertiary:.6f} | {total_loss_ref:.6f} | {tapvid_status} | {tap3d_status} | {reason} |"
        )

    report_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return payload


def _build_state_specs() -> List[VariantSpec]:
    return [
        VariantSpec(
            run_name="stage1_v2_state_legacy_mean5d_gru",
            group="state",
            state_variant="legacy_mean5d",
            backbone_variant="stage1_v2_backbone_multitoken_gru",
            model_preset="gru_baseline",
            loss_variant="coord_visibility_residual_velocity",
            enable_visibility=True,
            enable_residual=True,
            enable_velocity=True,
            enable_endpoint=False,
            notes="legacy mean-5d state with minimal GRU baseline",
        ),
        VariantSpec(
            run_name="stage1_v2_state_multitoken_gru",
            group="state",
            state_variant="multitoken",
            backbone_variant="stage1_v2_backbone_multitoken_gru",
            model_preset="gru_baseline",
            loss_variant="coord_visibility_residual_velocity",
            enable_visibility=True,
            enable_residual=True,
            enable_velocity=True,
            enable_endpoint=False,
            notes="multi-token state with matched GRU baseline",
        ),
    ]


def _build_backbone_specs() -> List[VariantSpec]:
    return [
        VariantSpec(
            run_name="stage1_v2_backbone_multitoken_gru",
            group="backbone",
            state_variant="multitoken",
            backbone_variant="stage1_v2_backbone_multitoken_gru",
            model_preset="gru_baseline",
            loss_variant="coord_visibility_residual_velocity",
            enable_visibility=True,
            enable_residual=True,
            enable_velocity=True,
            enable_endpoint=False,
            notes="multi-token state with GRU backbone",
        ),
        VariantSpec(
            run_name="stage1_v2_backbone_transformer_debugsmall",
            group="backbone",
            state_variant="multitoken",
            backbone_variant="stage1_v2_backbone_transformer_debugsmall",
            model_preset="debug_small",
            loss_variant="coord_visibility_residual_velocity",
            enable_visibility=True,
            enable_residual=True,
            enable_velocity=True,
            enable_endpoint=False,
            notes="multi-token state with transformer debug_small",
        ),
        VariantSpec(
            run_name="stage1_v2_backbone_transformer_prototype220m",
            group="backbone",
            state_variant="multitoken",
            backbone_variant="stage1_v2_backbone_transformer_prototype220m",
            model_preset="prototype_220m",
            loss_variant="coord_visibility_residual_velocity",
            enable_visibility=True,
            enable_residual=True,
            enable_velocity=True,
            enable_endpoint=False,
            notes="multi-token state with transformer prototype_220m",
        ),
    ]


def _build_loss_specs(selected_backbone: str) -> List[VariantSpec]:
    return [
        VariantSpec(
            run_name="stage1_v2_loss_coord_only",
            group="losses",
            state_variant="multitoken",
            backbone_variant=selected_backbone,
            model_preset="selected_backbone",
            loss_variant="coord_only",
            enable_visibility=False,
            enable_residual=False,
            enable_velocity=False,
            enable_endpoint=False,
            notes="coord-only loss",
        ),
        VariantSpec(
            run_name="stage1_v2_loss_coord_visibility",
            group="losses",
            state_variant="multitoken",
            backbone_variant=selected_backbone,
            model_preset="selected_backbone",
            loss_variant="coord_visibility",
            enable_visibility=True,
            enable_residual=False,
            enable_velocity=False,
            enable_endpoint=False,
            notes="coord + visibility",
        ),
        VariantSpec(
            run_name="stage1_v2_loss_coord_visibility_residual_velocity",
            group="losses",
            state_variant="multitoken",
            backbone_variant=selected_backbone,
            model_preset="selected_backbone",
            loss_variant="coord_visibility_residual_velocity",
            enable_visibility=True,
            enable_residual=True,
            enable_velocity=True,
            enable_endpoint=False,
            notes="coord + visibility + residual + velocity",
        ),
        VariantSpec(
            run_name="stage1_v2_loss_coord_visibility_residual_velocity_endpoint",
            group="losses",
            state_variant="multitoken",
            backbone_variant=selected_backbone,
            model_preset="selected_backbone",
            loss_variant="coord_visibility_residual_velocity_endpoint",
            enable_visibility=True,
            enable_residual=True,
            enable_velocity=True,
            enable_endpoint=True,
            notes="coord + visibility + residual + velocity + endpoint",
        ),
    ]


def _loss_flags_from_variant(loss_variant: str) -> Tuple[bool, bool, bool, bool]:
    lv = str(loss_variant).strip()
    if lv == "coord_only":
        return False, False, False, False
    if lv == "coord_visibility":
        return True, False, False, False
    if lv == "coord_visibility_residual_velocity":
        return True, True, True, False
    if lv == "coord_visibility_residual_velocity_endpoint":
        return True, True, True, True
    return True, True, True, False


def _build_mainline_replay_spec(
    best_state: Dict[str, Any] | None,
    best_backbone: Dict[str, Any] | None,
    best_loss: Dict[str, Any] | None,
) -> VariantSpec | None:
    if not best_state or not best_backbone or not best_loss:
        return None

    lv = str(best_loss.get("loss_variant", "coord_visibility"))
    vis, res, vel, endpoint = _loss_flags_from_variant(lv)

    return VariantSpec(
        run_name="stage1_v2_mainline_replay",
        group="mainline_replay",
        state_variant=str(best_state.get("state_variant", "multitoken")),
        backbone_variant=str(best_backbone.get("backbone_variant", "stage1_v2_backbone_transformer_debugsmall")),
        model_preset="replay_selected_combo",
        loss_variant=lv,
        enable_visibility=vis,
        enable_residual=res,
        enable_velocity=vel,
        enable_endpoint=endpoint,
        notes="true replay of candidate mainline combo chosen from ablation winners",
    )


def _is_220m_run(run: Dict[str, Any]) -> bool:
    model = run.get("model", {}) if isinstance(run.get("model", {}), dict) else {}
    return bool(model.get("target_220m_range_pass", False))


def _write_mainline_replay_report(
    *,
    report_json: Path,
    report_md: Path,
    replay_run: Dict[str, Any] | None,
    candidate_combo: Dict[str, Any],
) -> Dict[str, Any]:
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_md.parent.mkdir(parents=True, exist_ok=True)

    replay_status = "ok" if replay_run and str(replay_run.get("status", "")) == "ok" else "failed_or_skipped"

    payload = {
        "generated_at_utc": now_iso(),
        "objective": "Replay a real candidate mainline configuration selected from ablation winners.",
        "candidate_combo": candidate_combo,
        "replay_status": replay_status,
        "replay_run": replay_run,
        "summary_file": str(report_json),
    }
    report_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Stage1-v2 Mainline Replay",
        "",
        f"- generated_at_utc: {payload['generated_at_utc']}",
        f"- replay_status: {replay_status}",
        f"- candidate_state_variant: {candidate_combo.get('state_variant', 'none')}",
        f"- candidate_backbone_variant: {candidate_combo.get('backbone_variant', 'none')}",
        f"- candidate_loss_variant: {candidate_combo.get('loss_variant', 'none')}",
    ]

    if replay_run and str(replay_run.get("status", "")) == "ok":
        ev = replay_run.get("evaluation", {}) if isinstance(replay_run.get("evaluation", {}), dict) else {}
        model = replay_run.get("model", {}) if isinstance(replay_run.get("model", {}), dict) else {}
        lines.extend(
            [
                f"- replay_run_name: {replay_run.get('run_name', 'none')}",
                f"- parameter_count_estimated: {model.get('estimated_parameter_count', 0)}",
                f"- target_220m_range_pass: {model.get('target_220m_range_pass', False)}",
                "",
                "| metric | value |",
                "|---|---:|",
                f"| primary_endpoint_l2 | {_safe_float(ev.get('free_rollout_endpoint_l2'), 1e9):.6f} |",
                f"| secondary_mean_l2 | {_safe_float(ev.get('free_rollout_coord_mean_l2'), 1e9):.6f} |",
                f"| tertiary_external_l2 | {_tertiary_external_metric(ev):.6f} |",
                f"| total_loss_ref | {_safe_float((replay_run.get('train_final_metrics', {}) if isinstance(replay_run.get('train_final_metrics', {}), dict) else {}).get('total_loss', 0.0), 0.0):.6f} |",
            ]
        )
    else:
        lines.append("- replay did not produce a successful run record")

    report_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    args = parse_args()

    runtime = _load_runtime_config(args.recommended_runtime_json)
    print(
        "[stage1-v2-scireval] runtime "
        f"single_gpu_only={runtime.single_gpu_only} selected_gpu_id={runtime.selected_gpu_id} "
        f"workers={runtime.num_workers} pin_memory={runtime.pin_memory} "
        f"persistent_workers={runtime.persistent_workers} prefetch_factor={runtime.prefetch_factor}"
    )

    train_dataset = Stage1V2UnifiedDataset(
        dataset_names=[str(x) for x in args.dataset_names],
        split="train",
        contract_path=str(args.contract_path),
        obs_len=int(args.obs_len),
        fut_len=int(args.fut_len),
        max_tokens=int(args.max_tokens),
        max_samples_per_dataset=int(args.max_samples_per_dataset_train),
    )
    val_dataset = Stage1V2UnifiedDataset(
        dataset_names=[str(x) for x in args.dataset_names],
        split="val",
        contract_path=str(args.contract_path),
        obs_len=int(args.obs_len),
        fut_len=int(args.fut_len),
        max_tokens=int(args.max_tokens),
        max_samples_per_dataset=int(args.max_samples_per_dataset_val),
    )

    all_runs: List[Dict[str, Any]] = []

    state_specs = _build_state_specs()
    state_runs: List[Dict[str, Any]] = []
    for idx, spec in enumerate(state_specs):
        print(f"[stage1-v2-scireval] run={spec.run_name}")
        run = _train_variant(
            spec=spec,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            runtime=runtime,
            args=args,
            seed=int(args.seed) + idx,
        )
        state_runs.append(run)
        all_runs.append(run)

    best_state = _pick_best_run(state_runs)
    _write_group_report(
        report_json=Path(args.state_report_json),
        report_md=Path(args.state_report_md),
        group_name="state",
        objective="Is multi-token state beneficial vs legacy mean-5d under matched GRU backbone?",
        runs=state_runs,
        best_run=best_state,
        include_total_loss_reference=True,
        min_required_margin=float(args.state_min_win_delta),
    )

    backbone_specs = _build_backbone_specs()
    backbone_runs: List[Dict[str, Any]] = []
    for idx, spec in enumerate(backbone_specs):
        print(f"[stage1-v2-scireval] run={spec.run_name}")
        run = _train_variant(
            spec=spec,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            runtime=runtime,
            args=args,
            seed=int(args.seed) + 100 + idx,
        )
        backbone_runs.append(run)
        all_runs.append(run)

    best_backbone = _pick_best_run(backbone_runs)
    _write_group_report(
        report_json=Path(args.backbone_report_json),
        report_md=Path(args.backbone_report_md),
        group_name="backbone",
        objective="Under fixed multi-token state, compare GRU vs Transformer and debug_small vs prototype_220m.",
        runs=backbone_runs,
        best_run=best_backbone,
        include_total_loss_reference=True,
        min_required_margin=float(args.backbone_min_win_delta),
    )

    selected_backbone_variant = str(best_backbone.get("run_name", "stage1_v2_backbone_transformer_debugsmall")) if best_backbone else "stage1_v2_backbone_transformer_debugsmall"
    loss_specs = _build_loss_specs(selected_backbone=selected_backbone_variant)
    loss_runs: List[Dict[str, Any]] = []
    for idx, spec in enumerate(loss_specs):
        print(f"[stage1-v2-scireval] run={spec.run_name} on_backbone={selected_backbone_variant}")
        run = _train_variant(
            spec=spec,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            runtime=runtime,
            args=args,
            seed=int(args.seed) + 200 + idx,
        )
        loss_runs.append(run)
        all_runs.append(run)

    best_loss = _pick_best_run(loss_runs)
    _write_group_report(
        report_json=Path(args.losses_report_json),
        report_md=Path(args.losses_report_md),
        group_name="losses",
        objective="Under fixed state/backbone, compare loss families using external rollout metrics.",
        runs=loss_runs,
        best_run=best_loss,
        include_total_loss_reference=True,
        min_required_margin=0.0,
    )

    replay_spec = _build_mainline_replay_spec(best_state=best_state, best_backbone=best_backbone, best_loss=best_loss)
    replay_run: Dict[str, Any] | None = None
    if replay_spec is not None:
        print(f"[stage1-v2-scireval] run={replay_spec.run_name}")
        replay_run = _train_variant(
            spec=replay_spec,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            runtime=runtime,
            args=args,
            seed=int(args.seed) + 300,
            steps_override=int(args.mainline_replay_steps),
            eval_steps_override=int(args.mainline_replay_eval_steps),
        )
        all_runs.append(replay_run)

    replay_payload = _write_mainline_replay_report(
        report_json=Path(args.mainline_replay_json),
        report_md=Path(args.mainline_replay_md),
        replay_run=replay_run,
        candidate_combo={
            "state_variant": str(best_state.get("state_variant", "none")) if best_state else "none",
            "backbone_variant": str(best_backbone.get("backbone_variant", "none")) if best_backbone else "none",
            "loss_variant": str(best_loss.get("loss_variant", "none")) if best_loss else "none",
        },
    )

    run_details_json = Path(args.run_details_json)
    run_details_json.parent.mkdir(parents=True, exist_ok=True)
    run_details_json.write_text(
        json.dumps(
            {
                "generated_at_utc": now_iso(),
                "contract_path": str(args.contract_path),
                "runtime": runtime.__dict__,
                "runs": all_runs,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    best_state_variant = str(best_state.get("run_name", "none")) if best_state else "none"
    best_backbone_variant = str(best_backbone.get("run_name", "none")) if best_backbone else "none"
    best_loss_variant = str(best_loss.get("run_name", "none")) if best_loss else "none"

    state_winner_info = _group_winner_info(state_runs, best_state, float(args.state_min_win_delta))
    backbone_winner_info = _group_winner_info(backbone_runs, best_backbone, float(args.backbone_min_win_delta))

    loss_selection_uses_external = True
    replay_ok = bool(replay_run and str(replay_run.get("status", "")) == "ok")

    validation_gaps: List[str] = []

    if not bool(state_winner_info.get("clear_winner", False)):
        validation_gaps.append(
            "state ablation winner margin is below threshold "
            f"(margin={state_winner_info.get('primary_margin_vs_runner_up')}, required>={state_winner_info.get('min_required_margin')})"
        )

    if not bool(backbone_winner_info.get("clear_winner", False)):
        validation_gaps.append(
            "backbone ablation winner is not decisive by primary metric margin "
            f"(margin={backbone_winner_info.get('primary_margin_vs_runner_up')}, required>={backbone_winner_info.get('min_required_margin')})"
        )

    if not loss_selection_uses_external:
        validation_gaps.append("loss ablation winner is not selected by external metrics")

    if not replay_ok:
        validation_gaps.append("final mainline replay did not complete as a real run")

    replay_eval = replay_run.get("evaluation", {}) if replay_ok and isinstance(replay_run.get("evaluation", {}), dict) else {}
    tapvid_eval = replay_eval.get("tapvid_eval", {}) if isinstance(replay_eval.get("tapvid_eval", {}), dict) else {}
    tapvid3d_eval = replay_eval.get("tapvid3d_limited_eval", {}) if isinstance(replay_eval.get("tapvid3d_limited_eval", {}), dict) else {}

    tapvid_connected = str(tapvid_eval.get("status", "")) == TAP_STATUS_AVAILABLE
    tapvid3d_connected = str(tapvid3d_eval.get("status", "")) == TAP_STATUS_AVAILABLE

    if not tapvid_connected:
        validation_gaps.append(f"tapvid eval not connected: {tapvid_eval.get('status_reason', 'unknown')}")
    if not tapvid3d_connected:
        validation_gaps.append(f"tapvid3d limited eval not connected: {tapvid3d_eval.get('status_reason', 'unknown')}")

    core_strict_ok = bool(
        state_winner_info.get("clear_winner", False)
        and backbone_winner_info.get("clear_winner", False)
        and loss_selection_uses_external
        and replay_ok
    )

    if core_strict_ok and tapvid_connected and tapvid3d_connected:
        validation_status = "scientifically_validated"
    elif core_strict_ok:
        validation_status = "partially_validated"
    else:
        validation_status = "scientifically_not_validated"

    whether_validated_bool = bool(validation_status == "scientifically_validated")
    why_not_fully_validated = "" if whether_validated_bool else "; ".join(validation_gaps)

    backbone_valid = [r for r in backbone_runs if str(r.get("status", "")) == "ok"]
    small_candidates = [r for r in backbone_valid if not _is_220m_run(r)]
    m220_candidates = [r for r in backbone_valid if _is_220m_run(r)]

    best_small_run = _pick_best_run(small_candidates)
    best_220m_run = _pick_best_run(m220_candidates)

    best_small_model = str(best_small_run.get("run_name", "none")) if best_small_run else "none"
    best_220m_model = str(best_220m_run.get("run_name", "none")) if best_220m_run else "none"

    if best_220m_run is None:
        is_220m_competitive = False
        why_220m_not_competitive = "no 220m-qualified run found in backbone ablation"
    elif best_small_run is None:
        is_220m_competitive = True
        why_220m_not_competitive = ""
    else:
        m220_primary = _metric_tuple(best_220m_run)[0]
        small_primary = _metric_tuple(best_small_run)[0]
        primary_gap = float(m220_primary - small_primary)
        is_220m_competitive = bool(primary_gap <= float(args.backbone_min_win_delta))
        if is_220m_competitive:
            why_220m_not_competitive = ""
        else:
            why_220m_not_competitive = (
                "220m primary endpoint l2 is not yet competitive vs best small model "
                f"(gap={primary_gap:.6f}, required<={float(args.backbone_min_win_delta):.6f})"
            )

    should_promote_220m_now = bool(
        whether_validated_bool
        and best_220m_run is not None
        and best_backbone is not None
        and str(best_backbone.get("run_name", "")) == str(best_220m_run.get("run_name", ""))
        and is_220m_competitive
    )

    if replay_ok and replay_run is not None:
        final_mainline_model = (
            f"stage1_v2_mainline::{replay_run.get('state_variant', 'none')}::"
            f"{replay_run.get('backbone_variant', 'none')}::"
            f"{replay_run.get('loss_variant', 'none')}"
        )
        final_mainline_parameter_count = int((replay_run.get("model", {}) if isinstance(replay_run.get("model", {}), dict) else {}).get("estimated_parameter_count", 0))
        final_mainline_target_220m_range_pass = bool((replay_run.get("model", {}) if isinstance(replay_run.get("model", {}), dict) else {}).get("target_220m_range_pass", False))
    else:
        final_mainline_model = "none"
        final_mainline_parameter_count = 0
        final_mainline_target_220m_range_pass = False

    if validation_status != "scientifically_validated":
        next_step_choice = "fix_validation_gaps_then_revalidate"
    elif should_promote_220m_now:
        next_step_choice = "promote_220m_to_mainline"
    elif not is_220m_competitive:
        next_step_choice = "run_220m_competitiveness_gap_closure"
    else:
        next_step_choice = "continue_stage1_v2_real_training"

    final_payload = {
        "generated_at_utc": now_iso(),
        "contract_path": str(args.contract_path),
        "recommended_runtime_path": str(args.recommended_runtime_json),
        "single_gpu_only": bool(runtime.single_gpu_only),

        "best_state_variant": best_state_variant,
        "best_backbone_variant": best_backbone_variant,
        "best_loss_variant": best_loss_variant,

        "final_mainline_model": final_mainline_model,
        "final_mainline_parameter_count": int(final_mainline_parameter_count),
        "final_mainline_target_220m_range_pass": bool(final_mainline_target_220m_range_pass),
        "final_mainline_summary_path": str(args.mainline_replay_json),

        "validation_status": validation_status,
        "validation_gaps": validation_gaps,
        "why_not_fully_validated": why_not_fully_validated,
        "whether_v2_is_scientifically_validated": whether_validated_bool,

        "best_small_model": best_small_model,
        "best_220m_model": best_220m_model,
        "current_best_small_model": best_small_model,
        "current_best_220m_model": best_220m_model,
        "whether_220m_scientifically_competitive": bool(is_220m_competitive),
        "what_blocks_220m_competitiveness": why_220m_not_competitive,
        "should_promote_220m_now": bool(should_promote_220m_now),

        "next_step_choice": str(next_step_choice),

        "winner_analysis": {
            "state": state_winner_info,
            "backbone": backbone_winner_info,
            "loss_selection_uses_external_metrics": loss_selection_uses_external,
        },

        "evidence": {
            "state_ablation": str(args.state_report_json),
            "backbone_ablation": str(args.backbone_report_json),
            "loss_ablation": str(args.losses_report_json),
            "mainline_replay": str(args.mainline_replay_json),
            "run_details": str(args.run_details_json),
        },
    }

    final_report_json = Path(args.final_report_json)
    final_report_md = Path(args.final_report_md)
    final_report_json.parent.mkdir(parents=True, exist_ok=True)
    final_report_md.parent.mkdir(parents=True, exist_ok=True)
    final_report_json.write_text(json.dumps(final_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# TRACEWM Stage1-v2 Scientific Rigor-Fix Results",
        "",
        f"- generated_at_utc: {final_payload['generated_at_utc']}",
        f"- best_state_variant: {final_payload['best_state_variant']}",
        f"- best_backbone_variant: {final_payload['best_backbone_variant']}",
        f"- best_loss_variant: {final_payload['best_loss_variant']}",
        f"- final_mainline_model: {final_payload['final_mainline_model']}",
        f"- final_mainline_parameter_count: {final_payload['final_mainline_parameter_count']}",
        f"- final_mainline_target_220m_range_pass: {final_payload['final_mainline_target_220m_range_pass']}",
        f"- validation_status: {final_payload['validation_status']}",
        f"- whether_v2_is_scientifically_validated: {final_payload['whether_v2_is_scientifically_validated']}",
        f"- best_small_model: {final_payload['best_small_model']}",
        f"- best_220m_model: {final_payload['best_220m_model']}",
        f"- should_promote_220m_now: {final_payload['should_promote_220m_now']}",
        f"- next_step_choice: {final_payload['next_step_choice']}",
        "",
        "## Validation Gaps",
    ]

    if validation_gaps:
        for g in validation_gaps:
            lines.append(f"- {g}")
    else:
        lines.append("- none")

    lines.extend(
        [
            "",
            "## Evaluation Policy",
            "- primary: free_rollout_endpoint_l2",
            "- secondary: free_rollout_coord_mean_l2",
            "- tertiary: tapvid_eval.free_rollout_endpoint_l2 or tapvid3d_limited_eval.free_rollout_endpoint_l2",
            "- total_loss: reference_only_not_primary",
            "",
            "## Mainline Replay Evidence",
            f"- replay_report_json: {args.mainline_replay_json}",
            f"- replay_report_md: {args.mainline_replay_md}",
        ]
    )

    final_report_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"[stage1-v2-scireval] state_report={args.state_report_json}")
    print(f"[stage1-v2-scireval] backbone_report={args.backbone_report_json}")
    print(f"[stage1-v2-scireval] losses_report={args.losses_report_json}")
    print(f"[stage1-v2-scireval] mainline_replay={args.mainline_replay_json}")
    print(f"[stage1-v2-scireval] final_report={args.final_report_json}")
    print(f"[stage1-v2-scireval] validation_status={final_payload['validation_status']}")
    print(f"[stage1-v2-scireval] final_mainline_model={final_payload['final_mainline_model']}")
    print(f"[stage1-v2-scireval] should_promote_220m_now={final_payload['should_promote_220m_now']}")
    print(f"[stage1-v2-scireval] next_step_choice={final_payload['next_step_choice']}")


if __name__ == "__main__":
    main()
