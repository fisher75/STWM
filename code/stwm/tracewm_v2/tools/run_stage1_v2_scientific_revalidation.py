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
from torch.utils.data import DataLoader

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
    parser = ArgumentParser(description="Stage1-v2 scientific revalidation round")
    parser.add_argument("--contract-path", default="/home/chen034/workspace/data/_manifests/stage1_v2_trace_cache_contract_20260408.json")
    parser.add_argument("--recommended-runtime-json", default="/home/chen034/workspace/stwm/reports/stage1_v2_recommended_runtime_20260408.json")
    parser.add_argument("--state-report-json", default="/home/chen034/workspace/stwm/reports/stage1_v2_ablation_state_20260408.json")
    parser.add_argument("--state-report-md", default="/home/chen034/workspace/stwm/docs/STAGE1_V2_ABLATION_STATE_20260408.md")
    parser.add_argument("--backbone-report-json", default="/home/chen034/workspace/stwm/reports/stage1_v2_ablation_backbone_20260408.json")
    parser.add_argument("--backbone-report-md", default="/home/chen034/workspace/stwm/docs/STAGE1_V2_ABLATION_BACKBONE_20260408.md")
    parser.add_argument("--losses-report-json", default="/home/chen034/workspace/stwm/reports/stage1_v2_ablation_losses_20260408.json")
    parser.add_argument("--losses-report-md", default="/home/chen034/workspace/stwm/docs/STAGE1_V2_ABLATION_LOSSES_20260408.md")
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
    parser.add_argument("--gru-hidden-dim", type=int, default=384)
    parser.add_argument("--gru-num-layers", type=int, default=2)
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


def _evaluate_model(
    model: torch.nn.Module,
    criterion: StructuredTraceLoss,
    loader: DataLoader,
    device: torch.device,
    obs_len: int,
    fut_len: int,
    pin_memory: bool,
    eval_steps: int,
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
            fut_target = full_state[:, obs_len:]
            fut_valid = full_valid[:, obs_len:]
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

    tapvid_eval = {
        "supported": False,
        "status": "unavailable",
        "free_rollout_endpoint_l2": None,
    }
    tapvid3d_limited_eval = {
        "supported": False,
        "status": "unavailable",
        "free_rollout_endpoint_l2": None,
    }

    return {
        "teacher_forced_coord_loss": teacher_forced_coord_loss,
        "free_rollout_coord_mean_l2": free_rollout_coord_mean_l2,
        "free_rollout_endpoint_l2": free_rollout_endpoint_l2,
        "tapvid_eval": tapvid_eval,
        "tapvid3d_limited_eval": tapvid3d_limited_eval,
    }


def _train_variant(
    spec: VariantSpec,
    train_dataset: Stage1V2UnifiedDataset,
    val_dataset: Stage1V2UnifiedDataset,
    runtime: RuntimeConfig,
    args: Any,
    seed: int,
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
        while step_count < int(args.steps_per_variant):
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
        eval_steps=int(args.eval_steps),
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


def _metric_tuple(run: Dict[str, Any]) -> Tuple[float, float, float, str]:
    ev = run.get("evaluation", {}) if isinstance(run.get("evaluation", {}), dict) else {}
    endpoint = float(ev.get("free_rollout_endpoint_l2", 1e9) or 1e9)
    mean_l2 = float(ev.get("free_rollout_coord_mean_l2", 1e9) or 1e9)

    tapvid = ev.get("tapvid_eval", {}) if isinstance(ev.get("tapvid_eval", {}), dict) else {}
    tap_metric_raw = tapvid.get("free_rollout_endpoint_l2", None)
    tap_metric = float(tap_metric_raw) if tap_metric_raw is not None else 1e9
    return endpoint, mean_l2, tap_metric, str(run.get("run_name", ""))


def _pick_best_run(runs: List[Dict[str, Any]]) -> Dict[str, Any] | None:
    valid = [r for r in runs if str(r.get("status", "")) == "ok"]
    if not valid:
        return None
    return min(valid, key=_metric_tuple)


def _write_group_report(
    *,
    report_json: Path,
    report_md: Path,
    group_name: str,
    objective: str,
    runs: List[Dict[str, Any]],
    best_run: Dict[str, Any] | None,
    include_total_loss_reference: bool,
) -> None:
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_md.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "generated_at_utc": now_iso(),
        "group": group_name,
        "objective": objective,
        "selection_policy": {
            "primary": "free_rollout_endpoint_l2",
            "secondary": "free_rollout_coord_mean_l2",
            "tertiary": "tapvid_eval.free_rollout_endpoint_l2",
            "total_loss_usage": "reference_only" if include_total_loss_reference else "not_used",
        },
        "runs": runs,
        "best_variant": str(best_run.get("run_name", "none")) if best_run else "none",
    }
    report_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        f"# Stage1-v2 {group_name.title()} Ablation",
        "",
        f"- generated_at_utc: {payload['generated_at_utc']}",
        f"- objective: {objective}",
        f"- best_variant: {payload['best_variant']}",
        "- selection_policy: primary=free_rollout_endpoint_l2, secondary=free_rollout_coord_mean_l2, tertiary=tapvid_eval",
    ]
    if include_total_loss_reference:
        lines.append("- total_loss is reference only and not used as main selection key")

    lines.extend(
        [
            "",
            "| variant | status | teacher_forced_coord_loss | free_rollout_coord_mean_l2 | free_rollout_endpoint_l2 | tapvid_eval | tapvid3d_limited_eval | total_loss_ref |",
            "|---|---|---:|---:|---:|---|---|---:|",
        ]
    )

    for run in runs:
        ev = run.get("evaluation", {}) if isinstance(run.get("evaluation", {}), dict) else {}
        tf = float(ev.get("teacher_forced_coord_loss", 0.0) or 0.0)
        mean_l2 = float(ev.get("free_rollout_coord_mean_l2", 0.0) or 0.0)
        end_l2 = float(ev.get("free_rollout_endpoint_l2", 0.0) or 0.0)
        tapvid_status = str((ev.get("tapvid_eval", {}) if isinstance(ev.get("tapvid_eval", {}), dict) else {}).get("status", "unavailable"))
        tap3d_status = str((ev.get("tapvid3d_limited_eval", {}) if isinstance(ev.get("tapvid3d_limited_eval", {}), dict) else {}).get("status", "unavailable"))
        total_loss_ref = float((run.get("train_final_metrics", {}) if isinstance(run.get("train_final_metrics", {}), dict) else {}).get("total_loss", 0.0) or 0.0)
        lines.append(
            f"| {run.get('run_name', '-')} | {run.get('status', 'fail')} | {tf:.6f} | {mean_l2:.6f} | {end_l2:.6f} | {tapvid_status} | {tap3d_status} | {total_loss_ref:.6f} |"
        )

    report_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


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

    final_mainline_model = (
        f"stage1_v2_mainline::{best_state_variant}::{best_backbone_variant}::{best_loss_variant}"
        if best_state and best_backbone and best_loss
        else "none"
    )

    final_mainline_parameter_count = int(best_backbone.get("model", {}).get("estimated_parameter_count", 0)) if best_backbone else 0
    final_mainline_target_220m_range_pass = bool(best_backbone.get("model", {}).get("target_220m_range_pass", False)) if best_backbone else False

    all_groups_present = bool(best_state and best_backbone and best_loss)
    external_metrics_ready = all(
        float((run.get("evaluation", {}) if isinstance(run.get("evaluation", {}), dict) else {}).get("free_rollout_endpoint_l2", 0.0)) > 0.0
        for run in [best_state, best_backbone, best_loss]
        if run is not None
    )
    whether_validated = bool(all_groups_present and external_metrics_ready)

    if whether_validated and final_mainline_target_220m_range_pass:
        next_step_choice = "continue_stage1_v2_real_training"
    elif whether_validated:
        next_step_choice = "do_small_followup_ablation"
    else:
        next_step_choice = "stage1_v2_not_ready"

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
        "whether_v2_is_scientifically_validated": bool(whether_validated),
        "next_step_choice": str(next_step_choice),
        "evidence": {
            "state_ablation": str(args.state_report_json),
            "backbone_ablation": str(args.backbone_report_json),
            "loss_ablation": str(args.losses_report_json),
            "run_details": str(args.run_details_json),
        },
    }

    final_report_json = Path(args.final_report_json)
    final_report_md = Path(args.final_report_md)
    final_report_json.parent.mkdir(parents=True, exist_ok=True)
    final_report_md.parent.mkdir(parents=True, exist_ok=True)
    final_report_json.write_text(json.dumps(final_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# TRACEWM Stage1-v2 Scientific Revalidation Results",
        "",
        f"- generated_at_utc: {final_payload['generated_at_utc']}",
        f"- best_state_variant: {final_payload['best_state_variant']}",
        f"- best_backbone_variant: {final_payload['best_backbone_variant']}",
        f"- best_loss_variant: {final_payload['best_loss_variant']}",
        f"- final_mainline_model: {final_payload['final_mainline_model']}",
        f"- final_mainline_parameter_count: {final_payload['final_mainline_parameter_count']}",
        f"- final_mainline_target_220m_range_pass: {final_payload['final_mainline_target_220m_range_pass']}",
        f"- whether_v2_is_scientifically_validated: {final_payload['whether_v2_is_scientifically_validated']}",
        f"- next_step_choice: {final_payload['next_step_choice']}",
        "",
        "## Evaluation Policy",
        "- primary: free_rollout_endpoint_l2",
        "- secondary: free_rollout_coord_mean_l2",
        "- tertiary: tapvid_eval.free_rollout_endpoint_l2",
        "- total_loss: reference_only_not_primary",
    ]
    final_report_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"[stage1-v2-scireval] state_report={args.state_report_json}")
    print(f"[stage1-v2-scireval] backbone_report={args.backbone_report_json}")
    print(f"[stage1-v2-scireval] losses_report={args.losses_report_json}")
    print(f"[stage1-v2-scireval] final_report={args.final_report_json}")
    print(f"[stage1-v2-scireval] final_mainline_model={final_payload['final_mainline_model']}")
    print(f"[stage1-v2-scireval] whether_v2_is_scientifically_validated={final_payload['whether_v2_is_scientifically_validated']}")
    print(f"[stage1-v2-scireval] next_step_choice={final_payload['next_step_choice']}")


if __name__ == "__main__":
    main()
