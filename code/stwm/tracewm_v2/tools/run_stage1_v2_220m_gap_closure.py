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

from stwm.tracewm_v2.datasets.stage1_v2_unified import Stage1V2UnifiedDataset, stage1_v2_collate_fn
from stwm.tracewm_v2.losses.structured_trace_loss import StructuredTraceLoss, StructuredTraceLossConfig
from stwm.tracewm_v2.tools.run_stage1_v2_scientific_revalidation import (
    TAP_STATUS_AVAILABLE,
    _build_model,
    _free_rollout_coords,
    _future_pred,
    _load_runtime_config,
    _masked_sum,
    _to_device,
    _evaluate_tap_metrics_for_model,
)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass
class GapRunSpec:
    run_name: str
    mode: str
    backbone_variant: str
    state_variant: str
    loss_variant: str
    lr: float
    weight_decay: float
    warmup_steps: int
    batch_size: int
    grad_accum_steps: int
    clip_grad_norm: float
    coord_weight: float
    visibility_weight: float
    residual_weight: float
    velocity_weight: float
    endpoint_weight: float
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


def parse_args() -> Any:
    p = ArgumentParser(description="Stage1-v2 prototype_220m competitiveness gap-closure round")
    p.add_argument("--contract-path", default="/home/chen034/workspace/data/_manifests/stage1_v2_trace_cache_contract_20260408.json")
    p.add_argument("--recommended-runtime-json", default="/home/chen034/workspace/stwm/reports/stage1_v2_recommended_runtime_20260408.json")
    p.add_argument("--stage1-minisplit-path", default="/home/chen034/workspace/data/_manifests/stage1_minisplits_20260408.json")
    p.add_argument("--data-root", default="/home/chen034/workspace/data")

    p.add_argument("--runs-summary-json", default="/home/chen034/workspace/stwm/reports/stage1_v2_220m_gap_closure_runs_20260408.json")
    p.add_argument("--comparison-json", default="/home/chen034/workspace/stwm/reports/stage1_v2_220m_gap_closure_comparison_20260408.json")
    p.add_argument("--results-md", default="/home/chen034/workspace/stwm/docs/STAGE1_V2_220M_GAP_CLOSURE_RESULTS_20260408.md")

    p.add_argument("--dataset-names", nargs="*", default=["pointodyssey", "kubric"])
    p.add_argument("--obs-len", type=int, default=8)
    p.add_argument("--fut-len", type=int, default=8)
    p.add_argument("--max-tokens", type=int, default=64)
    p.add_argument("--max-samples-per-dataset-train", type=int, default=128)
    p.add_argument("--max-samples-per-dataset-val", type=int, default=64)

    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--train-steps", type=int, default=40)
    p.add_argument("--eval-steps", type=int, default=8)

    p.add_argument("--ref-lr", type=float, default=1e-4)
    p.add_argument("--ref-weight-decay", type=float, default=0.0)
    p.add_argument("--ref-batch-size", type=int, default=2)
    p.add_argument("--ref-grad-accum", type=int, default=1)
    p.add_argument("--ref-clip-grad", type=float, default=1.0)

    p.add_argument("--eval-max-tapvid-samples", type=int, default=6)
    p.add_argument("--eval-max-tapvid3d-samples", type=int, default=12)

    p.add_argument("--gru-hidden-dim", type=int, default=384)
    p.add_argument("--gru-num-layers", type=int, default=2)

    p.add_argument("--seed", type=int, default=20260408)
    return p.parse_args()


def _metric_key(run: Dict[str, Any]) -> Tuple[float, float, float, float, str]:
    ev = run.get("evaluation", {}) if isinstance(run.get("evaluation", {}), dict) else {}

    primary = _safe_float(ev.get("free_rollout_endpoint_l2"), 1e9)
    secondary = _safe_float(ev.get("free_rollout_coord_mean_l2"), 1e9)

    tap = ev.get("tapvid_eval", {}) if isinstance(ev.get("tapvid_eval", {}), dict) else {}
    tap3d = ev.get("tapvid3d_limited_eval", {}) if isinstance(ev.get("tapvid3d_limited_eval", {}), dict) else {}

    tertiary = 1e9
    if str(tap.get("status", "")) == TAP_STATUS_AVAILABLE:
        tertiary = _safe_float(tap.get("free_rollout_endpoint_l2"), 1e9)

    quaternary = 1e9
    if str(tap3d.get("status", "")) == TAP_STATUS_AVAILABLE:
        quaternary = _safe_float(tap3d.get("free_rollout_endpoint_l2"), 1e9)

    return primary, secondary, tertiary, quaternary, str(run.get("run_name", ""))


def _winner_reason(run: Dict[str, Any], best: Dict[str, Any]) -> str:
    if str(run.get("run_name", "")) == str(best.get("run_name", "")):
        return "winner by primary->secondary->tertiary->quaternary"

    rp, rs, rt, rq, _ = _metric_key(run)
    bp, bs, bt, bq, _ = _metric_key(best)

    if abs(rp - bp) > 1e-12:
        return f"loses on primary by +{(rp - bp):.6f}"
    if abs(rs - bs) > 1e-12:
        return f"primary tied, loses on secondary by +{(rs - bs):.6f}"
    if abs(rt - bt) > 1e-12:
        return f"primary+secondary tied, loses on tertiary by +{(rt - bt):.6f}"
    return f"primary+secondary+tertiary tied, loses on quaternary by +{(rq - bq):.6f}"


def _build_loss(spec: GapRunSpec) -> StructuredTraceLoss:
    cfg = StructuredTraceLossConfig(
        coord_weight=float(spec.coord_weight),
        visibility_weight=float(spec.visibility_weight),
        residual_weight=float(spec.residual_weight),
        velocity_weight=float(spec.velocity_weight),
        endpoint_weight=float(spec.endpoint_weight),
        enable_visibility=bool(spec.enable_visibility),
        enable_residual=bool(spec.enable_residual),
        enable_velocity=bool(spec.enable_velocity),
        enable_endpoint=bool(spec.enable_endpoint),
    )
    return StructuredTraceLoss(cfg)


def _set_optimizer_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = float(lr)


def _evaluate_in_domain(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    obs_len: int,
    fut_len: int,
    pin_memory: bool,
    eval_steps: int,
) -> Dict[str, float]:
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

    return {
        "teacher_forced_coord_loss": float(tf_coord_sum / max(tf_coord_count, 1.0)),
        "free_rollout_coord_mean_l2": float(rollout_l2_sum / max(rollout_l2_count, 1.0)),
        "free_rollout_endpoint_l2": float(endpoint_l2_sum / max(endpoint_l2_count, 1.0)),
    }


def _train_one_run(
    spec: GapRunSpec,
    train_dataset: Stage1V2UnifiedDataset,
    val_dataset: Stage1V2UnifiedDataset,
    runtime: Any,
    args: Any,
    seed: int,
) -> Dict[str, Any]:
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, model_meta = _build_model(
        spec=type(
            "SpecWrap",
            (),
            {
                "backbone_variant": spec.backbone_variant,
                "state_variant": spec.state_variant,
            },
        )(),
        max_tokens=int(args.max_tokens),
        gru_hidden_dim=int(args.gru_hidden_dim),
        gru_num_layers=int(args.gru_num_layers),
    )
    model = model.to(device)

    criterion = _build_loss(spec)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(spec.lr), weight_decay=float(spec.weight_decay))

    train_loader_kwargs: Dict[str, Any] = {
        "dataset": train_dataset,
        "batch_size": int(spec.batch_size),
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
        batch_size=max(1, int(spec.batch_size)),
        shuffle=False,
        num_workers=0,
        pin_memory=bool(runtime.pin_memory),
        collate_fn=stage1_v2_collate_fn,
    )

    train_steps = int(args.train_steps)
    grad_accum_steps = max(1, int(spec.grad_accum_steps))

    history: List[Dict[str, float]] = []
    data_iter = iter(train_loader)
    total_micro_steps = 0

    model.train()
    running = {
        "total_loss": 0.0,
        "coord_loss": 0.0,
        "visibility_loss": 0.0,
        "residual_loss": 0.0,
        "velocity_loss": 0.0,
        "endpoint_loss": 0.0,
    }

    for step in range(train_steps):
        if int(spec.warmup_steps) > 0 and step < int(spec.warmup_steps):
            lr_now = float(spec.lr) * float(step + 1) / float(max(int(spec.warmup_steps), 1))
        else:
            lr_now = float(spec.lr)
        _set_optimizer_lr(optimizer, lr_now)

        optimizer.zero_grad(set_to_none=True)

        for _micro in range(grad_accum_steps):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                batch = next(data_iter)

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

            (losses["total_loss"] / float(grad_accum_steps)).backward()

            for key in running:
                running[key] += float(losses[key].detach().cpu().item())
            total_micro_steps += 1

        if float(spec.clip_grad_norm) > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(spec.clip_grad_norm))
        optimizer.step()

    denom = float(max(total_micro_steps, 1))
    history.append({k: float(v / denom) for k, v in running.items()})
    train_final = history[-1]

    in_domain = _evaluate_in_domain(
        model=model,
        loader=val_loader,
        device=device,
        obs_len=int(args.obs_len),
        fut_len=int(args.fut_len),
        pin_memory=bool(runtime.pin_memory),
        eval_steps=int(args.eval_steps),
    )

    tap_eval = _evaluate_tap_metrics_for_model(model=model, device=device, args=args)

    evaluation = {
        "teacher_forced_coord_loss": float(in_domain["teacher_forced_coord_loss"]),
        "free_rollout_coord_mean_l2": float(in_domain["free_rollout_coord_mean_l2"]),
        "free_rollout_endpoint_l2": float(in_domain["free_rollout_endpoint_l2"]),
        "tapvid_eval": tap_eval["tapvid_eval"],
        "tapvid3d_limited_eval": tap_eval["tapvid3d_limited_eval"],
    }

    effective_batch = int(spec.batch_size) * int(grad_accum_steps)

    return {
        "run_name": spec.run_name,
        "mode": spec.mode,
        "status": "ok",
        "state_variant": spec.state_variant,
        "backbone_variant": spec.backbone_variant,
        "loss_variant": spec.loss_variant,
        "model": model_meta,
        "train_config": {
            "lr": float(spec.lr),
            "weight_decay": float(spec.weight_decay),
            "warmup_steps": int(spec.warmup_steps),
            "batch_size": int(spec.batch_size),
            "grad_accum_steps": int(grad_accum_steps),
            "clip_grad_norm": float(spec.clip_grad_norm),
            "coord_weight": float(spec.coord_weight),
            "visibility_weight": float(spec.visibility_weight),
            "residual_weight": float(spec.residual_weight),
            "velocity_weight": float(spec.velocity_weight),
            "endpoint_weight": float(spec.endpoint_weight),
            "enable_visibility": bool(spec.enable_visibility),
            "enable_residual": bool(spec.enable_residual),
            "enable_velocity": bool(spec.enable_velocity),
            "enable_endpoint": bool(spec.enable_endpoint),
        },
        "train_budget": {
            "epochs": int(args.epochs),
            "optimizer_steps": int(train_steps),
            "micro_steps": int(total_micro_steps),
            "eval_steps": int(args.eval_steps),
            "effective_batch": int(effective_batch),
        },
        "runtime": {
            "num_workers": int(runtime.num_workers),
            "pin_memory": bool(runtime.pin_memory),
            "persistent_workers": bool(runtime.persistent_workers),
            "prefetch_factor": int(runtime.prefetch_factor),
            "single_gpu_only": bool(runtime.single_gpu_only),
        },
        "train_final_metrics": train_final,
        "evaluation": evaluation,
        "notes": spec.notes,
    }


def _build_run_specs(args: Any) -> List[GapRunSpec]:
    ref_lr = float(args.ref_lr)
    ref_wd = float(args.ref_weight_decay)
    ref_bs = int(args.ref_batch_size)
    ref_acc = int(args.ref_grad_accum)
    ref_clip = float(args.ref_clip_grad)

    return [
        GapRunSpec(
            run_name="stage1_v2_gap_debugsmall_ref",
            mode="debugsmall_ref",
            backbone_variant="stage1_v2_backbone_transformer_debugsmall",
            state_variant="multitoken",
            loss_variant="coord_visibility",
            lr=ref_lr,
            weight_decay=ref_wd,
            warmup_steps=0,
            batch_size=ref_bs,
            grad_accum_steps=ref_acc,
            clip_grad_norm=ref_clip,
            coord_weight=1.0,
            visibility_weight=0.5,
            residual_weight=0.25,
            velocity_weight=0.25,
            endpoint_weight=0.1,
            enable_visibility=True,
            enable_residual=False,
            enable_velocity=False,
            enable_endpoint=False,
            notes="small-model reference under longer_train_short_eval budget",
        ),
        GapRunSpec(
            run_name="stage1_v2_gap_220m_ref",
            mode="220m_ref",
            backbone_variant="stage1_v2_backbone_transformer_prototype220m",
            state_variant="multitoken",
            loss_variant="coord_visibility",
            lr=ref_lr,
            weight_decay=ref_wd,
            warmup_steps=0,
            batch_size=ref_bs,
            grad_accum_steps=ref_acc,
            clip_grad_norm=ref_clip,
            coord_weight=1.0,
            visibility_weight=0.5,
            residual_weight=0.25,
            velocity_weight=0.25,
            endpoint_weight=0.1,
            enable_visibility=True,
            enable_residual=False,
            enable_velocity=False,
            enable_endpoint=False,
            notes="220m pure scaling reference with matched setup",
        ),
        GapRunSpec(
            run_name="stage1_v2_gap_220m_opt_lr",
            mode="220m_opt_lr",
            backbone_variant="stage1_v2_backbone_transformer_prototype220m",
            state_variant="multitoken",
            loss_variant="coord_visibility",
            lr=6e-5,
            weight_decay=0.01,
            warmup_steps=8,
            batch_size=ref_bs,
            grad_accum_steps=ref_acc,
            clip_grad_norm=ref_clip,
            coord_weight=1.0,
            visibility_weight=0.5,
            residual_weight=0.25,
            velocity_weight=0.25,
            endpoint_weight=0.1,
            enable_visibility=True,
            enable_residual=False,
            enable_velocity=False,
            enable_endpoint=False,
            notes="220m optimizer/lr/warmup/wd only",
        ),
        GapRunSpec(
            run_name="stage1_v2_gap_220m_opt_batch",
            mode="220m_opt_batch",
            backbone_variant="stage1_v2_backbone_transformer_prototype220m",
            state_variant="multitoken",
            loss_variant="coord_visibility",
            lr=ref_lr,
            weight_decay=ref_wd,
            warmup_steps=0,
            batch_size=1,
            grad_accum_steps=4,
            clip_grad_norm=0.8,
            coord_weight=1.0,
            visibility_weight=0.5,
            residual_weight=0.25,
            velocity_weight=0.25,
            endpoint_weight=0.1,
            enable_visibility=True,
            enable_residual=False,
            enable_velocity=False,
            enable_endpoint=False,
            notes="220m batch/accum/clip only",
        ),
        GapRunSpec(
            run_name="stage1_v2_gap_220m_opt_lossweights",
            mode="220m_opt_lossweights",
            backbone_variant="stage1_v2_backbone_transformer_prototype220m",
            state_variant="multitoken",
            loss_variant="coord_visibility",
            lr=ref_lr,
            weight_decay=ref_wd,
            warmup_steps=0,
            batch_size=ref_bs,
            grad_accum_steps=ref_acc,
            clip_grad_norm=ref_clip,
            coord_weight=1.2,
            visibility_weight=0.8,
            residual_weight=0.25,
            velocity_weight=0.25,
            endpoint_weight=0.1,
            enable_visibility=True,
            enable_residual=False,
            enable_velocity=False,
            enable_endpoint=False,
            notes="220m small loss-weight tuning within same coord_visibility family",
        ),
    ]


def _fmt_metric(run: Dict[str, Any]) -> Dict[str, float]:
    ev = run.get("evaluation", {}) if isinstance(run.get("evaluation", {}), dict) else {}
    tap = ev.get("tapvid_eval", {}) if isinstance(ev.get("tapvid_eval", {}), dict) else {}
    tap3d = ev.get("tapvid3d_limited_eval", {}) if isinstance(ev.get("tapvid3d_limited_eval", {}), dict) else {}
    return {
        "teacher_forced_coord_loss": _safe_float(ev.get("teacher_forced_coord_loss"), 1e9),
        "free_rollout_coord_mean_l2": _safe_float(ev.get("free_rollout_coord_mean_l2"), 1e9),
        "free_rollout_endpoint_l2": _safe_float(ev.get("free_rollout_endpoint_l2"), 1e9),
        "tapvid_endpoint_l2": _safe_float(tap.get("free_rollout_endpoint_l2"), 1e9),
        "tapvid3d_limited_endpoint_l2": _safe_float(tap3d.get("free_rollout_endpoint_l2"), 1e9),
    }


def main() -> None:
    args = parse_args()

    runtime = _load_runtime_config(args.recommended_runtime_json)
    print(
        "[stage1-v2-gap] runtime "
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

    specs = _build_run_specs(args)
    if len(specs) != 5:
        raise RuntimeError(f"gap closure matrix must contain exactly 5 runs, got {len(specs)}")

    runs: List[Dict[str, Any]] = []
    for idx, spec in enumerate(specs):
        print(f"[stage1-v2-gap] run={spec.run_name} mode={spec.mode}")
        run = _train_one_run(
            spec=spec,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            runtime=runtime,
            args=args,
            seed=int(args.seed) + idx,
        )
        runs.append(run)

    runs_summary = {
        "generated_at_utc": now_iso(),
        "objective": "prototype_220m competitiveness gap closure under fixed Stage1-v2 scientific protocol",
        "contract_path": str(args.contract_path),
        "recommended_runtime_path": str(args.recommended_runtime_json),
        "single_gpu_only": bool(runtime.single_gpu_only),
        "training_budget_policy": {
            "mode": "longer_train_short_eval",
            "optimizer_steps": int(args.train_steps),
            "eval_steps": int(args.eval_steps),
            "epochs": int(args.epochs),
        },
        "selection_policy": {
            "primary": "free_rollout_endpoint_l2",
            "secondary": "free_rollout_coord_mean_l2",
            "tertiary": "tapvid_eval.free_rollout_endpoint_l2",
            "quaternary": "tapvid3d_limited_eval.free_rollout_endpoint_l2",
            "total_loss_usage": "reference_only",
        },
        "runs": runs,
    }

    runs_summary_path = Path(args.runs_summary_json)
    runs_summary_path.parent.mkdir(parents=True, exist_ok=True)
    runs_summary_path.write_text(json.dumps(runs_summary, ensure_ascii=False, indent=2), encoding="utf-8")

    run_map = {str(r.get("run_name", "")): r for r in runs}
    small_ref = run_map["stage1_v2_gap_debugsmall_ref"]
    ref_220m = run_map["stage1_v2_gap_220m_ref"]

    all_220m = [r for r in runs if str(r.get("run_name", "")).startswith("stage1_v2_gap_220m")]
    best_220m = min(all_220m, key=_metric_key)
    best_overall = min(runs, key=_metric_key)

    p_small, s_small, t_small, q_small, _ = _metric_key(small_ref)
    p_220m_ref, s_220m_ref, t_220m_ref, q_220m_ref, _ = _metric_key(ref_220m)
    p_best_220m, s_best_220m, t_best_220m, q_best_220m, _ = _metric_key(best_220m)

    small_vs_220m_ref_gap = {
        "primary_endpoint_l2_gap": float(p_220m_ref - p_small),
        "secondary_mean_l2_gap": float(s_220m_ref - s_small),
        "tertiary_tapvid_gap": float(t_220m_ref - t_small),
        "quaternary_tapvid3d_limited_gap": float(q_220m_ref - q_small),
    }

    opt_candidates = [
        run_map["stage1_v2_gap_220m_opt_lr"],
        run_map["stage1_v2_gap_220m_opt_batch"],
        run_map["stage1_v2_gap_220m_opt_lossweights"],
    ]
    best_opt = min(opt_candidates, key=_metric_key)
    p_best_opt, s_best_opt, t_best_opt, q_best_opt, _ = _metric_key(best_opt)

    best_opt_vs_ref = {
        "primary_endpoint_l2_delta_vs_220m_ref": float(p_best_opt - p_220m_ref),
        "secondary_mean_l2_delta_vs_220m_ref": float(s_best_opt - s_220m_ref),
        "tertiary_tapvid_delta_vs_220m_ref": float(t_best_opt - t_220m_ref),
        "quaternary_tapvid3d_delta_vs_220m_ref": float(q_best_opt - q_220m_ref),
    }

    competitive = bool(
        p_best_220m <= p_small
        and s_best_220m <= s_small
        and t_best_220m <= t_small
        and q_best_220m <= q_small
    )

    should_promote_220m_now = bool(
        competitive and str(best_overall.get("run_name", "")).startswith("stage1_v2_gap_220m")
    )

    if should_promote_220m_now:
        next_step_choice = "promote_220m_as_mainline"
    else:
        improved_vs_220m_ref = bool(p_best_opt < p_220m_ref - 1e-9)
        if improved_vs_220m_ref:
            next_step_choice = "keep_debugsmall_as_mainline_and_continue_220m"
        else:
            next_step_choice = "stop_220m_for_now"

    comparison = {
        "generated_at_utc": now_iso(),
        "selection_policy": runs_summary["selection_policy"],
        "small_ref_run": str(small_ref.get("run_name", "")),
        "run_220m_ref": str(ref_220m.get("run_name", "")),
        "best_220m_run": str(best_220m.get("run_name", "")),
        "best_220m_optimization_run": str(best_opt.get("run_name", "")),
        "small_ref_vs_220m_ref_gap": small_vs_220m_ref_gap,
        "best_220m_optimization_effect": best_opt_vs_ref,
        "whether_220m_is_competitive": bool(competitive),
        "should_promote_220m_now": bool(should_promote_220m_now),
        "why_not_promote_220m": "" if should_promote_220m_now else (
            "best 220m run still lags debug_small on at least one ranked metric"
        ),
        "remaining_gap_to_small_ref": {
            "primary_endpoint_l2_gap": float(p_best_220m - p_small),
            "secondary_mean_l2_gap": float(s_best_220m - s_small),
            "tertiary_tapvid_gap": float(t_best_220m - t_small),
            "quaternary_tapvid3d_limited_gap": float(q_best_220m - q_small),
        },
        "next_step_choice": str(next_step_choice),
        "allowed_next_step_choice": [
            "promote_220m_as_mainline",
            "keep_debugsmall_as_mainline_and_continue_220m",
            "stop_220m_for_now",
        ],
        "evidence": {
            "runs_summary": str(args.runs_summary_json),
        },
    }

    comparison_path = Path(args.comparison_json)
    comparison_path.parent.mkdir(parents=True, exist_ok=True)
    comparison_path.write_text(json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8")

    results_md = Path(args.results_md)
    results_md.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Stage1-v2 220M Gap Closure Results",
        "",
        f"- generated_at_utc: {comparison['generated_at_utc']}",
        f"- small_ref_run: {comparison['small_ref_run']}",
        f"- run_220m_ref: {comparison['run_220m_ref']}",
        f"- best_220m_run: {comparison['best_220m_run']}",
        f"- best_220m_optimization_run: {comparison['best_220m_optimization_run']}",
        f"- should_promote_220m_now: {comparison['should_promote_220m_now']}",
        f"- next_step_choice: {comparison['next_step_choice']}",
        "",
        "## Key Answer",
        f"- small_ref vs 220m_ref primary gap: {comparison['small_ref_vs_220m_ref_gap']['primary_endpoint_l2_gap']:.6f}",
        f"- best 220m remaining primary gap to small_ref: {comparison['remaining_gap_to_small_ref']['primary_endpoint_l2_gap']:.6f}",
        "",
        "## Ranked Metrics Table",
        "| run | primary_endpoint_l2 | secondary_mean_l2 | tertiary_tapvid | quaternary_tapvid3d_limited | total_loss_ref | effective_batch | params_est | winner_reason_vs_best_overall |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]

    for r in sorted(runs, key=_metric_key):
        ev = r.get("evaluation", {}) if isinstance(r.get("evaluation", {}), dict) else {}
        tfm = r.get("train_final_metrics", {}) if isinstance(r.get("train_final_metrics", {}), dict) else {}
        budget = r.get("train_budget", {}) if isinstance(r.get("train_budget", {}), dict) else {}
        model = r.get("model", {}) if isinstance(r.get("model", {}), dict) else {}
        p, s, t, q, _ = _metric_key(r)
        reason = _winner_reason(r, best_overall)
        lines.append(
            f"| {r.get('run_name', '-')} | {p:.6f} | {s:.6f} | {t:.6f} | {q:.6f} | {_safe_float(tfm.get('total_loss'), 0.0):.6f} | {int(budget.get('effective_batch', 0))} | {int(model.get('estimated_parameter_count', 0))} | {reason} |"
        )

    lines.extend(
        [
            "",
            "## 220M Optimization Effect",
            f"- best optimization run: {comparison['best_220m_optimization_run']}",
            f"- delta primary vs 220m_ref: {comparison['best_220m_optimization_effect']['primary_endpoint_l2_delta_vs_220m_ref']:.6f}",
            f"- delta secondary vs 220m_ref: {comparison['best_220m_optimization_effect']['secondary_mean_l2_delta_vs_220m_ref']:.6f}",
            f"- delta tertiary vs 220m_ref: {comparison['best_220m_optimization_effect']['tertiary_tapvid_delta_vs_220m_ref']:.6f}",
            f"- delta quaternary vs 220m_ref: {comparison['best_220m_optimization_effect']['quaternary_tapvid3d_delta_vs_220m_ref']:.6f}",
        ]
    )

    results_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"[stage1-v2-gap] runs_summary={args.runs_summary_json}")
    print(f"[stage1-v2-gap] comparison={args.comparison_json}")
    print(f"[stage1-v2-gap] best_220m_run={comparison['best_220m_run']}")
    print(f"[stage1-v2-gap] should_promote_220m_now={comparison['should_promote_220m_now']}")
    print(f"[stage1-v2-gap] next_step_choice={comparison['next_step_choice']}")


if __name__ == "__main__":
    main()
