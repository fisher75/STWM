#!/usr/bin/env python3
from __future__ import annotations

from argparse import ArgumentParser
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple
import json
import os
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
from stwm.tracewm_v2.tools.run_stage1_v2_scientific_revalidation import (
    TAP_STATUS_AVAILABLE,
    _evaluate_model,
    _load_runtime_config,
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
    parser.add_argument("--recommended-runtime-json", default="/home/chen034/workspace/stwm/reports/stage1_v2_recommended_runtime_20260408.json")
    parser.add_argument("--use-recommended-runtime", action="store_true")
    parser.add_argument("--stage1-minisplit-path", default="/home/chen034/workspace/data/_manifests/stage1_minisplits_20260408.json")
    parser.add_argument("--data-root", default="/home/chen034/workspace/data")

    parser.add_argument("--dataset-names", nargs="*", default=["pointodyssey", "kubric"])
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--val-split", default="val")
    parser.add_argument("--obs-len", type=int, default=8)
    parser.add_argument("--fut-len", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--max-samples-per-dataset", type=int, default=256)
    parser.add_argument("--max-samples-per-dataset-val", type=int, default=64)

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
    parser.add_argument("--train-steps", type=int, default=0)
    parser.add_argument("--eval-interval", type=int, default=0)
    parser.add_argument("--eval-steps", type=int, default=12)
    parser.add_argument("--save-every-n-steps", type=int, default=1000)
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

    parser.add_argument("--eval-max-tapvid-samples", type=int, default=6)
    parser.add_argument("--eval-max-tapvid3d-samples", type=int, default=12)

    parser.add_argument("--output-dir", default="/home/chen034/workspace/stwm/outputs/training/tracewm_stage1_v2")
    parser.add_argument("--checkpoint-dir", default="")
    parser.add_argument("--resume-from", default="")
    parser.add_argument("--auto-resume-latest", action="store_true")

    parser.add_argument("--summary-json", default="/home/chen034/workspace/stwm/reports/tracewm_stage1_v2_train_summary_20260408.json")
    parser.add_argument("--progress-json", default="")
    parser.add_argument("--final-json", default="")
    parser.add_argument("--results-md", default="/home/chen034/workspace/stwm/docs/TRACEWM_STAGE1_V2_TRAIN_SUMMARY_20260408.md")
    parser.add_argument("--perf-step-timing-json", default="/home/chen034/workspace/stwm/reports/stage1_v2_perf_step_timing_20260408.json")

    parser.add_argument("--run-name", default="stage1_v2_train_mainline")
    parser.add_argument("--ablation-tag", default="mainline")
    parser.add_argument("--run-metadata-note", default="")
    parser.add_argument("--seed", type=int, default=20260408)
    parser.set_defaults(pin_memory=True, persistent_workers=True)
    return parser.parse_args()


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return float(default)
        return float(x)
    except Exception:
        return float(default)


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


def _ranked_metrics_from_evaluation(evaluation: Dict[str, Any]) -> Dict[str, float]:
    tap = evaluation.get("tapvid_eval", {}) if isinstance(evaluation.get("tapvid_eval", {}), dict) else {}
    tap3d = evaluation.get("tapvid3d_limited_eval", {}) if isinstance(evaluation.get("tapvid3d_limited_eval", {}), dict) else {}

    tertiary = 1e9
    if str(tap.get("status", "")) == TAP_STATUS_AVAILABLE:
        tertiary = _safe_float(tap.get("free_rollout_endpoint_l2"), 1e9)

    quaternary = 1e9
    if str(tap3d.get("status", "")) == TAP_STATUS_AVAILABLE:
        quaternary = _safe_float(tap3d.get("free_rollout_endpoint_l2"), 1e9)

    return {
        "teacher_forced_coord_loss": _safe_float(evaluation.get("teacher_forced_coord_loss"), 1e9),
        "free_rollout_coord_mean_l2": _safe_float(evaluation.get("free_rollout_coord_mean_l2"), 1e9),
        "free_rollout_endpoint_l2": _safe_float(evaluation.get("free_rollout_endpoint_l2"), 1e9),
        "tapvid_endpoint_l2": float(tertiary),
        "tapvid3d_limited_endpoint_l2": float(quaternary),
    }


def _rank_key(metrics: Dict[str, float]) -> Tuple[float, float, float, float]:
    return (
        float(metrics["free_rollout_endpoint_l2"]),
        float(metrics["free_rollout_coord_mean_l2"]),
        float(metrics["tapvid_endpoint_l2"]),
        float(metrics["tapvid3d_limited_endpoint_l2"]),
    )


def _atomic_torch_save(payload: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _checkpoint_payload(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    global_step: int,
    epoch: int,
    best_metric_so_far: Dict[str, Any] | None,
    config_payload: Dict[str, Any],
    run_metadata: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "global_step": int(global_step),
        "epoch": int(epoch),
        "best_metric_so_far": best_metric_so_far if isinstance(best_metric_so_far, dict) else None,
        "config": config_payload,
        "run_metadata": run_metadata,
    }


def _resolve_resume_path(resume_from: str, auto_resume_latest: bool, latest_path: Path) -> str:
    direct = str(resume_from).strip()
    if direct:
        return str(Path(direct).expanduser())
    if bool(auto_resume_latest) and latest_path.exists():
        return str(latest_path)
    return ""


def _build_progress_payload(
    *,
    args: Any,
    generated_at: str,
    device: torch.device,
    runtime_meta: Dict[str, Any],
    run_metadata: Dict[str, Any],
    checkpoint_dir: Path,
    best_path: Path,
    latest_path: Path,
    step_checkpoints: List[str],
    eval_history: List[Dict[str, Any]],
    best_metric_so_far: Dict[str, Any] | None,
    global_step: int,
    epoch: int,
    effective_batch: int,
) -> Dict[str, Any]:
    return {
        "generated_at_utc": generated_at,
        "run_name": str(args.run_name),
        "objective": "Stage1-v2 220M mainline long-train with recoverable checkpoint infrastructure",
        "device": str(device),
        "runtime": runtime_meta,
        "run_metadata": run_metadata,
        "contract_path": str(args.contract_path),
        "training_budget": {
            "optimizer_steps_target": int(args.train_steps),
            "optimizer_steps_current": int(global_step),
            "effective_batch": int(effective_batch),
            "epochs": int(epoch),
            "eval_interval": int(args.eval_interval),
            "save_every_n_steps": int(args.save_every_n_steps),
            "eval_steps": int(args.eval_steps),
        },
        "selection_policy": {
            "primary": "free_rollout_endpoint_l2",
            "secondary": "free_rollout_coord_mean_l2",
            "tertiary": "tapvid_eval.free_rollout_endpoint_l2",
            "quaternary": "tapvid3d_limited_eval.free_rollout_endpoint_l2",
            "total_loss_usage": "reference_only",
        },
        "checkpoint_inventory": {
            "checkpoint_dir": str(checkpoint_dir),
            "best": str(best_path),
            "latest": str(latest_path),
            "step_checkpoints": step_checkpoints,
        },
        "best_metric_so_far": best_metric_so_far if isinstance(best_metric_so_far, dict) else None,
        "eval_history": eval_history,
    }


def _write_results_md(
    *,
    path: Path,
    final_payload: Dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    best = final_payload.get("best_metric_so_far", {}) if isinstance(final_payload.get("best_metric_so_far", {}), dict) else {}
    best_metrics = best.get("metrics", {}) if isinstance(best.get("metrics", {}), dict) else {}
    ckpt = final_payload.get("checkpoint_inventory", {}) if isinstance(final_payload.get("checkpoint_inventory", {}), dict) else {}

    lines = [
        "# Stage1-v2 220M Long-Train Results",
        "",
        f"- generated_at_utc: {final_payload.get('generated_at_utc', '')}",
        f"- run_name: {final_payload.get('run_name', '')}",
        f"- final_stage1_backbone_decision_source: freeze_220m_as_stage1_backbone (from mainline freeze)",
        f"- next_step_choice: {final_payload.get('next_step_choice', '')}",
        "",
        "## Training Budget",
        f"- optimizer_steps: {int((final_payload.get('training_budget', {}) if isinstance(final_payload.get('training_budget', {}), dict) else {}).get('optimizer_steps_target', 0))}",
        f"- effective_batch: {int((final_payload.get('training_budget', {}) if isinstance(final_payload.get('training_budget', {}), dict) else {}).get('effective_batch', 0))}",
        f"- epochs: {int((final_payload.get('training_budget', {}) if isinstance(final_payload.get('training_budget', {}), dict) else {}).get('epochs', 0))}",
        f"- eval_interval: {int((final_payload.get('training_budget', {}) if isinstance(final_payload.get('training_budget', {}), dict) else {}).get('eval_interval', 0))}",
        f"- save_every_n_steps: {int((final_payload.get('training_budget', {}) if isinstance(final_payload.get('training_budget', {}), dict) else {}).get('save_every_n_steps', 0))}",
        "",
        "## Best Ranked Metrics",
        "| metric | value |",
        "|---|---:|",
        f"| free_rollout_endpoint_l2 (primary) | {float(best_metrics.get('free_rollout_endpoint_l2', 0.0)):.6f} |",
        f"| free_rollout_coord_mean_l2 (secondary) | {float(best_metrics.get('free_rollout_coord_mean_l2', 0.0)):.6f} |",
        f"| tapvid_endpoint_l2 (tertiary) | {float(best_metrics.get('tapvid_endpoint_l2', 0.0)):.6f} |",
        f"| tapvid3d_limited_endpoint_l2 (quaternary) | {float(best_metrics.get('tapvid3d_limited_endpoint_l2', 0.0)):.6f} |",
        f"| teacher_forced_coord_loss | {float(best_metrics.get('teacher_forced_coord_loss', 0.0)):.6f} |",
        "",
        "## Checkpoint Inventory",
        f"- checkpoint_dir: {ckpt.get('checkpoint_dir', '')}",
        f"- best: {ckpt.get('best', '')}",
        f"- latest: {ckpt.get('latest', '')}",
    ]

    for p in ckpt.get("step_checkpoints", []) if isinstance(ckpt.get("step_checkpoints", []), list) else []:
        lines.append(f"- step: {p}")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    set_seed(int(args.seed))

    output_dir = Path(args.output_dir)
    summary_json = Path(args.summary_json)
    results_md = Path(args.results_md)
    perf_step_timing_json = Path(args.perf_step_timing_json)

    progress_json = Path(args.progress_json) if str(args.progress_json).strip() else summary_json.with_name(summary_json.stem + "_progress.json")
    final_json = Path(args.final_json) if str(args.final_json).strip() else summary_json.with_name(summary_json.stem + "_final.json")

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    results_md.parent.mkdir(parents=True, exist_ok=True)
    perf_step_timing_json.parent.mkdir(parents=True, exist_ok=True)
    progress_json.parent.mkdir(parents=True, exist_ok=True)
    final_json.parent.mkdir(parents=True, exist_ok=True)

    target_steps = int(args.train_steps) if int(args.train_steps) > 0 else int(args.epochs) * int(args.steps_per_epoch)
    if target_steps <= 0:
        raise ValueError("target optimizer steps must be > 0; set --train-steps or (epochs*steps-per-epoch)")

    eval_interval = int(args.eval_interval)
    save_every_n_steps = int(args.save_every_n_steps)
    if save_every_n_steps <= 0:
        raise ValueError("save_every_n_steps must be > 0")

    runtime_meta: Dict[str, Any] = {
        "source": "manual_args",
        "num_workers": int(args.num_workers),
        "pin_memory": bool(args.pin_memory),
        "persistent_workers": bool(args.persistent_workers),
        "prefetch_factor": int(args.prefetch_factor),
        "single_gpu_only": True,
        "selected_gpu_id_runtime_json": -1,
        "required_mem_gb": None,
        "safety_margin_gb": None,
    }

    num_workers = int(args.num_workers)
    pin_memory = bool(args.pin_memory)
    persistent_workers = bool(args.persistent_workers)
    prefetch_factor = int(args.prefetch_factor)

    if bool(args.use_recommended_runtime):
        runtime = _load_runtime_config(args.recommended_runtime_json)
        num_workers = int(runtime.num_workers)
        pin_memory = bool(runtime.pin_memory)
        persistent_workers = bool(runtime.persistent_workers)
        prefetch_factor = int(runtime.prefetch_factor)
        runtime_meta = {
            "source": "recommended_runtime_json",
            "path": str(args.recommended_runtime_json),
            "num_workers": num_workers,
            "pin_memory": pin_memory,
            "persistent_workers": persistent_workers,
            "prefetch_factor": prefetch_factor,
            "single_gpu_only": bool(runtime.single_gpu_only),
            "selected_gpu_id_runtime_json": int(runtime.selected_gpu_id),
            "required_mem_gb": float(runtime.required_mem_gb),
            "safety_margin_gb": float(runtime.safety_margin_gb),
        }

    train_dataset = Stage1V2UnifiedDataset(
        dataset_names=[str(x) for x in args.dataset_names],
        split=str(args.train_split),
        contract_path=args.contract_path,
        obs_len=int(args.obs_len),
        fut_len=int(args.fut_len),
        max_tokens=int(args.max_tokens),
        max_samples_per_dataset=int(args.max_samples_per_dataset),
    )

    val_max_samples = int(args.max_samples_per_dataset_val)
    if val_max_samples <= 0:
        val_max_samples = int(args.max_samples_per_dataset)

    val_dataset = Stage1V2UnifiedDataset(
        dataset_names=[str(x) for x in args.dataset_names],
        split=str(args.val_split),
        contract_path=args.contract_path,
        obs_len=int(args.obs_len),
        fut_len=int(args.fut_len),
        max_tokens=int(args.max_tokens),
        max_samples_per_dataset=int(val_max_samples),
    )

    train_loader_kwargs: Dict[str, Any] = {
        "dataset": train_dataset,
        "batch_size": int(args.batch_size),
        "shuffle": True,
        "num_workers": int(num_workers),
        "pin_memory": bool(pin_memory),
        "collate_fn": stage1_v2_collate_fn,
    }
    if int(num_workers) > 0:
        train_loader_kwargs["persistent_workers"] = bool(persistent_workers)
        train_loader_kwargs["prefetch_factor"] = int(prefetch_factor)

    train_loader = DataLoader(**train_loader_kwargs)

    val_loader = DataLoader(
        val_dataset,
        batch_size=max(1, int(args.batch_size)),
        shuffle=False,
        num_workers=0,
        pin_memory=bool(pin_memory),
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
    scheduler = None

    checkpoint_dir = Path(str(args.checkpoint_dir)) if str(args.checkpoint_dir).strip() else (output_dir / "checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_checkpoint_path = checkpoint_dir / "best.pt"
    latest_checkpoint_path = checkpoint_dir / "latest.pt"

    run_metadata = {
        "run_name": str(args.run_name),
        "ablation_tag": str(args.ablation_tag),
        "run_metadata_note": str(args.run_metadata_note),
        "cuda_visible_devices": str(os.environ.get("CUDA_VISIBLE_DEVICES", "")),
        "gpu_selection_metadata": str(os.environ.get("TRACEWM_STAGE1_V2_GPU_SELECTION_METADATA", "")),
        "started_at_utc": now_iso(),
    }

    resolved_resume_path = _resolve_resume_path(
        resume_from=str(args.resume_from),
        auto_resume_latest=bool(args.auto_resume_latest),
        latest_path=latest_checkpoint_path,
    )

    global_step = 0
    epoch = 0
    best_metric_so_far: Dict[str, Any] | None = None

    if resolved_resume_path:
        resume_path = Path(resolved_resume_path)
        if not resume_path.exists():
            raise FileNotFoundError(f"resume checkpoint not found: {resume_path}")

        payload = torch.load(resume_path, map_location=device)
        if not isinstance(payload, dict):
            raise RuntimeError(f"unsupported checkpoint payload type for resume: {type(payload)}")

        if "model_state_dict" not in payload:
            raise RuntimeError("resume checkpoint missing model_state_dict")
        if "optimizer_state_dict" not in payload:
            raise RuntimeError("resume checkpoint missing optimizer_state_dict; refusing model-only resume")

        model.load_state_dict(payload["model_state_dict"])
        optimizer.load_state_dict(payload["optimizer_state_dict"])

        sched_state = payload.get("scheduler_state_dict")
        if scheduler is not None and sched_state is not None:
            scheduler.load_state_dict(sched_state)

        global_step = int(payload.get("global_step", 0) or 0)
        epoch = int(payload.get("epoch", 0) or 0)

        maybe_best = payload.get("best_metric_so_far")
        if isinstance(maybe_best, dict):
            best_metric_so_far = maybe_best

        run_metadata["resumed_from"] = str(resume_path)
    else:
        run_metadata["resumed_from"] = ""

    loss_sums = {
        "total_loss": 0.0,
        "coord_loss": 0.0,
        "visibility_loss": 0.0,
        "residual_loss": 0.0,
        "velocity_loss": 0.0,
        "endpoint_loss": 0.0,
    }
    steps_since_start = 0

    eval_history: List[Dict[str, Any]] = []
    step_checkpoints: List[str] = []
    all_step_timing: List[Dict[str, float]] = []
    latest_evaluation: Dict[str, Any] | None = None

    data_iter = iter(train_loader)

    def _save_step_and_latest(step_now: int, epoch_now: int) -> None:
        step_path = checkpoint_dir / f"step_{int(step_now):07d}.pt"
        payload = _checkpoint_payload(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            global_step=int(step_now),
            epoch=int(epoch_now),
            best_metric_so_far=best_metric_so_far,
            config_payload=vars(args),
            run_metadata=run_metadata,
        )
        _atomic_torch_save(payload, step_path)
        _atomic_torch_save(payload, latest_checkpoint_path)
        sp = str(step_path)
        if sp not in step_checkpoints:
            step_checkpoints.append(sp)

    while global_step < target_steps:
        wait_start = time.perf_counter()
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            epoch += 1
            batch = next(data_iter)
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
            pred=_future_pred(pred, obs_len=int(args.obs_len)),
            target_state=full_state[:, int(args.obs_len):],
            valid_mask=full_valid[:, int(args.obs_len):],
            token_mask=data["token_mask"],
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        forward_time = float(time.perf_counter() - forward_start)

        backward_start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        losses["total_loss"].backward()
        if float(args.clip_grad_norm) > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(args.clip_grad_norm))
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        backward_time = float(time.perf_counter() - backward_start)

        optim_start = time.perf_counter()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        optimizer_time = float(time.perf_counter() - optim_start)

        step_time = float(time.perf_counter() - step_start)
        samples_per_sec = float(data["obs_state"].shape[0] / max(step_time, 1e-8))

        all_step_timing.append(
            {
                "global_step": float(global_step + 1),
                "batch_wait_time": batch_wait_time,
                "h2d_time": h2d_time,
                "forward_time": forward_time,
                "backward_time": backward_time,
                "optimizer_time": optimizer_time,
                "step_time": step_time,
                "samples_per_sec": samples_per_sec,
            }
        )

        for key in loss_sums:
            loss_sums[key] += float(losses[key].detach().cpu().item())

        global_step += 1
        steps_since_start += 1

        should_eval = bool(eval_interval > 0 and global_step % eval_interval == 0)
        if global_step == target_steps and eval_interval <= 0:
            should_eval = False

        if should_eval:
            latest_evaluation = _evaluate_model(
                model=model,
                criterion=criterion,
                loader=val_loader,
                device=device,
                obs_len=int(args.obs_len),
                fut_len=int(args.fut_len),
                pin_memory=bool(pin_memory),
                eval_steps=int(args.eval_steps),
                args=args,
            )
            ranked = _ranked_metrics_from_evaluation(latest_evaluation)
            event = {
                "global_step": int(global_step),
                "epoch": int(epoch),
                "metrics": ranked,
                "rank_key": list(_rank_key(ranked)),
            }
            eval_history.append(event)

            if best_metric_so_far is None or tuple(event["rank_key"]) < tuple(best_metric_so_far.get("rank_key", [1e9, 1e9, 1e9, 1e9])):
                best_metric_so_far = {
                    "global_step": int(global_step),
                    "epoch": int(epoch),
                    "metrics": ranked,
                    "rank_key": list(_rank_key(ranked)),
                }
                payload_best = _checkpoint_payload(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    global_step=int(global_step),
                    epoch=int(epoch),
                    best_metric_so_far=best_metric_so_far,
                    config_payload=vars(args),
                    run_metadata=run_metadata,
                )
                _atomic_torch_save(payload_best, best_checkpoint_path)

        should_save = bool(global_step % save_every_n_steps == 0 or global_step == target_steps)
        if should_save:
            _save_step_and_latest(step_now=int(global_step), epoch_now=int(epoch))

        if should_eval or should_save:
            progress_payload = _build_progress_payload(
                args=args,
                generated_at=now_iso(),
                device=device,
                runtime_meta=runtime_meta,
                run_metadata=run_metadata,
                checkpoint_dir=checkpoint_dir,
                best_path=best_checkpoint_path,
                latest_path=latest_checkpoint_path,
                step_checkpoints=sorted(step_checkpoints),
                eval_history=eval_history,
                best_metric_so_far=best_metric_so_far,
                global_step=int(global_step),
                epoch=int(epoch),
                effective_batch=int(args.batch_size),
            )
            _write_json(progress_json, progress_payload)

    if eval_interval > 0 and (not eval_history or int(eval_history[-1].get("global_step", -1)) != int(global_step)):
        latest_evaluation = _evaluate_model(
            model=model,
            criterion=criterion,
            loader=val_loader,
            device=device,
            obs_len=int(args.obs_len),
            fut_len=int(args.fut_len),
            pin_memory=bool(pin_memory),
            eval_steps=int(args.eval_steps),
            args=args,
        )
        ranked = _ranked_metrics_from_evaluation(latest_evaluation)
        event = {
            "global_step": int(global_step),
            "epoch": int(epoch),
            "metrics": ranked,
            "rank_key": list(_rank_key(ranked)),
        }
        eval_history.append(event)

        if best_metric_so_far is None or tuple(event["rank_key"]) < tuple(best_metric_so_far.get("rank_key", [1e9, 1e9, 1e9, 1e9])):
            best_metric_so_far = {
                "global_step": int(global_step),
                "epoch": int(epoch),
                "metrics": ranked,
                "rank_key": list(_rank_key(ranked)),
            }
            payload_best = _checkpoint_payload(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                global_step=int(global_step),
                epoch=int(epoch),
                best_metric_so_far=best_metric_so_far,
                config_payload=vars(args),
                run_metadata=run_metadata,
            )
            _atomic_torch_save(payload_best, best_checkpoint_path)

    if best_metric_so_far is None and eval_history:
        best_metric_so_far = dict(eval_history[-1])
        payload_best = _checkpoint_payload(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            global_step=int(global_step),
            epoch=int(epoch),
            best_metric_so_far=best_metric_so_far,
            config_payload=vars(args),
            run_metadata=run_metadata,
        )
        _atomic_torch_save(payload_best, best_checkpoint_path)

    if global_step > 0 and not latest_checkpoint_path.exists():
        _save_step_and_latest(step_now=int(global_step), epoch_now=int(epoch))

    step_checkpoints = sorted(str(p) for p in checkpoint_dir.glob("step_*.pt"))

    denom = float(max(steps_since_start, 1))
    final_train_metrics = {k: float(v / denom) for k, v in loss_sums.items()}

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

    next_step_choice = "continue_to_10000_from_latest"
    if int(target_steps) >= 10000 and isinstance(best_metric_so_far, dict):
        primary_now = float((best_metric_so_far.get("metrics", {}) if isinstance(best_metric_so_far.get("metrics", {}), dict) else {}).get("free_rollout_endpoint_l2", 1e9))
        if primary_now <= 0.24:
            next_step_choice = "freeze_stage1_and_prepare_stage2"
        else:
            next_step_choice = "do_one_targeted_stage1_fix"

    final_payload = {
        "generated_at_utc": now_iso(),
        "run_name": str(args.run_name),
        "objective": "formal Stage1-v2 220M long-train backbone run",
        "contract_path": str(args.contract_path),
        "training_budget": {
            "optimizer_steps_target": int(target_steps),
            "optimizer_steps_completed": int(global_step),
            "effective_batch": int(args.batch_size),
            "epochs": int(epoch),
            "eval_interval": int(eval_interval),
            "save_every_n_steps": int(save_every_n_steps),
            "eval_steps": int(args.eval_steps),
        },
        "selection_policy": {
            "primary": "free_rollout_endpoint_l2",
            "secondary": "free_rollout_coord_mean_l2",
            "tertiary": "tapvid_eval.free_rollout_endpoint_l2",
            "quaternary": "tapvid3d_limited_eval.free_rollout_endpoint_l2",
            "total_loss_usage": "reference_only",
        },
        "runtime": runtime_meta,
        "run_metadata": run_metadata,
        "resume": {
            "resume_from": str(resolved_resume_path),
            "auto_resume_latest": bool(args.auto_resume_latest),
        },
        "model": {
            "preset": str(args.model_preset),
            "config": cfg.__dict__,
            "parameter_count": total_params,
            "estimated_parameter_count": estimated_params,
            "target_220m_range_pass": bool(200_000_000 <= estimated_params <= 240_000_000),
        },
        "best_metric_so_far": best_metric_so_far if isinstance(best_metric_so_far, dict) else None,
        "latest_eval_metrics": _ranked_metrics_from_evaluation(latest_evaluation) if isinstance(latest_evaluation, dict) else None,
        "checkpoint_inventory": {
            "checkpoint_dir": str(checkpoint_dir),
            "best": str(best_checkpoint_path),
            "latest": str(latest_checkpoint_path),
            "step_checkpoints": step_checkpoints,
        },
        "allowed_next_step_choice": [
            "continue_to_10000_from_latest",
            "freeze_stage1_and_prepare_stage2",
            "do_one_targeted_stage1_fix",
        ],
        "next_step_choice": str(next_step_choice),
    }

    summary = {
        "generated_at_utc": now_iso(),
        "ablation_tag": str(args.ablation_tag),
        "run_name": str(args.run_name),
        "device": str(device),
        "args": vars(args),
        "model": final_payload["model"],
        "dataset": {
            "train_size": int(len(train_dataset)),
            "val_size": int(len(val_dataset)),
            "train_split": str(args.train_split),
            "val_split": str(args.val_split),
            "dataset_names": [str(x) for x in args.dataset_names],
        },
        "loss_config": loss_cfg.__dict__,
        "final_metrics": final_train_metrics,
        "timing_stats": timing_stats,
        "best_metric_so_far": final_payload["best_metric_so_far"],
        "checkpoint_inventory": final_payload["checkpoint_inventory"],
        "p1_shape_contract": {
            "state_shape": "[B,T,K,D]",
            "state_dim": STATE_DIM,
        },
    }

    perf_payload = {
        "generated_at_utc": now_iso(),
        "ablation_tag": str(args.ablation_tag),
        "run_name": str(args.run_name),
        "model_preset": str(args.model_preset),
        "device": str(device),
        "num_steps": int(len(all_step_timing)),
        "timing_stats": timing_stats,
        "step_records": all_step_timing,
    }

    progress_payload = _build_progress_payload(
        args=args,
        generated_at=now_iso(),
        device=device,
        runtime_meta=runtime_meta,
        run_metadata=run_metadata,
        checkpoint_dir=checkpoint_dir,
        best_path=best_checkpoint_path,
        latest_path=latest_checkpoint_path,
        step_checkpoints=step_checkpoints,
        eval_history=eval_history,
        best_metric_so_far=best_metric_so_far,
        global_step=int(global_step),
        epoch=int(epoch),
        effective_batch=int(args.batch_size),
    )

    _write_json(summary_json, summary)
    _write_json(progress_json, progress_payload)
    _write_json(final_json, final_payload)
    _write_json(perf_step_timing_json, perf_payload)
    _write_results_md(path=results_md, final_payload=final_payload)

    print(f"[stage1-v2-train] run_name={args.run_name}")
    print(f"[stage1-v2-train] summary={summary_json}")
    print(f"[stage1-v2-train] progress_json={progress_json}")
    print(f"[stage1-v2-train] final_json={final_json}")
    print(f"[stage1-v2-train] results_md={results_md}")
    print(f"[stage1-v2-train] perf_step_timing={perf_step_timing_json}")
    print(f"[stage1-v2-train] checkpoint_dir={checkpoint_dir}")
    print(f"[stage1-v2-train] best_checkpoint={best_checkpoint_path}")
    print(f"[stage1-v2-train] latest_checkpoint={latest_checkpoint_path}")


if __name__ == "__main__":
    main()
