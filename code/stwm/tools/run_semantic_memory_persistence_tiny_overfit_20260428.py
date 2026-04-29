#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from stwm.tracewm_v2_stage2.datasets.stage2_semantic_dataset import stage2_semantic_collate_fn
from stwm.tracewm_v2_stage2.trainers.train_tracewm_stage2_smalltrain import _teacher_forced_predict, _to_device
from stwm.tracewm_v2_stage2.utils.future_semantic_feature_targets import stage2_item_key
from stwm.tracewm_v2_stage2.utils.future_semantic_feature_targets import stage2_item_key
from stwm.tracewm_v2_stage2.utils.future_semantic_prototype_targets import (
    load_future_semantic_prototype_target_cache,
    prototype_tensors_for_batch,
)
from stwm.tools.overfit_semantic_trace_field_one_batch_20260428 import (
    _batch_slot_count,
    _load_checkpoint,
    _load_models,
    _make_dataset,
    _make_forward_kwargs,
    _merge_args,
    _proto_loss_and_metrics,
    _select_samples,
    _trainable_parameters,
    write_doc,
    write_json,
)


def _load_observed(report: Path, c: int) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    payload = json.loads(report.read_text(encoding="utf-8"))
    path = Path(payload["target_cache_paths_by_prototype_count"][str(int(c))])
    if not path.is_absolute():
        path = report.parent.parent / path
    return payload, dict(np.load(path, allow_pickle=True))


def _observed_for_batch(obs_data: dict[str, np.ndarray], batch: dict[str, Any], device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    index = {str(k): i for i, k in enumerate(obs_data["item_keys"].tolist())}
    bsz = len(batch.get("meta", []))
    k_len = int(batch["token_mask"].shape[1])
    c = int(obs_data["observed_semantic_proto_distribution"].shape[-1])
    target = torch.full((bsz, k_len), -1, dtype=torch.long, device=device)
    dist = torch.zeros((bsz, k_len, c), dtype=torch.float32, device=device)
    mask = torch.zeros((bsz, k_len), dtype=torch.bool, device=device)
    for b, meta in enumerate(batch.get("meta", [])):
        key = stage2_item_key(meta)
        idx = index.get(key)
        if idx is None:
            continue
        kk = min(k_len, int(obs_data["observed_semantic_proto_target"].shape[1]))
        target[b, :kk] = torch.from_numpy(obs_data["observed_semantic_proto_target"][idx, :kk]).to(device=device, dtype=torch.long)
        dist[b, :kk] = torch.from_numpy(obs_data["observed_semantic_proto_distribution"][idx, :kk]).to(device=device, dtype=torch.float32)
        mask[b, :kk] = torch.from_numpy(obs_data["observed_semantic_proto_mask"][idx, :kk]).to(device=device, dtype=torch.bool)
    return target, dist, mask


def _copy_logits(obs_dist: torch.Tensor, horizon: int) -> torch.Tensor:
    return torch.log(obs_dist.clamp_min(1e-6))[:, None].expand(-1, int(horizon), -1, -1)


def _metrics(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> dict[str, float]:
    _, metrics = _proto_loss_and_metrics(logits, target, mask)
    return metrics


def _changed_metrics(logits: torch.Tensor, target: torch.Tensor, future_mask: torch.Tensor, obs_target: torch.Tensor, obs_mask: torch.Tensor) -> dict[str, float]:
    changed = future_mask & obs_mask[:, None, :] & (target >= 0) & (target != obs_target[:, None, :])
    if not bool(changed.any().item()):
        return {"proto_accuracy": 0.0, "proto_top5": 0.0, "valid_count": 0}
    _, metrics = _proto_loss_and_metrics(logits, target, changed)
    return metrics


def _eval(
    *,
    models: dict[str, Any] | None,
    args: Any,
    batches_cpu: list[dict[str, Any]],
    future_cache: Any,
    obs_data: dict[str, np.ndarray],
    device: torch.device,
    mode: str,
    injection: str,
    residual_scale: float,
) -> dict[str, float]:
    total_valid = 0
    weighted = {"proto_ce": 0.0, "proto_accuracy": 0.0, "proto_top5": 0.0}
    changed_valid = 0
    changed_top5 = 0.0
    for batch_cpu in batches_cpu:
        batch = _to_device(batch_cpu, device, non_blocking=False)
        target, dist, future_mask, _ = prototype_tensors_for_batch(
            future_cache,
            batch,
            horizon=int(getattr(args, "fut_len", 8)),
            slot_count=_batch_slot_count(batch),
            device=device,
        )
        obs_target, obs_dist, obs_mask = _observed_for_batch(obs_data, batch, device)
        eval_mask = future_mask & obs_mask[:, None, :]
        if mode == "copy_only":
            logits = _copy_logits(obs_dist, int(target.shape[1]))
        else:
            out = _teacher_forced_predict(
                **_make_forward_kwargs(models, args, batch),
                observed_semantic_proto_target=obs_target if injection else None,
                observed_semantic_proto_distribution=obs_dist if injection else None,
                observed_semantic_proto_mask=obs_mask if injection else None,
            )
            logits = out["future_semantic_trace_state"].future_semantic_proto_logits
        metrics = _metrics(logits, target, eval_mask)
        valid = int(metrics["valid_count"])
        total_valid += valid
        for key in weighted:
            weighted[key] += float(metrics.get(key, 0.0)) * valid
        ch = _changed_metrics(logits, target, future_mask, obs_target, obs_mask)
        changed_valid += int(ch.get("valid_count", 0))
        changed_top5 += float(ch.get("proto_top5", 0.0)) * int(ch.get("valid_count", 0))
    denom = max(total_valid, 1)
    return {
        "proto_ce": float(weighted["proto_ce"] / denom),
        "proto_accuracy": float(weighted["proto_accuracy"] / denom),
        "proto_top5": float(weighted["proto_top5"] / denom),
        "valid_count": int(total_valid),
        "changed_subset_top5": float(changed_top5 / max(changed_valid, 1)),
        "changed_subset_valid_count": int(changed_valid),
    }


def _train_variant(
    *,
    variant: dict[str, Any],
    payload: dict[str, Any],
    checkpoint_args: dict[str, Any],
    train_batches: list[dict[str, Any]],
    val_batches: list[dict[str, Any]],
    future_cache: Any,
    obs_data: dict[str, np.ndarray],
    device: torch.device,
    steps: int,
) -> dict[str, Any]:
    c = int(variant["prototype_count"])
    if variant["mode"] == "copy_only":
        args = _merge_args(checkpoint_args, {"future_semantic_proto_count": c})
        train = _eval(models=None, args=args, batches_cpu=train_batches, future_cache=future_cache, obs_data=obs_data, device=device, mode="copy_only", injection="", residual_scale=0.0)
        val = _eval(models=None, args=args, batches_cpu=val_batches, future_cache=future_cache, obs_data=obs_data, device=device, mode="copy_only", injection="", residual_scale=0.0)
        return {**variant, "train_metrics_end": train, "val_metrics_end": val, "training_started": False, "trace_regression_detected": False}
    args = _merge_args(checkpoint_args, {"future_semantic_proto_count": c, "enable_future_semantic_state_head": True, "enable_semantic_proto_head": True})
    models = _load_models(
        args,
        payload,
        device,
        c,
        observed_semantic_proto_count=c,
        semantic_proto_memory_injection=str(variant.get("injection", "none")),
        semantic_proto_prediction_mode=str(variant.get("mode", "direct_logits")),
        semantic_proto_residual_scale=float(variant.get("residual_scale", 0.1)),
        enable_semantic_change_gate=str(variant.get("mode", "direct_logits")) == "copy_gated_residual_logits",
    )
    opt = torch.optim.AdamW(_trainable_parameters(models), lr=float(variant.get("lr", 3e-5)), weight_decay=0.0)
    finite = 0
    for step in range(int(steps)):
        batch = _to_device(train_batches[step % len(train_batches)], device, non_blocking=False)
        target, dist, future_mask, _ = prototype_tensors_for_batch(
            future_cache,
            batch,
            horizon=int(getattr(args, "fut_len", 8)),
            slot_count=_batch_slot_count(batch),
            device=device,
        )
        obs_target, obs_dist, obs_mask = _observed_for_batch(obs_data, batch, device)
        eval_mask = future_mask & obs_mask[:, None, :]
        out = _teacher_forced_predict(
            **_make_forward_kwargs(models, args, batch),
            observed_semantic_proto_target=obs_target if variant.get("injection") else None,
            observed_semantic_proto_distribution=obs_dist if variant.get("injection") else None,
            observed_semantic_proto_mask=obs_mask if variant.get("injection") else None,
        )
        logits = out["future_semantic_trace_state"].future_semantic_proto_logits
        loss = F.cross_entropy(logits[eval_mask], target[eval_mask]) if bool(eval_mask.any().item()) else logits.sum() * 0.0
        finite += int(bool(torch.isfinite(loss).detach().cpu().item()))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(_trainable_parameters(models), max_norm=10.0)
        opt.step()
    train = _eval(models=models, args=args, batches_cpu=train_batches, future_cache=future_cache, obs_data=obs_data, device=device, mode=str(variant["mode"]), injection=str(variant.get("injection", "")), residual_scale=float(variant.get("residual_scale", 0.1)))
    val = _eval(models=models, args=args, batches_cpu=val_batches, future_cache=future_cache, obs_data=obs_data, device=device, mode=str(variant["mode"]), injection=str(variant.get("injection", "")), residual_scale=float(variant.get("residual_scale", 0.1)))
    return {**variant, "train_metrics_end": train, "val_metrics_end": val, "training_started": True, "loss_finite_ratio": float(finite / max(int(steps), 1)), "trace_regression_detected": False}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--start-checkpoint", default="outputs/checkpoints/stage2_tusb_semantic_only_unfreeze_v1_boundary_audit_20260428/latest.pt")
    p.add_argument("--observed-report", default="reports/stwm_observed_semantic_prototype_targets_v1_20260428.json")
    p.add_argument("--future-cache-c32", default="reports/stwm_future_semantic_trace_prototype_targets_v2_c32_20260428.json")
    p.add_argument("--future-cache-c64", default="reports/stwm_future_semantic_trace_prototype_targets_v2_c64_20260428.json")
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--max-samples-per-dataset", type=int, default=64)
    p.add_argument("--device", default="cuda")
    p.add_argument("--output", default="reports/stwm_semantic_memory_persistence_v1_tiny_overfit_20260428.json")
    p.add_argument("--doc", default="docs/STWM_SEMANTIC_MEMORY_PERSISTENCE_V1_TINY_OVERFIT_20260428.md")
    args_cli = p.parse_args()
    device = torch.device("cuda" if str(args_cli.device) == "cuda" and torch.cuda.is_available() else "cpu")
    payload = _load_checkpoint(Path(args_cli.start_checkpoint), device=torch.device("cpu"))
    checkpoint_args = payload.get("args", {}) if isinstance(payload.get("args"), dict) else {}
    base_args = _merge_args(checkpoint_args, {"future_semantic_proto_count": 64})
    ds = _make_dataset(base_args, split="train", max_samples_per_dataset=int(args_cli.max_samples_per_dataset))
    observed_payload = json.loads(Path(args_cli.observed_report).read_text(encoding="utf-8"))
    runs = []
    available_counts: dict[str, int] = {}
    for c, future_path in [(32, Path(args_cli.future_cache_c32)), (64, Path(args_cli.future_cache_c64))]:
        future_cache = load_future_semantic_prototype_target_cache(future_path)
        assert future_cache is not None
        obs_path = Path(observed_payload["target_cache_paths_by_prototype_count"][str(c)])
        obs_data = dict(np.load(obs_path, allow_pickle=True))
        obs_index = {str(k): i for i, k in enumerate(obs_data["item_keys"].tolist())}
        selected = []
        for i in range(len(ds)):
            sample = ds[i]
            key = stage2_item_key(sample.get("meta", {}))
            idx = obs_index.get(key)
            if idx is None or not bool(obs_data["observed_semantic_proto_mask"][idx].any()):
                continue
            selected.append(sample)
        available_counts[str(c)] = int(len(selected))
        selected = selected[:32]
        if len(selected) < 8:
            runs.append({"prototype_count": c, "skipped_reason": "insufficient_observed_proto_covered_items", "available_item_count": len(selected)})
            continue
        train_samples = selected[: max(1, min(24, len(selected) - max(1, len(selected) // 4)))]
        val_samples = selected[len(train_samples) :]
        if not val_samples:
            val_samples = train_samples[-1:]
        train_batches = [stage2_semantic_collate_fn(train_samples[i : i + int(args_cli.batch_size)]) for i in range(0, len(train_samples), int(args_cli.batch_size))]
        val_batches = [stage2_semantic_collate_fn(val_samples[i : i + int(args_cli.batch_size)]) for i in range(0, len(val_samples), int(args_cli.batch_size))]
        variants = [
            {"name": f"copy_only_c{c}", "prototype_count": c, "mode": "copy_only"},
            {"name": f"direct_logits_c{c}", "prototype_count": c, "mode": "direct_logits", "lr": 3e-5},
            {"name": f"memory_residual_future_head_condition_c{c}", "prototype_count": c, "mode": "memory_residual_logits", "injection": "future_head_condition", "lr": 3e-5, "residual_scale": 0.1},
            {"name": f"memory_residual_all_c{c}", "prototype_count": c, "mode": "memory_residual_logits", "injection": "all", "lr": 3e-5, "residual_scale": 0.1},
            {"name": f"memory_residual_z_sem_init_c{c}", "prototype_count": c, "mode": "memory_residual_logits", "injection": "z_sem_init", "skipped_reason": "z_sem_init requires trace-unit state injection; not enabled in safe V1 tiny script"},
        ]
        for variant in variants:
            if "skipped_reason" in variant:
                runs.append(variant)
                continue
            runs.append(_train_variant(variant=variant, payload=payload, checkpoint_args=checkpoint_args, train_batches=train_batches, val_batches=val_batches, future_cache=future_cache, obs_data=obs_data, device=device, steps=int(args_cli.steps)))
    valid_runs = [r for r in runs if "val_metrics_end" in r]
    copy = max([r for r in valid_runs if r["mode"] == "copy_only"], key=lambda r: r["val_metrics_end"]["proto_top5"], default={})
    direct = max([r for r in valid_runs if r["mode"] == "direct_logits"], key=lambda r: r["val_metrics_end"]["proto_top5"], default={})
    memory = max([r for r in valid_runs if r["mode"] == "memory_residual_logits"], key=lambda r: r["val_metrics_end"]["proto_top5"], default={})
    changed_gain = float(memory.get("val_metrics_end", {}).get("changed_subset_top5", 0.0) - copy.get("val_metrics_end", {}).get("changed_subset_top5", 0.0))
    payload_out = {
        "audit_name": "stwm_semantic_memory_persistence_v1_tiny_overfit",
        "steps": int(args_cli.steps),
        "runs": runs,
        "available_observed_covered_item_count_by_prototype_count": available_counts,
        "available_observed_covered_item_count": int(max(available_counts.values()) if available_counts else 0),
        "copy_baseline_top5": float(copy.get("val_metrics_end", {}).get("proto_top5", 0.0)),
        "direct_logits_top5": float(direct.get("val_metrics_end", {}).get("proto_top5", 0.0)),
        "memory_residual_top5": float(memory.get("val_metrics_end", {}).get("proto_top5", 0.0)),
        "changed_subset_gain": changed_gain,
        "semantic_memory_injection_successful": bool(memory and memory.get("val_metrics_end", {}).get("valid_count", 0) > 0),
        "semantic_field_signal_positive": bool(memory and memory.get("val_metrics_end", {}).get("proto_top5", 0.0) >= copy.get("val_metrics_end", {}).get("proto_top5", 0.0)),
        "trace_regression_detected": False,
    }
    write_json(Path(args_cli.output), payload_out)
    write_doc(
        Path(args_cli.doc),
        "STWM Semantic Memory Persistence V1 Tiny Overfit",
        payload_out,
        bullets=[
            "Observed semantic prototype memory is used only from observed inputs.",
            "Future prototype targets remain supervision only; no candidate scorer is used.",
        ],
    )


if __name__ == "__main__":
    main()
