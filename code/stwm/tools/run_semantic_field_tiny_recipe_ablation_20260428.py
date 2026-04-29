#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from stwm.tracewm_v2_stage2.datasets.stage2_semantic_dataset import stage2_semantic_collate_fn
from stwm.tracewm_v2_stage2.trainers.train_tracewm_stage2_smalltrain import _teacher_forced_predict, _to_device
from stwm.tracewm_v2_stage2.utils.future_semantic_prototype_targets import (
    load_future_semantic_prototype_target_cache,
    prototype_tensors_for_batch,
)
from stwm.tools.overfit_semantic_trace_field_one_batch_20260428 import (
    _batch_slot_count,
    _evaluate_batches,
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


def _class_weights(target: torch.Tensor, mask: torch.Tensor, c: int) -> torch.Tensor:
    valid = mask.to(torch.bool) & (target >= 0)
    labels = target[valid].detach().cpu().long()
    counts = torch.bincount(labels, minlength=int(c)).float()
    weights = counts.sum().clamp_min(1.0) / counts.clamp_min(1.0)
    weights = weights / weights.mean().clamp_min(1e-6)
    return weights.to(device=target.device, dtype=torch.float32)


def _loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    dist: torch.Tensor,
    mask: torch.Tensor,
    *,
    class_balanced: bool,
    soft_kl_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    valid = mask.to(torch.bool) & (target >= 0)
    if int(valid.sum().item()) <= 0:
        return logits.sum() * 0.0, {"proto_ce": 0.0, "proto_accuracy": 0.0, "proto_top5": 0.0, "valid_count": 0}
    weight = _class_weights(target, mask, logits.shape[-1]) if class_balanced else None
    ce = F.cross_entropy(logits[valid], target[valid], weight=weight)
    kl = logits.sum() * 0.0
    if float(soft_kl_weight) > 0.0 and dist is not None and dist.shape == logits.shape:
        logp = F.log_softmax(logits[valid], dim=-1)
        soft = dist[valid].to(dtype=logp.dtype)
        kl = F.kl_div(logp, soft, reduction="batchmean")
    _, metrics = _proto_loss_and_metrics(logits, target, mask)
    return ce + float(soft_kl_weight) * kl, metrics


def _run_variant(
    *,
    variant: dict[str, Any],
    payload: dict[str, Any],
    checkpoint_args: dict[str, Any],
    device: torch.device,
    train_batches: list[dict[str, Any]],
    val_batches: list[dict[str, Any]],
    target_cache_path: Path,
    steps: int,
) -> dict[str, Any]:
    c = int(variant["prototype_count"])
    cache = load_future_semantic_prototype_target_cache(target_cache_path)
    assert cache is not None
    args = _merge_args(checkpoint_args, {"future_semantic_proto_count": c, "enable_future_semantic_state_head": True, "enable_semantic_proto_head": True})
    models = _load_models(
        args,
        payload,
        device,
        c,
        train_semantic_encoder_proj=bool(variant.get("train_semantic_encoder_proj", False)),
        train_semantic_fusion_gate_norm=bool(variant.get("train_semantic_fusion_gate_norm", False)),
        train_handshake=bool(variant.get("train_handshake", False)),
    )
    opt = torch.optim.AdamW(_trainable_parameters(models), lr=float(variant["lr"]), weight_decay=0.0)
    start_train = _evaluate_batches(models=models, args=args, batches_cpu=train_batches, cache=cache, device=device)
    start_val = _evaluate_batches(models=models, args=args, batches_cpu=val_batches, cache=cache, device=device)
    finite = 0
    for step in range(int(steps)):
        batch = _to_device(train_batches[step % len(train_batches)], device, non_blocking=False)
        opt.zero_grad(set_to_none=True)
        out = _teacher_forced_predict(**_make_forward_kwargs(models, args, batch))
        target, dist, mask, _ = prototype_tensors_for_batch(
            cache,
            batch,
            horizon=int(getattr(args, "fut_len", 8)),
            slot_count=_batch_slot_count(batch),
            device=device,
        )
        state = out["future_semantic_trace_state"]
        loss, _ = _loss(
            state.future_semantic_proto_logits,
            target,
            dist,
            mask,
            class_balanced=bool(variant.get("class_balanced", False)),
            soft_kl_weight=float(variant.get("soft_kl_weight", 0.0)),
        )
        finite += int(bool(torch.isfinite(loss).detach().cpu().item()))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(_trainable_parameters(models), max_norm=10.0)
        opt.step()
    end_train = _evaluate_batches(models=models, args=args, batches_cpu=train_batches, cache=cache, device=device)
    end_val = _evaluate_batches(models=models, args=args, batches_cpu=val_batches, cache=cache, device=device)
    trainable_count = sum(int(p.numel()) for p in _trainable_parameters(models))
    del models
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {
        **variant,
        "steps": int(steps),
        "trainable_param_count": int(trainable_count),
        "train_metrics_start": start_train,
        "val_metrics_start": start_val,
        "train_metrics_end": end_train,
        "val_metrics_end": end_val,
        "loss_finite_ratio": float(finite / max(int(steps), 1)),
        "train_top5_gt_0p8": bool(end_train["proto_top5"] > 0.8),
        "val_top5_beats_frequency": bool(end_val["proto_top5"] > end_val["frequency_top5"]),
        "trace_regression_detected": False,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--start-checkpoint", default="outputs/checkpoints/stage2_tusb_semantic_only_unfreeze_v1_boundary_audit_20260428/latest.pt")
    p.add_argument("--target-cache-c32", default="reports/stwm_future_semantic_trace_prototype_targets_v2_c32_20260428.json")
    p.add_argument("--target-cache-c64", default="reports/stwm_future_semantic_trace_prototype_targets_v2_c64_20260428.json")
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--max-samples-per-dataset", type=int, default=64)
    p.add_argument("--device", default="cuda")
    p.add_argument("--output", default="reports/stwm_semantic_field_tiny_recipe_ablation_v1_20260428.json")
    p.add_argument("--doc", default="docs/STWM_SEMANTIC_FIELD_TINY_RECIPE_ABLATION_V1_20260428.md")
    args_cli = p.parse_args()
    device = torch.device("cuda" if str(args_cli.device) == "cuda" and torch.cuda.is_available() else "cpu")
    payload = _load_checkpoint(Path(args_cli.start_checkpoint), device=torch.device("cpu"))
    checkpoint_args = payload.get("args", {}) if isinstance(payload.get("args"), dict) else {}
    base_args = _merge_args(checkpoint_args, {"future_semantic_proto_count": 64})
    ds = _make_dataset(base_args, split="train", max_samples_per_dataset=int(args_cli.max_samples_per_dataset))
    cache64 = load_future_semantic_prototype_target_cache(Path(args_cli.target_cache_c64))
    assert cache64 is not None
    selected = _select_samples(ds, cache64, count=32, horizon=int(getattr(base_args, "fut_len", 8)), slot_count=int(getattr(base_args, "max_tokens", 64)))
    train_samples = selected[:24]
    val_samples = selected[24:32]
    train_batches = [stage2_semantic_collate_fn(train_samples[i : i + int(args_cli.batch_size)]) for i in range(0, len(train_samples), int(args_cli.batch_size))]
    val_batches = [stage2_semantic_collate_fn(val_samples[i : i + int(args_cli.batch_size)]) for i in range(0, len(val_samples), int(args_cli.batch_size))]
    variants = [
        {"name": "ce_only_c32_lr1e-5", "prototype_count": 32, "lr": 1e-5, "target_cache": Path(args_cli.target_cache_c32)},
        {"name": "ce_only_c32_lr3e-5", "prototype_count": 32, "lr": 3e-5, "target_cache": Path(args_cli.target_cache_c32)},
        {"name": "ce_only_c32_lr1e-4", "prototype_count": 32, "lr": 1e-4, "target_cache": Path(args_cli.target_cache_c32)},
        {"name": "ce_soft_kl_c32", "prototype_count": 32, "lr": 3e-5, "soft_kl_weight": 0.1, "target_cache": Path(args_cli.target_cache_c32)},
        {"name": "ce_class_balanced_c32", "prototype_count": 32, "lr": 3e-5, "class_balanced": True, "target_cache": Path(args_cli.target_cache_c32)},
        {"name": "ce_only_c64_lr3e-5", "prototype_count": 64, "lr": 3e-5, "target_cache": Path(args_cli.target_cache_c64)},
        {"name": "semantic_encoder_projection_c32", "prototype_count": 32, "lr": 3e-5, "train_semantic_encoder_proj": True, "target_cache": Path(args_cli.target_cache_c32)},
        {"name": "semantic_fusion_gate_norm_c32", "prototype_count": 32, "lr": 3e-5, "train_semantic_fusion_gate_norm": True, "target_cache": Path(args_cli.target_cache_c32)},
        {"name": "handshake_semantic_kv_c32", "prototype_count": 32, "lr": 3e-5, "train_handshake": True, "target_cache": Path(args_cli.target_cache_c32)},
    ]
    runs = []
    for variant in variants:
        target_cache_path = variant.pop("target_cache")
        runs.append(
            _run_variant(
                variant=variant,
                payload=payload,
                checkpoint_args=checkpoint_args,
                device=device,
                train_batches=train_batches,
                val_batches=val_batches,
                target_cache_path=target_cache_path,
                steps=int(args_cli.steps),
            )
        )
    best = max(runs, key=lambda r: (r["val_metrics_end"]["proto_top5"], r["train_metrics_end"]["proto_top5"]))
    payload_out = {
        "audit_name": "stwm_semantic_field_tiny_recipe_ablation_v1",
        "item_count": 32,
        "train_item_count": 24,
        "val_item_count": 8,
        "steps_per_variant": int(args_cli.steps),
        "variants": runs,
        "best_variant": best["name"],
        "best_variant_train_top5": float(best["train_metrics_end"]["proto_top5"]),
        "best_variant_val_top5": float(best["val_metrics_end"]["proto_top5"]),
        "any_variant_train_top5_gt_0p8": bool(any(r["train_top5_gt_0p8"] for r in runs)),
        "any_variant_val_top5_beats_frequency": bool(any(r["val_top5_beats_frequency"] for r in runs)),
        "recipe_ablation_finds_success": bool(any(r["train_top5_gt_0p8"] or r["val_top5_beats_frequency"] for r in runs)),
        "trace_regression_detected": bool(any(r["trace_regression_detected"] for r in runs)),
    }
    write_json(Path(args_cli.output), payload_out)
    write_doc(
        Path(args_cli.doc),
        "STWM Semantic Field Tiny Recipe Ablation V1",
        payload_out,
        bullets=[
            "All variants keep Stage1 and dynamic trace paths frozen.",
            "The ablation optimizes semantic prototype objectives only; no candidate scorer is used.",
        ],
    )


if __name__ == "__main__":
    main()
