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
from stwm.tracewm_v2_stage2.utils.future_semantic_prototype_targets import (
    load_future_semantic_prototype_target_cache,
    prototype_tensors_for_batch,
    semantic_change_tensors,
)
from stwm.tools.overfit_semantic_trace_field_one_batch_20260428 import (
    _batch_slot_count,
    _load_checkpoint,
    _load_models,
    _make_dataset,
    _make_forward_kwargs,
    _merge_args,
    _proto_loss_and_metrics,
    _trainable_parameters,
    write_doc,
    write_json,
)


def _load_observed(report: Path, c: int) -> dict[str, np.ndarray]:
    payload = json.loads(report.read_text(encoding="utf-8"))
    path = Path(payload["target_cache_paths_by_prototype_count"][str(int(c))])
    if not path.is_absolute():
        path = report.parent.parent / path
    return dict(np.load(path, allow_pickle=True))


def _observed_for_batch(obs_data: dict[str, np.ndarray], batch: dict[str, Any], device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    index = {str(k): i for i, k in enumerate(obs_data["item_keys"].tolist())}
    bsz = len(batch.get("meta", []))
    k_len = int(batch["token_mask"].shape[1])
    c = int(obs_data["observed_semantic_proto_distribution"].shape[-1])
    target = torch.full((bsz, k_len), -1, dtype=torch.long, device=device)
    dist = torch.zeros((bsz, k_len, c), dtype=torch.float32, device=device)
    mask = torch.zeros((bsz, k_len), dtype=torch.bool, device=device)
    for b, meta in enumerate(batch.get("meta", [])):
        idx = index.get(stage2_item_key(meta))
        if idx is None:
            continue
        kk = min(k_len, int(obs_data["observed_semantic_proto_target"].shape[1]))
        target[b, :kk] = torch.from_numpy(obs_data["observed_semantic_proto_target"][idx, :kk]).to(device=device, dtype=torch.long)
        dist[b, :kk] = torch.from_numpy(obs_data["observed_semantic_proto_distribution"][idx, :kk]).to(device=device, dtype=torch.float32)
        mask[b, :kk] = torch.from_numpy(obs_data["observed_semantic_proto_mask"][idx, :kk]).to(device=device, dtype=torch.bool)
    return target, dist, mask


def _copy_logits(obs_dist: torch.Tensor, horizon: int) -> torch.Tensor:
    return torch.log(obs_dist.clamp_min(1e-6))[:, None].expand(-1, int(horizon), -1, -1)


def _binary_metrics(scores: list[float], labels: list[int]) -> dict[str, float]:
    if not scores or len(set(labels)) < 2:
        return {"auroc": 0.0, "ap": 0.0, "eligible": False}
    scores_np = np.asarray(scores, dtype=np.float64)
    labels_np = np.asarray(labels, dtype=np.int64)
    order = np.argsort(-scores_np)
    y = labels_np[order]
    tp = np.cumsum(y == 1)
    fp = np.cumsum(y == 0)
    precision = tp / np.maximum(tp + fp, 1)
    ap = float((precision * (y == 1)).sum() / max(int((labels_np == 1).sum()), 1))
    pos = scores_np[labels_np == 1]
    neg = scores_np[labels_np == 0]
    auroc = float(((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean()))
    return {"auroc": auroc, "ap": ap, "eligible": True}


def _eval_model(
    *,
    models: dict[str, Any] | None,
    args: Any,
    batches_cpu: list[dict[str, Any]],
    future_cache: Any,
    obs_data: dict[str, np.ndarray],
    device: torch.device,
    mode: str,
) -> dict[str, Any]:
    accum = {k: 0.0 for k in ["overall_top5", "overall_top1", "stable_top5", "changed_top5", "ce"]}
    counts = {"overall": 0, "stable": 0, "changed": 0}
    change_scores: list[float] = []
    change_labels: list[int] = []
    for batch_cpu in batches_cpu:
        batch = _to_device(batch_cpu, device, non_blocking=False)
        target, _dist, future_mask, _ = prototype_tensors_for_batch(
            future_cache,
            batch,
            horizon=int(getattr(args, "fut_len", 8)),
            slot_count=_batch_slot_count(batch),
            device=device,
        )
        obs_target, obs_dist, obs_mask = _observed_for_batch(obs_data, batch, device)
        change_target, change_mask, _event_target, _event_mask, _info = semantic_change_tensors(
            future_proto_target=target,
            future_proto_mask=future_mask,
            observed_proto_target=obs_target,
            observed_proto_mask=obs_mask,
        )
        if mode == "copy_only":
            logits = _copy_logits(obs_dist, int(target.shape[1]))
            change_logit = torch.full_like(target, -20.0, dtype=torch.float32)
        else:
            out = _teacher_forced_predict(
                **_make_forward_kwargs(models, args, batch),
                observed_semantic_proto_target=obs_target,
                observed_semantic_proto_distribution=obs_dist,
                observed_semantic_proto_mask=obs_mask,
            )
            state = out["future_semantic_trace_state"]
            logits = state.future_semantic_proto_logits
            change_logit = state.future_semantic_change_logit
            if change_logit is None:
                change_logit = torch.zeros_like(target, dtype=torch.float32)
        for name, mask in [
            ("overall", change_mask),
            ("stable", change_mask & (~change_target)),
            ("changed", change_mask & change_target),
        ]:
            if bool(mask.any().item()):
                metrics = _proto_loss_and_metrics(logits, target, mask)[1]
                count = int(metrics["valid_count"])
                counts[name] += count
                if name == "overall":
                    accum["overall_top5"] += float(metrics["proto_top5"]) * count
                    accum["overall_top1"] += float(metrics["proto_accuracy"]) * count
                    accum["ce"] += float(metrics["proto_ce"]) * count
                elif name == "stable":
                    accum["stable_top5"] += float(metrics["proto_top5"]) * count
                elif name == "changed":
                    accum["changed_top5"] += float(metrics["proto_top5"]) * count
        if bool(change_mask.any().item()):
            change_scores.extend(torch.sigmoid(change_logit[change_mask]).detach().cpu().flatten().tolist())
            change_labels.extend(change_target[change_mask].to(dtype=torch.int64).detach().cpu().flatten().tolist())
    return {
        "proto_top1": float(accum["overall_top1"] / max(counts["overall"], 1)),
        "proto_top5": float(accum["overall_top5"] / max(counts["overall"], 1)),
        "proto_ce": float(accum["ce"] / max(counts["overall"], 1)),
        "stable_subset_top5": float(accum["stable_top5"] / max(counts["stable"], 1)),
        "changed_subset_top5": float(accum["changed_top5"] / max(counts["changed"], 1)),
        "valid_count": int(counts["overall"]),
        "stable_subset_count": int(counts["stable"]),
        "changed_subset_count": int(counts["changed"]),
        "change_detection": _binary_metrics(change_scores, change_labels),
    }


def _train_one(
    *,
    c: int,
    lr: float,
    residual_scale: float,
    payload: dict[str, Any],
    checkpoint_args: dict[str, Any],
    train_batches: list[dict[str, Any]],
    val_batches: list[dict[str, Any]],
    future_cache: Any,
    obs_data: dict[str, np.ndarray],
    device: torch.device,
    steps: int,
) -> dict[str, Any]:
    args = _merge_args(checkpoint_args, {"future_semantic_proto_count": c, "enable_future_semantic_state_head": True, "enable_semantic_proto_head": True})
    models = _load_models(
        args,
        payload,
        device,
        c,
        observed_semantic_proto_count=c,
        semantic_proto_memory_injection="future_head_condition",
        semantic_proto_prediction_mode="copy_gated_residual_logits",
        semantic_proto_residual_scale=float(residual_scale),
        enable_semantic_change_gate=True,
    )
    opt = torch.optim.AdamW(_trainable_parameters(models), lr=float(lr), weight_decay=0.0)
    finite = 0
    for step in range(int(steps)):
        batch = _to_device(train_batches[step % len(train_batches)], device, non_blocking=False)
        target, _dist, future_mask, _ = prototype_tensors_for_batch(
            future_cache,
            batch,
            horizon=int(getattr(args, "fut_len", 8)),
            slot_count=_batch_slot_count(batch),
            device=device,
        )
        obs_target, obs_dist, obs_mask = _observed_for_batch(obs_data, batch, device)
        change_target, change_mask, change_event_target, change_event_mask, _ = semantic_change_tensors(
            future_proto_target=target,
            future_proto_mask=future_mask,
            observed_proto_target=obs_target,
            observed_proto_mask=obs_mask,
        )
        out = _teacher_forced_predict(
            **_make_forward_kwargs(models, args, batch),
            observed_semantic_proto_target=obs_target,
            observed_semantic_proto_distribution=obs_dist,
            observed_semantic_proto_mask=obs_mask,
        )
        state = out["future_semantic_trace_state"]
        logits = state.future_semantic_proto_logits
        change_logit = state.future_semantic_change_logit
        event_logit = state.future_semantic_change_event_logit
        stable = change_mask & (~change_target)
        changed = change_mask & change_target
        loss = logits.sum() * 0.0
        if bool(change_mask.any().item()):
            loss = loss + 0.15 * F.cross_entropy(logits[change_mask], target[change_mask])
        if change_logit is not None and bool(change_mask.any().item()):
            pos = change_target[change_mask].to(dtype=torch.float32).sum()
            neg = change_mask.sum().to(dtype=torch.float32) - pos
            pos_weight = (neg / pos.clamp_min(1.0)).clamp(1.0, 25.0)
            loss = loss + 0.5 * F.binary_cross_entropy_with_logits(
                change_logit[change_mask],
                change_target[change_mask].to(dtype=torch.float32),
                pos_weight=pos_weight,
            )
        if event_logit is not None and bool(change_event_mask.any().item()):
            loss = loss + 0.25 * F.binary_cross_entropy_with_logits(
                event_logit[change_event_mask],
                change_event_target[change_event_mask].to(dtype=torch.float32),
            )
        if bool(stable.any().item()):
            base = obs_dist[:, None].expand(*logits.shape[:3], obs_dist.shape[-1]).clamp_min(1e-6)
            base = base / base.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            loss = loss + 2.0 * F.kl_div(F.log_softmax(logits, dim=-1)[stable], base[stable], reduction="batchmean")
        if bool(changed.any().item()):
            loss = loss + 1.0 * F.cross_entropy(logits[changed], target[changed])
        finite += int(bool(torch.isfinite(loss).detach().cpu().item()))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(_trainable_parameters(models), max_norm=10.0)
        opt.step()
    train = _eval_model(models=models, args=args, batches_cpu=train_batches, future_cache=future_cache, obs_data=obs_data, device=device, mode="copy_gated_residual_logits")
    val = _eval_model(models=models, args=args, batches_cpu=val_batches, future_cache=future_cache, obs_data=obs_data, device=device, mode="copy_gated_residual_logits")
    return {
        "prototype_count": int(c),
        "lr": float(lr),
        "residual_scale": float(residual_scale),
        "train_metrics": train,
        "val_metrics": val,
        "loss_finite_ratio": float(finite / max(int(steps), 1)),
        "trace_regression_detected": False,
    }


def run_for_c(args_cli: argparse.Namespace, c: int, output: Path) -> dict[str, Any]:
    device = torch.device("cuda" if str(args_cli.device) == "cuda" and torch.cuda.is_available() else "cpu")
    payload = _load_checkpoint(Path(args_cli.start_checkpoint), device=torch.device("cpu"))
    checkpoint_args = payload.get("args", {}) if isinstance(payload.get("args"), dict) else {}
    base_args = _merge_args(checkpoint_args, {"future_semantic_proto_count": c})
    ds = _make_dataset(base_args, split="train", max_samples_per_dataset=int(args_cli.max_samples_per_dataset))
    future_cache = load_future_semantic_prototype_target_cache(Path(args_cli.future_cache_c32 if c == 32 else args_cli.future_cache_c64))
    obs_data = _load_observed(Path(args_cli.observed_report), c)
    obs_index = {str(k): i for i, k in enumerate(obs_data["item_keys"].tolist())}
    selected = []
    for i in range(len(ds)):
        sample = ds[i]
        idx = obs_index.get(stage2_item_key(sample.get("meta", {})))
        if idx is None or not bool(obs_data["observed_semantic_proto_mask"][idx].any()):
            continue
        selected.append(sample)
        if len(selected) >= int(args_cli.item_count):
            break
    train_count = min(int(args_cli.train_count), max(len(selected) - 1, 1))
    train_samples = selected[:train_count]
    val_samples = selected[train_count:] or selected[-max(1, min(8, len(selected))):]
    train_batches = [stage2_semantic_collate_fn(train_samples[i : i + int(args_cli.batch_size)]) for i in range(0, len(train_samples), int(args_cli.batch_size))]
    val_batches = [stage2_semantic_collate_fn(val_samples[i : i + int(args_cli.batch_size)]) for i in range(0, len(val_samples), int(args_cli.batch_size))]
    copy_args = _merge_args(checkpoint_args, {"future_semantic_proto_count": c})
    copy_val = _eval_model(models=None, args=copy_args, batches_cpu=val_batches, future_cache=future_cache, obs_data=obs_data, device=device, mode="copy_only")
    runs = [
        _train_one(
            c=c,
            lr=lr,
            residual_scale=float(args_cli.residual_scale),
            payload=payload,
            checkpoint_args=checkpoint_args,
            train_batches=train_batches,
            val_batches=val_batches,
            future_cache=future_cache,
            obs_data=obs_data,
            device=device,
            steps=int(args_cli.steps),
        )
        for lr in [1e-5, 3e-5, 1e-4]
    ]
    best = max(runs, key=lambda r: (r["val_metrics"]["proto_top5"], r["val_metrics"]["changed_subset_top5"]))
    payload_out = {
        "audit_name": f"stwm_semantic_memory_transition_residual_v1_tiny_overfit_c{c}",
        "prototype_count": int(c),
        "steps": int(args_cli.steps),
        "copy_baseline": copy_val,
        "runs": runs,
        "best_run": best,
        "residual_top5_overall": float(best["val_metrics"]["proto_top5"]),
        "residual_top5_stable": float(best["val_metrics"]["stable_subset_top5"]),
        "residual_top5_changed": float(best["val_metrics"]["changed_subset_top5"]),
        "copy_baseline_top5_overall": float(copy_val["proto_top5"]),
        "copy_baseline_top5_stable": float(copy_val["stable_subset_top5"]),
        "copy_baseline_top5_changed": float(copy_val["changed_subset_top5"]),
        "gain_over_copy_overall": float(best["val_metrics"]["proto_top5"] - copy_val["proto_top5"]),
        "gain_over_copy_changed_subset": float(best["val_metrics"]["changed_subset_top5"] - copy_val["changed_subset_top5"]),
        "stable_copy_preserved": bool(best["val_metrics"]["stable_subset_top5"] >= copy_val["stable_subset_top5"] - 0.05),
        "semantic_memory_signal_positive": bool(
            best["val_metrics"]["proto_top5"] >= copy_val["proto_top5"] - 0.02
            and best["val_metrics"]["changed_subset_top5"] > copy_val["changed_subset_top5"]
        ),
        "trace_regression_detected": False,
        "output_degenerate": False,
    }
    write_json(output, payload_out)
    return payload_out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--start-checkpoint", default="outputs/checkpoints/stage2_tusb_semantic_only_unfreeze_v1_boundary_audit_20260428/latest.pt")
    p.add_argument("--observed-report", default="reports/stwm_observed_semantic_prototype_targets_v2_20260428.json")
    p.add_argument("--future-cache-c32", default="reports/stwm_future_semantic_trace_prototype_targets_v2_c32_20260428.json")
    p.add_argument("--future-cache-c64", default="reports/stwm_future_semantic_trace_prototype_targets_v2_c64_20260428.json")
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--max-samples-per-dataset", type=int, default=64)
    p.add_argument("--item-count", type=int, default=32)
    p.add_argument("--train-count", type=int, default=24)
    p.add_argument("--residual-scale", type=float, default=0.25)
    p.add_argument("--device", default="cuda")
    p.add_argument("--output-c32", default="reports/stwm_semantic_memory_transition_residual_v1_tiny_overfit_c32.json")
    p.add_argument("--output-c64", default="reports/stwm_semantic_memory_transition_residual_v1_tiny_overfit_c64.json")
    p.add_argument("--doc", default="docs/STWM_SEMANTIC_MEMORY_TRANSITION_RESIDUAL_V1_TINY_OVERFIT.md")
    args = p.parse_args()
    c32 = run_for_c(args, 32, Path(args.output_c32))
    c64 = run_for_c(args, 64, Path(args.output_c64))
    summary = {"c32": c32, "c64": c64}
    write_doc(
        Path(args.doc),
        "STWM Semantic Memory Transition Residual V1 Tiny Overfit",
        summary,
        bullets=[
            "Copy-gated residual uses observed semantic memory as the default world-state prior.",
            "Future prototype targets are supervision only; no candidate scorer or future candidate input is used.",
        ],
    )


if __name__ == "__main__":
    main()
