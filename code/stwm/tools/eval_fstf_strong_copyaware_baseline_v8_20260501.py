#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

from stwm.tracewm_v2_stage2.trainers.train_tracewm_stage2_smalltrain import _to_device
from stwm.tracewm_v2_stage2.utils.future_semantic_prototype_targets import (
    load_future_semantic_prototype_target_cache,
    prototype_tensors_for_batch,
    semantic_change_tensors,
)
from stwm.tools.run_semantic_memory_transition_residual_tiny_overfit_20260428 import _load_observed, _observed_for_batch
from stwm.tools.train_fstf_same_output_baseline_v7_20260501 import batch_slot_count, load_batches, proto_loss_and_metrics, write_json
from stwm.tools.train_fstf_strong_copyaware_baseline_v8_20260501 import (
    CopyAwareFSTFBaseline,
    apply_process_title,
    baseline_metadata,
    copy_logits,
)


def binary_metrics(scores: list[float], labels: list[int]) -> dict[str, Any]:
    if not scores or len(set(labels)) < 2:
        return {"eligible": False, "ap": 0.0, "auroc": 0.0, "status": "metric_invalid_or_untrained"}
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
    return {"eligible": True, "ap": ap, "auroc": auroc, "status": "computed"}


def bootstrap(vals: list[float], seed: int = 20260501, samples: int = 2000) -> dict[str, Any]:
    arr = np.asarray(vals, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"item_count": 0, "mean_delta": 0.0, "ci95": [0.0, 0.0], "zero_excluded": False, "bootstrap_win_rate": 0.0}
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(samples):
        idx = rng.integers(0, arr.size, size=arr.size)
        means.append(float(arr[idx].mean()))
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {"item_count": int(arr.size), "mean_delta": float(arr.mean()), "ci95": [float(lo), float(hi)], "zero_excluded": bool(lo > 0.0 or hi < 0.0), "bootstrap_win_rate": float(np.mean(np.asarray(means) > 0.0))}


def item_metrics(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> tuple[int, float, float, float]:
    _loss, m = proto_loss_and_metrics(logits, target, mask)
    return int(m["valid_count"]), float(m["proto_accuracy"]), float(m["proto_top5"]), float(m["proto_ce"])


def load_model(path: Path, device: torch.device) -> tuple[CopyAwareFSTFBaseline, dict[str, Any], int]:
    payload = torch.load(path, map_location="cpu")
    cfg = dict(payload["config"])
    model = CopyAwareFSTFBaseline(**cfg)
    model.load_state_dict(payload["model_state_dict"])
    model.to(device)
    model.eval()
    return model, cfg, int(payload.get("seed", -1))


def evaluate(
    *,
    baseline: str,
    model: CopyAwareFSTFBaseline | None,
    batches: list[dict[str, Any]],
    future_cache: Any,
    obs_data: dict[str, np.ndarray],
    device: torch.device,
    horizon: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    sums = {k: 0.0 for k in ["res_o_t1", "res_o_t5", "res_o_ce", "copy_o_t1", "copy_o_t5", "copy_o_ce", "res_s_t5", "copy_s_t5", "res_c_t5", "copy_c_t5"]}
    counts = {"overall": 0, "stable": 0, "changed": 0}
    item_scores: list[dict[str, Any]] = []
    trace_errors: list[float] = []
    change_scores: list[float] = []
    change_labels: list[int] = []
    with torch.no_grad():
        for batch_cpu in batches:
            batch = _to_device(batch_cpu, device, non_blocking=False)
            target, _dist, future_mask, _ = prototype_tensors_for_batch(future_cache, batch, horizon=horizon, slot_count=batch_slot_count(batch), device=device)
            obs_target, obs_dist, obs_mask = _observed_for_batch(obs_data, batch, device)
            change_target, change_mask, *_ = semantic_change_tensors(
                future_proto_target=target,
                future_proto_mask=future_mask,
                observed_proto_target=obs_target,
                observed_proto_mask=obs_mask,
            )
            copy = copy_logits(obs_dist, horizon)
            if baseline == "copy_semantic_memory_baseline":
                logits = copy
                gate_logit = torch.full_like(target, -20.0, dtype=torch.float32)
            elif baseline == "oracle_change_gate_upper_bound":
                logits = copy.clone()
                oracle = torch.full_like(logits, -20.0)
                oracle.scatter_(-1, target.clamp_min(0).unsqueeze(-1), 20.0)
                logits = torch.where(change_target[..., None], oracle, logits)
                gate_logit = torch.where(change_target, torch.full_like(target, 20.0, dtype=torch.float32), torch.full_like(target, -20.0, dtype=torch.float32))
            else:
                assert model is not None
                logits, gate_logit = model(batch, obs_dist)
                if gate_logit is None:
                    prob = torch.softmax(logits, dim=-1)
                    gate_logit = 1.0 - prob.gather(-1, obs_target[:, None, :, None].clamp_min(0).expand(-1, horizon, -1, 1)).squeeze(-1)
            if bool(change_mask.any().item()):
                scores = torch.sigmoid(gate_logit[change_mask]) if gate_logit.dtype.is_floating_point else gate_logit[change_mask].float()
                change_scores.extend(scores.detach().cpu().flatten().tolist())
                change_labels.extend(change_target[change_mask].to(torch.int64).detach().cpu().flatten().tolist())
            fut_valid = batch.get("fut_valid")
            if isinstance(fut_valid, torch.Tensor) and bool(fut_valid.any().item()):
                pred_coord = batch["obs_state"][:, -1:, :, :2].expand(-1, horizon, -1, -1)
                target_coord = batch["fut_state"][:, :horizon, :, :2]
                valid = fut_valid[:, :horizon].bool()
                err = torch.sqrt(((pred_coord - target_coord) ** 2).sum(dim=-1).clamp_min(1e-12))
                trace_errors.append(float(err[valid].mean().detach().cpu().item()))
            for b, meta in enumerate(batch["meta"]):
                key = str(meta.get("item_key", ""))
                per = {"item_key": key}
                for name, mask in [("overall", change_mask[b]), ("stable", change_mask[b] & (~change_target[b])), ("changed", change_mask[b] & change_target[b])]:
                    n, acc, top5, ce = item_metrics(logits[b : b + 1], target[b : b + 1], mask[None])
                    nc, accc, top5c, cec = item_metrics(copy[b : b + 1], target[b : b + 1], mask[None])
                    per[f"{name}_count"] = n
                    per[f"residual_{name}_top1"] = acc
                    per[f"residual_{name}_top5"] = top5
                    per[f"residual_{name}_ce"] = ce
                    per[f"copy_{name}_top1"] = accc
                    per[f"copy_{name}_top5"] = top5c
                    per[f"copy_{name}_ce"] = cec
                    if n <= 0:
                        continue
                    counts[name] += n
                    if name == "overall":
                        sums["res_o_t1"] += acc * n
                        sums["res_o_t5"] += top5 * n
                        sums["res_o_ce"] += ce * n
                        sums["copy_o_t1"] += accc * n
                        sums["copy_o_t5"] += top5c * n
                        sums["copy_o_ce"] += cec * n
                    elif name == "stable":
                        sums["res_s_t5"] += top5 * n
                        sums["copy_s_t5"] += top5c * n
                    else:
                        sums["res_c_t5"] += top5 * n
                        sums["copy_c_t5"] += top5c * n
                item_scores.append(per)
    metrics = {
        "proto_top1": sums["res_o_t1"] / max(counts["overall"], 1),
        "proto_top5": sums["res_o_t5"] / max(counts["overall"], 1),
        "proto_ce": sums["res_o_ce"] / max(counts["overall"], 1),
        "copy_proto_top1": sums["copy_o_t1"] / max(counts["overall"], 1),
        "copy_proto_top5": sums["copy_o_t5"] / max(counts["overall"], 1),
        "copy_proto_ce": sums["copy_o_ce"] / max(counts["overall"], 1),
        "stable_subset_top5": sums["res_s_t5"] / max(counts["stable"], 1),
        "copy_stable_subset_top5": sums["copy_s_t5"] / max(counts["stable"], 1),
        "changed_subset_top5": sums["res_c_t5"] / max(counts["changed"], 1),
        "copy_changed_subset_top5": sums["copy_c_t5"] / max(counts["changed"], 1),
        "valid_count": counts["overall"],
        "stable_subset_count": counts["stable"],
        "changed_subset_count": counts["changed"],
        "overall_gain_over_copy": sums["res_o_t5"] / max(counts["overall"], 1) - sums["copy_o_t5"] / max(counts["overall"], 1),
        "changed_subset_gain_over_copy": sums["res_c_t5"] / max(counts["changed"], 1) - sums["copy_c_t5"] / max(counts["changed"], 1),
        "stable_preservation_drop": sums["copy_s_t5"] / max(counts["stable"], 1) - sums["res_s_t5"] / max(counts["stable"], 1),
        "future_trace_coord_error": float(np.mean(trace_errors)) if trace_errors else 0.0,
        "visibility": {"eligible": False, "status": "metric_invalid_or_untrained", "ap": 0.0, "auroc": 0.0},
        "reappearance": {"eligible": False, "status": "metric_invalid_or_untrained", "ap": 0.0, "auroc": 0.0},
        "change_detection": binary_metrics(change_scores, change_labels),
    }
    return metrics, item_scores


def main() -> None:
    apply_process_title()
    p = argparse.ArgumentParser()
    p.add_argument("--baseline", required=True)
    p.add_argument("--checkpoint", default="")
    p.add_argument("--test-cache-report", default="reports/stwm_mixed_fullscale_v2_materialization_test_20260428.json")
    p.add_argument("--observed-report", default="reports/stwm_mixed_observed_semantic_prototype_targets_v2_20260428.json")
    p.add_argument("--future-cache-report", default="reports/stwm_fullscale_semantic_trace_prototype_targets_c32_v1_20260428.json")
    p.add_argument("--device", default="cuda")
    p.add_argument("--output", required=True)
    args = p.parse_args()
    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    model = None
    cfg: dict[str, Any] = {"prototype_count": 32, "horizon": 8}
    seed = -1
    if args.baseline not in {"copy_semantic_memory_baseline", "oracle_change_gate_upper_bound"}:
        model, cfg, seed = load_model(Path(args.checkpoint), device)
    batches, report = load_batches(Path(args.test_cache_report))
    future_cache = load_future_semantic_prototype_target_cache(Path(args.future_cache_report))
    obs_data = _load_observed(Path(args.observed_report), int(cfg.get("prototype_count", 32)))
    metrics, item_scores = evaluate(baseline=args.baseline, model=model, batches=batches, future_cache=future_cache, obs_data=obs_data, device=device, horizon=int(cfg.get("horizon", 8)))
    payload = {
        "audit_name": "stwm_fstf_strong_copyaware_baseline_v8_eval",
        **baseline_metadata(args.baseline),
        "seed": seed,
        "checkpoint_path": str(args.checkpoint),
        "heldout_item_count": int(report.get("final_eval_item_count", len(batches))),
        "free_rollout_path": "copy_aware_baseline_forward_observed_inputs_only",
        "teacher_forced_path_used": False,
        "candidate_scorer_used": False,
        "future_candidate_leakage": False,
        "metrics": metrics,
        "item_scores": item_scores,
        "visibility_reappearance_status": "metric_invalid_or_untrained",
    }
    write_json(args.output, payload)
    print(f"[fstf-v8-eval] done baseline={args.baseline} seed={seed} output={args.output}", flush=True)


if __name__ == "__main__":
    main()
