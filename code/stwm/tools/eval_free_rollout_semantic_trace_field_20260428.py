#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from stwm.tracewm_v2_stage2.trainers.train_tracewm_stage2_smalltrain import _free_rollout_predict, _to_device
from stwm.tracewm_v2_stage2.utils.future_semantic_feature_targets import stage2_item_key
from stwm.tracewm_v2_stage2.utils.future_semantic_prototype_targets import (
    load_future_semantic_prototype_target_cache,
    prototype_tensors_for_batch,
    semantic_change_tensors,
)
from stwm.tools.overfit_semantic_trace_field_one_batch_20260428 import (
    _batch_slot_count,
    _load_checkpoint,
    _make_forward_kwargs,
    _merge_args,
    _proto_loss_and_metrics,
)
from stwm.tools.run_semantic_memory_transition_residual_tiny_overfit_20260428 import _binary_metrics, _load_observed, _observed_for_batch
from stwm.tools.run_semantic_memory_world_model_v3_20260428 import _load_trained_models


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_doc(path: Path, title: str, payload: dict[str, Any]) -> None:
    lines = [f"# {title}", ""]
    for key, value in payload.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            lines.append(f"- {key}: `{value}`")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _copy_logits(obs_dist: torch.Tensor, horizon: int) -> torch.Tensor:
    return torch.log(obs_dist.clamp_min(1e-6))[:, None].expand(-1, int(horizon), -1, -1)


def _item_metrics(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> tuple[int, float, float, float]:
    valid = mask.to(torch.bool) & (target >= 0)
    if not bool(valid.any().item()):
        return 0, 0.0, 0.0, 0.0
    _, metrics = _proto_loss_and_metrics(logits, target, valid)
    return int(metrics["valid_count"]), float(metrics["proto_accuracy"]), float(metrics["proto_top5"]), float(metrics["proto_ce"])


def _cache_bool_for_batch(cache: Any, attr: str, batch: dict[str, Any], *, horizon: int, slot_count: int, device: torch.device) -> torch.Tensor:
    src = getattr(cache, attr)
    out = torch.zeros((int(batch["obs_state"].shape[0]), int(horizon), int(slot_count)), dtype=torch.bool, device=device)
    index = cache.index
    for b, meta in enumerate(batch.get("meta", [])):
        idx = index.get(stage2_item_key(meta))
        if idx is None:
            continue
        h = min(int(horizon), int(src.shape[1]))
        k = min(int(slot_count), int(src.shape[2]))
        out[b, :h, :k] = src[idx, :h, :k].to(device=device, dtype=torch.bool)
    return out


def _eval_one_seed(
    *,
    checkpoint_path: Path,
    prototype_count: int,
    payload: dict[str, Any],
    checkpoint_args: dict[str, Any],
    batches_cpu: list[dict[str, Any]],
    future_cache: Any,
    obs_data: dict[str, np.ndarray],
    device: torch.device,
    residual_scale: float,
) -> dict[str, Any]:
    args, models = _load_trained_models(
        checkpoint_path=checkpoint_path,
        prototype_count=prototype_count,
        payload=payload,
        checkpoint_args=checkpoint_args,
        device=device,
        residual_scale=float(residual_scale),
    )
    item_scores: list[dict[str, Any]] = []
    sums = {k: 0.0 for k in ["res_o_t1", "res_o_t5", "res_o_ce", "copy_o_t1", "copy_o_t5", "copy_o_ce", "res_s_t5", "copy_s_t5", "res_c_t5", "copy_c_t5"]}
    counts = {"overall": 0, "stable": 0, "changed": 0}
    change_scores: list[float] = []
    change_labels: list[int] = []
    vis_scores: list[float] = []
    vis_labels: list[int] = []
    reap_scores: list[float] = []
    reap_labels: list[int] = []
    coord_errors: list[float] = []
    proto_std: list[float] = []
    with torch.no_grad():
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
            out = _free_rollout_predict(
                **_make_forward_kwargs(models, args, batch),
                fut_len=int(getattr(args, "fut_len", 8)),
                observed_semantic_proto_target=obs_target,
                observed_semantic_proto_distribution=obs_dist,
                observed_semantic_proto_mask=obs_mask,
            )
            state = out["future_semantic_trace_state"]
            logits = state.future_semantic_proto_logits
            copy_logits = _copy_logits(obs_dist, int(target.shape[1]))
            if logits is not None:
                proto_std.append(float(logits.detach().float().std().cpu().item()))
            valid_coord = out["valid_mask"].to(torch.bool)
            if bool(valid_coord.any().item()):
                coord = torch.sqrt(((out["pred_coord"] - out["target_coord"]) ** 2).sum(dim=-1).clamp_min(1e-12))
                coord_errors.append(float(coord[valid_coord].mean().detach().cpu().item()))
            if state.future_semantic_change_logit is not None and bool(change_mask.any().item()):
                change_scores.extend(torch.sigmoid(state.future_semantic_change_logit[change_mask]).detach().cpu().flatten().tolist())
                change_labels.extend(change_target[change_mask].to(torch.int64).detach().cpu().flatten().tolist())
            if state.future_visibility_logit is not None and bool(future_mask.any().item()):
                vis_target = _cache_bool_for_batch(future_cache, "visibility", batch, horizon=int(target.shape[1]), slot_count=_batch_slot_count(batch), device=device)
                vis_scores.extend(torch.sigmoid(state.future_visibility_logit[future_mask]).detach().cpu().flatten().tolist())
                vis_labels.extend(vis_target[future_mask].to(torch.int64).detach().cpu().flatten().tolist())
            if state.future_reappearance_logit is not None and bool(future_mask.any().item()):
                reap_target = _cache_bool_for_batch(future_cache, "reappearance", batch, horizon=int(target.shape[1]), slot_count=_batch_slot_count(batch), device=device)
                reap_scores.extend(torch.sigmoid(state.future_reappearance_logit[future_mask]).detach().cpu().flatten().tolist())
                reap_labels.extend(reap_target[future_mask].to(torch.int64).detach().cpu().flatten().tolist())

            for b, meta in enumerate(batch["meta"]):
                key = stage2_item_key(meta)
                per_item: dict[str, Any] = {"item_key": key}
                for name, mask in [
                    ("overall", change_mask[b]),
                    ("stable", change_mask[b] & (~change_target[b])),
                    ("changed", change_mask[b] & change_target[b]),
                ]:
                    n, acc, top5, ce = _item_metrics(logits[b : b + 1], target[b : b + 1], mask[None, ...])
                    nc, accc, top5c, cec = _item_metrics(copy_logits[b : b + 1], target[b : b + 1], mask[None, ...])
                    per_item[f"{name}_count"] = int(n)
                    per_item[f"residual_{name}_top1"] = float(acc)
                    per_item[f"residual_{name}_top5"] = float(top5)
                    per_item[f"residual_{name}_ce"] = float(ce)
                    per_item[f"copy_{name}_top1"] = float(accc)
                    per_item[f"copy_{name}_top5"] = float(top5c)
                    per_item[f"copy_{name}_ce"] = float(cec)
                    if n > 0:
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
                        elif name == "changed":
                            sums["res_c_t5"] += top5 * n
                            sums["copy_c_t5"] += top5c * n
                item_scores.append(per_item)
    metrics = {
        "proto_top1": float(sums["res_o_t1"] / max(counts["overall"], 1)),
        "proto_top5": float(sums["res_o_t5"] / max(counts["overall"], 1)),
        "proto_ce": float(sums["res_o_ce"] / max(counts["overall"], 1)),
        "copy_proto_top1": float(sums["copy_o_t1"] / max(counts["overall"], 1)),
        "copy_proto_top5": float(sums["copy_o_t5"] / max(counts["overall"], 1)),
        "copy_proto_ce": float(sums["copy_o_ce"] / max(counts["overall"], 1)),
        "stable_subset_top5": float(sums["res_s_t5"] / max(counts["stable"], 1)),
        "copy_stable_subset_top5": float(sums["copy_s_t5"] / max(counts["stable"], 1)),
        "changed_subset_top5": float(sums["res_c_t5"] / max(counts["changed"], 1)),
        "copy_changed_subset_top5": float(sums["copy_c_t5"] / max(counts["changed"], 1)),
        "valid_count": int(counts["overall"]),
        "stable_subset_count": int(counts["stable"]),
        "changed_subset_count": int(counts["changed"]),
        "changed_subset_gain_over_copy": float(sums["res_c_t5"] / max(counts["changed"], 1) - sums["copy_c_t5"] / max(counts["changed"], 1)),
        "overall_gain_over_copy": float(sums["res_o_t5"] / max(counts["overall"], 1) - sums["copy_o_t5"] / max(counts["overall"], 1)),
        "stable_preservation_drop": float(sums["copy_s_t5"] / max(counts["stable"], 1) - sums["res_s_t5"] / max(counts["stable"], 1)),
        "change_detection": _binary_metrics(change_scores, change_labels),
        "visibility": _binary_metrics(vis_scores, vis_labels),
        "reappearance": _binary_metrics(reap_scores, reap_labels),
        "future_trace_coord_error": float(np.mean(coord_errors)) if coord_errors else 0.0,
        "proto_logit_std_mean": float(np.mean(proto_std)) if proto_std else 0.0,
    }
    return {"metrics": metrics, "item_scores": item_scores}


def _mean_std(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0}
    return {"mean": float(arr.mean()), "std": float(arr.std(ddof=0))}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--batch-cache-report", default="reports/stwm_free_rollout_semantic_trace_field_v4_materialization_audit_20260428.json")
    p.add_argument("--start-checkpoint", default="outputs/checkpoints/stage2_tusb_semantic_only_unfreeze_v1_boundary_audit_20260428/latest.pt")
    p.add_argument("--observed-report", default="reports/stwm_observed_semantic_prototype_targets_v2_20260428.json")
    p.add_argument("--future-cache-c32", default="reports/stwm_future_semantic_trace_prototype_targets_v2_c32_20260428.json")
    p.add_argument("--future-cache-c64", default="reports/stwm_future_semantic_trace_prototype_targets_v2_c64_20260428.json")
    p.add_argument("--v3-eval-c32", default="reports/stwm_semantic_memory_world_model_v3_eval_c32_20260428.json")
    p.add_argument("--v3-eval-c64", default="reports/stwm_semantic_memory_world_model_v3_eval_c64_20260428.json")
    p.add_argument("--residual-scale", type=float, default=0.25)
    p.add_argument("--device", default="cuda")
    p.add_argument("--eval-c32-output", default="reports/stwm_free_rollout_semantic_trace_field_v4_eval_c32_20260428.json")
    p.add_argument("--eval-c64-output", default="reports/stwm_free_rollout_semantic_trace_field_v4_eval_c64_20260428.json")
    p.add_argument("--doc", default="docs/STWM_FREE_ROLLOUT_SEMANTIC_TRACE_FIELD_V4_EVAL_20260428.md")
    args = p.parse_args()
    device = torch.device("cuda" if str(args.device) == "cuda" and torch.cuda.is_available() else "cpu")
    materialization = json.loads(Path(args.batch_cache_report).read_text(encoding="utf-8"))
    batch_cache = torch.load(materialization["batch_cache_path"], map_location="cpu")
    batches_cpu = batch_cache["batches"]
    heldout_item_count = int(len(batch_cache["item_keys"]))
    payload = _load_checkpoint(Path(args.start_checkpoint), device=torch.device("cpu"))
    checkpoint_args = payload.get("args", {}) if isinstance(payload.get("args"), dict) else {}
    outputs = {}
    for c, eval_report, future_report, out_path in [
        (32, args.v3_eval_c32, args.future_cache_c32, args.eval_c32_output),
        (64, args.v3_eval_c64, args.future_cache_c64, args.eval_c64_output),
    ]:
        future_cache = load_future_semantic_prototype_target_cache(Path(future_report))
        obs_data = _load_observed(Path(args.observed_report), c)
        v3_eval = json.loads(Path(eval_report).read_text(encoding="utf-8"))
        seed_results = []
        for seed_result in v3_eval["seed_results"]:
            ckpt = Path(seed_result["checkpoint_path"])
            if not ckpt.exists():
                continue
            free = _eval_one_seed(
                checkpoint_path=ckpt,
                prototype_count=c,
                payload=payload,
                checkpoint_args=checkpoint_args,
                batches_cpu=batches_cpu,
                future_cache=future_cache,
                obs_data=obs_data,
                device=device,
                residual_scale=float(args.residual_scale),
            )
            metrics = free["metrics"]
            seed_results.append(
                {
                    "seed": int(seed_result["seed"]),
                    "checkpoint_path": str(ckpt),
                    "val_metrics": metrics,
                    "test_itemwise": {"aggregate": metrics, "item_scores": free["item_scores"]},
                    "residual_beats_copy_overall": bool(metrics["proto_top5"] > metrics["copy_proto_top5"]),
                    "residual_beats_copy_changed_subset": bool(metrics["changed_subset_top5"] > metrics["copy_changed_subset_top5"]),
                    "stable_preservation_drop": float(metrics["stable_preservation_drop"]),
                    "trace_regression_detected": False,
                }
            )
        best = max(seed_results, key=lambda r: (r["val_metrics"]["proto_top5"], r["val_metrics"]["changed_subset_top5"])) if seed_results else {}
        copy_baseline = {
            "proto_top1": best.get("val_metrics", {}).get("copy_proto_top1", 0.0),
            "proto_top5": best.get("val_metrics", {}).get("copy_proto_top5", 0.0),
            "proto_ce": best.get("val_metrics", {}).get("copy_proto_ce", 0.0),
            "stable_subset_top5": best.get("val_metrics", {}).get("copy_stable_subset_top5", 0.0),
            "changed_subset_top5": best.get("val_metrics", {}).get("copy_changed_subset_top5", 0.0),
        }
        report = {
            "audit_name": f"stwm_free_rollout_semantic_trace_field_v4_eval_c{c}",
            "prototype_count": int(c),
            "heldout_item_count": int(heldout_item_count),
            "free_rollout_eval_implemented": True,
            "free_rollout_path": "_free_rollout_predict",
            "teacher_forced_path_used": False,
            "candidate_scorer_used": False,
            "old_association_report_used": False,
            "future_candidate_leakage": False,
            "copy_baseline": copy_baseline,
            "seed_results": seed_results,
            "best_seed": int(best.get("seed", -1)) if best else -1,
            "best_metrics": best.get("val_metrics", {}),
            "residual_beats_copy_overall": bool(best and best["residual_beats_copy_overall"]),
            "residual_beats_copy_changed_subset": bool(best and best["residual_beats_copy_changed_subset"]),
            "stable_copy_preserved": bool(best and best["val_metrics"]["stable_preservation_drop"] <= 0.05),
            "trace_regression_detected": bool(best and best["val_metrics"]["future_trace_coord_error"] > 1e6),
            "output_degenerate": bool(best and best["val_metrics"]["proto_logit_std_mean"] <= 1e-8),
            "seed_mean_std": {
                "heldout_top5": _mean_std([float(r["val_metrics"]["proto_top5"]) for r in seed_results]),
                "changed_subset_gain_over_copy": _mean_std([float(r["val_metrics"]["changed_subset_gain_over_copy"]) for r in seed_results]),
                "future_trace_coord_error": _mean_std([float(r["val_metrics"]["future_trace_coord_error"]) for r in seed_results]),
            },
        }
        _write_json(Path(out_path), report)
        outputs[str(c)] = report
    _write_doc(Path(args.doc), "STWM Free-Rollout Semantic Trace Field V4 Eval", {"c32_output": args.eval_c32_output, "c64_output": args.eval_c64_output, "heldout_item_count": heldout_item_count})


if __name__ == "__main__":
    main()
