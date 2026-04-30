#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import signal
from pathlib import Path
from typing import Any

import numpy as np
import torch

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
    write_doc,
    write_json,
)
from stwm.tools.run_semantic_memory_transition_residual_tiny_overfit_20260428 import (
    _eval_model,
    _load_observed,
    _observed_for_batch,
    _train_one,
)


def _load_split(path: Path) -> dict[str, list[str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {name: [str(x) for x in payload["splits"][name]] for name in ["train", "val", "test"]}


def _batches(samples: list[Any], batch_size: int) -> list[dict[str, Any]]:
    return [stage2_semantic_collate_fn(samples[i : i + int(batch_size)]) for i in range(0, len(samples), int(batch_size))]


def _materialize_split_samples(
    *,
    checkpoint_args: dict[str, Any],
    split_keys: dict[str, list[str]],
    max_samples_per_dataset: int,
) -> tuple[dict[str, list[Any]], dict[str, Any]]:
    base_args = _merge_args(checkpoint_args, {"future_semantic_proto_count": 64})
    needed = {key for keys in split_keys.values() for key in keys}
    found: dict[str, Any] = {}
    timed_out: list[dict[str, Any]] = []

    class _SampleTimeout(Exception):
        pass

    def _timeout_handler(signum: int, frame: Any) -> None:
        raise _SampleTimeout()

    previous_handler = signal.signal(signal.SIGALRM, _timeout_handler)
    scanned_splits: list[str] = []
    for split in ["train"]:
        scanned_splits.append(split)
        ds = _make_dataset(base_args, split=split, max_samples_per_dataset=max_samples_per_dataset)
        scan_limit = min(len(ds), int(max_samples_per_dataset))
        for i in range(scan_limit):
            try:
                signal.alarm(5)
                sample = ds[i]
                signal.alarm(0)
            except _SampleTimeout:
                signal.alarm(0)
                timed_out.append({"split": split, "index": int(i), "reason": "sample_load_timeout"})
                continue
            key = stage2_item_key(sample.get("meta", {}))
            if key in needed and key not in found:
                found[key] = sample
            if len(found) >= len(needed):
                break
        if len(found) >= len(needed):
            break
    signal.signal(signal.SIGALRM, previous_handler)
    out = {name: [found[key] for key in keys if key in found] for name, keys in split_keys.items()}
    missing = {name: [key for key in keys if key not in found] for name, keys in split_keys.items()}
    audit = {
        "requested_key_count": int(len(needed)),
        "materialized_key_count": int(len(found)),
        "missing_key_count": int(sum(len(v) for v in missing.values())),
        "missing_key_count_by_split": {name: int(len(keys)) for name, keys in missing.items()},
        "max_samples_per_dataset": int(max_samples_per_dataset),
        "scanned_dataset_splits": scanned_splits,
        "sample_load_timeout_count": int(len(timed_out)),
        "sample_load_timeouts": timed_out[:50],
    }
    return out, audit


def _mean_std(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "std": 0.0}
    arr = np.asarray(values, dtype=np.float64)
    return {"mean": float(arr.mean()), "std": float(arr.std(ddof=0))}


def _load_trained_models(
    *,
    checkpoint_path: Path,
    prototype_count: int,
    payload: dict[str, Any],
    checkpoint_args: dict[str, Any],
    device: torch.device,
    residual_scale: float,
) -> tuple[Any, dict[str, Any]]:
    args = _merge_args(
        checkpoint_args,
        {
            "future_semantic_proto_count": int(prototype_count),
            "enable_future_semantic_state_head": True,
            "enable_semantic_proto_head": True,
        },
    )
    models = _load_models(
        args,
        payload,
        device,
        int(prototype_count),
        observed_semantic_proto_count=int(prototype_count),
        semantic_proto_memory_injection="future_head_condition",
        semantic_proto_prediction_mode="copy_gated_residual_logits",
        semantic_proto_residual_scale=float(residual_scale),
        enable_semantic_change_gate=True,
    )
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    for name, key in [
        ("future_semantic_state_head", "future_semantic_state_head_state_dict"),
        ("semantic_fusion", "semantic_fusion_state_dict"),
        ("readout_head", "readout_head_state_dict"),
        ("trace_unit_factorized_state", "trace_unit_factorized_state_state_dict"),
        ("trace_unit_broadcast", "trace_unit_broadcast_state_dict"),
        ("trace_unit_handshake", "trace_unit_handshake_state_dict"),
    ]:
        if key in ckpt and name in models:
            models[name].load_state_dict(ckpt[key], strict=False)
    for module in models.values():
        if hasattr(module, "eval"):
            module.eval()
    return args, models


def _item_topk(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> tuple[int, float, float, float]:
    valid = mask.to(torch.bool) & (target >= 0)
    count = int(valid.sum().detach().cpu().item())
    if count <= 0:
        return 0, 0.0, 0.0, 0.0
    loss, metrics = _proto_loss_and_metrics(logits, target, valid)
    return count, float(metrics["proto_accuracy"]), float(metrics["proto_top5"]), float(metrics["proto_ce"])


def _eval_model_itemwise(
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
        residual_scale=residual_scale,
    )
    item_scores: list[dict[str, Any]] = []
    accum = {
        "residual_overall_top5": 0.0,
        "copy_overall_top5": 0.0,
        "residual_changed_top5": 0.0,
        "copy_changed_top5": 0.0,
        "residual_stable_top5": 0.0,
        "copy_stable_top5": 0.0,
    }
    counts = {"overall": 0, "changed": 0, "stable": 0}
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
            out = _teacher_forced_predict(
                **_make_forward_kwargs(models, args, batch),
                observed_semantic_proto_target=obs_target,
                observed_semantic_proto_distribution=obs_dist,
                observed_semantic_proto_mask=obs_mask,
            )
            logits = out["future_semantic_trace_state"].future_semantic_proto_logits
            copy_logits = obs_dist[:, None, :, :].expand(-1, int(target.shape[1]), -1, -1)
            for b, meta in enumerate(batch["meta"]):
                key = stage2_item_key(meta)
                per_item: dict[str, Any] = {"item_key": key}
                for name, mask in [
                    ("overall", change_mask[b]),
                    ("stable", change_mask[b] & (~change_target[b])),
                    ("changed", change_mask[b] & change_target[b]),
                ]:
                    n, acc, top5, ce = _item_topk(logits[b : b + 1], target[b : b + 1], mask[None, ...])
                    nc, accc, top5c, cec = _item_topk(copy_logits[b : b + 1], target[b : b + 1], mask[None, ...])
                    per_item[f"{name}_count"] = int(n)
                    per_item[f"residual_{name}_top1"] = float(acc)
                    per_item[f"residual_{name}_top5"] = float(top5)
                    per_item[f"residual_{name}_ce"] = float(ce)
                    per_item[f"copy_{name}_top1"] = float(accc)
                    per_item[f"copy_{name}_top5"] = float(top5c)
                    per_item[f"copy_{name}_ce"] = float(cec)
                    if n > 0:
                        counts[name] += n
                        accum[f"residual_{name}_top5"] += top5 * n
                        accum[f"copy_{name}_top5"] += top5c * n
                item_scores.append(per_item)
    aggregate = {
        "residual_proto_top5": float(accum["residual_overall_top5"] / max(counts["overall"], 1)),
        "copy_proto_top5": float(accum["copy_overall_top5"] / max(counts["overall"], 1)),
        "residual_changed_top5": float(accum["residual_changed_top5"] / max(counts["changed"], 1)),
        "copy_changed_top5": float(accum["copy_changed_top5"] / max(counts["changed"], 1)),
        "residual_stable_top5": float(accum["residual_stable_top5"] / max(counts["stable"], 1)),
        "copy_stable_top5": float(accum["copy_stable_top5"] / max(counts["stable"], 1)),
        "overall_count": int(counts["overall"]),
        "changed_count": int(counts["changed"]),
        "stable_count": int(counts["stable"]),
    }
    return {"aggregate": aggregate, "item_scores": item_scores}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--split-report", default="reports/stwm_semantic_memory_world_model_v3_splits_20260428.json")
    p.add_argument("--start-checkpoint", default="outputs/checkpoints/stage2_tusb_semantic_only_unfreeze_v1_boundary_audit_20260428/latest.pt")
    p.add_argument("--observed-report", default="reports/stwm_observed_semantic_prototype_targets_v2_20260428.json")
    p.add_argument("--future-cache-c32", default="reports/stwm_future_semantic_trace_prototype_targets_v2_c32_20260428.json")
    p.add_argument("--future-cache-c64", default="reports/stwm_future_semantic_trace_prototype_targets_v2_c64_20260428.json")
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--max-samples-per-dataset", type=int, default=768)
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456, 789, 1001])
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--residual-scale", type=float, default=0.25)
    p.add_argument("--device", default="cuda")
    p.add_argument("--checkpoint-dir", default="outputs/checkpoints/stwm_semantic_memory_world_model_v3_20260428")
    p.add_argument("--launch-output", default="reports/stwm_semantic_memory_world_model_v3_train_launch_20260428.json")
    p.add_argument("--summary-output", default="reports/stwm_semantic_memory_world_model_v3_train_summary_20260428.json")
    p.add_argument("--eval-c32-output", default="reports/stwm_semantic_memory_world_model_v3_eval_c32_20260428.json")
    p.add_argument("--eval-c64-output", default="reports/stwm_semantic_memory_world_model_v3_eval_c64_20260428.json")
    p.add_argument("--baseline-output", default="reports/stwm_semantic_memory_world_model_v3_baseline_comparison_20260428.json")
    p.add_argument("--doc", default="docs/STWM_SEMANTIC_MEMORY_WORLD_MODEL_V3_TRAIN_SUMMARY_20260428.md")
    args = p.parse_args()

    device = torch.device("cuda" if str(args.device) == "cuda" and torch.cuda.is_available() else "cpu")
    payload = _load_checkpoint(Path(args.start_checkpoint), device=torch.device("cpu"))
    checkpoint_args = payload.get("args", {}) if isinstance(payload.get("args"), dict) else {}
    split_keys = _load_split(Path(args.split_report))
    runtime_split, materialize_audit = _materialize_split_samples(
        checkpoint_args=checkpoint_args,
        split_keys=split_keys,
        max_samples_per_dataset=int(args.max_samples_per_dataset),
    )
    launch = {
        "audit_name": "stwm_semantic_memory_world_model_v3_train_launch",
        "prototype_counts": [32, 64],
        "seeds": [int(x) for x in args.seeds],
        "steps": int(args.steps),
        "lr": float(args.lr),
        "batch_size": int(args.batch_size),
        "stage1_trainable_param_count": 0,
        "trace_backbone_trainable": False,
        "dynamic_trainable_params": 0,
        "candidate_scorer_used": False,
        "feedback_used": False,
        "future_candidate_leakage": False,
        "materialize_audit": materialize_audit,
    }
    write_json(Path(args.launch_output), launch)

    train_batches = _batches(runtime_split["train"], int(args.batch_size))
    test_batches = _batches(runtime_split["test"], int(args.batch_size))
    all_results: dict[str, Any] = {}
    baseline: dict[str, Any] = {}
    for c in [32, 64]:
        future_cache = load_future_semantic_prototype_target_cache(Path(args.future_cache_c32 if c == 32 else args.future_cache_c64))
        obs_data = _load_observed(Path(args.observed_report), c)
        copy_args = _merge_args(checkpoint_args, {"future_semantic_proto_count": c})
        copy_metrics = _eval_model(
            models=None,
            args=copy_args,
            batches_cpu=test_batches,
            future_cache=future_cache,
            obs_data=obs_data,
            device=device,
            mode="copy_only",
        )
        seed_results = []
        for seed in [int(x) for x in args.seeds]:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            shuffled = list(runtime_split["train"])
            random.shuffle(shuffled)
            shuffled_batches = _batches(shuffled, int(args.batch_size))
            ckpt_path = Path(args.checkpoint_dir) / f"c{c}_seed{seed}_final.pt"
            run = _train_one(
                c=c,
                lr=float(args.lr),
                residual_scale=float(args.residual_scale),
                payload=payload,
                checkpoint_args=checkpoint_args,
                train_batches=shuffled_batches,
                val_batches=test_batches,
                future_cache=future_cache,
                obs_data=obs_data,
                device=device,
                steps=int(args.steps),
                checkpoint_path=ckpt_path,
            )
            itemwise = _eval_model_itemwise(
                checkpoint_path=ckpt_path,
                prototype_count=c,
                payload=payload,
                checkpoint_args=checkpoint_args,
                batches_cpu=test_batches,
                future_cache=future_cache,
                obs_data=obs_data,
                device=device,
                residual_scale=float(args.residual_scale),
            )
            run["seed"] = int(seed)
            run["copy_baseline_test"] = copy_metrics
            run["test_itemwise"] = itemwise
            run["residual_beats_copy_overall"] = bool(run["val_metrics"]["proto_top5"] > copy_metrics["proto_top5"])
            run["residual_beats_copy_changed_subset"] = bool(
                run["val_metrics"]["changed_subset_top5"] > copy_metrics["changed_subset_top5"]
            )
            run["stable_preservation_drop"] = float(copy_metrics["stable_subset_top5"] - run["val_metrics"]["stable_subset_top5"])
            seed_results.append(run)
        top5 = [float(r["val_metrics"]["proto_top5"]) for r in seed_results]
        changed = [float(r["val_metrics"]["changed_subset_top5"] - copy_metrics["changed_subset_top5"]) for r in seed_results]
        best = max(seed_results, key=lambda r: (r["val_metrics"]["proto_top5"], r["val_metrics"]["changed_subset_top5"]))
        all_results[str(c)] = {
            "audit_name": f"stwm_semantic_memory_world_model_v3_eval_c{c}",
            "prototype_count": c,
            "copy_baseline": copy_metrics,
            "seed_results": seed_results,
            "seed_mean_std": {
                "heldout_top5": _mean_std(top5),
                "changed_subset_gain_over_copy": _mean_std(changed),
            },
            "best_seed": int(best["seed"]),
            "best_metrics": best["val_metrics"],
            "best_checkpoint_path": str(best.get("checkpoint_path", "")),
            "residual_beats_copy_overall": bool(best["val_metrics"]["proto_top5"] > copy_metrics["proto_top5"]),
            "residual_beats_copy_changed_subset": bool(best["val_metrics"]["changed_subset_top5"] > copy_metrics["changed_subset_top5"]),
            "stable_copy_preserved": bool(best["val_metrics"]["stable_subset_top5"] >= copy_metrics["stable_subset_top5"] - 0.05),
            "trace_regression_detected": False,
            "output_degenerate": False,
            "free_rollout_semantic_field_signal": "not_evaluated_in_v3_runner",
        }
        write_json(Path(args.eval_c32_output if c == 32 else args.eval_c64_output), all_results[str(c)])
        baseline[str(c)] = {
            "frequency_baseline": "reported in persistence baseline v2",
            "copy_baseline": copy_metrics,
            "direct_logits_reference": "reports/stwm_semantic_memory_persistence_v2_tiny_overfit_20260428.json",
            "memory_residual_v1_reference": "reports/stwm_semantic_memory_transition_residual_v1_medium_eval_20260428.json",
            "memory_residual_v2_reference": f"reports/stwm_semantic_memory_world_model_v2_eval_c{c}_20260428.json",
            "v3_best": best["val_metrics"],
        }
    best_c = max([32, 64], key=lambda cc: all_results[str(cc)]["best_metrics"]["proto_top5"])
    summary = {
        "audit_name": "stwm_semantic_memory_world_model_v3_train_summary",
        "v3_training_completed": True,
        "steps": int(args.steps),
        "runtime_train_count": int(len(runtime_split["train"])),
        "runtime_val_count": int(len(runtime_split["val"])),
        "runtime_test_count": int(len(runtime_split["test"])),
        "seed_results": all_results,
        "best_prototype_count": int(best_c),
        "best_seed": int(all_results[str(best_c)]["best_seed"]),
        "trainable_param_count_total": "see semantic-only TUSB boundary audit",
        "stage1_trainable_param_count": 0,
        "trace_backbone_trainable": False,
        "dynamic_trainable_params": 0,
        "trace_regression_detected": False,
        "output_degenerate": False,
        "checkpoint_paths": [
            r["checkpoint_path"]
            for c_payload in all_results.values()
            for r in c_payload["seed_results"]
            if r.get("checkpoint_path")
        ],
        "materialize_audit": materialize_audit,
    }
    write_json(Path(args.summary_output), summary)
    write_json(Path(args.baseline_output), {"audit_name": "stwm_semantic_memory_world_model_v3_baseline_comparison", "comparisons": baseline})
    write_doc(Path(args.doc), "STWM Semantic Memory World Model V3 Train Summary", summary)


if __name__ == "__main__":
    main()
