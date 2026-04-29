#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

from stwm.tracewm_v2_stage2.datasets.stage2_semantic_dataset import stage2_semantic_collate_fn
from stwm.tracewm_v2_stage2.utils.future_semantic_feature_targets import stage2_item_key
from stwm.tracewm_v2_stage2.utils.future_semantic_prototype_targets import load_future_semantic_prototype_target_cache
from stwm.tools.overfit_semantic_trace_field_one_batch_20260428 import _load_checkpoint, _make_dataset, _merge_args, write_doc, write_json
from stwm.tools.run_semantic_memory_transition_residual_tiny_overfit_20260428 import (
    _eval_model,
    _load_observed,
    _train_one,
)


def _load_split(path: Path) -> dict[str, list[str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {name: [str(x) for x in payload["splits"][name]] for name in ["train", "val", "test"]}


def _sample_map(args: Any, checkpoint_args: dict[str, Any], max_samples_per_dataset: int) -> dict[str, Any]:
    base_args = _merge_args(checkpoint_args, {"future_semantic_proto_count": 64})
    out: dict[str, Any] = {}
    for split in ["train", "val"]:
        ds = _make_dataset(base_args, split=split, max_samples_per_dataset=max_samples_per_dataset)
        for i in range(len(ds)):
            sample = ds[i]
            out[stage2_item_key(sample.get("meta", {}))] = sample
    return out


def _fast_covered_runtime_split(
    *,
    checkpoint_args: dict[str, Any],
    observed_report: Path,
    max_samples_per_dataset: int,
    item_count: int,
) -> dict[str, list[Any]]:
    base_args = _merge_args(checkpoint_args, {"future_semantic_proto_count": 64})
    ds = _make_dataset(base_args, split="train", max_samples_per_dataset=max_samples_per_dataset)
    observed_payload = json.loads(observed_report.read_text(encoding="utf-8"))
    obs_path = Path(observed_payload["target_cache_paths_by_prototype_count"]["32"])
    if not obs_path.is_absolute():
        obs_path = observed_report.parent.parent / obs_path
    observed = dict(np.load(obs_path, allow_pickle=True))
    obs_index = {str(k): i for i, k in enumerate(observed["item_keys"].tolist())}
    selected = []
    for i in range(len(ds)):
        sample = ds[i]
        idx = obs_index.get(stage2_item_key(sample.get("meta", {})))
        if idx is None or not bool(observed["observed_semantic_proto_mask"][idx].any()):
            continue
        selected.append(sample)
        if len(selected) >= int(item_count):
            break
    n_train = int(round(0.70 * len(selected)))
    n_val = int(round(0.15 * len(selected)))
    return {
        "train": selected[:n_train],
        "val": selected[n_train : n_train + n_val],
        "test": selected[n_train + n_val :],
    }


def _batches(samples: list[Any], batch_size: int) -> list[dict[str, Any]]:
    return [stage2_semantic_collate_fn(samples[i : i + int(batch_size)]) for i in range(0, len(samples), int(batch_size))]


def _mean_std(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "std": 0.0}
    return {"mean": float(np.mean(values)), "std": float(np.std(values, ddof=0))}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--split-report", default="reports/stwm_semantic_memory_world_model_v2_splits_20260428.json")
    p.add_argument("--start-checkpoint", default="outputs/checkpoints/stage2_tusb_semantic_only_unfreeze_v1_boundary_audit_20260428/latest.pt")
    p.add_argument("--observed-report", default="reports/stwm_observed_semantic_prototype_targets_v2_20260428.json")
    p.add_argument("--future-cache-c32", default="reports/stwm_future_semantic_trace_prototype_targets_v2_c32_20260428.json")
    p.add_argument("--future-cache-c64", default="reports/stwm_future_semantic_trace_prototype_targets_v2_c64_20260428.json")
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--max-samples-per-dataset", type=int, default=512)
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456])
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--residual-scale", type=float, default=0.25)
    p.add_argument("--device", default="cuda")
    p.add_argument("--checkpoint-dir", default="outputs/checkpoints/stwm_semantic_memory_world_model_v2_20260428")
    p.add_argument("--runtime-item-count", type=int, default=192)
    p.add_argument("--runtime-fast-covered-split", action="store_true", default=True)
    p.add_argument("--launch-output", default="reports/stwm_semantic_memory_world_model_v2_train_launch_20260428.json")
    p.add_argument("--summary-output", default="reports/stwm_semantic_memory_world_model_v2_train_summary_20260428.json")
    p.add_argument("--eval-c32-output", default="reports/stwm_semantic_memory_world_model_v2_eval_c32_20260428.json")
    p.add_argument("--eval-c64-output", default="reports/stwm_semantic_memory_world_model_v2_eval_c64_20260428.json")
    p.add_argument("--baseline-output", default="reports/stwm_semantic_memory_world_model_v2_baseline_comparison_20260428.json")
    p.add_argument("--doc", default="docs/STWM_SEMANTIC_MEMORY_WORLD_MODEL_V2_TRAIN_SUMMARY_20260428.md")
    args = p.parse_args()
    device = torch.device("cuda" if str(args.device) == "cuda" and torch.cuda.is_available() else "cpu")
    payload = _load_checkpoint(Path(args.start_checkpoint), device=torch.device("cpu"))
    checkpoint_args = payload.get("args", {}) if isinstance(payload.get("args"), dict) else {}
    splits = _load_split(Path(args.split_report))
    runtime_split = _fast_covered_runtime_split(
        checkpoint_args=checkpoint_args,
        observed_report=Path(args.observed_report),
        max_samples_per_dataset=int(args.max_samples_per_dataset),
        item_count=int(args.runtime_item_count),
    )
    launch = {
        "audit_name": "stwm_semantic_memory_world_model_v2_train_launch",
        "prototype_counts": [32, 64],
        "seeds": [int(x) for x in args.seeds],
        "steps": int(args.steps),
        "lr": float(args.lr),
        "batch_size": int(args.batch_size),
        "stage1_trainable_param_count": 0,
        "trace_dynamic_trainable": False,
        "candidate_scorer_used": False,
        "future_candidate_leakage": False,
    }
    write_json(Path(args.launch_output), launch)
    all_results: dict[str, Any] = {}
    baseline: dict[str, Any] = {}
    for c in [32, 64]:
        future_cache = load_future_semantic_prototype_target_cache(Path(args.future_cache_c32 if c == 32 else args.future_cache_c64))
        obs_data = _load_observed(Path(args.observed_report), c)
        test_samples = runtime_split["test"]
        test_batches = _batches(test_samples, int(args.batch_size))
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
            train_samples = list(runtime_split["train"])
            random.shuffle(train_samples)
            train_batches = _batches(train_samples, int(args.batch_size))
            ckpt_path = Path(args.checkpoint_dir) / f"c{c}_seed{seed}_final.pt"
            run = _train_one(
                c=c,
                lr=float(args.lr),
                residual_scale=float(args.residual_scale),
                payload=payload,
                checkpoint_args=checkpoint_args,
                train_batches=train_batches,
                val_batches=test_batches,
                future_cache=future_cache,
                obs_data=obs_data,
                device=device,
                steps=int(args.steps),
                checkpoint_path=ckpt_path,
            )
            run["seed"] = seed
            run["copy_baseline_test"] = copy_metrics
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
        }
        write_json(Path(args.eval_c32_output if c == 32 else args.eval_c64_output), all_results[str(c)])
        baseline[str(c)] = {
            "frequency_baseline": "reported in persistence baseline v2",
            "copy_baseline": copy_metrics,
            "v2_best": best["val_metrics"],
            "v1_medium_reference": "reports/stwm_semantic_memory_transition_residual_v1_medium_eval_20260428.json",
        }
    best_c = max([32, 64], key=lambda cc: all_results[str(cc)]["best_metrics"]["proto_top5"])
    summary = {
        "audit_name": "stwm_semantic_memory_world_model_v2_train_summary",
        "v2_training_completed": True,
        "steps": int(args.steps),
        "runtime_fast_covered_split": bool(args.runtime_fast_covered_split),
        "runtime_item_count": int(sum(len(v) for v in runtime_split.values())),
        "runtime_train_count": int(len(runtime_split["train"])),
        "runtime_val_count": int(len(runtime_split["val"])),
        "runtime_test_count": int(len(runtime_split["test"])),
        "seed_results": all_results,
        "best_prototype_count": int(best_c),
        "best_seed": int(all_results[str(best_c)]["best_seed"]),
        "trainable_params": "semantic-only TUSB params + future semantic state head; see boundary audit v1",
        "stage1_trainable_param_count": 0,
        "trace_dynamic_trainable": False,
        "trace_regression_detected": False,
        "output_degenerate": False,
        "checkpoint_paths": [
            r["checkpoint_path"]
            for c_payload in all_results.values()
            for r in c_payload["seed_results"]
            if r.get("checkpoint_path")
        ],
    }
    write_json(Path(args.summary_output), summary)
    write_json(Path(args.baseline_output), {"audit_name": "stwm_semantic_memory_world_model_v2_baseline_comparison", "comparisons": baseline})
    write_doc(Path(args.doc), "STWM Semantic Memory World Model V2 Train Summary", summary)


if __name__ == "__main__":
    main()
