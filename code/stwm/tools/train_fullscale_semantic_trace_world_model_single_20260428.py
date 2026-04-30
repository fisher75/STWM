#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

from stwm.tracewm_v2_stage2.utils.future_semantic_prototype_targets import load_future_semantic_prototype_target_cache
from stwm.tools.overfit_semantic_trace_field_one_batch_20260428 import _load_checkpoint, _merge_args, write_doc, write_json
from stwm.tools.run_semantic_memory_transition_residual_tiny_overfit_20260428 import _load_observed, _train_one


def _apply_process_title_normalization() -> None:
    mode = str(os.environ.get("STWM_PROC_TITLE_MODE", "generic")).strip().lower()
    if mode == "off":
        return
    title = str(os.environ.get("STWM_PROC_TITLE", "python")).strip() or "python"
    try:
        import setproctitle  # type: ignore

        setproctitle.setproctitle(title)
    except Exception:
        pass


def _load_batches_from_report(report_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    cache_path = Path(str(payload.get("batch_cache_path") or ""))
    if not cache_path.is_absolute():
        cache_path = report_path.parent.parent / cache_path
    cache = torch.load(cache_path, map_location="cpu")
    return list(cache.get("batches", [])), payload


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--prototype-count", type=int, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--train-cache-report", required=True)
    p.add_argument("--val-cache-report", required=True)
    p.add_argument("--start-checkpoint", default="outputs/checkpoints/stage2_tusb_semantic_only_unfreeze_v1_boundary_audit_20260428/latest.pt")
    p.add_argument("--observed-report", required=True)
    p.add_argument("--future-cache-report", required=True)
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--residual-scale", type=float, default=0.25)
    p.add_argument("--device", default="cuda")
    p.add_argument("--checkpoint-output", required=True)
    p.add_argument("--summary-output", required=True)
    p.add_argument("--doc", required=True)
    p.add_argument("--torch-num-threads", type=int, default=int(os.environ.get("STWM_TORCH_NUM_THREADS", "16")))
    args = p.parse_args()

    _apply_process_title_normalization()
    if int(args.torch_num_threads) > 0:
        torch.set_num_threads(int(args.torch_num_threads))
        torch.set_num_interop_threads(max(1, min(4, int(args.torch_num_threads))))
    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))

    device = torch.device("cuda" if str(args.device) == "cuda" and torch.cuda.is_available() else "cpu")
    payload = _load_checkpoint(Path(args.start_checkpoint), device=torch.device("cpu"))
    checkpoint_args = payload.get("args", {}) if isinstance(payload.get("args"), dict) else {}
    train_batches, train_audit = _load_batches_from_report(Path(args.train_cache_report))
    val_batches, val_audit = _load_batches_from_report(Path(args.val_cache_report))
    shuffled = list(train_batches)
    random.shuffle(shuffled)
    future_cache = load_future_semantic_prototype_target_cache(Path(args.future_cache_report))
    obs_data = _load_observed(Path(args.observed_report), int(args.prototype_count))
    run = _train_one(
        c=int(args.prototype_count),
        lr=float(args.lr),
        residual_scale=float(args.residual_scale),
        payload=payload,
        checkpoint_args=checkpoint_args,
        train_batches=shuffled,
        val_batches=val_batches,
        future_cache=future_cache,
        obs_data=obs_data,
        device=device,
        steps=int(args.steps),
        checkpoint_path=Path(args.checkpoint_output),
    )
    summary = {
        "audit_name": "stwm_fullscale_semantic_trace_world_model_v1_single_train",
        "prototype_count": int(args.prototype_count),
        "seed": int(args.seed),
        "steps": int(args.steps),
        "lr": float(args.lr),
        "residual_scale": float(args.residual_scale),
        "device": str(device),
        "cuda_visible_devices": str(os.environ.get("CUDA_VISIBLE_DEVICES", "")),
        "train_batch_count": int(len(train_batches)),
        "val_batch_count": int(len(val_batches)),
        "train_materialization": train_audit,
        "val_materialization": val_audit,
        "stage1_trainable_param_count": 0,
        "trace_backbone_trainable": False,
        "dynamic_trainable_params": 0,
        "candidate_scorer_used": False,
        "future_candidate_leakage": False,
        **run,
    }
    write_json(Path(args.summary_output), summary)
    write_doc(Path(args.doc), "STWM Fullscale Semantic Trace World Model V1 Single Train", summary)


if __name__ == "__main__":
    main()
