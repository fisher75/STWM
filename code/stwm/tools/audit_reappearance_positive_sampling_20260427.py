#!/usr/bin/env python3
from __future__ import annotations

from argparse import ArgumentParser
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import json
import os
import sys

import torch


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


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve_repo_root(value: str | None) -> Path:
    if value:
        return Path(value).expanduser().resolve()
    env_root = os.environ.get("STWM_ROOT")
    if env_root:
        return Path(env_root).expanduser().resolve()
    return Path.cwd().resolve()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def write_doc(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# STWM Reappearance Positive Sampling Audit 20260427",
        "",
        f"- checkpoint: `{payload.get('checkpoint')}`",
        f"- total_train_samples: `{payload.get('total_train_samples')}`",
        f"- samples_with_reappearance_positive: `{payload.get('samples_with_reappearance_positive')}`",
        f"- sample_positive_rate: `{payload.get('sample_positive_rate')}`",
        f"- estimated_batches_with_positive_under_batch_size_1: `{payload.get('estimated_batches_with_positive_under_batch_size_1')}`",
        f"- recommended_oversample_factor: `{payload.get('recommended_oversample_factor')}`",
        f"- positive_sampling_ready: `{payload.get('positive_sampling_ready')}`",
        f"- exact_blocking_reason: `{payload.get('exact_blocking_reason')}`",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n")


def main() -> None:
    _apply_process_title_normalization()
    args = parse_args()
    repo_root = resolve_repo_root(args.repo_root)
    code_dir = repo_root / "code"
    if str(code_dir) not in sys.path:
        sys.path.insert(0, str(code_dir))

    from stwm.tools.export_future_semantic_trace_state_20260427 import _args_from_checkpoint_payload
    from stwm.tracewm_v2_stage2.trainers import train_tracewm_stage2_smalltrain as trainer

    checkpoint = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint.exists():
        payload = {
            "generated_at_utc": now_iso(),
            "checkpoint": str(checkpoint),
            "checkpoint_exists": False,
            "positive_sampling_ready": False,
            "exact_blocking_reason": f"checkpoint not found: {checkpoint}",
        }
        write_json(Path(args.out_report), payload)
        write_doc(Path(args.out_doc), payload)
        return

    ckpt = torch.load(checkpoint, map_location="cpu")
    if not isinstance(ckpt, dict):
        raise RuntimeError(f"checkpoint payload is not a dict: {checkpoint}")
    cargs = _args_from_checkpoint_payload(ckpt, repo_root, int(args.max_samples_train))
    cargs.max_samples_train = int(args.max_samples_train)
    train_cfg = trainer.Stage2SemanticDatasetConfig(
        dataset_names=[str(x) for x in cargs.dataset_names],
        split=str(cargs.train_split),
        contract_path=str(cargs.stage2_contract_path),
        obs_len=int(cargs.obs_len),
        fut_len=int(cargs.fut_len),
        max_tokens=int(cargs.max_tokens),
        max_samples_per_dataset=int(cargs.max_samples_train),
        semantic_patch_radius=int(cargs.semantic_patch_radius),
        semantic_crop_size=int(cargs.semantic_crop_size),
        semantic_source_mainline=str(cargs.semantic_source_mainline),
        semantic_temporal_window=int(cargs.local_temporal_window),
        predecode_cache_path=str(cargs.predecode_cache_path),
        teacher_semantic_cache_path=str(cargs.teacher_semantic_cache_path),
        max_entities_per_sample=int(cargs.max_entities_per_sample),
    )
    dataset = trainer.Stage2SemanticDataset(train_cfg)
    plan = trainer._build_reappearance_positive_sampling_plan(
        dataset,
        obs_len=int(cargs.obs_len),
        fut_len=int(cargs.fut_len),
        slot_count=int(cargs.max_tokens),
        target_min_batch_ratio=float(args.target_min_batch_ratio),
    )
    samples_with_positive = int(plan.get("samples_with_reappearance_positive", 0))
    sample_positive_rate = float(plan.get("sample_positive_rate", 0.0))
    ready = bool(samples_with_positive > 0)
    exact_blocking_reason = None if ready else "no training samples with event-level reappearance positives were found"
    payload = {
        "generated_at_utc": now_iso(),
        "repo_root": str(repo_root),
        "checkpoint": str(checkpoint),
        "checkpoint_exists": True,
        "dataset_summary": dict(dataset.dataset_summary),
        "target_min_batch_ratio": float(args.target_min_batch_ratio),
        "total_train_samples": int(plan.get("total_train_samples", len(dataset))),
        "samples_with_reappearance_positive": samples_with_positive,
        "sample_positive_rate": sample_positive_rate,
        "estimated_batches_with_positive_under_batch_size_1": float(plan.get("estimated_batches_with_positive_under_batch_size_1", sample_positive_rate)),
        "recommended_oversample_factor": float(plan.get("recommended_oversample_factor", 1.0)),
        "positive_indices_count": len(plan.get("positive_indices", [])),
        "positive_sampling_ready": ready,
        "sampler_cli_available": True,
        "sampler_cli": [
            "--reappearance-positive-oversample",
            "--reappearance-positive-min-batch-ratio",
            str(float(args.target_min_batch_ratio)),
        ],
        "exact_blocking_reason": exact_blocking_reason,
    }
    write_json(Path(args.out_report), payload)
    write_doc(Path(args.out_doc), payload)


def parse_args() -> Any:
    parser = ArgumentParser(description="Audit reappearance-positive coverage for Stage2SemanticDataset training samples.")
    parser.add_argument("--repo-root", default=None)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--max-samples-train", type=int, default=256)
    parser.add_argument("--target-min-batch-ratio", type=float, default=0.30)
    parser.add_argument("--out-report", required=True)
    parser.add_argument("--out-doc", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    main()
