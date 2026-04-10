#!/usr/bin/env python3
from __future__ import annotations

from argparse import ArgumentParser
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple
import json
import os
import shlex
import subprocess
import sys
import time
import traceback

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Subset

from stwm.infra.gpu_lease import acquire_lease, release_lease
from stwm.infra.gpu_selector import select_single_gpu
from stwm.tracewm_v2_stage2.datasets.stage2_semantic_dataset import (
    Stage2SemanticDataset,
    Stage2SemanticDatasetConfig,
)
from stwm.tools.run_tracewm_stage2_ljs_semantic_diagnosis_and_rescue_20260410 import (
    METRIC_KEYS,
    _evaluate_loaded_stage2,
    _f,
    _load_stage1_model,
    _load_stage2_modules,
    _make_dataset,
    _mean_std,
    _read_json,
    _write_json,
    _write_md,
)


WORK_ROOT = Path("/home/chen034/workspace/stwm")
SESSION = "tracewm_stage2_semantic_objective_redesign_v1_20260410"
DATE_TAG = "20260410"
CACHE_LIMIT_TRAIN_PER_DATASET = 128
CACHE_LIMIT_VAL_PER_DATASET = 64
PILOT_EXTRA_STEPS = 300
PILOT_BATCH_SIZE = 8
PILOT_EVAL_INTERVAL = 100
PILOT_SAVE_EVERY = 100
PILOT_EVAL_MAX_BATCHES = 32


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _python_bin_default() -> str:
    preferred = Path("/home/chen034/miniconda3/envs/stwm/bin/python")
    return str(preferred) if preferred.exists() else sys.executable


def _write_jsonl(path: str | Path, rows: Iterable[Dict[str, Any]]) -> int:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with p.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")
            count += 1
    return count


def _run_cmd(cmd: List[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, text=True, capture_output=True, **kwargs)


def _clip_cache_candidates() -> List[Path]:
    return [
        Path("/home/chen034/.cache/clip/ViT-B-32.pt"),
        Path("/raid/chen034/.cache/clip/ViT-B-32.pt"),
        WORK_ROOT / "models/checkpoints/clip/ViT-B-32.pt",
    ]


def _local_clip_cache_dir() -> Path:
    for p in _clip_cache_candidates():
        if p.exists():
            return p.parent
    return Path("/home/chen034/.cache/clip")


def _load_clip_cpu_probe() -> Dict[str, Any]:
    try:
        import clip  # type: ignore

        cache_file = next((p for p in _clip_cache_candidates() if p.exists()), None)
        if cache_file is None:
            return {"usable": False, "reason": "clip package is importable but no local ViT-B-32.pt cache was found"}
        model, _preprocess = clip.load("ViT-B/32", device="cpu", download_root=str(cache_file.parent))
        visual_out = int(getattr(getattr(model, "visual", None), "output_dim", 0) or 0)
        del model
        return {
            "usable": True,
            "package": "clip",
            "model": "ViT-B/32",
            "local_weight": str(cache_file),
            "feature_dim": int(visual_out or 512),
        }
    except Exception as exc:
        return {"usable": False, "reason": str(exc)}


def _backend_audit_payload() -> Dict[str, Any]:
    backends: List[Dict[str, Any]] = []
    for module_name in ["open_clip", "clip", "transformers", "timm", "segment_anything", "dinov2"]:
        try:
            module = __import__(module_name)
            backends.append({"name": module_name, "importable": True, "module_file": str(getattr(module, "__file__", ""))})
        except Exception as exc:
            backends.append({"name": module_name, "importable": False, "reason": str(exc)})

    sam2_probe: Dict[str, Any] = {
        "name": "sam2",
        "importable": False,
        "reason": "not checked",
        "local_weights": [
            str(WORK_ROOT / "models/checkpoints/sam2/sam2.1_hiera_base_plus.pt"),
            str(WORK_ROOT / "models/checkpoints/sam2/sam2.1_hiera_large.pt"),
        ],
    }
    try:
        sys.path.insert(0, str(WORK_ROOT / "third_party/sam2"))
        module = __import__("sam2")
        sam2_probe.update({"importable": True, "module_file": str(getattr(module, "__file__", "")), "reason": ""})
    except Exception as exc:
        sam2_probe.update({"importable": False, "reason": str(exc)})
    backends.append(sam2_probe)

    clip_probe = _load_clip_cpu_probe()
    verified: List[Dict[str, Any]] = []
    if bool(clip_probe.get("usable", False)):
        verified.append(
            {
                "name": "local_clip_vit_b32_mask_crop_visual_teacher",
                "usable": True,
                "backend": "clip",
                "model": "ViT-B/32",
                "feature_dim": int(clip_probe.get("feature_dim", 512)),
                "local_weight": str(clip_probe.get("local_weight", "")),
                "usage_boundary": "bootstrap/pseudo-label/cache/alignment target only; never the mainline semantic token source",
            }
        )

    chosen = verified[0]["name"] if verified else "none"
    return {
        "generated_at_utc": now_iso(),
        "audit_type": "stage2_real_bootstrap_backend_audit",
        "teacher_usage_policy": {
            "teacher_as_mainline_semantic_source": False,
            "allowed_uses": ["bootstrap", "pseudo_label", "cache", "semantic_alignment_target"],
            "forbidden_uses": ["replace_trainable_crop_semantic_encoder", "direct_teacher_semantic_token_input"],
        },
        "available_backends": backends,
        "verified_usable_backends": verified,
        "chosen_bootstrap_backend": chosen,
        "chosen_backend_usable": bool(verified),
        "chosen_backend_feature_dim": int(verified[0]["feature_dim"]) if verified else 0,
        "why_crop_stats_pseudo_target_cache_is_insufficient": (
            "crop_stats_pseudo_target_cache is a 10-d handcrafted color/area/foreground-stat target. "
            "It is useful as a fallback sanity signal but is not a real visual semantic teacher representation."
        ),
        "blocking_reason_if_no_real_backend": "" if verified else "No local importable visual teacher with local cached weights could be verified without downloading.",
        "minimal_missing_items_if_blocked": [] if verified else ["one verified local CLIP/SigLIP/DINO/SAM region-feature interface and local weights"],
    }


def write_backend_audit(audit_json: Path, audit_md: Path) -> Dict[str, Any]:
    payload = _backend_audit_payload()
    _write_json(audit_json, payload)
    lines = [
        "# Stage2 Real Bootstrap Backend Audit",
        "",
        f"- generated_at_utc: {payload['generated_at_utc']}",
        f"- chosen_bootstrap_backend: {payload['chosen_bootstrap_backend']}",
        f"- chosen_backend_usable: {payload['chosen_backend_usable']}",
        f"- chosen_backend_feature_dim: {payload['chosen_backend_feature_dim']}",
        f"- teacher_as_mainline_semantic_source: {payload['teacher_usage_policy']['teacher_as_mainline_semantic_source']}",
        f"- crop_stats_insufficient_reason: {payload['why_crop_stats_pseudo_target_cache_is_insufficient']}",
        f"- blocking_reason_if_no_real_backend: {payload['blocking_reason_if_no_real_backend']}",
        "",
        "## Verified Usable Backends",
    ]
    for backend in payload.get("verified_usable_backends", []):
        lines.append(f"- {backend.get('name', '')}: feature_dim={backend.get('feature_dim', 0)}, local_weight={backend.get('local_weight', '')}")
    _write_md(audit_md, lines)
    return payload


def write_protocol_doc(path: Path) -> None:
    lines = [
        "# Stage2 Semantic Objective Redesign V1 Protocol",
        "",
        f"- generated_at_utc: {now_iso()}",
        "- main_task: future trace / future state generation",
        "- full_video_reconstruction_is_main_task: false",
        "- stage1_status: frozen trace/state backbone",
        "- stage2_mainline_semantic_source: trainable object-region/mask-crop semantic encoder",
        "- teacher_usage: bootstrap / pseudo-label / cache / alignment target only",
        "- teacher_as_mainline_semantic_source: false",
        "- this_round_is_not: teacher mainline, full video reconstruction, paper framing, Stage1 rollback, DDP retrofit, batch/lr sweep",
        "",
        "## What This Round Fixes",
        "- semantic objective: add alignment and query/persistence auxiliary objectives without rewriting frozen Stage1 dynamics",
        "- bootstrap target quality: replace crop-stats pseudo bootstrap with verified local visual teacher cache when available",
        "- semantic-hard supervision: add bounded sample/loss reweighting for hard semantic/persistence clips",
    ]
    _write_md(path, lines)


def _dataset(
    dataset_names: List[str],
    split: str,
    contract_path: str,
    max_samples: int,
) -> Stage2SemanticDataset:
    return Stage2SemanticDataset(
        Stage2SemanticDatasetConfig(
            dataset_names=list(dataset_names),
            split=str(split),
            contract_path=str(contract_path),
            obs_len=8,
            fut_len=8,
            max_tokens=64,
            max_samples_per_dataset=int(max_samples),
            semantic_crop_size=64,
            semantic_source_mainline="crop_visual_encoder",
        )
    )


def _dataset_counts(dataset_names: List[str], split: str, contract_path: str, max_samples: int) -> Dict[str, int]:
    ds = _dataset(dataset_names, split, contract_path, max_samples=max_samples)
    return {str(k): int(v.get("sample_count", 0) or 0) for k, v in ds.dataset_summary.items()}


def _sample_to_clip_image(sample: Dict[str, Any]) -> Image.Image:
    rgb = sample["semantic_rgb_crop"][0].detach().cpu().float().clamp(0.0, 1.0)
    mask = sample["semantic_mask_crop"][0].detach().cpu().float().clamp(0.0, 1.0)
    if mask.ndim == 3:
        mask2d = mask[:1]
    else:
        mask2d = mask.reshape(1, *mask.shape[-2:])
    masked = rgb * mask2d + 0.5 * (1.0 - mask2d)
    arr = (masked.permute(1, 2, 0).numpy() * 255.0).round().clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def _encode_clip_rows(
    *,
    samples: List[Tuple[str, Dict[str, Any], Image.Image]],
    model: Any,
    preprocess: Any,
    device: torch.device,
    backend_name: str,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    batch_size = 32
    for start in range(0, len(samples), batch_size):
        chunk = samples[start : start + batch_size]
        images = torch.stack([preprocess(img) for _split, _sample, img in chunk], dim=0).to(device)
        with torch.no_grad():
            feats = model.encode_image(images).float()
            feats = feats / feats.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        for (split, sample, _img), feat in zip(chunk, feats.detach().cpu()):
            meta = sample.get("meta", {}) if isinstance(sample.get("meta", {}), dict) else {}
            rows.append(
                {
                    "dataset": str(meta.get("dataset", "")),
                    "clip_id": str(meta.get("clip_id", "")),
                    "split": str(split),
                    "dataset_index": int(sample.get("_dataset_index", -1)),
                    "backend": str(backend_name),
                    "feature_dim": int(feat.numel()),
                    "feature_target": [float(x) for x in feat.tolist()],
                    "semantic_frame_path": str(sample.get("semantic_frame_path", "")),
                    "semantic_mask_path": str(sample.get("semantic_mask_path", "")),
                    "teacher_as_mainline_semantic_source": False,
                }
            )
    return rows


def build_real_bootstrap_cache(
    *,
    audit: Dict[str, Any],
    contract_path: str,
    cache_json: Path,
    cache_md: Path,
) -> Dict[str, Any]:
    cache_path = WORK_ROOT / "data/processed/stage2_real_bootstrap_cache_20260410/clip_vit_b32_core_trainval_required_subset.jsonl"
    if not bool(audit.get("chosen_backend_usable", False)):
        payload = {
            "generated_at_utc": now_iso(),
            "cache_build_blocked": True,
            "blocking_reason": str(audit.get("blocking_reason_if_no_real_backend", "")),
            "cache_path": str(cache_path),
            "backend_used": "none",
            "train_val_coverage": {},
            "per_dataset_coverage": {},
            "missing_failed_items_count": 0,
            "teacher_as_mainline_semantic_source": False,
        }
        _write_json(cache_json, payload)
        _write_md(cache_md, ["# Stage2 Real Bootstrap Cache Build", "", "- cache_build_blocked: true", f"- blocking_reason: {payload['blocking_reason']}"])
        return payload

    try:
        import clip  # type: ignore
    except Exception as exc:
        payload = {
            "generated_at_utc": now_iso(),
            "cache_build_blocked": True,
            "blocking_reason": f"clip import failed during cache build: {exc}",
            "cache_path": str(cache_path),
            "backend_used": str(audit.get("chosen_bootstrap_backend", "none")),
            "train_val_coverage": {},
            "per_dataset_coverage": {},
            "missing_failed_items_count": 0,
            "teacher_as_mainline_semantic_source": False,
        }
        _write_json(cache_json, payload)
        _write_md(cache_md, ["# Stage2 Real Bootstrap Cache Build", "", "- cache_build_blocked: true", f"- blocking_reason: {payload['blocking_reason']}"])
        return payload

    # Keep cache generation off CUDA so the later evaluator can acquire a clean
    # GPU lease before CUDA is initialized in the process.
    device = torch.device("cpu")
    model, preprocess = clip.load("ViT-B/32", device=str(device), download_root=str(_local_clip_cache_dir()))
    model.eval()

    samples: List[Tuple[str, Dict[str, Any], Image.Image]] = []
    failed = 0
    per_dataset: Dict[str, Dict[str, int]] = {"train": {}, "val": {}}
    limits = {"train": CACHE_LIMIT_TRAIN_PER_DATASET, "val": CACHE_LIMIT_VAL_PER_DATASET}
    for split, limit in limits.items():
        ds = _dataset(["vspw", "vipseg"], split, str(contract_path), max_samples=int(limit))
        for idx in range(len(ds)):
            try:
                sample = ds[idx]
                sample["_dataset_index"] = int(idx)
                meta = sample.get("meta", {}) if isinstance(sample.get("meta", {}), dict) else {}
                dname = str(meta.get("dataset", ""))
                per_dataset.setdefault(split, {})[dname] = per_dataset.setdefault(split, {}).get(dname, 0) + 1
                samples.append((split, sample, _sample_to_clip_image(sample)))
            except Exception:
                failed += 1

    rows = _encode_clip_rows(
        samples=samples,
        model=model,
        preprocess=preprocess,
        device=device,
        backend_name=str(audit.get("chosen_bootstrap_backend", "local_clip_vit_b32_mask_crop_visual_teacher")),
    )
    count = _write_jsonl(cache_path, rows)
    payload = {
        "generated_at_utc": now_iso(),
        "cache_build_blocked": False,
        "cache_path": str(cache_path),
        "backend_used": str(audit.get("chosen_bootstrap_backend", "")),
        "feature_dim": int(audit.get("chosen_backend_feature_dim", 512)),
        "coverage_mode": "required_subset_cache_for_v1_pilot",
        "requested_limit_per_dataset_split": {
            "train": int(CACHE_LIMIT_TRAIN_PER_DATASET),
            "val": int(CACHE_LIMIT_VAL_PER_DATASET),
        },
        "train_val_coverage": {
            "total_rows": int(count),
            "train_rows": int(sum(per_dataset.get("train", {}).values())),
            "val_rows": int(sum(per_dataset.get("val", {}).values())),
        },
        "per_dataset_coverage": per_dataset,
        "missing_failed_items_count": int(failed),
        "teacher_as_mainline_semantic_source": False,
    }
    _write_json(cache_json, payload)
    lines = [
        "# Stage2 Real Bootstrap Cache Build",
        "",
        f"- generated_at_utc: {payload['generated_at_utc']}",
        f"- cache_build_blocked: {payload['cache_build_blocked']}",
        f"- cache_path: {payload['cache_path']}",
        f"- backend_used: {payload['backend_used']}",
        f"- feature_dim: {payload['feature_dim']}",
        f"- coverage_mode: {payload['coverage_mode']}",
        f"- train_val_coverage: {payload['train_val_coverage']}",
        f"- per_dataset_coverage: {payload['per_dataset_coverage']}",
        f"- missing_failed_items_count: {payload['missing_failed_items_count']}",
        f"- teacher_as_mainline_semantic_source: {payload['teacher_as_mainline_semantic_source']}",
    ]
    _write_md(cache_md, lines)
    return payload


def _load_ckpt_args(ckpt_path: str | Path) -> Dict[str, Any]:
    ckpt = torch.load(Path(ckpt_path), map_location="cpu", weights_only=False)
    args = ckpt.get("args", {}) if isinstance(ckpt.get("args", {}), dict) else {}
    return args


def _load_ckpt_step(ckpt_path: str | Path) -> int:
    ckpt = torch.load(Path(ckpt_path), map_location="cpu", weights_only=False)
    return int(ckpt.get("global_step", 0) or 0)


def _resume_ckpt_for_seed(seed: int) -> Path:
    p = WORK_ROOT / "outputs/checkpoints" / f"stage2_fullscale_core_cropenc_seed{int(seed)}_20260409" / "best.pt"
    if p.exists():
        return p
    return WORK_ROOT / "outputs/checkpoints/stage2_fullscale_core_cropenc_seed42_20260409/best.pt"


def _run_specs() -> List[Dict[str, Any]]:
    return [
        {
            "run_name": "stage2_semobjv1_align_seed42_20260410",
            "seed": 42,
            "objective_combo": "semantic_alignment_loss",
            "semantic_rescue_mode": "bootstrapplabel",
            "semantic_rescue_weight": 0.002,
            "semantic_alignment_loss_weight": 1.0,
            "query_persistence_consistency_loss_weight": 0.0,
            "semantic_hard_curriculum_weight": 0.0,
            "window_name": "semobjv1_align42",
        },
        {
            "run_name": "stage2_semobjv1_alignpersist_seed42_20260410",
            "seed": 42,
            "objective_combo": "semantic_alignment_loss+query_persistence_consistency_loss",
            "semantic_rescue_mode": "bootstrapplabel",
            "semantic_rescue_weight": 0.002,
            "semantic_alignment_loss_weight": 1.0,
            "query_persistence_consistency_loss_weight": 0.25,
            "semantic_hard_curriculum_weight": 0.0,
            "window_name": "semobjv1_ap42",
        },
        {
            "run_name": "stage2_semobjv1_alignhard_seed42_20260410",
            "seed": 42,
            "objective_combo": "semantic_alignment_loss+semantic_hard_curriculum_or_weighting",
            "semantic_rescue_mode": "bootstrapplabel",
            "semantic_rescue_weight": 0.002,
            "semantic_alignment_loss_weight": 1.0,
            "query_persistence_consistency_loss_weight": 0.0,
            "semantic_hard_curriculum_weight": 0.5,
            "window_name": "semobjv1_ah42",
        },
        {
            "run_name": "stage2_semobjv1_alignpersist_seed123_20260410",
            "seed": 123,
            "objective_combo": "semantic_alignment_loss+query_persistence_consistency_loss",
            "semantic_rescue_mode": "bootstrapplabel",
            "semantic_rescue_weight": 0.002,
            "semantic_alignment_loss_weight": 1.0,
            "query_persistence_consistency_loss_weight": 0.25,
            "semantic_hard_curriculum_weight": 0.0,
            "window_name": "semobjv1_ap123",
        },
    ]


def _select_gpu(run_name: str, lease_path: str) -> Dict[str, Any]:
    selector = select_single_gpu(
        required_mem_gb=40.0,
        safety_margin_gb=8.0,
        sample_count=3,
        interval_sec=0.5,
        lease_path=str(lease_path),
    )
    gpu_id = int(selector.get("selected_gpu_id", -1))
    if gpu_id < 0:
        raise RuntimeError("no GPU available for semantic objective redesign v1")
    lease = acquire_lease(
        gpu_id=gpu_id,
        owner=str(run_name),
        ttl_seconds=8 * 3600,
        lease_path=str(lease_path),
    )
    return {"selected_gpu_id": gpu_id, "lease_id": str(lease.get("lease_id", "")), "selector_payload": selector, "lease": lease}


def _release_lease_safe(lease_id: str, lease_path: str) -> None:
    if not str(lease_id).strip():
        return
    try:
        release_lease(lease_id=str(lease_id), lease_path=str(lease_path))
    except Exception:
        pass


def _tmux_windows(session_name: str) -> List[str]:
    proc = subprocess.run(["tmux", "list-windows", "-t", str(session_name), "-F", "#{window_name}"], text=True, capture_output=True)
    return proc.stdout.splitlines() if proc.returncode == 0 else []


def _status_for(meta: Dict[str, Any], session_name: str) -> Dict[str, Any]:
    final_path = Path(str(meta.get("final_json", "")))
    progress_path = Path(str(meta.get("progress_json", "")))
    detail: Dict[str, Any] = {}
    if str(meta.get("window_name", "")) in _tmux_windows(session_name):
        if progress_path.exists():
            try:
                detail = _read_json(progress_path)
            except Exception:
                detail = {}
        return {"status": "running", "detail": detail}
    if final_path.exists():
        try:
            detail = _read_json(final_path)
            status = str(detail.get("status", "launched")).lower()
            if status in {"completed", "failed"}:
                return {"status": status, "detail": detail}
        except Exception:
            pass
    if progress_path.exists():
        try:
            detail = _read_json(progress_path)
        except Exception:
            detail = {}
    return {"status": str(detail.get("status", "launched")).lower() if detail else "launched", "detail": detail}


def summarize(args: Any) -> Dict[str, Any]:
    launch = _read_json(args.launch_report)
    rows: List[Dict[str, Any]] = []
    running = completed = failed = 0
    for meta in launch.get("runs", []) if isinstance(launch.get("runs", []), list) else []:
        if not isinstance(meta, dict):
            continue
        status_info = _status_for(meta, session_name=str(args.tmux_session))
        status = str(status_info.get("status", "launched"))
        if status == "running":
            running += 1
        elif status == "completed":
            completed += 1
        elif status == "failed":
            failed += 1
        detail = status_info.get("detail", {}) if isinstance(status_info.get("detail", {}), dict) else {}
        best = detail.get("best_checkpoint_metric", {}) if isinstance(detail.get("best_checkpoint_metric", {}), dict) else {}
        latest = detail.get("latest_checkpoint_metric", {}) if isinstance(detail.get("latest_checkpoint_metric", {}), dict) else {}
        rows.append(
            {
                "run_name": str(meta.get("run_name", "")),
                "objective_combo": str(meta.get("objective_combo", "")),
                "seed": int(meta.get("seed", -1)),
                "selected_gpu_id": int(meta.get("selected_gpu_id", -1)),
                "lease_id": str(meta.get("lease_id", "")),
                "batch_size": int(meta.get("batch_size", 0)),
                "train_steps": int(meta.get("train_steps", 0)),
                "additional_train_steps": int(meta.get("additional_train_steps", 0)),
                "eval_interval": int(meta.get("eval_interval", 0)),
                "save_every_n_steps": int(meta.get("save_every_n_steps", 0)),
                "effective_train_sample_count_per_dataset": meta.get("effective_train_sample_count_per_dataset", {}),
                "effective_val_sample_count_per_dataset": meta.get("effective_val_sample_count_per_dataset", {}),
                "semantic_rescue_mode": str(meta.get("semantic_rescue_mode", "")),
                "semantic_rescue_weight": float(meta.get("semantic_rescue_weight", 0.0)),
                "semantic_alignment_loss_weight": float(meta.get("semantic_alignment_loss_weight", 0.0)),
                "query_persistence_consistency_loss_weight": float(meta.get("query_persistence_consistency_loss_weight", 0.0)),
                "semantic_hard_curriculum_weight": float(meta.get("semantic_hard_curriculum_weight", 0.0)),
                "status": status,
                "final_json": str(meta.get("final_json", "")),
                "best_checkpoint": str(Path(str(meta.get("output_dir", ""))) / "best.pt"),
                "latest_checkpoint": str(Path(str(meta.get("output_dir", ""))) / "latest.pt"),
                "best_checkpoint_metric": best,
                "latest_checkpoint_metric": latest,
            }
        )
    if failed:
        next_step = "redesign_stage2_semantic_objective_v2"
    elif completed == len(rows) and rows:
        next_step = "summarize_redesign_v1_after_completion"
    else:
        next_step = "continue_redesign_v1"

    payload = {
        "generated_at_utc": now_iso(),
        "redesign_v1_status": f"{running}_running_{completed}_completed_{failed}_failed",
        "runs": rows,
        "next_step_choice_internal": next_step,
    }
    _write_json(args.summary_report, payload)
    lines = [
        "# Stage2 Semantic Objective Redesign V1 Results",
        "",
        f"- generated_at_utc: {payload['generated_at_utc']}",
        f"- redesign_v1_status: {payload['redesign_v1_status']}",
        f"- next_step_choice_internal: {payload['next_step_choice_internal']}",
        "",
        "| run_name | combo | gpu | batch | steps | status | best_endpoint_l2 | latest_endpoint_l2 |",
        "|---|---|---:|---:|---:|---|---:|---:|",
    ]
    for row in rows:
        best_metrics = row.get("best_checkpoint_metric", {}).get("metrics", {}) if isinstance(row.get("best_checkpoint_metric", {}), dict) else {}
        latest_metrics = row.get("latest_checkpoint_metric", {}).get("metrics", {}) if isinstance(row.get("latest_checkpoint_metric", {}), dict) else {}
        lines.append(
            "| {run} | {combo} | {gpu} | {batch} | {steps} | {status} | {best:.8f} | {latest:.8f} |".format(
                run=row.get("run_name", ""),
                combo=row.get("objective_combo", ""),
                gpu=row.get("selected_gpu_id", -1),
                batch=row.get("batch_size", 0),
                steps=row.get("train_steps", 0),
                status=row.get("status", ""),
                best=float(best_metrics.get("free_rollout_endpoint_l2", 1e9)),
                latest=float(latest_metrics.get("free_rollout_endpoint_l2", 1e9)),
            )
        )
    _write_md(args.results_md, lines)
    return payload


def launch(args: Any) -> Dict[str, Any]:
    if subprocess.run(["tmux", "has-session", "-t", str(args.tmux_session)], capture_output=True).returncode != 0:
        subprocess.run(["tmux", "new-session", "-d", "-s", str(args.tmux_session), "bash"], check=True)

    anchor_args = _load_ckpt_args(_resume_ckpt_for_seed(42))
    obs_len = int(anchor_args.get("obs_len", 8) or 8)
    fut_len = int(anchor_args.get("fut_len", 8) or 8)
    max_tokens = int(anchor_args.get("max_tokens", 64) or 64)
    crop_size = int(anchor_args.get("semantic_crop_size", 64) or 64)
    train_counts = _dataset_counts(["vspw", "vipseg"], "train", args.stage2_contract_json, max_samples=CACHE_LIMIT_TRAIN_PER_DATASET)
    val_counts = _dataset_counts(["vspw", "vipseg"], "val", args.stage2_contract_json, max_samples=CACHE_LIMIT_VAL_PER_DATASET)

    runs = []
    for spec in _run_specs():
        run_name = str(spec["run_name"])
        resume_from = _resume_ckpt_for_seed(int(spec["seed"]))
        resume_step = _load_ckpt_step(resume_from)
        gpu = _select_gpu(run_name=run_name, lease_path=str(args.shared_lease_path))
        out_dir = Path(args.work_root) / "outputs" / "checkpoints" / run_name
        meta = {
            **spec,
            "selected_gpu_id": int(gpu["selected_gpu_id"]),
            "lease_id": str(gpu["lease_id"]),
            "dataset_names": ["vspw", "vipseg"],
            "obs_len": int(obs_len),
            "fut_len": int(fut_len),
            "max_tokens": int(max_tokens),
            "semantic_crop_size": int(crop_size),
            "semantic_source_mainline": "crop_visual_encoder",
            "legacy_semantic_source": "hand_crafted_stats",
            "batch_size": int(PILOT_BATCH_SIZE),
            "resume_from": str(resume_from),
            "resume_global_step": int(resume_step),
            "additional_train_steps": int(PILOT_EXTRA_STEPS),
            "train_steps": int(resume_step + PILOT_EXTRA_STEPS),
            "eval_interval": int(PILOT_EVAL_INTERVAL),
            "eval_max_batches": int(PILOT_EVAL_MAX_BATCHES),
            "save_every_n_steps": int(PILOT_SAVE_EVERY),
            "max_samples_train": int(CACHE_LIMIT_TRAIN_PER_DATASET),
            "max_samples_val": int(CACHE_LIMIT_VAL_PER_DATASET),
            "effective_train_sample_count_per_dataset": train_counts,
            "effective_val_sample_count_per_dataset": val_counts,
            "semantic_bootstrap_target_dim": 512,
            "output_dir": str(out_dir),
            "raw_json": str(Path(args.work_root) / "reports" / f"{run_name}_raw.json"),
            "progress_json": str(Path(args.work_root) / "reports" / f"{run_name}_progress.json"),
            "final_json": str(Path(args.work_root) / "reports" / f"{run_name}_final.json"),
            "log_path": str(Path(args.work_root) / "logs" / f"{run_name}.log"),
            "stage2_contract_json": str(args.stage2_contract_json),
            "stage1_runtime_json": str(args.stage1_runtime_json),
            "stage1_best_ckpt": str(args.stage1_best_ckpt),
            "shared_lease_path": str(args.shared_lease_path),
            "bootstrap_cache_jsonl": str(args.bootstrap_cache_jsonl),
            "work_root": str(args.work_root),
            "python_bin": str(args.python_bin),
        }
        meta_json = Path(args.work_root) / "reports" / "stage2_semantic_objective_redesign_v1_runs_20260410" / f"{run_name}_launch_meta.json"
        meta["meta_json"] = str(meta_json)
        _write_json(meta_json, meta)
        runs.append(meta)

        env = {
            "PYTHONPATH": f"{args.work_root}/code:{os.environ.get('PYTHONPATH', '')}",
            "CUDA_VISIBLE_DEVICES": str(meta["selected_gpu_id"]),
            "TRACEWM_STAGE1_V2_GPU_SELECTION_METADATA_JSON": json.dumps(
                {
                    "selected_gpu_id": int(meta["selected_gpu_id"]),
                    "lease_id": str(meta["lease_id"]),
                    "owner": run_name,
                    "mode": "single_gpu_only",
                },
                ensure_ascii=True,
            ),
        }
        env_prefix = " ".join(f"{k}={shlex.quote(v)}" for k, v in env.items())
        cmd = (
            f"{env_prefix} {shlex.quote(str(args.python_bin))} "
            f"{shlex.quote(str(Path(args.work_root) / 'code/stwm/tools/run_tracewm_stage2_semantic_objective_redesign_v1_20260410.py'))} "
            f"--mode run-one --meta-json {shlex.quote(str(meta_json))}"
        )
        subprocess.run(["tmux", "new-window", "-t", str(args.tmux_session), "-n", str(meta["window_name"]), cmd], check=True)

    launch_payload = {
        "generated_at_utc": now_iso(),
        "mode": "stage2_semantic_objective_redesign_v1_launch",
        "tmux_session": str(args.tmux_session),
        "pilot_policy": "bounded medium-budget pilot; no Stage1 training; teacher cache target only",
        "runs": runs,
    }
    _write_json(args.launch_report, launch_payload)
    return summarize(args)


def run_one(args: Any) -> None:
    meta = _read_json(args.meta_json)
    lease_id = str(meta.get("lease_id", ""))
    lease_path = str(meta.get("shared_lease_path", ""))
    trainer = Path(str(meta["work_root"])) / "code/stwm/tracewm_v2_stage2/trainers/train_tracewm_stage2_smalltrain.py"
    cmd = [
        str(meta["python_bin"]),
        str(trainer),
        "--stage2-contract-path",
        str(meta["stage2_contract_json"]),
        "--recommended-runtime-json",
        str(meta["stage1_runtime_json"]),
        "--use-recommended-runtime",
        "--stage1-backbone-checkpoint",
        str(meta["stage1_best_ckpt"]),
        "--dataset-names",
        "vspw",
        "vipseg",
        "--train-split",
        "train",
        "--val-split",
        "val",
        "--obs-len",
        str(meta["obs_len"]),
        "--fut-len",
        str(meta["fut_len"]),
        "--max-tokens",
        str(meta["max_tokens"]),
        "--max-samples-train",
        str(meta["max_samples_train"]),
        "--max-samples-val",
        str(meta["max_samples_val"]),
        "--batch-size",
        str(meta["batch_size"]),
        "--train-steps",
        str(meta["train_steps"]),
        "--eval-interval",
        str(meta["eval_interval"]),
        "--eval-max-batches",
        str(meta["eval_max_batches"]),
        "--save-every-n-steps",
        str(meta["save_every_n_steps"]),
        "--semantic-source-mainline",
        str(meta["semantic_source_mainline"]),
        "--legacy-semantic-source",
        str(meta["legacy_semantic_source"]),
        "--semantic-crop-size",
        str(meta["semantic_crop_size"]),
        "--semantic-rescue-mode",
        str(meta["semantic_rescue_mode"]),
        "--semantic-rescue-weight",
        str(meta["semantic_rescue_weight"]),
        "--semantic-bootstrap-cache-path",
        str(meta["bootstrap_cache_jsonl"]),
        "--semantic-bootstrap-target-dim",
        str(meta["semantic_bootstrap_target_dim"]),
        "--semantic-alignment-loss-weight",
        str(meta["semantic_alignment_loss_weight"]),
        "--query-persistence-consistency-loss-weight",
        str(meta["query_persistence_consistency_loss_weight"]),
        "--semantic-hard-curriculum-weight",
        str(meta["semantic_hard_curriculum_weight"]),
        "--resume-from",
        str(meta["resume_from"]),
        "--skip-resume-optimizer",
        "--output-dir",
        str(meta["output_dir"]),
        "--run-name",
        str(meta["run_name"]),
        "--run-summary-json",
        str(meta["raw_json"]),
        "--progress-json",
        str(meta["progress_json"]),
        "--seed",
        str(meta["seed"]),
    ]
    try:
        proc = subprocess.run(cmd, cwd=str(meta["work_root"]), text=True, capture_output=True, env=os.environ.copy())
        Path(str(meta["log_path"])).write_text(proc.stdout + ("\n" if proc.stdout else "") + proc.stderr, encoding="utf-8")
        if proc.returncode != 0:
            _write_json(
                meta["final_json"],
                {
                    "generated_at_utc": now_iso(),
                    "run_name": str(meta["run_name"]),
                    "status": "failed",
                    "returncode": int(proc.returncode),
                    "stderr_tail": proc.stderr[-4000:],
                    "stdout_tail": proc.stdout[-4000:],
                },
            )
            raise RuntimeError(f"trainer failed rc={proc.returncode}")
        raw = _read_json(meta["raw_json"])
        raw["generated_at_utc"] = now_iso()
        raw["status"] = "completed"
        raw["selected_gpu_id"] = int(meta["selected_gpu_id"])
        raw["lease_id"] = str(meta["lease_id"])
        raw["objective_combo"] = str(meta["objective_combo"])
        raw["resume_global_step"] = int(meta["resume_global_step"])
        _write_json(meta["final_json"], raw)
    except Exception as exc:
        _write_json(
            meta["final_json"],
            {
                "generated_at_utc": now_iso(),
                "run_name": str(meta.get("run_name", "")),
                "status": "failed",
                "message": str(exc),
                "traceback": traceback.format_exc(),
            },
        )
        raise
    finally:
        _release_lease_safe(lease_id=lease_id, lease_path=lease_path)


def _aggregate_rows(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key in METRIC_KEYS:
        out[key] = _mean_std([_f(row.get("metrics", {}).get(key), 1e9) for row in rows])
    return out


def _baseline_refs() -> Dict[str, Any]:
    diagnosis = _read_json(WORK_ROOT / "reports/stage2_semantic_value_diagnosis_20260410.json")
    full = diagnosis.get("full_validation_panel", {}).get("family_aggregates", {})
    return {
        "cropenc_fullscale_mean": full.get("cropenc", {}),
        "legacysem_fullscale_mean": full.get("legacysem", {}),
        "hard_subset_panels": diagnosis.get("hard_subset_panels", {}),
    }


def _metric_from_final(final_json: str | Path, key: str = "best_checkpoint_metric") -> Dict[str, float]:
    p = _read_json(final_json)
    block = p.get(key, {}) if isinstance(p.get(key, {}), dict) else {}
    metrics = block.get("metrics", {}) if isinstance(block.get("metrics", {}), dict) else {}
    return {k: _f(metrics.get(k), 1e9) for k in METRIC_KEYS}


def _v1_run_rows_from_summary(summary: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [row for row in summary.get("runs", []) if isinstance(row, dict)]


def _select_eval_gpu(lease_path: str) -> Dict[str, Any]:
    selector = select_single_gpu(required_mem_gb=40.0, safety_margin_gb=8.0, sample_count=3, interval_sec=0.5, lease_path=lease_path)
    gpu_id = int(selector.get("selected_gpu_id", -1))
    if gpu_id < 0:
        return {"selected_gpu_id": -1, "lease_id": ""}
    lease = acquire_lease(gpu_id=gpu_id, owner="stage2_semobjv1_diagnosis_eval", ttl_seconds=6 * 3600, lease_path=lease_path)
    return {"selected_gpu_id": gpu_id, "lease_id": str(lease.get("lease_id", ""))}


def diagnose(args: Any) -> Dict[str, Any]:
    summary = _read_json(args.summary_report)
    rows = _v1_run_rows_from_summary(summary)
    completed = [row for row in rows if str(row.get("status", "")) == "completed"]
    audit = _read_json(args.backend_audit_json) if Path(args.backend_audit_json).exists() else {"chosen_backend_usable": False}
    cache = _read_json(args.cache_build_json) if Path(args.cache_build_json).exists() else {"cache_build_blocked": True}
    refs = _baseline_refs()
    crop_ep = _f(refs["cropenc_fullscale_mean"].get("free_rollout_endpoint_l2", {}).get("mean"), 1e9)
    legacy_ep = _f(refs["legacysem_fullscale_mean"].get("free_rollout_endpoint_l2", {}).get("mean"), 1e9)

    terminal_ready = bool(rows and len(completed) == len(rows))
    eval_payload: Dict[str, Any] = {
        "generated_at_utc": now_iso(),
        "diagnosis_type": "stage2_semantic_objective_redesign_v1",
        "teacher_as_mainline_semantic_source": False,
        "chosen_real_bootstrap_backend": str(audit.get("chosen_bootstrap_backend", "none")),
        "real_bootstrap_backend_usable": bool(audit.get("chosen_backend_usable", False)),
        "cache_build_blocked": bool(cache.get("cache_build_blocked", True)),
        "v1_runs_terminal": bool(terminal_ready),
        "baseline_refs": {
            "current_cropenc_fullscale_mean_endpoint_l2": float(crop_ep),
            "legacysem_fullscale_mean_endpoint_l2": float(legacy_ep),
        },
        "full_validation_panel": {},
        "semantic_hard_subset_panel": {},
        "burst_persistence_hard_panel": {},
        "success_criteria": {},
    }
    if not terminal_ready:
        eval_payload["success_criteria"] = {
            "true_new_best_not_warm_start_inherited": False,
            "semantic_hard_positive_signal": False,
            "improved_vs_current_cropenc_baseline": False,
            "narrowed_or_won_vs_legacysem": False,
            "best_v1_objective_combo": "none",
            "next_step_choice": "redesign_stage2_semantic_objective_v2",
            "reason": "v1_runs_not_terminal",
        }
        _write_json(args.diagnosis_report, eval_payload)
        return eval_payload

    eval_gpu = _select_eval_gpu(str(args.shared_lease_path))
    if int(eval_gpu.get("selected_gpu_id", -1)) >= 0:
        if not str(os.environ.get("CUDA_VISIBLE_DEVICES", "")).strip():
            os.environ["CUDA_VISIBLE_DEVICES"] = str(eval_gpu["selected_gpu_id"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        print(f"[semobjv1] diagnosis device={device} eval_gpu={eval_gpu}", flush=True)
        stage1_model, _ = _load_stage1_model(device=device, stage1_ckpt=str(args.stage1_best_ckpt))
        full_ds = _make_dataset(["vspw", "vipseg"], "val", str(args.stage2_contract_json), max_samples=-1)
        subset_manifest = _read_json(WORK_ROOT / "manifests/protocol_v2/stage2_semantic_hard_subsets_20260410.json")
        core_ds = _make_dataset(["vspw", "vipseg"], "val", str(args.stage2_contract_json), max_samples=-1)
        burst_ds = None
        loaded: Dict[str, Tuple[Any, Any, Any, str]] = {}
        full_rows: List[Dict[str, Any]] = []
        for row in completed:
            ckpt = row.get("best_checkpoint", "")
            print(f"[semobjv1] full validation eval run={row['run_name']}", flush=True)
            loaded[str(row["run_name"])] = _load_stage2_modules(str(ckpt), device, stage1_model)
            metrics = _evaluate_loaded_stage2(
                loaded=loaded[str(row["run_name"])],
                stage1_model=stage1_model,
                dataset=full_ds,
                device=device,
            )
            full_rows.append(
                {
                    "run_name": str(row["run_name"]),
                    "objective_combo": str(row.get("objective_combo", "")),
                    "seed": int(row.get("seed", -1)),
                    "metrics": {k: _f(metrics.get(k), 1e9) for k in METRIC_KEYS},
                }
            )
        eval_payload["full_validation_panel"] = {
            "runs": full_rows,
            "aggregate": _aggregate_rows(full_rows),
            "dataset_binding": ["VSPW", "VIPSeg"],
            "eval_scope": "full_validation",
        }

        semantic_subset_names = [
            "occlusion_reappearance",
            "crossing_or_interaction_ambiguity",
            "small_object_or_low_area",
            "appearance_change_or_semantic_shift",
        ]
        hard_panels: Dict[str, Any] = {}
        for subset_name in semantic_subset_names:
            print(f"[semobjv1] semantic hard eval subset={subset_name}", flush=True)
            subset = subset_manifest.get("subsets", {}).get(subset_name, {})
            items = subset.get("items", []) if isinstance(subset.get("items", []), list) else []
            indices = [int(x["dataset_index"]) for x in items if isinstance(x, dict) and "dataset_index" in x]
            eval_ds = Subset(core_ds, indices)
            subset_rows: List[Dict[str, Any]] = []
            for row in completed:
                print(f"[semobjv1] subset={subset_name} eval run={row['run_name']}", flush=True)
                metrics = _evaluate_loaded_stage2(
                    loaded=loaded[str(row["run_name"])],
                    stage1_model=stage1_model,
                    dataset=eval_ds,
                    device=device,
                )
                subset_rows.append(
                    {
                        "run_name": str(row["run_name"]),
                        "objective_combo": str(row.get("objective_combo", "")),
                        "seed": int(row.get("seed", -1)),
                        "metrics": {k: _f(metrics.get(k), 1e9) for k in METRIC_KEYS},
                    }
                )
            hard_panels[subset_name] = {
                "clip_count": int(len(indices)),
                "runs": subset_rows,
                "aggregate": _aggregate_rows(subset_rows),
                "cropenc_baseline_aggregate": refs.get("hard_subset_panels", {})
                .get(subset_name, {})
                .get("families", {})
                .get("cropenc", {})
                .get("aggregate", {}),
                "legacysem_baseline_aggregate": refs.get("hard_subset_panels", {})
                .get(subset_name, {})
                .get("families", {})
                .get("legacysem", {})
                .get("aggregate", {}),
            }
        eval_payload["semantic_hard_subset_panel"] = hard_panels

        burst_subset = subset_manifest.get("subsets", {}).get("burst_persistence_stress", {})
        try:
            print("[semobjv1] optional BURST persistence-hard eval", flush=True)
            burst_ds = _make_dataset(["burst"], "val", str(args.stage2_contract_json), max_samples=-1)
            items = burst_subset.get("items", []) if isinstance(burst_subset.get("items", []), list) else []
            indices = [int(x["dataset_index"]) for x in items if isinstance(x, dict) and "dataset_index" in x]
            burst_eval_ds = Subset(burst_ds, indices)
            burst_rows = []
            for row in completed:
                print(f"[semobjv1] burst persistence eval run={row['run_name']}", flush=True)
                metrics = _evaluate_loaded_stage2(
                    loaded=loaded[str(row["run_name"])],
                    stage1_model=stage1_model,
                    dataset=burst_eval_ds,
                    device=device,
                )
                burst_rows.append(
                    {
                        "run_name": str(row["run_name"]),
                        "objective_combo": str(row.get("objective_combo", "")),
                        "seed": int(row.get("seed", -1)),
                        "metrics": {k: _f(metrics.get(k), 1e9) for k in METRIC_KEYS},
                    }
                )
            eval_payload["burst_persistence_hard_panel"] = {
                "status": "evaluated_optional_stress_panel",
                "clip_count": int(len(indices)),
                "runs": burst_rows,
                "aggregate": _aggregate_rows(burst_rows),
            }
        except Exception as exc:
            eval_payload["burst_persistence_hard_panel"] = {"status": "skipped", "reason": str(exc)}
    finally:
        _release_lease_safe(str(eval_gpu.get("lease_id", "")), str(args.shared_lease_path))

    true_new_best = False
    best_combo = "none"
    best_full_endpoint = 1e9
    for row in completed:
        final_path = str(row.get("final_json", ""))
        final = _read_json(final_path)
        best_step = int(final.get("best_checkpoint_metric", {}).get("global_step", -1))
        resume_step = int(final.get("resume_global_step", row.get("train_steps", 0) - PILOT_EXTRA_STEPS))
        true_new_best = bool(true_new_best or best_step > resume_step)
    for row in eval_payload.get("full_validation_panel", {}).get("runs", []):
        ep = _f(row.get("metrics", {}).get("free_rollout_endpoint_l2"), 1e9)
        if ep < best_full_endpoint:
            best_full_endpoint = ep
            best_combo = str(row.get("objective_combo", "none"))

    semantic_hard_positive = False
    for _name, panel in eval_payload.get("semantic_hard_subset_panel", {}).items():
        crop_base = _f(panel.get("cropenc_baseline_aggregate", {}).get("free_rollout_endpoint_l2", {}).get("mean"), 1e9)
        for row in panel.get("runs", []) if isinstance(panel.get("runs", []), list) else []:
            ep = _f(row.get("metrics", {}).get("free_rollout_endpoint_l2"), 1e9)
            if ep < crop_base:
                semantic_hard_positive = True

    improved_vs_cropenc = bool(best_full_endpoint < crop_ep)
    narrowed_vs_legacy = bool(abs(best_full_endpoint - legacy_ep) < abs(crop_ep - legacy_ep) or best_full_endpoint < legacy_ep)
    non_catastrophic = bool(best_full_endpoint <= crop_ep * 1.5)
    if not bool(audit.get("chosen_backend_usable", False)) or bool(cache.get("cache_build_blocked", True)):
        next_step = "bootstrap_backend_blocked_fix_first"
    elif true_new_best and semantic_hard_positive and non_catastrophic:
        next_step = "stage2_semantic_rescue_fullscale_wave1"
    else:
        next_step = "redesign_stage2_semantic_objective_v2"

    eval_payload["success_criteria"] = {
        "true_new_best_not_warm_start_inherited": bool(true_new_best),
        "semantic_hard_positive_signal": bool(semantic_hard_positive),
        "improved_vs_current_cropenc_baseline": bool(improved_vs_cropenc),
        "narrowed_or_won_vs_legacysem": bool(narrowed_vs_legacy),
        "full_validation_non_catastrophic": bool(non_catastrophic),
        "best_v1_objective_combo": str(best_combo),
        "best_v1_full_validation_endpoint_l2": float(best_full_endpoint),
        "current_cropenc_fullscale_mean_endpoint_l2": float(crop_ep),
        "legacysem_fullscale_mean_endpoint_l2": float(legacy_ep),
        "next_step_choice": str(next_step),
    }
    _write_json(args.diagnosis_report, eval_payload)
    return eval_payload


def wait_for_completion(args: Any) -> Dict[str, Any]:
    deadline = time.time() + float(args.wait_timeout_seconds)
    last = summarize(args)
    while time.time() < deadline:
        status = str(last.get("redesign_v1_status", ""))
        if status.startswith("0_running_"):
            return last
        time.sleep(float(args.poll_seconds))
        last = summarize(args)
    last["timed_out_waiting_for_completion"] = True
    _write_json(args.summary_report, last)
    return last


def run_all(args: Any) -> Dict[str, Any]:
    write_protocol_doc(Path(args.protocol_doc))
    audit = write_backend_audit(Path(args.backend_audit_json), Path(args.backend_audit_md))
    cache = build_real_bootstrap_cache(
        audit=audit,
        contract_path=str(args.stage2_contract_json),
        cache_json=Path(args.cache_build_json),
        cache_md=Path(args.cache_build_md),
    )
    if not bool(cache.get("cache_build_blocked", True)):
        args.bootstrap_cache_jsonl = str(cache.get("cache_path", args.bootstrap_cache_jsonl))
    launch(args)
    summary = wait_for_completion(args)
    diagnosis = diagnose(args)
    return {"summary": summary, "diagnosis": diagnosis}


def parse_args() -> Any:
    p = ArgumentParser(description="Stage2 semantic objective redesign v1 runner")
    p.add_argument("--mode", default="all", choices=["all", "audit-cache", "launch", "run-one", "summarize", "diagnose"])
    p.add_argument("--meta-json", default="")
    p.add_argument("--work-root", default=str(WORK_ROOT))
    p.add_argument("--python-bin", default=_python_bin_default())
    p.add_argument("--tmux-session", default=SESSION)
    p.add_argument("--stage2-contract-json", default=str(WORK_ROOT / "reports/stage2_bootstrap_data_contract_20260408.json"))
    p.add_argument("--stage1-runtime-json", default=str(WORK_ROOT / "reports/stage1_v2_recommended_runtime_20260408.json"))
    p.add_argument("--stage1-best-ckpt", default=str(WORK_ROOT / "outputs/checkpoints/stage1_v2_longtrain_220m_mainline_20260408/best.pt"))
    p.add_argument("--shared-lease-path", default=str(WORK_ROOT / "reports/stage1_v2_gpu_lease_20260408.json"))
    p.add_argument("--bootstrap-cache-jsonl", default=str(WORK_ROOT / "data/processed/stage2_real_bootstrap_cache_20260410/clip_vit_b32_core_trainval_required_subset.jsonl"))
    p.add_argument("--protocol-doc", default=str(WORK_ROOT / "docs/STAGE2_SEMANTIC_OBJECTIVE_REDESIGN_V1_PROTOCOL_20260410.md"))
    p.add_argument("--backend-audit-json", default=str(WORK_ROOT / "reports/stage2_real_bootstrap_backend_audit_20260410.json"))
    p.add_argument("--backend-audit-md", default=str(WORK_ROOT / "docs/STAGE2_REAL_BOOTSTRAP_BACKEND_AUDIT_20260410.md"))
    p.add_argument("--cache-build-json", default=str(WORK_ROOT / "reports/stage2_real_bootstrap_cache_build_20260410.json"))
    p.add_argument("--cache-build-md", default=str(WORK_ROOT / "docs/STAGE2_REAL_BOOTSTRAP_CACHE_BUILD_20260410.md"))
    p.add_argument("--launch-report", default=str(WORK_ROOT / "reports/stage2_semantic_objective_redesign_v1_launch_20260410.json"))
    p.add_argument("--summary-report", default=str(WORK_ROOT / "reports/stage2_semantic_objective_redesign_v1_summary_20260410.json"))
    p.add_argument("--results-md", default=str(WORK_ROOT / "docs/STAGE2_SEMANTIC_OBJECTIVE_REDESIGN_V1_RESULTS_20260410.md"))
    p.add_argument("--diagnosis-report", default=str(WORK_ROOT / "reports/stage2_semantic_objective_redesign_v1_diagnosis_20260410.json"))
    p.add_argument("--wait-timeout-seconds", type=int, default=21600)
    p.add_argument("--poll-seconds", type=int, default=60)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "all":
        print(json.dumps(run_all(args), ensure_ascii=True, indent=2))
        return
    if args.mode == "audit-cache":
        audit = write_backend_audit(Path(args.backend_audit_json), Path(args.backend_audit_md))
        print(json.dumps(build_real_bootstrap_cache(audit=audit, contract_path=args.stage2_contract_json, cache_json=Path(args.cache_build_json), cache_md=Path(args.cache_build_md)), ensure_ascii=True, indent=2))
        return
    if args.mode == "launch":
        print(json.dumps(launch(args), ensure_ascii=True, indent=2))
        return
    if args.mode == "summarize":
        print(json.dumps(summarize(args), ensure_ascii=True, indent=2))
        return
    if args.mode == "diagnose":
        print(json.dumps(diagnose(args), ensure_ascii=True, indent=2))
        return
    if args.mode == "run-one":
        run_one(args)
        return
    raise RuntimeError(f"unsupported mode={args.mode}")


if __name__ == "__main__":
    main()
