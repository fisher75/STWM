from __future__ import annotations

from collections import OrderedDict
import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

try:
    import cv2

    cv2.setNumThreads(0)
except Exception:
    # OpenCV may be absent in some environments; training should still run.
    cv2 = None

from argparse import ArgumentParser
from dataclasses import asdict
from pathlib import Path
import json
import math
import random
import shutil
import statistics
import subprocess
import sys
import time
from typing import Any

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from stwm.datasets.stwm_dataset import STWMDataset
from stwm.models.stwm_v4_2 import STWMV42, estimate_v4_2_parameter_budget, load_model_config_v4_2
from stwm.modules.semantic_adapter import SemanticAdapter
from stwm.modules.trace_adapter import TraceAdapter


class _SampleDataset(Dataset):
    def __init__(self, samples: list[Any]) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Any:
        return self.samples[index]


def _collate_samples(batch: list[Any]) -> list[Any]:
    return batch


class _FrontendCacheReader:
    def __init__(
        self,
        *,
        cache_dir: str | Path,
        index_path: str | Path,
        max_shards_in_memory: int = 4,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.index_path = Path(index_path)
        self.max_shards_in_memory = max(1, int(max_shards_in_memory))

        if not self.cache_dir.exists():
            raise FileNotFoundError(f"frontend cache dir not found: {self.cache_dir}")
        if not self.index_path.exists():
            raise FileNotFoundError(f"frontend cache index not found: {self.index_path}")

        payload = json.loads(self.index_path.read_text())
        entries = payload.get("entries", []) if isinstance(payload, dict) else []
        if not isinstance(entries, list):
            raise ValueError(f"invalid frontend cache index payload: {self.index_path}")

        self.index: dict[str, dict[str, Any]] = {}
        for item in entries:
            if not isinstance(item, dict):
                continue
            clip_id = str(item.get("clip_id", "")).strip()
            shard = str(item.get("shard", "")).strip()
            if not clip_id or not shard:
                continue
            try:
                offset = int(item.get("offset", 0))
            except (TypeError, ValueError):
                continue
            self.index[clip_id] = {"shard": shard, "offset": offset}

        if not self.index:
            raise RuntimeError(f"frontend cache index has no valid entries: {self.index_path}")

        self._shard_cache: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()

    def _load_shard_records(self, shard_name: str) -> list[dict[str, Any]]:
        if shard_name in self._shard_cache:
            records = self._shard_cache.pop(shard_name)
            self._shard_cache[shard_name] = records
            return records

        shard_path = self.cache_dir / shard_name
        if not shard_path.exists():
            raise FileNotFoundError(f"frontend cache shard not found: {shard_path}")

        try:
            payload = torch.load(shard_path, map_location="cpu", weights_only=False)
        except TypeError:
            payload = torch.load(shard_path, map_location="cpu")

        records = payload.get("records") if isinstance(payload, dict) else None
        if not isinstance(records, list):
            raise ValueError(f"invalid frontend cache shard payload: {shard_path}")

        self._shard_cache[shard_name] = records
        while len(self._shard_cache) > self.max_shards_in_memory:
            self._shard_cache.popitem(last=False)
        return records

    def get(self, clip_id: str) -> dict[str, Any] | None:
        key = str(clip_id)
        entry = self.index.get(key)
        if entry is None:
            return None

        records = self._load_shard_records(str(entry["shard"]))
        offset = int(entry["offset"])
        if offset < 0 or offset >= len(records):
            raise IndexError(
                f"frontend cache offset out of range: clip_id={key}, shard={entry['shard']}, offset={offset}"
            )

        record = records[offset]
        if not isinstance(record, dict):
            raise ValueError(f"invalid frontend cache record type for clip_id={key}")
        return record


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Train/Eval STWM V4.2 real-budget pipeline")
    parser.add_argument("--data-root", default="/home/chen034/workspace/stwm/data/external")
    parser.add_argument("--manifest", default="/home/chen034/workspace/stwm/manifests/minisplits/stwm_week1_mini.json")
    parser.add_argument("--output-dir", default="/home/chen034/workspace/stwm/outputs/training/stwm_v4_2_real/full_v4_2")
    parser.add_argument("--run-name", default="full_v4_2")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--target-epochs", type=float, default=0.0)
    parser.add_argument("--min-optimizer-steps", type=int, default=0)
    parser.add_argument("--max-optimizer-steps", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)

    parser.add_argument("--sample-limit", type=int, default=18)
    parser.add_argument("--micro-batch-per-gpu", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--persistent-workers", action="store_true")
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--data-mode", choices=("raw", "frontend_cache"), default="raw")
    parser.add_argument("--frontend-cache-dir", default="")
    parser.add_argument("--frontend-cache-index", default="")
    parser.add_argument("--frontend-cache-max-shards-in-memory", type=int, default=4)

    parser.add_argument("--model-preset", default="prototype_220m_v4_2")
    parser.add_argument("--preset-file", default="/home/chen034/workspace/stwm/code/stwm/configs/model_presets_v4_2.json")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--activation-checkpointing", action="store_true")

    parser.add_argument("--disable-semantics", action="store_true")
    parser.add_argument("--disable-identity-memory", action="store_true")
    parser.add_argument("--neutralize-object-bias", action="store_true")
    parser.add_argument("--object-bias-alpha", type=float, default=1.0)
    parser.add_argument("--object-bias-delay-steps", type=int, default=0)
    parser.add_argument("--object-bias-gated", action="store_true")
    parser.add_argument("--object-bias-gate-threshold", type=float, default=0.5)
    parser.add_argument("--use-teacher-priors", action="store_true")
    parser.add_argument("--enable-reconnect-loss", action="store_true")
    parser.add_argument("--contrastive-temperature", type=float, default=0.07)
    parser.add_argument("--reconnect-window", type=int, default=3)
    parser.add_argument("--reconnect-threshold", type=float, default=0.20)

    parser.add_argument("--lambda-traj", type=float, default=1.0)
    parser.add_argument("--lambda-vis", type=float, default=0.25)
    parser.add_argument("--lambda-sem", type=float, default=0.5)
    parser.add_argument("--lambda-reid", type=float, default=0.25)
    parser.add_argument("--lambda-query", type=float, default=0.25)
    parser.add_argument("--lambda-reconnect", type=float, default=0.1)

    parser.add_argument("--qstr-enable", action="store_true")
    parser.add_argument("--qstr-residual-scale", type=float, default=0.2)
    parser.add_argument("--qstr-neutral-path-weight", type=float, default=1.0)
    parser.add_argument("--qstr-route-temperature", type=float, default=0.25)
    parser.add_argument("--qstr-temporal-consistency-weight", type=float, default=0.0)

    parser.add_argument("--semantic-warmup", action="store_true")
    parser.add_argument("--semantic-warmup-start-ratio", type=float, default=0.10)
    parser.add_argument("--semantic-warmup-end-ratio", type=float, default=0.30)

    parser.add_argument("--gradient-audit-interval", type=int, default=0)
    parser.add_argument("--gradient-audit-output", default="")
    parser.add_argument("--gradient-audit-secondary-every", type=int, default=5)

    parser.add_argument("--summary-name", default="mini_val_summary.json")
    parser.add_argument("--log-name", default="train_log.jsonl")
    parser.add_argument("--save-checkpoint", action="store_true")
    parser.add_argument("--checkpoint-dir-name", default="checkpoints")
    parser.add_argument("--checkpoint-interval", type=int, default=100)
    parser.add_argument("--milestone-interval", type=int, default=0)
    parser.add_argument("--checkpoint-name", default="")
    parser.add_argument("--resume-checkpoint", default="")
    parser.add_argument("--auto-resume", action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--min-free-disk-gb", type=float, default=50.0)

    parser.add_argument("--protocol-eval-interval", type=int, default=0)
    parser.add_argument("--protocol-eval-manifest", default="")
    parser.add_argument("--protocol-eval-dataset", default="all")
    parser.add_argument("--protocol-eval-max-clips", type=int, default=0)
    parser.add_argument("--protocol-eval-seed", type=int, default=42)
    parser.add_argument("--protocol-eval-obs-steps", type=int, default=8)
    parser.add_argument("--protocol-eval-pred-steps", type=int, default=8)
    parser.add_argument("--protocol-eval-run-name", default="protocol_val_main")
    parser.add_argument("--protocol-diagnostics-manifest", default="")
    parser.add_argument("--protocol-diagnostics-dataset", default="all")
    parser.add_argument("--protocol-diagnostics-max-clips", type=int, default=0)
    parser.add_argument("--protocol-diagnostics-run-name", default="protocol_val_eventful")
    parser.add_argument("--protocol-version", default="v2_4_detached_frozen")
    parser.add_argument("--protocol-best-checkpoint-name", default="best_protocol_main.pt")
    parser.add_argument("--protocol-best-selection-name", default="best_protocol_main_selection.json")
    return parser


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def _read_mask_ratio(mask_path: str, target_label_id: int | None) -> float:
    p = Path(mask_path)
    if not p.exists():
        return 0.0
    arr = np.array(Image.open(p))
    if arr.ndim == 3:
        arr = arr[..., 0]
    if target_label_id is None:
        return float((arr > 0).mean())
    tgt = arr == int(target_label_id)
    if tgt.any():
        return float(tgt.mean())
    return float((arr > 0).mean())


def _build_features_for_sample(
    sample: Any,
    *,
    trace_adapter: TraceAdapter,
    semantic_adapter: SemanticAdapter,
    device: torch.device,
    disable_semantics: bool,
    use_teacher_priors: bool,
    data_mode: str = "raw",
    frontend_cache_reader: _FrontendCacheReader | None = None,
) -> dict[str, torch.Tensor | str | int]:
    if data_mode == "frontend_cache":
        if frontend_cache_reader is None:
            raise RuntimeError("frontend cache reader is required when data-mode=frontend_cache")
        cached = frontend_cache_reader.get(str(sample.clip_id))
        if cached is None:
            raise KeyError(f"clip_id missing in frontend cache index: {sample.clip_id}")

        trace_features = torch.as_tensor(cached["trace_features"], dtype=torch.float32, device=device)
        semantic_features = torch.as_tensor(cached["semantic_features"], dtype=torch.float32, device=device)
        target_trajectory = torch.as_tensor(cached["target_trajectory"], dtype=torch.float32, device=device)
        target_visibility = torch.as_tensor(cached["target_visibility"], dtype=torch.float32, device=device)
        target_semantic_probs = torch.as_tensor(cached["target_semantic_probs"], dtype=torch.float32, device=device)
        mask_ratio_t = torch.as_tensor(cached["mask_ratios"], dtype=torch.float32, device=device)

        if trace_features.ndim != 2:
            raise ValueError(f"cached trace_features must be [T,D], got {tuple(trace_features.shape)}")
        if semantic_features.ndim != 2:
            raise ValueError(f"cached semantic_features must be [T,D], got {tuple(semantic_features.shape)}")
        if target_trajectory.ndim != 2 or target_trajectory.shape[-1] != 2:
            raise ValueError(f"cached target_trajectory must be [T,2], got {tuple(target_trajectory.shape)}")
        if target_visibility.ndim == 1:
            target_visibility = target_visibility.unsqueeze(-1)
        if target_visibility.ndim != 2 or target_visibility.shape[-1] != 1:
            raise ValueError(f"cached target_visibility must be [T,1], got {tuple(target_visibility.shape)}")
        if target_semantic_probs.ndim != 2:
            raise ValueError(
                f"cached target_semantic_probs must be [T,C], got {tuple(target_semantic_probs.shape)}"
            )
        if mask_ratio_t.ndim != 1:
            mask_ratio_t = mask_ratio_t.reshape(-1)

        seq_len = int(
            min(
                trace_features.shape[0],
                semantic_features.shape[0],
                target_trajectory.shape[0],
                target_visibility.shape[0],
                target_semantic_probs.shape[0],
                mask_ratio_t.shape[0],
            )
        )
        if seq_len <= 1:
            raise RuntimeError(f"clip {sample.clip_id} has insufficient cached sequence length")

        trace_features = trace_features[:seq_len]
        semantic_features = semantic_features[:seq_len]
        target_trajectory = target_trajectory[:seq_len]
        target_visibility = target_visibility[:seq_len]
        target_semantic_probs = target_semantic_probs[:seq_len]
        mask_ratio_t = mask_ratio_t[:seq_len]

        if disable_semantics:
            semantic_features = torch.zeros_like(semantic_features)

        speed = torch.norm(trace_features[:, 2:4], dim=-1) if trace_features.shape[-1] >= 4 else torch.zeros_like(mask_ratio_t)
        semantic_conf = target_semantic_probs.max(dim=-1).values
        prior_features = torch.stack(
            [
                mask_ratio_t.clamp(0.0, 1.0),
                target_visibility[:, 0].clamp(0.0, 1.0),
                speed.clamp(0.0, 1.0),
                semantic_conf.clamp(0.0, 1.0),
            ],
            dim=-1,
        )

        if use_teacher_priors:
            teacher_objectness = (0.6 * mask_ratio_t + 0.4 * target_visibility[:, 0]).clamp(0.0, 1.0)
        else:
            teacher_objectness = target_visibility[:, 0].clamp(0.0, 1.0)

        return {
            "clip_id": sample.clip_id,
            "seq_len": seq_len,
            "trace_features": trace_features.unsqueeze(0),
            "semantic_features": semantic_features.unsqueeze(0),
            "prior_features": prior_features.unsqueeze(0),
            "teacher_objectness": teacher_objectness.unsqueeze(0),
            "target_trajectory": target_trajectory.unsqueeze(0),
            "target_visibility": target_visibility.unsqueeze(0),
            "target_semantic_probs": target_semantic_probs.unsqueeze(0),
        }

    trace_summary = trace_adapter.encode(sample.frame_paths, metadata=sample.metadata, clip_id=sample.clip_id)
    semantic_summary = semantic_adapter.encode(
        sample.text_labels,
        len(sample.frame_paths),
        metadata=sample.metadata,
        clip_id=sample.clip_id,
    )

    seq_len = int(min(trace_summary.centers.shape[0], semantic_summary.class_scores.shape[0]))
    if seq_len <= 1:
        raise RuntimeError(f"clip {sample.clip_id} has insufficient sequence length")

    centers = trace_summary.centers[:seq_len].to(device=device, dtype=torch.float32)
    velocities = trace_summary.velocities[:seq_len].to(device=device, dtype=torch.float32)
    visibility = trace_summary.visibility[:seq_len].to(device=device, dtype=torch.float32)
    trace_features = torch.cat([centers, velocities, visibility], dim=-1)

    sem_text = semantic_summary.text_embeddings[:seq_len].mean(dim=1).to(device=device, dtype=torch.float32)
    sem_scores = semantic_summary.class_scores[:seq_len].mean(dim=1).to(device=device, dtype=torch.float32)
    semantic_features = torch.cat([sem_text, sem_scores], dim=-1)
    if disable_semantics:
        semantic_features = torch.zeros_like(semantic_features)

    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    mask_paths = metadata.get("mask_paths") if isinstance(metadata.get("mask_paths"), list) else []
    target_label_id = metadata.get("target_label_id")
    try:
        target_label_id = int(target_label_id) if target_label_id is not None else None
    except (TypeError, ValueError):
        target_label_id = None

    mask_ratios: list[float] = []
    for i in range(seq_len):
        if i < len(mask_paths):
            mask_ratios.append(_read_mask_ratio(str(mask_paths[i]), target_label_id))
        else:
            mask_ratios.append(mask_ratios[-1] if mask_ratios else 0.0)
    mask_ratio_t = torch.tensor(mask_ratios, device=device, dtype=torch.float32)

    speed = torch.norm(velocities, dim=-1)
    semantic_conf = sem_scores.max(dim=-1).values
    prior_features = torch.stack(
        [
            mask_ratio_t.clamp(0.0, 1.0),
            visibility[:, 0].clamp(0.0, 1.0),
            speed.clamp(0.0, 1.0),
            semantic_conf.clamp(0.0, 1.0),
        ],
        dim=-1,
    )

    if use_teacher_priors:
        teacher_objectness = (0.6 * mask_ratio_t + 0.4 * visibility[:, 0]).clamp(0.0, 1.0)
    else:
        teacher_objectness = visibility[:, 0].clamp(0.0, 1.0)

    return {
        "clip_id": sample.clip_id,
        "seq_len": seq_len,
        "trace_features": trace_features.unsqueeze(0),
        "semantic_features": semantic_features.unsqueeze(0),
        "prior_features": prior_features.unsqueeze(0),
        "teacher_objectness": teacher_objectness.unsqueeze(0),
        "target_trajectory": centers.unsqueeze(0),
        "target_visibility": visibility.unsqueeze(0),
        "target_semantic_probs": sem_scores.unsqueeze(0),
    }


def _safe_zero(device: torch.device) -> torch.Tensor:
    return torch.zeros((), dtype=torch.float32, device=device)


def _detach_memory_state(memory_state: Any) -> Any:
    if memory_state is None:
        return None
    return type(memory_state)(
        keys=memory_state.keys.detach(),
        values=memory_state.values.detach(),
        valid_mask=memory_state.valid_mask.detach(),
    )


def _reappearance_indices(visibility: torch.Tensor, min_gap: int = 1) -> list[int]:
    flags = (visibility >= 0.5).to(dtype=torch.int32).detach().cpu().tolist()
    out: list[int] = []
    min_gap = max(1, int(min_gap))
    for i in range(1, len(flags)):
        if flags[i] != 1:
            continue
        j = i - 1
        gap = 0
        while j >= 0 and flags[j] == 0:
            gap += 1
            j -= 1
        had_visible_before = j >= 0 and flags[j] == 1
        if had_visible_before and gap >= min_gap:
            out.append(i)
    return out


def _disk_free_gb(path: Path) -> float:
    usage = shutil.disk_usage(path)
    return float(usage.free) / float(1024**3)


def _estimate_checkpoint_budget_gb(model_parameters: int, train_mode: bool, total_steps: int, milestone_interval: int) -> tuple[float, float]:
    model_gb = float(model_parameters) * 4.0 / float(1024**3)
    optim_gb = (float(model_parameters) * 8.0 / float(1024**3)) if train_mode else 0.0
    estimated_per_checkpoint = model_gb + optim_gb + 0.25
    milestone_count = 0
    if milestone_interval > 0 and total_steps > 0:
        milestone_count = total_steps // milestone_interval
    estimated_retained = 2 + milestone_count
    estimated_max = estimated_per_checkpoint * float(estimated_retained)
    return estimated_per_checkpoint, estimated_max


def _semantic_lambda_for_step(
    *,
    step: int,
    total_steps: int,
    target_lambda_sem: float,
    disable_semantics: bool,
    enable_warmup: bool,
    warmup_start_ratio: float,
    warmup_end_ratio: float,
) -> float:
    if disable_semantics:
        return 0.0
    target = float(target_lambda_sem)
    if not enable_warmup or total_steps <= 0:
        return target

    start = max(0.0, float(warmup_start_ratio))
    end = max(start, float(warmup_end_ratio))
    progress = float(step) / float(max(1, total_steps))
    if progress <= start:
        return 0.0
    if progress >= end or end <= start:
        return target
    alpha = (progress - start) / max(1e-12, end - start)
    return float(alpha * target)


def _shared_trunk_named_parameters(model: STWMV42) -> list[tuple[str, torch.nn.Parameter]]:
    prefixes = ("seq_input_proj", "seq_input_norm", "seq_backbone")
    named_params: list[tuple[str, torch.nn.Parameter]] = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith(prefixes):
            named_params.append((name, param))
    if named_params:
        return named_params
    fallback = [(f"fallback_param_{idx}", param) for idx, param in enumerate(model.parameters()) if param.requires_grad]
    return fallback


def _shared_trunk_parameters(model: STWMV42) -> list[torch.nn.Parameter]:
    return [param for _, param in _shared_trunk_named_parameters(model)]


def _loss_grads(loss: torch.Tensor, params: list[torch.nn.Parameter]) -> list[torch.Tensor | None]:
    if not isinstance(loss, torch.Tensor) or not loss.requires_grad:
        return [None for _ in params]
    grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
    return [grad for grad in grads]


def _tensor_grad(loss: torch.Tensor, anchor: torch.Tensor | None) -> torch.Tensor | None:
    if anchor is None or not isinstance(anchor, torch.Tensor):
        return None
    if not anchor.requires_grad:
        return None
    if not isinstance(loss, torch.Tensor) or not loss.requires_grad:
        return None
    grad = torch.autograd.grad(loss, [anchor], retain_graph=True, allow_unused=True)[0]
    return grad


def _tensor_norm(grad: torch.Tensor | None) -> float:
    if grad is None:
        return 0.0
    v = grad.detach().float().reshape(-1)
    if v.numel() == 0:
        return 0.0
    return float(torch.linalg.vector_norm(v).cpu().item())


def _tensor_cos(a: torch.Tensor | None, b: torch.Tensor | None) -> float:
    if a is None or b is None:
        return 0.0
    va = a.detach().float().reshape(-1)
    vb = b.detach().float().reshape(-1)
    if va.numel() == 0 or vb.numel() == 0:
        return 0.0
    na = float(torch.linalg.vector_norm(va).cpu().item())
    nb = float(torch.linalg.vector_norm(vb).cpu().item())
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    dot = float(torch.dot(va, vb).cpu().item())
    return float(dot / (na * nb + 1e-12))


def _grad_norm(grads: list[torch.Tensor | None]) -> float:
    total = 0.0
    for grad in grads:
        if grad is None:
            continue
        g = grad.detach().float()
        total += float(torch.sum(g * g).cpu().item())
    return float(math.sqrt(max(0.0, total)))


def _grad_cos(a: list[torch.Tensor | None], b: list[torch.Tensor | None]) -> float:
    dot = 0.0
    na = 0.0
    nb = 0.0
    for ga, gb in zip(a, b):
        if ga is not None:
            gfa = ga.detach().float()
            na += float(torch.sum(gfa * gfa).cpu().item())
        if gb is not None:
            gfb = gb.detach().float()
            nb += float(torch.sum(gfb * gfb).cpu().item())
        if ga is not None and gb is not None:
            dot += float(torch.sum(ga.detach().float() * gb.detach().float()).cpu().item())
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    return float(dot / (math.sqrt(na) * math.sqrt(nb) + 1e-12))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def main() -> None:
    args = build_parser().parse_args()
    _set_seed(args.seed)

    repo_root = Path(__file__).resolve().parents[3]
    evaluator_script = repo_root / "code" / "stwm" / "evaluators" / "eval_mini_val.py"
    protocol_update_script = repo_root / "code" / "stwm" / "tools" / "update_protocol_best_main.py"

    device = _resolve_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_log_path = output_dir / str(args.log_name)
    summary_path = output_dir / str(args.summary_name)
    checkpoint_dir = output_dir / str(args.checkpoint_dir_name)
    latest_checkpoint_path = checkpoint_dir / "latest.pt"
    best_checkpoint_path = checkpoint_dir / "best.pt"

    sample_limit = int(args.sample_limit) if int(args.sample_limit) > 0 else None
    dataset = STWMDataset(args.data_root, manifest=args.manifest, limit=sample_limit)
    samples = [sample for sample in dataset.samples if len(sample.frame_paths) >= 4]
    if not samples:
        raise RuntimeError("No eligible samples for STWM V4.2 real training")

    micro_batch = max(1, int(args.micro_batch_per_gpu))
    grad_accum = max(1, int(args.grad_accum))
    effective_batch = micro_batch * grad_accum
    steps_per_epoch = int(math.ceil(len(samples) / float(max(1, effective_batch))))

    if bool(args.eval_only):
        total_steps = max(1, int(args.steps))
        epoch_steps = 0
    else:
        requested_steps = max(0, int(args.steps))
        epoch_steps = int(math.ceil(max(0.0, float(args.target_epochs)) * float(steps_per_epoch))) if float(args.target_epochs) > 0 else 0
        min_steps = max(requested_steps, int(args.min_optimizer_steps), epoch_steps)
        total_steps = min_steps
        if int(args.max_optimizer_steps) > 0:
            total_steps = min(total_steps, int(args.max_optimizer_steps))
        if total_steps <= 0:
            raise RuntimeError("total optimizer steps resolved to 0; provide positive --steps or budget constraints")

    sample_dataset = _SampleDataset(samples)
    loader_kwargs: dict[str, Any] = {
        "batch_size": micro_batch,
        "shuffle": True,
        "num_workers": max(0, int(args.num_workers)),
        "drop_last": False,
        "collate_fn": _collate_samples,
        "pin_memory": bool(args.pin_memory),
    }
    if loader_kwargs["num_workers"] > 0:
        loader_kwargs["prefetch_factor"] = max(1, int(args.prefetch_factor))
        loader_kwargs["persistent_workers"] = bool(args.persistent_workers)

    train_loader = DataLoader(sample_dataset, **loader_kwargs)
    data_iter = iter(train_loader)

    data_mode = str(args.data_mode).strip().lower()
    resolved_frontend_cache_dir = ""
    resolved_frontend_cache_index = ""
    frontend_cache_reader: _FrontendCacheReader | None = None
    if data_mode == "frontend_cache":
        resolved_frontend_cache_dir = str(args.frontend_cache_dir).strip()
        resolved_frontend_cache_index = str(args.frontend_cache_index).strip()
        if not resolved_frontend_cache_dir:
            raise RuntimeError("--frontend-cache-dir is required when --data-mode frontend_cache")
        if not resolved_frontend_cache_index:
            resolved_frontend_cache_index = str(Path(resolved_frontend_cache_dir) / "index.json")
        frontend_cache_reader = _FrontendCacheReader(
            cache_dir=resolved_frontend_cache_dir,
            index_path=resolved_frontend_cache_index,
            max_shards_in_memory=int(args.frontend_cache_max_shards_in_memory),
        )

    def next_micro_batch() -> list[Any]:
        nonlocal data_iter
        try:
            return next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            return next(data_iter)

    trace_adapter = TraceAdapter()
    semantic_adapter = SemanticAdapter()

    warmup = _build_features_for_sample(
        samples[0],
        trace_adapter=trace_adapter,
        semantic_adapter=semantic_adapter,
        device=device,
        disable_semantics=bool(args.disable_semantics),
        use_teacher_priors=bool(args.use_teacher_priors),
        data_mode=data_mode,
        frontend_cache_reader=frontend_cache_reader,
    )

    config = load_model_config_v4_2(
        args.model_preset,
        trace_dim=int(warmup["trace_features"].shape[-1]),
        semantic_dim=int(warmup["semantic_features"].shape[-1]),
        prior_dim=int(warmup["prior_features"].shape[-1]),
        preset_path=args.preset_file,
    )
    config.activation_checkpointing = bool(args.activation_checkpointing)
    model = STWMV42(config).to(device)

    optimizer = None
    if not bool(args.eval_only):
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(args.learning_rate),
            weight_decay=float(args.weight_decay),
        )

    model_parameters = int(sum(param.numel() for param in model.parameters()))
    rough_budget = int(estimate_v4_2_parameter_budget(config))

    start_step = 0
    best_total_loss = float("inf")
    best_step = 0
    resolved_resume_path = ""

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if str(args.resume_checkpoint).strip():
        resolved_resume_path = str(Path(str(args.resume_checkpoint)).expanduser())
    elif bool(args.auto_resume) and latest_checkpoint_path.exists():
        resolved_resume_path = str(latest_checkpoint_path)

    if resolved_resume_path:
        resume_path = Path(resolved_resume_path)
        if not resume_path.exists():
            raise FileNotFoundError(f"resume checkpoint not found: {resume_path}")
        payload = torch.load(resume_path, map_location=device)
        if isinstance(payload, dict) and "model_state" in payload:
            model.load_state_dict(payload["model_state"], strict=True)
            start_step = int(payload.get("step", 0))
            best_total_loss = float(payload.get("best_total_loss", float("inf")))
            best_step = int(payload.get("best_step", 0))
            if optimizer is not None and "optimizer_state" in payload and payload["optimizer_state"] is not None:
                optimizer.load_state_dict(payload["optimizer_state"])
        elif isinstance(payload, dict):
            model.load_state_dict(payload, strict=True)
            start_step = 0
        else:
            raise RuntimeError(f"unsupported checkpoint payload type: {type(payload)}")

    if start_step >= total_steps:
        print(f"[stwm-v4.2-real] resume step {start_step} already reaches target steps {total_steps}; writing summary only")

    est_ckpt_gb, est_max_retained_gb = _estimate_checkpoint_budget_gb(
        model_parameters=model_parameters,
        train_mode=not bool(args.eval_only),
        total_steps=total_steps,
        milestone_interval=max(0, int(args.milestone_interval)),
    )

    checkpoint_interval = max(0, int(args.checkpoint_interval))
    milestone_interval = max(0, int(args.milestone_interval))

    if checkpoint_interval > 0:
        retention_text = f"latest_every_{checkpoint_interval}+best"
    else:
        retention_text = "latest_on_final+best"
    if milestone_interval > 0:
        retention_text = f"{retention_text}+milestone_every_{milestone_interval}"

    print(f"[stwm-v4.2-real] checkpoint_dir={checkpoint_dir}")
    print(f"[stwm-v4.2-real] retention_policy={retention_text}")
    print(f"[stwm-v4.2-real] checkpoint_interval={checkpoint_interval} milestone_interval={milestone_interval}")
    print(f"[stwm-v4.2-real] est_checkpoint_each_gb={est_ckpt_gb:.2f} est_max_retained_gb={est_max_retained_gb:.2f}")

    prev_identity_cache: dict[str, torch.Tensor] = {}
    memory_state = None
    log_rows: list[dict[str, float | int | str]] = []
    step_times: list[float] = []
    data_times: list[float] = []
    data_ratios: list[float] = []
    max_memory_gb_history: list[float] = []
    samples_processed = 0

    gradient_audit_interval = max(0, int(args.gradient_audit_interval))
    gradient_audit_secondary_every = max(1, int(args.gradient_audit_secondary_every))
    gradient_audit_path: Path | None = None
    if gradient_audit_interval > 0:
        if str(args.gradient_audit_output).strip():
            p = Path(str(args.gradient_audit_output))
            gradient_audit_path = p if p.is_absolute() else (repo_root / p)
        else:
            gradient_audit_path = output_dir / "gradient_audit.json"
    gradient_audit_rows: list[dict[str, Any]] = []
    gradient_audit_cycle = 0
    shared_trunk_named_params = _shared_trunk_named_parameters(model)
    shared_trunk_params = [param for _, param in shared_trunk_named_params]
    primary_anchor_name = "seq_backbone.output_features"
    query_path_anchor_name = "token_time_attention"
    secondary_anchor_name = shared_trunk_named_params[-1][0] if shared_trunk_named_params else ""
    secondary_anchor_param = shared_trunk_named_params[-1][1] if shared_trunk_named_params else None

    protocol_eval_interval = max(0, int(args.protocol_eval_interval))
    protocol_eval_manifest = str(args.protocol_eval_manifest).strip()
    if protocol_eval_manifest:
        p = Path(protocol_eval_manifest)
        protocol_eval_manifest = str(p if p.is_absolute() else (repo_root / p))
    protocol_diag_manifest = str(args.protocol_diagnostics_manifest).strip()
    if protocol_diag_manifest:
        p = Path(protocol_diag_manifest)
        protocol_diag_manifest = str(p if p.is_absolute() else (repo_root / p))
    protocol_best_checkpoint_path = checkpoint_dir / str(args.protocol_best_checkpoint_name)
    protocol_best_sidecar_path = checkpoint_dir / str(args.protocol_best_selection_name)
    protocol_eval_dir = checkpoint_dir / "protocol_eval"
    protocol_eval_rows: list[dict[str, Any]] = []

    amp_enabled = bool(args.bf16 and device.type == "cuda")

    def maybe_save_checkpoint(step: int, step_total_loss: float, force_final: bool = False, force: bool = False) -> None:
        nonlocal best_total_loss
        nonlocal best_step

        if not bool(args.save_checkpoint):
            return

        should_periodic = checkpoint_interval > 0 and step % checkpoint_interval == 0
        should_save = force_final or force or should_periodic
        if not should_save:
            return

        free_gb = _disk_free_gb(output_dir)
        if free_gb < float(args.min_free_disk_gb):
            print(
                f"[stwm-v4.2-real] skip checkpoint at step={step}: free_gb={free_gb:.2f} < min_free_disk_gb={float(args.min_free_disk_gb):.2f}"
            )
            return

        payload = {
            "step": int(step),
            "run_name": args.run_name,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
            "model_config": asdict(config),
            "args": vars(args),
            "best_total_loss": float(best_total_loss),
            "best_step": int(best_step),
        }

        torch.save(payload, latest_checkpoint_path)

        if step_total_loss <= best_total_loss:
            best_total_loss = float(step_total_loss)
            best_step = int(step)
            payload["best_total_loss"] = float(best_total_loss)
            payload["best_step"] = int(best_step)
            torch.save(payload, best_checkpoint_path)

        if milestone_interval > 0 and step % milestone_interval == 0:
            milestone_path = checkpoint_dir / f"milestone_step_{step:06d}.pt"
            torch.save(payload, milestone_path)

        if str(args.checkpoint_name).strip() and force_final:
            alias_path = output_dir / str(args.checkpoint_name)
            torch.save(payload, alias_path)

    def _build_eval_env() -> dict[str, str]:
        env = os.environ.copy()
        code_root = str(repo_root / "code")
        if env.get("PYTHONPATH"):
            env["PYTHONPATH"] = f"{code_root}:{env['PYTHONPATH']}"
        else:
            env["PYTHONPATH"] = code_root
        return env

    def _run_detached_eval(
        *,
        step: int,
        manifest_path: str,
        dataset_name: str,
        max_clips: int,
        run_name_prefix: str,
        env: dict[str, str],
    ) -> Path:
        eval_json = protocol_eval_dir / f"{run_name_prefix}_step_{step:06d}.json"
        run_name = f"{run_name_prefix}_step_{step:06d}"
        subprocess.run(
            [
                sys.executable,
                str(evaluator_script),
                "--data-root",
                str(args.data_root),
                "--manifest",
                str(manifest_path),
                "--dataset",
                str(dataset_name),
                "--max-clips",
                str(int(max_clips)),
                "--obs-steps",
                str(int(args.protocol_eval_obs_steps)),
                "--pred-steps",
                str(int(args.protocol_eval_pred_steps)),
                "--seed",
                str(int(args.protocol_eval_seed)),
                "--checkpoint",
                str(latest_checkpoint_path),
                "--model-preset",
                str(args.model_preset),
                "--preset-file",
                str(args.preset_file),
                "--protocol-version",
                str(args.protocol_version),
                "--run-name",
                run_name,
                "--output",
                str(eval_json),
            ],
            check=True,
            env=env,
        )
        return eval_json

    def maybe_run_protocol_eval(step: int) -> None:
        if bool(args.eval_only):
            return
        if protocol_eval_interval <= 0:
            return
        if step % protocol_eval_interval != 0:
            return

        maybe_save_checkpoint(step=step, step_total_loss=float(log_rows[-1].get("total_loss", 0.0)) if log_rows else 0.0, force=True)
        if not latest_checkpoint_path.exists():
            protocol_eval_rows.append(
                {
                    "step": int(step),
                    "kind": "official_main",
                    "status": "skipped",
                    "reason": "latest_checkpoint_missing",
                    "latest_checkpoint": str(latest_checkpoint_path),
                }
            )
            return

        protocol_eval_dir.mkdir(parents=True, exist_ok=True)
        env = _build_eval_env()

        main_eval_json = protocol_eval_dir / f"{args.protocol_eval_run_name}_step_{step:06d}.json"
        try:
            if not protocol_eval_manifest:
                raise FileNotFoundError("protocol_eval_manifest_empty")
            if not Path(protocol_eval_manifest).exists():
                raise FileNotFoundError(f"protocol_eval_manifest_not_found:{protocol_eval_manifest}")

            main_eval_json = _run_detached_eval(
                step=int(step),
                manifest_path=str(protocol_eval_manifest),
                dataset_name=str(args.protocol_eval_dataset),
                max_clips=int(args.protocol_eval_max_clips),
                run_name_prefix=str(args.protocol_eval_run_name),
                env=env,
            )

            subprocess.run(
                [
                    sys.executable,
                    str(protocol_update_script),
                    "--checkpoint-dir",
                    str(checkpoint_dir),
                    "--candidate-checkpoint",
                    str(latest_checkpoint_path),
                    "--eval-summary",
                    str(main_eval_json),
                    "--output-checkpoint",
                    str(protocol_best_checkpoint_path),
                    "--selection-sidecar",
                    str(protocol_best_sidecar_path),
                ],
                check=True,
                env=env,
            )

            action = ""
            improved = False
            if protocol_best_sidecar_path.exists():
                sidecar = json.loads(protocol_best_sidecar_path.read_text())
                action = str(sidecar.get("action", ""))
                improved = bool(sidecar.get("improved", False))
            protocol_eval_rows.append(
                {
                    "step": int(step),
                    "kind": "official_main",
                    "status": "ok",
                    "manifest": str(protocol_eval_manifest),
                    "eval_summary": str(main_eval_json),
                    "candidate_checkpoint": str(latest_checkpoint_path),
                    "official_best": str(protocol_best_checkpoint_path),
                    "selection_sidecar": str(protocol_best_sidecar_path),
                    "action": action,
                    "improved": bool(improved),
                }
            )
        except Exception as exc:
            protocol_eval_rows.append(
                {
                    "step": int(step),
                    "kind": "official_main",
                    "status": "error",
                    "manifest": str(protocol_eval_manifest),
                    "eval_summary": str(main_eval_json),
                    "candidate_checkpoint": str(latest_checkpoint_path),
                    "error": str(exc),
                }
            )

        if not protocol_diag_manifest:
            return

        diag_eval_json = protocol_eval_dir / f"{args.protocol_diagnostics_run_name}_step_{step:06d}.json"
        try:
            if not Path(protocol_diag_manifest).exists():
                raise FileNotFoundError(f"protocol_diag_manifest_not_found:{protocol_diag_manifest}")

            diag_eval_json = _run_detached_eval(
                step=int(step),
                manifest_path=str(protocol_diag_manifest),
                dataset_name=str(args.protocol_diagnostics_dataset),
                max_clips=int(args.protocol_diagnostics_max_clips),
                run_name_prefix=str(args.protocol_diagnostics_run_name),
                env=env,
            )

            protocol_eval_rows.append(
                {
                    "step": int(step),
                    "kind": "diagnostic_eventful",
                    "status": "ok",
                    "manifest": str(protocol_diag_manifest),
                    "eval_summary": str(diag_eval_json),
                    "candidate_checkpoint": str(latest_checkpoint_path),
                    "official_best_unchanged": True,
                }
            )
        except Exception as exc:
            protocol_eval_rows.append(
                {
                    "step": int(step),
                    "kind": "diagnostic_eventful",
                    "status": "error",
                    "manifest": str(protocol_diag_manifest),
                    "eval_summary": str(diag_eval_json),
                    "candidate_checkpoint": str(latest_checkpoint_path),
                    "official_best_unchanged": True,
                    "error": str(exc),
                }
            )

    if start_step < total_steps:
        for step in range(int(start_step) + 1, int(total_steps) + 1):
            step_start = time.perf_counter()
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)

            lambda_sem_effective = _semantic_lambda_for_step(
                step=int(step),
                total_steps=int(total_steps),
                target_lambda_sem=float(args.lambda_sem),
                disable_semantics=bool(args.disable_semantics),
                enable_warmup=bool(args.semantic_warmup),
                warmup_start_ratio=float(args.semantic_warmup_start_ratio),
                warmup_end_ratio=float(args.semantic_warmup_end_ratio),
            )
            do_grad_audit = bool(
                (not args.eval_only)
                and gradient_audit_interval > 0
                and step % gradient_audit_interval == 0
            )
            gradient_audit_row: dict[str, Any] | None = None

            if bool(args.eval_only):
                model.eval()
            else:
                model.train()

            data_time_s = 0.0
            metric_sum: dict[str, float] = {}
            metric_count = 0
            first_clip_id = ""

            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)

            for _ in range(grad_accum):
                fetch_t0 = time.perf_counter()
                micro_samples = next_micro_batch()
                data_time_s += time.perf_counter() - fetch_t0

                if not isinstance(micro_samples, list):
                    micro_samples = list(micro_samples)

                micro_divisor = max(1, len(micro_samples))
                for sample in micro_samples:
                    if not first_clip_id:
                        first_clip_id = str(sample.clip_id)

                    feature_t0 = time.perf_counter()
                    batch = _build_features_for_sample(
                        sample,
                        trace_adapter=trace_adapter,
                        semantic_adapter=semantic_adapter,
                        device=device,
                        disable_semantics=bool(args.disable_semantics),
                        use_teacher_priors=bool(args.use_teacher_priors),
                        data_mode=data_mode,
                        frontend_cache_reader=frontend_cache_reader,
                    )
                    data_time_s += time.perf_counter() - feature_t0

                    use_memory = not bool(args.disable_identity_memory)
                    model_prior_features = batch["prior_features"]
                    model_teacher_objectness = batch["teacher_objectness"]
                    model_semantic_features = batch["semantic_features"]

                    object_bias_alpha = float(args.object_bias_alpha)
                    object_bias_alpha = max(0.0, min(1.0, object_bias_alpha))
                    if int(args.object_bias_delay_steps) > 0 and int(step) <= int(args.object_bias_delay_steps):
                        object_bias_alpha = 0.0
                    if bool(args.neutralize_object_bias):
                        object_bias_alpha = 0.0

                    model_prior_features = model_prior_features * object_bias_alpha
                    neutral_teacher_objectness = torch.full_like(model_teacher_objectness, 0.5)
                    model_teacher_objectness = neutral_teacher_objectness + object_bias_alpha * (
                        model_teacher_objectness - neutral_teacher_objectness
                    )

                    object_bias_gate_mean = 1.0
                    if bool(args.object_bias_gated):
                        visibility_source = batch["target_visibility"][..., 0].clamp(0.0, 1.0)
                        gate = (visibility_source >= float(args.object_bias_gate_threshold)).to(dtype=model_prior_features.dtype)
                        object_bias_gate_mean = float(gate.mean().detach().float().cpu())
                        model_prior_features = model_prior_features * gate.unsqueeze(-1)
                        model_teacher_objectness = neutral_teacher_objectness + gate * (
                            model_teacher_objectness - neutral_teacher_objectness
                        )

                    qstr_route_mean = 0.0
                    if bool(args.qstr_enable):
                        # QSTR: keep neutral path while adding query-conditioned semantic residual routing.
                        sem_feat = model_semantic_features
                        bsz, _, sdim = sem_feat.shape
                        query_idx = torch.argmax(model_teacher_objectness, dim=-1)
                        query_feat = torch.gather(
                            sem_feat,
                            dim=1,
                            index=query_idx.view(bsz, 1, 1).expand(-1, 1, sdim),
                        ).squeeze(1)

                        neutral_sem = sem_feat.mean(dim=1, keepdim=True)
                        sem_residual = sem_feat - neutral_sem

                        sem_norm = F.normalize(sem_feat, dim=-1)
                        query_norm = F.normalize(query_feat, dim=-1).unsqueeze(1)
                        route_logits = torch.sum(sem_norm * query_norm, dim=-1, keepdim=True)
                        route_temp = max(1e-3, float(args.qstr_route_temperature))
                        route_gate = torch.sigmoid(route_logits / route_temp)
                        qstr_route_mean = float(route_gate.mean().detach().float().cpu())

                        neutral_w = max(0.0, float(args.qstr_neutral_path_weight))
                        residual_scale = max(0.0, float(args.qstr_residual_scale))
                        model_semantic_features = neutral_w * sem_feat + residual_scale * route_gate * sem_residual

                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=amp_enabled):
                        outputs = model(
                            trace_features=batch["trace_features"],
                            semantic_features=model_semantic_features,
                            prior_features=model_prior_features,
                            teacher_objectness=model_teacher_objectness,
                            memory_state=memory_state,
                            use_memory=use_memory,
                            update_memory=use_memory,
                            return_shared_trunk_features=do_grad_audit,
                        )

                        if use_memory:
                            memory_state = _detach_memory_state(outputs["memory_state"])
                        else:
                            memory_state = None

                        pred_traj = torch.sigmoid(outputs["trajectory"])
                        traj_loss = F.smooth_l1_loss(pred_traj, batch["target_trajectory"])
                        vis_loss = F.binary_cross_entropy_with_logits(outputs["visibility"], batch["target_visibility"])

                        frame_l1 = torch.abs(pred_traj - batch["target_trajectory"]).mean(dim=-1)
                        trajectory_l1 = float(frame_l1.mean().detach().float().cpu())

                        if bool(args.disable_semantics):
                            sem_loss = _safe_zero(device)
                        else:
                            token_attn = outputs["token_time_attention"]
                            token_targets = torch.einsum("bnt,btc->bnc", token_attn, batch["target_semantic_probs"])
                            token_targets = token_targets / token_targets.sum(dim=-1, keepdim=True).clamp(min=1e-6)
                            sem_log_probs = F.log_softmax(outputs["semantic_logits"], dim=-1)
                            sem_loss = -(token_targets * sem_log_probs).sum(dim=-1).mean()

                        if bool(args.disable_identity_memory):
                            reid_loss = _safe_zero(device)
                        else:
                            clip_id = str(batch["clip_id"])
                            curr = outputs["identity_embeddings"][0]
                            prev = prev_identity_cache.get(clip_id)
                            if prev is not None:
                                prev = prev.to(device=device, dtype=curr.dtype)
                                n = min(curr.shape[0], prev.shape[0])
                                logits = torch.matmul(curr[:n], prev[:n].transpose(0, 1)) / float(args.contrastive_temperature)
                                labels = torch.arange(n, device=device)
                                reid_loss = F.cross_entropy(logits, labels)
                            else:
                                reid_loss = _safe_zero(device)
                            prev_identity_cache[clip_id] = curr.detach().float().cpu()

                        token_attn = outputs["token_time_attention"]
                        q_idx = torch.argmax(batch["teacher_objectness"], dim=-1)
                        gather = torch.gather(
                            token_attn,
                            dim=-1,
                            index=q_idx.view(-1, 1, 1).expand(token_attn.shape[0], token_attn.shape[1], 1),
                        ).squeeze(-1)
                        query_loss = -torch.log(gather.max(dim=1).values.clamp(min=1e-6)).mean()

                        temporal_semantic_consistency_loss = _safe_zero(device)
                        if bool(args.qstr_enable) and float(args.qstr_temporal_consistency_weight) > 0.0:
                            sem_probs = F.softmax(outputs["semantic_logits"], dim=-1)
                            frame_sem_probs = torch.einsum("bnt,bnc->btc", token_attn, sem_probs)
                            if frame_sem_probs.shape[1] > 1:
                                temporal_semantic_consistency_loss = F.smooth_l1_loss(
                                    frame_sem_probs[:, 1:, :],
                                    frame_sem_probs[:, :-1, :],
                                )

                        query_token_index = int(torch.argmax(outputs["query_token_logits"][0]).item())
                        query_frame_scores = token_attn[0, query_token_index]
                        query_frame_idx = int(torch.argmax(query_frame_scores).item())
                        query_frame_idx = max(0, min(query_frame_idx, frame_l1.shape[1] - 1))
                        query_localization_error = float(frame_l1[0, query_frame_idx].detach().float().cpu())

                        q_pred = pred_traj[0, query_frame_idx].detach().float().cpu()
                        q_gt = batch["target_trajectory"][0, query_frame_idx].detach().float().cpu()

                        reconnect_loss = _safe_zero(device)
                        reconnect_success = 0.0
                        reconnect_min_error = 0.0
                        reappearance_count = 0
                        has_reappearance_event = 0.0

                        reappear = _reappearance_indices(batch["target_visibility"][0, :, 0], min_gap=1)
                        if reappear:
                            has_reappearance_event = 1.0
                            reappearance_count = int(len(reappear))
                            mins: list[float] = []
                            for idx in reappear:
                                end = min(frame_l1.shape[1], int(idx) + max(1, int(args.reconnect_window)))
                                w = frame_l1[0, int(idx):end]
                                if w.numel() > 0:
                                    mins.append(float(torch.min(w).detach().float().cpu()))
                            if mins:
                                reconnect_min_error = float(min(mins))
                                reconnect_success = float(any(x <= float(args.reconnect_threshold) for x in mins))

                        if bool(args.enable_reconnect_loss) and use_memory:
                            gate_mean = float(outputs["memory_diagnostics"].get("memory_gate_mean", 0.0))
                            low_vis = (batch["target_visibility"][..., 0] < 0.5).float().mean()
                            reconnect_target = low_vis.detach()
                            reconnect_loss = (reconnect_target - float(gate_mean)) ** 2

                        total_loss = (
                            float(args.lambda_traj) * traj_loss
                            + float(args.lambda_vis) * vis_loss
                            + float(lambda_sem_effective) * sem_loss
                            + float(args.lambda_reid) * reid_loss
                            + float(args.lambda_query) * query_loss
                            + float(args.lambda_reconnect) * reconnect_loss
                            + float(args.qstr_temporal_consistency_weight) * temporal_semantic_consistency_loss
                        )

                    if do_grad_audit and gradient_audit_row is None:
                        gradient_audit_cycle += 1
                        audit_t0 = time.perf_counter()
                        audit_mem_before_mb = 0.0
                        if device.type == "cuda":
                            audit_mem_before_mb = float(torch.cuda.memory_allocated(device) / float(1024**2))

                        primary_anchor = outputs.get("shared_trunk_features")
                        if isinstance(primary_anchor, torch.Tensor):
                            g_traj = _tensor_grad(traj_loss, primary_anchor)
                            g_query = _tensor_grad(query_loss, primary_anchor)
                            g_sem = _tensor_grad(sem_loss, primary_anchor)
                            primary_anchor_shape = list(primary_anchor.shape)
                        else:
                            g_traj = None
                            g_query = None
                            g_sem = None
                            primary_anchor_shape = []

                        query_path_anchor = outputs.get("token_time_attention")
                        if isinstance(query_path_anchor, torch.Tensor):
                            qpath_g_query = _tensor_grad(query_loss, query_path_anchor)
                            qpath_g_sem = _tensor_grad(sem_loss, query_path_anchor)
                            qpath_g_traj = _tensor_grad(traj_loss, query_path_anchor)
                            query_path_anchor_shape = list(query_path_anchor.shape)
                        else:
                            qpath_g_query = None
                            qpath_g_sem = None
                            qpath_g_traj = None
                            query_path_anchor_shape = []

                        secondary_due = bool(
                            secondary_anchor_param is not None
                            and gradient_audit_cycle % gradient_audit_secondary_every == 0
                        )
                        g_traj_secondary = None
                        g_query_secondary = None
                        g_sem_secondary = None
                        if secondary_due and secondary_anchor_param is not None:
                            g_traj_secondary = _loss_grads(traj_loss, [secondary_anchor_param])[0]
                            g_query_secondary = _loss_grads(query_loss, [secondary_anchor_param])[0]
                            g_sem_secondary = _loss_grads(sem_loss, [secondary_anchor_param])[0]

                        audit_mem_after_mb = audit_mem_before_mb
                        if device.type == "cuda":
                            audit_mem_after_mb = float(torch.cuda.memory_allocated(device) / float(1024**2))
                        audit_time_ms = float((time.perf_counter() - audit_t0) * 1000.0)

                        gradient_audit_row = {
                            "step": int(step),
                            "clip_id": str(batch["clip_id"]),
                            "lambda_sem_effective": float(lambda_sem_effective),
                            "audit_cycle": int(gradient_audit_cycle),
                            "primary_anchor": str(primary_anchor_name),
                            "primary_anchor_shape": primary_anchor_shape,
                            "g_traj_norm": float(_tensor_norm(g_traj)),
                            "g_query_norm": float(_tensor_norm(g_query)),
                            "g_sem_norm": float(_tensor_norm(g_sem)),
                            "cos_sem_traj": float(_tensor_cos(g_sem, g_traj)),
                            "cos_sem_query": float(_tensor_cos(g_sem, g_query)),
                            "query_path_anchor": str(query_path_anchor_name),
                            "query_path_anchor_shape": query_path_anchor_shape,
                            "qpath_g_query_norm": float(_tensor_norm(qpath_g_query)),
                            "qpath_g_sem_norm": float(_tensor_norm(qpath_g_sem)),
                            "qpath_g_traj_norm": float(_tensor_norm(qpath_g_traj)),
                            "qpath_cos_sem_query": float(_tensor_cos(qpath_g_sem, qpath_g_query)),
                            "qpath_cos_traj_query": float(_tensor_cos(qpath_g_traj, qpath_g_query)),
                            "qpath_cos_sem_traj": float(_tensor_cos(qpath_g_sem, qpath_g_traj)),
                            "secondary_anchor": str(secondary_anchor_name),
                            "secondary_due": bool(secondary_due),
                            "secondary_g_traj_norm": float(_grad_norm([g_traj_secondary])) if secondary_due else 0.0,
                            "secondary_g_query_norm": float(_grad_norm([g_query_secondary])) if secondary_due else 0.0,
                            "secondary_g_sem_norm": float(_grad_norm([g_sem_secondary])) if secondary_due else 0.0,
                            "secondary_cos_sem_traj": float(_grad_cos([g_sem_secondary], [g_traj_secondary])) if secondary_due else 0.0,
                            "secondary_cos_sem_query": float(_grad_cos([g_sem_secondary], [g_query_secondary])) if secondary_due else 0.0,
                            "audit_time_ms": float(audit_time_ms),
                            "audit_memory_delta_mb": float(audit_mem_after_mb - audit_mem_before_mb),
                        }

                    loss_div = float(grad_accum * micro_divisor)
                    loss_scaled = total_loss / max(1.0, loss_div)
                    if optimizer is not None:
                        loss_scaled.backward()

                    attn_mean = token_attn.mean(dim=1)
                    teacher_obj = batch["teacher_objectness"]
                    low_obj = (teacher_obj < 0.3).float()
                    high_obj = (teacher_obj >= 0.3).float()
                    bg_attn = float((attn_mean * low_obj).sum().detach().float().cpu() / low_obj.sum().clamp(min=1.0).detach().float().cpu())
                    fg_attn = float((attn_mean * high_obj).sum().detach().float().cpu() / high_obj.sum().clamp(min=1.0).detach().float().cpu())

                    tdiag = outputs["tokenizer_diagnostics"]
                    mdiag = outputs["memory_diagnostics"]

                    row_values = {
                        "total_loss": float(total_loss.detach().float().cpu()),
                        "trajectory_loss": float(traj_loss.detach().float().cpu()),
                        "trajectory_l1": float(trajectory_l1),
                        "visibility_loss": float(vis_loss.detach().float().cpu()),
                        "semantic_loss": float(sem_loss.detach().float().cpu()),
                        "qstr_temporal_consistency_loss": float(temporal_semantic_consistency_loss.detach().float().cpu()),
                        "qstr_temporal_consistency_weight": float(args.qstr_temporal_consistency_weight),
                        "lambda_sem_effective": float(lambda_sem_effective),
                        "reid_loss": float(reid_loss.detach().float().cpu()),
                        "query_loss": float(query_loss.detach().float().cpu()),
                        "query_localization_error": float(query_localization_error),
                        "query_frame_idx": float(query_frame_idx),
                        "query_token_index": float(query_token_index),
                        "query_pred_x": float(q_pred[0].item()),
                        "query_pred_y": float(q_pred[1].item()),
                        "query_gt_x": float(q_gt[0].item()),
                        "query_gt_y": float(q_gt[1].item()),
                        "query_traj_gap": float(query_localization_error - trajectory_l1),
                        "object_bias_alpha_effective": float(object_bias_alpha),
                        "object_bias_gate_mean": float(object_bias_gate_mean),
                        "reconnect_loss": float(reconnect_loss.detach().float().cpu()),
                        "has_reappearance_event": float(has_reappearance_event),
                        "reappearance_count": float(reappearance_count),
                        "reconnect_success": float(reconnect_success),
                        "reconnect_min_error": float(reconnect_min_error),
                        "assignment_entropy": float(tdiag.get("assignment_entropy", 0.0)),
                        "token_usage_entropy": float(tdiag.get("token_usage_entropy", 0.0)),
                        "objectness_mean": float(tdiag.get("objectness_mean", 0.0)),
                        "memory_gate_mean": float(mdiag.get("memory_gate_mean", 0.0)),
                        "bg_fg_attention_ratio": float(bg_attn / max(fg_attn, 1e-6)),
                        "qstr_enabled": float(bool(args.qstr_enable)),
                        "qstr_route_mean": float(qstr_route_mean),
                    }
                    for key, value in row_values.items():
                        metric_sum[key] = metric_sum.get(key, 0.0) + float(value)
                    metric_count += 1
                    samples_processed += 1

            grad_norm = 0.0
            if optimizer is not None:
                grad_norm_t = torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.max_grad_norm))
                grad_norm = float(grad_norm_t.detach().float().cpu())
                optimizer.step()

            step_time_s = float(time.perf_counter() - step_start)
            data_wait_ratio = float(data_time_s / step_time_s) if step_time_s > 0 else 0.0
            if device.type == "cuda":
                peak_memory_gb = float(torch.cuda.max_memory_allocated(device) / float(1024**3))
            else:
                peak_memory_gb = 0.0
            disk_free_gb = _disk_free_gb(output_dir)

            metric_div = float(max(1, metric_count))
            row = {
                "step": int(step),
                "clip_id": str(first_clip_id or samples[(step - 1) % len(samples)].clip_id),
                "effective_batch": int(effective_batch),
                "micro_batch_per_gpu": int(micro_batch),
                "grad_accum": int(grad_accum),
                "num_workers": int(loader_kwargs["num_workers"]),
                "prefetch_factor": int(loader_kwargs.get("prefetch_factor", 0)),
                "persistent_workers": int(bool(loader_kwargs.get("persistent_workers", False))),
                "pin_memory": int(bool(loader_kwargs.get("pin_memory", False))),
                "bf16": int(bool(args.bf16)),
                "activation_checkpointing": int(bool(args.activation_checkpointing)),
                "step_time_s": float(step_time_s),
                "data_time_s": float(data_time_s),
                "data_wait_ratio": float(data_wait_ratio),
                "gpu_peak_memory_gb": float(peak_memory_gb),
                "disk_free_gb": float(disk_free_gb),
                "grad_norm": float(grad_norm),
            }
            for key, value in metric_sum.items():
                row[key] = float(value / metric_div)

            log_rows.append(row)
            step_times.append(float(step_time_s))
            data_times.append(float(data_time_s))
            data_ratios.append(float(data_wait_ratio))
            max_memory_gb_history.append(float(peak_memory_gb))

            if gradient_audit_row is not None:
                gradient_audit_rows.append(gradient_audit_row)
                if gradient_audit_path is not None:
                    audit_time_vals = [float(x.get("audit_time_ms", 0.0)) for x in gradient_audit_rows]
                    audit_mem_vals = [float(x.get("audit_memory_delta_mb", 0.0)) for x in gradient_audit_rows]
                    _write_json(
                        gradient_audit_path,
                        {
                            "run_name": str(args.run_name),
                            "seed": int(args.seed),
                            "primary_anchor": str(primary_anchor_name),
                            "query_path_anchor": str(query_path_anchor_name),
                            "secondary_anchor": str(secondary_anchor_name),
                            "interval": int(gradient_audit_interval),
                            "secondary_every_audit_cycles": int(gradient_audit_secondary_every),
                            "average_audit_time_ms": float(sum(audit_time_vals) / max(1, len(audit_time_vals))),
                            "average_audit_memory_delta_mb": float(sum(audit_mem_vals) / max(1, len(audit_mem_vals))),
                            "rows": gradient_audit_rows,
                        },
                    )

            with train_log_path.open("a", encoding="utf-8") as fp:
                fp.write(json.dumps(row) + "\n")

            maybe_save_checkpoint(step=step, step_total_loss=float(row.get("total_loss", 0.0)), force_final=(step == total_steps))
            maybe_run_protocol_eval(step=step)

    if bool(args.save_checkpoint) and latest_checkpoint_path.exists():
        if not best_checkpoint_path.exists():
            payload = torch.load(latest_checkpoint_path, map_location="cpu")
            torch.save(payload, best_checkpoint_path)

    def _avg(key: str) -> float:
        vals = [float(x.get(key, 0.0)) for x in log_rows]
        return float(sum(vals) / max(1, len(vals)))

    def _pct(values: list[float], q: float) -> float:
        if not values:
            return 0.0
        if len(values) == 1:
            return float(values[0])
        return float(np.percentile(np.asarray(values, dtype=np.float64), q))

    first = log_rows[0] if log_rows else {}
    last = log_rows[-1] if log_rows else {}

    summary = {
        "run_name": args.run_name,
        "mode": "eval" if bool(args.eval_only) else "train",
        "steps": int(total_steps),
        "model_preset": args.model_preset,
        "model_parameters": model_parameters,
        "rough_parameter_budget": rough_budget,
        "ablation": {
            "disable_semantics": bool(args.disable_semantics),
            "disable_identity_memory": bool(args.disable_identity_memory),
            "neutralize_object_bias": bool(args.neutralize_object_bias),
            "object_bias_alpha": float(args.object_bias_alpha),
            "object_bias_delay_steps": int(args.object_bias_delay_steps),
            "object_bias_gated": bool(args.object_bias_gated),
            "object_bias_gate_threshold": float(args.object_bias_gate_threshold),
            "use_teacher_priors": bool(args.use_teacher_priors),
            "enable_reconnect_loss": bool(args.enable_reconnect_loss),
            "qstr_enable": bool(args.qstr_enable),
            "qstr_residual_scale": float(args.qstr_residual_scale),
            "qstr_neutral_path_weight": float(args.qstr_neutral_path_weight),
            "qstr_route_temperature": float(args.qstr_route_temperature),
            "qstr_temporal_consistency_weight": float(args.qstr_temporal_consistency_weight),
        },
        "loss_weights": {
            "lambda_traj": float(args.lambda_traj),
            "lambda_vis": float(args.lambda_vis),
            "lambda_sem_target": float(args.lambda_sem),
            "lambda_reid": float(args.lambda_reid),
            "lambda_query": float(args.lambda_query),
            "lambda_reconnect": float(args.lambda_reconnect),
            "qstr_temporal_consistency_weight": float(args.qstr_temporal_consistency_weight),
            "semantic_warmup": bool(args.semantic_warmup),
            "semantic_warmup_start_ratio": float(args.semantic_warmup_start_ratio),
            "semantic_warmup_end_ratio": float(args.semantic_warmup_end_ratio),
        },
        "average_losses": {
            "total": _avg("total_loss"),
            "trajectory": _avg("trajectory_loss"),
            "trajectory_l1": _avg("trajectory_l1"),
            "visibility": _avg("visibility_loss"),
            "semantic": _avg("semantic_loss"),
            "qstr_temporal_consistency": _avg("qstr_temporal_consistency_loss"),
            "lambda_sem_effective": _avg("lambda_sem_effective"),
            "reid": _avg("reid_loss"),
            "query": _avg("query_loss"),
            "query_localization_error": _avg("query_localization_error"),
            "query_traj_gap": _avg("query_traj_gap"),
        },
        "diagnostics": {
            "assignment_entropy": _avg("assignment_entropy"),
            "token_usage_entropy": _avg("token_usage_entropy"),
            "objectness_mean": _avg("objectness_mean"),
            "memory_gate_mean": _avg("memory_gate_mean"),
            "bg_fg_attention_ratio": _avg("bg_fg_attention_ratio"),
            "qstr_route_mean": _avg("qstr_route_mean"),
            "reappearance_event_ratio": _avg("has_reappearance_event"),
            "reconnect_success_rate": _avg("reconnect_success"),
            "reconnect_min_error": _avg("reconnect_min_error"),
            "gpu_peak_memory_gb": _avg("gpu_peak_memory_gb"),
            "data_wait_ratio": _avg("data_wait_ratio"),
        },
        "risk_flags": {
            "tokenizer_collapse_risk": bool(_avg("token_usage_entropy") < 0.25 or _avg("assignment_entropy") < 0.25),
            "background_bias_risk": bool(_avg("bg_fg_attention_ratio") > 1.1),
            "memory_inactive_risk": bool((not args.disable_identity_memory) and _avg("memory_gate_mean") < 0.1),
            "semantic_decorative_risk": bool((not args.disable_semantics) and float(last.get("semantic_loss", 0.0)) >= float(first.get("semantic_loss", 0.0))),
            "identity_decorative_risk": bool((not args.disable_identity_memory) and float(last.get("reid_loss", 0.0)) >= float(first.get("reid_loss", 0.0))),
        },
        "budget": {
            "sample_count": int(len(samples)),
            "steps_per_epoch": int(steps_per_epoch),
            "target_epochs": float(args.target_epochs),
            "steps_for_target_epochs": int(epoch_steps),
            "min_optimizer_steps": int(args.min_optimizer_steps),
            "max_optimizer_steps": int(args.max_optimizer_steps),
            "requested_steps": int(args.steps),
            "resolved_optimizer_steps": int(total_steps),
            "micro_batch_per_gpu": int(micro_batch),
            "grad_accum": int(grad_accum),
            "effective_batch": int(effective_batch),
        },
        "precision": {
            "bf16": bool(args.bf16),
            "activation_checkpointing": bool(args.activation_checkpointing),
            "device": str(device),
        },
        "dataloader": {
            "num_workers": int(loader_kwargs["num_workers"]),
            "prefetch_factor": int(loader_kwargs.get("prefetch_factor", 0)),
            "persistent_workers": bool(loader_kwargs.get("persistent_workers", False)),
            "pin_memory": bool(loader_kwargs.get("pin_memory", False)),
        },
        "data": {
            "data_mode": str(data_mode),
            "frontend_cache_dir": str(resolved_frontend_cache_dir),
            "frontend_cache_index": str(resolved_frontend_cache_index),
            "frontend_cache_max_shards_in_memory": int(args.frontend_cache_max_shards_in_memory),
        },
        "runtime": {
            "step_time_p50_s": _pct(step_times, 50.0),
            "step_time_p95_s": _pct(step_times, 95.0),
            "data_time_p50_s": _pct(data_times, 50.0),
            "data_wait_ratio_p50": _pct(data_ratios, 50.0),
            "data_wait_ratio_p95": _pct(data_ratios, 95.0),
            "gpu_peak_memory_gb_max": float(max(max_memory_gb_history) if max_memory_gb_history else 0.0),
            "samples_processed": int(samples_processed),
        },
        "checkpoint_policy": {
            "checkpoint_dir": str(checkpoint_dir),
            "retention": retention_text,
            "checkpoint_interval": int(checkpoint_interval),
            "milestone_interval": int(milestone_interval),
            "latest": str(latest_checkpoint_path) if latest_checkpoint_path.exists() else "",
            "best": str(best_checkpoint_path) if best_checkpoint_path.exists() else "",
            "official_best_protocol_main": str(protocol_best_checkpoint_path) if protocol_best_checkpoint_path.exists() else "",
            "official_best_selection_sidecar": str(protocol_best_sidecar_path) if protocol_best_sidecar_path.exists() else "",
            "min_free_disk_gb": float(args.min_free_disk_gb),
            "estimated_checkpoint_each_gb": float(est_ckpt_gb),
            "estimated_max_retained_gb": float(est_max_retained_gb),
        },
        "protocol_best_policy": {
            "protocol_eval_interval": int(protocol_eval_interval),
            "protocol_eval_manifest": str(protocol_eval_manifest),
            "protocol_eval_dataset": str(args.protocol_eval_dataset),
            "protocol_diagnostics_manifest": str(protocol_diag_manifest),
            "protocol_diagnostics_dataset": str(args.protocol_diagnostics_dataset),
            "protocol_version": str(args.protocol_version),
            "official_best_checkpoint_name": str(args.protocol_best_checkpoint_name),
            "official_best_selection_name": str(args.protocol_best_selection_name),
            "records": protocol_eval_rows,
        },
        "gradient_audit": {
            "enabled": bool(gradient_audit_interval > 0),
            "interval": int(gradient_audit_interval),
            "secondary_every_audit_cycles": int(gradient_audit_secondary_every),
            "output": str(gradient_audit_path) if gradient_audit_path is not None else "",
            "primary_anchor": str(primary_anchor_name),
            "query_path_anchor": str(query_path_anchor_name),
            "secondary_anchor": str(secondary_anchor_name),
            "average_audit_time_ms": float(sum(float(x.get("audit_time_ms", 0.0)) for x in gradient_audit_rows) / max(1, len(gradient_audit_rows))),
            "average_audit_memory_delta_mb": float(sum(float(x.get("audit_memory_delta_mb", 0.0)) for x in gradient_audit_rows) / max(1, len(gradient_audit_rows))),
            "rows": int(len(gradient_audit_rows)),
        },
        "paths": {
            "train_log": str(train_log_path),
            "summary": str(summary_path),
            "checkpoint": str(latest_checkpoint_path) if latest_checkpoint_path.exists() else "",
        },
        "resume": {
            "requested_resume_checkpoint": str(args.resume_checkpoint),
            "auto_resume": bool(args.auto_resume),
            "resolved_resume_checkpoint": str(resolved_resume_path),
            "start_step": int(start_step),
            "best_step": int(best_step),
            "best_total_loss": float(best_total_loss if math.isfinite(best_total_loss) else _avg("total_loss")),
            "eval_only": bool(args.eval_only),
            "log_name": str(args.log_name),
        },
    }

    summary_path.write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
