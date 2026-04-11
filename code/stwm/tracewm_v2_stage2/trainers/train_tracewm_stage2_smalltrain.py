#!/usr/bin/env python3
from __future__ import annotations

from argparse import ArgumentParser
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple
import json
import os
import random
import shutil

import numpy as np
import torch
from torch.utils.data import DataLoader

from stwm.tracewm_v2.models.causal_trace_transformer import (
    TraceCausalTransformer,
    build_tracewm_v2_config,
)
from stwm.tracewm_v2.tools.run_stage1_v2_scientific_revalidation import _load_runtime_config
from stwm.tracewm_v2_stage2.datasets.stage2_semantic_dataset import (
    Stage2SemanticDataset,
    Stage2SemanticDatasetConfig,
    stage2_semantic_collate_fn,
)
from stwm.tracewm_v2_stage2.models.semantic_encoder import SemanticEncoder, SemanticEncoderConfig
from stwm.tracewm_v2_stage2.models.semantic_fusion import SemanticFusion, SemanticFusionConfig


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args() -> Any:
    p = ArgumentParser(description="Stage2 small-train trainer on frozen Stage1 backbone")
    p.add_argument(
        "--stage2-contract-path",
        default="/home/chen034/workspace/stwm/reports/stage2_bootstrap_data_contract_20260408.json",
    )
    p.add_argument(
        "--recommended-runtime-json",
        default="/home/chen034/workspace/stwm/reports/stage1_v2_recommended_runtime_20260408.json",
    )
    p.add_argument("--use-recommended-runtime", action="store_true")

    p.add_argument(
        "--stage1-backbone-checkpoint",
        default="/home/chen034/workspace/stwm/outputs/checkpoints/stage1_v2_longtrain_220m_mainline_20260408/best.pt",
    )
    p.add_argument("--stage1-model-preset", default="prototype_220m")

    p.add_argument("--dataset-names", nargs="*", default=["vspw", "vipseg"])
    p.add_argument("--train-split", default="train")
    p.add_argument("--val-split", default="val")
    p.add_argument("--obs-len", type=int, default=8)
    p.add_argument("--fut-len", type=int, default=8)
    p.add_argument("--max-tokens", type=int, default=64)
    p.add_argument("--max-samples-train", type=int, default=24)
    p.add_argument("--max-samples-val", type=int, default=12)
    p.add_argument("--semantic-patch-radius", type=int, default=12)

    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--clip-grad-norm", type=float, default=1.0)

    p.add_argument("--train-steps", type=int, default=240)
    p.add_argument("--eval-interval", type=int, default=40)
    p.add_argument("--eval-max-batches", type=int, default=6)
    p.add_argument("--save-every-n-steps", type=int, default=1000)

    p.add_argument("--semantic-hidden-dim", type=int, default=256)
    p.add_argument("--semantic-embed-dim", type=int, default=256)
    p.add_argument("--semantic-source-mainline", default="crop_visual_encoder")
    p.add_argument("--legacy-semantic-source", default="hand_crafted_stats")
    p.add_argument("--semantic-crop-size", type=int, default=64)
    p.add_argument(
        "--semantic-rescue-mode",
        default="none",
        choices=["none", "align", "querypersist", "bootstrapplabel", "v2readoutalign", "v2readoutpersist", "v2readouthard"],
        help="Optional Stage2 semantic objective rescue pilot; default keeps the Wave1/Wave2 objective unchanged.",
    )
    p.add_argument("--semantic-rescue-weight", type=float, default=0.0)
    p.add_argument("--semantic-bootstrap-cache-path", default="")
    p.add_argument("--semantic-bootstrap-target-dim", type=int, default=10)
    p.add_argument("--semantic-alignment-loss-weight", type=float, default=0.0)
    p.add_argument("--query-persistence-consistency-loss-weight", type=float, default=0.0)
    p.add_argument("--semantic-hard-curriculum-weight", type=float, default=0.0)
    p.add_argument("--readout-semantic-alignment-loss-weight", type=float, default=0.0)
    p.add_argument("--persistence-contrastive-ranking-loss-weight", type=float, default=0.0)
    p.add_argument("--semantic-aux-subset-weighting-strength", type=float, default=0.0)

    p.add_argument("--output-dir", required=True)
    p.add_argument("--resume-from", default="")
    p.add_argument("--auto-resume-latest", action="store_true")
    p.add_argument(
        "--skip-resume-optimizer",
        action="store_true",
        help="Load module weights from --resume-from but start with a fresh optimizer state.",
    )

    p.add_argument("--run-name", required=True)
    p.add_argument("--run-summary-json", required=True)
    p.add_argument("--progress-json", default="")
    p.add_argument("--seed", type=int, default=20260408)
    return p.parse_args()


def _safe_json(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"json not found: {p}")
    payload = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"json payload must be dict: {p}")
    return payload


def _safe_load_checkpoint(path: str | Path, device: torch.device) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"checkpoint not found: {p}")
    try:
        payload = torch.load(p, map_location=device, weights_only=False)
    except TypeError:
        payload = torch.load(p, map_location=device)
    if not isinstance(payload, dict):
        raise RuntimeError(f"unsupported checkpoint payload type: {type(payload)}")
    return payload


def _load_bootstrap_cache(path_value: str) -> Dict[str, torch.Tensor]:
    target = str(path_value).strip()
    if not target:
        return {}
    p = Path(target)
    if not p.exists():
        return {}

    cache: Dict[str, torch.Tensor] = {}
    suffix = p.suffix.lower()
    if suffix == ".jsonl":
        for raw in p.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = raw.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except Exception:
                continue
            key = f"{str(item.get('dataset', '')).upper()}::{str(item.get('clip_id', ''))}"
            target_values = item.get("feature_target", [])
            if key != "::" and isinstance(target_values, list):
                try:
                    cache[key] = torch.tensor([float(x) for x in target_values], dtype=torch.float32)
                except Exception:
                    continue
        return cache

    try:
        payload = _safe_json(p)
    except Exception:
        return {}
    for item in payload.get("items", []) if isinstance(payload.get("items", []), list) else []:
        if not isinstance(item, dict):
            continue
        key = f"{str(item.get('dataset', '')).upper()}::{str(item.get('clip_id', ''))}"
        target_values = item.get("feature_target", [])
        if key != "::" and isinstance(target_values, list):
            try:
                cache[key] = torch.tensor([float(x) for x in target_values], dtype=torch.float32)
            except Exception:
                continue
    return cache


def _norm_name(name: str) -> str:
    return str(name).strip().upper()


def _extract_binding(contract: Dict[str, Any]) -> Dict[str, Any]:
    binding = contract.get("stage2_bootstrap_binding", {}) if isinstance(contract.get("stage2_bootstrap_binding", {}), dict) else {}
    core = [str(x).strip() for x in binding.get("core", [])] if isinstance(binding.get("core", []), list) else []
    optional = [str(x).strip() for x in binding.get("optional_extension", [])] if isinstance(binding.get("optional_extension", []), list) else []
    if not core:
        core = ["VSPW", "VIPSeg"]

    usage: Dict[str, Dict[str, bool]] = {}
    for ds in contract.get("datasets", []) if isinstance(contract.get("datasets", []), list) else []:
        if not isinstance(ds, dict):
            continue
        name = _norm_name(str(ds.get("dataset_name", "")))
        if not name:
            continue
        usage[name] = {
            "train": bool(ds.get("used_in_bootstrap_train", False)),
            "eval": bool(ds.get("used_in_bootstrap_eval", False)),
        }

    excluded = [
        {
            "dataset_name": str(x.get("dataset_name", "")),
            "reason": str(x.get("reason", "")),
        }
        for x in contract.get("excluded_datasets", [])
        if isinstance(x, dict)
    ]

    return {
        "core": core,
        "optional_extension": optional,
        "usage": usage,
        "excluded": excluded,
    }


def _summary_count(summary: Dict[str, Dict[str, Any]], name: str) -> int:
    target = _norm_name(name)
    for key, meta in summary.items():
        if _norm_name(str(key)) == target and isinstance(meta, dict):
            return int(meta.get("sample_count", 0))
    return 0


def _core_dataset_ready(
    train_summary: Dict[str, Dict[str, Any]],
    val_summary: Dict[str, Dict[str, Any]],
    core_names: List[str],
    usage: Dict[str, Dict[str, bool]],
) -> Tuple[bool, Dict[str, Any]]:
    details: Dict[str, Any] = {}
    ready = True
    for name in core_names:
        key = _norm_name(name)
        use_train = bool(usage.get(key, {}).get("train", True))
        use_eval = bool(usage.get(key, {}).get("eval", True))

        train_count = _summary_count(train_summary, key)
        val_count = _summary_count(val_summary, key)

        train_ok = (train_count > 0) if use_train else True
        val_ok = (val_count > 0) if use_eval else True
        item_ready = bool(train_ok and val_ok)
        ready = bool(ready and item_ready)

        details[key] = {
            "train_required": use_train,
            "eval_required": use_eval,
            "train_sample_count": int(train_count),
            "val_sample_count": int(val_count),
            "ready": bool(item_ready),
        }
    return bool(ready), details


def _atomic_torch_save(payload: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def _load_frozen_stage1_backbone(args: Any, device: torch.device) -> Tuple[TraceCausalTransformer, Dict[str, Any]]:
    ckpt = _safe_load_checkpoint(args.stage1_backbone_checkpoint, device=device)
    cfg_payload = ckpt.get("config", {}) if isinstance(ckpt.get("config", {}), dict) else {}
    preset = str(cfg_payload.get("model_preset", args.stage1_model_preset))
    cfg = build_tracewm_v2_config(preset)

    model = TraceCausalTransformer(cfg).to(device)
    state_dict = ckpt.get("model_state_dict") if isinstance(ckpt.get("model_state_dict"), dict) else None
    if state_dict is None:
        state_dict = {k: v for k, v in ckpt.items() if isinstance(v, torch.Tensor)}

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if unexpected:
        raise RuntimeError(f"unexpected stage1 checkpoint keys: {unexpected[:8]}")
    if len(missing) > 16:
        raise RuntimeError(f"too many missing stage1 checkpoint keys: {len(missing)}")

    for p in model.parameters():
        p.requires_grad = False
    model.eval()

    meta = {
        "checkpoint_path": str(args.stage1_backbone_checkpoint),
        "model_preset": preset,
        "missing_keys": list(missing),
        "unexpected_keys": list(unexpected),
        "parameter_count": int(sum(x.numel() for x in model.parameters())),
        "trainable_parameter_count": int(sum(x.numel() for x in model.parameters() if x.requires_grad)),
    }
    return model, meta


def _resolve_resume_path(resume_from: str, auto_resume_latest: bool, latest_path: Path) -> str:
    direct = str(resume_from).strip()
    if direct:
        return str(Path(direct).expanduser())
    if bool(auto_resume_latest) and latest_path.exists():
        return str(latest_path)
    return ""


def _to_device(batch: Dict[str, Any], device: torch.device, non_blocking: bool) -> Dict[str, Any]:
    out = dict(batch)
    for k in [
        "obs_state",
        "fut_state",
        "obs_valid",
        "fut_valid",
        "token_mask",
        "semantic_features",
        "semantic_mask",
        "semantic_rgb_crop",
        "semantic_mask_crop",
        "semantic_crop_valid",
        "semantic_mask_crop_valid",
    ]:
        out[k] = batch[k].to(device, non_blocking=non_blocking)
    return out


def _prepare_shifted(full_state: torch.Tensor) -> torch.Tensor:
    shifted = torch.zeros_like(full_state)
    shifted[:, 0] = full_state[:, 0]
    shifted[:, 1:] = full_state[:, :-1]
    return shifted


def _masked_mse_coord(pred_coord: torch.Tensor, target_coord: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    sq = ((pred_coord - target_coord) ** 2).sum(dim=-1)
    mask_f = valid_mask.float()
    denom = mask_f.sum().clamp_min(1.0)
    return (sq * mask_f).sum() / denom


def _masked_mean_l2(pred_coord: torch.Tensor, target_coord: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    l2 = torch.sqrt(((pred_coord - target_coord) ** 2).sum(dim=-1).clamp_min(1e-12))
    mask_f = valid_mask.float()
    denom = mask_f.sum().clamp_min(1.0)
    return (l2 * mask_f).sum() / denom


def _masked_endpoint_l2(pred_coord: torch.Tensor, target_coord: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    l2 = torch.sqrt(((pred_coord[:, -1] - target_coord[:, -1]) ** 2).sum(dim=-1).clamp_min(1e-12))
    mask_f = valid_mask[:, -1].float()
    denom = mask_f.sum().clamp_min(1.0)
    return (l2 * mask_f).sum() / denom


def _teacher_forced_predict(
    *,
    stage1_model: TraceCausalTransformer,
    semantic_encoder: SemanticEncoder,
    semantic_fusion: SemanticFusion,
    readout_head: torch.nn.Linear,
    batch: Dict[str, Any],
    obs_len: int,
    semantic_source_mainline: str,
) -> Dict[str, Any]:
    full_state = torch.cat([batch["obs_state"], batch["fut_state"]], dim=1)
    shifted = _prepare_shifted(full_state)
    token_mask = batch["token_mask"]

    with torch.no_grad():
        stage1_out = stage1_model(shifted, token_mask=token_mask)

    sem_enc = semantic_encoder(
        batch.get("semantic_features"),
        semantic_rgb_crop=batch.get("semantic_rgb_crop"),
        semantic_mask_crop=batch.get("semantic_mask_crop"),
        source_mode=str(semantic_source_mainline),
    )
    fused_hidden, aux = semantic_fusion(stage1_out["hidden"], sem_enc, token_mask=token_mask)
    pred_coord = readout_head(fused_hidden[:, int(obs_len) :])

    target_coord = batch["fut_state"][..., 0:2]
    valid_mask = batch["fut_valid"] & token_mask[:, None, :]

    return {
        "pred_coord": pred_coord,
        "target_coord": target_coord,
        "valid_mask": valid_mask,
        "semantic_tokens": sem_enc,
        "future_fused_hidden": fused_hidden[:, int(obs_len) :],
        "gate_mean": float(aux.get("gate_mean", 0.0)),
        "gate_std": float(aux.get("gate_std", 0.0)),
        "semantic_input_nonempty": bool((batch["semantic_mask"] & token_mask).any().item()),
    }


def _free_rollout_predict(
    *,
    stage1_model: TraceCausalTransformer,
    semantic_encoder: SemanticEncoder,
    semantic_fusion: SemanticFusion,
    readout_head: torch.nn.Linear,
    batch: Dict[str, Any],
    obs_len: int,
    fut_len: int,
    semantic_source_mainline: str,
) -> Dict[str, Any]:
    token_mask = batch["token_mask"]
    obs_state = batch["obs_state"]

    bsz, _, k_len, d_state = obs_state.shape
    total_len = int(obs_len) + int(fut_len)

    state_seq = torch.zeros((bsz, total_len, k_len, d_state), device=obs_state.device, dtype=obs_state.dtype)
    state_seq[:, : int(obs_len)] = obs_state

    sem_enc = semantic_encoder(
        batch.get("semantic_features"),
        semantic_rgb_crop=batch.get("semantic_rgb_crop"),
        semantic_mask_crop=batch.get("semantic_mask_crop"),
        source_mode=str(semantic_source_mainline),
    )
    gate_vals: List[float] = []

    for step in range(int(fut_len)):
        shifted = _prepare_shifted(state_seq)
        with torch.no_grad():
            stage1_out = stage1_model(shifted, token_mask=token_mask)

        fused_hidden, aux = semantic_fusion(stage1_out["hidden"], sem_enc, token_mask=token_mask)
        gate_vals.append(float(aux.get("gate_mean", 0.0)))

        pred_coord_all = readout_head(fused_hidden)
        time_idx = int(obs_len) + int(step)
        pred_coord_t = pred_coord_all[:, time_idx : time_idx + 1]

        pred_state_t = stage1_out["pred_state"][:, time_idx : time_idx + 1].detach().clone()
        pred_state_t[..., 0:2] = pred_coord_t.detach()
        state_seq[:, time_idx : time_idx + 1] = pred_state_t

    pred_future = state_seq[:, int(obs_len) :, :, 0:2]
    target_coord = batch["fut_state"][..., 0:2]
    valid_mask = batch["fut_valid"] & token_mask[:, None, :]

    return {
        "pred_coord": pred_future,
        "target_coord": target_coord,
        "valid_mask": valid_mask,
        "gate_mean": float(sum(gate_vals) / max(len(gate_vals), 1)),
    }


def _evaluate(
    *,
    stage1_model: TraceCausalTransformer,
    semantic_encoder: SemanticEncoder,
    semantic_fusion: SemanticFusion,
    readout_head: torch.nn.Linear,
    loader: DataLoader,
    device: torch.device,
    pin_memory: bool,
    obs_len: int,
    fut_len: int,
    max_batches: int,
    semantic_source_mainline: str,
) -> Dict[str, Any]:
    semantic_encoder.eval()
    semantic_fusion.eval()
    readout_head.eval()

    tf_sse = 0.0
    tf_count = 0.0
    free_l2_sum = 0.0
    free_l2_count = 0.0
    free_endpoint_sum = 0.0
    free_endpoint_count = 0.0
    total_loss_ref_sum = 0.0
    batch_count = 0
    gate_vals: List[float] = []
    nonempty_count = 0

    with torch.no_grad():
        for bi, raw_batch in enumerate(loader):
            if int(max_batches) > 0 and bi >= int(max_batches):
                break
            batch = _to_device(raw_batch, device=device, non_blocking=bool(pin_memory and device.type == "cuda"))

            tf_out = _teacher_forced_predict(
                stage1_model=stage1_model,
                semantic_encoder=semantic_encoder,
                semantic_fusion=semantic_fusion,
                readout_head=readout_head,
                batch=batch,
                obs_len=int(obs_len),
                semantic_source_mainline=str(semantic_source_mainline),
            )
            fr_out = _free_rollout_predict(
                stage1_model=stage1_model,
                semantic_encoder=semantic_encoder,
                semantic_fusion=semantic_fusion,
                readout_head=readout_head,
                batch=batch,
                obs_len=int(obs_len),
                fut_len=int(fut_len),
                semantic_source_mainline=str(semantic_source_mainline),
            )

            tf_sq = ((tf_out["pred_coord"] - tf_out["target_coord"]) ** 2).sum(dim=-1)
            tf_mask = tf_out["valid_mask"].float()
            tf_sse += float((tf_sq * tf_mask).sum().item())
            tf_count += float(tf_mask.sum().item())

            free_l2 = torch.sqrt(((fr_out["pred_coord"] - fr_out["target_coord"]) ** 2).sum(dim=-1).clamp_min(1e-12))
            free_mask = fr_out["valid_mask"].float()
            free_l2_sum += float((free_l2 * free_mask).sum().item())
            free_l2_count += float(free_mask.sum().item())

            endpoint_l2 = torch.sqrt(
                ((fr_out["pred_coord"][:, -1] - fr_out["target_coord"][:, -1]) ** 2).sum(dim=-1).clamp_min(1e-12)
            )
            endpoint_mask = fr_out["valid_mask"][:, -1].float()
            free_endpoint_sum += float((endpoint_l2 * endpoint_mask).sum().item())
            free_endpoint_count += float(endpoint_mask.sum().item())

            total_loss_ref_sum += float(_masked_mse_coord(tf_out["pred_coord"], tf_out["target_coord"], tf_out["valid_mask"]).item())
            gate_vals.append(float(tf_out["gate_mean"]))
            nonempty_count += 1 if bool(tf_out["semantic_input_nonempty"]) else 0
            batch_count += 1

    teacher_forced_coord_loss = float(tf_sse / max(tf_count, 1.0))
    free_rollout_coord_mean_l2 = float(free_l2_sum / max(free_l2_count, 1.0))
    free_rollout_endpoint_l2 = float(free_endpoint_sum / max(free_endpoint_count, 1.0))
    total_loss_reference = float(total_loss_ref_sum / max(batch_count, 1))

    return {
        "teacher_forced_coord_loss": teacher_forced_coord_loss,
        "free_rollout_coord_mean_l2": free_rollout_coord_mean_l2,
        "free_rollout_endpoint_l2": free_rollout_endpoint_l2,
        "total_loss_reference": total_loss_reference,
        "tapvid_style_eval": {
            "compatible": False,
            "status": "not_supported_in_current_stage2_trainer",
        },
        "tapvid3d_limited_eval": {
            "compatible": False,
            "status": "not_supported_in_current_stage2_trainer",
        },
        "semantic_branch_metrics": {
            "eval_gate_mean": float(sum(gate_vals) / max(len(gate_vals), 1)),
            "semantic_input_nonempty_ratio": float(nonempty_count / max(batch_count, 1)),
            "eval_batches": int(batch_count),
        },
    }


def _available_tertiary_metric(metrics: Dict[str, Any]) -> float:
    try:
        return float(metrics.get("teacher_forced_coord_loss", 1e9))
    except Exception:
        return 1e9


def _rank_key(metrics: Dict[str, Any]) -> Tuple[float, float, float]:
    return (
        float(metrics.get("free_rollout_endpoint_l2", 1e9)),
        float(metrics.get("free_rollout_coord_mean_l2", 1e9)),
        float(_available_tertiary_metric(metrics)),
    )


def _metric_triplet(metrics: Dict[str, Any]) -> Dict[str, float]:
    return {
        "teacher_forced_coord_loss": float(metrics.get("teacher_forced_coord_loss", 1e9)),
        "free_rollout_coord_mean_l2": float(metrics.get("free_rollout_coord_mean_l2", 1e9)),
        "free_rollout_endpoint_l2": float(metrics.get("free_rollout_endpoint_l2", 1e9)),
        "total_loss_reference": float(metrics.get("total_loss_reference", 1e9)),
    }


class SemanticRescueAuxHeads(torch.nn.Module):
    def __init__(self, semantic_dim: int, target_dim: int = 10, readout_dim: int | None = None) -> None:
        super().__init__()
        self.target_dim = int(target_dim)
        self.feature_head = torch.nn.Sequential(
            torch.nn.LayerNorm(int(semantic_dim)),
            torch.nn.Linear(int(semantic_dim), self.target_dim),
        )
        self.endpoint_head = torch.nn.Sequential(
            torch.nn.LayerNorm(int(semantic_dim)),
            torch.nn.Linear(int(semantic_dim), 2),
        )
        self.readout_feature_head: torch.nn.Module | None = None
        if readout_dim is not None and int(readout_dim) > 0:
            self.readout_feature_head = torch.nn.Sequential(
                torch.nn.LayerNorm(int(readout_dim)),
                torch.nn.Linear(int(readout_dim), self.target_dim),
            )

    def forward(self, semantic_tokens: torch.Tensor, readout_tokens: torch.Tensor | None = None) -> Dict[str, torch.Tensor]:
        out = {
            "feature_target": self.feature_head(semantic_tokens),
            "endpoint": self.endpoint_head(semantic_tokens),
        }
        if readout_tokens is not None and self.readout_feature_head is not None:
            out["readout_feature_target"] = self.readout_feature_head(readout_tokens)
        return out


def _bootstrap_targets_from_batch(
    *,
    batch: Dict[str, Any],
    cache: Dict[str, torch.Tensor],
    device: torch.device,
    target_dim: int,
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    fallback_raw = batch["semantic_features"].to(device=device, dtype=torch.float32)
    fallback = torch.zeros(
        (*fallback_raw.shape[:-1], int(target_dim)),
        device=device,
        dtype=torch.float32,
    )
    fallback_dim = min(int(fallback_raw.shape[-1]), int(target_dim))
    fallback[..., :fallback_dim] = fallback_raw[..., :fallback_dim]
    if not cache:
        valid = batch["semantic_mask"].to(device=device, dtype=torch.bool)
        return fallback, valid, 0.0

    targets = torch.zeros_like(fallback)
    valid = torch.zeros(batch["semantic_mask"].shape, dtype=torch.bool, device=device)
    hits = 0
    total = 0
    metas = batch.get("meta", [])
    for bi, meta in enumerate(metas if isinstance(metas, list) else []):
        if not isinstance(meta, dict):
            continue
        key = f"{str(meta.get('dataset', '')).upper()}::{str(meta.get('clip_id', ''))}"
        total += 1
        target = cache.get(key)
        if target is None:
            targets[bi] = fallback[bi]
            valid[bi] = batch["semantic_mask"][bi].to(device=device, dtype=torch.bool)
            continue
        dim = min(int(target.numel()), int(target_dim))
        targets[bi, :, :dim] = target[:dim].to(device=device, dtype=torch.float32)[None, :]
        valid[bi] = batch["semantic_mask"][bi].to(device=device, dtype=torch.bool)
        hits += 1
    coverage = float(hits / max(total, 1))
    return targets, valid, coverage


def _semantic_rescue_loss(
    *,
    mode: str,
    aux_heads: SemanticRescueAuxHeads | None,
    tf_out: Dict[str, Any],
    batch: Dict[str, Any],
    bootstrap_cache: Dict[str, torch.Tensor],
    device: torch.device,
    semantic_alignment_loss_weight: float,
    query_persistence_consistency_loss_weight: float,
    readout_semantic_alignment_loss_weight: float,
    persistence_contrastive_or_ranking_loss_weight: float,
    semantic_aux_subset_weighting_strength: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    if aux_heads is None or str(mode) == "none":
        zero = tf_out["pred_coord"].sum() * 0.0
        return zero, {
            "semantic_rescue_loss": 0.0,
            "semantic_alignment_loss": 0.0,
            "query_persistence_consistency_loss": 0.0,
            "readout_semantic_alignment_loss": 0.0,
            "persistence_contrastive_or_ranking_loss": 0.0,
            "semantic_aux_subset_weighting_strength": float(semantic_aux_subset_weighting_strength),
            "whether_main_rollout_loss_reweighted": False,
            "semantic_bootstrap_cache_hit_ratio": 0.0,
        }

    semantic_tokens = tf_out["semantic_tokens"]
    token_mask = batch["token_mask"].to(device=device, dtype=torch.bool)
    semantic_mask = batch["semantic_mask"].to(device=device, dtype=torch.bool) & token_mask
    readout_tokens = tf_out.get("future_fused_hidden")
    readout_tokens = readout_tokens[:, -1] if isinstance(readout_tokens, torch.Tensor) and readout_tokens.ndim == 4 else None
    aux = aux_heads(semantic_tokens, readout_tokens=readout_tokens)

    weighted_terms: List[torch.Tensor] = []
    weight_sum = 0.0
    cache_hit_ratio = 0.0
    mode_norm = str(mode).strip().lower()
    align_weight = float(semantic_alignment_loss_weight)
    query_weight = float(query_persistence_consistency_loss_weight)
    if mode_norm == "align" and align_weight <= 0.0:
        align_weight = 1.0
    if mode_norm == "querypersist" and query_weight <= 0.0:
        query_weight = 1.0
    if mode_norm.startswith("v2"):
        readout_align_weight = float(readout_semantic_alignment_loss_weight)
        contrastive_weight = float(persistence_contrastive_or_ranking_loss_weight)
        if mode_norm == "v2readoutalign" and readout_align_weight <= 0.0:
            readout_align_weight = 1.0
        if mode_norm == "v2readoutpersist":
            if readout_align_weight <= 0.0:
                readout_align_weight = 1.0
            if contrastive_weight <= 0.0:
                contrastive_weight = 0.25
        if mode_norm == "v2readouthard" and readout_align_weight <= 0.0:
            readout_align_weight = 1.0

        target, target_valid, cache_hit_ratio = _bootstrap_targets_from_batch(
            batch=batch,
            cache=bootstrap_cache,
            device=device,
            target_dim=int(aux_heads.target_dim),
        )
        valid = semantic_mask & target_valid
        sample_weights = _semantic_hard_sample_weights(
            batch=batch,
            device=device,
            strength=float(semantic_aux_subset_weighting_strength),
        )[:, None]
        aux_weights = torch.where(valid, sample_weights.expand_as(valid).to(torch.float32), torch.zeros_like(valid, dtype=torch.float32))
        denom = aux_weights.sum().clamp_min(1.0)
        readout_align_loss = semantic_tokens.sum() * 0.0
        contrastive_loss = semantic_tokens.sum() * 0.0

        readout_pred = aux.get("readout_feature_target", aux["feature_target"])
        if readout_align_weight > 0.0:
            pred = torch.nn.functional.normalize(readout_pred, dim=-1)
            tgt = torch.nn.functional.normalize(target, dim=-1)
            cosine = 1.0 - (pred * tgt).sum(dim=-1)
            readout_align_loss = (cosine * aux_weights).sum() / denom
            weighted_terms.append(float(readout_align_weight) * readout_align_loss)
            weight_sum += float(readout_align_weight)

        if contrastive_weight > 0.0:
            flat_valid = valid.reshape(-1)
            flat_pred = torch.nn.functional.normalize(readout_pred.reshape(-1, readout_pred.shape[-1])[flat_valid], dim=-1)
            flat_tgt = torch.nn.functional.normalize(target.reshape(-1, target.shape[-1])[flat_valid], dim=-1)
            if flat_pred.shape[0] > 1:
                logits = flat_pred @ flat_tgt.T / 0.07
                labels = torch.arange(flat_pred.shape[0], device=device)
                contrastive_loss = 0.5 * (
                    torch.nn.functional.cross_entropy(logits, labels)
                    + torch.nn.functional.cross_entropy(logits.T, labels)
                )
                weighted_terms.append(float(contrastive_weight) * contrastive_loss)
                weight_sum += float(contrastive_weight)

        if not weighted_terms:
            zero = tf_out["pred_coord"].sum() * 0.0
            return zero, {
                "semantic_rescue_loss": 0.0,
                "semantic_alignment_loss": 0.0,
                "query_persistence_consistency_loss": 0.0,
                "readout_semantic_alignment_loss": 0.0,
                "persistence_contrastive_or_ranking_loss": 0.0,
                "semantic_aux_subset_weighting_strength": float(semantic_aux_subset_weighting_strength),
                "whether_main_rollout_loss_reweighted": False,
                "semantic_bootstrap_cache_hit_ratio": float(cache_hit_ratio),
            }
        loss = sum(weighted_terms) / max(float(weight_sum), 1e-6)
        return loss, {
            "semantic_rescue_loss": float(loss.detach().cpu().item()),
            "semantic_alignment_loss": 0.0,
            "query_persistence_consistency_loss": 0.0,
            "readout_semantic_alignment_loss": float(readout_align_loss.detach().cpu().item()),
            "persistence_contrastive_or_ranking_loss": float(contrastive_loss.detach().cpu().item()),
            "semantic_aux_subset_weighting_strength": float(semantic_aux_subset_weighting_strength),
            "whether_main_rollout_loss_reweighted": False,
            "semantic_bootstrap_cache_hit_ratio": float(cache_hit_ratio),
        }

    align_loss = semantic_tokens.sum() * 0.0
    query_loss = semantic_tokens.sum() * 0.0
    if align_weight > 0.0:
        target, target_valid, cache_hit_ratio = _bootstrap_targets_from_batch(
            batch=batch,
            cache=bootstrap_cache if mode_norm == "bootstrapplabel" else {},
            device=device,
            target_dim=int(aux_heads.target_dim),
        )
        valid = semantic_mask & target_valid
        denom = valid.float().sum().clamp_min(1.0)
        pred = torch.nn.functional.normalize(aux["feature_target"], dim=-1)
        tgt = torch.nn.functional.normalize(target, dim=-1)
        cosine = 1.0 - (pred * tgt).sum(dim=-1)
        align_loss = (cosine * valid.float()).sum() / denom
        weighted_terms.append(float(align_weight) * align_loss)
        weight_sum += float(align_weight)

    if query_weight > 0.0:
        endpoint_target = batch["fut_state"][:, -1, :, 0:2].to(device=device, dtype=torch.float32)
        endpoint_valid = batch["fut_valid"][:, -1].to(device=device, dtype=torch.bool) & semantic_mask
        denom = endpoint_valid.float().sum().clamp_min(1.0)
        sq = ((aux["endpoint"] - endpoint_target) ** 2).sum(dim=-1)
        query_loss = (sq * endpoint_valid.float()).sum() / denom
        weighted_terms.append(float(query_weight) * query_loss)
        weight_sum += float(query_weight)

    if not weighted_terms:
        zero = tf_out["pred_coord"].sum() * 0.0
        return zero, {
            "semantic_rescue_loss": 0.0,
            "semantic_alignment_loss": 0.0,
            "query_persistence_consistency_loss": 0.0,
            "readout_semantic_alignment_loss": 0.0,
            "persistence_contrastive_or_ranking_loss": 0.0,
            "semantic_aux_subset_weighting_strength": float(semantic_aux_subset_weighting_strength),
            "whether_main_rollout_loss_reweighted": False,
            "semantic_bootstrap_cache_hit_ratio": 0.0,
        }

    loss = sum(weighted_terms) / max(float(weight_sum), 1e-6)
    return loss, {
        "semantic_rescue_loss": float(loss.detach().cpu().item()),
        "semantic_alignment_loss": float(align_loss.detach().cpu().item()),
        "query_persistence_consistency_loss": float(query_loss.detach().cpu().item()),
        "readout_semantic_alignment_loss": 0.0,
        "persistence_contrastive_or_ranking_loss": 0.0,
        "semantic_aux_subset_weighting_strength": float(semantic_aux_subset_weighting_strength),
        "whether_main_rollout_loss_reweighted": False,
        "semantic_bootstrap_cache_hit_ratio": float(cache_hit_ratio),
    }


def _semantic_hard_sample_weights(batch: Dict[str, Any], device: torch.device, strength: float) -> torch.Tensor:
    bsz = int(batch["obs_state"].shape[0])
    if float(strength) <= 0.0:
        return torch.ones((bsz,), device=device, dtype=torch.float32)
    state = torch.cat([batch["obs_state"], batch["fut_state"]], dim=1).to(device=device, dtype=torch.float32)
    valid = torch.cat([batch["obs_valid"], batch["fut_valid"]], dim=1).to(device=device, dtype=torch.bool)
    token_mask = batch["token_mask"].to(device=device, dtype=torch.bool)
    vt = valid & token_mask[:, None, :]
    coords = state[..., 0:2]
    area = (state[..., 6] * state[..., 7]).clamp(0.0, 1.0)
    step_motion = torch.sqrt(((coords[:, 1:] - coords[:, :-1]) ** 2).sum(dim=-1).clamp_min(1e-12))
    motion_valid = vt[:, 1:] & vt[:, :-1]
    motion = (step_motion * motion_valid.float()).sum(dim=(1, 2)) / motion_valid.float().sum(dim=(1, 2)).clamp_min(1.0)
    area_masked = torch.where(vt, area, torch.zeros_like(area))
    area_mean = area_masked.sum(dim=(1, 2)) / vt.float().sum(dim=(1, 2)).clamp_min(1.0)
    area_max = torch.where(vt, area, torch.full_like(area, -1.0)).amax(dim=(1, 2))
    area_min = torch.where(vt, area, torch.full_like(area, 2.0)).amin(dim=(1, 2))
    area_range = (area_max - area_min).clamp_min(0.0)
    small_score = (0.05 - area_mean).clamp_min(0.0) / 0.05
    hard_score = (0.5 * motion / 0.05 + 0.3 * area_range / 0.25 + 0.2 * small_score).clamp(0.0, 2.0)
    return 1.0 + float(strength) * hard_score.detach()


def _weighted_teacher_loss(
    pred_coord: torch.Tensor,
    target_coord: torch.Tensor,
    valid_mask: torch.Tensor,
    sample_weights: torch.Tensor,
) -> torch.Tensor:
    sq = ((pred_coord - target_coord) ** 2).sum(dim=-1)
    mask_f = valid_mask.float()
    weights = sample_weights[:, None, None].to(device=pred_coord.device, dtype=torch.float32)
    denom = (mask_f * weights).sum().clamp_min(1.0)
    return (sq * mask_f * weights).sum() / denom


def _split_counts_used(summary: Dict[str, Dict[str, Any]]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for name, meta in summary.items():
        if not isinstance(meta, dict):
            continue
        out[str(name)] = int(meta.get("sample_count", 0) or 0)
    return out


def _full_usage_flag(max_samples_value: Any) -> bool:
    try:
        return int(max_samples_value) < 0
    except Exception:
        return False


def _build_progress_payload(
    *,
    args: Any,
    status: str,
    global_step: int,
    target_steps: int,
    train_summary: Dict[str, Dict[str, Any]],
    val_summary: Dict[str, Dict[str, Any]],
    run_metadata: Dict[str, Any],
    runtime_meta: Dict[str, Any],
    checkpoint_dir: Path,
    best_ckpt: Path,
    latest_ckpt: Path,
    eval_history: List[Dict[str, Any]],
    best_metric_so_far: Dict[str, Any] | None,
) -> Dict[str, Any]:
    latest_event = eval_history[-1] if eval_history and isinstance(eval_history[-1], dict) else {}
    latest_metrics = latest_event.get("metrics", {}) if isinstance(latest_event.get("metrics", {}), dict) else {}
    return {
        "generated_at_utc": now_iso(),
        "run_name": str(args.run_name),
        "status": str(status),
        "current_mainline_semantic_source": str(args.semantic_source_mainline),
        "datasets_bound_for_train": [str(x) for x in args.dataset_names],
        "datasets_bound_for_eval": [str(x) for x in args.dataset_names],
        "runtime": runtime_meta,
        "run_metadata": run_metadata,
        "global_step": int(global_step),
        "train_steps_target": int(target_steps),
        "progress_ratio": float(float(global_step) / float(max(target_steps, 1))),
        "whether_full_train_used": bool(_full_usage_flag(args.max_samples_train)),
        "whether_full_val_used": bool(_full_usage_flag(args.max_samples_val)),
        "effective_train_sample_count_per_dataset": _split_counts_used(train_summary),
        "effective_val_sample_count_per_dataset": _split_counts_used(val_summary),
        "checkpoint_inventory": {
            "checkpoint_dir": str(checkpoint_dir),
            "best": str(best_ckpt),
            "latest": str(latest_ckpt),
            "best_exists": bool(best_ckpt.exists()),
            "latest_exists": bool(latest_ckpt.exists()),
        },
        "latest_eval_metrics": _metric_triplet(latest_metrics),
        "best_metric_so_far": best_metric_so_far if isinstance(best_metric_so_far, dict) else None,
    }


def _write_progress_snapshot(path_value: str, payload: Dict[str, Any]) -> None:
    target = str(path_value).strip()
    if not target:
        return
    _write_json(Path(target), payload)


def _checkpoint_payload(
    *,
    args: Any,
    global_step: int,
    best_metric_so_far: Dict[str, Any] | None,
    eval_history: List[Dict[str, Any]],
    semantic_encoder: SemanticEncoder,
    semantic_fusion: SemanticFusion,
    readout_head: torch.nn.Linear,
    semantic_rescue_heads: SemanticRescueAuxHeads | None,
    optimizer: torch.optim.Optimizer,
    run_metadata: Dict[str, Any],
) -> Dict[str, Any]:
    payload = {
        "run_name": str(args.run_name),
        "global_step": int(global_step),
        "best_metric_so_far": best_metric_so_far if isinstance(best_metric_so_far, dict) else None,
        "eval_history": eval_history,
        "semantic_encoder_state_dict": semantic_encoder.state_dict(),
        "semantic_fusion_state_dict": semantic_fusion.state_dict(),
        "readout_head_state_dict": readout_head.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "args": vars(args),
        "run_metadata": run_metadata,
    }
    if semantic_rescue_heads is not None:
        payload["semantic_rescue_heads_state_dict"] = semantic_rescue_heads.state_dict()
    return payload


def main() -> None:
    args = parse_args()
    set_seed(int(args.seed))

    if int(args.train_steps) <= 0:
        raise ValueError("train_steps must be > 0")
    if int(args.save_every_n_steps) <= 0:
        raise ValueError("save_every_n_steps must be > 0")

    output_dir = Path(str(args.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    run_summary_json = Path(str(args.run_summary_json))
    run_summary_json.parent.mkdir(parents=True, exist_ok=True)
    progress_json = str(args.progress_json).strip()

    best_ckpt = output_dir / "best.pt"
    latest_ckpt = output_dir / "latest.pt"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    runtime_meta: Dict[str, Any] = {
        "source": "manual",
        "num_workers": 0,
        "pin_memory": False,
        "persistent_workers": False,
        "prefetch_factor": 2,
        "single_gpu_only": True,
    }
    if bool(args.use_recommended_runtime):
        rt = _load_runtime_config(args.recommended_runtime_json)
        runtime_meta = {
            "source": "recommended_runtime_json",
            "path": str(args.recommended_runtime_json),
            "num_workers": int(rt.num_workers),
            "pin_memory": bool(rt.pin_memory),
            "persistent_workers": bool(rt.persistent_workers),
            "prefetch_factor": int(rt.prefetch_factor),
            "single_gpu_only": bool(rt.single_gpu_only),
            "selected_gpu_id_runtime_json": int(rt.selected_gpu_id),
            "required_mem_gb": float(rt.required_mem_gb),
            "safety_margin_gb": float(rt.safety_margin_gb),
        }

    stage2_contract = _safe_json(args.stage2_contract_path)
    binding = _extract_binding(stage2_contract)
    core_names = [str(x) for x in binding.get("core", [])]

    train_cfg = Stage2SemanticDatasetConfig(
        dataset_names=[str(x) for x in args.dataset_names],
        split=str(args.train_split),
        contract_path=str(args.stage2_contract_path),
        obs_len=int(args.obs_len),
        fut_len=int(args.fut_len),
        max_tokens=int(args.max_tokens),
        max_samples_per_dataset=int(args.max_samples_train),
        semantic_patch_radius=int(args.semantic_patch_radius),
        semantic_crop_size=int(args.semantic_crop_size),
        semantic_source_mainline=str(args.semantic_source_mainline),
    )
    val_cfg = Stage2SemanticDatasetConfig(
        dataset_names=[str(x) for x in args.dataset_names],
        split=str(args.val_split),
        contract_path=str(args.stage2_contract_path),
        obs_len=int(args.obs_len),
        fut_len=int(args.fut_len),
        max_tokens=int(args.max_tokens),
        max_samples_per_dataset=int(args.max_samples_val),
        semantic_patch_radius=int(args.semantic_patch_radius),
        semantic_crop_size=int(args.semantic_crop_size),
        semantic_source_mainline=str(args.semantic_source_mainline),
    )

    train_ds = Stage2SemanticDataset(train_cfg)
    val_ds = Stage2SemanticDataset(val_cfg)
    train_summary = dict(train_ds.dataset_summary)
    val_summary = dict(val_ds.dataset_summary)
    core_ready, core_details = _core_dataset_ready(
        train_summary=train_summary,
        val_summary=val_summary,
        core_names=core_names,
        usage=binding.get("usage", {}),
    )

    num_workers = int(runtime_meta.get("num_workers", 0))
    pin_memory = bool(runtime_meta.get("pin_memory", False))
    persistent_workers = bool(runtime_meta.get("persistent_workers", False))
    prefetch_factor = int(runtime_meta.get("prefetch_factor", 2))

    train_loader_kwargs: Dict[str, Any] = {
        "dataset": train_ds,
        "batch_size": int(args.batch_size),
        "shuffle": True,
        "num_workers": int(num_workers),
        "pin_memory": bool(pin_memory),
        "collate_fn": stage2_semantic_collate_fn,
    }
    if num_workers > 0:
        train_loader_kwargs["persistent_workers"] = bool(persistent_workers)
        train_loader_kwargs["prefetch_factor"] = int(prefetch_factor)
    train_loader = DataLoader(**train_loader_kwargs)
    val_loader = DataLoader(
        val_ds,
        batch_size=max(1, int(args.batch_size)),
        shuffle=False,
        num_workers=0,
        pin_memory=bool(pin_memory),
        collate_fn=stage2_semantic_collate_fn,
    )

    stage1_model, stage1_meta = _load_frozen_stage1_backbone(args=args, device=device)

    semantic_encoder = SemanticEncoder(
        SemanticEncoderConfig(
            input_dim=10,
            hidden_dim=int(args.semantic_hidden_dim),
            output_dim=int(args.semantic_embed_dim),
            dropout=0.1,
            mainline_source=str(args.semantic_source_mainline),
            legacy_source=str(args.legacy_semantic_source),
        )
    ).to(device)
    fusion_hidden_dim = int(stage1_model.config.d_model)
    semantic_fusion = SemanticFusion(
        SemanticFusionConfig(
            hidden_dim=fusion_hidden_dim,
            semantic_dim=int(args.semantic_embed_dim),
            dropout=0.1,
        )
    ).to(device)
    readout_head = torch.nn.Linear(fusion_hidden_dim, 2).to(device)
    semantic_rescue_heads: SemanticRescueAuxHeads | None = None
    rescue_mode = str(args.semantic_rescue_mode).strip().lower()
    rescue_weight = float(args.semantic_rescue_weight)
    bootstrap_cache = _load_bootstrap_cache(str(args.semantic_bootstrap_cache_path))
    if rescue_mode != "none" and rescue_weight > 0.0:
        semantic_rescue_heads = SemanticRescueAuxHeads(
            semantic_dim=int(args.semantic_embed_dim),
            target_dim=int(args.semantic_bootstrap_target_dim),
            readout_dim=int(fusion_hidden_dim),
        ).to(device)

    trainable_params: List[torch.nn.Parameter] = []
    modules_for_training: List[torch.nn.Module] = [semantic_encoder, semantic_fusion, readout_head]
    if semantic_rescue_heads is not None:
        modules_for_training.append(semantic_rescue_heads)
    for module in modules_for_training:
        trainable_params.extend([p for p in module.parameters() if p.requires_grad])

    pre_frozen_parameter_count = int(stage1_meta.get("parameter_count", 0))
    pre_stage1_trainable_parameter_count = int(stage1_meta.get("trainable_parameter_count", 0))
    pre_trainable_parameter_count = int(sum(p.numel() for p in trainable_params))

    print(f"[stage2-smalltrain] pre_frozen_parameter_count={pre_frozen_parameter_count}")
    print(f"[stage2-smalltrain] pre_stage1_trainable_parameter_count={pre_stage1_trainable_parameter_count}")
    print(f"[stage2-smalltrain] pre_stage2_trainable_parameter_count={pre_trainable_parameter_count}")

    optimizer = torch.optim.AdamW(trainable_params, lr=float(args.lr), weight_decay=float(args.weight_decay))

    gpu_selection = {}
    try:
        gpu_selection = json.loads(str(os.environ.get("TRACEWM_STAGE1_V2_GPU_SELECTION_METADATA_JSON", "")) or "{}")
        if not isinstance(gpu_selection, dict):
            gpu_selection = {}
    except Exception:
        gpu_selection = {}

    run_metadata: Dict[str, Any] = {
        "run_name": str(args.run_name),
        "started_at_utc": now_iso(),
        "cuda_visible_devices": str(os.environ.get("CUDA_VISIBLE_DEVICES", "")),
        "gpu_selection": gpu_selection,
        "current_mainline_semantic_source": str(args.semantic_source_mainline),
        "legacy_semantic_source": str(args.legacy_semantic_source),
        "semantic_rescue_mode": str(rescue_mode),
        "semantic_rescue_weight": float(rescue_weight),
        "semantic_bootstrap_cache_path": str(args.semantic_bootstrap_cache_path),
        "semantic_bootstrap_target_dim": int(args.semantic_bootstrap_target_dim),
        "semantic_alignment_loss_weight": float(args.semantic_alignment_loss_weight),
        "query_persistence_consistency_loss_weight": float(args.query_persistence_consistency_loss_weight),
        "semantic_hard_curriculum_weight": float(args.semantic_hard_curriculum_weight),
        "readout_semantic_alignment_loss_weight": float(args.readout_semantic_alignment_loss_weight),
        "persistence_contrastive_ranking_loss_weight": float(args.persistence_contrastive_ranking_loss_weight),
        "semantic_aux_subset_weighting_strength": float(args.semantic_aux_subset_weighting_strength),
        "whether_main_rollout_loss_reweighted": bool((not str(rescue_mode).startswith("v2")) and float(args.semantic_hard_curriculum_weight) > 0.0),
        "semantic_bootstrap_cache_item_count": int(len(bootstrap_cache)),
        "skip_resume_optimizer": bool(args.skip_resume_optimizer),
        "resume_optimizer_loaded": False,
        "resume_optimizer_skip_reason": "",
    }

    resolved_resume = _resolve_resume_path(
        resume_from=str(args.resume_from),
        auto_resume_latest=bool(args.auto_resume_latest),
        latest_path=latest_ckpt,
    )

    global_step = 0
    resume_global_step_loaded = 0
    new_best_written_this_run = False
    best_metric_so_far: Dict[str, Any] | None = None
    eval_history: List[Dict[str, Any]] = []
    if resolved_resume:
        payload = _safe_load_checkpoint(resolved_resume, device=device)
        semantic_encoder.load_state_dict(payload.get("semantic_encoder_state_dict", {}))
        semantic_fusion.load_state_dict(payload.get("semantic_fusion_state_dict", {}))
        readout_head.load_state_dict(payload.get("readout_head_state_dict", {}))
        if semantic_rescue_heads is not None and isinstance(payload.get("semantic_rescue_heads_state_dict", None), dict):
            semantic_rescue_heads.load_state_dict(payload["semantic_rescue_heads_state_dict"])
        if isinstance(payload.get("optimizer_state_dict", None), dict):
            if bool(args.skip_resume_optimizer):
                run_metadata["resume_optimizer_skip_reason"] = "explicit_skip_resume_optimizer"
            else:
                try:
                    optimizer.load_state_dict(payload["optimizer_state_dict"])
                    run_metadata["resume_optimizer_loaded"] = True
                except ValueError as exc:
                    run_metadata["resume_optimizer_skip_reason"] = f"incompatible_optimizer_state: {exc}"
        global_step = int(payload.get("global_step", 0) or 0)
        resume_global_step_loaded = int(global_step)
        if isinstance(payload.get("best_metric_so_far", None), dict):
            best_metric_so_far = payload.get("best_metric_so_far")
        if isinstance(payload.get("eval_history", None), list):
            eval_history = [x for x in payload.get("eval_history", []) if isinstance(x, dict)]
        run_metadata["resumed_from"] = str(resolved_resume)
    else:
        run_metadata["resumed_from"] = ""

    semantic_encoder.train()
    semantic_fusion.train()
    readout_head.train()
    if semantic_rescue_heads is not None:
        semantic_rescue_heads.train()

    train_iter = iter(train_loader)
    step_checkpoints: List[str] = sorted(str(p) for p in output_dir.glob("step_*.pt"))
    stage1_grad_detected_any = False
    semantic_grad_norm_latest = 0.0
    teacher_loss_history: List[float] = []
    rescue_loss_history: List[float] = []
    semantic_alignment_loss_history: List[float] = []
    query_persistence_loss_history: List[float] = []
    readout_alignment_loss_history: List[float] = []
    persistence_contrastive_loss_history: List[float] = []
    rescue_cache_hit_history: List[float] = []
    semantic_hard_weight_mean_history: List[float] = []
    gate_history: List[float] = []
    semantic_nonempty_count = 0
    optimizer_steps_this_run = 0

    target_steps = int(args.train_steps)
    eval_interval = int(args.eval_interval)
    save_every = int(args.save_every_n_steps)

    _write_progress_snapshot(
        progress_json,
        _build_progress_payload(
            args=args,
            status="initialized",
            global_step=int(global_step),
            target_steps=int(target_steps),
            train_summary=train_summary,
            val_summary=val_summary,
            run_metadata=run_metadata,
            runtime_meta=runtime_meta,
            checkpoint_dir=output_dir,
            best_ckpt=best_ckpt,
            latest_ckpt=latest_ckpt,
            eval_history=eval_history,
            best_metric_so_far=best_metric_so_far,
        ),
    )

    def _save_latest_and_optional_step(step_now: int, save_step: bool) -> None:
        payload = _checkpoint_payload(
            args=args,
            global_step=int(step_now),
            best_metric_so_far=best_metric_so_far,
            eval_history=eval_history,
            semantic_encoder=semantic_encoder,
            semantic_fusion=semantic_fusion,
            readout_head=readout_head,
            semantic_rescue_heads=semantic_rescue_heads,
            optimizer=optimizer,
            run_metadata=run_metadata,
        )
        _atomic_torch_save(payload, latest_ckpt)
        if save_step:
            step_path = output_dir / f"step_{int(step_now):07d}.pt"
            _atomic_torch_save(payload, step_path)
            sp = str(step_path)
            if sp not in step_checkpoints:
                step_checkpoints.append(sp)

    while global_step < target_steps:
        try:
            raw_batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            raw_batch = next(train_iter)

        batch = _to_device(raw_batch, device=device, non_blocking=bool(pin_memory and device.type == "cuda"))
        tf_out = _teacher_forced_predict(
            stage1_model=stage1_model,
            semantic_encoder=semantic_encoder,
            semantic_fusion=semantic_fusion,
            readout_head=readout_head,
            batch=batch,
            obs_len=int(args.obs_len),
            semantic_source_mainline=str(args.semantic_source_mainline),
        )

        main_rollout_reweight_strength = 0.0 if str(rescue_mode).startswith("v2") else float(args.semantic_hard_curriculum_weight)
        semantic_hard_weights = _semantic_hard_sample_weights(
            batch=batch,
            device=device,
            strength=float(main_rollout_reweight_strength),
        )
        teacher_loss = _weighted_teacher_loss(
            tf_out["pred_coord"],
            tf_out["target_coord"],
            tf_out["valid_mask"],
            semantic_hard_weights,
        )
        rescue_loss, rescue_info = _semantic_rescue_loss(
            mode=str(rescue_mode),
            aux_heads=semantic_rescue_heads,
            tf_out=tf_out,
            batch=batch,
            bootstrap_cache=bootstrap_cache,
            device=device,
            semantic_alignment_loss_weight=float(args.semantic_alignment_loss_weight),
            query_persistence_consistency_loss_weight=float(args.query_persistence_consistency_loss_weight),
            readout_semantic_alignment_loss_weight=float(args.readout_semantic_alignment_loss_weight),
            persistence_contrastive_or_ranking_loss_weight=float(args.persistence_contrastive_ranking_loss_weight),
            semantic_aux_subset_weighting_strength=float(args.semantic_aux_subset_weighting_strength),
        )
        total_train_loss = teacher_loss + float(rescue_weight) * rescue_loss

        optimizer.zero_grad(set_to_none=True)
        total_train_loss.backward()

        if float(args.clip_grad_norm) > 0.0:
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=float(args.clip_grad_norm))

        stage1_grad_detected = False
        for p in stage1_model.parameters():
            if p.grad is not None and float(p.grad.detach().abs().sum().item()) > 0.0:
                stage1_grad_detected = True
                break
        stage1_grad_detected_any = bool(stage1_grad_detected_any or stage1_grad_detected)

        semantic_grad_sq = 0.0
        for p in trainable_params:
            if p.grad is None:
                continue
            g = p.grad.detach()
            semantic_grad_sq += float((g * g).sum().item())
        semantic_grad_norm_latest = float(np.sqrt(max(semantic_grad_sq, 0.0)))

        optimizer.step()

        global_step += 1
        optimizer_steps_this_run += 1

        teacher_loss_history.append(float(teacher_loss.detach().cpu().item()))
        rescue_loss_history.append(float(rescue_info.get("semantic_rescue_loss", 0.0)))
        semantic_alignment_loss_history.append(float(rescue_info.get("semantic_alignment_loss", 0.0)))
        query_persistence_loss_history.append(float(rescue_info.get("query_persistence_consistency_loss", 0.0)))
        readout_alignment_loss_history.append(float(rescue_info.get("readout_semantic_alignment_loss", 0.0)))
        persistence_contrastive_loss_history.append(float(rescue_info.get("persistence_contrastive_or_ranking_loss", 0.0)))
        rescue_cache_hit_history.append(float(rescue_info.get("semantic_bootstrap_cache_hit_ratio", 0.0)))
        semantic_hard_weight_mean_history.append(float(semantic_hard_weights.detach().mean().cpu().item()))
        gate_history.append(float(tf_out["gate_mean"]))
        semantic_nonempty_count += 1 if bool(tf_out["semantic_input_nonempty"]) else 0

        should_eval = bool(eval_interval > 0 and global_step % eval_interval == 0)
        if global_step == target_steps:
            should_eval = True

        if should_eval:
            metrics = _evaluate(
                stage1_model=stage1_model,
                semantic_encoder=semantic_encoder,
                semantic_fusion=semantic_fusion,
                readout_head=readout_head,
                loader=val_loader,
                device=device,
                pin_memory=bool(pin_memory),
                obs_len=int(args.obs_len),
                fut_len=int(args.fut_len),
                max_batches=int(args.eval_max_batches),
                semantic_source_mainline=str(args.semantic_source_mainline),
            )
            rk = _rank_key(metrics)
            event = {
                "global_step": int(global_step),
                "metrics": metrics,
                "rank_key": [float(rk[0]), float(rk[1]), float(rk[2])],
            }
            eval_history.append(event)

            if best_metric_so_far is None or tuple(event["rank_key"]) < tuple(best_metric_so_far.get("rank_key", [1e9, 1e9, 1e9])):
                best_metric_so_far = {
                    "global_step": int(global_step),
                    "metrics": metrics,
                    "rank_key": [float(rk[0]), float(rk[1]), float(rk[2])],
                }
                best_payload = _checkpoint_payload(
                    args=args,
                    global_step=int(global_step),
                    best_metric_so_far=best_metric_so_far,
                    eval_history=eval_history,
                    semantic_encoder=semantic_encoder,
                    semantic_fusion=semantic_fusion,
                    readout_head=readout_head,
                    semantic_rescue_heads=semantic_rescue_heads,
                    optimizer=optimizer,
                    run_metadata=run_metadata,
                )
                _atomic_torch_save(best_payload, best_ckpt)
                new_best_written_this_run = True

            _write_progress_snapshot(
                progress_json,
                _build_progress_payload(
                    args=args,
                    status="running",
                    global_step=int(global_step),
                    target_steps=int(target_steps),
                    train_summary=train_summary,
                    val_summary=val_summary,
                    run_metadata=run_metadata,
                    runtime_meta=runtime_meta,
                    checkpoint_dir=output_dir,
                    best_ckpt=best_ckpt,
                    latest_ckpt=latest_ckpt,
                    eval_history=eval_history,
                    best_metric_so_far=best_metric_so_far,
                ),
            )

        should_save_step = bool(global_step % save_every == 0)
        should_save_latest = bool(should_save_step or global_step == target_steps)
        if should_save_latest:
            _save_latest_and_optional_step(step_now=int(global_step), save_step=bool(should_save_step))

    if best_metric_so_far is None:
        metrics = _evaluate(
            stage1_model=stage1_model,
            semantic_encoder=semantic_encoder,
            semantic_fusion=semantic_fusion,
            readout_head=readout_head,
            loader=val_loader,
            device=device,
            pin_memory=bool(pin_memory),
            obs_len=int(args.obs_len),
            fut_len=int(args.fut_len),
            max_batches=int(args.eval_max_batches),
            semantic_source_mainline=str(args.semantic_source_mainline),
        )
        rk = _rank_key(metrics)
        best_metric_so_far = {
            "global_step": int(global_step),
            "metrics": metrics,
            "rank_key": [float(rk[0]), float(rk[1]), float(rk[2])],
        }

    if not best_ckpt.exists():
        inherited_best_step = int((best_metric_so_far or {}).get("global_step", -1))
        if (
            resolved_resume
            and (not new_best_written_this_run)
            and inherited_best_step <= int(resume_global_step_loaded)
            and Path(str(resolved_resume)).resolve() != best_ckpt.resolve()
        ):
            shutil.copy2(str(resolved_resume), str(best_ckpt))
        else:
            best_payload = _checkpoint_payload(
                args=args,
                global_step=int(global_step),
                best_metric_so_far=best_metric_so_far,
                eval_history=eval_history,
                semantic_encoder=semantic_encoder,
                semantic_fusion=semantic_fusion,
                readout_head=readout_head,
                semantic_rescue_heads=semantic_rescue_heads,
                optimizer=optimizer,
                run_metadata=run_metadata,
            )
            _atomic_torch_save(best_payload, best_ckpt)
    if not latest_ckpt.exists():
        _save_latest_and_optional_step(step_now=int(global_step), save_step=False)

    final_metrics = dict(best_metric_so_far.get("metrics", {})) if isinstance(best_metric_so_far, dict) else {}
    final_metric_triplet = _metric_triplet(final_metrics)
    frozen_count = int(stage1_meta.get("parameter_count", 0))
    trainable_count = int(sum(p.numel() for p in trainable_params))
    stage1_trainable_count = int(stage1_meta.get("trainable_parameter_count", 0))

    best_checkpoint_metric = _metric_triplet(
        (best_metric_so_far.get("metrics", {}) if isinstance(best_metric_so_far, dict) and isinstance(best_metric_so_far.get("metrics", {}), dict) else final_metrics)
    )
    latest_event = eval_history[-1] if eval_history and isinstance(eval_history[-1], dict) else {}
    latest_metrics = latest_event.get("metrics", {}) if isinstance(latest_event.get("metrics", {}), dict) else final_metrics
    latest_checkpoint_metric = _metric_triplet(latest_metrics)

    train_split_counts_used = _split_counts_used(train_summary)
    val_split_counts_used = _split_counts_used(val_summary)

    boundary_ok = bool(stage1_trainable_count == 0 and (not stage1_grad_detected_any))
    run_stable = bool(
        np.isfinite(float(final_metrics.get("teacher_forced_coord_loss", np.inf)))
        and np.isfinite(float(final_metrics.get("free_rollout_coord_mean_l2", np.inf)))
        and np.isfinite(float(final_metrics.get("free_rollout_endpoint_l2", np.inf)))
    )

    train_total_count = int(sum(train_split_counts_used.values()))
    effective_batch = int(args.batch_size)
    epochs_completed = 0.0
    if train_total_count > 0:
        epochs_completed = float(optimizer_steps_this_run * effective_batch) / float(train_total_count)

    payload = {
        "generated_at_utc": now_iso(),
        "run_name": str(args.run_name),
        "objective": "Stage2 training run on frozen Stage1 220m backbone",
        "current_mainline_semantic_source": str(args.semantic_source_mainline),
        "legacy_semantic_source": str(args.legacy_semantic_source),
        "stage2_contract_path": str(args.stage2_contract_path),
        "stage2_data_binding": {
            "core": core_names,
            "optional_extension": binding.get("optional_extension", []),
            "excluded": binding.get("excluded", []),
            "run_datasets": [str(x) for x in args.dataset_names],
        },
        "datasets_bound_for_train": [str(x) for x in args.dataset_names],
        "datasets_bound_for_eval": [str(x) for x in args.dataset_names],
        "runtime": runtime_meta,
        "run_metadata": run_metadata,
        "training_budget": {
            "train_steps_target": int(target_steps),
            "train_steps_completed": int(global_step),
            "optimizer_steps_this_invocation": int(optimizer_steps_this_run),
            "batch_size": int(args.batch_size),
            "eval_interval": int(eval_interval),
            "eval_max_batches": int(args.eval_max_batches),
            "save_every_n_steps": int(save_every),
        },
        "dataset_summary": {
            "train": train_summary,
            "val": val_summary,
        },
        "whether_full_train_used": bool(_full_usage_flag(args.max_samples_train)),
        "whether_full_val_used": bool(_full_usage_flag(args.max_samples_val)),
        "effective_train_sample_count_per_dataset": train_split_counts_used,
        "effective_val_sample_count_per_dataset": val_split_counts_used,
        "core_dataset_inputs": {
            "ready": bool(core_ready),
            "details": core_details,
        },
        "stage1_backbone": {
            "load_success": True,
            **stage1_meta,
        },
        "parameter_count_frozen": int(frozen_count),
        "parameter_count_trainable": int(trainable_count),
        "freeze_trainable_boundary": {
            "stage1_trainable_parameter_count": int(stage1_trainable_count),
            "semantic_trainable_parameter_count": int(trainable_count),
            "stage1_grad_detected_after_backward": bool(stage1_grad_detected_any),
            "semantic_grad_norm_latest": float(semantic_grad_norm_latest),
            "boundary_ok": bool(boundary_ok),
        },
        "semantic_branch_metrics": {
            "train_gate_mean": float(sum(gate_history) / max(len(gate_history), 1)),
            "train_semantic_input_nonempty_ratio": float(semantic_nonempty_count / max(len(gate_history), 1)),
            "semantic_crop_size": int(args.semantic_crop_size),
            "current_mainline_semantic_source": str(args.semantic_source_mainline),
            "legacy_semantic_source": str(args.legacy_semantic_source),
            "legacy_semantic_feature_dim": 10,
            "semantic_rescue_mode": str(rescue_mode),
            "semantic_rescue_weight": float(rescue_weight),
            "semantic_bootstrap_target_dim": int(args.semantic_bootstrap_target_dim),
            "semantic_alignment_loss_weight": float(args.semantic_alignment_loss_weight),
            "query_persistence_consistency_loss_weight": float(args.query_persistence_consistency_loss_weight),
            "semantic_hard_curriculum_weight": float(args.semantic_hard_curriculum_weight),
            "readout_semantic_alignment_loss_weight": float(args.readout_semantic_alignment_loss_weight),
            "persistence_contrastive_ranking_loss_weight": float(args.persistence_contrastive_ranking_loss_weight),
            "semantic_aux_subset_weighting_strength": float(args.semantic_aux_subset_weighting_strength),
            "whether_main_rollout_loss_reweighted": bool((not str(rescue_mode).startswith("v2")) and float(args.semantic_hard_curriculum_weight) > 0.0),
            "semantic_rescue_loss_mean": float(sum(rescue_loss_history) / max(len(rescue_loss_history), 1)),
            "semantic_alignment_loss_mean": float(sum(semantic_alignment_loss_history) / max(len(semantic_alignment_loss_history), 1)),
            "query_persistence_consistency_loss_mean": float(sum(query_persistence_loss_history) / max(len(query_persistence_loss_history), 1)),
            "readout_semantic_alignment_loss_mean": float(sum(readout_alignment_loss_history) / max(len(readout_alignment_loss_history), 1)),
            "persistence_contrastive_or_ranking_loss_mean": float(sum(persistence_contrastive_loss_history) / max(len(persistence_contrastive_loss_history), 1)),
            "semantic_bootstrap_cache_hit_ratio_mean": float(sum(rescue_cache_hit_history) / max(len(rescue_cache_hit_history), 1)),
            "semantic_hard_sample_weight_mean": float(sum(semantic_hard_weight_mean_history) / max(len(semantic_hard_weight_mean_history), 1)),
            "semantic_bootstrap_cache_item_count": int(len(bootstrap_cache)),
        },
        "selection_policy": {
            "primary": "free_rollout_endpoint_l2",
            "secondary": "free_rollout_coord_mean_l2",
            "tertiary": "teacher_forced_coord_loss",
            "total_loss_usage": "reference_only",
        },
        "comparison_sorting": {
            "primary": "free_rollout_endpoint_l2",
            "secondary": "free_rollout_coord_mean_l2",
            "tertiary": "teacher_forced_coord_loss",
            "total_loss_usage": "reference_only",
        },
        "teacher_forced_coord_loss": float(final_metric_triplet["teacher_forced_coord_loss"]),
        "free_rollout_coord_mean_l2": float(final_metric_triplet["free_rollout_coord_mean_l2"]),
        "free_rollout_endpoint_l2": float(final_metric_triplet["free_rollout_endpoint_l2"]),
        "best_checkpoint_metric": {
            "global_step": int((best_metric_so_far or {}).get("global_step", -1)),
            "metrics": best_checkpoint_metric,
            "rank_key": _rank_key(best_checkpoint_metric),
        },
        "latest_checkpoint_metric": {
            "global_step": int(latest_event.get("global_step", -1) or -1),
            "metrics": latest_checkpoint_metric,
            "rank_key": _rank_key(latest_checkpoint_metric),
        },
        "train_split_counts_used": train_split_counts_used,
        "val_split_counts_used": val_split_counts_used,
        "train_split_total_count_used": int(train_total_count),
        "val_split_total_count_used": int(sum(val_split_counts_used.values())),
        "frozen_parameter_count": int(frozen_count),
        "trainable_parameter_count": int(trainable_count),
        "boundary_ok": bool(boundary_ok),
        "training_progress": {
            "optimizer_steps": int(optimizer_steps_this_run),
            "effective_batch": int(effective_batch),
            "epochs_completed": float(epochs_completed),
            "eval_interval": int(eval_interval),
            "save_every_n_steps": int(save_every),
        },
        "final_metrics": final_metrics,
        "best_metric_so_far": best_metric_so_far,
        "eval_history": eval_history,
        "checkpoint_inventory": {
            "checkpoint_dir": str(output_dir),
            "best": str(best_ckpt),
            "latest": str(latest_ckpt),
            "step_checkpoints": sorted(step_checkpoints),
            "resume_from": str(resolved_resume),
            "auto_resume_latest": bool(args.auto_resume_latest),
        },
        "run_stable": bool(run_stable),
    }

    _write_json(run_summary_json, payload)
    _write_progress_snapshot(
        progress_json,
        _build_progress_payload(
            args=args,
            status="completed",
            global_step=int(global_step),
            target_steps=int(target_steps),
            train_summary=train_summary,
            val_summary=val_summary,
            run_metadata=run_metadata,
            runtime_meta=runtime_meta,
            checkpoint_dir=output_dir,
            best_ckpt=best_ckpt,
            latest_ckpt=latest_ckpt,
            eval_history=eval_history,
            best_metric_so_far=best_metric_so_far,
        ),
    )

    print(f"[stage2-smalltrain] run_name={args.run_name}")
    print(f"[stage2-smalltrain] run_summary_json={run_summary_json}")
    print(f"[stage2-smalltrain] checkpoint_dir={output_dir}")
    print(f"[stage2-smalltrain] best_checkpoint={best_ckpt}")
    print(f"[stage2-smalltrain] latest_checkpoint={latest_ckpt}")
    print(f"[stage2-smalltrain] free_rollout_endpoint_l2={float(final_metrics.get('free_rollout_endpoint_l2', 1e9)):.6f}")
    print(f"[stage2-smalltrain] boundary_ok={bool(boundary_ok)}")


if __name__ == "__main__":
    main()
