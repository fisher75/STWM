#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable

import torch
import torch.nn.functional as F

from stwm.tracewm_v2_stage2.datasets.stage2_semantic_dataset import (
    Stage2SemanticDataset,
    Stage2SemanticDatasetConfig,
    stage2_semantic_collate_fn,
)
from stwm.tracewm_v2_stage2.models.semantic_encoder import SemanticEncoder, SemanticEncoderConfig
from stwm.tracewm_v2_stage2.models.semantic_fusion import SemanticFusion, SemanticFusionConfig
from stwm.tracewm_v2_stage2.models.semantic_trace_world_head import (
    SemanticTraceStateHead,
    SemanticTraceStateHeadConfig,
)
from stwm.tracewm_v2_stage2.models.trace_unit_broadcast import (
    TraceUnitBroadcast,
    TraceUnitBroadcastConfig,
)
from stwm.tracewm_v2_stage2.models.trace_unit_factorized_state import (
    TraceUnitFactorizedState,
    TraceUnitFactorizedStateConfig,
)
from stwm.tracewm_v2_stage2.models.trace_unit_handshake import (
    TraceUnitHandshake,
    TraceUnitHandshakeConfig,
)
from stwm.tracewm_v2_stage2.models.trace_unit_tokenizer import (
    TraceUnitTokenizer,
    TraceUnitTokenizerConfig,
)
from stwm.tracewm_v2_stage2.trainers.train_tracewm_stage2_smalltrain import (
    STATE_DIM,
    _build_future_semantic_tusb_unfreeze_audit,
    _configure_future_semantic_tusb_unfreeze_trainability,
    _load_frozen_stage1_backbone,
    _teacher_forced_predict,
    _to_device,
)
from stwm.tracewm_v2_stage2.utils.future_semantic_feature_targets import stage2_item_key
from stwm.tracewm_v2_stage2.utils.future_semantic_prototype_targets import (
    FutureSemanticPrototypeTargetCache,
    load_future_semantic_prototype_target_cache,
    prototype_tensors_for_batch,
)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, torch.Tensor):
        if obj.numel() == 1:
            return obj.detach().cpu().item()
        return obj.detach().cpu().tolist()
    return str(obj)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")


def write_doc(path: Path, title: str, payload: dict[str, Any], bullets: Iterable[str] = ()) -> None:
    lines = [f"# {title}", ""]
    for bullet in bullets:
        lines.append(f"- {bullet}")
    if bullets:
        lines.append("")
    for key, value in payload.items():
        if isinstance(value, (dict, list)):
            continue
        lines.append(f"- {key}: `{value}`")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return int(default)
        return int(value)
    except Exception:
        return int(default)


def _merge_args(checkpoint_args: dict[str, Any], overrides: dict[str, Any]) -> SimpleNamespace:
    merged = dict(checkpoint_args or {})
    merged.update({k: v for k, v in overrides.items() if v is not None})
    defaults = {
        "stage1_partial_unfreeze_mode": "none",
        "stage1_partial_unfreeze_layer_count": 1,
        "stage1_model_preset": "prototype_220m",
        "semantic_hidden_dim": 256,
        "semantic_embed_dim": 256,
        "semantic_source_mainline": "crop_visual_encoder",
        "legacy_semantic_source": "hand_crafted_stats",
        "local_temporal_window": 1,
        "local_temporal_fuse_weight": 0.5,
        "future_semantic_embedding_dim": 256,
        "future_measurement_feature_dim": 0,
        "future_hypothesis_count": 1,
        "enable_future_extent_head": False,
        "enable_future_multihypothesis_head": False,
        "trace_unit_teacher_prior_dim": 512,
        "trace_unit_dim": 384,
        "trace_unit_count": 16,
        "trace_unit_slot_iters": 3,
        "trace_unit_assignment_topk": 2,
        "trace_unit_assignment_temperature": 0.5,
        "trace_unit_use_instance_prior_bias": True,
        "trace_unit_dyn_update": "gru",
        "trace_unit_sem_update": "gated_residual",
        "trace_unit_sem_alpha_min": 0.05,
        "trace_unit_sem_alpha_max": 0.65,
        "trace_unit_handshake_dim": 192,
        "trace_unit_handshake_layers": 2,
        "trace_unit_handshake_writeback": "semantic",
        "trace_unit_broadcast_residual_weight": 0.25,
        "trace_unit_broadcast_stopgrad_semantic": False,
        "stage2_structure_mode": "trace_unit_semantic_binding",
        "trace_unit_disable_instance_path": False,
        "max_entities_per_sample": 8,
        "semantic_crop_size": 64,
        "semantic_patch_radius": 12,
        "include_entity_masks_over_time": False,
        "include_full_instance_id_map": False,
    }
    for key, value in defaults.items():
        merged.setdefault(key, value)
    return SimpleNamespace(**merged)


def _load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    payload = torch.load(path, map_location=device)
    if not isinstance(payload, dict):
        raise RuntimeError(f"checkpoint must be dict: {path}")
    return payload


def _make_dataset(args: SimpleNamespace, *, split: str, max_samples_per_dataset: int) -> Stage2SemanticDataset:
    cfg = Stage2SemanticDatasetConfig(
        dataset_names=list(getattr(args, "dataset_names", ["vspw", "vipseg"])),
        split=str(split),
        contract_path=str(getattr(args, "stage2_contract_path")),
        obs_len=_safe_int(getattr(args, "obs_len", 8), 8),
        fut_len=_safe_int(getattr(args, "fut_len", 8), 8),
        max_tokens=_safe_int(getattr(args, "max_tokens", 64), 64),
        max_samples_per_dataset=int(max_samples_per_dataset),
        semantic_patch_radius=_safe_int(getattr(args, "semantic_patch_radius", 12), 12),
        semantic_crop_size=_safe_int(getattr(args, "semantic_crop_size", 64), 64),
        semantic_source_mainline=str(getattr(args, "semantic_source_mainline", "crop_visual_encoder")),
        semantic_frame_index=_safe_int(getattr(args, "semantic_frame_index", 0), 0),
        semantic_temporal_window=_safe_int(getattr(args, "local_temporal_window", 1), 1),
        predecode_cache_path=str(getattr(args, "predecode_cache_path", "")),
        teacher_semantic_cache_path=str(getattr(args, "teacher_semantic_cache_path", "")),
        max_entities_per_sample=_safe_int(getattr(args, "max_entities_per_sample", 8), 8),
        include_entity_masks_over_time=bool(getattr(args, "include_entity_masks_over_time", False)),
        include_full_instance_id_map=bool(getattr(args, "include_full_instance_id_map", False)),
    )
    return Stage2SemanticDataset(cfg)


def _load_models(args: SimpleNamespace, payload: dict[str, Any], device: torch.device, prototype_count: int) -> dict[str, Any]:
    args.enable_future_semantic_state_head = True
    args.enable_semantic_proto_head = True
    args.future_semantic_proto_count = int(prototype_count)
    args.future_measurement_feature_dim = 0
    args.future_hypothesis_count = 1
    args.enable_future_extent_head = False
    args.enable_future_multihypothesis_head = False

    stage1_model, stage1_meta = _load_frozen_stage1_backbone(args=args, device=device)
    semantic_encoder = SemanticEncoder(
        SemanticEncoderConfig(
            input_dim=10,
            hidden_dim=_safe_int(getattr(args, "semantic_hidden_dim", 256), 256),
            output_dim=_safe_int(getattr(args, "semantic_embed_dim", 256), 256),
            dropout=0.0,
            mainline_source=str(getattr(args, "semantic_source_mainline", "crop_visual_encoder")),
            legacy_source=str(getattr(args, "legacy_semantic_source", "hand_crafted_stats")),
            local_temporal_window=_safe_int(getattr(args, "local_temporal_window", 1), 1),
            local_temporal_fuse_weight=_safe_float(getattr(args, "local_temporal_fuse_weight", 0.5)),
        )
    ).to(device)
    fusion_hidden_dim = int(stage1_model.config.d_model)
    semantic_fusion = SemanticFusion(
        SemanticFusionConfig(
            hidden_dim=fusion_hidden_dim,
            semantic_dim=_safe_int(getattr(args, "semantic_embed_dim", 256), 256),
            dropout=0.0,
        )
    ).to(device)
    readout_head = torch.nn.Linear(fusion_hidden_dim, 2).to(device)
    future_semantic_state_head = SemanticTraceStateHead(
        SemanticTraceStateHeadConfig(
            hidden_dim=fusion_hidden_dim,
            semantic_embedding_dim=_safe_int(getattr(args, "future_semantic_embedding_dim", 256), 256),
            identity_embedding_dim=_safe_int(getattr(args, "future_semantic_embedding_dim", 256), 256),
            measurement_feature_dim=0,
            semantic_proto_count=int(prototype_count),
            enable_semantic_proto_head=True,
            hypothesis_count=1,
            enable_extent_head=False,
            enable_multi_hypothesis_head=False,
            dropout=0.0,
        )
    ).to(device)

    trace_unit_tokenizer = TraceUnitTokenizer(
        TraceUnitTokenizerConfig(
            hidden_dim=fusion_hidden_dim,
            semantic_dim=_safe_int(getattr(args, "semantic_embed_dim", 256), 256),
            state_dim=STATE_DIM,
            teacher_prior_dim=_safe_int(getattr(args, "trace_unit_teacher_prior_dim", 512), 512),
            unit_dim=_safe_int(getattr(args, "trace_unit_dim", 384), 384),
            unit_count=_safe_int(getattr(args, "trace_unit_count", 16), 16),
            slot_iters=_safe_int(getattr(args, "trace_unit_slot_iters", 3), 3),
            assignment_topk=_safe_int(getattr(args, "trace_unit_assignment_topk", 2), 2),
            assignment_temperature=_safe_float(getattr(args, "trace_unit_assignment_temperature", 0.5)),
            use_instance_prior_bias=bool(getattr(args, "trace_unit_use_instance_prior_bias", True)),
        )
    ).to(device)
    trace_unit_factorized_state = TraceUnitFactorizedState(
        TraceUnitFactorizedStateConfig(
            unit_dim=_safe_int(getattr(args, "trace_unit_dim", 384), 384),
            dyn_update=str(getattr(args, "trace_unit_dyn_update", "gru")),
            sem_update=str(getattr(args, "trace_unit_sem_update", "gated_residual")),
            sem_alpha_min=_safe_float(getattr(args, "trace_unit_sem_alpha_min", 0.05)),
            sem_alpha_max=_safe_float(getattr(args, "trace_unit_sem_alpha_max", 0.65)),
        )
    ).to(device)
    trace_unit_handshake = TraceUnitHandshake(
        TraceUnitHandshakeConfig(
            unit_dim=_safe_int(getattr(args, "trace_unit_dim", 384), 384),
            handshake_dim=_safe_int(getattr(args, "trace_unit_handshake_dim", 192), 192),
            layers=_safe_int(getattr(args, "trace_unit_handshake_layers", 2), 2),
            writeback=str(getattr(args, "trace_unit_handshake_writeback", "semantic")),
        )
    ).to(device)
    trace_unit_broadcast = TraceUnitBroadcast(
        TraceUnitBroadcastConfig(
            hidden_dim=fusion_hidden_dim,
            unit_dim=_safe_int(getattr(args, "trace_unit_dim", 384), 384),
            residual_weight=_safe_float(getattr(args, "trace_unit_broadcast_residual_weight", 0.25)),
            stopgrad_semantic=bool(getattr(args, "trace_unit_broadcast_stopgrad_semantic", False)),
        )
    ).to(device)

    load_notes: dict[str, Any] = {}
    if isinstance(payload.get("semantic_encoder_state_dict"), dict):
        missing, unexpected = semantic_encoder.load_state_dict(payload["semantic_encoder_state_dict"], strict=False)
        load_notes["semantic_encoder_missing"] = list(missing)
        load_notes["semantic_encoder_unexpected"] = list(unexpected)
    if isinstance(payload.get("semantic_fusion_state_dict"), dict):
        semantic_fusion.load_state_dict(payload["semantic_fusion_state_dict"], strict=False)
    if isinstance(payload.get("readout_head_state_dict"), dict):
        readout_head.load_state_dict(payload["readout_head_state_dict"], strict=False)
    if isinstance(payload.get("future_semantic_state_head_state_dict"), dict):
        source = payload["future_semantic_state_head_state_dict"]
        target = future_semantic_state_head.state_dict()
        compatible = {k: v for k, v in source.items() if k in target and tuple(v.shape) == tuple(target[k].shape)}
        skipped = sorted(k for k, v in source.items() if k in target and tuple(v.shape) != tuple(target[k].shape))
        future_semantic_state_head.load_state_dict(compatible, strict=False)
        load_notes["future_head_skipped_shape_keys"] = skipped
    if isinstance(payload.get("trace_unit_tokenizer_state_dict"), dict):
        trace_unit_tokenizer.load_state_dict(payload["trace_unit_tokenizer_state_dict"], strict=False)
    if isinstance(payload.get("trace_unit_factorized_state_state_dict"), dict):
        trace_unit_factorized_state.load_state_dict(payload["trace_unit_factorized_state_state_dict"], strict=False)
    if isinstance(payload.get("trace_unit_handshake_state_dict"), dict):
        trace_unit_handshake.load_state_dict(payload["trace_unit_handshake_state_dict"], strict=False)
    if isinstance(payload.get("trace_unit_broadcast_state_dict"), dict):
        trace_unit_broadcast.load_state_dict(payload["trace_unit_broadcast_state_dict"], strict=False)
    if isinstance(payload.get("stage1_model_state_dict"), dict):
        stage1_model.load_state_dict(payload["stage1_model_state_dict"], strict=False)

    modules = {
        "stage1_model": stage1_model,
        "semantic_encoder": semantic_encoder,
        "semantic_fusion": semantic_fusion,
        "readout_head": readout_head,
        "future_semantic_state_head": future_semantic_state_head,
        "trace_unit_tokenizer": trace_unit_tokenizer,
        "trace_unit_factorized_state": trace_unit_factorized_state,
        "trace_unit_handshake": trace_unit_handshake,
        "trace_unit_broadcast": trace_unit_broadcast,
    }
    _configure_future_semantic_tusb_unfreeze_trainability(
        stage1_model=stage1_model,
        semantic_encoder=semantic_encoder,
        semantic_fusion=semantic_fusion,
        readout_head=readout_head,
        future_semantic_state_head=future_semantic_state_head,
        semantic_state_feedback_adapter=None,
        semantic_rescue_heads=None,
        trace_unit_tokenizer=trace_unit_tokenizer,
        trace_unit_factorized_state=trace_unit_factorized_state,
        trace_unit_handshake=trace_unit_handshake,
        trace_unit_broadcast=trace_unit_broadcast,
        train_factorized_state=True,
        train_handshake=False,
        train_broadcast=True,
        train_semantic_encoder_proj=False,
        train_tokenizer=False,
        train_semantic_fusion_proj=True,
        train_readout_head=True,
        allow_mixed_params=False,
    )
    audit = _build_future_semantic_tusb_unfreeze_audit(
        enabled=True,
        stage1_model=stage1_model,
        semantic_encoder=semantic_encoder,
        semantic_fusion=semantic_fusion,
        readout_head=readout_head,
        future_semantic_state_head=future_semantic_state_head,
        semantic_state_feedback_adapter=None,
        semantic_rescue_heads=None,
        trace_unit_tokenizer=trace_unit_tokenizer,
        trace_unit_factorized_state=trace_unit_factorized_state,
        trace_unit_handshake=trace_unit_handshake,
        trace_unit_broadcast=trace_unit_broadcast,
        allow_mixed_params=False,
    )
    for module in modules.values():
        module.train()
    stage1_model.eval()
    return {
        **modules,
        "stage1_meta": stage1_meta,
        "trainability_audit": audit,
        "load_notes": load_notes,
        "fusion_hidden_dim": fusion_hidden_dim,
    }


def _make_forward_kwargs(models: dict[str, Any], args: SimpleNamespace, batch: dict[str, Any]) -> dict[str, Any]:
    return {
        "stage1_model": models["stage1_model"],
        "semantic_encoder": models["semantic_encoder"],
        "semantic_fusion": models["semantic_fusion"],
        "readout_head": models["readout_head"],
        "future_semantic_state_head": models["future_semantic_state_head"],
        "structure_mode": str(getattr(args, "stage2_structure_mode", "trace_unit_semantic_binding")),
        "trace_unit_tokenizer": models["trace_unit_tokenizer"],
        "trace_unit_factorized_state": models["trace_unit_factorized_state"],
        "trace_unit_handshake": models["trace_unit_handshake"],
        "trace_unit_broadcast": models["trace_unit_broadcast"],
        "trace_unit_disable_instance_path": bool(getattr(args, "trace_unit_disable_instance_path", False)),
        "batch": batch,
        "obs_len": _safe_int(getattr(args, "obs_len", 8), 8),
        "semantic_source_mainline": str(getattr(args, "semantic_source_mainline", "crop_visual_encoder")),
        "allow_stage1_grad": False,
        "semantic_state_feedback_adapter": None,
        "semantic_state_feedback_enabled": False,
    }


def _proto_loss_and_metrics(
    logits: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    topk: int = 5,
) -> tuple[torch.Tensor, dict[str, float]]:
    valid = mask.to(torch.bool) & (target >= 0)
    valid_count = int(valid.sum().detach().cpu().item())
    if valid_count <= 0:
        zero = logits.sum() * 0.0
        return zero, {
            "valid_count": 0,
            "target_valid_ratio": 0.0,
            "proto_ce": 0.0,
            "proto_accuracy": 0.0,
            "proto_top5": 0.0,
        }
    flat_logits = logits[valid]
    flat_target = target[valid]
    loss = F.cross_entropy(flat_logits, flat_target)
    pred = flat_logits.argmax(dim=-1)
    acc = (pred == flat_target).float().mean()
    k = min(int(topk), int(flat_logits.shape[-1]))
    top = flat_logits.topk(k=k, dim=-1).indices
    top5 = (top == flat_target[:, None]).any(dim=-1).float().mean()
    return loss, {
        "valid_count": valid_count,
        "target_valid_ratio": float(valid.float().mean().detach().cpu().item()),
        "proto_ce": float(loss.detach().cpu().item()),
        "proto_accuracy": float(acc.detach().cpu().item()),
        "proto_top5": float(top5.detach().cpu().item()),
    }


def _frequency_baseline(target: torch.Tensor, mask: torch.Tensor, prototype_count: int) -> dict[str, float]:
    valid = mask.to(torch.bool) & (target >= 0)
    if int(valid.sum().item()) <= 0:
        return {"frequency_top1": 0.0, "frequency_top5": 0.0, "frequency_ce": 0.0}
    y = target[valid].detach().cpu().long()
    counts = torch.bincount(y, minlength=int(prototype_count)).float()
    probs = counts / counts.sum().clamp_min(1.0)
    top1_label = int(probs.argmax().item())
    k = min(5, int(prototype_count))
    topk_labels = set(int(x) for x in probs.topk(k).indices.tolist())
    top1 = float((y == top1_label).float().mean().item())
    top5 = float(torch.tensor([int(int(v) in topk_labels) for v in y.tolist()], dtype=torch.float32).mean().item())
    freq_probs = probs[y].clamp_min(1e-8)
    ce = float((-torch.log(freq_probs)).mean().item())
    return {"frequency_top1": top1, "frequency_top5": top5, "frequency_ce": ce}


def _batch_slot_count(batch: dict[str, Any]) -> int:
    token_mask = batch.get("token_mask")
    if isinstance(token_mask, torch.Tensor) and token_mask.ndim == 2:
        return int(token_mask.shape[1])
    obs_state = batch.get("obs_state")
    if isinstance(obs_state, torch.Tensor) and obs_state.ndim >= 3:
        return int(obs_state.shape[2])
    return 0


def _select_samples(
    dataset: Stage2SemanticDataset,
    cache: FutureSemanticPrototypeTargetCache,
    *,
    count: int,
    horizon: int,
    slot_count: int,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    cache_index = cache.index
    for idx in range(len(dataset)):
        sample = dataset[idx]
        key = stage2_item_key(sample.get("meta", {}))
        cache_idx = cache_index.get(key)
        if cache_idx is None:
            continue
        mask = cache.mask[cache_idx, :horizon, : min(slot_count, cache.mask.shape[2])]
        if bool(mask.any().item()):
            selected.append(sample)
        if len(selected) >= int(count):
            break
    if len(selected) < int(count):
        raise RuntimeError(f"only found {len(selected)} cache-hit samples with valid targets, need {count}")
    return selected


def run_target_alignment_audit(
    *,
    dataset: Stage2SemanticDataset,
    cache: FutureSemanticPrototypeTargetCache,
    output: Path,
    doc: Path,
    count: int,
    horizon: int,
    slot_count: int,
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    cache_hits = 0
    mask_valid_total = 0
    mask_total = 0
    mask_subset_ok_all = True
    proto_mask_ok_all = True
    h_ok_all = True
    k_ok_all = True
    label_hist = torch.zeros((int(cache.prototype_count),), dtype=torch.long)
    trace_examples: list[dict[str, Any]] = []
    rng = random.Random(20260428)
    for sample_idx in range(min(len(dataset), int(count))):
        sample = dataset[sample_idx]
        meta = dict(sample.get("meta", {}))
        key = stage2_item_key(meta)
        cache_idx = cache.index.get(key)
        hit = cache_idx is not None
        cache_hits += int(hit)
        rec: dict[str, Any] = {
            "sample_index": int(sample_idx),
            "batch_item_key": key,
            "cache_item_key": key if hit else None,
            "cache_hit": bool(hit),
            "dataset": meta.get("dataset", ""),
            "clip_id": meta.get("clip_id", ""),
            "split_or_sample_id_consistent": bool(hit),
        }
        if not hit:
            records.append(rec)
            continue
        proto = cache.proto_target[cache_idx]
        mask = cache.mask[cache_idx]
        fut_valid = sample["fut_valid"].to(torch.bool)
        h = min(int(horizon), int(proto.shape[0]), int(fut_valid.shape[0]))
        k = min(int(slot_count), int(proto.shape[1]), int(fut_valid.shape[1]))
        h_ok = int(proto.shape[0]) >= int(horizon)
        k_ok = int(proto.shape[1]) >= min(int(slot_count), int(sample["fut_valid"].shape[1]))
        h_ok_all = bool(h_ok_all and h_ok)
        k_ok_all = bool(k_ok_all and k_ok)
        local_mask = mask[:h, :k].to(torch.bool)
        local_fut = fut_valid[:h, :k]
        subset_ok = bool((local_mask & (~local_fut)).sum().item() == 0)
        exact_mask = bool(torch.equal(local_mask.cpu(), local_fut.cpu()))
        mask_subset_ok_all = bool(mask_subset_ok_all and subset_ok)
        local_proto = proto[:h, :k]
        proto_mask_ok = bool((((local_proto < 0) == (~local_mask)).all()).item())
        proto_mask_ok_all = bool(proto_mask_ok_all and proto_mask_ok)
        valid_labels = local_proto[local_mask & (local_proto >= 0)]
        if int(valid_labels.numel()) > 0:
            label_hist += torch.bincount(valid_labels.cpu().long(), minlength=int(cache.prototype_count))
        mask_valid_total += int(local_mask.sum().item())
        mask_total += int(local_mask.numel())
        rec.update(
            {
                "H_cache": int(proto.shape[0]),
                "H_batch": int(fut_valid.shape[0]),
                "H_consistent": bool(h_ok),
                "K_cache": int(proto.shape[1]),
                "K_batch": int(fut_valid.shape[1]),
                "K_consistent": bool(k_ok),
                "target_mask_matches_fut_valid_exact": bool(exact_mask),
                "target_mask_subset_of_fut_valid": bool(subset_ok),
                "target_mask_valid_count": int(local_mask.sum().item()),
                "fut_valid_count": int(local_fut.sum().item()),
                "proto_target_minus_one_only_when_mask_false": bool(proto_mask_ok),
                "slot_order_proxy_point_ids": sample.get("point_ids", torch.empty(0)).detach().cpu().tolist(),
                "target_label_distribution_nonzero_count": int((torch.bincount(valid_labels.cpu().long(), minlength=int(cache.prototype_count)) > 0).sum().item())
                if int(valid_labels.numel()) > 0
                else 0,
            }
        )
        possible = [(hh, kk) for hh in range(h) for kk in range(k) if bool(local_mask[hh, kk].item())]
        rng.shuffle(possible)
        for hh, kk in possible[: max(0, 5 - len(trace_examples))]:
            extent = cache.extent_box[cache_idx, hh, kk].detach().cpu().tolist()
            future_state = sample["fut_state"][hh, kk].detach().cpu().tolist()
            trace_examples.append(
                {
                    "item_key": key,
                    "h": int(hh),
                    "k": int(kk),
                    "proto_target": int(local_proto[hh, kk].item()),
                    "target_mask": bool(local_mask[hh, kk].item()),
                    "fut_valid": bool(local_fut[hh, kk].item()),
                    "future_state_xy": [float(future_state[0]), float(future_state[1])],
                    "future_extent_box_target": [float(x) for x in extent],
                }
            )
        records.append(rec)
    label_nonzero = int((label_hist > 0).sum().item())
    cache_hit_ratio = float(cache_hits / max(min(len(dataset), int(count)), 1))
    target_valid_ratio = float(mask_valid_total / max(mask_total, 1))
    alignment_ok = bool(cache_hit_ratio == 1.0 and h_ok_all and k_ok_all and mask_subset_ok_all and proto_mask_ok_all and target_valid_ratio > 0.0)
    payload = {
        "audit_name": "stwm_semantic_field_debug_v1_target_alignment_audit",
        "checked_item_count": int(min(len(dataset), int(count))),
        "cache_path": cache.cache_path,
        "prototype_count": int(cache.prototype_count),
        "cache_hit_ratio": cache_hit_ratio,
        "target_valid_ratio": target_valid_ratio,
        "target_mask_subset_of_fut_valid_all": bool(mask_subset_ok_all),
        "target_mask_exact_match_ratio": float(
            sum(1 for r in records if r.get("target_mask_matches_fut_valid_exact")) / max(sum(1 for r in records if r.get("cache_hit")), 1)
        ),
        "proto_target_minus_one_only_when_mask_false_all": bool(proto_mask_ok_all),
        "H_consistent_all": bool(h_ok_all),
        "K_consistent_all": bool(k_ok_all),
        "slot_order_audit_method": "point_ids proxy plus cache/sample key equality; visual crop trace examples recorded",
        "label_nonzero_count": label_nonzero,
        "label_histogram_top10": (
            [
                {"prototype": int(idx), "count": int(count)}
                for count, idx in zip(
                    torch.topk(label_hist, k=min(10, int(cache.prototype_count))).values.tolist(),
                    torch.topk(label_hist, k=min(10, int(cache.prototype_count))).indices.tolist(),
                )
            ]
            if int(label_hist.sum().item()) > 0
            else []
        ),
        "trace_examples": trace_examples,
        "records": records,
        "alignment_ok": alignment_ok,
    }
    write_json(output, payload)
    write_doc(
        doc,
        "STWM Semantic Field Debug V1 Target Alignment Audit",
        payload,
        bullets=[
            "Alignment passes only if cache hits are complete, shape is compatible, valid targets never supervise invalid future slots, and -1 labels only appear under mask=false.",
            "Exact target_mask == fut_valid is reported separately because feature-target validity can be a subset of future visibility.",
        ],
    )
    return payload


def _tensor_stats(x: torch.Tensor | None, mask: torch.Tensor | None = None) -> dict[str, float]:
    if x is None:
        return {"available": 0.0, "mean": 0.0, "std": 0.0, "norm_mean": 0.0, "var": 0.0}
    t = x.detach().float()
    if mask is not None:
        m = mask.to(device=t.device, dtype=torch.bool)
        while m.ndim < t.ndim:
            m = m.unsqueeze(-1)
        selected = t[m.expand_as(t)].reshape(-1)
        if selected.numel() == 0:
            selected = t.reshape(-1)
    else:
        selected = t.reshape(-1)
    norm = t.norm(dim=-1).reshape(-1) if t.ndim >= 1 else t.reshape(-1).abs()
    return {
        "available": 1.0,
        "mean": float(selected.mean().cpu().item()) if selected.numel() else 0.0,
        "std": float(selected.std(unbiased=False).cpu().item()) if selected.numel() else 0.0,
        "norm_mean": float(norm.mean().cpu().item()) if norm.numel() else 0.0,
        "var": float(selected.var(unbiased=False).cpu().item()) if selected.numel() else 0.0,
    }


def run_semantic_input_audit(
    *,
    models: dict[str, Any],
    args: SimpleNamespace,
    batch: dict[str, Any],
    cache: FutureSemanticPrototypeTargetCache,
    device: torch.device,
    output: Path,
    doc: Path,
) -> dict[str, Any]:
    for module_name in [
        "semantic_encoder",
        "semantic_fusion",
        "readout_head",
        "future_semantic_state_head",
        "trace_unit_tokenizer",
        "trace_unit_factorized_state",
        "trace_unit_handshake",
        "trace_unit_broadcast",
    ]:
        models[module_name].eval()
    with torch.no_grad():
        out = _teacher_forced_predict(**_make_forward_kwargs(models, args, batch))
        target, _, mask, info = prototype_tensors_for_batch(
            cache,
            batch,
            horizon=_safe_int(getattr(args, "fut_len", 8), 8),
            slot_count=_batch_slot_count(batch),
            device=device,
        )
    state = out["future_semantic_trace_state"]
    proto_logits = state.future_semantic_proto_logits if state is not None else None
    semantic_tokens = out.get("semantic_tokens")
    trace_aux = out.get("trace_unit_aux", {}) or {}
    z_sem = trace_aux.get("z_sem")
    future_hidden = out.get("future_fused_hidden")
    rgb = batch.get("semantic_rgb_crop")
    mask_crop = batch.get("semantic_mask_crop")
    sem_features = batch.get("semantic_features")
    rgb_nonzero = float((rgb.abs() > 1e-6).float().mean().detach().cpu().item()) if isinstance(rgb, torch.Tensor) else 0.0
    mask_nonzero = float((mask_crop.abs() > 1e-6).float().mean().detach().cpu().item()) if isinstance(mask_crop, torch.Tensor) else 0.0
    semantic_mask = batch.get("semantic_mask")
    semantic_input_nonempty = bool((semantic_mask & batch["token_mask"]).any().item()) if isinstance(semantic_mask, torch.Tensor) else False
    payload = {
        "audit_name": "stwm_semantic_field_debug_v1_semantic_input_audit",
        "semantic_input_nonempty": semantic_input_nonempty,
        "semantic_rgb_crop_nonzero_ratio": rgb_nonzero,
        "semantic_mask_crop_nonzero_ratio": mask_nonzero,
        "semantic_features_stats": _tensor_stats(sem_features, batch.get("semantic_mask")),
        "semantic_encoder_output_stats": _tensor_stats(semantic_tokens, batch.get("semantic_mask")),
        "semantic_fusion_gate_mean": float(out.get("gate_mean", 0.0)),
        "semantic_fusion_gate_std": float(out.get("gate_std", 0.0)),
        "semantic_tokens_variance_across_slots": float(semantic_tokens.detach().float().var(dim=1, unbiased=False).mean().cpu().item())
        if isinstance(semantic_tokens, torch.Tensor) and semantic_tokens.shape[1] > 1
        else 0.0,
        "z_sem_stats": _tensor_stats(z_sem),
        "future_hidden_stats": _tensor_stats(future_hidden),
        "future_semantic_proto_logits_stats": _tensor_stats(proto_logits),
        "prototype_target_info": info,
        "prototype_target_valid_ratio": float(info.get("target_valid_ratio", 0.0)),
        "semantic_input_valid": bool(semantic_input_nonempty and rgb_nonzero > 1e-4 and _tensor_stats(semantic_tokens)["var"] > 1e-8),
        "hidden_has_nonzero_semantic_variance": bool(
            _tensor_stats(z_sem)["var"] > 1e-8
            and _tensor_stats(future_hidden)["var"] > 1e-8
            and _tensor_stats(proto_logits)["var"] > 1e-8
        ),
    }
    write_json(output, payload)
    write_doc(
        doc,
        "STWM Semantic Field Debug V1 Semantic Input Audit",
        payload,
        bullets=[
            "This audit checks whether observed semantic crops/features are non-empty and whether the semantic path produces non-zero variance before any overfit claim.",
        ],
    )
    return payload


def _grad_norm(named_params: Iterable[tuple[str, torch.nn.Parameter]]) -> tuple[float, list[str]]:
    sq = 0.0
    names: list[str] = []
    for name, param in named_params:
        if param.grad is None:
            continue
        norm = float(param.grad.detach().float().norm().cpu().item())
        if norm > 0.0:
            names.append(str(name))
        sq += norm * norm
    return math.sqrt(sq), names[:20]


def _module_named_params(module: torch.nn.Module | None, prefix: str = "") -> list[tuple[str, torch.nn.Parameter]]:
    if module is None:
        return []
    return [(f"{prefix}{name}", p) for name, p in module.named_parameters()]


def _semantic_named_params(module: torch.nn.Module | None, prefix: str = "") -> list[tuple[str, torch.nn.Parameter]]:
    if module is None or not hasattr(module, "semantic_named_parameters"):
        return []
    return [(f"{prefix}{name}", p) for name, p in module.semantic_named_parameters()]


def _dynamic_named_params(module: torch.nn.Module | None, prefix: str = "") -> list[tuple[str, torch.nn.Parameter]]:
    if module is None or not hasattr(module, "dynamic_named_parameters"):
        return []
    return [(f"{prefix}{name}", p) for name, p in module.dynamic_named_parameters()]


def run_gradient_audit(
    *,
    models: dict[str, Any],
    args: SimpleNamespace,
    batch: dict[str, Any],
    cache: FutureSemanticPrototypeTargetCache,
    device: torch.device,
    output: Path,
    doc: Path,
) -> dict[str, Any]:
    for module_name in [
        "semantic_encoder",
        "semantic_fusion",
        "readout_head",
        "future_semantic_state_head",
        "trace_unit_tokenizer",
        "trace_unit_factorized_state",
        "trace_unit_handshake",
        "trace_unit_broadcast",
    ]:
        models[module_name].train()
        models[module_name].zero_grad(set_to_none=True)
    models["stage1_model"].zero_grad(set_to_none=True)
    out = _teacher_forced_predict(**_make_forward_kwargs(models, args, batch))
    target, _, mask, info = prototype_tensors_for_batch(
        cache,
        batch,
        horizon=_safe_int(getattr(args, "fut_len", 8), 8),
        slot_count=_batch_slot_count(batch),
        device=device,
    )
    state = out["future_semantic_trace_state"]
    if state is None or state.future_semantic_proto_logits is None or target is None or mask is None:
        raise RuntimeError("semantic proto logits/targets unavailable for gradient audit")
    loss, metrics = _proto_loss_and_metrics(state.future_semantic_proto_logits, target, mask)
    loss.backward()

    future_head_norm, future_head_names = _grad_norm(_module_named_params(models["future_semantic_state_head"], "future_semantic_state_head."))
    fusion_proj = getattr(models["semantic_fusion"], "semantic_proj", None)
    fusion_proj_norm, fusion_proj_names = _grad_norm(_module_named_params(fusion_proj, "semantic_fusion.semantic_proj."))
    factor_sem_norm, factor_sem_names = _grad_norm(_semantic_named_params(models["trace_unit_factorized_state"], "trace_unit_factorized_state."))
    broadcast_sem_norm, broadcast_sem_names = _grad_norm(_semantic_named_params(models["trace_unit_broadcast"], "trace_unit_broadcast."))
    handshake_sem_norm, handshake_sem_names = _grad_norm(_semantic_named_params(models["trace_unit_handshake"], "trace_unit_handshake."))
    factor_dyn_norm, factor_dyn_names = _grad_norm(_dynamic_named_params(models["trace_unit_factorized_state"], "trace_unit_factorized_state."))
    broadcast_dyn_norm, broadcast_dyn_names = _grad_norm(_dynamic_named_params(models["trace_unit_broadcast"], "trace_unit_broadcast."))
    stage1_norm, stage1_names = _grad_norm(_module_named_params(models["stage1_model"], "stage1_model."))

    semantic_grad_norm = float(math.sqrt(future_head_norm**2 + fusion_proj_norm**2 + factor_sem_norm**2 + broadcast_sem_norm**2 + handshake_sem_norm**2))
    dynamic_grad_norm = float(math.sqrt(factor_dyn_norm**2 + broadcast_dyn_norm**2))
    payload = {
        "audit_name": "stwm_semantic_field_debug_v1_gradient_audit",
        "loss_components": {
            "teacher_loss": 0.0,
            "rescue_loss": 0.0,
            "trace_unit_loss": 0.0,
            "future_semantic_proto_loss": float(metrics["proto_ce"]),
            "future_semantic_proto_soft_loss": 0.0,
            "future_visibility_loss": 0.0,
            "total_loss": float(metrics["proto_ce"]),
        },
        "prototype_target_info": info,
        "grad_norms": {
            "future_semantic_state_head": future_head_norm,
            "semantic_fusion.semantic_proj": fusion_proj_norm,
            "trace_unit_factorized_state_semantic": factor_sem_norm,
            "trace_unit_broadcast_semantic": broadcast_sem_norm,
            "trace_unit_handshake_semantic": handshake_sem_norm,
            "trace_unit_factorized_state_dynamic": factor_dyn_norm,
            "trace_unit_broadcast_dynamic": broadcast_dyn_norm,
            "stage1": stage1_norm,
            "all_enabled_semantic": semantic_grad_norm,
            "all_dynamic": dynamic_grad_norm,
        },
        "grad_nonzero_param_examples": {
            "future_semantic_state_head": future_head_names,
            "semantic_fusion.semantic_proj": fusion_proj_names,
            "trace_unit_factorized_state_semantic": factor_sem_names,
            "trace_unit_broadcast_semantic": broadcast_sem_names,
            "trace_unit_handshake_semantic": handshake_sem_names,
            "trace_unit_factorized_state_dynamic": factor_dyn_names,
            "trace_unit_broadcast_dynamic": broadcast_dyn_names,
            "stage1": stage1_names,
        },
        "proto_loss_grad_reaches_future_head": bool(future_head_norm > 0.0),
        "proto_loss_grad_reaches_tusb_semantic": bool(factor_sem_norm > 0.0 or broadcast_sem_norm > 0.0 or handshake_sem_norm > 0.0),
        "proto_loss_grad_reaches_tusb_dynamic": bool(dynamic_grad_norm > 0.0),
        "stage1_grad_detected": bool(stage1_norm > 0.0),
        "dynamic_grad_detected": bool(dynamic_grad_norm > 0.0),
        "semantic_grad_norm_by_module": {
            "future_semantic_state_head": future_head_norm,
            "semantic_fusion.semantic_proj": fusion_proj_norm,
            "trace_unit_factorized_state": factor_sem_norm,
            "trace_unit_broadcast": broadcast_sem_norm,
            "trace_unit_handshake": handshake_sem_norm,
        },
    }
    write_json(output, payload)
    write_doc(
        doc,
        "STWM Semantic Field Debug V1 Gradient Audit",
        payload,
        bullets=[
            "The backward pass optimizes only semantic prototype CE on a fixed batch.",
            "Stage1 and TUSB dynamic paths should have zero gradient.",
        ],
    )
    return payload


def _trainable_parameters(models: dict[str, Any]) -> list[torch.nn.Parameter]:
    params: list[torch.nn.Parameter] = []
    seen: set[int] = set()
    for name in [
        "semantic_encoder",
        "semantic_fusion",
        "readout_head",
        "future_semantic_state_head",
        "trace_unit_tokenizer",
        "trace_unit_factorized_state",
        "trace_unit_handshake",
        "trace_unit_broadcast",
    ]:
        for p in models[name].parameters():
            if p.requires_grad and id(p) not in seen:
                seen.add(id(p))
                params.append(p)
    return params


def run_one_batch_overfit(
    *,
    start_payload: dict[str, Any],
    checkpoint_args: dict[str, Any],
    device: torch.device,
    batch_cpu: dict[str, Any],
    target_cache_path: Path,
    prototype_count: int,
    lr_sweep: list[float],
    steps: int,
    output: Path,
    doc_fragments: list[str],
) -> dict[str, Any]:
    best: dict[str, Any] | None = None
    runs: list[dict[str, Any]] = []
    cache = load_future_semantic_prototype_target_cache(target_cache_path)
    assert cache is not None
    for lr in lr_sweep:
        args = _merge_args(
            checkpoint_args,
            {
                "future_semantic_proto_count": int(prototype_count),
                "enable_future_semantic_state_head": True,
                "enable_semantic_proto_head": True,
            },
        )
        models = _load_models(args, start_payload, device, int(prototype_count))
        optimizer = torch.optim.AdamW(_trainable_parameters(models), lr=float(lr), weight_decay=0.0)
        batch = _to_device(batch_cpu, device, non_blocking=False)
        start_metrics: dict[str, float] | None = None
        last_metrics: dict[str, float] | None = None
        finite_count = 0
        checkpoints: list[dict[str, float]] = []
        for step in range(int(steps) + 1):
            optimizer.zero_grad(set_to_none=True)
            out = _teacher_forced_predict(**_make_forward_kwargs(models, args, batch))
            target, _, mask, info = prototype_tensors_for_batch(
                cache,
                batch,
                horizon=_safe_int(getattr(args, "fut_len", 8), 8),
                slot_count=_batch_slot_count(batch),
                device=device,
            )
            state = out["future_semantic_trace_state"]
            if state is None or state.future_semantic_proto_logits is None:
                raise RuntimeError("future_semantic_proto_logits missing during overfit")
            loss, metrics = _proto_loss_and_metrics(state.future_semantic_proto_logits, target, mask)
            if step == 0:
                start_metrics = dict(metrics)
                freq = _frequency_baseline(target, mask, int(prototype_count))
            finite = bool(torch.isfinite(loss).detach().cpu().item())
            finite_count += int(finite)
            if step < int(steps):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(_trainable_parameters(models), max_norm=10.0)
                optimizer.step()
            last_metrics = dict(metrics)
            if step in {0, max(1, int(steps) // 4), max(1, int(steps) // 2), int(steps)}:
                checkpoints.append(
                    {
                        "step": float(step),
                        "proto_ce": float(metrics["proto_ce"]),
                        "proto_accuracy": float(metrics["proto_accuracy"]),
                        "proto_top5": float(metrics["proto_top5"]),
                    }
                )
        assert start_metrics is not None and last_metrics is not None
        ce_drop = float(start_metrics["proto_ce"] - last_metrics["proto_ce"])
        overfit_success = bool(last_metrics["proto_top5"] >= 0.8 or last_metrics["proto_ce"] < float(freq["frequency_ce"]) - 0.25)
        run = {
            "prototype_count": int(prototype_count),
            "lr": float(lr),
            "steps": int(steps),
            "train_proto_ce_start": float(start_metrics["proto_ce"]),
            "train_proto_ce_end": float(last_metrics["proto_ce"]),
            "train_proto_ce_drop": ce_drop,
            "train_proto_accuracy_start": float(start_metrics["proto_accuracy"]),
            "train_proto_accuracy_end": float(last_metrics["proto_accuracy"]),
            "train_proto_top5_start": float(start_metrics["proto_top5"]),
            "train_proto_top5_end": float(last_metrics["proto_top5"]),
            "frequency_baseline": freq,
            "target_valid_ratio": float(info.get("target_valid_ratio", 0.0)),
            "loss_finite_ratio": float(finite_count / max(int(steps) + 1, 1)),
            "metric_curve": checkpoints,
            "overfit_success": overfit_success,
        }
        runs.append(run)
        if best is None or run["train_proto_top5_end"] > best["train_proto_top5_end"] or (
            run["train_proto_top5_end"] == best["train_proto_top5_end"] and run["train_proto_ce_end"] < best["train_proto_ce_end"]
        ):
            best = run
    assert best is not None
    payload = {
        "audit_name": f"stwm_semantic_field_debug_v1_one_batch_overfit_c{prototype_count}",
        "prototype_count": int(prototype_count),
        "target_cache_path": str(target_cache_path),
        "batch_size": int(batch_cpu["obs_state"].shape[0]),
        "lr_sweep": [float(x) for x in lr_sweep],
        "steps_per_lr": int(steps),
        "loss_mode": "semantic_proto_ce_only",
        "teacher_loss_enabled": False,
        "rescue_loss_enabled": False,
        "trace_unit_loss_enabled": False,
        "candidate_scorer_enabled": False,
        "runs": runs,
        "best_lr": float(best["lr"]),
        "train_proto_ce_start": float(best["train_proto_ce_start"]),
        "train_proto_ce_end": float(best["train_proto_ce_end"]),
        "train_proto_accuracy_start": float(best["train_proto_accuracy_start"]),
        "train_proto_accuracy_end": float(best["train_proto_accuracy_end"]),
        "train_proto_top5_start": float(best["train_proto_top5_start"]),
        "train_proto_top5_end": float(best["train_proto_top5_end"]),
        "frequency_baseline": best["frequency_baseline"],
        "target_valid_ratio": float(best["target_valid_ratio"]),
        "loss_finite_ratio": float(best["loss_finite_ratio"]),
        "overfit_success": bool(best["overfit_success"]),
        "success_criteria": "top5 > 0.8 on fixed batch OR CE strongly below frequency baseline",
    }
    write_json(output, payload)
    doc_fragments.append(
        f"## C{prototype_count}\n\n"
        f"- best_lr: `{payload['best_lr']}`\n"
        f"- CE start/end: `{payload['train_proto_ce_start']}` -> `{payload['train_proto_ce_end']}`\n"
        f"- top5 start/end: `{payload['train_proto_top5_start']}` -> `{payload['train_proto_top5_end']}`\n"
        f"- frequency top5: `{payload['frequency_baseline']['frequency_top5']}`\n"
        f"- overfit_success: `{payload['overfit_success']}`\n"
    )
    return payload


def _build_tiny_overfit_skipped(reason: str, output: Path | None = None) -> dict[str, Any]:
    payload = {
        "audit_name": "stwm_semantic_field_debug_v1_tiny_overfit_summary",
        "tiny_overfit_started": False,
        "tiny_overfit_success": False,
        "skipped_reason": reason,
    }
    if output is not None:
        write_json(output, payload)
    return payload


def _evaluate_batches(
    *,
    models: dict[str, Any],
    args: SimpleNamespace,
    batches_cpu: list[dict[str, Any]],
    cache: FutureSemanticPrototypeTargetCache,
    device: torch.device,
) -> dict[str, float]:
    weighted_ce = 0.0
    weighted_acc = 0.0
    weighted_top5 = 0.0
    valid_total = 0
    freq_targets: list[torch.Tensor] = []
    freq_masks: list[torch.Tensor] = []
    with torch.no_grad():
        for batch_cpu in batches_cpu:
            batch = _to_device(batch_cpu, device, non_blocking=False)
            out = _teacher_forced_predict(**_make_forward_kwargs(models, args, batch))
            target, _, mask, _ = prototype_tensors_for_batch(
                cache,
                batch,
                horizon=_safe_int(getattr(args, "fut_len", 8), 8),
                slot_count=_batch_slot_count(batch),
                device=device,
            )
            state = out["future_semantic_trace_state"]
            if state is None or state.future_semantic_proto_logits is None:
                continue
            _, metrics = _proto_loss_and_metrics(state.future_semantic_proto_logits, target, mask)
            valid = int(metrics.get("valid_count", 0))
            valid_total += valid
            weighted_ce += float(metrics.get("proto_ce", 0.0)) * valid
            weighted_acc += float(metrics.get("proto_accuracy", 0.0)) * valid
            weighted_top5 += float(metrics.get("proto_top5", 0.0)) * valid
            freq_targets.append(target.detach().cpu())
            freq_masks.append(mask.detach().cpu())
    denom = max(valid_total, 1)
    if freq_targets:
        cat_target = torch.cat([x.reshape(-1) for x in freq_targets]).reshape(1, -1)
        cat_mask = torch.cat([x.reshape(-1) for x in freq_masks]).reshape(1, -1)
        freq = _frequency_baseline(cat_target, cat_mask, int(cache.prototype_count))
    else:
        freq = {"frequency_top1": 0.0, "frequency_top5": 0.0, "frequency_ce": 0.0}
    return {
        "proto_ce": float(weighted_ce / denom),
        "proto_accuracy": float(weighted_acc / denom),
        "proto_top5": float(weighted_top5 / denom),
        "valid_count": int(valid_total),
        **freq,
    }


def run_tiny_overfit(
    *,
    start_payload: dict[str, Any],
    checkpoint_args: dict[str, Any],
    device: torch.device,
    dataset: Stage2SemanticDataset,
    cache_paths: dict[int, Path],
    best_lrs: dict[int, float],
    steps: int,
    batch_size: int,
    output: Path,
    doc: Path,
) -> dict[str, Any]:
    cache64 = load_future_semantic_prototype_target_cache(cache_paths[64])
    assert cache64 is not None
    selected = _select_samples(
        dataset,
        cache64,
        count=32,
        horizon=_safe_int(checkpoint_args.get("fut_len", 8), 8),
        slot_count=_safe_int(checkpoint_args.get("max_tokens", 64), 64),
    )
    train_samples = selected[:24]
    val_samples = selected[24:32]

    def make_batches(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            stage2_semantic_collate_fn(samples[i : i + int(batch_size)])
            for i in range(0, len(samples), int(batch_size))
            if samples[i : i + int(batch_size)]
        ]

    train_batches = make_batches(train_samples)
    val_batches = make_batches(val_samples)
    run_payloads: list[dict[str, Any]] = []
    best_run: dict[str, Any] | None = None
    for c in [32, 64]:
        cache = load_future_semantic_prototype_target_cache(cache_paths[c])
        assert cache is not None
        lr = float(best_lrs.get(c, 1e-5) or 1e-5)
        args = _merge_args(
            checkpoint_args,
            {
                "future_semantic_proto_count": int(c),
                "enable_future_semantic_state_head": True,
                "enable_semantic_proto_head": True,
            },
        )
        models = _load_models(args, start_payload, device, int(c))
        optimizer = torch.optim.AdamW(_trainable_parameters(models), lr=lr, weight_decay=0.0)
        finite_count = 0
        curve: list[dict[str, float]] = []
        start_train = _evaluate_batches(models=models, args=args, batches_cpu=train_batches, cache=cache, device=device)
        start_val = _evaluate_batches(models=models, args=args, batches_cpu=val_batches, cache=cache, device=device)
        for step in range(int(steps)):
            batch_cpu = train_batches[step % len(train_batches)]
            batch = _to_device(batch_cpu, device, non_blocking=False)
            optimizer.zero_grad(set_to_none=True)
            out = _teacher_forced_predict(**_make_forward_kwargs(models, args, batch))
            target, _, mask, _ = prototype_tensors_for_batch(
                cache,
                batch,
                horizon=_safe_int(getattr(args, "fut_len", 8), 8),
                slot_count=_batch_slot_count(batch),
                device=device,
            )
            state = out["future_semantic_trace_state"]
            if state is None or state.future_semantic_proto_logits is None:
                raise RuntimeError("future_semantic_proto_logits missing during tiny overfit")
            loss, metrics = _proto_loss_and_metrics(state.future_semantic_proto_logits, target, mask)
            finite = bool(torch.isfinite(loss).detach().cpu().item())
            finite_count += int(finite)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(_trainable_parameters(models), max_norm=10.0)
            optimizer.step()
            if step in {0, max(1, int(steps) // 4), max(1, int(steps) // 2), int(steps) - 1}:
                curve.append(
                    {
                        "step": float(step + 1),
                        "batch_proto_ce": float(metrics["proto_ce"]),
                        "batch_proto_accuracy": float(metrics["proto_accuracy"]),
                        "batch_proto_top5": float(metrics["proto_top5"]),
                    }
                )
        end_train = _evaluate_batches(models=models, args=args, batches_cpu=train_batches, cache=cache, device=device)
        end_val = _evaluate_batches(models=models, args=args, batches_cpu=val_batches, cache=cache, device=device)
        train_top5_gain = float(end_train["proto_top5"] - start_train["proto_top5"])
        overfit_gap = float(end_train["proto_top5"] - end_val["proto_top5"])
        can_memorize = bool(end_train["proto_top5"] >= 0.7 or (end_train["proto_ce"] < end_train["frequency_ce"] - 0.25))
        run = {
            "prototype_count": int(c),
            "lr": lr,
            "steps": int(steps),
            "train_metrics_start": start_train,
            "tiny_val_metrics_start": start_val,
            "train_metrics_end": end_train,
            "tiny_val_metrics_end": end_val,
            "train_top5_gain": train_top5_gain,
            "overfit_gap_top5": overfit_gap,
            "loss_finite_ratio": float(finite_count / max(int(steps), 1)),
            "curve": curve,
            "can_memorize_semantic_field_when_target_aligned": can_memorize,
        }
        run_payloads.append(run)
        if best_run is None or run["train_metrics_end"]["proto_top5"] > best_run["train_metrics_end"]["proto_top5"]:
            best_run = run
        del models
        if device.type == "cuda":
            torch.cuda.empty_cache()
    assert best_run is not None
    tiny_success = bool(any(run["can_memorize_semantic_field_when_target_aligned"] for run in run_payloads))
    payload = {
        "audit_name": "stwm_semantic_field_debug_v1_tiny_overfit_summary",
        "tiny_overfit_started": True,
        "tiny_overfit_success": tiny_success,
        "item_count": 32,
        "train_item_count": 24,
        "tiny_val_item_count": 8,
        "batch_size": int(batch_size),
        "steps_per_prototype_count": int(steps),
        "loss_mode": "semantic_proto_ce_only",
        "runs": run_payloads,
        "best_prototype_count": int(best_run["prototype_count"]),
        "best_train_top5": float(best_run["train_metrics_end"]["proto_top5"]),
        "best_tiny_val_top5": float(best_run["tiny_val_metrics_end"]["proto_top5"]),
        "best_overfit_gap_top5": float(best_run["overfit_gap_top5"]),
        "model_can_memorize_semantic_field_when_target_aligned": tiny_success,
    }
    write_json(output, payload)
    write_doc(
        doc,
        "STWM Semantic Field Debug V1 Tiny Overfit Summary",
        payload,
        bullets=[
            "Tiny overfit uses 24 train items and 8 heldout tiny-val items from the same cache-aligned subset.",
            "The objective is semantic prototype CE only; no candidate scorer, teacher loss, rescue loss, or trace-unit auxiliary loss is enabled.",
        ],
    )
    return payload


def _root_cause_and_recommendation(
    *,
    alignment: dict[str, Any],
    semantic_input: dict[str, Any],
    gradient: dict[str, Any],
    one_c32: dict[str, Any],
    one_c64: dict[str, Any],
    tiny: dict[str, Any],
) -> tuple[str, str]:
    if not bool(alignment.get("alignment_ok")):
        return "target_cache_misaligned", "fix_target_alignment"
    if not bool(semantic_input.get("semantic_input_valid")):
        return "semantic_input_missing", "fix_semantic_input_pipeline"
    if not bool(gradient.get("proto_loss_grad_reaches_tusb_semantic")):
        return "gradient_not_reaching_semantic_branch", "fix_gradient_path"
    one_success = bool(one_c32.get("overfit_success")) or bool(one_c64.get("overfit_success"))
    tiny_success = bool(tiny.get("tiny_overfit_success"))
    if one_success and tiny_success:
        return "none_overfit_chain_passed", "run_semantic_field_medium_training"
    if one_success and not tiny_success:
        return "model_capacity_or_training_recipe", "improve_prototype_targets"
    c32_drop = float(one_c32.get("train_proto_ce_start", 0.0)) - float(one_c32.get("train_proto_ce_end", 0.0))
    c64_drop = float(one_c64.get("train_proto_ce_start", 0.0)) - float(one_c64.get("train_proto_ce_end", 0.0))
    if max(c32_drop, c64_drop) > 0.5:
        return "model_capacity_or_training_recipe", "run_semantic_field_medium_training"
    return "prototype_target_unlearnable", "redesign_semantic_target_space"


def main() -> None:
    parser = argparse.ArgumentParser(description="STWM semantic field training debug V1")
    parser.add_argument("--start-checkpoint", default="outputs/checkpoints/stage2_tusb_semantic_only_unfreeze_v1_boundary_audit_20260428/latest.pt")
    parser.add_argument("--target-cache-c32", default="reports/stwm_future_semantic_trace_prototype_targets_v2_c32_20260428.json")
    parser.add_argument("--target-cache-c64", default="reports/stwm_future_semantic_trace_prototype_targets_v2_c64_20260428.json")
    parser.add_argument("--split", default="train")
    parser.add_argument("--max-samples-per-dataset", type=int, default=64)
    parser.add_argument("--alignment-count", type=int, default=32)
    parser.add_argument("--overfit-batch-size", type=int, default=4)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--tiny-steps", type=int, default=1000)
    parser.add_argument("--lr-sweep", nargs="+", type=float, default=[1e-6, 3e-6, 1e-5])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=20260428)
    parser.add_argument("--reports-dir", default="reports")
    parser.add_argument("--docs-dir", default="docs")
    args_cli = parser.parse_args()

    os.environ.setdefault("STWM_PROC_TITLE", "python")
    random.seed(int(args_cli.seed))
    torch.manual_seed(int(args_cli.seed))
    device = torch.device(str(args_cli.device))
    start_ckpt = Path(args_cli.start_checkpoint)
    payload = _load_checkpoint(start_ckpt, device=torch.device("cpu"))
    checkpoint_args = payload.get("args", {}) if isinstance(payload.get("args", {}), dict) else {}
    base_args = _merge_args(checkpoint_args, {"future_semantic_proto_count": 64})
    dataset = _make_dataset(base_args, split=str(args_cli.split), max_samples_per_dataset=int(args_cli.max_samples_per_dataset))
    cache64 = load_future_semantic_prototype_target_cache(Path(args_cli.target_cache_c64))
    cache32 = load_future_semantic_prototype_target_cache(Path(args_cli.target_cache_c32))
    if cache64 is None or cache32 is None:
        raise RuntimeError("C32 and C64 prototype target caches are required")

    reports_dir = Path(args_cli.reports_dir)
    docs_dir = Path(args_cli.docs_dir)
    alignment_path = reports_dir / "stwm_semantic_field_debug_v1_target_alignment_audit_20260428.json"
    semantic_input_path = reports_dir / "stwm_semantic_field_debug_v1_semantic_input_audit_20260428.json"
    gradient_path = reports_dir / "stwm_semantic_field_debug_v1_gradient_audit_20260428.json"
    c32_path = reports_dir / "stwm_semantic_field_debug_v1_one_batch_overfit_c32.json"
    c64_path = reports_dir / "stwm_semantic_field_debug_v1_one_batch_overfit_c64.json"
    tiny_path = reports_dir / "stwm_semantic_field_debug_v1_tiny_overfit_summary_20260428.json"
    decision_path = reports_dir / "stwm_semantic_field_debug_v1_decision_20260428.json"
    guardrail_path = reports_dir / "stwm_world_model_no_drift_guardrail_v23_20260428.json"

    alignment = run_target_alignment_audit(
        dataset=dataset,
        cache=cache64,
        output=alignment_path,
        doc=docs_dir / "STWM_SEMANTIC_FIELD_DEBUG_V1_TARGET_ALIGNMENT_AUDIT_20260428.md",
        count=int(args_cli.alignment_count),
        horizon=_safe_int(getattr(base_args, "fut_len", 8), 8),
        slot_count=_safe_int(getattr(base_args, "max_tokens", 64), 64),
    )
    if not bool(alignment.get("alignment_ok")):
        semantic_input = {
            "semantic_input_valid": False,
            "skipped_reason": "target_alignment_failed",
        }
        gradient = {
            "proto_loss_grad_reaches_tusb_semantic": False,
            "stage1_grad_detected": False,
            "dynamic_grad_detected": False,
            "skipped_reason": "target_alignment_failed",
        }
        one_c32 = {
            "overfit_success": False,
            "skipped_reason": "target_alignment_failed",
        }
        one_c64 = {
            "overfit_success": False,
            "skipped_reason": "target_alignment_failed",
        }
        tiny = _build_tiny_overfit_skipped("target_alignment_failed", tiny_path)
    else:
        selected = _select_samples(
            dataset,
            cache64,
            count=int(args_cli.overfit_batch_size),
            horizon=_safe_int(getattr(base_args, "fut_len", 8), 8),
            slot_count=_safe_int(getattr(base_args, "max_tokens", 64), 64),
        )
        batch_cpu = stage2_semantic_collate_fn(selected)

        models64 = _load_models(
            _merge_args(checkpoint_args, {"future_semantic_proto_count": 64}),
            payload,
            device,
            prototype_count=64,
        )
        batch_dev = _to_device(batch_cpu, device, non_blocking=False)
        semantic_input = run_semantic_input_audit(
            models=models64,
            args=_merge_args(checkpoint_args, {"future_semantic_proto_count": 64}),
            batch=batch_dev,
            cache=cache64,
            device=device,
            output=semantic_input_path,
            doc=docs_dir / "STWM_SEMANTIC_FIELD_DEBUG_V1_SEMANTIC_INPUT_AUDIT_20260428.md",
        )
        gradient = run_gradient_audit(
            models=models64,
            args=_merge_args(checkpoint_args, {"future_semantic_proto_count": 64}),
            batch=batch_dev,
            cache=cache64,
            device=device,
            output=gradient_path,
            doc=docs_dir / "STWM_SEMANTIC_FIELD_DEBUG_V1_GRADIENT_AUDIT_20260428.md",
        )
        del models64
        if device.type == "cuda":
            torch.cuda.empty_cache()

        doc_fragments: list[str] = ["# STWM Semantic Field Debug V1 One-Batch Overfit", ""]
        one_c32 = run_one_batch_overfit(
            start_payload=payload,
            checkpoint_args=checkpoint_args,
            device=device,
            batch_cpu=batch_cpu,
            target_cache_path=Path(args_cli.target_cache_c32),
            prototype_count=32,
            lr_sweep=[float(x) for x in args_cli.lr_sweep],
            steps=int(args_cli.steps),
            output=c32_path,
            doc_fragments=doc_fragments,
        )
        if device.type == "cuda":
            torch.cuda.empty_cache()
        one_c64 = run_one_batch_overfit(
            start_payload=payload,
            checkpoint_args=checkpoint_args,
            device=device,
            batch_cpu=batch_cpu,
            target_cache_path=Path(args_cli.target_cache_c64),
            prototype_count=64,
            lr_sweep=[float(x) for x in args_cli.lr_sweep],
            steps=int(args_cli.steps),
            output=c64_path,
            doc_fragments=doc_fragments,
        )
        (docs_dir / "STWM_SEMANTIC_FIELD_DEBUG_V1_ONE_BATCH_OVERFIT.md").write_text(
            "\n".join(doc_fragments).rstrip() + "\n",
            encoding="utf-8",
        )
        if bool(one_c32.get("overfit_success")) or bool(one_c64.get("overfit_success")):
            tiny = run_tiny_overfit(
                start_payload=payload,
                checkpoint_args=checkpoint_args,
                device=device,
                dataset=dataset,
                cache_paths={32: Path(args_cli.target_cache_c32), 64: Path(args_cli.target_cache_c64)},
                best_lrs={32: float(one_c32.get("best_lr", 1e-5) or 1e-5), 64: float(one_c64.get("best_lr", 1e-5) or 1e-5)},
                steps=int(args_cli.tiny_steps),
                batch_size=int(args_cli.overfit_batch_size),
                output=tiny_path,
                doc=docs_dir / "STWM_SEMANTIC_FIELD_DEBUG_V1_TINY_OVERFIT_SUMMARY.md",
            )
        else:
            tiny = _build_tiny_overfit_skipped("one_batch_overfit_failed", tiny_path)

    root_cause, recommendation = _root_cause_and_recommendation(
        alignment=alignment,
        semantic_input=semantic_input,
        gradient=gradient,
        one_c32=one_c32,
        one_c64=one_c64,
        tiny=tiny,
    )
    decision = {
        "audit_name": "stwm_semantic_field_debug_v1_decision",
        "target_alignment_ok": bool(alignment.get("alignment_ok")),
        "semantic_input_valid": bool(semantic_input.get("semantic_input_valid")),
        "proto_loss_grad_reaches_tusb_semantic": bool(gradient.get("proto_loss_grad_reaches_tusb_semantic")),
        "stage1_grad_detected": bool(gradient.get("stage1_grad_detected")),
        "dynamic_grad_detected": bool(gradient.get("dynamic_grad_detected")),
        "one_batch_overfit_success": bool(one_c32.get("overfit_success")) or bool(one_c64.get("overfit_success")),
        "one_batch_overfit_c32_success": bool(one_c32.get("overfit_success")),
        "one_batch_overfit_c64_success": bool(one_c64.get("overfit_success")),
        "tiny_overfit_success": bool(tiny.get("tiny_overfit_success")),
        "root_cause": root_cause,
        "recommended_next_step_choice": recommendation,
        "target_alignment_audit": str(alignment_path),
        "semantic_input_audit": str(semantic_input_path),
        "gradient_audit": str(gradient_path),
        "one_batch_overfit_c32": str(c32_path),
        "one_batch_overfit_c64": str(c64_path),
        "tiny_overfit_summary": str(tiny_path),
    }
    write_json(decision_path, decision)
    write_doc(
        docs_dir / "STWM_SEMANTIC_FIELD_DEBUG_V1_DECISION.md",
        "STWM Semantic Field Debug V1 Decision",
        decision,
        bullets=[
            "One-batch overfit is the gate before any medium semantic-field training.",
            "If this gate fails with clean alignment/input/gradients, the semantic target space or optimization recipe must be redesigned before scaling.",
        ],
    )

    guardrail = {
        "guardrail": "stwm_world_model_no_drift_guardrail_v23",
        "allowed": [
            "overfit sanity for semantic trace field",
            "structured semantic prototype output",
            "Stage1 frozen",
            "dynamic trace path frozen",
        ],
        "forbidden": [
            "candidate scorer",
            "SAM2/CoTracker plugin",
            "future candidate leakage",
            "paper claim before overfit sanity",
            "continuing medium training without overfit success",
            "blaming target before checking cache alignment and gradients",
        ],
    }
    write_json(guardrail_path, guardrail)
    write_doc(
        docs_dir / "STWM_WORLD_MODEL_NO_DRIFT_GUARDRAIL_V23.md",
        "STWM World Model No-Drift Guardrail V23",
        guardrail,
        bullets=[
            "Allowed: overfit sanity, structured semantic prototype output, Stage1 frozen, dynamic trace path frozen.",
            "Forbidden: candidate scorer, plugin framing, future candidate leakage, premature paper claims, and medium training without overfit success.",
        ],
    )


if __name__ == "__main__":
    main()
