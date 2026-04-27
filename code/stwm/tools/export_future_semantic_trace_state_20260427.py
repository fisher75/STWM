#!/usr/bin/env python3
from __future__ import annotations

from argparse import ArgumentParser
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any
import hashlib
import json
import os
import sys

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


RAW_EXPORT_SCHEMA_VERSION = "future_semantic_trace_state_raw_export_v1"


def _bootstrap_repo_imports(repo_root: Path) -> None:
    code_dir = repo_root / "code"
    if str(code_dir) not in sys.path:
        sys.path.insert(0, str(code_dir))


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve_repo_root(value: str | None) -> Path:
    if value:
        return Path(value).expanduser().resolve()
    env_root = os.environ.get("STWM_ROOT")
    if env_root:
        return Path(env_root).expanduser().resolve()
    return Path.cwd().resolve()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def write_doc(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# STWM Future Semantic State Raw Export Repair V1 20260427",
        "",
        f"- raw_export_schema_version: `{payload.get('raw_export_schema_version')}`",
        f"- checkpoint_path: `{payload.get('checkpoint_path')}`",
        f"- checkpoint_loaded: `{payload.get('checkpoint_loaded')}`",
        f"- enable_future_semantic_state_head: `{payload.get('enable_future_semantic_state_head')}`",
        f"- free_rollout_used: `{payload.get('free_rollout_used')}`",
        f"- old_association_report_used: `{payload.get('old_association_report_used')}`",
        f"- total_items: `{payload.get('total_items')}`",
        f"- valid_items: `{payload.get('valid_items')}`",
        f"- valid_ratio: `{payload.get('valid_ratio')}`",
        "",
        "The export contains raw-output-derived shape/stat/variance fields for FutureSemanticTraceState. It does not export association top1/MRR/false-confuser metrics.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n")


def stable_seed(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)


def _as_bbox(value: Any) -> list[float] | None:
    if not isinstance(value, list) or len(value) != 4:
        return None
    try:
        return [float(x) for x in value]
    except Exception:
        return None


def _bbox_center(bbox: list[float] | None) -> list[float] | None:
    if bbox is None:
        return None
    x1, y1, x2, y2 = bbox
    return [(x1 + x2) / 2.0, (y1 + y2) / 2.0]


def _normalize_point(point: list[float] | None, scale: float = 1024.0) -> list[float] | None:
    if point is None:
        return None
    return [max(0.0, min(1.0, float(point[0]) / scale)), max(0.0, min(1.0, float(point[1]) / scale))]


def _extract_manifest_items(manifest_path: Path | None, max_items: int) -> list[dict[str, Any]]:
    if manifest_path is None:
        raise RuntimeError("--manifest is required for repair v1 raw export")
    if not manifest_path.exists():
        raise RuntimeError(f"manifest not found: {manifest_path}")
    manifest = load_json(manifest_path)
    raw_items = manifest.get("items") or manifest.get("materialized_items") or []
    if isinstance(raw_items, dict):
        raw_items = list(raw_items.values())
    if not isinstance(raw_items, list) or not raw_items:
        raise RuntimeError(f"manifest contains no item list: {manifest_path}")
    return [x for x in raw_items if isinstance(x, dict)][: int(max_items)]


def _extract_target_future_coord(item: dict[str, Any]) -> list[float] | None:
    gt = str(item.get("gt_candidate_id") or "")
    for cand in item.get("future_candidates") or []:
        if not isinstance(cand, dict):
            continue
        if str(cand.get("candidate_id")) == gt:
            return _normalize_point(_bbox_center(_as_bbox(cand.get("bbox"))))
    return None


def _extract_observed_coord(item: dict[str, Any]) -> list[float]:
    target = item.get("observed_target") if isinstance(item.get("observed_target"), dict) else {}
    point = _normalize_point(_bbox_center(_as_bbox(target.get("bbox"))))
    if point is not None:
        return point
    raw_point = target.get("point_prompt")
    if isinstance(raw_point, list) and len(raw_point) == 2:
        normalized = _normalize_point([float(raw_point[0]), float(raw_point[1])])
        if normalized is not None:
            return normalized
    seed = stable_seed(str(item.get("item_id") or item.get("protocol_item_id") or "item"))
    return [((seed % 997) / 997.0), (((seed // 997) % 991) / 991.0)]


def _tensor_stats(tensor: torch.Tensor) -> dict[str, float | list[int]]:
    t = tensor.detach().float().cpu()
    finite = torch.isfinite(t)
    finite_t = t[finite]
    if finite_t.numel() == 0:
        return {
            "shape": list(t.shape),
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
            "nan_inf_ratio": 1.0,
        }
    return {
        "shape": list(t.shape),
        "mean": float(finite_t.mean().item()),
        "std": float(finite_t.std(unbiased=False).item()),
        "min": float(finite_t.min().item()),
        "max": float(finite_t.max().item()),
        "nan_inf_ratio": float(1.0 - (finite_t.numel() / max(t.numel(), 1))),
    }


def _scalar_or_none(value: torch.Tensor) -> float | None:
    if value.numel() == 0:
        return None
    if not torch.isfinite(value).all():
        return None
    return float(value.detach().cpu().item())


def _load_head_from_checkpoint(repo_root: Path, checkpoint: Path, device: torch.device):
    _bootstrap_repo_imports(repo_root)
    from stwm.tracewm_v2_stage2.models.semantic_trace_world_head import SemanticTraceStateHead, SemanticTraceStateHeadConfig

    if not checkpoint.exists():
        raise RuntimeError(f"checkpoint not found: {checkpoint}")
    payload = torch.load(checkpoint, map_location="cpu")
    if not isinstance(payload, dict):
        raise RuntimeError(f"checkpoint payload is not a dict: {checkpoint}")
    state_dict = payload.get("future_semantic_state_head_state_dict")
    if not isinstance(state_dict, dict) or not state_dict:
        raise RuntimeError(f"checkpoint does not contain future_semantic_state_head_state_dict: {checkpoint}")

    def find_weight(suffix: str) -> torch.Tensor:
        for key, value in state_dict.items():
            if str(key).endswith(suffix) and isinstance(value, torch.Tensor) and value.ndim == 2:
                return value
        raise KeyError(suffix)

    sem_w = find_weight("semantic_embedding_head.2.weight")
    id_w = find_weight("identity_embedding_head.2.weight")
    hidden_dim = int(sem_w.shape[1])
    semantic_dim = int(sem_w.shape[0])
    identity_dim = int(id_w.shape[0])
    hypothesis_count = 1
    enable_multi = any(str(k).startswith("multi_hypothesis_head.") for k in state_dict.keys())
    if enable_multi:
        for key, value in state_dict.items():
            if str(key).endswith("multi_hypothesis_head.logit_head.1.weight"):
                hypothesis_count = int(value.shape[0])
                break
    cfg = SemanticTraceStateHeadConfig(
        hidden_dim=hidden_dim,
        semantic_embedding_dim=semantic_dim,
        identity_embedding_dim=identity_dim,
        hypothesis_count=hypothesis_count,
        enable_multi_hypothesis_head=enable_multi,
    )
    head = SemanticTraceStateHead(cfg).to(device)
    missing, unexpected = head.load_state_dict(state_dict, strict=False)
    head.eval()
    return head, payload, state_dict, cfg, list(missing), list(unexpected)


def _args_from_checkpoint_payload(payload: dict[str, Any], repo_root: Path, max_items: int) -> Any:
    raw_args = dict(payload.get("args") or {})
    defaults = {
        "stage2_contract_path": str(repo_root / "reports" / "stage2_bootstrap_data_contract_20260408.json"),
        "recommended_runtime_json": str(repo_root / "reports" / "stage1_v2_recommended_runtime_20260408.json"),
        "stage1_backbone_checkpoint": str(repo_root / "outputs" / "checkpoints" / "stage1_v2_longtrain_220m_mainline_20260408" / "best.pt"),
        "stage1_model_preset": "prototype_220m",
        "dataset_names": ["vspw", "vipseg"],
        "train_split": "train",
        "val_split": "val",
        "obs_len": 8,
        "fut_len": 8,
        "max_tokens": 64,
        "max_samples_val": int(max_items),
        "semantic_patch_radius": 12,
        "semantic_crop_size": 64,
        "semantic_source_mainline": "crop_visual_encoder",
        "local_temporal_window": 1,
        "predecode_cache_path": str(repo_root / "data" / "processed" / "stage2_tusb_v3_predecode_cache_20260418"),
        "teacher_semantic_cache_path": str(repo_root / "data" / "processed" / "stage2_teacher_semantic_cache_v4_appearance_20260418"),
        "max_entities_per_sample": 8,
        "semantic_hidden_dim": 256,
        "semantic_embed_dim": 256,
        "stage2_structure_mode": "trace_unit_semantic_binding",
        "trace_unit_count": 16,
        "trace_unit_dim": 384,
        "trace_unit_slot_iters": 3,
        "trace_unit_assignment_topk": 2,
        "trace_unit_assignment_temperature": 0.7,
        "trace_unit_use_instance_prior_bias": True,
        "trace_unit_disable_instance_path": False,
        "trace_unit_teacher_prior_dim": 512,
        "trace_unit_dyn_update": "gru",
        "trace_unit_sem_update": "gated_ema",
        "trace_unit_sem_alpha_min": 0.02,
        "trace_unit_sem_alpha_max": 0.12,
        "trace_unit_handshake_dim": 128,
        "trace_unit_handshake_layers": 1,
        "trace_unit_handshake_writeback": "dyn_only",
        "trace_unit_broadcast_residual_weight": 0.35,
        "trace_unit_broadcast_stopgrad_semantic": False,
        "future_semantic_embedding_dim": 256,
        "future_hypothesis_count": 1,
        "enable_future_extent_head": False,
        "enable_future_multihypothesis_head": False,
    }
    for key, value in defaults.items():
        raw_args.setdefault(key, value)
    raw_args["max_samples_val"] = int(max_items)
    return SimpleNamespace(**raw_args)


def _build_full_model_from_checkpoint(repo_root: Path, checkpoint: Path, device: torch.device, max_items: int) -> dict[str, Any]:
    _bootstrap_repo_imports(repo_root)
    from stwm.tracewm_v2_stage2.trainers import train_tracewm_stage2_smalltrain as trainer

    payload = torch.load(checkpoint, map_location=device)
    if not isinstance(payload, dict):
        raise RuntimeError(f"checkpoint payload is not a dict: {checkpoint}")
    if not isinstance(payload.get("future_semantic_state_head_state_dict"), dict):
        raise RuntimeError("checkpoint lacks future_semantic_state_head_state_dict; cannot run full-model semantic state export")
    args = _args_from_checkpoint_payload(payload, repo_root, max_items)
    stage1_model, stage1_meta = trainer._load_frozen_stage1_backbone(args=args, device=device)
    fusion_hidden_dim = int(stage1_model.config.d_model)
    semantic_encoder = trainer.SemanticEncoder(
        trainer.SemanticEncoderConfig(
            input_dim=10,
            hidden_dim=int(args.semantic_hidden_dim),
            output_dim=int(args.semantic_embed_dim),
        )
    ).to(device)
    semantic_fusion = trainer.SemanticFusion(
        trainer.SemanticFusionConfig(
            hidden_dim=fusion_hidden_dim,
            semantic_dim=int(args.semantic_embed_dim),
            dropout=0.1,
        )
    ).to(device)
    readout_head = torch.nn.Linear(fusion_hidden_dim, 2).to(device)
    future_semantic_state_head = trainer.SemanticTraceStateHead(
        trainer.SemanticTraceStateHeadConfig(
            hidden_dim=fusion_hidden_dim,
            semantic_embedding_dim=int(args.future_semantic_embedding_dim),
            identity_embedding_dim=int(args.future_semantic_embedding_dim),
            hypothesis_count=int(args.future_hypothesis_count),
            enable_extent_head=bool(args.enable_future_extent_head),
            enable_multi_hypothesis_head=bool(args.enable_future_multihypothesis_head)
            or int(args.future_hypothesis_count) > 1,
        )
    ).to(device)

    structure_mode = str(args.stage2_structure_mode).strip().lower()
    trace_unit_tokenizer = trace_unit_factorized_state = trace_unit_handshake = trace_unit_broadcast = None
    if structure_mode == "trace_unit_semantic_binding":
        trace_unit_tokenizer = trainer.TraceUnitTokenizer(
            trainer.TraceUnitTokenizerConfig(
                hidden_dim=fusion_hidden_dim,
                semantic_dim=int(args.semantic_embed_dim),
                state_dim=trainer.STATE_DIM,
                teacher_prior_dim=int(args.trace_unit_teacher_prior_dim),
                unit_dim=int(args.trace_unit_dim),
                unit_count=int(args.trace_unit_count),
                slot_iters=int(args.trace_unit_slot_iters),
                assignment_topk=int(args.trace_unit_assignment_topk),
                assignment_temperature=float(args.trace_unit_assignment_temperature),
                use_instance_prior_bias=bool(args.trace_unit_use_instance_prior_bias),
            )
        ).to(device)
        trace_unit_factorized_state = trainer.TraceUnitFactorizedState(
            trainer.TraceUnitFactorizedStateConfig(
                unit_dim=int(args.trace_unit_dim),
                dyn_update=str(args.trace_unit_dyn_update),
                sem_update=str(args.trace_unit_sem_update),
                sem_alpha_min=float(args.trace_unit_sem_alpha_min),
                sem_alpha_max=float(args.trace_unit_sem_alpha_max),
            )
        ).to(device)
        trace_unit_handshake = trainer.TraceUnitHandshake(
            trainer.TraceUnitHandshakeConfig(
                unit_dim=int(args.trace_unit_dim),
                handshake_dim=int(args.trace_unit_handshake_dim),
                layers=int(args.trace_unit_handshake_layers),
                writeback=str(args.trace_unit_handshake_writeback),
            )
        ).to(device)
        trace_unit_broadcast = trainer.TraceUnitBroadcast(
            trainer.TraceUnitBroadcastConfig(
                hidden_dim=fusion_hidden_dim,
                unit_dim=int(args.trace_unit_dim),
                residual_weight=float(args.trace_unit_broadcast_residual_weight),
                stopgrad_semantic=bool(args.trace_unit_broadcast_stopgrad_semantic),
            )
        ).to(device)

    load_report: dict[str, Any] = {}
    missing, unexpected = semantic_encoder.load_state_dict(payload.get("semantic_encoder_state_dict", {}), strict=False)
    load_report["semantic_encoder"] = {"loaded_keys": len(payload.get("semantic_encoder_state_dict", {}) or {}), "missing": len(missing), "unexpected": len(unexpected)}
    missing, unexpected = semantic_fusion.load_state_dict(payload.get("semantic_fusion_state_dict", {}), strict=False)
    load_report["semantic_fusion"] = {"loaded_keys": len(payload.get("semantic_fusion_state_dict", {}) or {}), "missing": len(missing), "unexpected": len(unexpected)}
    missing, unexpected = readout_head.load_state_dict(payload.get("readout_head_state_dict", {}), strict=False)
    load_report["readout_head"] = {"loaded_keys": len(payload.get("readout_head_state_dict", {}) or {}), "missing": len(missing), "unexpected": len(unexpected)}
    missing, unexpected = future_semantic_state_head.load_state_dict(payload.get("future_semantic_state_head_state_dict", {}), strict=False)
    load_report["future_semantic_state_head"] = {
        "loaded_keys": len(payload.get("future_semantic_state_head_state_dict", {}) or {}),
        "missing": len(missing),
        "unexpected": len(unexpected),
    }
    optional = [
        ("trace_unit_tokenizer", trace_unit_tokenizer, "trace_unit_tokenizer_state_dict"),
        ("trace_unit_factorized_state", trace_unit_factorized_state, "trace_unit_factorized_state_state_dict"),
        ("trace_unit_handshake", trace_unit_handshake, "trace_unit_handshake_state_dict"),
        ("trace_unit_broadcast", trace_unit_broadcast, "trace_unit_broadcast_state_dict"),
    ]
    for name, module, key in optional:
        if module is not None:
            missing, unexpected = module.load_state_dict(payload.get(key, {}), strict=False)
            load_report[name] = {"loaded_keys": len(payload.get(key, {}) or {}), "missing": len(missing), "unexpected": len(unexpected)}

    for module in [
        stage1_model,
        semantic_encoder,
        semantic_fusion,
        readout_head,
        future_semantic_state_head,
        trace_unit_tokenizer,
        trace_unit_factorized_state,
        trace_unit_handshake,
        trace_unit_broadcast,
    ]:
        if module is not None:
            module.eval()

    cfg = trainer.Stage2SemanticDatasetConfig(
        dataset_names=[str(x) for x in args.dataset_names],
        split=str(args.val_split),
        contract_path=str(args.stage2_contract_path),
        obs_len=int(args.obs_len),
        fut_len=int(args.fut_len),
        max_tokens=int(args.max_tokens),
        max_samples_per_dataset=int(max_items),
        semantic_patch_radius=int(args.semantic_patch_radius),
        semantic_crop_size=int(args.semantic_crop_size),
        semantic_source_mainline=str(args.semantic_source_mainline),
        semantic_temporal_window=int(args.local_temporal_window),
        predecode_cache_path=str(args.predecode_cache_path),
        teacher_semantic_cache_path=str(args.teacher_semantic_cache_path),
        max_entities_per_sample=int(args.max_entities_per_sample),
    )
    dataset = trainer.Stage2SemanticDataset(cfg)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        collate_fn=trainer.stage2_semantic_collate_fn,
    )
    return {
        "trainer": trainer,
        "args": args,
        "payload": payload,
        "stage1_model": stage1_model,
        "semantic_encoder": semantic_encoder,
        "semantic_fusion": semantic_fusion,
        "readout_head": readout_head,
        "future_semantic_state_head": future_semantic_state_head,
        "trace_unit_tokenizer": trace_unit_tokenizer,
        "trace_unit_factorized_state": trace_unit_factorized_state,
        "trace_unit_handshake": trace_unit_handshake,
        "trace_unit_broadcast": trace_unit_broadcast,
        "loader": loader,
        "dataset_summary": dict(dataset.dataset_summary),
        "stage1_meta": stage1_meta,
        "load_report": load_report,
        "structure_mode": structure_mode,
    }


def _item_subset_tags(raw: dict[str, Any]) -> Any:
    tags = raw.get("subset_tags", {})
    return tags if isinstance(tags, (dict, list)) else {}


def _item_forward(
    *,
    head: torch.nn.Module,
    cfg: Any,
    raw: dict[str, Any],
    horizon: int,
    slots: int,
    device: torch.device,
) -> dict[str, Any]:
    item_id = str(raw.get("item_id") or raw.get("protocol_item_id") or "unknown")
    generator = torch.Generator(device="cpu").manual_seed(stable_seed(item_id))
    observed = torch.tensor(_extract_observed_coord(raw), dtype=torch.float32).view(1, 1, 1, 2)
    base_coord = observed.repeat(1, int(horizon), int(slots), 1)
    base_coord = (base_coord + 0.01 * torch.randn(base_coord.shape, generator=generator)).clamp(0.0, 1.0).to(device)
    hidden = torch.randn((1, int(horizon), int(slots), int(cfg.hidden_dim)), generator=generator).to(device)
    with torch.no_grad():
        state = head(hidden, future_trace_coord=base_coord)
        validation = state.validate(strict=False)
        visibility_prob = torch.sigmoid(state.future_visibility_logit)
        uncertainty = F.softplus(state.future_uncertainty)
        sem = state.future_semantic_embedding
        ident = state.future_identity_embedding
        sem_norm = sem.norm(dim=-1)
        ident_norm = ident.norm(dim=-1)
        target_coord = _extract_target_future_coord(raw)
        coord_error = None
        if target_coord is not None:
            pred_last = state.future_trace_coord[0, -1, 0].detach().cpu()
            target = torch.tensor(target_coord, dtype=torch.float32)
            coord_error = float(torch.sqrt(((pred_last - target) ** 2).sum()).item())
        item = {
            "item_id": item_id,
            "protocol_item_id": raw.get("protocol_item_id", item_id),
            "subset_tags": _item_subset_tags(raw),
            "valid_output": bool(validation["valid"]),
            "failure_reason": "; ".join(validation.get("errors", [])) if not validation["valid"] else None,
            "future_semantic_trace_state_valid": bool(validation["valid"]),
            "future_trace_coord_shape": list(state.future_trace_coord.shape),
            "future_trace_coord_mean": _tensor_stats(state.future_trace_coord)["mean"],
            "future_trace_coord_std": _tensor_stats(state.future_trace_coord)["std"],
            "future_trace_coord_min": _tensor_stats(state.future_trace_coord)["min"],
            "future_trace_coord_max": _tensor_stats(state.future_trace_coord)["max"],
            "future_visibility_prob_shape": list(visibility_prob.shape),
            "future_visibility_prob_mean": _tensor_stats(visibility_prob)["mean"],
            "future_visibility_prob_std": _tensor_stats(visibility_prob)["std"],
            "future_visibility_prob_min": _tensor_stats(visibility_prob)["min"],
            "future_visibility_prob_max": _tensor_stats(visibility_prob)["max"],
            "future_semantic_embedding_shape": list(sem.shape),
            "future_semantic_embedding_norm_mean": _tensor_stats(sem_norm)["mean"],
            "future_semantic_embedding_norm_std": _tensor_stats(sem_norm)["std"],
            "future_semantic_embedding_var_unit": _scalar_or_none(sem.var(dim=2, unbiased=False).mean()),
            "future_semantic_embedding_var_horizon": _scalar_or_none(sem.var(dim=1, unbiased=False).mean()),
            "future_identity_embedding_shape": list(ident.shape),
            "future_identity_embedding_norm_mean": _tensor_stats(ident_norm)["mean"],
            "future_identity_embedding_norm_std": _tensor_stats(ident_norm)["std"],
            "future_identity_embedding_var_unit": _scalar_or_none(ident.var(dim=2, unbiased=False).mean()),
            "future_uncertainty_shape": list(uncertainty.shape),
            "future_uncertainty_mean": _tensor_stats(uncertainty)["mean"],
            "future_uncertainty_std": _tensor_stats(uncertainty)["std"],
            "future_uncertainty_min": _tensor_stats(uncertainty)["min"],
            "future_uncertainty_max": _tensor_stats(uncertainty)["max"],
            "future_trace_coord_error": coord_error,
            "target_visibility": 1 if raw.get("gt_candidate_id") is not None else None,
            "future_hypothesis_logits_shape": list(state.future_hypothesis_logits.shape) if state.future_hypothesis_logits is not None else None,
            "future_hypothesis_logits_mean": _tensor_stats(state.future_hypothesis_logits)["mean"] if state.future_hypothesis_logits is not None else None,
            "future_hypothesis_trace_coord_shape": list(state.future_hypothesis_trace_coord.shape) if state.future_hypothesis_trace_coord is not None else None,
        }
    return item


def _item_from_state(
    *,
    item_id: str,
    protocol_item_id: str,
    subset_tags: Any,
    state: Any,
    target_coord: torch.Tensor | None,
    valid_mask: torch.Tensor | None,
) -> dict[str, Any]:
    validation = state.validate(strict=False)
    visibility_prob = torch.sigmoid(state.future_visibility_logit)
    uncertainty = F.softplus(state.future_uncertainty)
    sem = state.future_semantic_embedding
    ident = state.future_identity_embedding
    sem_norm = sem.norm(dim=-1)
    ident_norm = ident.norm(dim=-1)
    coord_error = None
    if target_coord is not None:
        pred = state.future_trace_coord.detach()
        target = target_coord.detach().to(device=pred.device, dtype=pred.dtype)
        sq = ((pred - target) ** 2).sum(dim=-1).sqrt()
        if valid_mask is not None:
            mask = valid_mask.detach().to(device=pred.device, dtype=torch.bool)
            if mask.any():
                coord_error = float(sq[mask].mean().detach().cpu().item())
        else:
            coord_error = float(sq.mean().detach().cpu().item())
    return {
        "item_id": item_id,
        "protocol_item_id": protocol_item_id,
        "subset_tags": subset_tags,
        "valid_output": bool(validation["valid"]),
        "failure_reason": "; ".join(validation.get("errors", [])) if not validation["valid"] else None,
        "future_semantic_trace_state_valid": bool(validation["valid"]),
        "future_trace_coord_shape": list(state.future_trace_coord.shape),
        "future_trace_coord_mean": _tensor_stats(state.future_trace_coord)["mean"],
        "future_trace_coord_std": _tensor_stats(state.future_trace_coord)["std"],
        "future_trace_coord_min": _tensor_stats(state.future_trace_coord)["min"],
        "future_trace_coord_max": _tensor_stats(state.future_trace_coord)["max"],
        "future_visibility_prob_shape": list(visibility_prob.shape),
        "future_visibility_prob_mean": _tensor_stats(visibility_prob)["mean"],
        "future_visibility_prob_std": _tensor_stats(visibility_prob)["std"],
        "future_visibility_prob_min": _tensor_stats(visibility_prob)["min"],
        "future_visibility_prob_max": _tensor_stats(visibility_prob)["max"],
        "future_semantic_embedding_shape": list(sem.shape),
        "future_semantic_embedding_norm_mean": _tensor_stats(sem_norm)["mean"],
        "future_semantic_embedding_norm_std": _tensor_stats(sem_norm)["std"],
        "future_semantic_embedding_var_unit": _scalar_or_none(sem.var(dim=2, unbiased=False).mean()),
        "future_semantic_embedding_var_horizon": _scalar_or_none(sem.var(dim=1, unbiased=False).mean()),
        "future_identity_embedding_shape": list(ident.shape),
        "future_identity_embedding_norm_mean": _tensor_stats(ident_norm)["mean"],
        "future_identity_embedding_norm_std": _tensor_stats(ident_norm)["std"],
        "future_identity_embedding_var_unit": _scalar_or_none(ident.var(dim=2, unbiased=False).mean()),
        "future_uncertainty_shape": list(uncertainty.shape),
        "future_uncertainty_mean": _tensor_stats(uncertainty)["mean"],
        "future_uncertainty_std": _tensor_stats(uncertainty)["std"],
        "future_uncertainty_min": _tensor_stats(uncertainty)["min"],
        "future_uncertainty_max": _tensor_stats(uncertainty)["max"],
        "future_trace_coord_error": coord_error,
        "target_visibility": 1 if valid_mask is not None and bool(valid_mask.any().item()) else None,
        "future_hypothesis_logits_shape": list(state.future_hypothesis_logits.shape) if state.future_hypothesis_logits is not None else None,
        "future_hypothesis_logits_mean": _tensor_stats(state.future_hypothesis_logits)["mean"] if state.future_hypothesis_logits is not None else None,
        "future_hypothesis_trace_coord_shape": list(state.future_hypothesis_trace_coord.shape) if state.future_hypothesis_trace_coord is not None else None,
    }


def export(
    *,
    repo_root: Path,
    checkpoint: Path,
    manifest: Path,
    output: Path,
    max_items: int,
    device_name: str,
    mode: str,
) -> dict[str, Any]:
    device = torch.device(device_name if device_name != "cuda" or torch.cuda.is_available() else "cpu")
    exported_items: list[dict[str, Any]] = []
    full_report: dict[str, Any] = {}
    checkpoint_payload: dict[str, Any] = {}
    state_dict: dict[str, Any] = {}
    random_hidden_used = False
    full_model_forward_executed = False
    full_free_rollout_executed = False
    semantic_state_from_model_hidden = False
    if mode == "head_only_surrogate":
        head, checkpoint_payload, state_dict, cfg, missing, unexpected = _load_head_from_checkpoint(repo_root, checkpoint, device)
        raw_items = _extract_manifest_items(manifest, max_items)
        random_hidden_used = True
        for raw in raw_items:
            try:
                exported_items.append(
                    _item_forward(
                        head=head,
                        cfg=cfg,
                        raw=raw,
                        horizon=8,
                        slots=8,
                        device=device,
                    )
                )
            except Exception as exc:
                item_id = str(raw.get("item_id") or raw.get("protocol_item_id") or len(exported_items))
                exported_items.append(
                    {
                        "item_id": item_id,
                        "protocol_item_id": raw.get("protocol_item_id", item_id),
                        "subset_tags": _item_subset_tags(raw),
                        "valid_output": False,
                        "future_semantic_trace_state_valid": False,
                        "failure_reason": repr(exc),
                    }
                )
    elif mode in {"full_model_teacher_forced", "full_model_free_rollout"}:
        full = _build_full_model_from_checkpoint(repo_root, checkpoint, device, max_items)
        checkpoint_payload = full["payload"]
        state_dict = checkpoint_payload["future_semantic_state_head_state_dict"]
        full_model_forward_executed = True
        full_free_rollout_executed = mode == "full_model_free_rollout"
        semantic_state_from_model_hidden = True
        trainer = full["trainer"]
        args = full["args"]
        count = 0
        with torch.no_grad():
            for raw_batch in full["loader"]:
                if count >= int(max_items):
                    break
                batch = trainer._to_device(raw_batch, device=device, non_blocking=False)
                if mode == "full_model_teacher_forced":
                    out = trainer._teacher_forced_predict(
                        stage1_model=full["stage1_model"],
                        semantic_encoder=full["semantic_encoder"],
                        semantic_fusion=full["semantic_fusion"],
                        readout_head=full["readout_head"],
                        future_semantic_state_head=full["future_semantic_state_head"],
                        structure_mode=str(full["structure_mode"]),
                        trace_unit_tokenizer=full["trace_unit_tokenizer"],
                        trace_unit_factorized_state=full["trace_unit_factorized_state"],
                        trace_unit_handshake=full["trace_unit_handshake"],
                        trace_unit_broadcast=full["trace_unit_broadcast"],
                        trace_unit_disable_instance_path=bool(args.trace_unit_disable_instance_path),
                        batch=batch,
                        obs_len=int(args.obs_len),
                        semantic_source_mainline=str(args.semantic_source_mainline),
                    )
                else:
                    out = trainer._free_rollout_predict(
                        stage1_model=full["stage1_model"],
                        semantic_encoder=full["semantic_encoder"],
                        semantic_fusion=full["semantic_fusion"],
                        readout_head=full["readout_head"],
                        future_semantic_state_head=full["future_semantic_state_head"],
                        structure_mode=str(full["structure_mode"]),
                        trace_unit_tokenizer=full["trace_unit_tokenizer"],
                        trace_unit_factorized_state=full["trace_unit_factorized_state"],
                        trace_unit_handshake=full["trace_unit_handshake"],
                        trace_unit_broadcast=full["trace_unit_broadcast"],
                        trace_unit_disable_instance_path=bool(args.trace_unit_disable_instance_path),
                        batch=batch,
                        obs_len=int(args.obs_len),
                        fut_len=int(args.fut_len),
                        semantic_source_mainline=str(args.semantic_source_mainline),
                    )
                state = out.get("future_semantic_trace_state")
                if state is None:
                    raise RuntimeError(f"{mode} did not return future_semantic_trace_state")
                meta = (raw_batch.get("meta") or [{}])[0]
                item_id = f"{meta.get('dataset', 'stage2')}::{meta.get('clip_id', count)}::{count}"
                exported_items.append(
                    _item_from_state(
                        item_id=str(item_id),
                        protocol_item_id=str(item_id),
                        subset_tags={},
                        state=state,
                        target_coord=out.get("target_coord"),
                        valid_mask=out.get("valid_mask"),
                    )
                )
                count += 1
        full_report = {
            "checkpoint_path": str(checkpoint),
            "model_weights_loaded_count": {
                name: data.get("loaded_keys")
                for name, data in (full.get("load_report") or {}).items()
                if isinstance(data, dict)
            },
            "future_semantic_state_head_weights_loaded_count": full.get("load_report", {}).get("future_semantic_state_head", {}).get("loaded_keys"),
            "batch_source": "Stage2SemanticDataset validation split from checkpoint args",
            "manifest_path": str(manifest),
            "item_count": len(exported_items),
            "prediction_path": mode,
            "dataset_summary": full.get("dataset_summary"),
            "load_report": full.get("load_report"),
        }
    else:
        raise ValueError(f"unknown export mode: {mode}")

    valid_items = sum(1 for item in exported_items if bool(item.get("valid_output")))
    payload = {
        "generated_at_utc": now_iso(),
        "raw_export_schema_version": RAW_EXPORT_SCHEMA_VERSION,
        "export_mode": str(mode),
        "repo_root": str(repo_root),
        "checkpoint_path": str(checkpoint),
        "checkpoint_exists": checkpoint.exists(),
        "checkpoint_loaded": True,
        "consumed_checkpoint": str(checkpoint),
        "checkpoint_global_step": checkpoint_payload.get("global_step"),
        "future_semantic_state_head_keys_found": sorted(str(k) for k in state_dict.keys()),
        "future_semantic_state_head_key_count": len(state_dict),
        "enable_future_semantic_state_head": True,
        "state_dict_missing_keys": [str(x) for x in locals().get("missing", [])],
        "state_dict_unexpected_keys": [str(x) for x in locals().get("unexpected", [])],
        "manifest": str(manifest),
        "device": str(device),
        "random_hidden_used": bool(random_hidden_used),
        "observed_bbox_surrogate_coord_used": bool(mode == "head_only_surrogate"),
        "full_model_forward_executed": bool(full_model_forward_executed),
        "full_stage1_stage2_forward_executed": bool(full_model_forward_executed),
        "full_free_rollout_executed": bool(full_free_rollout_executed),
        "semantic_state_from_model_hidden": bool(semantic_state_from_model_hidden),
        "free_rollout_used": bool(full_free_rollout_executed),
        "world_model_output_claimable": bool(mode != "head_only_surrogate" and full_model_forward_executed and semantic_state_from_model_hidden),
        "old_association_report_used": False,
        "top1_mrr_false_confuser_exported": False,
        "total_items": len(exported_items),
        "valid_items": valid_items,
        "valid_ratio": valid_items / max(len(exported_items), 1),
        "full_model_loader_report": full_report,
        "items": exported_items,
    }
    write_json(output, payload)
    write_doc(output.with_suffix(".md"), payload)
    return payload


def parse_args() -> Any:
    p = ArgumentParser(description="Export raw-output-derived FutureSemanticTraceState repair-v1 diagnostics.")
    p.add_argument(
        "--mode",
        default="head_only_surrogate",
        choices=["head_only_surrogate", "full_model_teacher_forced", "full_model_free_rollout"],
    )
    p.add_argument("--repo-root", default=None)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--max-items", "--item-limit", dest="max_items", type=int, default=32)
    p.add_argument("--device", default="cpu")
    p.add_argument("--use-free-rollout", action="store_true", help="Deprecated alias for --mode full_model_free_rollout.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = resolve_repo_root(args.repo_root)
    mode = "full_model_free_rollout" if bool(args.use_free_rollout) and str(args.mode) == "head_only_surrogate" else str(args.mode)
    export(
        repo_root=repo_root,
        checkpoint=Path(args.checkpoint),
        manifest=Path(args.manifest),
        output=Path(args.output),
        max_items=int(args.max_items),
        device_name=str(args.device),
        mode=mode,
    )


if __name__ == "__main__":
    main()
