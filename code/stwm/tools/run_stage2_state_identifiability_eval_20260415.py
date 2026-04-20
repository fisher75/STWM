#!/usr/bin/env python3
from __future__ import annotations

from argparse import ArgumentParser
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Tuple
import gc
import json
import math
import os

import numpy as np
from PIL import Image
import torch
from pycocotools import mask as mask_utils

from stwm.infra.gpu_lease import acquire_lease, release_lease
from stwm.infra.gpu_selector import select_single_gpu
from stwm.tracewm_v2_stage2.datasets.stage2_semantic_dataset import (
    _box_from_mask_or_center,
    _build_semantic_crops,
    _build_state_from_boxes,
    _semantic_feature,
    stage2_semantic_collate_fn,
)
from stwm.tracewm_v2_stage2.models.semantic_encoder import SemanticEncoder, SemanticEncoderConfig
from stwm.tracewm_v2_stage2.models.semantic_fusion import SemanticFusion, SemanticFusionConfig
from stwm.tracewm_v2_stage2.models.trace_unit_broadcast import TraceUnitBroadcast, TraceUnitBroadcastConfig
from stwm.tracewm_v2_stage2.models.trace_unit_factorized_state import (
    TraceUnitFactorizedState,
    TraceUnitFactorizedStateConfig,
)
from stwm.tracewm_v2_stage2.models.trace_unit_handshake import TraceUnitHandshake, TraceUnitHandshakeConfig
from stwm.tracewm_v2_stage2.models.trace_unit_tokenizer import TraceUnitTokenizer, TraceUnitTokenizerConfig
from stwm.tracewm_v2_stage2.trainers import train_tracewm_stage2_smalltrain as trainer


def _repo_root() -> Path:
    for candidate in [
        Path("/raid/chen034/workspace/stwm"),
        Path("/home/chen034/workspace/stwm"),
    ]:
        if candidate.exists():
            return candidate
    raise RuntimeError("unable to resolve STWM repo root")


ROOT = _repo_root()
OBS_LEN = 8
FUT_LEN = 8
TOTAL_STEPS = OBS_LEN + FUT_LEN


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _apply_process_title_normalization(default_title: str = "python") -> None:
    mode = str(os.environ.get("STWM_PROC_TITLE_MODE", "generic")).strip().lower()
    if mode != "generic":
        return
    title = str(os.environ.get("STWM_PROC_TITLE", default_title)).strip() or default_title
    lowered = title.lower()
    if "stwm" in lowered or "tracewm" in lowered or "/home/" in lowered or "/raid/" in lowered:
        title = default_title
    try:
        import setproctitle  # type: ignore
        setproctitle.setproctitle(title)
    except Exception:
        pass


def read_json(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {}
    payload = json.loads(p.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def write_json(path: str | Path, payload: Dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def write_md(path: str | Path, lines: List[str]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _safe_load(path: str | Path, device: torch.device) -> Dict[str, Any]:
    try:
        payload = torch.load(Path(path), map_location=device, weights_only=False)
    except TypeError:
        payload = torch.load(Path(path), map_location=device)
    if not isinstance(payload, dict):
        raise RuntimeError(f"unsupported checkpoint payload at {path}")
    return payload


def _metric_triplet(metrics: Dict[str, Any]) -> Tuple[float, float, float]:
    return (
        float(metrics.get("free_rollout_endpoint_l2", 1e9)),
        float(metrics.get("free_rollout_coord_mean_l2", 1e9)),
        float(metrics.get("teacher_forced_coord_loss", 1e9)),
    )


def _best_family_row(rows: List[Dict[str, Any]], family_substring: str, worst: bool = False) -> Dict[str, Any]:
    matched = [row for row in rows if family_substring in str(row.get("run_name", ""))]
    if not matched:
        return {}
    key_fn = lambda row: _metric_triplet(((row.get("best_checkpoint_metric") or {}).get("metrics") or {}))
    return max(matched, key=key_fn) if worst else min(matched, key=key_fn)


def _burst_seq_map(annotation_file: Path) -> Dict[Tuple[str, str], Dict[str, Any]]:
    payload = read_json(annotation_file)
    sequences = payload.get("sequences", []) if isinstance(payload.get("sequences", []), list) else []
    return {
        (str(seq.get("dataset", "")), str(seq.get("seq_name", ""))): seq
        for seq in sequences
        if isinstance(seq, dict)
    }


def _vipseg_mask(path: Path) -> np.ndarray:
    arr = np.asarray(Image.open(path))
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr.astype(np.int32)


def _burst_mask(rle_counts: str, height: int, width: int) -> np.ndarray:
    arr = mask_utils.decode({"size": [int(height), int(width)], "counts": rle_counts.encode("utf-8")})
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr.astype(bool)


def _mask_centroid(mask: np.ndarray) -> Tuple[float, float]:
    ys, xs = np.where(mask)
    return float(xs.mean()), float(ys.mean())


def _mask_iou(a: np.ndarray | None, b: np.ndarray | None) -> float:
    if not isinstance(a, np.ndarray) or not isinstance(b, np.ndarray):
        return 0.0
    inter = float(np.logical_and(a, b).sum())
    union = float(np.logical_or(a, b).sum())
    return 0.0 if union <= 0.0 else inter / union


@dataclass
class MethodSpec:
    name: str
    run_name: str
    method_type: str
    checkpoint_path: str


@dataclass
class LoadedMethod:
    name: str
    run_name: str
    method_type: str
    checkpoint_path: str
    semantic_source_mainline: str
    stage1_model: Any
    semantic_encoder: Any | None
    semantic_fusion: Any | None
    readout_head: Any | None
    stage2_structure_mode: str = "calibration_only"
    trace_unit_disable_instance_path: bool = False
    trace_unit_tokenizer: Any | None = None
    trace_unit_factorized_state: Any | None = None
    trace_unit_handshake: Any | None = None
    trace_unit_broadcast: Any | None = None


def _load_stage1_model(checkpoint_path: Path, device: torch.device) -> Tuple[Any, Dict[str, Any]]:
    args = SimpleNamespace(
        stage1_backbone_checkpoint=str(checkpoint_path),
        stage1_model_preset="prototype_220m",
        stage1_partial_unfreeze_mode="none",
        stage1_partial_unfreeze_layer_count=1,
    )
    return trainer._load_frozen_stage1_backbone(args=args, device=device)


def _load_stage2_method(spec: MethodSpec, device: torch.device) -> LoadedMethod:
    checkpoint_path = Path(spec.checkpoint_path)
    payload = _safe_load(checkpoint_path, device=device)
    ckpt_args = payload.get("args", {}) if isinstance(payload.get("args", {}), dict) else {}
    stage1_path = Path(str(ckpt_args.get("stage1_backbone_checkpoint", ROOT / "outputs/checkpoints/stage1_v2_longtrain_220m_mainline_20260408/best.pt")))
    stage1_args = SimpleNamespace(
        stage1_backbone_checkpoint=str(stage1_path),
        stage1_model_preset=str(ckpt_args.get("stage1_model_preset", "prototype_220m")),
        stage1_partial_unfreeze_mode="none",
        stage1_partial_unfreeze_layer_count=1,
    )
    stage1_model, _ = trainer._load_frozen_stage1_backbone(args=stage1_args, device=device)
    hidden_dim = int(stage1_model.config.d_model)
    semantic_hidden_dim = int(ckpt_args.get("semantic_hidden_dim", 256))
    semantic_embed_dim = int(ckpt_args.get("semantic_embed_dim", 256))
    semantic_source_mainline = str(ckpt_args.get("semantic_source_mainline", "crop_visual_encoder"))
    legacy_source = str(ckpt_args.get("legacy_semantic_source", "hand_crafted_stats"))
    structure_mode = str(ckpt_args.get("stage2_structure_mode", "calibration_only")).strip().lower()
    trace_unit_disable_instance_path = bool(ckpt_args.get("trace_unit_disable_instance_path", False))
    semantic_encoder = SemanticEncoder(
        SemanticEncoderConfig(
            input_dim=10,
            hidden_dim=semantic_hidden_dim,
            output_dim=semantic_embed_dim,
            dropout=0.1,
            mainline_source=semantic_source_mainline,
            legacy_source=legacy_source,
        )
    ).to(device)
    semantic_fusion = SemanticFusion(
        SemanticFusionConfig(
            hidden_dim=hidden_dim,
            semantic_dim=semantic_embed_dim,
            dropout=0.1,
        )
    ).to(device)
    readout_head = torch.nn.Linear(hidden_dim, 2).to(device)
    semantic_encoder.load_state_dict(payload.get("semantic_encoder_state_dict", {}), strict=False)
    semantic_fusion.load_state_dict(payload.get("semantic_fusion_state_dict", {}), strict=False)
    readout_head.load_state_dict(payload.get("readout_head_state_dict", {}), strict=False)
    trace_unit_tokenizer = None
    trace_unit_factorized_state = None
    trace_unit_handshake = None
    trace_unit_broadcast = None
    has_tusb = any(
        isinstance(payload.get(key, None), dict)
        for key in [
            "trace_unit_tokenizer_state_dict",
            "trace_unit_factorized_state_state_dict",
            "trace_unit_handshake_state_dict",
            "trace_unit_broadcast_state_dict",
        ]
    )
    if structure_mode == "trace_unit_semantic_binding" or has_tusb:
        trace_unit_tokenizer = TraceUnitTokenizer(
            TraceUnitTokenizerConfig(
                hidden_dim=hidden_dim,
                semantic_dim=semantic_embed_dim,
                state_dim=8,
                teacher_prior_dim=int(ckpt_args.get("trace_unit_teacher_prior_dim", 512)),
                unit_dim=int(ckpt_args.get("trace_unit_dim", 384)),
                unit_count=int(ckpt_args.get("trace_unit_count", 16)),
                slot_iters=int(ckpt_args.get("trace_unit_slot_iters", 3)),
                assignment_topk=int(ckpt_args.get("trace_unit_assignment_topk", 2)),
                assignment_temperature=float(ckpt_args.get("trace_unit_assignment_temperature", 0.70)),
                use_instance_prior_bias=bool(ckpt_args.get("trace_unit_use_instance_prior_bias", False)),
            )
        ).to(device)
        trace_unit_factorized_state = TraceUnitFactorizedState(
            TraceUnitFactorizedStateConfig(
                unit_dim=int(ckpt_args.get("trace_unit_dim", 384)),
                dyn_update=str(ckpt_args.get("trace_unit_dyn_update", "gru")),
                sem_update=str(ckpt_args.get("trace_unit_sem_update", "gated_ema")),
                sem_alpha_min=float(ckpt_args.get("trace_unit_sem_alpha_min", 0.02)),
                sem_alpha_max=float(ckpt_args.get("trace_unit_sem_alpha_max", 0.12)),
            )
        ).to(device)
        trace_unit_handshake = TraceUnitHandshake(
            TraceUnitHandshakeConfig(
                unit_dim=int(ckpt_args.get("trace_unit_dim", 384)),
                handshake_dim=int(ckpt_args.get("trace_unit_handshake_dim", 128)),
                layers=int(ckpt_args.get("trace_unit_handshake_layers", 1)),
                writeback=str(ckpt_args.get("trace_unit_handshake_writeback", "dyn_only")),
            )
        ).to(device)
        trace_unit_broadcast = TraceUnitBroadcast(
            TraceUnitBroadcastConfig(
                hidden_dim=hidden_dim,
                unit_dim=int(ckpt_args.get("trace_unit_dim", 384)),
                residual_weight=float(ckpt_args.get("trace_unit_broadcast_residual_weight", 0.35)),
                stopgrad_semantic=bool(ckpt_args.get("trace_unit_broadcast_stopgrad_semantic", False)),
            )
        ).to(device)
        trace_unit_tokenizer.load_state_dict(payload.get("trace_unit_tokenizer_state_dict", {}), strict=False)
        trace_unit_factorized_state.load_state_dict(payload.get("trace_unit_factorized_state_state_dict", {}), strict=False)
        trace_unit_handshake.load_state_dict(payload.get("trace_unit_handshake_state_dict", {}), strict=False)
        trace_unit_broadcast.load_state_dict(payload.get("trace_unit_broadcast_state_dict", {}), strict=False)
        trace_unit_tokenizer.eval()
        trace_unit_factorized_state.eval()
        trace_unit_handshake.eval()
        trace_unit_broadcast.eval()
    stage1_model.eval()
    semantic_encoder.eval()
    semantic_fusion.eval()
    readout_head.eval()
    return LoadedMethod(
        name=spec.name,
        run_name=spec.run_name,
        method_type="stage2",
        checkpoint_path=str(checkpoint_path),
        semantic_source_mainline=semantic_source_mainline,
        stage1_model=stage1_model,
        semantic_encoder=semantic_encoder,
        semantic_fusion=semantic_fusion,
        readout_head=readout_head,
        stage2_structure_mode=structure_mode,
        trace_unit_disable_instance_path=trace_unit_disable_instance_path,
        trace_unit_tokenizer=trace_unit_tokenizer,
        trace_unit_factorized_state=trace_unit_factorized_state,
        trace_unit_handshake=trace_unit_handshake,
        trace_unit_broadcast=trace_unit_broadcast,
    )


def _load_method(spec: MethodSpec, device: torch.device) -> LoadedMethod:
    if spec.method_type == "stage1":
        stage1_model, _ = _load_stage1_model(Path(spec.checkpoint_path), device=device)
        return LoadedMethod(
            name=spec.name,
            run_name=spec.run_name,
            method_type="stage1",
            checkpoint_path=spec.checkpoint_path,
            semantic_source_mainline="none",
            stage1_model=stage1_model,
            semantic_encoder=None,
            semantic_fusion=None,
            readout_head=None,
        )
    return _load_stage2_method(spec, device=device)


def _release_method(method: LoadedMethod) -> None:
    for attr_name in [
        "trace_unit_broadcast",
        "trace_unit_handshake",
        "trace_unit_factorized_state",
        "trace_unit_tokenizer",
        "readout_head",
        "semantic_fusion",
        "semantic_encoder",
        "stage1_model",
    ]:
        mod = getattr(method, attr_name, None)
        if mod is None:
            continue
        try:
            mod.to("cpu")
        except Exception:
            pass
        setattr(method, attr_name, None)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _load_method_specs(args: Any) -> List[MethodSpec]:
    closure_diag = read_json(args.final_utility_diagnosis)
    calibration_run = str(closure_diag.get("overall_best_run_name", "stage2_calonly_topk1_seed123_longconfirm_v2_20260414"))
    closure_summary = read_json(args.final_utility_summary)
    closure_rows = closure_summary.get("run_rows", []) if isinstance(closure_summary.get("run_rows", []), list) else []
    noalign_row = _best_family_row(closure_rows, "noalign", worst=True)
    dense_row = _best_family_row(closure_rows, "densegate", worst=True)
    nodelay_row = _best_family_row(closure_rows, "nodelay", worst=True)

    methods: List[MethodSpec] = [
        MethodSpec(
            name="stage1_frozen_baseline",
            run_name="stage1_frozen_baseline",
            method_type="stage1",
            checkpoint_path=str(args.stage1_checkpoint),
        )
    ]
    named_ckpts = [
        ("legacysem_best", "stage2_fullscale_core_legacysem_seed456_wave2_20260409"),
        ("cropenc_baseline_best", "stage2_fullscale_core_cropenc_seed456_20260409"),
        ("calibration_only_mainline_best", calibration_run),
        ("noalign_failure", str(noalign_row.get("run_name", "stage2_calonly_noalign_seed42_ablate_20260414"))),
        ("densegate_failure", str(dense_row.get("run_name", "stage2_calonly_densegate_seed789_ablate_v2_20260414"))),
        ("nodelay_failure", str(nodelay_row.get("run_name", "stage2_calonly_nodelay_seed789_ablate_v2_20260414"))),
    ]
    for method_name, run_name in named_ckpts:
        ckpt = ROOT / "outputs/checkpoints" / run_name / "best.pt"
        if ckpt.exists():
            methods.append(
                MethodSpec(
                    name=method_name,
                    run_name=run_name,
                    method_type="stage2",
                    checkpoint_path=str(ckpt),
                )
            )
    return methods


def _select_eval_device(args: Any) -> Tuple[torch.device, Dict[str, Any]]:
    requested = str(args.device).strip().lower()
    if requested == "cpu":
        return torch.device("cpu"), {"mode": "forced_cpu", "selected_gpu_id": -1, "lease_id": ""}
    if requested == "cuda":
        if not torch.cuda.is_available():
            return torch.device("cpu"), {"mode": "forced_cuda_fallback_cpu", "selected_gpu_id": -1, "lease_id": ""}
        return torch.device("cuda"), {"mode": "ambient_cuda", "selected_gpu_id": 0, "lease_id": ""}
    if requested != "auto":
        raise RuntimeError(f"unsupported device mode: {requested}")
    if not torch.cuda.is_available():
        return torch.device("cpu"), {"mode": "auto_cpu_no_cuda", "selected_gpu_id": -1, "lease_id": ""}
    selector = select_single_gpu(
        required_mem_gb=float(args.eval_required_mem_gb),
        safety_margin_gb=float(args.eval_safety_margin_gb),
        sample_count=3,
        interval_sec=0.5,
        lease_path=str(args.lease_path),
    )
    gpu_id = int(selector.get("selected_gpu_id", -1))
    if gpu_id < 0:
        return torch.device("cpu"), {"mode": "auto_cpu_no_fit", "selected_gpu_id": -1, "lease_id": "", "selector_payload": selector}
    lease = acquire_lease(
        gpu_id=gpu_id,
        owner="stage2_state_identifiability_eval_20260415",
        ttl_seconds=6 * 3600,
        lease_path=str(args.lease_path),
        allow_shared=False,
    )
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    return torch.device("cuda"), {
        "mode": "auto_selected_gpu",
        "selected_gpu_id": gpu_id,
        "lease_id": str(lease.get("lease_id", "")),
        "selector_payload": selector,
    }


def _extract_entity_masks(
    item: Dict[str, Any],
    entity_id: str | int | None = None,
) -> Tuple[List[np.ndarray | None], List[Tuple[int, int]], Dict[str, np.ndarray], np.ndarray]:
    dataset = str(item.get("dataset", "")).strip().upper()
    if dataset == "VIPSEG":
        final_masks: Dict[str, np.ndarray] = {}
        target_masks: List[np.ndarray | None] = []
        sizes: List[Tuple[int, int]] = []
        selected_mask_paths = [Path(x) for x in item.get("selected_mask_paths", [])]
        target_id = int(item.get("target_id", -1) if entity_id is None else entity_id)
        future_step = int(item.get("future_step", FUT_LEN + OBS_LEN - 1))
        for step_i, mask_path in enumerate(selected_mask_paths):
            arr = _vipseg_mask(mask_path)
            sizes.append((int(arr.shape[1]), int(arr.shape[0])))
            mask = arr == target_id
            target_masks.append(mask if np.any(mask) else None)
            if step_i == future_step:
                for cand in [int(x) for x in np.unique(arr).tolist() if int(x) >= 125]:
                    cmask = arr == int(cand)
                    if np.any(cmask):
                        final_masks[str(cand)] = cmask
        target_future = target_masks[future_step]
        if target_future is None:
            raise RuntimeError(f"future target mask missing for {item.get('protocol_item_id')}")
        return target_masks, sizes, final_masks, target_future

    annotation_file = Path(str(item.get("burst_annotation_file", "")))
    seq_map = _burst_seq_map(annotation_file)
    seq = seq_map[(str(item.get("burst_dataset_name", "")), str(item.get("burst_seq_name", "")))]
    segs = seq.get("segmentations", []) if isinstance(seq.get("segmentations", []), list) else []
    height = int((item.get("image_size") or {}).get("height", seq.get("height", 0)))
    width = int((item.get("image_size") or {}).get("width", seq.get("width", 0)))
    target_id = str(item.get("target_id", "") if entity_id is None else entity_id)
    selected_indices = [int(x) for x in item.get("selected_frame_indices", [])]
    future_step = int(item.get("future_step", FUT_LEN + OBS_LEN - 1))
    target_masks: List[np.ndarray | None] = []
    final_masks: Dict[str, np.ndarray] = {}
    for local_step, global_idx in enumerate(selected_indices):
        seg = segs[global_idx] if global_idx < len(segs) and isinstance(segs[global_idx], dict) else {}
        payload = seg.get(target_id, {}) if isinstance(seg.get(target_id, {}), dict) else {}
        rle = str(payload.get("rle", ""))
        mask = _burst_mask(rle, height=height, width=width) if rle else None
        target_masks.append(mask if isinstance(mask, np.ndarray) and np.any(mask) else None)
        if local_step == future_step:
            for cand_id, cand_payload in seg.items():
                if not isinstance(cand_payload, dict):
                    continue
                cand_rle = str(cand_payload.get("rle", ""))
                if not cand_rle:
                    continue
                cand_mask = _burst_mask(cand_rle, height=height, width=width)
                if np.any(cand_mask):
                    final_masks[str(cand_id)] = cand_mask
    target_future = target_masks[future_step]
    if target_future is None:
        raise RuntimeError(f"future target mask missing for {item.get('protocol_item_id')}")
    sizes = [(int(width), int(height)) for _ in selected_indices]
    return target_masks, sizes, final_masks, target_future


def _extract_target_masks(item: Dict[str, Any]) -> Tuple[List[np.ndarray | None], List[Tuple[int, int]], Dict[str, np.ndarray], np.ndarray]:
    return _extract_entity_masks(item=item, entity_id=None)


def _protocol_observed_context_candidate_ids(item: Dict[str, Any], max_context_entities: int = 8) -> List[str]:
    max_context = max(int(max_context_entities), 1)
    dataset = str(item.get("dataset", "")).strip().upper()
    query_step = int(item.get("query_step", 0))
    target_id = str(item.get("target_id", ""))
    scored: Dict[str, Tuple[int, float]] = {}

    if dataset == "VIPSEG":
        selected_mask_paths = [Path(x) for x in item.get("selected_mask_paths", [])]
        obs_paths = selected_mask_paths[: min(OBS_LEN, len(selected_mask_paths))]
        for step_i, mask_path in enumerate(obs_paths):
            arr = _vipseg_mask(mask_path)
            for cand in [int(x) for x in np.unique(arr).tolist() if int(x) >= 125]:
                area = float((arr == cand).mean())
                key = str(int(cand))
                prev_presence, prev_area = scored.get(key, (0, 0.0))
                scored[key] = (prev_presence + 1, prev_area + area + (10.0 if step_i == query_step else 0.0))
    else:
        annotation_file = Path(str(item.get("burst_annotation_file", "")))
        seq_map = _burst_seq_map(annotation_file)
        seq = seq_map.get((str(item.get("burst_dataset_name", "")), str(item.get("burst_seq_name", ""))), {})
        segs = seq.get("segmentations", []) if isinstance(seq.get("segmentations", []), list) else []
        selected_indices = [int(x) for x in item.get("selected_frame_indices", [])][:OBS_LEN]
        height = int((item.get("image_size") or {}).get("height", seq.get("height", 0)))
        width = int((item.get("image_size") or {}).get("width", seq.get("width", 0)))
        denom = max(float(height * width), 1.0)
        for step_i, global_idx in enumerate(selected_indices):
            seg = segs[global_idx] if global_idx < len(segs) and isinstance(segs[global_idx], dict) else {}
            for cand_id, cand_payload in seg.items():
                if not isinstance(cand_payload, dict):
                    continue
                cand_rle = str(cand_payload.get("rle", ""))
                if not cand_rle:
                    continue
                cand_mask = _burst_mask(cand_rle, height=height, width=width)
                if not np.any(cand_mask):
                    continue
                area = float(cand_mask.sum() / denom)
                key = str(cand_id)
                prev_presence, prev_area = scored.get(key, (0, 0.0))
                scored[key] = (prev_presence + 1, prev_area + area + (10.0 if step_i == query_step else 0.0))

    ordered = [target_id]
    extras = [
        cand_id
        for cand_id, _ in sorted(
            scored.items(),
            key=lambda kv: (
                0 if str(kv[0]) == target_id else 1,
                -int(kv[1][0]),
                -float(kv[1][1]),
                str(kv[0]),
            ),
        )
        if str(cand_id) != target_id
    ]
    ordered.extend(extras[: max_context - 1])
    return [str(x) for x in ordered[:max_context]]


def _build_single_item_batch(item: Dict[str, Any]) -> Tuple[Dict[str, Any], np.ndarray, Dict[str, np.ndarray]]:
    frame_paths = [Path(x) for x in item.get("selected_frame_paths", [])]
    target_masks, sizes, future_masks, target_future_mask = _extract_target_masks(item)
    boxes: List[np.ndarray] = []
    present: List[bool] = []
    last_box = None
    query_step = int(item.get("query_step", 0))
    query_rgb = np.asarray(Image.open(frame_paths[query_step]).convert("RGB"), dtype=np.float32) / 255.0
    query_mask = target_masks[query_step]
    if query_mask is None:
        raise RuntimeError(f"query target mask missing for {item.get('protocol_item_id')}")
    sem_box, sem_mask_used, sem_fg_ratio = _box_from_mask_or_center(
        mask=query_mask.astype(np.uint8),
        width=int(query_rgb.shape[1]),
        height=int(query_rgb.shape[0]),
        radius=12,
    )
    for (width, height), mask in zip(sizes, target_masks):
        if mask is not None and np.any(mask):
            box, _, _ = _box_from_mask_or_center(mask.astype(np.uint8), width=width, height=height, radius=12)
            last_box = box
            present.append(True)
            boxes.append(box)
        else:
            fallback = last_box if last_box is not None else _box_from_mask_or_center(None, width=width, height=height, radius=12)[0]
            boxes.append(np.asarray(fallback, dtype=np.float32))
            present.append(False)
    state = _build_state_from_boxes(boxes=boxes, sizes=sizes)
    obs_state = state[:OBS_LEN, None, :]
    fut_state = state[OBS_LEN:, None, :]
    obs_valid = np.asarray(present[:OBS_LEN], dtype=bool)[:, None]
    fut_valid = np.asarray(present[OBS_LEN:], dtype=bool)[:, None]
    semantic_rgb_crop, semantic_mask_crop, mask_crop_available = _build_semantic_crops(
        rgb=query_rgb,
        mask=query_mask.astype(np.uint8),
        box_xyxy=sem_box,
        crop_size=64,
    )
    semantic_feat = _semantic_feature(
        rgb=query_rgb,
        mask=query_mask.astype(np.uint8),
        box_xyxy=sem_box,
        mask_used=bool(sem_mask_used),
        fg_ratio=float(sem_fg_ratio),
    )
    sample = {
        "obs_state": torch.from_numpy(obs_state).to(torch.float32),
        "fut_state": torch.from_numpy(fut_state).to(torch.float32),
        "obs_valid": torch.from_numpy(obs_valid).to(torch.bool),
        "fut_valid": torch.from_numpy(fut_valid).to(torch.bool),
        "point_ids": torch.tensor([0], dtype=torch.long),
        "meta": {
            "dataset": str(item.get("dataset", "")),
            "clip_id": str(item.get("clip_id", "")),
            "target_id": str(item.get("target_id", "")),
            "subset_tags": list(item.get("subset_tags", [])),
        },
        "semantic_features": torch.from_numpy(semantic_feat[None, :]).to(torch.float32),
        "semantic_boxes": torch.from_numpy(np.asarray([sem_box], dtype=np.float32)).to(torch.float32),
        "semantic_mask": torch.tensor([True], dtype=torch.bool),
        "semantic_rgb_crop": torch.from_numpy(semantic_rgb_crop[None, ...]).to(torch.float32),
        "semantic_mask_crop": torch.from_numpy(semantic_mask_crop[None, ...]).to(torch.float32),
        "semantic_crop_valid": torch.tensor([True], dtype=torch.bool),
        "semantic_mask_crop_valid": torch.tensor([bool(mask_crop_available)], dtype=torch.bool),
        "semantic_rgb_crop_temporal": torch.from_numpy(semantic_rgb_crop[None, None, ...]).to(torch.float32),
        "semantic_mask_crop_temporal": torch.from_numpy(semantic_mask_crop[None, None, ...]).to(torch.float32),
        "semantic_temporal_valid": torch.tensor([[True]], dtype=torch.bool),
        "semantic_instance_id_map": torch.from_numpy(query_mask.astype(np.int64)[None, ...]).to(torch.long),
        "semantic_instance_id_crop": torch.from_numpy((semantic_mask_crop > 0.5).astype(np.int64)[None, None, ...]).to(torch.long),
        "semantic_instance_id_temporal": torch.from_numpy((semantic_mask_crop > 0.5).astype(np.int64)[None, None, ...]).to(torch.long),
        "semantic_instance_valid": torch.tensor([[True]], dtype=torch.bool),
        "semantic_objectness_score": torch.tensor([max(float(sem_fg_ratio), 0.0)], dtype=torch.float32),
        "semantic_entity_dominant_instance_id": torch.tensor([1], dtype=torch.long),
        "semantic_entity_instance_overlap_score_over_time": torch.tensor([[1.0]], dtype=torch.float32),
        "semantic_entity_true_instance_confidence": torch.tensor([1.0], dtype=torch.float32),
        "semantic_teacher_prior": torch.zeros((1, 512), dtype=torch.float32),
        "entity_boxes_over_time": torch.from_numpy(np.asarray(boxes, dtype=np.float32)[:, None, :]).to(torch.float32),
        "entity_masks_over_time": [],
        "semantic_frame_path": str(frame_paths[query_step]),
        "semantic_mask_path": "",
        "semantic_source_mode": "object_region_or_mask_crop_visual_state",
        "current_mainline_semantic_source": "crop_visual_encoder",
        "legacy_semantic_source": "hand_crafted_stats",
        "semantic_source_summary": {},
    }
    return stage2_semantic_collate_fn([sample]), target_future_mask, future_masks


def _stage1_free_rollout_predict(stage1_model: Any, batch: Dict[str, Any], device: torch.device) -> np.ndarray:
    obs_state = batch["obs_state"].to(device)
    token_mask = batch["token_mask"].to(device=device, dtype=torch.bool)
    bsz, _, k_len, d_state = obs_state.shape
    state_seq = torch.zeros((bsz, TOTAL_STEPS, k_len, d_state), device=device, dtype=obs_state.dtype)
    state_seq[:, :OBS_LEN] = obs_state
    with torch.no_grad():
        for step in range(FUT_LEN):
            shifted = trainer._prepare_shifted(state_seq)
            out = stage1_model(shifted, token_mask=token_mask)
            pred_state_t = out["pred_state"][:, OBS_LEN + step : OBS_LEN + step + 1].detach()
            state_seq[:, OBS_LEN + step : OBS_LEN + step + 1] = pred_state_t
    return state_seq[:, OBS_LEN:, :, 0:2].detach().cpu().numpy()


def _stage2_free_rollout_predict(method: LoadedMethod, batch: Dict[str, Any], device: torch.device) -> np.ndarray:
    moved = trainer._to_device(batch, device=device, non_blocking=False)
    with torch.no_grad():
        out = trainer._free_rollout_predict(
            stage1_model=method.stage1_model,
            semantic_encoder=method.semantic_encoder,
            semantic_fusion=method.semantic_fusion,
            readout_head=method.readout_head,
            structure_mode=str(method.stage2_structure_mode),
            trace_unit_tokenizer=method.trace_unit_tokenizer,
            trace_unit_factorized_state=method.trace_unit_factorized_state,
            trace_unit_handshake=method.trace_unit_handshake,
            trace_unit_broadcast=method.trace_unit_broadcast,
            trace_unit_disable_instance_path=bool(method.trace_unit_disable_instance_path),
            batch=moved,
            obs_len=OBS_LEN,
            fut_len=FUT_LEN,
            semantic_source_mainline=method.semantic_source_mainline,
            allow_stage1_grad=False,
        )
    return out["pred_coord"].detach().cpu().numpy()


def _predict_final_coord(method: LoadedMethod, batch: Dict[str, Any], device: torch.device) -> Tuple[float, float]:
    pred = _stage1_free_rollout_predict(method.stage1_model, batch, device=device) if method.method_type == "stage1" else _stage2_free_rollout_predict(method, batch, device=device)
    coord = pred[0, -1, 0]
    return float(coord[0]), float(coord[1])


def _candidate_rankings(pred_xy_norm: Tuple[float, float], future_masks: Dict[str, np.ndarray], width: int, height: int) -> List[Dict[str, Any]]:
    px = min(max(pred_xy_norm[0] * float(width), 0.0), float(width - 1))
    py = min(max(pred_xy_norm[1] * float(height), 0.0), float(height - 1))
    diag = max(math.sqrt(float(width * width + height * height)), 1.0)
    rows: List[Dict[str, Any]] = []
    for cand_id, mask in future_masks.items():
        if not np.any(mask):
            continue
        inside = bool(mask[int(round(py)), int(round(px))]) if 0 <= int(round(py)) < mask.shape[0] and 0 <= int(round(px)) < mask.shape[1] else False
        cx, cy = _mask_centroid(mask)
        dist = math.sqrt((cx - px) ** 2 + (cy - py) ** 2) / diag
        rows.append(
            {
                "candidate_id": str(cand_id),
                "inside": bool(inside),
                "normalized_centroid_distance": float(dist),
            }
        )
    rows.sort(
        key=lambda row: (
            0 if bool(row.get("inside", False)) else 1,
            float(row.get("normalized_centroid_distance", 1e9)),
            str(row.get("candidate_id", "")),
        )
    )
    return rows


def _candidate_ranking(pred_xy_norm: Tuple[float, float], future_masks: Dict[str, np.ndarray], width: int, height: int) -> Tuple[str, float]:
    rows = _candidate_rankings(pred_xy_norm=pred_xy_norm, future_masks=future_masks, width=width, height=height)
    if not rows:
        return "none", 1e9
    top = rows[0]
    return str(top.get("candidate_id", "none")), float(top.get("normalized_centroid_distance", 1e9))


def _evaluate_item(method: LoadedMethod, item: Dict[str, Any], batch: Dict[str, Any], target_future_mask: np.ndarray, future_masks: Dict[str, np.ndarray], device: torch.device) -> Dict[str, Any]:
    width = int((item.get("image_size") or {}).get("width", target_future_mask.shape[1]))
    height = int((item.get("image_size") or {}).get("height", target_future_mask.shape[0]))
    pred_x_norm, pred_y_norm = _predict_final_coord(method, batch, device=device)
    pred_x = min(max(pred_x_norm * float(width), 0.0), float(width - 1))
    pred_y = min(max(pred_y_norm * float(height), 0.0), float(height - 1))
    target_cx, target_cy = _mask_centroid(target_future_mask)
    diag = max(math.sqrt(float(width * width + height * height)), 1.0)
    localization_error = float(math.sqrt((pred_x - target_cx) ** 2 + (pred_y - target_cy) ** 2) / diag)
    y_idx = int(round(pred_y))
    x_idx = int(round(pred_x))
    hit = bool(0 <= y_idx < target_future_mask.shape[0] and 0 <= x_idx < target_future_mask.shape[1] and target_future_mask[y_idx, x_idx])
    ranked_candidates = _candidate_rankings((pred_x_norm, pred_y_norm), future_masks, width=width, height=height)
    top1_id = str(ranked_candidates[0]["candidate_id"]) if ranked_candidates else "none"
    top1_mask = future_masks.get(str(top1_id))
    target_rank = 0
    for rank_idx, row in enumerate(ranked_candidates, start=1):
        if str(row.get("candidate_id", "")) == str(item.get("target_id")):
            target_rank = rank_idx
            break
    return {
        "query_future_top1_acc": 1.0 if str(top1_id) == str(item.get("target_id")) else 0.0,
        "query_future_hit_rate": 1.0 if hit else 0.0,
        "query_future_localization_error": float(localization_error),
        "future_mask_iou_at_top1": float(_mask_iou(top1_mask, target_future_mask)),
        "top1_candidate_id": str(top1_id),
        "target_rank": int(target_rank),
        "candidate_count": int(len(ranked_candidates)),
        "top5_hit": 1.0 if int(target_rank) > 0 and int(target_rank) <= 5 else 0.0,
        "mrr": float(1.0 / float(target_rank)) if int(target_rank) > 0 else 0.0,
        "ranked_candidate_ids": [str(row.get("candidate_id", "")) for row in ranked_candidates[:10]],
        "predicted_future_xy_norm": [float(pred_x_norm), float(pred_y_norm)],
        "predicted_future_xy_pixels": [float(pred_x), float(pred_y)],
    }


def _aggregate_item_metrics(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {
            "count": 0,
            "query_future_top1_acc": 0.0,
            "query_future_hit_rate": 0.0,
            "query_future_localization_error": 1e9,
            "future_mask_iou_at_top1": 0.0,
            "hard_subset_top1_acc": 0.0,
        }
    count = float(len(rows))
    return {
        "count": int(len(rows)),
        "query_future_top1_acc": float(sum(float(r["query_future_top1_acc"]) for r in rows) / count),
        "query_future_hit_rate": float(sum(float(r["query_future_hit_rate"]) for r in rows) / count),
        "query_future_localization_error": float(sum(float(r["query_future_localization_error"]) for r in rows) / count),
        "future_mask_iou_at_top1": float(sum(float(r["future_mask_iou_at_top1"]) for r in rows) / count),
        "hard_subset_top1_acc": float(sum(float(r["query_future_top1_acc"]) for r in rows) / count),
    }


def parse_args() -> Any:
    parser = ArgumentParser(description="Run real Stage2 state-identifiability / future grounding evaluation")
    parser.add_argument("--protocol-json", default=str(ROOT / "reports/stage2_state_identifiability_protocol_20260415.json"))
    parser.add_argument("--output-json", default=str(ROOT / "reports/stage2_state_identifiability_eval_20260415.json"))
    parser.add_argument("--output-md", default=str(ROOT / "docs/STAGE2_STATE_IDENTIFIABILITY_EVAL_20260415.md"))
    parser.add_argument("--stage1-checkpoint", default=str(ROOT / "outputs/checkpoints/stage1_v2_longtrain_220m_mainline_20260408/best.pt"))
    parser.add_argument("--final-utility-summary", default=str(ROOT / "reports/stage2_final_utility_closure_v2_summary_20260414.json"))
    parser.add_argument("--final-utility-diagnosis", default=str(ROOT / "reports/stage2_final_utility_closure_v2_diagnosis_20260414.json"))
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--lease-path", default=str(ROOT / "reports/stage1_v2_gpu_lease_20260408.json"))
    parser.add_argument("--eval-required-mem-gb", type=float, default=40.0)
    parser.add_argument("--eval-safety-margin-gb", type=float, default=8.0)
    return parser.parse_args()


def main() -> None:
    _apply_process_title_normalization()
    args = parse_args()
    protocol = read_json(args.protocol_json)
    items = protocol.get("items", []) if isinstance(protocol.get("items", []), list) else []
    device, device_info = _select_eval_device(args)
    specs = _load_method_specs(args)

    prepared_items: List[Tuple[Dict[str, Any], Dict[str, Any], np.ndarray, Dict[str, np.ndarray]]] = []
    per_item: List[Dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        batch, target_future_mask, future_masks = _build_single_item_batch(item)
        prepared_items.append((item, batch, target_future_mask, future_masks))
        per_item.append(
            {
                "protocol_item_id": str(item.get("protocol_item_id", "")),
                "dataset": str(item.get("dataset", "")),
                "clip_id": str(item.get("clip_id", "")),
                "subset_tags": list(item.get("subset_tags", [])),
                "target_id": str(item.get("target_id", "")),
                "methods": {},
            }
        )

    try:
        for spec in specs:
            method = _load_method(spec, device=device)
            for item_row, prepared in zip(per_item, prepared_items):
                item, batch, target_future_mask, future_masks = prepared
                item_row["methods"][method.name] = _evaluate_item(
                    method=method,
                    item=item,
                    batch=batch,
                    target_future_mask=target_future_mask,
                    future_masks=future_masks,
                    device=device,
                )
            _release_method(method)
    finally:
        lease_id = str(device_info.get("lease_id", "")).strip()
        if lease_id:
            try:
                release_lease(lease_id=lease_id, lease_path=str(args.lease_path))
            except Exception:
                pass

    panel_names = [
        "full_identifiability_panel",
        "occlusion_reappearance",
        "crossing_ambiguity",
        "small_object",
        "appearance_change",
        "long_gap_persistence",
    ]
    method_rows: List[Dict[str, Any]] = []
    for spec in specs:
        panel_metrics: Dict[str, Any] = {}
        all_rows = []
        hard_rows = []
        for item_row in per_item:
            score = (item_row.get("methods") or {}).get(spec.name)
            if not isinstance(score, dict):
                continue
            all_rows.append(score)
            if item_row.get("subset_tags"):
                hard_rows.append(score)
        panel_metrics["full_identifiability_panel"] = _aggregate_item_metrics(all_rows)
        panel_metrics["hard_subsets"] = _aggregate_item_metrics(hard_rows)
        for panel in panel_names[1:]:
            subset_rows = [
                (item_row.get("methods") or {}).get(spec.name)
                for item_row in per_item
                if panel in list(item_row.get("subset_tags", []))
            ]
            panel_metrics[panel] = _aggregate_item_metrics([r for r in subset_rows if isinstance(r, dict)])
        method_rows.append(
            {
                "name": spec.name,
                "run_name": spec.run_name,
                "method_type": spec.method_type,
                "checkpoint_path": spec.checkpoint_path,
                "panels": panel_metrics,
                "query_future_top1_acc": float(panel_metrics["full_identifiability_panel"]["query_future_top1_acc"]),
                "query_future_hit_rate": float(panel_metrics["full_identifiability_panel"]["query_future_hit_rate"]),
                "query_future_localization_error": float(panel_metrics["full_identifiability_panel"]["query_future_localization_error"]),
                "future_mask_iou_at_top1": float(panel_metrics["full_identifiability_panel"]["future_mask_iou_at_top1"]),
                "hard_subset_top1_acc": float(panel_metrics["hard_subsets"]["query_future_top1_acc"]),
                "ambiguous_case_top1_acc": float(panel_metrics["crossing_ambiguity"]["query_future_top1_acc"]),
                "small_object_query_top1_acc": float(panel_metrics["small_object"]["query_future_top1_acc"]),
                "appearance_change_query_top1_acc": float(panel_metrics["appearance_change"]["query_future_top1_acc"]),
            }
        )

    by_name = {row["name"]: row for row in method_rows}
    calibration = by_name.get("calibration_only_mainline_best", {})
    stage1 = by_name.get("stage1_frozen_baseline", {})
    legacysem = by_name.get("legacysem_best", {})
    cropenc = by_name.get("cropenc_baseline_best", {})

    def _ge(a: Dict[str, Any], b: Dict[str, Any], key: str) -> bool:
        if not a or not b:
            return False
        return float(a.get(key, -1.0)) >= float(b.get(key, -1.0))

    improved_stage1 = _ge(calibration, stage1, "query_future_top1_acc")
    improved_legacysem = _ge(calibration, legacysem, "query_future_top1_acc")
    improved_cropenc = _ge(calibration, cropenc, "query_future_top1_acc")
    improved_all = bool(improved_stage1 and improved_legacysem and improved_cropenc)
    improved_hard = bool(
        calibration
        and stage1
        and legacysem
        and cropenc
        and float(calibration.get("hard_subset_top1_acc", -1.0))
        >= max(
            float(stage1.get("hard_subset_top1_acc", -1.0)),
            float(legacysem.get("hard_subset_top1_acc", -1.0)),
            float(cropenc.get("hard_subset_top1_acc", -1.0)),
        )
    )
    protocol_success = bool(
        items
        and int((protocol.get("panel_counts") or {}).get("full_identifiability_panel", 0)) > 0
        and int((protocol.get("panel_counts") or {}).get("crossing_ambiguity", 0)) > 0
        and int((protocol.get("panel_counts") or {}).get("small_object", 0)) > 0
        and int((protocol.get("panel_counts") or {}).get("appearance_change", 0)) > 0
        and int((protocol.get("panel_counts") or {}).get("long_gap_persistence", 0)) > 0
    )

    payload = {
        "generated_at_utc": now_iso(),
        "benchmark_scope": "real state-identifiability / future grounding with true instance continuity and future masks",
        "official_benchmark": False,
        "protocol_contribution": True,
        "selected_device": str(device),
        "device_info": device_info,
        "protocol_item_count": int(len(per_item)),
        "panel_counts": dict(protocol.get("panel_counts", {})),
        "methods": method_rows,
        "per_item_results": per_item,
        "state_identifiability_protocol_success": bool(protocol_success),
        "future_grounding_usefulness_improved_vs_stage1": bool(improved_stage1),
        "future_grounding_usefulness_improved_vs_legacysem": bool(improved_legacysem),
        "future_grounding_usefulness_improved_vs_cropenc": bool(improved_cropenc),
        "future_grounding_usefulness_improved_vs_baselines": bool(improved_all),
        "future_grounding_usefulness_improved_on_hard_subsets": bool(improved_hard),
    }
    write_json(args.output_json, payload)

    lines = [
        "# Stage2 State-Identifiability Eval 20260415",
        "",
        "- scope: real future grounding with true instance identity / future mask continuity",
        "- official_benchmark: False",
        f"- protocol_item_count: {len(per_item)}",
        f"- selected_device: {device}",
        f"- state_identifiability_protocol_success: {protocol_success}",
        f"- future_grounding_usefulness_improved_vs_stage1: {improved_stage1}",
        f"- future_grounding_usefulness_improved_vs_legacysem: {improved_legacysem}",
        f"- future_grounding_usefulness_improved_vs_cropenc: {improved_cropenc}",
        f"- future_grounding_usefulness_improved_vs_baselines: {improved_all}",
        f"- future_grounding_usefulness_improved_on_hard_subsets: {improved_hard}",
        "",
        "| method | run_name | top1_acc | hit_rate | loc_error | top1_mask_iou | hard_top1 | ambiguity_top1 | small_top1 | appearance_top1 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in method_rows:
        lines.append(
            f"| {row['name']} | {row['run_name']} | "
            f"{row['query_future_top1_acc']:.4f} | {row['query_future_hit_rate']:.4f} | "
            f"{row['query_future_localization_error']:.6f} | {row['future_mask_iou_at_top1']:.4f} | "
            f"{row['hard_subset_top1_acc']:.4f} | {row['ambiguous_case_top1_acc']:.4f} | "
            f"{row['small_object_query_top1_acc']:.4f} | {row['appearance_change_query_top1_acc']:.4f} |"
        )
    write_md(args.output_md, lines)
    print(json.dumps(payload, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
