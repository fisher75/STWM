#!/usr/bin/env python3
from __future__ import annotations

from argparse import ArgumentParser
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple
import gc
import hashlib
import json
import math
import os
import sys

import numpy as np
from PIL import Image
import torch

for candidate in [
    Path("/raid/chen034/workspace/stwm/code"),
    Path("/home/chen034/workspace/stwm/code"),
]:
    if candidate.exists() and str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from stwm.tools import run_stage2_state_identifiability_eval_20260415 as prev
from stwm.tools import run_stage2_state_identifiability_eval_v3_20260416 as evalv3
from stwm.tracewm_v2_stage2.trainers import train_tracewm_stage2_smalltrain as trainer


ROOT = prev.ROOT
SEEDS = [42, 123, 456, 654, 789, 321]
PANEL_NAME = "densified_200_context_preserving"
TUSB_BEST = "TUSB-v3.1 best.pt"
TUSB_SIDECAR = "TUSB-v3.1 best_semantic_hard.pt"
CAL = "calibration-only::best.pt"
CROP = "cropenc::best.pt"
LEGACY = "legacysem::best.pt"
METHOD_ORDER = [TUSB_BEST, TUSB_SIDECAR, CAL, CROP, LEGACY]
VAL_BUCKETS = {0, 1, 2}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _apply_process_title_normalization(default_title: str = "python") -> None:
    title = str(os.environ.get("STWM_PROC_TITLE", default_title)).strip() or default_title
    mode = str(os.environ.get("STWM_PROC_TITLE_MODE", "generic")).strip().lower()
    if mode != "generic":
        return
    lowered = title.lower()
    if "stwm" in lowered or "tracewm" in lowered or "/raid/" in lowered or "/home/" in lowered:
        title = default_title
    try:
        import setproctitle  # type: ignore

        setproctitle.setproctitle(title)
    except Exception:
        pass


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def _write_md(path: Path, lines: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _sha256_json(payload: Any) -> str:
    raw = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _mean(values: Iterable[float]) -> float:
    vals = [float(v) for v in values]
    return float(sum(vals) / max(len(vals), 1))


def _std(values: Iterable[float]) -> float:
    vals = np.asarray([float(v) for v in values], dtype=np.float64)
    if vals.size <= 1:
        return 0.0
    return float(vals.std(ddof=0))


def _item_split(protocol_item_id: str) -> str:
    digest = hashlib.sha256(protocol_item_id.encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) % 10
    return "val" if bucket in VAL_BUCKETS else "test"


def _load_checkpoint_map(main_audit: Path, sidecar_audit: Path) -> Dict[str, Dict[int, Dict[str, Any]]]:
    main_payload = prev.read_json(main_audit)
    side_payload = prev.read_json(sidecar_audit)
    per_method = main_payload.get("per_method", {}) if isinstance(main_payload.get("per_method", {}), dict) else {}
    rows_tusb_side = side_payload.get("rows", []) if isinstance(side_payload.get("rows", []), list) else []

    def _rows(method_name: str) -> List[Dict[str, Any]]:
        block = per_method.get(method_name, {})
        return block.get("rows", []) if isinstance(block, dict) and isinstance(block.get("rows", []), list) else []

    mapping: Dict[str, Dict[int, Dict[str, Any]]] = {
        TUSB_BEST: {},
        TUSB_SIDECAR: {},
        CAL: {},
        CROP: {},
        LEGACY: {},
    }
    for row in _rows("TUSB-v3.1"):
        seed = int(row.get("seed", -1))
        if seed in SEEDS and bool(row.get("best_pt_exists", False)):
            mapping[TUSB_BEST][seed] = {
                "run_name": str(row.get("run_name", "")),
                "checkpoint_path": str(row.get("best_pt_path", "")),
            }
    for row in rows_tusb_side:
        seed = int(row.get("seed", -1))
        if seed in SEEDS and bool(row.get("best_semantic_hard_exists", False)):
            mapping[TUSB_SIDECAR][seed] = {
                "run_name": str(row.get("run_name", "")),
                "checkpoint_path": str(row.get("best_semantic_hard_path", "")),
            }
    for display_name, method_name in [
        (CAL, "calibration-only"),
        (CROP, "cropenc baseline"),
        (LEGACY, "legacysem baseline"),
    ]:
        for row in _rows(method_name):
            seed = int(row.get("seed", -1))
            if seed in SEEDS and bool(row.get("best_pt_exists", False)):
                mapping[display_name][seed] = {
                    "run_name": str(row.get("run_name", "")),
                    "checkpoint_path": str(row.get("best_pt_path", "")),
                }
    missing: List[str] = []
    for method_name, seeds in mapping.items():
        for seed in SEEDS:
            entry = seeds.get(seed)
            ckpt = Path(str((entry or {}).get("checkpoint_path", "")))
            if not entry or not ckpt.exists():
                missing.append(f"{method_name} seed={seed} missing checkpoint")
    if missing:
        raise RuntimeError("; ".join(missing))
    return mapping


def _prepare_candidate_inputs(item: Dict[str, Any], target_future_mask: np.ndarray, future_masks: Dict[str, np.ndarray]) -> Dict[str, Any]:
    frame_paths = [Path(x) for x in item.get("selected_frame_paths", [])]
    future_step = int(item.get("future_step", prev.FUT_LEN + prev.OBS_LEN - 1))
    with Image.open(frame_paths[future_step]) as img:
        future_rgb = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    width = int((item.get("image_size") or {}).get("width", target_future_mask.shape[1]))
    height = int((item.get("image_size") or {}).get("height", target_future_mask.shape[0]))
    target_cx, target_cy = prev._mask_centroid(target_future_mask)
    diag = max(math.sqrt(float(width * width + height * height)), 1.0)
    rows: Dict[str, Any] = {}
    for cand_id, cand_mask in future_masks.items():
        if not isinstance(cand_mask, np.ndarray) or not np.any(cand_mask):
            continue
        box_xyxy, mask_used, fg_ratio = prev._box_from_mask_or_center(
            mask=cand_mask.astype(np.uint8),
            width=width,
            height=height,
            radius=12,
        )
        rgb_crop, mask_crop, mask_valid = prev._build_semantic_crops(
            rgb=future_rgb,
            mask=cand_mask.astype(np.uint8),
            box_xyxy=box_xyxy,
            crop_size=64,
        )
        sem_feature = prev._semantic_feature(
            rgb=future_rgb,
            mask=cand_mask.astype(np.uint8),
            box_xyxy=box_xyxy,
            mask_used=bool(mask_used),
            fg_ratio=float(fg_ratio),
        )
        cand_cx, cand_cy = prev._mask_centroid(cand_mask)
        rows[str(cand_id)] = {
            "rgb_crop": rgb_crop.astype(np.float32),
            "mask_crop": mask_crop.astype(np.float32),
            "semantic_feature": sem_feature.astype(np.float32),
            "mask_valid": bool(mask_valid),
            "iou_with_target": float(prev._mask_iou(cand_mask, target_future_mask)),
            "centroid_distance_to_target": float(
                math.sqrt((cand_cx - target_cx) ** 2 + (cand_cy - target_cy) ** 2) / diag
            ),
        }
    return {
        "width": width,
        "height": height,
        "target_future_mask": target_future_mask,
        "future_masks": future_masks,
        "target_id": str(item.get("target_id", "")),
        "candidates": rows,
    }


def _prepare_items(protocol_path: Path, max_items: int = 0) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Counter]:
    protocol = prev.read_json(protocol_path)
    items = protocol.get("items", []) if isinstance(protocol.get("items", []), list) else []
    if int(max_items) > 0:
        items = items[: int(max_items)]
    prepared: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []
    skipped_counts: Counter = Counter()
    for item in items:
        if not isinstance(item, dict):
            skipped.append({"protocol_item_id": "unknown", "reason": "non_dict_item"})
            skipped_counts["non_dict_item"] += 1
            continue
        protocol_item_id = str(item.get("protocol_item_id", ""))
        try:
            batch, target_future_mask, future_masks = evalv3._build_context_preserving_item_batch_v3(
                item,
                temporal_window=5,
                max_context_entities=8,
            )
            candidate_inputs = _prepare_candidate_inputs(item, target_future_mask, future_masks)
            prepared.append(
                {
                    "item": item,
                    "batch": batch,
                    "candidate_inputs": candidate_inputs,
                    "protocol_item_id": protocol_item_id,
                }
            )
        except Exception as exc:
            reason = f"{type(exc).__name__}:{exc}"
            skipped.append({"protocol_item_id": protocol_item_id, "reason": reason})
            skipped_counts[reason] += 1
    return prepared, skipped, skipped_counts


def _coord_score_map(pred_xy_norm: Tuple[float, float], future_masks: Dict[str, np.ndarray], width: int, height: int) -> Dict[str, float]:
    rows = prev._candidate_rankings(pred_xy_norm=pred_xy_norm, future_masks=future_masks, width=width, height=height)
    scores: Dict[str, float] = {}
    for row in rows:
        inside_bonus = 1.0 if bool(row.get("inside", False)) else 0.0
        scores[str(row.get("candidate_id", ""))] = float(inside_bonus - float(row.get("normalized_centroid_distance", 1.0)))
    return scores


def _sorted_rank_from_scores(scores: Dict[str, float], target_id: str) -> Dict[str, Any]:
    ordered = sorted(scores.items(), key=lambda kv: (-float(kv[1]), str(kv[0])))
    ranked_ids = [str(cid) for cid, _ in ordered]
    target_rank = 0
    for idx, cid in enumerate(ranked_ids, start=1):
        if str(cid) == str(target_id):
            target_rank = idx
            break
    top1 = ranked_ids[0] if ranked_ids else "none"
    return {
        "top1_candidate_id": str(top1),
        "target_rank": int(target_rank),
        "top1": 1.0 if str(top1) == str(target_id) else 0.0,
        "top5_hit": 1.0 if 0 < int(target_rank) <= 5 else 0.0,
        "mrr": float(1.0 / float(target_rank)) if int(target_rank) > 0 else 0.0,
        "ranked_candidate_ids": ranked_ids[:10],
    }


def _ridge_project_query_to_semantic(
    unit_vectors: torch.Tensor,
    semantic_vectors: torch.Tensor,
    query_vector: torch.Tensor,
    ridge: float = 1e-3,
) -> torch.Tensor:
    if unit_vectors.ndim != 2 or semantic_vectors.ndim != 2:
        raise ValueError("ridge projection expects [N,D]")
    if unit_vectors.shape[0] < 2:
        dims = min(int(query_vector.shape[-1]), int(semantic_vectors.shape[-1]))
        out = torch.zeros((int(semantic_vectors.shape[-1]),), device=query_vector.device, dtype=query_vector.dtype)
        out[:dims] = query_vector[:dims]
        return out
    gram = unit_vectors @ unit_vectors.transpose(0, 1)
    eye = torch.eye(int(gram.shape[0]), device=gram.device, dtype=gram.dtype)
    coeff = torch.linalg.solve(gram + float(ridge) * eye, semantic_vectors)
    return (query_vector @ unit_vectors.transpose(0, 1)) @ coeff


def _extract_unit_representations(
    batch_gpu: Dict[str, Any],
    teacher_forced_out: Dict[str, Any],
) -> Dict[str, Any]:
    trace_aux = teacher_forced_out.get("trace_unit_aux", {}) if isinstance(teacher_forced_out.get("trace_unit_aux", {}), dict) else {}
    if not trace_aux:
        return {"available": False, "reason": "trace_unit_aux_missing"}
    assignment = trace_aux.get("assignment")
    z_sem = trace_aux.get("z_sem")
    z_dyn = trace_aux.get("z_dyn")
    if not isinstance(assignment, torch.Tensor) or not isinstance(z_sem, torch.Tensor) or not isinstance(z_dyn, torch.Tensor):
        return {"available": False, "reason": "trace_unit_tensor_missing"}
    obs_len = min(int(batch_gpu["obs_state"].shape[1]), int(assignment.shape[1]))
    assign_obs = assignment[0, :obs_len]
    z_sem_obs = z_sem[0, :obs_len]
    z_dyn_obs = z_dyn[0, :obs_len]
    obs_valid = batch_gpu.get("obs_valid")
    token_mask = batch_gpu.get("token_mask")
    if not isinstance(obs_valid, torch.Tensor) or not isinstance(token_mask, torch.Tensor):
        return {"available": False, "reason": "valid_mask_missing"}
    obs_valid = obs_valid[0, :obs_len].to(dtype=torch.bool)
    token_mask = token_mask[0].to(dtype=torch.bool)
    k_len = int(assign_obs.shape[1])
    entity_z_sem: List[torch.Tensor] = []
    entity_z_dyn: List[torch.Tensor] = []
    entity_assign: List[torch.Tensor] = []
    entity_valid: List[bool] = []
    for ent_idx in range(k_len):
        valid_t = obs_valid[:, ent_idx] & token_mask[ent_idx]
        if bool(valid_t.any().item()):
            idx = valid_t.nonzero(as_tuple=False).flatten()
            sem_steps = torch.einsum("tm,tmd->td", assign_obs[idx, ent_idx], z_sem_obs[idx])
            dyn_steps = torch.einsum("tm,tmd->td", assign_obs[idx, ent_idx], z_dyn_obs[idx])
            assign_steps = assign_obs[idx, ent_idx]
            entity_z_sem.append(sem_steps.mean(dim=0))
            entity_z_dyn.append(dyn_steps.mean(dim=0))
            entity_assign.append(assign_steps.mean(dim=0))
            entity_valid.append(True)
        else:
            entity_z_sem.append(torch.zeros_like(z_sem_obs[0, 0]))
            entity_z_dyn.append(torch.zeros_like(z_dyn_obs[0, 0]))
            entity_assign.append(torch.zeros_like(assign_obs[0, 0]))
            entity_valid.append(False)
    entity_z_sem_t = torch.stack(entity_z_sem, dim=0)
    entity_z_dyn_t = torch.stack(entity_z_dyn, dim=0)
    entity_assign_t = torch.stack(entity_assign, dim=0)
    dominant_unit = int(entity_assign_t[0].argmax().item())
    dominant_mass = float(entity_assign_t[0].max().item())
    return {
        "available": True,
        "entity_z_sem": entity_z_sem_t,
        "entity_z_dyn": entity_z_dyn_t,
        "entity_assign": entity_assign_t,
        "dominant_unit": dominant_unit,
        "dominant_mass": dominant_mass,
    }


def _encode_candidate_tokens(
    method: prev.LoadedMethod,
    candidate_inputs: Dict[str, Any],
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    candidate_rows = candidate_inputs.get("candidates", {})
    cand_ids = [str(x) for x in candidate_rows.keys()]
    if not cand_ids:
        return {}
    with torch.no_grad():
        if str(method.semantic_source_mainline).strip().lower() == "crop_visual_encoder":
            rgb = torch.from_numpy(np.stack([candidate_rows[cid]["rgb_crop"] for cid in cand_ids], axis=0)).to(device=device, dtype=torch.float32)[None, ...]
            mask = torch.from_numpy(np.stack([candidate_rows[cid]["mask_crop"] for cid in cand_ids], axis=0)).to(device=device, dtype=torch.float32)[None, ...]
            tokens = method.semantic_encoder(
                None,
                semantic_rgb_crop=rgb,
                semantic_mask_crop=mask,
                source_mode=str(method.semantic_source_mainline),
            )
        else:
            feats = torch.from_numpy(np.stack([candidate_rows[cid]["semantic_feature"] for cid in cand_ids], axis=0)).to(device=device, dtype=torch.float32)[None, ...]
            tokens = method.semantic_encoder(
                feats,
                source_mode=str(method.semantic_source_mainline),
            )
    return {cid: tokens[0, idx].detach() for idx, cid in enumerate(cand_ids)}


def _evaluate_coord_from_free_rollout(
    method: prev.LoadedMethod,
    batch: Dict[str, Any],
    target_future_mask: np.ndarray,
    future_masks: Dict[str, np.ndarray],
    target_id: str,
    width: int,
    height: int,
    device: torch.device,
) -> Tuple[Dict[str, Any], Dict[str, float]]:
    if method.method_type == "stage1":
        pred = prev._stage1_free_rollout_predict(method.stage1_model, batch, device=device)
    else:
        pred = prev._stage2_free_rollout_predict(method, batch, device=device)
    coord = pred[0, -1, 0]
    pred_x_norm = float(coord[0])
    pred_y_norm = float(coord[1])
    pred_x = min(max(pred_x_norm * float(width), 0.0), float(width - 1))
    pred_y = min(max(pred_y_norm * float(height), 0.0), float(height - 1))
    target_cx, target_cy = prev._mask_centroid(target_future_mask)
    diag = max(math.sqrt(float(width * width + height * height)), 1.0)
    localization_error = float(math.sqrt((pred_x - target_cx) ** 2 + (pred_y - target_cy) ** 2) / diag)
    y_idx = int(round(pred_y))
    x_idx = int(round(pred_x))
    hit = bool(0 <= y_idx < target_future_mask.shape[0] and 0 <= x_idx < target_future_mask.shape[1] and target_future_mask[y_idx, x_idx])
    coord_scores = _coord_score_map((pred_x_norm, pred_y_norm), future_masks, width=width, height=height)
    rank = _sorted_rank_from_scores(coord_scores, str(target_id))
    result = {
        "query_future_top1_acc": float(rank["top1"]),
        "query_future_hit_rate": 1.0 if hit else 0.0,
        "query_future_localization_error": float(localization_error),
        "future_mask_iou_at_top1": float(prev._mask_iou(future_masks.get(str(rank["top1_candidate_id"])), target_future_mask)),
        "top1_candidate_id": str(rank["top1_candidate_id"]),
        "target_rank": int(rank["target_rank"]),
        "candidate_count": int(len(coord_scores)),
        "top5_hit": float(rank["top5_hit"]),
        "mrr": float(rank["mrr"]),
        "ranked_candidate_ids": list(rank["ranked_candidate_ids"]),
        "predicted_future_xy_norm": [float(pred_x_norm), float(pred_y_norm)],
        "predicted_future_xy_pixels": [float(pred_x), float(pred_y)],
    }
    return result, coord_scores


def _evaluate_tusb_item(
    method: prev.LoadedMethod,
    prepared: Dict[str, Any],
    device: torch.device,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    item = prepared["item"]
    batch = prepared["batch"]
    candidate_inputs = prepared["candidate_inputs"]
    target_future_mask = candidate_inputs["target_future_mask"]
    future_masks = candidate_inputs["future_masks"]
    target_id = str(candidate_inputs["target_id"])
    width = int(candidate_inputs["width"])
    height = int(candidate_inputs["height"])

    coord_result, coord_scores = _evaluate_coord_from_free_rollout(
        method=method,
        batch=batch,
        target_future_mask=target_future_mask,
        future_masks=future_masks,
        target_id=target_id,
        width=width,
        height=height,
        device=device,
    )

    batch_gpu = trainer._to_device(batch, device=device, non_blocking=False)
    probe: Dict[str, Any] = {
        "available": False,
        "blocking_reason": "",
        "coord_scores": coord_scores,
    }
    if not (
        method.trace_unit_tokenizer is not None
        and method.trace_unit_factorized_state is not None
        and method.trace_unit_handshake is not None
        and method.trace_unit_broadcast is not None
    ):
        probe["blocking_reason"] = "trace_unit_modules_missing"
        return coord_result, probe

    with torch.no_grad():
        tf_out = trainer._teacher_forced_predict(
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
            batch=batch_gpu,
            obs_len=prev.OBS_LEN,
            semantic_source_mainline=method.semantic_source_mainline,
            allow_stage1_grad=False,
        )
    unit_info = _extract_unit_representations(batch_gpu, tf_out)
    if not bool(unit_info.get("available", False)):
        probe["blocking_reason"] = str(unit_info.get("reason", "unit_extraction_failed"))
        return coord_result, probe

    semantic_tokens = tf_out["semantic_tokens"][0]
    entity_z_sem = unit_info["entity_z_sem"]
    entity_z_dyn = unit_info["entity_z_dyn"]
    token_mask = batch_gpu["token_mask"][0].to(dtype=torch.bool)
    valid_entities = [idx for idx in range(int(token_mask.shape[0])) if bool(token_mask[idx].item())]
    if len(valid_entities) < 1:
        probe["blocking_reason"] = "no_valid_entities"
        return coord_result, probe
    x = entity_z_sem[valid_entities]
    y = semantic_tokens[valid_entities]
    target_proj = _ridge_project_query_to_semantic(x, y, entity_z_sem[0])
    target_sem = semantic_tokens[0]
    candidate_tokens = _encode_candidate_tokens(method, candidate_inputs, device=device)
    if not candidate_tokens:
        probe["blocking_reason"] = "candidate_tokens_missing"
        return coord_result, probe

    target_proj_n = torch.nn.functional.normalize(target_proj, dim=-1)
    target_sem_n = torch.nn.functional.normalize(target_sem, dim=-1)
    dominant_mass = float(unit_info.get("dominant_mass", 0.0))
    unit_scores: Dict[str, float] = {}
    semantic_scores: Dict[str, float] = {}
    for cand_id, token in candidate_tokens.items():
        token_n = torch.nn.functional.normalize(token, dim=-1)
        unit_sim = float(torch.dot(target_proj_n, token_n).detach().cpu().item())
        sem_sim = float(torch.dot(target_sem_n, token_n).detach().cpu().item())
        unit_scores[str(cand_id)] = float(unit_sim * (0.5 + 0.5 * dominant_mass))
        semantic_scores[str(cand_id)] = float(sem_sim)

    unit_rank = _sorted_rank_from_scores(unit_scores, target_id)
    probe = {
        "available": True,
        "coord_scores": coord_scores,
        "unit_scores": unit_scores,
        "semantic_scores": semantic_scores,
        "unit_identity_top1_candidate_id": str(unit_rank["top1_candidate_id"]),
        "unit_identity_target_rank": int(unit_rank["target_rank"]),
        "unit_identity_top1_acc": float(unit_rank["top1"]),
        "unit_identity_top5_hit": float(unit_rank["top5_hit"]),
        "unit_identity_mrr": float(unit_rank["mrr"]),
        "dominant_unit": int(unit_info["dominant_unit"]),
        "dominant_unit_mass": float(dominant_mass),
        "target_z_sem_norm": float(entity_z_sem[0].norm().detach().cpu().item()),
        "target_z_dyn_norm": float(entity_z_dyn[0].norm().detach().cpu().item()),
    }
    return coord_result, probe


def _aggregate_metric_rows(rows: List[Dict[str, Any]]) -> Dict[str, float]:
    if not rows:
        return {
            "overall_top1": 0.0,
            "hard_subset_top1": 0.0,
            "ambiguity_top1": 0.0,
            "appearance_change_top1": 0.0,
            "occlusion_reappearance_top1": 0.0,
            "long_gap_persistence_top1": 0.0,
            "MRR": 0.0,
            "top5_hit": 0.0,
        }
    def _subset_mean(tag: str) -> float:
        subset = [row for row in rows if tag in set(row.get("subset_tags", []))]
        return _mean(row.get("top1", 0.0) for row in subset) if subset else 0.0
    hard_rows = [row for row in rows if row.get("subset_tags")]
    return {
        "overall_top1": _mean(row.get("top1", 0.0) for row in rows),
        "hard_subset_top1": _mean(row.get("top1", 0.0) for row in hard_rows) if hard_rows else 0.0,
        "ambiguity_top1": _subset_mean("crossing_ambiguity"),
        "appearance_change_top1": _subset_mean("appearance_change"),
        "occlusion_reappearance_top1": _subset_mean("occlusion_reappearance"),
        "long_gap_persistence_top1": _subset_mean("long_gap_persistence"),
        "MRR": _mean(row.get("mrr", 0.0) for row in rows),
        "top5_hit": _mean(row.get("top5_hit", 0.0) for row in rows),
    }


def _bootstrap(values: List[float], seed: int = 0, n_boot: int = 4000) -> Dict[str, Any]:
    arr = np.asarray([float(v) for v in values], dtype=np.float64)
    if arr.size == 0:
        return {
            "count": 0,
            "mean_delta": 0.0,
            "ci95_low": 0.0,
            "ci95_high": 0.0,
            "zero_excluded": False,
            "bootstrap_win_rate": 0.0,
        }
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, int(arr.size), size=(n_boot, int(arr.size)))
    boot = arr[idx].mean(axis=1)
    low, high = np.percentile(boot, [2.5, 97.5]).tolist()
    return {
        "count": int(arr.size),
        "mean_delta": float(arr.mean()),
        "ci95_low": float(low),
        "ci95_high": float(high),
        "zero_excluded": bool(low > 0.0 or high < 0.0),
        "bootstrap_win_rate": float(np.mean(arr > 0.0)),
    }


def _build_hybrid_scores(
    coord_scores: Dict[str, float],
    unit_scores: Dict[str, float],
    semantic_scores: Dict[str, float],
    alpha: float,
    beta: float,
    gamma: float,
) -> Dict[str, float]:
    all_ids = sorted(set(coord_scores) | set(unit_scores) | set(semantic_scores))
    return {
        cid: float(alpha * coord_scores.get(cid, -1e9) + beta * unit_scores.get(cid, 0.0) + gamma * semantic_scores.get(cid, 0.0))
        for cid in all_ids
    }


def _aggregate_probe_rows(probe_rows: List[Dict[str, Any]], score_field: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in probe_rows:
        scores = row.get(score_field, {})
        if not isinstance(scores, dict):
            continue
        ranked = _sorted_rank_from_scores(scores, str(row.get("target_id", "")))
        out.append(
            {
                "protocol_item_id": str(row.get("protocol_item_id", "")),
                "seed": int(row.get("seed", -1)),
                "subset_tags": list(row.get("subset_tags", [])),
                "top1": float(ranked["top1"]),
                "top5_hit": float(ranked["top5_hit"]),
                "mrr": float(ranked["mrr"]),
                "top1_candidate_id": str(ranked["top1_candidate_id"]),
            }
        )
    return out


def _score_improvement(new_metrics: Dict[str, float], base_metrics: Dict[str, float]) -> bool:
    return bool(
        float(new_metrics.get("overall_top1", 0.0)) > float(base_metrics.get("overall_top1", 0.0))
        or float(new_metrics.get("hard_subset_top1", 0.0)) > float(base_metrics.get("hard_subset_top1", 0.0))
    )


def _representative_ids(rows: List[Dict[str, Any]], key: str) -> List[str]:
    freq: Counter = Counter(str(row.get("protocol_item_id", "")) for row in rows if str(row.get("category", "")) == key)
    return [item_id for item_id, _ in freq.most_common(20)]


def _semantic_heavy_condition(tags: List[str], top_wrong_cand: str, unit_scores: Dict[str, float], semantic_scores: Dict[str, float], target_id: str) -> bool:
    if "appearance_change" in set(tags):
        return True
    target_sem = float(semantic_scores.get(str(target_id), -1.0))
    wrong_sem = float(semantic_scores.get(str(top_wrong_cand), -1.0))
    return bool(target_sem > -0.5 and wrong_sem >= target_sem - 0.05)


def _confuser_collision_condition(
    top_wrong_cand: str,
    candidate_inputs: Dict[str, Any],
    tags: List[str],
) -> bool:
    cand = (candidate_inputs.get("candidates", {}) or {}).get(str(top_wrong_cand), {})
    return bool(
        float(cand.get("iou_with_target", 0.0)) > 0.0
        or float(cand.get("centroid_distance_to_target", 1.0)) < 0.10
        or "crossing_ambiguity" in set(tags)
    )


def _classify_rows(
    variant_name: str,
    flat_rows: List[Dict[str, Any]],
    probe_rows_by_key: Dict[Tuple[int, str], Dict[str, Any]],
    candidate_inputs_by_item: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    cat_rows: List[Dict[str, Any]] = []
    counts: Counter = Counter()
    subset_breakdown: Dict[str, Counter] = defaultdict(Counter)
    seed_breakdown: Dict[int, Counter] = defaultdict(Counter)
    tusb_fail_total = 0
    for row in flat_rows:
        methods = row.get("methods", {})
        tusb = methods.get(variant_name)
        legacy = methods.get(LEGACY)
        if not isinstance(tusb, dict) or not isinstance(legacy, dict):
            continue
        protocol_item_id = str(row.get("protocol_item_id", ""))
        seed = int(row.get("seed", -1))
        tags = list(row.get("subset_tags", []))
        target_id = str(row.get("target_id", ""))
        key = (seed, protocol_item_id)
        probe = probe_rows_by_key.get(key, {})
        category = "other"
        confuser_candidate_id = ""
        if float(tusb.get("query_future_top1_acc", 0.0)) > 0.5 and float(legacy.get("query_future_top1_acc", 0.0)) < 0.5 and (
            "occlusion_reappearance" in set(tags) or "long_gap_persistence" in set(tags)
        ):
            category = "tusb_win_continuity_case"
        elif float(legacy.get("query_future_top1_acc", 0.0)) > 0.5 and float(tusb.get("query_future_top1_acc", 0.0)) < 0.5:
            tusb_fail_total += 1
            top_wrong = str(tusb.get("top1_candidate_id", ""))
            confuser_candidate_id = top_wrong
            if str(probe.get("unit_identity_top1_candidate_id", "")) == target_id:
                category = "legacysem_win_tusb_unit_identity_correct_but_coord_wrong"
            elif _confuser_collision_condition(top_wrong, candidate_inputs_by_item.get(protocol_item_id, {}), tags):
                category = "legacysem_win_tusb_confuser_collision"
            elif _semantic_heavy_condition(tags, top_wrong, probe.get("unit_scores", {}), probe.get("semantic_scores", {}), target_id):
                category = "appearance_or_semantic_win_for_legacysem"
            elif float(tusb.get("query_future_localization_error", 1e9)) > float(legacy.get("query_future_localization_error", 1e9)):
                category = "legacysem_win_tusb_coord_far"
        counts[category] += 1
        seed_breakdown[seed][category] += 1
        for tag in tags:
            subset_breakdown[tag][category] += 1
        cat_rows.append(
            {
                "protocol_item_id": protocol_item_id,
                "seed": seed,
                "subset_tags": tags,
                "category": category,
                "target_id": target_id,
                "confuser_candidate_id": confuser_candidate_id,
            }
        )
    total = max(len(cat_rows), 1)
    ratios = {key: float(value / total) for key, value in sorted(counts.items())}
    summary = {
        "variant_name": variant_name,
        "category_counts": dict(sorted(counts.items())),
        "category_ratios": ratios,
        "per_subset_breakdown": {tag: dict(sorted(counter.items())) for tag, counter in sorted(subset_breakdown.items())},
        "per_seed_breakdown": {str(seed): dict(sorted(counter.items())) for seed, counter in sorted(seed_breakdown.items())},
        "top20_representative_item_ids": {
            key: _representative_ids(cat_rows, key)
            for key in [
                "legacysem_win_tusb_coord_far",
                "legacysem_win_tusb_confuser_collision",
                "legacysem_win_tusb_unit_identity_correct_but_coord_wrong",
                "tusb_win_continuity_case",
                "appearance_or_semantic_win_for_legacysem",
                "other",
            ]
        },
        "rows": cat_rows,
        "tusb_fail_total": int(tusb_fail_total),
    }
    return summary


def _dominant_failure_mode(
    variant_summary: Dict[str, Any],
    unit_state_contains_useful_signal: bool,
    hybrid_beats_legacysem: bool,
) -> str:
    counts = variant_summary.get("category_counts", {}) if isinstance(variant_summary.get("category_counts", {}), dict) else {}
    coord_far = int(counts.get("legacysem_win_tusb_coord_far", 0))
    confuser = int(counts.get("legacysem_win_tusb_confuser_collision", 0))
    readout = int(counts.get("legacysem_win_tusb_unit_identity_correct_but_coord_wrong", 0))
    semantic = int(counts.get("appearance_or_semantic_win_for_legacysem", 0))
    total_fail = max(int(variant_summary.get("tusb_fail_total", 0)), 1)
    if hybrid_beats_legacysem or readout / total_fail >= 0.30:
        return "readout_mismatch"
    if semantic / total_fail >= 0.35:
        return "semantic_teacher_weakness"
    if not unit_state_contains_useful_signal and (coord_far + confuser) / total_fail >= 0.55:
        return "representation_failure"
    return "mixed"


def build_reports(args: Any) -> Dict[str, Any]:
    checkpoint_map = _load_checkpoint_map(Path(args.main_checkpoint_audit), Path(args.sidecar_checkpoint_audit))
    prepared_items, skipped_items, skipped_reason_counts = _prepare_items(Path(args.protocol_json), max_items=int(args.max_items))
    candidate_inputs_by_item = {row["protocol_item_id"]: row["candidate_inputs"] for row in prepared_items}

    flat_rows: List[Dict[str, Any]] = []
    row_by_key: Dict[Tuple[int, str], Dict[str, Any]] = {}
    for seed in SEEDS:
        for prepared in prepared_items:
            item = prepared["item"]
            protocol_item_id = str(item.get("protocol_item_id", ""))
            row = {
                "protocol_item_id": protocol_item_id,
                "seed": int(seed),
                "dataset": str(item.get("dataset", "")),
                "clip_id": str(item.get("clip_id", "")),
                "subset_tags": list(item.get("subset_tags", [])),
                "target_id": str(item.get("target_id", "")),
                "split": _item_split(protocol_item_id),
                "methods": {},
            }
            flat_rows.append(row)
            row_by_key[(seed, protocol_item_id)] = row

    probe_rows_by_variant: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    device, device_info = prev._select_eval_device(args)
    eval_started = _now_iso()
    start_ts = datetime.now(timezone.utc).timestamp()
    try:
        for method_name in METHOD_ORDER:
            for seed in SEEDS:
                entry = checkpoint_map[method_name][seed]
                print(f"[{_now_iso()}] eval_start method={method_name} seed={seed}", flush=True)
                spec = prev.MethodSpec(
                    name=method_name,
                    run_name=str(entry["run_name"]),
                    method_type="stage2",
                    checkpoint_path=str(entry["checkpoint_path"]),
                )
                method = prev._load_method(spec, device=device)
                try:
                    for prepared in prepared_items:
                        item = prepared["item"]
                        key = (seed, str(item.get("protocol_item_id", "")))
                        row = row_by_key[key]
                        candidate_inputs = prepared["candidate_inputs"]
                        if method_name in {TUSB_BEST, TUSB_SIDECAR}:
                            coord_result, probe = _evaluate_tusb_item(method, prepared, device=device)
                            row["methods"][method_name] = coord_result
                            probe_rows_by_variant[method_name].append(
                                {
                                    "protocol_item_id": row["protocol_item_id"],
                                    "seed": seed,
                                    "subset_tags": list(row["subset_tags"]),
                                    "split": str(row["split"]),
                                    "target_id": str(row["target_id"]),
                                    "coord_scores": dict(probe.get("coord_scores", {})),
                                    "unit_identity_score": dict(probe.get("unit_scores", {})),
                                    "semantic_teacher_score": dict(probe.get("semantic_scores", {})),
                                    "unit_identity_top1_candidate_id": str(probe.get("unit_identity_top1_candidate_id", "")),
                                    "dominant_unit": int(probe.get("dominant_unit", -1)),
                                    "dominant_unit_mass": float(probe.get("dominant_unit_mass", 0.0)),
                                    "blocking_reason": str(probe.get("blocking_reason", "")),
                                }
                            )
                        else:
                            row["methods"][method_name] = prev._evaluate_item(
                                method=method,
                                item=item,
                                batch=prepared["batch"],
                                target_future_mask=candidate_inputs["target_future_mask"],
                                future_masks=candidate_inputs["future_masks"],
                                device=device,
                            )
                    print(f"[{_now_iso()}] eval_done method={method_name} seed={seed}", flush=True)
                finally:
                    prev._release_method(method)
    finally:
        lease_id = str(device_info.get("lease_id", "")).strip()
        if lease_id:
            try:
                prev.release_lease(lease_id=lease_id, lease_path=str(args.lease_path))
            except Exception:
                pass
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    eval_finished = _now_iso()
    wall_time = float(datetime.now(timezone.utc).timestamp() - start_ts)
    per_item_results_hash = _sha256_json(flat_rows)

    method_summary: Dict[str, Any] = {}
    for method_name in METHOD_ORDER:
        seed_rows: List[Dict[str, Any]] = []
        baseline_cmp = {CAL: [], CROP: [], LEGACY: []}
        for seed in SEEDS:
            rows = [row for row in flat_rows if int(row["seed"]) == int(seed) and isinstance(row["methods"].get(method_name), dict)]
            if not rows:
                continue
            scored = [row["methods"][method_name] for row in rows]
            subsets = lambda tag: [row["methods"][method_name] for row in rows if tag in set(row["subset_tags"])]
            seed_row = {
                "seed": int(seed),
                "overall_top1": _mean(r["query_future_top1_acc"] for r in scored),
                "hit_rate": _mean(r["query_future_hit_rate"] for r in scored),
                "localization_error": _mean(r["query_future_localization_error"] for r in scored),
                "mask_iou_at_top1": _mean(r["future_mask_iou_at_top1"] for r in scored),
                "hard_subset_top1": _mean(row["methods"][method_name]["query_future_top1_acc"] for row in rows if row["subset_tags"]),
                "ambiguity_top1": _mean(r["query_future_top1_acc"] for r in subsets("crossing_ambiguity")),
                "appearance_change_top1": _mean(r["query_future_top1_acc"] for r in subsets("appearance_change")),
                "occlusion_reappearance_top1": _mean(r["query_future_top1_acc"] for r in subsets("occlusion_reappearance")),
                "long_gap_persistence_top1": _mean(r["query_future_top1_acc"] for r in subsets("long_gap_persistence")),
                "small_object_top1": _mean(r["query_future_top1_acc"] for r in subsets("small_object")),
            }
            seed_rows.append(seed_row)
            for baseline_name in [CAL, CROP, LEGACY]:
                if method_name == baseline_name:
                    continue
                baseline_rows = [row for row in rows if isinstance(row["methods"].get(baseline_name), dict)]
                if not baseline_rows:
                    continue
                delta = _mean(
                    float(row["methods"][method_name]["query_future_top1_acc"]) - float(row["methods"][baseline_name]["query_future_top1_acc"])
                    for row in baseline_rows
                )
                baseline_cmp[baseline_name].append({"seed": int(seed), "delta_top1": float(delta)})
        mean_block = {
            key: _mean(row[key] for row in seed_rows)
            for key in [
                "overall_top1",
                "hit_rate",
                "localization_error",
                "mask_iou_at_top1",
                "hard_subset_top1",
                "ambiguity_top1",
                "appearance_change_top1",
                "occlusion_reappearance_top1",
                "long_gap_persistence_top1",
                "small_object_top1",
            ]
        }
        std_block = {
            key: _std(row[key] for row in seed_rows)
            for key in mean_block.keys()
        }
        method_summary[method_name] = {
            "seed_rows": seed_rows,
            "mean": mean_block,
            "std": std_block,
            "win_count_vs_calibration": int(sum(1 for row in baseline_cmp.get(CAL, []) if float(row["delta_top1"]) > 0.0)),
            "win_count_vs_cropenc": int(sum(1 for row in baseline_cmp.get(CROP, []) if float(row["delta_top1"]) > 0.0)),
            "win_count_vs_legacysem": int(sum(1 for row in baseline_cmp.get(LEGACY, []) if float(row["delta_top1"]) > 0.0)),
            "seedwise_delta_vs_legacysem": baseline_cmp.get(LEGACY, []),
        }

    val_test_results: Dict[str, Any] = {}
    best_hybrid_test_metrics: Dict[str, Dict[str, float]] = {}
    unit_signal = False
    hybrid_beats_legacysem = False
    legacy_test_metrics = _aggregate_metric_rows(
        [
            {
                "protocol_item_id": row["protocol_item_id"],
                "seed": row["seed"],
                "subset_tags": row["subset_tags"],
                "top1": float((row.get("methods", {}).get(LEGACY, {}) or {}).get("query_future_top1_acc", 0.0)),
                "top5_hit": float((row.get("methods", {}).get(LEGACY, {}) or {}).get("top5_hit", 0.0)),
                "mrr": float((row.get("methods", {}).get(LEGACY, {}) or {}).get("mrr", 0.0)),
            }
            for row in flat_rows
            if str(row.get("split")) == "test" and isinstance((row.get("methods", {}) or {}).get(LEGACY), dict)
        ]
    )
    for variant_name in [TUSB_BEST, TUSB_SIDECAR]:
        variant_rows = [row for row in probe_rows_by_variant[variant_name] if not str(row.get("blocking_reason", ""))]
        val_rows = [row for row in variant_rows if str(row.get("split")) == "val"]
        test_rows = [row for row in variant_rows if str(row.get("split")) == "test"]

        coord_val_metrics = _aggregate_metric_rows(_aggregate_probe_rows(val_rows, "coord_scores"))
        coord_test_metrics = _aggregate_metric_rows(_aggregate_probe_rows(test_rows, "coord_scores"))
        unit_val_metrics = _aggregate_metric_rows(_aggregate_probe_rows(val_rows, "unit_identity_score"))
        unit_test_metrics = _aggregate_metric_rows(_aggregate_probe_rows(test_rows, "unit_identity_score"))

        grid_rows: List[Dict[str, Any]] = []
        best_combo = {"alpha": 0.9, "beta": 0.1, "gamma": 0.0, "score": -1e9}
        for alpha in [0.5, 0.7, 0.9]:
            for beta in [0.1, 0.2, 0.4]:
                for gamma in [0.0, 0.1, 0.2]:
                    val_probe_rows = []
                    for row in val_rows:
                        hybrid = _build_hybrid_scores(
                            row["coord_scores"],
                            row["unit_identity_score"],
                            row["semantic_teacher_score"],
                            alpha=alpha,
                            beta=beta,
                            gamma=gamma,
                        )
                        val_probe_rows.append(
                            {
                                "protocol_item_id": row["protocol_item_id"],
                                "seed": row["seed"],
                                "subset_tags": row["subset_tags"],
                                "target_id": row["target_id"],
                                "hybrid_score": hybrid,
                            }
                        )
                    metrics = _aggregate_metric_rows(_aggregate_probe_rows(val_probe_rows, "hybrid_score"))
                    selection_score = float(metrics["overall_top1"] + 0.5 * metrics["hard_subset_top1"] + 0.1 * metrics["MRR"])
                    grid_rows.append(
                        {
                            "alpha": alpha,
                            "beta": beta,
                            "gamma": gamma,
                            "val_metrics": metrics,
                            "selection_score": selection_score,
                        }
                    )
                    if selection_score > float(best_combo["score"]):
                        best_combo = {"alpha": alpha, "beta": beta, "gamma": gamma, "score": selection_score}
        test_probe_rows = []
        for row in test_rows:
            hybrid = _build_hybrid_scores(
                row["coord_scores"],
                row["unit_identity_score"],
                row["semantic_teacher_score"],
                alpha=float(best_combo["alpha"]),
                beta=float(best_combo["beta"]),
                gamma=float(best_combo["gamma"]),
            )
            test_probe_rows.append(
                {
                    "protocol_item_id": row["protocol_item_id"],
                    "seed": row["seed"],
                    "subset_tags": row["subset_tags"],
                    "target_id": row["target_id"],
                    "hybrid_score": hybrid,
                }
            )
        hybrid_test_metrics = _aggregate_metric_rows(_aggregate_probe_rows(test_probe_rows, "hybrid_score"))
        best_hybrid_test_metrics[variant_name] = hybrid_test_metrics
        if _score_improvement(unit_test_metrics, coord_test_metrics) or _score_improvement(hybrid_test_metrics, coord_test_metrics):
            unit_signal = True
        if float(hybrid_test_metrics["overall_top1"]) > float(legacy_test_metrics["overall_top1"]):
            hybrid_beats_legacysem = True
        val_test_results[variant_name] = {
            "split_counts": {
                "val": int(len(val_rows)),
                "test": int(len(test_rows)),
            },
            "coord_only": {
                "val": coord_val_metrics,
                "test": coord_test_metrics,
            },
            "unit_identity_only": {
                "val": unit_val_metrics,
                "test": unit_test_metrics,
            },
            "hybrid_grid": grid_rows,
            "selected_hybrid_weights": {
                "alpha": float(best_combo["alpha"]),
                "beta": float(best_combo["beta"]),
                "gamma": float(best_combo["gamma"]),
            },
            "hybrid_score": {
                "test": hybrid_test_metrics,
            },
        }

    bestpt_variant_summary = _classify_rows(
        TUSB_BEST,
        flat_rows,
        {
            (int(row["seed"]), str(row["protocol_item_id"])): row
            for row in probe_rows_by_variant[TUSB_BEST]
        },
        candidate_inputs_by_item,
    )
    sidecar_variant_summary = _classify_rows(
        TUSB_SIDECAR,
        flat_rows,
        {
            (int(row["seed"]), str(row["protocol_item_id"])): row
            for row in probe_rows_by_variant[TUSB_SIDECAR]
        },
        candidate_inputs_by_item,
    )

    dominant_failure = _dominant_failure_mode(bestpt_variant_summary, unit_signal, hybrid_beats_legacysem)
    if hybrid_beats_legacysem:
        recommended_next_step = "integrate_unit_state_reranker_as_light_readout"
    elif unit_signal:
        recommended_next_step = "improve_semantic_teacher"
    elif float(method_summary[TUSB_BEST]["mean"]["overall_top1"]) > float(method_summary[CAL]["mean"]["overall_top1"]):
        recommended_next_step = "add_killer_baselines"
    else:
        recommended_next_step = "stop_stage2_escalation_and_write_moderate_claim"

    winmode_report = {
        "generated_at_utc": _now_iso(),
        "panel_name": PANEL_NAME,
        "eval_started_at": eval_started,
        "eval_finished_at": eval_finished,
        "wall_time_seconds": wall_time,
        "seed_count": len(SEEDS),
        "valid_items": int(len(prepared_items)),
        "skipped_items": int(len(skipped_items)),
        "skipped_reason_counts": dict(sorted(skipped_reason_counts.items())),
        "per_item_results_hash": per_item_results_hash,
        "per_method_seed_results": method_summary,
        "variants": {
            TUSB_BEST: {k: v for k, v in bestpt_variant_summary.items() if k != "rows"},
            TUSB_SIDECAR: {k: v for k, v in sidecar_variant_summary.items() if k != "rows"},
        },
        "legacysem_beats_tusb_because_bestpt": dominant_failure,
    }
    reranker_report = {
        "generated_at_utc": _now_iso(),
        "panel_name": PANEL_NAME,
        "valid_items": int(len(prepared_items)),
        "skipped_items": int(len(skipped_items)),
        "per_item_results_hash": per_item_results_hash,
        "method_coord_summary": {
            LEGACY: method_summary[LEGACY]["mean"],
            CAL: method_summary[CAL]["mean"],
            TUSB_BEST: method_summary[TUSB_BEST]["mean"],
            TUSB_SIDECAR: method_summary[TUSB_SIDECAR]["mean"],
        },
        "heldout_probe_results": val_test_results,
        "unit_state_contains_useful_signal": bool(unit_signal),
        "hybrid_beats_legacysem": bool(hybrid_beats_legacysem),
    }
    diagnosis_report = {
        "generated_at_utc": _now_iso(),
        "panel_name": PANEL_NAME,
        "legacysem_beats_tusb_because": dominant_failure,
        "unit_state_contains_useful_signal": bool(unit_signal),
        "hybrid_beats_legacysem": bool(hybrid_beats_legacysem),
        "recommended_next_step": recommended_next_step,
        "bestpt_mean_overall_top1": float(method_summary[TUSB_BEST]["mean"]["overall_top1"]),
        "sidecar_mean_overall_top1": float(method_summary[TUSB_SIDECAR]["mean"]["overall_top1"]),
        "legacysem_mean_overall_top1": float(method_summary[LEGACY]["mean"]["overall_top1"]),
        "exact_blocking_reason": "" if len(prepared_items) > 0 else "no_valid_items_after_context_preserving_build",
    }

    _write_json(Path(args.output_winmode_json), winmode_report)
    _write_md(
        Path(args.output_winmode_md),
        [
            "# STWM LegacySem Win-Mode Audit 20260422",
            "",
            f"- panel_name: {PANEL_NAME}",
            f"- valid_items: {winmode_report['valid_items']}",
            f"- skipped_items: {winmode_report['skipped_items']}",
            f"- per_item_results_hash: {per_item_results_hash}",
            f"- dominant_failure_mode_bestpt: {dominant_failure}",
        ],
    )
    _write_json(Path(args.output_reranker_json), reranker_report)
    _write_md(
        Path(args.output_reranker_md),
        [
            "# STWM TUSB Unit-State Reranker Probe 20260422",
            "",
            f"- panel_name: {PANEL_NAME}",
            f"- unit_state_contains_useful_signal: {reranker_report['unit_state_contains_useful_signal']}",
            f"- hybrid_beats_legacysem: {reranker_report['hybrid_beats_legacysem']}",
        ],
    )
    _write_json(Path(args.output_diagnosis_json), diagnosis_report)
    _write_md(
        Path(args.output_diagnosis_md),
        [
            "# STWM TUSB vs LegacySem Failure Diagnosis 20260422",
            "",
            f"- legacysem_beats_tusb_because: {diagnosis_report['legacysem_beats_tusb_because']}",
            f"- unit_state_contains_useful_signal: {diagnosis_report['unit_state_contains_useful_signal']}",
            f"- hybrid_beats_legacysem: {diagnosis_report['hybrid_beats_legacysem']}",
            f"- recommended_next_step: {diagnosis_report['recommended_next_step']}",
        ],
    )
    return {
        "winmode": winmode_report,
        "reranker": reranker_report,
        "diagnosis": diagnosis_report,
    }


def parse_args() -> Any:
    parser = ArgumentParser(description="Run STWM legacysem win-mode audit and TUSB unit-state reranker probe.")
    parser.add_argument("--protocol-json", default=str(ROOT / "reports/stage2_state_identifiability_protocol_v3_20260416.json"))
    parser.add_argument("--main-checkpoint-audit", default=str(ROOT / "reports/stwm_postfix_matched6seed_checkpoint_audit_20260421.json"))
    parser.add_argument("--sidecar-checkpoint-audit", default=str(ROOT / "reports/stwm_sidecar_checkpoint_audit_20260422.json"))
    parser.add_argument("--output-winmode-json", default=str(ROOT / "reports/stwm_legacysem_winmode_audit_20260422.json"))
    parser.add_argument("--output-winmode-md", default=str(ROOT / "docs/STWM_LEGACYSEM_WINMODE_AUDIT_20260422.md"))
    parser.add_argument("--output-reranker-json", default=str(ROOT / "reports/stwm_tusb_unit_state_reranker_probe_20260422.json"))
    parser.add_argument("--output-reranker-md", default=str(ROOT / "docs/STWM_TUSB_UNIT_STATE_RERANKER_PROBE_20260422.md"))
    parser.add_argument("--output-diagnosis-json", default=str(ROOT / "reports/stwm_tusb_vs_legacysem_failure_diagnosis_20260422.json"))
    parser.add_argument("--output-diagnosis-md", default=str(ROOT / "docs/STWM_TUSB_VS_LEGACYSEM_FAILURE_DIAGNOSIS_20260422.md"))
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--lease-path", default=str(ROOT / "reports/stage1_v2_gpu_lease_20260408.json"))
    parser.add_argument("--eval-required-mem-gb", type=float, default=40.0)
    parser.add_argument("--eval-safety-margin-gb", type=float, default=8.0)
    parser.add_argument("--max-items", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    _apply_process_title_normalization()
    args = parse_args()
    build_reports(args)


if __name__ == "__main__":
    main()
