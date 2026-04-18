#!/usr/bin/env python3
from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
from typing import Any, Dict, List, Tuple
import gc
import json

import numpy as np
import torch

from stwm.tools import run_stage2_state_identifiability_eval_20260415 as prev


ROOT = prev.ROOT


def _select_eval_device_v3(args: Any) -> Tuple[torch.device, Dict[str, Any]]:
    requested = str(args.device).strip().lower()
    if requested == "cpu":
        return torch.device("cpu"), {"mode": "forced_cpu", "selected_gpu_id": -1, "lease_id": ""}
    if requested == "cuda":
        if not torch.cuda.is_available():
            return torch.device("cpu"), {"mode": "forced_cuda_fallback_cpu", "selected_gpu_id": -1, "lease_id": ""}
        return torch.device("cuda:0"), {"mode": "forced_cuda0", "selected_gpu_id": 0, "lease_id": ""}
    if requested.startswith("cuda:"):
        if not torch.cuda.is_available():
            return torch.device("cpu"), {"mode": "forced_explicit_cuda_fallback_cpu", "selected_gpu_id": -1, "lease_id": ""}
        gpu_id = int(requested.split(":", 1)[1])
        if gpu_id < 0 or gpu_id >= torch.cuda.device_count():
            raise RuntimeError(f"requested cuda device out of range: {requested}")
        return torch.device(f"cuda:{gpu_id}"), {"mode": "forced_explicit_cuda", "selected_gpu_id": gpu_id, "lease_id": ""}
    if requested != "auto":
        raise RuntimeError(f"unsupported device mode: {requested}")
    if not torch.cuda.is_available():
        return torch.device("cpu"), {"mode": "auto_cpu_no_cuda", "selected_gpu_id": -1, "lease_id": ""}
    selector = prev.select_single_gpu(
        required_mem_gb=float(args.eval_required_mem_gb),
        safety_margin_gb=float(args.eval_safety_margin_gb),
        sample_count=3,
        interval_sec=0.5,
        lease_path=str(args.lease_path),
    )
    gpu_id = int(selector.get("selected_gpu_id", -1))
    if gpu_id < 0:
        return torch.device("cpu"), {"mode": "auto_cpu_no_fit", "selected_gpu_id": -1, "lease_id": "", "selector_payload": selector}
    lease = prev.acquire_lease(
        gpu_id=gpu_id,
        owner="stage2_state_identifiability_eval_v3_20260416",
        ttl_seconds=6 * 3600,
        lease_path=str(args.lease_path),
        allow_shared=False,
    )
    return torch.device(f"cuda:{gpu_id}"), {
        "mode": "auto_selected_gpu_explicit",
        "selected_gpu_id": gpu_id,
        "lease_id": str(lease.get("lease_id", "")),
        "selector_payload": selector,
    }


def _load_method_specs(args: Any) -> List[prev.MethodSpec]:
    final_diag = prev.read_json(args.final_utility_diagnosis)
    final_summary = prev.read_json(args.final_utility_summary)
    mechanism_summary = prev.read_json(args.mechanism_summary)
    calibration_run = str(final_diag.get("overall_best_run_name", "stage2_calonly_topk1_seed123_longconfirm_v2_20260414"))

    final_rows = final_summary.get("run_rows", []) if isinstance(final_summary.get("run_rows", []), list) else []
    mech_rows = mechanism_summary.get("run_rows", []) if isinstance(mechanism_summary.get("run_rows", []), list) else []

    def metric_triplet(row: Dict[str, Any]) -> Tuple[float, float, float]:
        metrics = ((row.get("best_checkpoint_metric") or {}).get("metrics") or {}) if isinstance(row, dict) else {}
        return (
            float(metrics.get("free_rollout_endpoint_l2", 1e9)),
            float(metrics.get("free_rollout_coord_mean_l2", 1e9)),
            float(metrics.get("teacher_forced_coord_loss", 1e9)),
        )

    def pick_failure(family: str, fallback: str) -> str:
        rows = [r for r in mech_rows if str(r.get("ablation_name", "")) == family and str(r.get("status", "")).lower() == "completed"]
        if not rows:
            return fallback
        return max(rows, key=metric_triplet).get("run_name", fallback)

    methods: List[prev.MethodSpec] = [
        prev.MethodSpec(
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
        ("noalign_failure", pick_failure("noalign", "stage2_calonly_noalign_seed123_ablate_fix_20260415")),
        ("densegate_failure", pick_failure("densegate", "stage2_calonly_densegate_seed456_ablate_fix_20260415")),
        ("nodelay_failure", pick_failure("nodelay", "stage2_calonly_nodelay_seed456_ablate_fix_20260415")),
    ]
    for method_name, run_name in named_ckpts:
        ckpt = ROOT / "outputs/checkpoints" / run_name / "best.pt"
        if ckpt.exists():
            methods.append(
                prev.MethodSpec(
                    name=method_name,
                    run_name=run_name,
                    method_type="stage2",
                    checkpoint_path=str(ckpt),
                )
            )
    return methods


def _metric_rows(per_item: List[Dict[str, Any]], calibration_name: str, comparator_name: str, metric_key: str, higher_better: bool, subset_tag: str | None = None) -> List[float]:
    diffs: List[float] = []
    for item in per_item:
        tags = list(item.get("subset_tags", []))
        if subset_tag == "__hard__" and not tags:
            continue
        if subset_tag and subset_tag != "__hard__" and subset_tag not in tags:
            continue
        methods = item.get("methods", {}) if isinstance(item.get("methods", {}), dict) else {}
        cal = methods.get(calibration_name)
        comp = methods.get(comparator_name)
        if not isinstance(cal, dict) or not isinstance(comp, dict):
            continue
        a = float(cal.get(metric_key, 0.0 if higher_better else 1e9))
        b = float(comp.get(metric_key, 0.0 if higher_better else 1e9))
        diffs.append((a - b) if higher_better else (b - a))
    return diffs


def _bootstrap_summary(diffs: List[float], seed: int = 0, n_boot: int = 4000) -> Dict[str, Any]:
    if not diffs:
        return {
            "count": 0,
            "mean_diff": 0.0,
            "win_rate": 0.0,
            "loss_rate": 0.0,
            "tie_rate": 0.0,
            "ci95_low": 0.0,
            "ci95_high": 0.0,
            "ci95_width": 0.0,
            "significant_positive": False,
        }
    arr = np.asarray(diffs, dtype=np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(arr), size=(n_boot, len(arr)))
    boot = arr[idx].mean(axis=1)
    low, high = np.percentile(boot, [2.5, 97.5]).tolist()
    return {
        "count": int(len(arr)),
        "mean_diff": float(arr.mean()),
        "win_rate": float(np.mean(arr > 0)),
        "loss_rate": float(np.mean(arr < 0)),
        "tie_rate": float(np.mean(arr == 0)),
        "ci95_low": float(low),
        "ci95_high": float(high),
        "ci95_width": float(high - low),
        "significant_positive": bool(low > 0.0),
    }


def _paired_comparison_bundle(per_item: List[Dict[str, Any]], calibration_name: str, comparator_name: str) -> Dict[str, Any]:
    return {
        "top1_acc": _bootstrap_summary(_metric_rows(per_item, calibration_name, comparator_name, "query_future_top1_acc", True), seed=7),
        "hit_rate": _bootstrap_summary(_metric_rows(per_item, calibration_name, comparator_name, "query_future_hit_rate", True), seed=11),
        "localization_error": _bootstrap_summary(_metric_rows(per_item, calibration_name, comparator_name, "query_future_localization_error", False), seed=13),
        "future_mask_iou_at_top1": _bootstrap_summary(_metric_rows(per_item, calibration_name, comparator_name, "future_mask_iou_at_top1", True), seed=17),
        "hard_top1_acc": _bootstrap_summary(_metric_rows(per_item, calibration_name, comparator_name, "query_future_top1_acc", True, subset_tag="__hard__"), seed=19),
    }


def _count_significant_top1(comparisons: Dict[str, Any], comparators: List[str]) -> Tuple[int, float]:
    count = 0
    widths: List[float] = []
    for comp in comparators:
        top1 = (((comparisons.get(comp) or {}).get("top1_acc")) or {})
        if not isinstance(top1, dict):
            continue
        if bool(top1.get("significant_positive", False)):
            count += 1
        widths.append(float(top1.get("ci95_width", 1e9)))
    return count, float(sum(widths) / max(len(widths), 1))


def parse_args() -> Any:
    parser = ArgumentParser(description="Run benchmark-scale Stage2 state-identifiability / future grounding evaluation v3")
    parser.add_argument("--protocol-json", default=str(ROOT / "reports/stage2_state_identifiability_protocol_v3_20260416.json"))
    parser.add_argument("--output-json", default=str(ROOT / "reports/stage2_state_identifiability_eval_v3_20260416.json"))
    parser.add_argument("--output-md", default=str(ROOT / "docs/STAGE2_STATE_IDENTIFIABILITY_EVAL_V3_20260416.md"))
    parser.add_argument("--stage1-checkpoint", default=str(ROOT / "outputs/checkpoints/stage1_v2_longtrain_220m_mainline_20260408/best.pt"))
    parser.add_argument("--final-utility-summary", default=str(ROOT / "reports/stage2_final_utility_closure_v2_summary_20260414.json"))
    parser.add_argument("--final-utility-diagnosis", default=str(ROOT / "reports/stage2_final_utility_closure_v2_diagnosis_20260414.json"))
    parser.add_argument("--mechanism-summary", default=str(ROOT / "reports/stage2_mechanism_ablation_fix_v2_summary_20260416.json"))
    parser.add_argument("--v2-eval-json", default=str(ROOT / "reports/stage2_state_identifiability_eval_v2_20260416.json"))
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--lease-path", default=str(ROOT / "reports/stage1_v2_gpu_lease_20260408.json"))
    parser.add_argument("--eval-required-mem-gb", type=float, default=40.0)
    parser.add_argument("--eval-safety-margin-gb", type=float, default=8.0)
    return parser.parse_args()


def _build_single_item_batch_v3(
    item: Dict[str, Any],
    temporal_window: int = 5,
    target_id: str | int | None = None,
) -> Tuple[Dict[str, Any], np.ndarray, Dict[str, np.ndarray]]:
    frame_paths = [Path(x) for x in item.get("selected_frame_paths", [])]
    target_masks, sizes, future_masks, target_future_mask = prev._extract_entity_masks(item, entity_id=target_id)
    boxes: List[np.ndarray] = []
    present: List[bool] = []
    last_box = None
    query_step = int(item.get("query_step", 0))
    with prev.Image.open(frame_paths[query_step]) as img:
        query_rgb = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    query_mask = target_masks[query_step]
    if query_mask is None:
        raise RuntimeError(f"query target mask missing for {item.get('protocol_item_id')}")
    sem_box, sem_mask_used, sem_fg_ratio = prev._box_from_mask_or_center(
        mask=query_mask.astype(np.uint8),
        width=int(query_rgb.shape[1]),
        height=int(query_rgb.shape[0]),
        radius=12,
    )
    for (width, height), mask in zip(sizes, target_masks):
        if mask is not None and np.any(mask):
            box, _, _ = prev._box_from_mask_or_center(mask.astype(np.uint8), width=width, height=height, radius=12)
            last_box = box
            present.append(True)
            boxes.append(box)
        else:
            fallback = last_box if last_box is not None else prev._box_from_mask_or_center(None, width=width, height=height, radius=12)[0]
            boxes.append(np.asarray(fallback, dtype=np.float32))
            present.append(False)
    state = prev._build_state_from_boxes(boxes=boxes, sizes=sizes)
    obs_state = state[:prev.OBS_LEN, None, :]
    fut_state = state[prev.OBS_LEN:, None, :]
    obs_valid = np.asarray(present[:prev.OBS_LEN], dtype=bool)[:, None]
    fut_valid = np.asarray(present[prev.OBS_LEN:], dtype=bool)[:, None]
    semantic_rgb_crop, semantic_mask_crop, mask_crop_available = prev._build_semantic_crops(
        rgb=query_rgb,
        mask=query_mask.astype(np.uint8),
        box_xyxy=sem_box,
        crop_size=64,
    )
    semantic_feat = prev._semantic_feature(
        rgb=query_rgb,
        mask=query_mask.astype(np.uint8),
        box_xyxy=sem_box,
        mask_used=bool(sem_mask_used),
        fg_ratio=float(sem_fg_ratio),
    )
    temporal_rgb_crops: List[np.ndarray] = []
    temporal_mask_crops: List[np.ndarray] = []
    temporal_valid_flags: List[bool] = []
    temporal_window = max(int(temporal_window), 1)
    for obs_idx in range(min(prev.OBS_LEN, temporal_window, len(frame_paths))):
        with prev.Image.open(frame_paths[obs_idx]) as img:
            rgb_t = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
        mask_t = target_masks[obs_idx]
        box_t = boxes[obs_idx]
        rgb_crop_t, mask_crop_t, mask_valid_t = prev._build_semantic_crops(
            rgb=rgb_t,
            mask=mask_t.astype(np.uint8) if isinstance(mask_t, np.ndarray) else None,
            box_xyxy=box_t,
            crop_size=64,
        )
        temporal_rgb_crops.append(rgb_crop_t)
        temporal_mask_crops.append(mask_crop_t)
        temporal_valid_flags.append(bool(mask_valid_t))
    while len(temporal_rgb_crops) < temporal_window:
        temporal_rgb_crops.append(np.zeros_like(semantic_rgb_crop))
        temporal_mask_crops.append(np.zeros_like(semantic_mask_crop))
        temporal_valid_flags.append(False)
    temporal_rgb = np.stack(temporal_rgb_crops[:temporal_window], axis=0).astype(np.float32)
    temporal_mask = np.stack(temporal_mask_crops[:temporal_window], axis=0).astype(np.float32)
    temporal_valid = np.asarray(temporal_valid_flags[:temporal_window], dtype=bool)
    semantic_instance_id_map = query_mask.astype(np.int64)[None, ...]
    semantic_instance_id_crop = (semantic_mask_crop > 0.5).astype(np.int64)[None, ...]
    semantic_instance_id_temporal = (temporal_mask > 0.5).astype(np.int64)
    semantic_instance_valid = temporal_valid[None, ...]
    semantic_objectness_score = np.asarray([max(float(sem_fg_ratio), 0.0)], dtype=np.float32)
    semantic_teacher_prior = np.zeros((1, 512), dtype=np.float32)
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
        "semantic_rgb_crop_temporal": torch.from_numpy(temporal_rgb[None, ...]).to(torch.float32),
        "semantic_mask_crop_temporal": torch.from_numpy(temporal_mask[None, ...]).to(torch.float32),
        "semantic_temporal_valid": torch.from_numpy(temporal_valid[None, ...]).to(torch.bool),
        "semantic_instance_id_map": torch.from_numpy(semantic_instance_id_map).to(torch.long),
        "semantic_instance_id_crop": torch.from_numpy(semantic_instance_id_crop[None, ...]).to(torch.long),
        "semantic_instance_id_temporal": torch.from_numpy(semantic_instance_id_temporal[None, ...]).to(torch.long),
        "semantic_instance_valid": torch.from_numpy(semantic_instance_valid).to(torch.bool),
        "semantic_objectness_score": torch.from_numpy(semantic_objectness_score).to(torch.float32),
        "semantic_teacher_prior": torch.from_numpy(semantic_teacher_prior).to(torch.float32),
        "semantic_frame_path": str(frame_paths[query_step]),
        "semantic_mask_path": "",
        "semantic_source_mode": "object_region_or_mask_crop_visual_state",
        "current_mainline_semantic_source": "crop_visual_encoder",
        "legacy_semantic_source": "hand_crafted_stats",
        "semantic_source_summary": {
            "mask_crop_used_tokens": int(1 if sem_mask_used else 0),
            "region_crop_used_tokens": int(0 if sem_mask_used else 1),
            "mask_available": True,
            "semantic_crop_size": 64,
            "current_mainline_semantic_source": "crop_visual_encoder",
            "legacy_semantic_source": "hand_crafted_stats",
            "legacy_semantic_feature_dim": 10,
            "temporal_window": int(temporal_window),
        },
    }
    return prev.stage2_semantic_collate_fn([sample]), target_future_mask, future_masks


def _build_context_preserving_item_batch_v3(
    item: Dict[str, Any],
    temporal_window: int = 5,
    max_context_entities: int = 8,
) -> Tuple[Dict[str, Any], np.ndarray, Dict[str, np.ndarray]]:
    frame_paths = [Path(x) for x in item.get("selected_frame_paths", [])]
    entity_ids = prev._protocol_observed_context_candidate_ids(item, max_context_entities=max_context_entities)
    if not entity_ids:
        return _build_single_item_batch_v3(item=item, temporal_window=temporal_window)

    target_id = str(item.get("target_id", ""))
    if str(entity_ids[0]) != target_id:
        entity_ids = [target_id] + [str(x) for x in entity_ids if str(x) != target_id]
    entity_ids = [str(x) for x in entity_ids[: max(int(max_context_entities), 1)]]

    query_step = int(item.get("query_step", 0))
    temporal_window = max(int(temporal_window), 1)
    query_rgbs: Dict[int, np.ndarray] = {}
    for step in range(min(prev.OBS_LEN, temporal_window, len(frame_paths))):
        with prev.Image.open(frame_paths[step]) as img:
            query_rgbs[step] = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    if query_step not in query_rgbs:
        with prev.Image.open(frame_paths[query_step]) as img:
            query_rgbs[query_step] = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0

    obs_states: List[np.ndarray] = []
    fut_states: List[np.ndarray] = []
    obs_valids: List[np.ndarray] = []
    fut_valids: List[np.ndarray] = []
    semantic_features: List[np.ndarray] = []
    semantic_boxes: List[np.ndarray] = []
    semantic_masks: List[bool] = []
    semantic_rgb_crops: List[np.ndarray] = []
    semantic_mask_crops: List[np.ndarray] = []
    semantic_crop_valids: List[bool] = []
    semantic_mask_crop_valids: List[bool] = []
    semantic_rgb_crop_temporals: List[np.ndarray] = []
    semantic_mask_crop_temporals: List[np.ndarray] = []
    semantic_temporal_valids: List[np.ndarray] = []
    semantic_instance_id_crops: List[np.ndarray] = []
    semantic_instance_id_temporals: List[np.ndarray] = []
    semantic_instance_valids: List[np.ndarray] = []
    semantic_objectness_scores: List[np.ndarray] = []
    semantic_teacher_priors: List[np.ndarray] = []
    entity_boxes_over_time: List[np.ndarray] = []
    entity_masks_over_time: List[List[np.ndarray | None]] = []
    query_instance_id_map: np.ndarray | None = None

    target_future_mask: np.ndarray | None = None
    future_masks: Dict[str, np.ndarray] = {}
    context_entity_indices: List[int] = []

    for entity_idx, entity_id in enumerate(entity_ids):
        target_masks, sizes, future_masks_local, target_future_mask_local = prev._extract_entity_masks(item, entity_id=entity_id)
        if entity_idx == 0:
            target_future_mask = target_future_mask_local
            future_masks = future_masks_local

        boxes: List[np.ndarray] = []
        present: List[bool] = []
        last_box = None
        for (width, height), mask in zip(sizes, target_masks):
            if mask is not None and np.any(mask):
                box, _, _ = prev._box_from_mask_or_center(mask.astype(np.uint8), width=width, height=height, radius=12)
                last_box = box
                present.append(True)
                boxes.append(box)
            else:
                fallback = last_box if last_box is not None else prev._box_from_mask_or_center(None, width=width, height=height, radius=12)[0]
                boxes.append(np.asarray(fallback, dtype=np.float32))
                present.append(False)
        state = prev._build_state_from_boxes(boxes=boxes, sizes=sizes)
        obs_states.append(state[:prev.OBS_LEN])
        fut_states.append(state[prev.OBS_LEN:])
        obs_valids.append(np.asarray(present[:prev.OBS_LEN], dtype=bool))
        fut_valids.append(np.asarray(present[prev.OBS_LEN:], dtype=bool))

        semantic_step = query_step
        if target_masks[semantic_step] is None:
            for alt_step in range(min(prev.OBS_LEN, len(target_masks))):
                if target_masks[alt_step] is not None and np.any(target_masks[alt_step]):
                    semantic_step = alt_step
                    break
        rgb_ref = query_rgbs.get(semantic_step)
        if rgb_ref is None:
            with prev.Image.open(frame_paths[semantic_step]) as img:
                rgb_ref = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
                query_rgbs[semantic_step] = rgb_ref
        mask_ref = target_masks[semantic_step]
        box_ref = boxes[semantic_step]
        sem_box, sem_mask_used, sem_fg_ratio = prev._box_from_mask_or_center(
            mask=mask_ref.astype(np.uint8) if isinstance(mask_ref, np.ndarray) else None,
            width=int(rgb_ref.shape[1]),
            height=int(rgb_ref.shape[0]),
            radius=12,
        )
        rgb_crop, mask_crop, mask_crop_available = prev._build_semantic_crops(
            rgb=rgb_ref,
            mask=mask_ref.astype(np.uint8) if isinstance(mask_ref, np.ndarray) else None,
            box_xyxy=sem_box,
            crop_size=64,
        )
        semantic_rgb_crops.append(rgb_crop)
        semantic_mask_crops.append(mask_crop)
        semantic_crop_valids.append(True)
        semantic_mask_crop_valids.append(bool(mask_crop_available))
        semantic_features.append(
            prev._semantic_feature(
                rgb=rgb_ref,
                mask=mask_ref.astype(np.uint8) if isinstance(mask_ref, np.ndarray) else None,
                box_xyxy=sem_box,
                mask_used=bool(sem_mask_used),
                fg_ratio=float(sem_fg_ratio),
            )
        )
        semantic_boxes.append(np.asarray(sem_box, dtype=np.float32))
        semantic_masks.append(True)
        semantic_objectness_scores.append(np.asarray([max(float(sem_fg_ratio), 0.0)], dtype=np.float32))
        semantic_teacher_priors.append(np.zeros((512,), dtype=np.float32))
        if entity_idx > 0:
            context_entity_indices.append(int(entity_idx))

        temporal_rgb_crops: List[np.ndarray] = []
        temporal_mask_crops: List[np.ndarray] = []
        temporal_valid_flags: List[bool] = []
        instance_temporal: List[np.ndarray] = []
        for obs_idx in range(min(prev.OBS_LEN, temporal_window, len(frame_paths))):
            rgb_t = query_rgbs.get(obs_idx)
            if rgb_t is None:
                with prev.Image.open(frame_paths[obs_idx]) as img:
                    rgb_t = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
                    query_rgbs[obs_idx] = rgb_t
            mask_t = target_masks[obs_idx]
            box_t = boxes[obs_idx]
            rgb_crop_t, mask_crop_t, mask_valid_t = prev._build_semantic_crops(
                rgb=rgb_t,
                mask=mask_t.astype(np.uint8) if isinstance(mask_t, np.ndarray) else None,
                box_xyxy=box_t,
                crop_size=64,
            )
            temporal_rgb_crops.append(rgb_crop_t)
            temporal_mask_crops.append(mask_crop_t)
            temporal_valid_flags.append(bool(mask_valid_t))
            instance_temporal.append((mask_crop_t > 0.5).astype(np.int64))
        while len(temporal_rgb_crops) < temporal_window:
            temporal_rgb_crops.append(np.zeros_like(semantic_rgb_crops[-1]))
            temporal_mask_crops.append(np.zeros_like(semantic_mask_crops[-1]))
            temporal_valid_flags.append(False)
            instance_temporal.append(np.zeros_like(semantic_mask_crops[-1], dtype=np.int64))
        semantic_rgb_crop_temporals.append(np.stack(temporal_rgb_crops[:temporal_window], axis=0).astype(np.float32))
        semantic_mask_crop_temporals.append(np.stack(temporal_mask_crops[:temporal_window], axis=0).astype(np.float32))
        semantic_temporal_valids.append(np.asarray(temporal_valid_flags[:temporal_window], dtype=bool))
        semantic_instance_id_crops.append((semantic_mask_crops[-1] > 0.5).astype(np.int64))
        semantic_instance_id_temporals.append(np.stack(instance_temporal[:temporal_window], axis=0).astype(np.int64))
        semantic_instance_valids.append(np.asarray(temporal_valid_flags[:temporal_window], dtype=bool))
        entity_boxes_over_time.append(np.stack(boxes, axis=0).astype(np.float32))
        entity_masks_over_time.append(target_masks)

    if target_future_mask is None:
        raise RuntimeError(f"context-preserving eval failed to resolve target future mask for {item.get('protocol_item_id')}")

    sample = {
        "obs_state": torch.from_numpy(np.stack(obs_states, axis=1)).to(torch.float32),
        "fut_state": torch.from_numpy(np.stack(fut_states, axis=1)).to(torch.float32),
        "obs_valid": torch.from_numpy(np.stack(obs_valids, axis=1)).to(torch.bool),
        "fut_valid": torch.from_numpy(np.stack(fut_valids, axis=1)).to(torch.bool),
        "point_ids": torch.arange(len(entity_ids), dtype=torch.long),
        "meta": {
            "dataset": str(item.get("dataset", "")),
            "clip_id": str(item.get("clip_id", "")),
            "target_id": str(item.get("target_id", "")),
            "subset_tags": list(item.get("subset_tags", [])),
            "target_entity_index": 0,
            "context_entity_indices": context_entity_indices,
            "protocol_eval_context_entity_count": int(len(entity_ids)),
            "protocol_eval_mode": "context_preserving",
        },
        "semantic_features": torch.from_numpy(np.stack(semantic_features, axis=0)).to(torch.float32),
        "semantic_boxes": torch.from_numpy(np.stack(semantic_boxes, axis=0)).to(torch.float32),
        "semantic_mask": torch.tensor(semantic_masks, dtype=torch.bool),
        "semantic_rgb_crop": torch.from_numpy(np.stack(semantic_rgb_crops, axis=0)).to(torch.float32),
        "semantic_mask_crop": torch.from_numpy(np.stack(semantic_mask_crops, axis=0)).to(torch.float32),
        "semantic_crop_valid": torch.tensor(semantic_crop_valids, dtype=torch.bool),
        "semantic_mask_crop_valid": torch.tensor(semantic_mask_crop_valids, dtype=torch.bool),
        "semantic_rgb_crop_temporal": torch.from_numpy(np.stack(semantic_rgb_crop_temporals, axis=0)).to(torch.float32),
        "semantic_mask_crop_temporal": torch.from_numpy(np.stack(semantic_mask_crop_temporals, axis=0)).to(torch.float32),
        "semantic_temporal_valid": torch.from_numpy(np.stack(semantic_temporal_valids, axis=0)).to(torch.bool),
        "semantic_instance_id_map": torch.zeros((1, 1), dtype=torch.long) if query_instance_id_map is None else torch.from_numpy(query_instance_id_map).to(torch.long),
        "semantic_instance_id_crop": torch.from_numpy(np.stack(semantic_instance_id_crops, axis=0)).to(torch.long),
        "semantic_instance_id_temporal": torch.from_numpy(np.stack(semantic_instance_id_temporals, axis=0)).to(torch.long),
        "semantic_instance_valid": torch.from_numpy(np.stack(semantic_instance_valids, axis=0)).to(torch.bool),
        "semantic_objectness_score": torch.from_numpy(np.concatenate(semantic_objectness_scores, axis=0)).to(torch.float32),
        "semantic_teacher_prior": torch.from_numpy(np.stack(semantic_teacher_priors, axis=0)).to(torch.float32),
        "semantic_frame_path": str(frame_paths[query_step]),
        "semantic_mask_path": "",
        "entity_boxes_over_time": torch.from_numpy(np.stack(entity_boxes_over_time, axis=0)).to(torch.float32),
        "entity_masks_over_time": entity_masks_over_time,
        "semantic_source_mode": "object_region_or_mask_crop_visual_state",
        "current_mainline_semantic_source": "crop_visual_encoder",
        "legacy_semantic_source": "hand_crafted_stats",
        "semantic_source_summary": {
            "mask_available": True,
            "semantic_crop_size": 64,
            "temporal_window": int(temporal_window),
            "protocol_eval_context_entity_count": int(len(entity_ids)),
        },
    }
    return prev.stage2_semantic_collate_fn([sample]), target_future_mask, future_masks


def main() -> None:
    args = parse_args()
    protocol = prev.read_json(args.protocol_json)
    items = protocol.get("items", []) if isinstance(protocol.get("items", []), list) else []
    device, device_info = _select_eval_device_v3(args)
    specs = _load_method_specs(args)

    prepared_items: List[Tuple[Dict[str, Any], Dict[str, Any], np.ndarray, Dict[str, np.ndarray]]] = []
    per_item: List[Dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        batch, target_future_mask, future_masks = _build_single_item_batch_v3(item)
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
            method = prev._load_method(spec, device=device)
            for item_row, prepared in zip(per_item, prepared_items):
                item, batch, target_future_mask, future_masks = prepared
                item_row["methods"][method.name] = prev._evaluate_item(
                    method=method,
                    item=item,
                    batch=batch,
                    target_future_mask=target_future_mask,
                    future_masks=future_masks,
                    device=device,
                )
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
        all_rows: List[Dict[str, Any]] = []
        hard_rows: List[Dict[str, Any]] = []
        for item_row in per_item:
            score = (item_row.get("methods") or {}).get(spec.name)
            if not isinstance(score, dict):
                continue
            all_rows.append(score)
            if item_row.get("subset_tags"):
                hard_rows.append(score)
        panel_metrics["full_identifiability_panel"] = prev._aggregate_item_metrics(all_rows)
        panel_metrics["hard_subsets"] = prev._aggregate_item_metrics(hard_rows)
        for panel in panel_names[1:]:
            subset_rows = [
                (item_row.get("methods") or {}).get(spec.name)
                for item_row in per_item
                if panel in list(item_row.get("subset_tags", []))
            ]
            panel_metrics[panel] = prev._aggregate_item_metrics([r for r in subset_rows if isinstance(r, dict)])
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
        calibration and stage1 and legacysem and cropenc and
        float(calibration.get("hard_subset_top1_acc", -1.0)) >= max(
            float(stage1.get("hard_subset_top1_acc", -1.0)),
            float(legacysem.get("hard_subset_top1_acc", -1.0)),
            float(cropenc.get("hard_subset_top1_acc", -1.0)),
        )
    )
    protocol_success = bool(
        items
        and int((protocol.get("panel_counts") or {}).get("full_identifiability_panel", 0)) >= 120
        and int((protocol.get("panel_counts") or {}).get("crossing_ambiguity", 0)) > 0
        and int((protocol.get("panel_counts") or {}).get("small_object", 0)) > 0
        and int((protocol.get("panel_counts") or {}).get("appearance_change", 0)) > 0
        and int((protocol.get("panel_counts") or {}).get("long_gap_persistence", 0)) > 0
    )

    comparisons = {
        "stage1_frozen_baseline": _paired_comparison_bundle(per_item, "calibration_only_mainline_best", "stage1_frozen_baseline"),
        "legacysem_best": _paired_comparison_bundle(per_item, "calibration_only_mainline_best", "legacysem_best"),
        "cropenc_baseline_best": _paired_comparison_bundle(per_item, "calibration_only_mainline_best", "cropenc_baseline_best"),
        "noalign_failure": _paired_comparison_bundle(per_item, "calibration_only_mainline_best", "noalign_failure"),
        "densegate_failure": _paired_comparison_bundle(per_item, "calibration_only_mainline_best", "densegate_failure"),
        "nodelay_failure": _paired_comparison_bundle(per_item, "calibration_only_mainline_best", "nodelay_failure"),
    }

    v2_eval = prev.read_json(args.v2_eval_json)
    v2_comparisons = v2_eval.get("paired_bootstrap_comparisons", {}) if isinstance(v2_eval.get("paired_bootstrap_comparisons", {}), dict) else {}
    baseline_names = ["stage1_frozen_baseline", "legacysem_best", "cropenc_baseline_best"]
    v3_sig_count, v3_avg_width = _count_significant_top1(comparisons, baseline_names)
    v2_sig_count, v2_avg_width = _count_significant_top1(v2_comparisons, baseline_names)
    v3_discriminative = bool(
        int(len(per_item)) >= 120
        and int(len(per_item)) > int(v2_eval.get("protocol_item_count", 0))
        and v3_sig_count >= v2_sig_count
        and (v3_avg_width < v2_avg_width or v3_sig_count > v2_sig_count or v2_avg_width <= 0.0)
    )
    discriminative_for_top_tier = bool(
        protocol_success
        and v3_discriminative
        and comparisons["stage1_frozen_baseline"]["top1_acc"]["significant_positive"]
        and comparisons["legacysem_best"]["top1_acc"]["significant_positive"]
        and comparisons["cropenc_baseline_best"]["top1_acc"]["significant_positive"]
    )

    payload = {
        "generated_at_utc": prev.now_iso(),
        "benchmark_scope": "real state-identifiability / future grounding with true instance continuity and future masks",
        "official_benchmark": False,
        "protocol_contribution": True,
        "selected_device": str(device),
        "device_info": device_info,
        "protocol_item_count": int(len(per_item)),
        "panel_counts": dict(protocol.get("panel_counts", {})),
        "methods": method_rows,
        "per_item_results": per_item,
        "paired_bootstrap_comparisons": comparisons,
        "state_identifiability_protocol_v3_success": bool(protocol_success),
        "state_identifiability_protocol_success": bool(protocol_success),
        "future_grounding_usefulness_improved_vs_stage1": bool(improved_stage1),
        "future_grounding_usefulness_improved_vs_legacysem": bool(improved_legacysem),
        "future_grounding_usefulness_improved_vs_cropenc": bool(improved_cropenc),
        "future_grounding_usefulness_improved_vs_baselines": bool(improved_all),
        "future_grounding_usefulness_improved_on_hard_subsets": bool(improved_hard),
        "protocol_v3_statistically_more_discriminative_than_v2": bool(v3_discriminative),
        "protocol_v3_discriminative_enough_for_top_tier": bool(discriminative_for_top_tier),
    }
    prev.write_json(args.output_json, payload)
    lines = [
        "# Stage2 State-Identifiability Eval V3 20260416",
        "",
        "- scope: real future grounding with true instance identity / future mask continuity",
        "- official_benchmark: False",
        "- protocol_contribution: True",
        f"- protocol_item_count: {len(per_item)}",
        f"- selected_device: {device}",
        f"- state_identifiability_protocol_v3_success: {protocol_success}",
        f"- future_grounding_usefulness_improved_vs_baselines: {improved_all}",
        f"- future_grounding_usefulness_improved_on_hard_subsets: {improved_hard}",
        f"- protocol_v3_statistically_more_discriminative_than_v2: {v3_discriminative}",
        f"- protocol_v3_discriminative_enough_for_top_tier: {discriminative_for_top_tier}",
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
    lines.extend(
        [
            "",
            "## Paired Bootstrap Comparisons",
            "",
            "| comparator | top1_mean_diff | top1_ci95 | top1_win_rate | locerr_mean_diff | locerr_ci95 |",
            "|---|---:|---|---:|---:|---|",
        ]
    )
    for comp_name, block in comparisons.items():
        top1 = block["top1_acc"]
        loc = block["localization_error"]
        lines.append(
            f"| {comp_name} | {top1['mean_diff']:.4f} | [{top1['ci95_low']:.4f}, {top1['ci95_high']:.4f}] | "
            f"{top1['win_rate']:.4f} | {loc['mean_diff']:.6f} | [{loc['ci95_low']:.6f}, {loc['ci95_high']:.6f}] |"
        )
    prev.write_md(args.output_md, lines)
    print(json.dumps(payload, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
