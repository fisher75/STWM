#!/usr/bin/env python3
from __future__ import annotations

from argparse import ArgumentParser
from collections import Counter
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
_MEASUREMENT_PROJECTION_CACHE: dict[tuple[str, int, int], torch.Tensor] = {}


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


def _bootstrap_repo_imports(repo_root: Path) -> None:
    code_dir = repo_root / "code"
    if str(code_dir) not in sys.path:
        sys.path.insert(0, str(code_dir))


def _import_visibility_builder(repo_root: Path):
    _bootstrap_repo_imports(repo_root)
    from stwm.tracewm_v2_stage2.utils.visibility_reappearance_targets import build_future_visibility_reappearance_targets

    return build_future_visibility_reappearance_targets


def _import_external_query_builder(repo_root: Path):
    _bootstrap_repo_imports(repo_root)
    from stwm.tools.external_hardcase_query_batch_builder_20260428 import (
        build_item_candidate_measurement_cache,
        build_candidate_measurement_features,
        build_query_batch_from_item,
        group_candidate_records,
    )

    return (
        build_query_batch_from_item,
        group_candidate_records,
        build_candidate_measurement_features,
        build_item_candidate_measurement_cache,
    )


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
        f"- manifest_mode: `{payload.get('manifest_mode')}`",
        f"- current_export_data_source: `{payload.get('current_export_data_source')}`",
        f"- external_manifest_consumed: `{payload.get('external_manifest_consumed')}`",
        f"- stage2_val_fallback_used: `{payload.get('stage2_val_fallback_used')}`",
        f"- future_candidate_used_as_input: `{payload.get('future_candidate_used_as_input')}`",
        f"- candidate_score_mode: `{payload.get('candidate_score_mode')}`",
        f"- candidate_feature_used_for_scoring: `{payload.get('candidate_feature_used_for_scoring')}`",
        f"- candidate_feature_used_for_rollout: `{payload.get('candidate_feature_used_for_rollout')}`",
        f"- raw_export_schema_version: `{payload.get('raw_export_schema_version')}`",
        f"- checkpoint_path: `{payload.get('checkpoint_path')}`",
        f"- checkpoint_loaded: `{payload.get('checkpoint_loaded')}`",
        f"- enable_future_semantic_state_head: `{payload.get('enable_future_semantic_state_head')}`",
        f"- free_rollout_used: `{payload.get('free_rollout_used')}`",
        f"- old_association_report_used: `{payload.get('old_association_report_used')}`",
        f"- total_items: `{payload.get('total_items')}`",
        f"- valid_items: `{payload.get('valid_items')}`",
        f"- valid_ratio: `{payload.get('valid_ratio')}`",
        f"- future_reappearance_head_available: `{payload.get('future_reappearance_head_available')}`",
        f"- future_reappearance_event_head_available: `{payload.get('future_reappearance_event_head_available')}`",
        f"- reappearance_prob_source: `{payload.get('reappearance_prob_source')}`",
        f"- reappearance_head_weights_random_init: `{payload.get('reappearance_head_weights_random_init')}`",
        f"- future_reappearance_mask_policy: `{payload.get('future_reappearance_mask_policy')}`",
        f"- future_reappearance_positive_rate_at_risk: `{payload.get('future_reappearance_positive_rate_at_risk')}`",
        f"- future_reappearance_event_positive_rate: `{payload.get('future_reappearance_event_positive_rate')}`",
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


def _candidate_center(candidate: dict[str, Any], image_size: Any) -> list[float] | None:
    bbox = _as_bbox(candidate.get("bbox")) if isinstance(candidate, dict) else None
    if bbox is None:
        return None
    width = height = 1.0
    if isinstance(image_size, list) and len(image_size) >= 2:
        try:
            width = max(float(image_size[0]), 1.0)
            height = max(float(image_size[1]), 1.0)
        except Exception:
            width = height = 1.0
    elif isinstance(image_size, dict):
        try:
            width = max(float(image_size.get("width", 1.0)), 1.0)
            height = max(float(image_size.get("height", 1.0)), 1.0)
        except Exception:
            width = height = 1.0
    x1, y1, x2, y2 = bbox
    return [
        max(0.0, min(1.0, ((x1 + x2) * 0.5) / width)),
        max(0.0, min(1.0, ((y1 + y2) * 0.5) / height)),
    ]


def _project_measurement_feature(feature_vector: list[float], dim: int, namespace: str) -> torch.Tensor:
    vec = torch.tensor(feature_vector, dtype=torch.float32)
    if vec.numel() == 0:
        vec = torch.zeros((1,), dtype=torch.float32)
    vec = (vec - vec.mean()) / (vec.std(unbiased=False) + 1e-6)
    key = (str(namespace), int(dim), int(vec.numel()))
    mat = _MEASUREMENT_PROJECTION_CACHE.get(key)
    if mat is None:
        seed = stable_seed(namespace)
        rows = torch.arange(int(dim), dtype=torch.float32).view(-1, 1) + 1.0
        cols = torch.arange(int(vec.numel()), dtype=torch.float32).view(1, -1) + 1.0
        phase = float(seed % 997) / 997.0
        mat = torch.sin(rows * cols * 0.173 + phase) + torch.cos(rows * cols * 0.071 + phase * 3.0)
        _MEASUREMENT_PROJECTION_CACHE[key] = mat
    projected = mat @ vec
    return F.normalize(projected, dim=0)


def _cosine01(lhs: torch.Tensor, rhs: torch.Tensor) -> float:
    lhs = F.normalize(lhs.detach().float().cpu(), dim=0)
    rhs = F.normalize(rhs.detach().float().cpu(), dim=0)
    return float(((torch.dot(lhs, rhs).clamp(-1.0, 1.0) + 1.0) * 0.5).item())


def _score_external_candidates(
    *,
    state: Any,
    item: dict[str, Any],
    candidate_score_mode: str = "posterior_v1",
    candidate_feature_builder: Any | None = None,
    candidate_measurement_cache: dict[str, Any] | None = None,
    candidate_measurement_feature_mode: str = "weak_rgb_bbox_stats",
    posterior_v4_weights: dict[str, float] | None = None,
) -> dict[str, Any]:
    import math

    candidates = item.get("future_candidates") if isinstance(item.get("future_candidates"), list) else []
    labels = [int(x) for x in item.get("candidate_labels", [])] if isinstance(item.get("candidate_labels"), list) else []
    image_size = item.get("image_size")
    pred = state.future_trace_coord[0, -1, 0, :2].detach().cpu()
    sem_vec = state.future_semantic_embedding[0, :, 0].detach().mean(dim=0).cpu()
    ident_vec = state.future_identity_embedding[0, :, 0].detach().mean(dim=0).cpu()
    visibility_prob = float(torch.sigmoid(state.future_visibility_logit[0, :, 0]).mean().detach().cpu().item())
    if getattr(state, "future_reappearance_event_logit", None) is not None:
        reappearance_event_prob = float(torch.sigmoid(state.future_reappearance_event_logit[0, 0]).detach().cpu().item())
    elif getattr(state, "future_reappearance_logit", None) is not None:
        reappearance_event_prob = float(torch.sigmoid(state.future_reappearance_logit[0, :, 0]).mean().detach().cpu().item())
    else:
        reappearance_event_prob = visibility_prob
    scores: list[float] = []
    distance_scores: list[float] = []
    semantic_scores: list[float] = []
    identity_scores: list[float] = []
    target_appearance_scores: list[float] = []
    predicted_semantic_scores: list[float] = []
    predicted_identity_scores: list[float] = []
    prior_scores: list[float] = []
    posterior_scores: list[float] = []
    posterior_v4_scores: list[float] = []
    distances: list[float | None] = []
    candidate_features: list[dict[str, Any]] = []
    feature_sources: list[str] = []
    mode = str(candidate_score_mode)
    weights = {
        "distance": 0.30,
        "target_appearance": 0.25,
        "predicted_semantic": 0.20,
        "predicted_identity": 0.15,
        "priors": 0.10,
    }
    if isinstance(posterior_v4_weights, dict):
        weights.update({k: float(v) for k, v in posterior_v4_weights.items() if k in weights})
    feature_needed = mode in {
        "semantic_only",
        "identity_only",
        "posterior_v1",
        "posterior_no_distance",
        "posterior_no_semantic",
        "weak_posterior_v3",
        "target_candidate_appearance_only",
        "predicted_semantic_to_candidate",
        "predicted_identity_to_candidate",
        "predicted_semantic_identity_to_candidate",
        "posterior_v4",
        "posterior_v4_no_distance",
        "posterior_v4_no_semantic_identity",
        "posterior_v4_no_target_candidate_appearance",
    }
    for cand in candidates:
        center = _candidate_center(cand if isinstance(cand, dict) else {}, image_size)
        if center is None:
            distances.append(None)
            distance_scores.append(0.0)
            semantic_scores.append(0.5)
            identity_scores.append(0.5)
            target_appearance_scores.append(0.5)
            predicted_semantic_scores.append(0.5)
            predicted_identity_scores.append(0.5)
            prior_scores.append(0.5)
            posterior_scores.append(0.0)
            posterior_v4_scores.append(0.0)
            scores.append(0.0)
            continue
        dx = float(pred[0].item()) - float(center[0])
        dy = float(pred[1].item()) - float(center[1])
        dist = math.sqrt(dx * dx + dy * dy)
        dist_score = math.exp(-8.0 * dist)
        prior_score = (0.5 + 0.5 * reappearance_event_prob) * (0.5 + 0.5 * visibility_prob)
        feature: dict[str, Any] = {
            "feature_source": "candidate_measurement_unavailable",
            "feature_vector": [],
            "observed_target_feature_vector": [],
            "candidate_feature_used_for_scoring": False,
            "candidate_feature_used_for_rollout": False,
        }
        cid = str(cand.get("candidate_id")) if isinstance(cand, dict) else ""
        if feature_needed and isinstance(candidate_measurement_cache, dict):
            feature = (
                candidate_measurement_cache.get("candidate_features", {}).get(cid)
                if isinstance(candidate_measurement_cache.get("candidate_features"), dict)
                else None
            ) or feature
        elif feature_needed and candidate_feature_builder is not None:
            feature = candidate_feature_builder(
                item,
                cand if isinstance(cand, dict) else {},
                item.get("frame_paths") if isinstance(item.get("frame_paths"), list) else [],
                item.get("future_frame_index"),
            )
        feature_vector = feature.get("feature_vector") if isinstance(feature.get("feature_vector"), list) else []
        observed_feature_vector = (
            feature.get("observed_target_feature_vector")
            if isinstance(feature.get("observed_target_feature_vector"), list)
            else []
        )
        if feature_vector:
            if len(feature_vector) == int(sem_vec.numel()):
                sem_measure = torch.tensor([float(x) for x in feature_vector], dtype=torch.float32)
            else:
                sem_measure = _project_measurement_feature([float(x) for x in feature_vector], int(sem_vec.numel()), "semantic_measurement_v4")
            if len(feature_vector) == int(ident_vec.numel()):
                id_measure = torch.tensor([float(x) for x in feature_vector], dtype=torch.float32)
            else:
                id_measure = _project_measurement_feature([float(x) for x in feature_vector], int(ident_vec.numel()), "identity_measurement_v4")
            semantic_score = _cosine01(sem_vec, sem_measure)
            identity_score = _cosine01(ident_vec, id_measure)
            feature["candidate_feature_used_for_scoring"] = True
        else:
            semantic_score = 0.5
            identity_score = 0.5
        if feature_vector and observed_feature_vector and len(feature_vector) == len(observed_feature_vector):
            target_appearance_score = _cosine01(
                torch.tensor([float(x) for x in observed_feature_vector], dtype=torch.float32),
                torch.tensor([float(x) for x in feature_vector], dtype=torch.float32),
            )
        else:
            target_appearance_score = 0.5
        posterior_score = (
            0.45 * dist_score
            + 0.20 * semantic_score
            + 0.20 * identity_score
            + 0.15 * prior_score
        )
        posterior_v4_score = (
            weights["distance"] * dist_score
            + weights["target_appearance"] * target_appearance_score
            + weights["predicted_semantic"] * semantic_score
            + weights["predicted_identity"] * identity_score
            + weights["priors"] * prior_score
        )
        if mode == "distance_only":
            score = dist_score
        elif mode == "weak_posterior_v3":
            score = posterior_score
        elif mode == "priors_only":
            score = prior_score
        elif mode == "semantic_only":
            score = semantic_score
        elif mode == "identity_only":
            score = identity_score
        elif mode == "target_candidate_appearance_only":
            score = target_appearance_score
        elif mode == "predicted_semantic_to_candidate":
            score = semantic_score
        elif mode == "predicted_identity_to_candidate":
            score = identity_score
        elif mode == "predicted_semantic_identity_to_candidate":
            score = 0.5 * semantic_score + 0.5 * identity_score
        elif mode == "posterior_v4":
            score = posterior_v4_score
        elif mode == "posterior_v4_no_distance":
            denom = max(weights["target_appearance"] + weights["predicted_semantic"] + weights["predicted_identity"] + weights["priors"], 1e-6)
            score = (
                weights["target_appearance"] * target_appearance_score
                + weights["predicted_semantic"] * semantic_score
                + weights["predicted_identity"] * identity_score
                + weights["priors"] * prior_score
            ) / denom
        elif mode == "posterior_v4_no_semantic_identity":
            denom = max(weights["distance"] + weights["target_appearance"] + weights["priors"], 1e-6)
            score = (
                weights["distance"] * dist_score
                + weights["target_appearance"] * target_appearance_score
                + weights["priors"] * prior_score
            ) / denom
        elif mode == "posterior_v4_no_target_candidate_appearance":
            denom = max(weights["distance"] + weights["predicted_semantic"] + weights["predicted_identity"] + weights["priors"], 1e-6)
            score = (
                weights["distance"] * dist_score
                + weights["predicted_semantic"] * semantic_score
                + weights["predicted_identity"] * identity_score
                + weights["priors"] * prior_score
            ) / denom
        elif mode == "posterior_no_distance":
            score = 0.40 * semantic_score + 0.40 * identity_score + 0.20 * prior_score
        elif mode == "posterior_no_semantic":
            score = 0.55 * dist_score + 0.30 * identity_score + 0.15 * prior_score
        else:
            score = posterior_score
        distances.append(dist)
        distance_scores.append(dist_score)
        semantic_scores.append(semantic_score)
        identity_scores.append(identity_score)
        target_appearance_scores.append(target_appearance_score)
        predicted_semantic_scores.append(semantic_score)
        predicted_identity_scores.append(identity_score)
        prior_scores.append(prior_score)
        posterior_scores.append(posterior_score)
        posterior_v4_scores.append(posterior_v4_score)
        scores.append(score)
        candidate_features.append(feature)
        feature_sources.append(str(feature.get("feature_source") or "unknown"))
    predicted_idx = int(max(range(len(scores)), key=lambda i: scores[i])) if scores else None
    rr = None
    if labels and scores and 1 in labels:
        gt_idx = labels.index(1)
        order = sorted(range(len(scores)), key=lambda i: float(scores[i]), reverse=True)
        rr = 1.0 / float(order.index(gt_idx) + 1) if gt_idx in order else 0.0
    score_tensor = torch.tensor(scores, dtype=torch.float32) if scores else torch.zeros((0,), dtype=torch.float32)
    all_equal = bool(score_tensor.numel() > 1 and float(score_tensor.std(unbiased=False).item()) <= 1e-12)
    components_by_mode = {
        "distance_only": ["future_trace_coord_distance"],
        "priors_only": ["future_reappearance_event_prob", "future_visibility_prob"],
        "semantic_only": ["future_semantic_embedding", "candidate_measurement_feature", "weak_semantic_measurement"],
        "identity_only": ["future_identity_embedding", "candidate_measurement_feature", "weak_identity_proxy"],
        "posterior_no_distance": [
            "future_semantic_embedding",
            "future_identity_embedding",
            "candidate_measurement_feature",
            "future_reappearance_event_prob",
            "future_visibility_prob",
        ],
        "posterior_no_semantic": [
            "future_trace_coord_distance",
            "future_identity_embedding",
            "candidate_measurement_feature",
            "future_reappearance_event_prob",
            "future_visibility_prob",
        ],
        "posterior_v1": [
            "future_trace_coord_distance",
            "future_semantic_embedding",
            "future_identity_embedding",
            "candidate_measurement_feature",
            "future_reappearance_event_prob",
            "future_visibility_prob",
        ],
        "weak_posterior_v3": [
            "future_trace_coord_distance",
            "future_semantic_embedding",
            "future_identity_embedding",
            "weak_candidate_measurement_feature",
            "future_reappearance_event_prob",
            "future_visibility_prob",
        ],
        "target_candidate_appearance_only": ["observed_target_crop_feature", "future_candidate_crop_feature"],
        "predicted_semantic_to_candidate": ["future_semantic_embedding", "future_candidate_crop_feature"],
        "predicted_identity_to_candidate": ["future_identity_embedding", "future_candidate_crop_feature"],
        "predicted_semantic_identity_to_candidate": [
            "future_semantic_embedding",
            "future_identity_embedding",
            "future_candidate_crop_feature",
        ],
        "posterior_v4": [
            "future_trace_coord_distance",
            "future_visibility_prob",
            "future_reappearance_event_prob",
            "observed_target_crop_feature",
            "future_semantic_embedding",
            "future_identity_embedding",
            "future_candidate_crop_feature",
        ],
        "posterior_v4_no_distance": [
            "future_visibility_prob",
            "future_reappearance_event_prob",
            "observed_target_crop_feature",
            "future_semantic_embedding",
            "future_identity_embedding",
            "future_candidate_crop_feature",
        ],
        "posterior_v4_no_semantic_identity": [
            "future_trace_coord_distance",
            "future_visibility_prob",
            "future_reappearance_event_prob",
            "observed_target_crop_feature",
        ],
        "posterior_v4_no_target_candidate_appearance": [
            "future_trace_coord_distance",
            "future_visibility_prob",
            "future_reappearance_event_prob",
            "future_semantic_embedding",
            "future_identity_embedding",
            "future_candidate_crop_feature",
        ],
    }
    return {
        "candidate_scores": scores,
        "candidate_distance_scores": distance_scores,
        "candidate_semantic_scores": semantic_scores,
        "candidate_identity_scores": identity_scores,
        "candidate_target_appearance_scores": target_appearance_scores,
        "candidate_predicted_semantic_scores": predicted_semantic_scores,
        "candidate_predicted_identity_scores": predicted_identity_scores,
        "candidate_prior_scores": prior_scores,
        "candidate_posterior_scores": posterior_scores,
        "candidate_posterior_v4_scores": posterior_v4_scores,
        "candidate_center_distances": distances,
        "candidate_measurement_features": candidate_features,
        "candidate_feature_source": "mixed:" + ",".join(sorted(set(feature_sources))) if feature_sources else "candidate_measurement_unavailable",
        "candidate_measurement_feature_mode": str(candidate_measurement_feature_mode),
        "candidate_feature_used_for_scoring": bool(any(bool(f.get("candidate_feature_used_for_scoring")) for f in candidate_features)),
        "candidate_feature_used_for_rollout": False,
        "candidate_labels": labels,
        "candidate_score_mode": str(candidate_score_mode),
        "all_candidate_score_equal": all_equal,
        "candidate_score_std": float(score_tensor.std(unbiased=False).item()) if score_tensor.numel() else None,
        "predicted_candidate_index": predicted_idx,
        "candidate_top1_correct": bool(labels and predicted_idx is not None and labels[predicted_idx] == 1),
        "candidate_mrr": rr,
        "future_visibility_prob_for_score": visibility_prob,
        "future_reappearance_event_prob_for_score": reappearance_event_prob,
        "score_components_used": components_by_mode.get(str(candidate_score_mode), components_by_mode["posterior_v1"]),
    }


def _external_query_item_from_state(
    *,
    item: dict[str, Any],
    state: Any,
    builder_meta: dict[str, Any],
    candidate_score_mode: str = "posterior_v1",
    candidate_feature_builder: Any | None = None,
    candidate_measurement_cache: dict[str, Any] | None = None,
    candidate_measurement_feature_mode: str = "weak_rgb_bbox_stats",
    posterior_v4_weights: dict[str, float] | None = None,
    feedback_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    validation = state.validate(strict=False)
    visibility_prob = torch.sigmoid(state.future_visibility_logit)
    uncertainty = F.softplus(state.future_uncertainty)
    sem = state.future_semantic_embedding
    ident = state.future_identity_embedding
    sem_norm = sem.norm(dim=-1)
    ident_norm = ident.norm(dim=-1)
    scoring = _score_external_candidates(
        state=state,
        item=item,
        candidate_score_mode=str(candidate_score_mode),
        candidate_feature_builder=candidate_feature_builder,
        candidate_measurement_cache=candidate_measurement_cache,
        candidate_measurement_feature_mode=str(candidate_measurement_feature_mode),
        posterior_v4_weights=posterior_v4_weights,
    )
    out = {
        "item_id": item.get("item_id"),
        "protocol_item_id": item.get("item_id"),
        "subset_tags": item.get("subset_tags") if isinstance(item.get("subset_tags"), dict) else {},
        "valid_output": bool(validation.get("valid", False)),
        "future_semantic_trace_state_valid": bool(validation.get("valid", False)),
        "validation_errors": validation.get("errors", []),
        "target_type": "external_candidate_expanded",
        "target_quality": "external_candidate_expanded",
        "future_candidate_used_as_input": False,
        "candidate_used_for_eval_scoring": True,
        "future_candidate_count": len(item.get("future_candidates") or []),
        "positive_candidate_count": int(sum(scoring.get("candidate_labels") or [])),
        "negative_candidate_count": int(len(scoring.get("candidate_labels") or []) - sum(scoring.get("candidate_labels") or [])),
        "future_trace_coord_shape": list(state.future_trace_coord.shape),
        "future_trace_coord_mean": _tensor_stats(state.future_trace_coord)["mean"],
        "future_trace_coord_std": _tensor_stats(state.future_trace_coord)["std"],
        "future_trace_coord_min": _tensor_stats(state.future_trace_coord)["min"],
        "future_trace_coord_max": _tensor_stats(state.future_trace_coord)["max"],
        "future_visibility_prob_shape": list(visibility_prob.shape),
        "future_visibility_prob_mean": _tensor_stats(visibility_prob)["mean"],
        "future_visibility_prob_std": _tensor_stats(visibility_prob)["std"],
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
        "semantic_state_feedback_enabled": bool((feedback_info or {}).get("semantic_state_feedback_enabled", False)),
        "feedback_gate_mean": (feedback_info or {}).get("feedback_gate_mean", 0.0),
        "feedback_gate_std": (feedback_info or {}).get("feedback_gate_std", 0.0),
        "feedback_gate_saturation_ratio": (feedback_info or {}).get("feedback_gate_saturation_ratio", 0.0),
        "feedback_delta_norm": (feedback_info or {}).get("feedback_delta_norm", 0.0),
    }
    out.update(scoring)
    out.update(builder_meta)
    return out


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


def _reset_reappearance_heads(head: torch.nn.Module, seed: int) -> list[str]:
    """Reset only reappearance heads for random-head baseline sweeps."""

    reset_names: list[str] = []
    torch.manual_seed(int(seed))
    for name in ["reappearance_head", "reappearance_event_head"]:
        module = getattr(head, name, None)
        if module is None:
            continue
        for child in module.modules():
            if hasattr(child, "reset_parameters"):
                child.reset_parameters()  # type: ignore[attr-defined]
        reset_names.append(name)
    return reset_names


def _load_head_from_checkpoint(
    repo_root: Path,
    checkpoint: Path,
    device: torch.device,
    *,
    reappearance_random_seed: int | None = None,
    force_random_reappearance_head: bool = False,
):
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
    reset_reappearance_heads: list[str] = []
    if force_random_reappearance_head and reappearance_random_seed is not None:
        reset_reappearance_heads = _reset_reappearance_heads(head, int(reappearance_random_seed))
    head.eval()
    return head, payload, state_dict, cfg, list(missing), list(unexpected), reset_reappearance_heads


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


def _build_full_model_from_checkpoint(
    repo_root: Path,
    checkpoint: Path,
    device: torch.device,
    max_items: int,
    *,
    reappearance_random_seed: int | None = None,
    force_random_reappearance_head: bool = False,
    enable_semantic_state_feedback: bool = False,
    semantic_state_feedback_alpha: float = 0.05,
) -> dict[str, Any]:
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
    semantic_state_feedback_adapter = None
    if bool(enable_semantic_state_feedback) or isinstance(payload.get("semantic_state_feedback_adapter_state_dict"), dict):
        semantic_state_feedback_adapter = trainer.SemanticStateFeedbackAdapter(
            trainer.SemanticStateFeedbackConfig(
                hidden_dim=fusion_hidden_dim,
                semantic_embedding_dim=int(args.future_semantic_embedding_dim),
                identity_embedding_dim=int(args.future_semantic_embedding_dim),
                alpha=float(semantic_state_feedback_alpha),
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
    missing_reappearance = [str(k) for k in missing if str(k).startswith("reappearance_head.")]
    missing_reappearance_event = [str(k) for k in missing if str(k).startswith("reappearance_event_head.")]
    reset_reappearance_heads: list[str] = []
    if force_random_reappearance_head and reappearance_random_seed is not None:
        reset_reappearance_heads = _reset_reappearance_heads(future_semantic_state_head, int(reappearance_random_seed))
    load_report["future_semantic_state_head"] = {
        "loaded_keys": len(payload.get("future_semantic_state_head_state_dict", {}) or {}),
        "missing": len(missing),
        "unexpected": len(unexpected),
        "missing_keys": [str(k) for k in missing],
        "unexpected_keys": [str(k) for k in unexpected],
        "missing_reappearance_head_weights": bool(missing_reappearance),
        "missing_reappearance_head_weight_keys": missing_reappearance,
        "missing_reappearance_event_head_weights": bool(missing_reappearance_event),
        "missing_reappearance_event_head_weight_keys": missing_reappearance_event,
        "force_random_reappearance_head": bool(force_random_reappearance_head),
        "reappearance_random_seed": reappearance_random_seed,
        "reset_reappearance_heads": reset_reappearance_heads,
    }
    if semantic_state_feedback_adapter is not None:
        missing, unexpected = semantic_state_feedback_adapter.load_state_dict(
            payload.get("semantic_state_feedback_adapter_state_dict", {}),
            strict=False,
        )
        load_report["semantic_state_feedback_adapter"] = {
            "loaded_keys": len(payload.get("semantic_state_feedback_adapter_state_dict", {}) or {}),
            "missing": len(missing),
            "unexpected": len(unexpected),
            "missing_keys": [str(k) for k in missing],
            "unexpected_keys": [str(k) for k in unexpected],
            "random_init": not isinstance(payload.get("semantic_state_feedback_adapter_state_dict"), dict),
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
        semantic_state_feedback_adapter,
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
        "semantic_state_feedback_adapter": semantic_state_feedback_adapter,
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
        reappearance_head_available = state.future_reappearance_logit is not None
        reappearance_prob = torch.sigmoid(state.future_reappearance_logit) if reappearance_head_available else None
        reappearance_event_head_available = state.future_reappearance_event_logit is not None
        reappearance_event_prob = torch.sigmoid(state.future_reappearance_event_logit) if reappearance_event_head_available else None
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
            "future_reappearance_head_available": bool(reappearance_head_available),
            "reappearance_prob_source": "future_reappearance_logit" if reappearance_head_available else "missing_reappearance_head",
            "future_reappearance_logit_shape": list(state.future_reappearance_logit.shape) if reappearance_head_available else None,
            "future_reappearance_prob_shape": list(reappearance_prob.shape) if reappearance_prob is not None else None,
            "future_reappearance_prob_mean": _tensor_stats(reappearance_prob)["mean"] if reappearance_prob is not None else None,
            "future_reappearance_prob_std": _tensor_stats(reappearance_prob)["std"] if reappearance_prob is not None else None,
            "future_reappearance_prob_min": _tensor_stats(reappearance_prob)["min"] if reappearance_prob is not None else None,
            "future_reappearance_prob_max": _tensor_stats(reappearance_prob)["max"] if reappearance_prob is not None else None,
            "future_reappearance_prob_values": [],
            "future_reappearance_event_head_available": bool(reappearance_event_head_available),
            "future_reappearance_event_logit_shape": list(state.future_reappearance_event_logit.shape) if reappearance_event_head_available else None,
            "future_reappearance_event_prob_shape": list(reappearance_event_prob.shape) if reappearance_event_prob is not None else None,
            "future_reappearance_event_prob_mean": _tensor_stats(reappearance_event_prob)["mean"] if reappearance_event_prob is not None else None,
            "future_reappearance_event_prob_std": _tensor_stats(reappearance_event_prob)["std"] if reappearance_event_prob is not None else None,
            "future_reappearance_event_prob_min": _tensor_stats(reappearance_event_prob)["min"] if reappearance_event_prob is not None else None,
            "future_reappearance_event_prob_max": _tensor_stats(reappearance_event_prob)["max"] if reappearance_event_prob is not None else None,
            "future_reappearance_event_prob_values": [],
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
    visibility_targets: Any | None = None,
    feedback_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    validation = state.validate(strict=False)
    visibility_prob = torch.sigmoid(state.future_visibility_logit)
    reappearance_head_available = state.future_reappearance_logit is not None
    reappearance_prob = torch.sigmoid(state.future_reappearance_logit) if reappearance_head_available else None
    reappearance_event_head_available = state.future_reappearance_event_logit is not None
    reappearance_event_prob = torch.sigmoid(state.future_reappearance_event_logit) if reappearance_event_head_available else None
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
    target_fields: dict[str, Any] = {
        "future_visibility_target_shape": None,
        "future_visibility_target_positive_rate": None,
        "future_visibility_target_source": "unavailable",
        "future_visibility_target_quality": "weak_unavailable",
        "future_visibility_supervised_ratio": 0.0,
        "future_reappearance_target_shape": None,
        "future_reappearance_target_positive_rate": None,
        "future_reappearance_supervised_ratio": 0.0,
        "future_reappearance_target_positive_rate": None,
        "future_visibility_prob_values": [],
        "future_visibility_target_values": [],
        "future_reappearance_prob_values": [],
        "future_reappearance_target_values": [],
        "future_reappearance_event_target_shape": None,
        "future_reappearance_event_target_positive_rate": None,
        "future_reappearance_event_supervised_ratio": 0.0,
        "future_reappearance_event_prob_values": [],
        "future_reappearance_event_target_values": [],
        "future_reappearance_risk_slot_ratio": 0.0,
        "future_reappearance_risk_entry_ratio": 0.0,
        "future_reappearance_positive_rate_all_slots": None,
        "future_reappearance_positive_rate_at_risk": None,
        "future_reappearance_mask_policy": "unavailable",
        "future_reappearance_negative_policy": "unavailable",
        "reappearance_prob_source": "future_reappearance_logit" if reappearance_head_available else "missing_reappearance_head",
    }
    if visibility_targets is not None:
        vis_t = visibility_targets.future_visibility_target.detach().to(device=visibility_prob.device, dtype=torch.float32)
        vis_m = visibility_targets.future_visibility_mask.detach().to(device=visibility_prob.device, dtype=torch.bool)
        rep_t = visibility_targets.future_reappearance_target.detach().to(device=visibility_prob.device, dtype=torch.float32)
        rep_m = visibility_targets.future_reappearance_mask.detach().to(device=visibility_prob.device, dtype=torch.bool)
        event_t = visibility_targets.future_reappearance_event_target.detach().to(device=visibility_prob.device, dtype=torch.float32)
        event_m = visibility_targets.future_reappearance_event_mask.detach().to(device=visibility_prob.device, dtype=torch.bool)
        vis_vals = visibility_prob.detach()[vis_m].float().cpu().tolist()
        vis_labels = vis_t.detach()[vis_m].float().cpu().tolist()
        rep_vals = reappearance_prob.detach()[rep_m].float().cpu().tolist() if reappearance_prob is not None else []
        rep_labels = rep_t.detach()[rep_m].float().cpu().tolist()
        event_vals = reappearance_event_prob.detach()[event_m].float().cpu().tolist() if reappearance_event_prob is not None else []
        event_labels = event_t.detach()[event_m].float().cpu().tolist()
        loss_info = visibility_targets.to_loss_info()
        target_fields = {
            "future_visibility_target_shape": list(vis_t.shape),
            "future_visibility_target_positive_rate": float(vis_t[vis_m].mean().detach().cpu().item()) if bool(vis_m.any().item()) else None,
            "future_visibility_target_source": str(visibility_targets.target_source),
            "future_visibility_target_quality": str(visibility_targets.target_quality),
            "future_visibility_target_reason": str(visibility_targets.target_reason),
            "future_visibility_supervised_ratio": float(vis_m.float().mean().detach().cpu().item()),
            "future_reappearance_target_shape": list(rep_t.shape),
            "future_reappearance_target_positive_rate": float(rep_t[rep_m].mean().detach().cpu().item()) if bool(rep_m.any().item()) else None,
            "future_reappearance_supervised_ratio": float(rep_m.float().mean().detach().cpu().item()),
            "future_visibility_prob_values": [float(x) for x in vis_vals],
            "future_visibility_target_values": [int(round(float(x))) for x in vis_labels],
            "future_reappearance_prob_values": [float(x) for x in rep_vals],
            "future_reappearance_target_values": [int(round(float(x))) for x in rep_labels],
            "future_reappearance_event_target_shape": list(event_t.shape),
            "future_reappearance_event_target_positive_rate": float(event_t[event_m].mean().detach().cpu().item()) if bool(event_m.any().item()) else None,
            "future_reappearance_event_supervised_ratio": float(event_m.float().mean().detach().cpu().item()),
            "future_reappearance_event_prob_values": [float(x) for x in event_vals],
            "future_reappearance_event_target_values": [int(round(float(x))) for x in event_labels],
            "future_reappearance_risk_slot_ratio": loss_info.get("future_reappearance_risk_slot_ratio"),
            "future_reappearance_risk_entry_ratio": loss_info.get("future_reappearance_risk_entry_ratio"),
            "future_reappearance_positive_rate_all_slots": loss_info.get("future_reappearance_positive_rate_all_slots"),
            "future_reappearance_positive_rate_at_risk": loss_info.get("future_reappearance_positive_rate_at_risk"),
            "future_reappearance_mask_policy": loss_info.get("future_reappearance_mask_policy"),
            "future_reappearance_negative_policy": loss_info.get("future_reappearance_negative_policy"),
            "reappearance_prob_source": "future_reappearance_logit" if reappearance_head_available else "missing_reappearance_head",
        }
    item = {
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
        "future_reappearance_head_available": bool(reappearance_head_available),
        "reappearance_prob_source": "future_reappearance_logit" if reappearance_head_available else "missing_reappearance_head",
        "future_reappearance_logit_shape": list(state.future_reappearance_logit.shape) if reappearance_head_available else None,
        "future_reappearance_prob_shape": list(reappearance_prob.shape) if reappearance_prob is not None else None,
        "future_reappearance_prob_mean": _tensor_stats(reappearance_prob)["mean"] if reappearance_prob is not None else None,
        "future_reappearance_prob_std": _tensor_stats(reappearance_prob)["std"] if reappearance_prob is not None else None,
        "future_reappearance_prob_min": _tensor_stats(reappearance_prob)["min"] if reappearance_prob is not None else None,
        "future_reappearance_prob_max": _tensor_stats(reappearance_prob)["max"] if reappearance_prob is not None else None,
        "future_reappearance_event_head_available": bool(reappearance_event_head_available),
        "future_reappearance_event_logit_shape": list(state.future_reappearance_event_logit.shape) if reappearance_event_head_available else None,
        "future_reappearance_event_prob_shape": list(reappearance_event_prob.shape) if reappearance_event_prob is not None else None,
        "future_reappearance_event_prob_mean": _tensor_stats(reappearance_event_prob)["mean"] if reappearance_event_prob is not None else None,
        "future_reappearance_event_prob_std": _tensor_stats(reappearance_event_prob)["std"] if reappearance_event_prob is not None else None,
        "future_reappearance_event_prob_min": _tensor_stats(reappearance_event_prob)["min"] if reappearance_event_prob is not None else None,
        "future_reappearance_event_prob_max": _tensor_stats(reappearance_event_prob)["max"] if reappearance_event_prob is not None else None,
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
        "semantic_state_feedback_enabled": bool((feedback_info or {}).get("semantic_state_feedback_enabled", False)),
        "semantic_state_feedback_mode": (feedback_info or {}).get("semantic_state_feedback_mode"),
        "semantic_state_feedback_alpha": (feedback_info or {}).get("semantic_state_feedback_alpha"),
        "feedback_gate_mean": (feedback_info or {}).get("feedback_gate_mean", 0.0),
        "feedback_gate_std": (feedback_info or {}).get("feedback_gate_std", 0.0),
        "feedback_gate_saturation_ratio": (feedback_info or {}).get("feedback_gate_saturation_ratio", 0.0),
        "feedback_delta_norm": (feedback_info or {}).get("feedback_delta_norm", 0.0),
        "future_hypothesis_logits_shape": list(state.future_hypothesis_logits.shape) if state.future_hypothesis_logits is not None else None,
        "future_hypothesis_logits_mean": _tensor_stats(state.future_hypothesis_logits)["mean"] if state.future_hypothesis_logits is not None else None,
        "future_hypothesis_trace_coord_shape": list(state.future_hypothesis_trace_coord.shape) if state.future_hypothesis_trace_coord is not None else None,
    }
    item.update(target_fields)
    return item


def export(
    *,
    repo_root: Path,
    checkpoint: Path,
    manifest: Path,
    output: Path,
    max_items: int,
    device_name: str,
    mode: str,
    reappearance_mask_policy: str = "at_risk_only",
    reappearance_random_seed: int | None = None,
    force_random_reappearance_head: bool = False,
    enable_semantic_state_feedback: bool = False,
    semantic_state_feedback_alpha: float = 0.05,
    semantic_state_feedback_mode: str = "readout_only",
    semantic_state_feedback_stopgrad_state: bool = True,
    manifest_mode: str = "stage2_val",
    external_semantic_state_manifest: Path | None = None,
    external_candidate_expanded_manifest: Path | None = None,
    strict_no_fallback: bool = False,
    candidate_score_mode: str = "posterior_v1",
    candidate_measurement_feature_mode: str = "weak_rgb_bbox_stats",
) -> dict[str, Any]:
    if str(manifest_mode) == "external_hardcase_semantic_state":
        if external_semantic_state_manifest is None:
            raise RuntimeError("--external-semantic-state-manifest is required for external_hardcase_semantic_state mode")
        if not bool(strict_no_fallback):
            raise RuntimeError("--strict-no-fallback is required for external_hardcase_semantic_state mode")
        external = load_json(external_semantic_state_manifest)
        external_items = external.get("items") if isinstance(external, dict) else []
        if not isinstance(external_items, list):
            external_items = []
        selected_items = external_items[: int(max_items)]
        exported_items: list[dict[str, Any]] = []
        target_type_counts: dict[str, int] = {}
        metric_eligible_count = 0
        for item in selected_items:
            if not isinstance(item, dict):
                continue
            target_type = str(item.get("target_type") or "unavailable")
            target_type_counts[target_type] = target_type_counts.get(target_type, 0) + 1
            metric_eligible = bool(item.get("usable_for_event_eval") or item.get("usable_for_candidate_eval"))
            if metric_eligible:
                metric_eligible_count += 1
            exported_items.append(
                {
                    "item_id": item.get("item_id"),
                    "protocol_item_id": item.get("sample_key") or item.get("item_id"),
                    "subset_tags": item.get("subset_tags") if isinstance(item.get("subset_tags"), dict) else {},
                    "valid_output": False,
                    "future_semantic_trace_state_valid": False,
                    "failure_reason": item.get("exclusion_reason") or "missing_stage2_dataset_mapping_for_full_model_forward",
                    "target_type": target_type,
                    "event_reappearance_target": item.get("event_reappearance_target"),
                    "candidate_match_target": item.get("candidate_match_target"),
                    "gt_candidate_id": item.get("gt_candidate_id"),
                    "future_candidate_count": len(item.get("future_candidates") or []),
                    "metric_eligible": metric_eligible,
                    "model_input_source": item.get("model_input_source"),
                    "stage2_dataset_mapping_key": item.get("stage2_dataset_mapping_key"),
                    "visibility_target_available": bool(item.get("visibility_target_available")),
                    "per_horizon_visibility_available": bool(item.get("per_horizon_visibility_available")),
                }
            )

        payload = {
            "generated_at_utc": now_iso(),
            "raw_export_schema_version": RAW_EXPORT_SCHEMA_VERSION,
            "export_mode": str(mode),
            "manifest_mode": "external_hardcase_semantic_state",
            "repo_root": str(repo_root),
            "checkpoint_path": str(checkpoint),
            "checkpoint_exists": checkpoint.exists(),
            "checkpoint_loaded": False,
            "checkpoint_loaded_reason": "blocked_before_model_load_due_to_missing_external_to_stage2_mapping",
            "manifest": str(manifest),
            "external_semantic_state_manifest": str(external_semantic_state_manifest),
            "current_export_data_source": "external_389_hardcase_semantic_state_manifest",
            "external_manifest_consumed": True,
            "stage2_val_fallback_used": False,
            "strict_no_fallback": True,
            "blocked_due_to_no_model_input_mapping": True,
            "blocked_due_to_no_model_input_mapping_reason": "semantic-state manifest items are external raw payloads without verified Stage2SemanticDataset/cache mapping; strict mode forbids fallback",
            "full_model_forward_executed": False,
            "full_stage1_stage2_forward_executed": False,
            "full_free_rollout_executed": False,
            "semantic_state_from_model_hidden": False,
            "random_hidden_used": False,
            "old_association_report_used": False,
            "top1_mrr_false_confuser_exported": False,
            "manifest_item_count": len(external_items),
            "total_items": len(exported_items),
            "exported_item_count": len(exported_items),
            "valid_items": 0,
            "valid_ratio": 0.0,
            "metric_eligible_items": metric_eligible_count,
            "target_type_counts": target_type_counts,
            "target_quality": "external_candidate_aligned" if target_type_counts.get("external_candidate_aligned") else "unavailable",
            "strong_slot_aligned": False,
            "items": exported_items,
        }
        write_json(output, payload)
        write_doc(output.with_suffix(".md"), payload)
        return payload

    if str(manifest_mode) == "external_hardcase_query":
        if external_candidate_expanded_manifest is None:
            raise RuntimeError("--external-candidate-expanded-manifest is required for external_hardcase_query mode")
        if not bool(strict_no_fallback):
            raise RuntimeError("--strict-no-fallback is required for external_hardcase_query mode")
        device = torch.device(device_name if device_name != "cuda" or torch.cuda.is_available() else "cpu")
        candidate_manifest = load_json(external_candidate_expanded_manifest)
        records = candidate_manifest.get("records") if isinstance(candidate_manifest, dict) else []
        if not isinstance(records, list):
            records = []
        (
            build_query_batch_from_item,
            group_candidate_records,
            build_candidate_measurement_features,
            build_item_candidate_measurement_cache,
        ) = _import_external_query_builder(repo_root)
        grouped_items = group_candidate_records([x for x in records if isinstance(x, dict)])
        selected_items = grouped_items[: int(max_items)]
        exported_items: list[dict[str, Any]] = []
        candidate_record_count = sum(len(item.get("future_candidates") or []) for item in selected_items)
        positive_candidate_count = sum(int(sum(item.get("candidate_labels") or [])) for item in selected_items)
        negative_candidate_count = candidate_record_count - positive_candidate_count
        score_components: set[str] = set()
        checkpoint_payload: dict[str, Any] = {}
        full_report: dict[str, Any] = {}
        full_model_forward_executed = False
        full_free_rollout_executed = False
        semantic_state_from_model_hidden = False
        checkpoint_loaded = False
        exact_block_reason = None
        try:
            full = _build_full_model_from_checkpoint(
                repo_root,
                checkpoint,
                device,
                max_items,
                reappearance_random_seed=reappearance_random_seed,
                force_random_reappearance_head=force_random_reappearance_head,
                enable_semantic_state_feedback=bool(enable_semantic_state_feedback),
                semantic_state_feedback_alpha=float(semantic_state_feedback_alpha),
            )
            checkpoint_payload = full["payload"]
            checkpoint_loaded = True
            full_model_forward_executed = True
            full_free_rollout_executed = mode == "full_model_free_rollout"
            semantic_state_from_model_hidden = True
            trainer = full["trainer"]
            args = full["args"]
            full_report = {
                "checkpoint_path": str(checkpoint),
                "model_weights_loaded_count": {
                    name: data.get("loaded_keys")
                    for name, data in (full.get("load_report") or {}).items()
                    if isinstance(data, dict)
                },
                "future_semantic_state_head_weights_loaded_count": full.get("load_report", {})
                .get("future_semantic_state_head", {})
                .get("loaded_keys"),
                "batch_source": "external hard-case query batch builder",
                "manifest_path": str(external_candidate_expanded_manifest),
                "item_count": len(selected_items),
                "prediction_path": mode,
                "dataset_summary": full.get("dataset_summary"),
                "load_report": full.get("load_report"),
            }
            with torch.no_grad():
                for item in selected_items:
                    try:
                        raw_batch, builder_meta = build_query_batch_from_item(
                            item,
                            obs_len=int(args.obs_len),
                            fut_len=int(args.fut_len),
                            crop_size=int(args.semantic_crop_size),
                            semantic_temporal_window=int(args.local_temporal_window),
                        )
                        batch = trainer._to_device(raw_batch, device=device, non_blocking=False)
                        if mode == "full_model_teacher_forced":
                            out = trainer._teacher_forced_predict(
                                stage1_model=full["stage1_model"],
                                semantic_encoder=full["semantic_encoder"],
                                semantic_fusion=full["semantic_fusion"],
                                readout_head=full["readout_head"],
                                future_semantic_state_head=full["future_semantic_state_head"],
                                semantic_state_feedback_adapter=full.get("semantic_state_feedback_adapter"),
                                semantic_state_feedback_enabled=bool(enable_semantic_state_feedback),
                                semantic_state_feedback_alpha=float(semantic_state_feedback_alpha),
                                semantic_state_feedback_stopgrad_state=bool(semantic_state_feedback_stopgrad_state),
                                semantic_state_feedback_mode=str(semantic_state_feedback_mode),
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
                        elif mode == "full_model_free_rollout":
                            out = trainer._free_rollout_predict(
                                stage1_model=full["stage1_model"],
                                semantic_encoder=full["semantic_encoder"],
                                semantic_fusion=full["semantic_fusion"],
                                readout_head=full["readout_head"],
                                future_semantic_state_head=full["future_semantic_state_head"],
                                semantic_state_feedback_adapter=full.get("semantic_state_feedback_adapter"),
                                semantic_state_feedback_enabled=bool(enable_semantic_state_feedback),
                                semantic_state_feedback_alpha=float(semantic_state_feedback_alpha),
                                semantic_state_feedback_stopgrad_state=bool(semantic_state_feedback_stopgrad_state),
                                semantic_state_feedback_mode=str(semantic_state_feedback_mode),
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
                        else:
                            raise RuntimeError(f"external_hardcase_query requires full-model mode, got {mode}")
                        state = out.get("future_semantic_trace_state")
                        if state is None:
                            raise RuntimeError(f"{mode} did not return future_semantic_trace_state")
                        measurement_cache = build_item_candidate_measurement_cache(
                            item,
                            semantic_encoder=full["semantic_encoder"],
                            device=device,
                            crop_size=int(args.semantic_crop_size),
                            feature_mode=str(candidate_measurement_feature_mode),
                        )
                        exported = _external_query_item_from_state(
                            item=item,
                            state=state,
                            builder_meta=builder_meta,
                            candidate_score_mode=str(candidate_score_mode),
                            candidate_feature_builder=build_candidate_measurement_features,
                            candidate_measurement_cache=measurement_cache,
                            candidate_measurement_feature_mode=str(candidate_measurement_feature_mode),
                            feedback_info=out.get("semantic_state_feedback_info", {}),
                        )
                        score_components.update(str(x) for x in exported.get("score_components_used", []) if x)
                        exported_items.append(exported)
                    except Exception as exc:
                        exported_items.append(
                            {
                                "item_id": item.get("item_id"),
                                "protocol_item_id": item.get("item_id"),
                                "subset_tags": item.get("subset_tags") if isinstance(item.get("subset_tags"), dict) else {},
                                "valid_output": False,
                                "future_semantic_trace_state_valid": False,
                                "failure_reason": repr(exc),
                                "target_type": "external_candidate_expanded",
                                "target_quality": "external_candidate_expanded",
                                "future_candidate_used_as_input": False,
                                "candidate_used_for_eval_scoring": True,
                                "future_target_leakage": False,
                                "candidate_scores": [],
                                "candidate_labels": item.get("candidate_labels") if isinstance(item.get("candidate_labels"), list) else [],
                            }
                        )
        except Exception as exc:
            exact_block_reason = repr(exc)
            checkpoint_loaded = False
            full_model_forward_executed = False
            full_free_rollout_executed = False
            semantic_state_from_model_hidden = False
            exported_items = [
                {
                    "item_id": item.get("item_id"),
                    "protocol_item_id": item.get("item_id"),
                    "subset_tags": item.get("subset_tags") if isinstance(item.get("subset_tags"), dict) else {},
                    "valid_output": False,
                    "future_semantic_trace_state_valid": False,
                    "failure_reason": f"full_model_build_or_forward_blocked: {repr(exc)}",
                    "target_type": "external_candidate_expanded",
                    "target_quality": "external_candidate_expanded",
                    "future_candidate_used_as_input": False,
                    "candidate_used_for_eval_scoring": True,
                    "future_target_leakage": False,
                    "candidate_scores": [],
                    "candidate_labels": item.get("candidate_labels") if isinstance(item.get("candidate_labels"), list) else [],
                }
                for item in selected_items
            ]

        valid_items = sum(1 for item in exported_items if bool(item.get("valid_output")))
        valid_ratio = valid_items / max(len(exported_items), 1)
        per_item_candidates = [len(item.get("candidate_labels") or []) for item in exported_items]
        payload = {
            "generated_at_utc": now_iso(),
            "raw_export_schema_version": RAW_EXPORT_SCHEMA_VERSION,
            "export_mode": str(mode),
            "manifest_mode": "external_hardcase_query",
            "repo_root": str(repo_root),
            "checkpoint_path": str(checkpoint),
            "checkpoint_exists": checkpoint.exists(),
            "checkpoint_loaded": bool(checkpoint_loaded),
            "checkpoint_loaded_reason": exact_block_reason,
            "consumed_checkpoint": str(checkpoint) if checkpoint_loaded else None,
            "checkpoint_global_step": checkpoint_payload.get("global_step"),
            "manifest": str(manifest),
            "external_candidate_expanded_manifest": str(external_candidate_expanded_manifest),
            "current_export_data_source": "external_hardcase_query_manifest",
            "external_manifest_consumed": True,
            "stage2_val_fallback_used": False,
            "strict_no_fallback": True,
            "old_association_report_used": False,
            "top1_mrr_false_confuser_exported": False,
            "future_candidate_used_as_input": False,
            "candidate_used_for_eval_scoring": True,
            "candidate_bbox_used_for_rollout_input": False,
            "candidate_bbox_used_for_eval_scoring": True,
            "observed_target_used_for_input": True,
            "future_target_leakage": False,
            "random_hidden_used": False,
            "full_model_forward_executed": bool(full_model_forward_executed),
            "full_stage1_stage2_forward_executed": bool(full_model_forward_executed),
            "full_free_rollout_executed": bool(full_free_rollout_executed),
            "semantic_state_from_model_hidden": bool(semantic_state_from_model_hidden),
            "free_rollout_used": bool(full_free_rollout_executed),
            "target_quality": "external_candidate_expanded",
            "target_type": "external_candidate_expanded",
            "candidate_score_mode": str(candidate_score_mode),
            "candidate_measurement_feature_mode": str(candidate_measurement_feature_mode),
            "manifest_item_count": candidate_manifest.get("original_item_count", len(grouped_items)) if isinstance(candidate_manifest, dict) else len(grouped_items),
            "selected_original_item_count": len(selected_items),
            "candidate_record_count": candidate_record_count,
            "positive_candidate_count": positive_candidate_count,
            "negative_candidate_count": negative_candidate_count,
            "total_items": len(exported_items),
            "exported_item_count": len(exported_items),
            "valid_items": valid_items,
            "valid_ratio": valid_ratio,
            "metric_eligible_items": sum(1 for item in exported_items if bool(item.get("valid_output")) and 1 in (item.get("candidate_labels") or [])),
            "candidate_count_min": min(per_item_candidates) if per_item_candidates else None,
            "candidate_count_max": max(per_item_candidates) if per_item_candidates else None,
            "score_components_used": sorted(score_components),
            "candidate_feature_source": "see_per_item_candidate_feature_source",
            "candidate_feature_used_for_scoring": True,
            "candidate_feature_used_for_rollout": False,
            "semantic_state_feedback_enabled": bool(enable_semantic_state_feedback),
            "semantic_state_feedback_mode": str(semantic_state_feedback_mode),
            "semantic_state_feedback_alpha": float(semantic_state_feedback_alpha),
            "current_best_is_feedback_mechanism": False,
            "feedback_branch_closed": True,
            "full_model_loader_report": full_report,
            "exact_block_reason": exact_block_reason,
            "items": exported_items,
        }
        write_json(output, payload)
        write_doc(output.with_suffix(".md"), payload)
        return payload

    device = torch.device(device_name if device_name != "cuda" or torch.cuda.is_available() else "cpu")
    exported_items: list[dict[str, Any]] = []
    full_report: dict[str, Any] = {}
    checkpoint_payload: dict[str, Any] = {}
    state_dict: dict[str, Any] = {}
    random_hidden_used = False
    full_model_forward_executed = False
    full_free_rollout_executed = False
    semantic_state_from_model_hidden = False
    reset_reappearance_heads: list[str] = []
    if mode == "head_only_surrogate":
        head, checkpoint_payload, state_dict, cfg, missing, unexpected, reset_reappearance_heads = _load_head_from_checkpoint(
            repo_root,
            checkpoint,
            device,
            reappearance_random_seed=reappearance_random_seed,
            force_random_reappearance_head=force_random_reappearance_head,
        )
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
        full = _build_full_model_from_checkpoint(
            repo_root,
            checkpoint,
            device,
            max_items,
            reappearance_random_seed=reappearance_random_seed,
            force_random_reappearance_head=force_random_reappearance_head,
            enable_semantic_state_feedback=bool(enable_semantic_state_feedback),
            semantic_state_feedback_alpha=float(semantic_state_feedback_alpha),
        )
        build_visibility_targets = _import_visibility_builder(repo_root)
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
                        semantic_state_feedback_adapter=full.get("semantic_state_feedback_adapter"),
                        semantic_state_feedback_enabled=bool(enable_semantic_state_feedback),
                        semantic_state_feedback_alpha=float(semantic_state_feedback_alpha),
                        semantic_state_feedback_stopgrad_state=bool(semantic_state_feedback_stopgrad_state),
                        semantic_state_feedback_mode=str(semantic_state_feedback_mode),
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
                        semantic_state_feedback_adapter=full.get("semantic_state_feedback_adapter"),
                        semantic_state_feedback_enabled=bool(enable_semantic_state_feedback),
                        semantic_state_feedback_alpha=float(semantic_state_feedback_alpha),
                        semantic_state_feedback_stopgrad_state=bool(semantic_state_feedback_stopgrad_state),
                        semantic_state_feedback_mode=str(semantic_state_feedback_mode),
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
                visibility_targets = build_visibility_targets(
                    batch=batch,
                    out=out,
                    obs_len=int(args.obs_len),
                    fut_len=int(args.fut_len),
                    slot_count=int(state.future_trace_coord.shape[2]),
                    reappearance_mask_policy=str(reappearance_mask_policy),
                )
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
                        visibility_targets=visibility_targets,
                        feedback_info=out.get("semantic_state_feedback_info", {}),
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
        reset_reappearance_heads = (
            full.get("load_report", {})
            .get("future_semantic_state_head", {})
            .get("reset_reappearance_heads", [])
        )
    else:
        raise ValueError(f"unknown export mode: {mode}")

    valid_items = sum(1 for item in exported_items if bool(item.get("valid_output")))
    valid_ratio = valid_items / max(len(exported_items), 1)
    target_sources = [str(item.get("future_visibility_target_source", "")) for item in exported_items if item.get("future_visibility_target_source")]
    target_qualities = [str(item.get("future_visibility_target_quality", "")) for item in exported_items if item.get("future_visibility_target_quality")]
    target_source = target_sources[0] if target_sources else "unavailable"
    target_quality = target_qualities[0] if target_qualities else "weak_unavailable"
    visibility_positive_rates = [
        float(item["future_visibility_target_positive_rate"])
        for item in exported_items
        if isinstance(item.get("future_visibility_target_positive_rate"), (int, float))
    ]
    reappearance_positive_rates = [
        float(item["future_reappearance_target_positive_rate"])
        for item in exported_items
        if isinstance(item.get("future_reappearance_target_positive_rate"), (int, float))
    ]
    reappearance_positive_rates_all_slots = [
        float(item["future_reappearance_positive_rate_all_slots"])
        for item in exported_items
        if isinstance(item.get("future_reappearance_positive_rate_all_slots"), (int, float))
    ]
    reappearance_positive_rates_at_risk = [
        float(item["future_reappearance_positive_rate_at_risk"])
        for item in exported_items
        if isinstance(item.get("future_reappearance_positive_rate_at_risk"), (int, float))
    ]
    reappearance_event_positive_rates = [
        float(item["future_reappearance_event_target_positive_rate"])
        for item in exported_items
        if isinstance(item.get("future_reappearance_event_target_positive_rate"), (int, float))
    ]
    feedback_gate_means = [
        float(item["feedback_gate_mean"])
        for item in exported_items
        if isinstance(item.get("feedback_gate_mean"), (int, float))
    ]
    feedback_gate_stds = [
        float(item["feedback_gate_std"])
        for item in exported_items
        if isinstance(item.get("feedback_gate_std"), (int, float))
    ]
    feedback_delta_norms = [
        float(item["feedback_delta_norm"])
        for item in exported_items
        if isinstance(item.get("feedback_delta_norm"), (int, float))
    ]
    feedback_gate_saturations = [
        float(item["feedback_gate_saturation_ratio"])
        for item in exported_items
        if isinstance(item.get("feedback_gate_saturation_ratio"), (int, float))
    ]
    reappearance_risk_slot_ratios = [
        float(item["future_reappearance_risk_slot_ratio"])
        for item in exported_items
        if isinstance(item.get("future_reappearance_risk_slot_ratio"), (int, float))
    ]
    reappearance_risk_entry_ratios = [
        float(item["future_reappearance_risk_entry_ratio"])
        for item in exported_items
        if isinstance(item.get("future_reappearance_risk_entry_ratio"), (int, float))
    ]
    visibility_supervised_ratios = [
        float(item["future_visibility_supervised_ratio"])
        for item in exported_items
        if isinstance(item.get("future_visibility_supervised_ratio"), (int, float))
    ]
    reappearance_supervised_ratios = [
        float(item["future_reappearance_supervised_ratio"])
        for item in exported_items
        if isinstance(item.get("future_reappearance_supervised_ratio"), (int, float))
    ]
    reappearance_head_available = bool(
        exported_items and all(bool(item.get("future_reappearance_head_available")) for item in exported_items if bool(item.get("valid_output")))
    )
    reappearance_event_head_available = bool(
        exported_items and all(bool(item.get("future_reappearance_event_head_available")) for item in exported_items if bool(item.get("valid_output")))
    )
    reappearance_sources = [str(item.get("reappearance_prob_source", "")) for item in exported_items if item.get("reappearance_prob_source")]
    reappearance_prob_source = "future_reappearance_logit" if reappearance_sources and all(x == "future_reappearance_logit" for x in reappearance_sources) else (
        reappearance_sources[0] if reappearance_sources else "missing_reappearance_head"
    )
    missing_reappearance_head_weights = bool(any(str(k).startswith("reappearance_head.") for k in locals().get("missing", [])))
    if full_report:
        missing_reappearance_head_weights = bool(
            full_report.get("load_report", {})
            .get("future_semantic_state_head", {})
            .get("missing_reappearance_head_weights", missing_reappearance_head_weights)
        )
    missing_reappearance_event_head_weights = bool(
        full_report.get("load_report", {})
        .get("future_semantic_state_head", {})
        .get("missing_reappearance_event_head_weights", False)
    ) if full_report else bool(any(str(k).startswith("reappearance_event_head.") for k in locals().get("missing", [])))
    visibility_metric_status = (
        "calibrated_visibility_available"
        if target_quality == "strong_slot_aligned"
        else "target_available_but_not_strong_slot_aligned"
        if target_quality == "medium_broadcast"
        else "target_unavailable"
    )
    engineering_output_claimable = bool(
        mode != "head_only_surrogate"
        and full_model_forward_executed
        and semantic_state_from_model_hidden
        and valid_ratio >= 0.95
    )
    current_export_data_source = (
        "Stage2SemanticDataset validation split from checkpoint args"
        if mode in {"full_model_teacher_forced", "full_model_free_rollout"}
        else "external/item manifest head-only surrogate sanity inputs"
    )
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
        "missing_reappearance_head_weights": bool(missing_reappearance_head_weights),
        "missing_reappearance_event_head_weights": bool(missing_reappearance_event_head_weights),
        "reappearance_head_weights_random_init": bool(missing_reappearance_head_weights and reappearance_head_available),
        "reappearance_event_head_weights_random_init": bool(missing_reappearance_event_head_weights and reappearance_event_head_available),
        "force_random_reappearance_head": bool(force_random_reappearance_head),
        "reappearance_random_seed": reappearance_random_seed,
        "reset_reappearance_heads": list(reset_reappearance_heads) if isinstance(reset_reappearance_heads, list) else [],
        "future_reappearance_head_available": bool(reappearance_head_available),
        "future_reappearance_event_head_available": bool(reappearance_event_head_available),
        "reappearance_prob_source": reappearance_prob_source,
        "manifest": str(manifest),
        "device": str(device),
        "random_hidden_used": bool(random_hidden_used),
        "observed_bbox_surrogate_coord_used": bool(mode == "head_only_surrogate"),
        "full_model_forward_executed": bool(full_model_forward_executed),
        "full_stage1_stage2_forward_executed": bool(full_model_forward_executed),
        "full_free_rollout_executed": bool(full_free_rollout_executed),
        "semantic_state_from_model_hidden": bool(semantic_state_from_model_hidden),
        "free_rollout_used": bool(full_free_rollout_executed),
        "engineering_output_claimable": bool(engineering_output_claimable),
        "paper_world_model_claimable": False,
        "world_model_output_claimable": bool(engineering_output_claimable),
        "world_model_output_claimable_scope": "engineering_output_only_not_paper_level",
        "visibility_metric_status": visibility_metric_status,
        "calibrated_visibility_available": bool(target_quality == "strong_slot_aligned"),
        "future_visibility_target_source": target_source,
        "future_visibility_target_quality": target_quality,
        "future_visibility_supervised_ratio": float(sum(visibility_supervised_ratios) / max(len(visibility_supervised_ratios), 1)),
        "future_reappearance_supervised_ratio": float(sum(reappearance_supervised_ratios) / max(len(reappearance_supervised_ratios), 1)),
        "future_visibility_positive_rate": float(sum(visibility_positive_rates) / max(len(visibility_positive_rates), 1)) if visibility_positive_rates else None,
        "future_reappearance_positive_rate": float(sum(reappearance_positive_rates) / max(len(reappearance_positive_rates), 1)) if reappearance_positive_rates else None,
        "future_reappearance_positive_rate_all_slots": float(sum(reappearance_positive_rates_all_slots) / max(len(reappearance_positive_rates_all_slots), 1)) if reappearance_positive_rates_all_slots else None,
        "future_reappearance_positive_rate_at_risk": float(sum(reappearance_positive_rates_at_risk) / max(len(reappearance_positive_rates_at_risk), 1)) if reappearance_positive_rates_at_risk else None,
        "future_reappearance_event_positive_rate": float(sum(reappearance_event_positive_rates) / max(len(reappearance_event_positive_rates), 1)) if reappearance_event_positive_rates else None,
        "future_reappearance_risk_slot_ratio": float(sum(reappearance_risk_slot_ratios) / max(len(reappearance_risk_slot_ratios), 1)) if reappearance_risk_slot_ratios else None,
        "future_reappearance_risk_entry_ratio": float(sum(reappearance_risk_entry_ratios) / max(len(reappearance_risk_entry_ratios), 1)) if reappearance_risk_entry_ratios else None,
        "future_reappearance_mask_policy": str(reappearance_mask_policy),
        "semantic_state_feedback_enabled": bool(enable_semantic_state_feedback),
        "semantic_state_feedback_mode": str(semantic_state_feedback_mode),
        "semantic_state_feedback_alpha": float(semantic_state_feedback_alpha),
        "semantic_state_feedback_stopgrad_state": bool(semantic_state_feedback_stopgrad_state),
        "feedback_gate_mean": float(sum(feedback_gate_means) / max(len(feedback_gate_means), 1)),
        "feedback_gate_std": float(sum(feedback_gate_stds) / max(len(feedback_gate_stds), 1)),
        "feedback_gate_saturation_ratio": float(sum(feedback_gate_saturations) / max(len(feedback_gate_saturations), 1)),
        "feedback_delta_norm": float(sum(feedback_delta_norms) / max(len(feedback_delta_norms), 1)),
        "current_export_data_source": current_export_data_source,
        "old_association_report_used": False,
        "top1_mrr_false_confuser_exported": False,
        "total_items": len(exported_items),
        "valid_items": valid_items,
        "valid_ratio": valid_ratio,
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
    p.add_argument(
        "--manifest-mode",
        default="stage2_val",
        choices=["stage2_val", "external_hardcase_semantic_state", "external_hardcase_query"],
    )
    p.add_argument("--external-semantic-state-manifest", default=None)
    p.add_argument("--external-candidate-expanded-manifest", default=None)
    p.add_argument("--strict-no-fallback", action="store_true")
    p.add_argument(
        "--candidate-score-mode",
        default="posterior_v1",
        choices=[
            "distance_only",
            "semantic_only",
            "identity_only",
            "priors_only",
            "posterior_v1",
            "posterior_no_distance",
            "posterior_no_semantic",
            "weak_posterior_v3",
            "target_candidate_appearance_only",
            "predicted_semantic_to_candidate",
            "predicted_identity_to_candidate",
            "predicted_semantic_identity_to_candidate",
            "posterior_v4",
            "posterior_v4_no_distance",
            "posterior_v4_no_semantic_identity",
            "posterior_v4_no_target_candidate_appearance",
        ],
    )
    p.add_argument(
        "--candidate-measurement-feature-mode",
        default="weak_rgb_bbox_stats",
        choices=["weak_rgb_bbox_stats", "crop_encoder_feature", "hybrid_crop_bbox_feature", "frozen_vlm_crop_feature"],
    )
    p.add_argument("--output", required=True)
    p.add_argument("--max-items", "--item-limit", dest="max_items", type=int, default=32)
    p.add_argument("--device", default="cpu")
    p.add_argument("--use-free-rollout", action="store_true", help="Deprecated alias for --mode full_model_free_rollout.")
    p.add_argument("--future-reappearance-mask-policy", default="at_risk_only", choices=["at_risk_only", "all_slots"])
    p.add_argument("--reappearance-random-seed", type=int, default=None)
    p.add_argument("--force-random-reappearance-head", action="store_true")
    p.add_argument("--enable-semantic-state-feedback", action="store_true")
    p.add_argument("--semantic-state-feedback-alpha", type=float, default=0.05)
    p.add_argument("--semantic-state-feedback-mode", default="readout_only", choices=["readout_only", "hidden_residual"])
    p.add_argument("--semantic-state-feedback-stopgrad-state", action="store_true", default=True)
    p.add_argument("--no-semantic-state-feedback-stopgrad-state", action="store_false", dest="semantic_state_feedback_stopgrad_state")
    return p.parse_args()


def main() -> None:
    _apply_process_title_normalization()
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
        reappearance_mask_policy=str(args.future_reappearance_mask_policy),
        reappearance_random_seed=args.reappearance_random_seed,
        force_random_reappearance_head=bool(args.force_random_reappearance_head),
        enable_semantic_state_feedback=bool(args.enable_semantic_state_feedback),
        semantic_state_feedback_alpha=float(args.semantic_state_feedback_alpha),
        semantic_state_feedback_mode=str(args.semantic_state_feedback_mode),
        semantic_state_feedback_stopgrad_state=bool(args.semantic_state_feedback_stopgrad_state),
        manifest_mode=str(args.manifest_mode),
        external_semantic_state_manifest=Path(args.external_semantic_state_manifest) if args.external_semantic_state_manifest else None,
        external_candidate_expanded_manifest=Path(args.external_candidate_expanded_manifest)
        if args.external_candidate_expanded_manifest
        else None,
        strict_no_fallback=bool(args.strict_no_fallback),
        candidate_score_mode=str(args.candidate_score_mode),
        candidate_measurement_feature_mode=str(args.candidate_measurement_feature_mode),
    )


if __name__ == "__main__":
    main()
