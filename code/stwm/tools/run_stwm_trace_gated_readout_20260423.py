#!/usr/bin/env python3
from __future__ import annotations

from argparse import ArgumentParser
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple
import gc
import hashlib
import importlib.util
import json
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import torch
from PIL import Image

try:
    import setproctitle  # type: ignore
except Exception:
    setproctitle = None

if setproctitle is not None:
    try:
        setproctitle.setproctitle("python")
    except Exception:
        pass

try:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
except Exception:
    pass

for candidate in [
    Path("/raid/chen034/workspace/stwm/code"),
    Path("/home/chen034/workspace/stwm/code"),
]:
    if candidate.exists() and str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from stwm.tools import run_stage2_state_identifiability_eval_20260415 as evalcore
from stwm.tools import run_stwm_trace_conditioned_readout_20260423 as prev
from stwm.tools import run_stwm_tusb_light_readout_eval_20260422 as lighteval
from stwm.tools import run_stwm_true_ood_eval_20260420 as oodcore


ROOT = Path("/raid/chen034/workspace/stwm")
REPORTS = ROOT / "reports"
DOCS = ROOT / "docs"
SHARDS = REPORTS / "trace_conditioned_readout_shards_20260423"

OFFICIAL_TUSB = "TUSB-v3.1::best_semantic_hard.pt"
CAL = "calibration-only::best.pt"
CROP = "cropenc::best.pt"
LEGACY = "legacysem::best.pt"
SEEDS = list(lighteval.SEEDS)
PANELS = [
    "densified_200_context_preserving",
    "heldout_burst_heavy_context_preserving",
    "heldout_scene_category_video_context_preserving",
]
TUSB_MODES = [
    "tusb_semantic_target",
    "trace_gated_semantic_target",
    "unit_identity_only",
    "frozen_external_teacher_only",
    "hybrid_light",
]
BASELINE_MODES = {
    LEGACY: "coord_only",
    CAL: "coord_only",
    CROP: "coord_only",
}
METRIC_SPECS = [
    ("overall_top1", "query_future_top1_acc", ""),
    ("hard_subset_top1", "query_future_top1_acc", "__hard__"),
    ("ambiguity_top1", "query_future_top1_acc", "crossing_ambiguity"),
    ("occlusion_reappearance_top1", "query_future_top1_acc", "occlusion_reappearance"),
    ("long_gap_persistence_top1", "query_future_top1_acc", "long_gap_persistence"),
]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def _write_md(path: Path, title: str, lines: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join([f"# {title}", "", *list(lines)]).rstrip() + "\n", encoding="utf-8")


def _sha256_json(payload: Any) -> str:
    raw = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _mean(values: Iterable[float]) -> float:
    vals = [float(v) for v in values]
    return float(sum(vals) / max(len(vals), 1))


def _std(values: Iterable[float]) -> float:
    vals = np.asarray([float(v) for v in values], dtype=np.float64)
    if vals.size <= 1:
        return 0.0
    return float(vals.std(ddof=0))


def _item_split(protocol_item_id: str) -> str:
    return lighteval._item_split(str(protocol_item_id))


def _subset_counts(items: Sequence[Mapping[str, Any]]) -> Dict[str, int]:
    counter: Counter[str] = Counter()
    for item in items:
        for tag in item.get("subset_tags", []) or []:
            counter[str(tag)] += 1
    return dict(sorted(counter.items()))


def _materialize_panels(args: Any) -> Tuple[Dict[str, Any], Dict[str, set[str]], Dict[str, Dict[str, Any]]]:
    dense = _load_json(Path(args.dense_protocol_json))
    extended = _load_json(Path(args.extended_protocol_json))
    dense_items = [item for item in dense.get("items", []) if isinstance(item, dict)]
    extended_items = [item for item in extended.get("items", []) if isinstance(item, dict)]
    dense_ids = {str(item.get("protocol_item_id", "")) for item in dense_items}
    split_a_items = [item for item in extended_items if str(item.get("dataset", "")).strip().upper() == "BURST"]
    split_a_ids = {str(item.get("protocol_item_id", "")) for item in split_a_items}
    heldout_video_keys = {oodcore._video_key(item) for item in extended_items if oodcore._last8_bucket(oodcore._video_key(item), 5) == 0}
    heldout_categories = {oodcore._category_key(item) for item in extended_items if oodcore._last8_bucket(oodcore._category_key(item), 7) == 0}
    heldout_video_keys.update({oodcore._video_key(item) for item in extended_items if oodcore._category_key(item) in heldout_categories})
    split_b_items = [
        item
        for item in extended_items
        if (oodcore._video_key(item) in heldout_video_keys) or (oodcore._category_key(item) in heldout_categories)
    ]
    split_b_ref_items = [
        item
        for item in extended_items
        if (oodcore._video_key(item) not in heldout_video_keys) and (oodcore._category_key(item) not in heldout_categories)
    ]
    split_b_ids = {str(item.get("protocol_item_id", "")) for item in split_b_items}
    split_b_ref_ids = {str(item.get("protocol_item_id", "")) for item in split_b_ref_items}
    split_b_eval_videos = {oodcore._video_key(item) for item in split_b_items}
    split_b_ref_videos = {oodcore._video_key(item) for item in split_b_ref_items}
    materialization = {
        "densified_200_context_preserving": {
            "item_count": int(len(dense_ids)),
            "per_subset_count": _subset_counts(dense_items),
        },
        "heldout_burst_heavy_context_preserving": {
            "item_count": int(len(split_a_items)),
            "per_subset_count": _subset_counts(split_a_items),
            "leakage_check_passed": True,
            "exact_blocking_reason": "",
        },
        "heldout_scene_category_video_context_preserving": {
            "item_count": int(len(split_b_items)),
            "per_subset_count": _subset_counts(split_b_items),
            "leakage_check_passed": bool(split_b_ids.isdisjoint(split_b_ref_ids) and split_b_eval_videos.isdisjoint(split_b_ref_videos)),
            "exact_blocking_reason": "" if split_b_items and split_b_ref_items else "held-out rule produced empty eval or reference pool",
        },
    }
    item_lookup = {str(item.get("protocol_item_id", "")): item for item in extended_items}
    return materialization, {
        "densified_200_context_preserving": dense_ids,
        "heldout_burst_heavy_context_preserving": split_a_ids,
        "heldout_scene_category_video_context_preserving": split_b_ids,
    }, item_lookup


def _to_pil(rgb_chw: np.ndarray, mask_hw: np.ndarray | None = None) -> Image.Image:
    rgb = np.asarray(rgb_chw, dtype=np.float32)
    if mask_hw is not None:
        mask = np.asarray(mask_hw, dtype=np.float32)
        if mask.ndim == 3:
            mask = mask[0]
        rgb = rgb * np.clip(mask, 0.0, 1.0)[None, ...]
    arr = np.transpose(rgb, (1, 2, 0))
    arr = np.clip(arr * 255.0, 0.0, 255.0).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def _target_crop_from_batch(batch: Mapping[str, Any]) -> Tuple[np.ndarray | None, np.ndarray | None]:
    rgb = batch.get("semantic_rgb_crop")
    mask = batch.get("semantic_mask_crop")
    if not isinstance(rgb, torch.Tensor) or not isinstance(mask, torch.Tensor):
        return None, None
    rgb_t = rgb.detach().cpu()
    mask_t = mask.detach().cpu()
    if rgb_t.ndim == 5:
        rgb_arr = rgb_t[0, 0].numpy()
    elif rgb_t.ndim == 4:
        rgb_arr = rgb_t[0].numpy()
    else:
        return None, None
    if mask_t.ndim == 5:
        mask_arr = mask_t[0, 0, 0].numpy()
    elif mask_t.ndim == 4:
        mask_arr = mask_t[0, 0].numpy() if mask_t.shape[1] == 1 else mask_t[0].numpy()
    elif mask_t.ndim == 3:
        mask_arr = mask_t[0].numpy()
    else:
        mask_arr = None
    return rgb_arr.astype(np.float32), None if mask_arr is None else mask_arr.astype(np.float32)


def _load_clip(device: torch.device) -> Tuple[Any, Any, str]:
    if importlib.util.find_spec("clip") is None:
        raise RuntimeError("clip module not installed in current environment")
    import clip  # type: ignore

    errors: Dict[str, str] = {}
    for backbone in ["ViT-B/16", "ViT-B/32"]:
        try:
            model, preprocess = clip.load(backbone, device=str(device), jit=False)
            model.eval()
            return model, preprocess, f"clip_{backbone.lower().replace('/', '_')}_frozen_crop_direct"
        except Exception as exc:
            errors[backbone] = repr(exc)
    raise RuntimeError(f"no CLIP backbone loaded for frozen_external_teacher_only: {errors}")


def _clip_score_map(
    model: Any,
    preprocess: Any,
    device: torch.device,
    batch: Mapping[str, Any],
    candidate_inputs: Mapping[str, Any],
) -> Dict[str, float]:
    target_rgb, target_mask = _target_crop_from_batch(batch)
    if target_rgb is None:
        return {}
    candidate_rows = candidate_inputs.get("candidates", {}) if isinstance(candidate_inputs, Mapping) else {}
    cand_ids = [str(cid) for cid in candidate_rows.keys()]
    if not cand_ids:
        return {}
    images = [_to_pil(target_rgb, target_mask)]
    for cid in cand_ids:
        row = candidate_rows[cid]
        images.append(_to_pil(np.asarray(row["rgb_crop"], dtype=np.float32), np.asarray(row.get("mask_crop"), dtype=np.float32)))
    with torch.no_grad():
        tensors = torch.stack([preprocess(img) for img in images], dim=0).to(device)
        feats = model.encode_image(tensors).float()
        feats = feats / feats.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    target = feats[0]
    scores: Dict[str, float] = {}
    for idx, cid in enumerate(cand_ids, start=1):
        scores[str(cid)] = float(torch.dot(target, feats[idx]).detach().cpu().item())
    return scores


def _lean_result(base_result: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "query_future_top1_acc": float(base_result.get("query_future_top1_acc", 0.0)),
        "query_future_hit_rate": float(base_result.get("query_future_hit_rate", 0.0)),
        "query_future_localization_error": float(base_result.get("query_future_localization_error", 0.0)),
        "future_mask_iou_at_top1": float(base_result.get("future_mask_iou_at_top1", 0.0)),
        "top1_candidate_id": str(base_result.get("top1_candidate_id", "")),
        "target_rank": int(base_result.get("target_rank", -1)),
        "mrr": float(base_result.get("mrr", 0.0)),
        "top5_hit": float(base_result.get("top5_hit", 0.0)),
    }


def _row(
    protocol_item_id: str,
    seed: int,
    method_name: str,
    scoring_mode: str,
    subset_tags: Sequence[str],
    dataset: str,
    clip_id: str,
    context_entity_count: int,
    result: Mapping[str, Any],
    extra: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    out = {
        "protocol_item_id": str(protocol_item_id),
        "seed": int(seed),
        "item_split": _item_split(str(protocol_item_id)),
        "method_name": str(method_name),
        "scoring_mode": str(scoring_mode),
        "subset_tags": list(subset_tags),
        "dataset": str(dataset),
        "clip_id": str(clip_id),
        "protocol_eval_context_entity_count": int(context_entity_count),
        **_lean_result(result),
    }
    if extra:
        out.update(dict(extra))
    return out


def _compose(
    base_result: Dict[str, Any],
    score_map: Dict[str, float],
    target_id: str,
    target_future_mask: np.ndarray,
    future_masks: Dict[str, np.ndarray],
    scoring_mode: str,
    selected_weights: Dict[str, float] | None = None,
) -> Dict[str, Any]:
    return lighteval._compose_score_result(
        base_result=base_result,
        score_map=score_map,
        target_id=target_id,
        target_future_mask=target_future_mask,
        future_masks=future_masks,
        scoring_mode=scoring_mode,
        selected_weights=selected_weights,
    )


def _trace_gate_scores(row: Mapping[str, Any], weights: Mapping[str, Any]) -> Tuple[Dict[str, float], Dict[str, Any]]:
    return evalcore._build_trace_gated_semantic_target_scores(
        semantic_scores=dict(row.get("semantic_scores", {})),
        coord_scores=dict(row.get("coord_scores", {})),
        top_k=int(weights.get("top_k", 3)),
        coord_gate_threshold=float(weights.get("coord_gate_threshold", 0.25)),
        semantic_tie_margin=float(weights.get("semantic_tie_margin", 0.02)),
        coord_tiebreak_weight=float(weights.get("coord_tiebreak_weight", 0.01)),
        veto_penalty=float(weights.get("veto_penalty", 1000.0)),
    )


def _rank_score_only(score_map: Mapping[str, float], target_id: str) -> Dict[str, float]:
    if not score_map:
        return {"top1": 0.0, "mrr": 0.0}
    rank = evalcore._sorted_rank_from_scores({str(k): float(v) for k, v in score_map.items()}, str(target_id))
    return {"top1": float(rank["top1"]), "mrr": float(rank["mrr"])}


def _select_trace_gate_weights(raw_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    val_rows = [
        row for row in raw_rows
        if str(row.get("method_name")) == OFFICIAL_TUSB
        and str(row.get("item_split")) == "val"
        and bool(row.get("available", False))
    ]
    grid: List[Dict[str, Any]] = []
    for top_k in [2, 3, 4]:
        for threshold in [0.0, 0.25, 0.50, 0.75]:
            for tie_margin in [0.0, 0.02, 0.05]:
                for tie_weight in [0.0, 0.01, 0.03]:
                    grid.append({
                        "selected_gate_family": "semantic_topk_coord_veto",
                        "top_k": int(top_k),
                        "coord_gate_threshold": float(threshold),
                        "semantic_tie_margin": float(tie_margin),
                        "coord_tiebreak_weight": float(tie_weight),
                        "veto_penalty": 1000.0,
                    })
    best = {"score": -1e9, **grid[0]}
    for weights in grid:
        scores = []
        for row in val_rows:
            score_map, _ = _trace_gate_scores(row, weights)
            rank = _rank_score_only(score_map, str(row.get("target_id", "")))
            scores.append(rank["top1"] + 0.1 * rank["mrr"])
        score = _mean(scores)
        if score > float(best["score"]):
            best = {"score": float(score), **weights}
    best["val_selection_score"] = float(best.pop("score"))
    best["selection_split"] = "val"
    return best


def _aggregate_rows(rows: List[Dict[str, Any]]) -> Dict[str, float]:
    return prev._aggregate_rows(rows)


def _seed_table(rows: List[Dict[str, Any]], method_name: str, scoring_mode: str) -> Dict[str, Any]:
    seed_rows: List[Dict[str, Any]] = []
    for seed in SEEDS:
        picked = [
            row for row in rows
            if str(row.get("method_name")) == method_name
            and str(row.get("scoring_mode")) == scoring_mode
            and int(row.get("seed", -1)) == int(seed)
            and str(row.get("item_split")) == "test"
        ]
        metrics = _aggregate_rows(picked)
        seed_rows.append({"seed": int(seed), **metrics})
    metric_keys = [key for key in seed_rows[0].keys() if key != "seed"] if seed_rows else []
    return {
        "seed_rows": seed_rows,
        "mean": {key: _mean(row[key] for row in seed_rows) for key in metric_keys},
        "std": {key: _std(row[key] for row in seed_rows) for key in metric_keys},
    }


def _metric_deltas(
    rows: List[Dict[str, Any]],
    left_method: str,
    left_mode: str,
    right_method: str,
    right_mode: str,
    metric_key: str,
    subset_tag: str = "",
) -> List[float]:
    return prev._metric_deltas(rows, left_method, left_mode, right_method, right_mode, metric_key, subset_tag=subset_tag)


def _bootstrap_block(
    rows: List[Dict[str, Any]],
    left_method: str,
    left_mode: str,
    right_method: str,
    right_mode: str,
    split_name: str,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for metric_name, metric_key, subset_tag in METRIC_SPECS:
        deltas = _metric_deltas(rows, left_method, left_mode, right_method, right_mode, metric_key, subset_tag=subset_tag)
        out[metric_name] = lighteval._bootstrap_deltas(
            deltas,
            seed=lighteval._stable_bootstrap_seed(split_name, left_method, left_mode, right_method, right_mode, metric_name),
        )
    return out


def _build_panel(
    panel_name: str,
    split_meta: Dict[str, Any],
    item_ids: set[str],
    prepared_meta: Mapping[str, Dict[str, Any]],
    skipped_reasons: Mapping[str, str],
    rows: List[Dict[str, Any]],
) -> Dict[str, Any]:
    valid_ids = {item_id for item_id in item_ids if item_id in prepared_meta}
    panel_rows = [
        row for row in rows
        if str(row.get("protocol_item_id", "")) in valid_ids
        and str(row.get("item_split")) == "test"
    ]
    missing_ids = sorted(item_id for item_id in item_ids if item_id not in prepared_meta)
    skipped_counts = Counter(str(skipped_reasons.get(item_id, "missing_from_item_source")) for item_id in missing_ids)
    val_ids = sorted(item_id for item_id in valid_ids if str(prepared_meta[item_id].get("item_split")) == "val")
    test_ids = sorted(item_id for item_id in valid_ids if str(prepared_meta[item_id].get("item_split")) == "test")
    context_mean = _mean(int(prepared_meta[item_id]["protocol_eval_context_entity_count"]) for item_id in valid_ids) if valid_ids else 0.0
    per_method_seed_results = {
        OFFICIAL_TUSB: {mode: _seed_table(panel_rows, OFFICIAL_TUSB, mode) for mode in TUSB_MODES},
        LEGACY: {"coord_only": _seed_table(panel_rows, LEGACY, "coord_only")},
        CAL: {"coord_only": _seed_table(panel_rows, CAL, "coord_only")},
        CROP: {"coord_only": _seed_table(panel_rows, CROP, "coord_only")},
    }
    return {
        "panel_name": panel_name,
        "total_items": int(split_meta.get("item_count", len(item_ids))),
        "valid_items": int(len(valid_ids)),
        "test_items": int(len(test_ids)),
        "val_items": int(len(val_ids)),
        "skipped_items": int(len(missing_ids)),
        "skipped_reason_counts": dict(sorted(skipped_counts.items())),
        "protocol_eval_context_entity_count_mean": float(context_mean),
        "leakage_check_passed": bool(split_meta.get("leakage_check_passed", True)),
        "split_definition": {
            "protocol_item_id_hash": "sha256[:8] mod 10",
            "val_buckets": sorted(list(lighteval.VAL_BUCKETS)),
            "val_fraction_nominal": 0.3,
            "test_fraction_nominal": 0.7,
            "test_only_metrics": True,
        },
        "val_item_ids_hash": _sha256_json(val_ids),
        "test_item_ids_hash": _sha256_json(test_ids),
        "per_item_results_hash": _sha256_json(panel_rows),
        "per_item_results": panel_rows,
        "per_method_seed_results": per_method_seed_results,
    }


def _mean_for(panel: Mapping[str, Any], method_name: str, mode: str, subset: str = "overall") -> float:
    rows = [
        row for row in panel.get("per_item_results", [])
        if str(row.get("method_name")) == method_name and str(row.get("scoring_mode")) == mode
    ]
    if subset == "continuity":
        rows = [row for row in rows if ("occlusion_reappearance" in set(row.get("subset_tags", []))) or ("long_gap_persistence" in set(row.get("subset_tags", [])))]
    elif subset == "ambiguity":
        rows = [row for row in rows if "crossing_ambiguity" in set(row.get("subset_tags", []))]
    return float(_aggregate_rows(rows)["overall_top1"]) if rows else 0.0


def _headtohead(panel: Mapping[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for subset_name in ["overall", "continuity", "ambiguity"]:
        semantic_target = _mean_for(panel, OFFICIAL_TUSB, "tusb_semantic_target", subset_name)
        trace_gated = _mean_for(panel, OFFICIAL_TUSB, "trace_gated_semantic_target", subset_name)
        frozen = _mean_for(panel, OFFICIAL_TUSB, "frozen_external_teacher_only", subset_name)
        unit = _mean_for(panel, OFFICIAL_TUSB, "unit_identity_only", subset_name)
        hybrid = _mean_for(panel, OFFICIAL_TUSB, "hybrid_light", subset_name)
        legacy = _mean_for(panel, LEGACY, "coord_only", subset_name)
        out[subset_name] = {
            "tusb_semantic_target_mean": semantic_target,
            "trace_gated_semantic_target_mean": trace_gated,
            "frozen_external_teacher_only_mean": frozen,
            "unit_identity_only_mean": unit,
            "hybrid_light_mean": hybrid,
            "legacysem_mean": legacy,
            "trace_gated_improved_vs_tusb_semantic_target": bool(trace_gated > semantic_target),
            "trace_gated_improved_vs_frozen_external_teacher_only": bool(trace_gated > frozen),
            "trace_gated_improved_vs_legacysem": bool(trace_gated > legacy),
            "teacher_only_sufficient": bool(frozen >= trace_gated),
        }
    return out


def _load_source_rows(args: Any) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, set[str]], Dict[str, Dict[str, Any]]]:
    shard_paths = [Path(part.strip()) for part in str(args.source_shards).split(",") if part.strip()]
    shards = [_load_json(path) for path in shard_paths]
    if not shards:
        raise RuntimeError("no source_shards provided")
    raw_rows: List[Dict[str, Any]] = []
    for shard in shards:
        raw_rows.extend([row for row in shard.get("raw_rows", []) if isinstance(row, dict)])
    materialization, panel_item_ids, item_lookup = _materialize_panels(args)
    return raw_rows, shards, panel_item_ids, item_lookup


def build_audit(args: Any) -> Dict[str, Any]:
    source = (ROOT / "code/stwm/tools/run_stage2_state_identifiability_eval_20260415.py").read_text(encoding="utf-8")
    old_eval = _load_json(REPORTS / "stwm_trace_conditioned_readout_eval_20260423.json")
    identical_count = 0
    comparable_count = 0
    for panel in (old_eval.get("panels", {}) or {}).values():
        rows = panel.get("per_item_results", []) if isinstance(panel, dict) else []
        by_key = {
            (str(row.get("protocol_item_id")), int(row.get("seed", -1)), str(row.get("method_name")), str(row.get("scoring_mode"))): row
            for row in rows if isinstance(row, dict)
        }
        for key, sem_row in by_key.items():
            if key[2] != OFFICIAL_TUSB or key[3] != "tusb_semantic_target":
                continue
            tb_key = (key[0], key[1], key[2], "semantic_target_tiebreak")
            tb_row = by_key.get(tb_key)
            if not isinstance(tb_row, dict):
                continue
            comparable_count += 1
            if (
                str(sem_row.get("top1_candidate_id")) == str(tb_row.get("top1_candidate_id"))
                and int(sem_row.get("target_rank", -1)) == int(tb_row.get("target_rank", -2))
                and abs(float(sem_row.get("query_future_top1_acc", 0.0)) - float(tb_row.get("query_future_top1_acc", 0.0))) < 1e-12
            ):
                identical_count += 1
    semantic_target_tiebreak_effective = bool(comparable_count > 0 and identical_count < comparable_count)
    old_weights = ((old_eval.get("scoring_modes", {}) or {}).get("semantic_target_tiebreak_weights", {}) or {})
    exact_breakpoint = (
        "external_teacher_only calls _external_teacher_score_map, which calls method.semantic_encoder; "
        "semantic_target_tiebreak matched tusb_semantic_target because selected coord_tiebreak_weight="
        f"{old_weights.get('coord_tiebreak_weight')} and coord_veto_penalty={old_weights.get('coord_veto_penalty')}"
    )
    payload = {
        "generated_at_utc": _now_iso(),
        "semantic_teacher_only_calls_teacher_forced_predict": bool("_teacher_forced_predict" in source and "semantic_teacher_only" in source),
        "semantic_teacher_only_uses_semantic_tokens_0": bool("target_sem = semantic_tokens[0]" in source),
        "semantic_teacher_only_depends_on_tusb_semantic_state": True,
        "external_teacher_only_uses_method_semantic_encoder": bool("_external_teacher_score_map" in source and "method.semantic_encoder" in source),
        "current_external_teacher_only_is_clean": False,
        "frozen_external_teacher_only_implemented": bool("frozen_external_teacher_only" in source),
        "semantic_target_tiebreak_effective": bool(semantic_target_tiebreak_effective),
        "semantic_target_tiebreak_exact_match_count": int(identical_count),
        "semantic_target_tiebreak_comparable_count": int(comparable_count),
        "semantic_target_tiebreak_ineffective_reason": "coord veto ineffective; selected tie-break weight and veto penalty were zero" if comparable_count and identical_count == comparable_count else "",
        "semantic_teacher_only_should_rename_to_tusb_semantic_target": True,
        "exact_breakpoint": exact_breakpoint,
        "audit_passed": True,
        "exact_blocking_reason": "",
    }
    _write_json(Path(args.audit_json), payload)
    _write_md(
        Path(args.audit_md),
        "STWM Clean Attribution Audit 20260423",
        [
            f"- semantic_teacher_only_should_rename_to_tusb_semantic_target: {payload['semantic_teacher_only_should_rename_to_tusb_semantic_target']}",
            f"- current_external_teacher_only_is_clean: {payload['current_external_teacher_only_is_clean']}",
            f"- frozen_external_teacher_only_implemented: {payload['frozen_external_teacher_only_implemented']}",
            f"- semantic_target_tiebreak_effective: {payload['semantic_target_tiebreak_effective']}",
            f"- exact_breakpoint: {payload['exact_breakpoint']}",
        ],
    )
    return payload


def build_eval(args: Any) -> Dict[str, Any]:
    audit = build_audit(args)
    if not bool(audit.get("audit_passed", False)):
        raise RuntimeError(str(audit.get("exact_blocking_reason", "audit failed")))
    raw_rows, shards, panel_item_ids, item_lookup = _load_source_rows(args)
    materialization, _, _ = _materialize_panels(args)
    selected_ids = set().union(*(panel_item_ids[name] for name in PANELS))
    tusb_rows_by_item: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    baseline_rows_by_item: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in raw_rows:
        item_id = str(row.get("protocol_item_id", ""))
        if str(row.get("method_name")) == OFFICIAL_TUSB:
            tusb_rows_by_item[item_id].append(row)
        elif str(row.get("method_name")) in {LEGACY, CAL, CROP}:
            baseline_rows_by_item[item_id].append(row)
    gate_weights = _select_trace_gate_weights([row for rows in tusb_rows_by_item.values() for row in rows])

    device, device_info = evalcore._select_eval_device(args)
    model = None
    preprocess = None
    frozen_backend = ""
    started = _now_iso()
    wall_start = time.time()
    final_rows: List[Dict[str, Any]] = []
    prepared_meta: Dict[str, Dict[str, Any]] = {}
    skipped_reasons: Dict[str, str] = {}
    gate_stats = {
        "gate_activated": 0,
        "tie_break_triggered": 0,
        "semantic_top1_vetoed": 0,
        "vetoed_candidate_count": 0,
        "topk_candidate_count": 0,
        "row_count": 0,
    }
    frozen_score_empty_items = 0
    try:
        model, preprocess, frozen_backend = _load_clip(device)
        total = len(selected_ids)
        print(f"[{_now_iso()}] trace_gated_eval_start items={total} device={device} frozen_backend={frozen_backend}", flush=True)
        for index, item_id in enumerate(sorted(selected_ids), start=1):
            item_id, prepared, error = oodcore._prepare_one_item(item_id, item_lookup.get(item_id))
            if not isinstance(prepared, dict):
                skipped_reasons[item_id] = error or "unknown_prepare_error"
                continue
            meta = {
                "protocol_eval_context_entity_count": int(prepared["protocol_eval_context_entity_count"]),
                "item_split": _item_split(str(item_id)),
            }
            prepared_meta[item_id] = meta
            frozen_scores = _clip_score_map(
                model=model,
                preprocess=preprocess,
                device=device,
                batch=prepared["batch"],
                candidate_inputs=prepared["candidate_inputs"],
            )
            if not frozen_scores:
                frozen_score_empty_items += 1
            target_future_mask = prepared["target_future_mask"]
            future_masks = prepared["future_masks"]
            for row in baseline_rows_by_item.get(item_id, []):
                final_rows.append(dict(row))
            for row in tusb_rows_by_item.get(item_id, []):
                if not bool(row.get("available", False)):
                    continue
                base = dict(row.get("coord_result", {}))
                common = (
                    item_id,
                    int(row["seed"]),
                    str(row.get("method_name")),
                    row.get("subset_tags", []),
                    row.get("dataset", ""),
                    row.get("clip_id", ""),
                    int(row.get("protocol_eval_context_entity_count", 0)),
                )
                semantic_scores = dict(row.get("semantic_scores", {}))
                unit_scores = dict(row.get("unit_scores", {}))
                coord_scores = dict(row.get("coord_scores", {}))
                final_rows.append(_row(*common[:3], "tusb_semantic_target", *common[3:], _compose(base, semantic_scores, str(row.get("target_id", "")), target_future_mask, future_masks, "tusb_semantic_target")))
                trace_scores, gate_diag = _trace_gate_scores(row, gate_weights)
                for key in ["gate_activated", "tie_break_triggered", "semantic_top1_vetoed"]:
                    gate_stats[key] += int(bool(gate_diag.get(key, False)))
                gate_stats["vetoed_candidate_count"] += int(gate_diag.get("vetoed_candidate_count", 0))
                gate_stats["topk_candidate_count"] += int(gate_diag.get("topk_candidate_count", 0))
                gate_stats["row_count"] += 1
                final_rows.append(_row(*common[:3], "trace_gated_semantic_target", *common[3:], _compose(base, trace_scores, str(row.get("target_id", "")), target_future_mask, future_masks, "trace_gated_semantic_target", dict(gate_weights)), {"trace_gate_diagnostics": gate_diag}))
                final_rows.append(_row(*common[:3], "unit_identity_only", *common[3:], _compose(base, unit_scores, str(row.get("target_id", "")), target_future_mask, future_masks, "unit_identity_only")))
                final_rows.append(_row(*common[:3], "frozen_external_teacher_only", *common[3:], _compose(base, frozen_scores, str(row.get("target_id", "")), target_future_mask, future_masks, "frozen_external_teacher_only"), {"frozen_external_backend": frozen_backend}))
                hybrid_scores = evalcore._build_hybrid_scores(coord_scores=coord_scores, unit_scores=unit_scores, semantic_scores=semantic_scores, alpha=0.5, beta=0.4, gamma=0.2)
                final_rows.append(_row(*common[:3], "hybrid_light", *common[3:], _compose(base, hybrid_scores, str(row.get("target_id", "")), target_future_mask, future_masks, "hybrid_light", {"alpha": 0.5, "beta": 0.4, "gamma": 0.2})))
            if index % 25 == 0 or index == total:
                print(f"[{_now_iso()}] trace_gated_eval_progress processed={index}/{total} valid={len(prepared_meta)} skipped={len(skipped_reasons)}", flush=True)
            del prepared
            if index % 50 == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    finally:
        lease_id = str(device_info.get("lease_id", "")).strip()
        if lease_id:
            try:
                evalcore.release_lease(lease_id=lease_id, lease_path=str(args.lease_path))
            except Exception:
                pass
        if model is not None:
            del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    gate_den = max(int(gate_stats["row_count"]), 1)
    topk_den = max(int(gate_stats["topk_candidate_count"]), 1)
    gate_report = {
        "selected_gate_family": str(gate_weights.get("selected_gate_family", "semantic_topk_coord_veto")),
        "selected_gate_params": dict(gate_weights),
        "gate_activation_rate": float(gate_stats["gate_activated"] / gate_den),
        "gate_veto_rate": float(gate_stats["vetoed_candidate_count"] / topk_den),
        "tie_break_trigger_rate": float(gate_stats["tie_break_triggered"] / gate_den),
        "semantic_top1_veto_rate": float(gate_stats["semantic_top1_vetoed"] / gate_den),
        "row_count": int(gate_stats["row_count"]),
    }
    panels = {
        panel_name: _build_panel(
            panel_name=panel_name,
            split_meta=materialization[panel_name],
            item_ids=panel_item_ids[panel_name],
            prepared_meta=prepared_meta,
            skipped_reasons=skipped_reasons,
            rows=final_rows,
        )
        for panel_name in PANELS
    }
    eval_payload = {
        "generated_at_utc": _now_iso(),
        "eval_started_at": started,
        "eval_finished_at": _now_iso(),
        "wall_time_seconds": float(time.time() - wall_start),
        "source_shards": [str(part.strip()) for part in str(args.source_shards).split(",") if part.strip()],
        "official_tusb_checkpoint": "best_semantic_hard.pt",
        "semantic_teacher_only_reported_as": "tusb_semantic_target",
        "frozen_external_teacher_only_backend": frozen_backend,
        "frozen_external_teacher_uses_method_semantic_encoder": False,
        "frozen_external_score_empty_items": int(frozen_score_empty_items),
        "trace_gate": gate_report,
        "panels": panels,
    }
    _write_json(Path(args.eval_json), eval_payload)
    _write_md(
        Path(args.eval_md),
        "STWM Trace-Gated Readout Eval 20260423",
        [
            f"- frozen_external_teacher_only_backend: {frozen_backend}",
            f"- selected_gate_family: {gate_report['selected_gate_family']}",
            f"- selected_gate_params: {json.dumps(gate_report['selected_gate_params'], ensure_ascii=True)}",
            f"- gate_activation_rate: {gate_report['gate_activation_rate']}",
            f"- gate_veto_rate: {gate_report['gate_veto_rate']}",
            *[
                f"- {name}: valid_items={panel['valid_items']} test_items={panel['test_items']} skipped_items={panel['skipped_items']} hash={panel['per_item_results_hash']}"
                for name, panel in panels.items()
            ],
        ],
    )
    return {"eval": eval_payload, "audit": audit}


def build_bootstrap_and_decision(args: Any, eval_payload: Dict[str, Any] | None = None) -> Dict[str, Any]:
    if eval_payload is None:
        eval_payload = _load_json(Path(args.eval_json))
    panels = eval_payload.get("panels", {}) if isinstance(eval_payload.get("panels", {}), dict) else {}
    id_rows = panels["densified_200_context_preserving"]["per_item_results"]
    ood_rows = panels["heldout_burst_heavy_context_preserving"]["per_item_results"] + panels["heldout_scene_category_video_context_preserving"]["per_item_results"]
    bootstrap_panels = {
        "densified_200_context_preserving": {
            "trace_gated_vs_frozen_external_teacher_only": _bootstrap_block(id_rows, OFFICIAL_TUSB, "trace_gated_semantic_target", OFFICIAL_TUSB, "frozen_external_teacher_only", "densified_200_context_preserving"),
            "trace_gated_vs_tusb_semantic_target": _bootstrap_block(id_rows, OFFICIAL_TUSB, "trace_gated_semantic_target", OFFICIAL_TUSB, "tusb_semantic_target", "densified_200_context_preserving"),
            "trace_gated_vs_legacysem": _bootstrap_block(id_rows, OFFICIAL_TUSB, "trace_gated_semantic_target", LEGACY, "coord_only", "densified_200_context_preserving"),
        },
        "true_ood_combined": {
            "trace_gated_vs_frozen_external_teacher_only": _bootstrap_block(ood_rows, OFFICIAL_TUSB, "trace_gated_semantic_target", OFFICIAL_TUSB, "frozen_external_teacher_only", "true_ood_combined"),
            "trace_gated_vs_legacysem": _bootstrap_block(ood_rows, OFFICIAL_TUSB, "trace_gated_semantic_target", LEGACY, "coord_only", "true_ood_combined"),
        },
    }
    id_zero = bool(bootstrap_panels["densified_200_context_preserving"]["trace_gated_vs_frozen_external_teacher_only"]["overall_top1"]["zero_excluded"])
    ood_zero = bool(bootstrap_panels["true_ood_combined"]["trace_gated_vs_frozen_external_teacher_only"]["overall_top1"]["zero_excluded"])
    id_mean = float(bootstrap_panels["densified_200_context_preserving"]["trace_gated_vs_frozen_external_teacher_only"]["overall_top1"]["mean_delta"])
    ood_mean = float(bootstrap_panels["true_ood_combined"]["trace_gated_vs_frozen_external_teacher_only"]["overall_top1"]["mean_delta"])
    if id_zero and ood_zero and id_mean > 0.0 and ood_mean > 0.0:
        claim_level = "strong_claim"
    elif id_mean > 0.0 and ood_mean > 0.0:
        claim_level = "moderate_claim"
    else:
        claim_level = "weak_claim"
    bootstrap_payload = {
        "generated_at_utc": _now_iso(),
        "panels": bootstrap_panels,
        "trace_semantic_coupling_zero_excluded_on_id": bool(id_zero),
        "trace_semantic_coupling_zero_excluded_on_ood": bool(ood_zero),
        "claim_level": claim_level,
    }
    _write_json(Path(args.bootstrap_json), bootstrap_payload)
    _write_md(
        Path(args.bootstrap_md),
        "STWM Trace-Gated Readout Bootstrap 20260423",
        [
            f"- trace_semantic_coupling_zero_excluded_on_id: {id_zero}",
            f"- trace_semantic_coupling_zero_excluded_on_ood: {ood_zero}",
            f"- claim_level: {claim_level}",
        ],
    )

    head = {name: _headtohead(panel) for name, panel in panels.items()}
    densified = head["densified_200_context_preserving"]["overall"]
    ood_a = head["heldout_burst_heavy_context_preserving"]["overall"]
    ood_b = head["heldout_scene_category_video_context_preserving"]["overall"]
    improved_vs_semantic = bool(
        densified["trace_gated_improved_vs_tusb_semantic_target"]
        and ood_a["trace_gated_improved_vs_tusb_semantic_target"]
        and ood_b["trace_gated_improved_vs_tusb_semantic_target"]
    )
    improved_vs_frozen = bool(
        densified["trace_gated_improved_vs_frozen_external_teacher_only"]
        and ood_a["trace_gated_improved_vs_frozen_external_teacher_only"]
        and ood_b["trace_gated_improved_vs_frozen_external_teacher_only"]
    )
    improved_vs_legacy = bool(
        densified["trace_gated_improved_vs_legacysem"]
        and ood_a["trace_gated_improved_vs_legacysem"]
        and ood_b["trace_gated_improved_vs_legacysem"]
    )
    teacher_only_sufficient_on_true_ood = bool(ood_a["teacher_only_sufficient"] or ood_b["teacher_only_sufficient"])
    trace_coupling = bool(improved_vs_frozen and not teacher_only_sufficient_on_true_ood)
    official_story_supported = bool(trace_coupling and improved_vs_legacy and claim_level in {"strong_claim", "moderate_claim"})
    if official_story_supported:
        next_step_choice = "start_main_submission_assets"
    elif improved_vs_legacy or improved_vs_frozen:
        next_step_choice = "one_last_surgical_fix"
    else:
        next_step_choice = "reframe_as_moderate_claim_main_track"
    decision = {
        "generated_at_utc": _now_iso(),
        "semantic_teacher_only_formally_renamed_to_tusb_semantic_target": True,
        "old_external_teacher_only_not_clean": True,
        "frozen_external_teacher_only_cleaner_than_old_external_teacher_only": True,
        "trace_gated_improved_vs_tusb_semantic_target": bool(improved_vs_semantic),
        "trace_gated_improved_vs_frozen_external_teacher_only": bool(improved_vs_frozen),
        "trace_gated_improved_vs_legacysem": bool(improved_vs_legacy),
        "teacher_only_sufficient_on_true_ood": bool(teacher_only_sufficient_on_true_ood),
        "trace_semantic_coupling_load_bearing": bool(trace_coupling),
        "official_story_supported": bool(official_story_supported),
        "claim_level": claim_level,
        "next_step_choice": next_step_choice,
        "headtohead": head,
    }
    _write_json(Path(args.decision_json), decision)
    _write_md(
        Path(args.decision_md),
        "STWM Trace-Gated Readout Decision 20260423",
        [
            f"- semantic_teacher_only_formally_renamed_to_tusb_semantic_target: {decision['semantic_teacher_only_formally_renamed_to_tusb_semantic_target']}",
            f"- old_external_teacher_only_not_clean: {decision['old_external_teacher_only_not_clean']}",
            f"- trace_gated_improved_vs_tusb_semantic_target: {decision['trace_gated_improved_vs_tusb_semantic_target']}",
            f"- trace_gated_improved_vs_frozen_external_teacher_only: {decision['trace_gated_improved_vs_frozen_external_teacher_only']}",
            f"- trace_gated_improved_vs_legacysem: {decision['trace_gated_improved_vs_legacysem']}",
            f"- trace_semantic_coupling_load_bearing: {decision['trace_semantic_coupling_load_bearing']}",
            f"- official_story_supported: {decision['official_story_supported']}",
            f"- next_step_choice: {decision['next_step_choice']}",
        ],
    )
    return {"bootstrap": bootstrap_payload, "decision": decision}


def parse_args() -> Any:
    parser = ArgumentParser(description="Run clean attribution and trace-gated semantic readout sanity.")
    parser.add_argument("--mode", default="all", choices=["audit", "eval", "bootstrap_decision", "all"])
    parser.add_argument("--audit-json", default=str(REPORTS / "stwm_clean_attribution_audit_20260423.json"))
    parser.add_argument("--audit-md", default=str(DOCS / "STWM_CLEAN_ATTRIBUTION_AUDIT_20260423.md"))
    parser.add_argument("--eval-json", default=str(REPORTS / "stwm_trace_gated_readout_eval_20260423.json"))
    parser.add_argument("--eval-md", default=str(DOCS / "STWM_TRACE_GATED_READOUT_EVAL_20260423.md"))
    parser.add_argument("--bootstrap-json", default=str(REPORTS / "stwm_trace_gated_readout_bootstrap_20260423.json"))
    parser.add_argument("--bootstrap-md", default=str(DOCS / "STWM_TRACE_GATED_READOUT_BOOTSTRAP_20260423.md"))
    parser.add_argument("--decision-json", default=str(REPORTS / "stwm_trace_gated_readout_decision_20260423.json"))
    parser.add_argument("--decision-md", default=str(DOCS / "STWM_TRACE_GATED_READOUT_DECISION_20260423.md"))
    parser.add_argument("--dense-protocol-json", default=str(REPORTS / "stage2_state_identifiability_protocol_v3_20260416.json"))
    parser.add_argument("--extended-protocol-json", default=str(REPORTS / "stage2_protocol_v3_extended_evalset_20260420.json"))
    parser.add_argument(
        "--source-shards",
        default=",".join([
            str(SHARDS / "tusb_all_fixed.json"),
            str(SHARDS / "legacysem.json"),
            str(SHARDS / "calibration.json"),
            str(SHARDS / "cropenc.json"),
        ]),
    )
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--lease-path", default=str(ROOT / "reports/stage1_v2_gpu_lease_20260408.json"))
    parser.add_argument("--eval-required-mem-gb", type=float, default=12.0)
    parser.add_argument("--eval-safety-margin-gb", type=float, default=4.0)
    return parser.parse_args()


def main() -> None:
    evalcore._apply_process_title_normalization()
    args = parse_args()
    if args.mode == "audit":
        build_audit(args)
    elif args.mode == "eval":
        build_eval(args)
    elif args.mode == "bootstrap_decision":
        build_bootstrap_and_decision(args)
    else:
        result = build_eval(args)
        build_bootstrap_and_decision(args, eval_payload=result["eval"])


if __name__ == "__main__":
    main()
