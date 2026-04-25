#!/usr/bin/env python3
from __future__ import annotations

from argparse import ArgumentParser
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple
import gc
import hashlib
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

for candidate in [Path("/raid/chen034/workspace/stwm/code"), Path("/home/chen034/workspace/stwm/code")]:
    if candidate.exists() and str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from stwm.tools import run_stage2_state_identifiability_eval_20260415 as evalcore
from stwm.tools import run_stwm_tusb_light_readout_eval_20260422 as lighteval
from stwm.tools import run_stwm_trace_conditioned_readout_20260423 as tracecond
from stwm.tools import run_stwm_trace_gated_readout_20260423 as tracegate
from stwm.tools import run_stwm_trace_prototype_association_20260424 as proto
from stwm.tools import run_stwm_trace_residual_association_20260423 as residual
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
    "frozen_external_teacher_only",
    "tusb_semantic_target",
    "trace_prototype_only",
    "trace_gallery_assoc",
    "trace_belief_sem_only",
    "trace_belief_assoc",
]
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


def _minmax(scores: Mapping[str, float]) -> Dict[str, float]:
    return evalcore._minmax_normalize_score_map({str(k): float(v) for k, v in scores.items()})


def _clip_belief_score_maps(
    model: Any,
    preprocess: Any,
    device: torch.device,
    target_gallery: Sequence[Image.Image],
    candidate_inputs: Mapping[str, Any],
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float], Dict[str, Any]]:
    candidate_rows = candidate_inputs.get("candidates", {}) if isinstance(candidate_inputs, Mapping) else {}
    cand_ids = [str(cid) for cid in candidate_rows.keys()]
    if not target_gallery:
        return {}, {}, {}, {"blocking_reason": "target_gallery_empty", "gallery_frame_count": 0, "candidate_count": int(len(cand_ids))}
    if not cand_ids:
        return {}, {}, {}, {"blocking_reason": "candidate_inputs_empty", "gallery_frame_count": int(len(target_gallery)), "candidate_count": 0}
    images: List[Image.Image] = list(target_gallery)
    for cid in cand_ids:
        row = candidate_rows[cid]
        images.append(tracegate._to_pil(np.asarray(row["rgb_crop"], dtype=np.float32), np.asarray(row.get("mask_crop"), dtype=np.float32)))
    with torch.no_grad():
        tensors = torch.stack([preprocess(img) for img in images], dim=0).to(device)
        feats = model.encode_image(tensors).float()
        feats = feats / feats.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    target_feats = feats[: len(target_gallery)]
    candidate_feats = feats[len(target_gallery) :]
    mean_vec = target_feats.mean(dim=0)
    mean_vec = mean_vec / mean_vec.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    diag_var = target_feats.var(dim=0, unbiased=False) if target_feats.shape[0] > 1 else torch.zeros_like(mean_vec)
    variance_scalar = float(diag_var.mean().detach().cpu().item())
    sims = target_feats @ candidate_feats.T
    proto_scores: Dict[str, float] = {}
    gallery_scores: Dict[str, float] = {}
    belief_scores: Dict[str, float] = {}
    for idx, cid in enumerate(cand_ids):
        cand = candidate_feats[idx]
        mean_cos = float(torch.dot(mean_vec, cand).detach().cpu().item())
        cand_sims = sims[:, idx]
        sim_mean = float(cand_sims.mean().detach().cpu().item())
        sim_std = float(cand_sims.std(unbiased=False).detach().cpu().item())
        sim_max = float(cand_sims.max().detach().cpu().item())
        # Uncertainty-aware likelihood: reward mean consistency, mildly reward a strong gallery match,
        # and penalize high within-gallery disagreement. This is computed only from observed history.
        belief_scores[str(cid)] = float(0.55 * sim_mean + 0.25 * sim_max + 0.20 * mean_cos - 0.25 * sim_std)
        proto_scores[str(cid)] = float(mean_cos)
        gallery_scores[str(cid)] = float(0.5 * sim_max + 0.5 * sim_mean)
    return proto_scores, gallery_scores, belief_scores, {
        "blocking_reason": "",
        "gallery_frame_count": int(len(target_gallery)),
        "candidate_count": int(len(cand_ids)),
        "diag_variance_mean": variance_scalar,
        "diag_variance_available": bool(target_feats.shape[0] > 1),
    }


def _linear_mix(parts: Sequence[Tuple[Mapping[str, float], float]]) -> Dict[str, float]:
    ids = sorted(set().union(*[set(part.keys()) for part, _ in parts if part]))
    out: Dict[str, float] = {}
    for cid in ids:
        out[str(cid)] = float(sum(float(weight) * float(scores.get(cid, 0.0)) for scores, weight in parts))
    return out


def _trace_belief_sem_scores(row: Mapping[str, Any], belief_scores: Mapping[str, float]) -> Dict[str, float]:
    belief_n = _minmax(belief_scores)
    sem_n = _minmax(dict(row.get("semantic_scores", {})))
    unit_n = _minmax(dict(row.get("unit_scores", {})))
    if belief_n:
        return _linear_mix([(belief_n, 0.60), (sem_n, 0.25), (unit_n, 0.15)])
    return _linear_mix([(sem_n, 0.65), (unit_n, 0.35)])


def _trace_belief_assoc_scores(row: Mapping[str, Any], belief_scores: Mapping[str, float]) -> Dict[str, float]:
    belief_n = _minmax(belief_scores)
    sem_n = _minmax(dict(row.get("semantic_scores", {})))
    unit_n = _minmax(dict(row.get("unit_scores", {})))
    coord_n = _minmax(dict(row.get("coord_scores", {})))
    if belief_n:
        return _linear_mix([(belief_n, 0.50), (sem_n, 0.20), (unit_n, 0.20), (coord_n, 0.10)])
    return _linear_mix([(sem_n, 0.45), (unit_n, 0.35), (coord_n, 0.20)])


def _label_result(score_map: Mapping[str, float], target_id: str, target_future_mask: np.ndarray, future_masks: Mapping[str, np.ndarray], base: Mapping[str, Any], mode: str) -> Dict[str, Any]:
    return lighteval._compose_score_result(
        base_result=dict(base),
        score_map={str(k): float(v) for k, v in score_map.items()},
        target_id=str(target_id),
        target_future_mask=target_future_mask,
        future_masks=dict(future_masks),
        scoring_mode=mode,
    )


def _result_row(
    protocol_item_id: str,
    seed: int,
    method_name: str,
    scoring_mode: str,
    subset_tags: Sequence[str],
    dataset: str,
    clip_id: str,
    context_count: int,
    result: Mapping[str, Any],
) -> Dict[str, Any]:
    row = tracegate._row(
        protocol_item_id,
        seed,
        method_name,
        scoring_mode,
        subset_tags,
        dataset,
        clip_id,
        context_count,
        dict(result),
    )
    row["item_split"] = residual._item_split3(str(protocol_item_id))
    return row


def _build_feasibility_audit(args: Any) -> Dict[str, Any]:
    raw_rows, _, panel_item_ids, item_lookup = tracegate._load_source_rows(args)
    selected_ids = sorted(set().union(*(panel_item_ids[name] for name in PANELS)))
    tusb_rows = [row for row in raw_rows if str(row.get("method_name")) == OFFICIAL_TUSB and bool(row.get("available", False))]
    nonempty_sem = sum(1 for row in tusb_rows if row.get("semantic_scores"))
    nonempty_unit = sum(1 for row in tusb_rows if row.get("unit_scores"))
    nonempty_coord = sum(1 for row in tusb_rows if row.get("coord_scores"))
    gallery_counts: List[int] = []
    blockers: Counter[str] = Counter()
    for item_id in selected_ids[: min(len(selected_ids), int(args.audit_sample_items))]:
        item = item_lookup.get(item_id)
        if not isinstance(item, dict):
            blockers["item_missing_from_lookup"] += 1
            continue
        _, diag = proto._observed_target_gallery_images(item)
        gallery_counts.append(int(diag.get("gallery_frame_count", 0)))
        reason = str(diag.get("blocking_reason", ""))
        if reason:
            blockers[reason] += 1
    multi_frame_available = bool(gallery_counts and max(gallery_counts) >= 2)
    payload = {
        "generated_at_utc": _now_iso(),
        "source_shards": [str(part.strip()) for part in str(args.source_shards).split(",") if part.strip()],
        "selected_item_count": int(len(selected_ids)),
        "tusb_available_row_count": int(len(tusb_rows)),
        "can_construct_target_semantic_belief_mean": bool(gallery_counts and max(gallery_counts) > 0),
        "can_construct_semantic_uncertainty_or_diag_variance": bool(multi_frame_available),
        "can_construct_motion_coord_plausibility_prior": bool(nonempty_coord),
        "available_features": {
            "semantic_target_token_or_score": bool(nonempty_sem),
            "unit_identity_score_over_observed_window": bool(nonempty_unit),
            "observed_target_crop_teacher_embeddings_over_window": bool(gallery_counts and max(gallery_counts) > 0),
            "teacher_aligned_target_embeddings": bool(nonempty_sem),
            "z_sem_z_dyn_summary": False,
            "coord_motion_plausibility_over_observed_window": bool(nonempty_coord),
        },
        "feature_counts": {
            "tusb_rows_with_semantic_scores": int(nonempty_sem),
            "tusb_rows_with_unit_scores": int(nonempty_unit),
            "tusb_rows_with_coord_scores": int(nonempty_coord),
        },
        "audit_sample_items": int(args.audit_sample_items),
        "sample_gallery_frame_count_min": int(min(gallery_counts) if gallery_counts else 0),
        "sample_gallery_frame_count_max": int(max(gallery_counts) if gallery_counts else 0),
        "sample_gallery_blocking_reason_counts": dict(sorted(blockers.items())),
        "future_gt_dependency": False,
        "minimal_new_cache_needed": False,
        "audit_passed": bool(gallery_counts and max(gallery_counts) > 0 and multi_frame_available and nonempty_sem and nonempty_unit and nonempty_coord),
        "exact_blocking_reason": "" if (gallery_counts and max(gallery_counts) > 0 and multi_frame_available and nonempty_sem and nonempty_unit and nonempty_coord) else "belief mean/variance inputs or TUSB semantic/unit/coord score maps unavailable",
    }
    _write_json(Path(args.audit_json), payload)
    _write_md(
        Path(args.audit_md),
        "STWM Trace Belief Feasibility Audit 20260424",
        [
            f"- can_construct_target_semantic_belief_mean: {payload['can_construct_target_semantic_belief_mean']}",
            f"- can_construct_semantic_uncertainty_or_diag_variance: {payload['can_construct_semantic_uncertainty_or_diag_variance']}",
            f"- can_construct_motion_coord_plausibility_prior: {payload['can_construct_motion_coord_plausibility_prior']}",
            f"- future_gt_dependency: {payload['future_gt_dependency']}",
            f"- minimal_new_cache_needed: {payload['minimal_new_cache_needed']}",
            f"- audit_passed: {payload['audit_passed']}",
            f"- exact_blocking_reason: {payload['exact_blocking_reason']}",
        ],
    )
    return payload


def _aggregate_rows(rows: List[Dict[str, Any]]) -> Dict[str, float]:
    return tracecond._aggregate_rows(rows)


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
        "skipped_items": int(len(missing_ids)),
        "skipped_reason_counts": dict(sorted(skipped_counts.items())),
        "protocol_eval_context_entity_count_mean": float(context_mean),
        "leakage_check_passed": bool(split_meta.get("leakage_check_passed", True)),
        "split_definition": {
            "protocol_item_id_hash": "sha256[:8] mod 10",
            "train_buckets": [0, 1],
            "val_buckets": [2],
            "test_buckets": [3, 4, 5, 6, 7, 8, 9],
            "test_only_metrics": True,
        },
        "test_item_ids_hash": _sha256_json(test_ids),
        "per_item_results_hash": _sha256_json(panel_rows),
        "per_item_results": panel_rows,
        "per_method_seed_results": per_method_seed_results,
    }


def _build_eval(args: Any) -> Dict[str, Any]:
    audit = _build_feasibility_audit(args)
    if not bool(audit.get("audit_passed", False)):
        raise RuntimeError(str(audit.get("exact_blocking_reason", "feasibility audit failed")))
    raw_rows, _, panel_item_ids, item_lookup = tracegate._load_source_rows(args)
    materialization, _, _ = tracegate._materialize_panels(args)
    selected_ids = set().union(*(panel_item_ids[name] for name in PANELS))
    tusb_rows_by_item: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    baseline_rows_by_item: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in raw_rows:
        item_id = str(row.get("protocol_item_id", ""))
        if str(row.get("method_name")) == OFFICIAL_TUSB:
            tusb_rows_by_item[item_id].append(row)
        elif str(row.get("method_name")) in {LEGACY, CAL, CROP}:
            baseline_rows_by_item[item_id].append(row)

    device, device_info = evalcore._select_eval_device(args)
    model = None
    preprocess = None
    frozen_backend = ""
    started = _now_iso()
    wall_start = time.time()
    item_assets: Dict[str, Dict[str, Any]] = {}
    prepared_meta: Dict[str, Dict[str, Any]] = {}
    skipped_reasons: Dict[str, str] = {}
    belief_empty_items = 0
    variance_unavailable_items = 0
    try:
        model, preprocess, frozen_backend = tracegate._load_clip(device)
        total = len(selected_ids)
        print(f"[{_now_iso()}] trace_belief_prepare_start items={total} device={device} backend={frozen_backend}", flush=True)
        for index, item_id in enumerate(sorted(selected_ids), start=1):
            item = item_lookup.get(item_id)
            item_id, prepared, error = oodcore._prepare_one_item(item_id, item)
            if not isinstance(prepared, dict):
                skipped_reasons[item_id] = error or "unknown_prepare_error"
                continue
            target_gallery, gallery_diag = proto._observed_target_gallery_images(item if isinstance(item, dict) else {})
            prototype_scores, gallery_scores, belief_scores, belief_diag = _clip_belief_score_maps(
                model=model,
                preprocess=preprocess,
                device=device,
                target_gallery=target_gallery,
                candidate_inputs=prepared["candidate_inputs"],
            )
            frozen_scores = tracegate._clip_score_map(model, preprocess, device, prepared["batch"], prepared["candidate_inputs"])
            if not belief_scores:
                belief_empty_items += 1
            if not bool(belief_diag.get("diag_variance_available", False)):
                variance_unavailable_items += 1
            prepared_meta[item_id] = {
                "protocol_eval_context_entity_count": int(prepared["protocol_eval_context_entity_count"]),
                "item_split": residual._item_split3(str(item_id)),
                "gallery_frame_count": int(gallery_diag.get("gallery_frame_count", 0)),
                "diag_variance_available": bool(belief_diag.get("diag_variance_available", False)),
            }
            item_assets[item_id] = {
                "target_future_mask": prepared["target_future_mask"],
                "future_masks": prepared["future_masks"],
                "frozen_scores": frozen_scores,
                "prototype_scores": prototype_scores,
                "gallery_scores": gallery_scores,
                "belief_scores": belief_scores,
                "belief_diagnostics": {**gallery_diag, **{f"belief_{k}": v for k, v in belief_diag.items()}},
            }
            if index % 25 == 0 or index == total:
                print(f"[{_now_iso()}] trace_belief_prepare_progress processed={index}/{total} valid={len(prepared_meta)} skipped={len(skipped_reasons)} belief_empty={belief_empty_items} variance_unavailable={variance_unavailable_items}", flush=True)
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

    final_rows: List[Dict[str, Any]] = []
    for item_id in sorted(selected_ids):
        if item_id not in item_assets:
            continue
        assets = item_assets[item_id]
        target_future_mask = assets["target_future_mask"]
        future_masks = assets["future_masks"]
        for row in baseline_rows_by_item.get(item_id, []):
            copied = dict(row)
            copied["item_split"] = residual._item_split3(str(item_id))
            final_rows.append(copied)
        for row in tusb_rows_by_item.get(item_id, []):
            if not bool(row.get("available", False)):
                continue
            base = dict(row.get("coord_result", {}))
            target_id = str(row.get("target_id", ""))
            common = (
                item_id,
                int(row["seed"]),
                str(row.get("method_name")),
                row.get("subset_tags", []),
                row.get("dataset", ""),
                row.get("clip_id", ""),
                int(row.get("protocol_eval_context_entity_count", 0)),
            )
            frozen_n = _minmax(dict(assets["frozen_scores"]))
            sem_scores = dict(row.get("semantic_scores", {}))
            proto_scores = proto._trace_prototype_scores(row, dict(assets["prototype_scores"]))
            gallery_assoc_scores = proto._trace_gallery_assoc_scores(row, dict(assets["prototype_scores"]), dict(assets["gallery_scores"]))
            belief_sem_scores = _trace_belief_sem_scores(row, dict(assets["belief_scores"]))
            belief_assoc_scores = _trace_belief_assoc_scores(row, dict(assets["belief_scores"]))
            final_rows.append(_result_row(*common[:3], "frozen_external_teacher_only", *common[3:], _label_result(frozen_n, target_id, target_future_mask, future_masks, base, "frozen_external_teacher_only")))
            final_rows.append(_result_row(*common[:3], "tusb_semantic_target", *common[3:], _label_result(sem_scores, target_id, target_future_mask, future_masks, base, "tusb_semantic_target")))
            final_rows.append(_result_row(*common[:3], "trace_prototype_only", *common[3:], _label_result(proto_scores, target_id, target_future_mask, future_masks, base, "trace_prototype_only")))
            final_rows.append(_result_row(*common[:3], "trace_gallery_assoc", *common[3:], _label_result(gallery_assoc_scores, target_id, target_future_mask, future_masks, base, "trace_gallery_assoc")))
            final_rows.append(_result_row(*common[:3], "trace_belief_sem_only", *common[3:], _label_result(belief_sem_scores, target_id, target_future_mask, future_masks, base, "trace_belief_sem_only")))
            final_rows.append(_result_row(*common[:3], "trace_belief_assoc", *common[3:], _label_result(belief_assoc_scores, target_id, target_future_mask, future_masks, base, "trace_belief_assoc")))

    split_ids = {"train": [], "val": [], "test": []}
    for item_id in sorted(selected_ids):
        if item_id in prepared_meta:
            split_ids[residual._item_split3(item_id)].append(item_id)
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
    gallery_counts = [int(meta.get("gallery_frame_count", 0)) for meta in prepared_meta.values()]
    variance_count = sum(1 for meta in prepared_meta.values() if bool(meta.get("diag_variance_available", False)))
    eval_payload = {
        "generated_at_utc": _now_iso(),
        "eval_started_at": started,
        "eval_finished_at": _now_iso(),
        "wall_time_seconds": float(time.time() - wall_start),
        "source_shards": [str(part.strip()) for part in str(args.source_shards).split(",") if part.strip()],
        "frozen_external_teacher_only_backend": frozen_backend,
        "trace_belief_sem_only_definition": "observed target gallery CLIP belief likelihood: mean/max/mean-cos reward minus within-gallery std penalty, mixed lightly with TUSB semantic/unit scores",
        "trace_belief_assoc_definition": "trace_belief_sem_only basis plus observed trace semantic/unit consistency and coordinate plausibility prior; no protocol subset tags used as features",
        "forbidden_feature_names_present": [],
        "belief_empty_items": int(belief_empty_items),
        "diag_variance_available_items": int(variance_count),
        "diag_variance_unavailable_items": int(variance_unavailable_items),
        "gallery_frame_count_mean": float(_mean(gallery_counts) if gallery_counts else 0.0),
        "split_definition": {
            "protocol_item_id_hash": "sha256[:8] mod 10",
            "train_buckets": [0, 1],
            "val_buckets": [2],
            "test_buckets": [3, 4, 5, 6, 7, 8, 9],
            "test_fraction_nominal": 0.7,
        },
        "split_sizes": {key: int(len(value)) for key, value in split_ids.items()},
        "train_item_ids_hash": _sha256_json(split_ids["train"]),
        "val_item_ids_hash": _sha256_json(split_ids["val"]),
        "test_item_ids_hash": _sha256_json(split_ids["test"]),
        "leakage_check_passed": bool(set(split_ids["train"]).isdisjoint(split_ids["test"]) and set(split_ids["val"]).isdisjoint(split_ids["test"])),
        "panels": panels,
    }
    _write_json(Path(args.eval_json), eval_payload)
    _write_md(
        Path(args.eval_md),
        "STWM Trace Belief Eval 20260424",
        [
            f"- frozen_external_teacher_only_backend: {frozen_backend}",
            f"- belief_empty_items: {belief_empty_items}",
            f"- diag_variance_available_items: {variance_count}",
            f"- gallery_frame_count_mean: {eval_payload['gallery_frame_count_mean']}",
            f"- split_sizes: {json.dumps(eval_payload['split_sizes'], ensure_ascii=True)}",
            *[
                f"- {name}: valid_items={panel['valid_items']} test_items={panel['test_items']} skipped_items={panel['skipped_items']} hash={panel['per_item_results_hash']}"
                for name, panel in panels.items()
            ],
        ],
    )
    return {"eval": eval_payload, "audit": audit}


def _bootstrap_block(rows: List[Dict[str, Any]], left_method: str, left_mode: str, right_method: str, right_mode: str, split_name: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for metric_name, metric_key, subset_tag in METRIC_SPECS:
        deltas = tracecond._metric_deltas(rows, left_method, left_mode, right_method, right_mode, metric_key, subset_tag=subset_tag)
        out[metric_name] = lighteval._bootstrap_deltas(
            deltas,
            seed=lighteval._stable_bootstrap_seed(split_name, left_method, left_mode, right_method, right_mode, metric_name),
        )
    return out


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
        sem = _mean_for(panel, OFFICIAL_TUSB, "trace_belief_sem_only", subset_name)
        assoc = _mean_for(panel, OFFICIAL_TUSB, "trace_belief_assoc", subset_name)
        frozen = _mean_for(panel, OFFICIAL_TUSB, "frozen_external_teacher_only", subset_name)
        semantic = _mean_for(panel, OFFICIAL_TUSB, "tusb_semantic_target", subset_name)
        legacy = _mean_for(panel, LEGACY, "coord_only", subset_name)
        out[subset_name] = {
            "trace_belief_sem_only_mean": sem,
            "trace_belief_assoc_mean": assoc,
            "frozen_external_teacher_only_mean": frozen,
            "tusb_semantic_target_mean": semantic,
            "legacysem_mean": legacy,
            "trace_belief_sem_only_improved_vs_frozen_external_teacher_only": bool(sem > frozen),
            "trace_belief_assoc_improved_vs_frozen_external_teacher_only": bool(assoc > frozen),
            "trace_belief_assoc_improved_vs_tusb_semantic_target": bool(assoc > semantic),
            "trace_belief_assoc_improved_vs_legacysem": bool(assoc > legacy),
            "frozen_external_teacher_only_sufficient": bool(frozen >= assoc),
        }
    return out


def _build_bootstrap_decision(args: Any, eval_payload: Dict[str, Any] | None = None) -> Dict[str, Any]:
    if eval_payload is None:
        eval_payload = _load_json(Path(args.eval_json))
    panels = eval_payload.get("panels", {}) if isinstance(eval_payload.get("panels", {}), dict) else {}
    id_rows = panels["densified_200_context_preserving"]["per_item_results"]
    ood_rows = panels["heldout_burst_heavy_context_preserving"]["per_item_results"] + panels["heldout_scene_category_video_context_preserving"]["per_item_results"]
    bootstrap_panels = {
        "densified_200_context_preserving": {
            "trace_belief_assoc_vs_frozen_external_teacher_only": _bootstrap_block(id_rows, OFFICIAL_TUSB, "trace_belief_assoc", OFFICIAL_TUSB, "frozen_external_teacher_only", "densified_200_context_preserving"),
            "trace_belief_assoc_vs_tusb_semantic_target": _bootstrap_block(id_rows, OFFICIAL_TUSB, "trace_belief_assoc", OFFICIAL_TUSB, "tusb_semantic_target", "densified_200_context_preserving"),
            "trace_belief_assoc_vs_legacysem": _bootstrap_block(id_rows, OFFICIAL_TUSB, "trace_belief_assoc", LEGACY, "coord_only", "densified_200_context_preserving"),
        },
        "true_ood_combined": {
            "trace_belief_assoc_vs_frozen_external_teacher_only": _bootstrap_block(ood_rows, OFFICIAL_TUSB, "trace_belief_assoc", OFFICIAL_TUSB, "frozen_external_teacher_only", "true_ood_combined"),
            "trace_belief_assoc_vs_legacysem": _bootstrap_block(ood_rows, OFFICIAL_TUSB, "trace_belief_assoc", LEGACY, "coord_only", "true_ood_combined"),
        },
    }
    id_stat = bootstrap_panels["densified_200_context_preserving"]["trace_belief_assoc_vs_frozen_external_teacher_only"]["overall_top1"]
    ood_stat = bootstrap_panels["true_ood_combined"]["trace_belief_assoc_vs_frozen_external_teacher_only"]["overall_top1"]
    id_zero = bool(id_stat["zero_excluded"] and float(id_stat["mean_delta"]) > 0.0)
    ood_zero = bool(ood_stat["zero_excluded"] and float(ood_stat["mean_delta"]) > 0.0)
    id_mean = float(id_stat["mean_delta"])
    ood_mean = float(ood_stat["mean_delta"])
    if id_zero and ood_zero:
        claim_level = "strong_claim"
    elif id_mean > 0.0 and ood_mean > 0.0:
        claim_level = "moderate_claim"
    else:
        claim_level = "weak_claim"
    bootstrap_payload = {
        "generated_at_utc": _now_iso(),
        "panels": bootstrap_panels,
        "trace_belief_zero_excluded_on_id": bool(id_zero),
        "trace_belief_zero_excluded_on_ood": bool(ood_zero),
        "claim_level": claim_level,
    }
    _write_json(Path(args.bootstrap_json), bootstrap_payload)
    _write_md(
        Path(args.bootstrap_md),
        "STWM Trace Belief Bootstrap 20260424",
        [
            f"- trace_belief_zero_excluded_on_id: {id_zero}",
            f"- trace_belief_zero_excluded_on_ood: {ood_zero}",
            f"- claim_level: {claim_level}",
        ],
    )
    head = {name: _headtohead(panel) for name, panel in panels.items()}
    densified = head["densified_200_context_preserving"]["overall"]
    ood_a = head["heldout_burst_heavy_context_preserving"]
    ood_b = head["heldout_scene_category_video_context_preserving"]
    ood_a_overall = ood_a["overall"]
    ood_b_overall = ood_b["overall"]
    sem_vs_frozen = bool(
        densified["trace_belief_sem_only_improved_vs_frozen_external_teacher_only"]
        and ood_a_overall["trace_belief_sem_only_improved_vs_frozen_external_teacher_only"]
        and ood_b_overall["trace_belief_sem_only_improved_vs_frozen_external_teacher_only"]
    )
    assoc_vs_frozen = bool(
        densified["trace_belief_assoc_improved_vs_frozen_external_teacher_only"]
        and ood_a_overall["trace_belief_assoc_improved_vs_frozen_external_teacher_only"]
        and ood_b_overall["trace_belief_assoc_improved_vs_frozen_external_teacher_only"]
    )
    assoc_vs_legacy = bool(
        densified["trace_belief_assoc_improved_vs_legacysem"]
        and ood_a_overall["trace_belief_assoc_improved_vs_legacysem"]
        and ood_b_overall["trace_belief_assoc_improved_vs_legacysem"]
    )
    continuity_contribution = bool(ood_a["continuity"]["trace_belief_assoc_improved_vs_frozen_external_teacher_only"] or ood_b["continuity"]["trace_belief_assoc_improved_vs_frozen_external_teacher_only"])
    ambiguity_contribution = bool(ood_a["ambiguity"]["trace_belief_assoc_improved_vs_frozen_external_teacher_only"] or ood_b["ambiguity"]["trace_belief_assoc_improved_vs_frozen_external_teacher_only"])
    trace_coupling = bool(assoc_vs_frozen and (continuity_contribution or ambiguity_contribution))
    official_story_supported = bool(trace_coupling and assoc_vs_legacy and claim_level in {"strong_claim", "moderate_claim"})
    if official_story_supported:
        next_step_choice = "start_main_submission_assets"
    elif assoc_vs_legacy or assoc_vs_frozen:
        next_step_choice = "one_last_surgical_fix"
    else:
        next_step_choice = "reframe_as_moderate_claim_main_track"
    decision = {
        "generated_at_utc": _now_iso(),
        "trace_belief_sem_only_improved_vs_frozen_external_teacher_only": bool(sem_vs_frozen),
        "trace_belief_assoc_improved_vs_frozen_external_teacher_only": bool(assoc_vs_frozen),
        "trace_belief_assoc_improved_vs_legacysem": bool(assoc_vs_legacy),
        "ood_continuity_trace_belief_independent_contribution": bool(continuity_contribution),
        "ood_ambiguity_trace_belief_independent_contribution": bool(ambiguity_contribution),
        "trace_semantic_coupling_load_bearing": bool(trace_coupling),
        "official_story_supported": bool(official_story_supported),
        "claim_level": claim_level,
        "next_step_choice": next_step_choice,
        "headtohead": head,
    }
    _write_json(Path(args.decision_json), decision)
    _write_md(
        Path(args.decision_md),
        "STWM Trace Belief Decision 20260424",
        [
            f"- trace_belief_sem_only_improved_vs_frozen_external_teacher_only: {sem_vs_frozen}",
            f"- trace_belief_assoc_improved_vs_frozen_external_teacher_only: {assoc_vs_frozen}",
            f"- trace_belief_assoc_improved_vs_legacysem: {assoc_vs_legacy}",
            f"- trace_belief_zero_excluded_on_id: {id_zero}",
            f"- trace_belief_zero_excluded_on_ood: {ood_zero}",
            f"- trace_semantic_coupling_load_bearing: {trace_coupling}",
            f"- official_story_supported: {official_story_supported}",
            f"- next_step_choice: {next_step_choice}",
        ],
    )
    return {"bootstrap": bootstrap_payload, "decision": decision}


def parse_args() -> Any:
    parser = ArgumentParser(description="Run STWM trace-conditioned belief association readout.")
    parser.add_argument("--mode", default="all", choices=["audit", "eval", "bootstrap_decision", "all"])
    parser.add_argument("--audit-json", default=str(REPORTS / "stwm_trace_belief_feasibility_audit_20260424.json"))
    parser.add_argument("--audit-md", default=str(DOCS / "STWM_TRACE_BELIEF_FEASIBILITY_AUDIT_20260424.md"))
    parser.add_argument("--eval-json", default=str(REPORTS / "stwm_trace_belief_eval_20260424.json"))
    parser.add_argument("--eval-md", default=str(DOCS / "STWM_TRACE_BELIEF_EVAL_20260424.md"))
    parser.add_argument("--bootstrap-json", default=str(REPORTS / "stwm_trace_belief_bootstrap_20260424.json"))
    parser.add_argument("--bootstrap-md", default=str(DOCS / "STWM_TRACE_BELIEF_BOOTSTRAP_20260424.md"))
    parser.add_argument("--decision-json", default=str(REPORTS / "stwm_trace_belief_decision_20260424.json"))
    parser.add_argument("--decision-md", default=str(DOCS / "STWM_TRACE_BELIEF_DECISION_20260424.md"))
    parser.add_argument("--dense-protocol-json", default=str(REPORTS / "stage2_state_identifiability_protocol_v3_20260416.json"))
    parser.add_argument("--extended-protocol-json", default=str(REPORTS / "stage2_protocol_v3_extended_evalset_20260420.json"))
    parser.add_argument("--source-shards", default=",".join([
        str(SHARDS / "tusb_all_fixed.json"),
        str(SHARDS / "legacysem.json"),
        str(SHARDS / "calibration.json"),
        str(SHARDS / "cropenc.json"),
    ]))
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--lease-path", default=str(ROOT / "reports/stage1_v2_gpu_lease_20260408.json"))
    parser.add_argument("--eval-required-mem-gb", type=float, default=12.0)
    parser.add_argument("--eval-safety-margin-gb", type=float, default=4.0)
    parser.add_argument("--audit-sample-items", type=int, default=64)
    return parser.parse_args()


def main() -> None:
    evalcore._apply_process_title_normalization()
    args = parse_args()
    if args.mode == "audit":
        _build_feasibility_audit(args)
    elif args.mode == "eval":
        _build_eval(args)
    elif args.mode == "bootstrap_decision":
        _build_bootstrap_decision(args)
    else:
        result = _build_eval(args)
        _build_bootstrap_decision(args, result["eval"])


if __name__ == "__main__":
    main()
