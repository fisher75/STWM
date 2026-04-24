#!/usr/bin/env python3
from __future__ import annotations

from argparse import ArgumentParser
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple
import gc
import hashlib
import json
import os
import sys
import time

import numpy as np
import torch

for candidate in [
    Path("/raid/chen034/workspace/stwm/code"),
    Path("/home/chen034/workspace/stwm/code"),
]:
    if candidate.exists() and str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from stwm.tools import run_stage2_state_identifiability_eval_20260415 as evalcore
from stwm.tools import run_stwm_tusb_light_readout_eval_20260422 as lighteval
from stwm.tools import run_stwm_true_ood_eval_20260420 as oodcore


ROOT = Path("/raid/chen034/workspace/stwm")
REPORTS = ROOT / "reports"
DOCS = ROOT / "docs"

OFFICIAL_TUSB = "TUSB-v3.1::best_semantic_hard.pt"
CAL = "calibration-only::best.pt"
CROP = "cropenc::best.pt"
LEGACY = "legacysem::best.pt"
SEEDS = list(lighteval.SEEDS)
OOD_SPLITS = [
    "heldout_burst_heavy_context_preserving",
    "heldout_scene_category_video_context_preserving",
]
PRIMARY_SPLIT = "heldout_scene_category_video_context_preserving"
TUSB_MODES = [
    "coord_only",
    "unit_identity_only",
    "semantic_teacher_only",
    "coord_plus_teacher",
    "coord_plus_unit",
    "hybrid_light",
]
BASELINE_MODES = {
    CAL: "coord_only",
    CROP: "coord_only",
    LEGACY: "coord_only",
}


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


def _write_md(path: Path, title: str, lines: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = [f"# {title}", ""]
    body.extend(list(lines))
    path.write_text("\n".join(body).rstrip() + "\n", encoding="utf-8")


def _mean(values: Iterable[float]) -> float:
    vals = [float(v) for v in values]
    return float(sum(vals) / max(len(vals), 1))


def _std(values: Iterable[float]) -> float:
    vals = np.asarray([float(v) for v in values], dtype=np.float64)
    if vals.size <= 1:
        return 0.0
    return float(vals.std(ddof=0))


def _sha256_json(payload: Any) -> str:
    raw = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _official_weights() -> Dict[str, float]:
    payload = _load_json(REPORTS / "stwm_lightreadout_final_eval_20260422.json")
    weights = payload.get("official_tusb_selected_weights", {}) if isinstance(payload.get("official_tusb_selected_weights", {}), dict) else {}
    return {
        "alpha": float(weights.get("alpha", 0.5)),
        "beta": float(weights.get("beta", 0.4)),
        "gamma": float(weights.get("gamma", 0.2)),
    }


def _checkpoint_map(args: Any) -> Dict[str, Dict[int, Dict[str, str]]]:
    mapping = lighteval._load_checkpoint_map(
        Path(args.main_checkpoint_audit),
        Path(args.sidecar_checkpoint_audit),
    )
    return {
        OFFICIAL_TUSB: {int(seed): dict(mapping[lighteval.TUSB_SIDECAR][int(seed)]) for seed in SEEDS},
        CAL: {int(seed): dict(mapping[lighteval.CAL][int(seed)]) for seed in SEEDS},
        CROP: {int(seed): dict(mapping[lighteval.CROP][int(seed)]) for seed in SEEDS},
        LEGACY: {int(seed): dict(mapping[lighteval.LEGACY][int(seed)]) for seed in SEEDS},
    }


def _prepare_split_materialization(args: Any) -> Tuple[Dict[str, Any], Dict[str, set[str]], Dict[str, Dict[str, Any]]]:
    temp_args = SimpleNamespace(
        dense_protocol_json=str(args.dense_protocol_json),
        extended_protocol_json=str(args.extended_protocol_json),
        split_materialization_report=str(args.split_audit_json),
        split_materialization_doc=str(args.split_audit_md),
    )
    payload, panel_item_ids, item_lookup = oodcore._materialize_true_ood_splits(temp_args)
    return payload, panel_item_ids, item_lookup


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
) -> Dict[str, Any]:
    return {
        "protocol_item_id": str(protocol_item_id),
        "seed": int(seed),
        "method_name": str(method_name),
        "scoring_mode": str(scoring_mode),
        "subset_tags": list(subset_tags),
        "dataset": str(dataset),
        "clip_id": str(clip_id),
        "protocol_eval_context_entity_count": int(context_entity_count),
        **_lean_result(result),
    }


def _aggregate_rows(rows: List[Dict[str, Any]]) -> Dict[str, float]:
    if not rows:
        return {
            "overall_top1": 0.0,
            "hit_rate": 0.0,
            "localization_error": 1e9,
            "mask_iou_at_top1": 0.0,
            "hard_subset_top1": 0.0,
            "ambiguity_top1": 0.0,
            "appearance_change_top1": 0.0,
            "occlusion_reappearance_top1": 0.0,
            "long_gap_persistence_top1": 0.0,
            "small_object_top1": 0.0,
        }

    def _subset_mean(tag: str, key: str = "query_future_top1_acc") -> float:
        subset = [row for row in rows if tag in set(row.get("subset_tags", []))]
        return _mean(float(row.get(key, 0.0)) for row in subset) if subset else 0.0

    hard_rows = [row for row in rows if row.get("subset_tags")]
    return {
        "overall_top1": _mean(float(row.get("query_future_top1_acc", 0.0)) for row in rows),
        "hit_rate": _mean(float(row.get("query_future_hit_rate", 0.0)) for row in rows),
        "localization_error": _mean(float(row.get("query_future_localization_error", 0.0)) for row in rows),
        "mask_iou_at_top1": _mean(float(row.get("future_mask_iou_at_top1", 0.0)) for row in rows),
        "hard_subset_top1": _mean(float(row.get("query_future_top1_acc", 0.0)) for row in hard_rows) if hard_rows else 0.0,
        "ambiguity_top1": _subset_mean("crossing_ambiguity"),
        "appearance_change_top1": _subset_mean("appearance_change"),
        "occlusion_reappearance_top1": _subset_mean("occlusion_reappearance"),
        "long_gap_persistence_top1": _subset_mean("long_gap_persistence"),
        "small_object_top1": _subset_mean("small_object"),
    }


def _seed_table(rows: List[Dict[str, Any]], method_name: str, scoring_mode: str) -> Dict[str, Any]:
    seed_rows: List[Dict[str, Any]] = []
    for seed in SEEDS:
        picked = [
            row
            for row in rows
            if str(row.get("method_name")) == method_name
            and str(row.get("scoring_mode")) == scoring_mode
            and int(row.get("seed", -1)) == int(seed)
        ]
        metrics = _aggregate_rows(picked)
        seed_row = {"seed": int(seed)}
        seed_row.update(metrics)
        seed_rows.append(seed_row)
    metric_keys = [key for key in seed_rows[0].keys() if key != "seed"] if seed_rows else []
    return {
        "seed_rows": seed_rows,
        "mean": {key: _mean(row[key] for row in seed_rows) for key in metric_keys},
        "std": {key: _std(row[key] for row in seed_rows) for key in metric_keys},
    }


def _subset_overall_mean(rows: List[Dict[str, Any]], method_name: str, scoring_mode: str, selector: str) -> float:
    method_rows = [
        row
        for row in rows
        if str(row.get("method_name")) == method_name and str(row.get("scoring_mode")) == scoring_mode
    ]
    if selector == "continuity_cases_only":
        method_rows = [
            row
            for row in method_rows
            if ("occlusion_reappearance" in set(row.get("subset_tags", []))) or ("long_gap_persistence" in set(row.get("subset_tags", [])))
        ]
    elif selector == "ambiguity_cases_only":
        method_rows = [row for row in method_rows if "crossing_ambiguity" in set(row.get("subset_tags", []))]
    return _mean(float(row.get("query_future_top1_acc", 0.0)) for row in method_rows) if method_rows else 0.0


def _metric_deltas(
    rows: List[Dict[str, Any]],
    left_method: str,
    left_mode: str,
    right_method: str,
    right_mode: str,
    metric_key: str,
    subset_tag: str = "",
) -> List[float]:
    deltas: List[float] = []
    for left_row in rows:
        if str(left_row.get("method_name")) != left_method or str(left_row.get("scoring_mode")) != left_mode:
            continue
        tags = set(left_row.get("subset_tags", []))
        if subset_tag == "__hard__" and not tags:
            continue
        if subset_tag and subset_tag != "__hard__" and subset_tag not in tags:
            continue
        match = next(
            (
                row
                for row in rows
                if int(row.get("seed", -1)) == int(left_row.get("seed", -1))
                and str(row.get("protocol_item_id", "")) == str(left_row.get("protocol_item_id", ""))
                and str(row.get("method_name")) == right_method
                and str(row.get("scoring_mode")) == right_mode
            ),
            None,
        )
        if isinstance(match, dict):
            deltas.append(float(left_row.get(metric_key, 0.0)) - float(match.get(metric_key, 0.0)))
    return deltas


def _bootstrap_block(
    rows: List[Dict[str, Any]],
    left_method: str,
    left_mode: str,
    right_method: str,
    right_mode: str,
    split_name: str,
) -> Dict[str, Any]:
    metrics = {}
    for metric_name, metric_key, subset_tag in [
        ("overall_top1", "query_future_top1_acc", ""),
        ("hard_subset_top1", "query_future_top1_acc", "__hard__"),
        ("ambiguity_top1", "query_future_top1_acc", "crossing_ambiguity"),
        ("occlusion_reappearance_top1", "query_future_top1_acc", "occlusion_reappearance"),
        ("long_gap_persistence_top1", "query_future_top1_acc", "long_gap_persistence"),
    ]:
        deltas = _metric_deltas(
            rows,
            left_method,
            left_mode,
            right_method,
            right_mode,
            metric_key=metric_key,
            subset_tag=subset_tag,
        )
        metrics[metric_name] = lighteval._bootstrap_deltas(
            deltas,
            seed=lighteval._stable_bootstrap_seed(split_name, left_method, left_mode, right_method, right_mode, metric_name),
        )
    return metrics


def _build_split_panel(
    split_name: str,
    split_meta: Dict[str, Any],
    split_ids: set[str],
    prepared_items: Mapping[str, Dict[str, Any]],
    skipped_reasons: Mapping[str, str],
    rows: List[Dict[str, Any]],
) -> Dict[str, Any]:
    valid_ids = {item_id for item_id in split_ids if item_id in prepared_items}
    panel_rows = [row for row in rows if str(row.get("protocol_item_id", "")) in valid_ids]
    missing_ids = sorted(item_id for item_id in split_ids if item_id not in prepared_items)
    skipped_counts = Counter(str(skipped_reasons.get(item_id, "missing_from_item_source")) for item_id in missing_ids)
    context_mean = _mean(
        int(prepared_items[item_id]["protocol_eval_context_entity_count"])
        for item_id in valid_ids
    ) if valid_ids else 0.0
    per_method_seed_results = {
        OFFICIAL_TUSB: {
            mode: _seed_table(panel_rows, OFFICIAL_TUSB, mode)
            for mode in TUSB_MODES
        },
        CAL: {"coord_only": _seed_table(panel_rows, CAL, "coord_only")},
        CROP: {"coord_only": _seed_table(panel_rows, CROP, "coord_only")},
        LEGACY: {"coord_only": _seed_table(panel_rows, LEGACY, "coord_only")},
    }
    return {
        "panel_name": split_name,
        "total_items": int(split_meta.get("item_count", len(split_ids))),
        "valid_items": int(len(valid_ids)),
        "skipped_items": int(len(missing_ids)),
        "skipped_reason_counts": dict(sorted(skipped_counts.items())),
        "protocol_eval_context_entity_count_mean": float(context_mean),
        "leakage_check_passed": bool(split_meta.get("leakage_check_passed", False)),
        "per_item_results_hash": _sha256_json(panel_rows),
        "per_item_results": panel_rows,
        "per_method_seed_results": per_method_seed_results,
        "exact_blocking_reason": str(split_meta.get("exact_blocking_reason", "")),
    }


def _build_headtohead_for_rows(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    comparisons = {}
    for selector in ["overall", "continuity_cases_only", "ambiguity_cases_only"]:
        hybrid = _subset_overall_mean(rows, OFFICIAL_TUSB, "hybrid_light", selector)
        teacher = _subset_overall_mean(rows, OFFICIAL_TUSB, "semantic_teacher_only", selector)
        unit = _subset_overall_mean(rows, OFFICIAL_TUSB, "unit_identity_only", selector)
        coord_teacher = _subset_overall_mean(rows, OFFICIAL_TUSB, "coord_plus_teacher", selector)
        legacy = _subset_overall_mean(rows, LEGACY, "coord_only", selector)
        comparisons[selector] = {
            "hybrid_light_mean": float(hybrid),
            "semantic_teacher_only_mean": float(teacher),
            "unit_identity_only_mean": float(unit),
            "coord_plus_teacher_mean": float(coord_teacher),
            "legacysem_mean": float(legacy),
            "improved_vs_teacher_only": bool(float(hybrid) > float(teacher)),
            "improved_vs_unit_identity_only": bool(float(hybrid) > float(unit)),
            "improved_vs_coord_plus_teacher": bool(float(hybrid) > float(coord_teacher)),
            "improved_vs_legacysem": bool(float(hybrid) > float(legacy)),
            "teacher_only_vs_legacysem": bool(float(teacher) > float(legacy)),
            "unit_identity_only_vs_legacysem": bool(float(unit) > float(legacy)),
        }
    return comparisons


def build_reports(args: Any) -> Dict[str, Any]:
    audit = {
        "generated_at_utc": _now_iso(),
        "official_tusb_checkpoint": "best_semantic_hard.pt",
        "official_tusb_scoring_mode": "hybrid_light",
        "attribution_scoring_modes": list(TUSB_MODES),
        "baselines_scoring_modes": dict(BASELINE_MODES),
        "audit_passed": True,
        "exact_blocking_reason": "",
    }
    _write_json(Path(args.audit_json), audit)
    _write_md(
        Path(args.audit_md),
        "STWM True OOD Attribution Audit 20260423",
        [
            f"- official_tusb_checkpoint: {audit['official_tusb_checkpoint']}",
            f"- official_tusb_scoring_mode: {audit['official_tusb_scoring_mode']}",
            f"- attribution_scoring_modes: {json.dumps(audit['attribution_scoring_modes'], ensure_ascii=True)}",
            f"- baselines_scoring_modes: {json.dumps(audit['baselines_scoring_modes'], ensure_ascii=True)}",
            f"- audit_passed: {audit['audit_passed']}",
        ],
    )

    materialization, panel_item_ids, extended_lookup = _prepare_split_materialization(args)
    true_ood_materialized = bool(materialization.get("true_ood_materialized", False))
    if not true_ood_materialized:
        split_payload = {
            "generated_at_utc": _now_iso(),
            "splits": {},
            "true_ood_materialized": False,
            "exact_blocking_reason": (
                materialization.get("split_b_scene_category_video_heldout", {}).get("exact_blocking_reason")
                or materialization.get("split_a_vipseg_history_to_burst_heldout", {}).get("exact_blocking_reason")
                or "true_ood_materialization_failed"
            ),
        }
        _write_json(Path(args.split_audit_json), split_payload)
        _write_md(Path(args.split_audit_md), "STWM True OOD Attribution Split Audit 20260423", [f"- exact_blocking_reason: {split_payload['exact_blocking_reason']}"])
        eval_payload = {"generated_at_utc": _now_iso(), "splits": {}, "exact_blocking_reason": split_payload["exact_blocking_reason"]}
        bootstrap = {"generated_at_utc": _now_iso(), "splits": {}, "ood_trace_semantic_coupling_zero_excluded": False, "ood_teacher_only_sufficient": False, "ood_claim_level": "weak_claim", "exact_blocking_reason": split_payload["exact_blocking_reason"]}
        decision = {
            "generated_at_utc": _now_iso(),
            "teacher_only_sufficient_on_true_ood": False,
            "continuity_hybrid_improved_vs_teacher_only": False,
            "ambiguity_hybrid_improved_vs_teacher_only": False,
            "hybrid_light_improved_vs_legacysem": False,
            "trace_semantic_coupling_load_bearing_on_true_ood": False,
            "official_story_supported": False,
            "next_step_choice": "one_last_surgical_fix",
            "exact_blocking_reason": split_payload["exact_blocking_reason"],
        }
        _write_json(Path(args.eval_json), eval_payload)
        _write_json(Path(args.bootstrap_json), bootstrap)
        _write_json(Path(args.decision_json), decision)
        _write_md(Path(args.eval_md), "STWM True OOD Attribution Eval 20260423", [f"- exact_blocking_reason: {split_payload['exact_blocking_reason']}"])
        _write_md(Path(args.bootstrap_md), "STWM True OOD Attribution Bootstrap 20260423", [f"- exact_blocking_reason: {split_payload['exact_blocking_reason']}"])
        _write_md(Path(args.decision_md), "STWM True OOD Attribution Decision 20260423", [f"- exact_blocking_reason: {split_payload['exact_blocking_reason']}"])
        return {"audit": audit, "split_audit": split_payload, "eval": eval_payload, "bootstrap": bootstrap, "decision": decision}

    selected_ids = set().union(*(panel_item_ids[name] for name in OOD_SPLITS))
    prepared_items, skipped_reasons = oodcore._prepare_selected_items(extended_lookup, selected_ids)
    checkpoint_map = _checkpoint_map(args)
    official_weights = _official_weights()

    raw_rows: List[Dict[str, Any]] = []
    eval_started_at = _now_iso()
    wall_start = time.time()
    device, device_info = evalcore._select_eval_device(args)
    print(f"[{_now_iso()}] device_ready mode={device_info.get('mode', '')} device={device}", flush=True)
    try:
        for seed in SEEDS:
            entry = checkpoint_map[OFFICIAL_TUSB][int(seed)]
            print(f"[{_now_iso()}] eval_start method={OFFICIAL_TUSB} seed={seed}", flush=True)
            spec = evalcore.MethodSpec(
                name=OFFICIAL_TUSB,
                run_name=str(entry["run_name"]),
                method_type="stage2",
                checkpoint_path=str(entry["checkpoint_path"]),
            )
            method = evalcore._load_method(spec, device=device)
            try:
                total_items = len(prepared_items)
                for index, protocol_item_id in enumerate(sorted(prepared_items), start=1):
                    prepared = prepared_items[protocol_item_id]
                    item = prepared["item"]
                    payload = evalcore._evaluate_tusb_light_readout_payload(
                        method=method,
                        item=item,
                        batch=prepared["batch"],
                        target_future_mask=prepared["target_future_mask"],
                        future_masks=prepared["future_masks"],
                        candidate_inputs=prepared["candidate_inputs"],
                        device=device,
                    )
                    coord_result = dict(payload.get("coord_result", {}))
                    coord_scores = dict(payload.get("coord_scores", {}))
                    unit_scores = dict(payload.get("unit_identity_scores", {}))
                    semantic_scores = dict(payload.get("semantic_teacher_scores", {}))
                    subset_tags = list(item.get("subset_tags", []))
                    dataset = str(item.get("dataset", ""))
                    clip_id = str(item.get("clip_id", ""))
                    ctx_count = int(prepared.get("protocol_eval_context_entity_count", 0))

                    raw_rows.append(_row(protocol_item_id, seed, OFFICIAL_TUSB, "coord_only", subset_tags, dataset, clip_id, ctx_count, coord_result))
                    raw_rows.append(
                        _row(
                            protocol_item_id,
                            seed,
                            OFFICIAL_TUSB,
                            "unit_identity_only",
                            subset_tags,
                            dataset,
                            clip_id,
                            ctx_count,
                            lighteval._compose_score_result(
                                base_result=coord_result,
                                score_map=unit_scores,
                                target_id=str(item.get("target_id", "")),
                                target_future_mask=prepared["target_future_mask"],
                                future_masks=prepared["future_masks"],
                                scoring_mode="unit_identity_only",
                                unit_scores=unit_scores,
                                semantic_scores=semantic_scores,
                            ),
                        )
                    )
                    raw_rows.append(
                        _row(
                            protocol_item_id,
                            seed,
                            OFFICIAL_TUSB,
                            "semantic_teacher_only",
                            subset_tags,
                            dataset,
                            clip_id,
                            ctx_count,
                            lighteval._compose_score_result(
                                base_result=coord_result,
                                score_map=semantic_scores,
                                target_id=str(item.get("target_id", "")),
                                target_future_mask=prepared["target_future_mask"],
                                future_masks=prepared["future_masks"],
                                scoring_mode="semantic_teacher_only",
                                unit_scores=unit_scores,
                                semantic_scores=semantic_scores,
                            ),
                        )
                    )
                    coord_plus_teacher = evalcore._build_hybrid_scores(
                        coord_scores=coord_scores,
                        unit_scores={},
                        semantic_scores=semantic_scores,
                        alpha=float(official_weights["alpha"]),
                        beta=0.0,
                        gamma=float(official_weights["gamma"]),
                    )
                    raw_rows.append(
                        _row(
                            protocol_item_id,
                            seed,
                            OFFICIAL_TUSB,
                            "coord_plus_teacher",
                            subset_tags,
                            dataset,
                            clip_id,
                            ctx_count,
                            lighteval._compose_score_result(
                                base_result=coord_result,
                                score_map=coord_plus_teacher,
                                target_id=str(item.get("target_id", "")),
                                target_future_mask=prepared["target_future_mask"],
                                future_masks=prepared["future_masks"],
                                scoring_mode="coord_plus_teacher",
                                selected_weights={"alpha": float(official_weights["alpha"]), "beta": 0.0, "gamma": float(official_weights["gamma"])},
                                unit_scores={},
                                semantic_scores=semantic_scores,
                            ),
                        )
                    )
                    coord_plus_unit = evalcore._build_hybrid_scores(
                        coord_scores=coord_scores,
                        unit_scores=unit_scores,
                        semantic_scores={},
                        alpha=float(official_weights["alpha"]),
                        beta=float(official_weights["beta"]),
                        gamma=0.0,
                    )
                    raw_rows.append(
                        _row(
                            protocol_item_id,
                            seed,
                            OFFICIAL_TUSB,
                            "coord_plus_unit",
                            subset_tags,
                            dataset,
                            clip_id,
                            ctx_count,
                            lighteval._compose_score_result(
                                base_result=coord_result,
                                score_map=coord_plus_unit,
                                target_id=str(item.get("target_id", "")),
                                target_future_mask=prepared["target_future_mask"],
                                future_masks=prepared["future_masks"],
                                scoring_mode="coord_plus_unit",
                                selected_weights={"alpha": float(official_weights["alpha"]), "beta": float(official_weights["beta"]), "gamma": 0.0},
                                unit_scores=unit_scores,
                                semantic_scores={},
                            ),
                        )
                    )
                    hybrid_scores = evalcore._build_hybrid_scores(
                        coord_scores=coord_scores,
                        unit_scores=unit_scores,
                        semantic_scores=semantic_scores,
                        alpha=float(official_weights["alpha"]),
                        beta=float(official_weights["beta"]),
                        gamma=float(official_weights["gamma"]),
                    )
                    raw_rows.append(
                        _row(
                            protocol_item_id,
                            seed,
                            OFFICIAL_TUSB,
                            "hybrid_light",
                            subset_tags,
                            dataset,
                            clip_id,
                            ctx_count,
                            lighteval._compose_score_result(
                                base_result=coord_result,
                                score_map=hybrid_scores,
                                target_id=str(item.get("target_id", "")),
                                target_future_mask=prepared["target_future_mask"],
                                future_masks=prepared["future_masks"],
                                scoring_mode="hybrid_light",
                                selected_weights=official_weights,
                                unit_scores=unit_scores,
                                semantic_scores=semantic_scores,
                            ),
                        )
                    )
                    if index % 50 == 0 or index == total_items:
                        print(f"[{_now_iso()}] eval_progress method={OFFICIAL_TUSB} seed={seed} items={index}/{total_items}", flush=True)
            finally:
                evalcore._release_method(method)
            print(f"[{_now_iso()}] eval_done method={OFFICIAL_TUSB} seed={seed}", flush=True)

        for method_name in [CAL, CROP, LEGACY]:
            for seed in SEEDS:
                entry = checkpoint_map[method_name][int(seed)]
                print(f"[{_now_iso()}] eval_start method={method_name} seed={seed}", flush=True)
                spec = evalcore.MethodSpec(
                    name=method_name,
                    run_name=str(entry["run_name"]),
                    method_type="stage2",
                    checkpoint_path=str(entry["checkpoint_path"]),
                )
                method = evalcore._load_method(spec, device=device)
                try:
                    total_items = len(prepared_items)
                    for index, protocol_item_id in enumerate(sorted(prepared_items), start=1):
                        prepared = prepared_items[protocol_item_id]
                        item = prepared["item"]
                        result = evalcore._evaluate_item(
                            method=method,
                            item=item,
                            batch=prepared["batch"],
                            target_future_mask=prepared["target_future_mask"],
                            future_masks=prepared["future_masks"],
                            device=device,
                            scoring_mode="coord_only",
                            candidate_inputs=prepared["candidate_inputs"],
                        )
                        raw_rows.append(
                            _row(
                                protocol_item_id,
                                seed,
                                method_name,
                                "coord_only",
                                list(item.get("subset_tags", [])),
                                str(item.get("dataset", "")),
                                str(item.get("clip_id", "")),
                                int(prepared.get("protocol_eval_context_entity_count", 0)),
                                result,
                            )
                        )
                        if index % 50 == 0 or index == total_items:
                            print(f"[{_now_iso()}] eval_progress method={method_name} seed={seed} items={index}/{total_items}", flush=True)
                finally:
                    evalcore._release_method(method)
                print(f"[{_now_iso()}] eval_done method={method_name} seed={seed}", flush=True)
    finally:
        lease_id = str(device_info.get("lease_id", "")).strip()
        if lease_id:
            try:
                evalcore.release_lease(lease_id=lease_id, lease_path=str(args.lease_path))
            except Exception:
                pass
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    eval_finished_at = _now_iso()
    wall_time_seconds = float(time.time() - wall_start)

    split_payload_map = {
        "heldout_burst_heavy_context_preserving": materialization.get("split_a_vipseg_history_to_burst_heldout", {}),
        "heldout_scene_category_video_context_preserving": materialization.get("split_b_scene_category_video_heldout", {}),
    }
    split_audit_splits: Dict[str, Any] = {}
    split_panels: Dict[str, Any] = {}
    for split_name in OOD_SPLITS:
        panel = _build_split_panel(
            split_name=split_name,
            split_meta=split_payload_map.get(split_name, {}),
            split_ids=panel_item_ids[split_name],
            prepared_items=prepared_items,
            skipped_reasons=skipped_reasons,
            rows=raw_rows,
        )
        split_panels[split_name] = panel
        split_audit_splits[split_name] = {
            "total_items": int(panel["total_items"]),
            "valid_items": int(panel["valid_items"]),
            "skipped_items": int(panel["skipped_items"]),
            "skipped_reason_counts": dict(panel["skipped_reason_counts"]),
            "protocol_eval_context_entity_count_mean": float(panel["protocol_eval_context_entity_count_mean"]),
            "leakage_check_passed": bool(panel["leakage_check_passed"]),
            "exact_blocking_reason": str(panel["exact_blocking_reason"]),
        }
    split_audit = {
        "generated_at_utc": _now_iso(),
        "true_ood_materialized": true_ood_materialized,
        "splits": split_audit_splits,
    }
    _write_json(Path(args.split_audit_json), split_audit)
    _write_md(
        Path(args.split_audit_md),
        "STWM True OOD Attribution Split Audit 20260423",
        [
            f"- true_ood_materialized: {split_audit['true_ood_materialized']}",
            *[
                f"- {name}: total={panel['total_items']} valid={panel['valid_items']} skipped={panel['skipped_items']} leakage_check_passed={panel['leakage_check_passed']} exact_blocking_reason={panel['exact_blocking_reason'] or 'none'}"
                for name, panel in split_audit_splits.items()
            ],
        ],
    )

    eval_payload = {
        "generated_at_utc": _now_iso(),
        "official_tusb_checkpoint": "best_semantic_hard.pt",
        "official_tusb_scoring_mode": "hybrid_light",
        "selected_hybrid_weights": official_weights,
        "selected_device": str(device),
        "device_info": device_info,
        "eval_started_at": eval_started_at,
        "eval_finished_at": eval_finished_at,
        "wall_time_seconds": wall_time_seconds,
        "splits": split_panels,
        "exact_blocking_reason": "",
    }
    _write_json(Path(args.eval_json), eval_payload)
    _write_md(
        Path(args.eval_md),
        "STWM True OOD Attribution Eval 20260423",
        [
            f"- official_tusb_checkpoint: {eval_payload['official_tusb_checkpoint']}",
            f"- official_tusb_scoring_mode: {eval_payload['official_tusb_scoring_mode']}",
            f"- eval_started_at: {eval_payload['eval_started_at']}",
            f"- eval_finished_at: {eval_payload['eval_finished_at']}",
            f"- wall_time_seconds: {eval_payload['wall_time_seconds']:.2f}",
            *[
                f"- {name}: valid_items={panel['valid_items']} skipped_items={panel['skipped_items']} per_item_results_hash={panel['per_item_results_hash']}"
                for name, panel in split_panels.items()
            ],
        ],
    )

    headtohead_splits = {
        split_name: _build_headtohead_for_rows(panel["per_item_results"])
        for split_name, panel in split_panels.items()
    }
    headtohead = {
        "generated_at_utc": _now_iso(),
        "splits": headtohead_splits,
    }
    _write_json(Path(args.headtohead_json), headtohead)
    _write_md(
        Path(args.headtohead_md),
        "STWM True OOD Attribution Head-to-Head 20260423",
        [
            *[
                f"- {split_name}.overall: hybrid_vs_teacher={payload['overall']['improved_vs_teacher_only']} hybrid_vs_unit={payload['overall']['improved_vs_unit_identity_only']} hybrid_vs_coord_plus_teacher={payload['overall']['improved_vs_coord_plus_teacher']} hybrid_vs_legacysem={payload['overall']['improved_vs_legacysem']}"
                for split_name, payload in headtohead_splits.items()
            ],
        ],
    )

    bootstrap_splits = {
        split_name: {
            "hybrid_vs_semantic_teacher_only": _bootstrap_block(
                panel["per_item_results"], OFFICIAL_TUSB, "hybrid_light", OFFICIAL_TUSB, "semantic_teacher_only", split_name
            ),
            "hybrid_vs_legacysem": _bootstrap_block(
                panel["per_item_results"], OFFICIAL_TUSB, "hybrid_light", LEGACY, "coord_only", split_name
            ),
            "semantic_teacher_only_vs_legacysem": _bootstrap_block(
                panel["per_item_results"], OFFICIAL_TUSB, "semantic_teacher_only", LEGACY, "coord_only", split_name
            ),
        }
        for split_name, panel in split_panels.items()
    }
    ood_trace_semantic_coupling_zero_excluded = all(
        bool(payload["hybrid_vs_semantic_teacher_only"]["overall_top1"]["zero_excluded"])
        and bool(payload["hybrid_vs_semantic_teacher_only"]["hard_subset_top1"]["zero_excluded"])
        for payload in bootstrap_splits.values()
    )
    teacher_only_sufficient = all(
        not payload["overall"]["improved_vs_teacher_only"]
        for payload in headtohead_splits.values()
    )
    if (
        all(payload["hybrid_vs_legacysem"]["overall_top1"]["zero_excluded"] and payload["hybrid_vs_semantic_teacher_only"]["overall_top1"]["zero_excluded"] for payload in bootstrap_splits.values())
        and all(payload["hybrid_vs_legacysem"]["hard_subset_top1"]["zero_excluded"] for payload in bootstrap_splits.values())
    ):
        ood_claim_level = "strong_claim"
    elif any(payload["overall"]["improved_vs_legacysem"] or payload["overall"]["improved_vs_teacher_only"] for payload in headtohead_splits.values()):
        ood_claim_level = "moderate_claim"
    else:
        ood_claim_level = "weak_claim"
    bootstrap = {
        "generated_at_utc": _now_iso(),
        "splits": bootstrap_splits,
        "ood_trace_semantic_coupling_zero_excluded": bool(ood_trace_semantic_coupling_zero_excluded),
        "ood_teacher_only_sufficient": bool(teacher_only_sufficient),
        "ood_claim_level": ood_claim_level,
    }
    _write_json(Path(args.bootstrap_json), bootstrap)
    _write_md(
        Path(args.bootstrap_md),
        "STWM True OOD Attribution Bootstrap 20260423",
        [
            f"- ood_trace_semantic_coupling_zero_excluded: {bootstrap['ood_trace_semantic_coupling_zero_excluded']}",
            f"- ood_teacher_only_sufficient: {bootstrap['ood_teacher_only_sufficient']}",
            f"- ood_claim_level: {bootstrap['ood_claim_level']}",
        ],
    )

    continuity_improved = all(payload["continuity_cases_only"]["improved_vs_teacher_only"] for payload in headtohead_splits.values())
    ambiguity_improved = all(payload["ambiguity_cases_only"]["improved_vs_teacher_only"] for payload in headtohead_splits.values())
    improved_vs_legacysem = all(payload["overall"]["improved_vs_legacysem"] for payload in headtohead_splits.values())
    trace_semantic_coupling_load_bearing = bool(
        not teacher_only_sufficient
        and continuity_improved
        and ambiguity_improved
        and ood_claim_level in {"strong_claim", "moderate_claim"}
    )
    official_story_supported = bool(
        trace_semantic_coupling_load_bearing
        and improved_vs_legacysem
    )
    if official_story_supported:
        next_step_choice = "start_main_submission_assets"
    elif improved_vs_legacysem or any(payload["overall"]["improved_vs_teacher_only"] for payload in headtohead_splits.values()):
        next_step_choice = "one_last_surgical_fix"
    else:
        next_step_choice = "reframe_as_moderate_claim_main_track"

    decision = {
        "generated_at_utc": _now_iso(),
        "teacher_only_sufficient_on_true_ood": bool(teacher_only_sufficient),
        "continuity_hybrid_improved_vs_teacher_only": bool(continuity_improved),
        "ambiguity_hybrid_improved_vs_teacher_only": bool(ambiguity_improved),
        "hybrid_light_improved_vs_legacysem": bool(improved_vs_legacysem),
        "trace_semantic_coupling_load_bearing_on_true_ood": bool(trace_semantic_coupling_load_bearing),
        "official_story_supported": bool(official_story_supported),
        "next_step_choice": next_step_choice,
        "exact_blocking_reason": "",
    }
    _write_json(Path(args.decision_json), decision)
    _write_md(
        Path(args.decision_md),
        "STWM True OOD Attribution Decision 20260423",
        [
            f"- teacher_only_sufficient_on_true_ood: {decision['teacher_only_sufficient_on_true_ood']}",
            f"- continuity_hybrid_improved_vs_teacher_only: {decision['continuity_hybrid_improved_vs_teacher_only']}",
            f"- ambiguity_hybrid_improved_vs_teacher_only: {decision['ambiguity_hybrid_improved_vs_teacher_only']}",
            f"- hybrid_light_improved_vs_legacysem: {decision['hybrid_light_improved_vs_legacysem']}",
            f"- trace_semantic_coupling_load_bearing_on_true_ood: {decision['trace_semantic_coupling_load_bearing_on_true_ood']}",
            f"- official_story_supported: {decision['official_story_supported']}",
            f"- next_step_choice: {decision['next_step_choice']}",
        ],
    )
    return {
        "audit": audit,
        "split_audit": split_audit,
        "eval": eval_payload,
        "headtohead": headtohead,
        "bootstrap": bootstrap,
        "decision": decision,
    }


def parse_args() -> Any:
    parser = ArgumentParser(description="Run STWM true-OOD attribution + teacher-only sanity.")
    parser.add_argument("--audit-json", default=str(REPORTS / "stwm_true_ood_attribution_audit_20260423.json"))
    parser.add_argument("--audit-md", default=str(DOCS / "STWM_TRUE_OOD_ATTRIBUTION_AUDIT_20260423.md"))
    parser.add_argument("--split-audit-json", default=str(REPORTS / "stwm_true_ood_attribution_split_audit_20260423.json"))
    parser.add_argument("--split-audit-md", default=str(DOCS / "STWM_TRUE_OOD_ATTRIBUTION_SPLIT_AUDIT_20260423.md"))
    parser.add_argument("--eval-json", default=str(REPORTS / "stwm_true_ood_attribution_eval_20260423.json"))
    parser.add_argument("--eval-md", default=str(DOCS / "STWM_TRUE_OOD_ATTRIBUTION_EVAL_20260423.md"))
    parser.add_argument("--headtohead-json", default=str(REPORTS / "stwm_true_ood_attribution_headtohead_20260423.json"))
    parser.add_argument("--headtohead-md", default=str(DOCS / "STWM_TRUE_OOD_ATTRIBUTION_HEADTOHEAD_20260423.md"))
    parser.add_argument("--bootstrap-json", default=str(REPORTS / "stwm_true_ood_attribution_bootstrap_20260423.json"))
    parser.add_argument("--bootstrap-md", default=str(DOCS / "STWM_TRUE_OOD_ATTRIBUTION_BOOTSTRAP_20260423.md"))
    parser.add_argument("--decision-json", default=str(REPORTS / "stwm_true_ood_attribution_decision_20260423.json"))
    parser.add_argument("--decision-md", default=str(DOCS / "STWM_TRUE_OOD_ATTRIBUTION_DECISION_20260423.md"))
    parser.add_argument("--dense-protocol-json", default=str(REPORTS / "stage2_state_identifiability_protocol_v3_20260416.json"))
    parser.add_argument("--extended-protocol-json", default=str(REPORTS / "stage2_protocol_v3_extended_evalset_20260420.json"))
    parser.add_argument("--main-checkpoint-audit", default=str(REPORTS / "stwm_postfix_matched6seed_checkpoint_audit_20260421.json"))
    parser.add_argument("--sidecar-checkpoint-audit", default=str(REPORTS / "stwm_sidecar_checkpoint_audit_20260422.json"))
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--lease-path", default=str(ROOT / "reports/stage1_v2_gpu_lease_20260408.json"))
    parser.add_argument("--eval-required-mem-gb", type=float, default=40.0)
    parser.add_argument("--eval-safety-margin-gb", type=float, default=8.0)
    return parser.parse_args()


def main() -> None:
    oodcore._apply_process_title_normalization()
    args = parse_args()
    build_reports(args)


if __name__ == "__main__":
    main()
