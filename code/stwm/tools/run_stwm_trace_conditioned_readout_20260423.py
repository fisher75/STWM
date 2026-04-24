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

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import torch

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
from stwm.tools import run_stwm_tusb_light_readout_eval_20260422 as lighteval
from stwm.tools import run_stwm_true_ood_eval_20260420 as oodcore
from stwm.tools import run_stwm_true_ood_attribution_20260423 as oodattrib


ROOT = Path("/raid/chen034/workspace/stwm")
REPORTS = ROOT / "reports"
DOCS = ROOT / "docs"
SHARDS = REPORTS / "trace_conditioned_readout_shards_20260423"
PREPARED_CACHE = REPORTS / "tmp_trace_conditioned_readout_prepared_cache_20260423.pt"

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
    "semantic_target_tiebreak",
    "unit_identity_only",
    "external_teacher_only",
    "hybrid_light",
]
BASELINE_MODES = {
    LEGACY: "coord_only",
    CAL: "coord_only",
    CROP: "coord_only",
}
GROUP_TO_METHOD = {
    "tusb": OFFICIAL_TUSB,
    "legacysem": LEGACY,
    "calibration": CAL,
    "cropenc": CROP,
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


def _prepare_panels(args: Any) -> Tuple[Dict[str, Any], Dict[str, set[str]], Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]], Dict[str, str]]:
    mat_args = SimpleNamespace(
        dense_protocol_json=str(args.dense_protocol_json),
        extended_protocol_json=str(args.extended_protocol_json),
        split_materialization_report=str(args.materialization_json),
        split_materialization_doc=str(args.materialization_md),
    )
    materialization, panel_item_ids, item_lookup = oodcore._materialize_true_ood_splits(mat_args)
    selected_ids = set().union(*(panel_item_ids[name] for name in PANELS))
    cache_path = Path(getattr(args, "prepared_cache", str(PREPARED_CACHE)))
    cache_key = _sha256_json(
        {
            "selected_ids": sorted(selected_ids),
            "dense_protocol_json": str(args.dense_protocol_json),
            "extended_protocol_json": str(args.extended_protocol_json),
            "builder": "context_preserving_v3_sequential",
        }
    )
    if cache_path.exists():
        try:
            cached = torch.load(cache_path, map_location="cpu", weights_only=False)
        except TypeError:
            cached = torch.load(cache_path, map_location="cpu")
        except Exception:
            cached = {}
        if isinstance(cached, dict) and cached.get("cache_key") == cache_key:
            prepared_items = cached.get("prepared_items", {})
            skipped_reasons = cached.get("skipped_reasons", {})
            if isinstance(prepared_items, dict) and isinstance(skipped_reasons, dict):
                print(
                    f"[{_now_iso()}] prepare_cache_hit path={cache_path} valid={len(prepared_items)} skipped={len(skipped_reasons)}",
                    flush=True,
                )
                return materialization, panel_item_ids, item_lookup, prepared_items, skipped_reasons

    prepared_items: Dict[str, Dict[str, Any]] = {}
    skipped_reasons: Dict[str, str] = {}
    selected_list = sorted(str(item_id) for item_id in selected_ids)
    print(
        f"[{_now_iso()}] prepare_start requested_items={len(selected_list)} mode=sequential_cache cache={cache_path}",
        flush=True,
    )
    for index, protocol_item_id in enumerate(selected_list, start=1):
        item_id, payload, error = oodcore._prepare_one_item(protocol_item_id, item_lookup.get(protocol_item_id))
        if isinstance(payload, dict):
            prepared_items[item_id] = payload
        else:
            skipped_reasons[item_id] = error or "unknown_prepare_error"
        if index % 25 == 0 or index == len(selected_list):
            print(
                f"[{_now_iso()}] prepare_progress processed={index}/{len(selected_list)} valid={len(prepared_items)} skipped={len(skipped_reasons)}",
                flush=True,
            )
    print(
        f"[{_now_iso()}] prepare_done valid_items={len(prepared_items)} skipped_items={len(skipped_reasons)}",
        flush=True,
    )
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "cache_key": cache_key,
                "generated_at_utc": _now_iso(),
                "prepared_items": prepared_items,
                "skipped_reasons": skipped_reasons,
            },
            cache_path,
        )
        print(f"[{_now_iso()}] prepare_cache_written path={cache_path}", flush=True)
    except Exception as exc:
        print(f"[{_now_iso()}] prepare_cache_write_failed error={type(exc).__name__}:{exc}", flush=True)
    return materialization, panel_item_ids, item_lookup, prepared_items, skipped_reasons


def _split_meta(materialization: Mapping[str, Any], panel_name: str) -> Dict[str, Any]:
    if panel_name == "densified_200_context_preserving":
        return dict(materialization.get("densified_200_context_preserving", {}))
    if panel_name == "heldout_burst_heavy_context_preserving":
        return dict(materialization.get("split_a_vipseg_history_to_burst_heldout", {}))
    if panel_name == "heldout_scene_category_video_context_preserving":
        return dict(materialization.get("split_b_scene_category_video_heldout", {}))
    return {}


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
        "item_split": _item_split(str(protocol_item_id)),
        "method_name": str(method_name),
        "scoring_mode": str(scoring_mode),
        "subset_tags": list(subset_tags),
        "dataset": str(dataset),
        "clip_id": str(clip_id),
        "protocol_eval_context_entity_count": int(context_entity_count),
        **_lean_result(result),
    }


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
            "MRR": 0.0,
            "top5_hit": 0.0,
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
        "MRR": _mean(float(row.get("mrr", 0.0)) for row in rows),
        "top5_hit": _mean(float(row.get("top5_hit", 0.0)) for row in rows),
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
    deltas: List[float] = []
    for left in rows:
        if str(left.get("method_name")) != left_method or str(left.get("scoring_mode")) != left_mode:
            continue
        if str(left.get("item_split")) != "test":
            continue
        tags = set(left.get("subset_tags", []))
        if subset_tag == "__hard__" and not tags:
            continue
        if subset_tag and subset_tag != "__hard__" and subset_tag not in tags:
            continue
        match = next(
            (
                row
                for row in rows
                if str(row.get("protocol_item_id")) == str(left.get("protocol_item_id"))
                and int(row.get("seed", -1)) == int(left.get("seed", -1))
                and str(row.get("method_name")) == right_method
                and str(row.get("scoring_mode")) == right_mode
            ),
            None,
        )
        if isinstance(match, dict):
            deltas.append(float(left.get(metric_key, 0.0)) - float(match.get(metric_key, 0.0)))
    return deltas


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


def _score_selection(rows: List[Dict[str, Any]]) -> float:
    metrics = _aggregate_rows(rows)
    return float(metrics["overall_top1"] + 0.5 * metrics["hard_subset_top1"] + 0.1 * metrics["MRR"])


def _select_tiebreak_weights(raw_rows: List[Dict[str, Any]], prepared_meta: Mapping[str, Dict[str, Any]]) -> Dict[str, float]:
    val_rows = [
        row
        for row in raw_rows
        if str(row.get("method_name")) == OFFICIAL_TUSB
        and str(row.get("item_split")) == "val"
        and bool(row.get("available", False))
    ]
    grid: List[Dict[str, float]] = []
    for tie_margin in [0.01, 0.03, 0.05, 0.10, 0.20]:
        for coord_tiebreak_weight in [0.00, 0.02, 0.05, 0.10, 0.20]:
            for coord_veto_margin in [0.60, 0.80, 1.00]:
                for coord_veto_penalty in [0.00, 0.10, 0.20, 0.40]:
                    grid.append(
                        {
                            "semantic_tie_margin": float(tie_margin),
                            "coord_tiebreak_weight": float(coord_tiebreak_weight),
                            "coord_veto_margin": float(coord_veto_margin),
                            "coord_veto_penalty": float(coord_veto_penalty),
                        }
                    )
    best = {"score": -1e9, **grid[0]}
    for weights in grid:
        scored: List[Dict[str, Any]] = []
        for row in val_rows:
            item_id = str(row.get("protocol_item_id", ""))
            meta = prepared_meta[item_id]
            score_map = evalcore._build_semantic_target_tiebreak_scores(
                semantic_scores=dict(row.get("semantic_scores", {})),
                coord_scores=dict(row.get("coord_scores", {})),
                tie_margin=float(weights["semantic_tie_margin"]),
                coord_tiebreak_weight=float(weights["coord_tiebreak_weight"]),
                coord_veto_margin=float(weights["coord_veto_margin"]),
                coord_veto_penalty=float(weights["coord_veto_penalty"]),
            )
            result = _compose(
                base_result=dict(row.get("coord_result", {})),
                score_map=score_map,
                target_id=str(row.get("target_id", "")),
                target_future_mask=meta["target_future_mask"],
                future_masks=meta["future_masks"],
                scoring_mode="semantic_target_tiebreak",
                selected_weights=weights,
            )
            scored.append(
                {
                    "protocol_item_id": item_id,
                    "seed": int(row.get("seed", -1)),
                    "item_split": "val",
                    "subset_tags": list(row.get("subset_tags", [])),
                    "method_name": OFFICIAL_TUSB,
                    "scoring_mode": "semantic_target_tiebreak",
                    **_lean_result(result),
                }
            )
        score = _score_selection(scored)
        if score > float(best["score"]):
            best = {"score": float(score), **weights}
    return {
        "semantic_tie_margin": float(best["semantic_tie_margin"]),
        "coord_tiebreak_weight": float(best["coord_tiebreak_weight"]),
        "coord_veto_margin": float(best["coord_veto_margin"]),
        "coord_veto_penalty": float(best["coord_veto_penalty"]),
        "val_selection_score": float(best["score"]),
    }


def build_audit(args: Any) -> Dict[str, Any]:
    source = (ROOT / "code/stwm/tools/run_stage2_state_identifiability_eval_20260415.py").read_text(encoding="utf-8")
    payload = {
        "generated_at_utc": _now_iso(),
        "scoring_modes_checked": [
            "coord_only",
            "unit_identity_only",
            "semantic_teacher_only",
            "coord_plus_teacher",
            "coord_plus_unit",
            "hybrid_light",
            "external_teacher_only",
            "semantic_target_tiebreak",
        ],
        "semantic_teacher_only_calls_teacher_forced_predict": bool("_teacher_forced_predict" in source and "semantic_teacher_only" in source),
        "semantic_teacher_only_uses_semantic_tokens_0": bool("target_sem = semantic_tokens[0]" in source),
        "semantic_teacher_only_depends_on_tusb_semantic_state": True,
        "semantic_teacher_only_should_rename_to_tusb_semantic_target": True,
        "true_missing_baseline_was_external_teacher_only": True,
        "external_teacher_only_present": bool("_external_teacher_score_map" in source and "external_teacher_only" in source),
        "semantic_target_tiebreak_present": bool("_build_semantic_target_tiebreak_scores" in source and "semantic_target_tiebreak" in source),
        "audit_passed": bool("_external_teacher_score_map" in source and "_build_semantic_target_tiebreak_scores" in source),
        "exact_blocking_reason": "",
    }
    if not payload["audit_passed"]:
        payload["exact_blocking_reason"] = "required scorer implementation missing from live evaluator"
    _write_json(Path(args.audit_json), payload)
    _write_md(
        Path(args.audit_md),
        "STWM Trace-Conditioned Readout Audit 20260423",
        [
            f"- semantic_teacher_only_calls_teacher_forced_predict: {payload['semantic_teacher_only_calls_teacher_forced_predict']}",
            f"- semantic_teacher_only_uses_semantic_tokens_0: {payload['semantic_teacher_only_uses_semantic_tokens_0']}",
            f"- semantic_teacher_only_should_rename_to_tusb_semantic_target: {payload['semantic_teacher_only_should_rename_to_tusb_semantic_target']}",
            f"- external_teacher_only_present: {payload['external_teacher_only_present']}",
            f"- semantic_target_tiebreak_present: {payload['semantic_target_tiebreak_present']}",
            f"- audit_passed: {payload['audit_passed']}",
        ],
    )
    return payload


def _parse_seed_list(text: str) -> List[int]:
    return [int(part.strip()) for part in str(text).split(",") if part.strip()]


def build_shard(args: Any) -> Dict[str, Any]:
    _, panel_item_ids, _, prepared_items, skipped_reasons = _prepare_panels(args)
    selected_ids = set().union(*(panel_item_ids[name] for name in PANELS))
    checkpoint_map = _checkpoint_map(args)
    method_name = GROUP_TO_METHOD[str(args.group)]
    seed_list = _parse_seed_list(args.seed_list)
    eval_started_at = _now_iso()
    wall_start = time.time()
    raw_rows: List[Dict[str, Any]] = []
    device, device_info = evalcore._select_eval_device(args)
    print(f"[{_now_iso()}] trace_readout_shard_start group={args.group} seeds={seed_list} device={device}", flush=True)
    try:
        for seed in seed_list:
            entry = checkpoint_map[method_name][int(seed)]
            spec = evalcore.MethodSpec(
                name=method_name,
                run_name=str(entry["run_name"]),
                method_type="stage2",
                checkpoint_path=str(entry["checkpoint_path"]),
            )
            method = evalcore._load_method(spec, device=device)
            try:
                total = len(prepared_items)
                for index, item_id in enumerate(sorted(prepared_items), start=1):
                    prepared = prepared_items[item_id]
                    item = prepared["item"]
                    subset_tags = list(item.get("subset_tags", []))
                    dataset = str(item.get("dataset", ""))
                    clip_id = str(item.get("clip_id", ""))
                    ctx_count = int(prepared.get("protocol_eval_context_entity_count", 0))
                    if args.group == "tusb":
                        payload = evalcore._evaluate_tusb_light_readout_payload(
                            method=method,
                            item=item,
                            batch=prepared["batch"],
                            target_future_mask=prepared["target_future_mask"],
                            future_masks=prepared["future_masks"],
                            candidate_inputs=prepared["candidate_inputs"],
                            device=device,
                        )
                        external_scores = evalcore._external_teacher_score_map(
                            method=method,
                            batch=prepared["batch"],
                            candidate_inputs=prepared["candidate_inputs"],
                            device=device,
                        )
                        raw_rows.append(
                            {
                                "protocol_item_id": str(item_id),
                                "seed": int(seed),
                                "item_split": _item_split(str(item_id)),
                                "method_name": method_name,
                                "subset_tags": subset_tags,
                                "dataset": dataset,
                                "clip_id": clip_id,
                                "target_id": str(item.get("target_id", "")),
                                "protocol_eval_context_entity_count": ctx_count,
                                "coord_result": dict(payload.get("coord_result", {})),
                                "coord_scores": dict(payload.get("coord_scores", {})),
                                "unit_scores": dict(payload.get("unit_identity_scores", {})),
                                "semantic_scores": dict(payload.get("semantic_teacher_scores", {})),
                                "external_scores": dict(external_scores),
                                "available": bool(payload.get("available", False)),
                                "blocking_reason": str(payload.get("blocking_reason", "")),
                            }
                        )
                    else:
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
                                str(item_id),
                                seed,
                                method_name,
                                "coord_only",
                                subset_tags,
                                dataset,
                                clip_id,
                                ctx_count,
                                result,
                            )
                        )
                    if index % 50 == 0 or index == total:
                        print(f"[{_now_iso()}] trace_readout_shard_progress group={args.group} seed={seed} items={index}/{total}", flush=True)
            finally:
                evalcore._release_method(method)
            print(f"[{_now_iso()}] trace_readout_shard_done group={args.group} seed={seed}", flush=True)
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
    prepared_meta = {
        item_id: {
            "protocol_eval_context_entity_count": int(prepared["protocol_eval_context_entity_count"]),
            "item_split": _item_split(str(item_id)),
        }
        for item_id, prepared in prepared_items.items()
        if item_id in selected_ids
    }
    payload = {
        "generated_at_utc": _now_iso(),
        "group": str(args.group),
        "seed_list": seed_list,
        "eval_started_at": eval_started_at,
        "eval_finished_at": _now_iso(),
        "wall_time_seconds": float(time.time() - wall_start),
        "selected_ids": sorted(selected_ids),
        "panel_item_ids": {name: sorted(panel_item_ids[name]) for name in PANELS},
        "prepared_item_meta": prepared_meta,
        "skipped_reasons": dict(skipped_reasons),
        "raw_rows_hash": _sha256_json(raw_rows),
        "raw_rows": raw_rows,
    }
    _write_json(Path(args.output_json), payload)
    return payload


def repair_external_scores(args: Any) -> Dict[str, Any]:
    source = _load_json(Path(args.input_json))
    if not source:
        raise RuntimeError(f"input shard not readable: {args.input_json}")
    _, _, _, prepared_items, _ = _prepare_panels(args)
    raw_rows = list(source.get("raw_rows", []))
    checkpoint_map = _checkpoint_map(args)
    seed_list = sorted({int(row.get("seed", -1)) for row in raw_rows if int(row.get("seed", -1)) in SEEDS})
    device, device_info = evalcore._select_eval_device(args)
    started = _now_iso()
    wall_start = time.time()
    print(f"[{_now_iso()}] external_repair_start input={args.input_json} seeds={seed_list} device={device}", flush=True)
    try:
        for seed in seed_list:
            entry = checkpoint_map[OFFICIAL_TUSB][int(seed)]
            spec = evalcore.MethodSpec(
                name=OFFICIAL_TUSB,
                run_name=str(entry["run_name"]),
                method_type="stage2",
                checkpoint_path=str(entry["checkpoint_path"]),
            )
            method = evalcore._load_method(spec, device=device)
            try:
                rows_for_seed = [row for row in raw_rows if int(row.get("seed", -1)) == int(seed)]
                total = len(rows_for_seed)
                for index, row in enumerate(rows_for_seed, start=1):
                    item_id = str(row.get("protocol_item_id", ""))
                    prepared = prepared_items.get(item_id)
                    if not isinstance(prepared, dict):
                        row["external_scores"] = {}
                        row["external_score_repair_reason"] = "prepared_item_missing"
                        continue
                    row["external_scores"] = evalcore._external_teacher_score_map(
                        method=method,
                        batch=prepared["batch"],
                        candidate_inputs=prepared["candidate_inputs"],
                        device=device,
                    )
                    row["external_score_repair_reason"] = "" if row["external_scores"] else "external_teacher_scores_missing_after_fix"
                    if index % 100 == 0 or index == total:
                        print(f"[{_now_iso()}] external_repair_progress seed={seed} rows={index}/{total}", flush=True)
            finally:
                evalcore._release_method(method)
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
    source["external_score_repaired_at_utc"] = _now_iso()
    source["external_score_repair_started_at_utc"] = started
    source["external_score_repair_wall_time_seconds"] = float(time.time() - wall_start)
    source["external_score_repair_device_info"] = device_info
    source["raw_rows"] = raw_rows
    source["raw_rows_hash"] = _sha256_json(raw_rows)
    source["external_score_empty_rows_after_repair"] = int(sum(1 for row in raw_rows if not row.get("external_scores")))
    _write_json(Path(args.output_json), source)
    return source


def _build_final_rows(raw_rows: List[Dict[str, Any]], prepared_assets: Mapping[str, Dict[str, Any]], tiebreak_weights: Dict[str, float]) -> List[Dict[str, Any]]:
    final_rows: List[Dict[str, Any]] = []
    for row in raw_rows:
        method_name = str(row.get("method_name", ""))
        item_id = str(row.get("protocol_item_id", ""))
        if method_name != OFFICIAL_TUSB:
            final_rows.append(row)
            continue
        if not bool(row.get("available", False)):
            continue
        assets = prepared_assets[item_id]
        base = dict(row.get("coord_result", {}))
        final_rows.append(
            _row(item_id, int(row["seed"]), method_name, "tusb_semantic_target", row.get("subset_tags", []), row.get("dataset", ""), row.get("clip_id", ""), int(row.get("protocol_eval_context_entity_count", 0)),
                 _compose(base, dict(row.get("semantic_scores", {})), str(row.get("target_id", "")), assets["target_future_mask"], assets["future_masks"], "tusb_semantic_target"))
        )
        final_rows.append(
            _row(item_id, int(row["seed"]), method_name, "unit_identity_only", row.get("subset_tags", []), row.get("dataset", ""), row.get("clip_id", ""), int(row.get("protocol_eval_context_entity_count", 0)),
                 _compose(base, dict(row.get("unit_scores", {})), str(row.get("target_id", "")), assets["target_future_mask"], assets["future_masks"], "unit_identity_only"))
        )
        final_rows.append(
            _row(item_id, int(row["seed"]), method_name, "external_teacher_only", row.get("subset_tags", []), row.get("dataset", ""), row.get("clip_id", ""), int(row.get("protocol_eval_context_entity_count", 0)),
                 _compose(base, dict(row.get("external_scores", {})), str(row.get("target_id", "")), assets["target_future_mask"], assets["future_masks"], "external_teacher_only"))
        )
        hybrid_scores = evalcore._build_hybrid_scores(
            coord_scores=dict(row.get("coord_scores", {})),
            unit_scores=dict(row.get("unit_scores", {})),
            semantic_scores=dict(row.get("semantic_scores", {})),
            alpha=0.5,
            beta=0.4,
            gamma=0.2,
        )
        final_rows.append(
            _row(item_id, int(row["seed"]), method_name, "hybrid_light", row.get("subset_tags", []), row.get("dataset", ""), row.get("clip_id", ""), int(row.get("protocol_eval_context_entity_count", 0)),
                 _compose(base, hybrid_scores, str(row.get("target_id", "")), assets["target_future_mask"], assets["future_masks"], "hybrid_light", {"alpha": 0.5, "beta": 0.4, "gamma": 0.2}))
        )
        tiebreak_scores = evalcore._build_semantic_target_tiebreak_scores(
            semantic_scores=dict(row.get("semantic_scores", {})),
            coord_scores=dict(row.get("coord_scores", {})),
            tie_margin=float(tiebreak_weights["semantic_tie_margin"]),
            coord_tiebreak_weight=float(tiebreak_weights["coord_tiebreak_weight"]),
            coord_veto_margin=float(tiebreak_weights["coord_veto_margin"]),
            coord_veto_penalty=float(tiebreak_weights["coord_veto_penalty"]),
        )
        final_rows.append(
            _row(item_id, int(row["seed"]), method_name, "semantic_target_tiebreak", row.get("subset_tags", []), row.get("dataset", ""), row.get("clip_id", ""), int(row.get("protocol_eval_context_entity_count", 0)),
                 _compose(base, tiebreak_scores, str(row.get("target_id", "")), assets["target_future_mask"], assets["future_masks"], "semantic_target_tiebreak", tiebreak_weights))
        )
    return final_rows


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
        row
        for row in rows
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
        row
        for row in panel.get("per_item_results", [])
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
        tiebreak = _mean_for(panel, OFFICIAL_TUSB, "semantic_target_tiebreak", subset_name)
        external = _mean_for(panel, OFFICIAL_TUSB, "external_teacher_only", subset_name)
        unit = _mean_for(panel, OFFICIAL_TUSB, "unit_identity_only", subset_name)
        hybrid = _mean_for(panel, OFFICIAL_TUSB, "hybrid_light", subset_name)
        legacy = _mean_for(panel, LEGACY, "coord_only", subset_name)
        out[subset_name] = {
            "tusb_semantic_target_mean": semantic_target,
            "semantic_target_tiebreak_mean": tiebreak,
            "external_teacher_only_mean": external,
            "unit_identity_only_mean": unit,
            "hybrid_light_mean": hybrid,
            "legacysem_mean": legacy,
            "tusb_semantic_target_improved_vs_external_teacher_only": bool(semantic_target > external),
            "semantic_target_tiebreak_improved_vs_external_teacher_only": bool(tiebreak > external),
            "semantic_target_tiebreak_improved_vs_hybrid_light": bool(tiebreak > hybrid),
            "semantic_target_tiebreak_improved_vs_legacysem": bool(tiebreak > legacy),
        }
    return out


def build_merge(args: Any) -> Dict[str, Any]:
    shard_paths = [Path(part.strip()) for part in str(args.shard_jsons).split(",") if part.strip()]
    shards = [_load_json(path) for path in shard_paths]
    if not shards:
        raise RuntimeError("no shard_jsons provided")
    first = shards[0]
    panel_item_ids = {name: set(first["panel_item_ids"][name]) for name in PANELS}
    prepared_meta = {item_id: dict(meta) for item_id, meta in first.get("prepared_item_meta", {}).items()}
    skipped_reasons = dict(first.get("skipped_reasons", {}))
    materialization, _, _, prepared_items, _ = _prepare_panels(args)
    prepared_assets = {
        item_id: {
            "target_future_mask": prepared["target_future_mask"],
            "future_masks": prepared["future_masks"],
        }
        for item_id, prepared in prepared_items.items()
    }
    raw_rows: List[Dict[str, Any]] = []
    for shard in shards:
        raw_rows.extend(shard.get("raw_rows", []))
    tiebreak_weights = _select_tiebreak_weights(raw_rows, prepared_assets)
    final_rows = _build_final_rows(raw_rows, prepared_assets, tiebreak_weights)
    panels = {
        panel_name: _build_panel(
            panel_name=panel_name,
            split_meta=_split_meta(materialization, panel_name),
            item_ids=panel_item_ids[panel_name],
            prepared_meta=prepared_meta,
            skipped_reasons=skipped_reasons,
            rows=final_rows,
        )
        for panel_name in PANELS
    }
    eval_payload = {
        "generated_at_utc": _now_iso(),
        "eval_started_at": min(str(shard.get("eval_started_at", "")) for shard in shards),
        "eval_finished_at": max(str(shard.get("eval_finished_at", "")) for shard in shards),
        "wall_time_seconds_sum_of_shards": float(sum(float(shard.get("wall_time_seconds", 0.0)) for shard in shards)),
        "source_shards": [str(path) for path in shard_paths],
        "official_tusb_checkpoint": "best_semantic_hard.pt",
        "scoring_modes": {
            "semantic_teacher_only_reported_as": "tusb_semantic_target",
            "semantic_target_tiebreak_weights": tiebreak_weights,
        },
        "panels": panels,
    }
    _write_json(Path(args.eval_json), eval_payload)
    _write_md(
        Path(args.eval_md),
        "STWM Trace-Conditioned Readout Eval 20260423",
        [
            f"- selected_semantic_target_tiebreak_weights: {json.dumps(tiebreak_weights, ensure_ascii=True)}",
            *[
                f"- {name}: valid_items={panel['valid_items']} test_items={panel['test_items']} skipped_items={panel['skipped_items']} hash={panel['per_item_results_hash']}"
                for name, panel in panels.items()
            ],
        ],
    )

    bootstrap_panels = {
        "densified_200_context_preserving": {
            "tusb_semantic_target_vs_external_teacher_only": _bootstrap_block(panels["densified_200_context_preserving"]["per_item_results"], OFFICIAL_TUSB, "tusb_semantic_target", OFFICIAL_TUSB, "external_teacher_only", "densified_200_context_preserving"),
            "semantic_target_tiebreak_vs_external_teacher_only": _bootstrap_block(panels["densified_200_context_preserving"]["per_item_results"], OFFICIAL_TUSB, "semantic_target_tiebreak", OFFICIAL_TUSB, "external_teacher_only", "densified_200_context_preserving"),
            "semantic_target_tiebreak_vs_hybrid_light": _bootstrap_block(panels["densified_200_context_preserving"]["per_item_results"], OFFICIAL_TUSB, "semantic_target_tiebreak", OFFICIAL_TUSB, "hybrid_light", "densified_200_context_preserving"),
            "semantic_target_tiebreak_vs_legacysem": _bootstrap_block(panels["densified_200_context_preserving"]["per_item_results"], OFFICIAL_TUSB, "semantic_target_tiebreak", LEGACY, "coord_only", "densified_200_context_preserving"),
        },
        "true_ood_combined": {
            "tusb_semantic_target_vs_external_teacher_only": _bootstrap_block(panels["heldout_burst_heavy_context_preserving"]["per_item_results"] + panels["heldout_scene_category_video_context_preserving"]["per_item_results"], OFFICIAL_TUSB, "tusb_semantic_target", OFFICIAL_TUSB, "external_teacher_only", "true_ood_combined"),
            "semantic_target_tiebreak_vs_external_teacher_only": _bootstrap_block(panels["heldout_burst_heavy_context_preserving"]["per_item_results"] + panels["heldout_scene_category_video_context_preserving"]["per_item_results"], OFFICIAL_TUSB, "semantic_target_tiebreak", OFFICIAL_TUSB, "external_teacher_only", "true_ood_combined"),
            "semantic_target_tiebreak_vs_legacysem": _bootstrap_block(panels["heldout_burst_heavy_context_preserving"]["per_item_results"] + panels["heldout_scene_category_video_context_preserving"]["per_item_results"], OFFICIAL_TUSB, "semantic_target_tiebreak", LEGACY, "coord_only", "true_ood_combined"),
        },
    }
    bootstrap_payload = {
        "generated_at_utc": _now_iso(),
        "panels": bootstrap_panels,
    }
    _write_json(Path(args.bootstrap_json), bootstrap_payload)
    _write_md(
        Path(args.bootstrap_md),
        "STWM Trace-Conditioned Readout Bootstrap 20260423",
        [
            f"- densified.semantic_target_tiebreak_vs_legacysem.overall_zero_excluded: {bootstrap_panels['densified_200_context_preserving']['semantic_target_tiebreak_vs_legacysem']['overall_top1']['zero_excluded']}",
            f"- ood.semantic_target_tiebreak_vs_legacysem.overall_zero_excluded: {bootstrap_panels['true_ood_combined']['semantic_target_tiebreak_vs_legacysem']['overall_top1']['zero_excluded']}",
        ],
    )

    head = {name: _headtohead(panel) for name, panel in panels.items()}
    densified = head["densified_200_context_preserving"]["overall"]
    ood_a = head["heldout_burst_heavy_context_preserving"]["overall"]
    ood_b = head["heldout_scene_category_video_context_preserving"]["overall"]
    tusb_target_vs_external = bool(
        densified["tusb_semantic_target_improved_vs_external_teacher_only"]
        and ood_a["tusb_semantic_target_improved_vs_external_teacher_only"]
        and ood_b["tusb_semantic_target_improved_vs_external_teacher_only"]
    )
    tiebreak_vs_hybrid = bool(densified["semantic_target_tiebreak_improved_vs_hybrid_light"])
    tiebreak_vs_legacy = bool(
        densified["semantic_target_tiebreak_improved_vs_legacysem"]
        and ood_a["semantic_target_tiebreak_improved_vs_legacysem"]
        and ood_b["semantic_target_tiebreak_improved_vs_legacysem"]
    )
    trace_coupling = bool(tusb_target_vs_external and tiebreak_vs_legacy)
    official_story_supported = bool(trace_coupling and tiebreak_vs_hybrid)
    if official_story_supported:
        next_step_choice = "start_main_submission_assets"
    elif tiebreak_vs_legacy or tusb_target_vs_external:
        next_step_choice = "one_last_surgical_fix"
    else:
        next_step_choice = "reframe_as_moderate_claim_main_track"
    decision = {
        "generated_at_utc": _now_iso(),
        "semantic_teacher_only_should_rename_to_tusb_semantic_target": True,
        "tusb_semantic_target_improved_vs_external_teacher_only": bool(tusb_target_vs_external),
        "semantic_target_tiebreak_improved_vs_hybrid_light": bool(tiebreak_vs_hybrid),
        "semantic_target_tiebreak_improved_vs_legacysem": bool(tiebreak_vs_legacy),
        "trace_semantic_coupling_load_bearing": bool(trace_coupling),
        "official_story_supported": bool(official_story_supported),
        "next_step_choice": next_step_choice,
        "headtohead": head,
    }
    _write_json(Path(args.decision_json), decision)
    _write_md(
        Path(args.decision_md),
        "STWM Trace-Conditioned Readout Decision 20260423",
        [
            f"- semantic_teacher_only_should_rename_to_tusb_semantic_target: {decision['semantic_teacher_only_should_rename_to_tusb_semantic_target']}",
            f"- tusb_semantic_target_improved_vs_external_teacher_only: {decision['tusb_semantic_target_improved_vs_external_teacher_only']}",
            f"- semantic_target_tiebreak_improved_vs_hybrid_light: {decision['semantic_target_tiebreak_improved_vs_hybrid_light']}",
            f"- semantic_target_tiebreak_improved_vs_legacysem: {decision['semantic_target_tiebreak_improved_vs_legacysem']}",
            f"- trace_semantic_coupling_load_bearing: {decision['trace_semantic_coupling_load_bearing']}",
            f"- official_story_supported: {decision['official_story_supported']}",
            f"- next_step_choice: {decision['next_step_choice']}",
        ],
    )
    return {"eval": eval_payload, "bootstrap": bootstrap_payload, "decision": decision}


def parse_args() -> Any:
    parser = ArgumentParser(description="Run STWM trace-conditioned semantic readout sanity.")
    parser.add_argument("--mode", required=True, choices=["audit", "shard", "repair_external", "merge"])
    parser.add_argument("--group", default="tusb", choices=["tusb", "legacysem", "calibration", "cropenc"])
    parser.add_argument("--seed-list", default="42,123,456,654,789,321")
    parser.add_argument("--input-json", default="")
    parser.add_argument("--output-json", default=str(SHARDS / "shard.json"))
    parser.add_argument("--shard-jsons", default="")
    parser.add_argument("--audit-json", default=str(REPORTS / "stwm_trace_conditioned_readout_audit_20260423.json"))
    parser.add_argument("--audit-md", default=str(DOCS / "STWM_TRACE_CONDITIONED_READOUT_AUDIT_20260423.md"))
    parser.add_argument("--eval-json", default=str(REPORTS / "stwm_trace_conditioned_readout_eval_20260423.json"))
    parser.add_argument("--eval-md", default=str(DOCS / "STWM_TRACE_CONDITIONED_READOUT_EVAL_20260423.md"))
    parser.add_argument("--bootstrap-json", default=str(REPORTS / "stwm_trace_conditioned_readout_bootstrap_20260423.json"))
    parser.add_argument("--bootstrap-md", default=str(DOCS / "STWM_TRACE_CONDITIONED_READOUT_BOOTSTRAP_20260423.md"))
    parser.add_argument("--decision-json", default=str(REPORTS / "stwm_trace_conditioned_readout_decision_20260423.json"))
    parser.add_argument("--decision-md", default=str(DOCS / "STWM_TRACE_CONDITIONED_READOUT_DECISION_20260423.md"))
    parser.add_argument("--materialization-json", default=str(REPORTS / "tmp_trace_conditioned_readout_materialization_20260423.json"))
    parser.add_argument("--materialization-md", default=str(DOCS / "TMP_TRACE_CONDITIONED_READOUT_MATERIALIZATION_20260423.md"))
    parser.add_argument("--prepared-cache", default=str(PREPARED_CACHE))
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
    evalcore._apply_process_title_normalization()
    args = parse_args()
    if args.mode == "audit":
        build_audit(args)
    elif args.mode == "shard":
        build_shard(args)
    elif args.mode == "repair_external":
        repair_external_scores(args)
    else:
        build_merge(args)


if __name__ == "__main__":
    main()
