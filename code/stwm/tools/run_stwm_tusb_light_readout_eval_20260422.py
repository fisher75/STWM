#!/usr/bin/env python3
from __future__ import annotations

from argparse import ArgumentParser
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple
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
from stwm.tools import run_stage2_state_identifiability_eval_v3_20260416 as evalv3


ROOT = evalcore.ROOT
PANEL_NAME = "densified_200_context_preserving"
SEEDS = [42, 123, 456, 654, 789, 321]
VAL_BUCKETS = {0, 1, 2}
TUSB_BEST = "TUSB-v3.1::best.pt"
TUSB_SIDECAR = "TUSB-v3.1::best_semantic_hard.pt"
CAL = "calibration-only::best.pt"
CROP = "cropenc::best.pt"
LEGACY = "legacysem::best.pt"
METHOD_ORDER = [TUSB_BEST, TUSB_SIDECAR, CAL, CROP, LEGACY]


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


def _load_checkpoint_map(main_audit: Path, sidecar_audit: Path) -> Dict[str, Dict[int, Dict[str, str]]]:
    main_payload = evalcore.read_json(main_audit)
    side_payload = evalcore.read_json(sidecar_audit)
    per_method = main_payload.get("per_method", {}) if isinstance(main_payload.get("per_method", {}), dict) else {}
    side_rows = side_payload.get("rows", []) if isinstance(side_payload.get("rows", []), list) else []

    def _rows(method_name: str) -> List[Dict[str, Any]]:
        block = per_method.get(method_name, {})
        return block.get("rows", []) if isinstance(block, dict) and isinstance(block.get("rows", []), list) else []

    mapping: Dict[str, Dict[int, Dict[str, str]]] = {
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
    for row in side_rows:
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
    for method_name, seed_map in mapping.items():
        for seed in SEEDS:
            entry = seed_map.get(seed, {})
            ckpt = Path(str(entry.get("checkpoint_path", "")))
            if not ckpt.exists():
                missing.append(f"{method_name} seed={seed} missing checkpoint: {ckpt}")
    if missing:
        raise RuntimeError("; ".join(missing))
    return mapping


def _prepare_items(protocol_path: Path, max_items: int = 0) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    protocol = evalcore.read_json(protocol_path)
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
            candidate_inputs = evalcore._prepare_candidate_inputs(
                item=item,
                target_future_mask=target_future_mask,
                future_masks=future_masks,
            )
            prepared.append(
                {
                    "item": item,
                    "batch": batch,
                    "target_future_mask": target_future_mask,
                    "future_masks": future_masks,
                    "candidate_inputs": candidate_inputs,
                    "protocol_item_id": protocol_item_id,
                    "split": _item_split(protocol_item_id),
                }
            )
        except Exception as exc:
            reason = f"{type(exc).__name__}:{exc}"
            skipped.append({"protocol_item_id": protocol_item_id, "reason": reason})
            skipped_counts[reason] += 1
    item_ids = sorted(str(row["protocol_item_id"]) for row in prepared)
    val_ids = sorted(str(row["protocol_item_id"]) for row in prepared if str(row["split"]) == "val")
    test_ids = sorted(str(row["protocol_item_id"]) for row in prepared if str(row["split"]) == "test")
    meta = {
        "total_requested_items": int(len(items)),
        "valid_items": int(len(prepared)),
        "skipped_items": int(len(skipped)),
        "skipped_reason_counts": dict(sorted(skipped_counts.items())),
        "split_definition": {
            "protocol_item_id_hash": "sha256[:8] mod 10",
            "val_buckets": sorted(list(VAL_BUCKETS)),
            "val_fraction_nominal": 0.3,
            "test_fraction_nominal": 0.7,
        },
        "split_sizes": {
            "val_items": int(len(val_ids)),
            "test_items": int(len(test_ids)),
        },
        "val_item_ids_hash": _sha256_json(val_ids),
        "test_item_ids_hash": _sha256_json(test_ids),
        "all_item_ids_hash": _sha256_json(item_ids),
    }
    return prepared, skipped, meta


def _compose_score_result(
    base_result: Dict[str, Any],
    score_map: Dict[str, float],
    target_id: str,
    target_future_mask: np.ndarray,
    future_masks: Dict[str, np.ndarray],
    scoring_mode: str,
    selected_weights: Dict[str, float] | None = None,
    unit_scores: Dict[str, float] | None = None,
    semantic_scores: Dict[str, float] | None = None,
) -> Dict[str, Any]:
    rank = evalcore._sorted_rank_from_scores(score_map, target_id)
    top1_id = str(rank["top1_candidate_id"])
    result = dict(base_result)
    result.update(
        {
            "scoring_mode": str(scoring_mode),
            "query_future_top1_acc": float(rank["top1"]),
            "future_mask_iou_at_top1": float(evalcore._mask_iou(future_masks.get(str(top1_id)), target_future_mask)),
            "top1_candidate_id": str(top1_id),
            "target_rank": int(rank["target_rank"]),
            "top5_hit": float(rank["top5_hit"]),
            "mrr": float(rank["mrr"]),
            "ranked_candidate_ids": list(rank["ranked_candidate_ids"]),
            "coord_only_scores": dict(base_result.get("coord_only_scores", {})),
            "unit_identity_scores": dict(unit_scores or {}),
            "semantic_teacher_scores": dict(semantic_scores or {}),
        }
    )
    if selected_weights is not None:
        result["selected_hybrid_weights"] = {
            "alpha": float(selected_weights.get("alpha", 0.0)),
            "beta": float(selected_weights.get("beta", 0.0)),
            "gamma": float(selected_weights.get("gamma", 0.0)),
        }
        result["hybrid_light_scores"] = dict(score_map)
    return result


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
            "MRR": 0.0,
            "top5_hit": 0.0,
        }

    def _subset_mean(tag: str, key: str = "query_future_top1_acc") -> float:
        subset = [row for row in rows if tag in set(row.get("subset_tags", []))]
        return _mean(row.get(key, 0.0) for row in subset) if subset else 0.0

    hard_rows = [row for row in rows if row.get("subset_tags")]
    return {
        "overall_top1": _mean(row.get("query_future_top1_acc", 0.0) for row in rows),
        "hit_rate": _mean(row.get("query_future_hit_rate", 0.0) for row in rows),
        "localization_error": _mean(row.get("query_future_localization_error", 0.0) for row in rows),
        "mask_iou_at_top1": _mean(row.get("future_mask_iou_at_top1", 0.0) for row in rows),
        "hard_subset_top1": _mean(row.get("query_future_top1_acc", 0.0) for row in hard_rows) if hard_rows else 0.0,
        "ambiguity_top1": _subset_mean("crossing_ambiguity"),
        "appearance_change_top1": _subset_mean("appearance_change"),
        "occlusion_reappearance_top1": _subset_mean("occlusion_reappearance"),
        "long_gap_persistence_top1": _subset_mean("long_gap_persistence"),
        "small_object_top1": _subset_mean("small_object"),
        "MRR": _mean(row.get("mrr", 0.0) for row in rows),
        "top5_hit": _mean(row.get("top5_hit", 0.0) for row in rows),
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
            and str(row.get("split")) == "test"
        ]
        metrics = _aggregate_rows(picked)
        seed_row = {"seed": int(seed)}
        seed_row.update(metrics)
        seed_rows.append(seed_row)
    metric_keys = list(seed_rows[0].keys())
    metric_keys.remove("seed")
    return {
        "seed_rows": seed_rows,
        "mean": {key: _mean(row[key] for row in seed_rows) for key in metric_keys},
        "std": {key: _std(row[key] for row in seed_rows) for key in metric_keys},
    }


def _seedwise_deltas(
    rows: List[Dict[str, Any]],
    left_method: str,
    left_scoring_mode: str,
    right_method: str,
    right_scoring_mode: str,
    key: str = "query_future_top1_acc",
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for seed in SEEDS:
        left_rows = [
            row
            for row in rows
            if str(row.get("method_name")) == left_method
            and str(row.get("scoring_mode")) == left_scoring_mode
            and int(row.get("seed", -1)) == int(seed)
            and str(row.get("split")) == "test"
        ]
        right_rows = [
            row
            for row in rows
            if str(row.get("method_name")) == right_method
            and str(row.get("scoring_mode")) == right_scoring_mode
            and int(row.get("seed", -1)) == int(seed)
            and str(row.get("split")) == "test"
        ]
        if not left_rows or not right_rows:
            continue
        delta = _mean(float(l.get(key, 0.0)) - float(r.get(key, 0.0)) for l, r in zip(left_rows, right_rows))
        out.append({"seed": int(seed), "delta_top1": float(delta)})
    return out


def _bootstrap_deltas(deltas: List[float], seed: int = 0, n_boot: int = 4000) -> Dict[str, Any]:
    arr = np.asarray([float(v) for v in deltas], dtype=np.float64)
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


def _stable_bootstrap_seed(*parts: str) -> int:
    digest = hashlib.sha256("::".join(parts).encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def _metric_delta_rows(
    rows: List[Dict[str, Any]],
    left_method: str,
    left_scoring_mode: str,
    right_method: str,
    right_scoring_mode: str,
    metric_key: str,
    subset_tag: str = "",
) -> List[float]:
    deltas: List[float] = []
    for left_row in rows:
        if str(left_row.get("method_name")) != left_method or str(left_row.get("scoring_mode")) != left_scoring_mode:
            continue
        if str(left_row.get("split")) != "test":
            continue
        if subset_tag and subset_tag not in set(left_row.get("subset_tags", [])):
            continue
        match = next(
            (
                row
                for row in rows
                if int(row.get("seed", -1)) == int(left_row.get("seed", -1))
                and str(row.get("protocol_item_id")) == str(left_row.get("protocol_item_id"))
                and str(row.get("method_name")) == right_method
                and str(row.get("scoring_mode")) == right_scoring_mode
            ),
            None,
        )
        if not isinstance(match, dict):
            continue
        deltas.append(float(left_row.get(metric_key, 0.0)) - float(match.get(metric_key, 0.0)))
    return deltas


def _method_row(
    protocol_item_id: str,
    seed: int,
    split: str,
    subset_tags: List[str],
    target_id: str,
    method_name: str,
    scoring_mode: str,
    result: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "protocol_item_id": str(protocol_item_id),
        "seed": int(seed),
        "split": str(split),
        "subset_tags": list(subset_tags),
        "target_id": str(target_id),
        "method_name": str(method_name),
        "scoring_mode": str(scoring_mode),
        **result,
    }


def build_reports(args: Any) -> Dict[str, Any]:
    checkpoint_map = _load_checkpoint_map(
        Path(args.main_checkpoint_audit),
        Path(args.sidecar_checkpoint_audit),
    )
    print(f"[{_now_iso()}] prepare_start panel={PANEL_NAME}", flush=True)
    prepared_items, skipped_items, prep_meta = _prepare_items(Path(args.protocol_json), max_items=int(args.max_items))
    print(
        f"[{_now_iso()}] prepare_done valid_items={prep_meta['valid_items']} skipped_items={prep_meta['skipped_items']}",
        flush=True,
    )
    item_assets = {
        str(row["protocol_item_id"]): {
            "target_future_mask": row["target_future_mask"],
            "future_masks": row["future_masks"],
            "candidate_inputs": row["candidate_inputs"],
        }
        for row in prepared_items
    }

    raw_rows: List[Dict[str, Any]] = []
    eval_started = _now_iso()
    wall_start = time.time()
    device, device_info = evalcore._select_eval_device(args)
    print(f"[{_now_iso()}] device_ready mode={device_info.get('mode', '')} device={device}", flush=True)
    try:
        for method_name in METHOD_ORDER:
            for seed in SEEDS:
                entry = checkpoint_map[method_name][seed]
                print(f"[{_now_iso()}] eval_start method={method_name} seed={seed}", flush=True)
                spec = evalcore.MethodSpec(
                    name=method_name,
                    run_name=str(entry["run_name"]),
                    method_type="stage2",
                    checkpoint_path=str(entry["checkpoint_path"]),
                )
                method = evalcore._load_method(spec, device=device)
                try:
                    for prepared in prepared_items:
                        item = prepared["item"]
                        if method_name in {TUSB_BEST, TUSB_SIDECAR}:
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
                            coord_result["coord_only_scores"] = dict(payload.get("coord_scores", {}))
                            raw_rows.append(
                                {
                                    "protocol_item_id": str(prepared["protocol_item_id"]),
                                    "seed": int(seed),
                                    "split": str(prepared["split"]),
                                    "subset_tags": list(item.get("subset_tags", [])),
                                    "target_id": str(item.get("target_id", "")),
                                    "method_name": str(method_name),
                                    "coord_result": coord_result,
                                    "coord_scores": dict(payload.get("coord_scores", {})),
                                    "unit_identity_scores": dict(payload.get("unit_identity_scores", {})),
                                    "semantic_teacher_scores": dict(payload.get("semantic_teacher_scores", {})),
                                    "available": bool(payload.get("available", False)),
                                    "blocking_reason": str(payload.get("blocking_reason", "")),
                                    "dominant_unit": int(payload.get("dominant_unit", -1)),
                                    "dominant_unit_mass": float(payload.get("dominant_unit_mass", 0.0)),
                                }
                            )
                        else:
                            coord_result = evalcore._evaluate_item(
                                method=method,
                                item=item,
                                batch=prepared["batch"],
                                target_future_mask=prepared["target_future_mask"],
                                future_masks=prepared["future_masks"],
                                device=device,
                                scoring_mode="coord_only",
                            )
                            raw_rows.append(
                                {
                                    "protocol_item_id": str(prepared["protocol_item_id"]),
                                    "seed": int(seed),
                                    "split": str(prepared["split"]),
                                    "subset_tags": list(item.get("subset_tags", [])),
                                    "target_id": str(item.get("target_id", "")),
                                    "method_name": str(method_name),
                                    "coord_result": dict(coord_result),
                                    "coord_scores": dict(coord_result.get("coord_only_scores", {})),
                                    "unit_identity_scores": {},
                                    "semantic_teacher_scores": {},
                                    "available": False,
                                    "blocking_reason": "",
                                    "dominant_unit": -1,
                                    "dominant_unit_mass": 0.0,
                                }
                            )
                    print(f"[{_now_iso()}] eval_done method={method_name} seed={seed}", flush=True)
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
    eval_finished = _now_iso()
    wall_time = float(time.time() - wall_start)

    selected_weights: Dict[str, Dict[str, float]] = {}
    for method_name in [TUSB_BEST, TUSB_SIDECAR]:
        val_rows = [
            row
            for row in raw_rows
            if str(row.get("method_name")) == method_name
            and str(row.get("split")) == "val"
            and bool(row.get("available", False))
        ]
        best_combo = {"alpha": 0.7, "beta": 0.2, "gamma": 0.1, "score": -1e9}
        for alpha in [0.5, 0.7, 0.9]:
            for beta in [0.1, 0.2, 0.4]:
                for gamma in [0.0, 0.1, 0.2]:
                    scored_rows: List[Dict[str, Any]] = []
                    for row in val_rows:
                        assets = item_assets[str(row["protocol_item_id"])]
                        score_map = evalcore._build_hybrid_scores(
                            coord_scores=dict(row["coord_scores"]),
                            unit_scores=dict(row["unit_identity_scores"]),
                            semantic_scores=dict(row["semantic_teacher_scores"]),
                            alpha=alpha,
                            beta=beta,
                            gamma=gamma,
                        )
                        scored_rows.append(
                            _compose_score_result(
                                base_result=dict(row["coord_result"]),
                                score_map=score_map,
                                target_id=str(row["target_id"]),
                                target_future_mask=assets["target_future_mask"],
                                future_masks=assets["future_masks"],
                                scoring_mode="hybrid_light",
                                selected_weights={"alpha": alpha, "beta": beta, "gamma": gamma},
                                unit_scores=dict(row["unit_identity_scores"]),
                                semantic_scores=dict(row["semantic_teacher_scores"]),
                            )
                            | {
                                "protocol_item_id": str(row["protocol_item_id"]),
                                "seed": int(row["seed"]),
                                "split": str(row["split"]),
                                "subset_tags": list(row["subset_tags"]),
                            }
                        )
                    metrics = _aggregate_rows(scored_rows)
                    selection_score = float(
                        metrics["overall_top1"] + 0.5 * metrics["hard_subset_top1"] + 0.1 * metrics["MRR"]
                    )
                    if selection_score > float(best_combo["score"]):
                        best_combo = {
                            "alpha": float(alpha),
                            "beta": float(beta),
                            "gamma": float(gamma),
                            "score": float(selection_score),
                        }
        selected_weights[method_name] = {
            "alpha": float(best_combo["alpha"]),
            "beta": float(best_combo["beta"]),
            "gamma": float(best_combo["gamma"]),
        }
        print(
            f"[{_now_iso()}] selected_weights method={method_name} alpha={best_combo['alpha']} beta={best_combo['beta']} gamma={best_combo['gamma']}",
            flush=True,
        )

    final_rows: List[Dict[str, Any]] = []
    for row in raw_rows:
        protocol_item_id = str(row["protocol_item_id"])
        assets = item_assets[protocol_item_id]
        coord_row = _method_row(
            protocol_item_id=protocol_item_id,
            seed=int(row["seed"]),
            split=str(row["split"]),
            subset_tags=list(row["subset_tags"]),
            target_id=str(row["target_id"]),
            method_name=str(row["method_name"]),
            scoring_mode="coord_only",
            result=dict(row["coord_result"]),
        )
        final_rows.append(coord_row)
        if str(row["method_name"]) not in {TUSB_BEST, TUSB_SIDECAR}:
            continue
        if not bool(row.get("available", False)):
            continue
        unit_row = _method_row(
            protocol_item_id=protocol_item_id,
            seed=int(row["seed"]),
            split=str(row["split"]),
            subset_tags=list(row["subset_tags"]),
            target_id=str(row["target_id"]),
            method_name=str(row["method_name"]),
            scoring_mode="unit_identity_only",
            result=_compose_score_result(
                base_result=dict(row["coord_result"]),
                score_map=dict(row["unit_identity_scores"]),
                target_id=str(row["target_id"]),
                target_future_mask=assets["target_future_mask"],
                future_masks=assets["future_masks"],
                scoring_mode="unit_identity_only",
                unit_scores=dict(row["unit_identity_scores"]),
                semantic_scores=dict(row["semantic_teacher_scores"]),
            ),
        )
        final_rows.append(unit_row)
        hybrid_weights = selected_weights[str(row["method_name"])]
        hybrid_row = _method_row(
            protocol_item_id=protocol_item_id,
            seed=int(row["seed"]),
            split=str(row["split"]),
            subset_tags=list(row["subset_tags"]),
            target_id=str(row["target_id"]),
            method_name=str(row["method_name"]),
            scoring_mode="hybrid_light",
            result=_compose_score_result(
                base_result=dict(row["coord_result"]),
                score_map=evalcore._build_hybrid_scores(
                    coord_scores=dict(row["coord_scores"]),
                    unit_scores=dict(row["unit_identity_scores"]),
                    semantic_scores=dict(row["semantic_teacher_scores"]),
                    alpha=float(hybrid_weights["alpha"]),
                    beta=float(hybrid_weights["beta"]),
                    gamma=float(hybrid_weights["gamma"]),
                ),
                target_id=str(row["target_id"]),
                target_future_mask=assets["target_future_mask"],
                future_masks=assets["future_masks"],
                scoring_mode="hybrid_light",
                selected_weights=hybrid_weights,
                unit_scores=dict(row["unit_identity_scores"]),
                semantic_scores=dict(row["semantic_teacher_scores"]),
            ),
        )
        final_rows.append(hybrid_row)

    per_item_results_hash = _sha256_json(final_rows)

    per_method_seed_results: Dict[str, Any] = {
        TUSB_BEST: {
            "coord_only": _seed_table(final_rows, TUSB_BEST, "coord_only"),
            "unit_identity_only": _seed_table(final_rows, TUSB_BEST, "unit_identity_only"),
            "hybrid_light": _seed_table(final_rows, TUSB_BEST, "hybrid_light"),
        },
        TUSB_SIDECAR: {
            "coord_only": _seed_table(final_rows, TUSB_SIDECAR, "coord_only"),
            "unit_identity_only": _seed_table(final_rows, TUSB_SIDECAR, "unit_identity_only"),
            "hybrid_light": _seed_table(final_rows, TUSB_SIDECAR, "hybrid_light"),
        },
        CAL: {"coord_only": _seed_table(final_rows, CAL, "coord_only")},
        CROP: {"coord_only": _seed_table(final_rows, CROP, "coord_only")},
        LEGACY: {"coord_only": _seed_table(final_rows, LEGACY, "coord_only")},
    }

    best_hybrid = per_method_seed_results[TUSB_BEST]["hybrid_light"]["mean"]
    sidecar_hybrid = per_method_seed_results[TUSB_SIDECAR]["hybrid_light"]["mean"]
    best_checkpoint_under_hybrid = "best_semantic_hard.pt"
    if (
        float(best_hybrid["overall_top1"]) > float(sidecar_hybrid["overall_top1"])
        or (
            float(best_hybrid["overall_top1"]) == float(sidecar_hybrid["overall_top1"])
            and float(best_hybrid["hard_subset_top1"]) >= float(sidecar_hybrid["hard_subset_top1"])
        )
    ):
        best_checkpoint_under_hybrid = "best.pt"
    official_hybrid_method = TUSB_BEST if best_checkpoint_under_hybrid == "best.pt" else TUSB_SIDECAR
    official_hybrid_mean = per_method_seed_results[official_hybrid_method]["hybrid_light"]["mean"]
    legacy_mean = per_method_seed_results[LEGACY]["coord_only"]["mean"]
    cal_mean = per_method_seed_results[CAL]["coord_only"]["mean"]
    crop_mean = per_method_seed_results[CROP]["coord_only"]["mean"]
    bestpt_coord_mean = per_method_seed_results[TUSB_BEST]["coord_only"]["mean"]

    headtohead = {
        "generated_at_utc": _now_iso(),
        "panel_name": PANEL_NAME,
        "selected_hybrid_weights": selected_weights,
        "best_checkpoint_under_hybrid": best_checkpoint_under_hybrid,
        "comparisons": {
            "official_hybrid_vs_bestpt_coordonly": {
                "left_method": official_hybrid_method,
                "left_scoring_mode": "hybrid_light",
                "right_method": TUSB_BEST,
                "right_scoring_mode": "coord_only",
                "left_mean": official_hybrid_mean,
                "right_mean": bestpt_coord_mean,
            },
            "official_hybrid_vs_legacysem": {
                "left_method": official_hybrid_method,
                "left_scoring_mode": "hybrid_light",
                "right_method": LEGACY,
                "right_scoring_mode": "coord_only",
                "left_mean": official_hybrid_mean,
                "right_mean": legacy_mean,
                "seedwise_delta_vs_legacysem": _seedwise_deltas(final_rows, official_hybrid_method, "hybrid_light", LEGACY, "coord_only"),
            },
            "official_hybrid_vs_calibration": {
                "left_mean": official_hybrid_mean,
                "right_mean": cal_mean,
            },
            "official_hybrid_vs_cropenc": {
                "left_mean": official_hybrid_mean,
                "right_mean": crop_mean,
            },
            "bestpt_hybrid_vs_sidecar_hybrid": {
                "bestpt_hybrid_mean": best_hybrid,
                "sidecar_hybrid_mean": sidecar_hybrid,
            },
        },
        "improved_vs_bestpt_coordonly": bool(float(official_hybrid_mean["overall_top1"]) > float(bestpt_coord_mean["overall_top1"])),
        "improved_vs_legacysem": bool(float(official_hybrid_mean["overall_top1"]) > float(legacy_mean["overall_top1"])),
        "improved_vs_calibration": bool(float(official_hybrid_mean["overall_top1"]) > float(cal_mean["overall_top1"])),
        "improved_vs_cropenc": bool(float(official_hybrid_mean["overall_top1"]) > float(crop_mean["overall_top1"])),
    }

    bootstrap_report = {
        "generated_at_utc": _now_iso(),
        "panel_name": PANEL_NAME,
        "split_used": "test",
        "best_checkpoint_under_hybrid": best_checkpoint_under_hybrid,
        "comparisons": {},
    }
    comparisons = [
        ("official_hybrid_vs_bestpt_coordonly", official_hybrid_method, "hybrid_light", TUSB_BEST, "coord_only"),
        ("official_hybrid_vs_legacysem", official_hybrid_method, "hybrid_light", LEGACY, "coord_only"),
    ]
    metric_specs = [
        ("overall_top1", "query_future_top1_acc", ""),
        ("hard_subset_top1", "query_future_top1_acc", "__hard__"),
        ("ambiguity_top1", "query_future_top1_acc", "crossing_ambiguity"),
        ("appearance_change_top1", "query_future_top1_acc", "appearance_change"),
        ("occlusion_reappearance_top1", "query_future_top1_acc", "occlusion_reappearance"),
        ("long_gap_persistence_top1", "query_future_top1_acc", "long_gap_persistence"),
    ]
    for comparison_name, left_method, left_mode, right_method, right_mode in comparisons:
        metric_rows: Dict[str, Any] = {}
        for metric_name, metric_key, subset_tag in metric_specs:
            deltas: List[float] = []
            if subset_tag == "__hard__":
                for left_row in final_rows:
                    if str(left_row.get("method_name")) != left_method or str(left_row.get("scoring_mode")) != left_mode:
                        continue
                    if str(left_row.get("split")) != "test" or not left_row.get("subset_tags"):
                        continue
                    match = next(
                        (
                            row
                            for row in final_rows
                            if int(row.get("seed", -1)) == int(left_row.get("seed", -1))
                            and str(row.get("protocol_item_id")) == str(left_row.get("protocol_item_id"))
                            and str(row.get("method_name")) == right_method
                            and str(row.get("scoring_mode")) == right_mode
                        ),
                        None,
                    )
                    if isinstance(match, dict):
                        deltas.append(float(left_row.get(metric_key, 0.0)) - float(match.get(metric_key, 0.0)))
            else:
                deltas = _metric_delta_rows(
                    final_rows,
                    left_method,
                    left_mode,
                    right_method,
                    right_mode,
                    metric_key=metric_key,
                    subset_tag=subset_tag,
                )
            metric_rows[metric_name] = _bootstrap_deltas(
                deltas,
                seed=_stable_bootstrap_seed(comparison_name, metric_name),
            )
        bootstrap_report["comparisons"][comparison_name] = metric_rows

    sidecar_vs_legacy_zero_excluded = bool(
        bootstrap_report["comparisons"]["official_hybrid_vs_legacysem"]["overall_top1"]["zero_excluded"]
    )
    bootstrap_report["bootstrap_zero_excluded_vs_legacysem"] = sidecar_vs_legacy_zero_excluded

    exact_blocking_reason = ""
    if int(prep_meta["valid_items"]) <= 0:
        exact_blocking_reason = "no_valid_items_after_context_preserving_build"
    if not exact_blocking_reason and int(prep_meta["valid_items"]) < 150:
        exact_blocking_reason = f"valid_items_below_threshold:{prep_meta['valid_items']}"

    decision = {
        "generated_at_utc": _now_iso(),
        "official_evaluator_light_hybrid_integrated": True,
        "improved_vs_bestpt_coordonly": bool(headtohead["improved_vs_bestpt_coordonly"]),
        "improved_vs_legacysem": bool(headtohead["improved_vs_legacysem"]),
        "improved_vs_calibration": bool(headtohead["improved_vs_calibration"]),
        "improved_vs_cropenc": bool(headtohead["improved_vs_cropenc"]),
        "bootstrap_zero_excluded_vs_legacysem": bool(sidecar_vs_legacy_zero_excluded),
        "official_checkpoint_under_hybrid": best_checkpoint_under_hybrid,
        "exact_blocking_reason": exact_blocking_reason,
    }
    if exact_blocking_reason:
        decision["next_step_choice"] = "stop_stage2_escalation_and_reframe_claims"
    elif bool(headtohead["improved_vs_legacysem"]):
        decision["next_step_choice"] = "promote_light_readout_and_run_final_top_tier_validation"
    elif bool(headtohead["improved_vs_calibration"]) or bool(headtohead["improved_vs_cropenc"]):
        decision["next_step_choice"] = "keep_moderate_claim_and_write_main_track"
    else:
        decision["next_step_choice"] = "stop_stage2_escalation_and_reframe_claims"

    eval_report = {
        "generated_at_utc": _now_iso(),
        "panel_name": PANEL_NAME,
        "split_definition": prep_meta["split_definition"],
        "split_sizes": prep_meta["split_sizes"],
        "val_item_ids_hash": prep_meta["val_item_ids_hash"],
        "test_item_ids_hash": prep_meta["test_item_ids_hash"],
        "selected_hybrid_weights": selected_weights,
        "eval_started_at": eval_started,
        "eval_finished_at": eval_finished,
        "wall_time_seconds": wall_time,
        "total_requested_items": prep_meta["total_requested_items"],
        "valid_items": prep_meta["valid_items"],
        "skipped_items": prep_meta["skipped_items"],
        "skipped_reason_counts": prep_meta["skipped_reason_counts"],
        "per_item_results_hash": per_item_results_hash,
        "per_method_seed_results": per_method_seed_results,
        "per_item_test_rows": [
            row
            for row in final_rows
            if str(row.get("split")) == "test"
        ],
    }

    _write_json(Path(args.output_eval_json), eval_report)
    _write_md(
        Path(args.output_eval_md),
        [
            "# STWM TUSB Light Readout Eval 20260422",
            "",
            f"- panel_name: {PANEL_NAME}",
            f"- valid_items: {eval_report['valid_items']}",
            f"- skipped_items: {eval_report['skipped_items']}",
            f"- per_item_results_hash: {per_item_results_hash}",
            f"- selected_weights_bestpt: {selected_weights[TUSB_BEST]}",
            f"- selected_weights_sidecar: {selected_weights[TUSB_SIDECAR]}",
        ],
    )
    _write_json(Path(args.output_headtohead_json), headtohead)
    _write_md(
        Path(args.output_headtohead_md),
        [
            "# STWM TUSB Light Readout Head-to-Head 20260422",
            "",
            f"- best_checkpoint_under_hybrid: {headtohead['best_checkpoint_under_hybrid']}",
            f"- improved_vs_bestpt_coordonly: {headtohead['improved_vs_bestpt_coordonly']}",
            f"- improved_vs_legacysem: {headtohead['improved_vs_legacysem']}",
            f"- improved_vs_calibration: {headtohead['improved_vs_calibration']}",
            f"- improved_vs_cropenc: {headtohead['improved_vs_cropenc']}",
        ],
    )
    _write_json(Path(args.output_bootstrap_json), bootstrap_report)
    _write_md(
        Path(args.output_bootstrap_md),
        [
            "# STWM TUSB Light Readout Strict Bootstrap 20260422",
            "",
            f"- best_checkpoint_under_hybrid: {bootstrap_report['best_checkpoint_under_hybrid']}",
            f"- bootstrap_zero_excluded_vs_legacysem: {bootstrap_report['bootstrap_zero_excluded_vs_legacysem']}",
        ],
    )
    _write_json(Path(args.output_decision_json), decision)
    _write_md(
        Path(args.output_decision_md),
        [
            "# STWM TUSB Light Readout Decision 20260422",
            "",
            f"- official_evaluator_light_hybrid_integrated: {decision['official_evaluator_light_hybrid_integrated']}",
            f"- improved_vs_bestpt_coordonly: {decision['improved_vs_bestpt_coordonly']}",
            f"- improved_vs_legacysem: {decision['improved_vs_legacysem']}",
            f"- bootstrap_zero_excluded_vs_legacysem: {decision['bootstrap_zero_excluded_vs_legacysem']}",
            f"- official_checkpoint_under_hybrid: {decision['official_checkpoint_under_hybrid']}",
            f"- next_step_choice: {decision['next_step_choice']}",
        ],
    )
    print(
        f"[{_now_iso()}] reports_written decision={decision['next_step_choice']} checkpoint={decision['official_checkpoint_under_hybrid']}",
        flush=True,
    )
    return {
        "eval": eval_report,
        "headtohead": headtohead,
        "bootstrap": bootstrap_report,
        "decision": decision,
    }


def parse_args() -> Any:
    parser = ArgumentParser(description="Run STWM TUSB-v3.1 light readout integration eval.")
    parser.add_argument("--protocol-json", default=str(ROOT / "reports/stage2_state_identifiability_protocol_v3_20260416.json"))
    parser.add_argument("--main-checkpoint-audit", default=str(ROOT / "reports/stwm_postfix_matched6seed_checkpoint_audit_20260421.json"))
    parser.add_argument("--sidecar-checkpoint-audit", default=str(ROOT / "reports/stwm_sidecar_checkpoint_audit_20260422.json"))
    parser.add_argument("--output-eval-json", default=str(ROOT / "reports/stwm_tusb_light_readout_eval_20260422.json"))
    parser.add_argument("--output-eval-md", default=str(ROOT / "docs/STWM_TUSB_LIGHT_READOUT_EVAL_20260422.md"))
    parser.add_argument("--output-headtohead-json", default=str(ROOT / "reports/stwm_tusb_light_readout_headtohead_20260422.json"))
    parser.add_argument("--output-headtohead-md", default=str(ROOT / "docs/STWM_TUSB_LIGHT_READOUT_HEADTOHEAD_20260422.md"))
    parser.add_argument("--output-bootstrap-json", default=str(ROOT / "reports/stwm_tusb_light_readout_strict_bootstrap_20260422.json"))
    parser.add_argument("--output-bootstrap-md", default=str(ROOT / "docs/STWM_TUSB_LIGHT_READOUT_STRICT_BOOTSTRAP_20260422.md"))
    parser.add_argument("--output-decision-json", default=str(ROOT / "reports/stwm_tusb_light_readout_decision_20260422.json"))
    parser.add_argument("--output-decision-md", default=str(ROOT / "docs/STWM_TUSB_LIGHT_READOUT_DECISION_20260422.md"))
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
