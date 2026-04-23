#!/usr/bin/env python3
from __future__ import annotations

from argparse import ArgumentParser
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
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
from stwm.tools import run_stage2_state_identifiability_eval_v3_20260416 as evalv3
from stwm.tools import run_stwm_tusb_light_readout_eval_20260422 as lighteval


ROOT = Path("/raid/chen034/workspace/stwm")
REPORTS = ROOT / "reports"
DOCS = ROOT / "docs"

OFFICIAL_TUSB = "TUSB-v3.1::official(best_semantic_hard.pt+hybrid_light)"
CALIBRATION = "calibration-only::best.pt"
CROPENC = "cropenc::best.pt"
LEGACYSEM = "legacysem::best.pt"
STAGE1 = "stage1_frozen::best.pt"
METHOD_ORDER = [OFFICIAL_TUSB, CALIBRATION, CROPENC, LEGACYSEM, STAGE1]
OFFICIAL_EVAL_SEED = 0
SEEDS = [OFFICIAL_EVAL_SEED]

SPLIT_A_NAME = "heldout_burst_heavy_context_preserving"
SPLIT_B_NAME = "heldout_scene_category_video_context_preserving"
PRIMARY_OOD_SPLIT = SPLIT_B_NAME
PREPARE_MAX_WORKERS = 8


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


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def write_md(path: Path, title: str, lines: Iterable[str]) -> None:
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


def _last8_bucket(value: str | int, mod: int) -> int:
    return int(hashlib.sha256(str(value).encode("utf-8")).hexdigest()[-8:], 16) % max(int(mod), 1)


def _video_key(item: Mapping[str, Any]) -> str:
    dataset = str(item.get("dataset", "")).strip().upper()
    if dataset == "BURST":
        return f"BURST::{item.get('burst_dataset_name')}::{item.get('burst_seq_name')}"
    return f"VIPSeg::{item.get('clip_id')}"


def _category_key(item: Mapping[str, Any]) -> str:
    return f"{str(item.get('dataset', '')).strip().upper()}::{item.get('category_id')}"


def _subset_counts(items: Sequence[Mapping[str, Any]]) -> Dict[str, int]:
    counter: Counter[str] = Counter()
    for item in items:
        for tag in item.get("subset_tags", []) or []:
            counter[str(tag)] += 1
    return dict(sorted(counter.items()))


def _official_weights() -> Dict[str, float]:
    payload = _load_json(REPORTS / "stwm_lightreadout_final_eval_20260422.json")
    weights = payload.get("official_tusb_selected_weights", {}) if isinstance(payload.get("official_tusb_selected_weights", {}), dict) else {}
    return {
        "alpha": float(weights.get("alpha", 0.5)),
        "beta": float(weights.get("beta", 0.4)),
        "gamma": float(weights.get("gamma", 0.2)),
    }


def _official_checkpoint_map(args: Any) -> Dict[str, Dict[int, Dict[str, str]]]:
    dualpanel = _load_json(REPORTS / "stage2_dualpanel_hardening_20260420.json")
    frozen_mainline = _load_json(REPORTS / "stage2_v3p1_frozen_mainline_20260420.json")
    dense_panel = dualpanel.get("densified_200_panel", {}) if isinstance(dualpanel.get("densified_200_panel", {}), dict) else {}
    method_rows = dense_panel.get("method_rows", []) if isinstance(dense_panel.get("method_rows", []), list) else []
    by_name = {
        str(row.get("name", "")): row
        for row in method_rows
        if isinstance(row, dict) and str(row.get("name", "")).strip()
    }

    def _row_checkpoint(method_name: str) -> Dict[int, Dict[str, str]]:
        row = by_name.get(method_name, {})
        run_name = str(row.get("run_name", "")).strip()
        checkpoint_path = str(row.get("checkpoint_path", "")).strip()
        if not run_name or not checkpoint_path:
            raise RuntimeError(f"missing official checkpoint mapping for {method_name}")
        path = Path(checkpoint_path)
        if not path.exists():
            raise RuntimeError(f"checkpoint missing for {method_name}: {path}")
        return {
            OFFICIAL_EVAL_SEED: {
                "run_name": run_name,
                "checkpoint_path": str(path),
            }
        }

    stage1_ckpt = Path(args.stage1_checkpoint)
    if not stage1_ckpt.exists():
        raise RuntimeError(f"missing stage1 checkpoint: {stage1_ckpt}")

    tusb_run = str(frozen_mainline.get("official_mainline_run_name", "")).strip() or "stage2_tusb_v3p1_seed123_20260418"
    tusb_ckpt = ROOT / "outputs" / "checkpoints" / tusb_run / "best_semantic_hard.pt"
    if not tusb_ckpt.exists():
        raise RuntimeError(f"missing official TUSB sidecar checkpoint: {tusb_ckpt}")

    return {
        OFFICIAL_TUSB: {
            OFFICIAL_EVAL_SEED: {
                "run_name": tusb_run,
                "checkpoint_path": str(tusb_ckpt),
            }
        },
        CALIBRATION: _row_checkpoint("current_calibration_only_best"),
        CROPENC: _row_checkpoint("cropenc_baseline_best"),
        LEGACYSEM: _row_checkpoint("legacysem_best"),
        STAGE1: {
            OFFICIAL_EVAL_SEED: {
                "run_name": "stage1_frozen_baseline",
                "checkpoint_path": str(stage1_ckpt),
            }
        },
    }


def _repo_sync_audit(args: Any) -> Dict[str, Any]:
    files = {
        "run_stage2_state_identifiability_eval_20260415.py": str(ROOT / "code/stwm/tools/run_stage2_state_identifiability_eval_20260415.py"),
        "run_stwm_true_ood_eval_20260420.py": str(ROOT / "code/stwm/tools/run_stwm_true_ood_eval_20260420.py"),
        "run_stwm_tusb_light_readout_eval_20260422.py": str(ROOT / "code/stwm/tools/run_stwm_tusb_light_readout_eval_20260422.py"),
    }
    payload = {
        "generated_at_utc": _now_iso(),
        "official_tusb_checkpoint": "best_semantic_hard.pt",
        "official_tusb_scoring_mode": "hybrid_light",
        "baselines_scoring_mode": "coord_only",
        "repo_files_present": {name: Path(path).exists() for name, path in files.items()},
        "utility_v3_repo_synced": bool((ROOT / "code/stwm/tools/run_stwm_lightreadout_downstream_utility_v3_20260422.py").exists()),
    }
    write_json(Path(args.repo_sync_audit_report), payload)
    write_md(
        Path(args.repo_sync_audit_doc),
        "STWM True OOD Repo Sync Audit 20260422",
        [
            f"- official_tusb_checkpoint: {payload['official_tusb_checkpoint']}",
            f"- official_tusb_scoring_mode: {payload['official_tusb_scoring_mode']}",
            f"- baselines_scoring_mode: {payload['baselines_scoring_mode']}",
            f"- repo_files_present: {json.dumps(payload['repo_files_present'], ensure_ascii=True)}",
            f"- utility_v3_repo_synced: {payload['utility_v3_repo_synced']}",
        ],
    )
    return payload


def _materialize_true_ood_splits(args: Any) -> Tuple[Dict[str, Any], Dict[str, set[str]], Dict[str, Dict[str, Any]]]:
    extended = _load_json(Path(args.extended_protocol_json))
    dense = _load_json(Path(args.dense_protocol_json))
    extended_items = [item for item in extended.get("items", []) if isinstance(item, dict)]
    dense_items = [item for item in dense.get("items", []) if isinstance(item, dict)]
    extended_item_ids = {str(item.get("protocol_item_id", "")) for item in extended_items}
    dense_ids = {str(item.get("protocol_item_id", "")) for item in dense_items}

    split_a_items = [item for item in extended_items if str(item.get("dataset", "")).strip().upper() == "BURST"]
    split_a_ref_items = [item for item in extended_items if str(item.get("dataset", "")).strip().upper() != "BURST"]
    split_a_eval_ids = {str(item.get("protocol_item_id", "")) for item in split_a_items}
    split_a_ref_ids = {str(item.get("protocol_item_id", "")) for item in split_a_ref_items}
    split_a_payload = {
        "split_name": SPLIT_A_NAME,
        "description": "VIPSeg-heavy history -> BURST-heavy held-out eval",
        "split_rules": {
            "eval_items": "dataset == BURST from protocol_v3_extended_600 items",
            "reference_pool": "dataset != BURST from protocol_v3_extended_600 items",
        },
        "item_count": int(len(split_a_items)),
        "per_subset_count": _subset_counts(split_a_items),
        "skipped_count": 0,
        "skipped_reason_counts": {},
        "leakage_check_passed": bool(split_a_eval_ids.isdisjoint(split_a_ref_ids)),
        "no_video_overlap": True,
        "no_item_leakage": bool(split_a_eval_ids.isdisjoint(split_a_ref_ids)),
        "exact_blocking_reason": "",
    }

    heldout_video_keys = {_video_key(item) for item in extended_items if _last8_bucket(_video_key(item), 5) == 0}
    heldout_categories = {_category_key(item) for item in extended_items if _last8_bucket(_category_key(item), 7) == 0}
    heldout_video_keys.update(
        {
            _video_key(item)
            for item in extended_items
            if _category_key(item) in heldout_categories
        }
    )
    split_b_items = [
        item
        for item in extended_items
        if (_video_key(item) in heldout_video_keys) or (_category_key(item) in heldout_categories)
    ]
    split_b_ref_items = [
        item
        for item in extended_items
        if (_video_key(item) not in heldout_video_keys) and (_category_key(item) not in heldout_categories)
    ]
    split_b_eval_ids = {str(item.get("protocol_item_id", "")) for item in split_b_items}
    split_b_ref_ids = {str(item.get("protocol_item_id", "")) for item in split_b_ref_items}
    split_b_eval_videos = {_video_key(item) for item in split_b_items}
    split_b_ref_videos = {_video_key(item) for item in split_b_ref_items}
    split_b_payload = {
        "split_name": SPLIT_B_NAME,
        "description": "conservative scene/category/video held-out split",
        "split_rules": {
            "heldout_video_rule": "video_key hash bucket via sha256(last8_hex) % 5 == 0",
            "heldout_category_rule": "dataset::category_id hash bucket via sha256(last8_hex) % 7 == 0",
            "eval_items": "items whose video_key is held out or dataset::category_id is held out",
            "reference_pool": "items whose video_key and dataset::category_id are both not held out",
        },
        "item_count": int(len(split_b_items)),
        "per_subset_count": _subset_counts(split_b_items),
        "skipped_count": 0,
        "skipped_reason_counts": {},
        "leakage_check_passed": bool(
            split_b_eval_ids.isdisjoint(split_b_ref_ids) and split_b_eval_videos.isdisjoint(split_b_ref_videos)
        ),
        "no_video_overlap": bool(split_b_eval_videos.isdisjoint(split_b_ref_videos)),
        "no_item_leakage": bool(split_b_eval_ids.isdisjoint(split_b_ref_ids)),
        "exact_blocking_reason": "" if split_b_items and split_b_ref_items else "held-out rule produced an empty eval or reference pool",
    }

    payload = {
        "generated_at_utc": _now_iso(),
        "source_dense_protocol": str(args.dense_protocol_json),
        "source_extended_protocol": str(args.extended_protocol_json),
        "densified_200_context_preserving": {
            "item_count": int(len(dense_ids)),
            "per_subset_count": _subset_counts(dense_items),
        },
        "split_a_vipseg_history_to_burst_heldout": split_a_payload,
        "split_b_scene_category_video_heldout": split_b_payload,
        "true_ood_materialized": bool(
            split_a_payload["item_count"] > 0
            and split_a_payload["leakage_check_passed"]
            and split_b_payload["item_count"] > 0
            and split_b_payload["leakage_check_passed"]
        ),
    }
    write_json(Path(args.split_materialization_report), payload)
    write_md(
        Path(args.split_materialization_doc),
        "STWM True OOD Split Materialization 20260422",
        [
            f"- true_ood_materialized: {payload['true_ood_materialized']}",
            f"- densified_200.item_count: {payload['densified_200_context_preserving']['item_count']}",
            f"- split_a.item_count: {split_a_payload['item_count']}",
            f"- split_a.leakage_check_passed: {split_a_payload['leakage_check_passed']}",
            f"- split_b.item_count: {split_b_payload['item_count']}",
            f"- split_b.leakage_check_passed: {split_b_payload['leakage_check_passed']}",
            f"- split_b.exact_blocking_reason: {split_b_payload['exact_blocking_reason'] or 'none'}",
        ],
    )
    return payload, {
        "densified_200_context_preserving": dense_ids,
        SPLIT_A_NAME: split_a_eval_ids,
        SPLIT_B_NAME: split_b_eval_ids,
    }, {str(item.get("protocol_item_id", "")): item for item in extended_items}


def _prepare_one_item(protocol_item_id: str, item: Any) -> Tuple[str, Dict[str, Any] | None, str]:
    if not isinstance(item, dict):
        return str(protocol_item_id), None, "missing_from_item_source"
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
        meta_list = batch.get("meta", []) if isinstance(batch.get("meta", []), list) else []
        meta0 = meta_list[0] if meta_list and isinstance(meta_list[0], dict) else {}
        return (
            str(protocol_item_id),
            {
                "item": item,
                "batch": batch,
                "target_future_mask": target_future_mask,
                "future_masks": future_masks,
                "candidate_inputs": candidate_inputs,
                "protocol_eval_context_entity_count": int(meta0.get("protocol_eval_context_entity_count", 0)),
                "protocol_eval_mode": str(meta0.get("protocol_eval_mode", "context_preserving")),
            },
            "",
        )
    except Exception as exc:
        return str(protocol_item_id), None, f"{type(exc).__name__}:{exc}"


def _prepare_selected_items(item_lookup: Mapping[str, Dict[str, Any]], selected_ids: set[str]) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, str]]:
    prepared: Dict[str, Dict[str, Any]] = {}
    skipped: Dict[str, str] = {}
    selected_list = sorted(str(item_id) for item_id in selected_ids)
    total_items = len(selected_list)
    worker_count = max(1, min(PREPARE_MAX_WORKERS, total_items))
    print(
        f"[{_now_iso()}] prepare_start requested_items={total_items} workers={worker_count}",
        flush=True,
    )
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = {
            executor.submit(_prepare_one_item, protocol_item_id, item_lookup.get(protocol_item_id)): protocol_item_id
            for protocol_item_id in selected_list
        }
        for index, future in enumerate(as_completed(futures), start=1):
            protocol_item_id, payload, error = future.result()
            if isinstance(payload, dict):
                prepared[protocol_item_id] = payload
            else:
                skipped[protocol_item_id] = error or "unknown_prepare_error"
            if index % 50 == 0 or index == total_items:
                print(
                    f"[{_now_iso()}] prepare_progress processed={index}/{total_items} valid={len(prepared)} skipped={len(skipped)}",
                    flush=True,
                )
    print(
        f"[{_now_iso()}] prepare_done valid_items={len(prepared)} skipped_items={len(skipped)}",
        flush=True,
    )
    return prepared, skipped


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


def _run_fresh_eval(
    *,
    prepared_items: Mapping[str, Dict[str, Any]],
    checkpoint_map: Dict[str, Dict[int, Dict[str, str]]],
    official_weights: Dict[str, float],
    args: Any,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    device, device_info = evalcore._select_eval_device(args)
    print(f"[{_now_iso()}] device_ready mode={device_info.get('mode', '')} device={device}", flush=True)
    try:
        for method_name in METHOD_ORDER:
            method_type = "stage1" if method_name == STAGE1 else "stage2"
            scoring_mode = "hybrid_light" if method_name == OFFICIAL_TUSB else "coord_only"
            for seed in SEEDS:
                entry = checkpoint_map[method_name][seed]
                print(f"[{_now_iso()}] eval_start method={method_name} seed={seed}", flush=True)
                spec = evalcore.MethodSpec(
                    name=method_name,
                    run_name=str(entry["run_name"]),
                    method_type=method_type,
                    checkpoint_path=str(entry["checkpoint_path"]),
                )
                method = evalcore._load_method(spec, device=device)
                try:
                    total_items = len(prepared_items)
                    for item_index, protocol_item_id in enumerate(sorted(prepared_items), start=1):
                        prepared = prepared_items[protocol_item_id]
                        item = prepared["item"]
                        result = evalcore._evaluate_item(
                            method=method,
                            item=item,
                            batch=prepared["batch"],
                            target_future_mask=prepared["target_future_mask"],
                            future_masks=prepared["future_masks"],
                            device=device,
                            scoring_mode=scoring_mode,
                            candidate_inputs=prepared["candidate_inputs"],
                            selected_weights=official_weights if method_name == OFFICIAL_TUSB else None,
                        )
                        rows.append(
                            {
                                "protocol_item_id": str(protocol_item_id),
                                "dataset": str(item.get("dataset", "")),
                                "clip_id": str(item.get("clip_id", "")),
                                "seed": int(seed),
                                "target_id": str(item.get("target_id", "")),
                                "subset_tags": list(item.get("subset_tags", [])),
                                "method_name": str(method_name),
                                "scoring_mode": str(scoring_mode),
                                "protocol_eval_context_entity_count": int(prepared["protocol_eval_context_entity_count"]),
                                "protocol_eval_mode": str(prepared["protocol_eval_mode"]),
                                **result,
                            }
                        )
                        if item_index % 50 == 0 or item_index == total_items:
                            print(
                                f"[{_now_iso()}] eval_progress method={method_name} seed={seed} items={item_index}/{total_items}",
                                flush=True,
                            )
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
    return rows, device_info


def _panel_report(
    *,
    panel_name: str,
    item_ids: set[str],
    prepared_items: Mapping[str, Dict[str, Any]],
    skipped_reasons: Mapping[str, str],
    raw_rows: List[Dict[str, Any]],
) -> Dict[str, Any]:
    valid_ids = {item_id for item_id in item_ids if item_id in prepared_items}
    panel_rows = []
    for row in raw_rows:
        protocol_item_id = str(row.get("protocol_item_id", ""))
        if protocol_item_id not in valid_ids:
            continue
        new_row = dict(row)
        new_row["panel_name"] = panel_name
        panel_rows.append(new_row)
    missing_ids = sorted(item_id for item_id in item_ids if item_id not in prepared_items)
    skipped_counts = Counter(str(skipped_reasons.get(item_id, "missing_from_item_source")) for item_id in missing_ids)
    context_mean = _mean(
        int(prepared_items[item_id]["protocol_eval_context_entity_count"])
        for item_id in valid_ids
    ) if valid_ids else 0.0
    return {
        "panel_name": panel_name,
        "total_requested_items": int(len(item_ids)),
        "valid_items": int(len(valid_ids)),
        "skipped_items": int(len(missing_ids)),
        "skipped_reason_counts": dict(sorted(skipped_counts.items())),
        "protocol_eval_context_entity_count_mean": float(context_mean),
        "per_item_results_hash": _sha256_json(panel_rows),
        "per_item_results": panel_rows,
        "per_method_seed_results": {
            method_name: {
                "hybrid_light" if method_name == OFFICIAL_TUSB else "coord_only": _seed_table(
                    panel_rows,
                    method_name,
                    "hybrid_light" if method_name == OFFICIAL_TUSB else "coord_only",
                )
            }
            for method_name in METHOD_ORDER
        },
    }


def _panel_method_mean(panel: Dict[str, Any], method_name: str) -> Dict[str, float]:
    mode = "hybrid_light" if method_name == OFFICIAL_TUSB else "coord_only"
    return (
        panel.get("per_method_seed_results", {})
        .get(method_name, {})
        .get(mode, {})
        .get("mean", {})
    )


def _metric_deltas(rows: List[Dict[str, Any]], left_method: str, right_method: str, metric_key: str, subset_tag: str = "") -> List[float]:
    deltas: List[float] = []
    for left_row in rows:
        if str(left_row.get("method_name")) != left_method:
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
                and str(row.get("protocol_item_id")) == str(left_row.get("protocol_item_id"))
                and str(row.get("method_name")) == right_method
            ),
            None,
        )
        if isinstance(match, dict):
            deltas.append(float(left_row.get(metric_key, 0.0)) - float(match.get(metric_key, 0.0)))
    return deltas


def build_lightreadout_ood_eval(
    *,
    report_path: Path,
    doc_path: Path,
    final_eval_report: Path,
    panel_name: str = "protocol_v3_extended_600_context_preserving",
    official_tusb_method: str = OFFICIAL_TUSB,
    calibration_method: str = CALIBRATION,
    cropenc_method: str = CROPENC,
    legacysem_method: str = LEGACYSEM,
) -> Dict[str, Any]:
    payload = _load_json(final_eval_report)
    panels = payload.get("panels", {}) if isinstance(payload.get("panels", {}), dict) else {}
    panel = panels.get(panel_name, {}) if isinstance(panels.get(panel_name, {}), dict) else {}
    rows = panel.get("per_item_results", []) if isinstance(panel.get("per_item_results", []), list) else []

    def _dataset(row: Dict[str, Any]) -> str:
        dataset = str(row.get("dataset", "")).strip()
        if dataset:
            return dataset
        protocol_item_id = str(row.get("protocol_item_id", "")).strip().lower()
        if protocol_item_id.startswith("burst::"):
            return "BURST"
        if protocol_item_id.startswith("vipseg::"):
            return "VIPSeg"
        return ""

    burst = [row for row in rows if _dataset(row) == "BURST"]
    vipseg = [row for row in rows if _dataset(row) == "VIPSeg"]

    def _agg(method_name: str, subset_rows: List[Dict[str, Any]]) -> Dict[str, float]:
        return _aggregate_rows([row for row in subset_rows if str(row.get("method_name")) == method_name])

    burst_tusb = _agg(official_tusb_method, burst)
    burst_cal = _agg(calibration_method, burst)
    burst_crop = _agg(cropenc_method, burst)
    burst_legacy = _agg(legacysem_method, burst)
    vip_tusb = _agg(official_tusb_method, vipseg)
    vip_cal = _agg(calibration_method, vipseg)
    vip_crop = _agg(cropenc_method, vipseg)
    vip_legacy = _agg(legacysem_method, vipseg)

    def _beats(left_a: Dict[str, float], left_b: Dict[str, float], right_a: Dict[str, float], right_b: Dict[str, float]) -> bool:
        return bool(float(left_a.get("overall_top1", 0.0)) > float(right_a.get("overall_top1", 0.0)) and float(left_b.get("overall_top1", 0.0)) > float(right_b.get("overall_top1", 0.0)))

    out = {
        "generated_at_utc": _now_iso(),
        "source_final_eval_report": str(final_eval_report),
        "panel_name": panel_name,
        "setting_a_vipseg_to_burst_heavy": {
            "official_tusb": burst_tusb,
            "calibration_only": burst_cal,
            "cropenc_baseline": burst_crop,
            "legacysem": burst_legacy,
        },
        "setting_b_burst_to_vipseg_heavy": {
            "official_tusb": vip_tusb,
            "calibration_only": vip_cal,
            "cropenc_baseline": vip_crop,
            "legacysem": vip_legacy,
        },
        "setting_c_conservative_heldout_split": {
            "supported": False,
            "reason": "current live repo still has no materialized true held-out scene/category/video split in this runner",
        },
        "ood_improved_vs_calibration": _beats(burst_tusb, vip_tusb, burst_cal, vip_cal),
        "ood_improved_vs_cropenc": _beats(burst_tusb, vip_tusb, burst_crop, vip_crop),
        "ood_improved_vs_legacysem": _beats(burst_tusb, vip_tusb, burst_legacy, vip_legacy),
        "ood_claim_ready": False,
        "proxy_only_vs_true_ood_boundary": "current rerun only refreshes the existing proxy domain split under official light readout; true held-out OOD is still unsupported here",
    }
    write_json(report_path, out)
    write_md(
        doc_path,
        "STWM Light Readout OOD Eval 20260422",
        [
            f"- panel_name: {panel_name}",
            f"- ood_improved_vs_calibration: {out['ood_improved_vs_calibration']}",
            f"- ood_improved_vs_cropenc: {out['ood_improved_vs_cropenc']}",
            f"- ood_improved_vs_legacysem: {out['ood_improved_vs_legacysem']}",
            f"- ood_claim_ready: {out['ood_claim_ready']}",
            f"- proxy_only_vs_true_ood_boundary: {out['proxy_only_vs_true_ood_boundary']}",
        ],
    )
    return out


def _run_true_ood_validation_20260422(args: Any) -> Dict[str, Any]:
    repo_audit = _repo_sync_audit(args)
    materialization, panel_item_ids, extended_item_lookup = _materialize_true_ood_splits(args)
    print(f"[{_now_iso()}] split_materialization true_ood_materialized={materialization.get('true_ood_materialized', False)}", flush=True)

    needed_ids = set().union(*panel_item_ids.values())
    prepared_items, skipped_reasons = _prepare_selected_items(extended_item_lookup, needed_ids)

    eval_started = _now_iso()
    wall_start = time.time()
    checkpoint_map = _official_checkpoint_map(args)
    official_weights = _official_weights()
    print(f"[{_now_iso()}] official_weights alpha={official_weights['alpha']} beta={official_weights['beta']} gamma={official_weights['gamma']}", flush=True)
    raw_rows, device_info = _run_fresh_eval(
        prepared_items=prepared_items,
        checkpoint_map=checkpoint_map,
        official_weights=official_weights,
        args=args,
    )
    eval_finished = _now_iso()
    wall_time = float(time.time() - wall_start)

    dense_panel = _panel_report(
        panel_name="densified_200_context_preserving",
        item_ids=panel_item_ids["densified_200_context_preserving"],
        prepared_items=prepared_items,
        skipped_reasons=skipped_reasons,
        raw_rows=raw_rows,
    )
    split_a_panel = _panel_report(
        panel_name=SPLIT_A_NAME,
        item_ids=panel_item_ids[SPLIT_A_NAME],
        prepared_items=prepared_items,
        skipped_reasons=skipped_reasons,
        raw_rows=raw_rows,
    )
    split_b_panel = _panel_report(
        panel_name=SPLIT_B_NAME,
        item_ids=panel_item_ids[SPLIT_B_NAME],
        prepared_items=prepared_items,
        skipped_reasons=skipped_reasons,
        raw_rows=raw_rows,
    )

    official_eval = {
        "generated_at_utc": _now_iso(),
        "eval_started_at": eval_started,
        "eval_finished_at": eval_finished,
        "wall_time_seconds": wall_time,
        "official_tusb_method": OFFICIAL_TUSB,
        "official_tusb_checkpoint": "best_semantic_hard.pt",
        "official_tusb_scoring_mode": "hybrid_light",
        "baselines_scoring_mode": "coord_only",
        "selected_device": str(device_info.get("mode", "")),
        "device_info": device_info,
        "panels": {
            "densified_200_context_preserving": dense_panel,
            SPLIT_A_NAME: split_a_panel,
            SPLIT_B_NAME: split_b_panel,
        },
        "exact_blocking_reason": "",
    }
    print(f"[{_now_iso()}] official_eval_ready wall_time_seconds={wall_time:.2f}", flush=True)
    write_json(Path(args.official_eval_report), official_eval)
    write_md(
        Path(args.official_eval_doc),
        "STWM True OOD Official Eval 20260422",
        [
            f"- official_tusb_method: `{OFFICIAL_TUSB}`",
            f"- official_tusb_checkpoint: `best_semantic_hard.pt`",
            f"- official_tusb_scoring_mode: `hybrid_light`",
            f"- eval_started_at: {eval_started}",
            f"- eval_finished_at: {eval_finished}",
            f"- wall_time_seconds: {wall_time:.2f}",
            f"- {SPLIT_A_NAME}.valid_items: {split_a_panel['valid_items']}",
            f"- {SPLIT_B_NAME}.valid_items: {split_b_panel['valid_items']}",
        ],
    )

    headtohead_splits: Dict[str, Any] = {}
    for split_name, panel in [
        ("densified_200_context_preserving", dense_panel),
        (SPLIT_A_NAME, split_a_panel),
        (SPLIT_B_NAME, split_b_panel),
    ]:
        official_mean = _panel_method_mean(panel, OFFICIAL_TUSB)
        cal_mean = _panel_method_mean(panel, CALIBRATION)
        crop_mean = _panel_method_mean(panel, CROPENC)
        legacy_mean = _panel_method_mean(panel, LEGACYSEM)
        headtohead_splits[split_name] = {
            "official_mean": official_mean,
            "calibration_mean": cal_mean,
            "cropenc_mean": crop_mean,
            "legacysem_mean": legacy_mean,
            "improved_vs_calibration": bool(float(official_mean.get("overall_top1", 0.0)) > float(cal_mean.get("overall_top1", 0.0))),
            "improved_vs_cropenc": bool(float(official_mean.get("overall_top1", 0.0)) > float(crop_mean.get("overall_top1", 0.0))),
            "improved_vs_legacysem": bool(float(official_mean.get("overall_top1", 0.0)) > float(legacy_mean.get("overall_top1", 0.0))),
            "hard_subsets_improved_vs_calibration": bool(float(official_mean.get("hard_subset_top1", 0.0)) > float(cal_mean.get("hard_subset_top1", 0.0))),
            "hard_subsets_improved_vs_legacysem": bool(float(official_mean.get("hard_subset_top1", 0.0)) > float(legacy_mean.get("hard_subset_top1", 0.0))),
        }
    headtohead = {
        "generated_at_utc": _now_iso(),
        "primary_heldout_split": PRIMARY_OOD_SPLIT,
        "splits": headtohead_splits,
    }
    print(f"[{_now_iso()}] headtohead_ready primary_split={PRIMARY_OOD_SPLIT}", flush=True)
    write_json(Path(args.headtohead_report), headtohead)
    write_md(
        Path(args.headtohead_doc),
        "STWM True OOD Head-to-Head 20260422",
        [
            f"- primary_heldout_split: `{PRIMARY_OOD_SPLIT}`",
            f"- split_a.improved_vs_calibration: {headtohead_splits[SPLIT_A_NAME]['improved_vs_calibration']}",
            f"- split_a.improved_vs_legacysem: {headtohead_splits[SPLIT_A_NAME]['improved_vs_legacysem']}",
            f"- split_b.improved_vs_calibration: {headtohead_splits[SPLIT_B_NAME]['improved_vs_calibration']}",
            f"- split_b.improved_vs_legacysem: {headtohead_splits[SPLIT_B_NAME]['improved_vs_legacysem']}",
        ],
    )

    primary_rows = split_b_panel["per_item_results"]
    comparisons: Dict[str, Any] = {}
    for comp_name, right_method in [
        ("official_vs_calibration", CALIBRATION),
        ("official_vs_legacysem", LEGACYSEM),
    ]:
        metrics = {}
        for metric_name, metric_key, subset_tag in [
            ("overall_top1", "query_future_top1_acc", ""),
            ("hard_subset_top1", "query_future_top1_acc", "__hard__"),
            ("ambiguity_top1", "query_future_top1_acc", "crossing_ambiguity"),
            ("occlusion_reappearance_top1", "query_future_top1_acc", "occlusion_reappearance"),
            ("long_gap_persistence_top1", "query_future_top1_acc", "long_gap_persistence"),
        ]:
            deltas = _metric_deltas(primary_rows, OFFICIAL_TUSB, right_method, metric_key, subset_tag=subset_tag)
            metrics[metric_name] = lighteval._bootstrap_deltas(
                deltas,
                seed=lighteval._stable_bootstrap_seed(comp_name, metric_name, PRIMARY_OOD_SPLIT),
            )
        comparisons[comp_name] = metrics

    ood_zero_excluded_vs_calibration = bool(
        comparisons["official_vs_calibration"]["overall_top1"]["zero_excluded"]
        and comparisons["official_vs_calibration"]["hard_subset_top1"]["zero_excluded"]
    )
    ood_zero_excluded_vs_legacysem = bool(
        comparisons["official_vs_legacysem"]["overall_top1"]["zero_excluded"]
        and comparisons["official_vs_legacysem"]["hard_subset_top1"]["zero_excluded"]
    )
    if ood_zero_excluded_vs_calibration and ood_zero_excluded_vs_legacysem:
        ood_claim_level = "strong_claim"
    elif headtohead_splits[SPLIT_B_NAME]["improved_vs_calibration"] or headtohead_splits[SPLIT_B_NAME]["improved_vs_legacysem"]:
        ood_claim_level = "moderate_claim"
    else:
        ood_claim_level = "weak_claim"
    bootstrap = {
        "generated_at_utc": _now_iso(),
        "primary_heldout_split": PRIMARY_OOD_SPLIT,
        "comparisons": comparisons,
        "ood_zero_excluded_vs_calibration": ood_zero_excluded_vs_calibration,
        "ood_zero_excluded_vs_legacysem": ood_zero_excluded_vs_legacysem,
        "ood_claim_level": ood_claim_level,
    }
    print(f"[{_now_iso()}] bootstrap_ready claim_level={ood_claim_level}", flush=True)
    write_json(Path(args.bootstrap_report), bootstrap)
    write_md(
        Path(args.bootstrap_doc),
        "STWM True OOD Strict Bootstrap 20260422",
        [
            f"- primary_heldout_split: `{PRIMARY_OOD_SPLIT}`",
            f"- ood_zero_excluded_vs_calibration: {ood_zero_excluded_vs_calibration}",
            f"- ood_zero_excluded_vs_legacysem: {ood_zero_excluded_vs_legacysem}",
            f"- ood_claim_level: `{ood_claim_level}`",
        ],
    )

    true_ood_materialized = bool(materialization.get("true_ood_materialized", False))
    improved_vs_calibration = bool(
        headtohead_splits[SPLIT_A_NAME]["improved_vs_calibration"]
        and headtohead_splits[SPLIT_B_NAME]["improved_vs_calibration"]
    )
    improved_vs_cropenc = bool(
        headtohead_splits[SPLIT_A_NAME]["improved_vs_cropenc"]
        and headtohead_splits[SPLIT_B_NAME]["improved_vs_cropenc"]
    )
    improved_vs_legacysem = bool(
        headtohead_splits[SPLIT_A_NAME]["improved_vs_legacysem"]
        and headtohead_splits[SPLIT_B_NAME]["improved_vs_legacysem"]
    )
    hard_subsets_improved = bool(
        headtohead_splits[SPLIT_A_NAME]["hard_subsets_improved_vs_calibration"]
        and headtohead_splits[SPLIT_B_NAME]["hard_subsets_improved_vs_calibration"]
    )

    if true_ood_materialized and improved_vs_calibration and improved_vs_cropenc and improved_vs_legacysem and ood_claim_level in {"strong_claim", "moderate_claim"}:
        next_step_choice = "start_main_submission_assets"
    elif true_ood_materialized and improved_vs_calibration and improved_vs_cropenc and not improved_vs_legacysem:
        next_step_choice = "one_last_surgical_fix"
    else:
        next_step_choice = "reframe_as_moderate_claim_main_track"

    decision = {
        "generated_at_utc": _now_iso(),
        "true_ood_materialized": true_ood_materialized,
        "improved_vs_calibration": improved_vs_calibration,
        "improved_vs_cropenc": improved_vs_cropenc,
        "improved_vs_legacysem": improved_vs_legacysem,
        "hard_subsets_improved": hard_subsets_improved,
        "ood_zero_excluded_vs_calibration": ood_zero_excluded_vs_calibration,
        "ood_zero_excluded_vs_legacysem": ood_zero_excluded_vs_legacysem,
        "ood_claim_level": ood_claim_level,
        "next_step_choice": next_step_choice,
        "exact_blocking_reason": (
            ""
            if true_ood_materialized
            else (
                materialization.get("split_b_scene_category_video_heldout", {}).get("exact_blocking_reason")
                or "true held-out OOD split materialization failed"
            )
        ),
    }
    print(f"[{_now_iso()}] decision_ready next_step_choice={next_step_choice}", flush=True)
    write_json(Path(args.final_decision_report), decision)
    write_md(
        Path(args.final_decision_doc),
        "STWM True OOD Final Decision 20260422",
        [
            f"- true_ood_materialized: {decision['true_ood_materialized']}",
            f"- improved_vs_calibration: {decision['improved_vs_calibration']}",
            f"- improved_vs_cropenc: {decision['improved_vs_cropenc']}",
            f"- improved_vs_legacysem: {decision['improved_vs_legacysem']}",
            f"- hard_subsets_improved: {decision['hard_subsets_improved']}",
            f"- ood_claim_level: `{decision['ood_claim_level']}`",
            f"- next_step_choice: `{decision['next_step_choice']}`",
            f"- exact_blocking_reason: {decision['exact_blocking_reason'] or 'none'}",
        ],
    )

    return {
        "repo_sync_audit": repo_audit,
        "split_materialization": materialization,
        "official_eval": official_eval,
        "headtohead": headtohead,
        "bootstrap": bootstrap,
        "decision": decision,
    }


def parse_args() -> Any:
    parser = ArgumentParser(description="Run STWM true held-out OOD validation assets.")
    parser.add_argument(
        "--mode",
        default="official_lightreadout_true_ood_20260422",
        choices=[
            "official_lightreadout_true_ood_20260422",
            "proxy_lightreadout_from_final_eval",
            "legacy_true_ood_20260420",
        ],
    )
    parser.add_argument("--final-eval-report", default=str(REPORTS / "stwm_lightreadout_final_eval_20260422.json"))
    parser.add_argument("--panel-name", default="protocol_v3_extended_600_context_preserving")
    parser.add_argument("--official-tusb-method", default=OFFICIAL_TUSB)
    parser.add_argument("--calibration-method", default=CALIBRATION)
    parser.add_argument("--cropenc-method", default=CROPENC)
    parser.add_argument("--legacysem-method", default=LEGACYSEM)
    parser.add_argument("--repo-sync-audit-report", default=str(REPORTS / "stwm_true_ood_repo_sync_audit_20260422.json"))
    parser.add_argument("--repo-sync-audit-doc", default=str(DOCS / "STWM_TRUE_OOD_REPO_SYNC_AUDIT_20260422.md"))
    parser.add_argument("--split-materialization-report", default=str(REPORTS / "stwm_true_ood_split_materialization_20260422.json"))
    parser.add_argument("--split-materialization-doc", default=str(DOCS / "STWM_TRUE_OOD_SPLIT_MATERIALIZATION_20260422.md"))
    parser.add_argument("--official-eval-report", default=str(REPORTS / "stwm_true_ood_official_eval_20260422.json"))
    parser.add_argument("--official-eval-doc", default=str(DOCS / "STWM_TRUE_OOD_OFFICIAL_EVAL_20260422.md"))
    parser.add_argument("--headtohead-report", default=str(REPORTS / "stwm_true_ood_headtohead_20260422.json"))
    parser.add_argument("--headtohead-doc", default=str(DOCS / "STWM_TRUE_OOD_HEADTOHEAD_20260422.md"))
    parser.add_argument("--bootstrap-report", default=str(REPORTS / "stwm_true_ood_strict_bootstrap_20260422.json"))
    parser.add_argument("--bootstrap-doc", default=str(DOCS / "STWM_TRUE_OOD_STRICT_BOOTSTRAP_20260422.md"))
    parser.add_argument("--final-decision-report", default=str(REPORTS / "stwm_true_ood_final_decision_20260422.json"))
    parser.add_argument("--final-decision-doc", default=str(DOCS / "STWM_TRUE_OOD_FINAL_DECISION_20260422.md"))
    parser.add_argument("--dense-protocol-json", default=str(REPORTS / "stage2_state_identifiability_protocol_v3_20260416.json"))
    parser.add_argument("--extended-protocol-json", default=str(REPORTS / "stage2_protocol_v3_extended_evalset_20260420.json"))
    parser.add_argument("--main-checkpoint-audit", default=str(REPORTS / "stwm_postfix_matched6seed_checkpoint_audit_20260421.json"))
    parser.add_argument("--sidecar-checkpoint-audit", default=str(REPORTS / "stwm_sidecar_checkpoint_audit_20260422.json"))
    parser.add_argument("--stage1-checkpoint", default=str(ROOT / "outputs/checkpoints/stage1_v2_longtrain_220m_mainline_20260408/best.pt"))
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--lease-path", default=str(ROOT / "reports/stage1_v2_gpu_lease_20260408.json"))
    parser.add_argument("--eval-required-mem-gb", type=float, default=40.0)
    parser.add_argument("--eval-safety-margin-gb", type=float, default=8.0)
    parser.add_argument("--output-report", default=str(REPORTS / "stwm_true_ood_eval_20260420.json"))
    parser.add_argument("--output-doc", default=str(DOCS / "STWM_TRUE_OOD_EVAL_20260420.md"))
    return parser.parse_args()


def main() -> None:
    _apply_process_title_normalization()
    args = parse_args()
    if args.mode == "proxy_lightreadout_from_final_eval":
        build_lightreadout_ood_eval(
            report_path=Path(args.output_report),
            doc_path=Path(args.output_doc),
            final_eval_report=Path(args.final_eval_report),
            panel_name=str(args.panel_name),
            official_tusb_method=str(args.official_tusb_method),
            calibration_method=str(args.calibration_method),
            cropenc_method=str(args.cropenc_method),
            legacysem_method=str(args.legacysem_method),
        )
        return
    if args.mode == "legacy_true_ood_20260420":
        from run_stwm_decisive_validation_20260420 import build_true_ood_eval_assets  # noqa: E402

        payload, md = build_true_ood_eval_assets()
        write_json(Path(args.output_report), payload)
        write_md(Path(args.output_doc), "STWM True OOD Eval 20260420", md)
        return

    _run_true_ood_validation_20260422(args)


if __name__ == "__main__":
    main()
