#!/usr/bin/env python3
from __future__ import annotations

import gc
import json
import time
from argparse import ArgumentParser
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch

import run_stage2_state_identifiability_eval_20260415 as evalcore
import run_stwm_tusb_light_readout_eval_20260422 as lighteval


ROOT = Path(__file__).resolve().parents[3]
SEEDS = list(lighteval.SEEDS)

OFFICIAL_TUSB = lighteval.TUSB_SIDECAR
CAL = lighteval.CAL
CROP = lighteval.CROP
LEGACY = lighteval.LEGACY
STAGE1 = "stage1 frozen::best.pt"

TUSB_SCORING_MODES = [
    "coord_only",
    "unit_identity_only",
    "semantic_teacher_only",
    "coord_plus_teacher",
    "coord_plus_unit",
    "hybrid_light",
]

PANEL_ORDER = [
    "densified_200_context_preserving",
    "legacy_85_context_preserving",
]


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _official_sidecar_weights(path: Path) -> Dict[str, float]:
    payload = _load_json(path)
    selected = payload.get("selected_hybrid_weights", {}) if isinstance(payload.get("selected_hybrid_weights", {}), dict) else {}
    sidecar = selected.get(OFFICIAL_TUSB, {}) if isinstance(selected.get(OFFICIAL_TUSB, {}), dict) else {}
    if sidecar:
        return {
            "alpha": float(sidecar.get("alpha", 0.5)),
            "beta": float(sidecar.get("beta", 0.4)),
            "gamma": float(sidecar.get("gamma", 0.2)),
        }
    return {"alpha": 0.5, "beta": 0.4, "gamma": 0.2}


def _legacy_85_item_ids(path: Path) -> set[str]:
    payload = _load_json(path)
    dense = payload.get("densified_200_context_preserving", {}) if isinstance(payload.get("densified_200_context_preserving", {}), dict) else {}
    rows = dense.get("per_item_results", []) if isinstance(dense.get("per_item_results", []), list) else []
    return {str(row.get("protocol_item_id", "")) for row in rows if isinstance(row, dict) and str(row.get("protocol_item_id", ""))}


def _load_checkpoint_map(args: Any) -> Dict[str, Dict[int, Dict[str, str]]]:
    mapping = lighteval._load_checkpoint_map(
        Path(args.main_checkpoint_audit),
        Path(args.sidecar_checkpoint_audit),
    )
    stage1_entry = {
        "run_name": "stage1_frozen_baseline",
        "checkpoint_path": str(args.stage1_checkpoint),
    }
    mapping[STAGE1] = {int(seed): dict(stage1_entry) for seed in SEEDS}
    return mapping


def _seed_table_all(rows: List[Dict[str, Any]], method_name: str, scoring_mode: str) -> Dict[str, Any]:
    seed_rows: List[Dict[str, Any]] = []
    for seed in SEEDS:
        picked = [
            row
            for row in rows
            if str(row.get("method_name")) == method_name
            and str(row.get("scoring_mode")) == scoring_mode
            and int(row.get("seed", -1)) == int(seed)
        ]
        metrics = lighteval._aggregate_rows(picked)
        seed_row = {"seed": int(seed)}
        seed_row.update(metrics)
        seed_rows.append(seed_row)
    metric_keys = list(seed_rows[0].keys()) if seed_rows else ["seed"]
    if "seed" in metric_keys:
        metric_keys.remove("seed")
    return {
        "seed_rows": seed_rows,
        "mean": {key: lighteval._mean(row[key] for row in seed_rows) for key in metric_keys},
        "std": {key: lighteval._std(row[key] for row in seed_rows) for key in metric_keys},
    }


def _panel_method_mean(panel: Dict[str, Any], method_name: str, scoring_mode: str) -> Dict[str, float]:
    return (
        panel.get("per_method_seed_results", {})
        .get(method_name, {})
        .get(scoring_mode, {})
        .get("mean", {})
    )


def _build_panel_report(
    panel_name: str,
    total_requested_items: int,
    subset_ids: set[str],
    rows: List[Dict[str, Any]],
    skipped_items: List[Dict[str, Any]],
) -> Dict[str, Any]:
    if subset_ids:
        panel_rows = [row for row in rows if str(row.get("protocol_item_id", "")) in subset_ids]
        skipped_subset = [row for row in skipped_items if str(row.get("protocol_item_id", "")) in subset_ids]
        valid_ids = {str(row.get("protocol_item_id", "")) for row in panel_rows}
        skipped_reason_counts = Counter(str(row.get("reason", "unknown")) for row in skipped_subset)
        residual_missing = int(total_requested_items - len(valid_ids) - len(skipped_subset))
        if residual_missing > 0:
            skipped_reason_counts["missing_after_subset_alignment"] += residual_missing
        skipped_count = int(total_requested_items - len(valid_ids))
    else:
        panel_rows = list(rows)
        valid_ids = {str(row.get("protocol_item_id", "")) for row in panel_rows}
        skipped_reason_counts = Counter(str(row.get("reason", "unknown")) for row in skipped_items)
        skipped_count = int(len(skipped_items))
    per_method_seed_results: Dict[str, Dict[str, Any]] = {
        OFFICIAL_TUSB: {
            mode: _seed_table_all(panel_rows, OFFICIAL_TUSB, mode)
            for mode in TUSB_SCORING_MODES
        },
        LEGACY: {"coord_only": _seed_table_all(panel_rows, LEGACY, "coord_only")},
        CAL: {"coord_only": _seed_table_all(panel_rows, CAL, "coord_only")},
        CROP: {"coord_only": _seed_table_all(panel_rows, CROP, "coord_only")},
        STAGE1: {"coord_only": _seed_table_all(panel_rows, STAGE1, "coord_only")},
    }
    return {
        "panel_name": panel_name,
        "total_requested_items": int(total_requested_items),
        "valid_items": int(len(valid_ids)),
        "skipped_items": int(skipped_count),
        "skipped_reason_counts": dict(sorted(skipped_reason_counts.items())),
        "per_item_results_hash": lighteval._sha256_json(panel_rows),
        "per_item_results": panel_rows,
        "per_method_seed_results": per_method_seed_results,
    }


def _metric_delta_rows(
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


def _bootstrap_rows(rows: List[Dict[str, Any]], left_method: str, left_mode: str, right_method: str, right_mode: str) -> Dict[str, Any]:
    metric_specs = [
        ("overall_top1", "query_future_top1_acc", ""),
        ("hard_subset_top1", "query_future_top1_acc", "__hard__"),
        ("ambiguity_top1", "query_future_top1_acc", "crossing_ambiguity"),
        ("appearance_change_top1", "query_future_top1_acc", "appearance_change"),
        ("occlusion_reappearance_top1", "query_future_top1_acc", "occlusion_reappearance"),
        ("long_gap_persistence_top1", "query_future_top1_acc", "long_gap_persistence"),
    ]
    out: Dict[str, Any] = {}
    for metric_name, metric_key, subset_tag in metric_specs:
        deltas = _metric_delta_rows(
            rows,
            left_method,
            left_mode,
            right_method,
            right_mode,
            metric_key=metric_key,
            subset_tag=subset_tag,
        )
        out[metric_name] = lighteval._bootstrap_deltas(
            deltas,
            seed=lighteval._stable_bootstrap_seed(left_method, left_mode, right_method, right_mode, metric_name),
        )
    return out


def _write_reports(
    args: Any,
    audit: Dict[str, Any],
    eval_report: Dict[str, Any],
    bootstrap: Dict[str, Any],
    decision: Dict[str, Any],
) -> None:
    lighteval._write_json(Path(args.output_audit_json), audit)
    lighteval._write_md(
        Path(args.output_audit_md),
        [
            "# STWM Light Readout Attribution Audit 20260422",
            "",
            f"- official_tusb_checkpoint: {audit['official_tusb_checkpoint']}",
            f"- official_tusb_scoring_mode: {audit['official_tusb_scoring_mode']}",
            f"- baseline_scoring_modes: {audit['baseline_scoring_modes']}",
            f"- per_item_results_source: {audit['per_item_results_source']}",
            f"- audit_passed: {audit['audit_passed']}",
        ],
    )
    lighteval._write_json(Path(args.output_eval_json), eval_report)
    lighteval._write_md(
        Path(args.output_eval_md),
        [
            "# STWM Light Readout Attribution Eval 20260422",
            "",
            f"- official_tusb_checkpoint: {audit['official_tusb_checkpoint']}",
            f"- official_tusb_scoring_mode: {audit['official_tusb_scoring_mode']}",
            f"- valid_items_dense: {eval_report['panels']['densified_200_context_preserving']['valid_items']}",
            f"- valid_items_legacy85: {eval_report['panels']['legacy_85_context_preserving']['valid_items']}",
            f"- per_item_results_hash_dense: {eval_report['panels']['densified_200_context_preserving']['per_item_results_hash']}",
        ],
    )
    lighteval._write_json(Path(args.output_bootstrap_json), bootstrap)
    lighteval._write_md(
        Path(args.output_bootstrap_md),
        [
            "# STWM Light Readout Attribution Bootstrap 20260422",
            "",
            f"- zero_excluded_vs_legacysem: {bootstrap['zero_excluded_vs_legacysem']}",
            f"- zero_excluded_vs_teacher_only: {bootstrap['zero_excluded_vs_teacher_only']}",
            f"- zero_excluded_vs_coord_only: {bootstrap['zero_excluded_vs_coord_only']}",
            f"- zero_excluded_vs_coord_plus_teacher: {bootstrap['zero_excluded_vs_coord_plus_teacher']}",
        ],
    )
    lighteval._write_json(Path(args.output_decision_json), decision)
    lighteval._write_md(
        Path(args.output_decision_md),
        [
            "# STWM Light Readout Attribution Decision 20260422",
            "",
            f"- hybrid_light_improved_vs_legacysem: {decision['hybrid_light_improved_vs_legacysem']}",
            f"- hybrid_light_improved_vs_teacher_only: {decision['hybrid_light_improved_vs_teacher_only']}",
            f"- hybrid_light_improved_vs_unit_identity_only: {decision['hybrid_light_improved_vs_unit_identity_only']}",
            f"- trace_semantic_coupling_load_bearing: {decision['trace_semantic_coupling_load_bearing']}",
            f"- teacher_only_sufficient: {decision['teacher_only_sufficient']}",
            f"- official_story_supported: {decision['official_story_supported']}",
            f"- next_step_choice: {decision['next_step_choice']}",
        ],
    )


def build_reports(args: Any) -> Dict[str, Any]:
    sidecar_weights = _official_sidecar_weights(Path(args.official_light_eval_report))
    legacy_85_ids = _legacy_85_item_ids(Path(args.legacy_85_source_report))
    audit = {
        "generated_at_utc": lighteval._now_iso(),
        "official_tusb_checkpoint": "best_semantic_hard.pt",
        "official_tusb_scoring_mode": "hybrid_light",
        "baseline_scoring_modes": {
            CAL: "coord_only",
            CROP: "coord_only",
            LEGACY: "coord_only",
            STAGE1: "coord_only",
        },
        "per_item_results_source": "fresh_eval_from_run_stwm_lightreadout_attribution_20260422.py",
        "audit_passed": bool(legacy_85_ids),
        "exact_blocking_reason": "" if legacy_85_ids else "legacy_85_item_ids_missing_from_source_report",
    }
    if not audit["audit_passed"]:
        eval_report = {
            "generated_at_utc": lighteval._now_iso(),
            "exact_blocking_reason": audit["exact_blocking_reason"],
            "panels": {},
        }
        bootstrap = {
            "generated_at_utc": lighteval._now_iso(),
            "exact_blocking_reason": audit["exact_blocking_reason"],
        }
        decision = {
            "generated_at_utc": lighteval._now_iso(),
            "dominant_gain_source": "teacher_only",
            "hybrid_light_improved_vs_legacysem": False,
            "hybrid_light_improved_vs_teacher_only": False,
            "hybrid_light_improved_vs_unit_identity_only": False,
            "trace_semantic_coupling_load_bearing": False,
            "teacher_only_sufficient": False,
            "unit_identity_only_sufficient": False,
            "official_story_supported": False,
            "next_step_choice": "reframe_as_moderate_claim_main_track",
            "exact_blocking_reason": audit["exact_blocking_reason"],
        }
        _write_reports(args, audit, eval_report, bootstrap, decision)
        return {"audit": audit, "eval": eval_report, "bootstrap": bootstrap, "decision": decision}

    checkpoint_map = _load_checkpoint_map(args)
    print(f"[{lighteval._now_iso()}] prepare_start protocol={args.protocol_json}", flush=True)
    prepared_items, skipped_items, prep_meta = lighteval._prepare_items(Path(args.protocol_json), max_items=int(args.max_items))
    print(
        f"[{lighteval._now_iso()}] prepare_done valid_items={prep_meta['valid_items']} skipped_items={prep_meta['skipped_items']}",
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
    eval_started_at = lighteval._now_iso()
    wall_start = time.time()
    device, device_info = evalcore._select_eval_device(args)
    print(f"[{lighteval._now_iso()}] device_ready mode={device_info.get('mode', '')} device={device}", flush=True)
    try:
        for method_name in [OFFICIAL_TUSB, CAL, CROP, LEGACY]:
            for seed in SEEDS:
                entry = checkpoint_map[method_name][int(seed)]
                print(f"[{lighteval._now_iso()}] eval_start method={method_name} seed={seed}", flush=True)
                spec = evalcore.MethodSpec(
                    name=str(method_name),
                    run_name=str(entry["run_name"]),
                    method_type="stage2",
                    checkpoint_path=str(entry["checkpoint_path"]),
                )
                method = evalcore._load_method(spec, device=device)
                try:
                    for prepared in prepared_items:
                        item = prepared["item"]
                        if method_name == OFFICIAL_TUSB:
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
                                    "subset_tags": list(item.get("subset_tags", [])),
                                    "target_id": str(item.get("target_id", "")),
                                    "method_name": str(method_name),
                                    "coord_result": coord_result,
                                    "coord_scores": dict(payload.get("coord_scores", {})),
                                    "unit_identity_scores": dict(payload.get("unit_identity_scores", {})),
                                    "semantic_teacher_scores": dict(payload.get("semantic_teacher_scores", {})),
                                    "available": bool(payload.get("available", False)),
                                    "blocking_reason": str(payload.get("blocking_reason", "")),
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
                                    "subset_tags": list(item.get("subset_tags", [])),
                                    "target_id": str(item.get("target_id", "")),
                                    "method_name": str(method_name),
                                    "coord_result": dict(coord_result),
                                    "coord_scores": dict(coord_result.get("coord_only_scores", {})),
                                    "unit_identity_scores": {},
                                    "semantic_teacher_scores": {},
                                    "available": False,
                                    "blocking_reason": "",
                                }
                            )
                finally:
                    evalcore._release_method(method)
                print(f"[{lighteval._now_iso()}] eval_done method={method_name} seed={seed}", flush=True)

        stage1_entry = checkpoint_map[STAGE1][SEEDS[0]]
        print(f"[{lighteval._now_iso()}] eval_start method={STAGE1} seeds=reused_single_checkpoint", flush=True)
        stage1_spec = evalcore.MethodSpec(
            name=STAGE1,
            run_name=str(stage1_entry["run_name"]),
            method_type="stage1",
            checkpoint_path=str(stage1_entry["checkpoint_path"]),
        )
        stage1_method = evalcore._load_method(stage1_spec, device=device)
        try:
            stage1_seed_rows: List[Dict[str, Any]] = []
            for prepared in prepared_items:
                item = prepared["item"]
                coord_result = evalcore._evaluate_item(
                    method=stage1_method,
                    item=item,
                    batch=prepared["batch"],
                    target_future_mask=prepared["target_future_mask"],
                    future_masks=prepared["future_masks"],
                    device=device,
                    scoring_mode="coord_only",
                )
                stage1_seed_rows.append(
                    {
                        "protocol_item_id": str(prepared["protocol_item_id"]),
                        "subset_tags": list(item.get("subset_tags", [])),
                        "target_id": str(item.get("target_id", "")),
                        "coord_result": dict(coord_result),
                    }
                )
            for seed in SEEDS:
                for row in stage1_seed_rows:
                    raw_rows.append(
                        {
                            "protocol_item_id": str(row["protocol_item_id"]),
                            "seed": int(seed),
                            "subset_tags": list(row["subset_tags"]),
                            "target_id": str(row["target_id"]),
                            "method_name": STAGE1,
                            "coord_result": dict(row["coord_result"]),
                            "coord_scores": dict(row["coord_result"].get("coord_only_scores", {})),
                            "unit_identity_scores": {},
                            "semantic_teacher_scores": {},
                            "available": False,
                            "blocking_reason": "",
                        }
                    )
        finally:
            evalcore._release_method(stage1_method)
        print(f"[{lighteval._now_iso()}] eval_done method={STAGE1} seeds=reused_single_checkpoint", flush=True)
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

    eval_finished_at = lighteval._now_iso()
    wall_time_seconds = float(time.time() - wall_start)

    dense_rows: List[Dict[str, Any]] = []
    for row in raw_rows:
        protocol_item_id = str(row["protocol_item_id"])
        assets = item_assets[protocol_item_id]
        dense_rows.append(
            lighteval._method_row(
                protocol_item_id=protocol_item_id,
                seed=int(row["seed"]),
                split="all",
                subset_tags=list(row["subset_tags"]),
                target_id=str(row["target_id"]),
                method_name=str(row["method_name"]),
                scoring_mode="coord_only",
                result=dict(row["coord_result"]),
            )
        )
        if str(row["method_name"]) != OFFICIAL_TUSB or not bool(row.get("available", False)):
            continue
        unit_scores = dict(row["unit_identity_scores"])
        semantic_scores = dict(row["semantic_teacher_scores"])
        coord_scores = dict(row["coord_scores"])
        dense_rows.append(
            lighteval._method_row(
                protocol_item_id=protocol_item_id,
                seed=int(row["seed"]),
                split="all",
                subset_tags=list(row["subset_tags"]),
                target_id=str(row["target_id"]),
                method_name=OFFICIAL_TUSB,
                scoring_mode="unit_identity_only",
                result=lighteval._compose_score_result(
                    base_result=dict(row["coord_result"]),
                    score_map=unit_scores,
                    target_id=str(row["target_id"]),
                    target_future_mask=assets["target_future_mask"],
                    future_masks=assets["future_masks"],
                    scoring_mode="unit_identity_only",
                    unit_scores=unit_scores,
                    semantic_scores=semantic_scores,
                ),
            )
        )
        dense_rows.append(
            lighteval._method_row(
                protocol_item_id=protocol_item_id,
                seed=int(row["seed"]),
                split="all",
                subset_tags=list(row["subset_tags"]),
                target_id=str(row["target_id"]),
                method_name=OFFICIAL_TUSB,
                scoring_mode="semantic_teacher_only",
                result=lighteval._compose_score_result(
                    base_result=dict(row["coord_result"]),
                    score_map=semantic_scores,
                    target_id=str(row["target_id"]),
                    target_future_mask=assets["target_future_mask"],
                    future_masks=assets["future_masks"],
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
            alpha=float(sidecar_weights["alpha"]),
            beta=0.0,
            gamma=float(sidecar_weights["gamma"]),
        )
        dense_rows.append(
            lighteval._method_row(
                protocol_item_id=protocol_item_id,
                seed=int(row["seed"]),
                split="all",
                subset_tags=list(row["subset_tags"]),
                target_id=str(row["target_id"]),
                method_name=OFFICIAL_TUSB,
                scoring_mode="coord_plus_teacher",
                result=lighteval._compose_score_result(
                    base_result=dict(row["coord_result"]),
                    score_map=coord_plus_teacher,
                    target_id=str(row["target_id"]),
                    target_future_mask=assets["target_future_mask"],
                    future_masks=assets["future_masks"],
                    scoring_mode="coord_plus_teacher",
                    selected_weights={
                        "alpha": float(sidecar_weights["alpha"]),
                        "beta": 0.0,
                        "gamma": float(sidecar_weights["gamma"]),
                    },
                    unit_scores={},
                    semantic_scores=semantic_scores,
                ),
            )
        )
        coord_plus_unit = evalcore._build_hybrid_scores(
            coord_scores=coord_scores,
            unit_scores=unit_scores,
            semantic_scores={},
            alpha=float(sidecar_weights["alpha"]),
            beta=float(sidecar_weights["beta"]),
            gamma=0.0,
        )
        dense_rows.append(
            lighteval._method_row(
                protocol_item_id=protocol_item_id,
                seed=int(row["seed"]),
                split="all",
                subset_tags=list(row["subset_tags"]),
                target_id=str(row["target_id"]),
                method_name=OFFICIAL_TUSB,
                scoring_mode="coord_plus_unit",
                result=lighteval._compose_score_result(
                    base_result=dict(row["coord_result"]),
                    score_map=coord_plus_unit,
                    target_id=str(row["target_id"]),
                    target_future_mask=assets["target_future_mask"],
                    future_masks=assets["future_masks"],
                    scoring_mode="coord_plus_unit",
                    selected_weights={
                        "alpha": float(sidecar_weights["alpha"]),
                        "beta": float(sidecar_weights["beta"]),
                        "gamma": 0.0,
                    },
                    unit_scores=unit_scores,
                    semantic_scores={},
                ),
            )
        )
        hybrid_scores = evalcore._build_hybrid_scores(
            coord_scores=coord_scores,
            unit_scores=unit_scores,
            semantic_scores=semantic_scores,
            alpha=float(sidecar_weights["alpha"]),
            beta=float(sidecar_weights["beta"]),
            gamma=float(sidecar_weights["gamma"]),
        )
        dense_rows.append(
            lighteval._method_row(
                protocol_item_id=protocol_item_id,
                seed=int(row["seed"]),
                split="all",
                subset_tags=list(row["subset_tags"]),
                target_id=str(row["target_id"]),
                method_name=OFFICIAL_TUSB,
                scoring_mode="hybrid_light",
                result=lighteval._compose_score_result(
                    base_result=dict(row["coord_result"]),
                    score_map=hybrid_scores,
                    target_id=str(row["target_id"]),
                    target_future_mask=assets["target_future_mask"],
                    future_masks=assets["future_masks"],
                    scoring_mode="hybrid_light",
                    selected_weights=sidecar_weights,
                    unit_scores=unit_scores,
                    semantic_scores=semantic_scores,
                ),
            )
        )

    dense_panel = _build_panel_report(
        panel_name="densified_200_context_preserving",
        total_requested_items=int(prep_meta["total_requested_items"]),
        subset_ids=set(),
        rows=dense_rows,
        skipped_items=skipped_items,
    )
    legacy_panel = _build_panel_report(
        panel_name="legacy_85_context_preserving",
        total_requested_items=int(len(legacy_85_ids)),
        subset_ids=legacy_85_ids,
        rows=dense_rows,
        skipped_items=skipped_items,
    )
    eval_report = {
        "generated_at_utc": lighteval._now_iso(),
        "official_tusb_checkpoint": "best_semantic_hard.pt",
        "official_tusb_scoring_mode": "hybrid_light",
        "selected_hybrid_weights": sidecar_weights,
        "selected_device": str(device),
        "device_info": device_info,
        "eval_started_at": eval_started_at,
        "eval_finished_at": eval_finished_at,
        "wall_time_seconds": wall_time_seconds,
        "panels": {
            "densified_200_context_preserving": dense_panel,
            "legacy_85_context_preserving": legacy_panel,
        },
        "exact_blocking_reason": "",
    }

    dense_mean = _panel_method_mean(dense_panel, OFFICIAL_TUSB, "hybrid_light")
    coord_mean = _panel_method_mean(dense_panel, OFFICIAL_TUSB, "coord_only")
    unit_mean = _panel_method_mean(dense_panel, OFFICIAL_TUSB, "unit_identity_only")
    teacher_mean = _panel_method_mean(dense_panel, OFFICIAL_TUSB, "semantic_teacher_only")
    coord_teacher_mean = _panel_method_mean(dense_panel, OFFICIAL_TUSB, "coord_plus_teacher")
    coord_unit_mean = _panel_method_mean(dense_panel, OFFICIAL_TUSB, "coord_plus_unit")
    legacy_mean = _panel_method_mean(dense_panel, LEGACY, "coord_only")
    cal_mean = _panel_method_mean(dense_panel, CAL, "coord_only")
    crop_mean = _panel_method_mean(dense_panel, CROP, "coord_only")

    headtohead = {
        "generated_at_utc": lighteval._now_iso(),
        "panel_name": "densified_200_context_preserving",
        "hybrid_light_improved_vs_coord_only": bool(float(dense_mean.get("overall_top1", 0.0)) > float(coord_mean.get("overall_top1", 0.0))),
        "hybrid_light_improved_vs_teacher_only": bool(float(dense_mean.get("overall_top1", 0.0)) > float(teacher_mean.get("overall_top1", 0.0))),
        "hybrid_light_improved_vs_unit_identity_only": bool(float(dense_mean.get("overall_top1", 0.0)) > float(unit_mean.get("overall_top1", 0.0))),
        "hybrid_light_improved_vs_coord_plus_teacher": bool(float(dense_mean.get("overall_top1", 0.0)) > float(coord_teacher_mean.get("overall_top1", 0.0))),
        "hybrid_light_improved_vs_coord_plus_unit": bool(float(dense_mean.get("overall_top1", 0.0)) > float(coord_unit_mean.get("overall_top1", 0.0))),
        "hybrid_light_improved_vs_legacysem": bool(float(dense_mean.get("overall_top1", 0.0)) > float(legacy_mean.get("overall_top1", 0.0))),
        "hybrid_light_improved_vs_calibration": bool(float(dense_mean.get("overall_top1", 0.0)) > float(cal_mean.get("overall_top1", 0.0))),
        "hybrid_light_improved_vs_cropenc": bool(float(dense_mean.get("overall_top1", 0.0)) > float(crop_mean.get("overall_top1", 0.0))),
    }

    bootstrap = {
        "generated_at_utc": lighteval._now_iso(),
        "panel_name": "densified_200_context_preserving",
        "comparisons": {
            "hybrid_light_vs_legacysem": _bootstrap_rows(dense_panel["per_item_results"], OFFICIAL_TUSB, "hybrid_light", LEGACY, "coord_only"),
            "hybrid_light_vs_semantic_teacher_only": _bootstrap_rows(dense_panel["per_item_results"], OFFICIAL_TUSB, "hybrid_light", OFFICIAL_TUSB, "semantic_teacher_only"),
            "hybrid_light_vs_coord_only": _bootstrap_rows(dense_panel["per_item_results"], OFFICIAL_TUSB, "hybrid_light", OFFICIAL_TUSB, "coord_only"),
            "hybrid_light_vs_coord_plus_teacher": _bootstrap_rows(dense_panel["per_item_results"], OFFICIAL_TUSB, "hybrid_light", OFFICIAL_TUSB, "coord_plus_teacher"),
        },
    }
    bootstrap["zero_excluded_vs_legacysem"] = bool(
        bootstrap["comparisons"]["hybrid_light_vs_legacysem"]["overall_top1"]["zero_excluded"]
    )
    bootstrap["zero_excluded_vs_teacher_only"] = bool(
        bootstrap["comparisons"]["hybrid_light_vs_semantic_teacher_only"]["overall_top1"]["zero_excluded"]
    )
    bootstrap["zero_excluded_vs_coord_only"] = bool(
        bootstrap["comparisons"]["hybrid_light_vs_coord_only"]["overall_top1"]["zero_excluded"]
    )
    bootstrap["zero_excluded_vs_coord_plus_teacher"] = bool(
        bootstrap["comparisons"]["hybrid_light_vs_coord_plus_teacher"]["overall_top1"]["zero_excluded"]
    )

    teacher_gap = float(dense_mean.get("overall_top1", 0.0)) - float(teacher_mean.get("overall_top1", 0.0))
    unit_gap = float(dense_mean.get("overall_top1", 0.0)) - float(unit_mean.get("overall_top1", 0.0))
    teacher_only_sufficient = bool(teacher_gap <= 0.005)
    unit_identity_only_sufficient = bool(unit_gap <= 0.005)
    trace_semantic_coupling_load_bearing = bool(
        headtohead["hybrid_light_improved_vs_teacher_only"]
        and headtohead["hybrid_light_improved_vs_unit_identity_only"]
    )
    official_story_supported = bool(
        trace_semantic_coupling_load_bearing
        and headtohead["hybrid_light_improved_vs_legacysem"]
        and not teacher_only_sufficient
    )

    if official_story_supported and bootstrap["zero_excluded_vs_legacysem"]:
        next_step_choice = "start_main_submission_assets"
    elif teacher_only_sufficient or not trace_semantic_coupling_load_bearing:
        next_step_choice = "reframe_as_moderate_claim_main_track"
    elif headtohead["hybrid_light_improved_vs_legacysem"]:
        next_step_choice = "run_true_ood_next"
    else:
        next_step_choice = "reframe_as_moderate_claim_main_track"

    dominant_gain_source = "trace+semantic coupling"
    candidate_sources = [
        ("teacher_only", float(teacher_mean.get("overall_top1", 0.0))),
        ("unit_identity_only", float(unit_mean.get("overall_top1", 0.0))),
        ("coord_plus_teacher", float(coord_teacher_mean.get("overall_top1", 0.0))),
        ("coord_plus_unit", float(coord_unit_mean.get("overall_top1", 0.0))),
        ("trace+semantic coupling", float(dense_mean.get("overall_top1", 0.0))),
    ]
    candidate_sources.sort(key=lambda kv: kv[1], reverse=True)
    dominant_gain_source = str(candidate_sources[0][0])

    decision = {
        "generated_at_utc": lighteval._now_iso(),
        "official_tusb_checkpoint": "best_semantic_hard.pt",
        "official_tusb_scoring_mode": "hybrid_light",
        "dominant_gain_source": dominant_gain_source,
        "hybrid_light_improved_vs_coord_only": bool(headtohead["hybrid_light_improved_vs_coord_only"]),
        "hybrid_light_improved_vs_teacher_only": bool(headtohead["hybrid_light_improved_vs_teacher_only"]),
        "hybrid_light_improved_vs_unit_identity_only": bool(headtohead["hybrid_light_improved_vs_unit_identity_only"]),
        "hybrid_light_improved_vs_coord_plus_teacher": bool(headtohead["hybrid_light_improved_vs_coord_plus_teacher"]),
        "hybrid_light_improved_vs_coord_plus_unit": bool(headtohead["hybrid_light_improved_vs_coord_plus_unit"]),
        "hybrid_light_improved_vs_legacysem": bool(headtohead["hybrid_light_improved_vs_legacysem"]),
        "trace_semantic_coupling_load_bearing": trace_semantic_coupling_load_bearing,
        "teacher_only_sufficient": teacher_only_sufficient,
        "unit_identity_only_sufficient": unit_identity_only_sufficient,
        "official_story_supported": official_story_supported,
        "exact_blocking_reason": "",
        "next_step_choice": next_step_choice,
    }

    _write_reports(args, audit, eval_report, bootstrap, decision)
    return {
        "audit": audit,
        "eval": eval_report,
        "bootstrap": bootstrap,
        "decision": decision,
    }


def parse_args() -> Any:
    parser = ArgumentParser(description="Run STWM light-readout attribution + killer baseline pack.")
    parser.add_argument("--protocol-json", default=str(ROOT / "reports/stage2_state_identifiability_protocol_v3_20260416.json"))
    parser.add_argument("--main-checkpoint-audit", default=str(ROOT / "reports/stwm_postfix_matched6seed_checkpoint_audit_20260421.json"))
    parser.add_argument("--sidecar-checkpoint-audit", default=str(ROOT / "reports/stwm_sidecar_checkpoint_audit_20260422.json"))
    parser.add_argument("--legacy-85-source-report", default=str(ROOT / "reports/stage2_v3p1_dualpanel_context_audit_20260420.json"))
    parser.add_argument("--official-light-eval-report", default=str(ROOT / "reports/stwm_tusb_light_readout_eval_20260422.json"))
    parser.add_argument("--stage1-checkpoint", default=str(ROOT / "outputs/checkpoints/stage1_v2_longtrain_220m_mainline_20260408/best.pt"))
    parser.add_argument("--output-audit-json", default=str(ROOT / "reports/stwm_lightreadout_attribution_audit_20260422.json"))
    parser.add_argument("--output-audit-md", default=str(ROOT / "docs/STWM_LIGHTREADOUT_ATTRIBUTION_AUDIT_20260422.md"))
    parser.add_argument("--output-eval-json", default=str(ROOT / "reports/stwm_lightreadout_attribution_eval_20260422.json"))
    parser.add_argument("--output-eval-md", default=str(ROOT / "docs/STWM_LIGHTREADOUT_ATTRIBUTION_EVAL_20260422.md"))
    parser.add_argument("--output-bootstrap-json", default=str(ROOT / "reports/stwm_lightreadout_attribution_bootstrap_20260422.json"))
    parser.add_argument("--output-bootstrap-md", default=str(ROOT / "docs/STWM_LIGHTREADOUT_ATTRIBUTION_BOOTSTRAP_20260422.md"))
    parser.add_argument("--output-decision-json", default=str(ROOT / "reports/stwm_lightreadout_attribution_decision_20260422.json"))
    parser.add_argument("--output-decision-md", default=str(ROOT / "docs/STWM_LIGHTREADOUT_ATTRIBUTION_DECISION_20260422.md"))
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--lease-path", default=str(ROOT / "reports/stage1_v2_gpu_lease_20260408.json"))
    parser.add_argument("--eval-required-mem-gb", type=float, default=40.0)
    parser.add_argument("--eval-safety-margin-gb", type=float, default=8.0)
    parser.add_argument("--max-items", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    evalcore._apply_process_title_normalization()
    args = parse_args()
    build_reports(args)


if __name__ == "__main__":
    main()
