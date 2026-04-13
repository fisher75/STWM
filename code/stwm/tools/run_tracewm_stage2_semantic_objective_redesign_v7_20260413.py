#!/usr/bin/env python3
from __future__ import annotations

from argparse import ArgumentParser
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple
import json
import os
import shlex
import subprocess
import time
import traceback
import gc

import torch
from torch.utils.data import Subset

from stwm.tools.run_tracewm_stage2_ljs_semantic_diagnosis_and_rescue_20260410 import (
    METRIC_KEYS,
    _evaluate_loaded_stage2,
    _f,
    _load_stage1_model,
    _load_stage2_modules,
    _make_dataset,
    _mean_std,
)
from stwm.tools.run_tracewm_stage2_semantic_objective_redesign_v1_20260410 import (
    _dataset_counts,
    _load_ckpt_args,
    _load_ckpt_step,
    _read_json,
    _resume_ckpt_for_seed,
    _write_json,
    _write_md,
)
from stwm.tools.run_tracewm_stage2_semantic_objective_redesign_v2_20260410 import (
    _python_bin_default,
    _release_lease_safe,
    _select_eval_gpu,
    _select_gpu,
    _tmux_windows,
)
from stwm.infra.gpu_lease import is_gpu_leased
from stwm.infra.gpu_telemetry import snapshot_gpu_telemetry


WORK_ROOT = Path("/home/chen034/workspace/stwm")
SESSION = "tracewm_stage2_semantic_objective_redesign_v7_20260413"
LOG_PATH = WORK_ROOT / "logs/stage2_semobjv7_20260413.log"
PILOT_EXTRA_STEPS = 360
PILOT_BATCH_SIZE = 8
PILOT_EVAL_INTERVAL = 100
PILOT_SAVE_EVERY = 100
PILOT_EVAL_MAX_BATCHES = 32
PILOT_MAX_TRAIN_PER_DATASET = 128
PILOT_MAX_VAL_PER_DATASET = 64


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _aggregate_rows(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {key: _mean_std([_f(row.get("metrics", {}).get(key), 1e9) for row in rows]) for key in METRIC_KEYS}


def _baseline_refs() -> Dict[str, Any]:
    diagnosis = _read_json(WORK_ROOT / "reports/stage2_semantic_value_diagnosis_20260410.json")
    full = diagnosis.get("full_validation_panel", {}).get("family_aggregates", {})
    return {
        "cropenc_fullscale_mean": full.get("cropenc", {}),
        "legacysem_fullscale_mean": full.get("legacysem", {}),
        "hard_subset_panels": diagnosis.get("hard_subset_panels", {}),
    }


def _v2_refs() -> Dict[str, Any]:
    return _read_json(WORK_ROOT / "reports/stage2_semantic_objective_redesign_v2_diagnosis_20260410.json")


def _v3_refs() -> Dict[str, Any]:
    return _read_json(WORK_ROOT / "reports/stage2_semantic_objective_redesign_v3_diagnosis_20260410.json")


def _v4_refs() -> Dict[str, Any]:
    return _read_json(WORK_ROOT / "reports/stage2_semantic_objective_redesign_v4_diagnosis_20260411.json")


def _v4_summary() -> Dict[str, Any]:
    return _read_json(WORK_ROOT / "reports/stage2_semantic_objective_redesign_v4_summary_20260411.json")


def _warm_start_anchor_refs() -> Dict[str, Any]:
    refs: Dict[str, Any] = {}
    for seed in [42, 123]:
        final_json = WORK_ROOT / "reports" / f"stage2_fullscale_core_cropenc_seed{seed}_20260409_final.json"
        if final_json.exists():
            payload = _read_json(final_json)
            best = payload.get("best_checkpoint_metric", {}) if isinstance(payload.get("best_checkpoint_metric", {}), dict) else {}
            refs[str(seed)] = {
                "checkpoint": str(WORK_ROOT / "outputs/checkpoints" / f"stage2_fullscale_core_cropenc_seed{seed}_20260409" / "best.pt"),
                "global_step": int(best.get("global_step", -1)),
                "metrics": best.get("metrics", {}) if isinstance(best.get("metrics", {}), dict) else {},
            }
    return refs


def _v4_best_gate_ratio(v4_diag: Dict[str, Any], v4_summary: Dict[str, Any]) -> float:
    best_combo = str(v4_diag.get("success_criteria", {}).get("best_v4_objective_combo", ""))
    vals: List[float] = []
    for row in v4_summary.get("runs", []) if isinstance(v4_summary.get("runs", []), list) else []:
        if not isinstance(row, dict):
            continue
        if best_combo and str(row.get("objective_combo", "")) != best_combo:
            continue
        vals.append(_f(row.get("sparse_gate_selected_ratio_mean"), 1.0))
    if vals:
        return float(sum(vals) / max(len(vals), 1))
    return 1.0


def _v2_best_hard_composite(v2: Dict[str, Any]) -> float:
    best_combo = str(v2.get("success_criteria", {}).get("best_v2_objective_combo", ""))
    hard_panels = v2.get("semantic_hard_subset_panel", {}) if isinstance(v2.get("semantic_hard_subset_panel", {}), dict) else {}
    per_run: Dict[str, List[float]] = {}
    for panel in hard_panels.values():
        rows = panel.get("runs", []) if isinstance(panel, dict) else []
        for row in rows if isinstance(rows, list) else []:
            if not isinstance(row, dict):
                continue
            if best_combo and str(row.get("objective_combo", "")) != best_combo:
                continue
            per_run.setdefault(str(row.get("run_name", "")), []).append(_f(row.get("metrics", {}).get("free_rollout_endpoint_l2"), 1e9))
    if not per_run:
        return 1e9
    return min(sum(vals) / max(len(vals), 1) for vals in per_run.values())


def _run_specs() -> List[Dict[str, Any]]:
    return [
        {
            "run_name": "stage2_semobjv7_alignonly_topk1_seed42_20260413",
            "seed": 42,
            "objective_combo": "v7_calibration_only_topk1_alignment",
            "objective_family": "calibration_only_family",
            "persistence_objective_declared": False,
            "semantic_rescue_mode": "v7alignonly",
            "semantic_rescue_weight": 0.00015,
            "confidence_gated_alignment_loss_weight": 1.0,
            "sparse_persistence_contrastive_loss_weight": 0.0,
            "confidence_gating_margin_threshold": 0.10,
            "confidence_gating_temperature": 0.05,
            "semantic_hard_score_threshold": 0.25,
            "aux_loss_delay_steps": 180,
            "aux_loss_ramp_steps": 360,
            "v6_gating_family": "hard_topk_query_gating_v2",
            "v6_topk_query_k": 1,
            "v6_capped_quantile": 0.85,
            "v6_max_affected_ratio": 0.15,
            "v6_gate_min_strength": 0.05,
            "v6_strict_max_pairs_per_sample": 2,
            "v6_hard_negative_cap": 6,
            "v6_pair_sampling_temperature": 0.35,
            "v6_guaranteed_min_pairs_per_sample": 0,
            "v6_two_level_pair_mining_enabled": False,
            "v6_relaxed_motion_threshold": 0.08,
            "v6_relaxed_area_jump_threshold": 0.06,
            "v6_relaxed_small_query_threshold": 0.20,
            "v6_relaxed_appearance_shift_threshold": 0.25,
            "v6_relaxed_center_interaction_threshold": 0.10,
            "window_name": "semobjv7_a_t1_s42",
        },
        {
            "run_name": "stage2_semobjv7_alignonly_qcap15_seed42_20260413",
            "seed": 42,
            "objective_combo": "v7_calibration_only_qcap15_alignment",
            "objective_family": "calibration_only_family",
            "persistence_objective_declared": False,
            "semantic_rescue_mode": "v7alignonly",
            "semantic_rescue_weight": 0.00015,
            "confidence_gated_alignment_loss_weight": 1.0,
            "sparse_persistence_contrastive_loss_weight": 0.0,
            "confidence_gating_margin_threshold": 0.10,
            "confidence_gating_temperature": 0.05,
            "semantic_hard_score_threshold": 0.25,
            "aux_loss_delay_steps": 180,
            "aux_loss_ramp_steps": 360,
            "v6_gating_family": "capped_quantile_sparse_gating_v2",
            "v6_topk_query_k": 1,
            "v6_capped_quantile": 0.80,
            "v6_max_affected_ratio": 0.15,
            "v6_gate_min_strength": 0.05,
            "v6_strict_max_pairs_per_sample": 2,
            "v6_hard_negative_cap": 6,
            "v6_pair_sampling_temperature": 0.35,
            "v6_guaranteed_min_pairs_per_sample": 0,
            "v6_two_level_pair_mining_enabled": False,
            "v6_relaxed_motion_threshold": 0.08,
            "v6_relaxed_area_jump_threshold": 0.06,
            "v6_relaxed_small_query_threshold": 0.20,
            "v6_relaxed_appearance_shift_threshold": 0.25,
            "v6_relaxed_center_interaction_threshold": 0.10,
            "window_name": "semobjv7_a_q15_s42",
        },
        {
            "run_name": "stage2_semobjv7_alignpersist_topk1_seed42_20260413",
            "seed": 42,
            "objective_combo": "v7_calibration_plus_active_persistence_topk1",
            "objective_family": "calibration_plus_active_persistence_family",
            "persistence_objective_declared": True,
            "semantic_rescue_mode": "v7alignpersist",
            "semantic_rescue_weight": 0.00015,
            "confidence_gated_alignment_loss_weight": 1.0,
            "sparse_persistence_contrastive_loss_weight": 0.05,
            "confidence_gating_margin_threshold": 0.10,
            "confidence_gating_temperature": 0.05,
            "semantic_hard_score_threshold": 0.25,
            "aux_loss_delay_steps": 180,
            "aux_loss_ramp_steps": 360,
            "v6_gating_family": "hard_topk_query_gating_v2",
            "v6_topk_query_k": 1,
            "v6_capped_quantile": 0.85,
            "v6_max_affected_ratio": 0.15,
            "v6_gate_min_strength": 0.05,
            "v6_strict_max_pairs_per_sample": 2,
            "v6_hard_negative_cap": 6,
            "v6_pair_sampling_temperature": 0.35,
            "v6_guaranteed_min_pairs_per_sample": 2,
            "v6_two_level_pair_mining_enabled": True,
            "v6_relaxed_motion_threshold": 0.08,
            "v6_relaxed_area_jump_threshold": 0.06,
            "v6_relaxed_small_query_threshold": 0.20,
            "v6_relaxed_appearance_shift_threshold": 0.25,
            "v6_relaxed_center_interaction_threshold": 0.10,
            "window_name": "semobjv7_p_t1_s42",
        },
        {
            "run_name": "stage2_semobjv7_alignpersist_qcap15_seed42_20260413",
            "seed": 42,
            "objective_combo": "v7_calibration_plus_active_persistence_qcap15",
            "objective_family": "calibration_plus_active_persistence_family",
            "persistence_objective_declared": True,
            "semantic_rescue_mode": "v7alignpersist",
            "semantic_rescue_weight": 0.00015,
            "confidence_gated_alignment_loss_weight": 1.0,
            "sparse_persistence_contrastive_loss_weight": 0.05,
            "confidence_gating_margin_threshold": 0.10,
            "confidence_gating_temperature": 0.05,
            "semantic_hard_score_threshold": 0.25,
            "aux_loss_delay_steps": 180,
            "aux_loss_ramp_steps": 360,
            "v6_gating_family": "capped_quantile_sparse_gating_v2",
            "v6_topk_query_k": 1,
            "v6_capped_quantile": 0.80,
            "v6_max_affected_ratio": 0.15,
            "v6_gate_min_strength": 0.05,
            "v6_strict_max_pairs_per_sample": 2,
            "v6_hard_negative_cap": 6,
            "v6_pair_sampling_temperature": 0.35,
            "v6_guaranteed_min_pairs_per_sample": 2,
            "v6_two_level_pair_mining_enabled": True,
            "v6_relaxed_motion_threshold": 0.08,
            "v6_relaxed_area_jump_threshold": 0.06,
            "v6_relaxed_small_query_threshold": 0.20,
            "v6_relaxed_appearance_shift_threshold": 0.25,
            "v6_relaxed_center_interaction_threshold": 0.10,
            "window_name": "semobjv7_p_q15_s42",
        },
        {
            "run_name": "stage2_semobjv7_alignonly_topk1_seed123_20260413",
            "seed": 123,
            "objective_combo": "v7_calibration_only_topk1_alignment",
            "objective_family": "calibration_only_family",
            "persistence_objective_declared": False,
            "semantic_rescue_mode": "v7alignonly",
            "semantic_rescue_weight": 0.00015,
            "confidence_gated_alignment_loss_weight": 1.0,
            "sparse_persistence_contrastive_loss_weight": 0.0,
            "confidence_gating_margin_threshold": 0.10,
            "confidence_gating_temperature": 0.05,
            "semantic_hard_score_threshold": 0.25,
            "aux_loss_delay_steps": 180,
            "aux_loss_ramp_steps": 360,
            "v6_gating_family": "hard_topk_query_gating_v2",
            "v6_topk_query_k": 1,
            "v6_capped_quantile": 0.85,
            "v6_max_affected_ratio": 0.15,
            "v6_gate_min_strength": 0.05,
            "v6_strict_max_pairs_per_sample": 2,
            "v6_hard_negative_cap": 6,
            "v6_pair_sampling_temperature": 0.35,
            "v6_guaranteed_min_pairs_per_sample": 0,
            "v6_two_level_pair_mining_enabled": False,
            "v6_relaxed_motion_threshold": 0.08,
            "v6_relaxed_area_jump_threshold": 0.06,
            "v6_relaxed_small_query_threshold": 0.20,
            "v6_relaxed_appearance_shift_threshold": 0.25,
            "v6_relaxed_center_interaction_threshold": 0.10,
            "window_name": "semobjv7_a_t1_s123",
        },
        {
            "run_name": "stage2_semobjv7_alignonly_qcap15_seed123_20260413",
            "seed": 123,
            "objective_combo": "v7_calibration_only_qcap15_alignment",
            "objective_family": "calibration_only_family",
            "persistence_objective_declared": False,
            "semantic_rescue_mode": "v7alignonly",
            "semantic_rescue_weight": 0.00015,
            "confidence_gated_alignment_loss_weight": 1.0,
            "sparse_persistence_contrastive_loss_weight": 0.0,
            "confidence_gating_margin_threshold": 0.10,
            "confidence_gating_temperature": 0.05,
            "semantic_hard_score_threshold": 0.25,
            "aux_loss_delay_steps": 180,
            "aux_loss_ramp_steps": 360,
            "v6_gating_family": "capped_quantile_sparse_gating_v2",
            "v6_topk_query_k": 1,
            "v6_capped_quantile": 0.80,
            "v6_max_affected_ratio": 0.15,
            "v6_gate_min_strength": 0.05,
            "v6_strict_max_pairs_per_sample": 2,
            "v6_hard_negative_cap": 6,
            "v6_pair_sampling_temperature": 0.35,
            "v6_guaranteed_min_pairs_per_sample": 0,
            "v6_two_level_pair_mining_enabled": False,
            "v6_relaxed_motion_threshold": 0.08,
            "v6_relaxed_area_jump_threshold": 0.06,
            "v6_relaxed_small_query_threshold": 0.20,
            "v6_relaxed_appearance_shift_threshold": 0.25,
            "v6_relaxed_center_interaction_threshold": 0.10,
            "window_name": "semobjv7_a_q15_s123",
        },
        {
            "run_name": "stage2_semobjv7_alignpersist_topk1_seed123_20260413",
            "seed": 123,
            "objective_combo": "v7_calibration_plus_active_persistence_topk1",
            "objective_family": "calibration_plus_active_persistence_family",
            "persistence_objective_declared": True,
            "semantic_rescue_mode": "v7alignpersist",
            "semantic_rescue_weight": 0.00015,
            "confidence_gated_alignment_loss_weight": 1.0,
            "sparse_persistence_contrastive_loss_weight": 0.05,
            "confidence_gating_margin_threshold": 0.10,
            "confidence_gating_temperature": 0.05,
            "semantic_hard_score_threshold": 0.25,
            "aux_loss_delay_steps": 180,
            "aux_loss_ramp_steps": 360,
            "v6_gating_family": "hard_topk_query_gating_v2",
            "v6_topk_query_k": 1,
            "v6_capped_quantile": 0.85,
            "v6_max_affected_ratio": 0.15,
            "v6_gate_min_strength": 0.05,
            "v6_strict_max_pairs_per_sample": 2,
            "v6_hard_negative_cap": 6,
            "v6_pair_sampling_temperature": 0.35,
            "v6_guaranteed_min_pairs_per_sample": 2,
            "v6_two_level_pair_mining_enabled": True,
            "v6_relaxed_motion_threshold": 0.08,
            "v6_relaxed_area_jump_threshold": 0.06,
            "v6_relaxed_small_query_threshold": 0.20,
            "v6_relaxed_appearance_shift_threshold": 0.25,
            "v6_relaxed_center_interaction_threshold": 0.10,
            "window_name": "semobjv7_p_t1_s123",
        },
        {
            "run_name": "stage2_semobjv7_alignpersist_qcap15_seed123_20260413",
            "seed": 123,
            "objective_combo": "v7_calibration_plus_active_persistence_qcap15",
            "objective_family": "calibration_plus_active_persistence_family",
            "persistence_objective_declared": True,
            "semantic_rescue_mode": "v7alignpersist",
            "semantic_rescue_weight": 0.00015,
            "confidence_gated_alignment_loss_weight": 1.0,
            "sparse_persistence_contrastive_loss_weight": 0.05,
            "confidence_gating_margin_threshold": 0.10,
            "confidence_gating_temperature": 0.05,
            "semantic_hard_score_threshold": 0.25,
            "aux_loss_delay_steps": 180,
            "aux_loss_ramp_steps": 360,
            "v6_gating_family": "capped_quantile_sparse_gating_v2",
            "v6_topk_query_k": 1,
            "v6_capped_quantile": 0.80,
            "v6_max_affected_ratio": 0.15,
            "v6_gate_min_strength": 0.05,
            "v6_strict_max_pairs_per_sample": 2,
            "v6_hard_negative_cap": 6,
            "v6_pair_sampling_temperature": 0.35,
            "v6_guaranteed_min_pairs_per_sample": 2,
            "v6_two_level_pair_mining_enabled": True,
            "v6_relaxed_motion_threshold": 0.08,
            "v6_relaxed_area_jump_threshold": 0.06,
            "v6_relaxed_small_query_threshold": 0.20,
            "v6_relaxed_appearance_shift_threshold": 0.25,
            "v6_relaxed_center_interaction_threshold": 0.10,
            "window_name": "semobjv7_p_q15_s123",
        },
    ]


def _semantic_hard_composite(values: List[float]) -> float:
    if not values:
        return 1e9
    return float(sum(values) / max(len(values), 1))


def _gpu_headroom_ok(lease_path: str, reserve_idle_gpu_count: int) -> bool:
    snap = snapshot_gpu_telemetry(prefer_nvml=True)
    gpu_ids = sorted({int(row.get("gpu_id", -1)) for row in snap.get("gpus", []) if int(row.get("gpu_id", -1)) >= 0})
    if not gpu_ids:
        return True
    free_unleased = 0
    for gpu_id in gpu_ids:
        if not is_gpu_leased(gpu_id=gpu_id, lease_path=lease_path):
            free_unleased += 1
    return bool(free_unleased > int(max(reserve_idle_gpu_count, 0)))


def write_protocol_doc(args: Any) -> None:
    _write_md(
        args.v6_protocol_doc,
        [
            "# Stage2 Semantic Objective Redesign V7 Protocol",
            "",
            f"- generated_at_utc: {now_iso()}",
            "- stage1_mutation_allowed: false",
            "- main_task: future trace / future state generation",
            "- teacher_as_mainline_semantic_source: false",
            "- v7_goal: final disambiguation between calibration-only and calibration-plus-active-persistence under unchanged Stage1 backbone.",
            "- objective_families: calibration_only_family vs calibration_plus_active_persistence_family.",
            "- hard_persistence_activation_rule: if persistence is declared, guaranteed_pair_count_mean must be >= 1.0 and valuable_pair_ratio_mean must be > 0.0, otherwise mark declared_but_inactive.",
            "- activation_audit_required_before_launch: true",
            "- persistence_mining_focus: strict/fallback telemetry + guaranteed pair activation check",
            "- selective_supervision_position: readout-side only, never overwriting frozen trace dynamics.",
            "- forbidden: teacher semantic token replacement; external-eval work; Stage1 rollback; codec/VAE upgrade; full-scale long train; batch/lr sweep; DDP retrofit.",
        ],
    )


def write_decision_rule(args: Any) -> Dict[str, Any]:
    payload = {
        "generated_at_utc": now_iso(),
        "rule_scope": "stage2_semantic_objective_redesign_v7",
        "strict_success_conditions": {
            "v7_runs_terminal": True,
            "semantic_hard_composite_improved_vs_v5": True,
            "improved_vs_v5_best_objective_combo": True,
            "full_validation_non_catastrophic": True,
        },
        "restricted_next_step_enum": [
            "alignment_only_is_true_mainline",
            "persistence_is_load_bearing",
            "persistence_declared_but_not_active",
            "still_cannot_disambiguate",
        ],
        "prohibited_if_disambiguation_missing": ["freeze_stage2_core_mainline"],
        "notes": [
            "The v7 diagnosis must report non-null booleans for alignment-only sufficiency and persistence contribution.",
            "If persistence is declared but inactive, it cannot be used as evidence for load-bearing persistence.",
        ],
    }
    _write_json(args.v6_decision_rule_report, payload)
    return payload


def activation_audit(args: Any) -> Dict[str, Any]:
    specs = _run_specs()
    max_queries = 8
    v5_rows: List[Dict[str, Any]] = []
    if Path(args.v5_reference_summary_report).exists():
        try:
            v5_summary = _read_json(args.v5_reference_summary_report)
            if isinstance(v5_summary.get("runs", []), list):
                v5_rows = [x for x in v5_summary.get("runs", []) if isinstance(x, dict)]
        except Exception:
            v5_rows = []

    def _find_v5_row(seed: int, family: str) -> Dict[str, Any] | None:
        for row in v5_rows:
            if int(row.get("seed", -1)) != int(seed):
                continue
            if str(row.get("v5_gating_family", "")) == family:
                return row
        return None

    audit_rows: List[Dict[str, Any]] = []
    underactive_count = 0
    for spec in specs:
        family = str(spec.get("v6_gating_family", ""))
        topk = int(spec.get("v6_topk_query_k", 1))
        cap_ratio = float(spec.get("v6_max_affected_ratio", 0.0))
        quantile = float(spec.get("v6_capped_quantile", 0.0))
        guaranteed = int(spec.get("v6_guaranteed_min_pairs_per_sample", 0))
        strict_budget = int(spec.get("v6_strict_max_pairs_per_sample", 0))
        two_level = bool(spec.get("v6_two_level_pair_mining_enabled", True))

        if family == "hard_topk_query_gating_v2":
            expected_gate_floor = float(min(max(topk, 1), max_queries) / float(max_queries))
        else:
            expected_gate_floor = float(max(min(cap_ratio, 1.0), 0.0))
        expected_min_pairs = float(max(min(guaranteed, strict_budget if strict_budget > 0 else guaranteed), 0))

        ref = _find_v5_row(seed=int(spec.get("seed", -1)), family=family)
        ref_gate = float(ref.get("actual_gate_positive_ratio_mean", -1.0)) if isinstance(ref, dict) else -1.0
        ref_pairs = float(ref.get("valuable_pair_ratio_mean", -1.0)) if isinstance(ref, dict) else -1.0
        ref_run = str(ref.get("run_name", "")) if isinstance(ref, dict) else ""

        if ref_gate >= 0.0 and ref_gate <= 0.30 and ref_pairs >= 0.0 and ref_pairs < 0.10:
            attribution = "persistence_mining_under_activated_despite_sparse_gate"
            underactive_count += 1
        elif ref_gate > 0.30:
            attribution = "gating_not_sparse_enough_in_reference_v5"
        elif ref_pairs >= 0.10:
            attribution = "persistence_mining_already_active_in_reference_v5"
        else:
            attribution = "insufficient_reference_signal"

        audit_rows.append(
            {
                "run_name": str(spec.get("run_name", "")),
                "seed": int(spec.get("seed", -1)),
                "objective_combo": str(spec.get("objective_combo", "")),
                "semantic_rescue_mode": str(spec.get("semantic_rescue_mode", "")),
                "v6_gating_family": family,
                "v6_topk_query_k": topk,
                "v6_capped_quantile": quantile,
                "v6_max_affected_ratio": cap_ratio,
                "v6_guaranteed_min_pairs_per_sample": guaranteed,
                "v6_strict_max_pairs_per_sample": strict_budget,
                "v6_two_level_pair_mining_enabled": two_level,
                "expected_gate_positive_ratio_floor": expected_gate_floor,
                "expected_min_pairs_per_sample_floor": expected_min_pairs,
                "v5_reference_run": ref_run,
                "v5_reference_gate_positive_ratio_mean": ref_gate,
                "v5_reference_valuable_pair_ratio_mean": ref_pairs,
                "attribution_decision": attribution,
            }
        )

    global_attribution = (
        "persistence_mining_under_activated_primary_bottleneck"
        if underactive_count >= max(1, len(audit_rows) // 2)
        else "mixed_or_non_persistence_primary_bottleneck"
    )

    payload = {
        "generated_at_utc": now_iso(),
        "audit_stage": "pre_launch",
        "diagnosis_source": str(args.v5_reference_summary_report),
        "run_count": int(len(audit_rows)),
        "runs": audit_rows,
        "global_attribution_decision": global_attribution,
        "launch_permitted": bool(len(audit_rows) > 0),
    }
    _write_json(args.v6_activation_audit_report, payload)
    return payload


def launch(args: Any) -> Dict[str, Any]:
    activation_audit(args)
    if subprocess.run(["tmux", "has-session", "-t", str(args.tmux_session)], capture_output=True).returncode != 0:
        subprocess.run(["tmux", "new-session", "-d", "-s", str(args.tmux_session), "bash"], check=True)

    anchor_args = _load_ckpt_args(_resume_ckpt_for_seed(42))
    obs_len = int(anchor_args.get("obs_len", 8) or 8)
    fut_len = int(anchor_args.get("fut_len", 8) or 8)
    max_tokens = int(anchor_args.get("max_tokens", 64) or 64)
    crop_size = int(anchor_args.get("semantic_crop_size", 64) or 64)
    train_counts = _dataset_counts(["vspw", "vipseg"], "train", args.stage2_contract_json, max_samples=PILOT_MAX_TRAIN_PER_DATASET)
    val_counts = _dataset_counts(["vspw", "vipseg"], "val", args.stage2_contract_json, max_samples=PILOT_MAX_VAL_PER_DATASET)

    runs: List[Dict[str, Any]] = []
    for spec in _run_specs():
        run_name = str(spec["run_name"])
        resume_from = _resume_ckpt_for_seed(int(spec["seed"]))
        resume_step = _load_ckpt_step(resume_from)

        out_dir = Path(args.work_root) / "outputs" / "checkpoints" / run_name
        meta = {
            **spec,
            "selected_gpu_id": -1,
            "lease_id": "",
            "dataset_names": ["vspw", "vipseg"],
            "obs_len": int(obs_len),
            "fut_len": int(fut_len),
            "max_tokens": int(max_tokens),
            "semantic_crop_size": int(crop_size),
            "semantic_source_mainline": "crop_visual_encoder",
            "legacy_semantic_source": "hand_crafted_stats",
            "batch_size": int(PILOT_BATCH_SIZE),
            "resume_from": str(resume_from),
            "resume_global_step": int(resume_step),
            "additional_train_steps": int(PILOT_EXTRA_STEPS),
            "train_steps": int(resume_step + PILOT_EXTRA_STEPS),
            "eval_interval": int(PILOT_EVAL_INTERVAL),
            "eval_max_batches": int(PILOT_EVAL_MAX_BATCHES),
            "save_every_n_steps": int(PILOT_SAVE_EVERY),
            "max_samples_train": int(PILOT_MAX_TRAIN_PER_DATASET),
            "max_samples_val": int(PILOT_MAX_VAL_PER_DATASET),
            "effective_train_sample_count_per_dataset": train_counts,
            "effective_val_sample_count_per_dataset": val_counts,
            "semantic_bootstrap_target_dim": 512,
            "semantic_hard_curriculum_weight": 0.0,
            "semantic_aux_subset_weighting_strength": 1.0,
            "output_dir": str(out_dir),
            "raw_json": str(Path(args.work_root) / "reports" / f"{run_name}_raw.json"),
            "progress_json": str(Path(args.work_root) / "reports" / f"{run_name}_progress.json"),
            "final_json": str(Path(args.work_root) / "reports" / f"{run_name}_final.json"),
            "log_path": str(Path(args.work_root) / "logs" / f"{run_name}.log"),
            "stage2_contract_json": str(args.stage2_contract_json),
            "stage1_runtime_json": str(args.stage1_runtime_json),
            "stage1_best_ckpt": str(args.stage1_best_ckpt),
            "shared_lease_path": str(args.shared_lease_path),
            "bootstrap_cache_jsonl": str(args.bootstrap_cache_jsonl),
            "semantic_hard_manifest_path": str(args.semantic_hard_manifest_path),
            "work_root": str(args.work_root),
            "python_bin": str(args.python_bin),
            "reserve_idle_gpu_count": int(args.reserve_idle_gpu_count),
            "gpu_acquire_timeout_seconds": int(args.gpu_acquire_timeout_seconds),
            "gpu_acquire_retry_seconds": int(args.gpu_acquire_retry_seconds),
        }
        meta_json = Path(args.work_root) / "reports" / "stage2_semantic_objective_redesign_v7_runs_20260413" / f"{run_name}_launch_meta.json"
        meta["meta_json"] = str(meta_json)
        _write_json(meta_json, meta)
        runs.append(meta)

        env = {
            "PYTHONPATH": f"{args.work_root}/code:{os.environ.get('PYTHONPATH', '')}",
        }
        env_prefix = " ".join(f"{k}={shlex.quote(v)}" for k, v in env.items())
        cmd = (
            f"{env_prefix} {shlex.quote(str(args.python_bin))} "
            f"{shlex.quote(str(Path(args.work_root) / 'code/stwm/tools/run_tracewm_stage2_semantic_objective_redesign_v7_20260413.py'))} "
            f"--mode run-one --meta-json {shlex.quote(str(meta_json))}"
        )
        subprocess.run(["tmux", "new-window", "-t", str(args.tmux_session), "-n", str(meta["window_name"]), cmd], check=True)

    payload = {
        "generated_at_utc": now_iso(),
        "mode": "stage2_semantic_objective_redesign_v7_launch",
        "tmux_session": str(args.tmux_session),
        "pilot_policy": "bounded pilot; no Stage1 mutation; keep at least one GPU idle as headroom; teacher remains target only; no codec/VAE upgrade",
        "runs": runs,
    }
    _write_json(args.v6_launch_report, payload)
    return summarize(args)


def run_one(args: Any) -> None:
    meta = _read_json(args.meta_json)
    lease_id = str(meta.get("lease_id", ""))
    lease_path = str(meta.get("shared_lease_path", ""))
    run_name = str(meta.get("run_name", ""))
    selected_gpu_id = int(meta.get("selected_gpu_id", -1))
    if selected_gpu_id < 0:
        acquire_deadline = time.time() + float(meta.get("gpu_acquire_timeout_seconds", 7200))
        last_gpu_error = ""
        while True:
            if not _gpu_headroom_ok(lease_path, int(meta.get("reserve_idle_gpu_count", 1))):
                last_gpu_error = (
                    f"gpu_headroom_blocked reserve_idle_gpu_count={int(meta.get('reserve_idle_gpu_count', 1))}"
                )
            else:
                try:
                    gpu = _select_gpu(run_name=run_name, lease_path=lease_path)
                    selected_gpu_id = int(gpu["selected_gpu_id"])
                    lease_id = str(gpu["lease_id"])
                    break
                except Exception as exc:
                    last_gpu_error = str(exc)
            if time.time() >= acquire_deadline:
                raise RuntimeError(f"gpu_acquire_timeout run={run_name} last_error={last_gpu_error}")
            time.sleep(float(meta.get("gpu_acquire_retry_seconds", 20)))

    log_path = Path(str(meta.get("log_path", "")))
    if str(log_path):
        log_path.parent.mkdir(parents=True, exist_ok=True)
    trainer = Path(str(meta["work_root"])) / "code/stwm/tracewm_v2_stage2/trainers/train_tracewm_stage2_smalltrain.py"
    cmd = [
        str(meta["python_bin"]), str(trainer),
        "--stage2-contract-path", str(meta["stage2_contract_json"]),
        "--recommended-runtime-json", str(meta["stage1_runtime_json"]),
        "--use-recommended-runtime",
        "--stage1-backbone-checkpoint", str(meta["stage1_best_ckpt"]),
        "--dataset-names", "vspw", "vipseg",
        "--train-split", "train",
        "--val-split", "val",
        "--obs-len", str(meta["obs_len"]),
        "--fut-len", str(meta["fut_len"]),
        "--max-tokens", str(meta["max_tokens"]),
        "--max-samples-train", str(meta["max_samples_train"]),
        "--max-samples-val", str(meta["max_samples_val"]),
        "--batch-size", str(meta["batch_size"]),
        "--train-steps", str(meta["train_steps"]),
        "--eval-interval", str(meta["eval_interval"]),
        "--eval-max-batches", str(meta["eval_max_batches"]),
        "--save-every-n-steps", str(meta["save_every_n_steps"]),
        "--semantic-source-mainline", str(meta["semantic_source_mainline"]),
        "--legacy-semantic-source", str(meta["legacy_semantic_source"]),
        "--semantic-crop-size", str(meta["semantic_crop_size"]),
        "--semantic-rescue-mode", str(meta["semantic_rescue_mode"]),
        "--semantic-rescue-weight", str(meta["semantic_rescue_weight"]),
        "--semantic-bootstrap-cache-path", str(meta["bootstrap_cache_jsonl"]),
        "--semantic-bootstrap-target-dim", str(meta["semantic_bootstrap_target_dim"]),
        "--semantic-hard-curriculum-weight", str(meta["semantic_hard_curriculum_weight"]),
        "--semantic-aux-subset-weighting-strength", str(meta["semantic_aux_subset_weighting_strength"]),
        "--confidence-gated-alignment-loss-weight", str(meta["confidence_gated_alignment_loss_weight"]),
        "--sparse-persistence-contrastive-loss-weight", str(meta["sparse_persistence_contrastive_loss_weight"]),
        "--confidence-gating-margin-threshold", str(meta["confidence_gating_margin_threshold"]),
        "--confidence-gating-temperature", str(meta["confidence_gating_temperature"]),
        "--semantic-hard-score-threshold", str(meta["semantic_hard_score_threshold"]),
        "--aux-loss-delay-steps", str(meta["aux_loss_delay_steps"]),
        "--aux-loss-ramp-steps", str(meta["aux_loss_ramp_steps"]),
        "--v6-gating-family", str(meta["v6_gating_family"]),
        "--v6-topk-query-k", str(meta["v6_topk_query_k"]),
        "--v6-capped-quantile", str(meta["v6_capped_quantile"]),
        "--v6-max-affected-ratio", str(meta["v6_max_affected_ratio"]),
        "--v6-gate-min-strength", str(meta["v6_gate_min_strength"]),
        "--v6-strict-max-pairs-per-sample", str(meta["v6_strict_max_pairs_per_sample"]),
        "--v6-hard-negative-cap", str(meta["v6_hard_negative_cap"]),
        "--v6-pair-sampling-temperature", str(meta["v6_pair_sampling_temperature"]),
        "--v6-guaranteed-min-pairs-per-sample", str(meta["v6_guaranteed_min_pairs_per_sample"]),
        "--v6-relaxed-motion-threshold", str(meta["v6_relaxed_motion_threshold"]),
        "--v6-relaxed-area-jump-threshold", str(meta["v6_relaxed_area_jump_threshold"]),
        "--v6-relaxed-small-query-threshold", str(meta["v6_relaxed_small_query_threshold"]),
        "--v6-relaxed-appearance-shift-threshold", str(meta["v6_relaxed_appearance_shift_threshold"]),
        "--v6-relaxed-center-interaction-threshold", str(meta["v6_relaxed_center_interaction_threshold"]),
        "--semantic-hard-manifest-path", str(meta["semantic_hard_manifest_path"]),
        "--resume-from", str(meta["resume_from"]),
        "--skip-resume-optimizer",
        "--semantic-hard-sidecar-enabled",
        "--output-dir", str(meta["output_dir"]),
        "--run-name", str(meta["run_name"]),
        "--run-summary-json", str(meta["raw_json"]),
        "--progress-json", str(meta["progress_json"]),
        "--seed", str(meta["seed"]),
    ]
    cmd.append("--v6-two-level-pair-mining-enabled" if bool(meta["v6_two_level_pair_mining_enabled"]) else "--no-v6-two-level-pair-mining-enabled")
    try:
        proc_env = os.environ.copy()
        proc_env["CUDA_VISIBLE_DEVICES"] = str(selected_gpu_id)
        proc_env["TRACEWM_STAGE1_V2_GPU_SELECTION_METADATA_JSON"] = json.dumps(
            {
                "selected_gpu_id": int(selected_gpu_id),
                "lease_id": str(lease_id),
                "owner": str(meta.get("run_name", "")),
                "mode": "single_gpu_only",
            },
            ensure_ascii=True,
        )
        with log_path.open("w", encoding="utf-8") as log_fh:
            proc = subprocess.run(
                cmd,
                cwd=str(meta["work_root"]),
                text=True,
                stdout=log_fh,
                stderr=subprocess.STDOUT,
                env=proc_env,
            )
        if proc.returncode != 0:
            log_tail = ""
            try:
                log_tail = log_path.read_text(encoding="utf-8")[-4000:]
            except Exception:
                log_tail = ""
            _write_json(
                meta["final_json"],
                {
                    "generated_at_utc": now_iso(),
                    "run_name": meta["run_name"],
                    "status": "failed",
                    "returncode": proc.returncode,
                    "log_tail": log_tail,
                },
            )
            raise RuntimeError(f"trainer failed rc={proc.returncode}")
        raw = _read_json(meta["raw_json"])
        raw.update(
            {
                "generated_at_utc": now_iso(),
                "status": "completed",
                "selected_gpu_id": int(selected_gpu_id),
                "lease_id": str(lease_id),
                "objective_combo": str(meta["objective_combo"]),
                "objective_family": str(meta.get("objective_family", "")),
                "persistence_objective_declared": bool(meta.get("persistence_objective_declared", False)),
                "resume_global_step": int(meta["resume_global_step"]),
            }
        )
        _write_json(meta["final_json"], raw)
    except Exception as exc:
        _write_json(
            meta["final_json"],
            {
                "generated_at_utc": now_iso(),
                "run_name": str(meta.get("run_name", "")),
                "status": "failed",
                "message": str(exc),
                "traceback": traceback.format_exc(),
            },
        )
        raise
    finally:
        _release_lease_safe(lease_id=lease_id, lease_path=lease_path)


def _status_for(meta: Dict[str, Any], session_name: str) -> Dict[str, Any]:
    final_path = Path(str(meta.get("final_json", "")))
    progress_path = Path(str(meta.get("progress_json", "")))
    if str(meta.get("window_name", "")) in _tmux_windows(session_name):
        detail = _read_json(progress_path) if progress_path.exists() else {}
        return {"status": "running", "detail": detail}
    if final_path.exists():
        try:
            detail = _read_json(final_path)
            status = str(detail.get("status", "launched")).lower()
            if status in {"completed", "failed"}:
                return {"status": status, "detail": detail}
        except Exception:
            pass
    detail = _read_json(progress_path) if progress_path.exists() else {}
    return {"status": str(detail.get("status", "launched")).lower() if detail else "launched", "detail": detail}


def _json_or_empty(path_like: Any) -> Dict[str, Any]:
    path = Path(str(path_like))
    if not str(path_like) or not path.exists():
        return {}
    try:
        payload = _read_json(path)
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _metric_block(block: Any) -> Dict[str, Any]:
    return block if isinstance(block, dict) else {}


def _metric_payload(block: Any) -> Dict[str, Any]:
    payload = _metric_block(block).get("metrics", {})
    return payload if isinstance(payload, dict) else {}


def _metric_rank_tuple(block: Any) -> Tuple[float, float, float]:
    metrics = _metric_payload(block)
    return (
        _f(metrics.get("free_rollout_endpoint_l2"), 1e9),
        _f(metrics.get("free_rollout_coord_mean_l2"), 1e9),
        _f(metrics.get("teacher_forced_coord_loss"), 1e9),
    )


def _run_family_label(payload: Dict[str, Any]) -> str:
    run_name = str(payload.get("run_name", ""))
    objective_family = str(payload.get("objective_family", ""))
    if "alignonly" in run_name or objective_family == "calibration_only_family":
        return "alignonly"
    if "alignpersist" in run_name or objective_family == "calibration_plus_active_persistence_family":
        return "alignpersist"
    return "unknown"


def _launch_meta_by_run(args: Any) -> Dict[str, Dict[str, Any]]:
    launch = _json_or_empty(args.v6_launch_report)
    return {
        str(meta.get("run_name", "")): meta
        for meta in launch.get("runs", [])
        if isinstance(meta, dict) and str(meta.get("run_name", "")).strip()
    }


def _best_block_from_payloads(final_payload: Dict[str, Any], raw_payload: Dict[str, Any], progress_payload: Dict[str, Any]) -> Dict[str, Any]:
    for payload in [final_payload, raw_payload, progress_payload]:
        block = payload.get("best_checkpoint_metric", {})
        if isinstance(block, dict) and isinstance(block.get("metrics", {}), dict):
            return block
        block = payload.get("best_metric_so_far", {})
        if isinstance(block, dict) and isinstance(block.get("metrics", {}), dict):
            return block
    return {}


def _latest_block_from_payloads(final_payload: Dict[str, Any], raw_payload: Dict[str, Any], progress_payload: Dict[str, Any]) -> Dict[str, Any]:
    for payload in [final_payload, raw_payload, progress_payload]:
        block = payload.get("latest_checkpoint_metric", {})
        if isinstance(block, dict) and isinstance(block.get("metrics", {}), dict):
            return block
        block = payload.get("latest_eval_metrics", {})
        if isinstance(block, dict) and isinstance(block.get("metrics", {}), dict):
            return block
    return {}


def _sidecar_block_from_payloads(final_payload: Dict[str, Any], raw_payload: Dict[str, Any]) -> Dict[str, Any]:
    for payload in [final_payload, raw_payload]:
        block = payload.get("semantic_hard_sidecar_metric", {})
        if isinstance(block, dict):
            return block
    return {}


def _branch_block_from_payloads(final_payload: Dict[str, Any], raw_payload: Dict[str, Any]) -> Dict[str, Any]:
    for payload in [final_payload, raw_payload]:
        block = payload.get("semantic_branch_metrics", {})
        if isinstance(block, dict):
            return block
    return {}


def _sidecar_selection_from_payloads(final_payload: Dict[str, Any], raw_payload: Dict[str, Any]) -> Dict[str, Any]:
    for payload in [final_payload, raw_payload]:
        block = payload.get("sidecar_checkpoint_selection", {})
        if isinstance(block, dict):
            return block
    return {}


def _v7_paths(args: Any, meta: Dict[str, Any], run_name: str) -> Dict[str, Path]:
    report_root = Path(args.work_root) / "reports"
    ckpt_root = Path(args.work_root) / "outputs" / "checkpoints" / run_name
    return {
        "launch": Path(args.work_root) / "reports" / "stage2_semantic_objective_redesign_v7_runs_20260413" / f"{run_name}_launch_meta.json",
        "progress": Path(str(meta.get("progress_json", report_root / f"{run_name}_progress.json"))),
        "final": Path(str(meta.get("final_json", report_root / f"{run_name}_final.json"))),
        "raw": Path(str(meta.get("raw_json", report_root / f"{run_name}_raw.json"))),
        "best": ckpt_root / "best.pt",
        "latest": ckpt_root / "latest.pt",
        "sidecar": ckpt_root / "best_semantic_hard.pt",
    }


def audit_v7_artifacts(args: Any) -> Dict[str, Any]:
    meta_by_run = _launch_meta_by_run(args)
    rows: List[Dict[str, Any]] = []
    launched = running = completed = failed = 0
    for spec in _run_specs():
        run_name = str(spec.get("run_name", ""))
        meta = meta_by_run.get(run_name, {})
        paths = _v7_paths(args, meta, run_name)
        launch_exists = paths["launch"].exists()
        progress_exists = paths["progress"].exists()
        final_exists = paths["final"].exists()
        raw_exists = paths["raw"].exists()
        best_ckpt_exists = paths["best"].exists()
        latest_ckpt_exists = paths["latest"].exists()
        sidecar_exists = paths["sidecar"].exists()

        launched += int(launch_exists)
        progress_payload = _json_or_empty(paths["progress"])
        final_payload = _json_or_empty(paths["final"])
        raw_payload = _json_or_empty(paths["raw"])
        status_info = _status_for(
            {
                **meta,
                "window_name": str(meta.get("window_name", spec.get("window_name", ""))),
                "progress_json": str(paths["progress"]),
                "final_json": str(paths["final"]),
            },
            session_name=str(args.tmux_session),
        )
        resolved_status = str(status_info.get("status", "launched")).lower()
        running += int(resolved_status == "running")
        completed += int(resolved_status == "completed")
        failed += int(resolved_status == "failed")

        progress_status = str(progress_payload.get("status", "")).lower()
        final_status = str(final_payload.get("status", "")).lower()
        best_block = _best_block_from_payloads(final_payload, raw_payload, progress_payload)
        latest_block = _latest_block_from_payloads(final_payload, raw_payload, progress_payload)
        sidecar_block = _sidecar_block_from_payloads(final_payload, raw_payload)
        checkpoint_inventory = progress_payload.get("checkpoint_inventory", {}) if isinstance(progress_payload.get("checkpoint_inventory", {}), dict) else {}

        notes: List[str] = []
        latest_step = int(_metric_block(latest_block).get("global_step", -1) or -1)
        progress_step = int(progress_payload.get("global_step", -1) or -1)
        inventory_best_ok = (
            True
            if not checkpoint_inventory
            else str(checkpoint_inventory.get("best", "")) == str(paths["best"]) and bool(checkpoint_inventory.get("best_exists", False)) == bool(best_ckpt_exists)
        )
        inventory_latest_ok = (
            True
            if not checkpoint_inventory
            else str(checkpoint_inventory.get("latest", "")) == str(paths["latest"]) and bool(checkpoint_inventory.get("latest_exists", False)) == bool(latest_ckpt_exists)
        )
        status_consistent = bool(
            resolved_status in {"completed", "failed", "running", "launched"}
            and (not progress_exists or progress_status in {"completed", "failed", "running", "launched", ""})
            and (not final_exists or final_status in {"completed", "failed", ""})
            and (not (progress_exists and final_exists) or progress_status == final_status or (progress_status == "completed" and final_status == "completed"))
            and (progress_step < 0 or latest_step < 0 or progress_step == latest_step)
            and inventory_best_ok
            and inventory_latest_ok
        )
        if final_exists and not final_payload.get("global_step", None) and progress_step >= 0:
            notes.append("final_global_step_missing_but_progress_global_step_present")
        if final_exists and latest_step >= 0 and progress_step >= 0 and latest_step == progress_step:
            notes.append("latest_checkpoint_step_matches_progress_global_step")
        if sidecar_exists and not checkpoint_inventory.get("sidecar", ""):
            notes.append("semantic_hard_sidecar_not_tracked_in_progress_checkpoint_inventory")
        if not status_consistent:
            notes.append("status_or_checkpoint_consistency_failed")

        salvageable_if_missing = bool(
            (not final_exists)
            and progress_exists
            and progress_status == "completed"
            and best_ckpt_exists
            and latest_ckpt_exists
        )
        salvage_source = "progress_json+checkpoint_inventory+checkpoints" if salvageable_if_missing else "none"
        if final_exists:
            salvage_source = "not_needed_final_exists"

        rows.append(
            {
                "run_name": run_name,
                "launch_exists": bool(launch_exists),
                "progress_exists": bool(progress_exists),
                "final_exists": bool(final_exists),
                "raw_exists": bool(raw_exists),
                "best_ckpt_exists": bool(best_ckpt_exists),
                "latest_ckpt_exists": bool(latest_ckpt_exists),
                "sidecar_exists": bool(sidecar_exists),
                "resolved_status": resolved_status,
                "progress_status": progress_status or "missing",
                "final_status": final_status or "missing",
                "progress_global_step": progress_step,
                "final_best_global_step": int(_metric_block(best_block).get("global_step", -1) or -1),
                "final_latest_global_step": latest_step,
                "final_sidecar_global_step": int(sidecar_block.get("global_step", -1) or -1),
                "status_consistent": bool(status_consistent),
                "salvageable_if_missing": bool(salvageable_if_missing),
                "salvage_source": salvage_source,
                "notes": notes,
            }
        )

    payload = {
        "generated_at_utc": now_iso(),
        "audit_type": "stage2_semantic_objective_redesign_v7_repair_audit",
        "run_count": int(len(rows)),
        "launch_count": int(launched),
        "running_count": int(running),
        "completed_count": int(completed),
        "failed_count": int(failed),
        "all_runs_terminal": bool(len(rows) > 0 and running == 0 and completed + failed == len(rows)),
        "run_rows": rows,
    }
    _write_json(args.v7_repair_audit_report, payload)
    lines = [
        "# Stage2 Semantic Objective Redesign V7 Repair Audit",
        "",
        f"- generated_at_utc: {payload['generated_at_utc']}",
        f"- run_count: {payload['run_count']}",
        f"- running_count: {payload['running_count']}",
        f"- completed_count: {payload['completed_count']}",
        f"- failed_count: {payload['failed_count']}",
        f"- all_runs_terminal: {payload['all_runs_terminal']}",
        "",
        "| run_name | launch | progress | final | best | latest | sidecar | status | consistent | salvageable_if_missing | notes |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        notes = "; ".join(row.get("notes", [])) if isinstance(row.get("notes", []), list) else ""
        lines.append(
            f"| {row['run_name']} | {row['launch_exists']} | {row['progress_exists']} | {row['final_exists']} | {row['best_ckpt_exists']} | {row['latest_ckpt_exists']} | {row['sidecar_exists']} | {row['resolved_status']} | {row['status_consistent']} | {row['salvageable_if_missing']} | {notes} |"
        )
    _write_md(args.v7_repair_audit_doc, lines)
    return payload


def _summary_overall_rank(row: Dict[str, Any]) -> Tuple[float, float, float, int, float, str]:
    best_block = row.get("best_checkpoint_metric", {}) if isinstance(row.get("best_checkpoint_metric", {}), dict) else {}
    endpoint, coord, teacher = _metric_rank_tuple(best_block)
    inactive_penalty = 1 if bool(row.get("persistence_declared_but_inactive", False)) else 0
    hard_score = _f(row.get("semantic_hard_sidecar_metric", {}).get("semantic_hard_sidecar_score"), 1e9) if isinstance(row.get("semantic_hard_sidecar_metric", {}), dict) else 1e9
    return (endpoint, coord, teacher, inactive_penalty, hard_score, str(row.get("run_name", "")))


def _summary_hard_rank(row: Dict[str, Any]) -> Tuple[float, float, float, float, str]:
    best_block = row.get("best_checkpoint_metric", {}) if isinstance(row.get("best_checkpoint_metric", {}), dict) else {}
    endpoint, coord, teacher = _metric_rank_tuple(best_block)
    hard_score = _f(row.get("semantic_hard_sidecar_metric", {}).get("semantic_hard_sidecar_score"), 1e9) if isinstance(row.get("semantic_hard_sidecar_metric", {}), dict) else 1e9
    return (hard_score, endpoint, coord, teacher, str(row.get("run_name", "")))


def summarize(args: Any) -> Dict[str, Any]:
    audit = audit_v7_artifacts(args)
    meta_by_run = _launch_meta_by_run(args)
    run_rows: List[Dict[str, Any]] = []
    for audit_row in audit.get("run_rows", []):
        if not isinstance(audit_row, dict):
            continue
        run_name = str(audit_row.get("run_name", ""))
        meta = meta_by_run.get(run_name, {})
        spec = next((x for x in _run_specs() if str(x.get("run_name", "")) == run_name), {})
        paths = _v7_paths(args, meta, run_name)
        progress_payload = _json_or_empty(paths["progress"])
        final_payload = _json_or_empty(paths["final"])
        raw_payload = _json_or_empty(paths["raw"])
        best_block = _best_block_from_payloads(final_payload, raw_payload, progress_payload)
        latest_block = _latest_block_from_payloads(final_payload, raw_payload, progress_payload)
        sidecar_block = _sidecar_block_from_payloads(final_payload, raw_payload)
        branch = _branch_block_from_payloads(final_payload, raw_payload)
        sidecar_sel = _sidecar_selection_from_payloads(final_payload, raw_payload)

        declared = bool(final_payload.get("persistence_objective_declared", meta.get("persistence_objective_declared", spec.get("persistence_objective_declared", False))))
        valuable_pair_ratio = float(branch.get("valuable_pair_ratio_mean", 0.0))
        guaranteed_pair_count = float(branch.get("guaranteed_pair_count_mean", 0.0))
        persistence_effective = bool(declared and guaranteed_pair_count >= 1.0 and valuable_pair_ratio > 0.0)
        global_step = int(progress_payload.get("global_step", _metric_block(latest_block).get("global_step", _metric_block(best_block).get("global_step", -1))) or -1)

        run_rows.append(
            {
                "run_name": run_name,
                "family": _run_family_label({**spec, **meta, **final_payload}),
                "objective_combo": str(meta.get("objective_combo", spec.get("objective_combo", ""))),
                "objective_family": str(meta.get("objective_family", spec.get("objective_family", ""))),
                "gating_family": str(meta.get("v6_gating_family", spec.get("v6_gating_family", ""))),
                "seed": int(meta.get("seed", spec.get("seed", -1))),
                "status": str(audit_row.get("resolved_status", "launched")),
                "global_step": global_step,
                "final_json_exists": bool(audit_row.get("final_exists", False)),
                "progress_json_exists": bool(audit_row.get("progress_exists", False)),
                "raw_json_exists": bool(audit_row.get("raw_exists", False)),
                "best_ckpt_exists": bool(audit_row.get("best_ckpt_exists", False)),
                "latest_ckpt_exists": bool(audit_row.get("latest_ckpt_exists", False)),
                "sidecar_exists": bool(audit_row.get("sidecar_exists", False)),
                "selected_gpu_id": int(final_payload.get("selected_gpu_id", meta.get("selected_gpu_id", -1))),
                "lease_id": str(final_payload.get("lease_id", meta.get("lease_id", ""))),
                "batch_size": int(meta.get("batch_size", 0)),
                "train_steps": int(meta.get("train_steps", 0)),
                "eval_interval": int(meta.get("eval_interval", 0)),
                "save_every_n_steps": int(meta.get("save_every_n_steps", 0)),
                "effective_train_sample_count_per_dataset": meta.get("effective_train_sample_count_per_dataset", {}),
                "effective_val_sample_count_per_dataset": meta.get("effective_val_sample_count_per_dataset", {}),
                "best_checkpoint_metric": best_block,
                "latest_checkpoint_metric": latest_block,
                "semantic_hard_sidecar_metric": sidecar_block,
                "actual_gate_positive_ratio_mean": float(branch.get("actual_gate_positive_ratio_mean", 1.0)),
                "valuable_pair_ratio_mean": float(valuable_pair_ratio),
                "guaranteed_pair_count_mean": float(guaranteed_pair_count),
                "persistence_objective_declared": bool(declared),
                "persistence_objective_effective": bool(persistence_effective),
                "persistence_declared_but_inactive": bool(declared and not persistence_effective),
                "notes": list(audit_row.get("notes", [])) if isinstance(audit_row.get("notes", []), list) else [],
            }
        )

    completed_rows = [row for row in run_rows if str(row.get("status", "")).lower() == "completed"]
    overall_best_run_name = "none"
    semantic_hard_best_run_name = "none"
    best_effective_persistence_run_name = "none"
    if completed_rows:
        overall_best_run_name = str(min(completed_rows, key=_summary_overall_rank).get("run_name", "none"))
        semantic_hard_best_run_name = str(min(completed_rows, key=_summary_hard_rank).get("run_name", "none"))
        effective_persist_rows = [row for row in completed_rows if bool(row.get("persistence_objective_effective", False))]
        if effective_persist_rows:
            best_effective_persistence_run_name = str(min(effective_persist_rows, key=_summary_hard_rank).get("run_name", "none"))

    payload = {
        "generated_at_utc": now_iso(),
        "redesign_v7_status": f"{audit.get('running_count', 0)}_running_{audit.get('completed_count', 0)}_completed_{audit.get('failed_count', 0)}_failed",
        "running_count": int(audit.get("running_count", 0)),
        "completed_count": int(audit.get("completed_count", 0)),
        "failed_count": int(audit.get("failed_count", 0)),
        "all_runs_terminal": bool(audit.get("all_runs_terminal", False)),
        "run_rows": run_rows,
        "runs": run_rows,
        "overall_best_run_name": overall_best_run_name,
        "semantic_hard_best_run_name": semantic_hard_best_run_name,
        "best_effective_persistence_run_name": best_effective_persistence_run_name,
        "next_step_choice_internal": "",
    }
    payload["next_step_choice_internal"] = (
        "ready_for_v7_repaired_diagnosis"
        if payload["all_runs_terminal"] and payload["failed_count"] == 0 and payload["completed_count"] == len(run_rows)
        else ("fix_failed_v7_runs" if payload["failed_count"] > 0 else "continue_v7_until_terminal")
    )
    _write_json(args.v6_summary_report, payload)
    return payload


def wait_for_completion(args: Any) -> Dict[str, Any]:
    deadline = time.time() + float(args.wait_timeout_seconds)
    last = summarize(args)
    while time.time() < deadline:
        if str(last.get("redesign_v7_status", "")).startswith("0_running_"):
            return last
        time.sleep(float(args.poll_seconds))
        last = summarize(args)
    last["timed_out_waiting_for_completion"] = True
    _write_json(args.v6_summary_report, last)
    return last


def _row_map(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {str(row.get("run_name", "")): row for row in rows if isinstance(row, dict)}


def _summary_row_anchor(row: Dict[str, Any]) -> Dict[str, Any]:
    best_block = row.get("best_checkpoint_metric", {}) if isinstance(row.get("best_checkpoint_metric", {}), dict) else {}
    return {
        "run_name": str(row.get("run_name", "none")),
        "family": str(row.get("family", "unknown")),
        "seed": int(row.get("seed", -1)),
        "best_checkpoint_metric": best_block,
        "semantic_hard_sidecar_metric": row.get("semantic_hard_sidecar_metric", {}) if isinstance(row.get("semantic_hard_sidecar_metric", {}), dict) else {},
        "actual_gate_positive_ratio_mean": float(row.get("actual_gate_positive_ratio_mean", 1.0)),
        "valuable_pair_ratio_mean": float(row.get("valuable_pair_ratio_mean", 0.0)),
        "guaranteed_pair_count_mean": float(row.get("guaranteed_pair_count_mean", 0.0)),
        "persistence_declared_but_inactive": bool(row.get("persistence_declared_but_inactive", False)),
    }


def _semantic_hard_score_from_row(row: Dict[str, Any]) -> float:
    sidecar = row.get("semantic_hard_sidecar_metric", {}) if isinstance(row.get("semantic_hard_sidecar_metric", {}), dict) else {}
    return _f(sidecar.get("semantic_hard_sidecar_score"), 1e9)


def _seed_level_alignment_support(seed_rows: List[Dict[str, Any]]) -> bool:
    align_rows = [row for row in seed_rows if str(row.get("family", "")) == "alignonly"]
    effective_persist_rows = [
        row for row in seed_rows
        if str(row.get("family", "")) == "alignpersist" and bool(row.get("persistence_objective_effective", False))
    ]
    if not align_rows:
        return False
    best_align = min(align_rows, key=_summary_overall_rank)
    if not effective_persist_rows:
        return True
    best_effective = min(effective_persist_rows, key=_summary_hard_rank)
    align_full = _metric_rank_tuple(best_align.get("best_checkpoint_metric", {}))[0]
    eff_full = _metric_rank_tuple(best_effective.get("best_checkpoint_metric", {}))[0]
    align_hard = _semantic_hard_score_from_row(best_align)
    eff_hard = _semantic_hard_score_from_row(best_effective)
    return not (eff_hard < align_hard * 0.98 and eff_full <= align_full * 1.02)


def _write_v7_results_md(args: Any, audit: Dict[str, Any], summary: Dict[str, Any], diagnosis: Dict[str, Any]) -> None:
    run_rows = [row for row in summary.get("run_rows", []) if isinstance(row, dict)]
    lines = [
        "# Stage2 Semantic Objective Redesign V7 Results",
        "",
        f"- generated_at_utc: {now_iso()}",
        f"- v7_runs_terminal: {bool(summary.get('all_runs_terminal', False))}",
        f"- running_count: {int(summary.get('running_count', 0))}",
        f"- completed_count: {int(summary.get('completed_count', 0))}",
        f"- failed_count: {int(summary.get('failed_count', 0))}",
        f"- overall_best_run_name: {diagnosis.get('overall_best_run_name', 'none')}",
        f"- semantic_hard_best_run_name: {diagnosis.get('semantic_hard_best_run_name', 'none')}",
        f"- best_effective_persistence_run_name: {diagnosis.get('best_effective_persistence_run_name', 'none')}",
        f"- true_new_best_not_warm_start_inherited: {bool(diagnosis.get('true_new_best_not_warm_start_inherited', False))}",
        f"- actual_gate_positive_ratio_below_0_30: {bool(diagnosis.get('actual_gate_positive_ratio_below_0_30', False))}",
        f"- semantic_hard_composite_improved_vs_v6: {bool(diagnosis.get('semantic_hard_composite_improved_vs_v6', False))}",
        f"- cross_seed_support_present: {bool(diagnosis.get('cross_seed_support_present', False))}",
        f"- alignment_only_is_already_sufficient: {bool(diagnosis.get('alignment_only_is_already_sufficient', False))}",
        f"- persistence_branch_actually_contributed: {bool(diagnosis.get('persistence_branch_actually_contributed', False))}",
        f"- persistence_declared_but_inactive_any: {bool(diagnosis.get('persistence_declared_but_inactive_any', False))}",
        f"- persistence_declared_but_inactive_all: {bool(diagnosis.get('persistence_declared_but_inactive_all', False))}",
        f"- next_step_choice: {diagnosis.get('next_step_choice', 'none')}",
        "",
        "## Final Read",
        "",
        f"- v7 的 8 个 run 真实终态: {int(audit.get('completed_count', 0))} completed / {int(audit.get('failed_count', 0))} failed / {int(audit.get('running_count', 0))} running。",
        f"- overall best: `{diagnosis.get('overall_best_run_name', 'none')}`。",
        f"- semantic-hard best: `{diagnosis.get('semantic_hard_best_run_name', 'none')}`。",
        f"- best effective persistence run: `{diagnosis.get('best_effective_persistence_run_name', 'none')}`。",
        "- 当前修复后的 family verdict 更支持 calibration-only / alignment-only line，而不支持把 persistence-aware line 写成已被证实的主线。",
        "- 若 semantic-hard best 来自 persistence-declared run，但 persistence telemetry 仍为 inactive，则该结果只能记为 sidecar probe，不得解释为 persistence 分支已实际起效。",
        "",
        "## Run Table",
        "",
        "| run_name | family | seed | status | global_step | best_endpoint_l2 | semantic_hard_composite | gate_ratio | valuable_pair_ratio | guaranteed_pair_count | declared_but_inactive |",
        "|---|---|---:|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in run_rows:
        best_metrics = _metric_payload(row.get("best_checkpoint_metric", {}))
        sidecar = row.get("semantic_hard_sidecar_metric", {}) if isinstance(row.get("semantic_hard_sidecar_metric", {}), dict) else {}
        lines.append(
            f"| {row.get('run_name', '')} | {row.get('family', '')} | {int(row.get('seed', -1))} | {row.get('status', '')} | {int(row.get('global_step', -1))} | {_f(best_metrics.get('free_rollout_endpoint_l2'), 1e9):.8f} | {_f(sidecar.get('semantic_hard_sidecar_score'), 1e9):.8f} | {float(row.get('actual_gate_positive_ratio_mean', 1.0)):.4f} | {float(row.get('valuable_pair_ratio_mean', 0.0)):.4f} | {float(row.get('guaranteed_pair_count_mean', 0.0)):.4f} | {bool(row.get('persistence_declared_but_inactive', False))} |"
        )
    _write_md(args.v6_results_md, lines)


def _repair_qualitative_v3_refs(args: Any, diagnosis: Dict[str, Any]) -> None:
    stage1_path = Path(str(args.stage1_qualitative_pack_v3_report))
    stage2_path = Path(str(args.stage2_qualitative_pack_v3_report))
    doc_path = Path(str(args.stage1_stage2_qualitative_pack_v3_doc))

    stage1_payload = _json_or_empty(stage1_path)
    stage2_payload = _json_or_empty(stage2_path)
    if stage2_payload:
        stage2_payload["best_alignment_only_run"] = str(diagnosis.get("overall_best_run_name", stage2_payload.get("best_alignment_only_run", "none")))
        stage2_payload["best_persistence_declared_run"] = str(
            diagnosis.get("semantic_hard_best_run_name", stage2_payload.get("best_persistence_declared_run", "none"))
        )
        stage2_payload["best_effective_persistence_run"] = str(diagnosis.get("best_effective_persistence_run_name", "none"))
        stage2_payload["persistence_declared_but_inactive_all"] = bool(diagnosis.get("persistence_declared_but_inactive_all", False))
        stage2_payload["v7_repair_verdict"] = {
            "alignment_only_is_already_sufficient": bool(diagnosis.get("alignment_only_is_already_sufficient", False)),
            "persistence_branch_actually_contributed": bool(diagnosis.get("persistence_branch_actually_contributed", False)),
            "next_step_choice": str(diagnosis.get("next_step_choice", "none")),
        }
        repaired_cases: List[Dict[str, Any]] = []
        for case in stage2_payload.get("cases", []) if isinstance(stage2_payload.get("cases", []), list) else []:
            if not isinstance(case, dict):
                continue
            case_copy = dict(case)
            if str(case_copy.get("group", "")) == "persistence_active_improved_cases":
                case_copy["group"] = "persistence_declared_probe_cases"
                case_copy["why_selected"] = (
                    "persistence-declared checkpoint looked favorable on this semantic-hard clip, "
                    "but v7 repair audit marked persistence telemetry inactive; keep only as probe case"
                )
                case_copy["qualitative_interpretation"] = (
                    "this clip visually favors the persistence-declared sidecar checkpoint, "
                    "but it is not valid evidence that the persistence branch became effective"
                )
            repaired_cases.append(case_copy)
        stage2_payload["cases"] = repaired_cases
        selection_policy = stage2_payload.get("selection_policy", {})
        if isinstance(selection_policy, dict) and isinstance(selection_policy.get("taxonomy", []), list):
            selection_policy["taxonomy"] = [
                "persistence_declared_probe_cases" if str(x) == "persistence_active_improved_cases" else x
                for x in selection_policy.get("taxonomy", [])
            ]
        _write_json(stage2_path, stage2_payload)

    if stage1_payload or stage2_payload:
        lines = [
            "# Stage1 / Stage2 Qualitative Pack V3",
            "",
            f"- generated_at_utc: {now_iso()}",
            f"- stage1_pack: {stage1_path}",
            f"- stage2_pack: {stage2_path}",
            f"- v7_repaired_overall_best: {diagnosis.get('overall_best_run_name', 'none')}",
            f"- v7_repaired_semantic_hard_best: {diagnosis.get('semantic_hard_best_run_name', 'none')}",
            f"- v7_repaired_best_effective_persistence: {diagnosis.get('best_effective_persistence_run_name', 'none')}",
            "",
        ]
        if stage1_payload:
            lines.extend(
                [
                    "## Stage1",
                    "",
                    "| case_id | bucket | dataset | clip_id | render |",
                    "|---|---|---|---|---|",
                ]
            )
            for case in stage1_payload.get("cases", []) if isinstance(stage1_payload.get("cases", []), list) else []:
                if not isinstance(case, dict):
                    continue
                lines.append(
                    f"| {case.get('case_id', '')} | {case.get('bucket', '')} | {case.get('dataset_source', '')} | {case.get('clip_id', '')} | {case.get('render_path', '')} |"
                )
            lines.append("")
        if stage2_payload:
            lines.extend(
                [
                    "## Stage2",
                    "",
                    "- 说明: `persistence_declared_probe_cases` 只表示 persistence-declared checkpoint 在个别 hard case 上看起来更好，但 v7 repair audit 判定 persistence telemetry 仍然 inactive，不能当作 persistence 已实际起效的证据。",
                    "",
                    "| case_id | group | dataset | clip_id | why_selected | qualitative_interpretation | render |",
                    "|---|---|---|---|---|---|---|",
                ]
            )
            for case in stage2_payload.get("cases", []) if isinstance(stage2_payload.get("cases", []), list) else []:
                if not isinstance(case, dict):
                    continue
                lines.append(
                    f"| {case.get('case_id', '')} | {case.get('group', '')} | {case.get('dataset_source', '')} | {case.get('clip_id', '')} | {case.get('why_selected', '')} | {case.get('qualitative_interpretation', '')} | {case.get('render_path', '')} |"
                )
        _write_md(doc_path, lines)


def diagnose_v7(args: Any) -> Dict[str, Any]:
    summary = summarize(args)
    audit = _json_or_empty(args.v7_repair_audit_report)
    refs = _baseline_refs()
    v5 = _json_or_empty(args.v5_reference_diagnosis_report)
    v6 = _json_or_empty(args.v6_reference_diagnosis_report)
    v6_summary = _json_or_empty(args.v6_reference_summary_report)
    warm_start = _warm_start_anchor_refs()

    rows = [row for row in summary.get("run_rows", []) if isinstance(row, dict)]
    row_by_name = _row_map(rows)
    completed = [row for row in rows if str(row.get("status", "")).lower() == "completed"]

    crop_ep = _f(refs.get("cropenc_fullscale_mean", {}).get("free_rollout_endpoint_l2", {}).get("mean"), 1e9)
    legacy_ep = _f(refs.get("legacysem_fullscale_mean", {}).get("free_rollout_endpoint_l2", {}).get("mean"), 1e9)
    v6_best_hard_comp = _f(v6.get("success_criteria", {}).get("best_v6_semantic_hard_composite_score"), 1e9)

    overall_best_run_name = str(summary.get("overall_best_run_name", "none"))
    semantic_hard_best_run_name = str(summary.get("semantic_hard_best_run_name", "none"))
    best_effective_persistence_run_name = str(summary.get("best_effective_persistence_run_name", "none"))

    overall_best_row = row_by_name.get(overall_best_run_name, {})
    semantic_hard_best_row = row_by_name.get(semantic_hard_best_run_name, {})
    effective_persist_row = row_by_name.get(best_effective_persistence_run_name, {})

    align_rows = [row for row in completed if str(row.get("family", "")) == "alignonly"]
    persist_rows = [row for row in completed if str(row.get("family", "")) == "alignpersist"]
    align_best_row = min(align_rows, key=_summary_overall_rank) if align_rows else {}
    persist_best_row = min(persist_rows, key=_summary_hard_rank) if persist_rows else {}

    overall_best_rank = _metric_rank_tuple(overall_best_row.get("best_checkpoint_metric", {}))
    align_best_rank = _metric_rank_tuple(align_best_row.get("best_checkpoint_metric", {}))
    effective_persist_rank = _metric_rank_tuple(effective_persist_row.get("best_checkpoint_metric", {}))
    align_best_hard = _semantic_hard_score_from_row(align_best_row) if align_best_row else 1e9
    semantic_hard_best_score = _semantic_hard_score_from_row(semantic_hard_best_row) if semantic_hard_best_row else 1e9
    effective_persist_hard = _semantic_hard_score_from_row(effective_persist_row) if effective_persist_row else 1e9

    overall_seed = str(overall_best_row.get("seed", ""))
    overall_best_step = int(_metric_block(overall_best_row.get("best_checkpoint_metric", {})).get("global_step", -1) or -1)
    warm_anchor_step = int(warm_start.get(overall_seed, {}).get("global_step", -1) or -1)
    true_new_best_not_warm_start_inherited = bool(overall_best_step > warm_anchor_step >= 0)

    gate_ratios = [float(row.get("actual_gate_positive_ratio_mean", 1.0)) for row in completed]
    actual_gate_positive_ratio_below_0_30 = bool(gate_ratios and max(gate_ratios) < 0.30)

    semantic_hard_composite_improved_vs_v6 = bool(semantic_hard_best_score < v6_best_hard_comp)
    persistence_declared_but_inactive_any = bool(any(bool(row.get("persistence_declared_but_inactive", False)) for row in persist_rows))
    persistence_declared_but_inactive_all = bool(bool(persist_rows) and all(bool(row.get("persistence_declared_but_inactive", False)) for row in persist_rows))

    persistence_branch_actually_contributed = bool(
        best_effective_persistence_run_name != "none"
        and bool(effective_persist_row.get("persistence_objective_effective", False))
        and effective_persist_hard < align_best_hard * 0.98
        and effective_persist_rank[0] <= align_best_rank[0] * 1.02
    )
    alignment_only_is_already_sufficient = bool(
        overall_best_run_name != "none"
        and str(overall_best_row.get("family", "")) == "alignonly"
        and (
            best_effective_persistence_run_name == "none"
            or not persistence_branch_actually_contributed
        )
    )

    seeds = sorted({int(row.get("seed", -1)) for row in completed if int(row.get("seed", -1)) >= 0})
    alignment_support_seeds = [
        seed for seed in seeds
        if _seed_level_alignment_support([row for row in completed if int(row.get("seed", -1)) == seed])
    ]
    cross_seed_support_present = bool(len(alignment_support_seeds) >= 2)

    if alignment_only_is_already_sufficient:
        next_step_choice = "alignment_only_is_true_mainline"
    elif persistence_branch_actually_contributed:
        next_step_choice = "persistence_is_load_bearing"
    elif persistence_declared_but_inactive_all:
        next_step_choice = "persistence_declared_but_not_active"
    else:
        next_step_choice = "still_cannot_disambiguate"

    full_validation_rows = [
        {
            "run_name": row.get("run_name", ""),
            "family": row.get("family", ""),
            "seed": int(row.get("seed", -1)),
            "metrics": {k: _f(_metric_payload(row.get("best_checkpoint_metric", {})).get(k), 1e9) for k in METRIC_KEYS},
        }
        for row in completed
    ]
    semantic_hard_rows = [
        {
            "run_name": row.get("run_name", ""),
            "family": row.get("family", ""),
            "seed": int(row.get("seed", -1)),
            "semantic_hard_composite_score": _semantic_hard_score_from_row(row),
            "actual_gate_positive_ratio": float(row.get("actual_gate_positive_ratio_mean", 1.0)),
            "valuable_pair_ratio": float(row.get("valuable_pair_ratio_mean", 0.0)),
            "guaranteed_pair_count": float(row.get("guaranteed_pair_count_mean", 0.0)),
            "persistence_declared_but_inactive": bool(row.get("persistence_declared_but_inactive", False)),
        }
        for row in completed
    ]

    payload: Dict[str, Any] = {
        "generated_at_utc": now_iso(),
        "diagnosis_type": "stage2_semantic_objective_redesign_v7_repair",
        "teacher_as_mainline_semantic_source": False,
        "chosen_bootstrap_backend": "local_clip_vit_b32_mask_crop_visual_teacher",
        "v7_runs_terminal": bool(summary.get("all_runs_terminal", False)),
        "current_cropenc_baseline_anchor": {
            "source_report": str(args.stage2_semantic_value_diagnosis_report),
            "aggregate_mean_free_rollout_endpoint_l2": float(crop_ep),
            "aggregate_mean_legacysem_free_rollout_endpoint_l2": float(legacy_ep),
        },
        "v6_best_objective_combo_anchor": {
            "source_report": str(args.v6_reference_diagnosis_report),
            "best_v6_objective_combo": str(v6.get("success_criteria", {}).get("best_v6_objective_combo", "none")),
            "overall_best_run_name": str(v6.get("success_criteria", {}).get("overall_best_run_name", "none")),
            "semantic_hard_best_run_name": str(v6.get("success_criteria", {}).get("semantic_hard_best_run_name", "none")),
            "best_v6_full_validation_endpoint_l2": float(_f(v6.get("success_criteria", {}).get("best_v6_full_validation_endpoint_l2"), 1e9)),
            "best_v6_semantic_hard_composite_score": float(v6_best_hard_comp),
        },
        "v7_overall_best_anchor": _summary_row_anchor(overall_best_row) if overall_best_row else {"run_name": "none"},
        "v7_semantic_hard_best_anchor": _summary_row_anchor(semantic_hard_best_row) if semantic_hard_best_row else {"run_name": "none"},
        "v7_best_effective_persistence_anchor": _summary_row_anchor(effective_persist_row) if effective_persist_row else {"run_name": "none"},
        "warm_start_anchor": warm_start,
        "summary_source": str(args.v6_summary_report),
        "audit_source": str(args.v7_repair_audit_report),
        "full_validation_panel": {
            "runs": full_validation_rows,
            "aggregate": _aggregate_rows(full_validation_rows),
            "dataset_binding": ["VSPW", "VIPSeg"],
            "eval_scope": "best_checkpoint_metric_from_repaired_summary",
        },
        "semantic_hard_subset_panel": {
            "status": "omitted_in_v7_repair",
            "reason": "use semantic_hard_sidecar_metric emitted by completed v7 artifacts; no re-eval allowed in repair round",
        },
        "burst_persistence_hard_panel": {
            "status": "omitted_in_v7_repair",
            "reason": "repair round is artifact truth audit only",
        },
        "semantic_hard_composite_panel": {"runs": semantic_hard_rows},
        "family_disambiguation_panel": {
            "alignment_only_family": {
                "best_run_name": str(align_best_row.get("run_name", "none")) if align_best_row else "none",
                "best_full_validation_rank": list(align_best_rank) if align_best_row else [1e9, 1e9, 1e9],
                "best_semantic_hard_composite_score": float(align_best_hard),
            },
            "calibration_plus_active_persistence_family": {
                "best_run_name": str(persist_best_row.get("run_name", "none")) if persist_best_row else "none",
                "best_semantic_hard_composite_score": float(_semantic_hard_score_from_row(persist_best_row) if persist_best_row else 1e9),
                "best_effective_run_name": best_effective_persistence_run_name,
                "best_effective_semantic_hard_composite_score": float(effective_persist_hard),
            },
            "alignment_only_is_already_sufficient_rule": "true iff overall best resolves to alignonly and no effective persistence run beats the best alignment-only run on semantic-hard score without violating the full-validation non-regression guardrail",
            "persistence_branch_actually_contributed_rule": "true iff a declared persistence run is effective, exists as a non-none effective candidate, and improves semantic-hard score by >=2% over alignment-only while keeping full-validation endpoint within +2%",
        },
        "true_new_best_not_warm_start_inherited": bool(true_new_best_not_warm_start_inherited),
        "actual_gate_positive_ratio_below_0_30": bool(actual_gate_positive_ratio_below_0_30),
        "semantic_hard_composite_improved_vs_v6": bool(semantic_hard_composite_improved_vs_v6),
        "cross_seed_support_present": bool(cross_seed_support_present),
        "overall_best_run_name": overall_best_run_name,
        "semantic_hard_best_run_name": semantic_hard_best_run_name,
        "best_effective_persistence_run_name": best_effective_persistence_run_name,
        "alignment_only_is_already_sufficient": bool(alignment_only_is_already_sufficient),
        "persistence_branch_actually_contributed": bool(persistence_branch_actually_contributed),
        "persistence_declared_but_inactive_any": bool(persistence_declared_but_inactive_any),
        "persistence_declared_but_inactive_all": bool(persistence_declared_but_inactive_all),
        "next_step_choice": next_step_choice,
        "success_criteria": {
            "true_new_best_not_warm_start_inherited": bool(true_new_best_not_warm_start_inherited),
            "actual_gate_positive_ratio_below_0_30": bool(actual_gate_positive_ratio_below_0_30),
            "semantic_hard_composite_improved_vs_v6": bool(semantic_hard_composite_improved_vs_v6),
            "cross_seed_support_present": bool(cross_seed_support_present),
            "overall_best_run_name": overall_best_run_name,
            "semantic_hard_best_run_name": semantic_hard_best_run_name,
            "best_effective_persistence_run_name": best_effective_persistence_run_name,
            "alignment_only_is_already_sufficient": bool(alignment_only_is_already_sufficient),
            "persistence_branch_actually_contributed": bool(persistence_branch_actually_contributed),
            "persistence_declared_but_inactive_any": bool(persistence_declared_but_inactive_any),
            "persistence_declared_but_inactive_all": bool(persistence_declared_but_inactive_all),
            "next_step_choice": next_step_choice,
        },
        "notes": [
            "true_new_best_not_warm_start_inherited uses checkpoint lineage via best checkpoint global_step > warm-start anchor global_step for the same seed",
            "cross_seed_support_present requires at least two seeds whose per-seed family verdict supports the repaired alignment-only conclusion",
            "best_effective_persistence_run_name stays 'none' unless persistence telemetry is both declared and effective",
        ],
    }
    _write_json(args.v6_diagnosis_report, payload)
    _write_v7_results_md(args, audit, summary, payload)
    _repair_qualitative_v3_refs(args, payload)
    return payload


def run_all(args: Any) -> Dict[str, Any]:
    write_protocol_doc(args)
    write_decision_rule(args)
    activation_audit(args)
    launch(args)
    wait_for_completion(args)
    diag = diagnose_v7(args)
    return {"v7_diagnosis": diag}


def repair_reports(args: Any) -> Dict[str, Any]:
    audit = audit_v7_artifacts(args)
    summary = summarize(args)
    diagnosis = diagnose_v7(args)
    return {
        "repair_audit": audit,
        "summary": summary,
        "diagnosis": diagnosis,
    }


def parse_args() -> Any:
    p = ArgumentParser(description="Stage2 semantic objective redesign v7")
    p.add_argument("--mode", default="repair", choices=["all", "activation-audit", "launch", "run-one", "summarize", "diagnose", "audit", "repair"])
    p.add_argument("--meta-json", default="")
    p.add_argument("--work-root", default=str(WORK_ROOT))
    p.add_argument("--python-bin", default=_python_bin_default())
    p.add_argument("--tmux-session", default=SESSION)
    p.add_argument("--stage2-contract-json", default=str(WORK_ROOT / "reports/stage2_bootstrap_data_contract_20260408.json"))
    p.add_argument("--stage1-runtime-json", default=str(WORK_ROOT / "reports/stage1_v2_recommended_runtime_20260408.json"))
    p.add_argument("--stage1-best-ckpt", default=str(WORK_ROOT / "outputs/checkpoints/stage1_v2_longtrain_220m_mainline_20260408/best.pt"))
    p.add_argument("--shared-lease-path", default=str(WORK_ROOT / "reports/stage1_v2_gpu_lease_20260408.json"))
    p.add_argument("--bootstrap-cache-jsonl", default=str(WORK_ROOT / "data/processed/stage2_real_bootstrap_cache_20260410/clip_vit_b32_core_trainval_required_subset.jsonl"))
    p.add_argument("--semantic-hard-manifest-path", default=str(WORK_ROOT / "manifests/protocol_v2/stage2_semantic_hard_subsets_20260410.json"))
    p.add_argument("--stage2-semantic-value-diagnosis-report", default=str(WORK_ROOT / "reports/stage2_semantic_value_diagnosis_20260410.json"))
    p.add_argument("--v5-reference-summary-report", default=str(WORK_ROOT / "reports/stage2_semantic_objective_redesign_v5_summary_20260411.json"))
    p.add_argument("--v5-reference-diagnosis-report", default=str(WORK_ROOT / "reports/stage2_semantic_objective_redesign_v5_diagnosis_20260411.json"))
    p.add_argument("--v6-reference-summary-report", default=str(WORK_ROOT / "reports/stage2_semantic_objective_redesign_v6_summary_20260411.json"))
    p.add_argument("--v6-reference-diagnosis-report", default=str(WORK_ROOT / "reports/stage2_semantic_objective_redesign_v6_diagnosis_20260411.json"))
    p.add_argument("--v6-activation-audit-report", default=str(WORK_ROOT / "reports/stage2_semantic_objective_redesign_v7_activation_audit_20260413.json"))
    p.add_argument("--v6-protocol-doc", default=str(WORK_ROOT / "docs/STAGE2_SEMANTIC_OBJECTIVE_REDESIGN_V7_PROTOCOL_20260413.md"))
    p.add_argument("--v6-decision-rule-report", default=str(WORK_ROOT / "reports/stage2_semantic_objective_redesign_v7_decision_rule_20260413.json"))
    p.add_argument("--v6-launch-report", default=str(WORK_ROOT / "reports/stage2_semantic_objective_redesign_v7_launch_20260413.json"))
    p.add_argument("--v6-summary-report", default=str(WORK_ROOT / "reports/stage2_semantic_objective_redesign_v7_summary_20260413.json"))
    p.add_argument("--v6-results-md", default=str(WORK_ROOT / "docs/STAGE2_SEMANTIC_OBJECTIVE_REDESIGN_V7_RESULTS_20260413.md"))
    p.add_argument("--v6-diagnosis-report", default=str(WORK_ROOT / "reports/stage2_semantic_objective_redesign_v7_diagnosis_20260413.json"))
    p.add_argument("--v6-diagnosis-cache-report", default=str(WORK_ROOT / "reports/stage2_semantic_objective_redesign_v7_diagnosis_cache_20260413.json"))
    p.add_argument("--v7-repair-audit-report", default=str(WORK_ROOT / "reports/stage2_semantic_objective_redesign_v7_repair_audit_20260413.json"))
    p.add_argument("--v7-repair-audit-doc", default=str(WORK_ROOT / "docs/STAGE2_SEMANTIC_OBJECTIVE_REDESIGN_V7_REPAIR_AUDIT_20260413.md"))
    p.add_argument("--stage1-qualitative-pack-v3-report", default=str(WORK_ROOT / "reports/stage1_qualitative_pack_v3_20260413.json"))
    p.add_argument("--stage2-qualitative-pack-v3-report", default=str(WORK_ROOT / "reports/stage2_qualitative_pack_v3_20260413.json"))
    p.add_argument("--stage1-stage2-qualitative-pack-v3-doc", default=str(WORK_ROOT / "docs/STAGE1_STAGE2_QUALITATIVE_PACK_V3_20260413.md"))
    p.add_argument("--diagnose-use-final-metrics-only", action="store_true", default=True)
    p.add_argument("--reserve-idle-gpu-count", type=int, default=1)
    p.add_argument("--gpu-acquire-timeout-seconds", type=int, default=7200)
    p.add_argument("--gpu-acquire-retry-seconds", type=int, default=20)
    p.add_argument("--wait-timeout-seconds", type=int, default=21600)
    p.add_argument("--poll-seconds", type=int, default=60)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "all":
        print(json.dumps(run_all(args), ensure_ascii=True, indent=2))
    elif args.mode == "activation-audit":
        print(json.dumps(activation_audit(args), ensure_ascii=True, indent=2))
    elif args.mode == "audit":
        print(json.dumps(audit_v7_artifacts(args), ensure_ascii=True, indent=2))
    elif args.mode == "launch":
        print(json.dumps(launch(args), ensure_ascii=True, indent=2))
    elif args.mode == "run-one":
        run_one(args)
    elif args.mode == "summarize":
        print(json.dumps(summarize(args), ensure_ascii=True, indent=2))
    elif args.mode == "diagnose":
        print(json.dumps(diagnose_v7(args), ensure_ascii=True, indent=2))
    elif args.mode == "repair":
        print(json.dumps(repair_reports(args), ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
