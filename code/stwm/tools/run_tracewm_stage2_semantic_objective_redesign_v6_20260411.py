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


WORK_ROOT = Path("/home/chen034/workspace/stwm")
SESSION = "tracewm_stage2_semantic_objective_redesign_v6_20260411"
LOG_PATH = WORK_ROOT / "logs/tracewm_stage2_semantic_objective_redesign_v6_20260411.log"
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
            "run_name": "stage2_semobjv6_topk1_g2_fb1_seed42_20260411",
            "seed": 42,
            "objective_combo": "hard_topk_query_gating_v6_k1+guaranteed_sparse_persistence_g2+two_level_pair_mining+conservative_delayed_aux_schedule",
            "semantic_rescue_mode": "v6sparse",
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
            "window_name": "semobjv6_t1g2s42",
        },
        {
            "run_name": "stage2_semobjv6_topk2_g2_fb1_seed42_20260411",
            "seed": 42,
            "objective_combo": "hard_topk_query_gating_v6_k2+guaranteed_sparse_persistence_g2+two_level_pair_mining+conservative_delayed_aux_schedule",
            "semantic_rescue_mode": "v6sparse",
            "semantic_rescue_weight": 0.00015,
            "confidence_gated_alignment_loss_weight": 1.0,
            "sparse_persistence_contrastive_loss_weight": 0.05,
            "confidence_gating_margin_threshold": 0.10,
            "confidence_gating_temperature": 0.05,
            "semantic_hard_score_threshold": 0.25,
            "aux_loss_delay_steps": 180,
            "aux_loss_ramp_steps": 360,
            "v6_gating_family": "hard_topk_query_gating_v2",
            "v6_topk_query_k": 2,
            "v6_capped_quantile": 0.85,
            "v6_max_affected_ratio": 0.25,
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
            "window_name": "semobjv6_t2g2s42",
        },
        {
            "run_name": "stage2_semobjv6_qcap15_g2_fb1_seed42_20260411",
            "seed": 42,
            "objective_combo": "capped_quantile_sparse_gating_v6_cap15+guaranteed_sparse_persistence_g2+two_level_pair_mining+conservative_delayed_aux_schedule",
            "semantic_rescue_mode": "v6sparse",
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
            "window_name": "semobjv6_q15g2s42",
        },
        {
            "run_name": "stage2_semobjv6_qcap25_g2_fb1_seed42_20260411",
            "seed": 42,
            "objective_combo": "capped_quantile_sparse_gating_v6_cap25+guaranteed_sparse_persistence_g2+two_level_pair_mining+conservative_delayed_aux_schedule",
            "semantic_rescue_mode": "v6sparse",
            "semantic_rescue_weight": 0.00015,
            "confidence_gated_alignment_loss_weight": 1.0,
            "sparse_persistence_contrastive_loss_weight": 0.05,
            "confidence_gating_margin_threshold": 0.10,
            "confidence_gating_temperature": 0.05,
            "semantic_hard_score_threshold": 0.25,
            "aux_loss_delay_steps": 180,
            "aux_loss_ramp_steps": 360,
            "v6_gating_family": "capped_quantile_sparse_gating_v2",
            "v6_topk_query_k": 2,
            "v6_capped_quantile": 0.70,
            "v6_max_affected_ratio": 0.25,
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
            "window_name": "semobjv6_q25g2s42",
        },
        {
            "run_name": "stage2_semobjv6_topk1_g2_fb1_seed123_20260411",
            "seed": 123,
            "objective_combo": "hard_topk_query_gating_v6_k1+guaranteed_sparse_persistence_g2+two_level_pair_mining+conservative_delayed_aux_schedule",
            "semantic_rescue_mode": "v6sparse",
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
            "window_name": "semobjv6_t1g2s123",
        },
        {
            "run_name": "stage2_semobjv6_qcap15_g2_fb0_seed123_20260411",
            "seed": 123,
            "objective_combo": "capped_quantile_sparse_gating_v6_cap15+guaranteed_sparse_persistence_g2+single_level_pair_mining+conservative_delayed_aux_schedule",
            "semantic_rescue_mode": "v6sparse",
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
            "v6_two_level_pair_mining_enabled": False,
            "v6_relaxed_motion_threshold": 0.08,
            "v6_relaxed_area_jump_threshold": 0.06,
            "v6_relaxed_small_query_threshold": 0.20,
            "v6_relaxed_appearance_shift_threshold": 0.25,
            "v6_relaxed_center_interaction_threshold": 0.10,
            "window_name": "semobjv6_q15g2f0s123",
        },
    ]


def _semantic_hard_composite(values: List[float]) -> float:
    if not values:
        return 1e9
    return float(sum(values) / max(len(values), 1))


def write_protocol_doc(args: Any) -> None:
    _write_md(
        args.v6_protocol_doc,
        [
            "# Stage2 Semantic Objective Redesign V6 Protocol",
            "",
            f"- generated_at_utc: {now_iso()}",
            "- stage1_mutation_allowed: false",
            "- main_task: future trace / future state generation",
            "- teacher_as_mainline_semantic_source: false",
            "- v5_failure_summary: sparse gating activated, but persistence mining was frequently inactive and sidecar gains did not transfer to overall best.",
            "- v6_core_principles: keep sparse query gating active; enforce guaranteed persistence pairs; add two-level fallback mining; keep delayed auxiliary schedule; sidecar remains independent.",
            "- activation_audit_required_before_launch: true",
            "- persistence_mining_repair_focus: guaranteed_min_pairs + strict/fallback pair ratio telemetry",
            "- selective_supervision_position: readout-side only, never overwriting frozen trace dynamics.",
            "- forbidden: teacher semantic token replacement; external-eval work; Stage1 rollback; codec/VAE upgrade; full-scale long train; batch/lr sweep; DDP retrofit.",
        ],
    )


def write_decision_rule(args: Any) -> Dict[str, Any]:
    payload = {
        "generated_at_utc": now_iso(),
        "rule_scope": "stage2_semantic_objective_redesign_v6",
        "strict_success_conditions": {
            "true_new_best_not_warm_start_inherited": True,
            "actual_gate_positive_ratio_below_0_30": True,
            "guaranteed_pair_count_mean_above_or_equal_1_0": True,
            "fallback_trigger_rate_reasonable_below_0_70": True,
            "semantic_hard_composite_improved_vs_prev": True,
            "full_validation_non_catastrophic": True,
        },
        "continue_rescue_conditions": {
            "description": "do not freeze; continue v6 rescue if there is signal but strict success conditions are not fully met",
            "requires": [
                "full_validation_non_catastrophic",
                "at least one of: improved_vs_current_cropenc_baseline, sidecar_truly_diverged, semantic_hard_composite_improved_vs_prev",
            ],
        },
        "fail_hard_conditions": {
            "description": "objective family still not working under current Stage2 design",
            "trigger_any": [
                "full_validation_non_catastrophic = false",
                "true_new_best_not_warm_start_inherited = false and semantic_hard_composite_improved_vs_prev = false and guaranteed_pair_count_mean_above_or_equal_1_0 = false",
            ],
        },
        "restricted_next_step_enum": [
            "stage2_semantic_rescue_fullscale_wave1",
            "redesign_stage2_semantic_objective_v7",
            "objective_family_still_not_working_under_current_stage2_design",
        ],
        "prohibited_if_strict_success_missing": [
            "freeze_stage2_core_mainline",
        ],
        "notes": [
            "This rule gates any later freeze recommendation; the v6 runner outputs only rescue-oriented next steps.",
            "A true v6 success should advance to stage2_semantic_rescue_fullscale_wave1 rather than direct freeze.",
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
        acquire_deadline = time.time() + float(args.gpu_acquire_timeout_seconds)
        gpu: Dict[str, Any] | None = None
        last_gpu_error = ""
        while gpu is None:
            try:
                gpu = _select_gpu(run_name=run_name, lease_path=str(args.shared_lease_path))
            except Exception as exc:
                last_gpu_error = str(exc)
                if time.time() >= acquire_deadline:
                    raise RuntimeError(f"gpu_acquire_timeout run={run_name} last_error={last_gpu_error}")
                time.sleep(float(args.gpu_acquire_retry_seconds))

        out_dir = Path(args.work_root) / "outputs" / "checkpoints" / run_name
        meta = {
            **spec,
            "selected_gpu_id": int(gpu["selected_gpu_id"]),
            "lease_id": str(gpu["lease_id"]),
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
        }
        meta_json = Path(args.work_root) / "reports" / "stage2_semantic_objective_redesign_v6_runs_20260411" / f"{run_name}_launch_meta.json"
        meta["meta_json"] = str(meta_json)
        _write_json(meta_json, meta)
        runs.append(meta)

        env = {
            "PYTHONPATH": f"{args.work_root}/code:{os.environ.get('PYTHONPATH', '')}",
            "CUDA_VISIBLE_DEVICES": str(meta["selected_gpu_id"]),
            "TRACEWM_STAGE1_V2_GPU_SELECTION_METADATA_JSON": json.dumps(
                {"selected_gpu_id": int(meta["selected_gpu_id"]), "lease_id": str(meta["lease_id"]), "owner": run_name, "mode": "single_gpu_only"},
                ensure_ascii=True,
            ),
        }
        env_prefix = " ".join(f"{k}={shlex.quote(v)}" for k, v in env.items())
        cmd = (
            f"{env_prefix} {shlex.quote(str(args.python_bin))} "
            f"{shlex.quote(str(Path(args.work_root) / 'code/stwm/tools/run_tracewm_stage2_semantic_objective_redesign_v6_20260411.py'))} "
            f"--mode run-one --meta-json {shlex.quote(str(meta_json))}"
        )
        subprocess.run(["tmux", "new-window", "-t", str(args.tmux_session), "-n", str(meta["window_name"]), cmd], check=True)

    payload = {
        "generated_at_utc": now_iso(),
        "mode": "stage2_semantic_objective_redesign_v6_launch",
        "tmux_session": str(args.tmux_session),
        "pilot_policy": "bounded pilot; no Stage1 mutation; teacher remains target only; no codec/VAE upgrade",
        "runs": runs,
    }
    _write_json(args.v6_launch_report, payload)
    return summarize(args)


def run_one(args: Any) -> None:
    meta = _read_json(args.meta_json)
    lease_id = str(meta.get("lease_id", ""))
    lease_path = str(meta.get("shared_lease_path", ""))
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
        with log_path.open("w", encoding="utf-8") as log_fh:
            proc = subprocess.run(
                cmd,
                cwd=str(meta["work_root"]),
                text=True,
                stdout=log_fh,
                stderr=subprocess.STDOUT,
                env=os.environ.copy(),
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
                "selected_gpu_id": int(meta["selected_gpu_id"]),
                "lease_id": str(meta["lease_id"]),
                "objective_combo": str(meta["objective_combo"]),
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


def summarize(args: Any) -> Dict[str, Any]:
    launch = _read_json(args.v6_launch_report)
    rows: List[Dict[str, Any]] = []
    running = completed = failed = 0
    for meta in launch.get("runs", []) if isinstance(launch.get("runs", []), list) else []:
        status_info = _status_for(meta, session_name=str(args.tmux_session))
        status = str(status_info.get("status", "launched"))
        running += int(status == "running")
        completed += int(status == "completed")
        failed += int(status == "failed")
        detail = status_info.get("detail", {}) if isinstance(status_info.get("detail", {}), dict) else {}
        branch = detail.get("semantic_branch_metrics", {}) if isinstance(detail.get("semantic_branch_metrics", {}), dict) else {}
        sidecar = detail.get("semantic_hard_sidecar_metric", {}) if isinstance(detail.get("semantic_hard_sidecar_metric", {}), dict) else {}
        sidecar_sel = detail.get("sidecar_checkpoint_selection", {}) if isinstance(detail.get("sidecar_checkpoint_selection", {}), dict) else {}
        rows.append(
            {
                "run_name": str(meta.get("run_name", "")),
                "objective_combo": str(meta.get("objective_combo", "")),
                "seed": int(meta.get("seed", -1)),
                "selected_gpu_id": int(meta.get("selected_gpu_id", -1)),
                "lease_id": str(meta.get("lease_id", "")),
                "batch_size": int(meta.get("batch_size", 0)),
                "train_steps": int(meta.get("train_steps", 0)),
                "additional_train_steps": int(meta.get("additional_train_steps", 0)),
                "eval_interval": int(meta.get("eval_interval", 0)),
                "save_every_n_steps": int(meta.get("save_every_n_steps", 0)),
                "effective_train_sample_count_per_dataset": meta.get("effective_train_sample_count_per_dataset", {}),
                "effective_val_sample_count_per_dataset": meta.get("effective_val_sample_count_per_dataset", {}),
                "status": status,
                "semantic_rescue_mode": str(meta.get("semantic_rescue_mode", "")),
                "v6_gating_family": str(meta.get("v6_gating_family", "")),
                "v6_guaranteed_min_pairs_per_sample": int(meta.get("v6_guaranteed_min_pairs_per_sample", 0)),
                "v6_two_level_pair_mining_enabled": bool(meta.get("v6_two_level_pair_mining_enabled", True)),
                "actual_gate_positive_ratio_mean": float(branch.get("actual_gate_positive_ratio_mean", 0.0)),
                "activated_query_count_mean": float(branch.get("activated_query_count_mean", 0.0)),
                "activated_query_ratio_mean": float(branch.get("activated_query_ratio_mean", 0.0)),
                "raw_quantile_ratio_mean": float(branch.get("raw_quantile_ratio_mean", 0.0)),
                "capped_ratio_mean": float(branch.get("capped_ratio_mean", 0.0)),
                "valuable_pair_ratio_mean": float(branch.get("valuable_pair_ratio_mean", 0.0)),
                "fallback_trigger_rate_mean": float(branch.get("fallback_trigger_rate_mean", 0.0)),
                "guaranteed_pair_count_mean": float(branch.get("guaranteed_pair_count_mean", 0.0)),
                "strict_pair_ratio_mean": float(branch.get("strict_pair_ratio_mean", 0.0)),
                "fallback_pair_ratio_mean": float(branch.get("fallback_pair_ratio_mean", 0.0)),
                "final_effective_aux_weight": float(branch.get("final_effective_aux_weight", 0.0)),
                "semantic_hard_sidecar_score": float(sidecar.get("semantic_hard_sidecar_score", 1e9)),
                "same_checkpoint_selected": bool(sidecar_sel.get("same_checkpoint_selected", False)),
                "sidecar_truly_diverged": bool(sidecar_sel.get("sidecar_truly_diverged", False)),
                "final_json": str(meta.get("final_json", "")),
                "best_checkpoint": str(Path(str(meta.get("output_dir", ""))) / "best.pt"),
                "latest_checkpoint": str(Path(str(meta.get("output_dir", ""))) / "latest.pt"),
                "best_semantic_hard_checkpoint": str(Path(str(meta.get("output_dir", ""))) / "best_semantic_hard.pt"),
                "best_checkpoint_metric": detail.get("best_checkpoint_metric", {}) if isinstance(detail.get("best_checkpoint_metric", {}), dict) else {},
                "latest_checkpoint_metric": detail.get("latest_checkpoint_metric", {}) if isinstance(detail.get("latest_checkpoint_metric", {}), dict) else {},
                "semantic_hard_sidecar_metric": sidecar,
            }
        )
    payload = {
        "generated_at_utc": now_iso(),
        "redesign_v6_status": f"{running}_running_{completed}_completed_{failed}_failed",
        "runs": rows,
        "next_step_choice_internal": "summarize_redesign_v6_after_completion" if completed == len(rows) and rows and failed == 0 else ("fix_failed_v6_runs" if failed else "continue_redesign_v6"),
    }
    _write_json(args.v6_summary_report, payload)
    lines = [
        "# Stage2 Semantic Objective Redesign V6 Results",
        "",
        f"- generated_at_utc: {payload['generated_at_utc']}",
        f"- redesign_v6_status: {payload['redesign_v6_status']}",
        "",
        "| run_name | combo | gate_family | gpu | batch | steps | status | best_endpoint_l2 | gate_ratio | guaranteed_pair_count | strict_pair_ratio | fallback_trigger_rate | fallback_pair_ratio | same_ckpt | sidecar_diverged |",
        "|---|---|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in rows:
        best = row.get("best_checkpoint_metric", {}).get("metrics", {}) if isinstance(row.get("best_checkpoint_metric", {}), dict) else {}
        lines.append(
            f"| {row['run_name']} | {row['objective_combo']} | {row['v6_gating_family']} | {row['selected_gpu_id']} | {row['batch_size']} | {row['train_steps']} | {row['status']} | {_f(best.get('free_rollout_endpoint_l2'), 1e9):.8f} | {float(row.get('actual_gate_positive_ratio_mean', 0.0)):.4f} | {float(row.get('guaranteed_pair_count_mean', 0.0)):.4f} | {float(row.get('strict_pair_ratio_mean', 0.0)):.4f} | {float(row.get('fallback_trigger_rate_mean', 0.0)):.4f} | {float(row.get('fallback_pair_ratio_mean', 0.0)):.4f} | {bool(row.get('same_checkpoint_selected', False))} | {bool(row.get('sidecar_truly_diverged', False))} |"
        )
    _write_md(args.v6_results_md, lines)
    return payload


def wait_for_completion(args: Any) -> Dict[str, Any]:
    deadline = time.time() + float(args.wait_timeout_seconds)
    last = summarize(args)
    while time.time() < deadline:
        if str(last.get("redesign_v6_status", "")).startswith("0_running_"):
            return last
        time.sleep(float(args.poll_seconds))
        last = summarize(args)
    last["timed_out_waiting_for_completion"] = True
    _write_json(args.v6_summary_report, last)
    return last


def diagnose_v6(args: Any) -> Dict[str, Any]:
    summary = _read_json(args.v6_summary_report)
    rows = [r for r in summary.get("runs", []) if isinstance(r, dict)]
    completed = [r for r in rows if str(r.get("status", "")) == "completed"]

    refs = _baseline_refs()
    v2 = _v2_refs()
    v3 = _v3_refs()
    v4 = _v4_refs()
    v4_summary = _v4_summary()
    warm_start = _warm_start_anchor_refs()
    activation_payload = _read_json(args.v6_activation_audit_report) if Path(args.v6_activation_audit_report).exists() else {}

    crop_ep = _f(refs["cropenc_fullscale_mean"].get("free_rollout_endpoint_l2", {}).get("mean"), 1e9)
    legacy_ep = _f(refs["legacysem_fullscale_mean"].get("free_rollout_endpoint_l2", {}).get("mean"), 1e9)
    v2_best_full_ep = _f(v2.get("success_criteria", {}).get("best_v2_full_validation_endpoint_l2"), 1e9)
    v2_best_hard_comp = _v2_best_hard_composite(v2)
    v3_best_full_ep = _f(v3.get("success_criteria", {}).get("best_v3_full_validation_endpoint_l2"), 1e9)
    v3_best_hard_comp = _f(v3.get("success_criteria", {}).get("best_v3_semantic_hard_composite_score"), 1e9)
    v4_best_full_ep = _f(v4.get("success_criteria", {}).get("best_v4_full_validation_endpoint_l2"), 1e9)
    v4_best_hard_comp = _f(v4.get("success_criteria", {}).get("best_v4_semantic_hard_composite_score"), 1e9)
    v4_best_gate_ratio = _v4_best_gate_ratio(v4, v4_summary)

    payload: Dict[str, Any] = {
        "generated_at_utc": now_iso(),
        "diagnosis_type": "stage2_semantic_objective_redesign_v6",
        "teacher_as_mainline_semantic_source": False,
        "chosen_bootstrap_backend": "local_clip_vit_b32_mask_crop_visual_teacher",
        "v6_runs_terminal": bool(rows and len(completed) == len(rows)),
        "activation_audit": activation_payload,
        "baseline_refs": {
            "current_cropenc_fullscale_mean_endpoint_l2": float(crop_ep),
            "legacysem_fullscale_mean_endpoint_l2": float(legacy_ep),
            "v2_best_objective_combo_endpoint_l2": float(v2_best_full_ep),
            "v2_best_semantic_hard_composite_score": float(v2_best_hard_comp),
            "v3_best_objective_combo_endpoint_l2": float(v3_best_full_ep),
            "v3_best_semantic_hard_composite_score": float(v3_best_hard_comp),
            "v4_best_objective_combo_endpoint_l2": float(v4_best_full_ep),
            "v4_best_semantic_hard_composite_score": float(v4_best_hard_comp),
            "v4_best_actual_gate_positive_ratio": float(v4_best_gate_ratio),
            "warm_start_anchor": warm_start,
        },
        "full_validation_panel": {},
        "semantic_hard_subset_panel": {},
        "burst_persistence_hard_panel": {},
        "semantic_hard_composite_panel": {},
        "success_criteria": {},
    }

    if not payload["v6_runs_terminal"]:
        payload["success_criteria"] = {
            "true_new_best_not_warm_start_inherited": False,
            "actual_gate_positive_ratio_below_0_30": False,
            "guaranteed_pair_count_mean_above_or_equal_1_0": False,
            "fallback_trigger_rate_reasonable_below_0_70": False,
            "semantic_hard_composite_improved_vs_v4": False,
            "cross_seed_support_present": False,
            "full_validation_non_catastrophic": False,
            "improved_vs_current_cropenc_baseline": False,
            "improved_vs_v4_best_objective_combo": False,
            "best_v6_objective_combo": "none",
            "best_v6_full_validation_endpoint_l2": 1e9,
            "best_v6_semantic_hard_composite_score": 1e9,
            "sidecar_truly_diverged": False,
            "restricted_next_step_enum": [
                "stage2_semantic_rescue_fullscale_wave1",
                "redesign_stage2_semantic_objective_v7",
                "objective_family_still_not_working_under_current_stage2_design",
            ],
            "next_step_choice": "objective_family_still_not_working_under_current_stage2_design",
        }
        _write_json(args.v6_diagnosis_report, payload)
        return payload

    eval_gpu = _select_eval_gpu(args)
    if int(eval_gpu.get("selected_gpu_id", -1)) >= 0 and not os.environ.get("CUDA_VISIBLE_DEVICES", "").strip():
        os.environ["CUDA_VISIBLE_DEVICES"] = str(eval_gpu["selected_gpu_id"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_aux: Dict[str, Dict[str, Any]] = {}
    try:
        print(f"[semobjv6] diagnosis device={device} eval_gpu={eval_gpu}", flush=True)
        stage1_model, _ = _load_stage1_model(device=device, stage1_ckpt=str(args.stage1_best_ckpt))
        full_ds = _make_dataset(["vspw", "vipseg"], "val", str(args.stage2_contract_json), max_samples=-1)
        core_ds = _make_dataset(["vspw", "vipseg"], "val", str(args.stage2_contract_json), max_samples=-1)
        subset_manifest = _read_json(args.semantic_hard_manifest_path)
        hard_subset_names = [
            "occlusion_reappearance",
            "crossing_or_interaction_ambiguity",
            "small_object_or_low_area",
            "appearance_change_or_semantic_shift",
        ]
        hard_subset_indices: Dict[str, List[int]] = {}
        hard_subset_eval_ds: Dict[str, Any] = {}
        hard_subset_rows: Dict[str, List[Dict[str, Any]]] = {}
        for subset_name in hard_subset_names:
            subset = subset_manifest.get("subsets", {}).get(subset_name, {})
            indices = [int(x["dataset_index"]) for x in subset.get("items", []) if isinstance(x, dict) and "dataset_index" in x]
            hard_subset_indices[subset_name] = indices
            hard_subset_eval_ds[subset_name] = Subset(core_ds, indices)
            hard_subset_rows[subset_name] = []

        print("[semobjv6] optional BURST persistence-hard eval", flush=True)
        burst_eval_ds = None
        burst_indices: List[int] = []
        burst_rows: List[Dict[str, Any]] = []
        burst_skip_reason = ""
        try:
            burst_ds = _make_dataset(["burst"], "val", str(args.stage2_contract_json), max_samples=-1)
            burst_subset = subset_manifest.get("subsets", {}).get("burst_persistence_stress", {})
            burst_indices = [int(x["dataset_index"]) for x in burst_subset.get("items", []) if isinstance(x, dict) and "dataset_index" in x]
            burst_eval_ds = Subset(burst_ds, burst_indices)
        except Exception as exc:
            burst_skip_reason = str(exc)

        cache_path = Path(str(args.v6_diagnosis_cache_report))
        cached_runs: Dict[str, Dict[str, Any]] = {}
        if cache_path.exists():
            try:
                existing_cache = _read_json(cache_path)
                cached_runs = existing_cache.get("runs", {}) if isinstance(existing_cache.get("runs", {}), dict) else {}
            except Exception:
                cached_runs = {}

        def _entry_complete(entry: Dict[str, Any]) -> bool:
            if not isinstance(entry, dict):
                return False
            full_metrics = entry.get("full_metrics", {})
            hard_metrics = entry.get("hard_subset_metrics", {})
            if not isinstance(full_metrics, dict) or not isinstance(hard_metrics, dict):
                return False
            if any(k not in full_metrics for k in METRIC_KEYS):
                return False
            for subset_name in hard_subset_names:
                subset_metrics = hard_metrics.get(subset_name, {})
                if not isinstance(subset_metrics, dict):
                    return False
                if any(k not in subset_metrics for k in METRIC_KEYS):
                    return False
            if burst_eval_ds is not None:
                burst_metrics = entry.get("burst_metrics", {})
                if not isinstance(burst_metrics, dict):
                    return False
                if any(k not in burst_metrics for k in METRIC_KEYS):
                    return False
            return True

        def _append_run_rows(run_name: str, row: Dict[str, Any], full_metrics: Dict[str, Any], hard_metrics: Dict[str, Dict[str, Any]], burst_metrics: Dict[str, Any]) -> None:
            full_rows.append(
                {
                    "run_name": run_name,
                    "objective_combo": str(row.get("objective_combo", "")),
                    "seed": int(row.get("seed", -1)),
                    "metrics": {k: _f(full_metrics.get(k), 1e9) for k in METRIC_KEYS},
                }
            )
            for subset_name in hard_subset_names:
                sm = hard_metrics.get(subset_name, {}) if isinstance(hard_metrics.get(subset_name, {}), dict) else {}
                hard_subset_rows[subset_name].append(
                    {
                        "run_name": run_name,
                        "objective_combo": str(row.get("objective_combo", "")),
                        "seed": int(row.get("seed", -1)),
                        "metrics": {k: _f(sm.get(k), 1e9) for k in METRIC_KEYS},
                    }
                )
                run_to_hard_values[run_name].append(_f(sm.get("free_rollout_endpoint_l2"), 1e9))
            if burst_eval_ds is not None and isinstance(burst_metrics, dict):
                burst_rows.append(
                    {
                        "run_name": run_name,
                        "objective_combo": str(row.get("objective_combo", "")),
                        "seed": int(row.get("seed", -1)),
                        "metrics": {k: _f(burst_metrics.get(k), 1e9) for k in METRIC_KEYS},
                    }
                )

        full_rows: List[Dict[str, Any]] = []
        run_to_hard_values: Dict[str, List[float]] = {str(row["run_name"]): [] for row in completed}
        cache_hits = 0
        cache_misses = 0
        for row in completed:
            run_name = str(row["run_name"])
            final = _read_json(str(row["final_json"]))
            branch = final.get("semantic_branch_metrics", {}) if isinstance(final.get("semantic_branch_metrics", {}), dict) else {}
            sidecar_sel = final.get("sidecar_checkpoint_selection", {}) if isinstance(final.get("sidecar_checkpoint_selection", {}), dict) else {}
            run_aux[run_name] = {
                "actual_gate_positive_ratio": float(branch.get("actual_gate_positive_ratio_mean", 1.0)),
                "valuable_pair_ratio": float(branch.get("valuable_pair_ratio_mean", 0.0)),
                "fallback_trigger_rate": float(branch.get("fallback_trigger_rate_mean", 1.0)),
                "guaranteed_pair_count": float(branch.get("guaranteed_pair_count_mean", 0.0)),
                "strict_pair_ratio": float(branch.get("strict_pair_ratio_mean", 0.0)),
                "fallback_pair_ratio": float(branch.get("fallback_pair_ratio_mean", 0.0)),
                "resume_global_step": int(final.get("resume_global_step", row.get("train_steps", 0) - PILOT_EXTRA_STEPS)),
                "overall_best_step": int(final.get("best_checkpoint_metric", {}).get("global_step", -1)),
                "same_checkpoint_selected": bool(sidecar_sel.get("same_checkpoint_selected", False)),
                "sidecar_truly_diverged": bool(sidecar_sel.get("sidecar_truly_diverged", False)),
            }

            cache_entry = cached_runs.get(run_name, {}) if isinstance(cached_runs.get(run_name, {}), dict) else {}
            if _entry_complete(cache_entry):
                cache_hits += 1
                _append_run_rows(
                    run_name=run_name,
                    row=row,
                    full_metrics=cache_entry.get("full_metrics", {}),
                    hard_metrics=cache_entry.get("hard_subset_metrics", {}),
                    burst_metrics=cache_entry.get("burst_metrics", {}),
                )
                continue

            cache_misses += 1
            print(f"[semobjv6] full validation eval run={run_name}", flush=True)
            loaded_run = _load_stage2_modules(str(row["best_checkpoint"]), device, stage1_model)
            try:
                full_metrics = _evaluate_loaded_stage2(loaded=loaded_run, stage1_model=stage1_model, dataset=full_ds, device=device)
                full_metrics_payload = {
                    k: _f(full_metrics.get(k), 1e9)
                    for k in METRIC_KEYS
                }
                hard_metrics_payload: Dict[str, Dict[str, Any]] = {}
                for subset_name in hard_subset_names:
                    print(f"[semobjv6] semantic hard eval subset={subset_name} run={run_name}", flush=True)
                    subset_metrics = _evaluate_loaded_stage2(
                        loaded=loaded_run,
                        stage1_model=stage1_model,
                        dataset=hard_subset_eval_ds[subset_name],
                        device=device,
                    )
                    hard_metrics_payload[subset_name] = {k: _f(subset_metrics.get(k), 1e9) for k in METRIC_KEYS}
                burst_metrics_payload: Dict[str, Any] = {}
                if burst_eval_ds is not None:
                    burst_metrics = _evaluate_loaded_stage2(loaded=loaded_run, stage1_model=stage1_model, dataset=burst_eval_ds, device=device)
                    burst_metrics_payload = {k: _f(burst_metrics.get(k), 1e9) for k in METRIC_KEYS}
            finally:
                del loaded_run
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

            cached_runs[run_name] = {
                "cached_at_utc": now_iso(),
                "full_metrics": full_metrics_payload,
                "hard_subset_metrics": hard_metrics_payload,
                "burst_metrics": burst_metrics_payload,
            }
            _write_json(
                cache_path,
                {
                    "generated_at_utc": now_iso(),
                    "cache_type": "stage2_semantic_objective_redesign_v6_diagnosis_cache",
                    "run_count": int(len(cached_runs)),
                    "runs": cached_runs,
                },
            )
            _append_run_rows(
                run_name=run_name,
                row=row,
                full_metrics=full_metrics_payload,
                hard_metrics=hard_metrics_payload,
                burst_metrics=burst_metrics_payload,
            )

        payload["evaluation_cache"] = {
            "path": str(cache_path),
            "cache_hits": int(cache_hits),
            "cache_misses": int(cache_misses),
            "cached_run_count": int(len(cached_runs)),
        }

        payload["full_validation_panel"] = {
            "runs": full_rows,
            "aggregate": _aggregate_rows(full_rows),
            "dataset_binding": ["VSPW", "VIPSeg"],
            "eval_scope": "full_validation",
        }

        hard_panels: Dict[str, Any] = {}
        for subset_name in hard_subset_names:
            subset_rows = hard_subset_rows[subset_name]
            indices = hard_subset_indices[subset_name]
            hard_panels[subset_name] = {
                "clip_count": int(len(indices)),
                "runs": subset_rows,
                "aggregate": _aggregate_rows(subset_rows),
                "cropenc_baseline_aggregate": refs.get("hard_subset_panels", {}).get(subset_name, {}).get("families", {}).get("cropenc", {}).get("aggregate", {}),
                "legacysem_baseline_aggregate": refs.get("hard_subset_panels", {}).get(subset_name, {}).get("families", {}).get("legacysem", {}).get("aggregate", {}),
                "v2_best_reference": v2.get("semantic_hard_subset_panel", {}).get(subset_name, {}),
                "v3_best_reference": v3.get("semantic_hard_subset_panel", {}).get(subset_name, {}),
                "v4_best_reference": v4.get("semantic_hard_subset_panel", {}).get(subset_name, {}),
            }
        payload["semantic_hard_subset_panel"] = hard_panels
        payload["semantic_hard_composite_panel"] = {
            "runs": [
                {
                    "run_name": run_name,
                    "objective_combo": next((str(x.get("objective_combo", "")) for x in completed if str(x.get("run_name", "")) == run_name), ""),
                    "seed": next((int(x.get("seed", -1)) for x in completed if str(x.get("run_name", "")) == run_name), -1),
                    "semantic_hard_composite_score": float(_semantic_hard_composite(vals)),
                    "actual_gate_positive_ratio": float(run_aux.get(run_name, {}).get("actual_gate_positive_ratio", 1.0)),
                    "valuable_pair_ratio": float(run_aux.get(run_name, {}).get("valuable_pair_ratio", 0.0)),
                    "fallback_trigger_rate": float(run_aux.get(run_name, {}).get("fallback_trigger_rate", 1.0)),
                    "guaranteed_pair_count": float(run_aux.get(run_name, {}).get("guaranteed_pair_count", 0.0)),
                    "strict_pair_ratio": float(run_aux.get(run_name, {}).get("strict_pair_ratio", 0.0)),
                    "fallback_pair_ratio": float(run_aux.get(run_name, {}).get("fallback_pair_ratio", 0.0)),
                    "same_checkpoint_selected": bool(run_aux.get(run_name, {}).get("same_checkpoint_selected", False)),
                    "sidecar_truly_diverged": bool(run_aux.get(run_name, {}).get("sidecar_truly_diverged", False)),
                }
                for run_name, vals in run_to_hard_values.items()
            ]
        }

        if burst_eval_ds is not None:
            payload["burst_persistence_hard_panel"] = {
                "status": "evaluated_optional_stress_panel",
                "clip_count": int(len(burst_indices)),
                "runs": burst_rows,
                "aggregate": _aggregate_rows(burst_rows),
            }
        else:
            payload["burst_persistence_hard_panel"] = {"status": "skipped", "reason": str(burst_skip_reason)}
    finally:
        _release_lease_safe(str(eval_gpu.get("lease_id", "")), str(args.shared_lease_path))

    run_to_full_ep = {str(row["run_name"]): _f(row.get("metrics", {}).get("free_rollout_endpoint_l2"), 1e9) for row in payload["full_validation_panel"]["runs"]}
    run_to_hard_comp = {str(row["run_name"]): float(row.get("semantic_hard_composite_score", 1e9)) for row in payload["semantic_hard_composite_panel"]["runs"]}

    best_full_run = min(run_to_full_ep, key=run_to_full_ep.get)
    best_hard_run = min(run_to_hard_comp, key=run_to_hard_comp.get)
    best_full_endpoint = float(run_to_full_ep[best_full_run])
    best_hard_composite = float(run_to_hard_comp[best_hard_run])
    best_combo = next((str(x.get("objective_combo", "")) for x in completed if str(x.get("run_name", "")) == best_hard_run), "none")

    true_new_best = any(
        int(run_aux.get(str(row["run_name"]), {}).get("overall_best_step", -1))
        > int(run_aux.get(str(row["run_name"]), {}).get("resume_global_step", 1e9))
        for row in completed
    )
    best_gate_ratio = float(run_aux.get(best_hard_run, {}).get("actual_gate_positive_ratio", 1.0))
    best_guaranteed_pairs = float(run_aux.get(best_hard_run, {}).get("guaranteed_pair_count", 0.0))
    best_fallback_trigger = float(run_aux.get(best_hard_run, {}).get("fallback_trigger_rate", 1.0))
    best_strict_ratio = float(run_aux.get(best_hard_run, {}).get("strict_pair_ratio", 0.0))
    best_fallback_ratio = float(run_aux.get(best_hard_run, {}).get("fallback_pair_ratio", 0.0))

    gate_sparse = bool(best_gate_ratio < min(0.30, float(v4_best_gate_ratio) - 1e-6))
    guaranteed_ok = bool(best_guaranteed_pairs >= 1.0)
    fallback_reasonable = bool(best_fallback_trigger <= 0.70)
    hard_improved_vs_v4 = bool(best_hard_composite < float(v4_best_hard_comp) * 0.98)
    non_catastrophic = bool(best_full_endpoint <= crop_ep * 1.5)
    improved_vs_cropenc = bool(best_full_endpoint < crop_ep)
    improved_vs_v4 = bool(best_hard_composite < float(v4_best_hard_comp))
    sidecar_truly_diverged = any(bool(run_aux.get(str(row["run_name"]), {}).get("sidecar_truly_diverged", False)) for row in completed)

    combo_rows = [x for x in payload["semantic_hard_composite_panel"]["runs"] if str(x.get("objective_combo", "")) == best_combo]
    cross_seed_support = False
    if len({int(x.get("seed", -1)) for x in combo_rows}) >= 2:
        vals = [float(x.get("semantic_hard_composite_score", 1e9)) for x in combo_rows]
        cross_seed_support = bool(min(vals) < float(v4_best_hard_comp) * 0.98 and max(vals) <= float(v4_best_hard_comp) * 1.05)

    if true_new_best and gate_sparse and guaranteed_ok and fallback_reasonable and hard_improved_vs_v4 and cross_seed_support and non_catastrophic:
        next_step = "stage2_semantic_rescue_fullscale_wave1"
    elif non_catastrophic and (improved_vs_cropenc or hard_improved_vs_v4 or sidecar_truly_diverged or guaranteed_ok):
        next_step = "redesign_stage2_semantic_objective_v7"
    else:
        next_step = "objective_family_still_not_working_under_current_stage2_design"

    payload["success_criteria"] = {
        "true_new_best_not_warm_start_inherited": bool(true_new_best),
        "actual_gate_positive_ratio_below_0_30": bool(gate_sparse),
        "guaranteed_pair_count_mean_above_or_equal_1_0": bool(guaranteed_ok),
        "fallback_trigger_rate_reasonable_below_0_70": bool(fallback_reasonable),
        "semantic_hard_composite_improved_vs_v4": bool(hard_improved_vs_v4),
        "cross_seed_support_present": bool(cross_seed_support),
        "full_validation_non_catastrophic": bool(non_catastrophic),
        "improved_vs_current_cropenc_baseline": bool(improved_vs_cropenc),
        "improved_vs_v4_best_objective_combo": bool(improved_vs_v4),
        "best_v6_objective_combo": str(best_combo),
        "best_v6_full_validation_endpoint_l2": float(best_full_endpoint),
        "best_v6_semantic_hard_composite_score": float(best_hard_composite),
        "best_v6_actual_gate_positive_ratio": float(best_gate_ratio),
        "best_v6_guaranteed_pair_count_mean": float(best_guaranteed_pairs),
        "best_v6_fallback_trigger_rate_mean": float(best_fallback_trigger),
        "best_v6_strict_pair_ratio_mean": float(best_strict_ratio),
        "best_v6_fallback_pair_ratio_mean": float(best_fallback_ratio),
        "same_checkpoint_selected": bool(next((x.get("same_checkpoint_selected", False) for x in payload["semantic_hard_composite_panel"]["runs"] if str(x.get("run_name", "")) == best_hard_run), False)),
        "sidecar_truly_diverged": bool(sidecar_truly_diverged),
        "overall_best_run_name": str(best_full_run),
        "semantic_hard_best_run_name": str(best_hard_run),
        "restricted_next_step_enum": [
            "stage2_semantic_rescue_fullscale_wave1",
            "redesign_stage2_semantic_objective_v7",
            "objective_family_still_not_working_under_current_stage2_design",
        ],
        "next_step_choice": str(next_step),
    }
    _write_json(args.v6_diagnosis_report, payload)
    return payload


def run_all(args: Any) -> Dict[str, Any]:
    write_protocol_doc(args)
    write_decision_rule(args)
    activation_audit(args)
    launch(args)
    wait_for_completion(args)
    diag = diagnose_v6(args)
    return {"v6_diagnosis": diag}


def parse_args() -> Any:
    p = ArgumentParser(description="Stage2 semantic objective redesign v6")
    p.add_argument("--mode", default="all", choices=["all", "activation-audit", "launch", "run-one", "summarize", "diagnose"])
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
    p.add_argument("--v5-reference-summary-report", default=str(WORK_ROOT / "reports/stage2_semantic_objective_redesign_v5_summary_20260411.json"))
    p.add_argument("--v6-activation-audit-report", default=str(WORK_ROOT / "reports/stage2_semantic_objective_redesign_v6_activation_audit_20260411.json"))
    p.add_argument("--v6-protocol-doc", default=str(WORK_ROOT / "docs/STAGE2_SEMANTIC_OBJECTIVE_REDESIGN_V6_PROTOCOL_20260411.md"))
    p.add_argument("--v6-decision-rule-report", default=str(WORK_ROOT / "reports/stage2_semantic_objective_redesign_v6_decision_rule_20260411.json"))
    p.add_argument("--v6-launch-report", default=str(WORK_ROOT / "reports/stage2_semantic_objective_redesign_v6_launch_20260411.json"))
    p.add_argument("--v6-summary-report", default=str(WORK_ROOT / "reports/stage2_semantic_objective_redesign_v6_summary_20260411.json"))
    p.add_argument("--v6-results-md", default=str(WORK_ROOT / "docs/STAGE2_SEMANTIC_OBJECTIVE_REDESIGN_V6_RESULTS_20260411.md"))
    p.add_argument("--v6-diagnosis-report", default=str(WORK_ROOT / "reports/stage2_semantic_objective_redesign_v6_diagnosis_20260411.json"))
    p.add_argument("--v6-diagnosis-cache-report", default=str(WORK_ROOT / "reports/stage2_semantic_objective_redesign_v6_diagnosis_cache_20260411.json"))
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
    elif args.mode == "launch":
        print(json.dumps(launch(args), ensure_ascii=True, indent=2))
    elif args.mode == "run-one":
        run_one(args)
    elif args.mode == "summarize":
        print(json.dumps(summarize(args), ensure_ascii=True, indent=2))
    elif args.mode == "diagnose":
        print(json.dumps(diagnose_v6(args), ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
