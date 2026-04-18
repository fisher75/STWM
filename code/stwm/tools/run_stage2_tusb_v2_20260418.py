#!/usr/bin/env python3
from __future__ import annotations

from argparse import ArgumentParser
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple
import gc
import json
import os
import shlex
import signal
import subprocess
import time
import traceback

import numpy as np
import torch

from stwm.infra.gpu_lease import list_active_leases
from stwm.tools import run_stage2_state_identifiability_eval_20260415 as prev_eval
from stwm.tools import run_stage2_state_identifiability_eval_v3_20260416 as evalv3
from stwm.tools import run_tracewm_stage2_calibration_only_fullscale_wave1_20260413 as base
from stwm.tracewm_v2_stage2.models.trace_unit_broadcast import (
    TraceUnitBroadcast,
    TraceUnitBroadcastConfig,
)
from stwm.tracewm_v2_stage2.models.trace_unit_factorized_state import (
    TraceUnitFactorizedState,
    TraceUnitFactorizedStateConfig,
)
from stwm.tracewm_v2_stage2.models.trace_unit_handshake import (
    TraceUnitHandshake,
    TraceUnitHandshakeConfig,
)
from stwm.tracewm_v2_stage2.models.trace_unit_tokenizer import (
    TraceUnitTokenizer,
    TraceUnitTokenizerConfig,
)
from stwm.tracewm_v2_stage2.trainers import train_tracewm_stage2_smalltrain as trainer


ROOT = prev_eval.ROOT
SESSION = "tracewm_stage2_tusb_v2_20260418"
LOG_PATH = ROOT / "logs/stage2_tusb_v2_20260418.log"
DATE_TAG = "20260418"
BOOTSTRAP_BACKEND = "local_clip_vit_b32_mask_crop_visual_teacher"
MAX_CONCURRENT_TUSB_TASKS = 4
MAX_CONCURRENT_CACHE_TASKS = 2
ADDITIONAL_TRAIN_STEPS = 1000
BATCH_SIZE = 8
EVAL_INTERVAL = 200
SAVE_EVERY = 200
EVAL_MAX_BATCHES = 0
MAX_TRAIN_PER_DATASET = 32
MAX_VAL_PER_DATASET = 32
TUSB_ALLOWED_PREFIXES = ("stage2_tusb_v2_",)
TUSB_RUNTIME_JSON = ROOT / "configs/recommended_stage2_runtime_tusb_v2_20260418.json"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _append_log(message: str) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as fh:
        fh.write(f"[{now_iso()}] {message}\n")


def _json_or_empty(path_like: Any) -> Dict[str, Any]:
    path = Path(str(path_like))
    if not str(path_like) or not path.exists():
        return {}
    try:
        payload = base._read_json(path)
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _repo_root_for_docs() -> Path:
    if ROOT.exists():
        return ROOT
    return Path("/raid/chen034/workspace/stwm")


def _resume_ckpt_for_seed(seed: int) -> Path:
    candidates = [
        ROOT / "outputs/checkpoints" / f"stage2_calonly_topk1_seed{int(seed)}_wave1_20260413" / "best.pt",
        ROOT / "outputs/checkpoints" / f"stage2_calonly_topk1_seed123_wave1_20260413" / "best.pt",
        ROOT / "outputs/checkpoints" / "stage2_calonly_topk1_seed123_longconfirm_v2_20260414" / "best.pt",
    ]
    for ckpt in candidates:
        if ckpt.exists():
            return ckpt
    raise FileNotFoundError(f"missing calibration resume checkpoint for seed {seed}: {candidates[-1]}")


def _inspect_tusb_checkpoint_liveness(ckpt_path: Path) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "checkpoint_path": str(ckpt_path),
        "exists": bool(ckpt_path.exists()),
        "stage2_structure_mode": "",
        "trace_unit_state_dict_keys_present": [],
        "trace_unit_state_dicts_complete": False,
        "teacher_prior_dim": None,
        "checkpoint_contains_trace_unit_modules": False,
        "blocking_reason": "",
    }
    if not ckpt_path.exists():
        result["blocking_reason"] = "checkpoint_missing"
        return result
    try:
        payload = torch.load(ckpt_path, map_location="cpu")
    except Exception as exc:
        result["blocking_reason"] = f"checkpoint_load_failed: {exc!r}"
        return result
    args_dict = payload.get("args", {})
    stage2_structure_mode = str(args_dict.get("stage2_structure_mode", "") or "")
    teacher_prior_dim = args_dict.get("trace_unit_teacher_prior_dim", None)
    required_keys = [
        "trace_unit_tokenizer_state_dict",
        "trace_unit_factorized_state_state_dict",
        "trace_unit_handshake_state_dict",
        "trace_unit_broadcast_state_dict",
    ]
    present = [key for key in required_keys if key in payload]
    result.update(
        {
            "stage2_structure_mode": stage2_structure_mode,
            "trace_unit_state_dict_keys_present": present,
            "trace_unit_state_dicts_complete": bool(len(present) == len(required_keys)),
            "teacher_prior_dim": teacher_prior_dim,
        }
    )
    result["checkpoint_contains_trace_unit_modules"] = bool(
        stage2_structure_mode == "trace_unit_semantic_binding" and result["trace_unit_state_dicts_complete"]
    )
    if not result["checkpoint_contains_trace_unit_modules"]:
        result["blocking_reason"] = "checkpoint_missing_tusb_state_dicts"
    return result


def _run_specs() -> List[Dict[str, Any]]:
    common = {
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
        "v6_strict_max_pairs_per_sample": 0,
        "v6_hard_negative_cap": 0,
        "v6_pair_sampling_temperature": 0.35,
        "v6_guaranteed_min_pairs_per_sample": 0,
        "v6_two_level_pair_mining_enabled": False,
        "v6_relaxed_motion_threshold": 0.08,
        "v6_relaxed_area_jump_threshold": 0.06,
        "v6_relaxed_small_query_threshold": 0.20,
        "v6_relaxed_appearance_shift_threshold": 0.25,
        "v6_relaxed_center_interaction_threshold": 0.10,
        "stage2_structure_mode": "trace_unit_semantic_binding",
        "trace_unit_count": 16,
        "trace_unit_dim": 384,
        "trace_unit_slot_iters": 3,
        "trace_unit_assignment_topk": 2,
        "trace_unit_assignment_temperature": 0.70,
        "trace_unit_use_instance_prior_bias": True,
        "trace_unit_dyn_update": "gru",
        "trace_unit_sem_update": "gated_ema",
        "trace_unit_sem_alpha_min": 0.02,
        "trace_unit_sem_alpha_max": 0.12,
        "trace_unit_handshake_type": "lowrank_cross_attn",
        "trace_unit_handshake_dim": 128,
        "trace_unit_handshake_layers": 1,
        "trace_unit_handshake_writeback": "dyn_only",
        "trace_unit_broadcast_residual_weight": 0.35,
        "trace_unit_broadcast_stopgrad_semantic": False,
        "trace_unit_assignment_sparsity_weight": 0.02,
        "trace_unit_assignment_temporal_consistency_weight": 0.05,
        "trace_unit_semantic_inertia_weight": 0.05,
        "trace_unit_instance_consistency_weight": 0.10,
        "trace_unit_dynsem_decorrelation_weight": 0.005,
        "trace_unit_utilization_weight": 0.06,
        "trace_unit_min_active_target": 4.0,
        "trace_unit_diversity_weight": 0.03,
        "trace_unit_top2_floor_weight": 0.03,
        "trace_unit_top2_mass_floor": 0.15,
        "trace_unit_teacher_prior_dim": 512,
        "trace_unit_disable_instance_path": False,
        "max_entities_per_sample": 8,
        "local_temporal_window": 1,
        "local_temporal_fuse_weight": 0.0,
    }
    specs: List[Dict[str, Any]] = []
    for seed, window_name in [(123, "tusb_s123"), (42, "tusb_s42"), (456, "tusb_s456")]:
        specs.append(
            {
                **common,
                "run_name": f"stage2_tusb_v2_seed{seed}_{DATE_TAG}",
                "seed": int(seed),
                "family": "tusb_v2_main",
                "ablation_name": "main",
                "objective_combo": f"tusb_v2_main_seed{seed}",
                "objective_family": "trace_unit_semantic_binding_v2",
                "window_name": str(window_name),
            }
        )
    specs.append(
        {
            **common,
            "run_name": f"stage2_tusb_v2_no_instance_path_seed123_{DATE_TAG}",
            "seed": 123,
            "family": "tusb_v2_ablation",
            "ablation_name": "no_instance_path",
            "objective_combo": "tusb_v2_no_instance_path_seed123",
            "objective_family": "trace_unit_semantic_binding_v2_ablation",
            "window_name": "tusb_noinst",
            "trace_unit_disable_instance_path": True,
            "trace_unit_instance_consistency_weight": 0.0,
        }
    )
    specs.append(
        {
            **common,
            "run_name": f"stage2_tusb_v2_no_teacher_prior_seed123_{DATE_TAG}",
            "seed": 123,
            "family": "tusb_v2_ablation",
            "ablation_name": "no_teacher_prior",
            "objective_combo": "tusb_v2_no_teacher_prior_seed123",
            "objective_family": "trace_unit_semantic_binding_v2_ablation",
            "window_name": "tusb_noprior",
            "trace_unit_teacher_prior_dim": 0,
            "teacher_semantic_cache_path": "",
        }
    )
    specs.append(
        {
            **common,
            "run_name": f"stage2_tusb_v2_no_anticollapse_seed123_{DATE_TAG}",
            "seed": 123,
            "family": "tusb_v2_ablation",
            "ablation_name": "no_anticollapse",
            "objective_combo": "tusb_v2_no_anticollapse_seed123",
            "objective_family": "trace_unit_semantic_binding_v2_ablation",
            "window_name": "tusb_nocollapse",
            "trace_unit_utilization_weight": 0.0,
            "trace_unit_diversity_weight": 0.0,
            "trace_unit_top2_floor_weight": 0.0,
            "trace_unit_min_active_target": 1.0,
        }
    )
    return specs


def _spec_by_run_name(run_name: str) -> Dict[str, Any]:
    for spec in _run_specs():
        if str(spec["run_name"]) == str(run_name):
            return spec
    raise KeyError(f"unknown TUSB run_name: {run_name}")


def _meta_dir(args: Any) -> Path:
    return Path(args.work_root) / "reports/stage2_tusb_v2_runs_20260418"


def _paths_for_run(args: Any, run_name: str) -> Dict[str, Path]:
    reports = Path(args.work_root) / "reports"
    out_dir = Path(args.work_root) / "outputs/checkpoints" / run_name
    return {
        "raw": reports / f"{run_name}_raw.json",
        "progress": reports / f"{run_name}_progress.json",
        "final": reports / f"{run_name}_final.json",
        "log": Path(args.work_root) / "logs" / f"{run_name}.log",
        "output_dir": out_dir,
        "best": out_dir / "best.pt",
        "latest": out_dir / "latest.pt",
        "sidecar": out_dir / "best_semantic_hard.pt",
        "launch": _meta_dir(args) / f"{run_name}_launch_meta.json",
    }


def _instance_cache_compatibility(predecode_cache_path: Path) -> Dict[str, Any]:
    result = {
        "predecode_cache_path": str(predecode_cache_path),
        "exists": bool(predecode_cache_path.exists()),
        "sample_npz_checked": "",
        "required_instance_keys": [
            "semantic_instance_id_map",
            "semantic_instance_id_crop",
            "semantic_instance_id_temporal",
            "semantic_instance_valid",
            "semantic_objectness_score",
        ],
        "missing_keys": [],
        "compatible": False,
        "exact_blocking_reason": "",
    }
    if not predecode_cache_path.exists():
        result["exact_blocking_reason"] = "predecode cache directory missing"
        return result
    sample = None
    for candidate in sorted(predecode_cache_path.rglob("VIPSeg*.npz")):
        sample = candidate
        break
    if sample is None:
        for candidate in sorted(predecode_cache_path.rglob("*.npz")):
            sample = candidate
            break
    if sample is None:
        result["exact_blocking_reason"] = "no npz entries found in predecode cache"
        return result
    result["sample_npz_checked"] = str(sample)
    try:
        with np.load(sample, allow_pickle=False) as payload:
            keys = set(payload.files)
    except Exception as exc:
        result["exact_blocking_reason"] = f"failed_to_open_sample_npz: {exc!r}"
        return result
    missing = [key for key in result["required_instance_keys"] if key not in keys]
    result["missing_keys"] = missing
    result["compatible"] = bool(not missing)
    if missing:
        result["exact_blocking_reason"] = (
            "existing 20260416 predecode cache was built before TUSB instance-aware fields; "
            "dataset now falls back to raw decode when cached payload misses semantic_instance_id_* or semantic_objectness_score"
        )
    return result


def _count_tusb_params() -> Dict[str, Any]:
    tokenizer = TraceUnitTokenizer(
        TraceUnitTokenizerConfig(
            hidden_dim=1152,
            semantic_dim=256,
            state_dim=8,
            teacher_prior_dim=512,
            unit_dim=384,
            unit_count=16,
            slot_iters=3,
            assignment_topk=2,
            assignment_temperature=0.70,
            use_instance_prior_bias=True,
        )
    )
    factorized = TraceUnitFactorizedState(
        TraceUnitFactorizedStateConfig(
            unit_dim=384,
            dyn_update="gru",
            sem_update="gated_ema",
            sem_alpha_min=0.02,
            sem_alpha_max=0.12,
        )
    )
    handshake = TraceUnitHandshake(
        TraceUnitHandshakeConfig(
            unit_dim=384,
            handshake_dim=128,
            layers=1,
            writeback="dyn_only",
        )
    )
    broadcast = TraceUnitBroadcast(
        TraceUnitBroadcastConfig(
            hidden_dim=1152,
            unit_dim=384,
            residual_weight=0.35,
            stopgrad_semantic=False,
        )
    )

    def _count(mod: torch.nn.Module) -> int:
        return int(sum(param.numel() for param in mod.parameters()))

    blocks = {
        "trace_unit_tokenizer": _count(tokenizer),
        "trace_unit_factorized_state": _count(factorized),
        "trace_unit_handshake": _count(handshake),
        "trace_unit_broadcast": _count(broadcast),
    }
    total = int(sum(blocks.values()))
    return {
        "generated_at_utc": now_iso(),
        "stage1_parameter_budget_changed": False,
        "trace_unit_count": 16,
        "trace_unit_dim": 384,
        "handshake_dim": 128,
        "slot_iters": 3,
        "assignment_topk": 2,
        "new_stage2_module_parameter_breakdown": blocks,
        "new_stage2_module_parameter_total": total,
        "under_12m_budget": bool(total < 12_000_000),
        "notes": [
            "parameter count is for newly introduced TUSB-V2 modules only",
            "hidden_dim assumes prototype_220m frozen Stage1 backbone with d_model=1152",
            "no Stage1 weight change included in this budget",
        ],
    }


def _write_frozen_fact_artifacts(args: Any) -> None:
    protocol_payload = {
        "generated_at_utc": now_iso(),
        "stage1_frozen_backbone": {
            "status": "frozen",
            "training_allowed": False,
            "unfreeze_allowed": False,
            "backbone_story": "trace-first future-state backbone remains the stable untouchable core",
        },
        "current_tusb_lite_state": {
            "already_landed": True,
            "freeze_ready_mainline": False,
            "key_blockers": [
                "protocol_v3_eval_path_likely_blind_to_tusb",
                "dataset_training_body_still_single_entity",
                "instance_aware_real_signal_not_used_in_main_training_body",
                "active_units_collapsed_near_single_unit",
                "20260416_predecode_cache_incompatible_with_instance_aware_fields",
            ],
        },
        "current_stage2_state": {
            "current_reasonable_mainline": "calibration_only",
            "interpretation": "semantic calibration/readout alignment rather than semanticized trace state",
        },
        "repair_goal": {
            "grounding_first_mainline": False,
            "continue_calibration_only_micro_fix": False,
            "target": "repair evaluation blindness, move dataset to multi-entity samples, inject real instance-aware supervision, deepen TUSB without abandoning the current story",
        },
        "forbidden": [
            "modify_stage1_weights",
            "restore_persistence_mainline",
            "continue_protocol_v4",
            "continue_large_qualitative_expansion",
            "video_or_render_head",
            "codec_or_vae",
            "full_architecture_search"
        ],
    }
    base._write_json(args.protocol_report, protocol_payload)
    base._write_md(
        args.protocol_doc,
        [
            "# Stage2 TUSB-V2 Repair Protocol 20260418",
            "",
            f"- generated_at_utc: {protocol_payload['generated_at_utc']}",
            "- stage1_backbone_status: frozen and untouchable for this round",
            "- stage1_training_allowed: false",
            "- stage1_unfreeze_allowed: false",
            "- current_stage2_interpretation: calibration-only remains the current reasonable mainline, but it is semantic calibration/readout alignment rather than semanticized trace state",
            "- current_tusb_lite_status: landed but not freeze-ready",
            "- key_blockers: eval blind risk; single-entity training body; weak real instance signal usage; active-unit collapse; old cache incompatibility",
            "- repair_target: multi-entity TUSB-V2 with real instance-aware path, anti-collapse unitization, and stronger frozen semantic prior",
            "- forbidden: Stage1 changes; persistence revival; protocol v4; qualitative expansion; video/render head; codec/VAE; full architecture search",
        ],
    )

    param_payload = _count_tusb_params()
    base._write_json(args.param_budget_report, param_payload)
    base._write_md(
        args.param_budget_doc,
        [
            "# Stage2 Trace-Unit Parameter Budget 20260418",
            "",
            f"- generated_at_utc: {param_payload['generated_at_utc']}",
            "- stage1_parameter_budget_changed: false",
            f"- trace_unit_count: {param_payload['trace_unit_count']}",
            f"- trace_unit_dim: {param_payload['trace_unit_dim']}",
            f"- handshake_dim: {param_payload['handshake_dim']}",
            f"- slot_iters: {param_payload['slot_iters']}",
            f"- assignment_topk: {param_payload['assignment_topk']}",
            f"- new_stage2_module_parameter_total: {param_payload['new_stage2_module_parameter_total']}",
            f"- under_12m_budget: {param_payload['under_12m_budget']}",
        ]
        + [
            f"- {name}: {count}"
            for name, count in param_payload["new_stage2_module_parameter_breakdown"].items()
        ],
    )
    runtime_payload = {
        "generated_at_utc": now_iso(),
        "mode": "stage2_runtime_pipeline_tusb_v2_compatible",
        "selected_gpu_policy": {
            "mode": "shared_gpu_selector",
            "selected_gpu_id": 3,
        },
        "recommended_num_workers": 4,
        "recommended_pin_memory": True,
        "recommended_persistent_workers": True,
        "recommended_prefetch_factor": 2,
        "required_mem_gb": 24.0,
        "safety_margin_gb": 4.0,
        "single_gpu_only": True,
        "notes": [
            "TUSB-v2 multi-entity samples caused host-side oversubscription under the 20260416 12-worker runtime",
            "this runtime intentionally trades peak single-run throughput for stable multi-run progress",
            "exact blocking reason for old runtime: 4 concurrent TUSB-v2 runs with num_workers=12 stalled before first eval/save boundary",
        ],
    }
    base._write_json(TUSB_RUNTIME_JSON, runtime_payload)


def _predict_stage2_with_hidden(method: prev_eval.LoadedMethod, batch: Dict[str, Any], device: torch.device) -> Dict[str, np.ndarray]:
    moved = trainer._to_device(batch, device=device, non_blocking=False)
    with torch.no_grad():
        out = trainer._free_rollout_predict(
            stage1_model=method.stage1_model,
            semantic_encoder=method.semantic_encoder,
            semantic_fusion=method.semantic_fusion,
            readout_head=method.readout_head,
            structure_mode=str(getattr(method, "stage2_structure_mode", "calibration_only")),
            trace_unit_tokenizer=getattr(method, "trace_unit_tokenizer", None),
            trace_unit_factorized_state=getattr(method, "trace_unit_factorized_state", None),
            trace_unit_handshake=getattr(method, "trace_unit_handshake", None),
            trace_unit_broadcast=getattr(method, "trace_unit_broadcast", None),
            trace_unit_disable_instance_path=bool(getattr(method, "trace_unit_disable_instance_path", False)),
            batch=moved,
            obs_len=prev_eval.OBS_LEN,
            fut_len=prev_eval.FUT_LEN,
            semantic_source_mainline=method.semantic_source_mainline,
            allow_stage1_grad=False,
        )
    return {
        "pred_coord": out["pred_coord"].detach().cpu().numpy(),
        "future_hidden": out["future_hidden"].detach().cpu().numpy(),
    }


def _current_calibration_best_run(args: Any) -> str:
    for path_like in [
        ROOT / "reports/stage2_final_utility_closure_v2_diagnosis_20260414.json",
        ROOT / "reports/stage2_calibration_only_final_pack_diagnosis_20260414.json",
    ]:
        payload = _json_or_empty(path_like)
        for key in ["current_best_overall_run_name", "overall_best_run_name", "best_overall_run_name"]:
            value = str(payload.get(key, "")).strip()
            if value:
                return value
    return "stage2_calonly_topk1_seed123_wave1_20260413"


def _best_tusb_lite_run(args: Any) -> str:
    for path_like in [
        ROOT / "reports/stage2_trace_unit_semantic_binding_diagnosis_20260417.json",
        ROOT / "reports/stage2_trace_unit_semantic_binding_summary_20260417.json",
    ]:
        payload = _json_or_empty(path_like)
        value = str(payload.get("best_tusb_run_name", "")).strip()
        if value:
            return value
    return "stage2_tusb_lite_seed123_20260417"


def _run_eval_liveness(args: Any) -> Dict[str, Any]:
    protocol = _json_or_empty(args.protocol_v3_json)
    items = protocol.get("items", []) if isinstance(protocol.get("items", []), list) else []
    mini_items = [item for item in items if isinstance(item, dict)][:16]
    if not mini_items:
        payload = {
            "generated_at_utc": now_iso(),
            "eval_path_tusb_liveness_passed": False,
            "blocking_reason": "empty_protocol_v3_items",
        }
        base._write_json(args.eval_liveness_report, payload)
        base._write_md(args.eval_liveness_doc, ["# Stage2 TUSB Eval Liveness 20260418", "", "- eval_path_tusb_liveness_passed: false", "- blocking_reason: empty_protocol_v3_items"])
        return payload

    cal_run = _current_calibration_best_run(args)
    tusb_run = _best_tusb_lite_run(args)
    cal_ckpt = ROOT / "outputs/checkpoints" / cal_run / "best.pt"
    tusb_ckpt = ROOT / "outputs/checkpoints" / tusb_run / "best.pt"
    tusb_ckpt_liveness = _inspect_tusb_checkpoint_liveness(tusb_ckpt)
    if not cal_ckpt.exists() or not tusb_ckpt.exists():
        payload = {
            "generated_at_utc": now_iso(),
            "eval_path_tusb_liveness_passed": False,
            "blocking_reason": "missing_calibration_or_tusb_checkpoint",
            "current_calibration_only_best": cal_run,
            "current_tusb_best": tusb_run,
            "tusb_checkpoint_liveness": tusb_ckpt_liveness,
        }
        base._write_json(args.eval_liveness_report, payload)
        base._write_md(args.eval_liveness_doc, ["# Stage2 TUSB Eval Liveness 20260418", "", "- eval_path_tusb_liveness_passed: false", f"- blocking_reason: {payload['blocking_reason']}"])
        return payload
    if not bool(tusb_ckpt_liveness.get("checkpoint_contains_trace_unit_modules", False)):
        payload = {
            "generated_at_utc": now_iso(),
            "current_calibration_only_best": cal_run,
            "current_tusb_best": tusb_run,
            "eval_path_tusb_liveness_passed": False,
            "protocol_v3_sees_tusb_modules": False,
            "previous_metric_flatness_likely_eval_blind_artifact": True,
            "blocking_reason": str(tusb_ckpt_liveness.get("blocking_reason", "checkpoint_missing_tusb_state_dicts")),
            "tusb_checkpoint_liveness": tusb_ckpt_liveness,
            "note": "current 20260417 TUSB-lite checkpoint does not contain trace-unit modules, so old protocol-v3 parity against calibration-only was primarily an eval/checkpoint blindness artifact",
        }
        base._write_json(args.eval_liveness_report, payload)
        base._write_md(
            args.eval_liveness_doc,
            [
                "# Stage2 TUSB Eval Liveness 20260418",
                "",
                f"- current_calibration_only_best: {cal_run}",
                f"- current_tusb_best: {tusb_run}",
                "- eval_path_tusb_liveness_passed: false",
                "- protocol_v3_sees_tusb_modules: false",
                "- previous_metric_flatness_likely_eval_blind_artifact: true",
                f"- blocking_reason: {payload['blocking_reason']}",
                "- note: old TUSB-lite checkpoint is missing trace-unit state dicts, so protocol-v3 parity with calibration-only is not scientifically meaningful",
            ],
        )
        return payload

    if not hasattr(args, "lease_path") or not str(getattr(args, "lease_path", "")).strip():
        setattr(args, "lease_path", str(args.shared_lease_path))
    device, device_info = evalv3._select_eval_device_v3(args)
    cal_spec = prev_eval.MethodSpec("current_calibration_only_best", cal_run, "stage2", str(cal_ckpt))
    tusb_spec = prev_eval.MethodSpec("current_tusb_best", tusb_run, "stage2", str(tusb_ckpt))
    cal_method = prev_eval._load_method(cal_spec, device=device)
    tusb_method = prev_eval._load_method(tusb_spec, device=device)
    pred_diffs: List[float] = []
    hidden_diffs: List[float] = []
    changed_top1 = 0
    comparisons: List[Dict[str, Any]] = []
    try:
        for item in mini_items:
            batch, _, future_masks = evalv3._build_single_item_batch_v3(item)
            cal_out = _predict_stage2_with_hidden(cal_method, batch, device=device)
            tusb_out = _predict_stage2_with_hidden(tusb_method, batch, device=device)
            pred_diff = float(np.sqrt(((cal_out["pred_coord"] - tusb_out["pred_coord"]) ** 2).sum(axis=-1)).mean())
            hid_diff = float(np.sqrt(((cal_out["future_hidden"] - tusb_out["future_hidden"]) ** 2).mean()))
            pred_diffs.append(pred_diff)
            hidden_diffs.append(hid_diff)
            width = int((item.get("image_size") or {}).get("width", 1))
            height = int((item.get("image_size") or {}).get("height", 1))
            cal_xy = cal_out["pred_coord"][0, -1, 0]
            tusb_xy = tusb_out["pred_coord"][0, -1, 0]
            cal_top1, _ = prev_eval._candidate_ranking((float(cal_xy[0]), float(cal_xy[1])), future_masks, width=width, height=height)
            tusb_top1, _ = prev_eval._candidate_ranking((float(tusb_xy[0]), float(tusb_xy[1])), future_masks, width=width, height=height)
            changed = str(cal_top1) != str(tusb_top1)
            changed_top1 += int(changed)
            comparisons.append(
                {
                    "protocol_item_id": str(item.get("protocol_item_id", "")),
                    "pred_coord_l2_diff": pred_diff,
                    "future_hidden_l2_diff": hid_diff,
                    "calibration_top1": str(cal_top1),
                    "tusb_top1": str(tusb_top1),
                    "top1_changed": bool(changed),
                }
            )
    finally:
        prev_eval._release_method(cal_method)
        prev_eval._release_method(tusb_method)
        lease_id = str(device_info.get("lease_id", "")).strip()
        if lease_id:
            try:
                prev_eval.release_lease(lease_id=lease_id, lease_path=str(args.shared_lease_path))
            except Exception:
                pass

    mean_pred_diff = float(sum(pred_diffs) / max(len(pred_diffs), 1))
    mean_hidden_diff = float(sum(hidden_diffs) / max(len(hidden_diffs), 1))
    top1_changed_ratio = float(changed_top1 / max(len(comparisons), 1))
    passed = bool(mean_hidden_diff > 1e-6 or mean_pred_diff > 1e-6 or top1_changed_ratio > 0.0)
    payload = {
        "generated_at_utc": now_iso(),
        "current_calibration_only_best": cal_run,
        "current_tusb_best": tusb_run,
        "protocol_item_count_checked": int(len(comparisons)),
        "selected_device": str(device),
        "device_info": device_info,
        "tusb_checkpoint_liveness": tusb_ckpt_liveness,
        "mean_pred_coord_l2_diff": mean_pred_diff,
        "mean_future_hidden_l2_diff": mean_hidden_diff,
        "top1_changed_ratio": top1_changed_ratio,
        "eval_path_tusb_liveness_passed": bool(passed),
        "protocol_v3_sees_tusb_modules": bool(passed),
        "previous_metric_flatness_likely_eval_blind_artifact": bool(not passed),
        "comparisons": comparisons,
    }
    base._write_json(args.eval_liveness_report, payload)
    base._write_md(
        args.eval_liveness_doc,
        [
            "# Stage2 TUSB Eval Liveness 20260418",
            "",
            f"- current_calibration_only_best: {cal_run}",
            f"- current_tusb_best: {tusb_run}",
            f"- protocol_item_count_checked: {len(comparisons)}",
            f"- mean_pred_coord_l2_diff: {mean_pred_diff:.8f}",
            f"- mean_future_hidden_l2_diff: {mean_hidden_diff:.8f}",
            f"- top1_changed_ratio: {top1_changed_ratio:.4f}",
            f"- eval_path_tusb_liveness_passed: {passed}",
            f"- protocol_v3_sees_tusb_modules: {passed}",
            f"- previous_metric_flatness_likely_eval_blind_artifact: {passed}",
        ],
    )
    return payload


def _common_launch_context(args: Any) -> Dict[str, Any]:
    lease_cleanup = base._cleanup_stale_leases(str(args.shared_lease_path), allowed_prefixes=TUSB_ALLOWED_PREFIXES)
    if subprocess.run(["tmux", "has-session", "-t", str(args.tmux_session)], capture_output=True).returncode != 0:
        subprocess.run(["tmux", "new-session", "-d", "-s", str(args.tmux_session), "bash"], check=True)
    existing_windows = set(base._tmux_windows(str(args.tmux_session)))
    anchor_args = base._load_ckpt_args(_resume_ckpt_for_seed(123))
    obs_len = int(anchor_args.get("obs_len", 8) or 8)
    fut_len = int(anchor_args.get("fut_len", 8) or 8)
    max_tokens = int(anchor_args.get("max_tokens", 64) or 64)
    crop_size = int(anchor_args.get("semantic_crop_size", 64) or 64)
    train_counts = base._dataset_counts(["vspw", "vipseg"], "train", args.stage2_contract_json, max_samples=MAX_TRAIN_PER_DATASET)
    val_counts = base._dataset_counts(["vspw", "vipseg"], "val", args.stage2_contract_json, max_samples=MAX_VAL_PER_DATASET)
    meta_dir = _meta_dir(args)
    meta_dir.mkdir(parents=True, exist_ok=True)
    return {
        "lease_cleanup": lease_cleanup,
        "existing_windows": existing_windows,
        "obs_len": obs_len,
        "fut_len": fut_len,
        "max_tokens": max_tokens,
        "crop_size": crop_size,
        "train_counts": train_counts,
        "val_counts": val_counts,
        "meta_dir": meta_dir,
    }


def _build_launch_meta(args: Any, spec: Dict[str, Any], ctx: Dict[str, Any]) -> Dict[str, Any]:
    run_name = str(spec["run_name"])
    resume_from = _resume_ckpt_for_seed(int(spec["seed"]))
    resume_step = base._load_ckpt_step(resume_from)
    out_dir = Path(args.work_root) / "outputs/checkpoints" / run_name
    cache_compat = _instance_cache_compatibility(Path(args.predecode_cache_path))
    meta = {
        **spec,
        "selected_gpu_id": -1,
        "lease_id": "",
        "dataset_names": ["vspw", "vipseg"],
        "obs_len": int(ctx["obs_len"]),
        "fut_len": int(ctx["fut_len"]),
        "max_tokens": int(ctx["max_tokens"]),
        "semantic_crop_size": int(ctx["crop_size"]),
        "semantic_source_mainline": "crop_visual_encoder",
        "legacy_semantic_source": "hand_crafted_stats",
        "batch_size": BATCH_SIZE,
        "resume_from": str(resume_from),
        "resume_global_step": int(resume_step),
        "additional_train_steps": ADDITIONAL_TRAIN_STEPS,
        "train_steps": int(resume_step + ADDITIONAL_TRAIN_STEPS),
        "eval_interval": EVAL_INTERVAL,
        "eval_max_batches": EVAL_MAX_BATCHES,
        "save_every_n_steps": SAVE_EVERY,
        "max_samples_train": MAX_TRAIN_PER_DATASET,
        "max_samples_val": MAX_VAL_PER_DATASET,
        "effective_train_sample_count_per_dataset": ctx["train_counts"],
        "effective_val_sample_count_per_dataset": ctx["val_counts"],
        "semantic_bootstrap_target_dim": 512,
        "semantic_hard_curriculum_weight": 0.0,
        "semantic_aux_subset_weighting_strength": 1.0,
        "output_dir": str(out_dir),
        "raw_json": str(_paths_for_run(args, run_name)["raw"]),
        "progress_json": str(_paths_for_run(args, run_name)["progress"]),
        "final_json": str(_paths_for_run(args, run_name)["final"]),
        "log_path": str(_paths_for_run(args, run_name)["log"]),
        "stage2_contract_json": str(args.stage2_contract_json),
        "runtime_json": str(args.runtime_json),
        "stage1_best_ckpt": str(args.stage1_best_ckpt),
        "shared_lease_path": str(args.shared_lease_path),
        "bootstrap_cache_jsonl": str(args.bootstrap_cache_jsonl),
        "semantic_hard_manifest_path": str(args.semantic_hard_manifest_path),
        "predecode_cache_path": str(args.predecode_cache_path),
        "teacher_semantic_cache_path": str(spec.get("teacher_semantic_cache_path", args.teacher_semantic_cache_path)),
        "work_root": str(args.work_root),
        "python_bin": str(args.python_bin),
        "worker_pid_file": str(ctx["meta_dir"] / f"{run_name}.pid"),
        "gpu_acquire_timeout_seconds": int(args.gpu_acquire_timeout_seconds),
        "gpu_acquire_retry_seconds": int(args.gpu_acquire_retry_seconds),
        "max_concurrent_tusb_tasks": int(args.max_concurrent_tusb_tasks),
        "predecode_cache_instance_aware_compatible": bool(cache_compat["compatible"]),
        "predecode_cache_exact_blocking_reason": str(cache_compat["exact_blocking_reason"]),
        "selector_payload": {},
    }
    meta_json = ctx["meta_dir"] / f"{run_name}_launch_meta.json"
    meta["meta_json"] = str(meta_json)
    return meta


def _tmux_window_command(args: Any, meta_json: Path, meta: Dict[str, Any]) -> str:
    pid_path = str(meta["worker_pid_file"])
    log_path = str(meta["log_path"])
    script_path = Path(args.work_root) / "code/stwm/tools/run_stage2_tusb_v2_20260418.py"
    pythonpath_value = f"{args.work_root}/code:{os.environ.get('PYTHONPATH', '')}"
    proc_title = str(os.environ.get("STWM_PROC_TITLE", "python"))
    proc_title_mode = str(os.environ.get("STWM_PROC_TITLE_MODE", "generic"))
    cmd = (
        f"PYTHONPATH={shlex.quote(pythonpath_value)} "
        f"STWM_PROC_TITLE={shlex.quote(proc_title)} "
        f"STWM_PROC_TITLE_MODE={shlex.quote(proc_title_mode)} "
        f"nohup {shlex.quote(str(args.python_bin))} {shlex.quote(str(script_path))} "
        f"--mode run-one --meta-json {shlex.quote(str(meta_json))} --work-root {shlex.quote(str(args.work_root))} "
        f">> {shlex.quote(log_path)} 2>&1 < /dev/null & echo $! > {shlex.quote(pid_path)}; "
        f"while kill -0 \"$(cat {shlex.quote(pid_path)})\" 2>/dev/null; do sleep 30; done"
    )
    return "bash -lc " + shlex.quote(
        f"cd {shlex.quote(str(args.work_root))}; rm -f {shlex.quote(pid_path)}; {cmd}; "
        f"printf '[%s] tmux_window_exit run_name={str(meta['run_name'])} observed_child_exit\\n' \"$(date -Iseconds)\" >> {shlex.quote(log_path)}"
    )


def _write_and_launch_meta(args: Any, meta: Dict[str, Any], existing_windows: set[str]) -> Dict[str, Any]:
    run_name = str(meta["run_name"])
    meta_json = Path(str(meta["meta_json"]))
    base._write_json(meta_json, meta)
    cmd = _tmux_window_command(args=args, meta_json=meta_json, meta=meta)
    if str(meta["window_name"]) in existing_windows:
        subprocess.run(["tmux", "kill-window", "-t", f"{args.tmux_session}:{meta['window_name']}"], check=False)
        existing_windows.discard(str(meta["window_name"]))
    subprocess.run(["tmux", "new-window", "-t", str(args.tmux_session), "-n", str(meta["window_name"]), cmd], check=True)
    existing_windows.add(str(meta["window_name"]))
    return {
        "run_name": run_name,
        "mode": "real_train",
        "resume_from": str(meta["resume_from"]),
        "resume_global_step": int(meta["resume_global_step"]),
        "stage2_structure_mode": str(meta["stage2_structure_mode"]),
    }


def launch(args: Any) -> Dict[str, Any]:
    _write_frozen_fact_artifacts(args)
    liveness = _run_eval_liveness(args)
    ctx = _common_launch_context(args)
    cleanup_actions: List[Dict[str, Any]] = []
    runs: List[Dict[str, Any]] = []
    for spec in _run_specs():
        run_name = str(spec["run_name"])
        meta = _build_launch_meta(args, spec, ctx)
        cleanup_actions.append(base._reset_run_artifacts(args=args, meta=meta, run_name=run_name))
        runs.append(_write_and_launch_meta(args, meta, ctx["existing_windows"]))
    payload = {
        "generated_at_utc": now_iso(),
        "tmux_session": str(args.tmux_session),
        "teacher_backend": BOOTSTRAP_BACKEND,
        "policy": "TUSB-V2 repair-and-deepen on frozen Stage1; multi-entity instance-aware path; no persistence; max 4 concurrent training tasks",
        "lease_cleanup": ctx["lease_cleanup"],
        "cleanup_actions": cleanup_actions,
        "recommended_runtime_json": str(args.runtime_json),
        "predecode_cache_path": str(args.predecode_cache_path),
        "teacher_semantic_cache_path": str(args.teacher_semantic_cache_path),
        "eval_liveness_report": str(args.eval_liveness_report),
        "eval_liveness": liveness,
        "runs": runs,
    }
    base._write_json(args.launch_report, payload)
    return summarize(args)


def _active_tusb_task_count(lease_path: str) -> int:
    leases = list_active_leases(lease_path=str(lease_path))
    count = 0
    for lease in leases:
        owner = str(lease.get("owner", ""))
        if owner.startswith(TUSB_ALLOWED_PREFIXES):
            count += 1
    return int(count)


def _append_trace_unit_flags(cmd: List[str], meta: Dict[str, Any]) -> None:
    cmd.extend(["--stage2-structure-mode", str(meta["stage2_structure_mode"])])
    cmd.extend(["--trace-unit-count", str(meta["trace_unit_count"])])
    cmd.extend(["--trace-unit-dim", str(meta["trace_unit_dim"])])
    cmd.extend(["--trace-unit-slot-iters", str(meta["trace_unit_slot_iters"])])
    cmd.extend(["--trace-unit-assignment-topk", str(meta["trace_unit_assignment_topk"])])
    cmd.extend(["--trace-unit-assignment-temperature", str(meta["trace_unit_assignment_temperature"])])
    cmd.extend(["--trace-unit-teacher-prior-dim", str(meta["trace_unit_teacher_prior_dim"])])
    cmd.extend(["--trace-unit-dyn-update", str(meta["trace_unit_dyn_update"])])
    cmd.extend(["--trace-unit-sem-update", str(meta["trace_unit_sem_update"])])
    cmd.extend(["--trace-unit-sem-alpha-min", str(meta["trace_unit_sem_alpha_min"])])
    cmd.extend(["--trace-unit-sem-alpha-max", str(meta["trace_unit_sem_alpha_max"])])
    cmd.extend(["--trace-unit-handshake-type", str(meta["trace_unit_handshake_type"])])
    cmd.extend(["--trace-unit-handshake-dim", str(meta["trace_unit_handshake_dim"])])
    cmd.extend(["--trace-unit-handshake-layers", str(meta["trace_unit_handshake_layers"])])
    cmd.extend(["--trace-unit-handshake-writeback", str(meta["trace_unit_handshake_writeback"])])
    cmd.extend(["--trace-unit-broadcast-residual-weight", str(meta["trace_unit_broadcast_residual_weight"])])
    cmd.extend(["--trace-unit-assignment-sparsity-weight", str(meta["trace_unit_assignment_sparsity_weight"])])
    cmd.extend(["--trace-unit-assignment-temporal-consistency-weight", str(meta["trace_unit_assignment_temporal_consistency_weight"])])
    cmd.extend(["--trace-unit-semantic-inertia-weight", str(meta["trace_unit_semantic_inertia_weight"])])
    cmd.extend(["--trace-unit-instance-consistency-weight", str(meta["trace_unit_instance_consistency_weight"])])
    cmd.extend(["--trace-unit-dynsem-decorrelation-weight", str(meta["trace_unit_dynsem_decorrelation_weight"])])
    cmd.extend(["--trace-unit-utilization-weight", str(meta["trace_unit_utilization_weight"])])
    cmd.extend(["--trace-unit-min-active-target", str(meta["trace_unit_min_active_target"])])
    cmd.extend(["--trace-unit-diversity-weight", str(meta["trace_unit_diversity_weight"])])
    cmd.extend(["--trace-unit-top2-floor-weight", str(meta["trace_unit_top2_floor_weight"])])
    cmd.extend(["--trace-unit-top2-mass-floor", str(meta["trace_unit_top2_mass_floor"])])
    cmd.extend(["--max-entities-per-sample", str(meta["max_entities_per_sample"])])
    if bool(meta.get("trace_unit_use_instance_prior_bias", False)):
        cmd.append("--trace-unit-use-instance-prior-bias")
    if bool(meta.get("trace_unit_broadcast_stopgrad_semantic", False)):
        cmd.append("--trace-unit-broadcast-stopgrad-semantic")
    if bool(meta.get("trace_unit_disable_instance_path", False)):
        cmd.append("--trace-unit-disable-instance-path")


def run_one(args: Any) -> None:
    meta = base._read_json(args.meta_json)
    lease_id = str(meta.get("lease_id", ""))
    lease_path = str(meta.get("shared_lease_path", ""))
    run_name = str(meta.get("run_name", ""))
    selected_gpu_id = int(meta.get("selected_gpu_id", -1))
    run_paths = _paths_for_run(args, run_name)
    log_path = Path(str(meta.get("log_path", "")))
    if str(log_path):
        log_path.parent.mkdir(parents=True, exist_ok=True)
    final_json_path = Path(str(meta.get("final_json", "")))
    abort_written = False
    trainer_proc: subprocess.Popen[str] | None = None

    def _append_worker_log(message: str) -> None:
        if not str(log_path):
            return
        with log_path.open("a", encoding="utf-8") as log_fh:
            log_fh.write(f"[{now_iso()}] {message}\n")
            log_fh.flush()

    def _write_abort_payload(message: str, *, signal_name: str = "", returncode: int | None = None) -> None:
        nonlocal abort_written
        if abort_written:
            return
        abort_written = True
        payload: Dict[str, Any] = {
            "generated_at_utc": now_iso(),
            "run_name": str(run_name),
            "status": "failed",
            "selected_gpu_id": int(selected_gpu_id),
            "lease_id": str(lease_id),
            "message": str(message),
        }
        if signal_name:
            payload["signal_name"] = str(signal_name)
        if returncode is not None:
            payload["returncode"] = int(returncode)
        try:
            base._write_json(final_json_path, payload)
        except Exception:
            pass

    def _signal_handler(signum: int, _frame: Any) -> None:
        sig_name = ""
        try:
            sig_name = signal.Signals(signum).name
        except Exception:
            sig_name = str(signum)
        _append_worker_log(f"run_one_received_signal run_name={run_name} signal={sig_name}")
        nonlocal trainer_proc
        if trainer_proc is not None and trainer_proc.poll() is None:
            try:
                os.killpg(int(trainer_proc.pid), signal.SIGTERM)
            except Exception:
                pass
        _write_abort_payload(f"run_one_terminated_by_signal_{sig_name}", signal_name=sig_name, returncode=128 + int(signum))
        raise SystemExit(128 + int(signum))

    try:
        signal.signal(signal.SIGHUP, signal.SIG_IGN)
    except Exception:
        pass
    for sig in [signal.SIGTERM, signal.SIGINT]:
        try:
            signal.signal(sig, _signal_handler)
        except Exception:
            pass

    deadline = time.time() + float(meta.get("gpu_acquire_timeout_seconds", args.gpu_acquire_timeout_seconds))
    while time.time() < deadline:
        active_tusb = _active_tusb_task_count(lease_path)
        if active_tusb < int(meta.get("max_concurrent_tusb_tasks", MAX_CONCURRENT_TUSB_TASKS)):
            break
        _append_worker_log(
            f"waiting_for_tusb_capacity run_name={run_name} active_tusb_tasks={active_tusb} "
            f"limit={int(meta.get('max_concurrent_tusb_tasks', MAX_CONCURRENT_TUSB_TASKS))}"
        )
        time.sleep(float(meta.get("gpu_acquire_retry_seconds", args.gpu_acquire_retry_seconds)))
    else:
        _write_abort_payload("tusb_capacity_timeout")
        raise RuntimeError("tusb_capacity_timeout")

    if selected_gpu_id < 0:
        gpu = base._select_clean_gpu_for_calibration(
            run_name=run_name,
            lease_path=lease_path,
            required_mem_gb=24.0,
            safety_margin_gb=4.0,
        )
        selected_gpu_id = int(gpu["selected_gpu_id"])
        lease_id = str(gpu["lease_id"])
        meta["selected_gpu_id"] = int(selected_gpu_id)
        meta["lease_id"] = str(lease_id)
        meta["selector_payload"] = gpu.get("selector_payload", {})
        base._write_json(args.meta_json, meta)
        _append_worker_log(f"gpu_acquired_threshold run_name={run_name} selected_gpu_id={selected_gpu_id} lease_id={lease_id}")

    # A same-name rerun must not inherit stale report/checkpoint artifacts from a
    # previous failed attempt, otherwise summarize/diagnose will keep reading the
    # old terminal state while the new trainer is still running.
    for stale_path in [
        run_paths["raw"],
        run_paths["progress"],
        run_paths["final"],
        run_paths["best"],
        run_paths["latest"],
        run_paths["sidecar"],
    ]:
        try:
            if stale_path.exists():
                stale_path.unlink()
        except Exception:
            pass

    trainer_path = Path(str(meta["work_root"])) / "code/stwm/tracewm_v2_stage2/trainers/train_tracewm_stage2_smalltrain.py"
    cmd = [
        str(meta["python_bin"]),
        str(trainer_path),
        "--stage2-contract-path", str(meta["stage2_contract_json"]),
        "--recommended-runtime-json", str(meta["runtime_json"]),
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
        "--local-temporal-window", str(meta["local_temporal_window"]),
        "--local-temporal-fuse-weight", str(meta["local_temporal_fuse_weight"]),
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
        "--teacher-semantic-cache-path", str(meta.get("teacher_semantic_cache_path", "")),
        "--output-dir", str(meta["output_dir"]),
        "--run-name", str(meta["run_name"]),
        "--run-summary-json", str(meta["raw_json"]),
        "--progress-json", str(meta["progress_json"]),
        "--seed", str(meta["seed"]),
    ]
    _append_trace_unit_flags(cmd, meta)
    predecode_cache_path = str(meta.get("predecode_cache_path", "") or "")
    if predecode_cache_path:
        cmd.extend(["--predecode-cache-path", predecode_cache_path])
    cmd.append("--v6-two-level-pair-mining-enabled" if bool(meta.get("v6_two_level_pair_mining_enabled", False)) else "--no-v6-two-level-pair-mining-enabled")

    try:
        proc_env = os.environ.copy()
        proc_env["CUDA_VISIBLE_DEVICES"] = str(selected_gpu_id)
        proc_env["STWM_PROC_TITLE"] = str(proc_env.get("STWM_PROC_TITLE", "python"))
        proc_env["STWM_PROC_TITLE_MODE"] = str(proc_env.get("STWM_PROC_TITLE_MODE", "generic"))
        proc_env["PYTHONUNBUFFERED"] = "1"
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
            log_fh.write(f"[run-one] run_name={run_name} selected_gpu_id={selected_gpu_id} lease_id={lease_id}\n")
            if isinstance(meta.get("selector_payload", {}), dict):
                log_fh.write(json.dumps(meta["selector_payload"], ensure_ascii=True) + "\n")
            log_fh.flush()
            trainer_proc = subprocess.Popen(
                [cmd[0], "-u", *cmd[1:]],
                cwd=str(meta["work_root"]),
                text=True,
                stdout=log_fh,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                env=proc_env,
                start_new_session=True,
            )
            _append_log(
                f"run_one_spawned_trainer run_name={run_name} "
                f"trainer_pid={int(trainer_proc.pid)} start_new_session=true"
            )
            returncode = trainer_proc.wait()
        trainer_proc = None
        proc = subprocess.CompletedProcess(args=[cmd[0], "-u", *cmd[1:]], returncode=returncode)
        if proc.returncode != 0:
            log_tail = log_path.read_text(encoding="utf-8", errors="ignore")[-4000:] if log_path.exists() else ""
            _write_abort_payload(f"trainer_failed_rc_{proc.returncode}", returncode=int(proc.returncode))
            final_payload = _json_or_empty(final_json_path)
            final_payload["log_tail"] = log_tail
            base._write_json(final_json_path, final_payload)
            raise RuntimeError(f"trainer failed rc={proc.returncode}")
        raw = base._read_json(meta["raw_json"])
        raw.update(
            {
                "generated_at_utc": now_iso(),
                "status": "completed",
                "selected_gpu_id": int(selected_gpu_id),
                "lease_id": str(lease_id),
                "objective_combo": str(meta["objective_combo"]),
                "objective_family": str(meta["objective_family"]),
                "resume_global_step": int(meta["resume_global_step"]),
                "teacher_backend": BOOTSTRAP_BACKEND,
                "stage2_structure_mode": str(meta["stage2_structure_mode"]),
            }
        )
        base._write_json(meta["final_json"], raw)
        abort_written = True
    except Exception as exc:
        _write_abort_payload(str(exc))
        final_payload = _json_or_empty(final_json_path)
        final_payload["traceback"] = traceback.format_exc()
        base._write_json(final_json_path, final_payload)
        raise
    finally:
        base._release_lease_safe(lease_id=lease_id, lease_path=lease_path)


def _trace_unit_block(final_payload: Dict[str, Any], raw_payload: Dict[str, Any], progress_payload: Dict[str, Any]) -> Dict[str, Any]:
    for payload in [final_payload, raw_payload]:
        block = payload.get("trace_unit_metrics", {})
        if isinstance(block, dict) and block:
            return block
    latest = progress_payload.get("latest_eval_metrics", {})
    if isinstance(latest, dict):
        block = latest.get("trace_unit_metrics", {})
        if isinstance(block, dict) and block:
            return block
    return {}


def summarize(args: Any) -> Dict[str, Any]:
    run_rows: List[Dict[str, Any]] = []
    running = completed = failed = 0
    meta_dir = _meta_dir(args)
    for spec in _run_specs():
        run_name = str(spec["run_name"])
        meta_json = meta_dir / f"{run_name}_launch_meta.json"
        meta = _json_or_empty(meta_json)
        paths = _paths_for_run(args, run_name)
        progress_payload = _json_or_empty(paths["progress"])
        final_payload = _json_or_empty(paths["final"])
        raw_payload = _json_or_empty(paths["raw"])
        status_info = base._status_for(
            {**meta, "window_name": str(meta.get("window_name", spec.get("window_name", ""))), "progress_json": str(paths["progress"]), "final_json": str(paths["final"])},
            session_name=str(args.tmux_session),
        )
        resolved_status = str(status_info.get("status", "launched")).lower()
        if resolved_status == "failed" and not paths["final"].exists():
            base._write_json(
                paths["final"],
                {
                    "generated_at_utc": now_iso(),
                    "run_name": run_name,
                    "status": "failed",
                    "message": str(status_info.get("salvage_reason", "failed_without_final_artifact")),
                    "salvaged_from_progress": bool(paths["progress"].exists()),
                },
            )
            final_payload = _json_or_empty(paths["final"])
        running += int(resolved_status == "running")
        completed += int(resolved_status == "completed")
        failed += int(resolved_status == "failed")
        best_ckpt_exists = bool(paths["best"].exists())
        latest_ckpt_exists = bool(paths["latest"].exists())
        sidecar_exists = bool(paths["sidecar"].exists())
        raw_json_exists = bool(paths["raw"].exists())
        scientific_result_valid = base._scientific_artifact_valid(
            resolved_status=resolved_status,
            best_ckpt_exists=best_ckpt_exists,
            latest_ckpt_exists=latest_ckpt_exists,
            raw_json_exists=raw_json_exists,
        )
        best_block = base._best_block(final_payload, raw_payload, progress_payload)
        latest_block = base._latest_block(final_payload, raw_payload, progress_payload)
        sidecar_block = base._sidecar_block(final_payload, raw_payload, progress_payload)
        trace_unit_block = _trace_unit_block(final_payload, raw_payload, progress_payload)
        if not scientific_result_valid:
            best_block = {}
            latest_block = {}
            sidecar_block = {}
            trace_unit_block = {}
        selected_gpu_id, lease_id = base._gpu_selection_from_payload(final_payload, progress_payload, meta)
        run_rows.append(
            {
                "run_name": run_name,
                "family": str(spec["family"]),
                "ablation_name": str(spec["ablation_name"]),
                "seed": int(spec["seed"]),
                "status": resolved_status,
                "global_step": int(progress_payload.get("global_step", best_block.get("global_step", -1))),
                "final_json_exists": bool(paths["final"].exists()),
                "progress_json_exists": bool(paths["progress"].exists()),
                "raw_json_exists": raw_json_exists,
                "best_ckpt_exists": best_ckpt_exists,
                "latest_ckpt_exists": latest_ckpt_exists,
                "sidecar_exists": sidecar_exists,
                "scientific_result_valid": bool(scientific_result_valid),
                "selected_gpu_id": int(selected_gpu_id),
                "lease_id": str(lease_id),
                "batch_size": int(meta.get("batch_size", BATCH_SIZE)),
                "train_steps": int(meta.get("train_steps", 0)),
                "best_checkpoint_metric": best_block,
                "latest_checkpoint_metric": latest_block,
                "semantic_hard_sidecar_metric": sidecar_block,
                "trace_unit_metrics": trace_unit_block,
            }
        )

    completed_rows = [row for row in run_rows if row["status"] == "completed"]
    best_main_run_name = "none"
    if completed_rows:
        main_rows = [row for row in completed_rows if str(row.get("family", "")) == "tusb_v2_main"] or completed_rows
        best_main = min(main_rows, key=base._summary_overall_rank)
        best_main_run_name = str(best_main["run_name"])
    payload = {
        "generated_at_utc": now_iso(),
        "tusb_status": f"{running}_running_{completed}_completed_{failed}_failed",
        "running_count": running,
        "completed_count": completed,
        "failed_count": failed,
        "all_runs_terminal": bool(len(run_rows) > 0 and running == 0 and completed + failed == len(run_rows)),
        "run_rows": run_rows,
        "best_tusb_run_name": best_main_run_name,
        "teacher_backend": BOOTSTRAP_BACKEND,
        "next_step_choice_internal": (
            "ready_for_tusb_diagnosis"
            if bool(len(run_rows) > 0 and running == 0 and failed == 0)
            else ("fix_failed_runs" if failed > 0 else "continue_running")
        ),
    }
    base._write_json(args.summary_report, payload)
    return payload


def wait_for_completion(args: Any) -> Dict[str, Any]:
    deadline = time.time() + float(args.wait_timeout_seconds)
    last = summarize(args)
    while time.time() < deadline:
        if bool(last.get("all_runs_terminal", False)):
            return last
        time.sleep(float(args.poll_seconds))
        last = summarize(args)
    last["timed_out_waiting_for_completion"] = True
    base._write_json(args.summary_report, last)
    return last


def _load_stage2_method_tusb_aware(spec: prev_eval.MethodSpec, device: torch.device) -> prev_eval.LoadedMethod:
    loaded = prev_eval._load_stage2_method(spec, device=device)
    payload = prev_eval._safe_load(Path(spec.checkpoint_path), device=device)
    ckpt_args = payload.get("args", {}) if isinstance(payload.get("args", {}), dict) else {}
    structure_mode = str(ckpt_args.get("stage2_structure_mode", "calibration_only"))
    trace_unit_disable_instance_path = bool(ckpt_args.get("trace_unit_disable_instance_path", False))
    setattr(loaded, "stage2_structure_mode", structure_mode)
    setattr(loaded, "trace_unit_disable_instance_path", trace_unit_disable_instance_path)
    setattr(loaded, "trace_unit_tokenizer", None)
    setattr(loaded, "trace_unit_factorized_state", None)
    setattr(loaded, "trace_unit_handshake", None)
    setattr(loaded, "trace_unit_broadcast", None)
    has_tusb = any(
        isinstance(payload.get(key, None), dict)
        for key in [
            "trace_unit_tokenizer_state_dict",
            "trace_unit_factorized_state_state_dict",
            "trace_unit_handshake_state_dict",
            "trace_unit_broadcast_state_dict",
        ]
    )
    if structure_mode != "trace_unit_semantic_binding" and not has_tusb:
        return loaded

    hidden_dim = int(getattr(loaded.stage1_model.config, "d_model", 1152))
    semantic_dim = int(ckpt_args.get("semantic_embed_dim", 256))
    tokenizer = TraceUnitTokenizer(
        TraceUnitTokenizerConfig(
            hidden_dim=hidden_dim,
            semantic_dim=semantic_dim,
            state_dim=8,
            teacher_prior_dim=int(ckpt_args.get("trace_unit_teacher_prior_dim", 512)),
            unit_dim=int(ckpt_args.get("trace_unit_dim", 384)),
            unit_count=int(ckpt_args.get("trace_unit_count", 16)),
            slot_iters=int(ckpt_args.get("trace_unit_slot_iters", 3)),
            assignment_topk=int(ckpt_args.get("trace_unit_assignment_topk", 2)),
            assignment_temperature=float(ckpt_args.get("trace_unit_assignment_temperature", 0.70)),
            use_instance_prior_bias=bool(ckpt_args.get("trace_unit_use_instance_prior_bias", False)),
        )
    ).to(device)
    factorized = TraceUnitFactorizedState(
        TraceUnitFactorizedStateConfig(
            unit_dim=int(ckpt_args.get("trace_unit_dim", 384)),
            dyn_update=str(ckpt_args.get("trace_unit_dyn_update", "gru")),
            sem_update=str(ckpt_args.get("trace_unit_sem_update", "gated_ema")),
            sem_alpha_min=float(ckpt_args.get("trace_unit_sem_alpha_min", 0.02)),
            sem_alpha_max=float(ckpt_args.get("trace_unit_sem_alpha_max", 0.12)),
        )
    ).to(device)
    handshake = TraceUnitHandshake(
        TraceUnitHandshakeConfig(
            unit_dim=int(ckpt_args.get("trace_unit_dim", 384)),
            handshake_dim=int(ckpt_args.get("trace_unit_handshake_dim", 128)),
            layers=int(ckpt_args.get("trace_unit_handshake_layers", 1)),
            writeback=str(ckpt_args.get("trace_unit_handshake_writeback", "dyn_only")),
        )
    ).to(device)
    broadcast = TraceUnitBroadcast(
        TraceUnitBroadcastConfig(
            hidden_dim=hidden_dim,
            unit_dim=int(ckpt_args.get("trace_unit_dim", 384)),
            residual_weight=float(ckpt_args.get("trace_unit_broadcast_residual_weight", 0.35)),
            stopgrad_semantic=bool(ckpt_args.get("trace_unit_broadcast_stopgrad_semantic", False)),
        )
    ).to(device)
    tokenizer.load_state_dict(payload.get("trace_unit_tokenizer_state_dict", {}), strict=False)
    factorized.load_state_dict(payload.get("trace_unit_factorized_state_state_dict", {}), strict=False)
    handshake.load_state_dict(payload.get("trace_unit_handshake_state_dict", {}), strict=False)
    broadcast.load_state_dict(payload.get("trace_unit_broadcast_state_dict", {}), strict=False)
    tokenizer.eval()
    factorized.eval()
    handshake.eval()
    broadcast.eval()
    setattr(loaded, "trace_unit_tokenizer", tokenizer)
    setattr(loaded, "trace_unit_factorized_state", factorized)
    setattr(loaded, "trace_unit_handshake", handshake)
    setattr(loaded, "trace_unit_broadcast", broadcast)
    return loaded


def _release_method_tusb_aware(method: prev_eval.LoadedMethod) -> None:
    prev_eval._release_method(method)
    for attr_name in [
        "trace_unit_tokenizer",
        "trace_unit_factorized_state",
        "trace_unit_handshake",
        "trace_unit_broadcast",
    ]:
        mod = getattr(method, attr_name, None)
        if mod is None:
            continue
        try:
            mod.to("cpu")
        except Exception:
            pass
        setattr(method, attr_name, None)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _build_single_item_batch_tusb(item: Dict[str, Any]) -> Tuple[Dict[str, Any], np.ndarray, Dict[str, np.ndarray]]:
    batch, target_future_mask, future_masks = evalv3._build_single_item_batch_v3(item)
    mask_crop = batch["semantic_mask_crop"].to(torch.float32)
    objectness = mask_crop.mean(dim=(-1, -2, -3)).clamp(min=0.0, max=1.0)
    temporal_valid = batch["semantic_temporal_valid"].clone()
    real_instance = str(item.get("dataset", "")).strip().lower() in {"vipseg", "burst"}
    batch["semantic_objectness_score"] = objectness
    batch["semantic_instance_valid"] = temporal_valid if real_instance else torch.zeros_like(temporal_valid)
    crop_h = int(batch["semantic_rgb_crop"].shape[-2])
    crop_w = int(batch["semantic_rgb_crop"].shape[-1])
    temporal_window = int(batch["semantic_rgb_crop_temporal"].shape[2])
    bsz = int(batch["batch_size"])
    k_len = int(batch["semantic_rgb_crop"].shape[1])
    batch["semantic_instance_id_crop"] = torch.zeros((bsz, k_len, 1, crop_h, crop_w), dtype=torch.long)
    batch["semantic_instance_id_temporal"] = torch.zeros((bsz, k_len, temporal_window, 1, crop_h, crop_w), dtype=torch.long)
    batch["semantic_instance_id_map"] = [torch.zeros((1, 1), dtype=torch.long) for _ in range(bsz)]
    return batch, target_future_mask, future_masks


def _stage2_free_rollout_predict_tusb_aware(method: prev_eval.LoadedMethod, batch: Dict[str, Any], device: torch.device) -> np.ndarray:
    moved = trainer._to_device(batch, device=device, non_blocking=False)
    with torch.no_grad():
        out = trainer._free_rollout_predict(
            stage1_model=method.stage1_model,
            semantic_encoder=method.semantic_encoder,
            semantic_fusion=method.semantic_fusion,
            readout_head=method.readout_head,
            structure_mode=str(getattr(method, "stage2_structure_mode", "calibration_only")),
            trace_unit_tokenizer=getattr(method, "trace_unit_tokenizer", None),
            trace_unit_factorized_state=getattr(method, "trace_unit_factorized_state", None),
            trace_unit_handshake=getattr(method, "trace_unit_handshake", None),
            trace_unit_broadcast=getattr(method, "trace_unit_broadcast", None),
            trace_unit_disable_instance_path=bool(getattr(method, "trace_unit_disable_instance_path", False)),
            batch=moved,
            obs_len=prev_eval.OBS_LEN,
            fut_len=prev_eval.FUT_LEN,
            semantic_source_mainline=method.semantic_source_mainline,
            allow_stage1_grad=False,
        )
    return out["pred_coord"].detach().cpu().numpy()


def _predict_final_coord_tusb_aware(method: prev_eval.LoadedMethod, batch: Dict[str, Any], device: torch.device) -> Tuple[float, float]:
    if method.method_type == "stage1":
        pred = prev_eval._stage1_free_rollout_predict(method.stage1_model, batch, device=device)
    else:
        pred = _stage2_free_rollout_predict_tusb_aware(method, batch, device=device)
    coord = pred[0, -1, 0]
    return float(coord[0]), float(coord[1])


def _evaluate_item_tusb_aware(
    method: prev_eval.LoadedMethod,
    item: Dict[str, Any],
    batch: Dict[str, Any],
    target_future_mask: np.ndarray,
    future_masks: Dict[str, np.ndarray],
    device: torch.device,
) -> Dict[str, Any]:
    width = int((item.get("image_size") or {}).get("width", target_future_mask.shape[1]))
    height = int((item.get("image_size") or {}).get("height", target_future_mask.shape[0]))
    pred_x_norm, pred_y_norm = _predict_final_coord_tusb_aware(method, batch, device=device)
    pred_x = min(max(pred_x_norm * float(width), 0.0), float(width - 1))
    pred_y = min(max(pred_y_norm * float(height), 0.0), float(height - 1))
    target_cx, target_cy = prev_eval._mask_centroid(target_future_mask)
    diag = max(np.sqrt(float(width * width + height * height)), 1.0)
    localization_error = float(np.sqrt((pred_x - target_cx) ** 2 + (pred_y - target_cy) ** 2) / diag)
    y_idx = int(round(pred_y))
    x_idx = int(round(pred_x))
    hit = bool(0 <= y_idx < target_future_mask.shape[0] and 0 <= x_idx < target_future_mask.shape[1] and target_future_mask[y_idx, x_idx])
    top1_id, _ = prev_eval._candidate_ranking((pred_x_norm, pred_y_norm), future_masks, width=width, height=height)
    top1_mask = future_masks.get(str(top1_id))
    return {
        "query_future_top1_acc": 1.0 if str(top1_id) == str(item.get("target_id")) else 0.0,
        "query_future_hit_rate": 1.0 if hit else 0.0,
        "query_future_localization_error": float(localization_error),
        "future_mask_iou_at_top1": float(prev_eval._mask_iou(top1_mask, target_future_mask)),
        "top1_candidate_id": str(top1_id),
        "predicted_future_xy_norm": [float(pred_x_norm), float(pred_y_norm)],
        "predicted_future_xy_pixels": [float(pred_x), float(pred_y)],
    }


def _load_eval_methods(args: Any, best_tusb_run_name: str) -> List[prev_eval.MethodSpec]:
    methods: List[prev_eval.MethodSpec] = [
        prev_eval.MethodSpec(
            name="stage1_frozen_baseline",
            run_name="stage1_frozen_baseline",
            method_type="stage1",
            checkpoint_path=str(args.stage1_best_ckpt),
        ),
        prev_eval.MethodSpec(
            name="legacysem_best",
            run_name="stage2_fullscale_core_legacysem_seed456_wave2_20260409",
            method_type="stage2",
            checkpoint_path=str(ROOT / "outputs/checkpoints/stage2_fullscale_core_legacysem_seed456_wave2_20260409/best.pt"),
        ),
        prev_eval.MethodSpec(
            name="cropenc_baseline_best",
            run_name="stage2_fullscale_core_cropenc_seed456_20260409",
            method_type="stage2",
            checkpoint_path=str(ROOT / "outputs/checkpoints/stage2_fullscale_core_cropenc_seed456_20260409/best.pt"),
        ),
        prev_eval.MethodSpec(
            name="current_calibration_only_best",
            run_name="stage2_calonly_topk1_seed123_wave1_20260413",
            method_type="stage2",
            checkpoint_path=str(ROOT / "outputs/checkpoints/stage2_calonly_topk1_seed123_wave1_20260413/best.pt"),
        ),
        prev_eval.MethodSpec(
            name="current_tusb_lite_best",
            run_name=_best_tusb_lite_run(args),
            method_type="stage2",
            checkpoint_path=str(ROOT / "outputs/checkpoints" / _best_tusb_lite_run(args) / "best.pt"),
        ),
        prev_eval.MethodSpec(
            name="tusb_v2_best",
            run_name=str(best_tusb_run_name),
            method_type="stage2",
            checkpoint_path=str(ROOT / "outputs/checkpoints" / str(best_tusb_run_name) / "best.pt"),
        ),
        prev_eval.MethodSpec(
            name="no_instance_path",
            run_name=f"stage2_tusb_v2_no_instance_path_seed123_{DATE_TAG}",
            method_type="stage2",
            checkpoint_path=str(ROOT / "outputs/checkpoints" / f"stage2_tusb_v2_no_instance_path_seed123_{DATE_TAG}" / "best.pt"),
        ),
        prev_eval.MethodSpec(
            name="no_teacher_prior",
            run_name=f"stage2_tusb_v2_no_teacher_prior_seed123_{DATE_TAG}",
            method_type="stage2",
            checkpoint_path=str(ROOT / "outputs/checkpoints" / f"stage2_tusb_v2_no_teacher_prior_seed123_{DATE_TAG}" / "best.pt"),
        ),
        prev_eval.MethodSpec(
            name="no_anticollapse",
            run_name=f"stage2_tusb_v2_no_anticollapse_seed123_{DATE_TAG}",
            method_type="stage2",
            checkpoint_path=str(ROOT / "outputs/checkpoints" / f"stage2_tusb_v2_no_anticollapse_seed123_{DATE_TAG}" / "best.pt"),
        ),
    ]
    out: List[prev_eval.MethodSpec] = []
    for spec in methods:
        if Path(spec.checkpoint_path).exists():
            out.append(spec)
    return out


def _evaluate_protocol_v3_for_tusb(args: Any, best_tusb_run_name: str) -> Dict[str, Any]:
    protocol = base._read_json(args.protocol_v3_json)
    items = protocol.get("items", []) if isinstance(protocol.get("items", []), list) else []
    if not hasattr(args, "lease_path") or not str(getattr(args, "lease_path", "")).strip():
        setattr(args, "lease_path", str(getattr(args, "shared_lease_path", ROOT / "reports/stage1_v2_gpu_lease_20260408.json")))
    device, device_info = evalv3._select_eval_device_v3(args)
    specs = _load_eval_methods(args, best_tusb_run_name=best_tusb_run_name)

    prepared_items: List[Tuple[Dict[str, Any], Dict[str, Any], np.ndarray, Dict[str, np.ndarray]]] = []
    per_item: List[Dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        batch, target_future_mask, future_masks = _build_single_item_batch_tusb(item)
        prepared_items.append((item, batch, target_future_mask, future_masks))
        per_item.append(
            {
                "protocol_item_id": str(item.get("protocol_item_id", "")),
                "dataset": str(item.get("dataset", "")),
                "clip_id": str(item.get("clip_id", "")),
                "subset_tags": list(item.get("subset_tags", [])),
                "target_id": str(item.get("target_id", "")),
                "methods": {},
            }
        )

    try:
        for spec in specs:
            method = prev_eval._load_method(spec, device=device) if spec.name in {"stage1_frozen_baseline", "legacysem_best", "cropenc_baseline_best", "current_calibration_only_best"} else _load_stage2_method_tusb_aware(spec, device=device)
            for item_row, prepared in zip(per_item, prepared_items):
                item, batch, target_future_mask, future_masks = prepared
                item_row["methods"][method.name] = _evaluate_item_tusb_aware(
                    method=method,
                    item=item,
                    batch=batch,
                    target_future_mask=target_future_mask,
                    future_masks=future_masks,
                    device=device,
                )
            _release_method_tusb_aware(method)
    finally:
        lease_id = str(device_info.get("lease_id", "")).strip()
        if lease_id:
            try:
                prev_eval.release_lease(lease_id=lease_id, lease_path=str(args.shared_lease_path))
            except Exception:
                pass
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    panel_names = [
        "full_identifiability_panel",
        "occlusion_reappearance",
        "crossing_ambiguity",
        "small_object",
        "appearance_change",
        "long_gap_persistence",
    ]
    method_rows: List[Dict[str, Any]] = []
    for spec in specs:
        panel_metrics: Dict[str, Any] = {}
        all_rows: List[Dict[str, Any]] = []
        hard_rows: List[Dict[str, Any]] = []
        for item_row in per_item:
            score = (item_row.get("methods") or {}).get(spec.name)
            if not isinstance(score, dict):
                continue
            all_rows.append(score)
            if item_row.get("subset_tags"):
                hard_rows.append(score)
        panel_metrics["full_identifiability_panel"] = prev_eval._aggregate_item_metrics(all_rows)
        panel_metrics["hard_subsets"] = prev_eval._aggregate_item_metrics(hard_rows)
        for panel in panel_names[1:]:
            subset_rows = [
                (item_row.get("methods") or {}).get(spec.name)
                for item_row in per_item
                if panel in list(item_row.get("subset_tags", []))
            ]
            panel_metrics[panel] = prev_eval._aggregate_item_metrics([r for r in subset_rows if isinstance(r, dict)])
        method_rows.append(
            {
                "name": spec.name,
                "run_name": spec.run_name,
                "method_type": spec.method_type,
                "checkpoint_path": spec.checkpoint_path,
                "panels": panel_metrics,
                "query_future_top1_acc": float(panel_metrics["full_identifiability_panel"]["query_future_top1_acc"]),
                "query_future_hit_rate": float(panel_metrics["full_identifiability_panel"]["query_future_hit_rate"]),
                "query_future_localization_error": float(panel_metrics["full_identifiability_panel"]["query_future_localization_error"]),
                "future_mask_iou_at_top1": float(panel_metrics["full_identifiability_panel"]["future_mask_iou_at_top1"]),
                "hard_subset_top1_acc": float(panel_metrics["hard_subsets"]["query_future_top1_acc"]),
                "ambiguous_case_top1_acc": float(panel_metrics["crossing_ambiguity"]["query_future_top1_acc"]),
                "small_object_query_top1_acc": float(panel_metrics["small_object"]["query_future_top1_acc"]),
                "appearance_change_query_top1_acc": float(panel_metrics["appearance_change"]["query_future_top1_acc"]),
            }
        )

    comparisons = {
        "stage1_frozen_baseline": evalv3._paired_comparison_bundle(per_item, "tusb_v2_best", "stage1_frozen_baseline"),
        "legacysem_best": evalv3._paired_comparison_bundle(per_item, "tusb_v2_best", "legacysem_best"),
        "cropenc_baseline_best": evalv3._paired_comparison_bundle(per_item, "tusb_v2_best", "cropenc_baseline_best"),
        "current_calibration_only_best": evalv3._paired_comparison_bundle(per_item, "tusb_v2_best", "current_calibration_only_best"),
        "current_tusb_lite_best": evalv3._paired_comparison_bundle(per_item, "tusb_v2_best", "current_tusb_lite_best"),
        "no_instance_path": evalv3._paired_comparison_bundle(per_item, "tusb_v2_best", "no_instance_path"),
        "no_teacher_prior": evalv3._paired_comparison_bundle(per_item, "tusb_v2_best", "no_teacher_prior"),
        "no_anticollapse": evalv3._paired_comparison_bundle(per_item, "tusb_v2_best", "no_anticollapse"),
    }
    by_name = {row["name"]: row for row in method_rows}
    tusb = by_name.get("tusb_v2_best", {})
    cal = by_name.get("current_calibration_only_best", {})
    improved_vs_cal = bool(
        tusb
        and cal
        and float(tusb.get("query_future_top1_acc", -1.0)) > float(cal.get("query_future_top1_acc", -1.0))
        and float(tusb.get("future_mask_iou_at_top1", -1.0)) >= float(cal.get("future_mask_iou_at_top1", -1.0))
        and float((comparisons["current_calibration_only_best"]["top1_acc"] or {}).get("mean_diff", 0.0)) > 0.0
    )
    hard_improved = bool(
        tusb
        and cal
        and float(tusb.get("hard_subset_top1_acc", -1.0)) > float(cal.get("hard_subset_top1_acc", -1.0))
        and float(tusb.get("ambiguous_case_top1_acc", -1.0)) >= float(cal.get("ambiguous_case_top1_acc", -1.0))
        and float(tusb.get("small_object_query_top1_acc", -1.0)) >= float(cal.get("small_object_query_top1_acc", -1.0))
        and float(tusb.get("appearance_change_query_top1_acc", -1.0)) >= float(cal.get("appearance_change_query_top1_acc", -1.0))
    )
    return {
        "generated_at_utc": now_iso(),
        "protocol_v3_path": str(args.protocol_v3_json),
        "protocol_item_count": int(len(per_item)),
        "selected_device": str(device),
        "device_info": device_info,
        "methods": method_rows,
        "per_item_results": per_item,
        "paired_bootstrap_comparisons": comparisons,
        "improved_vs_current_calonly": bool(improved_vs_cal),
        "hard_subsets_improved": bool(hard_improved),
    }


def diagnose(args: Any) -> Dict[str, Any]:
    summary = summarize(args)
    oral_diag = _json_or_empty(args.oral_hardening_diagnosis)
    rows = [row for row in summary.get("run_rows", []) if isinstance(row, dict)]
    completed = [row for row in rows if str(row.get("status", "")).lower() == "completed" and bool(row.get("scientific_result_valid", False))]
    if not bool(summary.get("all_runs_terminal", False)) or int(summary.get("failed_count", 0)) > 0 or not completed:
        payload = {
            "generated_at_utc": now_iso(),
            "status": "pending_tusb_completion",
            "stage1_frozen_still_correct": True,
            "best_tusb_run_name": str(summary.get("best_tusb_run_name", "none")),
            "protocol_v3_improved_vs_current_calonly": None,
            "hard_subsets_improved": None,
            "z_sem_slower_than_z_dyn": None,
            "assignment_sparse_and_interpretable": None,
            "active_unit_count_mean": None,
            "active_unit_count_mean_significantly_higher_than_tusb_lite": None,
            "instance_aware_real_signal_used": None,
            "paper_thickness_level": "cvpr_borderline",
            "next_step_choice": "keep_tusb_v2_but_refine_teacher_or_unitization",
        }
        base._write_json(args.diagnosis_report, payload)
        _write_results_md(args, summary, payload)
        return payload

    main_rows = [row for row in completed if str(row.get("family", "")) == "tusb_v2_main"]
    best_main = min(main_rows if main_rows else completed, key=base._summary_overall_rank)
    best_tusb_run_name = str(best_main["run_name"])
    protocol_eval = _evaluate_protocol_v3_for_tusb(args, best_tusb_run_name=best_tusb_run_name)

    trace_metrics = best_main.get("trace_unit_metrics", {}) if isinstance(best_main.get("trace_unit_metrics", {}), dict) else {}
    z_dyn_drift = float(trace_metrics.get("z_dyn_drift_mean", 1e9))
    z_sem_drift = float(trace_metrics.get("z_sem_drift_mean", 1e9))
    z_sem_ratio = float(trace_metrics.get("z_sem_to_z_dyn_drift_ratio_mean", trace_metrics.get("z_sem_to_z_dyn_drift_ratio", 1e9)))
    top2_ratio = float(trace_metrics.get("actual_top2_assignment_ratio_mean", trace_metrics.get("actual_top2_assignment_ratio", 0.0)))
    active_units = float(trace_metrics.get("active_unit_count_mean", 0.0))
    entropy = float(trace_metrics.get("assignment_entropy_mean", 0.0))

    z_sem_slower = bool(z_sem_drift < z_dyn_drift and z_sem_ratio < 0.90)
    assignment_interpretable = bool(0.01 < top2_ratio < 0.95 and active_units >= 2.0 and entropy > 0.0)
    instance_report = _json_or_empty(args.instance_data_report)
    vipseg_true = bool((((instance_report.get("dataset_instance_awareness") or {}).get("VIPSeg") or {}).get("true_instance_aware", False)))
    instance_signal_used = bool(vipseg_true and float(trace_metrics.get("same_instance_within_unit_consistency_mean", trace_metrics.get("same_instance_within_unit_consistency", 0.0))) > 0.0)
    lite_diag = _json_or_empty(ROOT / "reports/stage2_trace_unit_semantic_binding_diagnosis_20260417.json")
    lite_trace_metrics = {}
    lite_summary = _json_or_empty(ROOT / "reports/stage2_trace_unit_semantic_binding_summary_20260417.json")
    for row in lite_summary.get("run_rows", []) if isinstance(lite_summary.get("run_rows", []), list) else []:
        if isinstance(row, dict) and str(row.get("run_name", "")) == str(lite_diag.get("best_tusb_run_name", "")):
            lite_trace_metrics = row.get("trace_unit_metrics", {}) if isinstance(row.get("trace_unit_metrics", {}), dict) else {}
            break
    lite_active_units = float(lite_trace_metrics.get("active_unit_count_mean", 0.0))
    active_units_significantly_higher = bool(active_units >= max(lite_active_units + 1.5, lite_active_units * 1.8, 3.0))

    method_by_name = {
        str(row.get("name", "")): row
        for row in protocol_eval.get("methods", [])
        if isinstance(row, dict)
    }
    tusb_row = method_by_name.get("tusb_v2_best", {})
    cal_row = method_by_name.get("current_calibration_only_best", {})
    no_instance_row = method_by_name.get("no_instance_path", {})
    no_teacher_row = method_by_name.get("no_teacher_prior", {})
    no_anticollapse_row = method_by_name.get("no_anticollapse", {})

    instance_path_support = bool(
        tusb_row
        and no_instance_row
        and float(tusb_row.get("query_future_top1_acc", -1.0)) >= float(no_instance_row.get("query_future_top1_acc", -1.0))
        and float(tusb_row.get("hard_subset_top1_acc", -1.0)) >= float(no_instance_row.get("hard_subset_top1_acc", -1.0))
    )
    teacher_prior_support = bool(
        tusb_row
        and no_teacher_row
        and float(tusb_row.get("query_future_top1_acc", -1.0)) >= float(no_teacher_row.get("query_future_top1_acc", -1.0))
        and float(tusb_row.get("hard_subset_top1_acc", -1.0)) >= float(no_teacher_row.get("hard_subset_top1_acc", -1.0))
    )
    anticollapse_support = bool(
        tusb_row
        and no_anticollapse_row
        and float(tusb_row.get("query_future_top1_acc", -1.0)) >= float(no_anticollapse_row.get("query_future_top1_acc", -1.0))
        and float(tusb_row.get("hard_subset_top1_acc", -1.0)) >= float(no_anticollapse_row.get("hard_subset_top1_acc", -1.0))
        and active_units_significantly_higher
    )
    improved_vs_current_calonly = bool(protocol_eval.get("improved_vs_current_calonly", False))
    hard_improved = bool(protocol_eval.get("hard_subsets_improved", False))

    if improved_vs_current_calonly and hard_improved and z_sem_slower and assignment_interpretable and instance_signal_used and instance_path_support and teacher_prior_support and anticollapse_support:
        next_step = "freeze_tusb_v2_as_new_stage2_mainline"
        paper_level = "cvpr_or_eccv_main_candidate"
    elif improved_vs_current_calonly or hard_improved or z_sem_slower or instance_signal_used or active_units_significantly_higher:
        next_step = "keep_tusb_v2_but_refine_teacher_or_unitization"
        paper_level = "stronger_than_cvpr_borderline_but_not_freeze_ready"
    else:
        next_step = "rethink_stage2_story_if_multi_entity_tusb_still_not_supported"
        paper_level = str(oral_diag.get("current_paper_thickness_level", "cvpr_borderline"))

    anticollapse_payload = {
        "generated_at_utc": now_iso(),
        "best_tusb_v2_run_name": best_tusb_run_name,
        "best_tusb_lite_run_name": str(lite_diag.get("best_tusb_run_name", "none")),
        "active_unit_count_mean_tusb_v2": active_units,
        "active_unit_count_mean_tusb_lite": lite_active_units,
        "active_unit_count_mean_significantly_higher_than_tusb_lite": bool(active_units_significantly_higher),
        "assignment_entropy_mean": entropy,
        "actual_top2_assignment_ratio_mean": top2_ratio,
        "z_dyn_drift_mean": z_dyn_drift,
        "z_sem_drift_mean": z_sem_drift,
        "z_sem_to_z_dyn_drift_ratio": z_sem_ratio,
        "same_instance_within_unit_consistency": float(trace_metrics.get("same_instance_within_unit_consistency_mean", trace_metrics.get("same_instance_within_unit_consistency", 0.0))),
        "different_instance_between_unit_separation": float(trace_metrics.get("different_instance_between_unit_separation_mean", trace_metrics.get("different_instance_between_unit_separation", 0.0))),
        "unit_semantic_stability_over_time": float(trace_metrics.get("unit_semantic_stability_over_time_mean", trace_metrics.get("unit_semantic_stability_over_time", 0.0))),
    }
    base._write_json(args.anticollapse_report, anticollapse_payload)
    base._write_md(
        args.anticollapse_doc,
        [
            "# Stage2 TUSB Anticollapse 20260418",
            "",
            f"- best_tusb_v2_run_name: {anticollapse_payload['best_tusb_v2_run_name']}",
            f"- best_tusb_lite_run_name: {anticollapse_payload['best_tusb_lite_run_name']}",
            f"- active_unit_count_mean_tusb_v2: {anticollapse_payload['active_unit_count_mean_tusb_v2']:.4f}",
            f"- active_unit_count_mean_tusb_lite: {anticollapse_payload['active_unit_count_mean_tusb_lite']:.4f}",
            f"- active_unit_count_mean_significantly_higher_than_tusb_lite: {anticollapse_payload['active_unit_count_mean_significantly_higher_than_tusb_lite']}",
            f"- assignment_entropy_mean: {anticollapse_payload['assignment_entropy_mean']:.6f}",
            f"- actual_top2_assignment_ratio_mean: {anticollapse_payload['actual_top2_assignment_ratio_mean']:.6f}",
            f"- z_dyn_drift_mean: {anticollapse_payload['z_dyn_drift_mean']:.6f}",
            f"- z_sem_drift_mean: {anticollapse_payload['z_sem_drift_mean']:.6f}",
            f"- z_sem_to_z_dyn_drift_ratio: {anticollapse_payload['z_sem_to_z_dyn_drift_ratio']:.6f}",
        ],
    )

    payload = {
        "generated_at_utc": now_iso(),
        "status": "completed",
        "stage1_frozen_still_correct": True,
        "best_tusb_run_name": best_tusb_run_name,
        "protocol_v3_evaluation": protocol_eval,
        "protocol_v3_improved_vs_current_calonly": bool(improved_vs_current_calonly),
        "hard_subsets_improved": bool(hard_improved),
        "z_sem_slower_than_z_dyn": bool(z_sem_slower),
        "assignment_sparse_and_interpretable": bool(assignment_interpretable),
        "active_unit_count_mean": active_units,
        "active_unit_count_mean_significantly_higher_than_tusb_lite": bool(active_units_significantly_higher),
        "instance_aware_real_signal_used": bool(instance_signal_used),
        "instance_path_supports_design": bool(instance_path_support),
        "teacher_prior_supports_design": bool(teacher_prior_support),
        "anticollapse_supports_design": bool(anticollapse_support),
        "paper_thickness_level": paper_level,
        "next_step_choice": next_step,
    }
    base._write_json(args.diagnosis_report, payload)
    _write_results_md(args, summary, payload)
    return payload


def _write_results_md(args: Any, summary: Dict[str, Any], diagnosis: Dict[str, Any]) -> None:
    lines = [
        "# Stage2 TUSB-V2 20260418",
        "",
        f"- generated_at_utc: {now_iso()}",
        f"- tusb_status: {summary.get('tusb_status', 'unknown')}",
        f"- best_tusb_run_name: {diagnosis.get('best_tusb_run_name', 'none')}",
        f"- protocol_v3_improved_vs_current_calonly: {diagnosis.get('protocol_v3_improved_vs_current_calonly', None)}",
        f"- hard_subsets_improved: {diagnosis.get('hard_subsets_improved', None)}",
        f"- z_sem_slower_than_z_dyn: {diagnosis.get('z_sem_slower_than_z_dyn', None)}",
        f"- assignment_sparse_and_interpretable: {diagnosis.get('assignment_sparse_and_interpretable', None)}",
        f"- active_unit_count_mean_significantly_higher_than_tusb_lite: {diagnosis.get('active_unit_count_mean_significantly_higher_than_tusb_lite', None)}",
        f"- instance_aware_real_signal_used: {diagnosis.get('instance_aware_real_signal_used', None)}",
        f"- next_step_choice: {diagnosis.get('next_step_choice', 'none')}",
        "",
        "| run_name | family | seed | status | endpoint_l2 | hard_score | assign_entropy | top2_ratio | active_units | z_dyn_drift | z_sem_drift |",
        "|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary.get("run_rows", []) if isinstance(summary.get("run_rows", []), list) else []:
        if not isinstance(row, dict):
            continue
        valid = bool(row.get("scientific_result_valid", False))
        endpoint = base._metric_rank_tuple(row.get("best_checkpoint_metric", {}))[0] if valid else None
        hard_score = (
            float((row.get("semantic_hard_sidecar_metric", {}) if isinstance(row.get("semantic_hard_sidecar_metric", {}), dict) else {}).get("semantic_hard_sidecar_score", 0.0))
            if valid
            else None
        )
        trace_block = row.get("trace_unit_metrics", {}) if isinstance(row.get("trace_unit_metrics", {}), dict) else {}
        assign_entropy = f"{float(trace_block.get('assignment_entropy_mean', 0.0)):.4f}" if trace_block else "n/a"
        top2_ratio = (
            f"{float(trace_block.get('actual_top2_assignment_ratio_mean', trace_block.get('actual_top2_assignment_ratio', 0.0))):.4f}"
            if trace_block else "n/a"
        )
        active_units = f"{float(trace_block.get('active_unit_count_mean', 0.0)):.2f}" if trace_block else "n/a"
        z_dyn_drift = f"{float(trace_block.get('z_dyn_drift_mean', 0.0)):.4f}" if trace_block else "n/a"
        z_sem_drift = f"{float(trace_block.get('z_sem_drift_mean', 0.0)):.4f}" if trace_block else "n/a"
        lines.append(
            f"| {row.get('run_name', '')} | {row.get('family', '')} | {int(row.get('seed', -1))} | {row.get('status', '')} | "
            f"{(f'{endpoint:.6f}' if endpoint is not None else 'n/a')} | "
            f"{(f'{hard_score:.6f}' if hard_score is not None else 'n/a')} | "
            f"{assign_entropy} | {top2_ratio} | {active_units} | {z_dyn_drift} | {z_sem_drift} |"
        )
    protocol_eval = diagnosis.get("protocol_v3_evaluation", {}) if isinstance(diagnosis.get("protocol_v3_evaluation", {}), dict) else {}
    methods = protocol_eval.get("methods", []) if isinstance(protocol_eval.get("methods", []), list) else []
    if methods:
        lines.extend(
            [
                "",
                "## Protocol V3 Comparison",
                "",
                "| method | run_name | top1_acc | hit_rate | loc_error | mask_iou | hard_top1 | ambiguity_top1 | small_top1 | appearance_top1 |",
                "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in methods:
            if not isinstance(row, dict):
                continue
            lines.append(
                f"| {row.get('name', '')} | {row.get('run_name', '')} | "
                f"{float(row.get('query_future_top1_acc', 0.0)):.4f} | {float(row.get('query_future_hit_rate', 0.0)):.4f} | "
                f"{float(row.get('query_future_localization_error', 0.0)):.6f} | {float(row.get('future_mask_iou_at_top1', 0.0)):.4f} | "
                f"{float(row.get('hard_subset_top1_acc', 0.0)):.4f} | {float(row.get('ambiguous_case_top1_acc', 0.0)):.4f} | "
                f"{float(row.get('small_object_query_top1_acc', 0.0)):.4f} | {float(row.get('appearance_change_query_top1_acc', 0.0)):.4f} |"
            )
    base._write_md(args.results_md, lines)


def run_all(args: Any) -> Dict[str, Any]:
    launch(args)
    summary = wait_for_completion(args)
    return diagnose(args) if bool(summary.get("all_runs_terminal", False)) else summarize(args)


def parse_args() -> Any:
    parser = ArgumentParser(description="Run Stage2 TUSB-V2 repair-and-deepen pilot")
    parser.add_argument("--mode", default="run", choices=["run", "launch", "run-one", "summarize", "diagnose"])
    parser.add_argument("--work-root", default=str(ROOT))
    parser.add_argument("--tmux-session", default=SESSION)
    parser.add_argument("--python-bin", default=str(base._python_bin_default()))
    parser.add_argument("--stage2-contract-json", default=str(ROOT / "reports/stage2_bootstrap_data_contract_20260408.json"))
    parser.add_argument("--stage1-best-ckpt", default=str(ROOT / "outputs/checkpoints/stage1_v2_longtrain_220m_mainline_20260408/best.pt"))
    parser.add_argument("--shared-lease-path", default=str(ROOT / "reports/stage1_v2_gpu_lease_20260408.json"))
    parser.add_argument("--bootstrap-cache-jsonl", default=str(ROOT / "data/processed/stage2_real_bootstrap_cache_20260410/clip_vit_b32_core_trainval_required_subset.jsonl"))
    parser.add_argument("--semantic-hard-manifest-path", default=str(ROOT / "manifests/protocol_v2/stage2_semantic_hard_subsets_20260410.json"))
    parser.add_argument("--runtime-json", default=str(TUSB_RUNTIME_JSON))
    parser.add_argument("--predecode-cache-path", default=str(ROOT / "data/processed/stage2_tusb_v2_predecode_cache_20260418"))
    parser.add_argument("--teacher-semantic-cache-path", default=str(ROOT / "data/processed/stage2_teacher_semantic_cache_v2_20260418"))
    parser.add_argument("--protocol-v3-json", default=str(ROOT / "reports/stage2_state_identifiability_protocol_v3_20260416.json"))
    parser.add_argument("--oral-hardening-diagnosis", default=str(ROOT / "reports/stage2_oral_hardening_diagnosis_20260416.json"))
    parser.add_argument("--protocol-report", default=str(ROOT / "reports/stage2_tusb_v2_repair_protocol_20260418.json"))
    parser.add_argument("--protocol-doc", default=str(ROOT / "docs/STAGE2_TUSB_V2_REPAIR_PROTOCOL_20260418.md"))
    parser.add_argument("--instance-data-report", default=str(ROOT / "reports/stage2_multi_entity_tusb_data_20260418.json"))
    parser.add_argument("--instance-data-doc", default=str(ROOT / "docs/STAGE2_MULTI_ENTITY_TUSB_DATA_20260418.md"))
    parser.add_argument("--cache-health-report", default=str(ROOT / "reports/stage2_tusb_v2_cache_health_20260418.json"))
    parser.add_argument("--cache-health-doc", default=str(ROOT / "docs/STAGE2_TUSB_V2_CACHE_HEALTH_20260418.md"))
    parser.add_argument("--teacher-prior-report", default=str(ROOT / "reports/stage2_tusb_teacher_prior_v2_20260418.json"))
    parser.add_argument("--teacher-prior-doc", default=str(ROOT / "docs/STAGE2_TUSB_TEACHER_PRIOR_V2_20260418.md"))
    parser.add_argument("--eval-liveness-report", default=str(ROOT / "reports/stage2_tusb_eval_liveness_20260418.json"))
    parser.add_argument("--eval-liveness-doc", default=str(ROOT / "docs/STAGE2_TUSB_EVAL_LIVENESS_20260418.md"))
    parser.add_argument("--anticollapse-report", default=str(ROOT / "reports/stage2_tusb_anticollapse_20260418.json"))
    parser.add_argument("--anticollapse-doc", default=str(ROOT / "docs/STAGE2_TUSB_ANTICOLLAPSE_20260418.md"))
    parser.add_argument("--param-budget-report", default=str(ROOT / "reports/stage2_trace_unit_param_budget_20260417.json"))
    parser.add_argument("--param-budget-doc", default=str(ROOT / "docs/STAGE2_TRACE_UNIT_PARAM_BUDGET_20260417.md"))
    parser.add_argument("--launch-report", default=str(ROOT / "reports/stage2_tusb_v2_launch_20260418.json"))
    parser.add_argument("--summary-report", default=str(ROOT / "reports/stage2_tusb_v2_summary_20260418.json"))
    parser.add_argument("--diagnosis-report", default=str(ROOT / "reports/stage2_tusb_v2_diagnosis_20260418.json"))
    parser.add_argument("--results-md", default=str(ROOT / "docs/STAGE2_TUSB_V2_20260418.md"))
    parser.add_argument("--meta-json", default="")
    parser.add_argument("--gpu-acquire-timeout-seconds", type=int, default=28800)
    parser.add_argument("--gpu-acquire-retry-seconds", type=int, default=20)
    parser.add_argument("--wait-timeout-seconds", type=int, default=172800)
    parser.add_argument("--poll-seconds", type=int, default=120)
    parser.add_argument("--max-concurrent-tusb-tasks", type=int, default=MAX_CONCURRENT_TUSB_TASKS)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--eval-required-mem-gb", type=float, default=40.0)
    parser.add_argument("--eval-safety-margin-gb", type=float, default=8.0)
    return parser.parse_args()


def main() -> None:
    base._apply_process_title_normalization()
    args = parse_args()
    if args.mode == "run":
        print(json.dumps(run_all(args), ensure_ascii=True, indent=2))
    elif args.mode == "launch":
        print(json.dumps(launch(args), ensure_ascii=True, indent=2))
    elif args.mode == "run-one":
        run_one(args)
    elif args.mode == "summarize":
        print(json.dumps(summarize(args), ensure_ascii=True, indent=2))
    elif args.mode == "diagnose":
        print(json.dumps(diagnose(args), ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
