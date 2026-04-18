#!/usr/bin/env python3
from __future__ import annotations

from argparse import ArgumentParser
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple
import gc
import json
import subprocess
import time

import numpy as np
import torch

from stwm.tools import run_stage2_state_identifiability_eval_20260415 as prev_eval
from stwm.tools import run_stage2_state_identifiability_eval_v3_20260416 as evalv3
from stwm.tools import run_stage2_tusb_v2_20260418 as tusbbase
from stwm.tools import run_stage2_tusb_v2_context_aligned_20260418 as ctx
from stwm.tools import run_tracewm_stage2_calibration_only_fullscale_wave1_20260413 as base


ROOT = prev_eval.ROOT
SESSION = "tracewm_stage2_tusb_v3_identity_binding_20260418"
DATE_TAG = "20260418"
LOG_PATH = ROOT / "logs/stage2_tusb_v3_identity_binding_20260418.log"
TRAIN_ADDITIONAL_STEPS = 800
EVAL_INTERVAL = 100
SAVE_EVERY = 100
MAX_TRAIN_TASKS = 4
MAX_CACHE_TASKS = 2
K_CONTEXT = 8
TUSB_RUNTIME_JSON = ROOT / "configs/recommended_stage2_runtime_tusb_v2_20260418.json"
PREDECODE_CACHE_ROOT = ROOT / "data/processed/stage2_tusb_v3_predecode_cache_20260418"
TEACHER_CACHE_ROOT = ROOT / "data/processed/stage2_teacher_semantic_cache_v3_20260418"


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


def _write_protocol_artifacts(args: Any) -> Dict[str, Any]:
    v2_diag = _json_or_empty(ROOT / "reports/stage2_tusb_v2_diagnosis_20260418.json")
    ctx_diag = _json_or_empty(ROOT / "reports/stage2_tusb_v2_context_aligned_diagnosis_20260418.json")
    payload = {
        "generated_at_utc": now_iso(),
        "stage1_backbone": {
            "frozen": True,
            "training_allowed": False,
            "unfreeze_allowed": False,
            "swap_backbone_allowed": False,
        },
        "current_tusb_v2_state": {
            "already_landed": True,
            "anti_collapse_load_bearing": True,
            "z_sem_slower_than_z_dyn": True,
            "multi_entity_data_path_landed": True,
            "cache_landed": True,
            "context_preserving_eval_landed": True,
            "freeze_ready_mainline": False,
        },
        "core_unresolved_issue": {
            "instance_id_in_dataset_and_batch": True,
            "instance_id_used_as_real_training_supervision": False,
            "protocol_flat_cannot_be_primarily_blamed_on_eval_mismatch": True,
            "context_preserving_eval_still_flat": bool(not ctx_diag.get("context_preserving_protocol_improved_vs_current_calonly", False)),
            "goal": "turn true instance identity into explicit trace-unit binding supervision",
        },
        "current_truth": {
            "anti_collapse_load_bearing": True,
            "z_sem_slower_than_z_dyn": True,
            "instance_aware_real_signal_used": bool(ctx_diag.get("instance_aware_real_signal_used", False)),
            "next_step_choice_before_v3": str(ctx_diag.get("next_step_choice", "keep_tusb_v2_but_refine_teacher_or_protocol_alignment")),
            "current_stage2_ready_to_freeze": False,
        },
    }
    base._write_json(args.protocol_report, payload)
    base._write_md(
        args.protocol_doc,
        [
            "# Stage2 TUSB-V3 Identity-Binding Protocol 20260418",
            "",
            "- Stage1 remains frozen. No training, no unfreeze, no backbone swap.",
            "- Current TUSB-v2 already landed.",
            "- anti-collapse is load-bearing.",
            "- z_sem slower_than_z_dyn = true.",
            "- multi-entity data path, cache, and context-preserving eval already exist.",
            "- core unresolved issue: semantic_instance_id_* reaches dataset/batch, but current training mostly does not use true instance identity as supervision.",
            "- current protocol flatness is no longer primarily attributable to eval mismatch; context-preserving eval exists and remains flat.",
            "- this round only repairs identity binding. No protocol v4, no persistence, no Stage1 edits, no calibration-only micro-fix.",
        ],
    )
    return payload


def _run_subprocess(cmd: List[str], work_root: Path) -> None:
    env = dict(**subprocess.os.environ)
    env["STWM_PROC_TITLE"] = "python"
    env["STWM_PROC_TITLE_MODE"] = "generic"
    env["PYTHONUNBUFFERED"] = "1"
    subprocess.run(cmd, cwd=str(work_root), env=env, check=True)


def _ensure_teacher_smallrefine(args: Any) -> Dict[str, Any]:
    teacher_payload = _json_or_empty(args.teacher_prior_report)
    if not teacher_payload:
        cmd = [
            str(args.python_bin),
            str(ROOT / "code/stwm/tools/build_stage2_teacher_semantic_cache_v3_20260418.py"),
            "--predecode-cache-root",
            str(args.predecode_cache_path),
            "--teacher-cache-root",
            str(TEACHER_CACHE_ROOT),
            "--output-json",
            str(args.teacher_prior_report),
            "--output-md",
            str(args.teacher_prior_doc),
            "--device",
            str(args.eval_device),
        ]
        _append_log(f"teacher_smallrefine_build_start cmd={' '.join(cmd)}")
        _run_subprocess(cmd, ROOT)
        teacher_payload = _json_or_empty(args.teacher_prior_report)
    payload = {
        "generated_at_utc": now_iso(),
        "current_env_blocked_backends": dict(teacher_payload.get("current_env_blocked_backends", {})),
        "chosen_teacher_prior_v3": str(teacher_payload.get("chosen_teacher_prior_v3", "")),
        "teacher_cache_root": str(teacher_payload.get("teacher_cache_root", str(TEACHER_CACHE_ROOT))),
        "teacher_cache_index": str(teacher_payload.get("teacher_cache_index", "")),
        "why_this_is_strongest_available_frozen_prior_in_current_env": str(
            teacher_payload.get("why_this_is_strongest_available_frozen_prior_in_current_env", "")
        ),
        "teacher_is_mainline_semantic_source": False,
        "identity_binding_is_primary_action": True,
    }
    base._write_json(args.teacher_smallrefine_report, payload)
    base._write_md(
        args.teacher_smallrefine_doc,
        [
            "# Stage2 TUSB-V3 Teacher Smallrefine 20260418",
            "",
            f"- chosen_teacher_prior_v3: {payload['chosen_teacher_prior_v3']}",
            f"- teacher_is_mainline_semantic_source: {payload['teacher_is_mainline_semantic_source']}",
            f"- identity_binding_is_primary_action: {payload['identity_binding_is_primary_action']}",
            "",
            "## Blocked Backends",
            "",
            *[f"- {k}: {v}" for k, v in sorted(payload["current_env_blocked_backends"].items())],
        ],
    )
    return payload


def _ensure_predecode_cache(args: Any) -> Dict[str, Any]:
    payload = _json_or_empty(args.cache_health_report)
    if payload:
        return payload
    index_payload = _json_or_empty(Path(args.predecode_cache_path) / "index.json")
    existing_entries = index_payload.get("entries", {}) if isinstance(index_payload.get("entries", {}), dict) else {}
    if existing_entries:
        required_keys = {
            "semantic_instance_id_map",
            "semantic_instance_id_crop",
            "semantic_instance_id_temporal",
            "semantic_instance_valid",
            "semantic_objectness_score",
            "semantic_entity_dominant_instance_id",
            "semantic_entity_instance_overlap_score_over_time",
            "semantic_entity_true_instance_confidence",
            "entity_boxes_over_time",
            "entity_masks_over_time",
        }
        entity_histogram: Dict[str, int] = {}
        instance_source_counts: Dict[str, int] = {}
        per_dataset_cache_compatibility: Dict[str, Dict[str, Any]] = {}
        checked = 0
        missing_hits = 0
        true_instance_samples = 0
        sample_limit = 256
        for cache_path in list(existing_entries.values())[:sample_limit]:
            npz_path = Path(str(cache_path))
            if not npz_path.exists():
                continue
            with np.load(npz_path, allow_pickle=True) as payload_np:
                files = set(payload_np.files)
                meta = dict(payload_np["meta_json"].item())
            missing = sorted(required_keys - files)
            dataset_name = str(meta.get("dataset", "unknown"))
            block = per_dataset_cache_compatibility.setdefault(dataset_name, {"compatible": True, "missing_keys": []})
            if missing:
                missing_hits += 1
                block["compatible"] = False
                block["missing_keys"] = sorted(set(list(block.get("missing_keys", [])) + missing))
            entity_count = int(meta.get("entity_count", 0))
            entity_histogram[str(entity_count)] = entity_histogram.get(str(entity_count), 0) + 1
            source = str(meta.get("instance_source", "unknown"))
            instance_source_counts[source] = instance_source_counts.get(source, 0) + 1
            true_instance_samples += int(bool(meta.get("true_instance_aware", False)))
            checked += 1
        synthetic = {
            "generated_at_utc": now_iso(),
            "cache_root": str(args.predecode_cache_path),
            "index_json": str(Path(args.predecode_cache_path) / "index.json"),
            "dataset_names": ["vspw", "vipseg"],
            "splits": ["train", "val"],
            "semantic_temporal_window": int(index_payload.get("semantic_temporal_window", 5) or 5),
            "semantic_crop_size": int(index_payload.get("semantic_crop_size", 64) or 64),
            "max_entities_per_sample": 8,
            "total_cached_entries": int(len(existing_entries)),
            "newly_written_entries": 0,
            "reused_existing_entries": int(len(existing_entries)),
            "num_workers": 0,
            "prefetch_factor": 0,
            "total_duration_sec": 0.0,
            "aggregate_samples_per_sec": 0.0,
            "split_summaries": [],
            "cache_hit_rate": 1.0 if existing_entries else 0.0,
            "missing_keys_rate": float(missing_hits / max(checked, 1)),
            "fallback_to_raw_decode_ratio": float(missing_hits / max(checked, 1)),
            "per_dataset_cache_compatibility": per_dataset_cache_compatibility,
            "synthetic_from_existing_cache": True,
            "sampled_entry_count_for_health_check": int(checked),
        }
        base._write_json(args.cache_health_report, synthetic)
        base._write_md(
            args.cache_health_doc,
            [
                "# Stage2 TUSB-V3 Cache Health 20260418",
                "",
                "- synthetic_from_existing_cache: true",
                f"- total_cached_entries: {synthetic['total_cached_entries']}",
                f"- sampled_entry_count_for_health_check: {synthetic['sampled_entry_count_for_health_check']}",
                f"- missing_keys_rate: {synthetic['missing_keys_rate']:.4f}",
                f"- fallback_to_raw_decode_ratio: {synthetic['fallback_to_raw_decode_ratio']:.4f}",
            ],
        )
        multi_entity_report = {
            "generated_at_utc": now_iso(),
            "dataset_names": ["vspw", "vipseg"],
            "max_entities_per_sample": 8,
            "sample_count": int(checked),
            "entity_count_histogram": entity_histogram,
            "multi_entity_sample_coverage_ratio": float(sum(v for k, v in entity_histogram.items() if int(k) >= 2) / max(checked, 1)),
            "vipseg_true_instance_continuity_coverage_ratio": float(true_instance_samples / max(checked, 1)),
            "instance_source_counts": instance_source_counts,
            "dataset_instance_awareness": {
                "VIPSeg": {"true_instance_aware": True, "mode": "panoptic_instance_id"},
                "VSPW": {"true_instance_aware": False, "mode": "pseudo_or_null_component"},
            },
            "synthetic_from_existing_cache": True,
        }
        base._write_json(args.multi_entity_report, multi_entity_report)
        base._write_md(
            args.multi_entity_doc,
            [
                "# Stage2 TUSB-V3 Multi-Entity Data 20260418",
                "",
                "- synthetic_from_existing_cache: true",
                f"- sample_count: {multi_entity_report['sample_count']}",
                f"- multi_entity_sample_coverage_ratio: {multi_entity_report['multi_entity_sample_coverage_ratio']:.4f}",
                f"- vipseg_true_instance_continuity_coverage_ratio: {multi_entity_report['vipseg_true_instance_continuity_coverage_ratio']:.4f}",
            ],
        )
        return synthetic
    cmd = [
        str(args.python_bin),
        str(ROOT / "code/stwm/tools/build_stage2_predecode_cache_20260418.py"),
        "--contract-json",
        str(args.stage2_contract_json),
        "--cache-root",
        str(args.predecode_cache_path),
        "--output-json",
        str(args.cache_health_report),
        "--output-md",
        str(args.cache_health_doc),
        "--multi-entity-report",
        str(args.multi_entity_report),
        "--multi-entity-doc",
        str(args.multi_entity_doc),
        "--dataset-names",
        "vspw",
        "vipseg",
        "--splits",
        "train",
        "val",
        "--semantic-crop-size",
        "64",
        "--semantic-temporal-window",
        "5",
        "--max-entities-per-sample",
        "8",
        "--max-samples-per-dataset",
        "64",
        "--num-workers",
        "4",
        "--prefetch-factor",
        "2",
    ]
    _append_log(f"predecode_cache_v3_start cmd={' '.join(cmd)}")
    _run_subprocess(cmd, ROOT)
    return _json_or_empty(args.cache_health_report)


def _write_instance_binding_data_report(args: Any) -> Dict[str, Any]:
    index_payload = _json_or_empty(Path(args.predecode_cache_path) / "index.json")
    entries = index_payload.get("entries", {}) if isinstance(index_payload.get("entries", {}), dict) else {}
    same_pair_count = 0
    different_pair_count = 0
    dominant_cov = []
    noisy_ratio = []
    sample_true_ratio = []
    vipseg_true_ratio = []
    for cache_path in entries.values():
        npz_path = Path(str(cache_path))
        if not npz_path.exists():
            continue
        with np.load(npz_path, allow_pickle=True) as payload:
            meta = payload["meta_json"].item()
            dominant_ids = np.asarray(payload["semantic_entity_dominant_instance_id"], dtype=np.int64)
            conf = np.asarray(payload["semantic_entity_true_instance_confidence"], dtype=np.float32)
            temporal_valid = np.asarray(payload["semantic_instance_valid"], dtype=bool)
        confident = (dominant_ids > 0) & (conf >= 0.6)
        dominant_cov.append(float(np.mean(dominant_ids > 0)))
        noisy_ratio.append(float(np.mean((conf > 0.0) & (conf < 0.6))))
        sample_true_ratio.append(float(np.mean(confident)))
        if str(meta.get("dataset", "")).strip().upper() == "VIPSEG":
            vipseg_true_ratio.append(float(np.mean(confident)))
        for ent_idx in range(int(dominant_ids.shape[0])):
            if not bool(confident[ent_idx]):
                continue
            same_pair_count += max(int(np.sum(temporal_valid[ent_idx])) - 1, 0)
        confident_ids = [int(x) for x in dominant_ids[confident].tolist()]
        for i in range(len(confident_ids)):
            for j in range(i + 1, len(confident_ids)):
                if confident_ids[i] != confident_ids[j]:
                    different_pair_count += 1
    payload = {
        "generated_at_utc": now_iso(),
        "predecode_cache_root": str(args.predecode_cache_path),
        "same_instance_pair_count": int(same_pair_count),
        "different_instance_pair_count": int(different_pair_count),
        "dominant_instance_id_coverage": float(sum(dominant_cov) / max(len(dominant_cov), 1)),
        "noisy_or_ambiguous_pair_ratio": float(sum(noisy_ratio) / max(len(noisy_ratio), 1)),
        "true_instance_ratio_per_batch_proxy": float(sum(sample_true_ratio) / max(len(sample_true_ratio), 1)),
        "vipseg_true_instance_ratio_proxy": float(sum(vipseg_true_ratio) / max(len(vipseg_true_ratio), 1)),
    }
    base._write_json(args.instance_binding_data_report, payload)
    base._write_md(
        args.instance_binding_data_doc,
        [
            "# Stage2 TUSB-V3 Instance Binding Data 20260418",
            "",
            f"- same_instance_pair_count: {payload['same_instance_pair_count']}",
            f"- different_instance_pair_count: {payload['different_instance_pair_count']}",
            f"- dominant_instance_id_coverage: {payload['dominant_instance_id_coverage']:.4f}",
            f"- noisy_or_ambiguous_pair_ratio: {payload['noisy_or_ambiguous_pair_ratio']:.4f}",
            f"- true_instance_ratio_per_batch_proxy: {payload['true_instance_ratio_per_batch_proxy']:.4f}",
            f"- vipseg_true_instance_ratio_proxy: {payload['vipseg_true_instance_ratio_proxy']:.4f}",
        ],
    )
    return payload


def _resume_ckpt(run_name: str) -> Path:
    ckpt = ROOT / "outputs/checkpoints" / run_name / "best.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"missing resume checkpoint: {ckpt}")
    return ckpt


def _run_specs() -> List[Dict[str, Any]]:
    common = {
        "stage2_structure_mode": "trace_unit_semantic_binding",
        "trace_unit_count": 16,
        "trace_unit_dim": 384,
        "trace_unit_slot_iters": 3,
        "trace_unit_assignment_topk": 2,
        "trace_unit_assignment_temperature": 0.70,
        "trace_unit_use_instance_prior_bias": True,
        "trace_unit_disable_instance_path": False,
        "trace_unit_teacher_prior_dim": 512,
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
        "trace_unit_instance_consistency_weight": 0.0,
        "trace_unit_instance_binding_weight": 0.10,
        "trace_unit_interinstance_repulse_weight": 0.05,
        "trace_unit_unit_purity_weight": 0.03,
        "trace_unit_instance_id_source": "dominant_id",
        "trace_unit_instance_conf_threshold": 0.6,
        "trace_unit_dynsem_decorrelation_weight": 0.005,
        "trace_unit_utilization_weight": 0.06,
        "trace_unit_min_active_target": 4.0,
        "trace_unit_diversity_weight": 0.03,
        "trace_unit_top2_floor_weight": 0.03,
        "trace_unit_top2_mass_floor": 0.15,
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
        "max_entities_per_sample": 8,
        "local_temporal_window": 1,
        "local_temporal_fuse_weight": 0.0,
        "teacher_semantic_cache_path": str(TEACHER_CACHE_ROOT),
    }
    return [
        {**common, "run_name": f"stage2_tusb_v3_seed123_{DATE_TAG}", "seed": 123, "family": "tusb_v3_main", "ablation_name": "main", "objective_combo": "tusb_v3_seed123", "objective_family": "trace_unit_semantic_binding_v3", "window_name": "tusbv3_s123", "dataset_names": ["vspw", "vipseg"], "resume_from": str(_resume_ckpt("stage2_tusb_v2_seed123_20260418"))},
        {**common, "run_name": f"stage2_tusb_v3_seed42_{DATE_TAG}", "seed": 42, "family": "tusb_v3_main", "ablation_name": "main", "objective_combo": "tusb_v3_seed42", "objective_family": "trace_unit_semantic_binding_v3", "window_name": "tusbv3_s42", "dataset_names": ["vspw", "vipseg"], "resume_from": str(_resume_ckpt("stage2_tusb_v2_seed42_20260418"))},
        {**common, "run_name": f"stage2_tusb_v3_seed456_{DATE_TAG}", "seed": 456, "family": "tusb_v3_main", "ablation_name": "main", "objective_combo": "tusb_v3_seed456", "objective_family": "trace_unit_semantic_binding_v3", "window_name": "tusbv3_s456", "dataset_names": ["vspw", "vipseg"], "resume_from": str(_resume_ckpt("stage2_tusb_v2_seed456_20260418"))},
        {**common, "run_name": f"stage2_tusb_v3_no_identity_binding_seed123_{DATE_TAG}", "seed": 123, "family": "tusb_v3_ablation", "ablation_name": "no_identity_binding", "objective_combo": "tusb_v3_no_identity_binding_seed123", "objective_family": "trace_unit_semantic_binding_v3_ablation", "window_name": "tusbv3_noid", "dataset_names": ["vspw", "vipseg"], "resume_from": str(_resume_ckpt("stage2_tusb_v2_seed123_20260418")), "trace_unit_instance_binding_weight": 0.0, "trace_unit_interinstance_repulse_weight": 0.0, "trace_unit_unit_purity_weight": 0.0},
        {**common, "run_name": f"stage2_tusb_v3_no_interinstance_repulse_seed123_{DATE_TAG}", "seed": 123, "family": "tusb_v3_ablation", "ablation_name": "no_interinstance_repulse", "objective_combo": "tusb_v3_no_interinstance_repulse_seed123", "objective_family": "trace_unit_semantic_binding_v3_ablation", "window_name": "tusbv3_norep", "dataset_names": ["vspw", "vipseg"], "resume_from": str(_resume_ckpt("stage2_tusb_v2_seed123_20260418")), "trace_unit_interinstance_repulse_weight": 0.0},
        {**common, "run_name": f"stage2_tusb_v3_vipseg_upweight_seed123_{DATE_TAG}", "seed": 123, "family": "tusb_v3_density", "ablation_name": "vipseg_upweight", "objective_combo": "tusb_v3_vipseg_upweight_seed123", "objective_family": "trace_unit_semantic_binding_v3_density", "window_name": "tusbv3_upw", "dataset_names": ["vipseg", "vipseg", "vspw"], "resume_from": str(_resume_ckpt("stage2_tusb_v2_ctx_vipseg_upweight_seed123_20260418"))},
        {**common, "run_name": f"stage2_tusb_v3_vipseg_only_seed123_{DATE_TAG}", "seed": 123, "family": "tusb_v3_density", "ablation_name": "vipseg_only", "objective_combo": "tusb_v3_vipseg_only_seed123", "objective_family": "trace_unit_semantic_binding_v3_density", "window_name": "tusbv3_vip", "dataset_names": ["vipseg"], "resume_from": str(_resume_ckpt("stage2_tusb_v2_ctx_vipseg_only_seed123_20260418"))},
    ]


def _meta_dir(args: Any) -> Path:
    return Path(args.work_root) / "reports/stage2_tusb_v3_identity_binding_runs_20260418"


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


def _common_launch_context(args: Any) -> Dict[str, Any]:
    lease_cleanup = base._cleanup_stale_leases(str(args.shared_lease_path), allowed_prefixes=("stage2_tusb_v3_",))
    if subprocess.run(["tmux", "has-session", "-t", str(args.tmux_session)], capture_output=True).returncode != 0:
        subprocess.run(["tmux", "new-session", "-d", "-s", str(args.tmux_session), "bash"], check=True)
    existing_windows = set(base._tmux_windows(str(args.tmux_session)))
    anchor_args = base._load_ckpt_args(_resume_ckpt("stage2_tusb_v2_seed123_20260418"))
    meta_dir = _meta_dir(args)
    meta_dir.mkdir(parents=True, exist_ok=True)
    return {
        "lease_cleanup": lease_cleanup,
        "existing_windows": existing_windows,
        "obs_len": int(anchor_args.get("obs_len", 8) or 8),
        "fut_len": int(anchor_args.get("fut_len", 8) or 8),
        "max_tokens": int(anchor_args.get("max_tokens", 64) or 64),
        "crop_size": int(anchor_args.get("semantic_crop_size", 64) or 64),
        "meta_dir": meta_dir,
    }


def _build_launch_meta(args: Any, spec: Dict[str, Any], ctx_meta: Dict[str, Any]) -> Dict[str, Any]:
    run_name = str(spec["run_name"])
    resume_from = Path(str(spec["resume_from"]))
    resume_step = base._load_ckpt_step(resume_from)
    cache_compat = tusbbase._instance_cache_compatibility(Path(args.predecode_cache_path))
    train_counts = base._dataset_counts(list(spec["dataset_names"]), "train", args.stage2_contract_json, max_samples=32)
    val_counts = base._dataset_counts(list(spec["dataset_names"]), "val", args.stage2_contract_json, max_samples=32)
    out_dir = Path(args.work_root) / "outputs/checkpoints" / run_name
    meta = {
        **spec,
        "selected_gpu_id": -1,
        "lease_id": "",
        "obs_len": int(ctx_meta["obs_len"]),
        "fut_len": int(ctx_meta["fut_len"]),
        "max_tokens": int(ctx_meta["max_tokens"]),
        "semantic_crop_size": int(ctx_meta["crop_size"]),
        "semantic_source_mainline": "crop_visual_encoder",
        "legacy_semantic_source": "hand_crafted_stats",
        "batch_size": 8,
        "resume_from": str(resume_from),
        "resume_global_step": int(resume_step),
        "additional_train_steps": int(TRAIN_ADDITIONAL_STEPS),
        "train_steps": int(resume_step + TRAIN_ADDITIONAL_STEPS),
        "eval_interval": int(EVAL_INTERVAL),
        "eval_max_batches": 0,
        "save_every_n_steps": int(SAVE_EVERY),
        "max_samples_train": -1,
        "max_samples_val": -1,
        "effective_train_sample_count_per_dataset": train_counts,
        "effective_val_sample_count_per_dataset": val_counts,
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
        "teacher_semantic_cache_path": str(args.teacher_semantic_cache_path),
        "work_root": str(args.work_root),
        "python_bin": str(args.python_bin),
        "worker_pid_file": str(ctx_meta["meta_dir"] / f"{run_name}.pid"),
        "gpu_acquire_timeout_seconds": int(args.gpu_acquire_timeout_seconds),
        "gpu_acquire_retry_seconds": int(args.gpu_acquire_retry_seconds),
        "max_concurrent_tusb_tasks": int(args.max_concurrent_train_tasks),
        "predecode_cache_instance_aware_compatible": bool(cache_compat["compatible"]),
        "predecode_cache_exact_blocking_reason": str(cache_compat["exact_blocking_reason"]),
        "selector_payload": {},
    }
    meta["meta_json"] = str(ctx_meta["meta_dir"] / f"{run_name}_launch_meta.json")
    return meta


def launch(args: Any) -> Dict[str, Any]:
    _write_protocol_artifacts(args)
    _ensure_predecode_cache(args)
    _write_instance_binding_data_report(args)
    _ensure_teacher_smallrefine(args)
    ctx_meta = _common_launch_context(args)
    cleanup_actions: List[Dict[str, Any]] = []
    runs: List[Dict[str, Any]] = []
    for spec in _run_specs():
        meta = _build_launch_meta(args, spec, ctx_meta)
        cleanup_actions.append(base._reset_run_artifacts(args=args, meta=meta, run_name=str(spec["run_name"])))
        runs.append(tusbbase._write_and_launch_meta(args, meta, ctx_meta["existing_windows"]))
    payload = {
        "generated_at_utc": now_iso(),
        "tmux_session": str(args.tmux_session),
        "teacher_backend": "clip_vit-b_16_temporal_weighted_masked_mean_v3",
        "policy": "TUSB-v3 identity-binding repair on frozen Stage1; explicit instance-id supervision; no persistence; max 4 concurrent training tasks",
        "lease_cleanup": ctx_meta["lease_cleanup"],
        "cleanup_actions": cleanup_actions,
        "runs": runs,
    }
    base._write_json(args.launch_report, payload)
    return summarize(args)


def _summary_row_for_run(args: Any, spec: Dict[str, Any]) -> Dict[str, Any]:
    run_name = str(spec["run_name"])
    paths = _paths_for_run(args, run_name)
    progress_payload = _json_or_empty(paths["progress"])
    final_payload = _json_or_empty(paths["final"])
    raw_payload = _json_or_empty(paths["raw"])
    meta = _json_or_empty(paths["launch"])
    status_info = base._status_for(
        {**meta, "window_name": str(meta.get("window_name", spec.get("window_name", ""))), "progress_json": str(paths["progress"]), "final_json": str(paths["final"])},
        session_name=str(args.tmux_session),
    )
    best_block = base._best_block(final_payload, raw_payload, progress_payload)
    latest_block = base._latest_block(final_payload, raw_payload, progress_payload)
    sidecar_block = base._sidecar_block(final_payload, raw_payload, progress_payload)
    trace_block = tusbbase._trace_unit_block(final_payload, raw_payload, progress_payload)
    density_block = final_payload.get("instance_aware_density", {}) if isinstance(final_payload.get("instance_aware_density", {}), dict) else {}
    return {
        "run_name": run_name,
        "family": str(spec["family"]),
        "ablation_name": str(spec["ablation_name"]),
        "status": str(status_info.get("status", "launched")).lower(),
        "dataset_names": [str(x) for x in spec.get("dataset_names", [])],
        "best_checkpoint_metric": best_block,
        "latest_checkpoint_metric": latest_block,
        "semantic_hard_sidecar_metric": sidecar_block,
        "trace_unit_metrics": trace_block,
        "instance_aware_density": density_block,
    }


def summarize(args: Any) -> Dict[str, Any]:
    run_rows = [_summary_row_for_run(args, spec) for spec in _run_specs()]
    running = sum(int(row["status"] == "running") for row in run_rows)
    completed = sum(int(row["status"] == "completed") for row in run_rows)
    failed = sum(int(row["status"] == "failed") for row in run_rows)
    best_main = min(
        [row for row in run_rows if row["family"] == "tusb_v3_main" and row["status"] == "completed"],
        key=lambda row: (
            float(((row.get("best_checkpoint_metric") or {}).get("metrics") or {}).get("free_rollout_endpoint_l2", 1e9)),
            float(((row.get("best_checkpoint_metric") or {}).get("metrics") or {}).get("free_rollout_coord_mean_l2", 1e9)),
        ),
        default={},
    )
    payload = {
        "generated_at_utc": now_iso(),
        "status": f"{running}_running_{completed}_completed_{failed}_failed",
        "running_count": int(running),
        "completed_count": int(completed),
        "failed_count": int(failed),
        "all_runs_terminal": bool(running == 0 and (completed + failed) == len(run_rows)),
        "run_rows": run_rows,
        "best_tusb_v3_run_name": str(best_main.get("run_name", "")),
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


def _protocol_rank(row: Dict[str, Any]) -> Tuple[float, float, float, float, float, float]:
    return (
        -float(row.get("query_future_top1_acc", 0.0)),
        -float(row.get("hard_subset_top1_acc", 0.0)),
        -float(row.get("ambiguity_top1_acc", 0.0)),
        -float(row.get("small_object_top1_acc", 0.0)),
        -float(row.get("appearance_change_top1_acc", 0.0)),
        float(row.get("query_future_localization_error", 1e9)),
    )


def _build_method_specs(summary: Dict[str, Any]) -> List[prev_eval.MethodSpec]:
    best_main = str(summary.get("best_tusb_v3_run_name", "")).strip() or f"stage2_tusb_v3_seed123_{DATE_TAG}"
    specs = [
        prev_eval.MethodSpec(name="stage1_frozen_baseline", run_name="stage1_frozen_baseline", method_type="stage1", checkpoint_path=str(ROOT / "outputs/checkpoints/stage1_v2_longtrain_220m_mainline_20260408/best.pt")),
        prev_eval.MethodSpec(name="legacysem_best", run_name="stage2_fullscale_core_legacysem_seed456_wave2_20260409", method_type="stage2", checkpoint_path=str(ROOT / "outputs/checkpoints/stage2_fullscale_core_legacysem_seed456_wave2_20260409/best.pt")),
        prev_eval.MethodSpec(name="cropenc_baseline_best", run_name="stage2_fullscale_core_cropenc_seed456_20260409", method_type="stage2", checkpoint_path=str(ROOT / "outputs/checkpoints/stage2_fullscale_core_cropenc_seed456_20260409/best.pt")),
        prev_eval.MethodSpec(name="current_calibration_only_best", run_name=ctx._current_calibration_best_run(), method_type="stage2", checkpoint_path=str(ROOT / "outputs/checkpoints" / ctx._current_calibration_best_run() / "best.pt")),
        prev_eval.MethodSpec(name="current_tusb_v2_best_pt", run_name="stage2_tusb_v2_seed123_20260418", method_type="stage2", checkpoint_path=str(ROOT / "outputs/checkpoints/stage2_tusb_v2_seed123_20260418/best.pt")),
        prev_eval.MethodSpec(name="current_tusb_v3_best_pt", run_name=best_main, method_type="stage2", checkpoint_path=str(ROOT / "outputs/checkpoints" / best_main / "best.pt")),
        prev_eval.MethodSpec(name="current_tusb_v3_best_semantic_hard", run_name=best_main, method_type="stage2", checkpoint_path=str(ROOT / "outputs/checkpoints" / best_main / "best_semantic_hard.pt")),
        prev_eval.MethodSpec(name="no_identity_binding", run_name=f"stage2_tusb_v3_no_identity_binding_seed123_{DATE_TAG}", method_type="stage2", checkpoint_path=str(ROOT / f"outputs/checkpoints/stage2_tusb_v3_no_identity_binding_seed123_{DATE_TAG}/best.pt")),
        prev_eval.MethodSpec(name="no_interinstance_repulse", run_name=f"stage2_tusb_v3_no_interinstance_repulse_seed123_{DATE_TAG}", method_type="stage2", checkpoint_path=str(ROOT / f"outputs/checkpoints/stage2_tusb_v3_no_interinstance_repulse_seed123_{DATE_TAG}/best.pt")),
        prev_eval.MethodSpec(name="vipseg_upweight_tusb_v3", run_name=f"stage2_tusb_v3_vipseg_upweight_seed123_{DATE_TAG}", method_type="stage2", checkpoint_path=str(ROOT / f"outputs/checkpoints/stage2_tusb_v3_vipseg_upweight_seed123_{DATE_TAG}/best.pt")),
        prev_eval.MethodSpec(name="vipseg_only_tusb_v3", run_name=f"stage2_tusb_v3_vipseg_only_seed123_{DATE_TAG}", method_type="stage2", checkpoint_path=str(ROOT / f"outputs/checkpoints/stage2_tusb_v3_vipseg_only_seed123_{DATE_TAG}/best.pt")),
    ]
    return [spec for spec in specs if Path(spec.checkpoint_path).exists()]


def _run_checkpoint_protocol_judge(args: Any, summary: Dict[str, Any]) -> Dict[str, Any]:
    protocol = _json_or_empty(args.protocol_v3_json)
    items = protocol.get("items", []) if isinstance(protocol.get("items", []), list) else []
    specs = _build_method_specs(summary)
    if not hasattr(args, "lease_path") or not str(getattr(args, "lease_path", "")).strip():
        setattr(args, "lease_path", str(args.shared_lease_path))
    if not hasattr(args, "device"):
        setattr(args, "device", str(args.eval_device))
    result = ctx._run_eval_mode(
        args=args,
        protocol_items=items,
        specs=specs,
        mode_name="context_preserving",
        builder=lambda item: evalv3._build_context_preserving_item_batch_v3(item, temporal_window=5, max_context_entities=K_CONTEXT),
    )
    methods = ctx._aggregate_eval_methods(result["per_item_results"], specs)
    by_name = {str(row.get("name", "")): row for row in methods}
    best_pt = by_name.get("current_tusb_v3_best_pt", {})
    best_sidecar = by_name.get("current_tusb_v3_best_semantic_hard", {})
    cal = by_name.get("current_calibration_only_best", {})
    chosen_name = "best_semantic_hard.pt" if best_sidecar and _protocol_rank(best_sidecar) < _protocol_rank(best_pt if best_pt else {"query_future_localization_error": 1e9}) else "best.pt"
    chosen = best_sidecar if chosen_name == "best_semantic_hard.pt" else best_pt
    payload = {
        "generated_at_utc": now_iso(),
        "protocol_item_count": int(result.get("protocol_item_count", 0)),
        "protocol_eval_context_entity_count_mean": float(result.get("protocol_eval_context_entity_count_mean", 0.0)),
        "method_rows": methods,
        "best_tusb_v3_checkpoint_choice": chosen_name,
        "context_preserving_protocol_improved_vs_current_calonly": bool(
            chosen and cal and float(chosen.get("query_future_top1_acc", -1.0)) > float(cal.get("query_future_top1_acc", -1.0))
            and float(chosen.get("future_mask_iou_at_top1", -1.0)) >= float(cal.get("future_mask_iou_at_top1", -1.0))
        ),
        "hard_subsets_improved": bool(
            chosen and cal and float(chosen.get("hard_subset_top1_acc", -1.0)) > float(cal.get("hard_subset_top1_acc", -1.0))
            and float(chosen.get("ambiguity_top1_acc", -1.0)) >= float(cal.get("ambiguity_top1_acc", -1.0))
        ),
        "rollout_best_checkpoint": "best.pt",
        "protocol_best_checkpoint": chosen_name,
        "best_semantic_hard_more_aligned_with_protocol": bool(chosen_name == "best_semantic_hard.pt"),
    }
    base._write_json(args.checkpoint_judge_report, payload)
    base._write_md(
        args.checkpoint_judge_doc,
        [
            "# Stage2 TUSB-V3 Checkpoint Protocol Judge 20260418",
            "",
            f"- best_tusb_v3_checkpoint_choice: {payload['best_tusb_v3_checkpoint_choice']}",
            f"- context_preserving_protocol_improved_vs_current_calonly: {payload['context_preserving_protocol_improved_vs_current_calonly']}",
            f"- hard_subsets_improved: {payload['hard_subsets_improved']}",
            f"- best_semantic_hard_more_aligned_with_protocol: {payload['best_semantic_hard_more_aligned_with_protocol']}",
        ],
    )
    return payload


def _write_identity_binding_metrics(args: Any, summary: Dict[str, Any]) -> Dict[str, Any]:
    best_run = str(summary.get("best_tusb_v3_run_name", "")).strip() or f"stage2_tusb_v3_seed123_{DATE_TAG}"
    best_final = _json_or_empty(ROOT / "reports" / f"{best_run}_final.json")
    v2_final = _json_or_empty(ROOT / "reports/stage2_tusb_v2_seed123_20260418_final.json")
    trace = best_final.get("trace_unit_metrics", {}) if isinstance(best_final.get("trace_unit_metrics", {}), dict) else {}
    v2_trace = v2_final.get("trace_unit_metrics", {}) if isinstance(v2_final.get("trace_unit_metrics", {}), dict) else {}
    v2_proxy = float(v2_trace.get("same_instance_within_unit_consistency_mean", v2_trace.get("same_instance_within_unit_consistency", 0.0)) or 0.0)
    match_rate = float(trace.get("same_instance_dominant_unit_match_rate_mean", 0.0) or 0.0)
    payload = {
        "generated_at_utc": now_iso(),
        "best_tusb_v3_run_name": best_run,
        "best_tusb_v2_reference_run_name": "stage2_tusb_v2_seed123_20260418",
        "active_unit_count_mean": float(trace.get("active_unit_count_mean", 0.0) or 0.0),
        "assignment_entropy_mean": float(trace.get("assignment_entropy_mean", 0.0) or 0.0),
        "actual_top2_assignment_ratio_mean": float(trace.get("actual_top2_assignment_ratio_mean", trace.get("actual_top2_assignment_ratio", 0.0)) or 0.0),
        "same_instance_dominant_unit_match_rate_mean": match_rate,
        "same_instance_assignment_cosine_mean": float(trace.get("same_instance_assignment_cosine_mean", 0.0) or 0.0),
        "different_instance_dominant_unit_collision_rate_mean": float(trace.get("different_instance_dominant_unit_collision_rate_mean", 0.0) or 0.0),
        "unit_purity_by_instance_id_mean": float(trace.get("unit_purity_by_instance_id_mean", 0.0) or 0.0),
        "unit_track_stability_over_time_mean": float(trace.get("unit_track_stability_over_time_mean", 0.0) or 0.0),
        "target_entity_to_dominant_unit_consistency_mean": float(trace.get("target_entity_to_dominant_unit_consistency_mean", 0.0) or 0.0),
        "true_instance_ratio_per_batch_mean": float(best_final.get("instance_aware_density", {}).get("true_instance_ratio_per_batch_mean", 0.0) or 0.0),
        "v2_same_instance_proxy_baseline": v2_proxy,
        "same_instance_dominant_unit_match_rate_significantly_improved": bool(match_rate > max(v2_proxy + 0.10, 0.20)),
        "z_sem_slower_than_z_dyn": bool(float(trace.get("z_sem_drift_mean", 1e9)) < float(trace.get("z_dyn_drift_mean", 1e9))),
    }
    base._write_json(args.identity_metrics_report, payload)
    base._write_md(
        args.identity_metrics_doc,
        [
            "# Stage2 TUSB-V3 Identity-Binding Metrics 20260418",
            "",
            f"- best_tusb_v3_run_name: {payload['best_tusb_v3_run_name']}",
            f"- same_instance_dominant_unit_match_rate_mean: {payload['same_instance_dominant_unit_match_rate_mean']:.4f}",
            f"- different_instance_dominant_unit_collision_rate_mean: {payload['different_instance_dominant_unit_collision_rate_mean']:.4f}",
            f"- unit_purity_by_instance_id_mean: {payload['unit_purity_by_instance_id_mean']:.4f}",
            f"- same_instance_dominant_unit_match_rate_significantly_improved: {payload['same_instance_dominant_unit_match_rate_significantly_improved']}",
            f"- z_sem_slower_than_z_dyn: {payload['z_sem_slower_than_z_dyn']}",
        ],
    )
    return payload


def diagnose(args: Any) -> Dict[str, Any]:
    summary = summarize(args)
    judge = _json_or_empty(args.checkpoint_judge_report)
    if not judge:
        judge = _run_checkpoint_protocol_judge(args, summary)
    metrics = _json_or_empty(args.identity_metrics_report)
    if not metrics:
        metrics = _write_identity_binding_metrics(args, summary)
    density = _json_or_empty(args.instance_binding_data_report)
    improved = bool(judge.get("context_preserving_protocol_improved_vs_current_calonly", False))
    hard_improved = bool(judge.get("hard_subsets_improved", False))
    same_instance_improved = bool(metrics.get("same_instance_dominant_unit_match_rate_significantly_improved", False))
    z_sem_slower = bool(metrics.get("z_sem_slower_than_z_dyn", False))
    vipseg_gain_hint = bool(float(density.get("vipseg_true_instance_ratio_proxy", 0.0)) > float(density.get("true_instance_ratio_per_batch_proxy", 0.0)))
    if improved and hard_improved and same_instance_improved and z_sem_slower:
        next_step = "freeze_tusb_v3_as_new_stage2_mainline"
    elif same_instance_improved or vipseg_gain_hint or judge.get("best_semantic_hard_more_aligned_with_protocol", False):
        next_step = "keep_tusb_v3_but_refine_teacher_or_checkpoint_selection"
    else:
        next_step = "rethink_stage2_story_if_identity_binding_still_not_translating_to_protocol_gain"
    payload = {
        "generated_at_utc": now_iso(),
        "best_tusb_v3_run_name": str(summary.get("best_tusb_v3_run_name", "")),
        "best_tusb_v3_checkpoint_choice": str(judge.get("best_tusb_v3_checkpoint_choice", "best.pt")),
        "context_preserving_protocol_improved_vs_current_calonly": bool(improved),
        "hard_subsets_improved": bool(hard_improved),
        "same_instance_dominant_unit_match_rate_significantly_improved": bool(same_instance_improved),
        "z_sem_slower_than_z_dyn": bool(z_sem_slower),
        "next_step_choice": next_step,
    }
    base._write_json(args.diagnosis_report, payload)
    base._write_md(
        args.results_md,
        [
            "# Stage2 TUSB-V3 Identity Binding 20260418",
            "",
            f"- best_tusb_v3_run_name: {payload['best_tusb_v3_run_name']}",
            f"- best_tusb_v3_checkpoint_choice: {payload['best_tusb_v3_checkpoint_choice']}",
            f"- context_preserving_protocol_improved_vs_current_calonly: {payload['context_preserving_protocol_improved_vs_current_calonly']}",
            f"- hard_subsets_improved: {payload['hard_subsets_improved']}",
            f"- same_instance_dominant_unit_match_rate_significantly_improved: {payload['same_instance_dominant_unit_match_rate_significantly_improved']}",
            f"- z_sem_slower_than_z_dyn: {payload['z_sem_slower_than_z_dyn']}",
            f"- next_step_choice: {payload['next_step_choice']}",
        ],
    )
    return payload


def run_all(args: Any) -> Dict[str, Any]:
    _write_protocol_artifacts(args)
    _ensure_predecode_cache(args)
    _write_instance_binding_data_report(args)
    _ensure_teacher_smallrefine(args)
    launch(args)
    summary = wait_for_completion(args)
    if bool(summary.get("all_runs_terminal", False)):
        _run_checkpoint_protocol_judge(args, summary)
        _write_identity_binding_metrics(args, summary)
        diagnose(args)
    return {
        "generated_at_utc": now_iso(),
        "summary_report": str(args.summary_report),
        "diagnosis_report": str(args.diagnosis_report),
    }


def parse_args() -> Any:
    parser = ArgumentParser(description="Run STAGE2 TUSB-V3 identity-binding repair")
    parser.add_argument("--mode", default="run", choices=["run", "launch", "summarize", "diagnose"])
    parser.add_argument("--work-root", default=str(ROOT))
    parser.add_argument("--tmux-session", default=SESSION)
    parser.add_argument("--python-bin", default=str(base._python_bin_default()))
    parser.add_argument("--stage2-contract-json", default=str(ROOT / "reports/stage2_bootstrap_data_contract_20260408.json"))
    parser.add_argument("--stage1-best-ckpt", default=str(ROOT / "outputs/checkpoints/stage1_v2_longtrain_220m_mainline_20260408/best.pt"))
    parser.add_argument("--shared-lease-path", default=str(ROOT / "reports/stage1_v2_gpu_lease_20260408.json"))
    parser.add_argument("--bootstrap-cache-jsonl", default=str(ROOT / "data/processed/stage2_real_bootstrap_cache_20260410/clip_vit_b32_core_trainval_required_subset.jsonl"))
    parser.add_argument("--semantic-hard-manifest-path", default=str(ROOT / "manifests/protocol_v2/stage2_semantic_hard_subsets_20260410.json"))
    parser.add_argument("--runtime-json", default=str(TUSB_RUNTIME_JSON))
    parser.add_argument("--predecode-cache-path", default=str(PREDECODE_CACHE_ROOT))
    parser.add_argument("--teacher-semantic-cache-path", default=str(TEACHER_CACHE_ROOT))
    parser.add_argument("--protocol-v3-json", default=str(ROOT / "reports/stage2_state_identifiability_protocol_v3_20260416.json"))
    parser.add_argument("--eval-device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--protocol-report", default=str(ROOT / "reports/stage2_tusb_v3_identity_binding_protocol_20260418.json"))
    parser.add_argument("--protocol-doc", default=str(ROOT / "docs/STAGE2_TUSB_V3_IDENTITY_BINDING_PROTOCOL_20260418.md"))
    parser.add_argument("--instance-binding-data-report", default=str(ROOT / "reports/stage2_tusb_v3_instance_binding_data_20260418.json"))
    parser.add_argument("--instance-binding-data-doc", default=str(ROOT / "docs/STAGE2_TUSB_V3_INSTANCE_BINDING_DATA_20260418.md"))
    parser.add_argument("--identity-metrics-report", default=str(ROOT / "reports/stage2_tusb_v3_identity_binding_metrics_20260418.json"))
    parser.add_argument("--identity-metrics-doc", default=str(ROOT / "docs/STAGE2_TUSB_V3_IDENTITY_BINDING_METRICS_20260418.md"))
    parser.add_argument("--checkpoint-judge-report", default=str(ROOT / "reports/stage2_tusb_v3_checkpoint_protocol_judge_20260418.json"))
    parser.add_argument("--checkpoint-judge-doc", default=str(ROOT / "docs/STAGE2_TUSB_V3_CHECKPOINT_PROTOCOL_JUDGE_20260418.md"))
    parser.add_argument("--teacher-smallrefine-report", default=str(ROOT / "reports/stage2_tusb_v3_teacher_smallrefine_20260418.json"))
    parser.add_argument("--teacher-smallrefine-doc", default=str(ROOT / "docs/STAGE2_TUSB_V3_TEACHER_SMALLREFINE_20260418.md"))
    parser.add_argument("--launch-report", default=str(ROOT / "reports/stage2_tusb_v3_identity_binding_launch_20260418.json"))
    parser.add_argument("--summary-report", default=str(ROOT / "reports/stage2_tusb_v3_identity_binding_summary_20260418.json"))
    parser.add_argument("--diagnosis-report", default=str(ROOT / "reports/stage2_tusb_v3_identity_binding_diagnosis_20260418.json"))
    parser.add_argument("--results-md", default=str(ROOT / "docs/STAGE2_TUSB_V3_IDENTITY_BINDING_20260418.md"))
    parser.add_argument("--cache-health-report", default=str(ROOT / "reports/stage2_tusb_v3_cache_health_20260418.json"))
    parser.add_argument("--cache-health-doc", default=str(ROOT / "docs/STAGE2_TUSB_V3_CACHE_HEALTH_20260418.md"))
    parser.add_argument("--multi-entity-report", default=str(ROOT / "reports/stage2_tusb_v3_multi_entity_data_20260418.json"))
    parser.add_argument("--multi-entity-doc", default=str(ROOT / "docs/STAGE2_TUSB_V3_MULTI_ENTITY_DATA_20260418.md"))
    parser.add_argument("--teacher-prior-report", default=str(ROOT / "reports/stage2_tusb_teacher_prior_v3_20260418.json"))
    parser.add_argument("--teacher-prior-doc", default=str(ROOT / "docs/STAGE2_TUSB_TEACHER_PRIOR_V3_20260418.md"))
    parser.add_argument("--wait-timeout-seconds", type=int, default=172800)
    parser.add_argument("--poll-seconds", type=int, default=120)
    parser.add_argument("--max-concurrent-train-tasks", type=int, default=MAX_TRAIN_TASKS)
    parser.add_argument("--gpu-acquire-timeout-seconds", type=int, default=7200)
    parser.add_argument("--gpu-acquire-retry-seconds", type=int, default=60)
    parser.add_argument("--lease-path", default=str(ROOT / "reports/stage1_v2_gpu_lease_20260408.json"))
    parser.add_argument("--eval-required-mem-gb", type=float, default=24.0)
    parser.add_argument("--eval-safety-margin-gb", type=float, default=4.0)
    return parser.parse_args()


def main() -> None:
    base._apply_process_title_normalization()
    args = parse_args()
    if args.mode == "run":
        print(json.dumps(run_all(args), ensure_ascii=True, indent=2))
    elif args.mode == "launch":
        print(json.dumps(launch(args), ensure_ascii=True, indent=2))
    elif args.mode == "summarize":
        print(json.dumps(summarize(args), ensure_ascii=True, indent=2))
    elif args.mode == "diagnose":
        print(json.dumps(diagnose(args), ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
