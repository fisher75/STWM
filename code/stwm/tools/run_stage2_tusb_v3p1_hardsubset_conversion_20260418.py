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
from stwm.tools import run_stage2_tusb_v3_identity_binding_20260418 as v3
from stwm.tools import run_tracewm_stage2_calibration_only_fullscale_wave1_20260413 as base


ROOT = prev_eval.ROOT
SESSION = "tracewm_stage2_tusb_v3p1_hardsubset_conversion_20260418"
DATE_TAG = "20260418"
LOG_PATH = ROOT / "logs/stage2_tusb_v3p1_hardsubset_conversion_20260418.log"
TRAIN_ADDITIONAL_STEPS = 200
EVAL_INTERVAL = 100
SAVE_EVERY = 100
MAX_TRAIN_TASKS = 4
K_CONTEXT = 8
PREDECODE_CACHE_ROOT = ROOT / "data/processed/stage2_tusb_v3_predecode_cache_20260418"
TEACHER_CACHE_V3_ROOT = ROOT / "data/processed/stage2_teacher_semantic_cache_v3_20260418"
TEACHER_CACHE_V4_ROOT = ROOT / "data/processed/stage2_teacher_semantic_cache_v4_appearance_20260418"
RUNTIME_JSON = ROOT / "configs/recommended_stage2_runtime_tusb_v2_20260418.json"


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


def _meta_dir(args: Any) -> Path:
    return Path(args.work_root) / "reports/stage2_tusb_v3p1_hardsubset_conversion_runs_20260418"


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


def _existing_checkpoint_candidates(run_name: str) -> List[Tuple[str, Path]]:
    ckpt_dir = ROOT / "outputs/checkpoints" / run_name
    ordered = [
        ("best_semantic_hard.pt", ckpt_dir / "best_semantic_hard.pt"),
        ("best.pt", ckpt_dir / "best.pt"),
        ("latest.pt", ckpt_dir / "latest.pt"),
        ("step_0010900.pt", ckpt_dir / "step_0010900.pt"),
        ("resume_from_10900_repair.pt", ckpt_dir / "resume_from_10900_repair.pt"),
    ]
    return [(label, path) for label, path in ordered if path.exists()]


def _preferred_checkpoint_for_run(run_name: str) -> Tuple[str, Path]:
    candidates = _existing_checkpoint_candidates(run_name)
    if not candidates:
        return ("", Path())
    return candidates[0]


def _write_protocol_artifacts(args: Any) -> Dict[str, Any]:
    v3_diag = _json_or_empty(ROOT / "reports/stage2_tusb_v3_identity_binding_diagnosis_20260418.json")
    judge = _json_or_empty(ROOT / "reports/stage2_tusb_v3_checkpoint_protocol_judge_20260418.json")
    payload = {
        "generated_at_utc": now_iso(),
        "stage1_frozen": True,
        "stage1_train_or_unfreeze_allowed": False,
        "current_tusb_v3_state": {
            "identity_binding_basically_learned": True,
            "same_instance_metrics_strong": True,
            "z_sem_slower_than_z_dyn": True,
            "context_preserving_protocol_positive_vs_current_calonly": bool(
                v3_diag.get("context_preserving_protocol_improved_vs_current_calonly", False)
            ),
            "hard_subsets_still_not_improved": bool(not v3_diag.get("hard_subsets_improved", False)),
        },
        "current_main_contradiction": {
            "identity_gain_not_converted_to_hard_subset_gain": True,
            "occlusion_or_long_gap_already_helped": True,
            "crossing_ambiguity_still_down": True,
            "appearance_change_still_down": True,
            "different_instance_collision_still_high": True,
        },
        "checkpoint_truth": {
            "best_pt_not_sufficient_as_only_development_judge": True,
            "best_semantic_hard_more_aligned_with_protocol": bool(
                judge.get("best_semantic_hard_more_aligned_with_protocol", True)
            ),
        },
        "this_round_goal": {
            "protocol_v4": False,
            "new_unit_architecture": False,
            "priority": [
                "ambiguity-aware separation",
                "appearance-drift-aware teacher refinement",
                "hard-subset curriculum",
                "protocol-aligned checkpoint choice",
            ],
        },
    }
    base._write_json(args.protocol_report, payload)
    base._write_md(
        args.protocol_doc,
        [
            "# Stage2 TUSB-V3.1 Hard-Subset Conversion Protocol 20260418",
            "",
            "- Stage1 remains frozen. No training, no unfreeze, no backbone swap.",
            "- TUSB-v3 already learned identity binding and z_sem slower-than-z_dyn.",
            "- current unresolved issue is not identity entry anymore; it is hard-subset conversion failure.",
            "- occlusion / long-gap already benefit, but crossing ambiguity and appearance change still lag.",
            "- best.pt is no longer sufficient as the only development checkpoint for TUSB-family.",
            "- this round focuses on ambiguity-aware separation, appearance-drift-aware teacher refinement, hard-subset curriculum, and protocol-aligned checkpoint choice.",
        ],
    )
    return payload


def _ensure_teacher_prior_v4(args: Any) -> Dict[str, Any]:
    payload = _json_or_empty(args.appearance_teacher_report)
    if payload:
        return payload
    cmd = [
        str(args.python_bin),
        str(ROOT / "code/stwm/tools/build_stage2_teacher_semantic_cache_v4_appearance_20260418.py"),
        "--predecode-cache-root",
        str(args.predecode_cache_path),
        "--teacher-cache-root",
        str(args.teacher_semantic_cache_path),
        "--output-json",
        str(args.appearance_teacher_report),
        "--output-md",
        str(args.appearance_teacher_doc),
        "--device",
        str(args.eval_device),
    ]
    _append_log(f"teacher_v4_build_start cmd={' '.join(cmd)}")
    v3._run_subprocess(cmd, ROOT)
    return _json_or_empty(args.appearance_teacher_report)


def _pair_iou_xywh_np(box_i: np.ndarray, box_j: np.ndarray) -> float:
    wi = max(float(box_i[2]), 0.0)
    hi = max(float(box_i[3]), 0.0)
    wj = max(float(box_j[2]), 0.0)
    hj = max(float(box_j[3]), 0.0)
    li = float(box_i[0]) - wi * 0.5
    ri = float(box_i[0]) + wi * 0.5
    ti = float(box_i[1]) - hi * 0.5
    bi = float(box_i[1]) + hi * 0.5
    lj = float(box_j[0]) - wj * 0.5
    rj = float(box_j[0]) + wj * 0.5
    tj = float(box_j[1]) - hj * 0.5
    bj = float(box_j[1]) + hj * 0.5
    inter_w = max(min(ri, rj) - max(li, lj), 0.0)
    inter_h = max(min(bi, bj) - max(ti, tj), 0.0)
    inter = inter_w * inter_h
    union = wi * hi + wj * hj - inter
    return float(inter / union) if union > 1e-8 else 0.0


def _sample_curriculum_tags(npz_path: Path) -> Dict[str, bool]:
    with np.load(npz_path, allow_pickle=True) as payload:
        boxes = np.asarray(payload["entity_boxes_over_time"], dtype=np.float32)
        valid = np.asarray(payload["semantic_temporal_valid"], dtype=bool)
        rgb_temporal = np.asarray(payload["semantic_rgb_crop_temporal"], dtype=np.float32)
        mask_temporal = np.asarray(payload["semantic_mask_crop_temporal"], dtype=np.float32)
        meta = dict(payload["meta_json"].item())
    obs_len = int(meta.get("obs_len", min(boxes.shape[0], valid.shape[1]) or 1))
    boxes = boxes[:obs_len]
    ambiguity_risks: List[float] = []
    for t_idx in range(int(min(obs_len, boxes.shape[0]))):
        for i in range(int(valid.shape[0])):
            if not bool(valid[i, min(t_idx, valid.shape[1] - 1)]):
                continue
            for j in range(i + 1, int(valid.shape[0])):
                if not bool(valid[j, min(t_idx, valid.shape[1] - 1)]):
                    continue
                box_i = boxes[t_idx, i]
                box_j = boxes[t_idx, j]
                ci = np.asarray(box_i[:2], dtype=np.float32)
                cj = np.asarray(box_j[:2], dtype=np.float32)
                dist = float(np.linalg.norm(ci - cj))
                dist_risk = max(0.0, 1.0 - dist / 0.25)
                iou = _pair_iou_xywh_np(
                    np.asarray(box_i, dtype=np.float32),
                    np.asarray(box_j, dtype=np.float32),
                )
                ambiguity_risks.append(float(0.5 * dist_risk + 0.5 * iou))
    ambiguity_high = bool(ambiguity_risks and max(ambiguity_risks) >= 0.45)

    drift_vals: List[float] = []
    occlusion_vals: List[float] = []
    long_gap_vals: List[float] = []
    for ent_idx in range(int(valid.shape[0])):
        valid_steps = np.flatnonzero(valid[ent_idx])
        if valid_steps.size >= 2:
            first_idx = int(valid_steps[0])
            last_idx = int(valid_steps[-1])
            span = max(last_idx - first_idx + 1, 1)
            coverage = float(valid_steps.size) / float(span)
            gap_ratio = float(1.0 - coverage)
            occlusion_vals.append(gap_ratio)
            if span >= 3:
                long_gap_vals.append(gap_ratio)
            rgb_sig_early = np.asarray(rgb_temporal[ent_idx, first_idx], dtype=np.float32)
            rgb_sig_late = np.asarray(rgb_temporal[ent_idx, last_idx], dtype=np.float32)
            mask_early = np.asarray(mask_temporal[ent_idx, first_idx], dtype=np.float32)
            mask_late = np.asarray(mask_temporal[ent_idx, last_idx], dtype=np.float32)
            early_vec = (rgb_sig_early * mask_early).reshape(3, -1).sum(axis=-1) / max(float(mask_early.sum()), 1e-6)
            late_vec = (rgb_sig_late * mask_late).reshape(3, -1).sum(axis=-1) / max(float(mask_late.sum()), 1e-6)
            early_norm = float(np.linalg.norm(early_vec))
            late_norm = float(np.linalg.norm(late_vec))
            if early_norm > 1e-6 and late_norm > 1e-6:
                cosine = float(np.clip(np.dot(early_vec / early_norm, late_vec / late_norm), -1.0, 1.0))
                drift_vals.append(1.0 - cosine)
    return {
        "ambiguity_risk_high": bool(ambiguity_high),
        "appearance_drift_high": bool(drift_vals and float(np.mean(drift_vals)) >= 0.15),
        "occlusion_risk_high": bool(occlusion_vals and float(np.mean(occlusion_vals)) >= 0.20),
        "long_gap_like": bool(long_gap_vals and float(np.mean(long_gap_vals)) >= 0.20),
    }


def _write_hardsubset_curriculum(args: Any) -> Dict[str, Any]:
    payload = _json_or_empty(args.hardsubset_curriculum_report)
    if payload:
        return payload
    cache_index = _json_or_empty(Path(args.predecode_cache_path) / "index.json")
    entries = cache_index.get("entries", {}) if isinstance(cache_index.get("entries", {}), dict) else {}
    counts = {
        "ambiguity_risk_high": 0,
        "appearance_drift_high": 0,
        "occlusion_risk_high": 0,
        "long_gap_like": 0,
    }
    checked = 0
    for key, value in list(sorted(entries.items()))[:256]:
        parts = str(key).split("::", 2)
        if len(parts) != 3 or str(parts[1]).lower() != "train":
            continue
        npz_path = Path(str(value))
        if not npz_path.exists():
            continue
        tag_row = _sample_curriculum_tags(npz_path)
        for tag_name, enabled in tag_row.items():
            counts[tag_name] += int(bool(enabled))
        checked += 1
    payload = {
        "generated_at_utc": now_iso(),
        "sampled_train_entry_count": int(checked),
        "hard_tag_ratios": {
            key: float(value / max(checked, 1))
            for key, value in counts.items()
        },
        "curriculum_coefficients": {
            "trace_unit_hardsubset_curriculum_weight": 0.40,
            "trace_unit_hardsubset_ambiguity_weight": 1.20,
            "trace_unit_hardsubset_appearance_weight": 1.00,
            "trace_unit_hardsubset_occlusion_weight": 0.75,
            "trace_unit_hardsubset_longgap_weight": 0.75,
        },
        "designed_to_preserve_true_instance_ratio_and_unit_utilization": True,
    }
    base._write_json(args.hardsubset_curriculum_report, payload)
    base._write_md(
        args.hardsubset_curriculum_doc,
        [
            "# Stage2 TUSB-V3.1 Hard-Subset Curriculum 20260418",
            "",
            f"- sampled_train_entry_count: {payload['sampled_train_entry_count']}",
            *[
                f"- {name}: {ratio:.4f}"
                for name, ratio in sorted(payload["hard_tag_ratios"].items())
            ],
            "",
            "## Curriculum Coefficients",
            "",
            *[
                f"- {name}: {value}"
                for name, value in sorted(payload["curriculum_coefficients"].items())
            ],
        ],
    )
    return payload


def _write_ambiguity_repulse(args: Any) -> Dict[str, Any]:
    payload = {
        "generated_at_utc": now_iso(),
        "trace_unit_ambiguity_repulse_weight": 0.08,
        "trace_unit_ambiguity_risk_threshold": 0.40,
        "trace_unit_ambiguity_min_dist_weight": 0.45,
        "trace_unit_ambiguity_iou_weight": 0.35,
        "trace_unit_ambiguity_motion_cross_weight": 0.20,
        "goal": "lower different-instance collision specifically on high-risk crossing/ambiguity pairs",
    }
    base._write_json(args.ambiguity_repulse_report, payload)
    base._write_md(
        args.ambiguity_repulse_doc,
        [
            "# Stage2 TUSB-V3.1 Ambiguity Repulse 20260418",
            "",
            f"- trace_unit_ambiguity_repulse_weight: {payload['trace_unit_ambiguity_repulse_weight']}",
            f"- trace_unit_ambiguity_risk_threshold: {payload['trace_unit_ambiguity_risk_threshold']}",
            f"- trace_unit_ambiguity_min_dist_weight: {payload['trace_unit_ambiguity_min_dist_weight']}",
            f"- trace_unit_ambiguity_iou_weight: {payload['trace_unit_ambiguity_iou_weight']}",
            f"- trace_unit_ambiguity_motion_cross_weight: {payload['trace_unit_ambiguity_motion_cross_weight']}",
        ],
    )
    return payload


def _write_unitization_smallrefine(args: Any) -> Dict[str, Any]:
    payload = {
        "generated_at_utc": now_iso(),
        "structural_change_applied": False,
        "parameter_budget_changed": False,
        "applied_small_refine": "ambiguity-weighted top2/use-separation supervision only",
    }
    base._write_json(args.unitization_smallrefine_report, payload)
    base._write_md(
        args.unitization_smallrefine_doc,
        [
            "# Stage2 TUSB-V3.1 Unitization Smallrefine 20260418",
            "",
            f"- structural_change_applied: {payload['structural_change_applied']}",
            f"- parameter_budget_changed: {payload['parameter_budget_changed']}",
            f"- applied_small_refine: {payload['applied_small_refine']}",
        ],
    )
    return payload


def _current_v3_best_run_name() -> str:
    diag = _json_or_empty(ROOT / "reports/stage2_tusb_v3_identity_binding_diagnosis_20260418.json")
    return str(diag.get("best_tusb_v3_run_name", "")).strip() or f"stage2_tusb_v3_seed123_{DATE_TAG}"


def _resume_ckpt(run_name: str, *, prefer_semantic_hard: bool = True) -> Path:
    ckpt_dir = ROOT / "outputs/checkpoints" / run_name
    candidates = []
    if prefer_semantic_hard:
        candidates.append(ckpt_dir / "best_semantic_hard.pt")
    candidates.extend([ckpt_dir / "best.pt", ckpt_dir / "latest.pt"])
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"missing resume checkpoint for {run_name}")


def _run_specs() -> List[Dict[str, Any]]:
    template = dict(next(spec for spec in v3._run_specs() if str(spec["run_name"]) == f"stage2_tusb_v3_seed123_{DATE_TAG}"))
    common = {
        k: v
        for k, v in template.items()
        if k
        not in {
            "run_name",
            "seed",
            "family",
            "ablation_name",
            "objective_combo",
            "objective_family",
            "window_name",
            "dataset_names",
            "resume_from",
            "teacher_semantic_cache_path",
        }
    }
    common.update(
        {
            "teacher_semantic_cache_path": str(TEACHER_CACHE_V4_ROOT),
            "trace_unit_ambiguity_repulse_weight": 0.08,
            "trace_unit_ambiguity_risk_threshold": 0.40,
            "trace_unit_ambiguity_min_dist_weight": 0.45,
            "trace_unit_ambiguity_iou_weight": 0.35,
            "trace_unit_ambiguity_motion_cross_weight": 0.20,
            "trace_unit_appearance_refine_weight": 0.06,
            "trace_unit_hardsubset_curriculum_weight": 0.40,
            "trace_unit_hardsubset_ambiguity_weight": 1.20,
            "trace_unit_hardsubset_appearance_weight": 1.00,
            "trace_unit_hardsubset_occlusion_weight": 0.75,
            "trace_unit_hardsubset_longgap_weight": 0.75,
        }
    )
    return [
        {**common, "run_name": f"stage2_tusb_v3p1_seed123_{DATE_TAG}", "seed": 123, "family": "tusb_v3p1_main", "ablation_name": "main", "objective_combo": "tusb_v3p1_seed123", "objective_family": "trace_unit_semantic_binding_v3p1", "window_name": "tusbv31_s123", "dataset_names": ["vspw", "vipseg"], "resume_from": str(_resume_ckpt(f"stage2_tusb_v3_seed123_{DATE_TAG}"))},
        {**common, "run_name": f"stage2_tusb_v3p1_seed42_{DATE_TAG}", "seed": 42, "family": "tusb_v3p1_main", "ablation_name": "main", "objective_combo": "tusb_v3p1_seed42", "objective_family": "trace_unit_semantic_binding_v3p1", "window_name": "tusbv31_s42", "dataset_names": ["vspw", "vipseg"], "resume_from": str(_resume_ckpt(f"stage2_tusb_v3_seed42_{DATE_TAG}"))},
        {**common, "run_name": f"stage2_tusb_v3p1_seed456_{DATE_TAG}", "seed": 456, "family": "tusb_v3p1_main", "ablation_name": "main", "objective_combo": "tusb_v3p1_seed456", "objective_family": "trace_unit_semantic_binding_v3p1", "window_name": "tusbv31_s456", "dataset_names": ["vspw", "vipseg"], "resume_from": str(_resume_ckpt(f"stage2_tusb_v3_seed456_{DATE_TAG}"))},
        {**common, "run_name": f"stage2_tusb_v3p1_no_ambiguity_repulse_seed123_{DATE_TAG}", "seed": 123, "family": "tusb_v3p1_ablation", "ablation_name": "no_ambiguity_repulse", "objective_combo": "tusb_v3p1_no_ambiguity_repulse_seed123", "objective_family": "trace_unit_semantic_binding_v3p1_ablation", "window_name": "tusbv31_noamb", "dataset_names": ["vspw", "vipseg"], "resume_from": str(_resume_ckpt(f"stage2_tusb_v3_seed123_{DATE_TAG}")), "trace_unit_ambiguity_repulse_weight": 0.0},
        {**common, "run_name": f"stage2_tusb_v3p1_no_appearance_refine_seed123_{DATE_TAG}", "seed": 123, "family": "tusb_v3p1_ablation", "ablation_name": "no_appearance_refine", "objective_combo": "tusb_v3p1_no_appearance_refine_seed123", "objective_family": "trace_unit_semantic_binding_v3p1_ablation", "window_name": "tusbv31_noapp", "dataset_names": ["vspw", "vipseg"], "resume_from": str(_resume_ckpt(f"stage2_tusb_v3_seed123_{DATE_TAG}")), "teacher_semantic_cache_path": str(TEACHER_CACHE_V3_ROOT), "trace_unit_appearance_refine_weight": 0.0},
        {**common, "run_name": f"stage2_tusb_v3p1_no_hardsubset_curriculum_seed123_{DATE_TAG}", "seed": 123, "family": "tusb_v3p1_ablation", "ablation_name": "no_hardsubset_curriculum", "objective_combo": "tusb_v3p1_no_hardsubset_curriculum_seed123", "objective_family": "trace_unit_semantic_binding_v3p1_ablation", "window_name": "tusbv31_nocur", "dataset_names": ["vspw", "vipseg"], "resume_from": str(_resume_ckpt(f"stage2_tusb_v3_seed123_{DATE_TAG}")), "trace_unit_hardsubset_curriculum_weight": 0.0},
    ]


def _selected_run_specs(args: Any) -> List[Dict[str, Any]]:
    specs = _run_specs()
    raw_names = str(getattr(args, "run_names", "") or "").strip()
    if not raw_names:
        return specs
    wanted = {name.strip() for name in raw_names.split(",") if name.strip()}
    return [spec for spec in specs if str(spec["run_name"]) in wanted]


def _common_launch_context(args: Any) -> Dict[str, Any]:
    lease_cleanup = base._cleanup_stale_leases(str(args.shared_lease_path), allowed_prefixes=("stage2_tusb_v3p1_",))
    if subprocess.run(["tmux", "has-session", "-t", str(args.tmux_session)], capture_output=True).returncode != 0:
        subprocess.run(["tmux", "new-session", "-d", "-s", str(args.tmux_session), "bash"], check=True)
    existing_windows = set(base._tmux_windows(str(args.tmux_session)))
    anchor_args = base._load_ckpt_args(_resume_ckpt(f"stage2_tusb_v3_seed123_{DATE_TAG}"))
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
    meta = v3._build_launch_meta(args, spec, ctx_meta)
    run_name = str(spec["run_name"])
    resume_from = Path(str(spec["resume_from"]))
    resume_step = base._load_ckpt_step(resume_from)
    out_dir = Path(args.work_root) / "outputs/checkpoints" / run_name
    meta.update(
        {
            "resume_from": str(resume_from),
            "resume_global_step": int(resume_step),
            "additional_train_steps": int(TRAIN_ADDITIONAL_STEPS),
            "train_steps": int(resume_step + TRAIN_ADDITIONAL_STEPS),
            "eval_interval": int(EVAL_INTERVAL),
            "save_every_n_steps": int(SAVE_EVERY),
            "predecode_cache_path": str(args.predecode_cache_path),
            "teacher_semantic_cache_path": str(spec.get("teacher_semantic_cache_path", args.teacher_semantic_cache_path)),
            "runtime_json": str(args.runtime_json),
            "raw_json": str(_paths_for_run(args, run_name)["raw"]),
            "progress_json": str(_paths_for_run(args, run_name)["progress"]),
            "final_json": str(_paths_for_run(args, run_name)["final"]),
            "log_path": str(_paths_for_run(args, run_name)["log"]),
            "output_dir": str(out_dir),
            "meta_json": str(_meta_dir(args) / f"{run_name}_launch_meta.json"),
        }
    )
    for key in [
        "trace_unit_ambiguity_repulse_weight",
        "trace_unit_ambiguity_risk_threshold",
        "trace_unit_ambiguity_min_dist_weight",
        "trace_unit_ambiguity_iou_weight",
        "trace_unit_ambiguity_motion_cross_weight",
        "trace_unit_appearance_refine_weight",
        "trace_unit_hardsubset_curriculum_weight",
        "trace_unit_hardsubset_ambiguity_weight",
        "trace_unit_hardsubset_appearance_weight",
        "trace_unit_hardsubset_occlusion_weight",
        "trace_unit_hardsubset_longgap_weight",
    ]:
        if key in spec:
            meta[key] = spec[key]
    return meta


def launch(args: Any) -> Dict[str, Any]:
    _write_protocol_artifacts(args)
    _ensure_teacher_prior_v4(args)
    _write_ambiguity_repulse(args)
    _write_hardsubset_curriculum(args)
    _write_unitization_smallrefine(args)
    ctx_meta = _common_launch_context(args)
    cleanup_actions: List[Dict[str, Any]] = []
    runs: List[Dict[str, Any]] = []
    for spec in _selected_run_specs(args):
        meta = _build_launch_meta(args, spec, ctx_meta)
        cleanup_actions.append(base._reset_run_artifacts(args=args, meta=meta, run_name=str(spec["run_name"])))
        runs.append(tusbbase._write_and_launch_meta(args, meta, ctx_meta["existing_windows"]))
    payload = {
        "generated_at_utc": now_iso(),
        "tmux_session": str(args.tmux_session),
        "teacher_backend": "clip_vit-b_16_temporal_drift_aware_masked_mean_v4",
        "policy": "TUSB-v3.1 hard-subset conversion on frozen Stage1; ambiguity-aware repulse; appearance-drift teacher refine; hard-subset curriculum; max 4 concurrent training tasks",
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
    return {
        "run_name": run_name,
        "family": str(spec["family"]),
        "ablation_name": str(spec["ablation_name"]),
        "status": str(status_info.get("status", "launched")).lower(),
        "best_checkpoint_metric": best_block,
        "latest_checkpoint_metric": latest_block,
        "semantic_hard_sidecar_metric": sidecar_block,
        "trace_unit_metrics": trace_block,
    }


def summarize(args: Any) -> Dict[str, Any]:
    run_rows = [_summary_row_for_run(args, spec) for spec in _selected_run_specs(args)]
    running = sum(int(row["status"] == "running") for row in run_rows)
    completed = sum(int(row["status"] == "completed") for row in run_rows)
    failed = sum(int(row["status"] == "failed") for row in run_rows)
    best_main = min(
        [row for row in run_rows if row["family"] == "tusb_v3p1_main" and row["status"] == "completed"],
        key=lambda row: (
            float(((row.get("semantic_hard_sidecar_metric") or {}).get("semantic_hard_sidecar_score", 1e9))),
            -float(((row.get("semantic_hard_sidecar_metric") or {}).get("metrics") or {}).get("query_future_top1_acc", 0.0)),
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
        "best_tusb_v3p1_run_name": str(best_main.get("run_name", "")),
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
        -float(row.get("appearance_change_top1_acc", 0.0)),
        -float(row.get("future_mask_iou_at_top1", 0.0)),
        float(row.get("query_future_localization_error", 1e9)),
    )


def _run_checkpoint_alignment(args: Any, summary: Dict[str, Any]) -> Dict[str, Any]:
    protocol = _json_or_empty(args.protocol_v3_json)
    items = protocol.get("items", []) if isinstance(protocol.get("items", []), list) else []
    if not hasattr(args, "lease_path") or not str(getattr(args, "lease_path", "")).strip():
        setattr(args, "lease_path", str(args.shared_lease_path))
    if not hasattr(args, "device"):
        setattr(args, "device", str(args.eval_device))
    device, device_info = evalv3._select_eval_device_v3(args)
    try:
        prepared, per_item_proto, context_mean, skipped_items = ctx._prepare_protocol_items(
            items,
            lambda item: evalv3._build_context_preserving_item_batch_v3(item, temporal_window=5, max_context_entities=K_CONTEXT),
            mode_name="context_preserving",
        )
        run_map = {
            "current_tusb_v3_main": _current_v3_best_run_name(),
            "tusb_v3p1_seed123": f"stage2_tusb_v3p1_seed123_{DATE_TAG}",
            "tusb_v3p1_seed42": f"stage2_tusb_v3p1_seed42_{DATE_TAG}",
            "tusb_v3p1_seed456": f"stage2_tusb_v3p1_seed456_{DATE_TAG}",
            "no_ambiguity_repulse": f"stage2_tusb_v3p1_no_ambiguity_repulse_seed123_{DATE_TAG}",
            "no_appearance_refine": f"stage2_tusb_v3p1_no_appearance_refine_seed123_{DATE_TAG}",
            "no_hardsubset_curriculum": f"stage2_tusb_v3p1_no_hardsubset_curriculum_seed123_{DATE_TAG}",
        }
        rows: List[Dict[str, Any]] = []
        for logical_name, run_name in run_map.items():
            ckpt_specs = []
            for ckpt_label, ckpt_path in _existing_checkpoint_candidates(run_name):
                ckpt_specs.append(
                    prev_eval.MethodSpec(
                        name=f"{logical_name}::{ckpt_label}",
                        run_name=run_name,
                        method_type="stage2",
                        checkpoint_path=str(ckpt_path),
                    )
                )
            if not ckpt_specs:
                continue
            per_item = [
                {
                    "protocol_item_id": str(item_row.get("protocol_item_id", "")),
                    "dataset": str(item_row.get("dataset", "")),
                    "clip_id": str(item_row.get("clip_id", "")),
                    "subset_tags": list(item_row.get("subset_tags", [])),
                    "target_id": str(item_row.get("target_id", "")),
                    "methods": {},
                }
                for item_row in per_item_proto
            ]
            per_item = ctx._evaluate_prepared(ckpt_specs, prepared, per_item, device=device)
            method_rows = ctx._aggregate_eval_methods(per_item, ckpt_specs)
            method_rows = sorted(method_rows, key=_protocol_rank)
            protocol_best = method_rows[0] if method_rows else {}
            rows.append(
                {
                    "logical_name": logical_name,
                    "run_name": run_name,
                    "checkpoint_rows": method_rows,
                    "rollout_best_checkpoint": "best.pt",
                    "protocol_best_checkpoint": str(protocol_best.get("name", "")).split("::")[-1] if protocol_best else "none",
                    "rollout_best_and_protocol_best_match": bool(str(protocol_best.get("name", "")).endswith("best.pt")),
                }
            )
        main_rows = [row for row in rows if row["logical_name"].startswith("tusb_v3p1_seed")]
        best_choice_name = "best.pt"
        best_run_name = ""
        best_method: Dict[str, Any] = {}
        for row in main_rows:
            for method_row in row.get("checkpoint_rows", []):
                ckpt_name = str(method_row.get("name", "")).split("::")[-1]
                if ckpt_name not in {"best.pt", "best_semantic_hard.pt"}:
                    continue
                if not best_method or _protocol_rank(method_row) < _protocol_rank(best_method):
                    best_method = method_row
                    best_run_name = str(row["run_name"])
                    best_choice_name = ckpt_name or "best.pt"
        payload = {
            "generated_at_utc": now_iso(),
            "protocol_eval_context_entity_count_mean": float(context_mean),
            "skipped_protocol_item_count": int(len(skipped_items)),
            "run_rows": rows,
            "best_tusb_v3p1_run_name": best_run_name,
            "best_tusb_v3p1_checkpoint_choice": best_choice_name,
            "best_semantic_hard_more_aligned_with_protocol": bool(best_choice_name == "best_semantic_hard.pt"),
            "rollout_best_checkpoint": "best.pt",
            "protocol_best_checkpoint": best_choice_name,
        }
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
    base._write_json(args.checkpoint_policy_report, payload)
    base._write_md(
        args.checkpoint_policy_doc,
        [
            "# Stage2 TUSB-V3.1 Checkpoint Policy 20260418",
            "",
            f"- best_tusb_v3p1_run_name: {payload['best_tusb_v3p1_run_name']}",
            f"- best_tusb_v3p1_checkpoint_choice: {payload['best_tusb_v3p1_checkpoint_choice']}",
            f"- best_semantic_hard_more_aligned_with_protocol: {payload['best_semantic_hard_more_aligned_with_protocol']}",
        ],
    )
    return payload


def _context_method_specs(alignment: Dict[str, Any]) -> List[prev_eval.MethodSpec]:
    best_run_name = str(alignment.get("best_tusb_v3p1_run_name", "")).strip() or f"stage2_tusb_v3p1_seed123_{DATE_TAG}"
    best_choice = str(alignment.get("best_tusb_v3p1_checkpoint_choice", "best.pt"))
    v3_best = _current_v3_best_run_name()
    no_amb_label, no_amb_path = _preferred_checkpoint_for_run(f"stage2_tusb_v3p1_no_ambiguity_repulse_seed123_{DATE_TAG}")
    no_app_label, no_app_path = _preferred_checkpoint_for_run(f"stage2_tusb_v3p1_no_appearance_refine_seed123_{DATE_TAG}")
    no_cur_label, no_cur_path = _preferred_checkpoint_for_run(f"stage2_tusb_v3p1_no_hardsubset_curriculum_seed123_{DATE_TAG}")
    specs = [
        prev_eval.MethodSpec(name="stage1_frozen_baseline", run_name="stage1_frozen_baseline", method_type="stage1", checkpoint_path=str(ROOT / "outputs/checkpoints/stage1_v2_longtrain_220m_mainline_20260408/best.pt")),
        prev_eval.MethodSpec(name="legacysem_best", run_name="stage2_fullscale_core_legacysem_seed456_wave2_20260409", method_type="stage2", checkpoint_path=str(ROOT / "outputs/checkpoints/stage2_fullscale_core_legacysem_seed456_wave2_20260409/best.pt")),
        prev_eval.MethodSpec(name="cropenc_baseline_best", run_name="stage2_fullscale_core_cropenc_seed456_20260409", method_type="stage2", checkpoint_path=str(ROOT / "outputs/checkpoints/stage2_fullscale_core_cropenc_seed456_20260409/best.pt")),
        prev_eval.MethodSpec(name="current_calibration_only_best", run_name=ctx._current_calibration_best_run(), method_type="stage2", checkpoint_path=str(ROOT / "outputs/checkpoints" / ctx._current_calibration_best_run() / "best.pt")),
        prev_eval.MethodSpec(name="current_tusb_v3_best_semantic_hard", run_name=v3_best, method_type="stage2", checkpoint_path=str(ROOT / "outputs/checkpoints" / v3_best / "best_semantic_hard.pt")),
        prev_eval.MethodSpec(name="tusb_v3p1_best", run_name=best_run_name, method_type="stage2", checkpoint_path=str(ROOT / "outputs/checkpoints" / best_run_name / best_choice)),
        prev_eval.MethodSpec(name=f"no_ambiguity_repulse::{no_amb_label}", run_name=f"stage2_tusb_v3p1_no_ambiguity_repulse_seed123_{DATE_TAG}", method_type="stage2", checkpoint_path=str(no_amb_path)) if no_amb_label else None,
        prev_eval.MethodSpec(name=f"no_appearance_refine::{no_app_label}", run_name=f"stage2_tusb_v3p1_no_appearance_refine_seed123_{DATE_TAG}", method_type="stage2", checkpoint_path=str(no_app_path)) if no_app_label else None,
        prev_eval.MethodSpec(name=f"no_hardsubset_curriculum::{no_cur_label}", run_name=f"stage2_tusb_v3p1_no_hardsubset_curriculum_seed123_{DATE_TAG}", method_type="stage2", checkpoint_path=str(no_cur_path)) if no_cur_label else None,
    ]
    return [spec for spec in specs if spec is not None and Path(spec.checkpoint_path).exists()]


def diagnose(args: Any) -> Dict[str, Any]:
    summary = summarize(args)
    alignment = _json_or_empty(args.checkpoint_policy_report)
    if not alignment:
        alignment = _run_checkpoint_alignment(args, summary)
    protocol = _json_or_empty(args.protocol_v3_json)
    items = protocol.get("items", []) if isinstance(protocol.get("items", []), list) else []
    if not hasattr(args, "lease_path") or not str(getattr(args, "lease_path", "")).strip():
        setattr(args, "lease_path", str(args.shared_lease_path))
    if not hasattr(args, "device"):
        setattr(args, "device", str(args.eval_device))
    result = ctx._run_eval_mode(
        args=args,
        protocol_items=items,
        specs=_context_method_specs(alignment),
        mode_name="context_preserving",
        builder=lambda item: evalv3._build_context_preserving_item_batch_v3(item, temporal_window=5, max_context_entities=K_CONTEXT),
    )
    methods = {str(row.get("name", "")): row for row in result.get("methods", []) if isinstance(row, dict)}
    chosen = methods.get("tusb_v3p1_best", {})
    cal = methods.get("current_calibration_only_best", {})
    v3_best = methods.get("current_tusb_v3_best_semantic_hard", {})
    no_amb = methods.get("no_ambiguity_repulse", {})
    no_app = methods.get("no_appearance_refine", {})
    no_cur = methods.get("no_hardsubset_curriculum", {})
    best_run_name = str(alignment.get("best_tusb_v3p1_run_name", "")).strip() or f"stage2_tusb_v3p1_seed123_{DATE_TAG}"
    best_choice = str(alignment.get("best_tusb_v3p1_checkpoint_choice", "best.pt"))
    final_payload = _json_or_empty(ROOT / "reports" / f"{best_run_name}_final.json")
    trace = final_payload.get("trace_unit_metrics", {}) if isinstance(final_payload.get("trace_unit_metrics", {}), dict) else {}
    improved_vs_cal = bool(
        chosen and cal
        and float(chosen.get("query_future_top1_acc", -1.0)) > float(cal.get("query_future_top1_acc", -1.0))
        and float(chosen.get("future_mask_iou_at_top1", -1.0)) >= float(cal.get("future_mask_iou_at_top1", -1.0))
    )
    ambiguity_improved = bool(chosen and cal and float(chosen.get("ambiguity_top1_acc", -1.0)) > float(cal.get("ambiguity_top1_acc", -1.0)))
    appearance_improved = bool(chosen and cal and float(chosen.get("appearance_change_top1_acc", -1.0)) > float(cal.get("appearance_change_top1_acc", -1.0)))
    hard_improved = bool(
        chosen and cal
        and float(chosen.get("hard_subset_top1_acc", -1.0)) > float(cal.get("hard_subset_top1_acc", -1.0))
        and ambiguity_improved
        and appearance_improved
    )
    z_sem_slower = bool(float(trace.get("z_sem_drift_mean", 1e9)) < float(trace.get("z_dyn_drift_mean", 1e9)))
    no_amb_degrades = bool(no_amb and chosen and _protocol_rank(chosen) < _protocol_rank(no_amb))
    no_app_degrades = bool(no_app and chosen and _protocol_rank(chosen) < _protocol_rank(no_app))
    no_cur_degrades = bool(no_cur and chosen and _protocol_rank(chosen) < _protocol_rank(no_cur))
    if improved_vs_cal and hard_improved and no_amb_degrades and no_app_degrades and no_cur_degrades and z_sem_slower:
        next_step = "freeze_tusb_v3p1_as_new_stage2_mainline"
    elif improved_vs_cal or best_choice == "best_semantic_hard.pt" or no_amb_degrades or no_app_degrades:
        next_step = "keep_tusb_v3p1_but_refine_teacher_or_checkpoint_selection_further"
    else:
        next_step = "rethink_stage2_story_if_hard_subset_conversion_still_fails"
    payload = {
        "generated_at_utc": now_iso(),
        "best_tusb_v3p1_run_name": best_run_name,
        "best_tusb_v3p1_checkpoint_choice": best_choice,
        "context_preserving_protocol_improved_vs_current_calonly": bool(improved_vs_cal),
        "hard_subsets_improved": bool(hard_improved),
        "ambiguity_top1_acc_improved": bool(ambiguity_improved),
        "appearance_change_top1_acc_improved": bool(appearance_improved),
        "no_ambiguity_repulse_degraded": bool(no_amb_degrades),
        "no_appearance_refine_degraded": bool(no_app_degrades),
        "no_hardsubset_curriculum_degraded": bool(no_cur_degrades),
        "z_sem_slower_than_z_dyn": bool(z_sem_slower),
        "same_instance_dominant_unit_match_rate": float(trace.get("same_instance_dominant_unit_match_rate_mean", 0.0)),
        "different_instance_dominant_unit_collision_rate": float(trace.get("different_instance_dominant_unit_collision_rate_mean", 0.0)),
        "ambiguity_highrisk_pair_collision_rate": float(trace.get("ambiguity_highrisk_pair_collision_rate_mean", 0.0)),
        "appearance_drift_highrisk_same_instance_match_rate": float(trace.get("appearance_drift_highrisk_same_instance_match_rate_mean", 0.0)),
        "semantic_hard_sidecar_score": float(final_payload.get("semantic_hard_sidecar_metric", {}).get("semantic_hard_sidecar_score", 1e9)),
        "next_step_choice": next_step,
        "method_rows": list(methods.values()),
    }
    base._write_json(args.diagnosis_report, payload)
    base._write_md(
        args.results_md,
        [
            "# Stage2 TUSB-V3.1 Hard-Subset Conversion 20260418",
            "",
            f"- best_tusb_v3p1_run_name: {payload['best_tusb_v3p1_run_name']}",
            f"- best_tusb_v3p1_checkpoint_choice: {payload['best_tusb_v3p1_checkpoint_choice']}",
            f"- context_preserving_protocol_improved_vs_current_calonly: {payload['context_preserving_protocol_improved_vs_current_calonly']}",
            f"- hard_subsets_improved: {payload['hard_subsets_improved']}",
            f"- ambiguity_top1_acc_improved: {payload['ambiguity_top1_acc_improved']}",
            f"- appearance_change_top1_acc_improved: {payload['appearance_change_top1_acc_improved']}",
            f"- z_sem_slower_than_z_dyn: {payload['z_sem_slower_than_z_dyn']}",
            f"- next_step_choice: {payload['next_step_choice']}",
        ],
    )
    return payload


def run_all(args: Any) -> Dict[str, Any]:
    _write_protocol_artifacts(args)
    _ensure_teacher_prior_v4(args)
    _write_ambiguity_repulse(args)
    _write_hardsubset_curriculum(args)
    _write_unitization_smallrefine(args)
    launch(args)
    summary = wait_for_completion(args)
    if bool(summary.get("all_runs_terminal", False)):
        _run_checkpoint_alignment(args, summary)
        diagnose(args)
    return {
        "generated_at_utc": now_iso(),
        "summary_report": str(args.summary_report),
        "diagnosis_report": str(args.diagnosis_report),
    }


def parse_args() -> Any:
    parser = ArgumentParser(description="Run STAGE2 TUSB-V3.1 hard-subset conversion")
    parser.add_argument("--mode", default="run", choices=["run", "launch", "summarize", "diagnose"])
    parser.add_argument("--run-names", default="", help="comma-separated subset of run names to operate on for launch/summarize")
    parser.add_argument("--work-root", default=str(ROOT))
    parser.add_argument("--tmux-session", default=SESSION)
    parser.add_argument("--python-bin", default=str(base._python_bin_default()))
    parser.add_argument("--stage2-contract-json", default=str(ROOT / "reports/stage2_bootstrap_data_contract_20260408.json"))
    parser.add_argument("--stage1-best-ckpt", default=str(ROOT / "outputs/checkpoints/stage1_v2_longtrain_220m_mainline_20260408/best.pt"))
    parser.add_argument("--shared-lease-path", default=str(ROOT / "reports/stage1_v2_gpu_lease_20260408.json"))
    parser.add_argument("--bootstrap-cache-jsonl", default=str(ROOT / "data/processed/stage2_real_bootstrap_cache_20260410/clip_vit_b32_core_trainval_required_subset.jsonl"))
    parser.add_argument("--semantic-hard-manifest-path", default=str(ROOT / "manifests/protocol_v2/stage2_semantic_hard_subsets_20260410.json"))
    parser.add_argument("--runtime-json", default=str(RUNTIME_JSON))
    parser.add_argument("--predecode-cache-path", default=str(PREDECODE_CACHE_ROOT))
    parser.add_argument("--teacher-semantic-cache-path", default=str(TEACHER_CACHE_V4_ROOT))
    parser.add_argument("--protocol-v3-json", default=str(ROOT / "reports/stage2_state_identifiability_protocol_v3_20260416.json"))
    parser.add_argument("--eval-device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--protocol-report", default=str(ROOT / "reports/stage2_tusb_v3p1_hardsubset_conversion_protocol_20260418.json"))
    parser.add_argument("--protocol-doc", default=str(ROOT / "docs/STAGE2_TUSB_V3P1_HARDSUBSET_CONVERSION_PROTOCOL_20260418.md"))
    parser.add_argument("--checkpoint-policy-report", default=str(ROOT / "reports/stage2_tusb_v3p1_checkpoint_policy_20260418.json"))
    parser.add_argument("--checkpoint-policy-doc", default=str(ROOT / "docs/STAGE2_TUSB_V3P1_CHECKPOINT_POLICY_20260418.md"))
    parser.add_argument("--ambiguity-repulse-report", default=str(ROOT / "reports/stage2_tusb_v3p1_ambiguity_repulse_20260418.json"))
    parser.add_argument("--ambiguity-repulse-doc", default=str(ROOT / "docs/STAGE2_TUSB_V3P1_AMBIGUITY_REPULSE_20260418.md"))
    parser.add_argument("--appearance-teacher-report", default=str(ROOT / "reports/stage2_tusb_v3p1_appearance_teacher_20260418.json"))
    parser.add_argument("--appearance-teacher-doc", default=str(ROOT / "docs/STAGE2_TUSB_V3P1_APPEARANCE_TEACHER_20260418.md"))
    parser.add_argument("--hardsubset-curriculum-report", default=str(ROOT / "reports/stage2_tusb_v3p1_hardsubset_curriculum_20260418.json"))
    parser.add_argument("--hardsubset-curriculum-doc", default=str(ROOT / "docs/STAGE2_TUSB_V3P1_HARDSUBSET_CURRICULUM_20260418.md"))
    parser.add_argument("--unitization-smallrefine-report", default=str(ROOT / "reports/stage2_tusb_v3p1_unitization_smallrefine_20260418.json"))
    parser.add_argument("--unitization-smallrefine-doc", default=str(ROOT / "docs/STAGE2_TUSB_V3P1_UNITIZATION_SMALLREFINE_20260418.md"))
    parser.add_argument("--launch-report", default=str(ROOT / "reports/stage2_tusb_v3p1_hardsubset_conversion_launch_20260418.json"))
    parser.add_argument("--summary-report", default=str(ROOT / "reports/stage2_tusb_v3p1_hardsubset_conversion_summary_20260418.json"))
    parser.add_argument("--diagnosis-report", default=str(ROOT / "reports/stage2_tusb_v3p1_hardsubset_conversion_diagnosis_20260418.json"))
    parser.add_argument("--results-md", default=str(ROOT / "docs/STAGE2_TUSB_V3P1_HARDSUBSET_CONVERSION_20260418.md"))
    parser.add_argument("--wait-timeout-seconds", type=int, default=172800)
    parser.add_argument("--poll-seconds", type=int, default=120)
    parser.add_argument("--gpu-acquire-timeout-seconds", type=int, default=7200)
    parser.add_argument("--gpu-acquire-retry-seconds", type=int, default=60)
    parser.add_argument("--max-concurrent-train-tasks", type=int, default=MAX_TRAIN_TASKS)
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
