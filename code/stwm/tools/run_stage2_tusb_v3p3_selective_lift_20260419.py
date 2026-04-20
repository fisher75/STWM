#!/usr/bin/env python3
from __future__ import annotations

from argparse import ArgumentParser, Namespace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple
import json
import subprocess
import time

from stwm.tools import run_stage2_state_identifiability_eval_20260415 as prev_eval
from stwm.tools import run_stage2_state_identifiability_eval_v3_20260416 as evalv3
from stwm.tools import run_stage2_tusb_v2_20260418 as tusbbase
from stwm.tools import run_stage2_tusb_v2_context_aligned_20260418 as ctx
from stwm.tools import run_stage2_tusb_v3_identity_binding_20260418 as v3
from stwm.tools import run_stage2_tusb_v3p1_hardsubset_conversion_20260418 as v31
from stwm.tools import run_stage2_tusb_v3p2_ceiling_lift_20260419 as v32
from stwm.tools import run_tracewm_stage2_calibration_only_fullscale_wave1_20260413 as base


ROOT = prev_eval.ROOT
SESSION = "tracewm_stage2_tusb_v3p3_selective_lift_20260419"
DATE_TAG = "20260419"
LOG_PATH = ROOT / "logs/stage2_tusb_v3p3_selective_lift_20260419.log"
TRAIN_ADDITIONAL_STEPS = 200
EVAL_INTERVAL = 100
SAVE_EVERY = 100
MAX_TRAIN_TASKS = 4
K_CONTEXT = 8
PREDECODE_CACHE_ROOT = ROOT / "data/processed/stage2_tusb_v3_predecode_cache_20260418"
TEACHER_CACHE_V4_ROOT = ROOT / "data/processed/stage2_teacher_semantic_cache_v4_appearance_20260418"
TEACHER_CACHE_V5_ROOT = ROOT / "data/processed/stage2_teacher_semantic_cache_v5_driftcal_20260419"
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
    return Path(args.work_root) / "reports/stage2_tusb_v3p3_selective_lift_runs_20260419"


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


def _current_v31_best_run_name() -> str:
    payload = _json_or_empty(ROOT / "reports/stage2_tusb_v3p1_hardsubset_conversion_diagnosis_20260418.json")
    return str(payload.get("best_tusb_v3p1_run_name", "")).strip() or "stage2_tusb_v3p1_seed123_20260418"


def _current_v31_best_checkpoint_choice() -> str:
    payload = _json_or_empty(ROOT / "reports/stage2_tusb_v3p1_checkpoint_policy_20260418.json")
    return str(payload.get("best_tusb_v3p1_checkpoint_choice", "best.pt")).strip() or "best.pt"


def _current_v31_best_checkpoint() -> Path:
    return ROOT / "outputs/checkpoints" / _current_v31_best_run_name() / _current_v31_best_checkpoint_choice()


def _run_name_to_branch(run_name: str) -> str:
    mapping = {
        f"stage2_tusb_v3p3_anchor_replay_seed123_{DATE_TAG}": "v3p1_anchor_replay",
        f"stage2_tusb_v3p3_confuser_only_seed123_{DATE_TAG}": "v3p1_plus_confuser_only",
        f"stage2_tusb_v3p3_appearance_only_seed123_{DATE_TAG}": "v3p1_plus_appearance_only",
        f"stage2_tusb_v3p3_confuser_and_appearance_light_seed123_{DATE_TAG}": "v3p1_plus_confuser_and_appearance_light",
    }
    return mapping.get(str(run_name), str(run_name))


def _branch_to_run_name(branch_name: str) -> str:
    reverse = {
        "v3p1_anchor_replay": f"stage2_tusb_v3p3_anchor_replay_seed123_{DATE_TAG}",
        "v3p1_plus_confuser_only": f"stage2_tusb_v3p3_confuser_only_seed123_{DATE_TAG}",
        "v3p1_plus_appearance_only": f"stage2_tusb_v3p3_appearance_only_seed123_{DATE_TAG}",
        "v3p1_plus_confuser_and_appearance_light": f"stage2_tusb_v3p3_confuser_and_appearance_light_seed123_{DATE_TAG}",
    }
    return reverse.get(branch_name, branch_name)


def _selected_run_specs(args: Any, appearance_payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    template = dict(next(spec for spec in v31._run_specs() if str(spec["run_name"]) == "stage2_tusb_v3p1_seed123_20260418"))
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
    anchor_ckpt = _current_v31_best_checkpoint()
    appearance_threshold = float(appearance_payload.get("global_combined_drift_high_threshold", 0.18))
    common.update({"teacher_semantic_cache_path": str(TEACHER_CACHE_V4_ROOT)})
    specs: List[Dict[str, Any]] = [
        {
            **common,
            "run_name": f"stage2_tusb_v3p3_anchor_replay_seed123_{DATE_TAG}",
            "seed": 123,
            "family": "tusb_v3p3_selective",
            "ablation_name": "anchor_replay",
            "objective_combo": "tusb_v3p3_anchor_replay_seed123",
            "objective_family": "trace_unit_semantic_binding_v3p3_selective",
            "window_name": "tusbv33_anchor",
            "dataset_names": ["vspw", "vipseg"],
            "resume_from": str(anchor_ckpt),
        },
        {
            **common,
            "run_name": f"stage2_tusb_v3p3_confuser_only_seed123_{DATE_TAG}",
            "seed": 123,
            "family": "tusb_v3p3_selective",
            "ablation_name": "confuser_only",
            "objective_combo": "tusb_v3p3_confuser_only_seed123",
            "objective_family": "trace_unit_semantic_binding_v3p3_selective",
            "window_name": "tusbv33_conf",
            "dataset_names": ["vspw", "vipseg"],
            "resume_from": str(anchor_ckpt),
            "trace_unit_confuser_separation_weight": 0.08,
            "trace_unit_confuser_risk_threshold": 0.48,
            "trace_unit_confuser_appearance_weight": 0.40,
            "trace_unit_confuser_motion_weight": 0.30,
            "trace_unit_confuser_overlap_weight": 0.30,
        },
        {
            **common,
            "run_name": f"stage2_tusb_v3p3_appearance_only_seed123_{DATE_TAG}",
            "seed": 123,
            "family": "tusb_v3p3_selective",
            "ablation_name": "appearance_only",
            "objective_combo": "tusb_v3p3_appearance_only_seed123",
            "objective_family": "trace_unit_semantic_binding_v3p3_selective",
            "window_name": "tusbv33_app",
            "dataset_names": ["vspw", "vipseg"],
            "resume_from": str(anchor_ckpt),
            "teacher_semantic_cache_path": str(TEACHER_CACHE_V5_ROOT),
            "trace_unit_appearance_refine_weight": 0.08,
            "trace_unit_appearance_high_threshold": appearance_threshold,
            "trace_unit_appearance_high_quantile": 0.70,
        },
        {
            **common,
            "run_name": f"stage2_tusb_v3p3_confuser_and_appearance_light_seed123_{DATE_TAG}",
            "seed": 123,
            "family": "tusb_v3p3_selective",
            "ablation_name": "confuser_and_appearance_light",
            "objective_combo": "tusb_v3p3_confuser_and_appearance_light_seed123",
            "objective_family": "trace_unit_semantic_binding_v3p3_selective",
            "window_name": "tusbv33_cfal",
            "dataset_names": ["vspw", "vipseg"],
            "resume_from": str(anchor_ckpt),
            "teacher_semantic_cache_path": str(TEACHER_CACHE_V5_ROOT),
            "trace_unit_confuser_separation_weight": 0.04,
            "trace_unit_confuser_risk_threshold": 0.48,
            "trace_unit_confuser_appearance_weight": 0.40,
            "trace_unit_confuser_motion_weight": 0.30,
            "trace_unit_confuser_overlap_weight": 0.30,
            "trace_unit_appearance_refine_weight": 0.05,
            "trace_unit_appearance_high_threshold": appearance_threshold,
            "trace_unit_appearance_high_quantile": 0.70,
        },
    ]
    raw_names = str(getattr(args, "run_names", "") or "").strip()
    if not raw_names:
        return specs
    wanted = {name.strip() for name in raw_names.split(",") if name.strip()}
    return [spec for spec in specs if str(spec["run_name"]) in wanted]


def _common_launch_context(args: Any) -> Dict[str, Any]:
    lease_cleanup = base._cleanup_stale_leases(str(args.shared_lease_path), allowed_prefixes=("stage2_tusb_v3p3_",))
    if subprocess.run(["tmux", "has-session", "-t", str(args.tmux_session)], capture_output=True).returncode != 0:
        subprocess.run(["tmux", "new-session", "-d", "-s", str(args.tmux_session), "bash"], check=True)
    existing_windows = set(base._tmux_windows(str(args.tmux_session)))
    anchor_args = base._load_ckpt_args(_current_v31_best_checkpoint())
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
            "worker_pid_file": str(_meta_dir(args) / f"{run_name}.pid"),
            "max_concurrent_tusb_tasks": int(args.max_concurrent_train_tasks),
        }
    )
    for key in [
        "trace_unit_confuser_separation_weight",
        "trace_unit_confuser_risk_threshold",
        "trace_unit_confuser_appearance_weight",
        "trace_unit_confuser_motion_weight",
        "trace_unit_confuser_overlap_weight",
        "trace_unit_appearance_refine_weight",
        "trace_unit_appearance_high_threshold",
        "trace_unit_appearance_high_quantile",
    ]:
        if key in spec:
            meta[key] = spec[key]
    return meta


def _write_protocol_artifacts(args: Any) -> Dict[str, Any]:
    v31_diag = _json_or_empty(ROOT / "reports/stage2_tusb_v3p1_hardsubset_conversion_diagnosis_20260418.json")
    v32_diag = _json_or_empty(ROOT / "reports/stage2_tusb_v3p2_ceiling_lift_diagnosis_20260419.json")
    hardpanel = _json_or_empty(args.hardpanel_densified_report)
    payload = {
        "generated_at_utc": now_iso(),
        "stage1_frozen": True,
        "stage1_train_or_unfreeze_allowed": False,
        "current_tusb_v3p1_anchor": {
            "run_name": _current_v31_best_run_name(),
            "checkpoint_choice": _current_v31_best_checkpoint_choice(),
            "context_preserving_protocol_improved_vs_current_calonly": bool(
                v31_diag.get("context_preserving_protocol_improved_vs_current_calonly", False)
            ),
            "identity_binding_learned": True,
            "z_sem_slower_than_z_dyn": True,
        },
        "current_tusb_v3p2_truth": {
            "did_not_exceed_v3p1_ceiling": bool(not v32_diag.get("context_preserving_protocol_improved_vs_current_calonly", False)),
            "main_value": [
                "exposed bottlenecks",
                "confuser direction may help",
                "appearance signal offline can be non-zero",
            ],
        },
        "current_implementation_bottlenecks": {
            "densified_hard_panel_exists_but_not_in_main_judge_chain": True,
            "appearance_signal_offline_nonzero_but_training_metrics_near_zero": True,
            "hardpanel_legacy_count": int(hardpanel.get("old_effective_count", 85)),
            "hardpanel_densified_count": int(hardpanel.get("new_protocol_v3_count", 200)),
        },
        "this_round_goal": {
            "protocol_v4": False,
            "new_architecture": False,
            "priority": [
                "rollback_to_v3p1_anchor",
                "dualpannel_judge_alignment",
                "appearance_signal_plumbing_repair",
                "one_at_a_time_selective_add_back",
            ],
        },
    }
    base._write_json(args.protocol_report, payload)
    base._write_md(
        args.protocol_doc,
        [
            "# Stage2 TUSB-V3.3 Rollback-Anchored Protocol 20260419",
            "",
            "- Stage1 remains frozen. No training, no unfreeze, no backbone swap.",
            "- current strongest anchor is TUSB-v3.1, not v3.2.",
            "- TUSB-v3.2 did not exceed the v3.1 ceiling; its value is bottleneck exposure, not cumulative stacking.",
            "- densified hard panel already exists but is not yet the main judge chain.",
            "- appearance signal is non-zero offline but still weak in training-side metrics.",
            "- this round rolls back to v3.1 anchor, wires dual-panel judge, repairs appearance plumbing, and selectively adds back only confuser or appearance components.",
        ],
    )
    return payload


def _panel_subset_counts(per_item: List[Dict[str, Any]]) -> Dict[str, int]:
    counts = {
        "full_identifiability_panel": int(len(per_item)),
        "occlusion_reappearance": 0,
        "crossing_ambiguity": 0,
        "small_object": 0,
        "appearance_change": 0,
        "long_gap_persistence": 0,
    }
    for item in per_item:
        tags = set(str(x) for x in item.get("subset_tags", []))
        if "occlusion_reappearance" in tags:
            counts["occlusion_reappearance"] += 1
        if "crossing_ambiguity" in tags:
            counts["crossing_ambiguity"] += 1
        if "small_object" in tags:
            counts["small_object"] += 1
        if "appearance_change" in tags:
            counts["appearance_change"] += 1
        if "long_gap_persistence" in tags:
            counts["long_gap_persistence"] += 1
    return counts


def _method_name_map_for_branch(branch_name: str) -> Dict[str, str]:
    if branch_name == "current_tusb_v3p1_best":
        run_name = _current_v31_best_run_name()
        return {
            "best.pt": f"{branch_name}::best.pt",
            "best_semantic_hard.pt": f"{branch_name}::best_semantic_hard.pt",
            "run_name": run_name,
        }
    if branch_name == "v3p1_plus_dualpanel_sidecar_only":
        return {
            "best.pt": "current_tusb_v3p1_best::best.pt",
            "best_semantic_hard.pt": "current_tusb_v3p1_best::best_semantic_hard.pt",
            "run_name": _current_v31_best_run_name(),
        }
    run_name = _branch_to_run_name(branch_name)
    return {
        "best.pt": f"{branch_name}::best.pt",
        "best_semantic_hard.pt": f"{branch_name}::best_semantic_hard.pt",
        "run_name": run_name,
    }


def _method_specs_for_dualpanel() -> List[prev_eval.MethodSpec]:
    specs: List[prev_eval.MethodSpec] = [
        prev_eval.MethodSpec(
            name="stage1_frozen_baseline",
            run_name="stage1_frozen_baseline",
            method_type="stage1",
            checkpoint_path=str(ROOT / "outputs/checkpoints/stage1_v2_longtrain_220m_mainline_20260408/best.pt"),
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
            run_name=ctx._current_calibration_best_run(),
            method_type="stage2",
            checkpoint_path=str(ROOT / "outputs/checkpoints" / ctx._current_calibration_best_run() / "best.pt"),
        ),
    ]
    branch_names = [
        "current_tusb_v3p1_best",
        "v3p1_anchor_replay",
        "v3p1_plus_confuser_only",
        "v3p1_plus_appearance_only",
        "v3p1_plus_confuser_and_appearance_light",
    ]
    for branch_name in branch_names:
        mapping = _method_name_map_for_branch(branch_name)
        run_name = mapping["run_name"]
        ckpt_dir = ROOT / "outputs/checkpoints" / run_name
        for ckpt_name in ["best.pt", "best_semantic_hard.pt"]:
            ckpt_path = ckpt_dir / ckpt_name
            if ckpt_path.exists():
                specs.append(
                    prev_eval.MethodSpec(
                        name=mapping[ckpt_name],
                        run_name=run_name,
                        method_type="stage2",
                        checkpoint_path=str(ckpt_path),
                    )
                )
    return [spec for spec in specs if Path(spec.checkpoint_path).exists()]


def _choose_branch_checkpoint(methods_by_name: Dict[str, Dict[str, Any]], branch_name: str) -> Tuple[str, Dict[str, Any]]:
    mapping = _method_name_map_for_branch(branch_name)
    best_row = methods_by_name.get(mapping["best.pt"], {})
    side_row = methods_by_name.get(mapping["best_semantic_hard.pt"], {})
    if best_row and side_row:
        if ctx._protocol_rank(side_row) < ctx._protocol_rank(best_row):
            return "best_semantic_hard.pt", side_row
        return "best.pt", best_row
    if side_row:
        return "best_semantic_hard.pt", side_row
    if best_row:
        return "best.pt", best_row
    return "", {}


def _run_dualpanel_judge(args: Any) -> Dict[str, Any]:
    protocol = _json_or_empty(args.protocol_v3_json)
    items = protocol.get("items", []) if isinstance(protocol.get("items", []), list) else []
    specs = _method_specs_for_dualpanel()
    if not hasattr(args, "lease_path") or not str(getattr(args, "lease_path", "")).strip():
        setattr(args, "lease_path", str(args.shared_lease_path))
    if not hasattr(args, "device"):
        setattr(args, "device", str(args.eval_device))
    legacy_result = ctx._run_eval_mode(
        args=args,
        protocol_items=items,
        specs=specs,
        mode_name="legacy_85_context_preserving",
        builder=lambda item: evalv3._build_context_preserving_item_batch_v3(item, temporal_window=5, max_context_entities=K_CONTEXT),
    )
    dense_result = ctx._run_eval_mode(
        args=args,
        protocol_items=items,
        specs=specs,
        mode_name="densified_200_single_target",
        builder=lambda item: evalv3._build_single_item_batch_v3(item, temporal_window=5),
    )
    legacy_methods = ctx._method_by_name(legacy_result.get("methods", []))
    dense_methods = ctx._method_by_name(dense_result.get("methods", []))
    branch_names = [
        "current_tusb_v3p1_best",
        "v3p1_anchor_replay",
        "v3p1_plus_confuser_only",
        "v3p1_plus_appearance_only",
        "v3p1_plus_confuser_and_appearance_light",
        "v3p1_plus_dualpanel_sidecar_only",
    ]
    legacy_choices: Dict[str, Dict[str, Any]] = {}
    dense_choices: Dict[str, Dict[str, Any]] = {}
    for branch_name in branch_names:
        choice, row = _choose_branch_checkpoint(legacy_methods, branch_name)
        legacy_choices[branch_name] = {"checkpoint_choice": choice, "metrics": row}
        choice, row = _choose_branch_checkpoint(dense_methods, branch_name)
        dense_choices[branch_name] = {"checkpoint_choice": choice, "metrics": row}
    hardpanel = _json_or_empty(args.hardpanel_densified_report)
    payload = {
        "generated_at_utc": now_iso(),
        "legacy_85_count": int(legacy_result.get("protocol_item_count", 0)),
        "densified_200_count": int(dense_result.get("protocol_item_count", 0)),
        "legacy_85_panel": {
            "per_subset_counts": _panel_subset_counts(legacy_result.get("per_item_results", [])),
            "method_rows": legacy_result.get("methods", []),
            "branch_checkpoint_choices": legacy_choices,
        },
        "densified_200_panel": {
            "per_subset_counts": dict(hardpanel.get("per_subset_counts", {})) or _panel_subset_counts(dense_result.get("per_item_results", [])),
            "method_rows": dense_result.get("methods", []),
            "branch_checkpoint_choices": dense_choices,
        },
        "best_pt_and_sidecar_consistent_by_branch": {
            branch_name: bool(
                legacy_choices.get(branch_name, {}).get("checkpoint_choice", "") == dense_choices.get(branch_name, {}).get("checkpoint_choice", "")
            )
            for branch_name in branch_names
        },
        "densified_200_more_stable_hardsubset_conclusion": bool(
            int(dense_result.get("protocol_item_count", 0)) > int(legacy_result.get("protocol_item_count", 0))
        ),
    }
    base._write_json(args.dualpanel_judge_report, payload)
    base._write_md(
        args.dualpanel_judge_doc,
        [
            "# Stage2 TUSB-V3.3 Dualpanel Judge 20260419",
            "",
            f"- legacy_85_count: {payload['legacy_85_count']}",
            f"- densified_200_count: {payload['densified_200_count']}",
            f"- densified_200_more_stable_hardsubset_conclusion: {payload['densified_200_more_stable_hardsubset_conclusion']}",
            "",
            "## Densified 200 Branch Choices",
            "",
            *[
                f"- {branch}: {info.get('checkpoint_choice', '')}"
                for branch, info in sorted(payload["densified_200_panel"]["branch_checkpoint_choices"].items())
            ],
        ],
    )
    return payload


def _write_appearance_plumbing(args: Any) -> Dict[str, Any]:
    offline = _json_or_empty(args.appearance_signal_report)
    branch_rows: List[Dict[str, Any]] = []
    for run_name in [
        f"stage2_tusb_v3p3_anchor_replay_seed123_{DATE_TAG}",
        f"stage2_tusb_v3p3_appearance_only_seed123_{DATE_TAG}",
        f"stage2_tusb_v3p3_confuser_and_appearance_light_seed123_{DATE_TAG}",
    ]:
        final_payload = _json_or_empty(ROOT / "reports" / f"{run_name}_final.json")
        if str(final_payload.get("status", "")).lower() != "completed":
            continue
        trace = final_payload.get("trace_unit_metrics", {}) if isinstance(final_payload.get("trace_unit_metrics", {}), dict) else {}
        branch_rows.append(
            {
                "branch_name": _run_name_to_branch(run_name),
                "run_name": run_name,
                "batch_appearance_drift_high_ratio_mean": float(trace.get("batch_appearance_drift_high_ratio_mean", trace.get("appearance_drift_high_ratio_mean", 0.0))),
                "step_appearance_drift_high_count_mean": float(trace.get("step_appearance_drift_high_count_mean", 0.0)),
                "appearance_signal_valid_count_mean": float(trace.get("appearance_signal_valid_count_mean", 0.0)),
                "appearance_refine_loss_nonzero_ratio": float(trace.get("appearance_refine_loss_nonzero_ratio", 0.0)),
            }
        )
    best_branch = max(
        branch_rows,
        key=lambda row: (
            float(row.get("appearance_refine_loss_nonzero_ratio", 0.0)),
            float(row.get("batch_appearance_drift_high_ratio_mean", 0.0)),
            float(row.get("appearance_signal_valid_count_mean", 0.0)),
        ),
        default={},
    )
    payload = {
        "generated_at_utc": now_iso(),
        "offline_appearance_drift_high_ratio": float(offline.get("appearance_drift_high_ratio", 0.0)),
        "batch_level_appearance_drift_high_ratio_mean": float(best_branch.get("batch_appearance_drift_high_ratio_mean", 0.0)),
        "step_appearance_drift_high_count_mean": float(best_branch.get("step_appearance_drift_high_count_mean", 0.0)),
        "appearance_signal_valid_count_mean": float(best_branch.get("appearance_signal_valid_count_mean", 0.0)),
        "appearance_refine_loss_nonzero_ratio": float(best_branch.get("appearance_refine_loss_nonzero_ratio", 0.0)),
        "current_env_blocked_backends": dict(offline.get("current_env_blocked_backends", {})),
        "chosen_teacher_prior": str(offline.get("chosen_teacher_prior_v5", "")),
        "branch_rows": branch_rows,
        "appearance_signal_reaches_training": bool(
            float(best_branch.get("appearance_signal_valid_count_mean", 0.0)) > 0.0
            and float(best_branch.get("appearance_refine_loss_nonzero_ratio", 0.0)) > 0.0
        ),
    }
    base._write_json(args.appearance_plumbing_report, payload)
    base._write_md(
        args.appearance_plumbing_doc,
        [
            "# Stage2 TUSB-V3.3 Appearance Plumbing 20260419",
            "",
            f"- offline_appearance_drift_high_ratio: {payload['offline_appearance_drift_high_ratio']:.6f}",
            f"- batch_level_appearance_drift_high_ratio_mean: {payload['batch_level_appearance_drift_high_ratio_mean']:.6f}",
            f"- appearance_signal_valid_count_mean: {payload['appearance_signal_valid_count_mean']:.6f}",
            f"- appearance_refine_loss_nonzero_ratio: {payload['appearance_refine_loss_nonzero_ratio']:.6f}",
            f"- appearance_signal_reaches_training: {payload['appearance_signal_reaches_training']}",
        ],
    )
    return payload


def _write_checkpoint_dualtrack(args: Any, judge: Dict[str, Any]) -> Dict[str, Any]:
    legacy_choices = ((judge.get("legacy_85_panel") or {}).get("branch_checkpoint_choices") or {}) if isinstance((judge.get("legacy_85_panel") or {}).get("branch_checkpoint_choices"), dict) else {}
    dense_choices = ((judge.get("densified_200_panel") or {}).get("branch_checkpoint_choices") or {}) if isinstance((judge.get("densified_200_panel") or {}).get("branch_checkpoint_choices"), dict) else {}
    branches = sorted(set(legacy_choices.keys()) | set(dense_choices.keys()))
    payload = {
        "generated_at_utc": now_iso(),
        "branch_dualtrack": {
            branch: {
                "legacy_85_panel_checkpoint_choice": str((legacy_choices.get(branch) or {}).get("checkpoint_choice", "")),
                "densified_200_panel_checkpoint_choice": str((dense_choices.get(branch) or {}).get("checkpoint_choice", "")),
                "checkpoint_choice_split_persists": bool(
                    str((legacy_choices.get(branch) or {}).get("checkpoint_choice", ""))
                    != str((dense_choices.get(branch) or {}).get("checkpoint_choice", ""))
                ),
            }
            for branch in branches
        },
    }
    base._write_json(args.checkpoint_dualtrack_report, payload)
    base._write_md(
        args.checkpoint_dualtrack_doc,
        [
            "# Stage2 TUSB-V3.3 Checkpoint Dualtrack 20260419",
            "",
            *[
                f"- {branch}: legacy={row['legacy_85_panel_checkpoint_choice']} densified={row['densified_200_panel_checkpoint_choice']}"
                for branch, row in sorted(payload["branch_dualtrack"].items())
            ],
        ],
    )
    return payload


def _write_and_launch_spec(args: Any, spec: Dict[str, Any], ctx_meta: Dict[str, Any]) -> Dict[str, Any]:
    meta = _build_launch_meta(args, spec, ctx_meta)
    base._reset_run_artifacts(args=args, meta=meta, run_name=str(spec["run_name"]))
    return tusbbase._write_and_launch_meta(args, meta, ctx_meta["existing_windows"])


def launch(args: Any) -> Dict[str, Any]:
    _write_protocol_artifacts(args)
    v32._ensure_teacher_prior_v5(args)
    v32._write_hardpanel_densified(args)
    ctx_meta = _common_launch_context(args)
    runs: List[Dict[str, Any]] = []
    selected = _selected_run_specs(args, _json_or_empty(args.appearance_signal_report))
    for spec in selected:
        runs.append(_write_and_launch_spec(args, spec, ctx_meta))
    payload = {
        "generated_at_utc": now_iso(),
        "tmux_session": str(args.tmux_session),
        "policy": "rollback to v3.1 anchor; dual-panel judge; appearance plumbing repair; selective add-back only",
        "lease_cleanup": ctx_meta["lease_cleanup"],
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
    final_status = str(final_payload.get("status", "")).lower()
    if final_status in {"completed", "failed"}:
        status = final_status
    else:
        pid_alive = False
        pid_file = Path(str(meta.get("worker_pid_file", ""))) if str(meta.get("worker_pid_file", "")).strip() else Path()
        if pid_file and pid_file.exists():
            try:
                pid = int(pid_file.read_text(encoding="utf-8").strip())
                subprocess.os.kill(pid, 0)
                pid_alive = True
            except Exception:
                pid_alive = False
        if pid_alive:
            status = "running"
        else:
            status_info = base._status_for(
                {**meta, "window_name": str(meta.get("window_name", spec.get("window_name", ""))), "progress_json": str(paths["progress"]), "final_json": str(paths["final"])},
                session_name=str(args.tmux_session),
            )
            status = str(status_info.get("status", "launched")).lower()
    return {
        "run_name": run_name,
        "branch_name": _run_name_to_branch(run_name),
        "family": str(spec["family"]),
        "ablation_name": str(spec["ablation_name"]),
        "status": status,
        "best_checkpoint_metric": base._best_block(final_payload, raw_payload, progress_payload),
        "latest_checkpoint_metric": base._latest_block(final_payload, raw_payload, progress_payload),
        "semantic_hard_sidecar_metric": base._sidecar_block(final_payload, raw_payload, progress_payload),
        "trace_unit_metrics": tusbbase._trace_unit_block(final_payload, raw_payload, progress_payload),
    }


def summarize(args: Any) -> Dict[str, Any]:
    specs = _selected_run_specs(args, _json_or_empty(args.appearance_signal_report))
    run_rows = [_summary_row_for_run(args, spec) for spec in specs]
    running = sum(int(str(row["status"]).lower() == "running") for row in run_rows)
    completed = sum(int(str(row["status"]).lower() == "completed") for row in run_rows)
    failed = sum(int(str(row["status"]).lower() == "failed") for row in run_rows)
    payload = {
        "generated_at_utc": now_iso(),
        "status": f"{running}_running_{completed}_completed_{failed}_failed",
        "running_count": int(running),
        "completed_count": int(completed),
        "failed_count": int(failed),
        "all_runs_terminal": bool(running == 0 and (completed + failed) == len(run_rows)),
        "run_rows": run_rows,
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


def _panel_methods_map(judge: Dict[str, Any], panel_key: str) -> Dict[str, Dict[str, Any]]:
    panel = judge.get(panel_key, {}) if isinstance(judge.get(panel_key, {}), dict) else {}
    methods = panel.get("method_rows", []) if isinstance(panel.get("method_rows", []), list) else []
    return {str(row.get("name", "")): row for row in methods if isinstance(row, dict)}


def _selected_branch_rows(judge: Dict[str, Any], panel_key: str) -> Dict[str, Dict[str, Any]]:
    panel = judge.get(panel_key, {}) if isinstance(judge.get(panel_key, {}), dict) else {}
    choice_map = panel.get("branch_checkpoint_choices", {}) if isinstance(panel.get("branch_checkpoint_choices", {}), dict) else {}
    out: Dict[str, Dict[str, Any]] = {}
    for branch_name, row in choice_map.items():
        if isinstance(row, dict):
            metrics = row.get("metrics", {})
            if isinstance(metrics, dict):
                out[str(branch_name)] = metrics
    return out


def _trace_metrics_for_branch(branch_name: str) -> Dict[str, Any]:
    if branch_name in {"current_tusb_v3p1_best", "v3p1_plus_dualpanel_sidecar_only"}:
        run_name = _current_v31_best_run_name()
    else:
        run_name = _branch_to_run_name(branch_name)
    payload = _json_or_empty(ROOT / "reports" / f"{run_name}_final.json")
    trace = payload.get("trace_unit_metrics", {}) if isinstance(payload.get("trace_unit_metrics", {}), dict) else {}
    return trace


def diagnose(args: Any) -> Dict[str, Any]:
    summary = summarize(args)
    judge = _json_or_empty(args.dualpanel_judge_report)
    if not judge:
        judge = _run_dualpanel_judge(args)
    appearance_plumbing = _json_or_empty(args.appearance_plumbing_report)
    if not appearance_plumbing:
        appearance_plumbing = _write_appearance_plumbing(args)
    checkpoint_dualtrack = _json_or_empty(args.checkpoint_dualtrack_report)
    if not checkpoint_dualtrack:
        checkpoint_dualtrack = _write_checkpoint_dualtrack(args, judge)

    dense_methods = _panel_methods_map(judge, "densified_200_panel")
    dense_selected = _selected_branch_rows(judge, "densified_200_panel")
    cal = dense_methods.get("current_calibration_only_best", {})
    anchor_best_pt = dense_methods.get("current_tusb_v3p1_best::best.pt", {})
    branch_candidates = {
        branch: dense_selected.get(branch, {})
        for branch in [
            "v3p1_anchor_replay",
            "v3p1_plus_confuser_only",
            "v3p1_plus_appearance_only",
            "v3p1_plus_dualpanel_sidecar_only",
            "v3p1_plus_confuser_and_appearance_light",
        ]
        if dense_selected.get(branch, {})
    }
    best_branch_name = min(branch_candidates, key=lambda name: ctx._protocol_rank(branch_candidates[name]), default="")
    best_branch_metrics = branch_candidates.get(best_branch_name, {})
    best_choice = str(
        (((judge.get("densified_200_panel") or {}).get("branch_checkpoint_choices") or {}).get(best_branch_name, {}) or {}).get(
            "checkpoint_choice",
            "",
        )
    )
    improved_vs_cal = bool(
        best_branch_metrics
        and cal
        and float(best_branch_metrics.get("query_future_top1_acc", -1.0)) > float(cal.get("query_future_top1_acc", -1.0))
        and float(best_branch_metrics.get("future_mask_iou_at_top1", -1.0)) >= float(cal.get("future_mask_iou_at_top1", -1.0))
    )
    hard_improved = bool(
        best_branch_metrics
        and cal
        and float(best_branch_metrics.get("hard_subset_top1_acc", -1.0)) > float(cal.get("hard_subset_top1_acc", -1.0))
    )
    ambiguity_improved = bool(
        best_branch_metrics
        and cal
        and float(best_branch_metrics.get("ambiguity_top1_acc", -1.0)) > float(cal.get("ambiguity_top1_acc", -1.0))
    )
    appearance_improved = bool(
        best_branch_metrics
        and cal
        and float(best_branch_metrics.get("appearance_change_top1_acc", -1.0)) > float(cal.get("appearance_change_top1_acc", -1.0))
    )
    branch_exceeds_anchor = bool(
        best_branch_metrics and anchor_best_pt and ctx._protocol_rank(best_branch_metrics) < ctx._protocol_rank(anchor_best_pt)
    )
    trace = _trace_metrics_for_branch(best_branch_name)
    z_sem_slower = bool(float(trace.get("z_sem_drift_mean", 1e9)) < float(trace.get("z_dyn_drift_mean", 1e9)))
    if improved_vs_cal and hard_improved and (ambiguity_improved or appearance_improved) and branch_exceeds_anchor:
        next_step = "freeze_selective_lift_branch_as_new_stage2_mainline"
    elif improved_vs_cal or branch_exceeds_anchor or bool(appearance_plumbing.get("appearance_signal_reaches_training", False)):
        next_step = "keep_v3p3_but_refine_signal_or_data_density_further"
    else:
        next_step = "stop_stacking_and_write_if_no_branch_exceeds_v3p1_anchor"
    payload = {
        "generated_at_utc": now_iso(),
        "best_selective_lift_branch_name": best_branch_name,
        "best_selective_lift_checkpoint_choice": best_choice,
        "densified_200_panel_improved_vs_current_calonly": bool(improved_vs_cal),
        "densified_200_panel_hard_subsets_improved": bool(hard_improved),
        "ambiguity_top1_acc_improved": bool(ambiguity_improved),
        "appearance_change_top1_acc_improved": bool(appearance_improved),
        "z_sem_slower_than_z_dyn": bool(z_sem_slower),
        "branch_exceeds_current_tusb_v3p1_best": bool(branch_exceeds_anchor),
        "next_step_choice": next_step,
        "appearance_signal_reaches_training": bool(appearance_plumbing.get("appearance_signal_reaches_training", False)),
        "selected_branch_metrics_densified_200": best_branch_metrics,
    }
    base._write_json(args.diagnosis_report, payload)
    base._write_md(
        args.results_md,
        [
            "# Stage2 TUSB-V3.3 Selective Lift 20260419",
            "",
            f"- best_selective_lift_branch_name: {payload['best_selective_lift_branch_name']}",
            f"- best_selective_lift_checkpoint_choice: {payload['best_selective_lift_checkpoint_choice']}",
            f"- densified_200_panel_improved_vs_current_calonly: {payload['densified_200_panel_improved_vs_current_calonly']}",
            f"- densified_200_panel_hard_subsets_improved: {payload['densified_200_panel_hard_subsets_improved']}",
            f"- ambiguity_top1_acc_improved: {payload['ambiguity_top1_acc_improved']}",
            f"- appearance_change_top1_acc_improved: {payload['appearance_change_top1_acc_improved']}",
            f"- z_sem_slower_than_z_dyn: {payload['z_sem_slower_than_z_dyn']}",
            f"- next_step_choice: {payload['next_step_choice']}",
        ],
    )
    return payload


def run_all(args: Any) -> Dict[str, Any]:
    appearance_payload = v32._ensure_teacher_prior_v5(args)
    _write_protocol_artifacts(args)
    v32._write_hardpanel_densified(args)
    selected_specs = _selected_run_specs(args, appearance_payload)
    max_batch = max(1, int(getattr(args, "max_concurrent_train_tasks", MAX_TRAIN_TASKS)))
    summary: Dict[str, Any] = {}
    for start in range(0, len(selected_specs), max_batch):
        batch_specs = selected_specs[start : start + max_batch]
        batch_names = ",".join(str(spec["run_name"]) for spec in batch_specs)
        batch_args = Namespace(**vars(args))
        batch_args.run_names = batch_names
        _append_log(f"launch_batch run_names={batch_names}")
        launch(batch_args)
        summary = wait_for_completion(batch_args)
        _append_log(
            f"batch_complete run_names={batch_names} all_runs_terminal={summary.get('all_runs_terminal', False)} "
            f"completed_count={summary.get('completed_count', 0)} failed_count={summary.get('failed_count', 0)}"
        )
    args.run_names = ""
    summary = summarize(args)
    if bool(summary.get("all_runs_terminal", False)):
        judge = _run_dualpanel_judge(args)
        _write_appearance_plumbing(args)
        _write_checkpoint_dualtrack(args, judge)
        diagnose(args)
    return {"generated_at_utc": now_iso(), "summary_report": str(args.summary_report), "diagnosis_report": str(args.diagnosis_report)}


def parse_args() -> Any:
    parser = ArgumentParser(description="Run STAGE2 TUSB-V3.3 selective lift")
    parser.add_argument("--mode", default="run", choices=["run", "launch", "summarize", "diagnose"])
    parser.add_argument("--run-names", default="", help="comma-separated subset of run names to operate on")
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
    parser.add_argument("--hardpanel-densified-report", default=str(ROOT / "reports/stage2_protocol_v3_hardpanel_densified_20260419.json"))
    parser.add_argument("--hardpanel-densified-doc", default=str(ROOT / "docs/STAGE2_PROTOCOL_V3_HARDPANEL_DENSIFIED_20260419.md"))
    parser.add_argument("--appearance-signal-report", default=str(ROOT / "reports/stage2_tusb_v3p2_appearance_signal_20260419.json"))
    parser.add_argument("--appearance-signal-doc", default=str(ROOT / "docs/STAGE2_TUSB_V3P2_APPEARANCE_SIGNAL_20260419.md"))
    parser.add_argument("--eval-device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--lease-path", default=str(ROOT / "reports/stage1_v2_gpu_lease_20260408.json"))
    parser.add_argument("--eval-required-mem-gb", type=float, default=24.0)
    parser.add_argument("--eval-safety-margin-gb", type=float, default=4.0)
    parser.add_argument("--protocol-report", default=str(ROOT / "reports/stage2_tusb_v3p3_rollback_anchored_protocol_20260419.json"))
    parser.add_argument("--protocol-doc", default=str(ROOT / "docs/STAGE2_TUSB_V3P3_ROLLBACK_ANCHORED_PROTOCOL_20260419.md"))
    parser.add_argument("--dualpanel-judge-report", default=str(ROOT / "reports/stage2_tusb_v3p3_dualpanel_judge_20260419.json"))
    parser.add_argument("--dualpanel-judge-doc", default=str(ROOT / "docs/STAGE2_TUSB_V3P3_DUALPANEL_JUDGE_20260419.md"))
    parser.add_argument("--appearance-plumbing-report", default=str(ROOT / "reports/stage2_tusb_v3p3_appearance_plumbing_20260419.json"))
    parser.add_argument("--appearance-plumbing-doc", default=str(ROOT / "docs/STAGE2_TUSB_V3P3_APPEARANCE_PLUMBING_20260419.md"))
    parser.add_argument("--checkpoint-dualtrack-report", default=str(ROOT / "reports/stage2_tusb_v3p3_checkpoint_dualtrack_20260419.json"))
    parser.add_argument("--checkpoint-dualtrack-doc", default=str(ROOT / "docs/STAGE2_TUSB_V3P3_CHECKPOINT_DUALTRACK_20260419.md"))
    parser.add_argument("--launch-report", default=str(ROOT / "reports/stage2_tusb_v3p3_selective_lift_launch_20260419.json"))
    parser.add_argument("--summary-report", default=str(ROOT / "reports/stage2_tusb_v3p3_selective_lift_summary_20260419.json"))
    parser.add_argument("--diagnosis-report", default=str(ROOT / "reports/stage2_tusb_v3p3_selective_lift_diagnosis_20260419.json"))
    parser.add_argument("--results-md", default=str(ROOT / "docs/STAGE2_TUSB_V3P3_SELECTIVE_LIFT_20260419.md"))
    parser.add_argument("--wait-timeout-seconds", type=int, default=172800)
    parser.add_argument("--poll-seconds", type=int, default=120)
    parser.add_argument("--max-concurrent-train-tasks", type=int, default=MAX_TRAIN_TASKS)
    parser.add_argument("--gpu-acquire-timeout-seconds", type=int, default=7200)
    parser.add_argument("--gpu-acquire-retry-seconds", type=int, default=60)
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
