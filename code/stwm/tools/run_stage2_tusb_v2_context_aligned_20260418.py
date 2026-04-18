#!/usr/bin/env python3
from __future__ import annotations

from argparse import ArgumentParser
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple
import gc
import json
import subprocess
import time

import numpy as np
import torch

from stwm.tools import run_stage2_state_identifiability_eval_20260415 as prev_eval
from stwm.tools import run_stage2_state_identifiability_eval_v3_20260416 as evalv3
from stwm.tools import run_stage2_tusb_v2_20260418 as tusbbase
from stwm.tools import run_tracewm_stage2_calibration_only_fullscale_wave1_20260413 as base
ROOT = prev_eval.ROOT
SESSION = "tracewm_stage2_tusb_v2_context_aligned_20260418"
DATE_TAG = "20260418"
LOG_PATH = ROOT / "logs/stage2_tusb_v2_context_aligned_20260418.log"
TRAIN_ADDITIONAL_STEPS = 600
EVAL_INTERVAL = 100
SAVE_EVERY = 100
MAX_TRAIN_TASKS = 4
K_CONTEXT = 8


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


def _write_protocol_artifacts(args: Any) -> None:
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
            "multi_entity_training_sample_exists": True,
            "freeze_ready_mainline": False,
        },
        "current_context_mismatch": {
            "protocol_v3_single_target_eval_underestimates_multi_entity_context": True,
            "best_pt_rollout_selection_can_suppress_semantic_gain": True,
            "vipseg_true_continuity_coverage_approx": 0.5,
            "vspw_instance_path_mode": "pseudo_or_fallback",
        },
        "goal": {
            "grounding_first_mainline": False,
            "protocol_v4": False,
            "new_method_story": False,
            "target": "repair context-preserving eval, align checkpoint selection to protocol usefulness, increase true instance signal density, then judge whether TUSB-v2 deserves scale-up",
        },
    }
    base._write_json(args.protocol_report, payload)
    base._write_md(
        args.protocol_doc,
        [
            "# Stage2 TUSB-V2 Context-Aligned Protocol 20260418",
            "",
            "- Stage1 remains frozen. No training, no unfreeze, no backbone swap.",
            "- TUSB-v2 is already landed.",
            "- anti-collapse is load-bearing.",
            "- z_sem slower_than_z_dyn = true.",
            "- multi-entity training sample path already exists.",
            "- current flat protocol-v3 result does not automatically mean TUSB-v2 is ineffective.",
            "- strong suspicion 1: protocol eval still uses single-target observed context.",
            "- strong suspicion 2: best.pt checkpoint selection is rollout-aligned and can suppress semantic usefulness gains.",
            "- current instance-aware path is still weak in the main training body: VIPSeg true continuity is only partial, VSPW is mostly pseudo/fallback.",
            "- this round only repairs context alignment, checkpoint alignment, and true instance density. No protocol v4, no persistence, no Stage1 edits.",
        ],
    )


def _ctx_meta_dir(args: Any) -> Path:
    return Path(args.work_root) / "reports/stage2_tusb_v2_context_aligned_runs_20260418"


def _ctx_paths(args: Any, run_name: str) -> Dict[str, Path]:
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
        "launch": _ctx_meta_dir(args) / f"{run_name}_launch_meta.json",
    }


def _current_tusb_resume_ckpt() -> Path:
    return ROOT / "outputs/checkpoints/stage2_tusb_v2_seed123_20260418/best.pt"


def _current_tusb_run_name() -> str:
    return "stage2_tusb_v2_seed123_20260418"


def _current_calibration_best_run() -> str:
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


def _teacher_cache_v3_root(args: Any) -> Path:
    return Path(args.work_root) / "data/processed/stage2_teacher_semantic_cache_v3_20260418"


def _build_teacher_prior_v3(args: Any) -> Dict[str, Any]:
    report = Path(args.teacher_prior_report)
    if report.exists():
        payload = _json_or_empty(report)
        if payload:
            return payload
    cmd = [
        str(args.python_bin),
        str(Path(args.work_root) / "code/stwm/tools/build_stage2_teacher_semantic_cache_v3_20260418.py"),
        "--predecode-cache-root",
        str(args.predecode_cache_path),
        "--teacher-cache-root",
        str(_teacher_cache_v3_root(args)),
        "--output-json",
        str(args.teacher_prior_report),
        "--output-md",
        str(args.teacher_prior_doc),
        "--device",
        str(args.eval_device),
    ]
    _append_log(f"teacher_prior_v3_build_start cmd={' '.join(cmd)}")
    subprocess.run(cmd, cwd=str(args.work_root), check=True)
    payload = _json_or_empty(args.teacher_prior_report)
    _append_log(f"teacher_prior_v3_build_done chosen={payload.get('chosen_teacher_prior_v3', '')}")
    return payload


def _dataset_density_stats(args: Any, dataset_names: List[str], split: str = "train") -> Dict[str, Any]:
    cache_index = _json_or_empty(Path(args.predecode_cache_path) / "index.json")
    entries = cache_index.get("entries", {}) if isinstance(cache_index.get("entries", {}), dict) else {}
    dataset_weights: Dict[str, int] = {}
    for name in dataset_names:
        key = str(name).strip().upper()
        dataset_weights[key] = int(dataset_weights.get(key, 0)) + 1
    weighted_paths: List[Path] = []
    for key, source_path in sorted(entries.items()):
        parts = str(key).split("::", 2)
        if len(parts) != 3:
            continue
        dataset_name, key_split, _ = parts
        if str(key_split).strip().lower() != str(split).strip().lower():
            continue
        weight = int(dataset_weights.get(str(dataset_name).strip().upper(), 0))
        for _ in range(max(weight, 0)):
            weighted_paths.append(Path(str(source_path)))
    true_counts: List[float] = []
    pseudo_counts: List[float] = []
    fallback_counts: List[float] = []
    ratios: List[float] = []
    entity_counts: List[float] = []
    sources: Dict[str, int] = {}
    vipseg_true = 0
    vipseg_total = 0
    for npz_path in weighted_paths:
        if not npz_path.exists():
            continue
        with np.load(npz_path, allow_pickle=True) as payload:
            meta = payload["meta_json"].item()
        source = str(meta.get("instance_source", "unknown"))
        sources[source] = sources.get(source, 0) + 1
        entity_count = float(meta.get("entity_count", 0.0) or 0.0)
        true_instance_aware = bool(meta.get("true_instance_aware", False))
        if true_instance_aware and source == "true_instance_id":
            true_count = entity_count
            pseudo_count = 0.0
            fallback_count = 0.0
        elif source.startswith("pseudo"):
            true_count = 0.0
            pseudo_count = entity_count
            fallback_count = 0.0
        else:
            true_count = 0.0
            pseudo_count = 0.0
            fallback_count = entity_count
        ratio = float(true_count / max(entity_count, 1.0))
        true_counts.append(true_count)
        pseudo_counts.append(pseudo_count)
        fallback_counts.append(fallback_count)
        ratios.append(ratio)
        entity_counts.append(entity_count)
        if str(meta.get("dataset", "")).strip().upper() == "VIPSEG":
            vipseg_total += 1
            vipseg_true += int(bool(meta.get("true_instance_aware", False) and source == "true_instance_id"))
    return {
        "dataset_names": [str(x) for x in dataset_names],
        "split": str(split),
        "sample_count": int(len(weighted_paths)),
        "true_instance_entity_count_mean": float(sum(true_counts) / max(len(true_counts), 1)),
        "pseudo_entity_count_mean": float(sum(pseudo_counts) / max(len(pseudo_counts), 1)),
        "fallback_entity_count_mean": float(sum(fallback_counts) / max(len(fallback_counts), 1)),
        "entity_count_mean": float(sum(entity_counts) / max(len(entity_counts), 1)),
        "true_instance_ratio_mean": float(sum(ratios) / max(len(ratios), 1)),
        "instance_source_counts": sources,
        "vipseg_true_instance_continuity_coverage_ratio": float(vipseg_true / max(vipseg_total, 1)),
    }


def _write_true_instance_density(args: Any) -> Dict[str, Any]:
    _append_log("true_instance_density_scan_start")
    payload = {
        "generated_at_utc": now_iso(),
        "baseline_mixed": _dataset_density_stats(args, ["vspw", "vipseg"]),
        "vipseg_upweighted": _dataset_density_stats(args, ["vipseg", "vipseg", "vspw"]),
        "vipseg_only": _dataset_density_stats(args, ["vipseg"]),
    }
    payload["instance_aware_real_signal_used_if_upweighted"] = bool(
        float(payload["vipseg_upweighted"]["true_instance_ratio_mean"]) > float(payload["baseline_mixed"]["true_instance_ratio_mean"])
    )
    base._write_json(args.true_instance_density_report, payload)
    base._write_md(
        args.true_instance_density_doc,
        [
            "# Stage2 TUSB True Instance Density 20260418",
            "",
            f"- baseline_mixed.true_instance_ratio_mean: {payload['baseline_mixed']['true_instance_ratio_mean']:.4f}",
            f"- vipseg_upweighted.true_instance_ratio_mean: {payload['vipseg_upweighted']['true_instance_ratio_mean']:.4f}",
            f"- vipseg_only.true_instance_ratio_mean: {payload['vipseg_only']['true_instance_ratio_mean']:.4f}",
            f"- instance_aware_real_signal_used_if_upweighted: {payload['instance_aware_real_signal_used_if_upweighted']}",
        ],
    )
    _append_log(
        "true_instance_density_scan_done "
        f"baseline_ratio={payload['baseline_mixed']['true_instance_ratio_mean']:.4f} "
        f"vipseg_upweight_ratio={payload['vipseg_upweighted']['true_instance_ratio_mean']:.4f} "
        f"vipseg_only_ratio={payload['vipseg_only']['true_instance_ratio_mean']:.4f}"
    )
    return payload


def _base_tusb_spec_template() -> Dict[str, Any]:
    return next(spec for spec in tusbbase._run_specs() if str(spec["run_name"]) == "stage2_tusb_v2_seed123_20260418")


def _ctx_specs() -> List[Dict[str, Any]]:
    base_spec = dict(_base_tusb_spec_template())
    return [
        {
            **base_spec,
            "run_name": "stage2_tusb_v2_ctx_vipseg_upweight_seed123_20260418",
            "family": "tusb_v2_context_aligned_density",
            "ablation_name": "vipseg_upweight",
            "objective_combo": "tusb_v2_ctx_vipseg_upweight_seed123",
            "objective_family": "trace_unit_semantic_binding_v2_context_aligned",
            "window_name": "ctx_upw",
            "dataset_names": ["vipseg", "vipseg", "vspw"],
        },
        {
            **base_spec,
            "run_name": "stage2_tusb_v2_ctx_vipseg_only_seed123_20260418",
            "family": "tusb_v2_context_aligned_density",
            "ablation_name": "vipseg_only",
            "objective_combo": "tusb_v2_ctx_vipseg_only_seed123",
            "objective_family": "trace_unit_semantic_binding_v2_context_aligned",
            "window_name": "ctx_vip",
            "dataset_names": ["vipseg"],
        },
    ]


def _ensure_tmux_session(session_name: str) -> set[str]:
    if subprocess.run(["tmux", "has-session", "-t", session_name], capture_output=True).returncode != 0:
        subprocess.run(["tmux", "new-session", "-d", "-s", session_name, "bash"], check=True)
    return set(base._tmux_windows(session_name))


def _build_launch_meta(args: Any, spec: Dict[str, Any], existing_windows: set[str]) -> Dict[str, Any]:
    ctx = tusbbase._common_launch_context(args)
    meta = tusbbase._build_launch_meta(args, spec, ctx)
    dataset_names = [str(x) for x in spec.get("dataset_names", ["vspw", "vipseg"])]
    resume_from = _current_tusb_resume_ckpt()
    resume_step = base._load_ckpt_step(resume_from)
    meta.update(
        {
            "run_name": str(spec["run_name"]),
            "family": str(spec["family"]),
            "ablation_name": str(spec["ablation_name"]),
            "objective_combo": str(spec["objective_combo"]),
            "objective_family": str(spec["objective_family"]),
            "window_name": str(spec["window_name"]),
            "dataset_names": dataset_names,
            "resume_from": str(resume_from),
            "resume_global_step": int(resume_step),
            "additional_train_steps": int(TRAIN_ADDITIONAL_STEPS),
            "train_steps": int(resume_step + TRAIN_ADDITIONAL_STEPS),
            "eval_interval": int(EVAL_INTERVAL),
            "save_every_n_steps": int(SAVE_EVERY),
            "teacher_semantic_cache_path": str(_teacher_cache_v3_root(args)),
            "predecode_cache_path": str(args.predecode_cache_path),
            "runtime_json": str(args.runtime_json),
            "max_concurrent_tusb_tasks": int(args.max_concurrent_train_tasks),
            "effective_train_sample_count_per_dataset": base._dataset_counts(dataset_names, "train", args.stage2_contract_json, max_samples=32),
            "effective_val_sample_count_per_dataset": base._dataset_counts(dataset_names, "val", args.stage2_contract_json, max_samples=32),
        }
    )
    paths = _ctx_paths(args, str(spec["run_name"]))
    meta.update(
        {
            "raw_json": str(paths["raw"]),
            "progress_json": str(paths["progress"]),
            "final_json": str(paths["final"]),
            "log_path": str(paths["log"]),
            "output_dir": str(paths["output_dir"]),
            "worker_pid_file": str(_ctx_meta_dir(args) / f"{spec['run_name']}.pid"),
            "meta_json": str(paths["launch"]),
        }
    )
    _ctx_meta_dir(args).mkdir(parents=True, exist_ok=True)
    return meta


def _launch_training_runs(args: Any) -> Dict[str, Any]:
    existing_windows = _ensure_tmux_session(str(args.tmux_session))
    cleanup = base._cleanup_stale_leases(str(args.shared_lease_path), allowed_prefixes=tusbbase.TUSB_ALLOWED_PREFIXES)
    launched: List[Dict[str, Any]] = []
    run_rows: List[Dict[str, Any]] = []
    for spec in _ctx_specs():
        run_name = str(spec["run_name"])
        paths = _ctx_paths(args, run_name)
        final_payload = _json_or_empty(paths["final"])
        if str(final_payload.get("status", "")).lower() == "completed" and paths["best"].exists() and paths["latest"].exists():
            launched.append({"run_name": run_name, "mode": "already_completed"})
            continue
        meta = _build_launch_meta(args, spec, existing_windows)
        reset = base._reset_run_artifacts(args=args, meta=meta, run_name=run_name)
        launched.append({"run_name": run_name, "reset": reset})
        base._write_json(Path(meta["meta_json"]), meta)
        cmd = tusbbase._tmux_window_command(args=args, meta_json=Path(meta["meta_json"]), meta=meta)
        if str(meta["window_name"]) in existing_windows:
            subprocess.run(["tmux", "kill-window", "-t", f"{args.tmux_session}:{meta['window_name']}"], check=False)
            existing_windows.discard(str(meta["window_name"]))
        subprocess.run(["tmux", "new-window", "-t", str(args.tmux_session), "-n", str(meta["window_name"]), cmd], check=True)
        existing_windows.add(str(meta["window_name"]))
        run_rows.append({"run_name": run_name, "window_name": str(meta["window_name"]), "dataset_names": spec["dataset_names"]})
    payload = {
        "generated_at_utc": now_iso(),
        "tmux_session": str(args.tmux_session),
        "lease_cleanup": cleanup,
        "new_training_runs": launched,
        "existing_main_runs": [
            "stage2_tusb_v2_seed123_20260418",
            "stage2_tusb_v2_seed42_20260418",
            "stage2_tusb_v2_seed456_20260418",
        ],
        "eval_only_run": "stage2_tusb_v2_ctx_bestsidecar_eval_seed123_20260418",
    }
    base._write_json(args.launch_report, payload)
    return payload


def _summary_row_for_run(args: Any, spec: Dict[str, Any]) -> Dict[str, Any]:
    run_name = str(spec["run_name"])
    paths = _ctx_paths(args, run_name)
    progress_payload = _json_or_empty(paths["progress"])
    final_payload = _json_or_empty(paths["final"])
    raw_payload = _json_or_empty(paths["raw"])
    meta = _json_or_empty(paths["launch"])
    status_info = base._status_for(
        {**meta, "window_name": str(meta.get("window_name", spec.get("window_name", ""))), "progress_json": str(paths["progress"]), "final_json": str(paths["final"])},
        session_name=str(args.tmux_session),
    )
    status = str(status_info.get("status", "launched")).lower()
    best_block = base._best_block(final_payload, raw_payload, progress_payload)
    latest_block = base._latest_block(final_payload, raw_payload, progress_payload)
    sidecar_block = base._sidecar_block(final_payload, raw_payload, progress_payload)
    trace_block = tusbbase._trace_unit_block(final_payload, raw_payload, progress_payload)
    return {
        "run_name": run_name,
        "family": str(spec["family"]),
        "ablation_name": str(spec["ablation_name"]),
        "dataset_names": [str(x) for x in spec.get("dataset_names", [])],
        "status": status,
        "best_checkpoint_metric": best_block,
        "latest_checkpoint_metric": latest_block,
        "semantic_hard_sidecar_metric": sidecar_block,
        "trace_unit_metrics": trace_block,
    }


def summarize(args: Any) -> Dict[str, Any]:
    run_rows = [_summary_row_for_run(args, spec) for spec in _ctx_specs()]
    running = sum(int(str(r["status"]) == "running") for r in run_rows)
    completed = sum(int(str(r["status"]) == "completed") for r in run_rows)
    failed = sum(int(str(r["status"]) == "failed") for r in run_rows)
    payload = {
        "generated_at_utc": now_iso(),
        "status": f"{running}_running_{completed}_completed_{failed}_failed",
        "running_count": int(running),
        "completed_count": int(completed),
        "failed_count": int(failed),
        "all_runs_terminal": bool(running == 0 and completed + failed == len(run_rows)),
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


def _context_checkpoint_specs(args: Any) -> List[prev_eval.MethodSpec]:
    specs: List[prev_eval.MethodSpec] = [
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
            run_name=_current_calibration_best_run(),
            method_type="stage2",
            checkpoint_path=str(ROOT / "outputs/checkpoints" / _current_calibration_best_run() / "best.pt"),
        ),
        prev_eval.MethodSpec(
            name="current_tusb_v2_best_pt",
            run_name=_current_tusb_run_name(),
            method_type="stage2",
            checkpoint_path=str(ROOT / "outputs/checkpoints" / _current_tusb_run_name() / "best.pt"),
        ),
        prev_eval.MethodSpec(
            name="current_tusb_v2_best_semantic_hard",
            run_name=_current_tusb_run_name(),
            method_type="stage2",
            checkpoint_path=str(ROOT / "outputs/checkpoints" / _current_tusb_run_name() / "best_semantic_hard.pt"),
        ),
        prev_eval.MethodSpec(
            name="vipseg_upweighted_tusb_v2",
            run_name="stage2_tusb_v2_ctx_vipseg_upweight_seed123_20260418",
            method_type="stage2",
            checkpoint_path=str(ROOT / "outputs/checkpoints/stage2_tusb_v2_ctx_vipseg_upweight_seed123_20260418/best.pt"),
        ),
        prev_eval.MethodSpec(
            name="vipseg_only_tusb_v2",
            run_name="stage2_tusb_v2_ctx_vipseg_only_seed123_20260418",
            method_type="stage2",
            checkpoint_path=str(ROOT / "outputs/checkpoints/stage2_tusb_v2_ctx_vipseg_only_seed123_20260418/best.pt"),
        ),
    ]
    out: List[prev_eval.MethodSpec] = []
    for spec in specs:
        if Path(spec.checkpoint_path).exists():
            out.append(spec)
    return out


def _prepare_protocol_items(
    items: List[Dict[str, Any]],
    builder: Callable[[Dict[str, Any]], Tuple[Dict[str, Any], np.ndarray, Dict[str, np.ndarray]]],
    mode_name: str,
) -> Tuple[List[Tuple[Dict[str, Any], Dict[str, Any], np.ndarray, Dict[str, np.ndarray]]], List[Dict[str, Any]], float, List[Dict[str, Any]]]:
    prepared: List[Tuple[Dict[str, Any], Dict[str, Any], np.ndarray, Dict[str, np.ndarray]]] = []
    per_item: List[Dict[str, Any]] = []
    context_counts: List[float] = []
    skipped_items: List[Dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        try:
            batch, target_future_mask, future_masks = builder(item)
        except Exception as exc:
            skipped_items.append(
                {
                    "protocol_item_id": str(item.get("protocol_item_id", "")),
                    "dataset": str(item.get("dataset", "")),
                    "clip_id": str(item.get("clip_id", "")),
                    "subset_tags": list(item.get("subset_tags", [])),
                    "target_id": str(item.get("target_id", "")),
                    "protocol_eval_mode": str(mode_name),
                    "skip_reason": str(exc),
                }
            )
            continue
        prepared.append((item, batch, target_future_mask, future_masks))
        meta_rows = batch.get("meta", [])
        batch_meta = meta_rows[0] if isinstance(meta_rows, list) and meta_rows else {}
        context_counts.append(float(batch_meta.get("protocol_eval_context_entity_count", 1.0) or 1.0))
        per_item.append(
            {
                "protocol_item_id": str(item.get("protocol_item_id", "")),
                "dataset": str(item.get("dataset", "")),
                "clip_id": str(item.get("clip_id", "")),
                "subset_tags": list(item.get("subset_tags", [])),
                "target_id": str(item.get("target_id", "")),
                "protocol_eval_mode": str(mode_name),
                "protocol_eval_context_entity_count": int(batch_meta.get("protocol_eval_context_entity_count", 1)),
                "methods": {},
            }
        )
    return prepared, per_item, float(sum(context_counts) / max(len(context_counts), 1)), skipped_items


def _evaluate_prepared(
    specs: List[prev_eval.MethodSpec],
    prepared_items: List[Tuple[Dict[str, Any], Dict[str, Any], np.ndarray, Dict[str, np.ndarray]]],
    per_item: List[Dict[str, Any]],
    device: torch.device,
) -> List[Dict[str, Any]]:
    for spec in specs:
        method = prev_eval._load_method(spec, device=device)
        try:
            for item_row, prepared in zip(per_item, prepared_items):
                item, batch, target_future_mask, future_masks = prepared
                item_row["methods"][method.name] = prev_eval._evaluate_item(
                    method=method,
                    item=item,
                    batch=batch,
                    target_future_mask=target_future_mask,
                    future_masks=future_masks,
                    device=device,
                )
        finally:
            prev_eval._release_method(method)
    return per_item


def _aggregate_eval_methods(per_item: List[Dict[str, Any]], specs: List[prev_eval.MethodSpec]) -> List[Dict[str, Any]]:
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
                "checkpoint_path": spec.checkpoint_path,
                "panels": panel_metrics,
                "query_future_top1_acc": float(panel_metrics["full_identifiability_panel"]["query_future_top1_acc"]),
                "query_future_hit_rate": float(panel_metrics["full_identifiability_panel"]["query_future_hit_rate"]),
                "query_future_localization_error": float(panel_metrics["full_identifiability_panel"]["query_future_localization_error"]),
                "future_mask_iou_at_top1": float(panel_metrics["full_identifiability_panel"]["future_mask_iou_at_top1"]),
                "hard_subset_top1_acc": float(panel_metrics["hard_subsets"]["query_future_top1_acc"]),
                "ambiguity_top1_acc": float(panel_metrics["crossing_ambiguity"]["query_future_top1_acc"]),
                "small_object_top1_acc": float(panel_metrics["small_object"]["query_future_top1_acc"]),
                "appearance_change_top1_acc": float(panel_metrics["appearance_change"]["query_future_top1_acc"]),
            }
        )
    return method_rows


def _run_eval_mode(
    args: Any,
    protocol_items: List[Dict[str, Any]],
    specs: List[prev_eval.MethodSpec],
    mode_name: str,
    builder: Callable[[Dict[str, Any]], Tuple[Dict[str, Any], np.ndarray, Dict[str, np.ndarray]]],
) -> Dict[str, Any]:
    if not hasattr(args, "lease_path") or not str(getattr(args, "lease_path", "")).strip():
        setattr(args, "lease_path", str(args.shared_lease_path))
    device, device_info = evalv3._select_eval_device_v3(args)
    try:
        prepared, per_item, context_mean, skipped_items = _prepare_protocol_items(protocol_items, builder, mode_name=mode_name)
        per_item = _evaluate_prepared(specs, prepared, per_item, device=device)
        methods = _aggregate_eval_methods(per_item, specs)
        return {
            "selected_device": str(device),
            "device_info": device_info,
            "protocol_item_count": int(len(per_item)),
            "skipped_protocol_item_count": int(len(skipped_items)),
            "skipped_protocol_items": skipped_items,
            "protocol_eval_context_entity_count_mean": float(context_mean),
            "methods": methods,
            "per_item_results": per_item,
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


def _method_by_name(methods: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {str(row.get("name", "")): row for row in methods if isinstance(row, dict)}


def _run_context_eval(args: Any) -> Dict[str, Any]:
    protocol = _json_or_empty(args.protocol_v3_json)
    items = protocol.get("items", []) if isinstance(protocol.get("items", []), list) else []
    specs = _context_checkpoint_specs(args)
    single_target = _run_eval_mode(
        args=args,
        protocol_items=items,
        specs=specs,
        mode_name="single_target",
        builder=lambda item: evalv3._build_single_item_batch_v3(item, temporal_window=5),
    )
    context_eval = _run_eval_mode(
        args=args,
        protocol_items=items,
        specs=specs,
        mode_name="context_preserving",
        builder=lambda item: evalv3._build_context_preserving_item_batch_v3(item, temporal_window=5, max_context_entities=K_CONTEXT),
    )
    single_methods = _method_by_name(single_target["methods"])
    context_methods = _method_by_name(context_eval["methods"])
    tusb_single = single_methods.get("current_tusb_v2_best_pt", {})
    tusb_context = context_methods.get("current_tusb_v2_best_pt", {})
    cal_single = single_methods.get("current_calibration_only_best", {})
    cal_context = context_methods.get("current_calibration_only_best", {})
    single_gap = float(tusb_single.get("query_future_top1_acc", 0.0) - cal_single.get("query_future_top1_acc", 0.0))
    context_gap = float(tusb_context.get("query_future_top1_acc", 0.0) - cal_context.get("query_future_top1_acc", 0.0))
    hard_single_gap = float(tusb_single.get("hard_subset_top1_acc", 0.0) - cal_single.get("hard_subset_top1_acc", 0.0))
    hard_context_gap = float(tusb_context.get("hard_subset_top1_acc", 0.0) - cal_context.get("hard_subset_top1_acc", 0.0))
    payload = {
        "generated_at_utc": now_iso(),
        "protocol_v3_single_target_eval": single_target,
        "protocol_v3_context_preserving_eval": context_eval,
        "single_target_eval_underestimates_tusb_v2": bool(context_gap > single_gap or hard_context_gap > hard_single_gap),
        "context_preserving_eval_more_aligned_with_tusb_design": bool(float(context_eval["protocol_eval_context_entity_count_mean"]) > 1.5 and (context_gap > single_gap or hard_context_gap > hard_single_gap)),
        "protocol_eval_single_vs_context_gap": {
            "tusb_vs_calibration_top1_gap_single_target": float(single_gap),
            "tusb_vs_calibration_top1_gap_context": float(context_gap),
            "tusb_vs_calibration_hard_top1_gap_single_target": float(hard_single_gap),
            "tusb_vs_calibration_hard_top1_gap_context": float(hard_context_gap),
        },
    }
    base._write_json(args.context_eval_report, payload)
    base._write_md(
        args.context_eval_doc,
        [
            "# Stage2 TUSB Context Eval 20260418",
            "",
            f"- single_target_eval_underestimates_tusb_v2: {payload['single_target_eval_underestimates_tusb_v2']}",
            f"- context_preserving_eval_more_aligned_with_tusb_design: {payload['context_preserving_eval_more_aligned_with_tusb_design']}",
            f"- protocol_eval_context_entity_count_mean: {context_eval['protocol_eval_context_entity_count_mean']:.4f}",
            f"- single_target_gap_top1: {single_gap:.6f}",
            f"- context_gap_top1: {context_gap:.6f}",
            f"- single_target_gap_hard_top1: {hard_single_gap:.6f}",
            f"- context_gap_hard_top1: {hard_context_gap:.6f}",
        ],
    )
    return payload


def _protocol_rank(row: Dict[str, Any]) -> Tuple[float, float, float, float, float, float]:
    return (
        -float(row.get("query_future_top1_acc", 0.0)),
        -float(row.get("hard_subset_top1_acc", 0.0)),
        -float(row.get("ambiguity_top1_acc", 0.0)),
        -float(row.get("small_object_top1_acc", 0.0)),
        -float(row.get("appearance_change_top1_acc", 0.0)),
        float(row.get("query_future_localization_error", 1e9)),
    )


def _run_checkpoint_alignment(args: Any) -> Dict[str, Any]:
    protocol = _json_or_empty(args.protocol_v3_json)
    items = protocol.get("items", []) if isinstance(protocol.get("items", []), list) else []
    if not hasattr(args, "lease_path") or not str(getattr(args, "lease_path", "")).strip():
        setattr(args, "lease_path", str(args.shared_lease_path))
    device, device_info = evalv3._select_eval_device_v3(args)
    try:
        prepared, per_item_proto, context_mean, skipped_items = _prepare_protocol_items(
            items,
            lambda item: evalv3._build_context_preserving_item_batch_v3(item, temporal_window=5, max_context_entities=K_CONTEXT),
            mode_name="context_preserving",
        )
        run_map = {
            "current_tusb_v2_main": "stage2_tusb_v2_seed123_20260418",
            "no_instance_path": "stage2_tusb_v2_no_instance_path_seed123_20260418",
            "no_teacher_prior": "stage2_tusb_v2_no_teacher_prior_seed123_20260418",
            "no_anticollapse": "stage2_tusb_v2_no_anticollapse_seed123_20260418",
            "vipseg_upweighted": "stage2_tusb_v2_ctx_vipseg_upweight_seed123_20260418",
            "vipseg_only": "stage2_tusb_v2_ctx_vipseg_only_seed123_20260418",
        }
        rows: List[Dict[str, Any]] = []
        for logical_name, run_name in run_map.items():
            ckpt_dir = ROOT / "outputs/checkpoints" / run_name
            ckpt_specs = []
            for ckpt_label, ckpt_name in [("best.pt", "best.pt"), ("best_semantic_hard.pt", "best_semantic_hard.pt"), ("latest.pt", "latest.pt")]:
                ckpt_path = ckpt_dir / ckpt_name
                if ckpt_path.exists():
                    ckpt_specs.append(prev_eval.MethodSpec(name=f"{logical_name}::{ckpt_label}", run_name=run_name, method_type="stage2", checkpoint_path=str(ckpt_path)))
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
            per_item = _evaluate_prepared(ckpt_specs, prepared, per_item, device=device)
            method_rows = _aggregate_eval_methods(per_item, ckpt_specs)
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
        current_main = next((row for row in rows if row["logical_name"] == "current_tusb_v2_main"), {})
        best_choice = str(current_main.get("protocol_best_checkpoint", "best.pt"))
        payload = {
            "generated_at_utc": now_iso(),
            "protocol_eval_context_entity_count_mean": float(context_mean),
            "skipped_protocol_item_count": int(len(skipped_items)),
            "skipped_protocol_items": skipped_items,
            "run_rows": rows,
            "best_tusb_v2_checkpoint_choice": best_choice,
            "best_semantic_hard_more_aligned_with_protocol": bool(best_choice == "best_semantic_hard.pt"),
            "should_continue_using_best_pt_as_only_public_checkpoint": bool(best_choice == "best.pt"),
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
    base._write_json(args.checkpoint_alignment_report, payload)
    base._write_md(
        args.checkpoint_alignment_doc,
        [
            "# Stage2 TUSB Checkpoint Selection Alignment 20260418",
            "",
            f"- best_tusb_v2_checkpoint_choice: {payload['best_tusb_v2_checkpoint_choice']}",
            f"- best_semantic_hard_more_aligned_with_protocol: {payload['best_semantic_hard_more_aligned_with_protocol']}",
            f"- should_continue_using_best_pt_as_only_public_checkpoint: {payload['should_continue_using_best_pt_as_only_public_checkpoint']}",
        ],
    )
    return payload


def _write_unitization_refine(args: Any, best_run_name: str) -> Dict[str, Any]:
    final_payload = _json_or_empty(ROOT / "reports" / f"{best_run_name}_final.json")
    trace = final_payload.get("trace_unit_metrics", {}) if isinstance(final_payload.get("trace_unit_metrics", {}), dict) else {}
    payload = {
        "generated_at_utc": now_iso(),
        "best_run_name": best_run_name,
        "active_unit_count_mean": float(trace.get("active_unit_count_mean", 0.0)),
        "assignment_entropy_mean": float(trace.get("assignment_entropy_mean", 0.0)),
        "actual_top2_assignment_ratio_mean": float(trace.get("actual_top2_assignment_ratio_mean", trace.get("actual_top2_assignment_ratio", 0.0))),
        "same_instance_within_unit_consistency": float(trace.get("same_instance_within_unit_consistency_mean", trace.get("same_instance_within_unit_consistency", 0.0))),
        "different_instance_between_unit_separation": float(trace.get("different_instance_between_unit_separation_mean", trace.get("different_instance_between_unit_separation", 0.0))),
        "structural_change_applied": False,
    }
    base._write_json(args.unitization_refine_report, payload)
    base._write_md(
        args.unitization_refine_doc,
        [
            "# Stage2 TUSB Unitization Refine 20260418",
            "",
            f"- best_run_name: {best_run_name}",
            f"- active_unit_count_mean: {payload['active_unit_count_mean']:.4f}",
            f"- assignment_entropy_mean: {payload['assignment_entropy_mean']:.6f}",
            f"- actual_top2_assignment_ratio_mean: {payload['actual_top2_assignment_ratio_mean']:.6f}",
            f"- structural_change_applied: {payload['structural_change_applied']}",
        ],
    )
    return payload


def diagnose(args: Any) -> Dict[str, Any]:
    summary = summarize(args)
    context_eval = _json_or_empty(args.context_eval_report)
    alignment = _json_or_empty(args.checkpoint_alignment_report)
    density = _json_or_empty(args.true_instance_density_report)
    current_choice = str(alignment.get("best_tusb_v2_checkpoint_choice", "best.pt"))
    context_methods = _method_by_name((((context_eval.get("protocol_v3_context_preserving_eval") or {}).get("methods")) or []))
    chosen_method_name = "current_tusb_v2_best_semantic_hard" if current_choice == "best_semantic_hard.pt" else "current_tusb_v2_best_pt"
    chosen = context_methods.get(chosen_method_name, {})
    cal = context_methods.get("current_calibration_only_best", {})
    vipseg_up = context_methods.get("vipseg_upweighted_tusb_v2", {})
    vipseg_only = context_methods.get("vipseg_only_tusb_v2", {})

    improved_vs_cal = bool(
        chosen
        and cal
        and float(chosen.get("query_future_top1_acc", -1.0)) > float(cal.get("query_future_top1_acc", -1.0))
        and float(chosen.get("future_mask_iou_at_top1", -1.0)) >= float(cal.get("future_mask_iou_at_top1", -1.0))
    )
    hard_improved = bool(
        chosen
        and cal
        and float(chosen.get("hard_subset_top1_acc", -1.0)) > float(cal.get("hard_subset_top1_acc", -1.0))
        and float(chosen.get("ambiguity_top1_acc", -1.0)) >= float(cal.get("ambiguity_top1_acc", -1.0))
    )
    best_run_name = "stage2_tusb_v2_seed123_20260418"
    if vipseg_up and _protocol_rank(vipseg_up) < _protocol_rank(chosen if chosen else {"query_future_localization_error": 1e9}):
        best_run_name = "stage2_tusb_v2_ctx_vipseg_upweight_seed123_20260418"
        chosen = vipseg_up
    elif vipseg_only and _protocol_rank(vipseg_only) < _protocol_rank(chosen if chosen else {"query_future_localization_error": 1e9}):
        best_run_name = "stage2_tusb_v2_ctx_vipseg_only_seed123_20260418"
        chosen = vipseg_only

    best_final = _json_or_empty(ROOT / "reports" / f"{best_run_name}_final.json")
    trace = best_final.get("trace_unit_metrics", {}) if isinstance(best_final.get("trace_unit_metrics", {}), dict) else {}
    z_dyn = float(trace.get("z_dyn_drift_mean", 1e9))
    z_sem = float(trace.get("z_sem_drift_mean", 1e9))
    z_sem_slower = bool(z_sem < z_dyn)
    instance_signal_used = bool(float(density.get("vipseg_upweighted", {}).get("true_instance_ratio_mean", 0.0)) > float(density.get("baseline_mixed", {}).get("true_instance_ratio_mean", 0.0)))
    thickness = "cvpr_borderline"
    if improved_vs_cal and hard_improved and instance_signal_used:
        thickness = "stronger_cvpr_or_eccv_main_candidate"
    if improved_vs_cal and hard_improved and instance_signal_used and current_choice == "best_semantic_hard.pt":
        next_step = "freeze_tusb_v2_context_aligned_as_new_stage2_mainline"
    elif improved_vs_cal or hard_improved or current_choice == "best_semantic_hard.pt" or instance_signal_used:
        next_step = "keep_tusb_v2_but_refine_teacher_or_protocol_alignment"
    else:
        next_step = "rethink_stage2_story_if_context_preserving_eval_still_flat"
    payload = {
        "generated_at_utc": now_iso(),
        "best_tusb_v2_run_name": best_run_name,
        "best_tusb_v2_checkpoint_choice": current_choice,
        "single_target_eval_underestimates_tusb_v2": bool(context_eval.get("single_target_eval_underestimates_tusb_v2", False)),
        "context_preserving_protocol_improved_vs_current_calonly": bool(improved_vs_cal),
        "hard_subsets_improved": bool(hard_improved),
        "rollout_best_and_protocol_best_match": bool(not alignment.get("best_semantic_hard_more_aligned_with_protocol", False)),
        "best_semantic_hard_more_aligned_with_protocol": bool(alignment.get("best_semantic_hard_more_aligned_with_protocol", False)),
        "instance_aware_real_signal_used": bool(instance_signal_used),
        "vipseg_upweighted_clearer_gain": bool(vipseg_up and _protocol_rank(vipseg_up) < _protocol_rank(cal if cal else {"query_future_localization_error": 1e9})),
        "vipseg_only_clearer_gain": bool(vipseg_only and _protocol_rank(vipseg_only) < _protocol_rank(cal if cal else {"query_future_localization_error": 1e9})),
        "z_sem_slower_than_z_dyn": bool(z_sem_slower),
        "current_paper_thickness_level": thickness,
        "next_step_choice": next_step,
    }
    base._write_json(args.diagnosis_report, payload)
    base._write_md(
        args.results_md,
        [
            "# Stage2 TUSB-V2 Context-Aligned 20260418",
            "",
            f"- best_tusb_v2_run_name: {payload['best_tusb_v2_run_name']}",
            f"- best_tusb_v2_checkpoint_choice: {payload['best_tusb_v2_checkpoint_choice']}",
            f"- context_preserving_protocol_improved_vs_current_calonly: {payload['context_preserving_protocol_improved_vs_current_calonly']}",
            f"- hard_subsets_improved: {payload['hard_subsets_improved']}",
            f"- best_semantic_hard_more_aligned_with_protocol: {payload['best_semantic_hard_more_aligned_with_protocol']}",
            f"- instance_aware_real_signal_used: {payload['instance_aware_real_signal_used']}",
            f"- z_sem_slower_than_z_dyn: {payload['z_sem_slower_than_z_dyn']}",
            f"- current_paper_thickness_level: {payload['current_paper_thickness_level']}",
            f"- next_step_choice: {payload['next_step_choice']}",
        ],
    )
    return payload


def run_all(args: Any) -> Dict[str, Any]:
    _write_protocol_artifacts(args)
    teacher_prior = _build_teacher_prior_v3(args)
    density = _write_true_instance_density(args)
    _launch_training_runs(args)
    summary = wait_for_completion(args)
    context_eval = _run_context_eval(args)
    alignment = _run_checkpoint_alignment(args)
    best_run_name = "stage2_tusb_v2_seed123_20260418"
    _write_unitization_refine(args, best_run_name=best_run_name)
    diagnosis = diagnose(args)
    return {
        "generated_at_utc": now_iso(),
        "teacher_prior": teacher_prior.get("chosen_teacher_prior_v3", ""),
        "summary_status": summary.get("status", ""),
        "context_eval_report": str(args.context_eval_report),
        "checkpoint_alignment_report": str(args.checkpoint_alignment_report),
        "diagnosis_report": str(args.diagnosis_report),
    }


def parse_args() -> Any:
    parser = ArgumentParser(description="Run STAGE2 TUSB-V2 context-aligned repair")
    parser.add_argument("--mode", default="run", choices=["run", "launch", "summarize", "diagnose"])
    parser.add_argument("--work-root", default=str(ROOT))
    parser.add_argument("--tmux-session", default=SESSION)
    parser.add_argument("--python-bin", default=str(base._python_bin_default()))
    parser.add_argument("--stage2-contract-json", default=str(ROOT / "reports/stage2_bootstrap_data_contract_20260408.json"))
    parser.add_argument("--stage1-best-ckpt", default=str(ROOT / "outputs/checkpoints/stage1_v2_longtrain_220m_mainline_20260408/best.pt"))
    parser.add_argument("--shared-lease-path", default=str(ROOT / "reports/stage1_v2_gpu_lease_20260408.json"))
    parser.add_argument("--bootstrap-cache-jsonl", default=str(ROOT / "data/processed/stage2_real_bootstrap_cache_20260410/clip_vit_b32_core_trainval_required_subset.jsonl"))
    parser.add_argument("--semantic-hard-manifest-path", default=str(ROOT / "manifests/protocol_v2/stage2_semantic_hard_subsets_20260410.json"))
    parser.add_argument("--runtime-json", default=str(ROOT / "configs/recommended_stage2_runtime_tusb_v2_20260418.json"))
    parser.add_argument("--predecode-cache-path", default=str(ROOT / "data/processed/stage2_tusb_v2_predecode_cache_20260418"))
    parser.add_argument("--teacher-semantic-cache-path", default=str(ROOT / "data/processed/stage2_teacher_semantic_cache_v3_20260418"))
    parser.add_argument("--protocol-v3-json", default=str(ROOT / "reports/stage2_state_identifiability_protocol_v3_20260416.json"))
    parser.add_argument("--eval-device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--context-eval-report", default=str(ROOT / "reports/stage2_tusb_context_eval_20260418.json"))
    parser.add_argument("--context-eval-doc", default=str(ROOT / "docs/STAGE2_TUSB_CONTEXT_EVAL_20260418.md"))
    parser.add_argument("--checkpoint-alignment-report", default=str(ROOT / "reports/stage2_tusb_checkpoint_selection_alignment_20260418.json"))
    parser.add_argument("--checkpoint-alignment-doc", default=str(ROOT / "docs/STAGE2_TUSB_CHECKPOINT_SELECTION_ALIGNMENT_20260418.md"))
    parser.add_argument("--true-instance-density-report", default=str(ROOT / "reports/stage2_tusb_true_instance_density_20260418.json"))
    parser.add_argument("--true-instance-density-doc", default=str(ROOT / "docs/STAGE2_TUSB_TRUE_INSTANCE_DENSITY_20260418.md"))
    parser.add_argument("--teacher-prior-report", default=str(ROOT / "reports/stage2_tusb_teacher_prior_v3_20260418.json"))
    parser.add_argument("--teacher-prior-doc", default=str(ROOT / "docs/STAGE2_TUSB_TEACHER_PRIOR_V3_20260418.md"))
    parser.add_argument("--unitization-refine-report", default=str(ROOT / "reports/stage2_tusb_unitization_refine_20260418.json"))
    parser.add_argument("--unitization-refine-doc", default=str(ROOT / "docs/STAGE2_TUSB_UNITIZATION_REFINE_20260418.md"))
    parser.add_argument("--protocol-report", default=str(ROOT / "reports/stage2_tusb_v2_context_aligned_protocol_20260418.json"))
    parser.add_argument("--protocol-doc", default=str(ROOT / "docs/STAGE2_TUSB_V2_CONTEXT_ALIGNED_PROTOCOL_20260418.md"))
    parser.add_argument("--launch-report", default=str(ROOT / "reports/stage2_tusb_v2_context_aligned_launch_20260418.json"))
    parser.add_argument("--summary-report", default=str(ROOT / "reports/stage2_tusb_v2_context_aligned_summary_20260418.json"))
    parser.add_argument("--diagnosis-report", default=str(ROOT / "reports/stage2_tusb_v2_context_aligned_diagnosis_20260418.json"))
    parser.add_argument("--results-md", default=str(ROOT / "docs/STAGE2_TUSB_V2_CONTEXT_ALIGNED_20260418.md"))
    parser.add_argument("--wait-timeout-seconds", type=int, default=172800)
    parser.add_argument("--poll-seconds", type=int, default=120)
    parser.add_argument("--max-concurrent-train-tasks", type=int, default=MAX_TRAIN_TASKS)
    parser.add_argument("--max-concurrent-tusb-tasks", type=int, default=MAX_TRAIN_TASKS)
    parser.add_argument("--gpu-acquire-timeout-seconds", type=int, default=7200)
    parser.add_argument("--gpu-acquire-retry-seconds", type=int, default=60)
    parser.add_argument("--eval-required-mem-gb", type=float, default=24.0)
    parser.add_argument("--eval-safety-margin-gb", type=float, default=4.0)
    args = parser.parse_args()
    # The reused TUSB launcher expects these legacy names.
    if not hasattr(args, "max_concurrent_tusb_tasks"):
        setattr(args, "max_concurrent_tusb_tasks", int(args.max_concurrent_train_tasks))
    if not hasattr(args, "gpu_acquire_timeout_seconds"):
        setattr(args, "gpu_acquire_timeout_seconds", int(args.gpu_acquire_timeout_seconds))
    if not hasattr(args, "gpu_acquire_retry_seconds"):
        setattr(args, "gpu_acquire_retry_seconds", int(args.gpu_acquire_retry_seconds))
    if not hasattr(args, "device"):
        setattr(args, "device", str(args.eval_device))
    return args


def main() -> None:
    base._apply_process_title_normalization()
    args = parse_args()
    if args.mode == "run":
        print(json.dumps(run_all(args), ensure_ascii=True, indent=2))
    elif args.mode == "launch":
        print(json.dumps(_launch_training_runs(args), ensure_ascii=True, indent=2))
    elif args.mode == "summarize":
        print(json.dumps(summarize(args), ensure_ascii=True, indent=2))
    elif args.mode == "diagnose":
        print(json.dumps(diagnose(args), ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
