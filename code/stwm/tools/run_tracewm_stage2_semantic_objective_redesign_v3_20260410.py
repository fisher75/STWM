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
import sys
import time
import traceback

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
SESSION = "tracewm_stage2_semantic_objective_redesign_v3_20260410"
LOG_PATH = WORK_ROOT / "logs/tracewm_stage2_semantic_objective_redesign_v3_20260410.log"
PILOT_EXTRA_STEPS = 300
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


def _run_specs() -> List[Dict[str, Any]]:
    return [
        {
            "run_name": "stage2_semobjv3_confalign_seed42_20260410",
            "seed": 42,
            "objective_combo": "confidence_gated_readout_alignment",
            "semantic_rescue_mode": "v3confalign",
            "semantic_rescue_weight": 0.0002,
            "confidence_gated_alignment_loss_weight": 1.0,
            "sparse_persistence_contrastive_loss_weight": 0.0,
            "confidence_gating_margin_threshold": 0.10,
            "confidence_gating_temperature": 0.05,
            "semantic_hard_score_threshold": 0.25,
            "aux_loss_delay_steps": 0,
            "aux_loss_ramp_steps": 0,
            "semantic_hard_sidecar_enabled": False,
            "window_name": "semobjv3_ca42",
        },
        {
            "run_name": "stage2_semobjv3_confpersist_seed42_20260410",
            "seed": 42,
            "objective_combo": "confidence_gated_readout_alignment+sparse_persistence_contrastive_loss",
            "semantic_rescue_mode": "v3confpersist",
            "semantic_rescue_weight": 0.0002,
            "confidence_gated_alignment_loss_weight": 1.0,
            "sparse_persistence_contrastive_loss_weight": 0.05,
            "confidence_gating_margin_threshold": 0.10,
            "confidence_gating_temperature": 0.05,
            "semantic_hard_score_threshold": 0.25,
            "aux_loss_delay_steps": 0,
            "aux_loss_ramp_steps": 0,
            "semantic_hard_sidecar_enabled": False,
            "window_name": "semobjv3_cp42",
        },
        {
            "run_name": "stage2_semobjv3_confpersistdelay_seed42_20260410",
            "seed": 42,
            "objective_combo": "confidence_gated_readout_alignment+sparse_persistence_contrastive_loss+delayed_aux_loss_schedule",
            "semantic_rescue_mode": "v3confpersistdelay",
            "semantic_rescue_weight": 0.0002,
            "confidence_gated_alignment_loss_weight": 1.0,
            "sparse_persistence_contrastive_loss_weight": 0.05,
            "confidence_gating_margin_threshold": 0.10,
            "confidence_gating_temperature": 0.05,
            "semantic_hard_score_threshold": 0.25,
            "aux_loss_delay_steps": 50,
            "aux_loss_ramp_steps": 150,
            "semantic_hard_sidecar_enabled": False,
            "window_name": "semobjv3_cpd42",
        },
        {
            "run_name": "stage2_semobjv3_confpersistdelay_seed123_20260410",
            "seed": 123,
            "objective_combo": "confidence_gated_readout_alignment+sparse_persistence_contrastive_loss+delayed_aux_loss_schedule",
            "semantic_rescue_mode": "v3confpersistdelay",
            "semantic_rescue_weight": 0.0002,
            "confidence_gated_alignment_loss_weight": 1.0,
            "sparse_persistence_contrastive_loss_weight": 0.05,
            "confidence_gating_margin_threshold": 0.10,
            "confidence_gating_temperature": 0.05,
            "semantic_hard_score_threshold": 0.25,
            "aux_loss_delay_steps": 50,
            "aux_loss_ramp_steps": 150,
            "semantic_hard_sidecar_enabled": False,
            "window_name": "semobjv3_cpd123",
        },
        {
            "run_name": "stage2_semobjv3_confhardsidecar_seed42_20260410",
            "seed": 42,
            "objective_combo": "confidence_gated_readout_alignment+sparse_persistence_contrastive_loss+delayed_aux_loss_schedule+semantic_hard_best_sidecar_selection",
            "semantic_rescue_mode": "v3confhardsidecar",
            "semantic_rescue_weight": 0.0002,
            "confidence_gated_alignment_loss_weight": 1.0,
            "sparse_persistence_contrastive_loss_weight": 0.05,
            "confidence_gating_margin_threshold": 0.10,
            "confidence_gating_temperature": 0.05,
            "semantic_hard_score_threshold": 0.25,
            "aux_loss_delay_steps": 50,
            "aux_loss_ramp_steps": 150,
            "semantic_hard_sidecar_enabled": True,
            "window_name": "semobjv3_chs42",
        },
    ]


def _semantic_hard_composite(values: List[float]) -> float:
    if not values:
        return 1e9
    return float(sum(values) / max(len(values), 1))


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
    return min(_semantic_hard_composite(vals) for vals in per_run.values())


def write_protocol_doc(args: Any) -> None:
    _write_md(
        args.v3_protocol_doc,
        [
            "# Stage2 Semantic Objective Redesign V3 Protocol",
            "",
            f"- generated_at_utc: {now_iso()}",
            "- stage1_mutation_allowed: false",
            "- main_task: future trace / future state generation",
            "- teacher_as_mainline_semantic_source: false",
            "- v1_failure_summary: directly stacking semantic rescue losses hurt rollout optimum.",
            "- v2_summary: readout-side alignment plus persistence ranking was directionally right but did not create a true new global best.",
            "- v3_core_principles: semantics supervise and calibrate, not overwrite dynamics; semantic intervention is selective, confidence-aware, and hard-case-focused; semantic-hard evaluation is a first-class criterion.",
            "- forbidden: teacher semantic token replacement; stronger fusion trick; codec/VAE mainline replacement; full-scale long train; batch/lr sweep; DDP retrofit.",
        ],
    )


def launch(args: Any) -> Dict[str, Any]:
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
        gpu = _select_gpu(run_name=run_name, lease_path=str(args.shared_lease_path))
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
        meta_json = Path(args.work_root) / "reports" / "stage2_semantic_objective_redesign_v3_runs_20260410" / f"{run_name}_launch_meta.json"
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
            f"{shlex.quote(str(Path(args.work_root) / 'code/stwm/tools/run_tracewm_stage2_semantic_objective_redesign_v3_20260410.py'))} "
            f"--mode run-one --meta-json {shlex.quote(str(meta_json))}"
        )
        subprocess.run(["tmux", "new-window", "-t", str(args.tmux_session), "-n", str(meta["window_name"]), cmd], check=True)

    payload = {
        "generated_at_utc": now_iso(),
        "mode": "stage2_semantic_objective_redesign_v3_launch",
        "tmux_session": str(args.tmux_session),
        "pilot_policy": "bounded pilot; no Stage1 training; teacher is target only; no codec upgrade wave0",
        "runs": runs,
    }
    _write_json(args.v3_launch_report, payload)
    return summarize(args)


def run_one(args: Any) -> None:
    meta = _read_json(args.meta_json)
    lease_id = str(meta.get("lease_id", ""))
    lease_path = str(meta.get("shared_lease_path", ""))
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
        "--confidence-gated-alignment-loss-weight", str(meta["confidence_gated_alignment_loss_weight"]),
        "--sparse-persistence-contrastive-loss-weight", str(meta["sparse_persistence_contrastive_loss_weight"]),
        "--confidence-gating-margin-threshold", str(meta["confidence_gating_margin_threshold"]),
        "--confidence-gating-temperature", str(meta["confidence_gating_temperature"]),
        "--semantic-hard-score-threshold", str(meta["semantic_hard_score_threshold"]),
        "--aux-loss-delay-steps", str(meta["aux_loss_delay_steps"]),
        "--aux-loss-ramp-steps", str(meta["aux_loss_ramp_steps"]),
        "--semantic-hard-manifest-path", str(meta["semantic_hard_manifest_path"]),
        "--resume-from", str(meta["resume_from"]),
        "--skip-resume-optimizer",
        "--output-dir", str(meta["output_dir"]),
        "--run-name", str(meta["run_name"]),
        "--run-summary-json", str(meta["raw_json"]),
        "--progress-json", str(meta["progress_json"]),
        "--seed", str(meta["seed"]),
    ]
    if bool(meta.get("semantic_hard_sidecar_enabled", False)):
        cmd.append("--semantic-hard-sidecar-enabled")
    try:
        proc = subprocess.run(cmd, cwd=str(meta["work_root"]), text=True, capture_output=True, env=os.environ.copy())
        Path(str(meta["log_path"])).write_text(proc.stdout + ("\n" if proc.stdout else "") + proc.stderr, encoding="utf-8")
        if proc.returncode != 0:
            _write_json(meta["final_json"], {"generated_at_utc": now_iso(), "run_name": meta["run_name"], "status": "failed", "returncode": proc.returncode, "stderr_tail": proc.stderr[-4000:], "stdout_tail": proc.stdout[-4000:]})
            raise RuntimeError(f"trainer failed rc={proc.returncode}")
        raw = _read_json(meta["raw_json"])
        raw.update({
            "generated_at_utc": now_iso(),
            "status": "completed",
            "selected_gpu_id": int(meta["selected_gpu_id"]),
            "lease_id": str(meta["lease_id"]),
            "objective_combo": str(meta["objective_combo"]),
            "resume_global_step": int(meta["resume_global_step"]),
        })
        _write_json(meta["final_json"], raw)
    except Exception as exc:
        _write_json(meta["final_json"], {"generated_at_utc": now_iso(), "run_name": str(meta.get("run_name", "")), "status": "failed", "message": str(exc), "traceback": traceback.format_exc()})
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
    launch = _read_json(args.v3_launch_report)
    rows: List[Dict[str, Any]] = []
    running = completed = failed = 0
    for meta in launch.get("runs", []) if isinstance(launch.get("runs", []), list) else []:
        status_info = _status_for(meta, session_name=str(args.tmux_session))
        status = str(status_info.get("status", "launched"))
        running += int(status == "running")
        completed += int(status == "completed")
        failed += int(status == "failed")
        detail = status_info.get("detail", {}) if isinstance(status_info.get("detail", {}), dict) else {}
        sidecar = detail.get("semantic_hard_sidecar_metric", {}) if isinstance(detail.get("semantic_hard_sidecar_metric", {}), dict) else {}
        branch = detail.get("semantic_branch_metrics", {}) if isinstance(detail.get("semantic_branch_metrics", {}), dict) else {}
        rows.append({
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
            "confidence_gated_alignment_loss_weight": float(meta.get("confidence_gated_alignment_loss_weight", 0.0)),
            "sparse_persistence_contrastive_loss_weight": float(meta.get("sparse_persistence_contrastive_loss_weight", 0.0)),
            "aux_loss_delay_steps": int(meta.get("aux_loss_delay_steps", 0)),
            "aux_loss_ramp_steps": int(meta.get("aux_loss_ramp_steps", 0)),
            "semantic_hard_sidecar_enabled": bool(meta.get("semantic_hard_sidecar_enabled", False)),
            "confidence_gated_affected_sample_ratio_mean": float(branch.get("confidence_gated_affected_sample_ratio_mean", 0.0)),
            "semantic_hard_sidecar_score": float(sidecar.get("semantic_hard_sidecar_score", 1e9)),
            "final_json": str(meta.get("final_json", "")),
            "best_checkpoint": str(Path(str(meta.get("output_dir", ""))) / "best.pt"),
            "latest_checkpoint": str(Path(str(meta.get("output_dir", ""))) / "latest.pt"),
            "best_semantic_hard_checkpoint": str(Path(str(meta.get("output_dir", ""))) / "best_semantic_hard.pt"),
            "best_checkpoint_metric": detail.get("best_checkpoint_metric", {}) if isinstance(detail.get("best_checkpoint_metric", {}), dict) else {},
            "latest_checkpoint_metric": detail.get("latest_checkpoint_metric", {}) if isinstance(detail.get("latest_checkpoint_metric", {}), dict) else {},
            "semantic_hard_sidecar_metric": sidecar,
        })
    payload = {
        "generated_at_utc": now_iso(),
        "redesign_v3_status": f"{running}_running_{completed}_completed_{failed}_failed",
        "runs": rows,
        "next_step_choice_internal": "summarize_redesign_v3_after_completion" if completed == len(rows) and rows and failed == 0 else ("fix_failed_v3_runs" if failed else "continue_redesign_v3"),
    }
    _write_json(args.v3_summary_report, payload)
    lines = [
        "# Stage2 Semantic Objective Redesign V3 Results",
        "",
        f"- generated_at_utc: {payload['generated_at_utc']}",
        f"- redesign_v3_status: {payload['redesign_v3_status']}",
        "",
        "| run_name | combo | gpu | batch | steps | status | best_endpoint_l2 | hard_sidecar | conf_gate_ratio |",
        "|---|---|---:|---:|---:|---|---:|---:|---:|",
    ]
    for row in rows:
        best = row.get("best_checkpoint_metric", {}).get("metrics", {}) if isinstance(row.get("best_checkpoint_metric", {}), dict) else {}
        lines.append(
            f"| {row['run_name']} | {row['objective_combo']} | {row['selected_gpu_id']} | {row['batch_size']} | {row['train_steps']} | {row['status']} | {_f(best.get('free_rollout_endpoint_l2'), 1e9):.8f} | {float(row.get('semantic_hard_sidecar_score', 1e9)):.8f} | {float(row.get('confidence_gated_affected_sample_ratio_mean', 0.0)):.4f} |"
        )
    _write_md(args.v3_results_md, lines)
    return payload


def wait_for_completion(args: Any) -> Dict[str, Any]:
    deadline = time.time() + float(args.wait_timeout_seconds)
    last = summarize(args)
    while time.time() < deadline:
        if str(last.get("redesign_v3_status", "")).startswith("0_running_"):
            return last
        time.sleep(float(args.poll_seconds))
        last = summarize(args)
    last["timed_out_waiting_for_completion"] = True
    _write_json(args.v3_summary_report, last)
    return last


def diagnose_v3(args: Any) -> Dict[str, Any]:
    summary = _read_json(args.v3_summary_report)
    rows = [r for r in summary.get("runs", []) if isinstance(r, dict)]
    completed = [r for r in rows if str(r.get("status", "")) == "completed"]
    refs = _baseline_refs()
    v2 = _v2_refs()
    crop_ep = _f(refs["cropenc_fullscale_mean"].get("free_rollout_endpoint_l2", {}).get("mean"), 1e9)
    legacy_ep = _f(refs["legacysem_fullscale_mean"].get("free_rollout_endpoint_l2", {}).get("mean"), 1e9)
    v2_best_full_ep = _f(v2.get("success_criteria", {}).get("best_v2_full_validation_endpoint_l2"), 1e9)
    v2_best_hard_comp = _v2_best_hard_composite(v2)
    payload: Dict[str, Any] = {
        "generated_at_utc": now_iso(),
        "diagnosis_type": "stage2_semantic_objective_redesign_v3",
        "teacher_as_mainline_semantic_source": False,
        "chosen_bootstrap_backend": "local_clip_vit_b32_mask_crop_visual_teacher",
        "v3_runs_terminal": bool(rows and len(completed) == len(rows)),
        "baseline_refs": {
            "current_cropenc_fullscale_mean_endpoint_l2": float(crop_ep),
            "legacysem_fullscale_mean_endpoint_l2": float(legacy_ep),
            "v2_best_objective_combo_endpoint_l2": float(v2_best_full_ep),
            "v2_best_semantic_hard_composite_score": float(v2_best_hard_comp),
        },
        "full_validation_panel": {},
        "semantic_hard_subset_panel": {},
        "burst_persistence_hard_panel": {},
        "semantic_hard_composite_panel": {},
        "success_criteria": {},
    }
    if not payload["v3_runs_terminal"]:
        payload["success_criteria"] = {
            "true_new_best_not_warm_start_inherited": False,
            "semantic_hard_composite_improved_vs_v2": False,
            "cross_seed_support_present": False,
            "full_validation_non_catastrophic": False,
            "best_v3_objective_combo": "none",
            "overall_best_and_semantic_hard_best_diverged": False,
            "next_step_choice": "objective_family_still_not_working_under_current_stage2_design",
            "reason": "v3_runs_not_terminal",
        }
        _write_json(args.v3_diagnosis_report, payload)
        return payload

    eval_gpu = _select_eval_gpu(args)
    if int(eval_gpu.get("selected_gpu_id", -1)) >= 0 and not os.environ.get("CUDA_VISIBLE_DEVICES", "").strip():
        os.environ["CUDA_VISIBLE_DEVICES"] = str(eval_gpu["selected_gpu_id"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_aux: Dict[str, Dict[str, Any]] = {}
    try:
        print(f"[semobjv3] diagnosis device={device} eval_gpu={eval_gpu}", flush=True)
        stage1_model, _ = _load_stage1_model(device=device, stage1_ckpt=str(args.stage1_best_ckpt))
        full_ds = _make_dataset(["vspw", "vipseg"], "val", str(args.stage2_contract_json), max_samples=-1)
        core_ds = _make_dataset(["vspw", "vipseg"], "val", str(args.stage2_contract_json), max_samples=-1)
        subset_manifest = _read_json(args.semantic_hard_manifest_path)
        loaded: Dict[str, Tuple[Any, Any, Any, str]] = {}
        full_rows: List[Dict[str, Any]] = []
        for row in completed:
            run_name = str(row["run_name"])
            print(f"[semobjv3] full validation eval run={run_name}", flush=True)
            loaded[run_name] = _load_stage2_modules(str(row["best_checkpoint"]), device, stage1_model)
            metrics = _evaluate_loaded_stage2(loaded=loaded[run_name], stage1_model=stage1_model, dataset=full_ds, device=device)
            full_rows.append({
                "run_name": run_name,
                "objective_combo": str(row.get("objective_combo", "")),
                "seed": int(row.get("seed", -1)),
                "metrics": {k: _f(metrics.get(k), 1e9) for k in METRIC_KEYS},
            })
            final = _read_json(str(row["final_json"]))
            sidecar = final.get("semantic_hard_sidecar_metric", {}) if isinstance(final.get("semantic_hard_sidecar_metric", {}), dict) else {}
            branch = final.get("semantic_branch_metrics", {}) if isinstance(final.get("semantic_branch_metrics", {}), dict) else {}
            run_aux[run_name] = {
                "confidence_gated_affected_sample_ratio": float(branch.get("confidence_gated_affected_sample_ratio_mean", 0.0)),
                "low_confidence_sample_ratio": float(branch.get("low_confidence_sample_ratio_mean", 0.0)),
                "effective_pair_coverage_ratio": float(branch.get("effective_pair_coverage_ratio_mean", 0.0)),
                "positive_pair_count_mean": float(branch.get("positive_pair_count_mean", 0.0)),
                "hard_negative_count_mean": float(branch.get("hard_negative_count_mean", 0.0)),
                "semantic_hard_sidecar_metric": sidecar,
                "resume_global_step": int(final.get("resume_global_step", row.get("train_steps", 0) - PILOT_EXTRA_STEPS)),
                "overall_best_step": int(final.get("best_checkpoint_metric", {}).get("global_step", -1)),
            }
        payload["full_validation_panel"] = {"runs": full_rows, "aggregate": _aggregate_rows(full_rows), "dataset_binding": ["VSPW", "VIPSeg"], "eval_scope": "full_validation"}

        hard_panels: Dict[str, Any] = {}
        run_to_hard_values: Dict[str, List[float]] = {str(row["run_name"]): [] for row in completed}
        for subset_name in ["occlusion_reappearance", "crossing_or_interaction_ambiguity", "small_object_or_low_area", "appearance_change_or_semantic_shift"]:
            print(f"[semobjv3] semantic hard eval subset={subset_name}", flush=True)
            subset = subset_manifest.get("subsets", {}).get(subset_name, {})
            indices = [int(x["dataset_index"]) for x in subset.get("items", []) if isinstance(x, dict) and "dataset_index" in x]
            eval_ds = Subset(core_ds, indices)
            subset_rows: List[Dict[str, Any]] = []
            for row in completed:
                metrics = _evaluate_loaded_stage2(loaded=loaded[str(row["run_name"])], stage1_model=stage1_model, dataset=eval_ds, device=device)
                subset_rows.append({
                    "run_name": str(row["run_name"]),
                    "objective_combo": str(row.get("objective_combo", "")),
                    "seed": int(row.get("seed", -1)),
                    "metrics": {k: _f(metrics.get(k), 1e9) for k in METRIC_KEYS},
                })
                run_to_hard_values[str(row["run_name"])].append(_f(metrics.get("free_rollout_endpoint_l2"), 1e9))
            hard_panels[subset_name] = {
                "clip_count": int(len(indices)),
                "runs": subset_rows,
                "aggregate": _aggregate_rows(subset_rows),
                "cropenc_baseline_aggregate": refs.get("hard_subset_panels", {}).get(subset_name, {}).get("families", {}).get("cropenc", {}).get("aggregate", {}),
                "legacysem_baseline_aggregate": refs.get("hard_subset_panels", {}).get(subset_name, {}).get("families", {}).get("legacysem", {}).get("aggregate", {}),
                "v2_best_reference": v2.get("semantic_hard_subset_panel", {}).get(subset_name, {}),
            }
        payload["semantic_hard_subset_panel"] = hard_panels
        payload["semantic_hard_composite_panel"] = {
            "runs": [
                {
                    "run_name": run_name,
                    "objective_combo": next((str(x.get("objective_combo", "")) for x in completed if str(x.get("run_name", "")) == run_name), ""),
                    "seed": next((int(x.get("seed", -1)) for x in completed if str(x.get("run_name", "")) == run_name), -1),
                    "semantic_hard_composite_score": float(_semantic_hard_composite(vals)),
                    "confidence_gated_affected_sample_ratio": float(run_aux.get(run_name, {}).get("confidence_gated_affected_sample_ratio", 0.0)),
                }
                for run_name, vals in run_to_hard_values.items()
            ]
        }

        try:
            print("[semobjv3] optional BURST persistence-hard eval", flush=True)
            burst_ds = _make_dataset(["burst"], "val", str(args.stage2_contract_json), max_samples=-1)
            burst_subset = subset_manifest.get("subsets", {}).get("burst_persistence_stress", {})
            indices = [int(x["dataset_index"]) for x in burst_subset.get("items", []) if isinstance(x, dict) and "dataset_index" in x]
            eval_ds = Subset(burst_ds, indices)
            burst_rows = []
            for row in completed:
                metrics = _evaluate_loaded_stage2(loaded=loaded[str(row["run_name"])], stage1_model=stage1_model, dataset=eval_ds, device=device)
                burst_rows.append({
                    "run_name": str(row["run_name"]),
                    "objective_combo": str(row.get("objective_combo", "")),
                    "seed": int(row.get("seed", -1)),
                    "metrics": {k: _f(metrics.get(k), 1e9) for k in METRIC_KEYS},
                })
            payload["burst_persistence_hard_panel"] = {"status": "evaluated_optional_stress_panel", "clip_count": int(len(indices)), "runs": burst_rows, "aggregate": _aggregate_rows(burst_rows)}
        except Exception as exc:
            payload["burst_persistence_hard_panel"] = {"status": "skipped", "reason": str(exc)}
    finally:
        _release_lease_safe(str(eval_gpu.get("lease_id", "")), str(args.shared_lease_path))

    run_to_full_ep = {str(row["run_name"]): _f(row.get("metrics", {}).get("free_rollout_endpoint_l2"), 1e9) for row in payload["full_validation_panel"]["runs"]}
    run_to_hard_comp = {str(row["run_name"]): float(row.get("semantic_hard_composite_score", 1e9)) for row in payload["semantic_hard_composite_panel"]["runs"]}

    best_full_run = min(run_to_full_ep, key=run_to_full_ep.get)
    best_hard_run = min(run_to_hard_comp, key=run_to_hard_comp.get)
    best_full_endpoint = float(run_to_full_ep[best_full_run])
    best_hard_composite = float(run_to_hard_comp[best_hard_run])
    best_combo = next((str(x.get("objective_combo", "")) for x in completed if str(x.get("run_name", "")) == best_full_run), "none")

    true_new_best = any(int(run_aux.get(str(row["run_name"]), {}).get("overall_best_step", -1)) > int(run_aux.get(str(row["run_name"]), {}).get("resume_global_step", 1e9)) for row in completed)
    hard_improved_vs_v2 = bool(best_hard_composite < float(v2_best_hard_comp) * 0.95)
    cross_seed_runs = [row for row in payload["semantic_hard_composite_panel"]["runs"] if str(row.get("objective_combo", "")) == "confidence_gated_readout_alignment+sparse_persistence_contrastive_loss+delayed_aux_loss_schedule"]
    cross_seed_support = False
    if len({int(x.get("seed", -1)) for x in cross_seed_runs}) >= 2:
        vals = [float(x.get("semantic_hard_composite_score", 1e9)) for x in cross_seed_runs]
        cross_seed_support = bool(min(vals) < float(v2_best_hard_comp) and max(vals) <= float(v2_best_hard_comp) * 1.15)
    non_catastrophic = bool(best_full_endpoint <= crop_ep * 1.5)
    improved_vs_cropenc = bool(best_full_endpoint < crop_ep)
    improved_vs_v2 = bool(best_hard_composite < float(v2_best_hard_comp))
    overall_semantic_hard_diverged = bool(best_full_run != best_hard_run)
    sidecar_diverged = False
    for row in completed:
        final = _read_json(str(row["final_json"]))
        sidecar = final.get("semantic_hard_sidecar_metric", {}) if isinstance(final.get("semantic_hard_sidecar_metric", {}), dict) else {}
        if bool(sidecar.get("enabled", False)) and int(sidecar.get("global_step", -1)) >= 0:
            if int(sidecar.get("global_step", -1)) != int(final.get("best_checkpoint_metric", {}).get("global_step", -2)):
                sidecar_diverged = True
                break
    overall_semantic_hard_diverged = bool(overall_semantic_hard_diverged or sidecar_diverged)

    if true_new_best and hard_improved_vs_v2 and cross_seed_support and non_catastrophic:
        next_step = "stage2_semantic_rescue_fullscale_wave1"
    elif improved_vs_v2 or overall_semantic_hard_diverged:
        next_step = "redesign_stage2_semantic_objective_v4"
    else:
        next_step = "objective_family_still_not_working_under_current_stage2_design"

    payload["success_criteria"] = {
        "true_new_best_not_warm_start_inherited": bool(true_new_best),
        "semantic_hard_composite_improved_vs_v2": bool(hard_improved_vs_v2),
        "cross_seed_support_present": bool(cross_seed_support),
        "full_validation_non_catastrophic": bool(non_catastrophic),
        "semantic_hard_positive_signal_further_enhanced": bool(improved_vs_v2),
        "improved_vs_current_cropenc_baseline": bool(improved_vs_cropenc),
        "improved_vs_v2_best_objective_combo": bool(improved_vs_v2),
        "best_v3_objective_combo": str(next((str(x.get("objective_combo", "")) for x in completed if str(x.get("run_name", "")) == best_hard_run), "none")),
        "best_v3_full_validation_endpoint_l2": float(best_full_endpoint),
        "best_v3_semantic_hard_composite_score": float(best_hard_composite),
        "current_cropenc_fullscale_mean_endpoint_l2": float(crop_ep),
        "legacysem_fullscale_mean_endpoint_l2": float(legacy_ep),
        "v2_best_full_validation_endpoint_l2": float(v2_best_full_ep),
        "v2_best_semantic_hard_composite_score": float(v2_best_hard_comp),
        "overall_best_run_name": str(best_full_run),
        "semantic_hard_best_run_name": str(best_hard_run),
        "overall_best_and_semantic_hard_best_diverged": bool(overall_semantic_hard_diverged),
        "next_step_choice": str(next_step),
    }
    _write_json(args.v3_diagnosis_report, payload)
    return payload


def run_all(args: Any) -> Dict[str, Any]:
    write_protocol_doc(args)
    launch(args)
    wait_for_completion(args)
    diag = diagnose_v3(args)
    return {"v3_diagnosis": diag}


def parse_args() -> Any:
    p = ArgumentParser(description="Stage2 semantic objective redesign v3")
    p.add_argument("--mode", default="all", choices=["all", "launch", "run-one", "summarize", "diagnose"])
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
    p.add_argument("--v3-protocol-doc", default=str(WORK_ROOT / "docs/STAGE2_SEMANTIC_OBJECTIVE_REDESIGN_V3_PROTOCOL_20260410.md"))
    p.add_argument("--v3-launch-report", default=str(WORK_ROOT / "reports/stage2_semantic_objective_redesign_v3_launch_20260410.json"))
    p.add_argument("--v3-summary-report", default=str(WORK_ROOT / "reports/stage2_semantic_objective_redesign_v3_summary_20260410.json"))
    p.add_argument("--v3-results-md", default=str(WORK_ROOT / "docs/STAGE2_SEMANTIC_OBJECTIVE_REDESIGN_V3_RESULTS_20260410.md"))
    p.add_argument("--v3-diagnosis-report", default=str(WORK_ROOT / "reports/stage2_semantic_objective_redesign_v3_diagnosis_20260410.json"))
    p.add_argument("--wait-timeout-seconds", type=int, default=21600)
    p.add_argument("--poll-seconds", type=int, default=60)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "all":
        print(json.dumps(run_all(args), ensure_ascii=True, indent=2))
    elif args.mode == "launch":
        print(json.dumps(launch(args), ensure_ascii=True, indent=2))
    elif args.mode == "run-one":
        run_one(args)
    elif args.mode == "summarize":
        print(json.dumps(summarize(args), ensure_ascii=True, indent=2))
    elif args.mode == "diagnose":
        print(json.dumps(diagnose_v3(args), ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
