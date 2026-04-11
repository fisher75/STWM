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

import numpy as np
import torch
from torch.utils.data import Subset

from stwm.infra.gpu_lease import acquire_lease, release_lease
from stwm.infra.gpu_selector import select_single_gpu
from stwm.tracewm_v2_stage2.models.semantic_encoder import SemanticEncoder, SemanticEncoderConfig
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


WORK_ROOT = Path("/home/chen034/workspace/stwm")
SESSION = "tracewm_stage2_semobjv2_plus_codec_audit_20260410"
LOG_PATH = WORK_ROOT / "logs/tracewm_stage2_semobjv2_plus_codec_audit_20260410.log"
PILOT_EXTRA_STEPS = 200
PILOT_BATCH_SIZE = 8
PILOT_EVAL_INTERVAL = 100
PILOT_SAVE_EVERY = 100
PILOT_EVAL_MAX_BATCHES = 32
PILOT_MAX_TRAIN_PER_DATASET = 128
PILOT_MAX_VAL_PER_DATASET = 64


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _python_bin_default() -> str:
    preferred = Path("/home/chen034/miniconda3/envs/stwm/bin/python")
    return str(preferred) if preferred.exists() else sys.executable


def _aggregate_rows(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {key: _mean_std([_f(row.get("metrics", {}).get(key), 1e9) for row in rows]) for key in METRIC_KEYS}


def _baseline_refs() -> Dict[str, Any]:
    diagnosis = _read_json(WORK_ROOT / "reports/stage2_semantic_value_diagnosis_20260410.json")
    full = diagnosis.get("full_validation_panel", {}).get("family_aggregates", {})
    return {
        "cropenc_fullscale_mean": full.get("cropenc", {}),
        "legacysem_fullscale_mean": full.get("legacysem", {}),
        "hard_subset_panels": diagnosis.get("hard_subset_panels", {}),
        "stage2_semantic_value_diagnosis": diagnosis,
    }


def _v1_refs() -> Dict[str, Any]:
    p = WORK_ROOT / "reports/stage2_semantic_objective_redesign_v1_diagnosis_20260410.json"
    return _read_json(p) if p.exists() else {}


def write_protocol_docs(args: Any) -> None:
    _write_md(
        args.v2_protocol_doc,
        [
            "# Stage2 Semantic Objective Redesign V2 Protocol",
            "",
            f"- generated_at_utc: {now_iso()}",
            "- stage1_mutation_allowed: false",
            "- main_task: future trace / future state generation",
            "- teacher_as_mainline_semantic_source: false",
            "- v1_failure_summary: alignment / persistence / curriculum objective family hurt rollout optimum in pilot.",
            "- v2_core_principles: readout-side semantic rescue; contrastive/ranking persistence objective; non-destructive auxiliary supervision.",
            "- forbidden: teacher semantic token replacement; stronger fusion trick; full-scale long train; batch/lr sweep; DDP retrofit.",
        ],
    )
    _write_md(
        args.codec_protocol_doc,
        [
            "# Stage2 Codec Bottleneck Feasibility Audit Protocol",
            "",
            f"- generated_at_utc: {now_iso()}",
            "- purpose: test whether current 3M semantic visual branch / small codec is a key bottleneck.",
            "- not_this_round: mainline replacement; MotionCrafter/WAN/VAE full migration; full video reconstruction main task.",
            "- forbidden: full-scale codec replacement; full reconstruction task drift; Stage2 problem-definition change.",
        ],
    )


def _run_specs() -> List[Dict[str, Any]]:
    return [
        {
            "run_name": "stage2_semobjv2_readoutalign_seed42_20260410",
            "seed": 42,
            "objective_combo": "readout_semantic_alignment_head",
            "semantic_rescue_mode": "v2readoutalign",
            "semantic_rescue_weight": 0.0002,
            "readout_semantic_alignment_loss_weight": 1.0,
            "persistence_contrastive_ranking_loss_weight": 0.0,
            "semantic_aux_subset_weighting_strength": 0.0,
            "window_name": "semobjv2_ra42",
        },
        {
            "run_name": "stage2_semobjv2_readoutpersist_seed42_20260410",
            "seed": 42,
            "objective_combo": "readout_semantic_alignment_head+persistence_contrastive_or_ranking_loss",
            "semantic_rescue_mode": "v2readoutpersist",
            "semantic_rescue_weight": 0.0002,
            "readout_semantic_alignment_loss_weight": 1.0,
            "persistence_contrastive_ranking_loss_weight": 0.05,
            "semantic_aux_subset_weighting_strength": 0.0,
            "window_name": "semobjv2_rp42",
        },
        {
            "run_name": "stage2_semobjv2_readouthard_seed42_20260410",
            "seed": 42,
            "objective_combo": "readout_semantic_alignment_head+auxiliary_subset_weighting_only",
            "semantic_rescue_mode": "v2readouthard",
            "semantic_rescue_weight": 0.0002,
            "readout_semantic_alignment_loss_weight": 1.0,
            "persistence_contrastive_ranking_loss_weight": 0.0,
            "semantic_aux_subset_weighting_strength": 0.5,
            "window_name": "semobjv2_rh42",
        },
        {
            "run_name": "stage2_semobjv2_readoutpersist_seed123_20260410",
            "seed": 123,
            "objective_combo": "readout_semantic_alignment_head+persistence_contrastive_or_ranking_loss",
            "semantic_rescue_mode": "v2readoutpersist",
            "semantic_rescue_weight": 0.0002,
            "readout_semantic_alignment_loss_weight": 1.0,
            "persistence_contrastive_ranking_loss_weight": 0.05,
            "semantic_aux_subset_weighting_strength": 0.0,
            "window_name": "semobjv2_rp123",
        },
    ]


def _select_gpu(run_name: str, lease_path: str, required_mem_gb: float = 40.0) -> Dict[str, Any]:
    selector = select_single_gpu(
        required_mem_gb=float(required_mem_gb),
        safety_margin_gb=8.0,
        sample_count=3,
        interval_sec=0.5,
        lease_path=str(lease_path),
    )
    gpu_id = int(selector.get("selected_gpu_id", -1))
    if gpu_id < 0:
        raise RuntimeError(f"no GPU available for {run_name}")
    lease = acquire_lease(gpu_id=gpu_id, owner=str(run_name), ttl_seconds=8 * 3600, lease_path=str(lease_path))
    return {"selected_gpu_id": gpu_id, "lease_id": str(lease.get("lease_id", "")), "selector_payload": selector}


def _release_lease_safe(lease_id: str, lease_path: str) -> None:
    if not str(lease_id).strip():
        return
    try:
        release_lease(lease_id=str(lease_id), lease_path=str(lease_path))
    except Exception:
        pass


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

    runs = []
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
            "work_root": str(args.work_root),
            "python_bin": str(args.python_bin),
        }
        meta_json = Path(args.work_root) / "reports" / "stage2_semantic_objective_redesign_v2_runs_20260410" / f"{run_name}_launch_meta.json"
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
            f"{shlex.quote(str(Path(args.work_root) / 'code/stwm/tools/run_tracewm_stage2_semantic_objective_redesign_v2_20260410.py'))} "
            f"--mode run-one --meta-json {shlex.quote(str(meta_json))}"
        )
        subprocess.run(["tmux", "new-window", "-t", str(args.tmux_session), "-n", str(meta["window_name"]), cmd], check=True)

    payload = {
        "generated_at_utc": now_iso(),
        "mode": "stage2_semantic_objective_redesign_v2_launch",
        "tmux_session": str(args.tmux_session),
        "pilot_policy": "bounded pilot; no Stage1 training; CLIP cache is target only; no full-scale expansion",
        "runs": runs,
    }
    _write_json(args.v2_launch_report, payload)
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
        "--readout-semantic-alignment-loss-weight", str(meta["readout_semantic_alignment_loss_weight"]),
        "--persistence-contrastive-ranking-loss-weight", str(meta["persistence_contrastive_ranking_loss_weight"]),
        "--semantic-aux-subset-weighting-strength", str(meta["semantic_aux_subset_weighting_strength"]),
        "--resume-from", str(meta["resume_from"]),
        "--skip-resume-optimizer",
        "--output-dir", str(meta["output_dir"]),
        "--run-name", str(meta["run_name"]),
        "--run-summary-json", str(meta["raw_json"]),
        "--progress-json", str(meta["progress_json"]),
        "--seed", str(meta["seed"]),
    ]
    try:
        proc = subprocess.run(cmd, cwd=str(meta["work_root"]), text=True, capture_output=True, env=os.environ.copy())
        Path(str(meta["log_path"])).write_text(proc.stdout + ("\n" if proc.stdout else "") + proc.stderr, encoding="utf-8")
        if proc.returncode != 0:
            _write_json(meta["final_json"], {"generated_at_utc": now_iso(), "run_name": meta["run_name"], "status": "failed", "returncode": proc.returncode, "stderr_tail": proc.stderr[-4000:], "stdout_tail": proc.stdout[-4000:]})
            raise RuntimeError(f"trainer failed rc={proc.returncode}")
        raw = _read_json(meta["raw_json"])
        raw.update({"generated_at_utc": now_iso(), "status": "completed", "selected_gpu_id": int(meta["selected_gpu_id"]), "lease_id": str(meta["lease_id"]), "objective_combo": str(meta["objective_combo"]), "resume_global_step": int(meta["resume_global_step"])})
        _write_json(meta["final_json"], raw)
    except Exception as exc:
        _write_json(meta["final_json"], {"generated_at_utc": now_iso(), "run_name": str(meta.get("run_name", "")), "status": "failed", "message": str(exc), "traceback": traceback.format_exc()})
        raise
    finally:
        _release_lease_safe(lease_id=lease_id, lease_path=lease_path)


def _tmux_windows(session_name: str) -> List[str]:
    proc = subprocess.run(["tmux", "list-windows", "-t", str(session_name), "-F", "#{window_name}"], text=True, capture_output=True)
    return proc.stdout.splitlines() if proc.returncode == 0 else []


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
    launch = _read_json(args.v2_launch_report)
    rows: List[Dict[str, Any]] = []
    running = completed = failed = 0
    for meta in launch.get("runs", []) if isinstance(launch.get("runs", []), list) else []:
        status_info = _status_for(meta, session_name=str(args.tmux_session))
        status = str(status_info.get("status", "launched"))
        running += int(status == "running")
        completed += int(status == "completed")
        failed += int(status == "failed")
        detail = status_info.get("detail", {}) if isinstance(status_info.get("detail", {}), dict) else {}
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
            "semantic_rescue_mode": str(meta.get("semantic_rescue_mode", "")),
            "semantic_rescue_weight": float(meta.get("semantic_rescue_weight", 0.0)),
            "readout_semantic_alignment_loss_weight": float(meta.get("readout_semantic_alignment_loss_weight", 0.0)),
            "persistence_contrastive_ranking_loss_weight": float(meta.get("persistence_contrastive_ranking_loss_weight", 0.0)),
            "semantic_aux_subset_weighting_strength": float(meta.get("semantic_aux_subset_weighting_strength", 0.0)),
            "whether_main_rollout_loss_reweighted": False,
            "status": status,
            "final_json": str(meta.get("final_json", "")),
            "best_checkpoint": str(Path(str(meta.get("output_dir", ""))) / "best.pt"),
            "latest_checkpoint": str(Path(str(meta.get("output_dir", ""))) / "latest.pt"),
            "best_checkpoint_metric": detail.get("best_checkpoint_metric", {}) if isinstance(detail.get("best_checkpoint_metric", {}), dict) else {},
            "latest_checkpoint_metric": detail.get("latest_checkpoint_metric", {}) if isinstance(detail.get("latest_checkpoint_metric", {}), dict) else {},
        })
    payload = {"generated_at_utc": now_iso(), "redesign_v2_status": f"{running}_running_{completed}_completed_{failed}_failed", "runs": rows, "next_step_choice_internal": "summarize_redesign_v2_after_completion" if completed == len(rows) and rows and failed == 0 else ("redesign_stage2_semantic_objective_v3" if failed else "continue_redesign_v2")}
    _write_json(args.v2_summary_report, payload)
    lines = ["# Stage2 Semantic Objective Redesign V2 Results", "", f"- generated_at_utc: {payload['generated_at_utc']}", f"- redesign_v2_status: {payload['redesign_v2_status']}", f"- next_step_choice_internal: {payload['next_step_choice_internal']}", "", "| run_name | combo | gpu | batch | steps | status | best_endpoint_l2 | latest_endpoint_l2 |", "|---|---|---:|---:|---:|---|---:|---:|"]
    for row in rows:
        best = row.get("best_checkpoint_metric", {}).get("metrics", {}) if isinstance(row.get("best_checkpoint_metric", {}), dict) else {}
        latest = row.get("latest_checkpoint_metric", {}).get("metrics", {}) if isinstance(row.get("latest_checkpoint_metric", {}), dict) else {}
        lines.append(f"| {row['run_name']} | {row['objective_combo']} | {row['selected_gpu_id']} | {row['batch_size']} | {row['train_steps']} | {row['status']} | {_f(best.get('free_rollout_endpoint_l2'), 1e9):.8f} | {_f(latest.get('free_rollout_endpoint_l2'), 1e9):.8f} |")
    _write_md(args.v2_results_md, lines)
    return payload


def wait_for_completion(args: Any) -> Dict[str, Any]:
    deadline = time.time() + float(args.wait_timeout_seconds)
    last = summarize(args)
    while time.time() < deadline:
        if str(last.get("redesign_v2_status", "")).startswith("0_running_"):
            return last
        time.sleep(float(args.poll_seconds))
        last = summarize(args)
    last["timed_out_waiting_for_completion"] = True
    _write_json(args.v2_summary_report, last)
    return last


def _select_eval_gpu(args: Any) -> Dict[str, Any]:
    try:
        return _select_gpu("stage2_semobjv2_diagnosis_eval", str(args.shared_lease_path), required_mem_gb=40.0)
    except Exception:
        return {"selected_gpu_id": -1, "lease_id": ""}


def diagnose_v2(args: Any) -> Dict[str, Any]:
    summary = _read_json(args.v2_summary_report)
    rows = [r for r in summary.get("runs", []) if isinstance(r, dict)]
    completed = [r for r in rows if str(r.get("status", "")) == "completed"]
    refs = _baseline_refs()
    v1 = _v1_refs()
    crop_ep = _f(refs["cropenc_fullscale_mean"].get("free_rollout_endpoint_l2", {}).get("mean"), 1e9)
    legacy_ep = _f(refs["legacysem_fullscale_mean"].get("free_rollout_endpoint_l2", {}).get("mean"), 1e9)
    v1_best_ep = _f(v1.get("success_criteria", {}).get("best_v1_full_validation_endpoint_l2"), 1e9)
    payload: Dict[str, Any] = {
        "generated_at_utc": now_iso(),
        "diagnosis_type": "stage2_semantic_objective_redesign_v2",
        "teacher_as_mainline_semantic_source": False,
        "chosen_bootstrap_backend": "local_clip_vit_b32_mask_crop_visual_teacher",
        "v2_runs_terminal": bool(rows and len(completed) == len(rows)),
        "baseline_refs": {
            "current_cropenc_fullscale_mean_endpoint_l2": float(crop_ep),
            "legacysem_fullscale_mean_endpoint_l2": float(legacy_ep),
            "v1_best_objective_combo_endpoint_l2": float(v1_best_ep),
        },
        "full_validation_panel": {},
        "semantic_hard_subset_panel": {},
        "burst_persistence_hard_panel": {},
        "success_criteria": {},
    }
    if not payload["v2_runs_terminal"]:
        payload["success_criteria"] = {"true_new_best_not_warm_start_inherited": False, "semantic_hard_positive_signal": False, "more_stable_than_v1_best": False, "full_validation_non_catastrophic": False, "best_v2_objective_combo": "none", "next_step_choice": "redesign_stage2_semantic_objective_v3", "reason": "v2_runs_not_terminal"}
        _write_json(args.v2_diagnosis_report, payload)
        return payload

    eval_gpu = _select_eval_gpu(args)
    if int(eval_gpu.get("selected_gpu_id", -1)) >= 0 and not os.environ.get("CUDA_VISIBLE_DEVICES", "").strip():
        os.environ["CUDA_VISIBLE_DEVICES"] = str(eval_gpu["selected_gpu_id"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        print(f"[semobjv2] diagnosis device={device} eval_gpu={eval_gpu}", flush=True)
        stage1_model, _ = _load_stage1_model(device=device, stage1_ckpt=str(args.stage1_best_ckpt))
        full_ds = _make_dataset(["vspw", "vipseg"], "val", str(args.stage2_contract_json), max_samples=-1)
        core_ds = _make_dataset(["vspw", "vipseg"], "val", str(args.stage2_contract_json), max_samples=-1)
        subset_manifest = _read_json(WORK_ROOT / "manifests/protocol_v2/stage2_semantic_hard_subsets_20260410.json")
        loaded: Dict[str, Tuple[Any, Any, Any, str]] = {}
        full_rows: List[Dict[str, Any]] = []
        for row in completed:
            print(f"[semobjv2] full validation eval run={row['run_name']}", flush=True)
            loaded[str(row["run_name"])] = _load_stage2_modules(str(row["best_checkpoint"]), device, stage1_model)
            metrics = _evaluate_loaded_stage2(loaded=loaded[str(row["run_name"])], stage1_model=stage1_model, dataset=full_ds, device=device)
            full_rows.append({"run_name": str(row["run_name"]), "objective_combo": str(row.get("objective_combo", "")), "seed": int(row.get("seed", -1)), "metrics": {k: _f(metrics.get(k), 1e9) for k in METRIC_KEYS}})
        payload["full_validation_panel"] = {"runs": full_rows, "aggregate": _aggregate_rows(full_rows), "dataset_binding": ["VSPW", "VIPSeg"], "eval_scope": "full_validation"}

        hard_panels: Dict[str, Any] = {}
        for subset_name in ["occlusion_reappearance", "crossing_or_interaction_ambiguity", "small_object_or_low_area", "appearance_change_or_semantic_shift"]:
            print(f"[semobjv2] semantic hard eval subset={subset_name}", flush=True)
            subset = subset_manifest.get("subsets", {}).get(subset_name, {})
            indices = [int(x["dataset_index"]) for x in subset.get("items", []) if isinstance(x, dict) and "dataset_index" in x]
            eval_ds = Subset(core_ds, indices)
            subset_rows: List[Dict[str, Any]] = []
            for row in completed:
                metrics = _evaluate_loaded_stage2(loaded=loaded[str(row["run_name"])], stage1_model=stage1_model, dataset=eval_ds, device=device)
                subset_rows.append({"run_name": str(row["run_name"]), "objective_combo": str(row.get("objective_combo", "")), "seed": int(row.get("seed", -1)), "metrics": {k: _f(metrics.get(k), 1e9) for k in METRIC_KEYS}})
            hard_panels[subset_name] = {
                "clip_count": int(len(indices)),
                "runs": subset_rows,
                "aggregate": _aggregate_rows(subset_rows),
                "cropenc_baseline_aggregate": refs.get("hard_subset_panels", {}).get(subset_name, {}).get("families", {}).get("cropenc", {}).get("aggregate", {}),
                "legacysem_baseline_aggregate": refs.get("hard_subset_panels", {}).get(subset_name, {}).get("families", {}).get("legacysem", {}).get("aggregate", {}),
                "v1_best_reference": v1.get("semantic_hard_subset_panel", {}).get(subset_name, {}),
            }
        payload["semantic_hard_subset_panel"] = hard_panels

        try:
            print("[semobjv2] optional BURST persistence-hard eval", flush=True)
            burst_ds = _make_dataset(["burst"], "val", str(args.stage2_contract_json), max_samples=-1)
            burst_subset = subset_manifest.get("subsets", {}).get("burst_persistence_stress", {})
            indices = [int(x["dataset_index"]) for x in burst_subset.get("items", []) if isinstance(x, dict) and "dataset_index" in x]
            eval_ds = Subset(burst_ds, indices)
            burst_rows = []
            for row in completed:
                metrics = _evaluate_loaded_stage2(loaded=loaded[str(row["run_name"])], stage1_model=stage1_model, dataset=eval_ds, device=device)
                burst_rows.append({"run_name": str(row["run_name"]), "objective_combo": str(row.get("objective_combo", "")), "seed": int(row.get("seed", -1)), "metrics": {k: _f(metrics.get(k), 1e9) for k in METRIC_KEYS}})
            payload["burst_persistence_hard_panel"] = {"status": "evaluated_optional_stress_panel", "clip_count": int(len(indices)), "runs": burst_rows, "aggregate": _aggregate_rows(burst_rows)}
        except Exception as exc:
            payload["burst_persistence_hard_panel"] = {"status": "skipped", "reason": str(exc)}
    finally:
        _release_lease_safe(str(eval_gpu.get("lease_id", "")), str(args.shared_lease_path))

    true_new_best = False
    best_combo = "none"
    best_full_endpoint = 1e9
    latest_endpoints: List[float] = []
    for row in completed:
        final = _read_json(row["final_json"])
        best_step = int(final.get("best_checkpoint_metric", {}).get("global_step", -1))
        resume_step = int(final.get("resume_global_step", row.get("train_steps", 0) - PILOT_EXTRA_STEPS))
        true_new_best = bool(true_new_best or best_step > resume_step)
        latest_endpoints.append(_f(final.get("latest_checkpoint_metric", {}).get("metrics", {}).get("free_rollout_endpoint_l2"), 1e9))
    for row in payload.get("full_validation_panel", {}).get("runs", []):
        ep = _f(row.get("metrics", {}).get("free_rollout_endpoint_l2"), 1e9)
        if ep < best_full_endpoint:
            best_full_endpoint = ep
            best_combo = str(row.get("objective_combo", "none"))
    semantic_hard_positive = False
    for _name, panel in payload.get("semantic_hard_subset_panel", {}).items():
        crop_base = _f(panel.get("cropenc_baseline_aggregate", {}).get("free_rollout_endpoint_l2", {}).get("mean"), 1e9)
        for row in panel.get("runs", []) if isinstance(panel.get("runs", []), list) else []:
            semantic_hard_positive = bool(semantic_hard_positive or _f(row.get("metrics", {}).get("free_rollout_endpoint_l2"), 1e9) < crop_base)
    more_stable_than_v1 = bool(latest_endpoints and min(latest_endpoints) < _f(v1.get("success_criteria", {}).get("best_wave0_latest_endpoint_l2"), 1e9))
    non_catastrophic = bool(best_full_endpoint <= crop_ep * 1.5)
    payload["success_criteria"] = {
        "true_new_best_not_warm_start_inherited": bool(true_new_best),
        "semantic_hard_positive_signal": bool(semantic_hard_positive),
        "more_stable_than_v1_best": bool(more_stable_than_v1 or best_full_endpoint < v1_best_ep),
        "improved_vs_current_cropenc_baseline": bool(best_full_endpoint < crop_ep),
        "narrowed_or_won_vs_legacysem": bool(abs(best_full_endpoint - legacy_ep) < abs(crop_ep - legacy_ep) or best_full_endpoint < legacy_ep),
        "full_validation_non_catastrophic": bool(non_catastrophic),
        "best_v2_objective_combo": str(best_combo),
        "best_v2_full_validation_endpoint_l2": float(best_full_endpoint),
        "current_cropenc_fullscale_mean_endpoint_l2": float(crop_ep),
        "legacysem_fullscale_mean_endpoint_l2": float(legacy_ep),
        "v1_best_objective_combo_endpoint_l2": float(v1_best_ep),
    }
    _write_json(args.v2_diagnosis_report, payload)
    return payload


def _count_params(module: torch.nn.Module) -> int:
    return int(sum(p.numel() for p in module.parameters()))


def _candidate_paths(pattern: str, roots: List[Path], limit: int = 20) -> List[str]:
    out: List[str] = []
    for root in roots:
        if not root.exists():
            continue
        for p in root.rglob(pattern):
            out.append(str(p))
            if len(out) >= limit:
                return out
    return out


def codec_candidate_scan(args: Any) -> Dict[str, Any]:
    roots = [WORK_ROOT / "models", WORK_ROOT / "third_party", WORK_ROOT / "data/processed", Path("/home/chen034/.cache/clip")]
    motion = _candidate_paths("*MotionCrafter*", roots)
    vae = _candidate_paths("*vae*", roots)
    clip_weight = [str(p) for p in [Path("/home/chen034/.cache/clip/ViT-B-32.pt"), Path("/raid/chen034/.cache/clip/ViT-B-32.pt")] if p.exists()]
    sam2_weight = [str(p) for p in [WORK_ROOT / "models/checkpoints/sam2/sam2.1_hiera_base_plus.pt", WORK_ROOT / "models/checkpoints/sam2/sam2.1_hiera_large.pt"] if p.exists()]
    candidates = [
        {"candidate_name": "MotionCrafter-style VAE", "available_locally": bool(motion), "evidence_paths": motion, "integration_difficulty": "high", "frozen_feature_extractor_only": False, "would_force_task_drift_toward_full_reconstruction": True, "note": "No reusable local MotionCrafter VAE asset found." if not motion else "Local names found; direct Stage2 feature-export path still unverified."},
        {"candidate_name": "larger image/video VAE", "available_locally": bool(vae), "evidence_paths": vae, "integration_difficulty": "high", "frozen_feature_extractor_only": bool(vae), "would_force_task_drift_toward_full_reconstruction": True},
        {"candidate_name": "local CLIP ViT-B/32 mask-crop feature extractor", "available_locally": bool(clip_weight), "evidence_paths": clip_weight, "integration_difficulty": "low", "frozen_feature_extractor_only": True, "would_force_task_drift_toward_full_reconstruction": False},
        {"candidate_name": "SAM2 local visual feature candidate", "available_locally": bool(sam2_weight), "evidence_paths": sam2_weight, "integration_difficulty": "medium_high", "frozen_feature_extractor_only": True, "would_force_task_drift_toward_full_reconstruction": False, "note": "Weights exist but Stage2 region-feature export is not verified in this round."},
    ]
    payload = {"generated_at_utc": now_iso(), "scan_policy": "local-only; no large model downloads; no mainline replacement", "candidates": candidates}
    _write_json(args.codec_candidate_scan_report, payload)
    lines = ["# Stage2 Codec Candidate Feasibility Scan", "", f"- generated_at_utc: {payload['generated_at_utc']}"]
    for c in candidates:
        lines.append(f"- {c['candidate_name']}: available_locally={c['available_locally']}, difficulty={c['integration_difficulty']}, frozen_feature_extractor_only={c['frozen_feature_extractor_only']}, task_drift={c['would_force_task_drift_toward_full_reconstruction']}")
    _write_md(args.codec_candidate_scan_md, lines)
    return payload


def _load_clip_cache(path: str | Path) -> Dict[str, np.ndarray]:
    cache: Dict[str, np.ndarray] = {}
    p = Path(path)
    if not p.exists():
        return cache
    for raw in p.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not raw.strip():
            continue
        item = json.loads(raw)
        key = f"{str(item.get('dataset', '')).upper()}::{str(item.get('clip_id', ''))}"
        vals = item.get("feature_target", [])
        if isinstance(vals, list):
            cache[key] = np.asarray(vals, dtype=np.float32)
    return cache


def _separation_score(features: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    if features.shape[0] < 4 or len(set(labels.tolist())) < 2:
        return {"centroid_margin": 0.0, "one_nn_accuracy": 0.0, "count": int(features.shape[0])}
    x = features / np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-6)
    pos = x[labels == 1]
    neg = x[labels == 0]
    cp, cn = pos.mean(axis=0), neg.mean(axis=0)
    inter = float(np.linalg.norm(cp - cn))
    within = float(np.mean(np.linalg.norm(pos - cp, axis=1)) + np.mean(np.linalg.norm(neg - cn, axis=1)))
    sim = x @ x.T
    np.fill_diagonal(sim, -1e9)
    nn = sim.argmax(axis=1)
    acc = float(np.mean(labels[nn] == labels))
    return {"centroid_margin": float(inter / max(within, 1e-6)), "one_nn_accuracy": acc, "count": int(features.shape[0])}


def codec_side_probe(args: Any) -> Dict[str, Any]:
    subset_manifest = _read_json(WORK_ROOT / "manifests/protocol_v2/stage2_semantic_hard_subsets_20260410.json")
    hard_indices = set()
    for name in ["occlusion_reappearance", "crossing_or_interaction_ambiguity", "small_object_or_low_area", "appearance_change_or_semantic_shift"]:
        for item in subset_manifest.get("subsets", {}).get(name, {}).get("items", []):
            if isinstance(item, dict) and "dataset_index" in item:
                hard_indices.add(int(item["dataset_index"]))
    hard_indices = set(sorted(hard_indices)[:48])
    core_ds = _make_dataset(["vspw", "vipseg"], "val", str(args.stage2_contract_json), max_samples=-1)
    easy_indices = [i for i in range(min(len(core_ds), 160)) if i not in hard_indices][: len(hard_indices)]
    indices = list(sorted(hard_indices)) + easy_indices
    labels = np.asarray([1] * len(hard_indices) + [0] * len(easy_indices), dtype=np.int64)

    ckpt_path = _resume_ckpt_for_seed(42)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    ckpt_args = ckpt.get("args", {}) if isinstance(ckpt.get("args", {}), dict) else {}
    encoder = SemanticEncoder(
        SemanticEncoderConfig(
            input_dim=10,
            hidden_dim=int(ckpt_args.get("semantic_hidden_dim", 256) or 256),
            output_dim=int(ckpt_args.get("semantic_embed_dim", 256) or 256),
            dropout=0.1,
            mainline_source="crop_visual_encoder",
            legacy_source="hand_crafted_stats",
        )
    )
    encoder.load_state_dict(ckpt["semantic_encoder_state_dict"])
    encoder.eval()
    clip_cache = _load_clip_cache(args.bootstrap_cache_jsonl)
    crop_feats: List[np.ndarray] = []
    clip_feats: List[np.ndarray] = []
    kept_labels: List[int] = []
    missing_clip = 0
    with torch.no_grad():
        for idx, label in zip(indices, labels):
            sample = core_ds[int(idx)]
            meta = sample.get("meta", {}) if isinstance(sample.get("meta", {}), dict) else {}
            key = f"{str(meta.get('dataset', '')).upper()}::{str(meta.get('clip_id', ''))}"
            if key not in clip_cache:
                missing_clip += 1
                continue
            feat = encoder(
                sample["semantic_features"][None],
                semantic_rgb_crop=sample["semantic_rgb_crop"][None],
                semantic_mask_crop=sample["semantic_mask_crop"][None],
                source_mode="crop_visual_encoder",
            )[0, 0].detach().cpu().numpy().astype(np.float32)
            crop_feats.append(feat)
            clip_feats.append(clip_cache[key])
            kept_labels.append(int(label))
    crop_score = _separation_score(np.stack(crop_feats), np.asarray(kept_labels, dtype=np.int64)) if crop_feats else {"centroid_margin": 0.0, "one_nn_accuracy": 0.0, "count": 0}
    clip_score = _separation_score(np.stack(clip_feats), np.asarray(kept_labels, dtype=np.int64)) if clip_feats else {"centroid_margin": 0.0, "one_nn_accuracy": 0.0, "count": 0}
    clearly_better = bool(
        clip_score["count"] >= 8
        and (
            clip_score["centroid_margin"] > 1.25 * max(crop_score["centroid_margin"], 1e-6)
            or clip_score["one_nn_accuracy"] > crop_score["one_nn_accuracy"] + 0.10
        )
    )
    payload = {
        "generated_at_utc": now_iso(),
        "probe_scope": "required semantic-hard subset only; no full-scale training",
        "feature_sets": ["current_3m_semantic_crop_encoder_features", "CLIP_mask_crop_features"],
        "sample_counts": {"hard": int(len(hard_indices)), "easy": int(len(easy_indices)), "used_after_clip_cache_filter": int(len(kept_labels)), "missing_clip": int(missing_clip)},
        "current_3m_semantic_crop_encoder_features": crop_score,
        "clip_mask_crop_features": clip_score,
        "stronger_frozen_visual_vae_features_clearly_better": bool(clearly_better),
        "rule": "clear advantage if CLIP centroid_margin > 1.25x current or 1NN accuracy improves by >0.10 on identical samples",
    }
    _write_json(args.codec_side_probe_report, payload)
    _write_md(args.codec_side_probe_md, ["# Stage2 Codec Side Probe", "", f"- generated_at_utc: {payload['generated_at_utc']}", f"- used_after_clip_cache_filter: {payload['sample_counts']['used_after_clip_cache_filter']}", f"- current_crop_encoder: {crop_score}", f"- clip_mask_crop: {clip_score}", f"- stronger_frozen_visual_vae_features_clearly_better: {clearly_better}"])
    return payload


def codec_bottleneck_audit(args: Any, v2_diag: Dict[str, Any], side_probe: Dict[str, Any], candidate_scan: Dict[str, Any]) -> Dict[str, Any]:
    final = _read_json(WORK_ROOT / "reports/stage2_fullscale_core_cropenc_seed42_20260409_final.json")
    ckpt = torch.load(_resume_ckpt_for_seed(42), map_location="cpu", weights_only=False)
    ckpt_args = ckpt.get("args", {}) if isinstance(ckpt.get("args", {}), dict) else {}
    encoder = SemanticEncoder(SemanticEncoderConfig(input_dim=10, hidden_dim=int(ckpt_args.get("semantic_hidden_dim", 256) or 256), output_dim=int(ckpt_args.get("semantic_embed_dim", 256) or 256), dropout=0.1, mainline_source="crop_visual_encoder", legacy_source="hand_crafted_stats"))
    semantic_crop_encoder_params = _count_params(encoder.crop_encoder)
    semantic_encoder_total_params = _count_params(encoder)
    stage2_trainable = int(final.get("parameter_count_trainable", 0) or 0)
    stage1_frozen = int(final.get("parameter_count_frozen", 0) or 0)
    v1 = _v1_refs()
    v2_success = v2_diag.get("success_criteria", {})
    side_better = bool(side_probe.get("stronger_frozen_visual_vae_features_clearly_better", False))
    v2_failed = not (bool(v2_success.get("true_new_best_not_warm_start_inherited", False)) and bool(v2_success.get("semantic_hard_positive_signal", False)))
    if side_better and v2_failed:
        bottleneck_judgment = "possible_secondary_bottleneck"
    elif side_better:
        bottleneck_judgment = "possible_secondary_bottleneck"
    else:
        bottleneck_judgment = "likely_not_bottleneck"
    payload = {
        "generated_at_utc": now_iso(),
        "semantic_crop_encoder_params": int(semantic_crop_encoder_params),
        "semantic_encoder_total_params": int(semantic_encoder_total_params),
        "semantic_branch_total_trainable_params": int(stage2_trainable),
        "frozen_stage1_backbone_params": int(stage1_frozen),
        "trainable_to_frozen_param_ratio": float(stage2_trainable / max(stage1_frozen, 1)),
        "semantic_encoder_role": "object/mask crop -> trainable semantic token; fused into frozen Stage1 hidden via semantic_fusion gate; readout head predicts future trace coordinates.",
        "current_diagnosis_inputs": {
            "v1_failed": bool(not v1.get("success_criteria", {}).get("true_new_best_not_warm_start_inherited", False)),
            "v2_failed": bool(v2_failed),
            "clip_bootstrap_available": True,
            "side_probe_stronger_features_clearly_better": bool(side_better),
        },
        "bottleneck_judgment": bottleneck_judgment,
        "judgment_reason": "Capacity may matter as a secondary issue if CLIP side-probe wins, but v1/v2 objective failures do not prove the 3M crop branch is the primary bottleneck.",
    }
    _write_json(args.codec_bottleneck_audit_report, payload)
    _write_md(args.codec_bottleneck_audit_md, ["# Stage2 Codec Bottleneck Audit", "", f"- semantic_crop_encoder_params: {semantic_crop_encoder_params}", f"- semantic_branch_total_trainable_params: {stage2_trainable}", f"- frozen_stage1_backbone_params: {stage1_frozen}", f"- trainable_to_frozen_param_ratio: {payload['trainable_to_frozen_param_ratio']:.6f}", f"- bottleneck_judgment: {bottleneck_judgment}", f"- judgment_reason: {payload['judgment_reason']}"])
    return payload


def codec_feasibility_diagnosis(args: Any, audit: Dict[str, Any], side_probe: Dict[str, Any]) -> Dict[str, Any]:
    primary = bool(audit.get("bottleneck_judgment") == "likely_primary_bottleneck")
    side_better = bool(side_probe.get("stronger_frozen_visual_vae_features_clearly_better", False))
    payload = {
        "generated_at_utc": now_iso(),
        "current_3m_visual_branch_likely_primary_bottleneck": bool(primary),
        "if_not_primary_more_like": "secondary_bottleneck" if audit.get("bottleneck_judgment") == "possible_secondary_bottleneck" else "not_the_bottleneck_yet",
        "stronger_frozen_visual_vae_features_have_clear_side_probe_advantage": bool(side_better),
        "recommended_codec_next_step": "stage2_codec_upgrade_wave0" if primary and side_better else "keep_current_codec_and_continue_objective_redesign",
    }
    _write_json(args.codec_feasibility_diagnosis_report, payload)
    return payload


def final_decision(args: Any, v2_diag: Dict[str, Any], codec_diag: Dict[str, Any]) -> Dict[str, Any]:
    s = v2_diag.get("success_criteria", {})
    v2_positive = bool(s.get("true_new_best_not_warm_start_inherited", False) and s.get("semantic_hard_positive_signal", False) and s.get("full_validation_non_catastrophic", False))
    codec_primary = bool(codec_diag.get("current_3m_visual_branch_likely_primary_bottleneck", False))
    side_better = bool(codec_diag.get("stronger_frozen_visual_vae_features_have_clear_side_probe_advantage", False))
    if v2_positive and not codec_primary:
        next_step = "stage2_semantic_rescue_fullscale_wave1"
    elif (not v2_positive) and codec_primary and side_better:
        next_step = "stage2_codec_upgrade_wave0"
    elif (not v2_positive) and (not codec_primary):
        next_step = "redesign_stage2_semantic_objective_v3"
    else:
        next_step = "bootstrap_backend_ok_but_objective_and_codec_both_need_rethink"
    payload = {
        "generated_at_utc": now_iso(),
        "chosen_bootstrap_backend": "local_clip_vit_b32_mask_crop_visual_teacher",
        "best_v2_objective_combo": str(s.get("best_v2_objective_combo", "none")),
        "true_new_best_not_warm_start_inherited": bool(s.get("true_new_best_not_warm_start_inherited", False)),
        "semantic_hard_positive_signal": bool(s.get("semantic_hard_positive_signal", False)),
        "current_3m_visual_branch_likely_primary_bottleneck": bool(codec_primary),
        "stronger_frozen_visual_vae_features_clearly_better": bool(side_better),
        "next_step_choice": next_step,
    }
    _write_json(args.combined_decision_report, payload)
    _write_md(args.combined_decision_md, ["# Stage2 SemObjV2 Plus Codec Decision", "", f"- chosen_bootstrap_backend: {payload['chosen_bootstrap_backend']}", f"- best_v2_objective_combo: {payload['best_v2_objective_combo']}", f"- true_new_best_not_warm_start_inherited: {payload['true_new_best_not_warm_start_inherited']}", f"- semantic_hard_positive_signal: {payload['semantic_hard_positive_signal']}", f"- current_3m_visual_branch_likely_primary_bottleneck: {payload['current_3m_visual_branch_likely_primary_bottleneck']}", f"- stronger_frozen_visual_vae_features_clearly_better: {payload['stronger_frozen_visual_vae_features_clearly_better']}", f"- next_step_choice: {payload['next_step_choice']}"])
    return payload


def run_all(args: Any) -> Dict[str, Any]:
    write_protocol_docs(args)
    candidate_scan = codec_candidate_scan(args)
    launch(args)
    wait_for_completion(args)
    v2_diag = diagnose_v2(args)
    side_probe = codec_side_probe(args)
    audit = codec_bottleneck_audit(args, v2_diag, side_probe, candidate_scan)
    codec_diag = codec_feasibility_diagnosis(args, audit, side_probe)
    decision = final_decision(args, v2_diag, codec_diag)
    return {"v2_diagnosis": v2_diag, "codec_feasibility_diagnosis": codec_diag, "decision": decision}


def parse_args() -> Any:
    p = ArgumentParser(description="Stage2 semantic objective redesign v2 plus codec bottleneck audit")
    p.add_argument("--mode", default="all", choices=["all", "launch", "run-one", "summarize", "diagnose", "codec-audit"])
    p.add_argument("--meta-json", default="")
    p.add_argument("--work-root", default=str(WORK_ROOT))
    p.add_argument("--python-bin", default=_python_bin_default())
    p.add_argument("--tmux-session", default=SESSION)
    p.add_argument("--stage2-contract-json", default=str(WORK_ROOT / "reports/stage2_bootstrap_data_contract_20260408.json"))
    p.add_argument("--stage1-runtime-json", default=str(WORK_ROOT / "reports/stage1_v2_recommended_runtime_20260408.json"))
    p.add_argument("--stage1-best-ckpt", default=str(WORK_ROOT / "outputs/checkpoints/stage1_v2_longtrain_220m_mainline_20260408/best.pt"))
    p.add_argument("--shared-lease-path", default=str(WORK_ROOT / "reports/stage1_v2_gpu_lease_20260408.json"))
    p.add_argument("--bootstrap-cache-jsonl", default=str(WORK_ROOT / "data/processed/stage2_real_bootstrap_cache_20260410/clip_vit_b32_core_trainval_required_subset.jsonl"))
    p.add_argument("--v2-protocol-doc", default=str(WORK_ROOT / "docs/STAGE2_SEMANTIC_OBJECTIVE_REDESIGN_V2_PROTOCOL_20260410.md"))
    p.add_argument("--codec-protocol-doc", default=str(WORK_ROOT / "docs/STAGE2_CODEC_BOTTLENECK_FEASIBILITY_AUDIT_PROTOCOL_20260410.md"))
    p.add_argument("--v2-launch-report", default=str(WORK_ROOT / "reports/stage2_semantic_objective_redesign_v2_launch_20260410.json"))
    p.add_argument("--v2-summary-report", default=str(WORK_ROOT / "reports/stage2_semantic_objective_redesign_v2_summary_20260410.json"))
    p.add_argument("--v2-results-md", default=str(WORK_ROOT / "docs/STAGE2_SEMANTIC_OBJECTIVE_REDESIGN_V2_RESULTS_20260410.md"))
    p.add_argument("--v2-diagnosis-report", default=str(WORK_ROOT / "reports/stage2_semantic_objective_redesign_v2_diagnosis_20260410.json"))
    p.add_argument("--codec-bottleneck-audit-report", default=str(WORK_ROOT / "reports/stage2_codec_bottleneck_audit_20260410.json"))
    p.add_argument("--codec-bottleneck-audit-md", default=str(WORK_ROOT / "docs/STAGE2_CODEC_BOTTLENECK_AUDIT_20260410.md"))
    p.add_argument("--codec-candidate-scan-report", default=str(WORK_ROOT / "reports/stage2_codec_candidate_feasibility_scan_20260410.json"))
    p.add_argument("--codec-candidate-scan-md", default=str(WORK_ROOT / "docs/STAGE2_CODEC_CANDIDATE_FEASIBILITY_SCAN_20260410.md"))
    p.add_argument("--codec-side-probe-report", default=str(WORK_ROOT / "reports/stage2_codec_side_probe_20260410.json"))
    p.add_argument("--codec-side-probe-md", default=str(WORK_ROOT / "docs/STAGE2_CODEC_SIDE_PROBE_20260410.md"))
    p.add_argument("--codec-feasibility-diagnosis-report", default=str(WORK_ROOT / "reports/stage2_codec_bottleneck_feasibility_diagnosis_20260410.json"))
    p.add_argument("--combined-decision-report", default=str(WORK_ROOT / "reports/stage2_semobjv2_plus_codec_decision_20260410.json"))
    p.add_argument("--combined-decision-md", default=str(WORK_ROOT / "docs/STAGE2_SEMOBJV2_PLUS_CODEC_DECISION_20260410.md"))
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
        print(json.dumps(diagnose_v2(args), ensure_ascii=True, indent=2))
    elif args.mode == "codec-audit":
        scan = codec_candidate_scan(args)
        v2_diag = _read_json(args.v2_diagnosis_report)
        side = codec_side_probe(args)
        audit = codec_bottleneck_audit(args, v2_diag, side, scan)
        codec_diag = codec_feasibility_diagnosis(args, audit, side)
        print(json.dumps(final_decision(args, v2_diag, codec_diag), ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
