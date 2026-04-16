#!/usr/bin/env python3
from __future__ import annotations

from argparse import ArgumentParser
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple
import json
import os
import shlex
import shutil
import signal
import subprocess
import time

from stwm.infra.gpu_lease import acquire_lease, release_lease
from stwm.infra.gpu_selector import select_single_gpu
from stwm.tools import run_tracewm_stage2_calibration_only_fullscale_wave1_20260413 as base


def _repo_root() -> Path:
    for candidate in [
        Path("/raid/chen034/workspace/stwm"),
        Path("/home/chen034/workspace/stwm"),
    ]:
        if candidate.exists():
            return candidate
    raise RuntimeError("unable to resolve STWM repo root")


ROOT = _repo_root()
SESSION = "tracewm_stage2_top_tier_closure_20260415"
LOG_PATH = ROOT / "logs/stage2_top_tier_closure_20260415.log"
BOOTSTRAP_BACKEND = "local_clip_vit_b32_mask_crop_visual_teacher"
DATE_TAG = "20260415"
EXTRA_STEPS = 4000
BATCH_SIZE = 8
EVAL_INTERVAL = 500
SAVE_EVERY = 500
EVAL_MAX_BATCHES = 0
MAX_TRAIN = -1
MAX_VAL = -1


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _append_log(message: str) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as fh:
        fh.write(f"[{now_iso()}] {message}\n")


def _read_json(path_like: Any) -> Dict[str, Any]:
    p = Path(path_like)
    if not p.exists():
        return {}
    payload = json.loads(p.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _write_json(path_like: Any, payload: Dict[str, Any]) -> None:
    p = Path(path_like)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def _write_md(path_like: Any, lines: List[str]) -> None:
    p = Path(path_like)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _apply_process_title() -> None:
    base._apply_process_title_normalization(default_title="python")


def _spec_base(run_name: str, seed: int, ablation_name: str, window_name: str, reuse_source_run_name: str = "") -> Dict[str, Any]:
    return {
        "run_name": run_name,
        "seed": int(seed),
        "ablation_name": ablation_name,
        "objective_combo": f"closure_{ablation_name}_seed{int(seed)}",
        "track": "ablation_fix",
        "family": "calibration_only_ablation_fix",
        "objective_family": "calibration_only_mechanism_ablation_fix",
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
        "persistence_objective_declared": False,
        "reuse_source_run_name": reuse_source_run_name,
        "window_name": window_name,
    }


def _run_specs() -> List[Dict[str, Any]]:
    specs: List[Dict[str, Any]] = []
    reuse_map = {
        ("noalign", 42): "stage2_calonly_noalign_seed42_ablate_20260414",
        ("noalign", 123): "stage2_calonly_noalign_seed123_ablate_20260414",
        ("noalign", 456): "stage2_calonly_noalign_seed456_ablate_20260414",
        ("densegate", 42): "stage2_calonly_densegate_seed42_ablate_20260414",
        ("densegate", 123): "stage2_calonly_densegate_seed123_ablate_20260414",
        ("nodelay", 42): "stage2_calonly_nodelay_seed42_ablate_20260414",
        ("nodelay", 123): "stage2_calonly_nodelay_seed123_ablate_20260414",
    }
    for ablation_name in ["noalign", "densegate", "nodelay"]:
        for seed in [42, 123, 456, 654]:
            run_name = f"stage2_calonly_{ablation_name}_seed{seed}_ablate_fix_{DATE_TAG}"
            spec = _spec_base(
                run_name=run_name,
                seed=seed,
                ablation_name=ablation_name,
                window_name=f"abfix_{ablation_name[:4]}_{seed}",
                reuse_source_run_name=reuse_map.get((ablation_name, seed), ""),
            )
            if ablation_name == "noalign":
                spec["semantic_rescue_weight"] = 0.0
                spec["confidence_gated_alignment_loss_weight"] = 0.0
            elif ablation_name == "densegate":
                spec["v6_gating_family"] = "capped_quantile_sparse_gating_v2"
                spec["v6_capped_quantile"] = 0.0
                spec["v6_max_affected_ratio"] = 1.0
            elif ablation_name == "nodelay":
                spec["aux_loss_delay_steps"] = 0
                spec["aux_loss_ramp_steps"] = 0
            specs.append(spec)
    return specs


def _run_spec_map() -> Dict[str, Dict[str, Any]]:
    return {str(spec["run_name"]): spec for spec in _run_specs()}


def _meta_dir(args: Any) -> Path:
    return Path(args.work_root) / "reports/stage2_mechanism_ablation_fix_runs_20260415"


def _shared_select_gpu(run_name: str, lease_path: str, required_mem_gb: float = 24.0, safety_margin_gb: float = 4.0) -> Dict[str, Any]:
    selector = select_single_gpu(
        required_mem_gb=float(required_mem_gb),
        safety_margin_gb=float(safety_margin_gb),
        sample_count=3,
        interval_sec=0.5,
        lease_path=str(lease_path),
    )
    gpu_id = int(selector.get("selected_gpu_id", -1))
    if gpu_id < 0:
        raise RuntimeError(f"no_gpu_available_for_{run_name}")
    lease = acquire_lease(
        gpu_id=gpu_id,
        owner=str(run_name),
        ttl_seconds=12 * 3600,
        lease_path=str(lease_path),
        allow_shared=True,
    )
    return {
        "selected_gpu_id": int(gpu_id),
        "lease_id": str(lease.get("lease_id", "")),
        "selector_payload": selector,
    }


def _resume_ckpt_for_seed(seed: int) -> Path:
    candidates = [
        ROOT / f"outputs/checkpoints/stage2_calonly_topk1_seed{seed}_longconfirm_v2_20260414/best.pt",
        ROOT / f"outputs/checkpoints/stage2_calonly_topk1_seed{seed}_wave2_20260414/best.pt",
        ROOT / f"outputs/checkpoints/stage2_calonly_topk1_seed{seed}_wave1_20260413/best.pt",
    ]
    for ckpt in candidates:
        if ckpt.exists():
            return ckpt
    return base._resume_ckpt_for_seed(seed)


def _paths_for_run(args: Any, run_name: str) -> Dict[str, Path]:
    reports = Path(args.work_root) / "reports"
    return {
        "raw": reports / f"{run_name}_raw.json",
        "progress": reports / f"{run_name}_progress.json",
        "final": reports / f"{run_name}_final.json",
        "log": Path(args.work_root) / "logs" / f"{run_name}.log",
        "output_dir": Path(args.work_root) / "outputs/checkpoints" / run_name,
        "best": Path(args.work_root) / "outputs/checkpoints" / run_name / "best.pt",
        "latest": Path(args.work_root) / "outputs/checkpoints" / run_name / "latest.pt",
        "sidecar": Path(args.work_root) / "outputs/checkpoints" / run_name / "best_semantic_hard.pt",
    }


def _reset_real_run_artifacts(args: Any, run_name: str) -> Dict[str, Any]:
    paths = _paths_for_run(args, run_name)
    deleted: List[str] = []
    for key in ["raw", "progress", "final", "log"]:
        if paths[key].exists():
            paths[key].unlink()
            deleted.append(str(paths[key]))
    out_dir = paths["output_dir"]
    if out_dir.exists() and not out_dir.is_symlink():
        shutil.rmtree(out_dir)
        deleted.append(str(out_dir))
    meta_json = _meta_dir(args) / f"{run_name}_launch_meta.json"
    pid_file = _meta_dir(args) / f"{run_name}.pid"
    for extra in [meta_json, pid_file]:
        if extra.exists():
            extra.unlink()
            deleted.append(str(extra))
    return {"run_name": run_name, "deleted": deleted}


def _common_launch_context(args: Any) -> Dict[str, Any]:
    lease_cleanup = base._cleanup_stale_leases(str(args.shared_lease_path), allowed_prefixes=("stage2_calonly_",))
    if subprocess.run(["tmux", "has-session", "-t", str(args.tmux_session)], capture_output=True).returncode != 0:
        subprocess.run(["tmux", "new-session", "-d", "-s", str(args.tmux_session), "bash"], check=True)
    existing_windows = set(base._tmux_windows(str(args.tmux_session)))
    anchor_args = base._load_ckpt_args(base._resume_ckpt_for_seed(42))
    obs_len = int(anchor_args.get("obs_len", 8) or 8)
    fut_len = int(anchor_args.get("fut_len", 8) or 8)
    max_tokens = int(anchor_args.get("max_tokens", 64) or 64)
    crop_size = int(anchor_args.get("semantic_crop_size", 64) or 64)
    train_counts = base._dataset_counts(["vspw", "vipseg"], "train", args.stage2_contract_json, max_samples=MAX_TRAIN)
    val_counts = base._dataset_counts(["vspw", "vipseg"], "val", args.stage2_contract_json, max_samples=MAX_VAL)
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
        "additional_train_steps": EXTRA_STEPS,
        "train_steps": int(resume_step + EXTRA_STEPS),
        "eval_interval": EVAL_INTERVAL,
        "eval_max_batches": EVAL_MAX_BATCHES,
        "save_every_n_steps": SAVE_EVERY,
        "max_samples_train": MAX_TRAIN,
        "max_samples_val": MAX_VAL,
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
        "stage1_runtime_json": str(args.runtime_json),
        "stage1_best_ckpt": str(args.stage1_best_ckpt),
        "shared_lease_path": str(args.shared_lease_path),
        "bootstrap_cache_jsonl": str(args.bootstrap_cache_jsonl),
        "semantic_hard_manifest_path": str(args.semantic_hard_manifest_path),
        "work_root": str(args.work_root),
        "python_bin": str(args.python_bin),
        "worker_pid_file": str(Path(ctx["meta_dir"]) / f"{run_name}.pid"),
        "gpu_acquire_timeout_seconds": int(args.gpu_acquire_timeout_seconds),
        "gpu_acquire_retry_seconds": int(args.gpu_acquire_retry_seconds),
        "selector_payload": {},
    }
    meta_json = Path(ctx["meta_dir"]) / f"{run_name}_launch_meta.json"
    meta["meta_json"] = str(meta_json)
    return meta


def _write_and_launch_meta(args: Any, meta: Dict[str, Any], existing_windows: set[str]) -> Dict[str, Any]:
    run_name = str(meta["run_name"])
    meta_json = Path(str(meta["meta_json"]))
    _write_json(meta_json, meta)
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
    }


def _alias_equivalent_run(args: Any, spec: Dict[str, Any]) -> Dict[str, Any]:
    source_run = str(spec.get("reuse_source_run_name", ""))
    if not source_run:
        return {"run_name": str(spec["run_name"]), "reused": False}
    src_paths = _paths_for_run(args, source_run)
    dst_paths = _paths_for_run(args, str(spec["run_name"]))
    dst_paths["output_dir"].parent.mkdir(parents=True, exist_ok=True)
    if dst_paths["output_dir"].exists() or dst_paths["output_dir"].is_symlink():
        if dst_paths["output_dir"].is_symlink():
            target = os.readlink(dst_paths["output_dir"])
            if str(target) != str(src_paths["output_dir"]):
                dst_paths["output_dir"].unlink()
                os.symlink(str(src_paths["output_dir"]), str(dst_paths["output_dir"]), target_is_directory=True)
        else:
            raise RuntimeError(f"alias destination exists and is not symlink: {dst_paths['output_dir']}")
    else:
        os.symlink(str(src_paths["output_dir"]), str(dst_paths["output_dir"]), target_is_directory=True)
    for key in ["raw", "progress", "final"]:
        payload = _read_json(src_paths[key])
        payload["run_name"] = str(spec["run_name"])
        payload["reused_equivalent_source_run_name"] = source_run
        payload["reused_equivalent_artifacts"] = True
        payload["ablation_fix_date_tag"] = DATE_TAG
        _write_json(dst_paths[key], payload)
    return {
        "run_name": str(spec["run_name"]),
        "reused": True,
        "source_run_name": source_run,
        "aliased_checkpoint_dir": str(dst_paths["output_dir"]),
    }


def _tmux_window_command(args: Any, meta_json: Path, meta: Dict[str, Any]) -> str:
    script_path = Path(args.work_root) / "code/stwm/tools/run_stage2_mechanism_ablation_fix_20260415.py"
    pid_path = str(meta["worker_pid_file"])
    log_path = str(meta["log_path"])
    pythonpath_value = f"{args.work_root}/code:{os.environ.get('PYTHONPATH', '')}"
    proc_title = str(os.environ.get("STWM_PROC_TITLE", "python"))
    proc_title_mode = str(os.environ.get("STWM_PROC_TITLE_MODE", "generic"))
    cmd = (
        f"PYTHONPATH={shlex.quote(pythonpath_value)} "
        f"STWM_PROC_TITLE={shlex.quote(proc_title)} "
        f"STWM_PROC_TITLE_MODE={shlex.quote(proc_title_mode)} "
        f"nohup {shlex.quote(str(args.python_bin))} {shlex.quote(str(script_path))} --mode run-one --meta-json {shlex.quote(str(meta_json))} "
        f">> {shlex.quote(log_path)} 2>&1 < /dev/null & echo $! > {shlex.quote(pid_path)}; "
        f"while kill -0 \"$(cat {shlex.quote(pid_path)})\" 2>/dev/null; do sleep 30; done"
    )
    return "bash -lc " + shlex.quote(
        f"cd {shlex.quote(str(args.work_root))}; rm -f {shlex.quote(pid_path)}; {cmd}; "
        f"printf '[%s] tmux_window_exit run_name={str(meta['run_name'])} observed_child_exit\\n' \"$(date -Iseconds)\" >> {shlex.quote(log_path)}"
    )


def launch(args: Any) -> Dict[str, Any]:
    _append_log("launch_start")
    ctx = _common_launch_context(args)

    actions: List[Dict[str, Any]] = []
    runs: List[Dict[str, Any]] = []
    for spec in _run_specs():
        run_name = str(spec["run_name"])
        if str(spec.get("reuse_source_run_name", "")).strip():
            actions.append(_alias_equivalent_run(args, spec))
            runs.append({"run_name": run_name, "mode": "reused_equivalent", "source_run_name": str(spec["reuse_source_run_name"])})
            continue
        actions.append(_reset_real_run_artifacts(args, run_name))
        meta = _build_launch_meta(args, spec, ctx)
        runs.append(_write_and_launch_meta(args, meta, ctx["existing_windows"]))
    payload = {
        "generated_at_utc": now_iso(),
        "tmux_session": str(args.tmux_session),
        "teacher_backend": BOOTSTRAP_BACKEND,
        "policy": "reuse exact-equivalent completed ablations when present; real-train only missing 20260415 fix runs; persistence disabled",
        "lease_cleanup": ctx["lease_cleanup"],
        "actions": actions,
        "runs": runs,
        "reused_equivalent_count": int(sum(1 for x in runs if x["mode"] == "reused_equivalent")),
        "real_train_count": int(sum(1 for x in runs if x["mode"] == "real_train")),
    }
    _write_json(args.launch_report, payload)
    _append_log(f"launch_complete reused={payload['reused_equivalent_count']} real={payload['real_train_count']}")
    return payload


def run_one(args: Any) -> None:
    meta = _read_json(args.meta_json)
    run_name = str(meta.get("run_name", ""))
    if not run_name:
        raise RuntimeError("meta missing run_name")
    if not str(meta.get("objective_combo", "")).strip():
        meta["objective_combo"] = f"closure_{str(meta.get('ablation_name', 'unknown'))}_seed{int(meta.get('seed', -1))}"
        _write_json(args.meta_json, meta)
    selected_gpu_id = int(meta.get("selected_gpu_id", -1))
    lease_id = str(meta.get("lease_id", ""))
    if selected_gpu_id < 0:
        deadline = time.time() + float(meta.get("gpu_acquire_timeout_seconds", args.gpu_acquire_timeout_seconds))
        retry_seconds = float(meta.get("gpu_acquire_retry_seconds", args.gpu_acquire_retry_seconds))
        last_error = ""
        while time.time() < deadline:
            try:
                gpu = _shared_select_gpu(run_name=run_name, lease_path=str(meta.get("shared_lease_path", args.shared_lease_path)))
                selected_gpu_id = int(gpu["selected_gpu_id"])
                lease_id = str(gpu["lease_id"])
                meta["selected_gpu_id"] = selected_gpu_id
                meta["lease_id"] = lease_id
                meta["selector_payload"] = gpu.get("selector_payload", {})
                _write_json(args.meta_json, meta)
                break
            except Exception as exc:
                last_error = str(exc)
                time.sleep(retry_seconds)
        if selected_gpu_id < 0:
            payload = {
                "generated_at_utc": now_iso(),
                "run_name": run_name,
                "status": "failed",
                "message": f"gpu_acquire_timeout_shared_selector last_error={last_error}",
            }
            _write_json(meta["final_json"], payload)
            raise RuntimeError(payload["message"])
    try:
        signal.signal(signal.SIGHUP, signal.SIG_IGN)
    except Exception:
        pass
    try:
        base.run_one(args)
    finally:
        if lease_id:
            try:
                release_lease(lease_id=lease_id, lease_path=str(meta.get("shared_lease_path", args.shared_lease_path)))
            except Exception:
                pass


def _collect_rows(args: Any) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    meta_dir = _meta_dir(args)
    for spec in _run_specs():
        run_name = str(spec["run_name"])
        meta_json = meta_dir / f"{run_name}_launch_meta.json"
        meta = _read_json(meta_json) if meta_json.exists() else {}
        paths = _paths_for_run(args, run_name)
        raw_payload = _read_json(paths["raw"])
        progress_payload = _read_json(paths["progress"])
        final_payload = _read_json(paths["final"])
        if str(spec.get("reuse_source_run_name", "")).strip():
            resolved_status = "completed"
        else:
            status_info = base._status_for(
                {
                    **meta,
                    "window_name": str(meta.get("window_name", spec.get("window_name", ""))),
                    "progress_json": str(paths["progress"]),
                    "final_json": str(paths["final"]),
                },
                session_name=str(args.tmux_session),
            )
            resolved_status = str(status_info.get("status", "pending"))
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
        if not scientific_result_valid:
            best_block = {}
            latest_block = {}
            sidecar_block = {}
        branch = raw_payload.get("semantic_branch_summary", {}) if isinstance(raw_payload.get("semantic_branch_summary", {}), dict) else {}
        if not branch and isinstance(final_payload.get("semantic_branch_summary", {}), dict):
            branch = final_payload.get("semantic_branch_summary", {})
        if not branch and isinstance(progress_payload.get("semantic_branch_summary", {}), dict):
            branch = progress_payload.get("semantic_branch_summary", {})
        selected_gpu_id, lease_id = base._gpu_selection_from_payload(final_payload, progress_payload, meta)
        rows.append(
            {
                "run_name": run_name,
                "seed": int(spec["seed"]),
                "ablation_name": str(spec["ablation_name"]),
                "status": resolved_status,
                "reused_equivalent": bool(str(spec.get("reuse_source_run_name", "")).strip()),
                "reused_equivalent_source_run_name": str(spec.get("reuse_source_run_name", "")),
                "global_step": int(progress_payload.get("global_step", best_block.get("global_step", -1))),
                "final_json_exists": bool(paths["final"].exists()),
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
                "actual_gate_positive_ratio_mean": float(branch.get("actual_gate_positive_ratio_mean", branch.get("eval_gate_mean", 1.0))) if scientific_result_valid and branch else None,
            }
        )
    return rows


def summarize(args: Any) -> Dict[str, Any]:
    rows = _collect_rows(args)
    running = sum(str(r.get("status", "")).lower() == "running" for r in rows)
    completed = sum(str(r.get("status", "")).lower() == "completed" for r in rows)
    failed = sum(str(r.get("status", "")).lower() == "failed" for r in rows)
    payload = {
        "generated_at_utc": now_iso(),
        "mechanism_ablation_fix_status": f"{running}_running_{completed}_completed_{failed}_failed",
        "running_count": int(running),
        "completed_count": int(completed),
        "failed_count": int(failed),
        "all_runs_terminal": bool(rows and running == 0 and completed + failed == len(rows)),
        "run_rows": rows,
    }
    _write_json(args.summary_report, payload)
    return payload


def diagnose(args: Any) -> Dict[str, Any]:
    summary = summarize(args)
    rows = summary.get("run_rows", []) if isinstance(summary.get("run_rows", []), list) else []
    mainline_summary = _read_json(args.final_pack_summary_report)
    mainline_rows = mainline_summary.get("run_rows", []) if isinstance(mainline_summary.get("run_rows", []), list) else []
    ref_by_seed = {
        int(r.get("seed", -1)): r
        for r in mainline_rows
        if isinstance(r, dict) and str(r.get("family", "")) == "topk1" and str(r.get("status", "")).lower() == "completed"
    }

    def _ep(row: Dict[str, Any]) -> float:
        return base._metric_rank_tuple(row.get("best_checkpoint_metric", {}))[0]

    def _hard(row: Dict[str, Any]) -> float:
        block = row.get("semantic_hard_sidecar_metric", {}) if isinstance(row.get("semantic_hard_sidecar_metric", {}), dict) else {}
        return base._f(block.get("semantic_hard_sidecar_score"), 1e9)

    family_payloads: Dict[str, Any] = {}
    family_flags: Dict[str, bool] = {}
    for family in ["noalign", "densegate", "nodelay"]:
        family_rows = [r for r in rows if str(r.get("ablation_name", "")) == family]
        anomalies: List[Dict[str, Any]] = []
        judgments: List[Dict[str, Any]] = []
        for row in family_rows:
            seed = int(row.get("seed", -1))
            ref = ref_by_seed.get(seed, {})
            degraded = bool(ref and (_ep(row) > _ep(ref) or _hard(row) > _hard(ref)))
            anomaly = bool(ref and (_ep(row) < _ep(ref) or _hard(row) < _hard(ref)))
            judgment = {
                "run_name": str(row.get("run_name", "")),
                "seed": seed,
                "reference_run_name": str(ref.get("run_name", "none")) if ref else "none",
                "reference_endpoint_l2": float(_ep(ref)) if ref else 1e9,
                "reference_semantic_hard_sidecar_score": float(_hard(ref)) if ref else 1e9,
                "endpoint_l2": float(_ep(row)),
                "semantic_hard_sidecar_score": float(_hard(row)),
                "degraded_vs_reference": degraded,
                "anomaly_better_than_reference": anomaly,
                "reused_equivalent": bool(row.get("reused_equivalent", False)),
            }
            judgments.append(judgment)
            if anomaly:
                anomalies.append(judgment)
        family_flag = bool(
            family_rows
            and all(str(r.get("status", "")).lower() == "completed" and bool(r.get("scientific_result_valid", False)) for r in family_rows)
            and len(judgments) == 4
            and all(j["degraded_vs_reference"] and not j["anomaly_better_than_reference"] for j in judgments)
        )
        family_flags[family] = family_flag
        family_payloads[family] = {"judgments": judgments, "anomalies": anomalies, "cross_seed_support": family_flag}

    payload = {
        "generated_at_utc": now_iso(),
        "all_runs_terminal": bool(summary.get("all_runs_terminal", False)),
        "alignment_load_bearing_cross_seed": bool(family_flags.get("noalign", False)),
        "sparse_gating_load_bearing_cross_seed": bool(family_flags.get("densegate", False)),
        "delayed_schedule_load_bearing_cross_seed": bool(family_flags.get("nodelay", False)),
        "families": family_payloads,
    }
    _write_json(args.diagnosis_report, payload)
    lines = [
        "# Stage2 Mechanism Ablation Fix 20260415",
        "",
        f"- alignment_load_bearing_cross_seed: {payload['alignment_load_bearing_cross_seed']}",
        f"- sparse_gating_load_bearing_cross_seed: {payload['sparse_gating_load_bearing_cross_seed']}",
        f"- delayed_schedule_load_bearing_cross_seed: {payload['delayed_schedule_load_bearing_cross_seed']}",
        "",
        "| family | seed | run_name | degraded_vs_reference | anomaly_better_than_reference | reused_equivalent |",
        "|---|---:|---|---|---|---|",
    ]
    for family in ["noalign", "densegate", "nodelay"]:
        for row in family_payloads[family]["judgments"]:
            lines.append(
                f"| {family} | {row['seed']} | {row['run_name']} | {row['degraded_vs_reference']} | {row['anomaly_better_than_reference']} | {row['reused_equivalent']} |"
            )
    _write_md(args.results_md, lines)
    return payload


def run_all(args: Any) -> Dict[str, Any]:
    launch_payload = launch(args)
    deadline = time.time() + float(args.wait_timeout_seconds)
    while time.time() < deadline:
        summary = summarize(args)
        if bool(summary.get("all_runs_terminal", False)):
            diag = diagnose(args)
            return {"launch": launch_payload, "summary": summary, "diagnosis": diag}
        time.sleep(float(args.poll_seconds))
    diag = diagnose(args)
    return {"launch": launch_payload, "summary": summarize(args), "diagnosis": diag, "timeout": True}


def rerun_selected(args: Any) -> Dict[str, Any]:
    requested = [x.strip() for x in str(args.rerun_run_names).split(",") if x.strip()]
    if not requested:
        raise RuntimeError("rerun_selected requires --rerun-run-names")
    spec_map = _run_spec_map()
    unknown = [name for name in requested if name not in spec_map]
    if unknown:
        raise RuntimeError(f"unknown rerun run names: {unknown}")
    ctx = _common_launch_context(args)
    actions: List[Dict[str, Any]] = []
    runs: List[Dict[str, Any]] = []
    for run_name in requested:
        spec = spec_map[run_name]
        if str(spec.get("reuse_source_run_name", "")).strip():
            raise RuntimeError(f"refusing to rerun reused-equivalent completed run: {run_name}")
        actions.append(_reset_real_run_artifacts(args, run_name))
        meta = _build_launch_meta(args, spec, ctx)
        runs.append(_write_and_launch_meta(args, meta, ctx["existing_windows"]))
    payload = {
        "generated_at_utc": now_iso(),
        "tmux_session": str(args.tmux_session),
        "teacher_backend": BOOTSTRAP_BACKEND,
        "policy": "rerun only explicitly requested failed real-train runs; completed equivalent runs untouched",
        "lease_cleanup": ctx["lease_cleanup"],
        "actions": actions,
        "runs": runs,
        "reused_equivalent_count": 0,
        "real_train_count": len(runs),
        "rerun_selected_only": True,
        "rerun_run_names": requested,
    }
    _write_json(args.launch_report, payload)
    _append_log(f"rerun_selected_complete count={len(runs)} runs={requested}")
    return payload


def parse_args() -> Any:
    parser = ArgumentParser()
    parser.add_argument("--mode", default="all", choices=["all", "launch", "run-one", "summarize", "diagnose", "rerun-selected"])
    parser.add_argument("--meta-json", default="")
    parser.add_argument("--rerun-run-names", default="")
    parser.add_argument("--work-root", default=str(ROOT))
    parser.add_argument("--python-bin", default=base._python_bin_default())
    parser.add_argument("--tmux-session", default=SESSION)
    parser.add_argument("--stage2-contract-json", default=str(ROOT / "reports/stage2_bootstrap_data_contract_20260408.json"))
    parser.add_argument("--stage1-best-ckpt", default=str(ROOT / "outputs/checkpoints/stage1_v2_longtrain_220m_mainline_20260408/best.pt"))
    parser.add_argument("--shared-lease-path", default=str(ROOT / "reports/stage1_v2_gpu_lease_20260408.json"))
    parser.add_argument("--bootstrap-cache-jsonl", default=str(ROOT / "data/processed/stage2_real_bootstrap_cache_20260410/clip_vit_b32_core_trainval_required_subset.jsonl"))
    parser.add_argument("--semantic-hard-manifest-path", default=str(ROOT / "manifests/protocol_v2/stage2_semantic_hard_subsets_20260410.json"))
    parser.add_argument("--runtime-json", default=str(ROOT / "reports/stage2_calibration_only_wave2_runtime_20260414.json"))
    parser.add_argument("--final-pack-summary-report", default=str(ROOT / "reports/stage2_calibration_only_final_pack_summary_20260414.json"))
    parser.add_argument("--launch-report", default=str(ROOT / "reports/stage2_mechanism_ablation_fix_launch_20260415.json"))
    parser.add_argument("--summary-report", default=str(ROOT / "reports/stage2_mechanism_ablation_fix_summary_20260415.json"))
    parser.add_argument("--diagnosis-report", default=str(ROOT / "reports/stage2_mechanism_ablation_fix_diagnosis_20260415.json"))
    parser.add_argument("--results-md", default=str(ROOT / "docs/STAGE2_MECHANISM_ABLATION_FIX_20260415.md"))
    parser.add_argument("--gpu-acquire-timeout-seconds", type=int, default=28800)
    parser.add_argument("--gpu-acquire-retry-seconds", type=int, default=20)
    parser.add_argument("--wait-timeout-seconds", type=int, default=172800)
    parser.add_argument("--poll-seconds", type=int, default=120)
    return parser.parse_args()


def main() -> None:
    _apply_process_title()
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
        print(json.dumps(diagnose(args), ensure_ascii=True, indent=2))
    elif args.mode == "rerun-selected":
        print(json.dumps(rerun_selected(args), ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
