#!/usr/bin/env python3
from __future__ import annotations

from argparse import ArgumentParser
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List
import json
import subprocess
import time

from stwm.tools import run_stage2_mechanism_ablation_fix_20260415 as prevfix
from stwm.tools import run_tracewm_stage2_calibration_only_fullscale_wave1_20260413 as base


ROOT = prevfix.ROOT
SESSION = "tracewm_stage2_oral_hardening_20260416"
LOG_PATH = ROOT / "logs/stage2_oral_hardening_20260416.log"
BOOTSTRAP_BACKEND = "local_clip_vit_b32_mask_crop_visual_teacher"
DATE_TAG = "20260416"
BATCH_SIZE = 8
EXTRA_STEPS = 2500
EVAL_INTERVAL = 250
EVAL_MAX_BATCHES = 0
SAVE_EVERY = 250
MAX_TRAIN = 4096
MAX_VAL = 1024
RESUME_RUN_BY_SEED = {
    42: "stage2_calonly_topk1_seed42_wave1_20260413",
    123: "stage2_calonly_topk1_seed123_wave1_20260413",
}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _append_log(message: str) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as fh:
        fh.write(f"[{now_iso()}] {message}\n")


def _resume_ckpt_for_seed(seed: int) -> Path:
    run_name = RESUME_RUN_BY_SEED.get(int(seed), RESUME_RUN_BY_SEED[123])
    ckpt = ROOT / "outputs/checkpoints" / run_name / "best.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"missing resume checkpoint for seed {seed}: {ckpt}")
    return ckpt


def _spec_base(run_name: str, seed: int, local_temporal_window: int, window_name: str) -> Dict[str, Any]:
    spec = prevfix._spec_base(run_name=run_name, seed=seed, ablation_name="localtemp", window_name=window_name, reuse_source_run_name="")
    spec["objective_combo"] = f"oral_hardening_localtemp_w{int(local_temporal_window)}_seed{int(seed)}"
    spec["track"] = "local_temporal_pilot"
    spec["family"] = f"localtemp_w{int(local_temporal_window)}"
    spec["objective_family"] = "local_temporal_semantic_branch"
    spec["local_temporal_window"] = int(local_temporal_window)
    spec["local_temporal_fuse_weight"] = 0.5
    return spec


def _run_specs() -> List[Dict[str, Any]]:
    return [
        _spec_base("stage2_localtemp_w3_seed42_20260416", 42, 3, "ltv3_w3_42"),
        _spec_base("stage2_localtemp_w3_seed123_20260416", 123, 3, "ltv3_w3_123"),
        _spec_base("stage2_localtemp_w5_seed42_20260416", 42, 5, "ltv3_w5_42"),
        _spec_base("stage2_localtemp_w5_seed123_20260416", 123, 5, "ltv3_w5_123"),
    ]


def _meta_dir(args: Any) -> Path:
    return Path(args.work_root) / "reports/stage2_local_temporal_semantic_branch_runs_20260416"


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


def _common_launch_context(args: Any) -> Dict[str, Any]:
    lease_cleanup = base._cleanup_stale_leases(str(args.shared_lease_path), allowed_prefixes=("stage2_calonly_", "stage2_localtemp_"))
    if subprocess.run(["tmux", "has-session", "-t", str(args.tmux_session)], capture_output=True).returncode != 0:
        subprocess.run(["tmux", "new-session", "-d", "-s", str(args.tmux_session), "bash"], check=True)
    existing_windows = set(base._tmux_windows(str(args.tmux_session)))
    anchor_args = base._load_ckpt_args(_resume_ckpt_for_seed(42))
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
        "predecode_cache_path": str(args.predecode_cache_path),
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


def _tmux_window_command(args: Any, meta_json: Path, meta: Dict[str, Any]) -> str:
    import os
    import shlex
    pid_path = str(meta["worker_pid_file"])
    log_path = str(meta["log_path"])
    pythonpath_value = f"{args.work_root}/code:{os.environ.get('PYTHONPATH', '')}"
    proc_title = str(os.environ.get("STWM_PROC_TITLE", "python"))
    proc_title_mode = str(os.environ.get("STWM_PROC_TITLE_MODE", "generic"))
    script_path = Path(args.work_root) / "code/stwm/tools/run_stage2_local_temporal_semantic_branch_20260416.py"
    cmd = (
        f"PYTHONPATH={shlex.quote(pythonpath_value)} "
        f"STWM_PROC_TITLE={shlex.quote(proc_title)} "
        f"STWM_PROC_TITLE_MODE={shlex.quote(proc_title_mode)} "
        f"nohup {shlex.quote(str(args.python_bin))} {shlex.quote(str(script_path))} --mode run-one --meta-json {shlex.quote(str(meta_json))} --work-root {shlex.quote(str(args.work_root))} "
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
    prevfix._write_json(meta_json, meta)
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
        "local_temporal_window": int(meta["local_temporal_window"]),
    }


def launch(args: Any) -> Dict[str, Any]:
    _append_log("local_temporal_launch_start")
    ctx = _common_launch_context(args)
    actions: List[Dict[str, Any]] = []
    runs: List[Dict[str, Any]] = []
    for spec in _run_specs():
        run_name = str(spec["run_name"])
        actions.append(prevfix._reset_real_run_artifacts(args, run_name))
        meta = _build_launch_meta(args, spec, ctx)
        runs.append(_write_and_launch_meta(args, meta, ctx["existing_windows"]))
    payload = {
        "generated_at_utc": now_iso(),
        "tmux_session": str(args.tmux_session),
        "teacher_backend": BOOTSTRAP_BACKEND,
        "policy": "small-scale local temporal semantic branch pilot; calibration-only backbone unchanged; persistence disabled",
        "lease_cleanup": ctx["lease_cleanup"],
        "actions": actions,
        "runs": runs,
        "real_train_count": len(runs),
    }
    prevfix._write_json(args.launch_report, payload)
    return payload


def run_one(args: Any) -> None:
    base.run_one(args)


def _collect_rows(args: Any) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    meta_dir = _meta_dir(args)
    for spec in _run_specs():
        run_name = str(spec["run_name"])
        meta_json = meta_dir / f"{run_name}_launch_meta.json"
        meta = prevfix._read_json(meta_json) if meta_json.exists() else {}
        paths = _paths_for_run(args, run_name)
        raw_payload = prevfix._read_json(paths["raw"])
        progress_payload = prevfix._read_json(paths["progress"])
        final_payload = prevfix._read_json(paths["final"])
        status_info = base._status_for(
            {**meta, "window_name": str(meta.get("window_name", spec.get("window_name", ""))), "progress_json": str(paths["progress"]), "final_json": str(paths["final"])},
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
        branch = base._branch_block(final_payload, raw_payload, progress_payload)
        if not scientific_result_valid:
            best_block = {}
            latest_block = {}
            sidecar_block = {}
            branch = {}
        selected_gpu_id, lease_id = base._gpu_selection_from_payload(final_payload, progress_payload, meta)
        rows.append(
            {
                "run_name": run_name,
                "family": str(spec["family"]),
                "seed": int(spec["seed"]),
                "local_temporal_window": int(spec["local_temporal_window"]),
                "status": resolved_status,
                "global_step": int(progress_payload.get("global_step", best_block.get("global_step", -1))),
                "final_json_exists": bool(paths["final"].exists()),
                "best_ckpt_exists": best_ckpt_exists,
                "latest_ckpt_exists": latest_ckpt_exists,
                "sidecar_exists": sidecar_exists,
                "scientific_result_valid": bool(scientific_result_valid),
                "selected_gpu_id": int(selected_gpu_id),
                "lease_id": str(lease_id),
                "best_checkpoint_metric": best_block,
                "semantic_hard_sidecar_metric": sidecar_block,
                "actual_gate_positive_ratio_mean": (
                    float(branch.get("actual_gate_positive_ratio_mean", branch.get("eval_gate_mean", 1.0)))
                    if scientific_result_valid and isinstance(branch, dict) and branch
                    else None
                ),
            }
        )
    return rows


def summarize(args: Any) -> Dict[str, Any]:
    rows = _collect_rows(args)
    running = sum(str(r.get("status", "")).lower() == "running" for r in rows)
    completed = sum(str(r.get("status", "")).lower() == "completed" for r in rows)
    failed = sum(str(r.get("status", "")).lower() == "failed" for r in rows)
    completed_rows = [row for row in rows if str(row.get("status", "")).lower() == "completed" and bool(row.get("scientific_result_valid", False))]
    best_run_name = "none"
    best_window = -1
    case_candidates: List[Dict[str, Any]] = []
    if completed_rows:
        def _hard_score(row: Dict[str, Any]) -> float:
            block = row.get("semantic_hard_sidecar_metric", {}) if isinstance(row.get("semantic_hard_sidecar_metric", {}), dict) else {}
            return float(base._f(block.get("semantic_hard_sidecar_score"), 1e9))
        best = min(completed_rows, key=lambda row: (
            _hard_score(row),
            base._metric_rank_tuple(row.get("best_checkpoint_metric", {}))[0],
            str(row.get("run_name", "")),
        ))
        best_run_name = str(best.get("run_name", "none"))
        best_window = int(best.get("local_temporal_window", -1))
        case_candidates = [
            {
                "run_name": str(row.get("run_name", "")),
                "seed": int(row.get("seed", -1)),
                "local_temporal_window": int(row.get("local_temporal_window", -1)),
                "endpoint_l2": float(base._metric_rank_tuple(row.get("best_checkpoint_metric", {}))[0]),
                "semantic_hard_sidecar_score": _hard_score(row),
            }
            for row in sorted(
                completed_rows,
                key=lambda row: (
                    _hard_score(row),
                    base._metric_rank_tuple(row.get("best_checkpoint_metric", {}))[0],
                    str(row.get("run_name", "")),
                ),
            )[:4]
        ]
    payload = {
        "generated_at_utc": now_iso(),
        "local_temporal_branch_status": f"{running}_running_{completed}_completed_{failed}_failed",
        "running_count": int(running),
        "completed_count": int(completed),
        "failed_count": int(failed),
        "all_runs_terminal": bool(rows and running == 0 and completed + failed == len(rows)),
        "best_run_name": best_run_name,
        "best_window": int(best_window),
        "case_candidates": case_candidates,
        "run_rows": rows,
    }
    prevfix._write_json(args.summary_report, payload)
    return payload


def diagnose(args: Any) -> Dict[str, Any]:
    summary = summarize(args)
    final_pack_diag = prevfix._read_json(args.final_pack_diagnosis)
    final_pack_summary = prevfix._read_json(args.final_pack_summary)
    baseline_run_name = str(final_pack_diag.get("current_best_overall_run_name", "stage2_calonly_topk1_seed123_wave1_20260413"))
    baseline_row = next(
        (
            row for row in (final_pack_summary.get("run_rows", []) if isinstance(final_pack_summary.get("run_rows", []), list) else [])
            if isinstance(row, dict) and str(row.get("run_name", "")) == baseline_run_name
        ),
        {},
    )
    completed_rows = [row for row in summary.get("run_rows", []) if isinstance(row, dict) and str(row.get("status", "")).lower() == "completed" and bool(row.get("scientific_result_valid", False))]
    baseline_endpoint = float(base._metric_rank_tuple(baseline_row.get("best_checkpoint_metric", {}))[0]) if baseline_row else 1e9
    baseline_hard = float(base._f((baseline_row.get("semantic_hard_sidecar_metric", {}) if isinstance(baseline_row.get("semantic_hard_sidecar_metric", {}), dict) else {}).get("semantic_hard_sidecar_score"), 1e9)) if baseline_row else 1e9
    best_local = None
    if completed_rows:
        best_local = min(
            completed_rows,
            key=lambda row: (
                float(base._f((row.get("semantic_hard_sidecar_metric", {}) if isinstance(row.get("semantic_hard_sidecar_metric", {}), dict) else {}).get("semantic_hard_sidecar_score"), 1e9)),
                float(base._metric_rank_tuple(row.get("best_checkpoint_metric", {}))[0]),
                str(row.get("run_name", "")),
            ),
        )
    best_endpoint = float(base._metric_rank_tuple((best_local or {}).get("best_checkpoint_metric", {}))[0]) if best_local else 1e9
    best_hard = float(base._f((((best_local or {}).get("semantic_hard_sidecar_metric", {}) if isinstance((best_local or {}).get("semantic_hard_sidecar_metric", {}), dict) else {}).get("semantic_hard_sidecar_score")), 1e9)) if best_local else 1e9
    hard_subset_improved = bool(best_local and best_hard < baseline_hard)
    non_catastrophic = bool(best_local and best_endpoint <= max(baseline_endpoint * 1.05, baseline_endpoint + 5e-5))
    local_temporal_branch_promising = bool(best_local and hard_subset_improved and non_catastrophic)
    payload = {
        "generated_at_utc": now_iso(),
        "all_runs_terminal": bool(summary.get("all_runs_terminal", False)),
        "baseline_run_name": baseline_run_name,
        "baseline_endpoint_l2": baseline_endpoint,
        "baseline_semantic_hard_sidecar_score": baseline_hard,
        "best_local_temporal_run_name": str((best_local or {}).get("run_name", "none")),
        "best_local_temporal_window": int((best_local or {}).get("local_temporal_window", -1)),
        "best_local_temporal_endpoint_l2": best_endpoint,
        "best_local_temporal_semantic_hard_sidecar_score": best_hard,
        "hard_subset_improved": bool(hard_subset_improved),
        "non_catastrophic_vs_baseline": bool(non_catastrophic),
        "local_temporal_branch_promising": bool(local_temporal_branch_promising),
        "case_candidates": list(summary.get("case_candidates", [])),
    }
    prevfix._write_json(args.diagnosis_report, payload)
    prevfix._write_md(
        args.results_md,
        [
            "# Stage2 Local Temporal Semantic Branch 20260416",
            "",
            f"- baseline_run_name: {payload['baseline_run_name']}",
            f"- best_local_temporal_run_name: {payload['best_local_temporal_run_name']}",
            f"- best_local_temporal_window: {payload['best_local_temporal_window']}",
            f"- hard_subset_improved: {payload['hard_subset_improved']}",
            f"- non_catastrophic_vs_baseline: {payload['non_catastrophic_vs_baseline']}",
            f"- local_temporal_branch_promising: {payload['local_temporal_branch_promising']}",
        ],
    )
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


def parse_args() -> Any:
    parser = ArgumentParser()
    parser.add_argument("--mode", default="all", choices=["all", "launch", "run-one", "summarize", "diagnose"])
    parser.add_argument("--meta-json", default="")
    parser.add_argument("--work-root", default=str(ROOT))
    parser.add_argument("--python-bin", default=base._python_bin_default())
    parser.add_argument("--tmux-session", default=SESSION)
    parser.add_argument("--stage2-contract-json", default=str(ROOT / "reports/stage2_bootstrap_data_contract_20260408.json"))
    parser.add_argument("--stage1-best-ckpt", default=str(ROOT / "outputs/checkpoints/stage1_v2_longtrain_220m_mainline_20260408/best.pt"))
    parser.add_argument("--shared-lease-path", default=str(ROOT / "reports/stage1_v2_gpu_lease_20260408.json"))
    parser.add_argument("--bootstrap-cache-jsonl", default=str(ROOT / "data/processed/stage2_real_bootstrap_cache_20260410/clip_vit_b32_core_trainval_required_subset.jsonl"))
    parser.add_argument("--semantic-hard-manifest-path", default=str(ROOT / "manifests/protocol_v2/stage2_semantic_hard_subsets_20260410.json"))
    parser.add_argument("--runtime-json", default=str(ROOT / "configs/recommended_stage2_runtime_20260416.json"))
    parser.add_argument("--predecode-cache-path", default=str(ROOT / "data/processed/stage2_predecode_cache_20260416"))
    parser.add_argument("--final-pack-summary", default=str(ROOT / "reports/stage2_calibration_only_final_pack_summary_20260414.json"))
    parser.add_argument("--final-pack-diagnosis", default=str(ROOT / "reports/stage2_calibration_only_final_pack_diagnosis_20260414.json"))
    parser.add_argument("--launch-report", default=str(ROOT / "reports/stage2_local_temporal_semantic_branch_launch_20260416.json"))
    parser.add_argument("--summary-report", default=str(ROOT / "reports/stage2_local_temporal_semantic_branch_summary_20260416.json"))
    parser.add_argument("--diagnosis-report", default=str(ROOT / "reports/stage2_local_temporal_semantic_branch_diagnosis_20260416.json"))
    parser.add_argument("--results-md", default=str(ROOT / "docs/STAGE2_LOCAL_TEMPORAL_SEMANTIC_BRANCH_20260416.md"))
    parser.add_argument("--gpu-acquire-timeout-seconds", type=int, default=28800)
    parser.add_argument("--gpu-acquire-retry-seconds", type=int, default=20)
    parser.add_argument("--wait-timeout-seconds", type=int, default=172800)
    parser.add_argument("--poll-seconds", type=int, default=120)
    return parser.parse_args()


def main() -> None:
    prevfix._apply_process_title()
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


if __name__ == "__main__":
    main()
