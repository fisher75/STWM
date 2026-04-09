#!/usr/bin/env python3
from __future__ import annotations

from argparse import ArgumentParser
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List
import json
import os
import shlex
import subprocess
import sys
import traceback

import torch

from stwm.infra.gpu_lease import acquire_lease, release_lease
from stwm.infra.gpu_selector import select_single_gpu
from stwm.tracewm_v2_stage2.datasets.stage2_semantic_dataset import (
    Stage2SemanticDataset,
    Stage2SemanticDatasetConfig,
)


DATE_TAG = "20260409"
WORK_ROOT = Path("/home/chen034/workspace/stwm")
DATA_ROOT = Path("/home/chen034/workspace/data")


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"json not found: {p}")
    payload = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"json payload must be dict: {p}")
    return payload


def _write_json(path: str | Path, payload: Dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def _write_md(path: str | Path, lines: List[str]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _tail(text: str, limit: int = 4000) -> str:
    s = str(text)
    if len(s) <= limit:
        return s
    return s[-limit:]


def _f(x: Any, default: float = 1e9) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def _i(x: Any, default: int = -1) -> int:
    try:
        return int(x)
    except Exception:
        return int(default)


def _split_counts_used(summary: Dict[str, Dict[str, Any]]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for name, meta in summary.items():
        if not isinstance(meta, dict):
            continue
        out[str(name)] = int(meta.get("sample_count", 0) or 0)
    return out


def _python_bin_default() -> str:
    preferred = WORK_ROOT / ".." / ".." / "miniconda3" / "envs" / "stwm" / "bin" / "python"
    preferred = preferred.resolve()
    if preferred.exists():
        return str(preferred)
    return sys.executable


def parse_args() -> Any:
    p = ArgumentParser(description="Stage2 full-scale wave1 launcher / worker")
    p.add_argument("--mode", default="orchestrate", choices=["orchestrate", "run-one"])
    p.add_argument("--meta-json", default="")
    p.add_argument("--work-root", default=str(WORK_ROOT))
    p.add_argument("--data-root", default=str(DATA_ROOT))
    p.add_argument("--python-bin", default=_python_bin_default())
    p.add_argument("--tmux-session", default="tracewm_stage2_fullscale_wave1_20260409")
    p.add_argument("--stage2-eval-fix-json", default=str(WORK_ROOT / "reports" / "stage2_eval_fix_comparison_20260408.json"))
    p.add_argument("--stage2-mainline-final-json", default=str(WORK_ROOT / "reports" / "stage2_core_mainline_train_final_20260408.json"))
    p.add_argument("--stage2-mainline-raw-json", default=str(WORK_ROOT / "reports" / "stage2_core_mainline_train_raw_20260408.json"))
    p.add_argument("--stage2-contract-json", default=str(WORK_ROOT / "reports" / "stage2_bootstrap_data_contract_20260408.json"))
    p.add_argument("--stage1-runtime-json", default=str(WORK_ROOT / "reports" / "stage1_v2_recommended_runtime_20260408.json"))
    p.add_argument("--project-readiness-json", default=str(WORK_ROOT / "reports" / "tracewm_project_readiness_20260409.json"))
    p.add_argument("--external-fidelity-json", default=str(WORK_ROOT / "reports" / "stage2_external_eval_fidelity_audit_20260409.json"))
    p.add_argument("--stage1-best-ckpt", default=str(WORK_ROOT / "outputs" / "checkpoints" / "stage1_v2_longtrain_220m_mainline_20260408" / "best.pt"))
    p.add_argument("--shared-lease-path", default=str(WORK_ROOT / "reports" / "stage1_v2_gpu_lease_20260408.json"))
    p.add_argument("--runtime-preflight-report", default=str(WORK_ROOT / "reports" / "stage2_fullscale_wave1_runtime_preflight_20260409.json"))
    p.add_argument("--launch-report", default=str(WORK_ROOT / "reports" / "stage2_fullscale_wave1_launch_20260409.json"))
    p.add_argument("--summary-report", default=str(WORK_ROOT / "reports" / "stage2_fullscale_wave1_summary_20260409.json"))
    p.add_argument("--results-md", default=str(WORK_ROOT / "docs" / "STAGE2_FULLSCALE_WAVE1_RESULTS_20260409.md"))
    p.add_argument("--protocol-md", default=str(WORK_ROOT / "docs" / "TRACEWM_STAGE2_FULLSCALE_WAVE1_PROTOCOL_20260409.md"))
    return p.parse_args()


def _validate_frozen_facts(args: Any) -> Dict[str, Any]:
    eval_fix = _read_json(args.stage2_eval_fix_json)
    final_payload = _read_json(args.stage2_mainline_final_json)
    contract = _read_json(args.stage2_contract_json)
    readiness = _read_json(args.project_readiness_json)
    fidelity = _read_json(args.external_fidelity_json)

    if str(eval_fix.get("final_recommended_mainline", "")) != "stage2_core_cropenc":
        raise RuntimeError("final_recommended_mainline must remain stage2_core_cropenc")
    if str(final_payload.get("current_mainline_semantic_source", "")) != "crop_visual_encoder":
        raise RuntimeError("current_mainline_semantic_source must remain crop_visual_encoder")
    if [str(x).lower() for x in final_payload.get("datasets_bound_for_train", [])] != ["vspw", "vipseg"]:
        raise RuntimeError("datasets_bound_for_train must remain VSPW+VIPSeg")
    if [str(x).lower() for x in final_payload.get("datasets_bound_for_eval", [])] != ["vspw", "vipseg"]:
        raise RuntimeError("datasets_bound_for_eval must remain VSPW+VIPSeg")
    if str(readiness.get("project_readiness", "")) != "training_ready_but_eval_gap_remains":
        raise RuntimeError("project_readiness must remain training_ready_but_eval_gap_remains before wave1 launch")
    if not bool(fidelity.get("official_evaluator_invoked", False)):
        raise RuntimeError("official_evaluator_invoked must remain true")
    if bool(fidelity.get("official_task_faithfully_instantiated", True)):
        raise RuntimeError("official_task_faithfully_instantiated must remain false")
    if str(fidelity.get("tap_style_eval_status", "")) != "partially_bridged":
        raise RuntimeError("tap_style_eval_status must remain partially_bridged")
    if str(fidelity.get("tap3d_style_eval_status", "")) != "not_yet_implemented":
        raise RuntimeError("tap3d_style_eval_status must remain not_yet_implemented")

    excluded = contract.get("excluded_datasets", []) if isinstance(contract.get("excluded_datasets", []), list) else []
    ex_map = {str(x.get("dataset_name", "")).upper(): x for x in excluded if isinstance(x, dict)}
    if str(ex_map.get("TAO", {}).get("reason", "")) != "access_ready":
        raise RuntimeError("TAO must remain access_ready / excluded")
    if str(ex_map.get("VISOR", {}).get("reason", "")) != "manual_gate":
        raise RuntimeError("VISOR must remain manual_gate / excluded")

    return {
        "eval_fix": eval_fix,
        "final_payload": final_payload,
        "contract": contract,
        "readiness": readiness,
        "fidelity": fidelity,
    }


def _load_anchor_args(best_ckpt_path: str | Path) -> Dict[str, Any]:
    ckpt = torch.load(Path(best_ckpt_path), map_location="cpu", weights_only=False)
    args = ckpt.get("args", {}) if isinstance(ckpt.get("args", {}), dict) else {}
    return args


def _estimate_effective_counts(
    *,
    dataset_names: List[str],
    split: str,
    stage2_contract_path: str,
    obs_len: int,
    fut_len: int,
    max_tokens: int,
    max_samples_per_dataset: int,
    semantic_crop_size: int,
    semantic_source_mainline: str,
) -> Dict[str, int]:
    ds = Stage2SemanticDataset(
        Stage2SemanticDatasetConfig(
            dataset_names=list(dataset_names),
            split=str(split),
            contract_path=str(stage2_contract_path),
            obs_len=int(obs_len),
            fut_len=int(fut_len),
            max_tokens=int(max_tokens),
            max_samples_per_dataset=int(max_samples_per_dataset),
            semantic_patch_radius=12,
            semantic_crop_size=int(semantic_crop_size),
            semantic_source_mainline=str(semantic_source_mainline),
        )
    )
    return _split_counts_used(ds.dataset_summary)


def _select_gpu_with_lease(
    *,
    required_mem_gb: float,
    safety_margin_gb: float,
    lease_path: str,
    owner: str,
    ttl_seconds: int,
    sample_count: int,
    interval_sec: float,
) -> Dict[str, Any]:
    selector_payload = select_single_gpu(
        required_mem_gb=float(required_mem_gb),
        safety_margin_gb=float(safety_margin_gb),
        sample_count=int(sample_count),
        interval_sec=float(interval_sec),
        lease_path=str(lease_path),
    )
    selected_gpu_id = int(selector_payload.get("selected_gpu_id", -1))
    if selected_gpu_id < 0:
        raise RuntimeError("no GPU candidate available after selector + lease filter")
    lease = acquire_lease(
        gpu_id=int(selected_gpu_id),
        owner=str(owner),
        ttl_seconds=int(ttl_seconds),
        lease_path=str(lease_path),
    )
    selected_row = {}
    for row in selector_payload.get("gpus", []) if isinstance(selector_payload.get("gpus", []), list) else []:
        if int(row.get("gpu_id", -1)) == int(selected_gpu_id):
            selected_row = dict(row)
            break
    return {
        "selected_gpu_id": int(selected_gpu_id),
        "lease_id": str(lease.get("lease_id", "")),
        "selector_payload": selector_payload,
        "selected_row": selected_row,
        "lease": lease,
    }


def _release_lease_safe(lease_id: str, lease_path: str) -> None:
    if not str(lease_id).strip():
        return
    try:
        release_lease(lease_id=str(lease_id), lease_path=str(lease_path))
    except Exception:
        pass


def _preflight_log_path(work_root: Path, batch_size: int) -> Path:
    return work_root / "logs" / f"stage2_fullscale_wave1_preflight_bs{int(batch_size)}_{DATE_TAG}.log"


def _run_preflight(args: Any, anchor_args: Dict[str, Any], runtime_json: Dict[str, Any]) -> Dict[str, Any]:
    required_mem_gb = float(runtime_json.get("required_mem_gb", 40.0) or 40.0)
    safety_margin_gb = float(runtime_json.get("safety_margin_gb", 8.0) or 8.0)
    gpu_pick = _select_gpu_with_lease(
        required_mem_gb=required_mem_gb,
        safety_margin_gb=safety_margin_gb,
        lease_path=str(args.shared_lease_path),
        owner="tracewm_stage2_fullscale_wave1_preflight_20260409",
        ttl_seconds=2 * 3600,
        sample_count=4,
        interval_sec=0.5,
    )
    selected_gpu_id = int(gpu_pick["selected_gpu_id"])
    lease_id = str(gpu_pick["lease_id"])

    attempted: List[Dict[str, Any]] = []
    selected_batch_size = -1
    selected_counts_train: Dict[str, int] = {}
    selected_counts_val: Dict[str, int] = {}

    try:
        for batch_size in [2, 4, 8]:
            run_name = f"stage2_fullscale_wave1_preflight_bs{int(batch_size)}_{DATE_TAG}"
            checkpoint_dir = Path(args.work_root) / "outputs" / "checkpoints" / run_name
            raw_json = Path(args.work_root) / "reports" / f"{run_name}_raw.json"
            progress_json = Path(args.work_root) / "reports" / f"{run_name}_progress.json"
            log_path = _preflight_log_path(Path(args.work_root), int(batch_size))

            cmd = [
                str(args.python_bin),
                str(Path(args.work_root) / "code" / "stwm" / "tracewm_v2_stage2" / "trainers" / "train_tracewm_stage2_smalltrain.py"),
                "--stage2-contract-path",
                str(args.stage2_contract_json),
                "--recommended-runtime-json",
                str(args.stage1_runtime_json),
                "--use-recommended-runtime",
                "--stage1-backbone-checkpoint",
                str(args.stage1_best_ckpt),
                "--dataset-names",
                "vspw",
                "vipseg",
                "--train-split",
                "train",
                "--val-split",
                "val",
                "--obs-len",
                str(int(anchor_args.get("obs_len", 8) or 8)),
                "--fut-len",
                str(int(anchor_args.get("fut_len", 8) or 8)),
                "--max-tokens",
                str(int(anchor_args.get("max_tokens", 64) or 64)),
                "--max-samples-train",
                "-1",
                "--max-samples-val",
                "-1",
                "--batch-size",
                str(int(batch_size)),
                "--lr",
                str(float(anchor_args.get("lr", 1e-4) or 1e-4)),
                "--weight-decay",
                str(float(anchor_args.get("weight_decay", 0.0) or 0.0)),
                "--clip-grad-norm",
                str(float(anchor_args.get("clip_grad_norm", 1.0) or 1.0)),
                "--train-steps",
                "30",
                "--eval-interval",
                "15",
                "--eval-max-batches",
                "2",
                "--save-every-n-steps",
                "1000",
                "--semantic-hidden-dim",
                str(int(anchor_args.get("semantic_hidden_dim", 256) or 256)),
                "--semantic-embed-dim",
                str(int(anchor_args.get("semantic_embed_dim", 256) or 256)),
                "--semantic-source-mainline",
                "crop_visual_encoder",
                "--legacy-semantic-source",
                str(anchor_args.get("legacy_semantic_source", "hand_crafted_stats") or "hand_crafted_stats"),
                "--semantic-crop-size",
                str(int(anchor_args.get("semantic_crop_size", 64) or 64)),
                "--output-dir",
                str(checkpoint_dir),
                "--run-name",
                run_name,
                "--run-summary-json",
                str(raw_json),
                "--progress-json",
                str(progress_json),
                "--seed",
                "42",
            ]
            env = os.environ.copy()
            env["PYTHONPATH"] = f"{args.work_root}/code:{env.get('PYTHONPATH', '')}"
            env["CUDA_VISIBLE_DEVICES"] = str(selected_gpu_id)
            gpu_meta = {
                "selected_gpu_id": int(selected_gpu_id),
                "lease_id": str(lease_id),
                "owner": "tracewm_stage2_fullscale_wave1_preflight_20260409",
                "mode": "single_gpu_only",
            }
            env["TRACEWM_STAGE1_V2_GPU_SELECTION_METADATA"] = ";".join(f"{k}={v}" for k, v in gpu_meta.items())
            env["TRACEWM_STAGE1_V2_GPU_SELECTION_METADATA_JSON"] = json.dumps(gpu_meta, ensure_ascii=True)

            proc = subprocess.run(
                cmd,
                cwd=str(args.work_root),
                env=env,
                text=True,
                capture_output=True,
            )
            log_path.write_text(
                proc.stdout + ("\n" if proc.stdout else "") + proc.stderr,
                encoding="utf-8",
            )
            raw_payload = _read_json(raw_json) if raw_json.exists() else {}
            success = bool(
                proc.returncode == 0
                and raw_json.exists()
                and bool(raw_payload.get("run_stable", False))
                and bool(raw_payload.get("boundary_ok", False))
            )
            row = {
                "batch_size": int(batch_size),
                "selected_gpu_id": int(selected_gpu_id),
                "lease_id": str(lease_id),
                "returncode": int(proc.returncode),
                "success": bool(success),
                "raw_json": str(raw_json),
                "progress_json": str(progress_json),
                "log_path": str(log_path),
                "stdout_tail": _tail(proc.stdout),
                "stderr_tail": _tail(proc.stderr),
                "run_stable": bool(raw_payload.get("run_stable", False)) if raw_payload else False,
                "boundary_ok": bool(raw_payload.get("boundary_ok", False)) if raw_payload else False,
                "effective_train_sample_count_per_dataset": raw_payload.get("effective_train_sample_count_per_dataset", {}),
                "effective_val_sample_count_per_dataset": raw_payload.get("effective_val_sample_count_per_dataset", {}),
                "final_metrics": raw_payload.get("final_metrics", {}),
            }
            attempted.append(row)
            if success:
                selected_batch_size = int(batch_size)
                selected_counts_train = dict(raw_payload.get("effective_train_sample_count_per_dataset", {}))
                selected_counts_val = dict(raw_payload.get("effective_val_sample_count_per_dataset", {}))
            else:
                break
    finally:
        _release_lease_safe(lease_id=lease_id, lease_path=str(args.shared_lease_path))

    if selected_batch_size < 0:
        raise RuntimeError("preflight failed for all batch sizes {2,4,8}")

    payload = {
        "generated_at_utc": now_iso(),
        "mode": "stage2_fullscale_wave1_runtime_preflight",
        "selected_gpu_id": int(selected_gpu_id),
        "lease_id": str(lease_id),
        "lease_released_after_preflight": True,
        "candidate_batch_sizes": [2, 4, 8],
        "attempted": attempted,
        "selected_batch_size": int(selected_batch_size),
        "whether_full_train_used": True,
        "whether_full_val_used": True,
        "effective_train_sample_count_per_dataset": selected_counts_train,
        "effective_val_sample_count_per_dataset": selected_counts_val,
    }
    _write_json(args.runtime_preflight_report, payload)
    return payload


def _run_placeholder_status(meta: Dict[str, Any], status: str, message: str = "") -> Dict[str, Any]:
    return {
        "generated_at_utc": now_iso(),
        "run_name": str(meta.get("run_name", "")),
        "status": str(status),
        "selected_gpu_id": int(meta.get("selected_gpu_id", -1)),
        "lease_id": str(meta.get("lease_id", "")),
        "datasets_bound_for_train": meta.get("dataset_names", []),
        "datasets_bound_for_eval": meta.get("dataset_names", []),
        "batch_size": int(meta.get("batch_size", 0)),
        "train_steps": int(meta.get("train_steps", 0)),
        "eval_interval": int(meta.get("eval_interval", 0)),
        "save_every_n_steps": int(meta.get("save_every_n_steps", 0)),
        "effective_train_sample_count_per_dataset": meta.get("effective_train_sample_count_per_dataset", {}),
        "effective_val_sample_count_per_dataset": meta.get("effective_val_sample_count_per_dataset", {}),
        "message": str(message),
    }


def _build_run_final_from_raw(raw: Dict[str, Any], meta: Dict[str, Any]) -> Dict[str, Any]:
    payload = dict(raw)
    payload["generated_at_utc"] = now_iso()
    payload["status"] = "completed"
    payload["selected_gpu_id"] = int(meta.get("selected_gpu_id", -1))
    payload["lease_id"] = str(meta.get("lease_id", ""))
    payload["raw_json_path"] = str(meta.get("raw_json", ""))
    payload["progress_json_path"] = str(meta.get("progress_json", ""))
    payload["trainer_mode"] = "full_scale" if bool(raw.get("whether_full_train_used", False)) or bool(raw.get("whether_full_val_used", False)) else "pilot_limited"
    return payload


def _run_final_md_lines(final_payload: Dict[str, Any]) -> List[str]:
    fm = final_payload.get("final_metrics", {}) if isinstance(final_payload.get("final_metrics", {}), dict) else {}
    return [
        f"# {final_payload.get('run_name', '')}",
        "",
        f"- status: {final_payload.get('status', '')}",
        f"- selected_gpu_id: {final_payload.get('selected_gpu_id', -1)}",
        f"- lease_id: {final_payload.get('lease_id', '')}",
        f"- current_mainline_semantic_source: {final_payload.get('current_mainline_semantic_source', '')}",
        f"- datasets_bound_for_train: {final_payload.get('datasets_bound_for_train', [])}",
        f"- whether_full_train_used: {bool(final_payload.get('whether_full_train_used', False))}",
        f"- whether_full_val_used: {bool(final_payload.get('whether_full_val_used', False))}",
        f"- effective_train_sample_count_per_dataset: {final_payload.get('effective_train_sample_count_per_dataset', {})}",
        f"- effective_val_sample_count_per_dataset: {final_payload.get('effective_val_sample_count_per_dataset', {})}",
        f"- free_rollout_endpoint_l2: {_f(fm.get('free_rollout_endpoint_l2'), 1e9):.6f}",
        f"- free_rollout_coord_mean_l2: {_f(fm.get('free_rollout_coord_mean_l2'), 1e9):.6f}",
        f"- teacher_forced_coord_loss: {_f(fm.get('teacher_forced_coord_loss'), 1e9):.6f}",
    ]


def _tmux_window_exists(session_name: str, window_name: str) -> bool:
    proc = subprocess.run(
        ["tmux", "list-windows", "-t", str(session_name), "-F", "#{window_name}"],
        text=True,
        capture_output=True,
    )
    if proc.returncode != 0:
        return False
    names = [x.strip() for x in proc.stdout.splitlines() if x.strip()]
    return str(window_name) in names


def _collect_run_status(meta: Dict[str, Any], session_name: str) -> Dict[str, Any]:
    progress_path = Path(str(meta.get("progress_json", "")))
    final_path = Path(str(meta.get("final_json", "")))
    raw_path = Path(str(meta.get("raw_json", "")))
    status = "launched"
    detail = {}
    final_detail: Dict[str, Any] = {}
    if final_path.exists():
        try:
            final_detail = _read_json(final_path)
            final_status = str(final_detail.get("status", status)).strip().lower()
            if final_status in {"completed", "failed"}:
                status = final_status
                return {
                    "status": str(status),
                    "detail": final_detail,
                }
        except Exception:
            status = "completed"
            return {
                "status": str(status),
                "detail": final_detail,
            }
    if progress_path.exists():
        try:
            detail = _read_json(progress_path)
            progress_status = str(detail.get("status", "")).strip().lower()
            if progress_status in {"completed", "failed"}:
                status = progress_status
            elif _tmux_window_exists(session_name, str(meta.get("window_name", ""))):
                status = "running"
            else:
                status = progress_status or "launched"
        except Exception:
            status = "running" if _tmux_window_exists(session_name, str(meta.get("window_name", ""))) else "launched"
    elif raw_path.exists():
        status = "completed"
    elif _tmux_window_exists(session_name, str(meta.get("window_name", ""))):
        status = "running"
    return {
        "status": str(status),
        "detail": detail,
    }


def _build_wave1_summary_payload(launch_payload: Dict[str, Any], session_name: str) -> Dict[str, Any]:
    runs = launch_payload.get("runs", []) if isinstance(launch_payload.get("runs", []), list) else []
    rows: List[Dict[str, Any]] = []
    running = 0
    completed = 0
    failed = 0
    for meta in runs:
        if not isinstance(meta, dict):
            continue
        status_info = _collect_run_status(meta, session_name=session_name)
        status = str(status_info.get("status", "launched"))
        if status == "running":
            running += 1
        elif status == "completed":
            completed += 1
        elif status == "failed":
            failed += 1
        final_detail = status_info.get("detail", {}) if isinstance(status_info.get("detail", {}), dict) else {}
        best_metric = final_detail.get("best_checkpoint_metric", {}) if isinstance(final_detail.get("best_checkpoint_metric", {}), dict) else {}
        rows.append(
            {
                "run_name": str(meta.get("run_name", "")),
                "selected_gpu_id": int(meta.get("selected_gpu_id", -1)),
                "lease_id": str(meta.get("lease_id", "")),
                "batch_size": int(meta.get("batch_size", 0)),
                "train_steps": int(meta.get("train_steps", 0)),
                "eval_interval": int(meta.get("eval_interval", 0)),
                "save_every_n_steps": int(meta.get("save_every_n_steps", 0)),
                "effective_train_sample_count_per_dataset": meta.get("effective_train_sample_count_per_dataset", {}),
                "effective_val_sample_count_per_dataset": meta.get("effective_val_sample_count_per_dataset", {}),
                "status": status,
                "raw_json": str(meta.get("raw_json", "")),
                "progress_json": str(meta.get("progress_json", "")),
                "final_json": str(meta.get("final_json", "")),
                "best_checkpoint_metric": best_metric,
            }
        )

    if failed > 0:
        next_step_choice = "fix_failed_runs_and_resume"
    elif completed == len(rows) and len(rows) > 0:
        next_step_choice = "summarize_wave1_after_completion"
    else:
        next_step_choice = "continue_stage2_fullscale_wave1"

    return {
        "generated_at_utc": now_iso(),
        "wave1_status": f"{running}_running_{completed}_completed_{failed}_failed",
        "runs": rows,
        "current_strongest_candidate_mainline": "core-only + crop_visual_encoder",
        "next_step_choice": next_step_choice,
    }


def _wave1_results_md_lines(summary_payload: Dict[str, Any]) -> List[str]:
    lines = [
        "# Stage2 Fullscale Wave1 Results",
        "",
        f"- wave1_status: {summary_payload.get('wave1_status', '')}",
        f"- current_strongest_candidate_mainline: {summary_payload.get('current_strongest_candidate_mainline', '')}",
        f"- next_step_choice: {summary_payload.get('next_step_choice', '')}",
        "",
        "| run_name | gpu | batch_size | train_steps | eval_interval | save_every_n_steps | status | best_endpoint_l2 |",
        "|---|---:|---:|---:|---:|---:|---|---:|",
    ]
    for row in summary_payload.get("runs", []):
        if not isinstance(row, dict):
            continue
        best_metric = row.get("best_checkpoint_metric", {}) if isinstance(row.get("best_checkpoint_metric", {}), dict) else {}
        metrics = best_metric.get("metrics", {}) if isinstance(best_metric.get("metrics", {}), dict) else {}
        lines.append(
            "| {run} | {gpu} | {bs} | {steps} | {ev} | {save} | {status} | {endpoint:.6f} |".format(
                run=str(row.get("run_name", "")),
                gpu=int(row.get("selected_gpu_id", -1)),
                bs=int(row.get("batch_size", 0)),
                steps=int(row.get("train_steps", 0)),
                ev=int(row.get("eval_interval", 0)),
                save=int(row.get("save_every_n_steps", 0)),
                status=str(row.get("status", "")),
                endpoint=_f(metrics.get("free_rollout_endpoint_l2"), 1e9),
            )
        )
    return lines


def _write_launch_placeholder(meta: Dict[str, Any]) -> None:
    progress_payload = _run_placeholder_status(meta, status="launched", message="wave1 run launched in tmux")
    final_payload = _run_placeholder_status(meta, status="launched", message="final result pending")
    _write_json(meta["progress_json"], progress_payload)
    _write_json(meta["final_json"], final_payload)


def _launch_window_for_run(args: Any, meta: Dict[str, Any]) -> None:
    _write_launch_placeholder(meta)
    env_exports = {
        "PYTHONPATH": f"{args.work_root}/code:{os.environ.get('PYTHONPATH', '')}",
        "CUDA_VISIBLE_DEVICES": str(meta["selected_gpu_id"]),
        "TRACEWM_STAGE1_V2_GPU_SELECTION_METADATA": str(meta["gpu_metadata_str"]),
        "TRACEWM_STAGE1_V2_GPU_SELECTION_METADATA_JSON": str(meta["gpu_metadata_json"]),
    }
    env_prefix = " ".join(f"{k}={shlex.quote(v)}" for k, v in env_exports.items())
    cmd = (
        f"{env_prefix} {shlex.quote(str(args.python_bin))} "
        f"{shlex.quote(str(Path(args.work_root) / 'code' / 'stwm' / 'tools' / 'run_tracewm_stage2_fullscale_wave1_20260409.py'))} "
        f"--mode run-one --meta-json {shlex.quote(str(meta['meta_json']))}"
    )
    subprocess.run(
        ["tmux", "new-window", "-t", str(args.tmux_session), "-n", str(meta["window_name"]), cmd],
        check=True,
        cwd=str(args.work_root),
    )


def _build_protocol_lines(anchor_args: Dict[str, Any], preflight_payload: Dict[str, Any]) -> List[str]:
    return [
        "# TRACEWM Stage2 Fullscale Wave1 Protocol",
        "",
        "- Stage1 remains frozen and untouched in this round.",
        "- Stage2 current mainline remains `stage2_core_cropenc` with `crop_visual_encoder` as the mainline semantic source.",
        "- Core train/eval binding remains `VSPW + VIPSeg`.",
        "- BURST is allowed only for the explicit `coreplusburst` control run.",
        "- TAO and VISOR remain excluded from this wave.",
        "",
        "## Preserved Mainline Hyperparameters",
        f"- obs_len: {int(anchor_args.get('obs_len', 8) or 8)}",
        f"- fut_len: {int(anchor_args.get('fut_len', 8) or 8)}",
        f"- max_tokens: {int(anchor_args.get('max_tokens', 64) or 64)}",
        f"- semantic_crop_size: {int(anchor_args.get('semantic_crop_size', 64) or 64)}",
        f"- semantic_hidden_dim: {int(anchor_args.get('semantic_hidden_dim', 256) or 256)}",
        f"- semantic_embed_dim: {int(anchor_args.get('semantic_embed_dim', 256) or 256)}",
        "",
        "## Runtime Preflight",
        f"- candidate_batch_sizes: {preflight_payload.get('candidate_batch_sizes', [])}",
        f"- selected_batch_size: {int(preflight_payload.get('selected_batch_size', -1))}",
        f"- whether_full_train_used: {bool(preflight_payload.get('whether_full_train_used', False))}",
        f"- whether_full_val_used: {bool(preflight_payload.get('whether_full_val_used', False))}",
        "",
        "## Wave1 Run Set",
        "- stage2_fullscale_core_cropenc_seed42_20260409",
        "- stage2_fullscale_core_cropenc_seed123_20260409",
        "- stage2_fullscale_core_cropenc_seed456_20260409",
        "- stage2_fullscale_core_legacysem_seed42_20260409",
        "- stage2_fullscale_coreplusburst_cropenc_seed42_20260409",
        "",
        "## Training Budget",
        "- train_steps: 10000",
        "- eval_interval: 1000",
        "- save_every_n_steps: 1000",
        "- eval_max_batches: -1 (full validation)",
        "",
        "## Scheduling",
        "- single-card per run",
        "- selector + shared lease path reused",
        "- 5 runs launched in parallel, leaving headroom on the 8xB200 node",
    ]


def orchestrate(args: Any) -> None:
    facts = _validate_frozen_facts(args)
    runtime_json = _read_json(args.stage1_runtime_json)
    anchor_args = _load_anchor_args(args.stage1_best_ckpt)
    preflight_payload = _run_preflight(args, anchor_args, runtime_json)

    batch_size = int(preflight_payload["selected_batch_size"])
    common = {
        "obs_len": int(anchor_args.get("obs_len", 8) or 8),
        "fut_len": int(anchor_args.get("fut_len", 8) or 8),
        "max_tokens": int(anchor_args.get("max_tokens", 64) or 64),
        "semantic_crop_size": int(anchor_args.get("semantic_crop_size", 64) or 64),
        "semantic_hidden_dim": int(anchor_args.get("semantic_hidden_dim", 256) or 256),
        "semantic_embed_dim": int(anchor_args.get("semantic_embed_dim", 256) or 256),
        "lr": float(anchor_args.get("lr", 1e-4) or 1e-4),
        "weight_decay": float(anchor_args.get("weight_decay", 0.0) or 0.0),
        "clip_grad_norm": float(anchor_args.get("clip_grad_norm", 1.0) or 1.0),
        "train_steps": 10000,
        "eval_interval": 1000,
        "eval_max_batches": -1,
        "save_every_n_steps": 1000,
        "batch_size": int(batch_size),
    }

    run_specs = [
        {
            "run_name": "stage2_fullscale_core_cropenc_seed42_20260409",
            "dataset_names": ["vspw", "vipseg"],
            "semantic_source_mainline": "crop_visual_encoder",
            "legacy_semantic_source": "hand_crafted_stats",
            "seed": 42,
            "window_name": "core_s42",
        },
        {
            "run_name": "stage2_fullscale_core_cropenc_seed123_20260409",
            "dataset_names": ["vspw", "vipseg"],
            "semantic_source_mainline": "crop_visual_encoder",
            "legacy_semantic_source": "hand_crafted_stats",
            "seed": 123,
            "window_name": "core_s123",
        },
        {
            "run_name": "stage2_fullscale_core_cropenc_seed456_20260409",
            "dataset_names": ["vspw", "vipseg"],
            "semantic_source_mainline": "crop_visual_encoder",
            "legacy_semantic_source": "hand_crafted_stats",
            "seed": 456,
            "window_name": "core_s456",
        },
        {
            "run_name": "stage2_fullscale_core_legacysem_seed42_20260409",
            "dataset_names": ["vspw", "vipseg"],
            "semantic_source_mainline": "hand_crafted_stats",
            "legacy_semantic_source": "hand_crafted_stats",
            "seed": 42,
            "window_name": "legacy_s42",
        },
        {
            "run_name": "stage2_fullscale_coreplusburst_cropenc_seed42_20260409",
            "dataset_names": ["vspw", "vipseg", "burst"],
            "semantic_source_mainline": "crop_visual_encoder",
            "legacy_semantic_source": "hand_crafted_stats",
            "seed": 42,
            "window_name": "burst_s42",
        },
    ]

    runs_meta: List[Dict[str, Any]] = []
    required_mem_gb = float(runtime_json.get("required_mem_gb", 40.0) or 40.0)
    safety_margin_gb = float(runtime_json.get("safety_margin_gb", 8.0) or 8.0)
    for spec in run_specs:
        counts_train = _estimate_effective_counts(
            dataset_names=spec["dataset_names"],
            split="train",
            stage2_contract_path=str(args.stage2_contract_json),
            obs_len=common["obs_len"],
            fut_len=common["fut_len"],
            max_tokens=common["max_tokens"],
            max_samples_per_dataset=-1,
            semantic_crop_size=common["semantic_crop_size"],
            semantic_source_mainline=spec["semantic_source_mainline"],
        )
        counts_val = _estimate_effective_counts(
            dataset_names=spec["dataset_names"],
            split="val",
            stage2_contract_path=str(args.stage2_contract_json),
            obs_len=common["obs_len"],
            fut_len=common["fut_len"],
            max_tokens=common["max_tokens"],
            max_samples_per_dataset=-1,
            semantic_crop_size=common["semantic_crop_size"],
            semantic_source_mainline=spec["semantic_source_mainline"],
        )
        gpu_pick = _select_gpu_with_lease(
            required_mem_gb=required_mem_gb,
            safety_margin_gb=safety_margin_gb,
            lease_path=str(args.shared_lease_path),
            owner=str(spec["run_name"]),
            ttl_seconds=18 * 3600,
            sample_count=4,
            interval_sec=0.5,
        )
        selected_gpu_id = int(gpu_pick["selected_gpu_id"])
        lease_id = str(gpu_pick["lease_id"])
        gpu_metadata = {
            "selected_gpu_id": int(selected_gpu_id),
            "lease_id": str(lease_id),
            "selected_reason": str((gpu_pick.get("selected_row", {}) if isinstance(gpu_pick.get("selected_row", {}), dict) else {}).get("selected_reason", "")),
            "owner": str(spec["run_name"]),
            "mode": "single_gpu_only",
        }
        run_root = Path(args.work_root) / "outputs" / "checkpoints" / spec["run_name"]
        meta = {
            **spec,
            **common,
            "selected_gpu_id": int(selected_gpu_id),
            "lease_id": str(lease_id),
            "gpu_metadata_json": json.dumps(gpu_metadata, ensure_ascii=True),
            "gpu_metadata_str": ";".join(f"{k}={v}" for k, v in gpu_metadata.items()),
            "effective_train_sample_count_per_dataset": counts_train,
            "effective_val_sample_count_per_dataset": counts_val,
            "stage1_best_ckpt": str(args.stage1_best_ckpt),
            "stage2_contract_json": str(args.stage2_contract_json),
            "stage1_runtime_json": str(args.stage1_runtime_json),
            "shared_lease_path": str(args.shared_lease_path),
            "work_root": str(args.work_root),
            "python_bin": str(args.python_bin),
            "output_dir": str(run_root),
            "raw_json": str(Path(args.work_root) / "reports" / f"{spec['run_name']}_raw.json"),
            "progress_json": str(Path(args.work_root) / "reports" / f"{spec['run_name']}_progress.json"),
            "final_json": str(Path(args.work_root) / "reports" / f"{spec['run_name']}_final.json"),
            "results_md": str(Path(args.work_root) / "docs" / f"{spec['run_name']}_results.md"),
            "log_path": str(Path(args.work_root) / "logs" / f"{spec['run_name']}.log"),
            "summary_report_path": str(args.summary_report),
            "launch_report_path": str(args.launch_report),
        }
        meta_json = Path(args.work_root) / "reports" / "stage2_fullscale_wave1_runs_20260409" / f"{spec['run_name']}_launch_meta.json"
        meta["meta_json"] = str(meta_json)
        _write_json(meta_json, meta)
        _launch_window_for_run(args, meta)
        runs_meta.append(meta)

    protocol_lines = _build_protocol_lines(anchor_args, preflight_payload)
    _write_md(args.protocol_md, protocol_lines)

    launch_payload = {
        "generated_at_utc": now_iso(),
        "mode": "stage2_fullscale_wave1_launch",
        "tmux_session": str(args.tmux_session),
        "runtime_preflight_report": str(args.runtime_preflight_report),
        "shared_lease_path": str(args.shared_lease_path),
        "runs": runs_meta,
        "current_strongest_candidate_mainline": "core-only + crop_visual_encoder",
    }
    _write_json(args.launch_report, launch_payload)

    summary_payload = _build_wave1_summary_payload(launch_payload, session_name=str(args.tmux_session))
    _write_json(args.summary_report, summary_payload)
    _write_md(args.results_md, _wave1_results_md_lines(summary_payload))

    print(json.dumps(
        {
            "runtime_preflight_report": str(args.runtime_preflight_report),
            "launch_report": str(args.launch_report),
            "summary_report": str(args.summary_report),
            "wave1_status": summary_payload.get("wave1_status", ""),
            "current_strongest_candidate_mainline": summary_payload.get("current_strongest_candidate_mainline", ""),
            "next_step_choice": summary_payload.get("next_step_choice", ""),
        },
        ensure_ascii=True,
        indent=2,
    ))


def run_one(args: Any) -> None:
    meta = _read_json(args.meta_json)
    raw_json = Path(str(meta["raw_json"]))
    progress_json = Path(str(meta["progress_json"]))
    final_json = Path(str(meta["final_json"]))
    results_md = Path(str(meta["results_md"]))
    log_path = Path(str(meta["log_path"]))
    lease_id = str(meta.get("lease_id", ""))
    lease_path = str(meta.get("shared_lease_path", ""))
    python_bin = str(meta.get("python_bin", args.python_bin))
    trainer_path = Path(meta["work_root"]) / "code" / "stwm" / "tracewm_v2_stage2" / "trainers" / "train_tracewm_stage2_smalltrain.py"

    progress_payload = _run_placeholder_status(meta, status="running", message="wave1 run entered trainer")
    _write_json(progress_json, progress_payload)
    _write_json(final_json, _run_placeholder_status(meta, status="launched", message="final result pending"))

    cmd = [
        python_bin,
        str(trainer_path),
        "--stage2-contract-path",
        str(meta["stage2_contract_json"]),
        "--recommended-runtime-json",
        str(meta["stage1_runtime_json"]),
        "--use-recommended-runtime",
        "--stage1-backbone-checkpoint",
        str(meta["stage1_best_ckpt"]),
        "--dataset-names",
        *[str(x) for x in meta.get("dataset_names", [])],
        "--train-split",
        "train",
        "--val-split",
        "val",
        "--obs-len",
        str(int(meta["obs_len"])),
        "--fut-len",
        str(int(meta["fut_len"])),
        "--max-tokens",
        str(int(meta["max_tokens"])),
        "--max-samples-train",
        "-1",
        "--max-samples-val",
        "-1",
        "--batch-size",
        str(int(meta["batch_size"])),
        "--lr",
        str(float(meta["lr"])),
        "--weight-decay",
        str(float(meta["weight_decay"])),
        "--clip-grad-norm",
        str(float(meta["clip_grad_norm"])),
        "--train-steps",
        str(int(meta["train_steps"])),
        "--eval-interval",
        str(int(meta["eval_interval"])),
        "--eval-max-batches",
        str(int(meta["eval_max_batches"])),
        "--save-every-n-steps",
        str(int(meta["save_every_n_steps"])),
        "--semantic-hidden-dim",
        str(int(meta["semantic_hidden_dim"])),
        "--semantic-embed-dim",
        str(int(meta["semantic_embed_dim"])),
        "--semantic-source-mainline",
        str(meta["semantic_source_mainline"]),
        "--legacy-semantic-source",
        str(meta["legacy_semantic_source"]),
        "--semantic-crop-size",
        str(int(meta["semantic_crop_size"])),
        "--output-dir",
        str(meta["output_dir"]),
        "--run-name",
        str(meta["run_name"]),
        "--run-summary-json",
        str(raw_json),
        "--progress-json",
        str(progress_json),
        "--seed",
        str(int(meta["seed"])),
    ]

    try:
        proc = subprocess.run(
            cmd,
            cwd=str(meta["work_root"]),
            text=True,
            capture_output=True,
            env=os.environ.copy(),
        )
        log_path.write_text(proc.stdout + ("\n" if proc.stdout else "") + proc.stderr, encoding="utf-8")
        if proc.returncode != 0:
            fail_payload = _run_placeholder_status(meta, status="failed", message=f"trainer returncode={proc.returncode}")
            fail_payload["stdout_tail"] = _tail(proc.stdout)
            fail_payload["stderr_tail"] = _tail(proc.stderr)
            _write_json(progress_json, fail_payload)
            _write_json(final_json, fail_payload)
            raise RuntimeError(f"trainer failed for {meta['run_name']} rc={proc.returncode}")

        raw_payload = _read_json(raw_json)
        final_payload = _build_run_final_from_raw(raw_payload, meta)
        _write_json(final_json, final_payload)
        _write_md(results_md, _run_final_md_lines(final_payload))
    except Exception as exc:
        fail_payload = _run_placeholder_status(meta, status="failed", message=str(exc))
        fail_payload["traceback"] = traceback.format_exc()
        _write_json(progress_json, fail_payload)
        _write_json(final_json, fail_payload)
        _write_md(results_md, [f"# {meta.get('run_name', '')}", "", f"- status: failed", f"- message: {exc}"])
        raise
    finally:
        _release_lease_safe(lease_id=lease_id, lease_path=lease_path)


def main() -> None:
    args = parse_args()
    if args.mode == "orchestrate":
        orchestrate(args)
        return
    if args.mode == "run-one":
        if not str(args.meta_json).strip():
            raise RuntimeError("--meta-json is required for --mode run-one")
        run_one(args)
        return
    raise RuntimeError(f"unsupported mode={args.mode}")


if __name__ == "__main__":
    main()
