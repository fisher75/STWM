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


WORK_ROOT = Path("/home/chen034/workspace/stwm")
DATE_TAG = "20260410"


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


def _python_bin_default() -> str:
    preferred = Path("/home/chen034/miniconda3/envs/stwm/bin/python")
    return str(preferred) if preferred.exists() else sys.executable


def parse_args() -> Any:
    p = ArgumentParser(description="Stage2 semantic rescue wave0 launcher")
    p.add_argument("--mode", default="launch", choices=["launch", "run-one", "summarize"])
    p.add_argument("--meta-json", default="")
    p.add_argument("--work-root", default=str(WORK_ROOT))
    p.add_argument("--python-bin", default=_python_bin_default())
    p.add_argument("--tmux-session", default="tracewm_stage2_ljs_aligned_semantic_diagnosis_and_rescue_20260410")
    p.add_argument("--stage2-contract-json", default=str(WORK_ROOT / "reports/stage2_bootstrap_data_contract_20260408.json"))
    p.add_argument("--stage1-runtime-json", default=str(WORK_ROOT / "reports/stage1_v2_recommended_runtime_20260408.json"))
    p.add_argument("--stage1-best-ckpt", default=str(WORK_ROOT / "outputs/checkpoints/stage1_v2_longtrain_220m_mainline_20260408/best.pt"))
    p.add_argument("--shared-lease-path", default=str(WORK_ROOT / "reports/stage1_v2_gpu_lease_20260408.json"))
    p.add_argument("--bootstrap-cache-jsonl", default=str(WORK_ROOT / "data/processed/stage2_semantic_bootstrap_cache_20260410/core_trainval_pseudo_targets.jsonl"))
    p.add_argument("--launch-report", default=str(WORK_ROOT / "reports/stage2_semantic_rescue_wave0_launch_20260410.json"))
    p.add_argument("--summary-report", default=str(WORK_ROOT / "reports/stage2_semantic_rescue_wave0_summary_20260410.json"))
    p.add_argument("--results-md", default=str(WORK_ROOT / "docs/STAGE2_SEMANTIC_RESCUE_WAVE0_RESULTS_20260410.md"))
    return p.parse_args()


def _dataset_counts(dataset_names: List[str], split: str, contract_path: str, obs_len: int, fut_len: int, max_tokens: int, crop_size: int) -> Dict[str, int]:
    ds = Stage2SemanticDataset(
        Stage2SemanticDatasetConfig(
            dataset_names=list(dataset_names),
            split=str(split),
            contract_path=str(contract_path),
            obs_len=int(obs_len),
            fut_len=int(fut_len),
            max_tokens=int(max_tokens),
            max_samples_per_dataset=-1,
            semantic_crop_size=int(crop_size),
            semantic_source_mainline="crop_visual_encoder",
        )
    )
    return {str(k): int(v.get("sample_count", 0) or 0) for k, v in ds.dataset_summary.items()}


def _load_anchor_args(best_ckpt: str | Path) -> Dict[str, Any]:
    ckpt = torch.load(Path(best_ckpt), map_location="cpu", weights_only=False)
    args = ckpt.get("args", {}) if isinstance(ckpt.get("args", {}), dict) else {}
    return args


def _select_gpu(run_name: str, lease_path: str) -> Dict[str, Any]:
    selector = select_single_gpu(
        required_mem_gb=40.0,
        safety_margin_gb=8.0,
        sample_count=3,
        interval_sec=0.5,
        lease_path=str(lease_path),
    )
    gpu_id = int(selector.get("selected_gpu_id", -1))
    if gpu_id < 0:
        raise RuntimeError("no GPU available for semantic rescue wave0")
    lease = acquire_lease(
        gpu_id=gpu_id,
        owner=str(run_name),
        ttl_seconds=8 * 3600,
        lease_path=str(lease_path),
    )
    return {"selected_gpu_id": gpu_id, "lease_id": str(lease.get("lease_id", "")), "selector_payload": selector, "lease": lease}


def _release_lease_safe(lease_id: str, lease_path: str) -> None:
    if not str(lease_id).strip():
        return
    try:
        release_lease(lease_id=str(lease_id), lease_path=str(lease_path))
    except Exception:
        pass


def _run_specs() -> List[Dict[str, Any]]:
    return [
        {
            "run_name": "stage2_semrescue_align_seed42_wave0_20260410",
            "semantic_rescue_mode": "align",
            "semantic_rescue_weight": 0.05,
            "window_name": "semres_align",
        },
        {
            "run_name": "stage2_semrescue_querypersist_seed42_wave0_20260410",
            "semantic_rescue_mode": "querypersist",
            "semantic_rescue_weight": 0.05,
            "window_name": "semres_qpersist",
        },
        {
            "run_name": "stage2_semrescue_bootstrapplabel_seed42_wave0_20260410",
            "semantic_rescue_mode": "bootstrapplabel",
            "semantic_rescue_weight": 0.05,
            "window_name": "semres_plabel",
        },
    ]


def _status_for(meta: Dict[str, Any], session_name: str) -> Dict[str, Any]:
    final_path = Path(str(meta.get("final_json", "")))
    progress_path = Path(str(meta.get("progress_json", "")))
    detail: Dict[str, Any] = {}
    proc = subprocess.run(
        ["tmux", "list-windows", "-t", str(session_name), "-F", "#{window_name}"],
        text=True,
        capture_output=True,
    )
    windows = proc.stdout.splitlines() if proc.returncode == 0 else []
    if str(meta.get("window_name", "")) in windows:
        if progress_path.exists():
            try:
                detail = _read_json(progress_path)
            except Exception:
                detail = {}
        return {"status": "running", "detail": detail}

    if final_path.exists():
        try:
            detail = _read_json(final_path)
            status = str(detail.get("status", "launched")).lower()
            if status in {"completed", "failed"}:
                return {"status": status, "detail": detail}
        except Exception:
            pass
    if progress_path.exists():
        try:
            detail = _read_json(progress_path)
        except Exception:
            detail = {}
    return {"status": str(detail.get("status", "launched")).lower() if detail else "launched", "detail": detail}


def summarize(args: Any) -> Dict[str, Any]:
    launch = _read_json(args.launch_report)
    rows: List[Dict[str, Any]] = []
    running = completed = failed = 0
    for meta in launch.get("runs", []) if isinstance(launch.get("runs", []), list) else []:
        if not isinstance(meta, dict):
            continue
        status_info = _status_for(meta, session_name=str(args.tmux_session))
        status = str(status_info.get("status", "launched"))
        if status == "running":
            running += 1
        elif status == "completed":
            completed += 1
        elif status == "failed":
            failed += 1
        detail = status_info.get("detail", {}) if isinstance(status_info.get("detail", {}), dict) else {}
        best = detail.get("best_checkpoint_metric", {}) if isinstance(detail.get("best_checkpoint_metric", {}), dict) else {}
        latest = detail.get("latest_checkpoint_metric", {}) if isinstance(detail.get("latest_checkpoint_metric", {}), dict) else {}
        rows.append(
            {
                "run_name": str(meta.get("run_name", "")),
                "semantic_rescue_mode": str(meta.get("semantic_rescue_mode", "")),
                "selected_gpu_id": int(meta.get("selected_gpu_id", -1)),
                "lease_id": str(meta.get("lease_id", "")),
                "batch_size": int(meta.get("batch_size", 0)),
                "train_steps": int(meta.get("train_steps", 0)),
                "eval_interval": int(meta.get("eval_interval", 0)),
                "save_every_n_steps": int(meta.get("save_every_n_steps", 0)),
                "effective_train_sample_count_per_dataset": meta.get("effective_train_sample_count_per_dataset", {}),
                "effective_val_sample_count_per_dataset": meta.get("effective_val_sample_count_per_dataset", {}),
                "status": status,
                "final_json": str(meta.get("final_json", "")),
                "best_checkpoint_metric": best,
                "latest_checkpoint_metric": latest,
            }
        )
    if failed:
        next_step = "redesign_stage2_semantic_objective"
    elif completed == len(rows) and rows:
        next_step = "summarize_semantic_rescue_wave0_after_completion"
    else:
        next_step = "continue_semantic_rescue_wave0"
    payload = {
        "generated_at_utc": now_iso(),
        "wave0_status": f"{running}_running_{completed}_completed_{failed}_failed",
        "runs": rows,
        "next_step_choice_internal": next_step,
    }
    _write_json(args.summary_report, payload)
    lines = [
        "# Stage2 Semantic Rescue Wave0 Results",
        "",
        f"- wave0_status: {payload['wave0_status']}",
        f"- next_step_choice_internal: {payload['next_step_choice_internal']}",
        "",
        "| run_name | mode | gpu | batch | steps | status | best_endpoint_l2 | latest_endpoint_l2 |",
        "|---|---|---:|---:|---:|---|---:|---:|",
    ]
    for row in rows:
        metrics = (row.get("best_checkpoint_metric", {}).get("metrics", {}) if isinstance(row.get("best_checkpoint_metric", {}), dict) else {})
        latest_metrics = (row.get("latest_checkpoint_metric", {}).get("metrics", {}) if isinstance(row.get("latest_checkpoint_metric", {}), dict) else {})
        lines.append(
            "| {run} | {mode} | {gpu} | {batch} | {steps} | {status} | {endpoint:.6f} | {latest_endpoint:.6f} |".format(
                run=row.get("run_name", ""),
                mode=row.get("semantic_rescue_mode", ""),
                gpu=row.get("selected_gpu_id", -1),
                batch=row.get("batch_size", 0),
                steps=row.get("train_steps", 0),
                status=row.get("status", ""),
                endpoint=float(metrics.get("free_rollout_endpoint_l2", 1e9)),
                latest_endpoint=float(latest_metrics.get("free_rollout_endpoint_l2", 1e9)),
            )
        )
    _write_md(args.results_md, lines)
    return payload


def launch(args: Any) -> Dict[str, Any]:
    anchor = _load_anchor_args(args.stage1_best_ckpt)
    resume_from = WORK_ROOT / "outputs/checkpoints/stage2_fullscale_core_cropenc_seed42_20260409/best.pt"
    resume_payload = torch.load(resume_from, map_location="cpu", weights_only=False)
    resume_step = int(resume_payload.get("global_step", 0) or 0)
    obs_len = int(anchor.get("obs_len", 8) or 8)
    fut_len = int(anchor.get("fut_len", 8) or 8)
    max_tokens = int(anchor.get("max_tokens", 64) or 64)
    crop_size = int(anchor.get("semantic_crop_size", 64) or 64)
    train_counts = _dataset_counts(["vspw", "vipseg"], "train", args.stage2_contract_json, obs_len, fut_len, max_tokens, crop_size)
    val_counts = _dataset_counts(["vspw", "vipseg"], "val", args.stage2_contract_json, obs_len, fut_len, max_tokens, crop_size)

    runs = []
    for spec in _run_specs():
        run_name = str(spec["run_name"])
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
            "batch_size": 8,
            "resume_from": str(resume_from),
            "resume_global_step": int(resume_step),
            "additional_train_steps": 200,
            "train_steps": int(resume_step + 200),
            "eval_interval": 50,
            "eval_max_batches": 32,
            "save_every_n_steps": 100,
            "seed": 42,
            "effective_train_sample_count_per_dataset": train_counts,
            "effective_val_sample_count_per_dataset": val_counts,
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
        meta_json = Path(args.work_root) / "reports" / "stage2_semantic_rescue_wave0_runs_20260410" / f"{run_name}_launch_meta.json"
        meta["meta_json"] = str(meta_json)
        _write_json(meta_json, meta)
        runs.append(meta)

        env = {
            "PYTHONPATH": f"{args.work_root}/code:{os.environ.get('PYTHONPATH', '')}",
            "CUDA_VISIBLE_DEVICES": str(meta["selected_gpu_id"]),
            "TRACEWM_STAGE1_V2_GPU_SELECTION_METADATA_JSON": json.dumps(
                {
                    "selected_gpu_id": int(meta["selected_gpu_id"]),
                    "lease_id": str(meta["lease_id"]),
                    "owner": run_name,
                    "mode": "single_gpu_only",
                },
                ensure_ascii=True,
            ),
        }
        env_prefix = " ".join(f"{k}={shlex.quote(v)}" for k, v in env.items())
        cmd = (
            f"{env_prefix} {shlex.quote(str(args.python_bin))} "
            f"{shlex.quote(str(Path(args.work_root) / 'code/stwm/tools/run_tracewm_stage2_semantic_rescue_wave0_20260410.py'))} "
            f"--mode run-one --meta-json {shlex.quote(str(meta_json))}"
        )
        subprocess.run(["tmux", "new-window", "-t", str(args.tmux_session), "-n", str(meta["window_name"]), cmd], check=True)

    launch_payload = {
        "generated_at_utc": now_iso(),
        "mode": "stage2_semantic_rescue_wave0_launch",
        "tmux_session": str(args.tmux_session),
        "runs": runs,
    }
    _write_json(args.launch_report, launch_payload)
    return summarize(args)


def run_one(args: Any) -> None:
    meta = _read_json(args.meta_json)
    lease_id = str(meta.get("lease_id", ""))
    lease_path = str(meta.get("shared_lease_path", ""))
    trainer = Path(str(meta["work_root"])) / "code/stwm/tracewm_v2_stage2/trainers/train_tracewm_stage2_smalltrain.py"
    cmd = [
        str(meta["python_bin"]),
        str(trainer),
        "--stage2-contract-path",
        str(meta["stage2_contract_json"]),
        "--recommended-runtime-json",
        str(meta["stage1_runtime_json"]),
        "--use-recommended-runtime",
        "--stage1-backbone-checkpoint",
        str(meta["stage1_best_ckpt"]),
        "--dataset-names",
        "vspw",
        "vipseg",
        "--train-split",
        "train",
        "--val-split",
        "val",
        "--obs-len",
        str(meta["obs_len"]),
        "--fut-len",
        str(meta["fut_len"]),
        "--max-tokens",
        str(meta["max_tokens"]),
        "--max-samples-train",
        "-1",
        "--max-samples-val",
        "-1",
        "--batch-size",
        str(meta["batch_size"]),
        "--train-steps",
        str(meta["train_steps"]),
        "--eval-interval",
        str(meta["eval_interval"]),
        "--eval-max-batches",
        str(meta["eval_max_batches"]),
        "--save-every-n-steps",
        str(meta["save_every_n_steps"]),
        "--semantic-source-mainline",
        str(meta["semantic_source_mainline"]),
        "--legacy-semantic-source",
        str(meta["legacy_semantic_source"]),
        "--semantic-crop-size",
        str(meta["semantic_crop_size"]),
        "--semantic-rescue-mode",
        str(meta["semantic_rescue_mode"]),
        "--semantic-rescue-weight",
        str(meta["semantic_rescue_weight"]),
        "--semantic-bootstrap-cache-path",
        str(meta["bootstrap_cache_jsonl"]),
        "--resume-from",
        str(meta["resume_from"]),
        "--skip-resume-optimizer",
        "--output-dir",
        str(meta["output_dir"]),
        "--run-name",
        str(meta["run_name"]),
        "--run-summary-json",
        str(meta["raw_json"]),
        "--progress-json",
        str(meta["progress_json"]),
        "--seed",
        str(meta["seed"]),
    ]
    try:
        proc = subprocess.run(cmd, cwd=str(meta["work_root"]), text=True, capture_output=True, env=os.environ.copy())
        Path(str(meta["log_path"])).write_text(proc.stdout + ("\n" if proc.stdout else "") + proc.stderr, encoding="utf-8")
        if proc.returncode != 0:
            _write_json(
                meta["final_json"],
                {
                    "generated_at_utc": now_iso(),
                    "run_name": str(meta["run_name"]),
                    "status": "failed",
                    "returncode": int(proc.returncode),
                    "stderr_tail": proc.stderr[-4000:],
                },
            )
            raise RuntimeError(f"trainer failed rc={proc.returncode}")
        raw = _read_json(meta["raw_json"])
        raw["generated_at_utc"] = now_iso()
        raw["status"] = "completed"
        raw["selected_gpu_id"] = int(meta["selected_gpu_id"])
        raw["lease_id"] = str(meta["lease_id"])
        _write_json(meta["final_json"], raw)
    except Exception as exc:
        _write_json(
            meta["final_json"],
            {
                "generated_at_utc": now_iso(),
                "run_name": str(meta.get("run_name", "")),
                "status": "failed",
                "message": str(exc),
                "traceback": traceback.format_exc(),
            },
        )
        raise
    finally:
        _release_lease_safe(lease_id=lease_id, lease_path=lease_path)


def main() -> None:
    args = parse_args()
    if args.mode == "launch":
        print(json.dumps(launch(args), ensure_ascii=True, indent=2))
        return
    if args.mode == "summarize":
        print(json.dumps(summarize(args), ensure_ascii=True, indent=2))
        return
    if args.mode == "run-one":
        run_one(args)
        return
    raise RuntimeError(f"unsupported mode={args.mode}")


if __name__ == "__main__":
    main()
