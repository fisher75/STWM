#!/usr/bin/env python3
from __future__ import annotations

from argparse import ArgumentParser, Namespace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple
import gc
import hashlib
import json
import os
import shlex
import subprocess
import time

import numpy as np
import torch

from stwm.infra.gpu_lease import acquire_lease, list_active_leases
from stwm.tools import run_stage2_state_identifiability_eval_20260415 as prev_eval
from stwm.tools import run_stage2_state_identifiability_eval_v3_20260416 as evalv3
from stwm.tools import run_stage2_tusb_v2_context_aligned_20260418 as ctxeval
from stwm.tools import run_tracewm_stage2_calibration_only_fullscale_wave1_20260413 as base


ROOT = Path("/raid/chen034/workspace/stwm")
DATE_TAG = "20260420"
SESSION = "stwm_matched6seed_real_completion_20260420"
LOG_PATH = ROOT / "logs" / "stwm_matched6seed_real_completion_20260420.log"
DEFAULT_PYTHON = Path("/home/chen034/miniconda3/envs/stwm/bin/python")
MATCHED_SEEDS = [42, 123, 456, 654, 789, 321]
MAX_CONCURRENT_TRAIN = 4
POLL_SECONDS = 60
TRAIN_TIMEOUT_SECONDS = 72 * 3600
RESERVED_GPU_IDS = {2}
MAX_STWM_TASKS_PER_GPU = 2
BASELINE_GPU_POOL = [0, 1]
TUSB_GPU_POOL = [0, 1, 4, 5, 7]

PLAN_REPORT = ROOT / "reports" / f"stwm_matched6seed_real_completion_plan_{DATE_TAG}.json"
PLAN_DOC = ROOT / "docs" / f"STWM_MATCHED6SEED_REAL_COMPLETION_PLAN_{DATE_TAG}.md"
LAUNCH_REPORT = ROOT / "reports" / f"stwm_matched6seed_real_completion_launch_{DATE_TAG}.json"
SUMMARY_REPORT = ROOT / "reports" / f"stwm_matched6seed_real_completion_summary_{DATE_TAG}.json"
SUMMARY_DOC = ROOT / "docs" / f"STWM_MATCHED6SEED_REAL_COMPLETION_{DATE_TAG}.md"
MAIN_EVAL_REPORT = ROOT / "reports" / f"stwm_matched6seed_main_eval_{DATE_TAG}.json"
MAIN_EVAL_DOC = ROOT / "docs" / f"STWM_MATCHED6SEED_MAIN_EVAL_{DATE_TAG}.md"
BOOTSTRAP_REPORT = ROOT / "reports" / f"stwm_matched6seed_strict_bootstrap_{DATE_TAG}.json"
BOOTSTRAP_DOC = ROOT / "docs" / f"STWM_MATCHED6SEED_STRICT_BOOTSTRAP_{DATE_TAG}.md"

LEGACY_DUALPANEL_AUDIT = ROOT / "reports" / "stage2_v3p1_dualpanel_context_audit_20260420.json"
PROTOCOL_V3_JSON = ROOT / "reports" / "stage2_state_identifiability_protocol_v3_20260416.json"
SHARED_LEASE_PATH = ROOT / "reports" / "stage1_v2_gpu_lease_20260408.json"
STAGE1_BEST = ROOT / "outputs/checkpoints/stage1_v2_longtrain_220m_mainline_20260408/best.pt"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _append_log(message: str) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as fh:
        fh.write(f"[{now_iso()}] {message}\n")


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def _write_md(path: Path, lines: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _apply_process_title_normalization(default_title: str = "python") -> None:
    mode = str(os.environ.get("STWM_PROC_TITLE_MODE", "generic")).strip().lower()
    if mode != "generic":
        return
    title = str(os.environ.get("STWM_PROC_TITLE", default_title)).strip() or default_title
    lowered = title.lower()
    if "stwm" in lowered or "tracewm" in lowered or "/home/" in lowered or "/raid/" in lowered:
        title = default_title
    try:
        import setproctitle  # type: ignore

        setproctitle.setproctitle(title)
    except Exception:
        pass


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _method_seed_dirs() -> Dict[str, Dict[int, str]]:
    return {
        "TUSB-v3.1": {
            42: "stage2_tusb_v3p1_seed42_20260418",
            123: "stage2_tusb_v3p1_seed123_20260418",
            456: "stage2_tusb_v3p1_seed456_20260418",
            654: "stage2_tusb_v3p1_seed654_matched6_20260420",
            789: "stage2_tusb_v3p1_seed789_matched6_20260420",
            321: "stage2_tusb_v3p1_seed321_matched6_20260420",
        },
        "calibration-only": {
            42: "stage2_calonly_topk1_seed42_wave1_20260413",
            123: "stage2_calonly_topk1_seed123_longconfirm_v2_20260414",
            456: "stage2_calonly_topk1_seed456_wave1_20260413",
            654: "stage2_calonly_topk1_seed654_longconfirm_20260414",
            789: "stage2_calonly_topk1_seed789_wave2_20260414",
            321: "stage2_calonly_topk1_seed321_longconfirm_v2_20260414",
        },
        "cropenc baseline": {
            42: "stage2_fullscale_core_cropenc_seed42_20260409",
            123: "stage2_fullscale_core_cropenc_seed123_20260409",
            456: "stage2_fullscale_core_cropenc_seed456_20260409",
            654: "stage2_fullscale_core_cropenc_seed654_matched6_20260420",
            789: "stage2_fullscale_core_cropenc_seed789_wave2_20260409",
            321: "stage2_fullscale_core_cropenc_seed321_matched6_20260420",
        },
        "legacysem baseline": {
            42: "stage2_fullscale_core_legacysem_seed42_20260409",
            123: "stage2_fullscale_core_legacysem_seed123_wave2_20260409",
            456: "stage2_fullscale_core_legacysem_seed456_wave2_20260409",
            654: "stage2_fullscale_core_legacysem_seed654_matched6_20260420",
            789: "stage2_fullscale_core_legacysem_seed789_matched6_20260420",
            321: "stage2_fullscale_core_legacysem_seed321_matched6_20260420",
        },
    }


def _calibration_resume_map() -> Dict[int, str]:
    return {
        42: "stage2_calonly_topk1_seed42_wave1_20260413",
        123: "stage2_calonly_topk1_seed123_longconfirm_v2_20260414",
        456: "stage2_calonly_topk1_seed456_wave1_20260413",
        654: "stage2_calonly_topk1_seed654_longconfirm_20260414",
        789: "stage2_calonly_topk1_seed789_wave2_20260414",
        321: "stage2_calonly_topk1_seed321_longconfirm_v2_20260414",
    }


def _coverage_row(method: str, seed: int, run_name: str) -> Dict[str, Any]:
    ckpt_dir = ROOT / "outputs" / "checkpoints" / run_name
    best = ckpt_dir / "best.pt"
    sidecar = ckpt_dir / "best_semantic_hard.pt"
    exists = bool(best.exists())
    return {
        "method": method,
        "seed": int(seed),
        "run_name": run_name,
        "checkpoint_exists": exists,
        "best_pt_exists": exists,
        "best_semantic_hard_exists": bool(sidecar.exists()),
        "train_needed": not exists,
        "exact_checkpoint_path_or_missing_reason": str(best) if exists else f"checkpoint_missing:{best}",
    }


def _build_plan_payload() -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    coverage_summary: Dict[str, Any] = {}
    for method, mapping in _method_seed_dirs().items():
        per_method = []
        for seed in MATCHED_SEEDS:
            row = _coverage_row(method, seed, mapping[seed])
            rows.append(row)
            per_method.append(row)
        coverage_summary[method] = {
            "seed_count_present": int(sum(int(bool(r["best_pt_exists"])) for r in per_method)),
            "fully_covered": bool(all(bool(r["best_pt_exists"]) for r in per_method)),
        }
    payload = {
        "generated_at_utc": now_iso(),
        "matched_seeds": MATCHED_SEEDS,
        "rows": rows,
        "coverage_summary": coverage_summary,
    }
    return payload


def _write_plan_artifacts() -> Dict[str, Any]:
    payload = _build_plan_payload()
    _write_json(PLAN_REPORT, payload)
    lines = [
        "# STWM Matched-6seed Real Completion Plan 20260420",
        "",
        "- Scope: TUSB-v3.1 / calibration-only / cropenc / legacysem only.",
        "- This round only audits coverage, launches missing real training, runs matched 6-seed main eval, and runs one strict bootstrap.",
        "",
        "## Coverage Summary",
    ]
    for method, summary in payload["coverage_summary"].items():
        lines.append(
            f"- {method}: seed_count_present={summary['seed_count_present']}, fully_covered={summary['fully_covered']}"
        )
    _write_md(PLAN_DOC, lines)
    return payload


def _gpu_rows() -> List[Dict[str, Any]]:
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,memory.total,memory.used,memory.free,utilization.gpu,utilization.memory",
        "--format=csv,noheader,nounits",
    ]
    proc = subprocess.run(cmd, text=True, capture_output=True, check=True)
    rows: List[Dict[str, Any]] = []
    for line in proc.stdout.strip().splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) != 6:
            continue
        idx, total, used, free, gpu_util, mem_util = parts
        rows.append(
            {
                "gpu_id": int(idx),
                "total_mem_gb": float(total) / 1024.0,
                "used_mem_gb": float(used) / 1024.0,
                "free_mem_gb": float(free) / 1024.0,
                "gpu_util": float(gpu_util),
                "mem_util": float(mem_util),
            }
        )
    return rows


def _probe_gpu_from_pool(
    pool: List[int],
    required_mem_gb: float,
    safety_margin_gb: float,
) -> Dict[str, Any] | None:
    lease_path = str(SHARED_LEASE_PATH)
    candidates: List[Dict[str, Any]] = []
    leases = list_active_leases(lease_path=lease_path)
    for row in _gpu_rows():
        gpu_id = int(row["gpu_id"])
        if gpu_id in RESERVED_GPU_IDS:
            continue
        if gpu_id not in pool:
            continue
        lease_count = int(sum(int(int(lease.get("gpu_id", -1)) == gpu_id) for lease in leases))
        if lease_count >= MAX_STWM_TASKS_PER_GPU:
            continue
        if float(row["free_mem_gb"]) < float(required_mem_gb + safety_margin_gb):
            continue
        row = dict(row)
        row["lease_count"] = lease_count
        candidates.append(row)
    if not candidates:
        return None
    candidates.sort(key=lambda r: (-float(r["free_mem_gb"]), float(r["gpu_util"]), float(r["mem_util"]), int(r["gpu_id"])))
    return candidates[0]


def _select_gpu_from_pool(
    pool: List[int],
    required_mem_gb: float,
    safety_margin_gb: float,
    owner: str,
    wait_timeout_seconds: int = TRAIN_TIMEOUT_SECONDS,
) -> Dict[str, Any] | None:
    deadline = time.time() + max(0, int(wait_timeout_seconds))
    lease_path = str(SHARED_LEASE_PATH)
    while True:
        selected = _probe_gpu_from_pool(
            pool=pool,
            required_mem_gb=required_mem_gb,
            safety_margin_gb=safety_margin_gb,
        )
        if selected is not None:
            lease = acquire_lease(
                gpu_id=int(selected["gpu_id"]),
                owner=str(owner),
                ttl_seconds=24 * 3600,
                lease_path=lease_path,
                allow_shared=bool(int(selected.get("lease_count", 0)) > 0),
            )
            return {
                "selected_gpu_id": int(selected["gpu_id"]),
                "lease_id": str(lease["lease_id"]),
                "selected_reason": "custom_pool_highest_free_mem",
                "telemetry": selected,
            }
        if int(wait_timeout_seconds) <= 0 or time.time() >= deadline:
            return None
        time.sleep(30)


def _ensure_tmux_session(session_name: str) -> None:
    if subprocess.run(["tmux", "has-session", "-t", session_name], capture_output=True).returncode != 0:
        subprocess.run(["tmux", "new-session", "-d", "-s", session_name, "bash"], check=True, cwd=str(ROOT))


def _tmux_windows(session_name: str) -> List[str]:
    proc = subprocess.run(["tmux", "list-windows", "-t", session_name, "-F", "#{window_name}"], text=True, capture_output=True)
    if proc.returncode != 0:
        return []
    return [line.strip() for line in proc.stdout.splitlines() if line.strip()]


def _launch_tmux_task(session_name: str, window_name: str, command: str) -> None:
    existing = set(_tmux_windows(session_name))
    if window_name in existing:
        subprocess.run(["tmux", "kill-window", "-t", f"{session_name}:{window_name}"], check=False, cwd=str(ROOT))
    subprocess.run(["tmux", "new-window", "-t", session_name, "-n", window_name, command], check=True, cwd=str(ROOT))


def _paths_for_run(run_name: str) -> Dict[str, Path]:
    ckpt_dir = ROOT / "outputs" / "checkpoints" / run_name
    return {
        "output_dir": ckpt_dir,
        "best": ckpt_dir / "best.pt",
        "sidecar": ckpt_dir / "best_semantic_hard.pt",
        "raw": ROOT / "reports" / f"{run_name}_raw.json",
        "progress": ROOT / "reports" / f"{run_name}_progress.json",
        "final": ROOT / "reports" / f"{run_name}_final.json",
        "log": ROOT / "logs" / f"{run_name}.log",
        "results_md": ROOT / "docs" / f"{run_name}_results.md",
        "pid": ROOT / "reports" / "stwm_matched6seed_real_completion_runs_20260420" / f"{run_name}.pid",
    }


def _load_template(path: Path) -> Dict[str, Any]:
    payload = _read_json(path)
    if not payload:
        raise FileNotFoundError(path)
    return payload


def _baseline_template(method: str) -> Tuple[Dict[str, Any], Path]:
    if method == "cropenc baseline":
        return (
            _load_template(ROOT / "reports/stage2_fullscale_wave2_runs_20260409/stage2_fullscale_core_cropenc_seed789_wave2_20260409_launch_meta.json"),
            ROOT / "code/stwm/tools/run_tracewm_stage2_fullscale_wave2_20260409.py",
        )
    if method == "legacysem baseline":
        return (
            _load_template(ROOT / "reports/stage2_fullscale_wave2_runs_20260409/stage2_fullscale_core_legacysem_seed123_wave2_20260409_launch_meta.json"),
            ROOT / "code/stwm/tools/run_tracewm_stage2_fullscale_wave2_20260409.py",
        )
    raise KeyError(method)


def _tusb_v2_template() -> Tuple[Dict[str, Any], Path]:
    return (
        _load_template(ROOT / "reports/stage2_tusb_v2_runs_20260418/stage2_tusb_v2_seed123_20260418_launch_meta.json"),
        ROOT / "code/stwm/tools/run_stage2_tusb_v2_20260418.py",
    )


def _tusb_v3_template() -> Tuple[Dict[str, Any], Path]:
    return (
        _load_template(ROOT / "reports/stage2_tusb_v3_identity_binding_runs_20260418/stage2_tusb_v3_seed123_20260418_launch_meta.json"),
        ROOT / "code/stwm/tools/run_stage2_tusb_v3_identity_binding_20260418.py",
    )


def _tusb_v3p1_template() -> Tuple[Dict[str, Any], Path]:
    return (
        _load_template(ROOT / "reports/stage2_tusb_v3p1_hardsubset_conversion_runs_20260418/stage2_tusb_v3p1_seed123_20260418_launch_meta.json"),
        ROOT / "code/stwm/tools/run_stage2_tusb_v3p1_hardsubset_conversion_20260418.py",
    )


def _load_ckpt_step(path: Path) -> int:
    return base._load_ckpt_step(path)


def _calibration_resume_ckpt(seed: int) -> Path:
    run_name = _calibration_resume_map()[int(seed)]
    ckpt = ROOT / "outputs" / "checkpoints" / run_name / "best.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"missing calibration checkpoint for seed {seed}: {ckpt}")
    return ckpt


def _build_gpu_metadata(selected_gpu_id: int, lease_id: str, owner: str, selected_reason: str) -> Tuple[str, str]:
    payload = {
        "selected_gpu_id": int(selected_gpu_id),
        "lease_id": str(lease_id),
        "selected_reason": str(selected_reason),
        "owner": str(owner),
        "mode": "single_gpu_only",
    }
    return json.dumps(payload, ensure_ascii=True), ";".join(f"{k}={v}" for k, v in payload.items())


def _task_spec_list() -> List[Dict[str, Any]]:
    specs: List[Dict[str, Any]] = []
    for seed in [654, 321]:
        specs.append(
            {
                "task_name": f"cropenc_seed{seed}",
                "method": "cropenc baseline",
                "seed": int(seed),
                "run_name": _method_seed_dirs()["cropenc baseline"][seed],
                "window_name": f"crop{seed}",
                "task_type": "baseline",
                "gpu_pool": BASELINE_GPU_POOL,
                "required_mem_gb": 40.0,
                "safety_margin_gb": 8.0,
                "deps": [],
            }
        )
    for seed in [654, 789, 321]:
        specs.extend(
            [
                {
                    "task_name": f"tusb_v2_seed{seed}",
                    "method": "TUSB-v3.1",
                    "seed": int(seed),
                    "run_name": f"stage2_tusb_v2_seed{seed}_matched6_20260420",
                    "window_name": f"tv2{seed}",
                    "task_type": "tusb_v2",
                    "gpu_pool": TUSB_GPU_POOL,
                    "required_mem_gb": 24.0,
                    "safety_margin_gb": 4.0,
                    "deps": [],
                },
                {
                    "task_name": f"tusb_v3_seed{seed}",
                    "method": "TUSB-v3.1",
                    "seed": int(seed),
                    "run_name": f"stage2_tusb_v3_seed{seed}_matched6_20260420",
                    "window_name": f"tv3{seed}",
                    "task_type": "tusb_v3",
                    "gpu_pool": TUSB_GPU_POOL,
                    "required_mem_gb": 24.0,
                    "safety_margin_gb": 4.0,
                    "deps": [f"tusb_v2_seed{seed}"],
                },
                {
                    "task_name": f"tusb_v3p1_seed{seed}",
                    "method": "TUSB-v3.1",
                    "seed": int(seed),
                    "run_name": _method_seed_dirs()["TUSB-v3.1"][seed],
                    "window_name": f"tv31{seed}",
                    "task_type": "tusb_v3p1",
                    "gpu_pool": TUSB_GPU_POOL,
                    "required_mem_gb": 24.0,
                    "safety_margin_gb": 4.0,
                    "deps": [f"tusb_v3_seed{seed}"],
                },
            ]
        )
    for seed in [654, 789, 321]:
        specs.append(
            {
                "task_name": f"legacysem_seed{seed}",
                "method": "legacysem baseline",
                "seed": int(seed),
                "run_name": _method_seed_dirs()["legacysem baseline"][seed],
                "window_name": f"leg{seed}",
                "task_type": "baseline",
                "gpu_pool": BASELINE_GPU_POOL,
                "required_mem_gb": 40.0,
                "safety_margin_gb": 8.0,
                "deps": [],
            }
        )
    return specs


def _baseline_meta(spec: Dict[str, Any], selected_gpu_id: int, lease_id: str, selected_reason: str, python_bin: str) -> Tuple[Dict[str, Any], Path]:
    template, script_path = _baseline_template(str(spec["method"]))
    run_name = str(spec["run_name"])
    seed = int(spec["seed"])
    paths = _paths_for_run(run_name)
    gpu_meta_json, gpu_meta_str = _build_gpu_metadata(selected_gpu_id, lease_id, run_name, selected_reason)
    meta = dict(template)
    meta.update(
        {
            "run_name": run_name,
            "seed": int(seed),
            "window_name": str(spec["window_name"]),
            "selected_gpu_id": int(selected_gpu_id),
            "lease_id": str(lease_id),
            "gpu_metadata_json": gpu_meta_json,
            "gpu_metadata_str": gpu_meta_str,
            "python_bin": str(python_bin),
            "work_root": str(ROOT),
            "shared_lease_path": str(SHARED_LEASE_PATH),
            "output_dir": str(paths["output_dir"]),
            "raw_json": str(paths["raw"]),
            "progress_json": str(paths["progress"]),
            "final_json": str(paths["final"]),
            "results_md": str(paths["results_md"]),
            "log_path": str(paths["log"]),
            "summary_report_path": str(SUMMARY_REPORT),
            "launch_report_path": str(LAUNCH_REPORT),
        }
    )
    meta_json = ROOT / "reports" / "stwm_matched6seed_real_completion_runs_20260420" / f"{run_name}_launch_meta.json"
    meta["meta_json"] = str(meta_json)
    return meta, script_path


def _tusb_meta_from_template(
    template: Dict[str, Any],
    script_path: Path,
    spec: Dict[str, Any],
    selected_gpu_id: int,
    lease_id: str,
    selected_reason: str,
    python_bin: str,
    resume_from: Path,
    resume_global_step: int,
    train_steps: int,
    output_dir: Path,
    extra: Dict[str, Any] | None = None,
) -> Tuple[Dict[str, Any], Path]:
    run_name = str(spec["run_name"])
    paths = _paths_for_run(run_name)
    gpu_meta_json, gpu_meta_str = _build_gpu_metadata(selected_gpu_id, lease_id, run_name, selected_reason)
    meta = dict(template)
    meta.update(
        {
            "run_name": run_name,
            "seed": int(spec["seed"]),
            "window_name": str(spec["window_name"]),
            "selected_gpu_id": int(selected_gpu_id),
            "lease_id": str(lease_id),
            "gpu_metadata_json": gpu_meta_json,
            "gpu_metadata_str": gpu_meta_str,
            "python_bin": str(python_bin),
            "work_root": str(ROOT),
            "shared_lease_path": str(SHARED_LEASE_PATH),
            "resume_from": str(resume_from),
            "resume_global_step": int(resume_global_step),
            "train_steps": int(train_steps),
            "output_dir": str(output_dir),
            "raw_json": str(paths["raw"]),
            "progress_json": str(paths["progress"]),
            "final_json": str(paths["final"]),
            "log_path": str(paths["log"]),
            "worker_pid_file": str(paths["pid"]),
        }
    )
    if extra:
        meta.update(extra)
    meta_json = ROOT / "reports" / "stwm_matched6seed_real_completion_runs_20260420" / f"{run_name}_launch_meta.json"
    meta["meta_json"] = str(meta_json)
    return meta, script_path


def _build_task_launch(spec: Dict[str, Any], python_bin: str, selected: Dict[str, Any] | None = None) -> Tuple[Dict[str, Any], Path]:
    selected = selected or _select_gpu_from_pool(
        pool=[int(x) for x in spec["gpu_pool"]],
        required_mem_gb=float(spec["required_mem_gb"]),
        safety_margin_gb=float(spec["safety_margin_gb"]),
        owner=str(spec["run_name"]),
    )
    if selected is None:
        raise RuntimeError(f"gpu_not_available_now task={spec['task_name']}")
    gpu_id = int(selected["selected_gpu_id"])
    lease_id = str(selected["lease_id"])
    selected_reason = str(selected["selected_reason"])
    if spec["task_type"] == "baseline":
        return _baseline_meta(spec, gpu_id, lease_id, selected_reason, python_bin)
    if spec["task_type"] == "tusb_v2":
        template, script_path = _tusb_v2_template()
        resume_from = _calibration_resume_ckpt(int(spec["seed"]))
        resume_step = _load_ckpt_step(resume_from)
        return _tusb_meta_from_template(
            template,
            script_path,
            spec,
            gpu_id,
            lease_id,
            selected_reason,
            python_bin,
            resume_from,
            resume_step,
            int(resume_step + 1000),
            ROOT / "outputs" / "checkpoints" / str(spec["run_name"]),
        )
    if spec["task_type"] == "tusb_v3":
        template, script_path = _tusb_v3_template()
        resume_from = ROOT / "outputs" / "checkpoints" / f"stage2_tusb_v2_seed{int(spec['seed'])}_matched6_20260420" / "best.pt"
        if not resume_from.exists():
            raise FileNotFoundError(f"resume checkpoint missing for {spec['task_name']}: {resume_from}")
        resume_step = _load_ckpt_step(resume_from)
        return _tusb_meta_from_template(
            template,
            script_path,
            spec,
            gpu_id,
            lease_id,
            selected_reason,
            python_bin,
            resume_from,
            resume_step,
            int(resume_step + 800),
            ROOT / "outputs" / "checkpoints" / str(spec["run_name"]),
        )
    if spec["task_type"] == "tusb_v3p1":
        template, script_path = _tusb_v3p1_template()
        resume_from = ROOT / "outputs" / "checkpoints" / f"stage2_tusb_v3_seed{int(spec['seed'])}_matched6_20260420" / "best_semantic_hard.pt"
        if not resume_from.exists():
            raise FileNotFoundError(f"resume sidecar missing for {spec['task_name']}: {resume_from}")
        resume_step = _load_ckpt_step(resume_from)
        return _tusb_meta_from_template(
            template,
            script_path,
            spec,
            gpu_id,
            lease_id,
            selected_reason,
            python_bin,
            resume_from,
            resume_step,
            int(resume_step + 200),
            ROOT / "outputs" / "checkpoints" / str(spec["run_name"]),
        )
    raise KeyError(spec["task_type"])


def _tmux_window_command(script_path: Path, meta_json: Path, log_path: Path, pid_path: Path, python_bin: str) -> str:
    pythonpath = f"{ROOT}/code{os.environ.get('PYTHONPATH', '') and ':' + os.environ.get('PYTHONPATH', '')}"
    cmd = (
        f"PYTHONPATH={shlex.quote(pythonpath)} "
        f"STWM_PROC_TITLE={shlex.quote(str(os.environ.get('STWM_PROC_TITLE', 'python')))} "
        f"STWM_PROC_TITLE_MODE={shlex.quote(str(os.environ.get('STWM_PROC_TITLE_MODE', 'generic')))} "
        f"nohup {shlex.quote(str(python_bin))} {shlex.quote(str(script_path))} "
        f"--mode run-one --meta-json {shlex.quote(str(meta_json))} --work-root {shlex.quote(str(ROOT))} "
        f">> {shlex.quote(str(log_path))} 2>&1 < /dev/null & echo $! > {shlex.quote(str(pid_path))}; "
        f"while kill -0 \"$(cat {shlex.quote(str(pid_path))})\" 2>/dev/null; do sleep 30; done"
    )
    return "bash -lc " + shlex.quote(
        f"cd {shlex.quote(str(ROOT))}; rm -f {shlex.quote(str(pid_path))}; {cmd}; "
        f"printf '[%s] tmux_window_exit\\n' \"$(date -Iseconds)\" >> {shlex.quote(str(log_path))}"
    )


def _task_success(task: Dict[str, Any]) -> bool:
    paths = _paths_for_run(str(task["run_name"]))
    if not paths["best"].exists():
        return False
    if str(task["task_type"]).startswith("tusb_v3p1"):
        return paths["sidecar"].exists() or paths["best"].exists()
    return True


def _task_state(task: Dict[str, Any], session_name: str) -> Tuple[str, str]:
    paths = _paths_for_run(str(task["run_name"]))
    if _task_success(task):
        return "completed", ""
    final_payload = _read_json(paths["final"])
    progress_payload = _read_json(paths["progress"])
    message = str(final_payload.get("message", "") or progress_payload.get("message", "") or "")
    if final_payload and str(final_payload.get("status", "")).lower() == "failed":
        return "failed", message or "trainer_failed"
    if progress_payload and str(progress_payload.get("status", "")).lower() == "failed":
        return "failed", message or "trainer_failed"
    if str(task.get("window_name", "")) in set(_tmux_windows(session_name)):
        return "running", ""
    if bool(task.get("launched", False)):
        if final_payload or progress_payload:
            if message:
                return "failed", message
        return "failed", "window_exited_without_best_checkpoint"
    return "pending", ""


def _all_coverage_after_run() -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    per_method: Dict[str, Any] = {}
    for method, mapping in _method_seed_dirs().items():
        method_rows = []
        for seed in MATCHED_SEEDS:
            row = _coverage_row(method, seed, mapping[seed])
            rows.append(row)
            method_rows.append(row)
        per_method[method] = {
            "rows": method_rows,
            "coverage_count": int(sum(int(bool(r["best_pt_exists"])) for r in method_rows)),
            "fully_covered": bool(all(bool(r["best_pt_exists"]) for r in method_rows)),
        }
    return {"rows": rows, "per_method": per_method}


def _write_launch_report(tasks: List[Dict[str, Any]]) -> Dict[str, Any]:
    payload = {
        "generated_at_utc": now_iso(),
        "training_jobs_launched": [
            {
                "task_name": str(t["task_name"]),
                "method": str(t["method"]),
                "seed": int(t["seed"]),
                "run_name": str(t["run_name"]),
            }
            for t in tasks
            if bool(t.get("launched", False))
        ],
        "tmux_sessions": [SESSION],
        "log_paths": [str(_paths_for_run(str(t["run_name"]))["log"]) for t in tasks if bool(t.get("launched", False))],
        "output_dirs": [str(_paths_for_run(str(t["run_name"]))["output_dir"]) for t in tasks if bool(t.get("launched", False))],
    }
    _write_json(LAUNCH_REPORT, payload)
    return payload


def _write_summary_report(tasks: List[Dict[str, Any]]) -> Dict[str, Any]:
    coverage = _all_coverage_after_run()
    failed_tasks: List[Dict[str, Any]] = []
    completed_count = 0
    for task in tasks:
        status, reason = _task_state(task, SESSION)
        if status == "completed":
            completed_count += 1
        elif status == "failed":
            failed_tasks.append(
                {
                    "task_name": str(task["task_name"]),
                    "method": str(task["method"]),
                    "seed": int(task["seed"]),
                    "run_name": str(task["run_name"]),
                    "exact_fail_reason": reason or "unknown_failure",
                }
            )
    payload = {
        "generated_at_utc": now_iso(),
        "per_method_seed_coverage_after_run": coverage["per_method"],
        "completed_count": int(completed_count),
        "failed_count": int(len(failed_tasks)),
        "still_missing_count": int(sum(int(not bool(row["best_pt_exists"])) for row in coverage["rows"])),
        "exact_fail_reasons": failed_tasks,
    }
    _write_json(SUMMARY_REPORT, payload)
    lines = [
        "# STWM Matched-6seed Real Completion 20260420",
        "",
        f"- completed_count: {payload['completed_count']}",
        f"- failed_count: {payload['failed_count']}",
        f"- still_missing_count: {payload['still_missing_count']}",
    ]
    for method, row in payload["per_method_seed_coverage_after_run"].items():
        lines.append(f"- {method}: coverage_count={row['coverage_count']}, fully_covered={row['fully_covered']}")
    if failed_tasks:
        lines.append("")
        lines.append("## Fail Reasons")
        for row in failed_tasks:
            lines.append(f"- {row['task_name']}: {row['exact_fail_reason']}")
    _write_md(SUMMARY_DOC, lines)
    return payload


def _coverage_complete_for(method: str) -> bool:
    summary = _read_json(SUMMARY_REPORT)
    per_method = summary.get("per_method_seed_coverage_after_run", {})
    block = per_method.get(method, {})
    return bool(block.get("fully_covered", False))


def _legacy_85_ids() -> List[str]:
    audit = _read_json(LEGACY_DUALPANEL_AUDIT)
    per_item = ((audit.get("densified_200_context_preserving") or {}).get("per_item_results")) or []
    ids = [str(row.get("protocol_item_id", "")) for row in per_item if str(row.get("protocol_item_id", "")).strip()]
    return ids


def _context_eval_args() -> Namespace:
    return Namespace(
        shared_lease_path=str(SHARED_LEASE_PATH),
        lease_path=str(SHARED_LEASE_PATH),
        eval_device="auto",
        eval_required_mem_gb=24.0,
        eval_safety_margin_gb=4.0,
    )


def _main_eval_method_specs() -> Tuple[List[prev_eval.MethodSpec], List[str], List[str]]:
    mapping = _method_seed_dirs()
    specs: List[prev_eval.MethodSpec] = []
    main_methods: List[str] = []
    secondary_methods: List[str] = []
    for method in ["TUSB-v3.1", "calibration-only", "cropenc baseline", "legacysem baseline"]:
        if _coverage_complete_for(method):
            main_methods.append(method)
        else:
            secondary_methods.append(method)
            continue
        for seed in MATCHED_SEEDS:
            run_name = mapping[method][seed]
            ckpt_path = ROOT / "outputs" / "checkpoints" / run_name / "best.pt"
            specs.append(
                prev_eval.MethodSpec(
                    name=f"{method}::seed{seed}",
                    run_name=run_name,
                    method_type="stage2" if method != "stage1 frozen" else "stage1",
                    checkpoint_path=str(ckpt_path),
                )
            )
    return specs, main_methods, secondary_methods


def _aggregate_method_seed_rows(methods: List[Dict[str, Any]], prefix: str) -> Dict[str, Any]:
    seed_rows = [row for row in methods if str(row.get("name", "")).startswith(prefix)]
    out_rows: List[Dict[str, Any]] = []
    for row in seed_rows:
        name = str(row.get("name", ""))
        seed = int(name.split("seed")[-1])
        out_rows.append(
            {
                "seed": seed,
                "top1": float(row.get("query_future_top1_acc", 0.0)),
                "hit_rate": float(row.get("query_future_hit_rate", 0.0)),
                "localization_error": float(row.get("query_future_localization_error", 0.0)),
                "mask_iou_at_top1": float(row.get("future_mask_iou_at_top1", 0.0)),
                "hard_subset_top1": float(row.get("hard_subset_top1_acc", 0.0)),
                "ambiguity_top1": float(row.get("ambiguity_top1_acc", 0.0)),
                "appearance_change_top1": float(row.get("appearance_change_top1_acc", 0.0)),
                "occlusion_reappearance_top1": float((row.get("panels", {}) or {}).get("occlusion_reappearance", {}).get("query_future_top1_acc", 0.0)),
                "long_gap_persistence_top1": float((row.get("panels", {}) or {}).get("long_gap_persistence", {}).get("query_future_top1_acc", 0.0)),
                "small_object_top1": float(row.get("small_object_top1_acc", 0.0)),
            }
        )
    out_rows.sort(key=lambda r: int(r["seed"]))
    summary: Dict[str, Any] = {"seed_rows": out_rows, "mean": {}, "std": {}}
    metrics = [
        "top1",
        "hit_rate",
        "localization_error",
        "mask_iou_at_top1",
        "hard_subset_top1",
        "ambiguity_top1",
        "appearance_change_top1",
        "occlusion_reappearance_top1",
        "long_gap_persistence_top1",
        "small_object_top1",
    ]
    for metric in metrics:
        vals = [float(r[metric]) for r in out_rows]
        summary["mean"][metric] = float(np.mean(vals)) if vals else 0.0
        summary["std"][metric] = float(np.std(vals)) if vals else 0.0
    return summary


def _add_win_counts(panel_summary: Dict[str, Any]) -> None:
    tusb_rows = {int(r["seed"]): r for r in panel_summary["TUSB-v3.1"]["seed_rows"]}
    for baseline in ["calibration-only", "cropenc baseline", "legacysem baseline"]:
        base_rows = {int(r["seed"]): r for r in panel_summary[baseline]["seed_rows"]}
        wins = 0
        delta_table: List[Dict[str, Any]] = []
        for seed in MATCHED_SEEDS:
            if seed not in tusb_rows or seed not in base_rows:
                continue
            delta = float(tusb_rows[seed]["top1"] - base_rows[seed]["top1"])
            if delta > 0:
                wins += 1
            delta_table.append({"seed": seed, "delta_top1": delta})
        panel_summary["TUSB-v3.1"][f"win_count_vs_{baseline.replace(' ', '_')}"] = int(wins)
        panel_summary["TUSB-v3.1"][f"seedwise_delta_vs_{baseline.replace(' ', '_')}"] = delta_table


def _run_main_eval() -> Dict[str, Any]:
    _apply_process_title_normalization()
    protocol = _read_json(PROTOCOL_V3_JSON)
    all_items = [item for item in protocol.get("items", []) if isinstance(item, dict)]
    legacy_ids = set(_legacy_85_ids())
    legacy_items = [item for item in all_items if str(item.get("protocol_item_id", "")) in legacy_ids]
    dense_items = list(all_items)
    specs, main_methods, secondary_methods = _main_eval_method_specs()
    eval_args = _context_eval_args()
    if not specs:
        payload = {
            "generated_at_utc": now_iso(),
            "main_table_methods": [],
            "secondary_methods": secondary_methods,
            "exact_blocking_reason": "no_method_with_full_6seed_coverage",
        }
        _write_json(MAIN_EVAL_REPORT, payload)
        _write_md(MAIN_EVAL_DOC, ["# STWM Matched-6seed Main Eval 20260420", "", "- blocked: no method has full 6-seed coverage."])
        return payload

    legacy_eval = ctxeval._run_eval_mode(
        args=eval_args,
        protocol_items=legacy_items,
        specs=specs,
        mode_name="legacy_85_context_preserving",
        builder=lambda item: evalv3._build_context_preserving_item_batch_v3(item, temporal_window=5, max_context_entities=8),
    )
    dense_eval = ctxeval._run_eval_mode(
        args=eval_args,
        protocol_items=dense_items,
        specs=specs,
        mode_name="densified_200_context_preserving",
        builder=lambda item: evalv3._build_context_preserving_item_batch_v3(item, temporal_window=5, max_context_entities=8),
    )
    legacy_summary = {
        method: _aggregate_method_seed_rows(legacy_eval["methods"], f"{method}::seed") for method in main_methods
    }
    dense_summary = {
        method: _aggregate_method_seed_rows(dense_eval["methods"], f"{method}::seed") for method in main_methods
    }
    _add_win_counts(legacy_summary)
    _add_win_counts(dense_summary)
    tusb_mean = float(dense_summary["TUSB-v3.1"]["mean"]["top1"])
    cal_mean = float(dense_summary["calibration-only"]["mean"]["top1"])
    tusb_hard = float(dense_summary["TUSB-v3.1"]["mean"]["hard_subset_top1"])
    cal_hard = float(dense_summary["calibration-only"]["mean"]["hard_subset_top1"])
    payload = {
        "generated_at_utc": now_iso(),
        "main_table_methods": main_methods,
        "secondary_methods": secondary_methods,
        "legacy_85_context_preserving": {
            "protocol_item_count": int(legacy_eval.get("protocol_item_count", 0)),
            "skipped_protocol_item_count": int(legacy_eval.get("skipped_protocol_item_count", 0)),
            "protocol_eval_context_entity_count_mean": float(legacy_eval.get("protocol_eval_context_entity_count_mean", 0.0)),
            "per_method": legacy_summary,
        },
        "densified_200_context_preserving": {
            "protocol_item_count": int(dense_eval.get("protocol_item_count", 0)),
            "skipped_protocol_item_count": int(dense_eval.get("skipped_protocol_item_count", 0)),
            "protocol_eval_context_entity_count_mean": float(dense_eval.get("protocol_eval_context_entity_count_mean", 0.0)),
            "per_method": dense_summary,
            "skipped_protocol_items": dense_eval.get("skipped_protocol_items", []),
        },
        "matched_6seed_improved_vs_calibration": bool(tusb_mean > cal_mean),
        "densified_200_context_preserving_hard_subsets_improved": bool(tusb_hard > cal_hard),
    }
    _write_json(MAIN_EVAL_REPORT, payload)
    _write_md(
        MAIN_EVAL_DOC,
        [
            "# STWM Matched-6seed Main Eval 20260420",
            "",
            f"- main_table_methods: {', '.join(main_methods)}",
            f"- secondary_methods: {', '.join(secondary_methods) if secondary_methods else 'none'}",
            f"- legacy_85_context_preserving.count: {payload['legacy_85_context_preserving']['protocol_item_count']}",
            f"- densified_200_context_preserving.count: {payload['densified_200_context_preserving']['protocol_item_count']}",
            f"- matched_6seed_improved_vs_calibration: {payload['matched_6seed_improved_vs_calibration']}",
            f"- densified_200_context_preserving_hard_subsets_improved: {payload['densified_200_context_preserving_hard_subsets_improved']}",
        ],
    )
    return payload


def _bootstrap_delta(a: np.ndarray, b: np.ndarray, n_boot: int = 4000, seed: int = 0) -> Dict[str, Any]:
    if len(a) == 0 or len(a) != len(b):
        return {"count": 0, "mean_delta": 0.0, "ci95_low": 0.0, "ci95_high": 0.0, "zero_excluded": False, "bootstrap_win_rate": 0.0}
    delta = a - b
    rng = np.random.default_rng(seed)
    samples = np.empty(n_boot, dtype=np.float64)
    n = len(delta)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        samples[i] = float(np.mean(delta[idx]))
    low = float(np.quantile(samples, 0.025))
    high = float(np.quantile(samples, 0.975))
    return {
        "count": int(n),
        "mean_delta": float(np.mean(delta)),
        "ci95_low": low,
        "ci95_high": high,
        "zero_excluded": bool(low > 0.0 or high < 0.0),
        "bootstrap_win_rate": float(np.mean(samples > 0.0)),
    }


def _run_strict_bootstrap() -> Dict[str, Any]:
    _apply_process_title_normalization()
    main_eval = _read_json(MAIN_EVAL_REPORT)
    if not main_eval or not bool(main_eval.get("matched_6seed_improved_vs_calibration", False) or True):
        pass
    protocol = _read_json(PROTOCOL_V3_JSON)
    all_items = [item for item in protocol.get("items", []) if isinstance(item, dict)]
    eval_args = _context_eval_args()
    mapping = _method_seed_dirs()
    tusb_specs = [
        prev_eval.MethodSpec(
            name=f"TUSB-v3.1::seed{seed}",
            run_name=mapping["TUSB-v3.1"][seed],
            method_type="stage2",
            checkpoint_path=str(ROOT / "outputs/checkpoints" / mapping["TUSB-v3.1"][seed] / "best.pt"),
        )
        for seed in MATCHED_SEEDS
        if _coverage_complete_for("TUSB-v3.1")
    ]
    cal_specs = [
        prev_eval.MethodSpec(
            name=f"calibration-only::seed{seed}",
            run_name=mapping["calibration-only"][seed],
            method_type="stage2",
            checkpoint_path=str(ROOT / "outputs/checkpoints" / mapping["calibration-only"][seed] / "best.pt"),
        )
        for seed in MATCHED_SEEDS
        if _coverage_complete_for("calibration-only")
    ]
    specs = tusb_specs + cal_specs
    dense_eval = ctxeval._run_eval_mode(
        args=eval_args,
        protocol_items=all_items,
        specs=specs,
        mode_name="densified_200_context_preserving",
        builder=lambda item: evalv3._build_context_preserving_item_batch_v3(item, temporal_window=5, max_context_entities=8),
    )
    per_item = dense_eval.get("per_item_results", [])
    pairs: Dict[str, Tuple[List[float], List[float]]] = {
        "overall_top1": ([], []),
        "hard_subset_top1": ([], []),
        "ambiguity_top1": ([], []),
        "appearance_change_top1": ([], []),
    }
    for item_row in per_item:
        tags = set(str(x) for x in item_row.get("subset_tags", []))
        methods = item_row.get("methods", {})
        for seed in MATCHED_SEEDS:
            ta = methods.get(f"TUSB-v3.1::seed{seed}")
            tb = methods.get(f"calibration-only::seed{seed}")
            if not isinstance(ta, dict) or not isinstance(tb, dict):
                continue
            pairs["overall_top1"][0].append(float(ta.get("query_future_top1_acc", 0.0)))
            pairs["overall_top1"][1].append(float(tb.get("query_future_top1_acc", 0.0)))
            if tags:
                pairs["hard_subset_top1"][0].append(float(ta.get("query_future_top1_acc", 0.0)))
                pairs["hard_subset_top1"][1].append(float(tb.get("query_future_top1_acc", 0.0)))
            if "crossing_ambiguity" in tags:
                pairs["ambiguity_top1"][0].append(float(ta.get("query_future_top1_acc", 0.0)))
                pairs["ambiguity_top1"][1].append(float(tb.get("query_future_top1_acc", 0.0)))
            if "appearance_change" in tags:
                pairs["appearance_change_top1"][0].append(float(ta.get("query_future_top1_acc", 0.0)))
                pairs["appearance_change_top1"][1].append(float(tb.get("query_future_top1_acc", 0.0)))
    metrics = {
        key: _bootstrap_delta(np.asarray(a), np.asarray(b), seed=abs(hash(key)) % (2**32))
        for key, (a, b) in pairs.items()
    }
    payload = {
        "generated_at_utc": now_iso(),
        "panel": "densified_200_context_preserving",
        "comparison": "TUSB-v3.1 vs calibration-only",
        "metrics": metrics,
        "zero_excluded": bool(metrics["overall_top1"]["zero_excluded"]),
    }
    _write_json(BOOTSTRAP_REPORT, payload)
    _write_md(
        BOOTSTRAP_DOC,
        [
            "# STWM Matched-6seed Strict Bootstrap 20260420",
            "",
            f"- overall_top1.mean_delta: {metrics['overall_top1']['mean_delta']:.6f}",
            f"- overall_top1.ci95: [{metrics['overall_top1']['ci95_low']:.6f}, {metrics['overall_top1']['ci95_high']:.6f}]",
            f"- overall_top1.zero_excluded: {metrics['overall_top1']['zero_excluded']}",
            f"- hard_subset_top1.zero_excluded: {metrics['hard_subset_top1']['zero_excluded']}",
            f"- ambiguity_top1.zero_excluded: {metrics['ambiguity_top1']['zero_excluded']}",
            f"- appearance_change_top1.zero_excluded: {metrics['appearance_change_top1']['zero_excluded']}",
        ],
    )
    return payload


def _run_queue(python_bin: str) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    _ensure_tmux_session(SESSION)
    tasks = _task_spec_list()
    task_by_name = {str(t["task_name"]): t for t in tasks}
    for task in tasks:
        task["launched"] = False
        task["final_state"] = "pending"
        task["exact_reason"] = ""

    started_tasks: List[Dict[str, Any]] = []
    _write_launch_report(tasks)
    deadline = time.time() + TRAIN_TIMEOUT_SECONDS

    while time.time() < deadline:
        running = 0
        terminal = 0
        for task in tasks:
            state, reason = _task_state(task, SESSION)
            task["final_state"] = state
            if state != "pending":
                task["launched"] = True
            if reason:
                task["exact_reason"] = reason
            if state == "running":
                running += 1
            if state in {"completed", "failed", "blocked"}:
                terminal += 1
        if terminal == len(tasks):
            break

        launched_now = False
        while running < MAX_CONCURRENT_TRAIN:
            ready: Dict[str, Any] | None = None
            ready_selected: Dict[str, Any] | None = None
            for task in tasks:
                if bool(task["launched"]):
                    continue
                deps = [task_by_name[x] for x in task.get("deps", [])]
                if any(str(dep.get("final_state")) == "failed" for dep in deps):
                    task["launched"] = True
                    task["final_state"] = "blocked"
                    task["exact_reason"] = "dependency_failed"
                    continue
                if any(str(dep.get("final_state")) != "completed" for dep in deps):
                    continue
                selected = _select_gpu_from_pool(
                    pool=[int(x) for x in task["gpu_pool"]],
                    required_mem_gb=float(task["required_mem_gb"]),
                    safety_margin_gb=float(task["safety_margin_gb"]),
                    owner=str(task["run_name"]),
                    wait_timeout_seconds=0,
                )
                if selected is None:
                    task["exact_reason"] = "waiting_for_gpu"
                    continue
                ready = task
                ready_selected = selected
                break
            if ready is None:
                break
            try:
                meta, script_path = _build_task_launch(ready, python_bin=python_bin, selected=ready_selected)
                meta_json = Path(str(meta["meta_json"]))
                _write_json(meta_json, meta)
                paths = _paths_for_run(str(ready["run_name"]))
                paths["pid"].parent.mkdir(parents=True, exist_ok=True)
                cmd = _tmux_window_command(script_path, meta_json, paths["log"], paths["pid"], python_bin)
                _launch_tmux_task(SESSION, str(ready["window_name"]), cmd)
                ready["launched"] = True
                ready["final_state"] = "running"
                started_tasks.append(
                    {
                        "task_name": str(ready["task_name"]),
                        "method": str(ready["method"]),
                        "seed": int(ready["seed"]),
                        "run_name": str(ready["run_name"]),
                        "tmux_session": SESSION,
                        "window_name": str(ready["window_name"]),
                        "log_path": str(paths["log"]),
                        "output_dir": str(paths["output_dir"]),
                        "meta_json": str(meta_json),
                    }
                )
                _append_log(f"launched task={ready['task_name']} run_name={ready['run_name']}")
                running += 1
                launched_now = True
                _write_launch_report(tasks)
            except Exception as exc:
                ready["launched"] = True
                ready["final_state"] = "failed"
                ready["exact_reason"] = str(exc)
                _append_log(f"failed_to_launch task={ready['task_name']} reason={exc}")
        if not launched_now:
            time.sleep(POLL_SECONDS)
        _write_summary_report(tasks)

    summary = _write_summary_report(tasks)
    launch = _write_launch_report(tasks)
    plan = _write_plan_artifacts()
    return plan, launch, summary


def _final_next_step(summary: Dict[str, Any], main_eval: Dict[str, Any], bootstrap: Dict[str, Any]) -> str:
    tusb_ok = bool((summary.get("per_method_seed_coverage_after_run", {}) or {}).get("TUSB-v3.1", {}).get("fully_covered", False))
    cal_ok = bool((summary.get("per_method_seed_coverage_after_run", {}) or {}).get("calibration-only", {}).get("fully_covered", False))
    crop_ok = bool((summary.get("per_method_seed_coverage_after_run", {}) or {}).get("cropenc baseline", {}).get("fully_covered", False))
    legacy_ok = bool((summary.get("per_method_seed_coverage_after_run", {}) or {}).get("legacysem baseline", {}).get("fully_covered", False))
    if not (tusb_ok and cal_ok and crop_ok and legacy_ok):
        return "one_last_surgical_fix"
    if bool(main_eval.get("matched_6seed_improved_vs_calibration", False)) and bool(main_eval.get("densified_200_context_preserving_hard_subsets_improved", False)) and bool(bootstrap.get("zero_excluded", False)):
        return "start_writing_main_submission"
    if bool(main_eval.get("matched_6seed_improved_vs_calibration", False)):
        return "one_last_surgical_fix"
    return "stop_stage2_escalation_and_reframe_claims"


def run_all(args: Any) -> Dict[str, Any]:
    _append_log("matched6seed_real_completion_start")
    plan = _write_plan_artifacts()
    python_bin = str(args.python_bin)
    if not Path(python_bin).exists():
        raise FileNotFoundError(f"python_bin_missing:{python_bin}")
    plan, launch, summary = _run_queue(python_bin=python_bin)
    main_eval: Dict[str, Any] = {}
    bootstrap: Dict[str, Any] = {}
    if bool((summary.get("per_method_seed_coverage_after_run", {}) or {}).get("TUSB-v3.1", {}).get("fully_covered", False)) and bool((summary.get("per_method_seed_coverage_after_run", {}) or {}).get("calibration-only", {}).get("fully_covered", False)):
        main_eval = _run_main_eval()
        bootstrap = _run_strict_bootstrap()
    next_step = _final_next_step(summary, main_eval, bootstrap)
    result = {
        "generated_at_utc": now_iso(),
        "plan_report": str(PLAN_REPORT),
        "launch_report": str(LAUNCH_REPORT),
        "summary_report": str(SUMMARY_REPORT),
        "main_eval_report": str(MAIN_EVAL_REPORT),
        "bootstrap_report": str(BOOTSTRAP_REPORT),
        "next_step_choice": next_step,
    }
    _append_log(f"matched6seed_real_completion_done next_step_choice={next_step}")
    return result


def parse_args() -> Any:
    parser = ArgumentParser(description="STWM matched 6-seed real completion + mainline re-eval")
    parser.add_argument("--mode", default="run", choices=["run"])
    parser.add_argument("--python-bin", default=str(DEFAULT_PYTHON))
    return parser.parse_args()


def main() -> None:
    _apply_process_title_normalization()
    args = parse_args()
    print(json.dumps(run_all(args), ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
