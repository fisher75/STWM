from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import json
import statistics
import subprocess
import time
from typing import Any

import numpy as np


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Run raw/frontend_cache A/B benchmark with process CPU/GPU sampling")
    parser.add_argument("--repo-root", default="/home/chen034/workspace/stwm")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--frontend-cache-dir", required=True)
    parser.add_argument("--frontend-cache-index", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--gpu-index", type=int, default=7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample-interval", type=float, default=1.0)
    return parser


def _safe_float(value: str) -> float:
    v = str(value).strip()
    if not v or v == "-":
        return 0.0
    try:
        return float(v)
    except ValueError:
        return 0.0


def _find_train_pid(run_name: str) -> tuple[int | None, float]:
    out = subprocess.run(["ps", "-eo", "pid,pcpu,args"], capture_output=True, text=True, check=True).stdout
    for line in out.splitlines()[1:]:
        if run_name not in line:
            continue
        if "train_stwm_v4_2_real.py" not in line:
            continue
        if "conda run" in line:
            continue
        parts = line.strip().split(None, 2)
        if len(parts) < 3:
            continue
        try:
            pid = int(parts[0])
            cpu = float(parts[1])
        except ValueError:
            continue
        return pid, cpu
    return None, 0.0


def _sample_pmon(gpu_index: int, pid: int | None) -> tuple[float, float]:
    if pid is None:
        return 0.0, 0.0

    out = subprocess.run(
        ["nvidia-smi", "pmon", "-i", str(gpu_index), "-s", "um", "-c", "1"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout

    for line in out.splitlines():
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        cols = text.split()
        if len(cols) < 5:
            continue
        try:
            row_pid = int(cols[1])
        except ValueError:
            continue
        if row_pid != pid:
            continue
        sm = _safe_float(cols[3])
        mem = _safe_float(cols[4])
        return sm, mem

    return 0.0, 0.0


def _sample_used_mem(gpu_index: int, pid: int | None) -> float:
    if pid is None:
        return 0.0

    out = subprocess.run(
        [
            "nvidia-smi",
            "--id",
            str(gpu_index),
            "--query-compute-apps=pid,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout

    for line in out.splitlines():
        cols = [x.strip() for x in line.split(",")]
        if len(cols) < 2:
            continue
        try:
            row_pid = int(cols[0])
        except ValueError:
            continue
        if row_pid == pid:
            return _safe_float(cols[1])
    return 0.0


def _pct(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def _summarize_train_log(path: Path) -> dict[str, float]:
    rows = []
    if path.exists():
        for line in path.read_text().splitlines():
            text = line.strip()
            if not text:
                continue
            try:
                rows.append(json.loads(text))
            except json.JSONDecodeError:
                continue

    step_vals = [float(r.get("step_time_s", 0.0)) for r in rows]
    data_vals = [float(r.get("data_time_s", 0.0)) for r in rows]
    wait_vals = [float(r.get("data_wait_ratio", 0.0)) for r in rows]

    return {
        "rows": float(len(rows)),
        "step_time_mean": float(statistics.mean(step_vals)) if step_vals else 0.0,
        "step_time_p50": _pct(step_vals, 50.0),
        "step_time_p95": _pct(step_vals, 95.0),
        "data_time_mean": float(statistics.mean(data_vals)) if data_vals else 0.0,
        "data_time_p50": _pct(data_vals, 50.0),
        "data_time_p95": _pct(data_vals, 95.0),
        "data_wait_mean": float(statistics.mean(wait_vals)) if wait_vals else 0.0,
        "data_wait_p50": _pct(wait_vals, 50.0),
        "data_wait_p95": _pct(wait_vals, 95.0),
    }


def _build_cmd(
    *,
    repo_root: Path,
    manifest: Path,
    output_dir: Path,
    run_name: str,
    seed: int,
    steps: int,
    data_mode: str,
    frontend_cache_dir: Path,
    frontend_cache_index: Path,
    gpu_index: int,
) -> list[str]:
    cmd = [
        "env",
        f"CUDA_VISIBLE_DEVICES={gpu_index}",
        f"PYTHONPATH={repo_root / 'code'}",
        "conda",
        "run",
        "--no-capture-output",
        "-n",
        "stwm",
        "python",
        str(repo_root / "code" / "stwm" / "trainers" / "train_stwm_v4_2_real.py"),
        "--data-root",
        str(repo_root / "data" / "external"),
        "--manifest",
        str(manifest),
        "--output-dir",
        str(output_dir),
        "--run-name",
        run_name,
        "--seed",
        str(seed),
        "--steps",
        str(steps),
        "--target-epochs",
        "0",
        "--min-optimizer-steps",
        "0",
        "--max-optimizer-steps",
        "0",
        "--sample-limit",
        "0",
        "--model-preset",
        "prototype_220m_v4_2",
        "--preset-file",
        str(repo_root / "code" / "stwm" / "configs" / "model_presets_v4_2.json"),
        "--use-teacher-priors",
        "--save-checkpoint",
        "--checkpoint-dir-name",
        "checkpoints",
        "--checkpoint-interval",
        "1000",
        "--milestone-interval",
        "0",
        "--micro-batch-per-gpu",
        "2",
        "--grad-accum",
        "8",
        "--num-workers",
        "12",
        "--prefetch-factor",
        "2",
        "--persistent-workers",
        "--pin-memory",
        "--bf16",
        "--activation-checkpointing",
        "--lambda-traj",
        "1.0",
        "--lambda-vis",
        "0.25",
        "--lambda-sem",
        "0.5",
        "--lambda-reid",
        "0.25",
        "--lambda-query",
        "0.25",
        "--lambda-reconnect",
        "0.1",
        "--gradient-audit-interval",
        "0",
        "--protocol-eval-interval",
        "0",
        "--data-mode",
        data_mode,
    ]

    if data_mode == "frontend_cache":
        cmd.extend(
            [
                "--frontend-cache-dir",
                str(frontend_cache_dir),
                "--frontend-cache-index",
                str(frontend_cache_index),
            ]
        )
    return cmd


def _run_mode(
    *,
    repo_root: Path,
    manifest: Path,
    output_root: Path,
    mode_name: str,
    data_mode: str,
    seed: int,
    steps: int,
    gpu_index: int,
    sample_interval: float,
    frontend_cache_dir: Path,
    frontend_cache_index: Path,
) -> dict[str, Any]:
    output_dir = output_root / mode_name
    output_dir.mkdir(parents=True, exist_ok=True)
    run_name = f"ab_{mode_name}_seed{seed}"

    cmd = _build_cmd(
        repo_root=repo_root,
        manifest=manifest,
        output_dir=output_dir,
        run_name=run_name,
        seed=seed,
        steps=steps,
        data_mode=data_mode,
        frontend_cache_dir=frontend_cache_dir,
        frontend_cache_index=frontend_cache_index,
        gpu_index=gpu_index,
    )

    started = time.perf_counter()
    proc = subprocess.Popen(cmd, cwd=str(repo_root), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    samples: list[dict[str, float]] = []
    while proc.poll() is None:
        pid, cpu_pct = _find_train_pid(run_name)
        sm_util, mem_util = _sample_pmon(gpu_index=gpu_index, pid=pid)
        used_mem = _sample_used_mem(gpu_index=gpu_index, pid=pid)
        samples.append(
            {
                "cpu_pct": float(cpu_pct),
                "gpu_sm_util": float(sm_util),
                "gpu_mem_util": float(mem_util),
                "gpu_used_mem_mib": float(used_mem),
            }
        )
        time.sleep(max(0.2, float(sample_interval)))

    stdout_text = ""
    if proc.stdout is not None:
        try:
            stdout_text = proc.stdout.read()
        except Exception:
            stdout_text = ""

    duration_s = float(time.perf_counter() - started)
    train_log = output_dir / "train_log.jsonl"
    summary_file = output_dir / "mini_val_summary.json"

    run_summary = {}
    if summary_file.exists():
        try:
            run_summary = json.loads(summary_file.read_text())
        except json.JSONDecodeError:
            run_summary = {}

    cpu_vals = [float(x["cpu_pct"]) for x in samples]
    sm_vals = [float(x["gpu_sm_util"]) for x in samples]
    mem_vals = [float(x["gpu_mem_util"]) for x in samples]
    used_vals = [float(x["gpu_used_mem_mib"]) for x in samples]

    return {
        "mode": mode_name,
        "data_mode": data_mode,
        "exit_code": int(proc.returncode if proc.returncode is not None else -1),
        "duration_s": float(duration_s),
        "output_dir": str(output_dir),
        "train_log": str(train_log),
        "summary_file": str(summary_file),
        "train_log_metrics": _summarize_train_log(train_log),
        "process_samples": int(len(samples)),
        "cpu_pct_mean": float(statistics.mean(cpu_vals)) if cpu_vals else 0.0,
        "cpu_pct_p95": _pct(cpu_vals, 95.0),
        "gpu_sm_util_mean": float(statistics.mean(sm_vals)) if sm_vals else 0.0,
        "gpu_sm_util_p95": _pct(sm_vals, 95.0),
        "gpu_mem_util_mean": float(statistics.mean(mem_vals)) if mem_vals else 0.0,
        "gpu_mem_util_p95": _pct(mem_vals, 95.0),
        "gpu_used_mem_mib_mean": float(statistics.mean(used_vals)) if used_vals else 0.0,
        "gpu_used_mem_mib_p95": _pct(used_vals, 95.0),
        "summary_runtime": run_summary.get("runtime", {}),
        "stdout_tail": stdout_text[-4000:],
    }


def main() -> None:
    args = build_parser().parse_args()

    repo_root = Path(args.repo_root)
    manifest = Path(args.manifest)
    frontend_cache_dir = Path(args.frontend_cache_dir)
    frontend_cache_index = Path(args.frontend_cache_index)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    raw_result = _run_mode(
        repo_root=repo_root,
        manifest=manifest,
        output_root=output_root,
        mode_name="raw",
        data_mode="raw",
        seed=int(args.seed),
        steps=int(args.steps),
        gpu_index=int(args.gpu_index),
        sample_interval=float(args.sample_interval),
        frontend_cache_dir=frontend_cache_dir,
        frontend_cache_index=frontend_cache_index,
    )

    frontend_result = _run_mode(
        repo_root=repo_root,
        manifest=manifest,
        output_root=output_root,
        mode_name="frontend_cache",
        data_mode="frontend_cache",
        seed=int(args.seed),
        steps=int(args.steps),
        gpu_index=int(args.gpu_index),
        sample_interval=float(args.sample_interval),
        frontend_cache_dir=frontend_cache_dir,
        frontend_cache_index=frontend_cache_index,
    )

    raw_metrics = raw_result.get("train_log_metrics", {})
    fe_metrics = frontend_result.get("train_log_metrics", {})

    compare = {
        "step_time_mean_delta": float(fe_metrics.get("step_time_mean", 0.0) - raw_metrics.get("step_time_mean", 0.0)),
        "step_time_mean_ratio": float(
            fe_metrics.get("step_time_mean", 0.0) / max(1e-9, raw_metrics.get("step_time_mean", 0.0))
        ),
        "data_time_mean_delta": float(fe_metrics.get("data_time_mean", 0.0) - raw_metrics.get("data_time_mean", 0.0)),
        "data_time_mean_ratio": float(
            fe_metrics.get("data_time_mean", 0.0) / max(1e-9, raw_metrics.get("data_time_mean", 0.0))
        ),
        "data_wait_mean_delta": float(fe_metrics.get("data_wait_mean", 0.0) - raw_metrics.get("data_wait_mean", 0.0)),
        "data_wait_mean_ratio": float(
            fe_metrics.get("data_wait_mean", 0.0) / max(1e-9, raw_metrics.get("data_wait_mean", 0.0))
        ),
        "cpu_pct_mean_delta": float(frontend_result.get("cpu_pct_mean", 0.0) - raw_result.get("cpu_pct_mean", 0.0)),
        "gpu_sm_util_mean_delta": float(frontend_result.get("gpu_sm_util_mean", 0.0) - raw_result.get("gpu_sm_util_mean", 0.0)),
        "gpu_used_mem_mib_mean_delta": float(
            frontend_result.get("gpu_used_mem_mib_mean", 0.0) - raw_result.get("gpu_used_mem_mib_mean", 0.0)
        ),
        "raw_exit_ok": bool(raw_result.get("exit_code", 1) == 0),
        "frontend_exit_ok": bool(frontend_result.get("exit_code", 1) == 0),
    }

    report = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "manifest": str(manifest),
            "frontend_cache_dir": str(frontend_cache_dir),
            "frontend_cache_index": str(frontend_cache_index),
            "seed": int(args.seed),
            "steps": int(args.steps),
            "gpu_index": int(args.gpu_index),
            "sample_interval": float(args.sample_interval),
        },
        "raw": raw_result,
        "frontend_cache": frontend_result,
        "compare": compare,
    }

    out_path = output_root / "ab_report.json"
    out_path.write_text(json.dumps(report, indent=2))
    print(str(out_path))


if __name__ == "__main__":
    main()
