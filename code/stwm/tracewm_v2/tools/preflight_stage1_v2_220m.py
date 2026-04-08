#!/usr/bin/env python3
from __future__ import annotations

from argparse import ArgumentParser
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List
import json
import math
import subprocess
import sys


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_args() -> Any:
    parser = ArgumentParser(description="Stage1-v2 220M preflight dry run")
    parser.add_argument("--python-bin", default="/home/chen034/miniconda3/envs/stwm/bin/python")
    parser.add_argument("--work-root", default="/home/chen034/workspace/stwm")
    parser.add_argument("--contract-path", default="/home/chen034/workspace/data/_manifests/stage1_v2_trace_cache_contract_20260408.json")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--batch-size-candidates", default="2,1")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-samples-per-dataset", type=int, default=128)
    parser.add_argument("--report-json", default="/home/chen034/workspace/stwm/reports/stage1_v2_220m_preflight_20260408.json")
    parser.add_argument("--report-md", default="/home/chen034/workspace/stwm/docs/STAGE1_V2_220M_PREFLIGHT_20260408.md")
    return parser.parse_args()


def _isfinite_metrics(summary_payload: Dict[str, Any]) -> bool:
    fm = summary_payload.get("final_metrics", {})
    if not isinstance(fm, dict):
        return False
    for key, value in fm.items():
        if key == "epoch":
            continue
        try:
            v = float(value)
        except Exception:
            return False
        if not math.isfinite(v):
            return False
    return True


def _classify_failure(stdout: str, stderr: str, return_code: int) -> str:
    merged = (stdout + "\n" + stderr).lower()
    if "out of memory" in merged or "cuda error: out of memory" in merged or "cudnn_status_alloc_failed" in merged:
        return "oom"
    if return_code != 0:
        return "process_error"
    return "unknown"


def main() -> None:
    args = parse_args()
    work_root = Path(args.work_root)
    report_json = Path(args.report_json)
    report_md = Path(args.report_md)

    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_md.parent.mkdir(parents=True, exist_ok=True)

    trainer = work_root / "code/stwm/tracewm_v2/trainers/train_tracewm_stage1_v2.py"

    attempts: List[Dict[str, Any]] = []
    passed = False
    selected_batch_size = -1
    failure_reason = "not_run"

    try_batches = []
    for raw in str(args.batch_size_candidates).split(","):
        raw = raw.strip()
        if not raw:
            continue
        try:
            try_batches.append(int(raw))
        except Exception:
            continue
    if not try_batches:
        try_batches = [1]

    for bs in try_batches:
        attempt_tag = f"preflight_220m_bs{bs}"
        summary_json = work_root / "reports" / f"stage1_v2_220m_preflight_bs{bs}_summary_20260408.json"
        results_md = work_root / "docs" / f"STAGE1_V2_220M_PREFLIGHT_BS{bs}_20260408.md"
        timing_json = work_root / "reports" / f"stage1_v2_220m_preflight_bs{bs}_timing_20260408.json"
        output_dir = work_root / "outputs/training" / f"stage1_v2_220m_preflight_bs{bs}_20260408"

        cmd = [
            str(args.python_bin),
            str(trainer),
            "--contract-path", str(args.contract_path),
            "--dataset-names", "pointodyssey", "kubric",
            "--train-split", "train",
            "--obs-len", "8",
            "--fut-len", "8",
            "--max-tokens", "64",
            "--max-samples-per-dataset", str(int(args.max_samples_per_dataset)),
            "--model-preset", "prototype_220m",
            "--epochs", "1",
            "--steps-per-epoch", str(int(args.steps)),
            "--batch-size", str(int(bs)),
            "--num-workers", str(int(args.num_workers)),
            "--pin-memory",
            "--persistent-workers",
            "--prefetch-factor", "2",
            "--enable-visibility",
            "--enable-residual",
            "--enable-velocity",
            "--ablation-tag", attempt_tag,
            "--output-dir", str(output_dir),
            "--summary-json", str(summary_json),
            "--results-md", str(results_md),
            "--perf-step-timing-json", str(timing_json),
        ]

        proc = subprocess.run(cmd, cwd=str(work_root), text=True, capture_output=True)

        attempt: Dict[str, Any] = {
            "batch_size": int(bs),
            "return_code": int(proc.returncode),
            "summary_json": str(summary_json),
            "results_md": str(results_md),
            "timing_json": str(timing_json),
            "stdout_tail": "\n".join(proc.stdout.strip().splitlines()[-20:]),
            "stderr_tail": "\n".join(proc.stderr.strip().splitlines()[-20:]),
        }

        metrics_finite = False
        if proc.returncode == 0 and summary_json.exists():
            try:
                summary = json.loads(summary_json.read_text(encoding="utf-8"))
                metrics_finite = _isfinite_metrics(summary)
                attempt["final_metrics"] = dict(summary.get("final_metrics", {}))
                attempt["estimated_parameter_count"] = int(summary.get("model", {}).get("estimated_parameter_count", 0))
                attempt["parameter_count"] = int(summary.get("model", {}).get("parameter_count", 0))
            except Exception as exc:
                attempt["summary_parse_error"] = str(exc)

        attempt["metrics_finite"] = bool(metrics_finite)
        if proc.returncode == 0 and metrics_finite:
            attempt["status"] = "pass"
            attempts.append(attempt)
            passed = True
            selected_batch_size = int(bs)
            failure_reason = ""
            break

        failure_reason = _classify_failure(proc.stdout, proc.stderr, proc.returncode)
        attempt["status"] = "fail"
        attempt["failure_reason"] = failure_reason
        attempts.append(attempt)

    payload = {
        "generated_at_utc": now_iso(),
        "contract_path": str(args.contract_path),
        "preflight_pass": bool(passed),
        "target_preset": "prototype_220m",
        "steps": int(args.steps),
        "selected_batch_size": int(selected_batch_size),
        "failure_reason": "" if passed else str(failure_reason),
        "attempts": attempts,
    }
    report_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Stage1-v2 220M Preflight",
        "",
        f"- generated_at_utc: {payload['generated_at_utc']}",
        f"- contract_path: {payload['contract_path']}",
        f"- target_preset: {payload['target_preset']}",
        f"- steps: {payload['steps']}",
        f"- preflight_pass: {payload['preflight_pass']}",
        f"- selected_batch_size: {payload['selected_batch_size']}",
        f"- failure_reason: {payload['failure_reason']}",
        "",
        "| attempt | batch_size | return_code | metrics_finite | status | failure_reason |",
        "|---|---:|---:|---|---|---|",
    ]
    for i, att in enumerate(attempts, start=1):
        lines.append(
            "| {idx} | {bs} | {rc} | {mf} | {st} | {fr} |".format(
                idx=i,
                bs=int(att.get("batch_size", -1)),
                rc=int(att.get("return_code", -1)),
                mf=bool(att.get("metrics_finite", False)),
                st=str(att.get("status", "unknown")),
                fr=str(att.get("failure_reason", "")),
            )
        )

    report_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"[stage1-v2-220m-preflight] report_json={report_json}")
    print(f"[stage1-v2-220m-preflight] report_md={report_md}")
    print(f"[stage1-v2-220m-preflight] preflight_pass={passed}")

    if not passed:
        sys.exit(31)


if __name__ == "__main__":
    main()
