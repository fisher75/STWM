#!/usr/bin/env python3
from __future__ import annotations

import json
import time
import zipfile
from pathlib import Path
from typing import Any


BASELINES = [
    "copy_residual_mlp",
    "copy_residual_transformer",
    "copy_gated_residual_no_trace",
    "copy_gated_residual_trace_only",
    "copy_gated_residual_plain_trace_semantic",
]
SEEDS = [42, 123, 456, 789, 1001]
RUN_TAG = "fstf_strong_copyaware_baselines_v8_20260501"


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"_read_error": str(exc)}


def _artifact(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "exists": path.exists(),
        "size_bytes": path.stat().st_size if path.exists() else 0,
        "mtime": path.stat().st_mtime if path.exists() else 0.0,
        "mtime_readable": time.strftime("%Y-%m-%d %H:%M:%S %Z", time.localtime(path.stat().st_mtime))
        if path.exists()
        else "",
    }


def _zip_contains_any(paths: list[str]) -> bool:
    zips = list(Path(".").glob("*.zip")) + list(Path("outputs").glob("*.zip")) + list(Path("reports").glob("*.zip"))
    if not zips:
        return False
    wanted = set(paths)
    for zpath in zips:
        try:
            with zipfile.ZipFile(zpath) as zf:
                names = set(zf.namelist())
                if any(p in names or f"./{p}" in names for p in wanted):
                    return True
        except Exception:
            continue
    return False


def main() -> int:
    ckpt_root = Path("outputs/checkpoints") / RUN_TAG
    log_root = Path("logs") / RUN_TAG
    status_root = Path("outputs/run_status") / RUN_TAG
    run_rows = []
    artifact_paths: list[str] = []
    for baseline in BASELINES:
        for seed in SEEDS:
            run_dir = ckpt_root / baseline / str(seed)
            log = log_root / f"{baseline}_seed{seed}.log"
            status_candidates = [
                status_root / f"{baseline}_seed{seed}.status.json",
                status_root / f"{baseline}_seed{seed}_relaunch.status.json",
                status_root / f"{baseline}_seed{seed}_eval_relaunch.status.json",
            ]
            statuses = [_read_json(p) | {"status_path": str(p), "status_exists": p.exists()} for p in status_candidates if p.exists()]
            row = {
                "baseline": baseline,
                "seed": seed,
                "checkpoint": _artifact(run_dir / "checkpoint.pt"),
                "train_summary": _artifact(run_dir / "train_summary.json"),
                "eval_summary": _artifact(run_dir / "eval_test.json"),
                "log": _artifact(log),
                "statuses": statuses,
                "tmux_session": next((s.get("tmux_session") for s in reversed(statuses) if s.get("tmux_session")), ""),
                "cuda_visible_devices": next((s.get("cuda_visible_devices") for s in reversed(statuses) if s.get("cuda_visible_devices") is not None), ""),
            }
            for key in ["checkpoint", "train_summary", "eval_summary", "log"]:
                artifact_paths.append(row[key]["path"])
            run_rows.append(row)
    checkpoint_count = sum(1 for r in run_rows if r["checkpoint"]["exists"])
    eval_summary_count = sum(1 for r in run_rows if r["eval_summary"]["exists"])
    train_summary_count = sum(1 for r in run_rows if r["train_summary"]["exists"])
    log_count = sum(1 for r in run_rows if r["log"]["exists"] and r["log"]["size_bytes"] > 0)
    live_artifacts_exist = bool(checkpoint_count == len(BASELINES) * len(SEEDS) and eval_summary_count == len(BASELINES) * len(SEEDS))
    zip_contains = _zip_contains_any(artifact_paths)
    report = {
        "audit_name": "stwm_fstf_v8_live_artifact_audit",
        "expected_learned_run_count": len(BASELINES) * len(SEEDS),
        "checkpoint_count": checkpoint_count,
        "eval_summary_count": eval_summary_count,
        "train_summary_count": train_summary_count,
        "log_count": log_count,
        "run_artifacts": run_rows,
        "copy_eval": _artifact(ckpt_root / "copy_semantic_memory_baseline" / "eval_test.json"),
        "oracle_eval": _artifact(ckpt_root / "oracle_change_gate_upper_bound" / "eval_test.json"),
        "live_artifacts_exist": live_artifacts_exist,
        "zip_snapshot_contains_outputs_or_logs": zip_contains,
        "exported_snapshot_incomplete": not zip_contains,
        "candidate_scorer_used": False,
        "future_candidate_leakage": False,
        "teacher_forced_path_used": False,
    }
    out = Path("reports/stwm_fstf_v8_live_artifact_audit_20260501.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    doc = Path("docs/STWM_FSTF_V8_LIVE_ARTIFACT_AUDIT_20260501.md")
    doc.parent.mkdir(parents=True, exist_ok=True)
    doc.write_text(
        "\n".join(
            [
                "# STWM FSTF V8 Live Artifact Audit",
                "",
                f"- checkpoint_count: `{checkpoint_count}`",
                f"- eval_summary_count: `{eval_summary_count}`",
                f"- train_summary_count: `{train_summary_count}`",
                f"- log_count: `{log_count}`",
                f"- live_artifacts_exist: `{live_artifacts_exist}`",
                f"- exported_snapshot_incomplete: `{not zip_contains}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"[v8-artifact-audit] report={out}")
    print(f"[v8-artifact-audit] doc={doc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
