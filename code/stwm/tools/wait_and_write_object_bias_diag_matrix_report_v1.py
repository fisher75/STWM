from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import json
import time
from typing import Any


DEFAULT_RUN_NAMES = [
    "full_v4_2_seed42_fixed_nowarm_lambda1_objdiag_v1",
    "wo_object_bias_v4_2_seed42_objdiag_v1",
    "full_v4_2_seed42_objbias_alpha025_objdiag_v1",
    "full_v4_2_seed42_objbias_alpha050_objdiag_v1",
    "full_v4_2_seed42_objbias_delayed200_objdiag_v1",
    "full_v4_2_seed42_objbias_gated_objdiag_v1",
]


def build_parser() -> ArgumentParser:
    p = ArgumentParser(description="Wait for object-bias diag matrix and write summary report")
    p.add_argument(
        "--queue-status-dir",
        default=(
            "/home/chen034/workspace/stwm/outputs/queue/"
            "stwm_protocol_v2_frontend_default_v1/d1_train/status"
        ),
    )
    p.add_argument(
        "--run-root",
        default=(
            "/home/chen034/workspace/stwm/outputs/training/"
            "stwm_v4_2_220m_protocol_object_bias_diag_v1"
        ),
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--run-names", default=",".join(DEFAULT_RUN_NAMES))
    p.add_argument("--poll-seconds", type=int, default=180)
    p.add_argument("--timeout-seconds", type=int, default=0)
    p.add_argument(
        "--out-report",
        default="/home/chen034/workspace/stwm/reports/stwm_object_bias_diag_matrix_v1.json",
    )
    p.add_argument(
        "--out-doc",
        default="/home/chen034/workspace/stwm/docs/STWM_OBJECT_BIAS_DIAG_MATRIX_REPORT_V1.md",
    )
    return p


def _safe_float(v: Any) -> float | None:
    try:
        x = float(v)
    except Exception:
        return None
    if x != x:
        return None
    if x == float("inf") or x == float("-inf"):
        return None
    return x


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text())
    except Exception:
        return None
    if isinstance(payload, dict):
        return payload
    return None


def _find_latest_status(status_dir: Path, run_name: str) -> Path | None:
    cands = sorted(status_dir.glob(f"*_{run_name}.status.json"))
    if not cands:
        return None
    return cands[-1]


def _collect_one(run_root: Path, status_dir: Path, run_name: str) -> dict[str, Any]:
    info: dict[str, Any] = {
        "run_name": run_name,
        "state": "missing",
    }
    status_path = _find_latest_status(status_dir, run_name)
    if status_path is None:
        return info

    status = _load_json(status_path)
    info["status_path"] = str(status_path)
    if status is None:
        info["state"] = "status_invalid"
        return info

    state = str(status.get("state", "unknown")).lower()
    info["state"] = state
    info["job_id"] = str(status.get("job_id", ""))
    info["main_log"] = str(status.get("main_log", ""))

    out_dir = run_root / run_name
    ckpt_dir = out_dir / "checkpoints"
    sidecar_path = ckpt_dir / "best_protocol_main_selection.json"
    info["output_dir"] = str(out_dir)
    info["selection_sidecar"] = str(sidecar_path)

    if not sidecar_path.exists():
        return info

    sidecar = _load_json(sidecar_path)
    if sidecar is None:
        return info

    eval_path = Path(str(sidecar.get("eval_summary", "")))
    info["eval_summary"] = str(eval_path)
    info["selection_metrics"] = sidecar.get("metrics", {})

    eval_payload = _load_json(eval_path)
    if eval_payload is None:
        return info

    metrics = eval_payload.get("metrics", {})
    if not isinstance(metrics, dict):
        metrics = {}
    info["protocol_metrics"] = {
        "query_localization_error": _safe_float(metrics.get("query_localization_error")),
        "query_top1_acc": _safe_float(metrics.get("query_top1_acc")),
        "future_trajectory_l1": _safe_float(metrics.get("future_trajectory_l1")),
        "future_mask_iou": _safe_float(metrics.get("future_mask_iou")),
        "query_hit_rate": _safe_float(metrics.get("query_hit_rate")),
        "identity_consistency": _safe_float(metrics.get("identity_consistency")),
        "identity_switch_rate": _safe_float(metrics.get("identity_switch_rate")),
    }
    return info


def _all_terminal(infos: list[dict[str, Any]]) -> bool:
    for info in infos:
        state = str(info.get("state", ""))
        if state not in {"done", "failed"}:
            return False
    return True


def _rank_key(info: dict[str, Any]) -> tuple[float, float, float, str]:
    metrics = info.get("protocol_metrics")
    if not isinstance(metrics, dict):
        return (float("inf"), float("-inf"), float("inf"), str(info.get("run_name", "")))
    qloc = _safe_float(metrics.get("query_localization_error"))
    qtop1 = _safe_float(metrics.get("query_top1_acc"))
    ftraj = _safe_float(metrics.get("future_trajectory_l1"))
    if qloc is None:
        qloc = float("inf")
    if qtop1 is None:
        qtop1 = float("-inf")
    if ftraj is None:
        ftraj = float("inf")
    return (float(qloc), float(-qtop1), float(ftraj), str(info.get("run_name", "")))


def _fmt(v: float | None) -> str:
    if v is None:
        return "na"
    return f"{float(v):.6f}"


def _build_markdown(report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# STWM Object Bias Diagnostic Matrix Report V1")
    lines.append("")
    lines.append(f"Generated: {report['generated_at']}")
    lines.append(f"Queue status dir: {report['queue_status_dir']}")
    lines.append(f"Run root: {report['run_root']}")
    lines.append("")

    if report.get("best_run_by_protocol_rule"):
        best = report["best_run_by_protocol_rule"]
        lines.append("## Best Candidate (Protocol Rule)")
        lines.append("")
        lines.append(f"- run: {best.get('run_name', '')}")
        lines.append(f"- query_localization_error: {_fmt(best.get('query_localization_error'))}")
        lines.append(f"- query_top1_acc: {_fmt(best.get('query_top1_acc'))}")
        lines.append(f"- future_trajectory_l1: {_fmt(best.get('future_trajectory_l1'))}")
        lines.append("")

    lines.append("## Runs")
    lines.append("")
    lines.append("| run | state | job_id | query_loc | query_top1 | future_traj_l1 | status_file | main_log |")
    lines.append("|---|---|---|---:|---:|---:|---|---|")
    for item in report.get("runs", []):
        metrics = item.get("protocol_metrics", {})
        if not isinstance(metrics, dict):
            metrics = {}
        lines.append(
            "| "
            f"{item.get('run_name', '')} | {item.get('state', '')} | {item.get('job_id', '')} | "
            f"{_fmt(_safe_float(metrics.get('query_localization_error')))} | "
            f"{_fmt(_safe_float(metrics.get('query_top1_acc')))} | "
            f"{_fmt(_safe_float(metrics.get('future_trajectory_l1')))} | "
            f"{item.get('status_path', '')} | {item.get('main_log', '')} |"
        )
    lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = build_parser().parse_args()
    status_dir = Path(args.queue_status_dir)
    run_root = Path(args.run_root) / f"seed_{int(args.seed)}"
    run_names = [x.strip() for x in str(args.run_names).split(",") if x.strip()]
    poll_seconds = max(10, int(args.poll_seconds))
    timeout_seconds = max(0, int(args.timeout_seconds))

    started = time.time()
    while True:
        infos = [_collect_one(run_root=run_root, status_dir=status_dir, run_name=run) for run in run_names]
        if _all_terminal(infos):
            break
        if timeout_seconds > 0 and (time.time() - started) >= timeout_seconds:
            break
        summary = ", ".join(f"{i['run_name']}={i.get('state', 'missing')}" for i in infos)
        print(f"[object-bias-diag-wait] {summary}", flush=True)
        time.sleep(poll_seconds)

    infos = [_collect_one(run_root=run_root, status_dir=status_dir, run_name=run) for run in run_names]

    finished = [i for i in infos if str(i.get("state", "")) == "done" and isinstance(i.get("protocol_metrics"), dict)]
    ranked = sorted(finished, key=_rank_key)

    best_summary: dict[str, Any] | None = None
    if ranked:
        bm = ranked[0].get("protocol_metrics", {})
        if not isinstance(bm, dict):
            bm = {}
        best_summary = {
            "run_name": ranked[0].get("run_name", ""),
            "query_localization_error": _safe_float(bm.get("query_localization_error")),
            "query_top1_acc": _safe_float(bm.get("query_top1_acc")),
            "future_trajectory_l1": _safe_float(bm.get("future_trajectory_l1")),
        }

    report: dict[str, Any] = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "queue_status_dir": str(status_dir),
        "run_root": str(run_root),
        "run_names": run_names,
        "runs": infos,
        "best_run_by_protocol_rule": best_summary,
    }

    out_report = Path(args.out_report)
    out_doc = Path(args.out_doc)
    out_report.parent.mkdir(parents=True, exist_ok=True)
    out_doc.parent.mkdir(parents=True, exist_ok=True)
    out_report.write_text(json.dumps(report, indent=2))
    out_doc.write_text(_build_markdown(report))

    print(str(out_report))
    print(str(out_doc))


if __name__ == "__main__":
    main()
