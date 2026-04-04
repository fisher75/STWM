from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import json
import time
from typing import Any


DEFAULT_RUN_NAMES = [
    "full_v4_2_seed42_fixed_nowarm_lambda1_rerun_v2",
    "full_v4_2_seed42_objbias_alpha050_replacement_v1",
    "wo_semantics_v4_2_seed42_control_v1",
    "wo_object_bias_v4_2_seed42_control_v1",
]


def build_parser() -> ArgumentParser:
    p = ArgumentParser(description="Watch Wave1 seed42 replacement clean matrix and write report")
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
            "stwm_v4_2_220m_protocol_frozen_frontend_replacement_seed42_v1"
        ),
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--run-names", default=",".join(DEFAULT_RUN_NAMES))
    p.add_argument("--poll-seconds", type=int, default=120)
    p.add_argument("--timeout-seconds", type=int, default=0)
    p.add_argument(
        "--out-report",
        default="/home/chen034/workspace/stwm/reports/stwm_replacement_clean_matrix_seed42_report_v1.json",
    )
    p.add_argument(
        "--out-doc",
        default="/home/chen034/workspace/stwm/docs/STWM_REPLACEMENT_CLEAN_MATRIX_SEED42_REPORT_V1.md",
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


def _count_train_rows_and_step(train_log: Path) -> tuple[int, int]:
    if not train_log.exists():
        return 0, -1
    rows = 0
    max_step = -1
    with train_log.open("r") as f:
        for line in f:
            text = line.strip()
            if not text:
                continue
            rows += 1
            try:
                obj = json.loads(text)
                step = int(obj.get("step", -1))
                if step > max_step:
                    max_step = step
            except Exception:
                continue
    return rows, max_step


def _collect(
    run_name: str,
    status_dir: Path,
    run_root: Path,
) -> dict[str, Any]:
    info: dict[str, Any] = {
        "run_name": run_name,
        "state": "missing",
    }
    status_path = _find_latest_status(status_dir, run_name)
    if status_path is not None:
        info["status_path"] = str(status_path)
        status = _load_json(status_path)
        if status is not None:
            info["state"] = str(status.get("state", "unknown")).lower()
            info["job_id"] = str(status.get("job_id", ""))
            info["main_log"] = str(status.get("main_log", ""))
            info["worker_session"] = str(status.get("worker_session", ""))
            info["gpu_index"] = str(status.get("gpu_index", ""))
            info["status_update_ts"] = str(status.get("update_ts", ""))

    out_dir = run_root / run_name
    train_log = out_dir / "train_log.jsonl"
    sidecar = out_dir / "checkpoints" / "best_protocol_main_selection.json"
    rows, max_step = _count_train_rows_and_step(train_log)

    info["output_dir"] = str(out_dir)
    info["train_log"] = str(train_log)
    info["train_log_rows"] = int(rows)
    info["max_step"] = int(max_step)
    info["selection_sidecar"] = str(sidecar)
    info["selection_sidecar_exists"] = bool(sidecar.exists())

    if sidecar.exists():
        sc = _load_json(sidecar)
        if sc is not None:
            eval_summary = Path(str(sc.get("eval_summary", "")))
            info["eval_summary"] = str(eval_summary)
            metrics = sc.get("metrics", {})
            if not isinstance(metrics, dict):
                metrics = {}
            info["selection_metrics"] = {
                "query_localization_error": _safe_float(metrics.get("query_localization_error")),
                "query_top1_acc": _safe_float(metrics.get("query_top1_acc")),
                "future_trajectory_l1": _safe_float(metrics.get("future_trajectory_l1")),
            }
    return info


def _terminal(infos: list[dict[str, Any]]) -> bool:
    for info in infos:
        if str(info.get("state", "")) not in {"done", "failed"}:
            return False
    return True


def _fmt(v: float | None) -> str:
    if v is None:
        return "na"
    return f"{float(v):.6f}"


def _write_markdown(report: dict[str, Any], out_doc: Path) -> None:
    lines: list[str] = []
    lines.append("# STWM Replacement Clean Matrix Seed42 Report V1")
    lines.append("")
    lines.append(f"Generated: {report['generated_at']}")
    lines.append(f"Queue status dir: {report['queue_status_dir']}")
    lines.append(f"Run root: {report['run_root']}")
    lines.append("")
    lines.append("## Monitoring Summary")
    lines.append("")
    lines.append(f"- all_entered_running: {report['monitoring']['all_entered_running']}")
    lines.append(f"- all_train_log_growth_observed: {report['monitoring']['all_train_log_growth_observed']}")
    lines.append(f"- all_sidecar_generated: {report['monitoring']['all_sidecar_generated']}")
    lines.append(f"- all_terminal(done/failed): {report['monitoring']['all_terminal']}")
    lines.append("")
    lines.append("## Runs")
    lines.append("")
    lines.append("| run | state | max_step | train_log_rows | growth_observed | sidecar | job_id | status_file | main_log |")
    lines.append("|---|---|---:|---:|---|---|---|---|---|")
    for item in report.get("runs", []):
        lines.append(
            "| "
            f"{item.get('run_name','')} | {item.get('state','')} | {item.get('max_step',-1)} | "
            f"{item.get('train_log_rows',0)} | {item.get('train_log_growth_observed',False)} | "
            f"{item.get('selection_sidecar_exists',False)} | {item.get('job_id','')} | "
            f"{item.get('status_path','')} | {item.get('main_log','')} |"
        )
    lines.append("")

    ranked = report.get("official_ranking", [])
    if isinstance(ranked, list) and ranked:
        lines.append("## Official Ranking Snapshot")
        lines.append("")
        lines.append("| rank | run | query_localization_error | query_top1_acc | future_trajectory_l1 |")
        lines.append("|---|---|---:|---:|---:|")
        for r in ranked:
            lines.append(
                "| "
                f"{r.get('rank','')} | {r.get('run_name','')} | {_fmt(r.get('query_localization_error'))} | "
                f"{_fmt(r.get('query_top1_acc'))} | {_fmt(r.get('future_trajectory_l1'))} |"
            )
        lines.append("")

    out_doc.parent.mkdir(parents=True, exist_ok=True)
    out_doc.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = build_parser().parse_args()
    status_dir = Path(args.queue_status_dir)
    run_root = Path(args.run_root) / f"seed_{int(args.seed)}"
    run_names = [x.strip() for x in str(args.run_names).split(",") if x.strip()]
    poll_seconds = max(15, int(args.poll_seconds))
    timeout_seconds = max(0, int(args.timeout_seconds))

    history: dict[str, dict[str, Any]] = {
        run: {
            "running_seen": False,
            "running_seen_ts": "",
            "train_log_growth_observed": False,
            "first_train_log_nonzero_ts": "",
            "prev_train_rows": 0,
            "sidecar_seen": False,
            "sidecar_seen_ts": "",
        }
        for run in run_names
    }

    started = time.time()
    while True:
        infos = [_collect(run, status_dir=status_dir, run_root=run_root) for run in run_names]

        now_ts = time.strftime("%Y-%m-%d %H:%M:%S")
        for info in infos:
            run = str(info.get("run_name", ""))
            h = history[run]

            state = str(info.get("state", ""))
            if state == "running" and not h["running_seen"]:
                h["running_seen"] = True
                h["running_seen_ts"] = now_ts

            cur_rows = int(info.get("train_log_rows", 0))
            prev_rows = int(h.get("prev_train_rows", 0))
            if cur_rows > prev_rows:
                h["train_log_growth_observed"] = True
            if cur_rows > 0 and not h["first_train_log_nonzero_ts"]:
                h["first_train_log_nonzero_ts"] = now_ts
            h["prev_train_rows"] = cur_rows

            if bool(info.get("selection_sidecar_exists", False)) and not h["sidecar_seen"]:
                h["sidecar_seen"] = True
                h["sidecar_seen_ts"] = now_ts

        all_terminal = _terminal(infos)
        if all_terminal:
            break
        if timeout_seconds > 0 and (time.time() - started) >= timeout_seconds:
            break

        summary = ", ".join(
            f"{i['run_name']}:{i.get('state','missing')}/step={i.get('max_step',-1)}/rows={i.get('train_log_rows',0)}"
            for i in infos
        )
        print(f"[replacement-seed42-watch] {summary}", flush=True)
        time.sleep(poll_seconds)

    infos = [_collect(run, status_dir=status_dir, run_root=run_root) for run in run_names]
    merged_runs: list[dict[str, Any]] = []
    for info in infos:
        run = str(info.get("run_name", ""))
        h = history[run]
        merged = dict(info)
        merged["running_seen"] = bool(h["running_seen"])
        merged["running_seen_ts"] = str(h["running_seen_ts"])
        merged["train_log_growth_observed"] = bool(h["train_log_growth_observed"])
        merged["first_train_log_nonzero_ts"] = str(h["first_train_log_nonzero_ts"])
        merged["sidecar_seen"] = bool(h["sidecar_seen"])
        merged["sidecar_seen_ts"] = str(h["sidecar_seen_ts"])
        merged_runs.append(merged)

    all_entered_running = all(bool(r.get("running_seen", False)) for r in merged_runs)
    all_growth = all(bool(r.get("train_log_growth_observed", False)) for r in merged_runs)
    all_sidecar = all(bool(r.get("selection_sidecar_exists", False)) for r in merged_runs)
    all_terminal = _terminal(merged_runs)

    # Ranking snapshot using official selection rule keys from sidecar metrics when available.
    ranking_candidates: list[dict[str, Any]] = []
    for r in merged_runs:
        sm = r.get("selection_metrics")
        if not isinstance(sm, dict):
            continue
        qloc = _safe_float(sm.get("query_localization_error"))
        qtop1 = _safe_float(sm.get("query_top1_acc"))
        ftraj = _safe_float(sm.get("future_trajectory_l1"))
        if qloc is None or qtop1 is None or ftraj is None:
            continue
        ranking_candidates.append(
            {
                "run_name": str(r.get("run_name", "")),
                "query_localization_error": float(qloc),
                "query_top1_acc": float(qtop1),
                "future_trajectory_l1": float(ftraj),
            }
        )
    ranking_candidates.sort(
        key=lambda x: (
            x["query_localization_error"],
            -x["query_top1_acc"],
            x["future_trajectory_l1"],
            x["run_name"],
        )
    )
    for i, item in enumerate(ranking_candidates, 1):
        item["rank"] = i

    first_running = sorted(
        [r for r in merged_runs if str(r.get("running_seen_ts", ""))],
        key=lambda x: str(x.get("running_seen_ts", "")),
    )
    first_training = sorted(
        [r for r in merged_runs if str(r.get("first_train_log_nonzero_ts", ""))],
        key=lambda x: str(x.get("first_train_log_nonzero_ts", "")),
    )

    report: dict[str, Any] = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "queue_status_dir": str(status_dir),
        "run_root": str(run_root),
        "run_names": run_names,
        "monitoring": {
            "all_entered_running": bool(all_entered_running),
            "all_train_log_growth_observed": bool(all_growth),
            "all_sidecar_generated": bool(all_sidecar),
            "all_terminal": bool(all_terminal),
        },
        "first_running_job": {
            "run_name": first_running[0].get("run_name", "") if first_running else "",
            "running_seen_ts": first_running[0].get("running_seen_ts", "") if first_running else "",
        },
        "first_training_job": {
            "run_name": first_training[0].get("run_name", "") if first_training else "",
            "first_train_log_nonzero_ts": first_training[0].get("first_train_log_nonzero_ts", "") if first_training else "",
        },
        "runs": merged_runs,
        "official_ranking": ranking_candidates,
    }

    out_report = Path(args.out_report)
    out_doc = Path(args.out_doc)
    out_report.parent.mkdir(parents=True, exist_ok=True)
    out_doc.parent.mkdir(parents=True, exist_ok=True)
    out_report.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    _write_markdown(report, out_doc)

    print(str(out_report))
    print(str(out_doc))


if __name__ == "__main__":
    main()
