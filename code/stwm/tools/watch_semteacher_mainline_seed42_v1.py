from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import csv
import json
import time
from typing import Any


def build_parser() -> ArgumentParser:
    p = ArgumentParser(description="Watch semteacher seed42 mainline and emit full+hard subset report")
    p.add_argument(
        "--submit-tsv",
        default=(
            "/home/chen034/workspace/stwm/reports/"
            "stwm_semteacher_mainline_seed42_submit_v1.tsv"
        ),
    )
    p.add_argument(
        "--queue-status-dir",
        default=(
            "/home/chen034/workspace/stwm/outputs/queue/"
            "stwm_protocol_v2_frontend_default_v1/d1_train/status"
        ),
    )
    p.add_argument(
        "--semantic-hard-manifest",
        default=(
            "/home/chen034/workspace/stwm/manifests/protocol_v2/"
            "semantic_hard_seed42_v1.json"
        ),
    )
    p.add_argument("--poll-seconds", type=int, default=120)
    p.add_argument("--timeout-seconds", type=int, default=0)
    p.add_argument(
        "--out-report",
        default=(
            "/home/chen034/workspace/stwm/reports/"
            "stwm_semteacher_mainline_seed42_report_v1.json"
        ),
    )
    p.add_argument(
        "--out-doc",
        default=(
            "/home/chen034/workspace/stwm/docs/"
            "STWM_SEMTEACHER_MAINLINE_SEED42_REPORT_V1.md"
        ),
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


def _read_submit_rows(tsv_path: Path) -> list[dict[str, str]]:
    if not tsv_path.exists():
        return []
    with tsv_path.open("r", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        rows = [dict(r) for r in reader]
    return rows


def _collect_one(row: dict[str, str], status_dir: Path) -> dict[str, Any]:
    run_name = str(row.get("run_name", "")).strip()
    unique_change_point = str(row.get("unique_change_point", "")).strip()
    output_dir = Path(str(row.get("output_dir", "")).strip())

    info: dict[str, Any] = {
        "run_name": run_name,
        "unique_change_point": unique_change_point,
        "state": "missing",
        "output_dir": str(output_dir),
    }

    status_path = None
    tsv_status = str(row.get("status_file", "")).strip()
    if tsv_status:
        p = Path(tsv_status)
        if p.exists():
            status_path = p
    if status_path is None:
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

    train_log = output_dir / "train_log.jsonl"
    sidecar = output_dir / "checkpoints" / "best_protocol_main_selection.json"
    rows, max_step = _count_train_rows_and_step(train_log)

    info["train_log"] = str(train_log)
    info["train_log_rows"] = int(rows)
    info["max_step"] = int(max_step)
    info["selection_sidecar"] = str(sidecar)
    info["selection_sidecar_exists"] = bool(sidecar.exists())

    if sidecar.exists():
        sc = _load_json(sidecar)
        if sc is not None:
            metrics = sc.get("metrics", {})
            if not isinstance(metrics, dict):
                metrics = {}
            info["selection_metrics"] = {
                "query_localization_error": _safe_float(metrics.get("query_localization_error")),
                "query_top1_acc": _safe_float(metrics.get("query_top1_acc")),
                "future_trajectory_l1": _safe_float(metrics.get("future_trajectory_l1")),
                "future_mask_iou": _safe_float(metrics.get("future_mask_iou")),
                "identity_consistency": _safe_float(metrics.get("identity_consistency")),
                "identity_switch_rate": _safe_float(metrics.get("identity_switch_rate")),
            }
            info["eval_summary"] = str(sc.get("eval_summary", ""))
            info["selection_action"] = str(sc.get("action", ""))
            info["selection_improved"] = bool(sc.get("improved", False))
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


def _load_hard_clip_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    payload = json.loads(path.read_text())
    out: set[str] = set()
    if isinstance(payload, list):
        for item in payload:
            if isinstance(item, dict) and str(item.get("clip_id", "")).strip():
                out.add(str(item.get("clip_id", "")).strip())
            elif isinstance(item, str) and item.strip():
                out.add(item.strip())
    elif isinstance(payload, dict):
        arr = payload.get("selected_clip_ids")
        if isinstance(arr, list):
            for item in arr:
                if str(item).strip():
                    out.add(str(item).strip())
    return out


def _collect_hard_subset_metrics(eval_summary_path: Path, hard_clip_ids: set[str]) -> dict[str, float] | None:
    if not eval_summary_path.exists():
        return None
    payload = _load_json(eval_summary_path)
    if payload is None:
        return None
    rows = payload.get("per_clip", [])
    if not isinstance(rows, list):
        return None

    filtered = [r for r in rows if isinstance(r, dict) and str(r.get("clip_id", "")) in hard_clip_ids]
    if not filtered:
        return None

    def avg(key: str) -> float:
        vals = [_safe_float(r.get(key)) for r in filtered]
        vals = [float(v) for v in vals if v is not None]
        if not vals:
            return 0.0
        return float(sum(vals) / max(1, len(vals)))

    return {
        "query_localization_error": avg("query_localization_error"),
        "query_top1_acc": avg("query_top1_acc"),
        "future_trajectory_l1": avg("future_trajectory_l1"),
        "num_rows": int(len(filtered)),
        "num_unique_clips": int(len({str(r.get('clip_id', '')) for r in filtered})),
    }


def _write_markdown(report: dict[str, Any], out_doc: Path) -> None:
    lines: list[str] = []
    lines.append("# STWM Semteacher Mainline Seed42 Report V1")
    lines.append("")
    lines.append(f"Generated: {report['generated_at']}")
    lines.append(f"Submit TSV: {report['submit_tsv']}")
    lines.append(f"Queue status dir: {report['queue_status_dir']}")
    lines.append(f"Semantic hard manifest: {report.get('semantic_hard_manifest', '')}")
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
    lines.append("| run | unique_change_point | state | max_step | train_log_rows | sidecar | job_id | status_file | main_log |")
    lines.append("|---|---|---|---:|---:|---|---|---|---|")
    for item in report.get("runs", []):
        lines.append(
            "| "
            f"{item.get('run_name','')} | {item.get('unique_change_point','')} | {item.get('state','')} | "
            f"{item.get('max_step',-1)} | {item.get('train_log_rows',0)} | "
            f"{item.get('selection_sidecar_exists',False)} | {item.get('job_id','')} | "
            f"{item.get('status_path','')} | {item.get('main_log','')} |"
        )
    lines.append("")

    full_ranked = report.get("official_ranking_full", [])
    if isinstance(full_ranked, list) and full_ranked:
        lines.append("## Full-Set Official Ranking")
        lines.append("")
        lines.append("| rank | run | query_localization_error | query_top1_acc | future_trajectory_l1 |")
        lines.append("|---|---|---:|---:|---:|")
        for r in full_ranked:
            lines.append(
                "| "
                f"{r.get('rank','')} | {r.get('run_name','')} | {_fmt(r.get('query_localization_error'))} | "
                f"{_fmt(r.get('query_top1_acc'))} | {_fmt(r.get('future_trajectory_l1'))} |"
            )
        lines.append("")

    hard_ranked = report.get("official_ranking_semantic_hard", [])
    if isinstance(hard_ranked, list) and hard_ranked:
        lines.append("## Semantic-Hard Ranking")
        lines.append("")
        lines.append("| rank | run | query_localization_error | query_top1_acc | future_trajectory_l1 | hard_rows | hard_unique_clips |")
        lines.append("|---|---|---:|---:|---:|---:|---:|")
        for r in hard_ranked:
            lines.append(
                "| "
                f"{r.get('rank','')} | {r.get('run_name','')} | {_fmt(r.get('query_localization_error'))} | "
                f"{_fmt(r.get('query_top1_acc'))} | {_fmt(r.get('future_trajectory_l1'))} | "
                f"{r.get('hard_rows',0)} | {r.get('hard_unique_clips',0)} |"
            )
        lines.append("")

    verdict = report.get("hard_subset_verdict", {})
    lines.append("## Verdict")
    lines.append("")
    lines.append(f"- hard_subset_cleanly_established: {verdict.get('hard_subset_cleanly_established', False)}")
    lines.append(f"- reason: {verdict.get('reason', '')}")
    lines.append(f"- best_full: {report.get('best_run_by_official_rule_full', '')}")
    lines.append(f"- best_semantic_hard: {report.get('best_run_by_official_rule_semantic_hard', '')}")
    lines.append("")

    out_doc.parent.mkdir(parents=True, exist_ok=True)
    out_doc.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = build_parser().parse_args()
    submit_tsv = Path(args.submit_tsv)
    status_dir = Path(args.queue_status_dir)
    hard_manifest = Path(args.semantic_hard_manifest)
    poll_seconds = max(15, int(args.poll_seconds))
    timeout_seconds = max(0, int(args.timeout_seconds))

    submit_rows = _read_submit_rows(submit_tsv)
    if not submit_rows:
        raise SystemExit(f"submit tsv not found or empty: {submit_tsv}")

    run_names = [str(r.get("run_name", "")).strip() for r in submit_rows if str(r.get("run_name", "")).strip()]
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
        infos = [_collect_one(r, status_dir=status_dir) for r in submit_rows]

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
        print(f"[semteacher-seed42-watch] {summary}", flush=True)
        time.sleep(poll_seconds)

    infos = [_collect_one(r, status_dir=status_dir) for r in submit_rows]
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

    full_ranking: list[dict[str, Any]] = []
    for r in merged_runs:
        sm = r.get("selection_metrics")
        if not isinstance(sm, dict):
            continue
        qloc = _safe_float(sm.get("query_localization_error"))
        qtop1 = _safe_float(sm.get("query_top1_acc"))
        ftraj = _safe_float(sm.get("future_trajectory_l1"))
        if qloc is None or qtop1 is None or ftraj is None:
            continue
        full_ranking.append(
            {
                "run_name": str(r.get("run_name", "")),
                "query_localization_error": float(qloc),
                "query_top1_acc": float(qtop1),
                "future_trajectory_l1": float(ftraj),
            }
        )
    full_ranking.sort(
        key=lambda x: (
            x["query_localization_error"],
            -x["query_top1_acc"],
            x["future_trajectory_l1"],
            x["run_name"],
        )
    )
    for i, item in enumerate(full_ranking, 1):
        item["rank"] = i

    hard_clip_ids = _load_hard_clip_ids(hard_manifest)
    hard_ranking: list[dict[str, Any]] = []
    for r in merged_runs:
        eval_summary_raw = str(r.get("eval_summary", "")).strip()
        if not eval_summary_raw:
            continue
        hard_metrics = _collect_hard_subset_metrics(Path(eval_summary_raw), hard_clip_ids)
        if hard_metrics is None:
            continue
        hard_ranking.append(
            {
                "run_name": str(r.get("run_name", "")),
                "query_localization_error": float(hard_metrics["query_localization_error"]),
                "query_top1_acc": float(hard_metrics["query_top1_acc"]),
                "future_trajectory_l1": float(hard_metrics["future_trajectory_l1"]),
                "hard_rows": int(hard_metrics["num_rows"]),
                "hard_unique_clips": int(hard_metrics["num_unique_clips"]),
            }
        )
    hard_ranking.sort(
        key=lambda x: (
            x["query_localization_error"],
            -x["query_top1_acc"],
            x["future_trajectory_l1"],
            x["run_name"],
        )
    )
    for i, item in enumerate(hard_ranking, 1):
        item["rank"] = i

    control_run = "trace_baseline_seed42_semteacher_control_v1"
    hard_top = hard_ranking[0]["run_name"] if hard_ranking else ""
    hard_subset_cleanly_established = bool(
        hard_ranking
        and hard_top != control_run
        and all_terminal
    )
    if not hard_ranking:
        verdict_reason = "semantic hard ranking unavailable (missing eval summaries or subset IDs)"
    elif not all_terminal:
        verdict_reason = "jobs not all terminal yet"
    elif hard_top == control_run:
        verdict_reason = "control still best on semantic hard subset"
    else:
        verdict_reason = "challenge run outranks control on semantic hard subset under official rule"

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
        "submit_tsv": str(submit_tsv),
        "queue_status_dir": str(status_dir),
        "semantic_hard_manifest": str(hard_manifest),
        "semantic_hard_clip_count": int(len(hard_clip_ids)),
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
        "official_ranking_full": full_ranking,
        "official_ranking_semantic_hard": hard_ranking,
        "best_run_by_official_rule_full": full_ranking[0]["run_name"] if full_ranking else "",
        "best_run_by_official_rule_semantic_hard": hard_ranking[0]["run_name"] if hard_ranking else "",
        "hard_subset_verdict": {
            "hard_subset_cleanly_established": bool(hard_subset_cleanly_established),
            "reason": verdict_reason,
            "control_run": control_run,
        },
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
