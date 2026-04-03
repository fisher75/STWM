from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import json
import re
import time
from typing import Any


RUNS = [
    "full_v4_2_seed42_fixed_nowarm_lambda1_objdiag_v1",
    "wo_object_bias_v4_2_seed42_objdiag_v1",
    "full_v4_2_seed42_objbias_alpha025_objdiag_v1",
    "full_v4_2_seed42_objbias_alpha050_objdiag_v1",
    "full_v4_2_seed42_objbias_delayed200_objdiag_v1",
    "full_v4_2_seed42_objbias_gated_objdiag_v1",
]


def build_parser() -> ArgumentParser:
    p = ArgumentParser(description="Generate strict object-bias promotion decision")
    p.add_argument("--repo-root", default="/home/chen034/workspace/stwm")
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
    p.add_argument("--expected-steps", type=int, default=1200)
    p.add_argument("--baseline-run", default="full_v4_2_seed42_fixed_nowarm_lambda1_objdiag_v1")
    p.add_argument(
        "--watcher-report",
        default="/home/chen034/workspace/stwm/reports/stwm_object_bias_diag_matrix_v1.json",
    )
    p.add_argument(
        "--watcher-doc",
        default="/home/chen034/workspace/stwm/docs/STWM_OBJECT_BIAS_DIAG_MATRIX_REPORT_V1.md",
    )
    p.add_argument(
        "--out-report",
        default="/home/chen034/workspace/stwm/reports/stwm_object_bias_promotion_decision_v1.json",
    )
    p.add_argument(
        "--out-doc",
        default="/home/chen034/workspace/stwm/docs/STWM_OBJECT_BIAS_PROMOTION_DECISION_V1.md",
    )
    p.add_argument(
        "--out-plan-doc",
        default="/home/chen034/workspace/stwm/docs/STWM_NEXT_CLEAN_MATRIX_PLAN_V1.md",
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


def _latest_step(train_log: Path) -> int:
    if not train_log.exists():
        return 0
    last_line = ""
    with train_log.open("rb") as fh:
        fh.seek(0, 2)
        size = fh.tell()
        offset = min(size, 65536)
        fh.seek(size - offset)
        tail = fh.read().decode("utf-8", errors="ignore").splitlines()
        for line in reversed(tail):
            if line.strip():
                last_line = line
                break
    if not last_line:
        return 0
    try:
        obj = json.loads(last_line)
    except Exception:
        return 0
    step = obj.get("step")
    if isinstance(step, (int, float)):
        return int(step)
    return 0


def _extract_step_from_path(path_str: str) -> int | None:
    m = re.search(r"step_(\d+)", path_str)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def _pick_selected_step(sidecar: dict[str, Any]) -> int | None:
    for key in ("selected_best_step", "selected_step", "best_step", "step"):
        v = sidecar.get(key)
        if isinstance(v, (int, float)):
            return int(v)
    eval_summary = str(sidecar.get("eval_summary", ""))
    return _extract_step_from_path(eval_summary)


def _rank_key(item: dict[str, Any]) -> tuple[float, float, float, str]:
    q_loc = _safe_float(item.get("query_localization_error"))
    q_top1 = _safe_float(item.get("query_top1_acc"))
    fut_l1 = _safe_float(item.get("future_trajectory_l1"))
    if q_loc is None:
        q_loc = float("inf")
    if q_top1 is None:
        q_top1 = float("-inf")
    if fut_l1 is None:
        fut_l1 = float("inf")
    return (float(q_loc), float(-q_top1), float(fut_l1), str(item.get("run_name", "")))


def _fmt(v: float | int | None) -> str:
    if v is None:
        return "na"
    if isinstance(v, int):
        return str(v)
    return f"{float(v):.6f}"


def _build_plan_doc(result: dict[str, Any], plan_path: Path) -> None:
    recommendation = result["promotion_recommendation"]
    winner = recommendation.get("best_variant")
    promote = bool(recommendation.get("recommend_promotion", False))

    lines: list[str] = []
    lines.append("# STWM Next Clean Matrix Plan V1")
    lines.append("")
    lines.append(f"Generated: {result['generated_at']}")
    lines.append("")

    if not promote or not winner:
        lines.append("## Decision")
        lines.append("")
        lines.append("- No promotion candidate met strict replacement criteria.")
        lines.append("- Do not launch replacement clean matrix yet.")
        lines.append("")
        lines.append("## Action")
        lines.append("")
        lines.append("- Hold current full baseline for official clean matrix path.")
        lines.append("- Re-open promotion only after stronger evidence arrives.")
        lines.append("")
    else:
        lines.append("## Winner")
        lines.append("")
        lines.append(f"- Selected variant for replacement: {winner}")
        lines.append("")
        lines.append("## Seed42 Replacement Clean Matrix (Plan Only)")
        lines.append("")
        lines.append("- Keep protocol/evaluator rule unchanged.")
        lines.append("- Replace current full lane by winner variant in D1 clean matrix submission.")
        lines.append("- Suggested command (plan only, do not auto-run here):")
        lines.append(
            "  STWM_D1_VARIANT_OVERRIDE_FULL="
            + winner
            + " bash scripts/enqueue_stwm_protocol_v2_d1_matrix_v1.sh"
        )
        lines.append("")
        lines.append("## Seed123 Replication Clean Matrix (Plan Only)")
        lines.append("")
        lines.append("- Run same replacement matrix with seed=123 for replication.")
        lines.append("- Suggested command (plan only, do not auto-run here):")
        lines.append(
            "  STWM_D1_SEED=123 STWM_D1_VARIANT_OVERRIDE_FULL="
            + winner
            + " bash scripts/enqueue_stwm_protocol_v2_d1_matrix_v1.sh"
        )
        lines.append("")
        lines.append("## Launch Policy")
        lines.append("")
        lines.append("- No 1B launch in this stage.")
        lines.append("- Prefer seed42 first, then seed123 replication unless schedule allows dual launch.")
        lines.append("")

    plan_path.parent.mkdir(parents=True, exist_ok=True)
    plan_path.write_text("\n".join(lines) + "\n")


def _build_doc(result: dict[str, Any], doc_path: Path) -> None:
    ranking = result["ranking"]
    recommendation = result["promotion_recommendation"]

    lines: list[str] = []
    lines.append("# STWM Object Bias Promotion Decision V1")
    lines.append("")
    lines.append(f"Generated: {result['generated_at']}")
    lines.append(f"Seed: {result['seed']}")
    lines.append(f"Baseline: {result['baseline_run']}")
    lines.append("")

    lines.append("## Completion Validation")
    lines.append("")
    lines.append(f"- all_runs_done: {result['completion']['all_runs_done']}")
    lines.append(f"- all_runs_reached_expected_steps: {result['completion']['all_runs_reached_expected_steps']}")
    lines.append(f"- all_selection_sidecars_present: {result['completion']['all_selection_sidecars_present']}")
    lines.append(f"- watcher_matrix_report_present: {result['completion']['watcher_matrix_report_present']}")
    lines.append("")

    lines.append("## Official Ranking")
    lines.append("")
    lines.append("Rule: query_localization_error asc, query_top1_acc desc, future_trajectory_l1 asc")
    lines.append("")
    lines.append("| rank | run | selected_best_step | query_localization_error | query_top1_acc | future_trajectory_l1 | future_mask_iou |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for row in ranking:
        lines.append(
            "| "
            + f"{row['rank']} | {row['run_name']} | {_fmt(row.get('selected_best_step'))} | "
            + f"{_fmt(row.get('query_localization_error'))} | {_fmt(row.get('query_top1_acc'))} | "
            + f"{_fmt(row.get('future_trajectory_l1'))} | {_fmt(row.get('future_mask_iou'))} |"
        )
    lines.append("")

    lines.append("## Promotion Verdict")
    lines.append("")
    lines.append(f"- best_variant: {recommendation.get('best_variant')}")
    lines.append(f"- best_vs_baseline_significant: {recommendation.get('best_vs_baseline_significant')}")
    lines.append(f"- recommend_promotion: {recommendation.get('recommend_promotion')}")
    lines.append(f"- reason: {recommendation.get('reason')}")
    lines.append("")

    doc_path.parent.mkdir(parents=True, exist_ok=True)
    doc_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = build_parser().parse_args()

    repo_root = Path(args.repo_root)
    status_dir = Path(args.queue_status_dir)
    run_root = Path(args.run_root) / f"seed_{int(args.seed)}"
    watcher_report = Path(args.watcher_report)
    watcher_doc = Path(args.watcher_doc)

    rows: list[dict[str, Any]] = []
    completion_issues: list[dict[str, Any]] = []

    for run in RUNS:
        status_path = _find_latest_status(status_dir, run)
        status_obj = _load_json(status_path) if status_path else None
        state = str(status_obj.get("state", "missing")).lower() if status_obj else "missing"

        out_dir = run_root / run
        train_log = out_dir / "train_log.jsonl"
        final_step = _latest_step(train_log)

        sidecar_path = out_dir / "checkpoints" / "best_protocol_main_selection.json"
        sidecar_obj = _load_json(sidecar_path)
        eval_summary = ""
        metrics: dict[str, Any] = {}
        selected_best_step: int | None = None
        if sidecar_obj is not None:
            eval_summary = str(sidecar_obj.get("eval_summary", ""))
            metrics_obj = sidecar_obj.get("metrics", {})
            if isinstance(metrics_obj, dict):
                metrics = dict(metrics_obj)
            selected_best_step = _pick_selected_step(sidecar_obj)

        eval_metrics: dict[str, Any] = {}
        if eval_summary:
            eval_obj = _load_json(Path(eval_summary))
            if isinstance(eval_obj, dict) and isinstance(eval_obj.get("metrics"), dict):
                eval_metrics = dict(eval_obj["metrics"])

        query_loc = _safe_float(metrics.get("query_localization_error"))
        query_top1 = _safe_float(metrics.get("query_top1_acc"))
        fut_l1 = _safe_float(metrics.get("future_trajectory_l1"))
        fut_iou = _safe_float(eval_metrics.get("future_mask_iou"))

        row = {
            "run_name": run,
            "status_path": str(status_path) if status_path else "",
            "state": state,
            "job_id": str(status_obj.get("job_id", "")) if status_obj else "",
            "output_dir": str(out_dir),
            "train_log": str(train_log),
            "final_step": int(final_step),
            "expected_step": int(args.expected_steps),
            "selection_sidecar": str(sidecar_path),
            "selection_sidecar_exists": bool(sidecar_path.exists()),
            "selected_best_step": selected_best_step,
            "eval_summary": eval_summary,
            "query_localization_error": query_loc,
            "query_top1_acc": query_top1,
            "future_trajectory_l1": fut_l1,
            "future_mask_iou": fut_iou,
        }
        rows.append(row)

        if state != "done":
            completion_issues.append({"run": run, "issue": "state_not_done", "state": state})
        if final_step < int(args.expected_steps):
            completion_issues.append(
                {
                    "run": run,
                    "issue": "expected_step_not_reached",
                    "final_step": int(final_step),
                    "expected_step": int(args.expected_steps),
                }
            )
        if not sidecar_path.exists():
            completion_issues.append({"run": run, "issue": "selection_sidecar_missing", "path": str(sidecar_path)})

    all_runs_done = all(r["state"] == "done" for r in rows)
    all_steps_ok = all(int(r["final_step"]) >= int(args.expected_steps) for r in rows)
    all_sidecars = all(bool(r["selection_sidecar_exists"]) for r in rows)
    watcher_ready = watcher_report.exists() and watcher_doc.exists()

    ranked = sorted(rows, key=_rank_key)
    ranking: list[dict[str, Any]] = []
    for i, item in enumerate(ranked, start=1):
        row = dict(item)
        row["rank"] = i
        ranking.append(row)

    baseline = next((r for r in rows if r["run_name"] == args.baseline_run), None)
    best = ranking[0] if ranking else None

    recommend_promotion = False
    significant = False
    reason = "no_valid_winner"

    if best is not None and baseline is not None:
        b_q = _safe_float(best.get("query_localization_error"))
        b_t = _safe_float(best.get("query_top1_acc"))
        b_f = _safe_float(best.get("future_trajectory_l1"))
        base_q = _safe_float(baseline.get("query_localization_error"))
        base_t = _safe_float(baseline.get("query_top1_acc"))
        base_f = _safe_float(baseline.get("future_trajectory_l1"))

        if best["run_name"] == baseline["run_name"]:
            reason = "best_is_already_baseline"
        elif None in (b_q, b_t, b_f, base_q, base_t, base_f):
            reason = "insufficient_metric_values"
        else:
            assert b_q is not None and b_t is not None and b_f is not None
            assert base_q is not None and base_t is not None and base_f is not None
            all_three_better = (b_q < base_q) and (b_t > base_t) and (b_f < base_f)
            rel_q = (base_q - b_q) / max(base_q, 1e-12)
            rel_f = (base_f - b_f) / max(base_f, 1e-12)
            abs_top1 = b_t - base_t

            significant = bool(all_three_better and rel_q >= 0.05 and rel_f >= 0.05 and abs_top1 >= 0.005)
            recommend_promotion = bool(significant)
            if recommend_promotion:
                reason = "best_variant_significantly_better_than_baseline"
            else:
                reason = (
                    "best_variant_not_significant_vs_baseline"
                    + f"(all_three_better={all_three_better}, rel_q={rel_q:.4f}, rel_f={rel_f:.4f}, abs_top1={abs_top1:.4f})"
                )

    result: dict[str, Any] = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "repo_root": str(repo_root),
        "seed": int(args.seed),
        "baseline_run": str(args.baseline_run),
        "completion": {
            "all_runs_done": all_runs_done,
            "all_runs_reached_expected_steps": all_steps_ok,
            "all_selection_sidecars_present": all_sidecars,
            "watcher_matrix_report_present": watcher_ready,
            "watcher_report": str(watcher_report),
            "watcher_doc": str(watcher_doc),
            "issues": completion_issues,
        },
        "ranking_rule": {
            "primary_metric": "query_localization_error",
            "primary_direction": "lower_better",
            "tie_break_1": "query_top1_acc",
            "tie_break_1_direction": "higher_better",
            "tie_break_2": "future_trajectory_l1",
            "tie_break_2_direction": "lower_better",
        },
        "ranking": ranking,
        "promotion_recommendation": {
            "best_variant": best.get("run_name") if best else None,
            "best_vs_baseline_significant": significant,
            "recommend_promotion": recommend_promotion,
            "reason": reason,
            "baseline": baseline,
            "best": best,
        },
    }

    out_report = Path(args.out_report)
    out_doc = Path(args.out_doc)
    out_plan = Path(args.out_plan_doc)
    out_report.parent.mkdir(parents=True, exist_ok=True)
    out_doc.parent.mkdir(parents=True, exist_ok=True)
    out_plan.parent.mkdir(parents=True, exist_ok=True)

    out_report.write_text(json.dumps(result, indent=2))
    _build_doc(result, out_doc)
    _build_plan_doc(result, out_plan)

    print(str(out_report))
    print(str(out_doc))
    print(str(out_plan))


if __name__ == "__main__":
    main()
