from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import json
import statistics
from typing import Any


RUNS = [
    "full_v4_2_seed42_fixed_nowarm_lambda1",
    "full_v4_2_seed42_fixed_warmup_lambda1",
    "wo_semantics_v4_2_seed42",
    "wo_object_bias_v4_2_seed42",
]

HIGHER_BETTER = {
    "query_top1_acc": True,
    "future_mask_iou": True,
    "identity_consistency": True,
}

LOWER_BETTER = {
    "query_localization_error": True,
    "future_trajectory_l1": True,
    "identity_switch_rate": True,
}


def build_parser() -> ArgumentParser:
    p = ArgumentParser(description="Generate final D1 clean matrix report")
    p.add_argument("--repo-root", default="/home/chen034/workspace/stwm")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--queue-status-dir",
        default="/home/chen034/workspace/stwm/outputs/queue/stwm_protocol_v2_frontend_default_v1/d1_train/status",
    )
    p.add_argument(
        "--output-root",
        default="/home/chen034/workspace/stwm/outputs/training/stwm_v4_2_220m_protocol_frozen_frontend_default_v1",
    )
    p.add_argument(
        "--grad-report-dir",
        default="/home/chen034/workspace/stwm/reports/frontend_default_v1",
    )
    p.add_argument(
        "--confirm-report",
        default="/home/chen034/workspace/stwm/reports/stwm_frontend_cache_confirm_v1.json",
    )
    p.add_argument(
        "--out-report",
        default="/home/chen034/workspace/stwm/reports/stwm_d1_clean_matrix_final_report_v1.json",
    )
    p.add_argument(
        "--out-doc",
        default="/home/chen034/workspace/stwm/docs/STWM_D1_CLEAN_MATRIX_FINAL_REPORT_V1.md",
    )
    return p


def _safe_float(v: Any) -> float | None:
    try:
        x = float(v)
        if x != x:  # NaN
            return None
        if x == float("inf") or x == float("-inf"):
            return None
        return x
    except Exception:
        return None


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


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text().splitlines():
        text = line.strip()
        if not text:
            continue
        try:
            row = json.loads(text)
        except Exception:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _find_latest_status(status_dir: Path, run_name: str) -> Path | None:
    candidates = sorted(status_dir.glob(f"*_{run_name}.status.json"))
    if not candidates:
        return None
    return candidates[-1]


def _find_selection_sidecar(checkpoint_dir: Path) -> Path | None:
    preferred = checkpoint_dir / "best_protocol_main_selection.json"
    if preferred.exists():
        return preferred

    ranked = []
    for p in checkpoint_dir.glob("*selection*.json"):
        name = p.name.lower()
        score = 0
        if "protocol" in name:
            score += 3
        if "main" in name:
            score += 2
        if "best" in name:
            score += 1
        ranked.append((score, name, p))
    if not ranked:
        return None
    ranked.sort(reverse=True)
    return ranked[0][2]


def _stats_from_rows(rows: list[dict[str, Any]], key: str) -> dict[str, float | int]:
    vals: list[float] = []
    for row in rows:
        if key not in row:
            continue
        fv = _safe_float(row.get(key))
        if fv is None:
            continue
        vals.append(fv)

    if not vals:
        return {
            "count": 0,
            "first": 0.0,
            "median": 0.0,
            "last": 0.0,
            "min": 0.0,
            "max": 0.0,
        }
    return {
        "count": len(vals),
        "first": float(vals[0]),
        "median": float(statistics.median(vals)),
        "last": float(vals[-1]),
        "min": float(min(vals)),
        "max": float(max(vals)),
    }


def _neg_ratio(rows: list[dict[str, Any]], key: str) -> float:
    vals: list[float] = []
    for row in rows:
        fv = _safe_float(row.get(key))
        if fv is not None:
            vals.append(fv)
    if not vals:
        return 1.0
    neg = sum(1 for v in vals if v < 0)
    return float(neg) / float(len(vals))


def _flatten_numeric(d: Any, out: dict[str, float], prefix: str = "") -> None:
    if isinstance(d, dict):
        for k, v in d.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            _flatten_numeric(v, out, key)
    elif isinstance(d, list):
        for i, v in enumerate(d):
            key = f"{prefix}[{i}]"
            _flatten_numeric(v, out, key)
    else:
        fv = _safe_float(d)
        if fv is not None and prefix:
            out[prefix] = fv


def _extract_selected_record(sidecar: dict[str, Any]) -> dict[str, Any]:
    if isinstance(sidecar.get("selected_record"), dict):
        return dict(sidecar["selected_record"])

    selected_step = None
    for k in ("selected_best_step", "selected_step", "best_step", "step"):
        if k in sidecar:
            selected_step = sidecar.get(k)
            break

    records = sidecar.get("records")
    if isinstance(records, list):
        if selected_step is not None:
            for rec in records:
                if isinstance(rec, dict) and str(rec.get("step")) == str(selected_step):
                    return dict(rec)
        for rec in records:
            if isinstance(rec, dict):
                return dict(rec)

    return dict(sidecar)


def _pick_metric(rec: dict[str, Any], sidecar: dict[str, Any], candidates: list[str]) -> float | None:
    for key in candidates:
        if key in rec:
            fv = _safe_float(rec.get(key))
            if fv is not None:
                return fv
    for key in candidates:
        if key in sidecar:
            fv = _safe_float(sidecar.get(key))
            if fv is not None:
                return fv

    flat_rec: dict[str, float] = {}
    _flatten_numeric(rec, flat_rec)
    flat_side: dict[str, float] = {}
    _flatten_numeric(sidecar, flat_side)

    for key in candidates:
        key_l = key.lower()
        for fk, fv in flat_rec.items():
            if key_l in fk.lower():
                return fv
        for fk, fv in flat_side.items():
            if key_l in fk.lower():
                return fv
    return None


def _selection_summary(sidecar: dict[str, Any]) -> str:
    for k in (
        "selection_rule",
        "selection_basis",
        "selection_metric",
        "selection_objective",
        "best_selection_reason",
        "reason",
    ):
        if k in sidecar:
            return str(sidecar.get(k))

    keys = sorted(sidecar.keys())
    return "keys=" + ",".join(keys[:12])


def _pct(vals: list[float], q: float) -> float:
    if not vals:
        return 0.0
    if len(vals) == 1:
        return float(vals[0])
    s = sorted(vals)
    pos = (len(s) - 1) * q / 100.0
    lo = int(pos)
    hi = min(lo + 1, len(s) - 1)
    w = pos - lo
    return float(s[lo] * (1.0 - w) + s[hi] * w)


def _efficiency_summary(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    step_vals = [_safe_float(r.get("step_time_s")) for r in rows]
    data_vals = [_safe_float(r.get("data_time_s")) for r in rows]
    wait_vals = [_safe_float(r.get("data_wait_ratio")) for r in rows]

    step_vals = [v for v in step_vals if v is not None]
    data_vals = [v for v in data_vals if v is not None]
    wait_vals = [v for v in wait_vals if v is not None]

    recent = rows[-500:] if len(rows) > 500 else rows
    recent_step = [_safe_float(r.get("step_time_s")) for r in recent]
    recent_wait = [_safe_float(r.get("data_wait_ratio")) for r in recent]
    recent_step = [v for v in recent_step if v is not None]
    recent_wait = [v for v in recent_wait if v is not None]

    def _mean(xs: list[float]) -> float:
        return float(sum(xs) / max(1, len(xs)))

    return {
        "rows": int(len(rows)),
        "full_mean_step_time_s": _mean(step_vals),
        "full_mean_data_time_s": _mean(data_vals),
        "full_mean_data_wait_ratio": _mean(wait_vals),
        "recent500_mean_step_time_s": _mean(recent_step),
        "recent500_mean_data_wait_ratio": _mean(recent_wait),
        "step_time_p50_s": _pct(step_vals, 50.0),
        "step_time_p95_s": _pct(step_vals, 95.0),
    }


def _metric_compare(a: float | None, b: float | None, higher_better: bool) -> str:
    if a is None or b is None:
        return "na"
    if abs(a - b) <= 1e-12:
        return "tie"
    if higher_better:
        return "a_better" if a > b else "b_better"
    return "a_better" if a < b else "b_better"


def _run_comparison(a: dict[str, Any], b: dict[str, Any], a_name: str, b_name: str) -> dict[str, Any]:
    metrics = [
        "query_localization_error",
        "query_top1_acc",
        "future_trajectory_l1",
        "future_mask_iou",
        "identity_consistency",
        "identity_switch_rate",
    ]
    details: dict[str, Any] = {}
    a_wins = 0
    b_wins = 0
    ties = 0
    for m in metrics:
        av = a.get(m)
        bv = b.get(m)
        hb = bool(HIGHER_BETTER.get(m, False))
        lb = bool(LOWER_BETTER.get(m, False))
        higher_better = hb and not lb
        cmpv = _metric_compare(av, bv, higher_better=higher_better)
        details[m] = {
            a_name: av,
            b_name: bv,
            "comparison": cmpv,
            "higher_better": higher_better,
        }
        if cmpv == "a_better":
            a_wins += 1
        elif cmpv == "b_better":
            b_wins += 1
        elif cmpv == "tie":
            ties += 1

    return {
        "pair": f"{a_name} vs {b_name}",
        "a_wins": int(a_wins),
        "b_wins": int(b_wins),
        "ties": int(ties),
        "details": details,
    }


def _format_stats_block(name: str, stats: dict[str, float | int]) -> str:
    return (
        f"- {name}: first={stats['first']:.9g}, median={stats['median']:.9g}, "
        f"last={stats['last']:.9g}, min={stats['min']:.9g}, max={stats['max']:.9g}, n={int(stats['count'])}"
    )


def main() -> None:
    args = build_parser().parse_args()

    repo_root = Path(args.repo_root)
    status_dir = Path(args.queue_status_dir)
    output_root = Path(args.output_root)
    run_root = output_root / f"seed_{int(args.seed)}"
    grad_dir = Path(args.grad_report_dir)
    confirm_path = Path(args.confirm_report)
    out_report = Path(args.out_report)
    out_doc = Path(args.out_doc)

    result: dict[str, Any] = {
        "generated_at": __import__("time").strftime("%Y-%m-%d %H:%M:%S"),
        "repo_root": str(repo_root),
        "seed": int(args.seed),
        "queue_status_dir": str(status_dir),
        "run_root": str(run_root),
        "runs": {},
        "validation_ok": False,
    }

    anomalies: list[dict[str, Any]] = []

    nowarm_grad = grad_dir / "stwm_v4_2_gradient_audit_220m_seed42_full_v4_2_seed42_fixed_nowarm_lambda1_frontend_default_v1.json"
    warmup_grad = grad_dir / "stwm_v4_2_gradient_audit_220m_seed42_full_v4_2_seed42_fixed_warmup_lambda1_frontend_default_v1.json"

    if not nowarm_grad.exists():
        anomalies.append({"scope": "gradient_audit", "file": str(nowarm_grad), "issue": "missing"})
    if not warmup_grad.exists():
        anomalies.append({"scope": "gradient_audit", "file": str(warmup_grad), "issue": "missing"})

    for run in RUNS:
        run_info: dict[str, Any] = {"run_name": run}
        status_path = _find_latest_status(status_dir, run)
        run_info["status_path"] = str(status_path) if status_path is not None else ""

        status = _load_json(status_path) if status_path is not None else None
        if status is None:
            anomalies.append({"run": run, "issue": "status_missing_or_invalid", "path": str(status_path)})
            result["runs"][run] = run_info
            continue

        run_info["job_state"] = str(status.get("state", ""))
        run_info["job_id"] = str(status.get("job_id", ""))
        run_info["main_log"] = str(status.get("main_log", ""))

        if str(status.get("state", "")).lower() != "done":
            anomalies.append(
                {
                    "run": run,
                    "issue": "job_not_done",
                    "state": str(status.get("state", "")),
                    "status_path": str(status_path),
                }
            )

        out_dir = run_root / run
        run_info["output_dir"] = str(out_dir)
        train_log = out_dir / "train_log.jsonl"
        run_info["train_log"] = str(train_log)

        rows = _load_jsonl(train_log)
        steps = [int(r.get("step", 0)) for r in rows if isinstance(r.get("step", None), (int, float))]
        final_step = max(steps) if steps else 0
        run_info["train_log_rows"] = int(len(rows))
        run_info["final_step"] = int(final_step)
        if final_step < 2000:
            anomalies.append(
                {
                    "run": run,
                    "issue": "step_not_reached",
                    "final_step": int(final_step),
                    "expected": 2000,
                    "train_log": str(train_log),
                }
            )

        ckpt_dir = out_dir / "checkpoints"
        latest_pt = ckpt_dir / "latest.pt"
        best_main = ckpt_dir / "best_protocol_main.pt"
        selection_sidecar = _find_selection_sidecar(ckpt_dir)
        run_info["checkpoint_dir"] = str(ckpt_dir)
        run_info["latest_pt"] = str(latest_pt)
        run_info["best_protocol_main_pt"] = str(best_main)
        run_info["best_protocol_main_selection"] = str(selection_sidecar) if selection_sidecar else ""

        if not latest_pt.exists():
            anomalies.append({"run": run, "issue": "latest_pt_missing", "path": str(latest_pt)})
        if not best_main.exists():
            anomalies.append({"run": run, "issue": "best_protocol_main_missing", "path": str(best_main)})
        if selection_sidecar is None or not Path(selection_sidecar).exists():
            anomalies.append(
                {
                    "run": run,
                    "issue": "selection_sidecar_missing",
                    "checkpoint_dir": str(ckpt_dir),
                }
            )

        result["runs"][run] = run_info

    if anomalies:
        result["validation_ok"] = False
        result["stage"] = "A_validation_failed"
        result["anomalies"] = anomalies

        out_report.parent.mkdir(parents=True, exist_ok=True)
        out_report.write_text(json.dumps(result, indent=2, ensure_ascii=False))

        lines = [
            "# STWM D1 Clean Matrix Final Report V1",
            "",
            "Status: BLOCKED (validation failed)",
            "",
            "## A. Completion Validation",
            "",
            "At least one run is incomplete or missing required artifacts, so downstream analysis is stopped.",
            "",
            "## Anomalies",
            "",
        ]
        for item in anomalies:
            lines.append(f"- {json.dumps(item, ensure_ascii=False)}")
        out_doc.parent.mkdir(parents=True, exist_ok=True)
        out_doc.write_text("\n".join(lines) + "\n")
        raise SystemExit(2)

    result["validation_ok"] = True
    result["stage"] = "A_validation_passed"

    # B. Gradient audit final report
    nowarm_payload = _load_json(nowarm_grad) or {}
    warmup_payload = _load_json(warmup_grad) or {}
    nowarm_rows = sorted(nowarm_payload.get("rows", []), key=lambda x: int(x.get("step", 0))) if isinstance(nowarm_payload.get("rows", []), list) else []
    warmup_rows = sorted(warmup_payload.get("rows", []), key=lambda x: int(x.get("step", 0))) if isinstance(warmup_payload.get("rows", []), list) else []

    grad_keys = ["g_traj_norm", "g_sem_norm", "cos_sem_traj"]
    qpath_keys = ["qpath_g_query_norm", "qpath_cos_sem_query", "qpath_cos_traj_query"]

    grad_summary = {
        "nowarm": {k: _stats_from_rows(nowarm_rows, k) for k in grad_keys},
        "warmup": {k: _stats_from_rows(warmup_rows, k) for k in grad_keys},
        "nowarm_rows": int(len(nowarm_rows)),
        "warmup_rows": int(len(warmup_rows)),
    }

    qpath_present = any(k in row for row in nowarm_rows + warmup_rows for k in qpath_keys)
    qpath_summary = {
        "present": bool(qpath_present),
        "nowarm": {k: _stats_from_rows(nowarm_rows, k) for k in qpath_keys},
        "warmup": {k: _stats_from_rows(warmup_rows, k) for k in qpath_keys},
    }

    nowarm_neg = _neg_ratio(nowarm_rows, "cos_sem_traj")
    warmup_neg = _neg_ratio(warmup_rows, "cos_sem_traj")
    nowarm_med = float(grad_summary["nowarm"]["cos_sem_traj"]["median"])
    warmup_med = float(grad_summary["warmup"]["cos_sem_traj"]["median"])
    warmup_sustained_mitigation = bool((warmup_neg + 0.10) <= nowarm_neg and warmup_med > nowarm_med)

    query_nonzero = False
    query_reason = ""
    if qpath_present:
        q_nowarm_med = float(qpath_summary["nowarm"]["qpath_g_query_norm"]["median"])
        q_warmup_med = float(qpath_summary["warmup"]["qpath_g_query_norm"]["median"])
        query_nonzero = (q_nowarm_med > 0.0) and (q_warmup_med > 0.0)
        if query_nonzero:
            query_reason = "qpath_g_query_norm median is non-zero for both nowarm and warmup"
        else:
            query_reason = "qpath fields exist but at least one run still has zero median qpath_g_query_norm"
    else:
        query_reason = "qpath fields not found in clean-matrix gradient audit rows"

    result["gradient"] = {
        "nowarm_report": str(nowarm_grad),
        "warmup_report": str(warmup_grad),
        "summary": grad_summary,
        "qpath": qpath_summary,
        "warmup_sustained_conflict_mitigation": bool(warmup_sustained_mitigation),
        "warmup_vs_nowarm_neg_ratio": {
            "nowarm": float(nowarm_neg),
            "warmup": float(warmup_neg),
        },
        "query_gradient_nonzero_after_fix": bool(query_nonzero),
        "query_gradient_reason": query_reason,
    }

    # C. Official selection scorecard
    scorecard: dict[str, Any] = {}
    for run in RUNS:
        sidecar_path = Path(str(result["runs"][run]["best_protocol_main_selection"]))
        sidecar = _load_json(sidecar_path) or {}
        rec = _extract_selected_record(sidecar)

        entry = {
            "selection_sidecar": str(sidecar_path),
            "selected_best_step": _pick_metric(rec, sidecar, ["selected_best_step", "selected_step", "best_step", "step"]),
            "query_localization_error": _pick_metric(rec, sidecar, ["query_localization_error"]),
            "query_top1_acc": _pick_metric(rec, sidecar, ["query_top1_acc", "query_top1_accuracy"]),
            "future_trajectory_l1": _pick_metric(rec, sidecar, ["future_trajectory_l1", "future_traj_l1"]),
            "future_mask_iou": _pick_metric(rec, sidecar, ["future_mask_iou", "mask_iou_future"]),
            "identity_consistency": _pick_metric(rec, sidecar, ["identity_consistency"]),
            "identity_switch_rate": _pick_metric(rec, sidecar, ["identity_switch_rate", "switch_rate"]),
            "selection_basis_summary": _selection_summary(sidecar),
        }
        scorecard[run] = entry

    cmp_full_vs_wo_sem = _run_comparison(
        scorecard["full_v4_2_seed42_fixed_nowarm_lambda1"],
        scorecard["wo_semantics_v4_2_seed42"],
        "full_nowarm",
        "wo_semantics",
    )
    cmp_full_vs_wo_obj = _run_comparison(
        scorecard["full_v4_2_seed42_fixed_nowarm_lambda1"],
        scorecard["wo_object_bias_v4_2_seed42"],
        "full_nowarm",
        "wo_object_bias",
    )
    cmp_warmup_vs_nowarm = _run_comparison(
        scorecard["full_v4_2_seed42_fixed_warmup_lambda1"],
        scorecard["full_v4_2_seed42_fixed_nowarm_lambda1"],
        "full_warmup",
        "full_nowarm",
    )

    full_beats_ablations = bool(cmp_full_vs_wo_sem["a_wins"] >= cmp_full_vs_wo_sem["b_wins"] and cmp_full_vs_wo_obj["a_wins"] >= cmp_full_vs_wo_obj["b_wins"])
    warmup_net_gain = bool(cmp_warmup_vs_nowarm["a_wins"] > cmp_warmup_vs_nowarm["b_wins"])

    result["selection_scorecard"] = {
        "runs": scorecard,
        "comparisons": {
            "full_nowarm_vs_wo_semantics": cmp_full_vs_wo_sem,
            "full_nowarm_vs_wo_object_bias": cmp_full_vs_wo_obj,
            "full_warmup_vs_full_nowarm": cmp_warmup_vs_nowarm,
        },
        "full_stably_beats_wo_semantics_and_wo_object_bias": bool(full_beats_ablations),
        "warmup_net_protocol_best_gain": bool(warmup_net_gain),
    }

    # D. Frontend-cache efficiency scorecard
    eff: dict[str, Any] = {}
    for run in RUNS:
        train_log = Path(str(result["runs"][run]["train_log"]))
        rows = _load_jsonl(train_log)
        eff[run] = _efficiency_summary(rows)

    full_runs = [
        eff["full_v4_2_seed42_fixed_nowarm_lambda1"],
        eff["full_v4_2_seed42_fixed_warmup_lambda1"],
    ]

    def _mean_key(items: list[dict[str, Any]], key: str) -> float:
        vals = [_safe_float(x.get(key)) for x in items]
        vals = [v for v in vals if v is not None]
        return float(sum(vals) / max(1, len(vals)))

    full_avg = {
        "full_mean_step_time_s": _mean_key(full_runs, "full_mean_step_time_s"),
        "full_mean_data_time_s": _mean_key(full_runs, "full_mean_data_time_s"),
        "full_mean_data_wait_ratio": _mean_key(full_runs, "full_mean_data_wait_ratio"),
        "recent500_mean_step_time_s": _mean_key(full_runs, "recent500_mean_step_time_s"),
        "recent500_mean_data_wait_ratio": _mean_key(full_runs, "recent500_mean_data_wait_ratio"),
        "step_time_p50_s": _mean_key(full_runs, "step_time_p50_s"),
        "step_time_p95_s": _mean_key(full_runs, "step_time_p95_s"),
    }

    confirm = _load_json(confirm_path) or {}
    compare = confirm.get("compare", {}) if isinstance(confirm.get("compare", {}), dict) else {}
    raw_confirm_step = _safe_float(compare.get("step_time_mean_raw_s")) or 0.0
    fe_confirm_step = _safe_float(compare.get("step_time_mean_frontend_s")) or 0.0
    raw_confirm_wait = _safe_float(compare.get("data_wait_mean_raw")) or 0.0
    fe_confirm_wait = _safe_float(compare.get("data_wait_mean_frontend")) or 0.0

    speedup_vs_raw_confirm = raw_confirm_step / max(1e-9, float(full_avg["full_mean_step_time_s"]))

    if speedup_vs_raw_confirm >= 9.0:
        speedup_bucket = "接近10x"
    elif speedup_vs_raw_confirm >= 7.0:
        speedup_bucket = "8x"
    elif speedup_vs_raw_confirm >= 4.0:
        speedup_bucket = "5x"
    else:
        speedup_bucket = "3x"

    sustained_speedup = bool(speedup_vs_raw_confirm >= 3.0)
    wait_acceptable = bool(float(full_avg["recent500_mean_data_wait_ratio"]) < 0.20)

    result["efficiency"] = {
        "per_run": eff,
        "full_runs_average": full_avg,
        "confirm_baseline": {
            "raw_confirm_step_time_mean_s": raw_confirm_step,
            "frontend_confirm_step_time_mean_s": fe_confirm_step,
            "raw_confirm_data_wait_mean": raw_confirm_wait,
            "frontend_confirm_data_wait_mean": fe_confirm_wait,
        },
        "comparison": {
            "speedup_vs_raw_confirm": float(speedup_vs_raw_confirm),
            "speedup_bucket": speedup_bucket,
            "frontend_cache_sustained_significant_speedup": bool(sustained_speedup),
            "data_wait_ratio_acceptable": bool(wait_acceptable),
        },
    }

    # E. Final 5 conclusions
    all_done = all(str(result["runs"][r].get("job_state", "")).lower() == "done" and int(result["runs"][r].get("final_step", 0)) >= 2000 for r in RUNS)

    result["final_conclusions"] = {
        "all_four_runs_complete": bool(all_done),
        "warmup_vs_nowarm_gradient_conflict": (
            "warmup持续缓解" if warmup_sustained_mitigation else "warmup未呈现持续缓解"
        ),
        "query_gradient_fixed_nonzero": bool(query_nonzero),
        "full_beats_wo_sem_and_wo_obj": bool(full_beats_ablations),
        "frontend_cache_default_path_justified": bool(sustained_speedup and wait_acceptable),
        "frontend_cache_speedup_bucket": speedup_bucket,
    }

    out_report.parent.mkdir(parents=True, exist_ok=True)
    out_report.write_text(json.dumps(result, indent=2, ensure_ascii=False))

    lines = [
        "# STWM D1 Clean Matrix Final Report V1",
        "",
        f"Generated at: {result['generated_at']}",
        "",
        "## A. Completion Validation",
        "",
    ]
    for run in RUNS:
        info = result["runs"][run]
        lines.append(
            f"- {run}: state={info.get('job_state','')}, final_step={info.get('final_step',0)}, "
            f"latest.pt={Path(str(info.get('latest_pt',''))).exists()}, "
            f"best_protocol_main.pt={Path(str(info.get('best_protocol_main_pt',''))).exists()}, "
            f"selection_sidecar={info.get('best_protocol_main_selection','')}"
        )

    lines.extend(
        [
            "",
            "## B. Gradient Audit",
            "",
            "### nowarm",
            _format_stats_block("||g_traj||", result["gradient"]["summary"]["nowarm"]["g_traj_norm"]),
            _format_stats_block("||g_sem||", result["gradient"]["summary"]["nowarm"]["g_sem_norm"]),
            _format_stats_block("cos(g_sem, g_traj)", result["gradient"]["summary"]["nowarm"]["cos_sem_traj"]),
            "",
            "### warmup",
            _format_stats_block("||g_traj||", result["gradient"]["summary"]["warmup"]["g_traj_norm"]),
            _format_stats_block("||g_sem||", result["gradient"]["summary"]["warmup"]["g_sem_norm"]),
            _format_stats_block("cos(g_sem, g_traj)", result["gradient"]["summary"]["warmup"]["cos_sem_traj"]),
            "",
            f"- warmup持续缓解冲突: {result['gradient']['warmup_sustained_conflict_mitigation']}",
            f"- query修复后非零: {result['gradient']['query_gradient_nonzero_after_fix']} ({result['gradient']['query_gradient_reason']})",
            "",
            "## C. Selection Scorecard",
            "",
        ]
    )

    for run in RUNS:
        s = result["selection_scorecard"]["runs"][run]
        lines.append(
            f"- {run}: step={s.get('selected_best_step')}, q_loc={s.get('query_localization_error')}, "
            f"q_top1={s.get('query_top1_acc')}, fut_l1={s.get('future_trajectory_l1')}, "
            f"fut_iou={s.get('future_mask_iou')}, id_cons={s.get('identity_consistency')}, "
            f"id_sw={s.get('identity_switch_rate')}"
        )

    lines.extend(
        [
            "",
            f"- full优于wo_sem/wo_object_bias: {result['selection_scorecard']['full_stably_beats_wo_semantics_and_wo_object_bias']}",
            f"- warmup相对nowarm净收益: {result['selection_scorecard']['warmup_net_protocol_best_gain']}",
            "",
            "## D. Efficiency",
            "",
        ]
    )

    for run in RUNS:
        e = result["efficiency"]["per_run"][run]
        lines.append(
            f"- {run}: mean_step={e.get('full_mean_step_time_s'):.4f}, mean_data={e.get('full_mean_data_time_s'):.4f}, "
            f"mean_wait={e.get('full_mean_data_wait_ratio'):.4f}, recent500_step={e.get('recent500_mean_step_time_s'):.4f}, "
            f"recent500_wait={e.get('recent500_mean_data_wait_ratio'):.4f}, p50={e.get('step_time_p50_s'):.4f}, p95={e.get('step_time_p95_s'):.4f}"
        )

    c = result["efficiency"]["comparison"]
    lines.extend(
        [
            "",
            f"- frontend_cache在2000-step是否持续显著提速: {c.get('frontend_cache_sustained_significant_speedup')}",
            f"- 提速档位: {c.get('speedup_bucket')} (factor={c.get('speedup_vs_raw_confirm'):.4f})",
            f"- data_wait_ratio是否可接受: {c.get('data_wait_ratio_acceptable')}",
            "",
            "## E. Final 5 Conclusions",
            "",
            f"1) 四个run完整成功结束: {result['final_conclusions']['all_four_runs_complete']}",
            f"2) warmup vs nowarm梯度冲突: {result['final_conclusions']['warmup_vs_nowarm_gradient_conflict']}",
            f"3) query gradient修复后是否正常: {result['final_conclusions']['query_gradient_fixed_nonzero']}",
            f"4) full是否赢下wo_semantics/wo_object_bias: {result['final_conclusions']['full_beats_wo_sem_and_wo_obj']}",
            (
                "5) frontend_cache默认主线是否成立: "
                f"{result['final_conclusions']['frontend_cache_default_path_justified']} "
                f"(提速档位={result['final_conclusions']['frontend_cache_speedup_bucket']})"
            ),
        ]
    )

    out_doc.parent.mkdir(parents=True, exist_ok=True)
    out_doc.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
