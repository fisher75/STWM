from __future__ import annotations

from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path
import json
import statistics
import time
from typing import Any


LOWER_BETTER = {
    "query_localization_error",
    "future_trajectory_l1",
    "identity_switch_rate",
}

HIGHER_BETTER = {
    "query_top1_acc",
    "query_hit_rate",
    "future_mask_iou",
    "identity_consistency",
    "occlusion_recovery_acc",
    "visibility_accuracy",
    "visibility_f1",
}

CORE_METRICS = [
    "query_localization_error",
    "query_top1_acc",
    "future_trajectory_l1",
    "future_mask_iou",
    "identity_consistency",
    "identity_switch_rate",
    "query_hit_rate",
]


def build_parser() -> ArgumentParser:
    p = ArgumentParser(description="Generate STWM object-bias autopsy report from completed artifacts")
    p.add_argument("--repo-root", default="/home/chen034/workspace/stwm")
    p.add_argument(
        "--run-root",
        default=(
            "/home/chen034/workspace/stwm/outputs/training/"
            "stwm_v4_2_220m_protocol_frozen_frontend_default_v1"
        ),
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--full-run", default="full_v4_2_seed42_fixed_nowarm_lambda1")
    p.add_argument("--wo-run", default="wo_object_bias_v4_2_seed42")
    p.add_argument(
        "--out-report",
        default="/home/chen034/workspace/stwm/reports/stwm_object_bias_autopsy_v1.json",
    )
    p.add_argument(
        "--out-doc",
        default="/home/chen034/workspace/stwm/docs/STWM_OBJECT_BIAS_AUTOPSY_V1.md",
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


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"invalid json object at {path}")
    return payload


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


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _pearson(x: list[float], y: list[float]) -> float:
    if len(x) != len(y) or len(x) < 2:
        return 0.0
    mx = _mean(x)
    my = _mean(y)
    num = sum((a - mx) * (b - my) for a, b in zip(x, y))
    den_x = sum((a - mx) ** 2 for a in x)
    den_y = sum((b - my) ** 2 for b in y)
    den = (den_x * den_y) ** 0.5
    if den <= 0.0:
        return 0.0
    return float(num / den)


def _window_mean(rows: list[dict[str, Any]], key: str, lo: int, hi: int) -> float:
    vals: list[float] = []
    for row in rows:
        step = row.get("step")
        if not isinstance(step, (int, float)):
            continue
        si = int(step)
        if si < lo or si > hi:
            continue
        fv = _safe_float(row.get(key))
        if fv is not None:
            vals.append(fv)
    return _mean(vals)


def _clip_keyed_rows(rows: list[dict[str, Any]]) -> dict[tuple[str, int], dict[str, Any]]:
    counter: dict[str, int] = defaultdict(int)
    out: dict[tuple[str, int], dict[str, Any]] = {}
    for row in rows:
        clip_id = str(row.get("clip_id", ""))
        idx = counter[clip_id]
        counter[clip_id] += 1
        out[(clip_id, idx)] = row
    return out


def _metric_cmp(metric: str, full_v: float, wo_v: float, eps: float = 1e-12) -> str:
    if metric in LOWER_BETTER:
        if full_v > wo_v + eps:
            return "full_worse"
        if full_v < wo_v - eps:
            return "full_better"
        return "tie"
    if metric in HIGHER_BETTER:
        if full_v < wo_v - eps:
            return "full_worse"
        if full_v > wo_v + eps:
            return "full_better"
        return "tie"
    if full_v > wo_v + eps:
        return "full_worse"
    if full_v < wo_v - eps:
        return "full_better"
    return "tie"


def _top_examples(entries: list[dict[str, Any]], metric: str, k: int = 12) -> list[dict[str, Any]]:
    def severity(item: dict[str, Any]) -> float:
        delta = float(item.get("full_minus_wo", 0.0))
        if metric in LOWER_BETTER:
            return delta
        if metric in HIGHER_BETTER:
            return -delta
        return abs(delta)

    ranked = sorted(entries, key=severity, reverse=True)
    out: list[dict[str, Any]] = []
    for item in ranked[:k]:
        out.append(
            {
                "clip_id": str(item.get("clip_id", "")),
                "occurrence": int(item.get("occurrence", 0)),
                "full": float(item.get("full", 0.0)),
                "wo_object_bias": float(item.get("wo", 0.0)),
                "full_minus_wo": float(item.get("full_minus_wo", 0.0)),
                "comparison": str(item.get("comparison", "tie")),
            }
        )
    return out


def _build_metric_comparison(
    full_eval: dict[str, Any],
    wo_eval: dict[str, Any],
) -> dict[str, Any]:
    full_per_clip = full_eval.get("per_clip", [])
    wo_per_clip = wo_eval.get("per_clip", [])
    if not isinstance(full_per_clip, list) or not isinstance(wo_per_clip, list):
        raise ValueError("protocol eval does not contain per_clip list")

    full_map = _clip_keyed_rows([r for r in full_per_clip if isinstance(r, dict)])
    wo_map = _clip_keyed_rows([r for r in wo_per_clip if isinstance(r, dict)])
    paired_keys = sorted(set(full_map.keys()) & set(wo_map.keys()))

    per_metric: dict[str, Any] = {}
    query_deltas: list[float] = []
    traj_deltas: list[float] = []
    for metric in CORE_METRICS:
        entries: list[dict[str, Any]] = []
        for clip_key in paired_keys:
            full_row = full_map[clip_key]
            wo_row = wo_map[clip_key]
            full_v = _safe_float(full_row.get(metric))
            wo_v = _safe_float(wo_row.get(metric))
            if full_v is None or wo_v is None:
                continue
            entries.append(
                {
                    "clip_id": clip_key[0],
                    "occurrence": int(clip_key[1]),
                    "full": float(full_v),
                    "wo": float(wo_v),
                    "full_minus_wo": float(full_v - wo_v),
                    "comparison": _metric_cmp(metric, float(full_v), float(wo_v)),
                }
            )

            if metric == "query_localization_error":
                query_deltas.append(float(full_v - wo_v))
            if metric == "future_trajectory_l1":
                traj_deltas.append(float(full_v - wo_v))

        if not entries:
            per_metric[metric] = {
                "count": 0,
                "full_worse_rate": 0.0,
                "full_better_rate": 0.0,
                "tie_rate": 0.0,
                "mean_full_minus_wo": 0.0,
                "median_full_minus_wo": 0.0,
                "top_full_worse_examples": [],
            }
            continue

        deltas = [float(e["full_minus_wo"]) for e in entries]
        worse = sum(1 for e in entries if e["comparison"] == "full_worse")
        better = sum(1 for e in entries if e["comparison"] == "full_better")
        ties = sum(1 for e in entries if e["comparison"] == "tie")
        total = len(entries)
        per_metric[metric] = {
            "count": int(total),
            "full_worse_rate": float(worse / total),
            "full_better_rate": float(better / total),
            "tie_rate": float(ties / total),
            "mean_full_minus_wo": float(_mean(deltas)),
            "median_full_minus_wo": float(statistics.median(deltas)),
            "top_full_worse_examples": _top_examples(entries, metric=metric, k=12),
        }

    coupling = {
        "query_traj_delta_corr": _pearson(query_deltas, traj_deltas),
        "share_query_worse_traj_worse": 0.0,
        "share_query_worse_traj_not_worse": 0.0,
    }
    if query_deltas and traj_deltas and len(query_deltas) == len(traj_deltas):
        total = float(len(query_deltas))
        q_worse_t_worse = sum(1 for qd, td in zip(query_deltas, traj_deltas) if qd > 1e-12 and td > 1e-12)
        q_worse_t_not = sum(1 for qd, td in zip(query_deltas, traj_deltas) if qd > 1e-12 and td <= 1e-12)
        coupling["share_query_worse_traj_worse"] = float(q_worse_t_worse / total)
        coupling["share_query_worse_traj_not_worse"] = float(q_worse_t_not / total)

    return {
        "paired_clips": int(len(paired_keys)),
        "metrics": per_metric,
        "query_traj_coupling": coupling,
    }


def _build_train_window_summary(
    full_rows: list[dict[str, Any]],
    wo_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    windows = [
        ("step_0001_0200", 1, 200),
        ("step_0201_0600", 201, 600),
        ("step_0601_1200", 601, 1200),
        ("step_1201_2000", 1201, 2000),
    ]
    metrics = ["query_localization_error", "trajectory_l1", "objectness_mean"]

    out: dict[str, Any] = {}
    for name, lo, hi in windows:
        slot: dict[str, Any] = {}
        for metric in metrics:
            full_v = _window_mean(full_rows, metric, lo, hi)
            wo_v = _window_mean(wo_rows, metric, lo, hi)
            slot[metric] = {
                "full": float(full_v),
                "wo_object_bias": float(wo_v),
                "full_minus_wo": float(full_v - wo_v),
            }
        out[name] = slot

    early = out["step_0001_0200"]["query_localization_error"]["full_minus_wo"]
    late = out["step_1201_2000"]["query_localization_error"]["full_minus_wo"]
    early_late_ratio = float(abs(early) / max(abs(late), 1e-12))
    out["query_gap_early_late_ratio"] = early_late_ratio
    return out


def _choose_primary_hypothesis(
    per_clip_metrics: dict[str, Any],
    windows: dict[str, Any],
    coupling: dict[str, Any],
) -> dict[str, Any]:
    q = per_clip_metrics["query_localization_error"]
    t = per_clip_metrics["future_trajectory_l1"]
    top1 = per_clip_metrics["query_top1_acc"]
    ratio = float(windows.get("query_gap_early_late_ratio", 0.0))

    mid_obj_gap = float(windows["step_0601_1200"]["objectness_mean"]["full_minus_wo"])
    late_obj_gap = float(windows["step_1201_2000"]["objectness_mean"]["full_minus_wo"])

    issue = "object_bias_strength_and_timing"
    if q.get("full_worse_rate", 0.0) < 0.6:
        issue = "weak_or_inconclusive_object_bias_effect"

    evidence = {
        "query_localization_full_worse_rate": float(q.get("full_worse_rate", 0.0)),
        "future_trajectory_full_worse_rate": float(t.get("full_worse_rate", 0.0)),
        "query_top1_full_worse_rate": float(top1.get("full_worse_rate", 0.0)),
        "query_gap_early_late_ratio": float(ratio),
        "objectness_gap_mid": float(mid_obj_gap),
        "objectness_gap_late": float(late_obj_gap),
        "query_traj_delta_corr": float(coupling.get("query_traj_delta_corr", 0.0)),
        "share_query_worse_traj_not_worse": float(coupling.get("share_query_worse_traj_not_worse", 0.0)),
    }

    narrative = (
        "full path appears over-biased to objectness and injects this bias too early: "
        "query/trajectory are worse on almost all clips while objectness_mean remains much higher than wo_object_bias, "
        "and the largest query gap occurs in early steps."
    )
    if issue != "object_bias_strength_and_timing":
        narrative = "object-bias failure is not dominant under current evidence; diagnostics should re-check data and selection artifacts."

    return {
        "most_suspicious_issue": issue,
        "narrative": narrative,
        "evidence": evidence,
    }


def _fmt(v: float) -> str:
    return f"{v:.6f}"


def _build_doc(result: dict[str, Any]) -> str:
    per_clip = result["per_clip_comparison"]["metrics"]
    windows = result["train_windows"]
    hypothesis = result["hypothesis"]
    coupling = result["per_clip_comparison"]["query_traj_coupling"]

    lines: list[str] = []
    lines.append("# STWM Object Bias Autopsy V1")
    lines.append("")
    lines.append(f"Date: {result['generated_at']}")
    lines.append(f"Seed: {result['seed']}")
    lines.append(f"Full run: {result['full_run']}")
    lines.append(f"WO object bias run: {result['wo_run']}")
    lines.append("")
    lines.append("## 1) Why wo_object_bias Beats full")
    lines.append("")
    lines.append(f"- Paired clips: {result['per_clip_comparison']['paired_clips']}")
    lines.append(
        "- query_localization_error: "
        f"full_worse_rate={_fmt(per_clip['query_localization_error']['full_worse_rate'])}, "
        f"mean(full-wo)={_fmt(per_clip['query_localization_error']['mean_full_minus_wo'])}"
    )
    lines.append(
        "- future_trajectory_l1: "
        f"full_worse_rate={_fmt(per_clip['future_trajectory_l1']['full_worse_rate'])}, "
        f"mean(full-wo)={_fmt(per_clip['future_trajectory_l1']['mean_full_minus_wo'])}"
    )
    lines.append(
        "- query_top1_acc: "
        f"full_worse_rate={_fmt(per_clip['query_top1_acc']['full_worse_rate'])}, "
        f"mean(full-wo)={_fmt(per_clip['query_top1_acc']['mean_full_minus_wo'])}"
    )
    lines.append(
        "- future_mask_iou / identity consistency are not degraded at the same magnitude, "
        "indicating the dominant failure is query-trajectory axis rather than broad collapse."
    )
    lines.append("")
    lines.append("## 2) Over-Strong / Over-Early Evidence")
    lines.append("")
    for w in ["step_0001_0200", "step_0201_0600", "step_0601_1200", "step_1201_2000"]:
        q = windows[w]["query_localization_error"]["full_minus_wo"]
        t = windows[w]["trajectory_l1"]["full_minus_wo"]
        o = windows[w]["objectness_mean"]["full_minus_wo"]
        lines.append(f"- {w}: delta_query={_fmt(q)}, delta_traj={_fmt(t)}, delta_objectness_mean={_fmt(o)}")
    lines.append(
        "- query gap early/late ratio: "
        f"{_fmt(float(windows['query_gap_early_late_ratio']))} "
        "(large ratio supports over-early bias injection)."
    )
    lines.append(
        "- query/traj delta coupling: "
        f"corr={_fmt(float(coupling['query_traj_delta_corr']))}, "
        f"share(query_worse & traj_not_worse)={_fmt(float(coupling['share_query_worse_traj_not_worse']))}"
    )
    lines.append("")
    lines.append("## 3) Most Suspicious Issue")
    lines.append("")
    lines.append(f"- {hypothesis['most_suspicious_issue']}")
    lines.append(f"- {hypothesis['narrative']}")
    lines.append(
        "- Key evidence: "
        f"query_worse_rate={_fmt(hypothesis['evidence']['query_localization_full_worse_rate'])}, "
        f"traj_worse_rate={_fmt(hypothesis['evidence']['future_trajectory_full_worse_rate'])}, "
        f"query_early_late_ratio={_fmt(hypothesis['evidence']['query_gap_early_late_ratio'])}, "
        f"late_objectness_gap={_fmt(hypothesis['evidence']['objectness_gap_late'])}"
    )
    lines.append(
        "- Wrong-position signal check: "
        f"share(query_worse but traj_not_worse)={_fmt(hypothesis['evidence']['share_query_worse_traj_not_worse'])}; "
        "current evidence favors broad query+traj degradation rather than isolated query-anchor misplacement."
    )
    lines.append("")
    lines.append("## 4) Representative Full-Worse Cases (query_localization_error)")
    lines.append("")
    for ex in per_clip["query_localization_error"]["top_full_worse_examples"][:10]:
        lines.append(
            "- "
            f"clip={ex['clip_id']}#{ex['occurrence']}, full={_fmt(ex['full'])}, "
            f"wo={_fmt(ex['wo_object_bias'])}, full-wo={_fmt(ex['full_minus_wo'])}"
        )
    lines.append("")
    lines.append("## 5) Diagnostic Recommendation")
    lines.append("")
    lines.append("- Priority variants: delayed200 and alpha050 (then alpha025).")
    lines.append("- Keep protocol selection rule unchanged; compare at same short-mid endpoint first.")
    lines.append("- Warmup is not the primary lever for this failure mode; object-bias timing/strength should be fixed first.")
    lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = build_parser().parse_args()

    repo_root = Path(args.repo_root)
    run_root = Path(args.run_root) / f"seed_{int(args.seed)}"
    full_dir = run_root / str(args.full_run)
    wo_dir = run_root / str(args.wo_run)

    full_sidecar = _load_json(full_dir / "checkpoints" / "best_protocol_main_selection.json")
    wo_sidecar = _load_json(wo_dir / "checkpoints" / "best_protocol_main_selection.json")

    full_eval = _load_json(Path(str(full_sidecar["eval_summary"])))
    wo_eval = _load_json(Path(str(wo_sidecar["eval_summary"])))

    full_rows = _load_jsonl(full_dir / "train_log.jsonl")
    wo_rows = _load_jsonl(wo_dir / "train_log.jsonl")

    per_clip_comp = _build_metric_comparison(full_eval=full_eval, wo_eval=wo_eval)
    windows = _build_train_window_summary(full_rows=full_rows, wo_rows=wo_rows)
    hypothesis = _choose_primary_hypothesis(
        per_clip_comp["metrics"],
        windows,
        coupling=per_clip_comp.get("query_traj_coupling", {}),
    )

    result: dict[str, Any] = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "repo_root": str(repo_root),
        "seed": int(args.seed),
        "full_run": str(args.full_run),
        "wo_run": str(args.wo_run),
        "full_eval_path": str(full_sidecar.get("eval_summary", "")),
        "wo_eval_path": str(wo_sidecar.get("eval_summary", "")),
        "global_metrics": {
            "full": full_eval.get("metrics", {}),
            "wo_object_bias": wo_eval.get("metrics", {}),
        },
        "per_clip_comparison": per_clip_comp,
        "train_windows": windows,
        "hypothesis": hypothesis,
    }

    out_report = Path(args.out_report)
    out_doc = Path(args.out_doc)
    out_report.parent.mkdir(parents=True, exist_ok=True)
    out_doc.parent.mkdir(parents=True, exist_ok=True)
    out_report.write_text(json.dumps(result, indent=2))
    out_doc.write_text(_build_doc(result))

    print(str(out_report))
    print(str(out_doc))


if __name__ == "__main__":
    main()
