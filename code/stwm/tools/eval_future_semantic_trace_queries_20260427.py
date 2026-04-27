#!/usr/bin/env python3
from __future__ import annotations

from argparse import ArgumentParser
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import hashlib
import json

ROOT = Path("/home/chen034/workspace/stwm")
REPORTS = ROOT / "reports"
DOCS = ROOT / "docs"

DEFAULT_SOURCE = REPORTS / "stwm_trace_belief_eval_20260424.json"
DEFAULT_REPORT = REPORTS / "stwm_future_semantic_query_eval_20260427.json"
DEFAULT_DOC = DOCS / "STWM_FUTURE_SEMANTIC_QUERY_EVAL_20260427.md"

OFFICIAL_METHOD = "TUSB-v3.1::best_semantic_hard.pt"
OFFICIAL_SCORING = "trace_belief_assoc"
SUBSETS = [
    "long_gap_persistence",
    "occlusion_reappearance",
    "crossing_ambiguity",
    "OOD_hard",
    "appearance_change",
]


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def write_doc(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# STWM Future Semantic Query Eval 20260427",
        "",
        "This is a read-only utility evaluation over existing per-item reports. It does not train, rerun inference, or modify official results.",
        "",
        "## Overall",
        f"- source_report: `{payload['source_report']}`",
        f"- future_semantic_trace_field_available: `{payload['future_semantic_trace_field_available']}`",
        f"- exact_blocking_reason_for_visibility_auroc: `{payload['exact_blocking_reason_for_visibility_auroc']}`",
        f"- exact_blocking_reason_for_uncertainty_ece: `{payload['exact_blocking_reason_for_uncertainty_ece']}`",
        "",
        "## Panels",
        "| panel | rows | top1 | MRR | false confuser | visibility metric | uncertainty metric |",
        "|---|---:|---:|---:|---:|---|---|",
    ]
    for name, panel in payload.get("panels", {}).items():
        overall = panel.get("overall", {})
        lines.append(
            f"| {name} | {overall.get('row_count', 0)} | {fmt(overall.get('top1'))} | "
            f"{fmt(overall.get('MRR'))} | {fmt(overall.get('false_confuser_rate'))} | "
            f"{overall.get('visibility_metric_status')} | {overall.get('uncertainty_metric_status')} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n")


def fmt(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def row_is_official(row: dict[str, Any]) -> bool:
    return (
        str(row.get("method_name")) == OFFICIAL_METHOD
        and str(row.get("scoring_mode")) in {OFFICIAL_SCORING, "trace_belief_assoc"}
    )


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "row_count": 0,
            "top1": None,
            "MRR": None,
            "false_confuser_rate": None,
            "visibility_AUROC": None,
            "visibility_accuracy": None,
            "uncertainty_ECE": None,
            "gap_length_decay": None,
            "visibility_metric_status": "unavailable_no_future_visibility_field",
            "uncertainty_metric_status": "unavailable_no_confidence_or_uncertainty_field",
        }
    top1 = [float(r.get("query_future_top1_acc") or 0.0) for r in rows]
    mrr = [float(r.get("mrr") or 0.0) for r in rows]
    return {
        "row_count": len(rows),
        "unique_item_count": len({str(r.get("protocol_item_id")) for r in rows}),
        "top1": sum(top1) / max(len(top1), 1),
        "MRR": sum(mrr) / max(len(mrr), 1),
        "false_confuser_rate": 1.0 - (sum(top1) / max(len(top1), 1)),
        "visibility_AUROC": None,
        "visibility_accuracy": None,
        "uncertainty_ECE": None,
        "gap_length_decay": None,
        "visibility_metric_status": "unavailable_no_future_visibility_field",
        "uncertainty_metric_status": "unavailable_no_confidence_or_uncertainty_field",
    }


def subset_rows(rows: list[dict[str, Any]], subset: str) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        tags = row.get("subset_tags")
        if isinstance(tags, list) and subset in {str(x) for x in tags}:
            out.append(row)
    return out


def hash_rows(rows: list[dict[str, Any]]) -> str:
    compact = [
        {
            "protocol_item_id": r.get("protocol_item_id"),
            "seed": r.get("seed"),
            "method_name": r.get("method_name"),
            "scoring_mode": r.get("scoring_mode"),
            "top1": r.get("query_future_top1_acc"),
            "rank": r.get("target_rank"),
        }
        for r in rows
    ]
    return hashlib.sha256(json.dumps(compact, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def run(source_report: Path, out_report: Path, out_doc: Path) -> dict[str, Any]:
    source = load_json(source_report)
    panels_out: dict[str, Any] = {}
    all_rows: list[dict[str, Any]] = []
    for panel_name, panel in (source.get("panels") or {}).items():
        rows = [r for r in (panel.get("per_item_results") or []) if isinstance(r, dict) and row_is_official(r)]
        all_rows.extend(rows)
        panels_out[panel_name] = {
            "panel_name": panel_name,
            "task_definitions": {
                "future_target_grounding": "given observed query entity, rank future candidate entity",
                "future_visibility_reappearance_prediction": "blocked until explicit future visibility logits are emitted",
                "false_confuser_rejection": "measured as 1 - future target grounding top1",
                "semantic_hard_subset_breakdown": SUBSETS,
            },
            "overall": aggregate(rows),
            "semantic_hard_subset_breakdown": {subset: aggregate(subset_rows(rows, subset)) for subset in SUBSETS},
            "per_item_results_hash": hash_rows(rows),
        }
    payload = {
        "generated_at_utc": now_iso(),
        "source_report": str(source_report),
        "official_method": OFFICIAL_METHOD,
        "official_scoring_mode": OFFICIAL_SCORING,
        "fresh_eval_executed": False,
        "read_only_existing_per_item_eval": True,
        "future_semantic_trace_field_available": False,
        "future_semantic_query_eval_added": True,
        "task_metrics": [
            "top1",
            "MRR",
            "false_confuser_rate",
            "visibility_AUROC_or_accuracy",
            "uncertainty_ECE_if_available",
            "gap_length_decay_if_available",
        ],
        "exact_blocking_reason_for_visibility_auroc": "current official per-item reports do not contain explicit future_visibility_logit or visibility labels emitted by a FutureSemanticTraceState head",
        "exact_blocking_reason_for_uncertainty_ece": "current official per-item reports do not contain calibrated future_uncertainty/confidence fields from a FutureSemanticTraceState head",
        "panels": panels_out,
        "overall_all_panels": aggregate(all_rows),
        "all_panels_per_item_results_hash": hash_rows(all_rows),
    }
    write_json(out_report, payload)
    write_doc(out_doc, payload)
    return payload


def parse_args() -> Any:
    p = ArgumentParser(description="Evaluate future semantic trace query utility from existing STWM per-item reports.")
    p.add_argument("--source-report", default=str(DEFAULT_SOURCE))
    p.add_argument("--out-report", default=str(DEFAULT_REPORT))
    p.add_argument("--out-doc", default=str(DEFAULT_DOC))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run(Path(args.source_report), Path(args.out_report), Path(args.out_doc))


if __name__ == "__main__":
    main()
