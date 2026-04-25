#!/usr/bin/env python3
from __future__ import annotations

from argparse import ArgumentParser
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple
import hashlib
import json
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

try:
    import setproctitle  # type: ignore
except Exception:
    setproctitle = None

if setproctitle is not None:
    try:
        setproctitle.setproctitle("python")
    except Exception:
        pass

for candidate in [Path("/raid/chen034/workspace/stwm/code"), Path("/home/chen034/workspace/stwm/code")]:
    if candidate.exists() and str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from stwm.tools import run_stage2_state_identifiability_eval_20260415 as evalcore
from stwm.tools import run_stwm_downstream_utility_v2_20260420 as utilitycore
from stwm.tools import run_stwm_top_tier_final_validation_20260420 as finalcore
from stwm.tools import run_stwm_trace_belief_filter_20260424 as beliefcore
from stwm.tools import run_stwm_tusb_light_readout_eval_20260422 as lighteval
from stwm.tools import run_stwm_true_ood_eval_20260420 as oodcore


ROOT = Path("/raid/chen034/workspace/stwm")
REPORTS = ROOT / "reports"
DOCS = ROOT / "docs"
LOGS = ROOT / "logs"

BASE_TUSB = "TUSB-v3.1::best_semantic_hard.pt"
OFFICIAL_TUSB = "TUSB-v3.1::official(best_semantic_hard.pt+trace_belief_assoc)"
CAL = "calibration-only::best.pt"
CROP = "cropenc::best.pt"
LEGACY = "legacysem::best.pt"
STAGE1 = "stage1_frozen::best.pt"
BASELINE_METHODS = [CAL, CROP, LEGACY]
METHOD_ORDER = [OFFICIAL_TUSB, CAL, CROP, LEGACY, STAGE1]
SEEDS = list(lighteval.SEEDS)
OFFICIAL_MODE = "trace_belief_assoc"
BASELINE_MODE = "coord_only"
FINAL_PANEL_NAMES = [
    "legacy_85_context_preserving",
    "densified_200_context_preserving",
    "protocol_v3_extended_600_context_preserving",
]
OOD_PANEL_NAMES = [
    "heldout_burst_heavy_context_preserving",
    "heldout_scene_category_video_context_preserving",
]
STRICT_METRICS = [
    ("overall_top1", "query_future_top1_acc", ""),
    ("hard_subset_top1", "query_future_top1_acc", "__hard__"),
    ("ambiguity_top1", "query_future_top1_acc", "crossing_ambiguity"),
    ("appearance_change_top1", "query_future_top1_acc", "appearance_change"),
    ("occlusion_reappearance_top1", "query_future_top1_acc", "occlusion_reappearance"),
    ("long_gap_persistence_top1", "query_future_top1_acc", "long_gap_persistence"),
]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def _write_md(path: Path, title: str, lines: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join([f"# {title}", "", *list(lines)]).rstrip() + "\n", encoding="utf-8")


def _sha256_json(payload: Any) -> str:
    raw = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _mean(values: Iterable[float]) -> float:
    vals = [float(v) for v in values]
    return float(sum(vals) / max(len(vals), 1))


def _std(values: Iterable[float]) -> float:
    vals = [float(v) for v in values]
    if len(vals) <= 1:
        return 0.0
    m = _mean(vals)
    return float((sum((v - m) ** 2 for v in vals) / len(vals)) ** 0.5)


def _canonical_rows(rows: Sequence[Mapping[str, Any]], panel_name: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in rows:
        method = str(row.get("method_name", ""))
        mode = str(row.get("scoring_mode", ""))
        if method == BASE_TUSB and mode == OFFICIAL_MODE:
            copied = dict(row)
            copied["method_name"] = OFFICIAL_TUSB
            copied["scoring_mode"] = OFFICIAL_MODE
        elif method in BASELINE_METHODS and mode == BASELINE_MODE:
            copied = dict(row)
            copied["method_name"] = method
            copied["scoring_mode"] = BASELINE_MODE
        else:
            continue
        copied["panel_name"] = panel_name
        copied.setdefault("split", copied.get("item_split", "test"))
        copied.setdefault("item_split", copied.get("split", "test"))
        out.append(copied)
    return out


def _dedupe_rows(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    by_key: Dict[Tuple[str, int, str, str], Dict[str, Any]] = {}
    for row in rows:
        key = (
            str(row.get("protocol_item_id", "")),
            int(row.get("seed", -1)),
            str(row.get("method_name", "")),
            str(row.get("scoring_mode", "")),
        )
        by_key[key] = dict(row)
    return [by_key[key] for key in sorted(by_key.keys())]


def _aggregate(rows: Sequence[Mapping[str, Any]]) -> Dict[str, float]:
    return lighteval._aggregate_rows([dict(row) for row in rows])


def _seed_table(rows: Sequence[Mapping[str, Any]], method_name: str, scoring_mode: str) -> Dict[str, Any]:
    seed_rows: List[Dict[str, Any]] = []
    for seed in SEEDS:
        picked = [
            dict(row)
            for row in rows
            if str(row.get("method_name")) == method_name
            and str(row.get("scoring_mode")) == scoring_mode
            and int(row.get("seed", -1)) == int(seed)
        ]
        seed_rows.append({"seed": int(seed), **_aggregate(picked)})
    keys = [key for key in seed_rows[0].keys() if key != "seed"] if seed_rows else []
    return {
        "seed_rows": seed_rows,
        "mean": {key: _mean(row[key] for row in seed_rows) for key in keys},
        "std": {key: _std(row[key] for row in seed_rows) for key in keys},
    }


def _panel_report(
    *,
    panel_name: str,
    rows: Sequence[Mapping[str, Any]],
    total_requested_items: int,
    valid_items: int | None = None,
    skipped_items: int | None = None,
    skipped_reason_counts: Mapping[str, int] | None = None,
    source_panel_name: str = "",
    exact_blocking_reason: str = "",
) -> Dict[str, Any]:
    final_rows = _dedupe_rows(rows)
    unique_ids = {str(row.get("protocol_item_id", "")) for row in final_rows if str(row.get("protocol_item_id", ""))}
    valid = int(valid_items if valid_items is not None else len(unique_ids))
    skipped = int(skipped_items if skipped_items is not None else max(int(total_requested_items) - valid, 0))
    per_method_seed_results: Dict[str, Dict[str, Any]] = {
        OFFICIAL_TUSB: {OFFICIAL_MODE: _seed_table(final_rows, OFFICIAL_TUSB, OFFICIAL_MODE)},
        CAL: {BASELINE_MODE: _seed_table(final_rows, CAL, BASELINE_MODE)},
        CROP: {BASELINE_MODE: _seed_table(final_rows, CROP, BASELINE_MODE)},
        LEGACY: {BASELINE_MODE: _seed_table(final_rows, LEGACY, BASELINE_MODE)},
        STAGE1: {
            BASELINE_MODE: {
                "available": False,
                "exact_blocking_reason": "stage1 frozen rows are not present in the current belief source shards; Stage1 remains frozen and is not used in the belief readout comparisons",
                "seed_rows": [],
                "mean": {},
                "std": {},
            }
        },
    }
    return {
        "panel_name": panel_name,
        "source_panel_name": source_panel_name or panel_name,
        "total_requested_items": int(total_requested_items),
        "valid_items": valid,
        "skipped_items": skipped,
        "skipped_reason_counts": dict(sorted((skipped_reason_counts or {}).items())),
        "per_item_results_hash": _sha256_json(final_rows),
        "per_item_results": final_rows,
        "per_method_seed_results": per_method_seed_results,
        "exact_blocking_reason": exact_blocking_reason,
    }


def _panel_mean(panel: Mapping[str, Any], method_name: str, mode: str) -> Dict[str, float]:
    return (
        panel.get("per_method_seed_results", {})
        .get(method_name, {})
        .get(mode, {})
        .get("mean", {})
        if isinstance(panel.get("per_method_seed_results", {}), dict)
        else {}
    )


def _beats_on_panels(panels: Sequence[Mapping[str, Any]], right_method: str, metric: str = "overall_top1") -> bool:
    if not panels:
        return False
    for panel in panels:
        left = _panel_mean(panel, OFFICIAL_TUSB, OFFICIAL_MODE)
        right = _panel_mean(panel, right_method, BASELINE_MODE)
        if float(left.get(metric, 0.0)) <= float(right.get(metric, 0.0)):
            return False
    return True


def _metric_deltas(
    rows: Sequence[Mapping[str, Any]],
    left_method: str,
    left_mode: str,
    right_method: str,
    right_mode: str,
    metric_key: str,
    subset_tag: str,
) -> List[float]:
    right_by_key: Dict[Tuple[str, int], Mapping[str, Any]] = {}
    for row in rows:
        if str(row.get("method_name")) == right_method and str(row.get("scoring_mode")) == right_mode:
            right_by_key[(str(row.get("protocol_item_id", "")), int(row.get("seed", -1)))] = row
    deltas: List[float] = []
    for row in rows:
        if str(row.get("method_name")) != left_method or str(row.get("scoring_mode")) != left_mode:
            continue
        tags = set(row.get("subset_tags", []) or [])
        if subset_tag == "__hard__" and not tags:
            continue
        if subset_tag and subset_tag != "__hard__" and subset_tag not in tags:
            continue
        match = right_by_key.get((str(row.get("protocol_item_id", "")), int(row.get("seed", -1))))
        if isinstance(match, Mapping):
            deltas.append(float(row.get(metric_key, 0.0)) - float(match.get(metric_key, 0.0)))
    return deltas


def _bootstrap_panel(rows: Sequence[Mapping[str, Any]], panel_name: str, right_method: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    compare_name = f"{OFFICIAL_TUSB}_vs_{right_method}"
    for metric_name, metric_key, subset_tag in STRICT_METRICS:
        deltas = _metric_deltas(rows, OFFICIAL_TUSB, OFFICIAL_MODE, right_method, BASELINE_MODE, metric_key, subset_tag)
        out[metric_name] = lighteval._bootstrap_deltas(
            deltas,
            seed=lighteval._stable_bootstrap_seed(panel_name, compare_name, metric_name, subset_tag),
        )
    return out


def _build_promotion_audit(args: Any, feasibility: Mapping[str, Any]) -> Dict[str, Any]:
    payload = {
        "generated_at_utc": _now_iso(),
        "official_tusb_checkpoint": "best_semantic_hard.pt",
        "official_tusb_scoring_mode": OFFICIAL_MODE,
        "official_tusb_method": OFFICIAL_TUSB,
        "baselines_scoring_mode": BASELINE_MODE,
        "baseline_methods": {
            "calibration-only": "best.pt + coord_only",
            "cropenc": "best.pt + coord_only",
            "legacysem": "best.pt + coord_only",
            "stage1_frozen": "best.pt + coord_only; frozen and not retrained",
        },
        "belief_feasibility_audit_passed": bool(feasibility.get("audit_passed", False)),
        "official_evaluator_supports_belief": True,
        "belief_promotion_passed": bool(feasibility.get("audit_passed", False)),
        "exact_blocking_reason": "" if bool(feasibility.get("audit_passed", False)) else str(feasibility.get("exact_blocking_reason", "")),
    }
    _write_json(Path(args.promotion_audit_report), payload)
    _write_md(
        Path(args.promotion_audit_doc),
        "STWM Belief Promotion Audit 20260424",
        [
            f"- official_tusb_checkpoint: {payload['official_tusb_checkpoint']}",
            f"- official_tusb_scoring_mode: {payload['official_tusb_scoring_mode']}",
            f"- baselines_scoring_mode: {payload['baselines_scoring_mode']}",
            f"- belief_promotion_passed: {payload['belief_promotion_passed']}",
            f"- exact_blocking_reason: {payload['exact_blocking_reason']}",
        ],
    )
    return payload


def _build_final_eval(args: Any) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    belief_args = SimpleNamespace(
        audit_json=str(args.promotion_audit_report),
        audit_md=str(args.promotion_audit_doc),
        eval_json=str(args.final_eval_report),
        eval_md=str(args.final_eval_doc),
        bootstrap_json=str(REPORTS / "stwm_belief_internal_unused_bootstrap_20260424.json"),
        bootstrap_md=str(DOCS / "STWM_BELIEF_INTERNAL_UNUSED_BOOTSTRAP_20260424.md"),
        decision_json=str(REPORTS / "stwm_belief_internal_unused_decision_20260424.json"),
        decision_md=str(DOCS / "STWM_BELIEF_INTERNAL_UNUSED_DECISION_20260424.md"),
        dense_protocol_json=str(args.dense_protocol_json),
        extended_protocol_json=str(args.extended_protocol_json),
        source_shards=str(args.source_shards),
        device=str(args.device),
        lease_path=str(args.lease_path),
        eval_required_mem_gb=float(args.eval_required_mem_gb),
        eval_safety_margin_gb=float(args.eval_safety_margin_gb),
        audit_sample_items=int(args.audit_sample_items),
    )
    result = beliefcore._build_eval(belief_args)
    raw_eval = result["eval"]
    audit = _build_promotion_audit(args, result["audit"])
    raw_panels = raw_eval.get("panels", {}) if isinstance(raw_eval.get("panels", {}), dict) else {}

    dense_raw = raw_panels.get("densified_200_context_preserving", {})
    dense_rows = _canonical_rows(dense_raw.get("per_item_results", []), "densified_200_context_preserving")
    dense_panel = _panel_report(
        panel_name="densified_200_context_preserving",
        rows=dense_rows,
        total_requested_items=int(dense_raw.get("total_items", dense_raw.get("total_requested_items", 200))),
        valid_items=int(dense_raw.get("valid_items", 0)),
        skipped_items=int(dense_raw.get("skipped_items", 0)),
        skipped_reason_counts=dense_raw.get("skipped_reason_counts", {}),
    )

    legacy_ids = finalcore._legacy_85_item_ids()
    legacy_rows = [row for row in dense_rows if str(row.get("protocol_item_id", "")) in legacy_ids]
    legacy_panel = _panel_report(
        panel_name="legacy_85_context_preserving",
        rows=legacy_rows,
        total_requested_items=int(len(legacy_ids)),
        valid_items=int(len({str(row.get("protocol_item_id", "")) for row in legacy_rows})),
        skipped_items=max(int(len(legacy_ids)) - int(len({str(row.get("protocol_item_id", "")) for row in legacy_rows})), 0),
        skipped_reason_counts={"not_in_densified_test_split": max(int(len(legacy_ids)) - int(len({str(row.get("protocol_item_id", "")) for row in legacy_rows})), 0)} if len(legacy_ids) else {},
        source_panel_name="densified_200_context_preserving",
    )

    ood_rows: List[Dict[str, Any]] = []
    for ood_name in OOD_PANEL_NAMES:
        ood_panel = raw_panels.get(ood_name, {})
        ood_rows.extend(_canonical_rows(ood_panel.get("per_item_results", []), "protocol_v3_extended_600_context_preserving"))
    extended_payload = _load_json(Path(args.extended_protocol_json))
    extended_count = len(extended_payload.get("items", [])) if isinstance(extended_payload.get("items", []), list) else 0
    protocol_rows = _dedupe_rows(ood_rows)
    protocol_valid_ids = {str(row.get("protocol_item_id", "")) for row in protocol_rows if str(row.get("protocol_item_id", ""))}
    protocol_skipped = max(int(extended_count) - int(len(protocol_valid_ids)), 0)
    protocol_panel = _panel_report(
        panel_name="protocol_v3_extended_600_context_preserving",
        rows=protocol_rows,
        total_requested_items=int(extended_count),
        valid_items=int(len(protocol_valid_ids)),
        skipped_items=protocol_skipped,
        skipped_reason_counts={"not_in_current_true_ood_test_union": protocol_skipped} if protocol_skipped else {},
        source_panel_name="heldout_burst_heavy + heldout_scene_category_video union",
        exact_blocking_reason=(
            ""
            if protocol_valid_ids
            else "protocol_v3_extended_600_context_preserving has no belief rows in the current materialized true OOD union"
        ),
    )

    primary_panels = [dense_panel]
    if protocol_panel["valid_items"] > 0:
        primary_panels.append(protocol_panel)
    final_eval = {
        "generated_at_utc": _now_iso(),
        "official_tusb_method": OFFICIAL_TUSB,
        "official_tusb_checkpoint": "best_semantic_hard.pt",
        "official_tusb_scoring_mode": OFFICIAL_MODE,
        "baselines_scoring_mode": BASELINE_MODE,
        "stage1_frozen": True,
        "selected_device": raw_eval.get("selected_device", "belief_runner_device"),
        "source_trace_belief_eval_fresh_rerun": True,
        "source_trace_belief_wall_time_seconds": raw_eval.get("wall_time_seconds"),
        "eval_started_at": raw_eval.get("eval_started_at"),
        "eval_finished_at": raw_eval.get("eval_finished_at"),
        "wall_time_seconds": raw_eval.get("wall_time_seconds"),
        "belief_empty_items": raw_eval.get("belief_empty_items"),
        "diag_variance_available_items": raw_eval.get("diag_variance_available_items"),
        "panels": {
            "legacy_85_context_preserving": legacy_panel,
            "densified_200_context_preserving": dense_panel,
            "protocol_v3_extended_600_context_preserving": protocol_panel,
        },
        "auxiliary_true_ood_panel_names": list(OOD_PANEL_NAMES),
        "raw_true_ood_panels": {
            name: {
                **raw_panels.get(name, {}),
                "per_item_results": _canonical_rows(raw_panels.get(name, {}).get("per_item_results", []), name),
                "per_item_results_hash": _sha256_json(_canonical_rows(raw_panels.get(name, {}).get("per_item_results", []), name)),
            }
            for name in OOD_PANEL_NAMES
        },
    }
    final_eval["improved_vs_calibration"] = _beats_on_panels(primary_panels, CAL)
    final_eval["improved_vs_cropenc"] = _beats_on_panels(primary_panels, CROP)
    final_eval["improved_vs_legacysem"] = _beats_on_panels(primary_panels, LEGACY)
    final_eval["hard_subsets_improved_vs_calibration"] = _beats_on_panels(primary_panels, CAL, "hard_subset_top1")
    final_eval["hard_subsets_improved_vs_legacysem"] = _beats_on_panels(primary_panels, LEGACY, "hard_subset_top1")
    final_eval["exact_blocking_reason"] = (
        protocol_panel["exact_blocking_reason"]
        or ("protocol_v3_extended_600 skipped_items > 0: see skipped_reason_counts" if protocol_panel["skipped_items"] else "")
    )
    _write_json(Path(args.final_eval_report), final_eval)
    _write_md(
        Path(args.final_eval_doc),
        "STWM Belief Final Eval 20260424",
        [
            f"- official_tusb_method: `{OFFICIAL_TUSB}`",
            f"- official_tusb_checkpoint: `best_semantic_hard.pt`",
            f"- official_tusb_scoring_mode: `{OFFICIAL_MODE}`",
            f"- improved_vs_calibration: {final_eval['improved_vs_calibration']}",
            f"- improved_vs_cropenc: {final_eval['improved_vs_cropenc']}",
            f"- improved_vs_legacysem: {final_eval['improved_vs_legacysem']}",
            f"- hard_subsets_improved_vs_calibration: {final_eval['hard_subsets_improved_vs_calibration']}",
            f"- hard_subsets_improved_vs_legacysem: {final_eval['hard_subsets_improved_vs_legacysem']}",
            *[
                f"- {name}: valid_items={panel['valid_items']} skipped_items={panel['skipped_items']} hash={panel['per_item_results_hash']}"
                for name, panel in final_eval["panels"].items()
            ],
            f"- exact_blocking_reason: {final_eval['exact_blocking_reason'] or 'none'}",
        ],
    )
    return final_eval, audit


def _build_strict_bootstrap(args: Any, final_eval: Mapping[str, Any]) -> Dict[str, Any]:
    panels = final_eval.get("panels", {}) if isinstance(final_eval.get("panels", {}), dict) else {}
    panel_payloads: Dict[str, Any] = {}
    for panel_name in ["densified_200_context_preserving", "protocol_v3_extended_600_context_preserving"]:
        panel = panels.get(panel_name, {})
        rows = panel.get("per_item_results", []) if isinstance(panel.get("per_item_results", []), list) else []
        if not rows:
            continue
        panel_payloads[panel_name] = {
            "vs_calibration": _bootstrap_panel(rows, panel_name, CAL),
            "vs_cropenc": _bootstrap_panel(rows, panel_name, CROP),
            "vs_legacysem": _bootstrap_panel(rows, panel_name, LEGACY),
        }

    def _positive_zero(compare_key: str, metric_name: str, panel_name: str = "densified_200_context_preserving") -> bool:
        block = panel_payloads.get(panel_name, {}).get(compare_key, {}).get(metric_name, {})
        return bool(block.get("zero_excluded", False) and float(block.get("mean_delta", 0.0)) > 0.0)

    zero_excluded_vs_calibration = bool(_positive_zero("vs_calibration", "overall_top1"))
    zero_excluded_vs_cropenc = bool(_positive_zero("vs_cropenc", "overall_top1"))
    zero_excluded_vs_legacysem = bool(_positive_zero("vs_legacysem", "overall_top1"))
    strong = bool(
        zero_excluded_vs_calibration
        and zero_excluded_vs_cropenc
        and zero_excluded_vs_legacysem
        and _positive_zero("vs_calibration", "hard_subset_top1")
        and _positive_zero("vs_cropenc", "hard_subset_top1")
        and _positive_zero("vs_legacysem", "hard_subset_top1")
    )
    if strong:
        claim_level = "strong_claim"
    elif final_eval.get("improved_vs_calibration") and final_eval.get("improved_vs_cropenc") and final_eval.get("improved_vs_legacysem"):
        claim_level = "moderate_claim"
    else:
        claim_level = "weak_claim"
    payload = {
        "generated_at_utc": _now_iso(),
        "official_tusb_method": OFFICIAL_TUSB,
        "panels": panel_payloads,
        "zero_excluded_vs_calibration": zero_excluded_vs_calibration,
        "zero_excluded_vs_cropenc": zero_excluded_vs_cropenc,
        "zero_excluded_vs_legacysem": zero_excluded_vs_legacysem,
        "claim_level": claim_level,
    }
    _write_json(Path(args.strict_bootstrap_report), payload)
    _write_md(
        Path(args.strict_bootstrap_doc),
        "STWM Belief Strict Bootstrap 20260424",
        [
            f"- zero_excluded_vs_calibration: {zero_excluded_vs_calibration}",
            f"- zero_excluded_vs_cropenc: {zero_excluded_vs_cropenc}",
            f"- zero_excluded_vs_legacysem: {zero_excluded_vs_legacysem}",
            f"- claim_level: {claim_level}",
        ],
    )
    return payload


def _build_downstream_utility(args: Any) -> Dict[str, Any]:
    result = utilitycore.build_downstream_utility_from_final_eval(
        report_path=Path(args.downstream_utility_report),
        doc_path=Path(args.downstream_utility_doc),
        final_eval_report=Path(args.final_eval_report),
        official_tusb_method=OFFICIAL_TUSB,
        calibration_method=CAL,
        cropenc_method=CROP,
        legacysem_method=LEGACY,
        dense_panel_name="densified_200_context_preserving",
        extended_panel_name="protocol_v3_extended_600_context_preserving",
    )
    _write_md(
        Path(args.downstream_utility_doc),
        "STWM Belief Downstream Utility 20260424",
        [
            f"- official_tusb_method: `{OFFICIAL_TUSB}`",
            f"- utility_improved_vs_calibration: {result['utility_improved_vs_calibration']}",
            f"- utility_improved_vs_cropenc: {result['utility_improved_vs_cropenc']}",
            f"- utility_improved_vs_legacysem: {result['utility_improved_vs_legacysem']}",
            f"- utility_hard_subset_improved: {result['utility_hard_subset_improved']}",
            f"- leakage_check_passed: {result['leakage_check_passed']}",
            f"- utility_claim_ready: {result['utility_claim_ready']}",
        ],
    )
    return result


def _build_ood_eval(args: Any, final_eval: Mapping[str, Any]) -> Dict[str, Any]:
    raw_ood = final_eval.get("raw_true_ood_panels", {}) if isinstance(final_eval.get("raw_true_ood_panels", {}), dict) else {}
    split_payloads: Dict[str, Any] = {}
    for split_name in OOD_PANEL_NAMES:
        panel = raw_ood.get(split_name, {})
        rows = panel.get("per_item_results", []) if isinstance(panel.get("per_item_results", []), list) else []
        report_panel = _panel_report(
            panel_name=split_name,
            rows=rows,
            total_requested_items=int(panel.get("total_items", panel.get("total_requested_items", 0))),
            valid_items=int(panel.get("valid_items", 0)),
            skipped_items=int(panel.get("skipped_items", 0)),
            skipped_reason_counts=panel.get("skipped_reason_counts", {}),
        )
        official = _panel_mean(report_panel, OFFICIAL_TUSB, OFFICIAL_MODE)
        cal = _panel_mean(report_panel, CAL, BASELINE_MODE)
        crop = _panel_mean(report_panel, CROP, BASELINE_MODE)
        legacy = _panel_mean(report_panel, LEGACY, BASELINE_MODE)
        split_payloads[split_name] = {
            "panel": report_panel,
            "official_mean": official,
            "calibration_mean": cal,
            "cropenc_mean": crop,
            "legacysem_mean": legacy,
            "improved_vs_calibration": bool(float(official.get("overall_top1", 0.0)) > float(cal.get("overall_top1", 0.0))),
            "improved_vs_cropenc": bool(float(official.get("overall_top1", 0.0)) > float(crop.get("overall_top1", 0.0))),
            "improved_vs_legacysem": bool(float(official.get("overall_top1", 0.0)) > float(legacy.get("overall_top1", 0.0))),
            "hard_subsets_improved_vs_calibration": bool(float(official.get("hard_subset_top1", 0.0)) > float(cal.get("hard_subset_top1", 0.0))),
            "hard_subsets_improved_vs_legacysem": bool(float(official.get("hard_subset_top1", 0.0)) > float(legacy.get("hard_subset_top1", 0.0))),
        }
    ood_improved_vs_calibration = all(bool(split_payloads[name]["improved_vs_calibration"]) for name in OOD_PANEL_NAMES)
    ood_improved_vs_cropenc = all(bool(split_payloads[name]["improved_vs_cropenc"]) for name in OOD_PANEL_NAMES)
    ood_improved_vs_legacysem = all(bool(split_payloads[name]["improved_vs_legacysem"]) for name in OOD_PANEL_NAMES)
    hard_subsets_improved = all(
        bool(split_payloads[name]["hard_subsets_improved_vs_calibration"])
        and bool(split_payloads[name]["hard_subsets_improved_vs_legacysem"])
        for name in OOD_PANEL_NAMES
    )
    payload = {
        "generated_at_utc": _now_iso(),
        "official_tusb_method": OFFICIAL_TUSB,
        "official_tusb_checkpoint": "best_semantic_hard.pt",
        "official_tusb_scoring_mode": OFFICIAL_MODE,
        "baselines_scoring_mode": BASELINE_MODE,
        "true_ood_split_source": "current materialized heldout_burst_heavy + heldout_scene_category_video panels from fresh belief rerun",
        "splits": split_payloads,
        "ood_improved_vs_calibration": bool(ood_improved_vs_calibration),
        "ood_improved_vs_cropenc": bool(ood_improved_vs_cropenc),
        "ood_improved_vs_legacysem": bool(ood_improved_vs_legacysem),
        "hard_subsets_improved": bool(hard_subsets_improved),
        "ood_claim_ready": bool(ood_improved_vs_calibration and ood_improved_vs_cropenc and ood_improved_vs_legacysem and hard_subsets_improved),
    }
    _write_json(Path(args.true_ood_eval_report), payload)
    _write_md(
        Path(args.true_ood_eval_doc),
        "STWM Belief True OOD Eval 20260424",
        [
            f"- ood_improved_vs_calibration: {payload['ood_improved_vs_calibration']}",
            f"- ood_improved_vs_cropenc: {payload['ood_improved_vs_cropenc']}",
            f"- ood_improved_vs_legacysem: {payload['ood_improved_vs_legacysem']}",
            f"- hard_subsets_improved: {payload['hard_subsets_improved']}",
            f"- ood_claim_ready: {payload['ood_claim_ready']}",
        ],
    )
    return payload


def _build_decision(
    args: Any,
    audit: Mapping[str, Any],
    final_eval: Mapping[str, Any],
    bootstrap: Mapping[str, Any],
    utility: Mapping[str, Any],
    ood: Mapping[str, Any],
) -> Dict[str, Any]:
    light = _load_json(REPORTS / "stwm_lightreadout_final_eval_20260422.json")
    light_dense = (
        light.get("panels", {})
        .get("densified_200_context_preserving", {})
        .get("per_method_seed_results", {})
        .get("TUSB-v3.1::official(best_semantic_hard.pt+hybrid_light)", {})
        .get("hybrid_light", {})
        .get("mean", {})
        if isinstance(light.get("panels", {}), dict)
        else {}
    )
    belief_dense = _panel_mean(final_eval.get("panels", {}).get("densified_200_context_preserving", {}), OFFICIAL_TUSB, OFFICIAL_MODE)
    belief_not_worse_than_light = bool(float(belief_dense.get("overall_top1", 0.0)) >= float(light_dense.get("overall_top1", -1.0)))
    improved_vs_calibration = bool(final_eval.get("improved_vs_calibration", False))
    improved_vs_cropenc = bool(final_eval.get("improved_vs_cropenc", False))
    improved_vs_legacysem = bool(final_eval.get("improved_vs_legacysem", False))
    hard_subsets_improved = bool(final_eval.get("hard_subsets_improved_vs_calibration", False) and final_eval.get("hard_subsets_improved_vs_legacysem", False))
    bootstrap_claim_level = str(bootstrap.get("claim_level", "weak_claim"))
    utility_improved = bool(utility.get("utility_claim_ready", False))
    ood_improved = bool(ood.get("ood_claim_ready", False))
    if not belief_not_worse_than_light:
        next_step = "reframe_as_moderate_claim_main_track"
    elif improved_vs_legacysem and bootstrap_claim_level in {"strong_claim", "moderate_claim"} and utility_improved:
        next_step = "start_main_submission_assets"
    elif improved_vs_calibration or improved_vs_cropenc or improved_vs_legacysem:
        next_step = "reframe_as_moderate_claim_main_track"
    else:
        next_step = "one_last_surgical_fix"
    payload = {
        "generated_at_utc": _now_iso(),
        "belief_official_readout_integrated": bool(audit.get("belief_promotion_passed", False)),
        "official_tusb_method": OFFICIAL_TUSB,
        "official_tusb_checkpoint": "best_semantic_hard.pt",
        "official_tusb_scoring_mode": OFFICIAL_MODE,
        "belief_not_worse_than_light_readout": belief_not_worse_than_light,
        "improved_vs_calibration": improved_vs_calibration,
        "improved_vs_cropenc": improved_vs_cropenc,
        "improved_vs_legacysem": improved_vs_legacysem,
        "hard_subsets_improved": hard_subsets_improved,
        "bootstrap_claim_level": bootstrap_claim_level,
        "utility_improved": utility_improved,
        "utility_claim_ready": bool(utility.get("utility_claim_ready", False)),
        "true_ood_improved": ood_improved,
        "ood_claim_ready": bool(ood.get("ood_claim_ready", False)),
        "exact_blocking_reason": "" if bool(audit.get("belief_promotion_passed", False)) else str(audit.get("exact_blocking_reason", "")),
        "next_step_choice": next_step,
    }
    _write_json(Path(args.final_decision_report), payload)
    _write_md(
        Path(args.final_decision_doc),
        "STWM Belief Final Decision 20260424",
        [
            f"- belief_official_readout_integrated: {payload['belief_official_readout_integrated']}",
            f"- improved_vs_calibration: {payload['improved_vs_calibration']}",
            f"- improved_vs_cropenc: {payload['improved_vs_cropenc']}",
            f"- improved_vs_legacysem: {payload['improved_vs_legacysem']}",
            f"- hard_subsets_improved: {payload['hard_subsets_improved']}",
            f"- bootstrap_claim_level: {payload['bootstrap_claim_level']}",
            f"- utility_improved: {payload['utility_improved']}",
            f"- true_ood_improved: {payload['true_ood_improved']}",
            f"- belief_not_worse_than_light_readout: {payload['belief_not_worse_than_light_readout']}",
            f"- next_step_choice: {payload['next_step_choice']}",
        ],
    )
    return payload


def parse_args() -> Any:
    parser = ArgumentParser(description="Promote STWM trace belief association as official readout and rerun final validation.")
    parser.add_argument("--mode", default="all", choices=["all", "audit_only", "bootstrap_decision_only"])
    parser.add_argument("--promotion-audit-report", default=str(REPORTS / "stwm_belief_promotion_audit_20260424.json"))
    parser.add_argument("--promotion-audit-doc", default=str(DOCS / "STWM_BELIEF_PROMOTION_AUDIT_20260424.md"))
    parser.add_argument("--final-eval-report", default=str(REPORTS / "stwm_belief_final_eval_20260424.json"))
    parser.add_argument("--final-eval-doc", default=str(DOCS / "STWM_BELIEF_FINAL_EVAL_20260424.md"))
    parser.add_argument("--strict-bootstrap-report", default=str(REPORTS / "stwm_belief_strict_bootstrap_20260424.json"))
    parser.add_argument("--strict-bootstrap-doc", default=str(DOCS / "STWM_BELIEF_STRICT_BOOTSTRAP_20260424.md"))
    parser.add_argument("--downstream-utility-report", default=str(REPORTS / "stwm_belief_downstream_utility_20260424.json"))
    parser.add_argument("--downstream-utility-doc", default=str(DOCS / "STWM_BELIEF_DOWNSTREAM_UTILITY_20260424.md"))
    parser.add_argument("--true-ood-eval-report", default=str(REPORTS / "stwm_belief_true_ood_eval_20260424.json"))
    parser.add_argument("--true-ood-eval-doc", default=str(DOCS / "STWM_BELIEF_TRUE_OOD_EVAL_20260424.md"))
    parser.add_argument("--final-decision-report", default=str(REPORTS / "stwm_belief_final_decision_20260424.json"))
    parser.add_argument("--final-decision-doc", default=str(DOCS / "STWM_BELIEF_FINAL_DECISION_20260424.md"))
    parser.add_argument("--dense-protocol-json", default=str(REPORTS / "stage2_state_identifiability_protocol_v3_20260416.json"))
    parser.add_argument("--extended-protocol-json", default=str(REPORTS / "stage2_protocol_v3_extended_evalset_20260420.json"))
    parser.add_argument("--source-shards", default=",".join([
        str(REPORTS / "trace_conditioned_readout_shards_20260423/tusb_all_fixed.json"),
        str(REPORTS / "trace_conditioned_readout_shards_20260423/legacysem.json"),
        str(REPORTS / "trace_conditioned_readout_shards_20260423/calibration.json"),
        str(REPORTS / "trace_conditioned_readout_shards_20260423/cropenc.json"),
    ]))
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--lease-path", default=str(REPORTS / "stage1_v2_gpu_lease_20260408.json"))
    parser.add_argument("--eval-required-mem-gb", type=float, default=12.0)
    parser.add_argument("--eval-safety-margin-gb", type=float, default=4.0)
    parser.add_argument("--audit-sample-items", type=int, default=64)
    return parser.parse_args()


def main() -> None:
    evalcore._apply_process_title_normalization()
    args = parse_args()
    if args.mode == "audit_only":
        feasibility = beliefcore._build_feasibility_audit(args)
        _build_promotion_audit(args, feasibility)
        return
    if args.mode == "bootstrap_decision_only":
        final_eval = _load_json(Path(args.final_eval_report))
        audit = _load_json(Path(args.promotion_audit_report))
    else:
        final_eval, audit = _build_final_eval(args)
    bootstrap = _build_strict_bootstrap(args, final_eval)
    utility = _build_downstream_utility(args)
    ood = _build_ood_eval(args, final_eval)
    _build_decision(args, audit, final_eval, bootstrap, utility, ood)


if __name__ == "__main__":
    main()
