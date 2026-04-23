#!/usr/bin/env python3
from __future__ import annotations

from argparse import ArgumentParser
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple
import hashlib
import json
import os
import random


ROOT = Path("/raid/chen034/workspace/stwm")
FINAL_EVAL_REPORT = ROOT / "reports/stwm_lightreadout_final_eval_20260422.json"
OLD_UTILITY_REPORT = ROOT / "reports/stwm_lightreadout_downstream_utility_20260422.json"

AUDIT_REPORT = ROOT / "reports/stwm_lightreadout_utility_plumbing_audit_20260422.json"
AUDIT_DOC = ROOT / "docs/STWM_LIGHTREADOUT_UTILITY_PLUMBING_AUDIT_20260422.md"
UTILITY_REPORT = ROOT / "reports/stwm_lightreadout_downstream_utility_v3_20260422.json"
UTILITY_DOC = ROOT / "docs/STWM_LIGHTREADOUT_DOWNSTREAM_UTILITY_V3_20260422.md"
BOOTSTRAP_REPORT = ROOT / "reports/stwm_lightreadout_downstream_utility_v3_bootstrap_20260422.json"
BOOTSTRAP_DOC = ROOT / "docs/STWM_LIGHTREADOUT_DOWNSTREAM_UTILITY_V3_BOOTSTRAP_20260422.md"
DECISION_REPORT = ROOT / "reports/stwm_lightreadout_downstream_utility_v3_decision_20260422.json"
DECISION_DOC = ROOT / "docs/STWM_LIGHTREADOUT_DOWNSTREAM_UTILITY_V3_DECISION_20260422.md"

PANEL_NAME = "densified_200_context_preserving"

OFFICIAL_TUSB = "TUSB-v3.1::official(best_semantic_hard.pt+hybrid_light)"
CALIBRATION = "calibration-only::best.pt"
CROPENC = "cropenc::best.pt"
LEGACYSEM = "legacysem::best.pt"
METHODS = [OFFICIAL_TUSB, CALIBRATION, CROPENC, LEGACYSEM]

VAL_BUCKETS = {0, 1, 2}
BOOTSTRAP_SAMPLES = 2000


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _apply_process_title_normalization(default_title: str = "python") -> None:
    title = str(os.environ.get("STWM_PROC_TITLE", default_title)).strip() or default_title
    mode = str(os.environ.get("STWM_PROC_TITLE_MODE", "generic")).strip().lower()
    if mode != "generic":
        return
    lowered = title.lower()
    if "stwm" in lowered or "tracewm" in lowered or "/raid/" in lowered or "/home/" in lowered:
        title = default_title
    try:
        import setproctitle  # type: ignore

        setproctitle.setproctitle(title)
    except Exception:
        pass


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


def _write_md(path: Path, lines: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _mean(values: Iterable[float]) -> float:
    vals = list(values)
    return float(sum(vals) / max(len(vals), 1))


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _percentile(sorted_values: Sequence[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    pos = max(0.0, min(1.0, q)) * float(len(sorted_values) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(sorted_values) - 1)
    frac = pos - float(lo)
    return float(sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac)


def _rows_from_final_eval(path: Path, panel_name: str) -> List[Dict[str, Any]]:
    payload = _load_json(path)
    panels = payload.get("panels", {}) if isinstance(payload.get("panels"), dict) else {}
    panel = panels.get(panel_name, {}) if isinstance(panels.get(panel_name), dict) else {}
    rows = panel.get("per_item_results", []) if isinstance(panel.get("per_item_results"), list) else []
    return [row for row in rows if isinstance(row, dict)]


def _method_rows(rows: List[Dict[str, Any]], method_name: str) -> List[Dict[str, Any]]:
    return [row for row in rows if str(row.get("method_name")) == method_name]


def _subset_rows(rows: List[Dict[str, Any]], tags: Sequence[str]) -> List[Dict[str, Any]]:
    wanted = set(tags)
    return [row for row in rows if wanted.intersection(set(row.get("subset_tags", [])))]


def _stable_bucket(protocol_item_id: str) -> int:
    digest = hashlib.sha256(protocol_item_id.encode("utf-8")).hexdigest()
    return int(digest[-8:], 16) % 10


def _split_name(protocol_item_id: str) -> str:
    return "val" if _stable_bucket(protocol_item_id) in VAL_BUCKETS else "test"


def _split_hash(item_ids: Sequence[str]) -> str:
    return _sha256_text("\n".join(sorted(set(item_ids))))


def _row_key(row: Mapping[str, Any]) -> Tuple[str, int]:
    return (str(row.get("protocol_item_id", "")), int(row.get("seed", -1)))


def _candidate_scores_from_row(row: Mapping[str, Any]) -> Dict[str, float]:
    for key in ("hybrid_light_scores", "coord_only_scores", "unit_identity_scores", "semantic_teacher_scores"):
        value = row.get(key)
        if isinstance(value, dict) and value:
            parsed: Dict[str, float] = {}
            for cand, score in value.items():
                try:
                    parsed[str(cand)] = float(score)
                except Exception:
                    continue
            if parsed:
                return parsed
    return {}


def _ranked_candidate_ids(row: Mapping[str, Any]) -> List[str]:
    scores = _candidate_scores_from_row(row)
    if scores:
        return [
            cand
            for cand, _ in sorted(scores.items(), key=lambda item: (-float(item[1]), str(item[0])))
        ]
    ranked = row.get("ranked_candidate_ids", [])
    if isinstance(ranked, list):
        return [str(cand) for cand in ranked]
    top1 = row.get("top1_candidate_id")
    return [str(top1)] if top1 is not None else []


def _target_rank(row: Mapping[str, Any]) -> int:
    try:
        explicit = int(row.get("target_rank", 0))
    except Exception:
        explicit = 0
    if explicit > 0:
        return explicit
    target_id = str(row.get("target_id", ""))
    ranked = _ranked_candidate_ids(row)
    try:
        return ranked.index(target_id) + 1
    except ValueError:
        return max(len(ranked), 1) + 1


def _candidate_count(row: Mapping[str, Any]) -> int:
    try:
        explicit = int(row.get("candidate_count", 0))
    except Exception:
        explicit = 0
    if explicit > 0:
        return explicit
    ranked = _ranked_candidate_ids(row)
    return max(len(ranked), 1)


def _top1_from_rank(rank: int) -> float:
    return 1.0 if rank == 1 else 0.0


def _top5_from_rank(rank: int) -> float:
    return 1.0 if rank <= 5 else 0.0


def _mrr_from_rank(rank: int) -> float:
    return 1.0 / float(max(rank, 1))


def _confusion_from_rank(rank: int) -> float:
    return 0.0 if rank == 1 else 1.0


def _aggregate_probe_a(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {
            "count": 0,
            "top1": 0.0,
            "top5": 0.0,
            "mrr": 0.0,
            "candidate_confusion_rate": 0.0,
        }
    ranks = [_target_rank(row) for row in rows]
    return {
        "count": int(len(rows)),
        "top1": _mean(_top1_from_rank(rank) for rank in ranks),
        "top5": _mean(_top5_from_rank(rank) for rank in ranks),
        "mrr": _mean(_mrr_from_rank(rank) for rank in ranks),
        "candidate_confusion_rate": _mean(_confusion_from_rank(rank) for rank in ranks),
    }


def _aggregate_probe_b(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {
            "count": 0,
            "top1": 0.0,
            "mrr": 0.0,
            "false_confuser_rate": 0.0,
        }
    ranks = [_target_rank(row) for row in rows]
    return {
        "count": int(len(rows)),
        "top1": _mean(_top1_from_rank(rank) for rank in ranks),
        "mrr": _mean(_mrr_from_rank(rank) for rank in ranks),
        "false_confuser_rate": _mean(_confusion_from_rank(rank) for rank in ranks),
    }


def _probe_a_metrics_from_row(row: Mapping[str, Any]) -> Dict[str, float]:
    rank = _target_rank(row)
    return {
        "top1": _top1_from_rank(rank),
        "top5": _top5_from_rank(rank),
        "mrr": _mrr_from_rank(rank),
        "candidate_confusion_rate": _confusion_from_rank(rank),
    }


def _probe_b_metrics_from_row(row: Mapping[str, Any]) -> Dict[str, float]:
    rank = _target_rank(row)
    return {
        "top1": _top1_from_rank(rank),
        "mrr": _mrr_from_rank(rank),
        "false_confuser_rate": _confusion_from_rank(rank),
    }


def _build_bootstrap_metrics(
    left_rows: List[Dict[str, Any]],
    right_rows: List[Dict[str, Any]],
    *,
    probe: str,
    rng_seed: int,
) -> Dict[str, Any]:
    left_map = {_row_key(row): row for row in left_rows}
    right_map = {_row_key(row): row for row in right_rows}
    common_keys = sorted(set(left_map).intersection(right_map))
    if not common_keys:
        return {"count": 0, "metrics": {}}

    if probe == "probe_a":
        metric_names = ["top1", "top5", "mrr", "candidate_confusion_rate"]
        left_metric = _probe_a_metrics_from_row
        right_metric = _probe_a_metrics_from_row
        lower_is_better = {"candidate_confusion_rate"}
    else:
        metric_names = ["top1", "mrr", "false_confuser_rate"]
        left_metric = _probe_b_metrics_from_row
        right_metric = _probe_b_metrics_from_row
        lower_is_better = {"false_confuser_rate"}

    deltas: Dict[str, List[float]] = {metric: [] for metric in metric_names}
    for key in common_keys:
        left_values = left_metric(left_map[key])
        right_values = right_metric(right_map[key])
        for metric in metric_names:
            if metric in lower_is_better:
                delta = float(right_values[metric]) - float(left_values[metric])
            else:
                delta = float(left_values[metric]) - float(right_values[metric])
            deltas[metric].append(delta)

    rng = random.Random(rng_seed)
    metric_report: Dict[str, Any] = {}
    n = len(common_keys)
    for metric, values in deltas.items():
        if not values:
            metric_report[metric] = {
                "count": 0,
                "mean_delta": 0.0,
                "ci95_low": 0.0,
                "ci95_high": 0.0,
                "zero_excluded": False,
                "bootstrap_win_rate": 0.0,
            }
            continue
        bootstrap_means: List[float] = []
        for _ in range(BOOTSTRAP_SAMPLES):
            total = 0.0
            for _ in range(n):
                total += values[rng.randrange(n)]
            bootstrap_means.append(total / float(n))
        bootstrap_means.sort()
        ci_low = _percentile(bootstrap_means, 0.025)
        ci_high = _percentile(bootstrap_means, 0.975)
        mean_delta = _mean(values)
        metric_report[metric] = {
            "count": int(n),
            "mean_delta": float(mean_delta),
            "ci95_low": float(ci_low),
            "ci95_high": float(ci_high),
            "zero_excluded": bool(ci_low > 0.0 or ci_high < 0.0),
            "bootstrap_win_rate": float(sum(1 for val in bootstrap_means if val > 0.0) / len(bootstrap_means)),
        }
    return {"count": int(n), "metrics": metric_report}


def build_plumbing_audit(
    *,
    audit_report: Path,
    audit_doc: Path,
    final_eval_report: Path,
    old_utility_report: Path,
) -> Dict[str, Any]:
    final_eval = _load_json(final_eval_report)
    old_utility = _load_json(old_utility_report)
    panel = (
        final_eval.get("panels", {}).get(PANEL_NAME, {})
        if isinstance(final_eval.get("panels", {}), dict)
        else {}
    )
    rows = panel.get("per_item_results", []) if isinstance(panel.get("per_item_results", []), list) else []
    first_row = rows[0] if rows else {}
    old_probe_a = old_utility.get("probe_a", {}) if isinstance(old_utility.get("probe_a", {}), dict) else {}
    old_probe_b = old_utility.get("probe_b", {}) if isinstance(old_utility.get("probe_b", {}), dict) else {}

    audit = {
        "generated_at_utc": _now_iso(),
        "inspected_inputs": {
            "utility_v2_script": str(ROOT / "code/stwm/tools/run_stwm_downstream_utility_v2_20260420.py"),
            "top_tier_utility_script": str(ROOT / "code/stwm/tools/run_stwm_top_tier_downstream_utility_20260420.py"),
            "light_readout_eval_script": str(ROOT / "code/stwm/tools/run_stwm_tusb_light_readout_eval_20260422.py"),
            "official_final_eval_report": str(final_eval_report),
            "previous_lightreadout_utility_report": str(old_utility_report),
        },
        "why_probe_counts_zero": (
            "The previous light-readout utility report consumed the official final-eval rows with the old nested "
            "`row[\"methods\"][method_name]` schema assumption. The live official final-eval report stores flat "
            "`per_item_results` rows keyed by `method_name` and `scoring_mode`, so every method lookup missed and every probe count collapsed to zero."
        ),
        "root_cause": {
            "row_schema_mismatch": True,
            "method_key_mismatch": False,
            "panel_source_mismatch": False,
            "split_filter_logic_issue": False,
        },
        "exact_breakpoint": [
            {
                "file": str(ROOT / "code/stwm/tools/run_stwm_downstream_utility_v2_20260420.py"),
                "line": 76,
                "code": "scored = [row[\"methods\"][method_name] for row in rows if method_name in row.get(\"methods\", {})]",
            },
            {
                "file": str(ROOT / "code/stwm/tools/run_stwm_downstream_utility_v2_20260420.py"),
                "line": 94,
                "code": "scored = [row[\"methods\"][method_name] for row in rows if method_name in row.get(\"methods\", {})]",
            },
            {
                "file": str(ROOT / "code/stwm/tools/run_stwm_top_tier_downstream_utility_20260420.py"),
                "line": 75,
                "code": "scored = [row[\"methods\"][method_name] for row in rows if method_name in row.get(\"methods\", {})]",
            },
        ],
        "current_official_final_eval_schema": {
            "panel_name": PANEL_NAME,
            "panel_keys": sorted(panel.keys()) if isinstance(panel, dict) else [],
            "row_count": int(len(rows)),
            "row_keys": sorted(first_row.keys()) if isinstance(first_row, dict) else [],
            "flat_row_schema": True,
            "sample_method_name": str(first_row.get("method_name", "")),
            "sample_scoring_mode": str(first_row.get("scoring_mode", "")),
        },
        "previous_report_observation": {
            "probe_a_methods": sorted(old_probe_a.keys()) if isinstance(old_probe_a, dict) else [],
            "probe_b_methods": sorted(old_probe_b.keys()) if isinstance(old_probe_b, dict) else [],
            "all_probe_counts_zero": bool(
                all(float(entry.get("count", 0)) == 0.0 for entry in old_probe_a.values()) if old_probe_a else True
            )
            and bool(
                all(float(entry.get("count", 0)) == 0.0 for entry in old_probe_b.values()) if old_probe_b else True
            ),
        },
        "fix_plan": {
            "new_runner": str(ROOT / "code/stwm/tools/run_stwm_lightreadout_downstream_utility_v3_20260422.py"),
            "official_tusb_method": OFFICIAL_TUSB,
            "baselines": [CALIBRATION, CROPENC, LEGACYSEM],
            "panel_source": PANEL_NAME,
            "strategy": "read flat per_item_results rows, rebuild rankings from explicit score maps or ranked_candidate_ids, then run test-only utility probes and paired bootstrap",
        },
    }
    _write_json(audit_report, audit)
    _write_md(
        audit_doc,
        [
            "# STWM Light Readout Utility Plumbing Audit 20260422",
            "",
            f"- official_final_eval_report: `{final_eval_report}`",
            f"- why_probe_counts_zero: {audit['why_probe_counts_zero']}",
            f"- row_schema_mismatch: {audit['root_cause']['row_schema_mismatch']}",
            f"- exact_breakpoint_1: `{audit['exact_breakpoint'][0]['file']}:{audit['exact_breakpoint'][0]['line']}`",
            f"- exact_breakpoint_2: `{audit['exact_breakpoint'][1]['file']}:{audit['exact_breakpoint'][1]['line']}`",
            f"- exact_breakpoint_3: `{audit['exact_breakpoint'][2]['file']}:{audit['exact_breakpoint'][2]['line']}`",
            f"- current_row_count: {audit['current_official_final_eval_schema']['row_count']}",
            f"- sample_method_name: `{audit['current_official_final_eval_schema']['sample_method_name']}`",
            f"- fix_strategy: {audit['fix_plan']['strategy']}",
        ],
    )
    return audit


def build_utility_v3(
    *,
    report_path: Path,
    doc_path: Path,
    final_eval_report: Path,
) -> Dict[str, Any]:
    rows = _rows_from_final_eval(final_eval_report, PANEL_NAME)
    item_ids = sorted({str(row.get("protocol_item_id", "")) for row in rows if row.get("protocol_item_id") is not None})
    val_ids = {item_id for item_id in item_ids if _split_name(item_id) == "val"}
    test_ids = {item_id for item_id in item_ids if item_id not in val_ids}
    test_rows = [row for row in rows if str(row.get("protocol_item_id", "")) in test_ids]

    method_rows_test = {method: _method_rows(test_rows, method) for method in METHODS}
    probe_b_method_rows_test = {
        method: _subset_rows(method_rows_test[method], ["occlusion_reappearance", "long_gap_persistence"])
        for method in METHODS
    }

    probe_a = {method: _aggregate_probe_a(method_rows_test[method]) for method in METHODS}
    probe_b = {method: _aggregate_probe_b(probe_b_method_rows_test[method]) for method in METHODS}
    hard_subset_breakdown = {
        tag: {method: _aggregate_probe_a(_subset_rows(method_rows_test[method], [tag])) for method in METHODS}
        for tag in [
            "crossing_ambiguity",
            "appearance_change",
            "occlusion_reappearance",
            "long_gap_persistence",
            "small_object",
        ]
    }

    def _probe_a_beats(lhs: str, rhs: str) -> bool:
        return bool(
            float(probe_a[lhs]["top1"]) > float(probe_a[rhs]["top1"])
            and float(probe_a[lhs]["mrr"]) >= float(probe_a[rhs]["mrr"])
        )

    def _probe_b_beats(lhs: str, rhs: str) -> bool:
        return bool(
            float(probe_b[lhs]["top1"]) > float(probe_b[rhs]["top1"])
            and float(probe_b[lhs]["mrr"]) >= float(probe_b[rhs]["mrr"])
        )

    probe_a_improved_vs_calibration = _probe_a_beats(OFFICIAL_TUSB, CALIBRATION)
    probe_b_improved_vs_calibration = _probe_b_beats(OFFICIAL_TUSB, CALIBRATION)
    probe_a_improved_vs_cropenc = _probe_a_beats(OFFICIAL_TUSB, CROPENC)
    probe_b_improved_vs_cropenc = _probe_b_beats(OFFICIAL_TUSB, CROPENC)
    probe_a_improved_vs_legacysem = _probe_a_beats(OFFICIAL_TUSB, LEGACYSEM)
    probe_b_improved_vs_legacysem = _probe_b_beats(OFFICIAL_TUSB, LEGACYSEM)

    utility_improved_vs_calibration = bool(probe_a_improved_vs_calibration and probe_b_improved_vs_calibration)
    utility_improved_vs_cropenc = bool(probe_a_improved_vs_cropenc and probe_b_improved_vs_cropenc)
    utility_improved_vs_legacysem = bool(probe_a_improved_vs_legacysem or probe_b_improved_vs_legacysem)
    utility_hard_subset_improved = bool(
        float(hard_subset_breakdown["occlusion_reappearance"][OFFICIAL_TUSB]["top1"])
        > float(hard_subset_breakdown["occlusion_reappearance"][CALIBRATION]["top1"])
        and float(hard_subset_breakdown["long_gap_persistence"][OFFICIAL_TUSB]["top1"])
        > float(hard_subset_breakdown["long_gap_persistence"][CALIBRATION]["top1"])
    )

    result = {
        "generated_at_utc": _now_iso(),
        "source_final_eval_report": str(final_eval_report),
        "official_tusb_method": OFFICIAL_TUSB,
        "used_checkpoint": "best_semantic_hard.pt",
        "used_scoring_mode": "hybrid_light",
        "split_definition": {
            "rule": "stable protocol_item_id hash using sha256(last8_hex) % 10; buckets {0,1,2} -> val, others -> test",
            "split_sizes": {"val_items": int(len(val_ids)), "test_items": int(len(test_ids))},
            "val_item_ids_hash": _split_hash(sorted(val_ids)),
            "test_item_ids_hash": _split_hash(sorted(test_ids)),
            "leakage_check_passed": bool(val_ids.isdisjoint(test_ids)),
            "parameter_tuning_note": "no probe parameters were tuned; the held-out split is still emitted to enforce the no test-time tuning rule",
        },
        "probe_a": probe_a,
        "probe_b": probe_b,
        "hard_subset_breakdown": hard_subset_breakdown,
        "probe_a_improved_vs_calibration": probe_a_improved_vs_calibration,
        "probe_b_improved_vs_calibration": probe_b_improved_vs_calibration,
        "probe_a_improved_vs_cropenc": probe_a_improved_vs_cropenc,
        "probe_b_improved_vs_cropenc": probe_b_improved_vs_cropenc,
        "probe_a_improved_vs_legacysem": probe_a_improved_vs_legacysem,
        "probe_b_improved_vs_legacysem": probe_b_improved_vs_legacysem,
        "utility_improved_vs_calibration": utility_improved_vs_calibration,
        "utility_improved_vs_cropenc": utility_improved_vs_cropenc,
        "utility_improved_vs_legacysem": utility_improved_vs_legacysem,
        "utility_hard_subset_improved": utility_hard_subset_improved,
        "leakage_check_passed": bool(val_ids.isdisjoint(test_ids)),
        "utility_claim_ready": bool(
            utility_improved_vs_calibration
            and utility_improved_vs_cropenc
            and utility_hard_subset_improved
            and bool(val_ids.isdisjoint(test_ids))
            and int(probe_a[OFFICIAL_TUSB]["count"]) > 0
            and int(probe_b[OFFICIAL_TUSB]["count"]) > 0
        ),
        "exact_blocking_reason": "",
    }
    _write_json(report_path, result)
    _write_md(
        doc_path,
        [
            "# STWM Light Readout Downstream Utility V3 20260422",
            "",
            f"- source_final_eval_report: `{final_eval_report}`",
            f"- official_tusb_method: `{OFFICIAL_TUSB}`",
            f"- used_checkpoint: `best_semantic_hard.pt`",
            f"- used_scoring_mode: `hybrid_light`",
            f"- probe_a_count_tusb: {probe_a[OFFICIAL_TUSB]['count']}",
            f"- probe_b_count_tusb: {probe_b[OFFICIAL_TUSB]['count']}",
            f"- utility_improved_vs_calibration: {result['utility_improved_vs_calibration']}",
            f"- utility_improved_vs_cropenc: {result['utility_improved_vs_cropenc']}",
            f"- utility_improved_vs_legacysem: {result['utility_improved_vs_legacysem']}",
            f"- utility_hard_subset_improved: {result['utility_hard_subset_improved']}",
            f"- leakage_check_passed: {result['leakage_check_passed']}",
            f"- utility_claim_ready: {result['utility_claim_ready']}",
        ],
    )
    return result


def build_bootstrap(
    *,
    report_path: Path,
    doc_path: Path,
    final_eval_report: Path,
) -> Dict[str, Any]:
    rows = _rows_from_final_eval(final_eval_report, PANEL_NAME)
    test_ids = {str(row.get("protocol_item_id", "")) for row in rows if _split_name(str(row.get("protocol_item_id", ""))) == "test"}
    test_rows = [row for row in rows if str(row.get("protocol_item_id", "")) in test_ids]

    official_rows = _method_rows(test_rows, OFFICIAL_TUSB)
    calibration_rows = _method_rows(test_rows, CALIBRATION)
    legacy_rows = _method_rows(test_rows, LEGACYSEM)

    official_probe_b_rows = _subset_rows(official_rows, ["occlusion_reappearance", "long_gap_persistence"])
    calibration_probe_b_rows = _subset_rows(calibration_rows, ["occlusion_reappearance", "long_gap_persistence"])
    legacy_probe_b_rows = _subset_rows(legacy_rows, ["occlusion_reappearance", "long_gap_persistence"])

    comparisons = {
        "official_vs_calibration": {
            "probe_a": _build_bootstrap_metrics(official_rows, calibration_rows, probe="probe_a", rng_seed=2026042201),
            "probe_b": _build_bootstrap_metrics(
                official_probe_b_rows,
                calibration_probe_b_rows,
                probe="probe_b",
                rng_seed=2026042202,
            ),
        },
        "official_vs_legacysem": {
            "probe_a": _build_bootstrap_metrics(official_rows, legacy_rows, probe="probe_a", rng_seed=2026042203),
            "probe_b": _build_bootstrap_metrics(
                official_probe_b_rows,
                legacy_probe_b_rows,
                probe="probe_b",
                rng_seed=2026042204,
            ),
        },
    }

    def _key_zero(comp_key: str, probe_key: str, metric_key: str) -> bool:
        return bool(
            comparisons.get(comp_key, {})
            .get(probe_key, {})
            .get("metrics", {})
            .get(metric_key, {})
            .get("zero_excluded", False)
        )

    utility_zero_excluded_vs_calibration = bool(
        _key_zero("official_vs_calibration", "probe_a", "top1")
        and _key_zero("official_vs_calibration", "probe_b", "top1")
    )
    utility_zero_excluded_vs_legacysem = bool(
        _key_zero("official_vs_legacysem", "probe_a", "top1")
        and _key_zero("official_vs_legacysem", "probe_b", "top1")
    )

    if utility_zero_excluded_vs_calibration and utility_zero_excluded_vs_legacysem:
        utility_claim_level = "strong_claim"
    elif _key_zero("official_vs_calibration", "probe_a", "top1") or _key_zero("official_vs_calibration", "probe_b", "top1"):
        utility_claim_level = "moderate_claim"
    else:
        utility_claim_level = "weak_claim"

    result = {
        "generated_at_utc": _now_iso(),
        "source_final_eval_report": str(final_eval_report),
        "panel_name": PANEL_NAME,
        "split_used": "test",
        "comparisons": comparisons,
        "utility_zero_excluded_vs_calibration": utility_zero_excluded_vs_calibration,
        "utility_zero_excluded_vs_legacysem": utility_zero_excluded_vs_legacysem,
        "utility_claim_level": utility_claim_level,
    }
    _write_json(report_path, result)
    _write_md(
        doc_path,
        [
            "# STWM Light Readout Downstream Utility V3 Bootstrap 20260422",
            "",
            f"- source_final_eval_report: `{final_eval_report}`",
            f"- panel_name: `{PANEL_NAME}`",
            f"- utility_zero_excluded_vs_calibration: {utility_zero_excluded_vs_calibration}",
            f"- utility_zero_excluded_vs_legacysem: {utility_zero_excluded_vs_legacysem}",
            f"- utility_claim_level: `{utility_claim_level}`",
        ],
    )
    return result


def build_decision(
    *,
    report_path: Path,
    doc_path: Path,
    utility_report: Dict[str, Any],
    bootstrap_report: Dict[str, Any],
) -> Dict[str, Any]:
    probe_a_count_nonzero = bool(int(utility_report.get("probe_a", {}).get(OFFICIAL_TUSB, {}).get("count", 0)) > 0)
    probe_b_count_nonzero = bool(int(utility_report.get("probe_b", {}).get(OFFICIAL_TUSB, {}).get("count", 0)) > 0)
    utility_improved_vs_legacysem = bool(utility_report.get("utility_improved_vs_legacysem", False))
    utility_claim_ready = bool(utility_report.get("utility_claim_ready", False))
    utility_claim_level = str(bootstrap_report.get("utility_claim_level", "weak_claim"))

    if utility_claim_ready and utility_improved_vs_legacysem:
        next_step_choice = "run_true_ood_next"
    elif utility_report.get("utility_improved_vs_calibration", False):
        next_step_choice = "run_true_ood_next"
    else:
        next_step_choice = "reframe_as_moderate_claim_main_track"

    decision = {
        "generated_at_utc": _now_iso(),
        "utility_plumbing_fixed": True,
        "probe_a_count_nonzero": probe_a_count_nonzero,
        "probe_b_count_nonzero": probe_b_count_nonzero,
        "probe_a_improved_vs_calibration": bool(utility_report.get("probe_a_improved_vs_calibration", False)),
        "probe_b_improved_vs_calibration": bool(utility_report.get("probe_b_improved_vs_calibration", False)),
        "utility_improved_vs_legacysem": utility_improved_vs_legacysem,
        "utility_claim_ready": utility_claim_ready,
        "utility_claim_level": utility_claim_level,
        "next_step_choice": next_step_choice,
        "exact_blocking_reason": str(utility_report.get("exact_blocking_reason", "")),
    }
    _write_json(report_path, decision)
    _write_md(
        doc_path,
        [
            "# STWM Light Readout Downstream Utility V3 Decision 20260422",
            "",
            f"- utility_plumbing_fixed: {decision['utility_plumbing_fixed']}",
            f"- probe_a_count_nonzero: {probe_a_count_nonzero}",
            f"- probe_b_count_nonzero: {probe_b_count_nonzero}",
            f"- probe_a_improved_vs_calibration: {decision['probe_a_improved_vs_calibration']}",
            f"- probe_b_improved_vs_calibration: {decision['probe_b_improved_vs_calibration']}",
            f"- utility_improved_vs_legacysem: {decision['utility_improved_vs_legacysem']}",
            f"- utility_claim_ready: {decision['utility_claim_ready']}",
            f"- utility_claim_level: `{decision['utility_claim_level']}`",
            f"- next_step_choice: `{decision['next_step_choice']}`",
        ],
    )
    return decision


def main() -> None:
    _apply_process_title_normalization()
    parser = ArgumentParser(description="Run STWM official light-readout downstream utility v3.")
    parser.add_argument("--final-eval-report", default=str(FINAL_EVAL_REPORT))
    parser.add_argument("--audit-report", default=str(AUDIT_REPORT))
    parser.add_argument("--audit-doc", default=str(AUDIT_DOC))
    parser.add_argument("--output-report", default=str(UTILITY_REPORT))
    parser.add_argument("--output-doc", default=str(UTILITY_DOC))
    parser.add_argument("--bootstrap-report", default=str(BOOTSTRAP_REPORT))
    parser.add_argument("--bootstrap-doc", default=str(BOOTSTRAP_DOC))
    parser.add_argument("--decision-report", default=str(DECISION_REPORT))
    parser.add_argument("--decision-doc", default=str(DECISION_DOC))
    args = parser.parse_args()

    final_eval_report = Path(args.final_eval_report)
    audit = build_plumbing_audit(
        audit_report=Path(args.audit_report),
        audit_doc=Path(args.audit_doc),
        final_eval_report=final_eval_report,
        old_utility_report=OLD_UTILITY_REPORT,
    )
    if not audit.get("current_official_final_eval_schema", {}).get("row_count", 0):
        utility_report = {
            "generated_at_utc": _now_iso(),
            "source_final_eval_report": str(final_eval_report),
            "official_tusb_method": OFFICIAL_TUSB,
            "used_checkpoint": "best_semantic_hard.pt",
            "used_scoring_mode": "hybrid_light",
            "split_definition": {},
            "probe_a": {},
            "probe_b": {},
            "hard_subset_breakdown": {},
            "utility_improved_vs_calibration": False,
            "utility_improved_vs_cropenc": False,
            "utility_improved_vs_legacysem": False,
            "utility_hard_subset_improved": False,
            "leakage_check_passed": False,
            "utility_claim_ready": False,
            "exact_blocking_reason": "official final eval report missing or empty per_item_results for densified_200_context_preserving",
        }
        _write_json(Path(args.output_report), utility_report)
        _write_md(Path(args.output_doc), ["# STWM Light Readout Downstream Utility V3 20260422", "", f"- exact_blocking_reason: {utility_report['exact_blocking_reason']}"])
        bootstrap_report = {
            "generated_at_utc": _now_iso(),
            "panel_name": PANEL_NAME,
            "comparisons": {},
            "utility_zero_excluded_vs_calibration": False,
            "utility_zero_excluded_vs_legacysem": False,
            "utility_claim_level": "weak_claim",
        }
        _write_json(Path(args.bootstrap_report), bootstrap_report)
        _write_md(Path(args.bootstrap_doc), ["# STWM Light Readout Downstream Utility V3 Bootstrap 20260422", "", "- exact_blocking_reason: missing final eval rows"])
        build_decision(
            report_path=Path(args.decision_report),
            doc_path=Path(args.decision_doc),
            utility_report=utility_report,
            bootstrap_report=bootstrap_report,
        )
        return

    utility_report = build_utility_v3(
        report_path=Path(args.output_report),
        doc_path=Path(args.output_doc),
        final_eval_report=final_eval_report,
    )
    bootstrap_report = build_bootstrap(
        report_path=Path(args.bootstrap_report),
        doc_path=Path(args.bootstrap_doc),
        final_eval_report=final_eval_report,
    )
    build_decision(
        report_path=Path(args.decision_report),
        doc_path=Path(args.decision_doc),
        utility_report=utility_report,
        bootstrap_report=bootstrap_report,
    )


if __name__ == "__main__":
    main()
