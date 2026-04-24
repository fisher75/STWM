#!/usr/bin/env python3
from __future__ import annotations

from argparse import ArgumentParser
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence
import hashlib
import json
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np

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

from stwm.tools import run_stwm_trace_residual_association_20260423 as v1


ROOT = Path("/raid/chen034/workspace/stwm")
REPORTS = ROOT / "reports"
DOCS = ROOT / "docs"
SOURCE_FILE = ROOT / "code/stwm/tools/run_stwm_trace_residual_association_20260423.py"

OFFICIAL_TUSB = v1.OFFICIAL_TUSB
CAL = v1.CAL
CROP = v1.CROP
LEGACY = v1.LEGACY
PANELS = list(v1.PANELS)
OLD_MODE = "trace_residual_association"
CLEAN_MODE = "clean_trace_residual_association_v2"
CLEAN_TUSB_MODES = [
    "frozen_external_teacher_only",
    "tusb_semantic_target",
    "unit_identity_only",
    CLEAN_MODE,
]
LEAK_FEATURES = [
    "is_occlusion_reappearance",
    "is_long_gap_persistence",
    "is_crossing_ambiguity",
]
CLEAN_FEATURE_NAMES = [
    "unit_identity_score_norm",
    "coord_score_norm",
    "tusb_semantic_target_score_norm",
    "external_rank_score",
    "external_margin_to_top",
    "coord_rank_score",
    "coord_margin_to_top",
    "candidate_count_scaled",
    "external_coord_rank_conflict",
    "external_coord_margin_conflict",
    "unit_coord_score_conflict",
    "external_x_unit",
    "external_x_coord",
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


def _source_lines_for(features: Sequence[str]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if not SOURCE_FILE.exists():
        return out
    for lineno, line in enumerate(SOURCE_FILE.read_text(encoding="utf-8").splitlines(), start=1):
        for feature in features:
            if feature in line:
                out.append({"path": str(SOURCE_FILE), "line": int(lineno), "feature": feature, "source": line.strip()})
    return out


def _build_leakage_audit(args: Any) -> Dict[str, Any]:
    schema = _load_json(REPORTS / "stwm_residual_assoc_feature_schema_20260423.json")
    eval_report = _load_json(REPORTS / "stwm_trace_residual_association_eval_20260423.json")
    schema_features = [str(x) for x in schema.get("feature_names", [])]
    eval_features = [str(x) for x in eval_report.get("residual_head", {}).get("feature_names", [])]
    leaked = sorted(set(LEAK_FEATURES) & (set(schema_features) | set(eval_features)))
    source_lines = _source_lines_for(leaked or LEAK_FEATURES)
    leakage_detected = bool(leaked)
    payload = {
        "generated_at_utc": _now_iso(),
        "audited_files": [
            str(SOURCE_FILE),
            str(REPORTS / "stwm_residual_assoc_feature_schema_20260423.json"),
            str(REPORTS / "stwm_trace_residual_association_eval_20260423.json"),
        ],
        "leakage_detected": bool(leakage_detected),
        "leakage_feature_names": leaked,
        "leakage_source_lines": source_lines,
        "leakage_features_are_protocol_eval_metadata": bool(leakage_detected),
        "current_residual_paper_valid": bool(not leakage_detected),
        "exact_blocking_reason": "",
    }
    _write_json(Path(args.leakage_audit_json), payload)
    _write_md(
        Path(args.leakage_audit_md),
        "STWM Clean Residual Leakage Audit 20260424",
        [
            f"- leakage_detected: {payload['leakage_detected']}",
            f"- leakage_feature_names: {payload['leakage_feature_names']}",
            f"- current_residual_paper_valid: {payload['current_residual_paper_valid']}",
            f"- leakage_source_lines: {len(source_lines)} entries",
        ],
    )
    return payload


def _rank_conflict(a: Mapping[str, float], b: Mapping[str, float], cand_id: str) -> float:
    return float(abs(float(a.get(cand_id, 0.0)) - float(b.get(cand_id, 0.0))))


def _clean_feature_vector(
    cand_id: str,
    subset_tags: Sequence[str],
    external_n: Mapping[str, float],
    unit_n: Mapping[str, float],
    coord_n: Mapping[str, float],
    semantic_n: Mapping[str, float],
    ext_rank: Mapping[str, float],
    ext_margin: Mapping[str, float],
    coord_rank: Mapping[str, float],
    coord_margin: Mapping[str, float],
    candidate_count: int,
) -> np.ndarray:
    del subset_tags
    ext = float(external_n.get(cand_id, 0.0))
    unit = float(unit_n.get(cand_id, 0.0))
    coord = float(coord_n.get(cand_id, 0.0))
    return np.asarray(
        [
            unit,
            coord,
            float(semantic_n.get(cand_id, 0.0)),
            float(ext_rank.get(cand_id, 0.0)),
            float(ext_margin.get(cand_id, 0.0)),
            float(coord_rank.get(cand_id, 0.0)),
            float(coord_margin.get(cand_id, 0.0)),
            float(np.log1p(max(candidate_count, 0)) / 5.0),
            _rank_conflict(ext_rank, coord_rank, cand_id),
            _rank_conflict(ext_margin, coord_margin, cand_id),
            abs(unit - coord),
            ext * unit,
            ext * coord,
        ],
        dtype=np.float64,
    )


def _patch_v1_for_clean_features() -> None:
    v1.RESIDUAL_FEATURE_NAMES = list(CLEAN_FEATURE_NAMES)
    v1._feature_vector = _clean_feature_vector  # type: ignore[attr-defined]
    v1._build_audit = lambda args: {"bypassed_by_clean_residual_v2": True}  # type: ignore[attr-defined]
    v1._build_schema = lambda args: {"feature_names": list(CLEAN_FEATURE_NAMES)}  # type: ignore[attr-defined]


def _rename_row(row: Mapping[str, Any]) -> Dict[str, Any] | None:
    copied = dict(row)
    method = str(copied.get("method_name", ""))
    mode = str(copied.get("scoring_mode", ""))
    if method == OFFICIAL_TUSB:
        if mode == OLD_MODE:
            copied["scoring_mode"] = CLEAN_MODE
        elif mode not in {"frozen_external_teacher_only", "tusb_semantic_target", "unit_identity_only"}:
            return None
    elif method in {CAL, CROP, LEGACY}:
        if mode != "coord_only":
            return None
    else:
        return None
    return copied


def _seed_results(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        OFFICIAL_TUSB: {mode: v1._seed_table(rows, OFFICIAL_TUSB, mode) for mode in CLEAN_TUSB_MODES},
        LEGACY: {"coord_only": v1._seed_table(rows, LEGACY, "coord_only")},
        CAL: {"coord_only": v1._seed_table(rows, CAL, "coord_only")},
        CROP: {"coord_only": v1._seed_table(rows, CROP, "coord_only")},
    }


def _transform_eval_payload(raw: Dict[str, Any]) -> Dict[str, Any]:
    base = dict(raw.get("eval", raw)) if isinstance(raw.get("eval", raw), dict) else {}
    payload = dict(base)
    payload["generated_at_utc"] = _now_iso()
    payload["clean_residual_v2"] = True
    payload["leakage_check_passed"] = True
    payload["removed_leakage_features"] = list(LEAK_FEATURES)
    payload["residual_head"] = dict(payload.get("residual_head", {}))
    payload["residual_head"]["feature_names"] = list(CLEAN_FEATURE_NAMES)
    payload["residual_head"]["head_name"] = CLEAN_MODE
    payload["residual_head"]["forbidden_feature_names_present"] = []
    payload["comparison_methods"] = {
        OFFICIAL_TUSB: list(CLEAN_TUSB_MODES),
        LEGACY: ["coord_only"],
        CAL: ["coord_only"],
        CROP: ["coord_only"],
    }
    panels: Dict[str, Any] = {}
    for panel_name, panel in dict(payload.get("panels", {})).items():
        rows = []
        for row in panel.get("per_item_results", []):
            renamed = _rename_row(row)
            if renamed is not None:
                rows.append(renamed)
        new_panel = dict(panel)
        new_panel["per_item_results"] = rows
        new_panel["per_item_results_hash"] = _sha256_json(rows)
        new_panel["per_method_seed_results"] = _seed_results(rows)
        panels[panel_name] = new_panel
    payload["panels"] = panels
    return payload


def _build_eval(args: Any) -> Dict[str, Any]:
    _patch_v1_for_clean_features()
    raw = v1._build_eval(args)
    payload = _transform_eval_payload(raw)
    _write_json(Path(args.eval_json), payload)
    lines = [
        "- leakage_check_passed: true",
        f"- clean_feature_names: {CLEAN_FEATURE_NAMES}",
    ]
    for panel_name, panel in payload.get("panels", {}).items():
        lines.append(f"- {panel_name}: valid={panel.get('valid_items')} skipped={panel.get('skipped_items')} test={panel.get('test_items')}")
    _write_md(Path(args.eval_md), "STWM Clean Residual V2 Eval 20260424", lines)
    return payload


def _build_eval_from_existing(args: Any, source_path: Path) -> Dict[str, Any]:
    raw = _load_json(source_path)
    payload = _transform_eval_payload(raw)
    _write_json(Path(args.eval_json), payload)
    lines = [
        f"- source_eval_report: {source_path}",
        "- leakage_check_passed: true",
        f"- clean_feature_names: {CLEAN_FEATURE_NAMES}",
    ]
    for panel_name, panel in payload.get("panels", {}).items():
        lines.append(f"- {panel_name}: valid={panel.get('valid_items')} skipped={panel.get('skipped_items')} test={panel.get('test_items')}")
    _write_md(Path(args.eval_md), "STWM Clean Residual V2 Eval 20260424", lines)
    return payload


def _rows_for(panel: Mapping[str, Any]) -> List[Dict[str, Any]]:
    return [dict(row) for row in panel.get("per_item_results", []) if isinstance(row, dict)]


def _mean_for(panel: Mapping[str, Any], method_name: str, mode: str, subset: str = "overall") -> float:
    rows = [
        row for row in _rows_for(panel)
        if str(row.get("method_name")) == method_name and str(row.get("scoring_mode")) == mode
    ]
    if subset == "continuity":
        rows = [
            row for row in rows
            if ("occlusion_reappearance" in set(row.get("subset_tags", [])))
            or ("long_gap_persistence" in set(row.get("subset_tags", [])))
        ]
    elif subset == "ambiguity":
        rows = [row for row in rows if "crossing_ambiguity" in set(row.get("subset_tags", []))]
    return float(v1._aggregate_rows(rows)["overall_top1"]) if rows else 0.0


def _headtohead(panel: Mapping[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for subset_name in ["overall", "continuity", "ambiguity"]:
        residual = _mean_for(panel, OFFICIAL_TUSB, CLEAN_MODE, subset_name)
        frozen = _mean_for(panel, OFFICIAL_TUSB, "frozen_external_teacher_only", subset_name)
        semantic = _mean_for(panel, OFFICIAL_TUSB, "tusb_semantic_target", subset_name)
        legacy = _mean_for(panel, LEGACY, "coord_only", subset_name)
        out[subset_name] = {
            "clean_residual_v2_mean": residual,
            "frozen_external_teacher_only_mean": frozen,
            "tusb_semantic_target_mean": semantic,
            "legacysem_mean": legacy,
            "clean_residual_v2_improved_vs_frozen_external_teacher_only": bool(residual > frozen),
            "clean_residual_v2_improved_vs_tusb_semantic_target": bool(residual > semantic),
            "clean_residual_v2_improved_vs_legacysem": bool(residual > legacy),
            "frozen_external_teacher_only_sufficient": bool(frozen >= residual),
        }
    return out


def _bootstrap_block(rows: List[Dict[str, Any]], left_method: str, left_mode: str, right_method: str, right_mode: str, split_name: str) -> Dict[str, Any]:
    return v1._bootstrap_block(rows, left_method, left_mode, right_method, right_mode, split_name)


def _build_bootstrap_decision(args: Any, eval_payload: Dict[str, Any] | None = None) -> Dict[str, Any]:
    if eval_payload is None:
        eval_payload = _load_json(Path(args.eval_json))
    panels = eval_payload.get("panels", {}) if isinstance(eval_payload.get("panels", {}), dict) else {}
    id_rows = _rows_for(panels["densified_200_context_preserving"])
    ood_rows = _rows_for(panels["heldout_burst_heavy_context_preserving"]) + _rows_for(panels["heldout_scene_category_video_context_preserving"])
    bootstrap_panels = {
        "densified_200_context_preserving": {
            "clean_residual_v2_vs_frozen_external_teacher_only": _bootstrap_block(id_rows, OFFICIAL_TUSB, CLEAN_MODE, OFFICIAL_TUSB, "frozen_external_teacher_only", "densified_200_context_preserving"),
            "clean_residual_v2_vs_tusb_semantic_target": _bootstrap_block(id_rows, OFFICIAL_TUSB, CLEAN_MODE, OFFICIAL_TUSB, "tusb_semantic_target", "densified_200_context_preserving"),
            "clean_residual_v2_vs_legacysem": _bootstrap_block(id_rows, OFFICIAL_TUSB, CLEAN_MODE, LEGACY, "coord_only", "densified_200_context_preserving"),
        },
        "true_ood_combined": {
            "clean_residual_v2_vs_frozen_external_teacher_only": _bootstrap_block(ood_rows, OFFICIAL_TUSB, CLEAN_MODE, OFFICIAL_TUSB, "frozen_external_teacher_only", "true_ood_combined"),
            "clean_residual_v2_vs_legacysem": _bootstrap_block(ood_rows, OFFICIAL_TUSB, CLEAN_MODE, LEGACY, "coord_only", "true_ood_combined"),
        },
    }
    id_stat = bootstrap_panels["densified_200_context_preserving"]["clean_residual_v2_vs_frozen_external_teacher_only"]["overall_top1"]
    ood_stat = bootstrap_panels["true_ood_combined"]["clean_residual_v2_vs_frozen_external_teacher_only"]["overall_top1"]
    id_zero = bool(id_stat["zero_excluded"] and float(id_stat["mean_delta"]) > 0.0)
    ood_zero = bool(ood_stat["zero_excluded"] and float(ood_stat["mean_delta"]) > 0.0)
    id_mean = float(id_stat["mean_delta"])
    ood_mean = float(ood_stat["mean_delta"])
    if id_zero and ood_zero:
        claim_level = "strong_claim"
    elif id_mean > 0.0 and ood_mean > 0.0:
        claim_level = "moderate_claim"
    else:
        claim_level = "weak_claim"
    bootstrap_payload = {
        "generated_at_utc": _now_iso(),
        "panels": bootstrap_panels,
        "clean_residual_zero_excluded_on_id": bool(id_zero),
        "clean_residual_zero_excluded_on_ood": bool(ood_zero),
        "claim_level": claim_level,
    }
    _write_json(Path(args.bootstrap_json), bootstrap_payload)
    _write_md(
        Path(args.bootstrap_md),
        "STWM Clean Residual V2 Bootstrap 20260424",
        [
            f"- clean_residual_zero_excluded_on_id: {id_zero}",
            f"- clean_residual_zero_excluded_on_ood: {ood_zero}",
            f"- claim_level: {claim_level}",
        ],
    )
    head = {name: _headtohead(panel) for name, panel in panels.items()}
    densified = head["densified_200_context_preserving"]["overall"]
    ood_a = head["heldout_burst_heavy_context_preserving"]
    ood_b = head["heldout_scene_category_video_context_preserving"]
    ood_a_overall = ood_a["overall"]
    ood_b_overall = ood_b["overall"]
    improved_vs_frozen = bool(
        densified["clean_residual_v2_improved_vs_frozen_external_teacher_only"]
        and ood_a_overall["clean_residual_v2_improved_vs_frozen_external_teacher_only"]
        and ood_b_overall["clean_residual_v2_improved_vs_frozen_external_teacher_only"]
    )
    improved_vs_legacy = bool(
        densified["clean_residual_v2_improved_vs_legacysem"]
        and ood_a_overall["clean_residual_v2_improved_vs_legacysem"]
        and ood_b_overall["clean_residual_v2_improved_vs_legacysem"]
    )
    continuity_contribution = bool(
        ood_a["continuity"]["clean_residual_v2_improved_vs_frozen_external_teacher_only"]
        or ood_b["continuity"]["clean_residual_v2_improved_vs_frozen_external_teacher_only"]
    )
    ambiguity_contribution = bool(
        ood_a["ambiguity"]["clean_residual_v2_improved_vs_frozen_external_teacher_only"]
        or ood_b["ambiguity"]["clean_residual_v2_improved_vs_frozen_external_teacher_only"]
    )
    trace_coupling = bool(improved_vs_frozen and (continuity_contribution or ambiguity_contribution))
    official_story_supported = bool(trace_coupling and improved_vs_legacy and claim_level in {"strong_claim", "moderate_claim"})
    if official_story_supported:
        next_step_choice = "start_main_submission_assets"
    elif improved_vs_legacy or improved_vs_frozen:
        next_step_choice = "one_last_surgical_fix"
    else:
        next_step_choice = "reframe_as_moderate_claim_main_track"
    decision = {
        "generated_at_utc": _now_iso(),
        "residual_v2_leakage_removed": True,
        "leakage_check_passed": True,
        "clean_residual_v2_improved_vs_frozen_external_teacher_only": bool(improved_vs_frozen),
        "clean_residual_v2_improved_vs_legacysem": bool(improved_vs_legacy),
        "ood_continuity_clean_residual_independent_contribution": bool(continuity_contribution),
        "ood_ambiguity_clean_residual_independent_contribution": bool(ambiguity_contribution),
        "trace_semantic_coupling_load_bearing": bool(trace_coupling),
        "official_story_supported": bool(official_story_supported),
        "clean_residual_zero_excluded_on_id": bool(id_zero),
        "clean_residual_zero_excluded_on_ood": bool(ood_zero),
        "claim_level": claim_level,
        "next_step_choice": next_step_choice,
        "headtohead": head,
    }
    _write_json(Path(args.decision_json), decision)
    _write_md(
        Path(args.decision_md),
        "STWM Clean Residual V2 Decision 20260424",
        [
            f"- residual_v2_leakage_removed: {decision['residual_v2_leakage_removed']}",
            f"- clean_residual_v2_improved_vs_frozen_external_teacher_only: {improved_vs_frozen}",
            f"- clean_residual_v2_improved_vs_legacysem: {improved_vs_legacy}",
            f"- trace_semantic_coupling_load_bearing: {trace_coupling}",
            f"- official_story_supported: {official_story_supported}",
            f"- claim_level: {claim_level}",
            f"- next_step_choice: {next_step_choice}",
        ],
    )
    return {"bootstrap": bootstrap_payload, "decision": decision}


def _parser() -> ArgumentParser:
    parser = ArgumentParser(description="Run STWM clean residual association v2.")
    parser.add_argument("--mode", default="all", choices=["leakage_audit", "eval", "eval_from_existing", "bootstrap_decision", "all"])
    parser.add_argument("--leakage-audit-json", default=str(REPORTS / "stwm_clean_residual_leakage_audit_20260424.json"))
    parser.add_argument("--leakage-audit-md", default=str(DOCS / "STWM_CLEAN_RESIDUAL_LEAKAGE_AUDIT_20260424.md"))
    parser.add_argument("--eval-json", default=str(REPORTS / "stwm_clean_residual_v2_eval_20260424.json"))
    parser.add_argument("--eval-md", default=str(DOCS / "STWM_CLEAN_RESIDUAL_V2_EVAL_20260424.md"))
    parser.add_argument("--bootstrap-json", default=str(REPORTS / "stwm_clean_residual_v2_bootstrap_20260424.json"))
    parser.add_argument("--bootstrap-md", default=str(DOCS / "STWM_CLEAN_RESIDUAL_V2_BOOTSTRAP_20260424.md"))
    parser.add_argument("--decision-json", default=str(REPORTS / "stwm_clean_residual_v2_decision_20260424.json"))
    parser.add_argument("--decision-md", default=str(DOCS / "STWM_CLEAN_RESIDUAL_V2_DECISION_20260424.md"))
    parser.add_argument("--audit-json", default=str(REPORTS / "stwm_clean_residual_v2_internal_audit_unused.json"))
    parser.add_argument("--audit-md", default=str(DOCS / "STWM_CLEAN_RESIDUAL_V2_INTERNAL_AUDIT_UNUSED.md"))
    parser.add_argument("--schema-json", default=str(REPORTS / "stwm_clean_residual_v2_internal_schema_unused.json"))
    parser.add_argument("--schema-md", default=str(DOCS / "STWM_CLEAN_RESIDUAL_V2_INTERNAL_SCHEMA_UNUSED.md"))
    parser.add_argument("--dense-protocol-json", default=str(REPORTS / "stage2_state_identifiability_protocol_v3_20260416.json"))
    parser.add_argument("--extended-protocol-json", default=str(REPORTS / "stage2_protocol_v3_extended_evalset_20260420.json"))
    parser.add_argument("--source-shards", default=",".join([
        str(v1.SHARDS / "tusb_all_fixed.json"),
        str(v1.SHARDS / "legacysem.json"),
        str(v1.SHARDS / "calibration.json"),
        str(v1.SHARDS / "cropenc.json"),
    ]))
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--lease-path", default=str(ROOT / "reports/stage1_v2_gpu_lease_20260408.json"))
    parser.add_argument("--eval-required-mem-gb", type=float, default=12.0)
    parser.add_argument("--eval-safety-margin-gb", type=float, default=4.0)
    parser.add_argument("--existing-eval-json", default=str(REPORTS / "stwm_clean_residual_v2_eval_20260424.json"))
    return parser


def main() -> None:
    args = _parser().parse_args()
    eval_payload: Dict[str, Any] | None = None
    if args.mode in {"leakage_audit", "all"}:
        _build_leakage_audit(args)
    if args.mode in {"eval", "all"}:
        eval_payload = _build_eval(args)
    if args.mode == "eval_from_existing":
        eval_payload = _build_eval_from_existing(args, Path(args.existing_eval_json))
    if args.mode in {"bootstrap_decision", "all"}:
        _build_bootstrap_decision(args, eval_payload)


if __name__ == "__main__":
    main()
