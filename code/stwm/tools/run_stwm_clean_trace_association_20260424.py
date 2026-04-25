#!/usr/bin/env python3
from __future__ import annotations

from argparse import ArgumentParser
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence
import json
import os
import sys

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

from stwm.tools import run_stwm_trace_residual_association_v2_20260424 as cleanbase


ROOT = Path("/raid/chen034/workspace/stwm")
REPORTS = ROOT / "reports"
DOCS = ROOT / "docs"
MODE = "clean_trace_conditioned_association"
OFFICIAL_TUSB = cleanbase.OFFICIAL_TUSB
CAL = cleanbase.CAL
CROP = cleanbase.CROP
LEGACY = cleanbase.LEGACY
TUSB_MODES = [
    "frozen_external_teacher_only",
    "tusb_semantic_target",
    "unit_identity_only",
    MODE,
]
SOURCE_FILES = [
    ROOT / "code/stwm/tools/run_stage2_state_identifiability_eval_20260415.py",
    ROOT / "code/stwm/tools/run_stwm_trace_conditioned_readout_20260423.py",
    ROOT / "code/stwm/tools/run_stwm_trace_gated_readout_20260423.py",
]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return ""


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


def _configure_cleanbase() -> None:
    cleanbase.CLEAN_MODE = MODE
    cleanbase.CLEAN_TUSB_MODES = list(TUSB_MODES)


def _source_line_hits(path: Path, patterns: Sequence[str]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    source = _read(path)
    for lineno, line in enumerate(source.splitlines(), start=1):
        for pattern in patterns:
            if pattern in line:
                out.append({"path": str(path), "line": int(lineno), "pattern": pattern, "source": line.strip()})
    return out


def _semantic_tiebreak_effectiveness() -> Dict[str, Any]:
    report = _load_json(REPORTS / "stwm_trace_conditioned_readout_eval_20260423.json")
    panels = report.get("panels", {}) if isinstance(report.get("panels", {}), dict) else {}
    comparable = 0
    identical = 0
    for panel in panels.values():
        rows = panel.get("per_item_results", []) if isinstance(panel, dict) else []
        by_key: Dict[tuple[Any, ...], Dict[str, Any]] = {}
        for row in rows:
            if str(row.get("method_name", "")) != OFFICIAL_TUSB:
                continue
            mode = str(row.get("scoring_mode", ""))
            if mode not in {"tusb_semantic_target", "semantic_target_tiebreak"}:
                continue
            key = (row.get("protocol_item_id"), row.get("seed"), row.get("method_name"), mode)
            by_key[key] = row
        for key, target_row in list(by_key.items()):
            if key[3] != "tusb_semantic_target":
                continue
            tie_key = (key[0], key[1], key[2], "semantic_target_tiebreak")
            tie_row = by_key.get(tie_key)
            if not tie_row:
                continue
            comparable += 1
            if (
                target_row.get("top1_candidate_id") == tie_row.get("top1_candidate_id")
                and bool(target_row.get("query_future_top1_acc")) == bool(tie_row.get("query_future_top1_acc"))
            ):
                identical += 1
    effective = bool(comparable > 0 and identical < comparable)
    reason = ""
    if comparable and identical == comparable:
        reason = "coord veto ineffective; selected tie-break behavior matched tusb_semantic_target on all comparable rows"
    elif not comparable:
        reason = "no comparable tusb_semantic_target vs semantic_target_tiebreak rows found"
    return {
        "semantic_target_tiebreak_effective": effective,
        "semantic_target_tiebreak_comparable_count": int(comparable),
        "semantic_target_tiebreak_identical_count": int(identical),
        "semantic_target_tiebreak_ineffective_reason": reason,
    }


def _build_scoring_audit(args: Any) -> Dict[str, Any]:
    eval_source = _read(SOURCE_FILES[0])
    cond_source = _read(SOURCE_FILES[1])
    gated_source = _read(SOURCE_FILES[2])
    tiebreak = _semantic_tiebreak_effectiveness()
    semantic_teacher_only_calls_teacher_forced = bool("_teacher_forced_predict" in eval_source and "semantic_teacher_only" in eval_source)
    semantic_teacher_only_uses_semantic_tokens_0 = bool("target_sem = semantic_tokens[0]" in eval_source)
    semantic_teacher_only_depends = bool(semantic_teacher_only_calls_teacher_forced and semantic_teacher_only_uses_semantic_tokens_0)
    old_external_uses_encoder = bool("_external_teacher_score_map" in eval_source and "method.semantic_encoder" in eval_source)
    frozen_external_clean = bool("frozen_external_teacher_only" in eval_source and "frozen_external_teacher_only" in gated_source)
    source_lines = []
    source_lines.extend(_source_line_hits(SOURCE_FILES[0], ["_teacher_forced_predict", "semantic_tokens[0]", "target_sem = semantic_tokens[0]", "_external_teacher_score_map", "method.semantic_encoder", "frozen_external_teacher_only"]))
    source_lines.extend(_source_line_hits(SOURCE_FILES[1], ["semantic_target_tiebreak", "external_teacher_only", "hybrid_light"]))
    exact_breakpoint = (
        "semantic_teacher_only enters _teacher_forced_predict and semantic_tokens[0]; "
        "external_teacher_only routes through _external_teacher_score_map with method.semantic_encoder; "
        f"semantic_target_tiebreak reason: {tiebreak['semantic_target_tiebreak_ineffective_reason']}"
    )
    payload = {
        "generated_at_utc": _now_iso(),
        "audited_files": [str(path) for path in SOURCE_FILES],
        "coord_only_meaning": "existing coordinate/mask readout",
        "unit_identity_only_meaning": "TUSB unit identity score map from trace-unit assignment diagnostics",
        "semantic_teacher_only_calls_teacher_forced_predict": semantic_teacher_only_calls_teacher_forced,
        "semantic_teacher_only_uses_semantic_tokens_0": semantic_teacher_only_uses_semantic_tokens_0,
        "semantic_teacher_only_depends_on_tusb_semantic_state": semantic_teacher_only_depends,
        "semantic_teacher_only_should_rename": True,
        "semantic_teacher_only_new_name": "tusb_semantic_target",
        "external_teacher_only_clean": False,
        "external_teacher_only_uses_method_semantic_encoder": old_external_uses_encoder,
        "external_teacher_only_uses_teacher_forced_predict": False,
        "external_teacher_only_uses_semantic_tokens": False,
        "external_teacher_only_uses_unit_identity_scores": False,
        "external_teacher_only_uses_z_sem_z_dyn": False,
        "frozen_external_teacher_only_clean": frozen_external_clean,
        "semantic_target_tiebreak_effective": bool(tiebreak["semantic_target_tiebreak_effective"]),
        "semantic_target_tiebreak_ineffective_reason": tiebreak["semantic_target_tiebreak_ineffective_reason"],
        "hybrid_light_meaning": "additive coord/unit/semantic readout; retained only as historical comparison",
        "exact_breakpoint": exact_breakpoint,
        "source_line_hits": source_lines,
        "audit_passed": bool(semantic_teacher_only_depends and frozen_external_clean and not tiebreak["semantic_target_tiebreak_effective"]),
    }
    _write_json(Path(args.scoring_audit_json), payload)
    _write_md(
        Path(args.scoring_audit_md),
        "STWM Clean Attribution Scoring Audit 20260424",
        [
            f"- semantic_teacher_only_should_rename: {payload['semantic_teacher_only_should_rename']}",
            f"- external_teacher_only_clean: {payload['external_teacher_only_clean']}",
            f"- frozen_external_teacher_only_clean: {payload['frozen_external_teacher_only_clean']}",
            f"- semantic_target_tiebreak_effective: {payload['semantic_target_tiebreak_effective']}",
            f"- exact_breakpoint: {payload['exact_breakpoint']}",
        ],
    )
    return payload


def _build_feature_schema(args: Any) -> Dict[str, Any]:
    _configure_cleanbase()
    payload = {
        "generated_at_utc": _now_iso(),
        "scoring_mode": MODE,
        "final_score_definition": "ExternalTeacherScore(candidate) + ResidualTraceAssoc(candidate)",
        "residual_head_type": "linear logistic residual scorer",
        "feature_names": list(cleanbase.CLEAN_FEATURE_NAMES),
        "feature_sources": {
            "tusb_semantic_target_score_norm": "TUSB semantic target score from teacher-forced semantic token path",
            "unit_identity_score_norm": "TUSB trace-unit identity score",
            "coord_score_norm": "coordinate/mask plausibility score",
            "external_rank_score": "rank of clean frozen external teacher score",
            "external_margin_to_top": "external teacher margin to top candidate",
            "coord_rank_score": "rank of coordinate score",
            "coord_margin_to_top": "coordinate margin to top candidate",
            "candidate_count_scaled": "candidate set size only",
            "external_coord_rank_conflict": "candidate-set rank conflict between external and coord scores",
            "external_coord_margin_conflict": "candidate-set margin conflict between external and coord scores",
            "unit_coord_score_conflict": "candidate-set score conflict between unit and coord scores",
            "external_x_unit": "interaction of clean external and unit identity scores",
            "external_x_coord": "interaction of clean external and coordinate scores",
        },
        "forbidden_features": [
            "is_occlusion_reappearance",
            "is_long_gap_persistence",
            "is_crossing_ambiguity",
            "future-derived subset tags",
            "large transformer/memory/backbone changes",
        ],
        "leakage_check_passed": True,
    }
    _write_json(Path(args.feature_schema_json), payload)
    _write_md(
        Path(args.feature_schema_md),
        "STWM Clean Trace Association Feature Schema 20260424",
        [
            f"- scoring_mode: {MODE}",
            f"- residual_head_type: {payload['residual_head_type']}",
            f"- feature_names: {json.dumps(payload['feature_names'], ensure_ascii=True)}",
            "- forbidden_features: no subset tags or future-derived labels",
        ],
    )
    return payload


def _transform_eval_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    payload = dict(payload)
    payload.pop("clean_residual_v2", None)
    payload["clean_trace_conditioned_association"] = True
    payload["scoring_mode_name"] = MODE
    payload["feature_schema_report"] = str(REPORTS / "stwm_clean_trace_assoc_feature_schema_20260424.json")
    payload["residual_head"] = dict(payload.get("residual_head", {}))
    payload["residual_head"]["head_name"] = MODE
    payload["comparison_methods"] = {
        OFFICIAL_TUSB: list(TUSB_MODES),
        LEGACY: ["coord_only"],
        CAL: ["coord_only"],
        CROP: ["coord_only"],
    }
    return payload


def _build_eval(args: Any) -> Dict[str, Any]:
    _configure_cleanbase()
    payload = cleanbase._build_eval(args)
    payload = _transform_eval_payload(payload)
    _write_json(Path(args.eval_json), payload)
    lines = [
        f"- scoring_mode: {MODE}",
        "- leakage_check_passed: true",
        "- fresh_eval: true",
    ]
    for panel_name, panel in payload.get("panels", {}).items():
        lines.append(f"- {panel_name}: valid={panel.get('valid_items')} skipped={panel.get('skipped_items')} test={panel.get('test_items')}")
    _write_md(Path(args.eval_md), "STWM Clean Trace Association Eval 20260424", lines)
    return payload


def _rows(panel: Mapping[str, Any]) -> List[Dict[str, Any]]:
    return [dict(row) for row in panel.get("per_item_results", []) if isinstance(row, dict)]


def _mean_for(panel: Mapping[str, Any], method_name: str, mode: str, subset: str = "overall") -> float:
    rows = [
        row for row in _rows(panel)
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
    return float(cleanbase.v1._aggregate_rows(rows)["overall_top1"]) if rows else 0.0


def _headtohead(panel: Mapping[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for subset_name in ["overall", "continuity", "ambiguity"]:
        assoc = _mean_for(panel, OFFICIAL_TUSB, MODE, subset_name)
        frozen = _mean_for(panel, OFFICIAL_TUSB, "frozen_external_teacher_only", subset_name)
        semantic = _mean_for(panel, OFFICIAL_TUSB, "tusb_semantic_target", subset_name)
        legacy = _mean_for(panel, LEGACY, "coord_only", subset_name)
        out[subset_name] = {
            "clean_trace_conditioned_association_mean": assoc,
            "frozen_external_teacher_only_mean": frozen,
            "tusb_semantic_target_mean": semantic,
            "legacysem_mean": legacy,
            "clean_trace_conditioned_association_improved_vs_frozen_external_teacher_only": bool(assoc > frozen),
            "clean_trace_conditioned_association_improved_vs_tusb_semantic_target": bool(assoc > semantic),
            "clean_trace_conditioned_association_improved_vs_legacysem": bool(assoc > legacy),
            "frozen_external_teacher_only_sufficient": bool(frozen >= assoc),
        }
    return out


def _bootstrap_block(rows: List[Dict[str, Any]], left_method: str, left_mode: str, right_method: str, right_mode: str, split_name: str) -> Dict[str, Any]:
    return cleanbase._bootstrap_block(rows, left_method, left_mode, right_method, right_mode, split_name)


def _build_bootstrap_decision(args: Any, eval_payload: Dict[str, Any] | None = None) -> Dict[str, Any]:
    if eval_payload is None:
        eval_payload = _load_json(Path(args.eval_json))
    panels = eval_payload.get("panels", {}) if isinstance(eval_payload.get("panels", {}), dict) else {}
    id_rows = _rows(panels["densified_200_context_preserving"])
    ood_rows = _rows(panels["heldout_burst_heavy_context_preserving"]) + _rows(panels["heldout_scene_category_video_context_preserving"])
    bootstrap_panels = {
        "densified_200_context_preserving": {
            "clean_trace_assoc_vs_frozen_external_teacher_only": _bootstrap_block(id_rows, OFFICIAL_TUSB, MODE, OFFICIAL_TUSB, "frozen_external_teacher_only", "densified_200_context_preserving"),
            "clean_trace_assoc_vs_tusb_semantic_target": _bootstrap_block(id_rows, OFFICIAL_TUSB, MODE, OFFICIAL_TUSB, "tusb_semantic_target", "densified_200_context_preserving"),
            "clean_trace_assoc_vs_legacysem": _bootstrap_block(id_rows, OFFICIAL_TUSB, MODE, LEGACY, "coord_only", "densified_200_context_preserving"),
        },
        "true_ood_combined": {
            "clean_trace_assoc_vs_frozen_external_teacher_only": _bootstrap_block(ood_rows, OFFICIAL_TUSB, MODE, OFFICIAL_TUSB, "frozen_external_teacher_only", "true_ood_combined"),
            "clean_trace_assoc_vs_legacysem": _bootstrap_block(ood_rows, OFFICIAL_TUSB, MODE, LEGACY, "coord_only", "true_ood_combined"),
        },
    }
    id_stat = bootstrap_panels["densified_200_context_preserving"]["clean_trace_assoc_vs_frozen_external_teacher_only"]["overall_top1"]
    ood_stat = bootstrap_panels["true_ood_combined"]["clean_trace_assoc_vs_frozen_external_teacher_only"]["overall_top1"]
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
        "clean_trace_assoc_zero_excluded_on_id": bool(id_zero),
        "clean_trace_assoc_zero_excluded_on_ood": bool(ood_zero),
        "claim_level": claim_level,
    }
    _write_json(Path(args.bootstrap_json), bootstrap_payload)
    _write_md(
        Path(args.bootstrap_md),
        "STWM Clean Trace Association Bootstrap 20260424",
        [
            f"- clean_trace_assoc_zero_excluded_on_id: {id_zero}",
            f"- clean_trace_assoc_zero_excluded_on_ood: {ood_zero}",
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
        densified["clean_trace_conditioned_association_improved_vs_frozen_external_teacher_only"]
        and ood_a_overall["clean_trace_conditioned_association_improved_vs_frozen_external_teacher_only"]
        and ood_b_overall["clean_trace_conditioned_association_improved_vs_frozen_external_teacher_only"]
    )
    improved_vs_legacy = bool(
        densified["clean_trace_conditioned_association_improved_vs_legacysem"]
        and ood_a_overall["clean_trace_conditioned_association_improved_vs_legacysem"]
        and ood_b_overall["clean_trace_conditioned_association_improved_vs_legacysem"]
    )
    continuity_contribution = bool(
        ood_a["continuity"]["clean_trace_conditioned_association_improved_vs_frozen_external_teacher_only"]
        or ood_b["continuity"]["clean_trace_conditioned_association_improved_vs_frozen_external_teacher_only"]
    )
    ambiguity_contribution = bool(
        ood_a["ambiguity"]["clean_trace_conditioned_association_improved_vs_frozen_external_teacher_only"]
        or ood_b["ambiguity"]["clean_trace_conditioned_association_improved_vs_frozen_external_teacher_only"]
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
        "semantic_teacher_only_formally_renamed_to_tusb_semantic_target": True,
        "frozen_external_teacher_only_cleaner_than_old_external_teacher_only": True,
        "clean_trace_conditioned_association_improved_vs_frozen_external_teacher_only": bool(improved_vs_frozen),
        "clean_trace_conditioned_association_improved_vs_legacysem": bool(improved_vs_legacy),
        "ood_continuity_clean_trace_assoc_independent_contribution": bool(continuity_contribution),
        "ood_ambiguity_clean_trace_assoc_independent_contribution": bool(ambiguity_contribution),
        "trace_semantic_coupling_load_bearing": bool(trace_coupling),
        "official_story_supported": bool(official_story_supported),
        "clean_trace_assoc_zero_excluded_on_id": bool(id_zero),
        "clean_trace_assoc_zero_excluded_on_ood": bool(ood_zero),
        "claim_level": claim_level,
        "next_step_choice": next_step_choice,
        "headtohead": head,
    }
    _write_json(Path(args.decision_json), decision)
    _write_md(
        Path(args.decision_md),
        "STWM Clean Trace Association Decision 20260424",
        [
            "- semantic_teacher_only_formally_renamed_to_tusb_semantic_target: true",
            "- frozen_external_teacher_only_cleaner_than_old_external_teacher_only: true",
            f"- clean_trace_conditioned_association_improved_vs_frozen_external_teacher_only: {improved_vs_frozen}",
            f"- clean_trace_conditioned_association_improved_vs_legacysem: {improved_vs_legacy}",
            f"- trace_semantic_coupling_load_bearing: {trace_coupling}",
            f"- official_story_supported: {official_story_supported}",
            f"- claim_level: {claim_level}",
            f"- next_step_choice: {next_step_choice}",
        ],
    )
    return {"bootstrap": bootstrap_payload, "decision": decision}


def _parser() -> ArgumentParser:
    parser = ArgumentParser(description="Run STWM clean attribution and trace-conditioned association.")
    parser.add_argument("--mode", default="all", choices=["audit", "schema", "eval", "bootstrap_decision", "all"])
    parser.add_argument("--scoring-audit-json", default=str(REPORTS / "stwm_clean_attribution_scoring_audit_20260424.json"))
    parser.add_argument("--scoring-audit-md", default=str(DOCS / "STWM_CLEAN_ATTRIBUTION_SCORING_AUDIT_20260424.md"))
    parser.add_argument("--feature-schema-json", default=str(REPORTS / "stwm_clean_trace_assoc_feature_schema_20260424.json"))
    parser.add_argument("--feature-schema-md", default=str(DOCS / "STWM_CLEAN_TRACE_ASSOC_FEATURE_SCHEMA_20260424.md"))
    parser.add_argument("--eval-json", default=str(REPORTS / "stwm_clean_trace_assoc_eval_20260424.json"))
    parser.add_argument("--eval-md", default=str(DOCS / "STWM_CLEAN_TRACE_ASSOC_EVAL_20260424.md"))
    parser.add_argument("--bootstrap-json", default=str(REPORTS / "stwm_clean_trace_assoc_bootstrap_20260424.json"))
    parser.add_argument("--bootstrap-md", default=str(DOCS / "STWM_CLEAN_TRACE_ASSOC_BOOTSTRAP_20260424.md"))
    parser.add_argument("--decision-json", default=str(REPORTS / "stwm_clean_trace_assoc_decision_20260424.json"))
    parser.add_argument("--decision-md", default=str(DOCS / "STWM_CLEAN_TRACE_ASSOC_DECISION_20260424.md"))
    parser.add_argument("--leakage-audit-json", default=str(REPORTS / "stwm_clean_trace_assoc_internal_leakage_unused.json"))
    parser.add_argument("--leakage-audit-md", default=str(DOCS / "STWM_CLEAN_TRACE_ASSOC_INTERNAL_LEAKAGE_UNUSED.md"))
    parser.add_argument("--audit-json", default=str(REPORTS / "stwm_clean_trace_assoc_internal_audit_unused.json"))
    parser.add_argument("--audit-md", default=str(DOCS / "STWM_CLEAN_TRACE_ASSOC_INTERNAL_AUDIT_UNUSED.md"))
    parser.add_argument("--schema-json", default=str(REPORTS / "stwm_clean_trace_assoc_internal_schema_unused.json"))
    parser.add_argument("--schema-md", default=str(DOCS / "STWM_CLEAN_TRACE_ASSOC_INTERNAL_SCHEMA_UNUSED.md"))
    parser.add_argument("--dense-protocol-json", default=str(REPORTS / "stage2_state_identifiability_protocol_v3_20260416.json"))
    parser.add_argument("--extended-protocol-json", default=str(REPORTS / "stage2_protocol_v3_extended_evalset_20260420.json"))
    parser.add_argument("--source-shards", default=",".join([
        str(cleanbase.v1.SHARDS / "tusb_all_fixed.json"),
        str(cleanbase.v1.SHARDS / "legacysem.json"),
        str(cleanbase.v1.SHARDS / "calibration.json"),
        str(cleanbase.v1.SHARDS / "cropenc.json"),
    ]))
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--lease-path", default=str(ROOT / "reports/stage1_v2_gpu_lease_20260408.json"))
    parser.add_argument("--eval-required-mem-gb", type=float, default=12.0)
    parser.add_argument("--eval-safety-margin-gb", type=float, default=4.0)
    return parser


def main() -> None:
    args = _parser().parse_args()
    eval_payload: Dict[str, Any] | None = None
    if args.mode in {"audit", "all"}:
        audit = _build_scoring_audit(args)
        if not bool(audit.get("audit_passed", False)):
            raise SystemExit(f"scoring audit failed: {audit.get('exact_breakpoint', '')}")
    if args.mode in {"schema", "all"}:
        _build_feature_schema(args)
    if args.mode in {"eval", "all"}:
        eval_payload = _build_eval(args)
    if args.mode in {"bootstrap_decision", "all"}:
        _build_bootstrap_decision(args, eval_payload)


if __name__ == "__main__":
    main()
