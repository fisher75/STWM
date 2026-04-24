from __future__ import annotations

import json
from argparse import ArgumentParser
from pathlib import Path
from typing import Any, Dict, List

from stwm.tools import run_stwm_true_ood_attribution_20260423 as full


ROOT = Path("/raid/chen034/workspace/stwm")
REPORTS = ROOT / "reports"
DOCS = ROOT / "docs"


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text())


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))


def _write_md(path: Path, title: str, lines: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join([f"# {title}", ""] + lines) + "\n")


def merge(args: Any) -> Dict[str, Any]:
    shard_paths = [Path(part.strip()) for part in str(args.shard_jsons).split(",") if part.strip()]
    shards = [_load_json(path) for path in shard_paths]
    if not shards:
        raise RuntimeError("no shard_jsons provided")

    first = shards[0]
    split_item_ids = {k: set(v) for k, v in first["split_item_ids"].items()}
    prepared_meta = {k: dict(v) for k, v in first["prepared_item_meta"].items()}
    skipped_reasons = dict(first.get("skipped_reasons", {}))
    split_meta = dict(first.get("split_meta", {}))
    official_weights = dict(first.get("official_weights", {}))
    all_rows: List[Dict[str, Any]] = []
    for shard in shards:
        all_rows.extend(shard.get("raw_rows", []))

    fake_prepared = {
        item_id: {"protocol_eval_context_entity_count": int(meta.get("protocol_eval_context_entity_count", 0))}
        for item_id, meta in prepared_meta.items()
    }

    eval_started_at = min(str(shard.get("eval_started_at", "")) for shard in shards)
    eval_finished_at = max(str(shard.get("eval_finished_at", "")) for shard in shards)
    wall_time_seconds = float(sum(float(shard.get("wall_time_seconds", 0.0)) for shard in shards))

    split_panels: Dict[str, Any] = {}
    split_audit_splits: Dict[str, Any] = {}
    for split_name in full.OOD_SPLITS:
        panel = full._build_split_panel(
            split_name=split_name,
            split_meta=split_meta.get(split_name, {}),
            split_ids=split_item_ids[split_name],
            prepared_items=fake_prepared,
            skipped_reasons=skipped_reasons,
            rows=all_rows,
        )
        split_panels[split_name] = panel
        split_audit_splits[split_name] = {
            "total_items": int(panel["total_items"]),
            "valid_items": int(panel["valid_items"]),
            "skipped_items": int(panel["skipped_items"]),
            "skipped_reason_counts": dict(panel["skipped_reason_counts"]),
            "protocol_eval_context_entity_count_mean": float(panel["protocol_eval_context_entity_count_mean"]),
            "leakage_check_passed": bool(panel["leakage_check_passed"]),
            "exact_blocking_reason": str(panel.get("exact_blocking_reason", "")),
        }

    split_audit = {
        "generated_at_utc": full._now_iso(),
        "splits": split_audit_splits,
        "true_ood_materialized": True,
        "official_weights": official_weights,
        "source_shards": [str(path) for path in shard_paths],
    }
    _write_json(Path(args.split_audit_json), split_audit)
    _write_md(
        Path(args.split_audit_md),
        "STWM True OOD Attribution Split Audit 20260423",
        [f"- {name}: valid_items={payload['valid_items']} skipped_items={payload['skipped_items']}" for name, payload in split_audit_splits.items()],
    )

    eval_payload = {
        "generated_at_utc": full._now_iso(),
        "eval_started_at": eval_started_at,
        "eval_finished_at": eval_finished_at,
        "wall_time_seconds": wall_time_seconds,
        "splits": split_panels,
        "source_shards": [str(path) for path in shard_paths],
    }
    _write_json(Path(args.eval_json), eval_payload)
    _write_md(
        Path(args.eval_md),
        "STWM True OOD Attribution Eval 20260423",
        [f"- {name}: valid_items={panel['valid_items']} hash={panel['per_item_results_hash']}" for name, panel in split_panels.items()],
    )

    headtohead_splits = {
        split_name: full._build_headtohead_for_rows(panel["per_item_results"])
        for split_name, panel in split_panels.items()
    }
    _write_json(Path(args.headtohead_json), {"generated_at_utc": full._now_iso(), "splits": headtohead_splits})
    _write_md(
        Path(args.headtohead_md),
        "STWM True OOD Attribution HeadToHead 20260423",
        [f"- {name}: improved_vs_teacher_only={payload['overall']['improved_vs_teacher_only']} improved_vs_legacysem={payload['overall']['improved_vs_legacysem']}" for name, payload in headtohead_splits.items()],
    )

    bootstrap_splits = {
        split_name: {
            "hybrid_vs_semantic_teacher_only": full._bootstrap_block(
                panel["per_item_results"], full.OFFICIAL_TUSB, "hybrid_light", full.OFFICIAL_TUSB, "semantic_teacher_only", split_name
            ),
            "hybrid_vs_legacysem": full._bootstrap_block(
                panel["per_item_results"], full.OFFICIAL_TUSB, "hybrid_light", full.LEGACY, "coord_only", split_name
            ),
            "semantic_teacher_only_vs_legacysem": full._bootstrap_block(
                panel["per_item_results"], full.OFFICIAL_TUSB, "semantic_teacher_only", full.LEGACY, "coord_only", split_name
            ),
        }
        for split_name, panel in split_panels.items()
    }
    teacher_only_sufficient = all(not payload["overall"]["improved_vs_teacher_only"] for payload in headtohead_splits.values())
    ood_trace_semantic_coupling_zero_excluded = (
        all(payload["hybrid_vs_legacysem"]["overall_top1"]["zero_excluded"] and payload["hybrid_vs_semantic_teacher_only"]["overall_top1"]["zero_excluded"] for payload in bootstrap_splits.values())
        and all(payload["hybrid_vs_legacysem"]["hard_subset_top1"]["zero_excluded"] for payload in bootstrap_splits.values())
    )
    if ood_trace_semantic_coupling_zero_excluded:
        ood_claim_level = "strong_claim"
    elif any(payload["overall"]["improved_vs_legacysem"] or payload["overall"]["improved_vs_teacher_only"] for payload in headtohead_splits.values()):
        ood_claim_level = "moderate_claim"
    else:
        ood_claim_level = "weak_claim"
    bootstrap = {
        "generated_at_utc": full._now_iso(),
        "splits": bootstrap_splits,
        "ood_trace_semantic_coupling_zero_excluded": bool(ood_trace_semantic_coupling_zero_excluded),
        "ood_teacher_only_sufficient": bool(teacher_only_sufficient),
        "ood_claim_level": ood_claim_level,
    }
    _write_json(Path(args.bootstrap_json), bootstrap)
    _write_md(
        Path(args.bootstrap_md),
        "STWM True OOD Attribution Bootstrap 20260423",
        [
            f"- ood_trace_semantic_coupling_zero_excluded: {bootstrap['ood_trace_semantic_coupling_zero_excluded']}",
            f"- ood_teacher_only_sufficient: {bootstrap['ood_teacher_only_sufficient']}",
            f"- ood_claim_level: {bootstrap['ood_claim_level']}",
        ],
    )

    continuity_improved = all(payload["continuity_cases_only"]["improved_vs_teacher_only"] for payload in headtohead_splits.values())
    ambiguity_improved = all(payload["ambiguity_cases_only"]["improved_vs_teacher_only"] for payload in headtohead_splits.values())
    improved_vs_legacysem = all(payload["overall"]["improved_vs_legacysem"] for payload in headtohead_splits.values())
    trace_semantic_coupling_load_bearing = (
        (not teacher_only_sufficient)
        and continuity_improved
        and ambiguity_improved
        and ood_claim_level in {"strong_claim", "moderate_claim"}
    )
    official_story_supported = bool(trace_semantic_coupling_load_bearing and improved_vs_legacysem)
    if official_story_supported:
        next_step_choice = "start_main_submission_assets"
    elif improved_vs_legacysem or any(payload["overall"]["improved_vs_teacher_only"] for payload in headtohead_splits.values()):
        next_step_choice = "one_last_surgical_fix"
    else:
        next_step_choice = "reframe_as_moderate_claim_main_track"

    decision = {
        "generated_at_utc": full._now_iso(),
        "teacher_only_sufficient_on_true_ood": bool(teacher_only_sufficient),
        "continuity_hybrid_improved_vs_teacher_only": bool(continuity_improved),
        "ambiguity_hybrid_improved_vs_teacher_only": bool(ambiguity_improved),
        "hybrid_light_improved_vs_legacysem": bool(improved_vs_legacysem),
        "trace_semantic_coupling_load_bearing_on_true_ood": bool(trace_semantic_coupling_load_bearing),
        "official_story_supported": bool(official_story_supported),
        "next_step_choice": next_step_choice,
    }
    _write_json(Path(args.decision_json), decision)
    _write_md(
        Path(args.decision_md),
        "STWM True OOD Attribution Decision 20260423",
        [
            f"- teacher_only_sufficient_on_true_ood: {decision['teacher_only_sufficient_on_true_ood']}",
            f"- continuity_hybrid_improved_vs_teacher_only: {decision['continuity_hybrid_improved_vs_teacher_only']}",
            f"- ambiguity_hybrid_improved_vs_teacher_only: {decision['ambiguity_hybrid_improved_vs_teacher_only']}",
            f"- hybrid_light_improved_vs_legacysem: {decision['hybrid_light_improved_vs_legacysem']}",
            f"- trace_semantic_coupling_load_bearing_on_true_ood: {decision['trace_semantic_coupling_load_bearing_on_true_ood']}",
            f"- official_story_supported: {decision['official_story_supported']}",
            f"- next_step_choice: {decision['next_step_choice']}",
        ],
    )
    return {
        "split_audit": split_audit,
        "eval": eval_payload,
        "bootstrap": bootstrap,
        "decision": decision,
    }


def main() -> None:
    parser = ArgumentParser(description="Merge STWM true-OOD attribution shard outputs.")
    parser.add_argument("--shard-jsons", required=True)
    parser.add_argument("--split-audit-json", default=str(REPORTS / "stwm_true_ood_attribution_split_audit_20260423.json"))
    parser.add_argument("--split-audit-md", default=str(DOCS / "STWM_TRUE_OOD_ATTRIBUTION_SPLIT_AUDIT_20260423.md"))
    parser.add_argument("--eval-json", default=str(REPORTS / "stwm_true_ood_attribution_eval_20260423.json"))
    parser.add_argument("--eval-md", default=str(DOCS / "STWM_TRUE_OOD_ATTRIBUTION_EVAL_20260423.md"))
    parser.add_argument("--headtohead-json", default=str(REPORTS / "stwm_true_ood_attribution_headtohead_20260423.json"))
    parser.add_argument("--headtohead-md", default=str(DOCS / "STWM_TRUE_OOD_ATTRIBUTION_HEADTOHEAD_20260423.md"))
    parser.add_argument("--bootstrap-json", default=str(REPORTS / "stwm_true_ood_attribution_bootstrap_20260423.json"))
    parser.add_argument("--bootstrap-md", default=str(DOCS / "STWM_TRUE_OOD_ATTRIBUTION_BOOTSTRAP_20260423.md"))
    parser.add_argument("--decision-json", default=str(REPORTS / "stwm_true_ood_attribution_decision_20260423.json"))
    parser.add_argument("--decision-md", default=str(DOCS / "STWM_TRUE_OOD_ATTRIBUTION_DECISION_20260423.md"))
    args = parser.parse_args()
    merge(args)


if __name__ == "__main__":
    main()
