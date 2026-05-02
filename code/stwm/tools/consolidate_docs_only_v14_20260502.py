#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
DOCS = ROOT / "docs"
ARTIFACTS = ROOT / "artifacts"
REPORTS = ROOT / "reports"


KEEP_NAMES = {
    "ARCHITECTURE.md",
    "REPO_ARTIFACT_POLICY.md",
    "STATUS.md",
    "STWM_DOCS_CLEANUP_V12_20260502.md",
    "STWM_REPO_REDUNDANCY_CLEANUP_V13_20260502.md",
    "STWM_REPO_SUPERSEDED_DOCS_CLEANUP_V13_20260502.md",
    "STWM_REPO_SUPERSEDED_LARGE_REPORT_CLEANUP_V13_20260502.md",
    "STWM_WORLD_MODEL_NO_DRIFT_GUARDRAIL_V44.md",
    # Current FSTF evidence chain.
    "STWM_FSTF_FINAL_CLAIM_BOUNDARY_V13_20260502.md",
    "STWM_FSTF_V13_CVPR_READINESS_GATE_20260502.md",
    "STWM_FSTF_SCALING_CLAIM_VERIFICATION_V13_20260502.md",
    "STWM_FSTF_TRACE_CONDITIONING_HORIZON_V13_20260502.md",
    "STWM_FSTF_TRACE_DENSITY_VALID_UNITS_AUDIT_V13_20260502.md",
    "STWM_FSTF_VISUALIZATION_ARTIFACT_AUDIT_V13_20260502.md",
    "STWM_FSTF_FULL_SCALING_LAWS_V12_20260502.md",
    "STWM_FSTF_VISUALIZATION_V12_20260502.md",
    "STWM_FSTF_TRACE_CONDITIONING_AUDIT_V10_20260502.md",
    "STWM_FSTF_STRONG_COPYAWARE_BASELINE_SUITE_V8_20260501.md",
    "STWM_FSTF_V7_BASELINE_FAILURE_DIAGNOSIS_20260501.md",
    "STWM_BASELINE_OFFICIALITY_AUDIT_V7_20260501.md",
    "STWM_FSTF_BENCHMARK_PROTOCOL_V6_20260501.md",
    # Main mixed/VIPSeg/LODO protocol evidence.
    "STWM_MIXED_FULLSCALE_V2_PROTOCOL_AUDIT_20260428.md",
    "STWM_MIXED_FULLSCALE_V2_TRAIN_SUMMARY_COMPLETE_20260428.md",
    "STWM_MIXED_FULLSCALE_V2_VAL_SELECTION_COMPLETE_20260428.md",
    "STWM_MIXED_FULLSCALE_V2_TEST_EVAL_COMPLETE_20260428.md",
    "STWM_MIXED_FULLSCALE_V2_SIGNIFICANCE_COMPLETE_20260428.md",
    "STWM_MIXED_FULLSCALE_V2_SEED_ROBUSTNESS_COMPLETE_20260428.md",
    "STWM_MIXED_FULLSCALE_V2_COMPLETE_DECISION_20260428.md",
    "STWM_FINAL_LODO_V3_20260428.md",
    "STWM_FINAL_LODO_CONSISTENCY_AUDIT_V5_20260428.md",
    "STWM_FINAL_LODO_DOMAIN_SHIFT_DIAGNOSIS_V5_20260428.md",
    "STWM_VIPSEG_RAW_OBSERVED_MEMORY_V2_DECISION_20260428.md",
    "STWM_VIPSEG_RAW_OBSERVED_MEMORY_V2_ROOT_CAUSE_AUDIT_20260428.md",
    "STWM_VIPSEG_RAW_OBSERVED_SEMANTIC_FEATURES_V2_20260428.md",
    "STWM_VIPSEG_OBSERVED_SEMANTIC_PROTOTYPE_TARGETS_V2_20260428.md",
    # Paper positioning and external boundary.
    "STWM_FINAL_RELATED_WORK_POSITIONING_V5_20260428.md",
    "STWM_FINAL_PAPER_OUTLINE_V5_20260428.md",
    "STWM_EXTERNAL_BASELINE_FULL_EVAL_BOOTSTRAP_20260426.md",
    "STWM_EXTERNAL_BASELINE_SAM2_FULL_EVAL_20260426.md",
    "STWM_EXTERNAL_BASELINE_COTRACKER_FULL_EVAL_20260426.md",
    "STWM_EXTERNAL_BASELINE_CUTIE_FULL_EVAL_20260426.md",
    "STWM_EXTERNAL_CONSUMER_UTILITY_BOUNDARY_20260427.md",
}

NEW_DOCS = {
    "STWM_DOCS_CURRENT_INDEX_20260502.md",
    "STWM_CURRENT_EVIDENCE_BRIEF_20260502.md",
    "STWM_DOCS_ARCHIVE_MANIFEST_20260502.md",
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _dump_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _doc_link(name: str) -> str:
    return f"[{name}]({name})"


def _write_current_index(*, archive_path: Path, kept: list[Path], archived: list[dict[str, Any]]) -> None:
    groups = {
        "Start Here": [
            "STWM_CURRENT_EVIDENCE_BRIEF_20260502.md",
            "STWM_FSTF_FINAL_CLAIM_BOUNDARY_V13_20260502.md",
            "STWM_FSTF_V13_CVPR_READINESS_GATE_20260502.md",
            "STWM_FSTF_BENCHMARK_PROTOCOL_V6_20260501.md",
        ],
        "Final FSTF Evidence": [
            "STWM_FSTF_SCALING_CLAIM_VERIFICATION_V13_20260502.md",
            "STWM_FSTF_TRACE_CONDITIONING_HORIZON_V13_20260502.md",
            "STWM_FSTF_TRACE_DENSITY_VALID_UNITS_AUDIT_V13_20260502.md",
            "STWM_FSTF_FULL_SCALING_LAWS_V12_20260502.md",
            "STWM_FSTF_TRACE_CONDITIONING_AUDIT_V10_20260502.md",
            "STWM_FSTF_STRONG_COPYAWARE_BASELINE_SUITE_V8_20260501.md",
            "STWM_FSTF_VISUALIZATION_ARTIFACT_AUDIT_V13_20260502.md",
            "STWM_FSTF_VISUALIZATION_V12_20260502.md",
        ],
        "Mixed/VIPSeg/LODO": [
            "STWM_MIXED_FULLSCALE_V2_COMPLETE_DECISION_20260428.md",
            "STWM_MIXED_FULLSCALE_V2_TRAIN_SUMMARY_COMPLETE_20260428.md",
            "STWM_MIXED_FULLSCALE_V2_VAL_SELECTION_COMPLETE_20260428.md",
            "STWM_MIXED_FULLSCALE_V2_TEST_EVAL_COMPLETE_20260428.md",
            "STWM_MIXED_FULLSCALE_V2_SIGNIFICANCE_COMPLETE_20260428.md",
            "STWM_FINAL_LODO_CONSISTENCY_AUDIT_V5_20260428.md",
            "STWM_FINAL_LODO_DOMAIN_SHIFT_DIAGNOSIS_V5_20260428.md",
            "STWM_VIPSEG_RAW_OBSERVED_MEMORY_V2_DECISION_20260428.md",
        ],
        "External Boundary and Paper Assets": [
            "STWM_EXTERNAL_BASELINE_FULL_EVAL_BOOTSTRAP_20260426.md",
            "STWM_EXTERNAL_CONSUMER_UTILITY_BOUNDARY_20260427.md",
            "STWM_FINAL_RELATED_WORK_POSITIONING_V5_20260428.md",
            "STWM_FINAL_PAPER_OUTLINE_V5_20260428.md",
            "STWM_WORLD_MODEL_NO_DRIFT_GUARDRAIL_V44.md",
        ],
        "Repo/Docs Maintenance": [
            "STWM_DOCS_ARCHIVE_MANIFEST_20260502.md",
            "STWM_DOCS_CLEANUP_V12_20260502.md",
            "STWM_REPO_REDUNDANCY_CLEANUP_V13_20260502.md",
        ],
    }
    kept_names = {p.name for p in kept}
    lines = [
        "# STWM Docs Current Index",
        "",
        "This directory has been consolidated so day-to-day reading starts from a small current evidence set.",
        "",
        f"- current_docs_kept: `{len(kept)}`",
        f"- historical_docs_archived: `{len(archived)}`",
        f"- historical_archive: `{archive_path.relative_to(ROOT)}`",
        "",
    ]
    for title, names in groups.items():
        lines.append(f"## {title}")
        for name in names:
            if name in kept_names or name in NEW_DOCS:
                lines.append(f"- {_doc_link(name)}")
        lines.append("")
    lines.extend(
        [
            "## Notes",
            "- Historical documents were archived, not silently destroyed.",
            "- Shard-level docs from earlier cleanup are in `artifacts/stwm_redundant_shard_reports_v13_20260502.tar.gz`.",
            "- Reports and experiment artifacts remain outside this docs-only consolidation.",
        ]
    )
    (DOCS / "STWM_DOCS_CURRENT_INDEX_20260502.md").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _write_evidence_brief() -> None:
    readiness = _load_json(REPORTS / "stwm_fstf_v13_cvpr_readiness_gate_20260502.json")
    claim = _load_json(REPORTS / "stwm_fstf_final_claim_boundary_v13_20260502.json")
    scaling = _load_json(REPORTS / "stwm_fstf_scaling_claim_verification_v13_20260502.json")
    trace_h = _load_json(REPORTS / "stwm_fstf_trace_conditioning_horizon_v13_20260502.json")
    density = _load_json(REPORTS / "stwm_fstf_trace_density_valid_units_audit_v13_20260502.json")
    lines = [
        "# STWM Current Evidence Brief",
        "",
        "## Fixed Definition",
        "- STWM-FSTF is Future Semantic Trace Field Prediction.",
        "- Input: frozen video-derived trace state plus observed semantic memory.",
        "- Output: future trace units and future semantic prototype field, with visibility/reappearance/identity auxiliaries only when valid.",
        "- Forbidden: candidate scorer, SAM2/CoTracker as same-output baseline, future candidate leakage, raw-video end-to-end training claim, dense trace-field claim without valid K evidence.",
        "",
        "## Current Claim Flags",
        f"- prototype_scaling_positive: `{readiness.get('corrected_prototype_scaling_positive')}`",
        f"- horizon_scaling_positive: `{readiness.get('corrected_horizon_scaling_positive')}`",
        f"- trace_density_scaling_positive: `{readiness.get('corrected_trace_density_scaling_positive')}`",
        f"- model_size_scaling_positive: `{readiness.get('corrected_model_size_scaling_positive')}`",
        f"- dense_trace_field_claim_allowed: `{readiness.get('dense_trace_field_claim_allowed')}`",
        f"- long_horizon_claim_allowed: `{readiness.get('long_horizon_claim_allowed')}`",
        f"- raw_visualization_pack_ready: `{readiness.get('raw_visualization_pack_ready')}`",
        f"- next_step_choice: `{readiness.get('next_step_choice')}`",
        "",
        "## Important Evidence",
        "- V8: STWM significantly beats the strongest controlled copy-aware same-output baseline (`copy_residual_mlp`).",
        "- V10: H8 future-hidden trace-rollout representation is load-bearing; old V9 no-trace ablation was not valid.",
        f"- V13 H16/H24 hidden audit: H16=`{trace_h.get('future_hidden_load_bearing_at_H16')}`, H24=`{trace_h.get('future_hidden_load_bearing_at_H24')}`.",
        f"- C selection: selected_C=`{scaling.get('selected_C')}`, C128_overfit_or_fail=`{scaling.get('C128_overfit_or_fail')}`.",
        f"- K wording: `{density.get('recommended_wording')}`; dense claim allowed=`{density.get('dense_trace_field_claim_allowed')}`.",
        "",
        "## Allowed Strong Claims",
    ]
    for item in claim.get("allowed_strong_claims", []):
        lines.append(f"- {item}")
    lines.append("")
    lines.append("## Forbidden Claims")
    for item in claim.get("forbidden_claims", []):
        lines.append(f"- {item}")
    lines.append("")
    lines.append("## Primary Report Pointers")
    for path in [
        "reports/stwm_fstf_v13_cvpr_readiness_gate_20260502.json",
        "reports/stwm_fstf_final_claim_boundary_v13_20260502.json",
        "reports/stwm_fstf_scaling_claim_verification_v13_20260502.json",
        "reports/stwm_fstf_trace_conditioning_horizon_v13_20260502.json",
        "reports/stwm_fstf_trace_density_valid_units_audit_v13_20260502.json",
        "reports/stwm_fstf_strong_copyaware_baseline_suite_v8_20260501.json",
    ]:
        lines.append(f"- `{path}`")
    (DOCS / "STWM_CURRENT_EVIDENCE_BRIEF_20260502.md").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _write_archive_manifest(*, archive_path: Path, archived: list[dict[str, Any]], kept_count: int) -> None:
    by_prefix = Counter(row["prefix"] for row in archived)
    lines = [
        "# STWM Docs Archive Manifest",
        "",
        f"- kept_current_docs: `{kept_count}`",
        f"- archived_historical_docs: `{len(archived)}`",
        f"- archive_path: `{archive_path.relative_to(ROOT)}`",
        "",
        "## Archived Prefix Counts",
    ]
    for prefix, count in by_prefix.most_common():
        lines.append(f"- `{prefix}`: `{count}`")
    lines.extend(
        [
            "",
            "## Restore",
            "To inspect an archived doc without restoring everything:",
            "",
            "```bash",
            f"tar -tzf {archive_path.relative_to(ROOT)} | grep '<DOC_NAME>'",
            f"tar -xzf {archive_path.relative_to(ROOT)} docs/<DOC_NAME>",
            "```",
        ]
    )
    (DOCS / "STWM_DOCS_ARCHIVE_MANIFEST_20260502.md").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _prefix(name: str) -> str:
    parts = name.split("_")
    if len(parts) >= 2:
        return "_".join(parts[:2])
    return parts[0]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    execute = bool(args.execute)
    all_docs = sorted(p for p in DOCS.glob("*.md") if p.is_file())
    keep_names = set(KEEP_NAMES) | set(NEW_DOCS)
    kept = [p for p in all_docs if p.name in KEEP_NAMES]
    to_archive = [p for p in all_docs if p.name not in keep_names]
    archive_path = ARTIFACTS / "stwm_docs_historical_archive_v14_20260502.tar.gz"
    archived_rows: list[dict[str, Any]] = []
    for path in to_archive:
        archived_rows.append(
            {
                "path": str(path.relative_to(ROOT)),
                "size_bytes": int(path.stat().st_size),
                "sha256": _sha256(path),
                "prefix": _prefix(path.name),
            }
        )
    if execute:
        ARTIFACTS.mkdir(parents=True, exist_ok=True)
        if to_archive:
            with tarfile.open(archive_path, "w:gz") as tar:
                for path in to_archive:
                    tar.add(path, arcname=str(path.relative_to(ROOT)))
        _write_evidence_brief()
        # Recompute kept after creating new docs.
        kept_after = sorted(p for p in DOCS.glob("*.md") if p.is_file() and p.name in keep_names)
        _write_current_index(archive_path=archive_path, kept=kept_after, archived=archived_rows)
        _write_archive_manifest(archive_path=archive_path, archived=archived_rows, kept_count=len(kept_after))
        for path in to_archive:
            if path.exists():
                path.unlink()
    payload = {
        "audit_name": "stwm_docs_only_consolidation_v14",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "execute": execute,
        "docs_before_count": len(all_docs),
        "kept_current_doc_count": len([p for p in DOCS.glob("*.md") if p.is_file() and p.name in keep_names]) if execute else len(kept) + len(NEW_DOCS),
        "archived_historical_doc_count": len(to_archive),
        "archive_path": str(archive_path.relative_to(ROOT)),
        "archive_size_bytes": int(archive_path.stat().st_size) if archive_path.exists() else 0,
        "new_index_doc": "docs/STWM_DOCS_CURRENT_INDEX_20260502.md",
        "new_evidence_brief": "docs/STWM_CURRENT_EVIDENCE_BRIEF_20260502.md",
        "new_archive_manifest": "docs/STWM_DOCS_ARCHIVE_MANIFEST_20260502.md",
        "kept_current_docs": sorted(keep_names),
        "archived_docs": archived_rows,
        "policy": "Docs-only consolidation: keep current evidence chain in docs/, archive historical/intermediate docs in artifacts/ without touching reports/logs/outputs/data.",
    }
    _dump_json(REPORTS / "stwm_docs_only_consolidation_v14_20260502.json", payload)
    print(REPORTS / "stwm_docs_only_consolidation_v14_20260502.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
