#!/usr/bin/env python3
from __future__ import annotations

from argparse import ArgumentParser
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import json


SUBSETS = ["long_gap_persistence", "occlusion_reappearance", "crossing_ambiguity", "OOD_hard", "appearance_change"]


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _usable_item(item: dict[str, Any]) -> tuple[bool, str | None]:
    if not isinstance(item.get("frame_paths"), list) or not item.get("frame_paths"):
        return False, "missing_frame_paths"
    if item.get("future_frame_index") is None:
        return False, "missing_future_frame_index"
    target = item.get("observed_target") if isinstance(item.get("observed_target"), dict) else {}
    if not (target.get("bbox") or target.get("mask_rle") or target.get("mask_path") or target.get("point_prompt")):
        return False, "missing_observed_target_prompt"
    cands = item.get("future_candidates") if isinstance(item.get("future_candidates"), list) else []
    if not cands:
        return False, "missing_future_candidates"
    if item.get("gt_candidate_id") is None:
        return False, "missing_gt_candidate_id"
    return True, None


def build(source: Path, output: Path, doc: Path) -> dict[str, Any]:
    raw = load_json(source)
    items = raw.get("items") if isinstance(raw, dict) else []
    if not isinstance(items, list):
        items = []
    records: list[dict[str, Any]] = []
    pos = neg = 0
    subset_counts: dict[str, Counter[str]] = {s: Counter() for s in SUBSETS}
    exclusion_counts: Counter[str] = Counter()
    item_count = 0
    usable_item_count = 0
    for item in items:
        if not isinstance(item, dict):
            continue
        item_count += 1
        usable, reason = _usable_item(item)
        cands = item.get("future_candidates") if isinstance(item.get("future_candidates"), list) else []
        if not usable:
            exclusion_counts[reason or "unusable"] += max(len(cands), 1)
            continue
        usable_item_count += 1
        gt = str(item.get("gt_candidate_id"))
        tags = item.get("subset_tags") if isinstance(item.get("subset_tags"), dict) else {}
        for cand_idx, cand in enumerate(cands):
            if not isinstance(cand, dict):
                exclusion_counts["invalid_candidate_record"] += 1
                continue
            cand_id = str(cand.get("candidate_id"))
            if not cand_id:
                exclusion_counts["missing_candidate_id"] += 1
                continue
            if not (cand.get("bbox") or cand.get("mask_rle") or cand.get("mask_path")):
                exclusion_counts["missing_candidate_geometry"] += 1
                continue
            label = 1 if cand_id == gt else 0
            pos += int(label == 1)
            neg += int(label == 0)
            for subset in SUBSETS:
                if bool(tags.get(subset)):
                    subset_counts[subset]["positive" if label else "negative"] += 1
            item_id = str(item.get("item_id") or item.get("protocol_item_id") or item_count)
            records.append(
                {
                    "record_id": f"{item_id}::{cand_id}",
                    "item_id": item_id,
                    "source_dataset": item.get("dataset"),
                    "video_id": item.get("video_id"),
                    "frame_paths": item.get("frame_paths"),
                    "observed_frame_indices": item.get("observed_frame_indices"),
                    "future_frame_index": item.get("future_frame_index"),
                    "observed_target": item.get("observed_target"),
                    "candidate": cand,
                    "candidate_index": cand_idx,
                    "label_same_identity": label,
                    "subset_tags": tags,
                    "image_size": item.get("image_size"),
                    "target_quality": "external_candidate_expanded",
                    "model_input_mode": "query_observed_target_only",
                    "future_candidate_used_as_input": False,
                    "future_candidate_used_for_eval_only": True,
                    "exclusion_reason": None,
                }
            )
    payload = {
        "generated_at_utc": now_iso(),
        "source_manifest": str(source),
        "manifest_schema_version": "external_hardcase_candidate_expanded_v2",
        "original_item_count": item_count,
        "usable_original_item_count": usable_item_count,
        "candidate_record_count": len(records),
        "positive_candidate_count": pos,
        "negative_candidate_count": neg,
        "subset_candidate_counts": {k: dict(v) for k, v in subset_counts.items()},
        "target_quality": "external_candidate_expanded",
        "model_input_mode": "query_observed_target_only",
        "future_candidate_used_as_input": False,
        "future_candidate_used_for_eval_only": True,
        "exclusion_reason_counts": dict(exclusion_counts),
        "records": records,
    }
    write_json(output, payload)
    lines = [
        "# STWM External Hardcase Candidate Expanded Manifest V2",
        "",
        f"- original_item_count: `{item_count}`",
        f"- usable_original_item_count: `{usable_item_count}`",
        f"- candidate_record_count: `{len(records)}`",
        f"- positive_candidate_count: `{pos}`",
        f"- negative_candidate_count: `{neg}`",
        "- future_candidate_used_as_input: `false`",
        "- future_candidate_used_for_eval_only: `true`",
    ]
    doc.parent.mkdir(parents=True, exist_ok=True)
    doc.write_text("\n".join(lines).rstrip() + "\n")
    return payload


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--source", default="reports/stwm_external_baseline_item_manifest_20260426.json")
    parser.add_argument("--output", default="reports/stwm_external_hardcase_candidate_expanded_manifest_v2_20260428.json")
    parser.add_argument("--doc", default="docs/STWM_EXTERNAL_HARDCASE_CANDIDATE_EXPANDED_MANIFEST_V2_20260428.md")
    args = parser.parse_args()
    build(Path(args.source), Path(args.output), Path(args.doc))


if __name__ == "__main__":
    main()
