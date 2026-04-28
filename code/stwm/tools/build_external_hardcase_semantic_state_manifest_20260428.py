#!/usr/bin/env python3
from __future__ import annotations

from argparse import ArgumentParser
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import json
import os


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _has_prompt(target: Any) -> bool:
    if not isinstance(target, dict):
        return False
    return bool(target.get("bbox") or target.get("mask_path") or target.get("mask_rle") or target.get("point_prompt"))


def _candidate_target(item: dict[str, Any]) -> tuple[int | None, str | None]:
    gt = item.get("gt_candidate_id")
    if gt is None:
        return None, "missing_gt_candidate_id"
    cands = item.get("future_candidates")
    if not isinstance(cands, list) or not cands:
        return None, "missing_future_candidates"
    for idx, cand in enumerate(cands):
        if isinstance(cand, dict) and str(cand.get("candidate_id")) == str(gt):
            return idx, None
    return None, "gt_candidate_id_not_in_future_candidates"


def _subset_tags(item: dict[str, Any]) -> dict[str, Any]:
    tags = item.get("subset_tags")
    return tags if isinstance(tags, dict) else {}


def build_manifest(source: Path, output: Path, doc: Path) -> dict[str, Any]:
    raw = load_json(source)
    items = raw.get("items") if isinstance(raw, dict) else []
    if not isinstance(items, list):
        items = []

    out_items: list[dict[str, Any]] = []
    target_type_counts: Counter[str] = Counter()
    model_input_counts: Counter[str] = Counter()
    usable_counts: Counter[str] = Counter()
    exclusion_counts: Counter[str] = Counter()

    for item in items:
        if not isinstance(item, dict):
            continue
        item_id = str(item.get("item_id") or item.get("protocol_item_id") or len(out_items))
        observed_target = item.get("observed_target") if isinstance(item.get("observed_target"), dict) else {}
        future_candidates = item.get("future_candidates") if isinstance(item.get("future_candidates"), list) else []
        candidate_index, candidate_reason = _candidate_target(item)
        has_frames = isinstance(item.get("frame_paths"), list) and bool(item.get("frame_paths"))
        has_future_frame = item.get("future_frame_index") is not None
        has_prompt = _has_prompt(observed_target)
        has_candidate_geometry = bool(
            future_candidates
            and all(isinstance(c, dict) and (c.get("bbox") or c.get("mask_path") or c.get("mask_rle")) for c in future_candidates)
        )

        # Current full STWM Stage2 export can only run Stage2SemanticDataset
        # samples. The external manifest has raw frames/prompts/candidates but no
        # verified Stage2 sample key/cache mapping, so do not pretend it is
        # full-model runnable.
        model_input_source = "external_item_only" if has_frames and has_prompt and has_future_frame else "unavailable"
        usable_for_full_model_export = False
        exclusion_reason = "missing_stage2_dataset_mapping_for_full_model_forward"
        if not has_frames:
            exclusion_reason = "missing_frame_paths"
        elif not has_prompt:
            exclusion_reason = "missing_observed_prompt"
        elif not has_future_frame:
            exclusion_reason = "missing_future_frame_index"

        if candidate_index is not None:
            target_type = "external_candidate_aligned"
            event_target = 1
            usable_for_candidate_eval = True
            usable_for_event_eval = True
        elif has_prompt and future_candidates:
            target_type = "weak_event_only"
            event_target = None
            usable_for_candidate_eval = False
            usable_for_event_eval = False
            exclusion_counts[candidate_reason or "candidate_target_unavailable"] += 1
        else:
            target_type = "unavailable"
            event_target = None
            usable_for_candidate_eval = False
            usable_for_event_eval = False
            exclusion_counts[candidate_reason or exclusion_reason] += 1

        per_horizon_visibility_available = False
        visibility_target_available = False
        target_type_counts[target_type] += 1
        model_input_counts[model_input_source] += 1
        if usable_for_event_eval:
            usable_counts["usable_for_event_eval"] += 1
        if usable_for_candidate_eval:
            usable_counts["usable_for_candidate_eval"] += 1
        if usable_for_full_model_export:
            usable_counts["usable_for_full_model_export"] += 1
        else:
            usable_counts["blocked"] += 1
            exclusion_counts[exclusion_reason] += 1

        out_items.append(
            {
                "item_id": item_id,
                "original_external_item": item,
                "source_dataset": item.get("dataset"),
                "source_video_id": item.get("video_id"),
                "sample_key": item.get("protocol_item_id"),
                "observed_range": item.get("observed_frame_indices"),
                "future_range": None,
                "future_frame": item.get("future_frame_index"),
                "observed_target": observed_target,
                "future_candidates": future_candidates,
                "gt_candidate_id": item.get("gt_candidate_id"),
                "subset_tags": _subset_tags(item),
                "model_input_source": model_input_source,
                "target_type": target_type,
                "event_reappearance_target": event_target,
                "candidate_match_target": candidate_index,
                "visibility_target_available": visibility_target_available,
                "per_horizon_visibility_available": per_horizon_visibility_available,
                "usable_for_full_model_export": usable_for_full_model_export,
                "usable_for_event_eval": usable_for_event_eval,
                "usable_for_candidate_eval": usable_for_candidate_eval,
                "exclusion_reason": None if usable_for_full_model_export else exclusion_reason,
                "target_exclusion_reason": candidate_reason,
                "stage2_dataset_mapping_key": None,
            }
        )

    payload = {
        "generated_at_utc": now_iso(),
        "source_manifest": str(source),
        "manifest_schema_version": "external_hardcase_semantic_state_manifest_v1",
        "item_count": len(out_items),
        "items": out_items,
        "usable_subsets": {
            "usable_for_full_model_export": usable_counts["usable_for_full_model_export"],
            "usable_for_event_eval": usable_counts["usable_for_event_eval"],
            "usable_for_candidate_eval": usable_counts["usable_for_candidate_eval"],
            "blocked": usable_counts["blocked"],
        },
        "target_type_counts": dict(target_type_counts),
        "model_input_source_counts": dict(model_input_counts),
        "exclusion_reason_counts": dict(exclusion_counts),
        "stage2_dataset_mapping_filled": False,
        "full_model_forward_possible_count": usable_counts["usable_for_full_model_export"],
    }
    write_json(output, payload)

    lines = [
        "# STWM External Hardcase Semantic-State Manifest V1",
        "",
        f"- source_manifest: `{source}`",
        f"- item_count: `{len(out_items)}`",
        f"- usable_for_full_model_export: `{usable_counts['usable_for_full_model_export']}`",
        f"- usable_for_event_eval: `{usable_counts['usable_for_event_eval']}`",
        f"- usable_for_candidate_eval: `{usable_counts['usable_for_candidate_eval']}`",
        f"- blocked: `{usable_counts['blocked']}`",
        "",
        "The manifest preserves external candidate-aligned targets but does not fabricate Stage2 dataset mappings. Current items are external raw payloads only, so full-model export remains blocked until a real Stage2 batch bridge exists.",
    ]
    doc.parent.mkdir(parents=True, exist_ok=True)
    doc.write_text("\n".join(lines).rstrip() + "\n")
    return payload


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--source", default="reports/stwm_external_baseline_item_manifest_20260426.json")
    parser.add_argument("--output", default="reports/stwm_external_hardcase_semantic_state_manifest_v1_20260428.json")
    parser.add_argument("--doc", default="docs/STWM_EXTERNAL_HARDCASE_SEMANTIC_STATE_MANIFEST_V1_20260428.md")
    args = parser.parse_args()
    build_manifest(Path(args.source), Path(args.output), Path(args.doc))


if __name__ == "__main__":
    main()
