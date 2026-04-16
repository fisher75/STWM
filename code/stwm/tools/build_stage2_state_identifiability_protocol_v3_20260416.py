#!/usr/bin/env python3
from __future__ import annotations

from argparse import ArgumentParser
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple
import json

from stwm.tools import build_stage2_state_identifiability_protocol_20260415 as prev


ROOT = prev.ROOT
PANEL_TARGETS_V3 = {
    "occlusion_reappearance": 40,
    "crossing_ambiguity": 40,
    "small_object": 40,
    "appearance_change": 40,
    "long_gap_persistence": 40,
}
MIN_TOTAL_ITEMS = 120
IDEAL_TOTAL_ITEMS = 200


def _candidate_tags(item: Dict[str, Any]) -> List[str]:
    return [str(x) for x in item.get("subset_tags", []) if str(x) in PANEL_TARGETS_V3]


def _difficulty(item: Dict[str, Any]) -> float:
    stats = item.get("stats", {}) if isinstance(item.get("stats", {}), dict) else {}
    return float(stats.get("difficulty_score", 0.0))


def _sorted_candidates(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(
        items,
        key=lambda item: (-_difficulty(item), str(item.get("protocol_item_id", ""))),
    )


def _select_candidates_v3(all_candidates: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, List[str]], Dict[str, Any]]:
    ranked = _sorted_candidates(all_candidates)
    panel_members: Dict[str, List[str]] = {key: [] for key in PANEL_TARGETS_V3}
    selected: List[Dict[str, Any]] = []
    selected_ids: set[str] = set()
    panel_candidate_counts = {
        panel: int(sum(panel in _candidate_tags(item) for item in ranked))
        for panel in PANEL_TARGETS_V3
    }

    for item in ranked:
        tags = _candidate_tags(item)
        if not tags:
            continue
        useful = [tag for tag in tags if len(panel_members[tag]) < int(PANEL_TARGETS_V3[tag])]
        if not useful:
            continue
        item_id = str(item.get("protocol_item_id", ""))
        if item_id not in selected_ids:
            selected.append(item)
            selected_ids.add(item_id)
        for tag in useful:
            panel_members[tag].append(item_id)
        if (
            len(selected) >= IDEAL_TOTAL_ITEMS
            and all(len(panel_members[tag]) >= int(PANEL_TARGETS_V3[tag]) for tag in PANEL_TARGETS_V3)
        ):
            break

    for item in ranked:
        if len(selected) >= IDEAL_TOTAL_ITEMS:
            break
        item_id = str(item.get("protocol_item_id", ""))
        if item_id in selected_ids:
            continue
        tags = _candidate_tags(item)
        if not tags:
            continue
        selected.append(item)
        selected_ids.add(item_id)

    shortage_reasons: Dict[str, str] = {}
    for panel, target in PANEL_TARGETS_V3.items():
        actual = len(panel_members[panel])
        if actual < int(target):
            shortage_reasons[panel] = (
                f"candidate_pool_shortage: target={int(target)} actual={actual} available={panel_candidate_counts[panel]}"
            )
    if len(selected) < MIN_TOTAL_ITEMS:
        shortage_reasons["full_protocol"] = (
            f"candidate_pool_shortage: min_total={MIN_TOTAL_ITEMS} actual={len(selected)}"
        )

    selection_meta = {
        "panel_candidate_counts": panel_candidate_counts,
        "shortage_reasons": shortage_reasons,
        "selected_item_count": int(len(selected)),
    }
    return selected, panel_members, selection_meta


def parse_args() -> Any:
    parser = ArgumentParser(description="Build benchmark-scale Stage2 state-identifiability protocol v3")
    parser.add_argument("--contract-json", default=str(ROOT / "reports/stage2_bootstrap_data_contract_20260408.json"))
    parser.add_argument("--output-json", default=str(ROOT / "reports/stage2_state_identifiability_protocol_v3_20260416.json"))
    parser.add_argument("--output-md", default=str(ROOT / "docs/STAGE2_STATE_IDENTIFIABILITY_PROTOCOL_V3_20260416.md"))
    parser.add_argument("--vipseg-max-clips", type=int, default=1000000)
    parser.add_argument("--burst-max-seqs", type=int, default=1000000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    contract = prev.read_json(args.contract_json)
    ds_map = {
        prev._norm_name(str(rec.get("dataset_name", ""))): rec
        for rec in contract.get("datasets", [])
        if isinstance(rec, dict)
    }
    vipseg = ds_map["VIPSEG"]
    burst = ds_map["BURST"]

    vipseg_split = Path(vipseg["split_mapping"]["val"]["split_file"])
    vipseg_frame_root = Path(vipseg["split_mapping"]["val"]["frame_root"])
    vipseg_mask_root = Path(vipseg["split_mapping"]["val"]["mask_root"])
    vipseg_ids = prev._read_split_ids(vipseg_split)[: max(int(args.vipseg_max_clips), 1)]

    all_candidates: List[Dict[str, Any]] = []
    vipseg_candidate_count = 0
    for clip_id in vipseg_ids:
        frame_paths = prev._list_visible_files(vipseg_frame_root / clip_id, [".jpg", ".jpeg", ".png"])
        mask_paths = prev._list_visible_files(vipseg_mask_root / clip_id, [".png"])
        if len(frame_paths) < prev.TOTAL_STEPS or len(mask_paths) < prev.TOTAL_STEPS:
            continue
        candidates = prev._vipseg_candidates_for_clip(clip_id=clip_id, frame_paths=frame_paths, mask_paths=mask_paths)
        vipseg_candidate_count += len(candidates)
        all_candidates.extend(candidates)

    burst_cfg = burst["split_mapping"]["val"]
    burst_annotation_file = Path(burst_cfg["annotation_file"])
    burst_frames_root = Path(burst_cfg["frames_root"])
    burst_payload = prev.read_json(burst_annotation_file)
    burst_sequences = burst_payload.get("sequences", []) if isinstance(burst_payload.get("sequences", []), list) else []
    burst_candidate_count = 0
    for seq in burst_sequences[: max(int(args.burst_max_seqs), 1)]:
        if not isinstance(seq, dict):
            continue
        candidates = prev._burst_candidates(seq=seq, annotation_file=burst_annotation_file, frames_root=burst_frames_root)
        burst_candidate_count += len(candidates)
        all_candidates.extend(candidates)

    selected_items, panel_members, selection_meta = _select_candidates_v3(all_candidates)
    selected_ids = {str(item.get("protocol_item_id", "")) for item in selected_items}
    per_dataset_counts = dict(Counter(str(item.get("dataset", "")) for item in selected_items))
    panel_counts = {
        "full_identifiability_panel": int(len(selected_items)),
        **{panel: int(len(ids)) for panel, ids in panel_members.items()},
    }
    payload = {
        "generated_at_utc": prev.now_iso(),
        "protocol_name": "Stage2 real state-identifiability / future grounding protocol v3",
        "protocol_version": "20260416",
        "protocol_contribution": True,
        "task_definition": {
            "query": "given a historical object / mask query, recover the same target from future state / future candidate masks",
            "query_frame_index_convention": int(prev.QUERY_STEP),
            "future_target_frame_index_convention": int(prev.FUTURE_STEP),
            "uses_real_instance_identity": True,
            "uses_real_future_masks": True,
            "rollout_error_proxy_only": False,
        },
        "dataset_policy": {
            "VIPSeg": "primary real instance-identifiability panel",
            "BURST": "primary long-gap persistence / continuity panel",
            "VSPW": "scene/stuff supplemental only; excluded from main protocol",
        },
        "item_selection_rules": {
            "primary_sort_key": "difficulty_score descending",
            "panel_balance_policy": "fill panel targets first, then top up globally with highest-ranked remaining items",
            "minimum_total_items": int(MIN_TOTAL_ITEMS),
            "ideal_total_items": int(IDEAL_TOTAL_ITEMS),
            "no_leakage_policy": "validation split only, no train overlap, unique protocol_item_id per selected target",
            "no_overlap_policy": "same protocol_item_id cannot appear twice in full protocol",
        },
        "panel_targets": {k: int(v) for k, v in PANEL_TARGETS_V3.items()},
        "panel_counts": panel_counts,
        "per_dataset_counts": per_dataset_counts,
        "scan_stats": {
            "vipseg_val_clip_count_scanned": int(len(vipseg_ids)),
            "vipseg_candidate_count": int(vipseg_candidate_count),
            "burst_val_sequence_count_scanned": int(min(len(burst_sequences), int(args.burst_max_seqs))),
            "burst_candidate_count": int(burst_candidate_count),
            "total_candidate_count": int(len(all_candidates)),
            "selected_protocol_item_count": int(len(selected_items)),
        },
        "selection_meta": selection_meta,
        "panel_members": panel_members,
        "selected_protocol_item_ids": sorted(selected_ids),
        "items": selected_items,
    }
    prev.write_json(args.output_json, payload)
    prev.write_md(
        args.output_md,
        [
            "# Stage2 State-Identifiability Protocol V3 20260416",
            "",
            "- protocol_contribution: true",
            "- rollout_error_proxy_only: false",
            "- primary datasets: VIPSeg, BURST",
            "- VSPW excluded from main instance-identifiability protocol",
            f"- selected_protocol_item_count: {len(selected_items)}",
            f"- per_dataset_counts: {json.dumps(per_dataset_counts, ensure_ascii=True)}",
            "",
            "## Panel Counts",
            "",
            *[f"- {k}: {v}" for k, v in panel_counts.items()],
            "",
            "## Selection Policy",
            "",
            f"- minimum_total_items: {MIN_TOTAL_ITEMS}",
            f"- ideal_total_items: {IDEAL_TOTAL_ITEMS}",
            "- panel targets filled first, then global top-up from remaining highest-difficulty items",
            "- no leakage: validation split only",
            "- no overlap: unique protocol_item_id",
            "",
            "## Shortage Notes",
            "",
            *(
                [f"- {k}: {v}" for k, v in selection_meta["shortage_reasons"].items()]
                if selection_meta["shortage_reasons"]
                else ["- none"]
            ),
        ],
    )
    print(json.dumps(payload, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
