from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import json
import math
from typing import Any


QUERY_TYPES = [
    "same_category_distractor",
    "spatial_disambiguation",
    "relation_conditioned_query",
    "future_conditioned_reappearance_aware",
]


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Build STWM V4.2 state-identifiability protocol manifest")
    parser.add_argument("--source-manifest", default="/home/chen034/workspace/stwm/manifests/minisplits/stwm_week2_minival_v2.json")
    parser.add_argument("--eventful-report", default="/home/chen034/workspace/stwm/reports/stwm_v4_2_eventful_protocol_v1.json")
    parser.add_argument("--hard-query-report", default="/home/chen034/workspace/stwm/reports/stwm_v4_2_hard_query_protocol_v1.json")
    parser.add_argument("--target-size", type=int, default=18)
    parser.add_argument("--relation-min-label-count", type=float, default=5.0)
    parser.add_argument("--relation-min-ambiguity", type=float, default=0.25)
    parser.add_argument(
        "--output-manifest",
        default="/home/chen034/workspace/stwm/manifests/minisplits/stwm_v4_2_state_identifiability_v1.json",
    )
    parser.add_argument(
        "--output-clip-ids",
        default="/home/chen034/workspace/stwm/manifests/minisplits/stwm_v4_2_state_identifiability_clip_ids_v1.json",
    )
    parser.add_argument(
        "--output-report",
        default="/home/chen034/workspace/stwm/reports/stwm_v4_2_state_identifiability_protocol_v1.json",
    )
    return parser


def _dedupe_keep_order(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        key = str(value)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out


def _candidate_rows(eventful_report: dict[str, Any]) -> list[dict[str, Any]]:
    selected = eventful_report.get("selected_clips", [])
    if isinstance(selected, list) and selected:
        return [x for x in selected if isinstance(x, dict)]
    fallback = eventful_report.get("all_candidates_top30", [])
    if isinstance(fallback, list):
        return [x for x in fallback if isinstance(x, dict)]
    return []


def _hard_query_type_map(hard_query_report: dict[str, Any]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    rows = hard_query_report.get("selected_clips", [])
    if not isinstance(rows, list):
        return out
    for row in rows:
        if not isinstance(row, dict):
            continue
        clip_id = str(row.get("clip_id", ""))
        if not clip_id:
            continue
        htypes = row.get("hard_query_types", [])
        if not isinstance(htypes, list):
            htypes = []
        out[clip_id] = [str(x) for x in htypes if str(x)]
    return out


def _spatial_tags(cx: float, cy: float) -> tuple[str, str, str]:
    if cx < 0.33:
        horizontal = "left"
    elif cx > 0.67:
        horizontal = "right"
    else:
        horizontal = "center"

    if cy < 0.33:
        vertical = "top"
    elif cy > 0.67:
        vertical = "bottom"
    else:
        vertical = "middle"

    region = f"{vertical}-{horizontal}"
    return horizontal, vertical, region


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _mean_std(vals: list[float]) -> dict[str, float | int]:
    if not vals:
        return {"count": 0, "mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    mean = float(sum(vals) / len(vals))
    var = float(sum((v - mean) ** 2 for v in vals) / max(1, len(vals)))
    return {
        "count": int(len(vals)),
        "mean": mean,
        "std": float(math.sqrt(max(0.0, var))),
        "min": float(min(vals)),
        "max": float(max(vals)),
    }


def _difficulty_bucket(score: float) -> str:
    if score < 0.34:
        return "easy"
    if score < 0.67:
        return "medium"
    return "hard"


def _difficulty_components(row: dict[str, Any]) -> dict[str, float]:
    label_stats = row.get("label_stats", {}) if isinstance(row.get("label_stats"), dict) else {}

    ambiguity = _clamp01(float(row.get("identity_ambiguity_score", 0.0)))
    crowd = _clamp01(float(row.get("avg_label_count_per_frame", 0.0)) / 20.0)
    motion = _clamp01(float(label_stats.get("mean_motion", 0.0)) / 0.15)
    area_cv = _clamp01(float(label_stats.get("area_cv", 0.0)) / 3.0)
    missing_span = _clamp01(float(label_stats.get("longest_missing_span", 0.0)) / 24.0)
    reappearance = _clamp01(float(label_stats.get("reappearance_events", 0.0)) / 4.0)
    reconnect = _clamp01(float(label_stats.get("reconnect_events", 0.0)) / 4.0)
    cx = float(label_stats.get("centroid_mean_x", 0.5))
    cy = float(label_stats.get("centroid_mean_y", 0.5))
    center_dist = min(1.0, (abs(cx - 0.5) + abs(cy - 0.5)) * 1.2)
    spatial_ambiguity = 1.0 - center_dist

    return {
        "ambiguity": ambiguity,
        "crowd": crowd,
        "motion": motion,
        "area_cv": area_cv,
        "missing_span": missing_span,
        "reappearance": reappearance,
        "reconnect": reconnect,
        "spatial_ambiguity": _clamp01(spatial_ambiguity),
    }


def _difficulty_for_type(query_type: str, comps: dict[str, float]) -> float:
    if query_type == "same_category_distractor":
        return _clamp01(0.55 * comps["ambiguity"] + 0.25 * comps["crowd"] + 0.20 * comps["area_cv"])
    if query_type == "spatial_disambiguation":
        return _clamp01(0.50 * comps["spatial_ambiguity"] + 0.30 * comps["motion"] + 0.20 * comps["crowd"])
    if query_type == "relation_conditioned_query":
        return _clamp01(0.40 * comps["crowd"] + 0.35 * comps["ambiguity"] + 0.25 * comps["motion"])
    if query_type == "future_conditioned_reappearance_aware":
        return _clamp01(
            0.35 * comps["missing_span"]
            + 0.30 * comps["reappearance"]
            + 0.20 * comps["reconnect"]
            + 0.15 * comps["motion"]
        )
    return 0.0


def _taxonomy() -> list[dict[str, str]]:
    return [
        {
            "query_type": "same_category_distractor",
            "definition": "Target and distractor share category-level semantics; model must identify the intended instance.",
            "difficulty_definition": "0.55*ambiguity + 0.25*crowd + 0.20*area_cv",
        },
        {
            "query_type": "spatial_disambiguation",
            "definition": "Target is identified by spatial disambiguation cues such as left/right/top/bottom/region.",
            "difficulty_definition": "0.50*spatial_ambiguity + 0.30*motion + 0.20*crowd",
        },
        {
            "query_type": "relation_conditioned_query",
            "definition": "Target is selected by relation-conditioned grounding against another object context.",
            "difficulty_definition": "0.40*crowd + 0.35*ambiguity + 0.25*motion",
        },
        {
            "query_type": "future_conditioned_reappearance_aware",
            "definition": "Target requires future-aware grounding under reappearance/reconnect or visibility transitions.",
            "difficulty_definition": "0.35*missing_span + 0.30*reappearance + 0.20*reconnect + 0.15*motion",
        },
    ]


def _select_query_types(
    row: dict[str, Any],
    hard_types: list[str],
    relation_min_label_count: float,
    relation_min_ambiguity: float,
) -> tuple[list[str], list[str]]:
    reasons = [str(x) for x in row.get("selected_reasons", []) if str(x)]
    secondary_id = row.get("secondary_label_id")
    try:
        secondary_id = int(secondary_id) if secondary_id is not None else None
    except (TypeError, ValueError):
        secondary_id = None

    label_stats = row.get("label_stats", {}) if isinstance(row.get("label_stats"), dict) else {}
    cx = float(label_stats.get("centroid_mean_x", 0.5))
    cy = float(label_stats.get("centroid_mean_y", 0.5))
    h, v, region = _spatial_tags(cx, cy)

    ambiguity = float(row.get("identity_ambiguity_score", 0.0))
    crowd = float(row.get("avg_label_count_per_frame", 0.0))

    query_types: list[str] = []
    query_tags: list[str] = []

    if secondary_id is not None or "same_category_distractor" in hard_types:
        query_types.append("same_category_distractor")
        query_tags.append("same-category-distractor-query")

    query_types.append("spatial_disambiguation")
    query_tags.extend([f"spatial-{h}", f"spatial-{v}", f"region-{region}"])

    if secondary_id is not None and (crowd >= float(relation_min_label_count) or ambiguity >= float(relation_min_ambiguity)):
        query_types.append("relation_conditioned_query")
        query_tags.append("relation-conditioned-query")
    elif "identity_ambiguity" in reasons and secondary_id is not None:
        query_types.append("relation_conditioned_query")
        query_tags.append("relation-conditioned-query")

    has_future = (
        "reappearance" in reasons
        or "reconnect" in reasons
        or "visibility_flip" in reasons
        or "reappearing_object_query" in hard_types
    )
    if has_future:
        query_types.append("future_conditioned_reappearance_aware")
        query_tags.append("future-reappearance-aware-query")

    return _dedupe_keep_order(query_types), _dedupe_keep_order(query_tags)


def main() -> None:
    args = build_parser().parse_args()

    items = json.loads(Path(args.source_manifest).read_text())
    item_map = {str(item.get("clip_id", "")): item for item in items}

    eventful_report = json.loads(Path(args.eventful_report).read_text())
    hard_query_report = json.loads(Path(args.hard_query_report).read_text())

    hard_type_map = _hard_query_type_map(hard_query_report)

    candidates = _candidate_rows(eventful_report)
    ranked = sorted(candidates, key=lambda x: float(x.get("selection_score", 0.0)), reverse=True)
    selected_rows = ranked[: max(1, int(args.target_size))]

    out_manifest_data: list[dict[str, Any]] = []
    selected_summary: list[dict[str, Any]] = []
    type_counts = {q: 0 for q in QUERY_TYPES}
    difficulty_values = {q: [] for q in QUERY_TYPES}
    difficulty_buckets = {q: {"easy": 0, "medium": 0, "hard": 0} for q in QUERY_TYPES}

    for row in selected_rows:
        clip_id = str(row.get("clip_id", ""))
        if not clip_id or clip_id not in item_map:
            continue

        src = dict(item_map[clip_id])
        md = dict(src.get("metadata", {}))

        hard_types = hard_type_map.get(clip_id, [])
        query_types, query_tags = _select_query_types(
            row,
            hard_types=hard_types,
            relation_min_label_count=float(args.relation_min_label_count),
            relation_min_ambiguity=float(args.relation_min_ambiguity),
        )

        if not query_types:
            continue

        comps = _difficulty_components(row)
        difficulty_by_type: dict[str, float] = {}
        difficulty_bucket_by_type: dict[str, str] = {}
        for qtype in query_types:
            score = _difficulty_for_type(qtype, comps)
            bucket = _difficulty_bucket(score)
            difficulty_by_type[qtype] = float(score)
            difficulty_bucket_by_type[qtype] = bucket

            if qtype in type_counts:
                type_counts[qtype] += 1
                difficulty_values[qtype].append(float(score))
                difficulty_buckets[qtype][bucket] = int(difficulty_buckets[qtype][bucket] + 1)

        target_id = row.get("target_label_id")
        if target_id is not None:
            try:
                md["target_label_id"] = int(target_id)
            except (TypeError, ValueError):
                pass

        text_labels = [str(x) for x in src.get("text_labels", []) if str(x)]
        src["text_labels"] = _dedupe_keep_order(text_labels + query_tags)

        md["state_identifiability_protocol"] = {
            "query_types": query_types,
            "query_tags": query_tags,
            "selected_reasons": [str(x) for x in row.get("selected_reasons", []) if str(x)],
            "selection_score": float(row.get("selection_score", 0.0)),
            "target_label_id": int(target_id) if target_id is not None else None,
            "secondary_label_id": int(row.get("secondary_label_id")) if row.get("secondary_label_id") is not None else None,
            "difficulty_by_type": difficulty_by_type,
            "difficulty_bucket_by_type": difficulty_bucket_by_type,
            "difficulty_components": comps,
            "source_eventful_report": str(args.eventful_report),
            "source_hard_query_report": str(args.hard_query_report),
        }
        src["metadata"] = md
        out_manifest_data.append(src)

        selected_summary.append(
            {
                "clip_id": clip_id,
                "query_types": query_types,
                "query_tags": query_tags,
                "selected_reasons": [str(x) for x in row.get("selected_reasons", []) if str(x)],
                "selection_score": float(row.get("selection_score", 0.0)),
                "target_label_id": int(target_id) if target_id is not None else None,
                "secondary_label_id": int(row.get("secondary_label_id")) if row.get("secondary_label_id") is not None else None,
                "difficulty_by_type": difficulty_by_type,
                "difficulty_bucket_by_type": difficulty_bucket_by_type,
            }
        )

    selected_count = len(out_manifest_data)
    type_ratios = {q: (float(type_counts[q]) / max(1, selected_count)) for q in QUERY_TYPES}

    type_difficulty_stats: dict[str, Any] = {}
    for q in QUERY_TYPES:
        type_difficulty_stats[q] = {
            **_mean_std(difficulty_values[q]),
            "bucket_counts": difficulty_buckets[q],
        }

    coverage_insufficient = bool(
        selected_count < max(1, int(args.target_size) // 2)
        or any(int(type_counts[q]) <= 0 for q in QUERY_TYPES)
    )

    report = {
        "source_manifest": str(args.source_manifest),
        "eventful_report": str(args.eventful_report),
        "hard_query_report": str(args.hard_query_report),
        "target_size": int(args.target_size),
        "selected_count": int(selected_count),
        "coverage_insufficient": coverage_insufficient,
        "taxonomy": _taxonomy(),
        "query_type_counts": type_counts,
        "query_type_ratios": type_ratios,
        "query_type_difficulty": type_difficulty_stats,
        "selected_clip_ids": [str(item.get("clip_id", "")) for item in selected_summary],
        "selected_clips": selected_summary,
    }

    out_manifest = Path(args.output_manifest)
    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    out_manifest.write_text(json.dumps(out_manifest_data, indent=2))

    out_clip_ids = Path(args.output_clip_ids)
    out_clip_ids.parent.mkdir(parents=True, exist_ok=True)
    out_clip_ids.write_text(json.dumps([str(item.get("clip_id", "")) for item in selected_summary], indent=2))

    out_report = Path(args.output_report)
    out_report.parent.mkdir(parents=True, exist_ok=True)
    out_report.write_text(json.dumps(report, indent=2))

    print(
        json.dumps(
            {
                "output_manifest": str(out_manifest),
                "output_clip_ids": str(out_clip_ids),
                "output_report": str(out_report),
                "selected_count": selected_count,
                "coverage_insufficient": coverage_insufficient,
                "query_type_counts": type_counts,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
