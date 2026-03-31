from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import json
from typing import Any


HARD_QUERY_TYPES = [
    "same_category_distractor",
    "spatial_disambiguation",
    "reappearing_object_query",
]


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Build STWM V4.2 hard-query protocol manifest")
    parser.add_argument("--source-manifest", default="/home/chen034/workspace/stwm/manifests/minisplits/stwm_week2_minival_v2.json")
    parser.add_argument(
        "--eventful-report",
        default="/home/chen034/workspace/stwm/reports/stwm_v4_2_eventful_protocol_v1.json",
    )
    parser.add_argument("--target-size", type=int, default=18)
    parser.add_argument(
        "--output-manifest",
        default="/home/chen034/workspace/stwm/manifests/minisplits/stwm_v4_2_hard_query_minival_v1.json",
    )
    parser.add_argument(
        "--output-clip-ids",
        default="/home/chen034/workspace/stwm/manifests/minisplits/stwm_v4_2_hard_query_clip_ids_v1.json",
    )
    parser.add_argument(
        "--output-report",
        default="/home/chen034/workspace/stwm/reports/stwm_v4_2_hard_query_protocol_v1.json",
    )
    return parser


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


def _dedupe_keep_order(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for val in values:
        key = str(val)
        if key in seen:
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


def _build_hard_query_types(row: dict[str, Any]) -> tuple[list[str], list[str]]:
    reasons = [str(x) for x in row.get("selected_reasons", []) if str(x)]
    label_stats = row.get("label_stats", {}) if isinstance(row.get("label_stats"), dict) else {}

    target_id = row.get("target_label_id")
    secondary_id = row.get("secondary_label_id")
    cx = float(label_stats.get("centroid_mean_x", 0.5))
    cy = float(label_stats.get("centroid_mean_y", 0.5))
    h, v, region = _spatial_tags(cx, cy)

    types: list[str] = []
    tags: list[str] = []

    if secondary_id is not None:
        types.append("same_category_distractor")
        tags.extend(
            [
                "same-category-distractor-query",
                f"target-object-{int(target_id)}" if target_id is not None else "target-object",
                f"distractor-object-{int(secondary_id)}",
            ]
        )

    types.append("spatial_disambiguation")
    tags.extend([f"spatial-{h}", f"spatial-{v}", f"region-{region}"])

    if "reappearance" in reasons:
        types.append("reappearing_object_query")
        tags.append("reappearing-object-query")

    if "visibility_flip" in reasons:
        tags.append("visibility-flip-query")

    return _dedupe_keep_order(types), _dedupe_keep_order(tags)


def main() -> None:
    args = build_parser().parse_args()

    source_manifest = Path(args.source_manifest)
    items = json.loads(source_manifest.read_text())
    item_map = {str(item.get("clip_id", "")): item for item in items}

    eventful_report = json.loads(Path(args.eventful_report).read_text())
    candidate_rows = _candidate_rows(eventful_report)

    ranked_candidates = sorted(
        candidate_rows,
        key=lambda x: float(x.get("selection_score", 0.0)),
        reverse=True,
    )

    selected_rows = ranked_candidates[: max(1, int(args.target_size))]
    selected_ids = [str(x.get("clip_id", "")) for x in selected_rows if str(x.get("clip_id", ""))]

    out_manifest_data: list[dict[str, Any]] = []
    type_counts = {k: 0 for k in HARD_QUERY_TYPES}
    selected_summary: list[dict[str, Any]] = []

    for row in selected_rows:
        clip_id = str(row.get("clip_id", ""))
        if not clip_id or clip_id not in item_map:
            continue

        src = dict(item_map[clip_id])
        md = dict(src.get("metadata", {}))

        hard_types, hard_tags = _build_hard_query_types(row)
        for t in hard_types:
            if t in type_counts:
                type_counts[t] += 1

        target_id = row.get("target_label_id")
        if target_id is not None:
            try:
                md["target_label_id"] = int(target_id)
            except (TypeError, ValueError):
                pass

        text_labels = [str(x) for x in src.get("text_labels", []) if str(x)]
        text_labels = _dedupe_keep_order(text_labels + hard_tags)
        src["text_labels"] = text_labels

        md["hard_query_protocol"] = {
            "hard_query_types": hard_types,
            "hard_query_tags": hard_tags,
            "selected_reasons": [str(x) for x in row.get("selected_reasons", []) if str(x)],
            "selection_score": float(row.get("selection_score", 0.0)),
            "target_label_id": int(target_id) if target_id is not None else None,
            "secondary_label_id": int(row.get("secondary_label_id")) if row.get("secondary_label_id") is not None else None,
            "source_eventful_report": str(args.eventful_report),
        }
        src["metadata"] = md
        out_manifest_data.append(src)

        selected_summary.append(
            {
                "clip_id": clip_id,
                "hard_query_types": hard_types,
                "hard_query_tags": hard_tags,
                "selected_reasons": [str(x) for x in row.get("selected_reasons", []) if str(x)],
                "selection_score": float(row.get("selection_score", 0.0)),
                "target_label_id": int(target_id) if target_id is not None else None,
                "secondary_label_id": int(row.get("secondary_label_id")) if row.get("secondary_label_id") is not None else None,
            }
        )

    selected_count = len(out_manifest_data)
    ratios = {k: (float(v) / max(1, selected_count)) for k, v in type_counts.items()}

    coverage_insufficient = bool(
        selected_count < max(1, int(args.target_size) // 2)
        or type_counts.get("same_category_distractor", 0) == 0
        or type_counts.get("spatial_disambiguation", 0) == 0
        or type_counts.get("reappearing_object_query", 0) == 0
    )

    report = {
        "source_manifest": str(source_manifest),
        "eventful_report": str(args.eventful_report),
        "target_size": int(args.target_size),
        "selected_count": selected_count,
        "coverage_insufficient": coverage_insufficient,
        "hard_query_type_counts": type_counts,
        "hard_query_type_ratios": ratios,
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
                "hard_query_type_counts": type_counts,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
