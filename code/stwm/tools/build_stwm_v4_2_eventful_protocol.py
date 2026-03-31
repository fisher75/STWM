from __future__ import annotations

from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path
import json
from typing import Any

import numpy as np
from PIL import Image


EVENT_TYPES = [
    "occlusion",
    "reappearance",
    "reconnect",
    "visibility_flip",
    "identity_ambiguity",
]


@dataclass
class LabelEventStats:
    label_id: int
    score: float
    disappear_events: int
    reappearance_events: int
    reconnect_events: int
    visibility_flip_count: int
    longest_missing_span: int
    mean_area: float
    area_cv: float
    mean_motion: float
    centroid_mean_x: float
    centroid_mean_y: float


@dataclass
class ClipCandidate:
    clip_id: str
    dataset: str
    target_label_id: int | None
    secondary_label_id: int | None
    selection_score: float
    reasons: list[str]
    label_stats: LabelEventStats | None
    identity_ambiguity_score: float
    avg_label_count_per_frame: float
    diagnostics: dict[str, float]


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Build STWM V4.2 eventful protocol manifest")
    parser.add_argument("--source-manifest", default="/home/chen034/workspace/stwm/manifests/minisplits/stwm_week2_minival_v2.json")
    parser.add_argument(
        "--runs-root",
        default="/home/chen034/workspace/stwm/outputs/training/stwm_v4_2_minival_multiseed",
    )
    parser.add_argument("--diag-seeds", default="42,123,456")
    parser.add_argument("--diag-run", default="full_v4_2")
    parser.add_argument("--max-frames", type=int, default=64)
    parser.add_argument("--candidate-labels", type=int, default=12)
    parser.add_argument("--min-area-ratio", type=float, default=5e-4)
    parser.add_argument("--min-disappear-frames", type=int, default=1)
    parser.add_argument("--reconnect-distance-threshold", type=float, default=0.20)
    parser.add_argument("--identity-ambiguity-ratio", type=float, default=0.35)
    parser.add_argument("--identity-ambiguity-min-ratio", type=float, default=0.20)
    parser.add_argument("--target-size", type=int, default=18)
    parser.add_argument("--min-selected", type=int, default=10)
    parser.add_argument("--min-reappearance", type=int, default=3)
    parser.add_argument("--min-reconnect", type=int, default=2)
    parser.add_argument(
        "--output-manifest",
        default="/home/chen034/workspace/stwm/manifests/minisplits/stwm_v4_2_eventful_minival_v1.json",
    )
    parser.add_argument(
        "--output-clip-ids",
        default="/home/chen034/workspace/stwm/manifests/minisplits/stwm_v4_2_eventful_clip_ids_v1.json",
    )
    parser.add_argument(
        "--output-report",
        default="/home/chen034/workspace/stwm/reports/stwm_v4_2_eventful_protocol_v1.json",
    )
    return parser


def _read_mask(path: str) -> np.ndarray:
    arr = np.asarray(Image.open(path))
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr.astype(np.int32)


def _valid_label_ids(arr: np.ndarray, dataset: str) -> np.ndarray:
    ids = np.unique(arr)
    ids = ids[ids > 0]
    if dataset == "vspw":
        ids = ids[ids != 255]
    return ids


def _clip_frame_info(mask_paths: list[str], dataset: str, max_frames: int) -> tuple[list[np.ndarray], list[dict[int, float]], dict[int, float], dict[int, int]]:
    masks: list[np.ndarray] = []
    area_maps: list[dict[int, float]] = []
    area_sum: dict[int, float] = {}
    presence_count: dict[int, int] = {}

    for path in mask_paths[: max(1, int(max_frames))]:
        if not Path(path).exists():
            continue
        arr = _read_mask(path)
        masks.append(arr)
        total = float(arr.shape[0] * arr.shape[1])
        ids, counts = np.unique(arr, return_counts=True)

        area_map: dict[int, float] = {}
        for raw_id, raw_count in zip(ids.tolist(), counts.tolist()):
            lid = int(raw_id)
            if lid <= 0:
                continue
            if dataset == "vspw" and lid == 255:
                continue
            ratio = float(raw_count) / max(1.0, total)
            area_map[lid] = ratio
            area_sum[lid] = float(area_sum.get(lid, 0.0) + ratio)
            presence_count[lid] = int(presence_count.get(lid, 0) + 1)
        area_maps.append(area_map)

    return masks, area_maps, area_sum, presence_count


def _reappearance_indices(vis: list[int], min_gap: int) -> list[int]:
    out: list[int] = []
    min_gap = max(1, int(min_gap))
    for i in range(1, len(vis)):
        if vis[i] != 1:
            continue
        j = i - 1
        gap = 0
        while j >= 0 and vis[j] == 0:
            gap += 1
            j -= 1
        had_visible_before = j >= 0 and vis[j] == 1
        if had_visible_before and gap >= min_gap:
            out.append(i)
    return out


def _centroid(arr: np.ndarray, label_id: int) -> tuple[float, float] | None:
    fg = arr == int(label_id)
    if not np.any(fg):
        return None
    ys, xs = np.nonzero(fg)
    h, w = arr.shape[:2]
    cx = float(xs.mean() / max(1, w - 1))
    cy = float(ys.mean() / max(1, h - 1))
    return (cx, cy)


def _label_stats(
    masks: list[np.ndarray],
    area_maps: list[dict[int, float]],
    label_id: int,
    min_area_ratio: float,
    min_disappear_frames: int,
    reconnect_distance_threshold: float,
) -> LabelEventStats:
    vis: list[int] = []
    areas: list[float] = []
    centroids: list[tuple[float, float] | None] = []

    for i, arr in enumerate(masks):
        area = float(area_maps[i].get(int(label_id), 0.0))
        visible = 1 if area >= float(min_area_ratio) else 0
        vis.append(visible)
        areas.append(area)
        centroids.append(_centroid(arr, int(label_id)) if visible else None)

    transitions = 0
    disappear_events = 0
    for i in range(1, len(vis)):
        if vis[i] != vis[i - 1]:
            transitions += 1
        if vis[i - 1] == 1 and vis[i] == 0:
            disappear_events += 1

    reappear_idx = _reappearance_indices(vis, min_gap=int(min_disappear_frames))
    reappearance_events = int(len(reappear_idx))

    reconnect_events = 0
    reconnect_distances: list[float] = []
    for idx in reappear_idx:
        j = idx - 1
        while j >= 0 and vis[j] == 0:
            j -= 1
        if j < 0:
            continue
        c_prev = centroids[j]
        c_now = centroids[idx]
        if c_prev is None or c_now is None:
            continue
        dx = float(c_now[0] - c_prev[0])
        dy = float(c_now[1] - c_prev[1])
        dist = float((dx * dx + dy * dy) ** 0.5)
        reconnect_distances.append(dist)
        if dist <= float(reconnect_distance_threshold):
            reconnect_events += 1

    longest_missing = 0
    cur_missing = 0
    for flag in vis:
        if flag == 0:
            cur_missing += 1
            longest_missing = max(longest_missing, cur_missing)
        else:
            cur_missing = 0

    motions: list[float] = []
    for i in range(1, len(centroids)):
        a = centroids[i - 1]
        b = centroids[i]
        if a is None or b is None:
            continue
        dx = float(b[0] - a[0])
        dy = float(b[1] - a[1])
        motions.append(float((dx * dx + dy * dy) ** 0.5))

    visible_centroids = [c for c in centroids if c is not None]
    if visible_centroids:
        cx = float(np.mean([x[0] for x in visible_centroids]))
        cy = float(np.mean([x[1] for x in visible_centroids]))
    else:
        cx, cy = 0.5, 0.5

    mean_area = float(np.mean(areas)) if areas else 0.0
    area_cv = float(np.std(areas) / (mean_area + 1e-8)) if areas else 0.0
    mean_motion = float(np.mean(motions)) if motions else 0.0

    score = (
        5.0 * float(reappearance_events)
        + 3.0 * float(disappear_events)
        + 2.0 * float(reconnect_events)
        + 1.5 * float(transitions)
        + 0.5 * float(longest_missing)
        + 1.0 * float(area_cv)
        + 1.0 * float(mean_motion)
    )

    return LabelEventStats(
        label_id=int(label_id),
        score=float(score),
        disappear_events=int(disappear_events),
        reappearance_events=int(reappearance_events),
        reconnect_events=int(reconnect_events),
        visibility_flip_count=int(transitions),
        longest_missing_span=int(longest_missing),
        mean_area=float(mean_area),
        area_cv=float(area_cv),
        mean_motion=float(mean_motion),
        centroid_mean_x=float(cx),
        centroid_mean_y=float(cy),
    )


def _load_clip_diagnostics(runs_root: Path, seeds: list[str], run_name: str) -> dict[str, dict[str, float]]:
    clip_values: dict[str, dict[str, list[float]]] = {}

    for seed in seeds:
        log_path = runs_root / f"seed_{seed}" / run_name / "train_log.jsonl"
        if not log_path.exists():
            continue
        for line in log_path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            clip_id = str(row.get("clip_id", ""))
            if not clip_id:
                continue
            slot = clip_values.setdefault(
                clip_id,
                {
                    "trajectory_l1": [],
                    "query_localization_error": [],
                    "query_traj_gap": [],
                    "semantic_loss": [],
                    "reid_loss": [],
                },
            )
            for k in list(slot.keys()):
                slot[k].append(float(row.get(k, 0.0)))

    out: dict[str, dict[str, float]] = {}
    for clip_id, vals in clip_values.items():
        out[clip_id] = {
            k: float(np.mean(v)) if v else 0.0
            for k, v in vals.items()
        }
    return out


def _identity_ambiguity_score(
    area_maps: list[dict[int, float]],
    target_label_id: int,
    ratio_threshold: float,
) -> tuple[float, float]:
    if not area_maps:
        return 0.0, 0.0
    ambiguous = 0
    avg_label_count = float(np.mean([len(m) for m in area_maps]))

    for amap in area_maps:
        target_area = float(amap.get(int(target_label_id), 0.0))
        if target_area <= 0.0:
            continue
        other_areas = sorted([float(v) for k, v in amap.items() if int(k) != int(target_label_id)], reverse=True)
        if not other_areas:
            continue
        if other_areas[0] / max(target_area, 1e-8) >= float(ratio_threshold):
            ambiguous += 1

    return float(ambiguous / max(1, len(area_maps))), float(avg_label_count)


def _clip_candidate(
    item: dict[str, Any],
    diagnostics: dict[str, dict[str, float]],
    max_frames: int,
    candidate_labels: int,
    min_area_ratio: float,
    min_disappear_frames: int,
    reconnect_distance_threshold: float,
    identity_ambiguity_ratio: float,
    identity_ambiguity_min_ratio: float,
) -> ClipCandidate | None:
    clip_id = str(item.get("clip_id", ""))
    md = dict(item.get("metadata", {}))
    dataset = str(md.get("dataset", "unknown")).lower()
    mask_paths = md.get("mask_paths") if isinstance(md.get("mask_paths"), list) else []
    if not mask_paths:
        return None

    masks, area_maps, area_sum, presence_count = _clip_frame_info(mask_paths, dataset, max_frames=max_frames)
    if not masks:
        return None

    existing_target = md.get("target_label_id")
    try:
        existing_target = int(existing_target) if existing_target is not None else None
    except (TypeError, ValueError):
        existing_target = None

    ranked = sorted(area_sum.keys(), key=lambda lid: (presence_count.get(lid, 0), area_sum.get(lid, 0.0)), reverse=True)
    label_ids = ranked[: max(1, int(candidate_labels))]
    if existing_target is not None and existing_target in area_sum and existing_target not in label_ids:
        label_ids = [existing_target] + label_ids

    all_stats: list[LabelEventStats] = []
    for lid in label_ids:
        all_stats.append(
            _label_stats(
                masks,
                area_maps,
                label_id=int(lid),
                min_area_ratio=float(min_area_ratio),
                min_disappear_frames=int(min_disappear_frames),
                reconnect_distance_threshold=float(reconnect_distance_threshold),
            )
        )

    if not all_stats:
        return None

    best = max(all_stats, key=lambda s: s.score)
    secondary = None
    alt = [x for x in all_stats if int(x.label_id) != int(best.label_id)]
    if alt:
        secondary = max(alt, key=lambda s: s.score)

    ambiguity_score, avg_label_count = _identity_ambiguity_score(
        area_maps,
        target_label_id=int(best.label_id),
        ratio_threshold=float(identity_ambiguity_ratio),
    )

    reasons: list[str] = []
    if best.disappear_events > 0:
        reasons.append("occlusion")
    if best.reappearance_events > 0:
        reasons.append("reappearance")
    if best.reconnect_events > 0:
        reasons.append("reconnect")
    if best.visibility_flip_count >= 2:
        reasons.append("visibility_flip")
    if ambiguity_score >= float(identity_ambiguity_min_ratio):
        reasons.append("identity_ambiguity")

    diag = diagnostics.get(clip_id, {})
    diag_difficulty = float(diag.get("query_localization_error", 0.0)) + 0.5 * float(diag.get("trajectory_l1", 0.0))
    selection_score = float(best.score + 8.0 * diag_difficulty)

    return ClipCandidate(
        clip_id=clip_id,
        dataset=dataset,
        target_label_id=int(best.label_id),
        secondary_label_id=int(secondary.label_id) if secondary is not None else None,
        selection_score=selection_score,
        reasons=reasons,
        label_stats=best,
        identity_ambiguity_score=float(ambiguity_score),
        avg_label_count_per_frame=float(avg_label_count),
        diagnostics={
            "trajectory_l1": float(diag.get("trajectory_l1", 0.0)),
            "query_localization_error": float(diag.get("query_localization_error", 0.0)),
            "query_traj_gap": float(diag.get("query_traj_gap", 0.0)),
            "semantic_loss": float(diag.get("semantic_loss", 0.0)),
            "reid_loss": float(diag.get("reid_loss", 0.0)),
        },
    )


def _greedy_select(candidates: list[ClipCandidate], target_size: int) -> list[ClipCandidate]:
    ranked = sorted(candidates, key=lambda x: (len(x.reasons), x.selection_score), reverse=True)
    out: list[ClipCandidate] = []
    used: set[str] = set()

    for event_type in EVENT_TYPES:
        options = [c for c in ranked if event_type in c.reasons and c.clip_id not in used]
        if not options:
            continue
        pick = options[0]
        out.append(pick)
        used.add(pick.clip_id)

    for c in ranked:
        if len(out) >= max(1, int(target_size)):
            break
        if c.clip_id in used:
            continue
        out.append(c)
        used.add(c.clip_id)

    return out


def main() -> None:
    args = build_parser().parse_args()

    source_manifest = Path(args.source_manifest)
    items = json.loads(source_manifest.read_text())

    seeds = [s.strip() for s in str(args.diag_seeds).split(",") if s.strip()]
    diagnostics = _load_clip_diagnostics(Path(args.runs_root), seeds=seeds, run_name=str(args.diag_run))

    candidates: list[ClipCandidate] = []
    for item in items:
        cand = _clip_candidate(
            item=item,
            diagnostics=diagnostics,
            max_frames=int(args.max_frames),
            candidate_labels=int(args.candidate_labels),
            min_area_ratio=float(args.min_area_ratio),
            min_disappear_frames=int(args.min_disappear_frames),
            reconnect_distance_threshold=float(args.reconnect_distance_threshold),
            identity_ambiguity_ratio=float(args.identity_ambiguity_ratio),
            identity_ambiguity_min_ratio=float(args.identity_ambiguity_min_ratio),
        )
        if cand is None:
            continue
        if not cand.reasons:
            continue
        candidates.append(cand)

    selected = _greedy_select(candidates, target_size=int(args.target_size))
    selected_ids = [x.clip_id for x in selected]

    item_map = {str(item.get("clip_id", "")): item for item in items}
    out_manifest_data: list[dict[str, Any]] = []

    for c in selected:
        src = dict(item_map[c.clip_id])
        md = dict(src.get("metadata", {}))
        if c.target_label_id is not None:
            md["target_label_id"] = int(c.target_label_id)
        md["eventful_protocol"] = {
            "selected_reasons": list(c.reasons),
            "selection_score": float(c.selection_score),
            "target_label_id": int(c.target_label_id) if c.target_label_id is not None else None,
            "secondary_label_id": int(c.secondary_label_id) if c.secondary_label_id is not None else None,
            "identity_ambiguity_score": float(c.identity_ambiguity_score),
            "avg_label_count_per_frame": float(c.avg_label_count_per_frame),
            "label_stats": {
                "disappear_events": int(c.label_stats.disappear_events) if c.label_stats else 0,
                "reappearance_events": int(c.label_stats.reappearance_events) if c.label_stats else 0,
                "reconnect_events": int(c.label_stats.reconnect_events) if c.label_stats else 0,
                "visibility_flip_count": int(c.label_stats.visibility_flip_count) if c.label_stats else 0,
                "longest_missing_span": int(c.label_stats.longest_missing_span) if c.label_stats else 0,
                "mean_area": float(c.label_stats.mean_area) if c.label_stats else 0.0,
                "area_cv": float(c.label_stats.area_cv) if c.label_stats else 0.0,
                "mean_motion": float(c.label_stats.mean_motion) if c.label_stats else 0.0,
                "centroid_mean_x": float(c.label_stats.centroid_mean_x) if c.label_stats else 0.5,
                "centroid_mean_y": float(c.label_stats.centroid_mean_y) if c.label_stats else 0.5,
            },
            "diagnostics": dict(c.diagnostics),
            "source_manifest": str(source_manifest),
        }
        src["metadata"] = md
        out_manifest_data.append(src)

    event_counts = {k: 0 for k in EVENT_TYPES}
    for c in selected:
        for r in c.reasons:
            if r in event_counts:
                event_counts[r] += 1

    selected_count = len(selected)
    event_ratios = {
        k: (float(v) / max(1, selected_count))
        for k, v in event_counts.items()
    }

    coverage_insufficient = bool(
        selected_count < int(args.min_selected)
        or event_counts.get("reappearance", 0) < int(args.min_reappearance)
        or event_counts.get("reconnect", 0) < int(args.min_reconnect)
    )

    report = {
        "source_manifest": str(source_manifest),
        "runs_root": str(args.runs_root),
        "diag_run": str(args.diag_run),
        "diag_seeds": seeds,
        "candidate_count": len(candidates),
        "selected_count": selected_count,
        "target_size": int(args.target_size),
        "coverage_insufficient": coverage_insufficient,
        "coverage_thresholds": {
            "min_selected": int(args.min_selected),
            "min_reappearance": int(args.min_reappearance),
            "min_reconnect": int(args.min_reconnect),
        },
        "event_type_counts": event_counts,
        "event_type_ratios": event_ratios,
        "selected_clip_ids": selected_ids,
        "selected_clips": [
            {
                "clip_id": c.clip_id,
                "dataset": c.dataset,
                "selected_reasons": c.reasons,
                "selection_score": float(c.selection_score),
                "target_label_id": int(c.target_label_id) if c.target_label_id is not None else None,
                "secondary_label_id": int(c.secondary_label_id) if c.secondary_label_id is not None else None,
                "identity_ambiguity_score": float(c.identity_ambiguity_score),
                "avg_label_count_per_frame": float(c.avg_label_count_per_frame),
                "label_stats": {
                    "disappear_events": int(c.label_stats.disappear_events) if c.label_stats else 0,
                    "reappearance_events": int(c.label_stats.reappearance_events) if c.label_stats else 0,
                    "reconnect_events": int(c.label_stats.reconnect_events) if c.label_stats else 0,
                    "visibility_flip_count": int(c.label_stats.visibility_flip_count) if c.label_stats else 0,
                    "longest_missing_span": int(c.label_stats.longest_missing_span) if c.label_stats else 0,
                    "mean_area": float(c.label_stats.mean_area) if c.label_stats else 0.0,
                    "area_cv": float(c.label_stats.area_cv) if c.label_stats else 0.0,
                    "mean_motion": float(c.label_stats.mean_motion) if c.label_stats else 0.0,
                    "centroid_mean_x": float(c.label_stats.centroid_mean_x) if c.label_stats else 0.5,
                    "centroid_mean_y": float(c.label_stats.centroid_mean_y) if c.label_stats else 0.5,
                },
                "diagnostics": dict(c.diagnostics),
            }
            for c in selected
        ],
        "all_candidates_top30": [
            {
                "clip_id": c.clip_id,
                "dataset": c.dataset,
                "selected_reasons": c.reasons,
                "selection_score": float(c.selection_score),
                "target_label_id": int(c.target_label_id) if c.target_label_id is not None else None,
                "secondary_label_id": int(c.secondary_label_id) if c.secondary_label_id is not None else None,
            }
            for c in sorted(candidates, key=lambda x: (len(x.reasons), x.selection_score), reverse=True)[:30]
        ],
    }

    out_manifest = Path(args.output_manifest)
    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    out_manifest.write_text(json.dumps(out_manifest_data, indent=2))

    out_clip_ids = Path(args.output_clip_ids)
    out_clip_ids.parent.mkdir(parents=True, exist_ok=True)
    out_clip_ids.write_text(json.dumps(selected_ids, indent=2))

    out_report = Path(args.output_report)
    out_report.parent.mkdir(parents=True, exist_ok=True)
    out_report.write_text(json.dumps(report, indent=2))

    print(
        json.dumps(
            {
                "output_manifest": str(out_manifest),
                "output_clip_ids": str(out_clip_ids),
                "output_report": str(out_report),
                "candidate_count": len(candidates),
                "selected_count": selected_count,
                "coverage_insufficient": coverage_insufficient,
                "event_type_counts": event_counts,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
