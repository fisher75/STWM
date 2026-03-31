from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import json
from typing import Any

from PIL import Image, ImageDraw

from stwm.datasets.stwm_dataset import STWMDataset


INSTANCE_TYPES = {
    "same_category_distractor",
    "spatial_disambiguation",
    "relation_conditioned_query",
}

FUTURE_TYPE = "future_conditioned_reappearance_aware"


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Build STWM V4.2 state-identifiability figure casebook")
    parser.add_argument("--runs-root", default="/home/chen034/workspace/stwm/outputs/training/stwm_v4_2_state_identifiability")
    parser.add_argument("--seeds", default="42,123")
    parser.add_argument("--manifest", default="/home/chen034/workspace/stwm/manifests/minisplits/stwm_v4_2_state_identifiability_v1.json")
    parser.add_argument("--data-root", default="/home/chen034/workspace/stwm/data/external")
    parser.add_argument(
        "--output-dir",
        default="/home/chen034/workspace/stwm/outputs/visualizations/stwm_v4_2_state_identifiability_figures",
    )
    parser.add_argument("--cases-per-group", type=int, default=8)
    parser.add_argument("--min-consistent-seeds", type=int, default=2)
    return parser


def _read_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _last_by_clip(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        clip_id = str(row.get("clip_id", ""))
        if not clip_id:
            continue
        prev = out.get(clip_id)
        if prev is None or int(row.get("step", 0)) >= int(prev.get("step", 0)):
            out[clip_id] = row
    return out


def _draw_point(draw: ImageDraw.ImageDraw, x: float, y: float, color: tuple[int, int, int], radius: int = 5) -> None:
    draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)


def _render_panel(
    clip_id: str,
    seed: str,
    frame_path: Path,
    full_row: dict[str, Any],
    other_row: dict[str, Any],
    other_name: str,
) -> Image.Image:
    frame = Image.open(frame_path).convert("RGB")
    left = frame.copy()
    right = frame.copy()
    dl = ImageDraw.Draw(left)
    dr = ImageDraw.Draw(right)

    for drawer in [dl, dr]:
        drawer.rectangle((0, 0, frame.width, 60), fill=(0, 0, 0))

    gt_x = float(full_row.get("query_gt_x", 0.5)) * max(1, frame.width - 1)
    gt_y = float(full_row.get("query_gt_y", 0.5)) * max(1, frame.height - 1)
    _draw_point(dl, gt_x, gt_y, (30, 200, 90), radius=6)
    _draw_point(dr, gt_x, gt_y, (30, 200, 90), radius=6)

    full_px = float(full_row.get("query_pred_x", 0.5)) * max(1, frame.width - 1)
    full_py = float(full_row.get("query_pred_y", 0.5)) * max(1, frame.height - 1)
    other_px = float(other_row.get("query_pred_x", 0.5)) * max(1, frame.width - 1)
    other_py = float(other_row.get("query_pred_y", 0.5)) * max(1, frame.height - 1)
    _draw_point(dl, full_px, full_py, (60, 120, 255), radius=6)
    _draw_point(dr, other_px, other_py, (240, 120, 40), radius=6)

    dl.text((8, 8), f"seed={seed} | full_v4_2 | {clip_id}", fill=(255, 255, 255))
    dl.text(
        (8, 30),
        "traj={:.4f} q={:.4f}".format(
            float(full_row.get("trajectory_l1", 0.0)),
            float(full_row.get("query_localization_error", 0.0)),
        ),
        fill=(255, 255, 255),
    )

    dr.text((8, 8), f"seed={seed} | {other_name}", fill=(255, 255, 255))
    dr.text(
        (8, 30),
        "traj={:.4f} q={:.4f}".format(
            float(other_row.get("trajectory_l1", 0.0)),
            float(other_row.get("query_localization_error", 0.0)),
        ),
        fill=(255, 255, 255),
    )

    canvas = Image.new("RGB", (left.width + right.width, max(left.height, right.height)), color=(255, 255, 255))
    canvas.paste(left, (0, 0))
    canvas.paste(right, (left.width, 0))
    return canvas


def _collect_seed_maps(runs_root: Path, seeds: list[str]) -> dict[str, dict[str, dict[str, dict[str, Any]]]]:
    out: dict[str, dict[str, dict[str, dict[str, Any]]]] = {}
    for seed in seeds:
        out[seed] = {}
        for run in ["full_v4_2", "wo_semantics_v4_2", "wo_object_bias_v4_2"]:
            rows = _read_rows(runs_root / f"seed_{seed}" / run / "train_log.jsonl")
            out[seed][run] = _last_by_clip(rows)
    return out


def _load_query_types(manifest: Path) -> dict[str, list[str]]:
    data = json.loads(manifest.read_text())
    out: dict[str, list[str]] = {}
    for item in data:
        clip_id = str(item.get("clip_id", ""))
        if not clip_id:
            continue
        md = item.get("metadata", {}) if isinstance(item.get("metadata"), dict) else {}
        proto = md.get("state_identifiability_protocol", {}) if isinstance(md.get("state_identifiability_protocol"), dict) else {}
        qtypes = [str(x) for x in proto.get("query_types", []) if str(x)]
        out[clip_id] = qtypes
    return out


def _select_cases(candidates: list[dict[str, Any]], cases_per_group: int, min_consistent_seeds: int) -> list[dict[str, Any]]:
    if not candidates:
        return []

    def rank_key(x: dict[str, Any]) -> tuple[int, int, float]:
        return (
            int(x.get("consistency_count", 0)),
            int(x.get("event_seed_count", 0)),
            float(x.get("avg_delta", 0.0)),
        )

    strict = [
        c
        for c in candidates
        if int(c.get("consistency_count", 0)) >= int(min_consistent_seeds) and float(c.get("avg_delta", 0.0)) > 0.0
    ]
    strict = sorted(strict, key=rank_key, reverse=True)

    selected: list[dict[str, Any]] = strict[:cases_per_group]
    if len(selected) >= cases_per_group:
        return selected

    selected_ids = {str(x.get("clip_id", "")) for x in selected}
    relaxed = [
        c
        for c in candidates
        if str(c.get("clip_id", "")) not in selected_ids
        and int(c.get("consistency_count", 0)) >= 1
        and float(c.get("avg_delta", 0.0)) > 0.0
    ]
    relaxed = sorted(relaxed, key=rank_key, reverse=True)
    for c in relaxed:
        if len(selected) >= cases_per_group:
            break
        c = dict(c)
        c["relaxed_pick"] = True
        selected.append(c)
    return selected


def main() -> None:
    args = build_parser().parse_args()

    runs_root = Path(args.runs_root)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    seeds = [x.strip() for x in str(args.seeds).split(",") if x.strip()]
    cases_per_group = max(1, int(args.cases_per_group))
    min_consistent = max(1, int(args.min_consistent_seeds))

    dataset = STWMDataset(args.data_root, manifest=args.manifest, limit=None)
    sample_map = {s.clip_id: s for s in dataset.samples}
    query_type_map = _load_query_types(Path(args.manifest))
    seed_maps = _collect_seed_maps(runs_root, seeds)

    clip_ids = sorted(
        set(sample_map.keys()).intersection(
            {
                cid
                for seed in seeds
                for cid in seed_maps.get(seed, {}).get("full_v4_2", {}).keys()
            }
        )
    )

    semantic_candidates: list[dict[str, Any]] = []
    instance_candidates: list[dict[str, Any]] = []
    future_candidates: list[dict[str, Any]] = []

    for clip_id in clip_ids:
        qtypes = set(query_type_map.get(clip_id, []))
        has_instance_type = bool(qtypes.intersection(INSTANCE_TYPES))
        has_future_type = FUTURE_TYPE in qtypes

        sem_per_seed: dict[str, dict[str, Any]] = {}
        inst_per_seed: dict[str, dict[str, Any]] = {}
        fut_per_seed: dict[str, dict[str, Any]] = {}

        for seed in seeds:
            full_row = seed_maps.get(seed, {}).get("full_v4_2", {}).get(clip_id)
            ws_row = seed_maps.get(seed, {}).get("wo_semantics_v4_2", {}).get(clip_id)
            wob_row = seed_maps.get(seed, {}).get("wo_object_bias_v4_2", {}).get(clip_id)

            if full_row is not None and ws_row is not None:
                d_sem = (
                    float(ws_row.get("trajectory_l1", 0.0))
                    + float(ws_row.get("query_localization_error", 0.0))
                    - float(full_row.get("trajectory_l1", 0.0))
                    - float(full_row.get("query_localization_error", 0.0))
                )
                sem_per_seed[seed] = {
                    "delta": d_sem,
                    "comparator": "wo_semantics_v4_2",
                }

            if has_instance_type and full_row is not None and wob_row is not None:
                d_inst = (
                    float(wob_row.get("query_localization_error", 0.0))
                    + 0.5 * float(wob_row.get("trajectory_l1", 0.0))
                    - float(full_row.get("query_localization_error", 0.0))
                    - 0.5 * float(full_row.get("trajectory_l1", 0.0))
                )
                inst_per_seed[seed] = {
                    "delta": d_inst,
                    "comparator": "wo_object_bias_v4_2",
                }

            if has_future_type and full_row is not None and wob_row is not None:
                d_fut = (
                    float(wob_row.get("query_localization_error", 0.0))
                    + 0.5 * float(wob_row.get("trajectory_l1", 0.0))
                    + 0.25 * float(wob_row.get("reconnect_min_error", 0.0))
                    - float(full_row.get("query_localization_error", 0.0))
                    - 0.5 * float(full_row.get("trajectory_l1", 0.0))
                    - 0.25 * float(full_row.get("reconnect_min_error", 0.0))
                )
                has_event = bool(
                    float(full_row.get("has_reappearance_event", 0.0)) > 0.5
                    or float(wob_row.get("has_reappearance_event", 0.0)) > 0.5
                )
                fut_per_seed[seed] = {
                    "delta": d_fut,
                    "comparator": "wo_object_bias_v4_2",
                    "has_event": has_event,
                }

        def to_candidate(kind: str, per_seed_map: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
            if not per_seed_map:
                return None
            deltas = [float(v.get("delta", 0.0)) for v in per_seed_map.values()]
            consistency = int(sum(1 for x in deltas if x > 0.0))
            avg_delta = float(sum(deltas) / max(1, len(deltas)))
            event_seed_count = int(sum(1 for v in per_seed_map.values() if bool(v.get("has_event", False))))
            best_seed = max(per_seed_map.items(), key=lambda kv: float(kv[1].get("delta", 0.0)))[0]
            best_comp = str(per_seed_map[best_seed].get("comparator", "wo_semantics_v4_2"))
            return {
                "clip_id": clip_id,
                "kind": kind,
                "query_types": sorted(list(qtypes)),
                "available_seed_count": int(len(per_seed_map)),
                "consistency_count": int(consistency),
                "avg_delta": float(avg_delta),
                "event_seed_count": int(event_seed_count),
                "best_seed": str(best_seed),
                "best_comparator": best_comp,
                "per_seed": per_seed_map,
            }

        sem_c = to_candidate("semantic_sensitive", sem_per_seed)
        if sem_c is not None:
            semantic_candidates.append(sem_c)

        inst_c = to_candidate("instance_disambiguation", inst_per_seed)
        if inst_c is not None:
            instance_candidates.append(inst_c)

        fut_c = to_candidate("future_grounding", fut_per_seed)
        if fut_c is not None:
            future_candidates.append(fut_c)

    selected = {
        "semantic_sensitive": _select_cases(semantic_candidates, cases_per_group, min_consistent),
        "instance_disambiguation": _select_cases(instance_candidates, cases_per_group, min_consistent),
        "future_grounding": _select_cases(future_candidates, cases_per_group, min_consistent),
    }

    artifacts: dict[str, list[str]] = {k: [] for k in selected}
    case_summaries: dict[str, dict[str, Any]] = {k: {} for k in selected}

    for group, cases in selected.items():
        group_dir = out_dir / f"{group}_cases"
        group_dir.mkdir(parents=True, exist_ok=True)

        for case in cases:
            clip_id = str(case.get("clip_id", ""))
            seed = str(case.get("best_seed", ""))
            comparator = str(case.get("best_comparator", "wo_semantics_v4_2"))
            sample = sample_map.get(clip_id)
            if sample is None or not sample.frame_paths:
                continue

            full_row = seed_maps.get(seed, {}).get("full_v4_2", {}).get(clip_id)
            other_row = seed_maps.get(seed, {}).get(comparator, {}).get(clip_id)
            if full_row is None or other_row is None:
                continue

            frame_idx = int(full_row.get("query_frame_idx", 0))
            frame_idx = max(0, min(frame_idx, len(sample.frame_paths) - 1))
            frame_path = Path(sample.frame_paths[frame_idx])
            if not frame_path.exists():
                continue

            panel = _render_panel(
                clip_id=clip_id,
                seed=seed,
                frame_path=frame_path,
                full_row=full_row,
                other_row=other_row,
                other_name=comparator,
            )
            out_path = group_dir / f"{clip_id}_seed{seed}.png"
            panel.save(out_path)

            artifacts[group].append(str(out_path))
            case_summaries[group][clip_id] = {
                "representative_seed": seed,
                "representative_comparator": comparator,
                "consistency_count": int(case.get("consistency_count", 0)),
                "available_seed_count": int(case.get("available_seed_count", 0)),
                "event_seed_count": int(case.get("event_seed_count", 0)),
                "avg_delta": float(case.get("avg_delta", 0.0)),
                "query_types": case.get("query_types", []),
                "relaxed_pick": bool(case.get("relaxed_pick", False)),
                "per_seed": case.get("per_seed", {}),
                "artifact": str(out_path),
            }

    figure_manifest = {
        "runs_root": str(runs_root),
        "manifest": str(args.manifest),
        "seeds": seeds,
        "output_dir": str(out_dir),
        "cases_per_group": cases_per_group,
        "min_consistent_seeds": min_consistent,
        "selected_case_ids": {
            group: [str(x.get("clip_id", "")) for x in cases]
            for group, cases in selected.items()
        },
        "selected_case_count": {
            group: len(cases) for group, cases in selected.items()
        },
        "case_summaries": case_summaries,
        "artifacts": artifacts,
    }

    (out_dir / "figure_manifest.json").write_text(json.dumps(figure_manifest, indent=2))

    print(
        json.dumps(
            {
                "output_dir": str(out_dir),
                "figure_manifest": str(out_dir / "figure_manifest.json"),
                "selected_case_count": figure_manifest["selected_case_count"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
