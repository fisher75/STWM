from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import json
from typing import Any

from PIL import Image, ImageDraw

from stwm.datasets.stwm_dataset import STWMDataset


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Build STWM V4.2 multi-seed shared qualitative casebook")
    parser.add_argument("--runs-root", default="/home/chen034/workspace/stwm/outputs/training/stwm_v4_2_minival_multiseed")
    parser.add_argument("--seeds", default="42,123,456")
    parser.add_argument("--manifest", default="/home/chen034/workspace/stwm/manifests/minisplits/stwm_week2_minival_v2.json")
    parser.add_argument("--data-root", default="/home/chen034/workspace/stwm/data/external")
    parser.add_argument("--output-dir", default="/home/chen034/workspace/stwm/outputs/visualizations/stwm_v4_2_multiseed_casebook")
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
    for r in rows:
        cid = str(r.get("clip_id", ""))
        if not cid:
            continue
        prev = out.get(cid)
        if prev is None or int(r.get("step", 0)) >= int(prev.get("step", 0)):
            out[cid] = r
    return out


def _draw_point(draw: ImageDraw.ImageDraw, x: float, y: float, color: tuple[int, int, int], r: int = 5) -> None:
    draw.ellipse((x - r, y - r, x + r, y + r), fill=color)


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

    for d in [dl, dr]:
        d.rectangle((0, 0, frame.width, 60), fill=(0, 0, 0))

    fgx = float(full_row.get("query_gt_x", 0.5)) * max(1, frame.width - 1)
    fgy = float(full_row.get("query_gt_y", 0.5)) * max(1, frame.height - 1)
    _draw_point(dl, fgx, fgy, (30, 200, 90), 6)
    _draw_point(dr, fgx, fgy, (30, 200, 90), 6)

    fpx = float(full_row.get("query_pred_x", 0.5)) * max(1, frame.width - 1)
    fpy = float(full_row.get("query_pred_y", 0.5)) * max(1, frame.height - 1)
    opx = float(other_row.get("query_pred_x", 0.5)) * max(1, frame.width - 1)
    opy = float(other_row.get("query_pred_y", 0.5)) * max(1, frame.height - 1)
    _draw_point(dl, fpx, fpy, (60, 120, 255), 6)
    _draw_point(dr, opx, opy, (240, 120, 40), 6)

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


def _collect_seed_maps(
    runs_root: Path,
    seeds: list[str],
) -> dict[str, dict[str, dict[str, dict[str, Any]]]]:
    out: dict[str, dict[str, dict[str, dict[str, Any]]]] = {}
    for seed in seeds:
        out[seed] = {}
        for run in ["full_v4_2", "wo_semantics_v4_2", "wo_identity_v4_2"]:
            rows = _read_rows(runs_root / f"seed_{seed}" / run / "train_log.jsonl")
            out[seed][run] = _last_by_clip(rows)
    return out


def main() -> None:
    args = build_parser().parse_args()

    runs_root = Path(args.runs_root)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    seeds = [s.strip() for s in str(args.seeds).split(",") if s.strip()]
    cases_per_group = max(1, int(args.cases_per_group))
    min_consistent = max(1, int(args.min_consistent_seeds))

    dataset = STWMDataset(args.data_root, manifest=args.manifest, limit=None)
    sample_map = {s.clip_id: s for s in dataset.samples}

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

    sem_candidates: list[dict[str, Any]] = []
    id_candidates: list[dict[str, Any]] = []
    qry_candidates: list[dict[str, Any]] = []

    for cid in clip_ids:
        sem_per_seed: dict[str, dict[str, Any]] = {}
        id_per_seed: dict[str, dict[str, Any]] = {}
        qry_per_seed: dict[str, dict[str, Any]] = {}

        for seed in seeds:
            full_row = seed_maps.get(seed, {}).get("full_v4_2", {}).get(cid)
            ws_row = seed_maps.get(seed, {}).get("wo_semantics_v4_2", {}).get(cid)
            wi_row = seed_maps.get(seed, {}).get("wo_identity_v4_2", {}).get(cid)

            if full_row is not None and ws_row is not None:
                sem_delta = (
                    float(ws_row.get("trajectory_l1", 0.0))
                    + float(ws_row.get("query_localization_error", 0.0))
                    - float(full_row.get("trajectory_l1", 0.0))
                    - float(full_row.get("query_localization_error", 0.0))
                )
                sem_per_seed[seed] = {"delta": sem_delta}

            if full_row is not None and wi_row is not None:
                wi_delta = (
                    float(wi_row.get("reconnect_min_error", 0.0))
                    - float(full_row.get("reconnect_min_error", 0.0))
                    + 0.6
                    * (
                        float(full_row.get("reconnect_success", 0.0))
                        - float(wi_row.get("reconnect_success", 0.0))
                    )
                    + 0.3
                    * (
                        float(wi_row.get("trajectory_l1", 0.0))
                        + float(wi_row.get("query_localization_error", 0.0))
                        - float(full_row.get("trajectory_l1", 0.0))
                        - float(full_row.get("query_localization_error", 0.0))
                    )
                )
                has_event = bool(
                    float(full_row.get("has_reappearance_event", 0.0)) > 0.5
                    or float(wi_row.get("has_reappearance_event", 0.0)) > 0.5
                )
                id_per_seed[seed] = {"delta": wi_delta, "has_event": has_event}

            if full_row is not None and (ws_row is not None or wi_row is not None):
                ws_delta = (
                    float(ws_row.get("query_localization_error", 0.0))
                    - float(full_row.get("query_localization_error", 0.0))
                    if ws_row is not None
                    else float("-inf")
                )
                wi_delta = (
                    float(wi_row.get("query_localization_error", 0.0))
                    - float(full_row.get("query_localization_error", 0.0))
                    if wi_row is not None
                    else float("-inf")
                )
                if ws_delta >= wi_delta:
                    qry_per_seed[seed] = {"delta": ws_delta, "worse_ablation": "wo_semantics_v4_2"}
                else:
                    qry_per_seed[seed] = {"delta": wi_delta, "worse_ablation": "wo_identity_v4_2"}

        def build_candidate(kind: str, per_seed_map: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
            if len(per_seed_map) < min_consistent:
                return None
            deltas = [float(v.get("delta", 0.0)) for v in per_seed_map.values()]
            consistency_count = int(sum(1 for x in deltas if x > 0.0))
            avg_delta = float(sum(deltas) / max(1, len(deltas)))
            if consistency_count < min_consistent or avg_delta <= 0.0:
                return None
            event_seed_count = int(sum(1 for v in per_seed_map.values() if bool(v.get("has_event", False))))
            best_seed = max(per_seed_map.items(), key=lambda kv: float(kv[1].get("delta", 0.0)))[0]
            return {
                "clip_id": cid,
                "kind": kind,
                "available_seed_count": int(len(per_seed_map)),
                "consistency_count": consistency_count,
                "avg_delta": avg_delta,
                "event_seed_count": event_seed_count,
                "best_seed": str(best_seed),
                "per_seed": per_seed_map,
            }

        sem_c = build_candidate("semantic_sensitive", sem_per_seed)
        if sem_c is not None:
            sem_candidates.append(sem_c)

        id_c = build_candidate("identity_reconnect", id_per_seed)
        if id_c is not None:
            id_candidates.append(id_c)

        qry_c = build_candidate("query_grounding", qry_per_seed)
        if qry_c is not None:
            qry_candidates.append(qry_c)

    def rank_key(x: dict[str, Any]) -> tuple[int, int, float]:
        return (
            int(x.get("consistency_count", 0)),
            int(x.get("event_seed_count", 0)),
            float(x.get("avg_delta", 0.0)),
        )

    sem_candidates.sort(key=rank_key, reverse=True)
    id_candidates.sort(key=rank_key, reverse=True)
    qry_candidates.sort(key=rank_key, reverse=True)

    selected = {
        "semantic_sensitive": sem_candidates[:cases_per_group],
        "identity_reconnect": id_candidates[:cases_per_group],
        "query_grounding": qry_candidates[:cases_per_group],
    }

    artifacts: dict[str, list[str]] = {k: [] for k in selected}
    case_summaries: dict[str, dict[str, Any]] = {k: {} for k in selected}

    for group, cases in selected.items():
        group_dir = out_dir / f"{group}_cases"
        group_dir.mkdir(parents=True, exist_ok=True)

        for case in cases:
            cid = str(case["clip_id"])
            seed = str(case["best_seed"])
            sample = sample_map.get(cid)
            if sample is None or not sample.frame_paths:
                continue

            full_row = seed_maps.get(seed, {}).get("full_v4_2", {}).get(cid)
            if full_row is None:
                continue

            other_name = "wo_semantics_v4_2"
            if group == "semantic_sensitive":
                other_name = "wo_semantics_v4_2"
            elif group == "identity_reconnect":
                other_name = "wo_identity_v4_2"
            else:
                best_seed_row = case.get("per_seed", {}).get(seed, {})
                other_name = str(best_seed_row.get("worse_ablation", "wo_semantics_v4_2"))

            other_row = seed_maps.get(seed, {}).get(other_name, {}).get(cid)
            if other_row is None:
                continue

            frame_idx = int(full_row.get("query_frame_idx", 0))
            frame_idx = max(0, min(frame_idx, len(sample.frame_paths) - 1))
            frame_path = Path(sample.frame_paths[frame_idx])
            if not frame_path.exists():
                continue

            panel = _render_panel(
                clip_id=cid,
                seed=seed,
                frame_path=frame_path,
                full_row=full_row,
                other_row=other_row,
                other_name=other_name,
            )
            out_path = group_dir / f"{cid}_seed{seed}.png"
            panel.save(out_path)
            artifacts[group].append(str(out_path))

            case_summaries[group][cid] = {
                "representative_seed": seed,
                "representative_ablation": other_name,
                "consistency_count": int(case.get("consistency_count", 0)),
                "available_seed_count": int(case.get("available_seed_count", 0)),
                "event_seed_count": int(case.get("event_seed_count", 0)),
                "avg_delta": float(case.get("avg_delta", 0.0)),
                "per_seed": case.get("per_seed", {}),
                "artifact": str(out_path),
            }

    manifest = {
        "runs_root": str(runs_root),
        "seeds": seeds,
        "output_dir": str(out_dir),
        "cases_per_group": cases_per_group,
        "min_consistent_seeds": min_consistent,
        "selected_case_ids": {
            "semantic_sensitive": [x["clip_id"] for x in selected["semantic_sensitive"]],
            "identity_reconnect": [x["clip_id"] for x in selected["identity_reconnect"]],
            "query_grounding": [x["clip_id"] for x in selected["query_grounding"]],
        },
        "case_summaries": case_summaries,
        "artifacts": artifacts,
    }
    (out_dir / "figure_manifest.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
