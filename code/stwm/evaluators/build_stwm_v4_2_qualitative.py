from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import json
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from stwm.datasets.stwm_dataset import STWMDataset


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Build STWM V4.2 qualitative case packs")
    parser.add_argument("--runs-root", default="/home/chen034/workspace/stwm/outputs/training/stwm_v4_2_minival_seed42")
    parser.add_argument("--manifest", default="/home/chen034/workspace/stwm/manifests/minisplits/stwm_week2_minival_v2.json")
    parser.add_argument("--data-root", default="/home/chen034/workspace/stwm/data/external")
    parser.add_argument("--output-dir", default="/home/chen034/workspace/stwm/outputs/visualizations/stwm_v4_2_minival_seed42")
    parser.add_argument("--cases-per-group", type=int, default=8)
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
        d.rectangle((0, 0, frame.width, 52), fill=(0, 0, 0))

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

    dl.text((8, 8), f"full_v4_2 | {clip_id}", fill=(255, 255, 255))
    dl.text(
        (8, 28),
        "traj={:.4f} q={:.4f}".format(
            float(full_row.get("trajectory_l1", 0.0)),
            float(full_row.get("query_localization_error", 0.0)),
        ),
        fill=(255, 255, 255),
    )

    dr.text((8, 8), f"{other_name}", fill=(255, 255, 255))
    dr.text(
        (8, 28),
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


def main() -> None:
    args = build_parser().parse_args()
    runs_root = Path(args.runs_root)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = STWMDataset(args.data_root, manifest=args.manifest, limit=None)
    sample_map = {s.clip_id: s for s in dataset.samples}

    full_rows = _last_by_clip(_read_rows(runs_root / "full_v4_2" / "train_log.jsonl"))
    ws_rows = _last_by_clip(_read_rows(runs_root / "wo_semantics_v4_2" / "train_log.jsonl"))
    wi_rows = _last_by_clip(_read_rows(runs_root / "wo_identity_v4_2" / "train_log.jsonl"))

    common_ws = sorted(set(full_rows).intersection(ws_rows).intersection(sample_map))
    common_wi = sorted(set(full_rows).intersection(wi_rows).intersection(sample_map))

    k = max(1, int(args.cases_per_group))

    sem_rank = []
    for cid in common_ws:
        f = full_rows[cid]
        w = ws_rows[cid]
        score = (
            float(w.get("trajectory_l1", 0.0))
            + float(w.get("query_localization_error", 0.0))
            - float(f.get("trajectory_l1", 0.0))
            - float(f.get("query_localization_error", 0.0))
        )
        sem_rank.append((cid, score))
    sem_rank.sort(key=lambda x: x[1], reverse=True)
    semantic_ids = [x[0] for x in sem_rank[:k]]

    id_rank = []
    for cid in common_wi:
        f = full_rows[cid]
        w = wi_rows[cid]
        score = (
            float(w.get("reconnect_min_error", 0.0))
            - float(f.get("reconnect_min_error", 0.0))
            + 0.6 * (float(f.get("reconnect_success", 0.0)) - float(w.get("reconnect_success", 0.0)))
        )
        # Prioritize clips where reappearance actually exists.
        if float(f.get("has_reappearance_event", 0.0)) > 0.5 or float(w.get("has_reappearance_event", 0.0)) > 0.5:
            score += 0.5
        id_rank.append((cid, score))
    id_rank.sort(key=lambda x: x[1], reverse=True)
    identity_ids = [x[0] for x in id_rank[:k]]

    query_rank = []
    for cid in sorted(set(common_ws).intersection(common_wi)):
        f = full_rows[cid]
        ws = ws_rows[cid]
        wi = wi_rows[cid]
        d_ws = float(ws.get("query_localization_error", 0.0)) - float(f.get("query_localization_error", 0.0))
        d_wi = float(wi.get("query_localization_error", 0.0)) - float(f.get("query_localization_error", 0.0))
        query_rank.append((cid, min(d_ws, d_wi), d_ws, d_wi))
    query_rank.sort(key=lambda x: x[1], reverse=True)
    query_ids = [x[0] for x in query_rank[:k]]

    groups = {
        "semantic_sensitive_cases": semantic_ids,
        "identity_reconnect_cases": identity_ids,
        "query_grounding_cases": query_ids,
    }

    artifacts: dict[str, list[str]] = {k: [] for k in groups}

    for group, ids in groups.items():
        gdir = out_dir / group
        gdir.mkdir(parents=True, exist_ok=True)
        for cid in ids:
            sample = sample_map.get(cid)
            if sample is None or not sample.frame_paths:
                continue

            f = full_rows.get(cid)
            if f is None:
                continue

            if group == "semantic_sensitive_cases":
                other = ws_rows.get(cid)
                other_name = "wo_semantics_v4_2"
            elif group == "identity_reconnect_cases":
                other = wi_rows.get(cid)
                other_name = "wo_identity_v4_2"
            else:
                ws = ws_rows.get(cid)
                wi = wi_rows.get(cid)
                if ws is None and wi is None:
                    continue
                ws_q = float(ws.get("query_localization_error", 0.0)) if ws is not None else -1e9
                wi_q = float(wi.get("query_localization_error", 0.0)) if wi is not None else -1e9
                if ws_q >= wi_q:
                    other = ws
                    other_name = "wo_semantics_v4_2"
                else:
                    other = wi
                    other_name = "wo_identity_v4_2"

            if other is None:
                continue

            frame_idx = int(f.get("query_frame_idx", 0))
            frame_idx = max(0, min(frame_idx, len(sample.frame_paths) - 1))
            frame_path = Path(sample.frame_paths[frame_idx])
            if not frame_path.exists():
                continue

            panel = _render_panel(cid, frame_path, f, other, other_name)
            out_path = gdir / f"{cid}.png"
            panel.save(out_path)
            artifacts[group].append(str(out_path))

    manifest = {
        "runs_root": str(runs_root),
        "output_dir": str(out_dir),
        "cases_per_group": int(k),
        "selected": groups,
        "artifacts": artifacts,
    }
    (out_dir / "figure_manifest.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
