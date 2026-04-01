from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import json
import re
from typing import Any


def natural_key(text: str) -> list[object]:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", text)]


def visible_files(root: Path, suffixes: tuple[str, ...]) -> list[Path]:
    if not root.exists():
        return []
    return sorted(
        [
            path
            for path in root.iterdir()
            if path.is_file() and path.suffix.lower() in suffixes and not path.name.startswith(".")
        ],
        key=lambda p: natural_key(p.name),
    )


def read_split_ids(path: Path) -> list[str]:
    items = [line.strip() for line in path.read_text().splitlines()]
    out = [item for item in items if item and not item.startswith(".")]
    return out


def to_sample(
    *,
    dataset: str,
    clip_id: str,
    frame_paths: list[Path],
    mask_paths: list[Path],
    max_frames: int,
    source_split_file: Path,
) -> dict[str, Any] | None:
    if not frame_paths or not mask_paths:
        return None

    keep = min(len(frame_paths), len(mask_paths), max_frames)
    if keep < 4:
        return None

    frames = [str(p) for p in frame_paths[:keep]]
    masks = [str(p) for p in mask_paths[:keep]]

    if dataset == "vspw":
        text_labels = ["scene", "object"]
    else:
        text_labels = ["thing", "stuff", "object"]

    return {
        "clip_id": clip_id,
        "frame_paths": frames,
        "text_labels": text_labels,
        "metadata": {
            "dataset": dataset,
            "split": "train",
            "mask_paths": masks,
            "source_split_file": str(source_split_file),
            "num_frames": keep,
        },
    }


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Build STWM V4.2 real train manifest from VSPW/VIPSeg train splits")
    parser.add_argument("--data-root", default="/home/chen034/workspace/stwm/data/external")
    parser.add_argument("--vspw-train-list", default="/home/chen034/workspace/stwm/data/external/vspw/VSPW/train.txt")
    parser.add_argument("--vipseg-train-list", default="/home/chen034/workspace/stwm/data/external/vipseg/VIPSeg/train.txt")
    parser.add_argument("--max-frames", type=int, default=64)
    parser.add_argument(
        "--output-manifest",
        default="/home/chen034/workspace/stwm/manifests/realsplits/stwm_v4_2_vspw_vipseg_train_v1.json",
    )
    parser.add_argument(
        "--output-report",
        default="/home/chen034/workspace/stwm/reports/stwm_v4_2_real_train_manifest_report_v1.json",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    data_root = Path(args.data_root)
    vspw_train_file = Path(args.vspw_train_list)
    vipseg_train_file = Path(args.vipseg_train_list)

    vspw_ids = read_split_ids(vspw_train_file)
    vipseg_ids = read_split_ids(vipseg_train_file)

    vspw_data_root = data_root / "vspw" / "VSPW" / "data"
    vipseg_img_root = data_root / "vipseg" / "VIPSeg" / "imgs"
    vipseg_mask_root = data_root / "vipseg" / "VIPSeg" / "panomasks"

    samples: list[dict[str, Any]] = []
    vspw_missing = 0
    vipseg_missing = 0

    for clip_id in vspw_ids:
        clip_root = vspw_data_root / clip_id
        frame_paths = visible_files(clip_root / "origin", (".jpg", ".jpeg", ".png"))
        mask_paths = visible_files(clip_root / "mask", (".png",))
        sample = to_sample(
            dataset="vspw",
            clip_id=clip_id,
            frame_paths=frame_paths,
            mask_paths=mask_paths,
            max_frames=max(4, int(args.max_frames)),
            source_split_file=vspw_train_file,
        )
        if sample is None:
            vspw_missing += 1
            continue
        samples.append(sample)

    for clip_id in vipseg_ids:
        frame_paths = visible_files(vipseg_img_root / clip_id, (".jpg", ".jpeg", ".png"))
        mask_paths = visible_files(vipseg_mask_root / clip_id, (".png",))
        sample = to_sample(
            dataset="vipseg",
            clip_id=clip_id,
            frame_paths=frame_paths,
            mask_paths=mask_paths,
            max_frames=max(4, int(args.max_frames)),
            source_split_file=vipseg_train_file,
        )
        if sample is None:
            vipseg_missing += 1
            continue
        samples.append(sample)

    out_manifest = Path(args.output_manifest)
    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    out_manifest.write_text(json.dumps(samples, indent=2))

    vspw_kept = len([x for x in samples if x.get("metadata", {}).get("dataset") == "vspw"])
    vipseg_kept = len([x for x in samples if x.get("metadata", {}).get("dataset") == "vipseg"])

    report = {
        "manifest": str(out_manifest),
        "max_frames": int(args.max_frames),
        "vspw": {
            "train_list": str(vspw_train_file),
            "listed_clips": len(vspw_ids),
            "kept_clips": vspw_kept,
            "missing_or_invalid": vspw_missing,
        },
        "vipseg": {
            "train_list": str(vipseg_train_file),
            "listed_clips": len(vipseg_ids),
            "kept_clips": vipseg_kept,
            "missing_or_invalid": vipseg_missing,
        },
        "combined": {
            "kept_clips": len(samples),
        },
    }

    out_report = Path(args.output_report)
    out_report.parent.mkdir(parents=True, exist_ok=True)
    out_report.write_text(json.dumps(report, indent=2))

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
