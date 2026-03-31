from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import json
import random
import re
import zipfile


def natural_key(text: str) -> list[object]:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", text)]


def visible_files(root: Path, suffixes: tuple[str, ...]) -> list[Path]:
    if not root.exists():
        return []
    return sorted(
        [
            path
            for path in root.iterdir()
            if path.is_file() and path.suffix.lower() in suffixes and not path.name.startswith("._") and not path.name.startswith(".")
        ],
        key=lambda p: natural_key(p.name),
    )


def sample_paths(paths: list[Path], count: int, rng: random.Random) -> list[Path]:
    if len(paths) <= count:
        return sorted(paths, key=lambda p: natural_key(p.name))
    return sorted(rng.sample(paths, count), key=lambda p: natural_key(p.name))


def take_frames(paths: list[Path], max_frames: int) -> list[str]:
    return [str(path) for path in paths[:max_frames]]


def unpack_visor_sequence(zip_path: Path, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    extracted = visible_files(out_dir, (".jpg", ".jpeg", ".png"))
    if extracted:
        return extracted

    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(out_dir)

    all_files = sorted(
        [path for path in out_dir.rglob("*") if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}],
        key=lambda p: natural_key(p.name),
    )
    return all_files


def build_vspw_samples(data_root: Path, clip_count: int, max_frames: int, rng: random.Random) -> list[dict]:
    root = data_root / "vspw" / "VSPW" / "data"
    if not root.exists():
        return []
    clip_dirs = [path for path in root.iterdir() if path.is_dir()]
    selected = sample_paths(clip_dirs, clip_count, rng)
    samples = []
    for clip_dir in selected:
        frame_paths = visible_files(clip_dir / "origin", (".jpg", ".jpeg", ".png"))
        mask_paths = visible_files(clip_dir / "mask", (".png",))
        if not frame_paths:
            continue
        samples.append(
            {
                "clip_id": clip_dir.name,
                "frame_paths": take_frames(frame_paths, max_frames),
                "text_labels": ["scene", "object"],
                "metadata": {
                    "dataset": "vspw",
                    "clip_dir": str(clip_dir),
                    "mask_paths": take_frames(mask_paths, max_frames),
                },
            }
        )
    return samples


def build_vipseg_samples(data_root: Path, clip_count: int, max_frames: int, rng: random.Random) -> list[dict]:
    root = data_root / "vipseg" / "VIPSeg"
    frame_root = root / "imgs"
    mask_root = root / "panomasks"
    if not frame_root.exists():
        return []
    clip_dirs = [path for path in frame_root.iterdir() if path.is_dir()]
    selected = sample_paths(clip_dirs, clip_count, rng)
    samples = []
    for clip_dir in selected:
        frame_paths = visible_files(clip_dir, (".jpg", ".jpeg", ".png"))
        mask_paths = visible_files(mask_root / clip_dir.name, (".png",))
        if not frame_paths:
            continue
        samples.append(
            {
                "clip_id": clip_dir.name,
                "frame_paths": take_frames(frame_paths, max_frames),
                "text_labels": ["thing", "stuff", "object"],
                "metadata": {
                    "dataset": "vipseg",
                    "clip_dir": str(clip_dir),
                    "mask_paths": take_frames(mask_paths, max_frames),
                },
            }
        )
    return samples


def burst_leaf_dirs(root: Path) -> list[Path]:
    leaves = []
    for path in root.rglob("*"):
        if path.is_dir():
            files = visible_files(path, (".jpg", ".jpeg", ".png"))
            if files:
                leaves.append(path)
    return leaves


def build_burst_samples(data_root: Path, clip_count: int, max_frames: int, rng: random.Random) -> list[dict]:
    roots = []
    for split in ("train", "val"):
        candidate = data_root / "burst" / "images" / split / "frames" / split
        if candidate.exists():
            roots.append((split, candidate))

    leaves: list[tuple[str, Path]] = []
    for split, root in roots:
        for leaf in burst_leaf_dirs(root):
            leaves.append((split, leaf))

    selected = sample_paths([leaf for _, leaf in leaves], clip_count, rng)
    selected_set = {path for path in selected}
    samples = []
    for split, leaf in leaves:
        if leaf not in selected_set:
            continue
        frame_paths = visible_files(leaf, (".jpg", ".jpeg", ".png"))
        samples.append(
            {
                "clip_id": f"{leaf.parent.name}_{leaf.name}",
                "frame_paths": take_frames(frame_paths, max_frames),
                "text_labels": ["object", "open-world"],
                "metadata": {
                    "dataset": "burst",
                    "split": split,
                    "clip_dir": str(leaf),
                },
            }
        )
    return samples


def build_visor_samples(
    data_root: Path,
    cache_root: Path,
    clip_count: int,
    max_frames: int,
    rng: random.Random,
) -> list[dict]:
    root = data_root / "visor" / "2v6cgv1x04ol22qp9rm9x2j6a7" / "GroundTruth-SparseAnnotations"
    annotations_root = root / "annotations"
    rgb_root = root / "rgb_frames"

    annotation_files: list[tuple[str, Path]] = []
    for split in ("train", "val"):
        split_root = annotations_root / split
        if not split_root.exists():
            continue
        for path in split_root.glob("*.json"):
            if path.name.startswith("."):
                continue
            annotation_files.append((split, path))

    selected = sample_paths([path for _, path in annotation_files], clip_count, rng)
    selected_set = {path for path in selected}
    samples = []
    for split, annotation_path in annotation_files:
        if annotation_path not in selected_set:
            continue
        seq_id = annotation_path.stem
        participant = seq_id.split("_")[0]
        zip_path = rgb_root / split / participant / f"{seq_id}.zip"
        if not zip_path.exists():
            continue
        cache_dir = cache_root / "visor_sequences" / split / seq_id
        frame_paths = unpack_visor_sequence(zip_path, cache_dir)
        if not frame_paths:
            continue
        samples.append(
            {
                "clip_id": seq_id,
                "frame_paths": take_frames(frame_paths, max_frames),
                "text_labels": ["hand", "active object"],
                "metadata": {
                    "dataset": "visor",
                    "split": split,
                    "annotation_path": str(annotation_path),
                    "frame_archive": str(zip_path),
                    "cache_dir": str(cache_dir),
                },
            }
        )
    return samples


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Build STWM mini split manifests from local datasets")
    parser.add_argument("--data-root", default="/home/chen034/workspace/stwm/data/external")
    parser.add_argument("--cache-root", default="/home/chen034/workspace/stwm/data/cache")
    parser.add_argument("--manifest-dir", default="/home/chen034/workspace/stwm/manifests/minisplits")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--max-frames", type=int, default=64)
    parser.add_argument("--vspw-count", type=int, default=20)
    parser.add_argument("--vipseg-count", type=int, default=20)
    parser.add_argument("--burst-count", type=int, default=20)
    parser.add_argument("--visor-count", type=int, default=10)
    return parser


def write_manifest(path: Path, samples: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(samples, indent=2))


def main() -> None:
    args = build_parser().parse_args()
    rng = random.Random(args.seed)
    data_root = Path(args.data_root)
    cache_root = Path(args.cache_root)
    manifest_dir = Path(args.manifest_dir)

    manifests = {
        "vspw_mini.json": build_vspw_samples(data_root, args.vspw_count, args.max_frames, rng),
        "vipseg_mini.json": build_vipseg_samples(data_root, args.vipseg_count, args.max_frames, rng),
        "burst_mini.json": build_burst_samples(data_root, args.burst_count, args.max_frames, rng),
        "visor_mini.json": build_visor_samples(data_root, cache_root, args.visor_count, args.max_frames, rng),
    }

    for name, samples in manifests.items():
        write_manifest(manifest_dir / name, samples)
        print(f"{name}: {len(samples)} samples")

    combined = []
    for samples in manifests.values():
        combined.extend(samples)
    write_manifest(manifest_dir / "stwm_week1_mini.json", combined)
    print(f"stwm_week1_mini.json: {len(combined)} samples")


if __name__ == "__main__":
    main()
