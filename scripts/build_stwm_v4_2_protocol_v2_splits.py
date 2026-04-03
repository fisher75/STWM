#!/usr/bin/env python3
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_DIR = REPO_ROOT / "manifests" / "protocol_v2"

DATASET_CONFIG = {
    "vspw": {
        "split_files": {
            "train": REPO_ROOT / "data" / "external" / "vspw" / "VSPW" / "train.txt",
            "val": REPO_ROOT / "data" / "external" / "vspw" / "VSPW" / "val.txt",
            "test": REPO_ROOT / "data" / "external" / "vspw" / "VSPW" / "test.txt",
        },
        "frame_dir": REPO_ROOT / "data" / "external" / "vspw" / "VSPW" / "data",
        "frame_subdir": "origin",
        "mask_subdir": "mask",
        "text_labels": ["scene", "object"],
    },
    "vipseg": {
        "split_files": {
            "train": REPO_ROOT / "data" / "external" / "vipseg" / "VIPSeg" / "train.txt",
            "val": REPO_ROOT / "data" / "external" / "vipseg" / "VIPSeg" / "val.txt",
            "test": REPO_ROOT / "data" / "external" / "vipseg" / "VIPSeg" / "test.txt",
        },
        "frame_dir": REPO_ROOT / "data" / "external" / "vipseg" / "VIPSeg" / "imgs",
        "frame_subdir": "",
        "mask_subdir": "../panomasks",
        "text_labels": ["thing", "stuff", "object"],
    },
}


@dataclass
class EventStats:
    sampled_frames: int
    eventful_clip: bool
    reappearance_label_count: int
    reappearance_event_count: int
    max_disappear_gap: int


def _read_split_ids(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"split file missing: {path}")
    ids = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    return sorted(set(ids))


def _pair_frames_and_masks(frame_paths: list[Path], mask_paths: list[Path]) -> tuple[list[str], list[str]]:
    frame_map = {path.stem: path for path in frame_paths if not path.name.startswith("._")}
    mask_map = {path.stem: path for path in mask_paths if not path.name.startswith("._")}
    common = sorted(set(frame_map.keys()) & set(mask_map.keys()))
    paired_frames = [str(frame_map[stem]) for stem in common]
    paired_masks = [str(mask_map[stem]) for stem in common]
    return paired_frames, paired_masks


def _list_clip_entry(dataset: str, split: str, clip_id: str) -> dict[str, Any] | None:
    cfg = DATASET_CONFIG[dataset]

    if dataset == "vspw":
        clip_root = cfg["frame_dir"] / clip_id
        frame_dir = clip_root / cfg["frame_subdir"]
        mask_dir = clip_root / cfg["mask_subdir"]
    else:
        frame_dir = cfg["frame_dir"] / clip_id
        mask_dir = (cfg["frame_dir"] / cfg["mask_subdir"] / clip_id).resolve()

    if not frame_dir.exists() or not mask_dir.exists():
        return None

    frame_paths = sorted(frame_dir.glob("*.jpg")) + sorted(frame_dir.glob("*.png"))
    mask_paths = sorted(mask_dir.glob("*.png"))
    paired_frames, paired_masks = _pair_frames_and_masks(frame_paths, mask_paths)
    if len(paired_frames) < 16:
        return None

    return {
        "clip_id": clip_id,
        "frame_paths": paired_frames,
        "text_labels": list(cfg["text_labels"]),
        "metadata": {
            "dataset": dataset,
            "split": split,
            "source_split_file": str(cfg["split_files"][split]),
            "mask_paths": paired_masks,
            "num_frames": len(paired_frames),
            "protocol_v2": True,
        },
    }


def _subsample_indices(length: int, max_frames: int) -> list[int]:
    if length <= max_frames:
        return list(range(length))
    idx = np.linspace(0, length - 1, max_frames)
    return sorted(set(int(round(x)) for x in idx.tolist()))


def _labels_present(mask_path: str, min_area_ratio: float = 0.001) -> set[int]:
    try:
        arr = np.array(Image.open(mask_path))
    except Exception:
        return set()
    if arr.ndim == 3:
        arr = arr[..., 0]
    arr = arr[::4, ::4]

    values, counts = np.unique(arr, return_counts=True)
    total = float(max(1, arr.size))
    present: set[int] = set()
    for label, count in zip(values.tolist(), counts.tolist()):
        label_i = int(label)
        if label_i == 0:
            continue
        if float(count) / total >= float(min_area_ratio):
            present.add(label_i)
    return present


def _compute_event_stats(mask_paths: list[str], max_frames: int = 48, min_gap: int = 2) -> EventStats:
    if not mask_paths:
        return EventStats(0, False, 0, 0, 0)

    indices = _subsample_indices(len(mask_paths), max_frames=max_frames)
    label_presence: dict[int, list[int]] = {}

    for idx in indices:
        present = _labels_present(mask_paths[idx])
        for label in list(label_presence.keys()):
            label_presence[label].append(1 if label in present else 0)
        for label in present:
            if label not in label_presence:
                label_presence[label] = [0] * (len(indices) - 1) + [1]

    reappearance_label_count = 0
    reappearance_event_count = 0
    max_disappear_gap = 0

    for seq in label_presence.values():
        had_event = False
        for i in range(1, len(seq)):
            if seq[i] != 1:
                continue
            j = i - 1
            gap = 0
            while j >= 0 and seq[j] == 0:
                gap += 1
                j -= 1
            had_visible_before = j >= 0 and seq[j] == 1
            if had_visible_before and gap >= min_gap:
                had_event = True
                reappearance_event_count += 1
                if gap > max_disappear_gap:
                    max_disappear_gap = int(gap)
        if had_event:
            reappearance_label_count += 1

    return EventStats(
        sampled_frames=len(indices),
        eventful_clip=(reappearance_event_count > 0),
        reappearance_label_count=int(reappearance_label_count),
        reappearance_event_count=int(reappearance_event_count),
        max_disappear_gap=int(max_disappear_gap),
    )


def _build_split(dataset: str, split: str) -> list[dict[str, Any]]:
    ids = _read_split_ids(Path(DATASET_CONFIG[dataset]["split_files"][split]))
    out: list[dict[str, Any]] = []
    for clip_id in ids:
        item = _list_clip_entry(dataset=dataset, split=split, clip_id=clip_id)
        if item is not None:
            out.append(item)
    out.sort(key=lambda x: x["clip_id"])
    return out


def _video_key(item: dict[str, Any]) -> str:
    dataset = str(item.get("metadata", {}).get("dataset", "unknown"))
    clip_id = str(item.get("clip_id", ""))
    return f"{dataset}:{clip_id}"


def _audit_split(items: list[dict[str, Any]]) -> dict[str, Any]:
    by_dataset: dict[str, int] = {}
    videos: set[str] = set()
    for item in items:
        dataset = str(item.get("metadata", {}).get("dataset", "unknown"))
        by_dataset[dataset] = by_dataset.get(dataset, 0) + 1
        videos.add(_video_key(item))
    total = max(1, len(items))
    ratio = {k: float(v) / float(total) for k, v in sorted(by_dataset.items())}
    return {
        "clip_count": len(items),
        "video_count": len(videos),
        "dataset_clip_count": dict(sorted(by_dataset.items())),
        "dataset_clip_ratio": ratio,
    }


def _stable_bucket(token: str) -> int:
    return sum(ord(ch) for ch in token) % 100


def _split_val_main_and_internal_final(items: list[dict[str, Any]], main_ratio: float = 0.7) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    cutoff = int(round(float(main_ratio) * 100.0))
    by_dataset: dict[str, list[dict[str, Any]]] = {}
    for item in items:
        dataset = str(item.get("metadata", {}).get("dataset", "unknown"))
        by_dataset.setdefault(dataset, []).append(item)

    main_items: list[dict[str, Any]] = []
    final_items: list[dict[str, Any]] = []
    for dataset, rows in sorted(by_dataset.items()):
        rows = sorted(rows, key=lambda x: str(x.get("clip_id", "")))
        ds_main: list[dict[str, Any]] = []
        ds_final: list[dict[str, Any]] = []
        for item in rows:
            bucket = _stable_bucket(f"{dataset}:{item.get('clip_id', '')}")
            if bucket < cutoff:
                ds_main.append(item)
            else:
                ds_final.append(item)

        if not ds_main and ds_final:
            ds_main.append(ds_final.pop(0))
        if not ds_final and ds_main:
            ds_final.append(ds_main.pop())

        main_items.extend(ds_main)
        final_items.extend(ds_final)

    main_items.sort(key=lambda x: (str(x.get("metadata", {}).get("dataset", "")), x["clip_id"]))
    final_items.sort(key=lambda x: (str(x.get("metadata", {}).get("dataset", "")), x["clip_id"]))
    return main_items, final_items


def main() -> None:
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)

    train_items: list[dict[str, Any]] = []
    val_items: list[dict[str, Any]] = []
    test_items_official: list[dict[str, Any]] = []

    for dataset in ["vspw", "vipseg"]:
        train_items.extend(_build_split(dataset, "train"))
        val_items.extend(_build_split(dataset, "val"))
        test_items_official.extend(_build_split(dataset, "test"))

    train_items.sort(key=lambda x: (str(x.get("metadata", {}).get("dataset", "")), x["clip_id"]))
    val_items.sort(key=lambda x: (str(x.get("metadata", {}).get("dataset", "")), x["clip_id"]))
    test_items_official.sort(key=lambda x: (str(x.get("metadata", {}).get("dataset", "")), x["clip_id"]))

    # If official test masks are unavailable (common for benchmark-held-out test),
    # build an internal final test split from official val by deterministic video-level partition.
    if len(test_items_official) == 0:
        val_main_items, internal_final_items = _split_val_main_and_internal_final(val_items, main_ratio=0.7)
        final_test_source = "internal_from_official_val"
    else:
        val_main_items = list(val_items)
        internal_final_items = list(test_items_official)
        final_test_source = "official_test"

    train_keys = {_video_key(x) for x in train_items}
    val_keys = {_video_key(x) for x in val_main_items}
    test_keys = {_video_key(x) for x in internal_final_items}

    if train_keys & val_keys:
        raise RuntimeError("train and val are not disjoint at video level")
    if train_keys & test_keys:
        raise RuntimeError("train and test are not disjoint at video level")
    if val_keys & test_keys:
        raise RuntimeError("val and test are not disjoint at video level")

    event_candidates: list[tuple[dict[str, Any], EventStats]] = []
    for item in val_main_items:
        stats = _compute_event_stats(item.get("metadata", {}).get("mask_paths", []))
        event_candidates.append((item, stats))

    per_dataset_target = 80
    eventful_items: list[dict[str, Any]] = []
    coverage_rows: list[dict[str, Any]] = []

    for dataset in ["vspw", "vipseg"]:
        rows = [(item, stats) for item, stats in event_candidates if item.get("metadata", {}).get("dataset") == dataset]
        rows.sort(
            key=lambda pair: (
                1 if pair[1].eventful_clip else 0,
                pair[1].reappearance_event_count,
                pair[1].reappearance_label_count,
                pair[1].max_disappear_gap,
                pair[0]["clip_id"],
            ),
            reverse=True,
        )
        selected = [pair for pair in rows if pair[1].eventful_clip][:per_dataset_target]
        if not selected:
            selected = rows[: min(20, len(rows))]

        for item, stats in selected:
            item_copy = json.loads(json.dumps(item))
            item_copy["metadata"]["eventful_protocol_v2"] = True
            item_copy["metadata"]["eventful_stats"] = {
                "sampled_frames": int(stats.sampled_frames),
                "eventful_clip": bool(stats.eventful_clip),
                "reappearance_label_count": int(stats.reappearance_label_count),
                "reappearance_event_count": int(stats.reappearance_event_count),
                "max_disappear_gap": int(stats.max_disappear_gap),
            }
            eventful_items.append(item_copy)

        coverage_rows.append(
            {
                "dataset": dataset,
                "val_clip_count": len(rows),
                "selected_eventful_clip_count": len(selected),
                "selected_with_reappearance": int(sum(1 for _, s in selected if s.eventful_clip)),
                "selected_reappearance_event_count": int(sum(s.reappearance_event_count for _, s in selected)),
                "selected_avg_max_disappear_gap": float(
                    np.mean([s.max_disappear_gap for _, s in selected]) if selected else 0.0
                ),
            }
        )

    eventful_items.sort(key=lambda x: (str(x.get("metadata", {}).get("dataset", "")), x["clip_id"]))

    train_path = MANIFEST_DIR / "train_v2.json"
    val_main_path = MANIFEST_DIR / "protocol_val_main_v1.json"
    val_eventful_path = MANIFEST_DIR / "protocol_val_eventful_v1.json"
    final_test_path = MANIFEST_DIR / "internal_final_test_v1.json"
    audit_path = MANIFEST_DIR / "protocol_v2_split_audit.json"

    train_path.write_text(json.dumps(train_items, indent=2))
    val_main_path.write_text(json.dumps(val_main_items, indent=2))
    val_eventful_path.write_text(json.dumps(eventful_items, indent=2))
    final_test_path.write_text(json.dumps(internal_final_items, indent=2))

    audit = {
        "policy_version": "protocol_v2",
        "split_role": {
            "train_v2": "training only",
            "protocol_val_main_v1": "official model selection",
            "protocol_val_eventful_v1": "secondary diagnostics only",
            "internal_final_test_v1": "final locked reporting only (no model selection)",
        },
        "paths": {
            "train_v2": str(train_path),
            "protocol_val_main_v1": str(val_main_path),
            "protocol_val_eventful_v1": str(val_eventful_path),
            "internal_final_test_v1": str(final_test_path),
        },
        "stats": {
            "train_v2": _audit_split(train_items),
            "protocol_val_main_v1": _audit_split(val_main_items),
            "protocol_val_eventful_v1": _audit_split(eventful_items),
            "internal_final_test_v1": _audit_split(internal_final_items),
        },
        "final_test_source": final_test_source,
        "official_test_available_in_manifest": bool(len(test_items_official) > 0),
        "disjoint_check": {
            "train_vs_val_overlap": int(len(train_keys & val_keys)),
            "train_vs_test_overlap": int(len(train_keys & test_keys)),
            "val_vs_test_overlap": int(len(val_keys & test_keys)),
        },
        "eventful_coverage": {
            "rows": coverage_rows,
            "combined": {
                "selected_eventful_clip_count": int(len(eventful_items)),
                "selected_with_reappearance": int(
                    sum(1 for item in eventful_items if bool(item.get("metadata", {}).get("eventful_stats", {}).get("eventful_clip", False)))
                ),
                "selected_reappearance_event_count": int(
                    sum(int(item.get("metadata", {}).get("eventful_stats", {}).get("reappearance_event_count", 0)) for item in eventful_items)
                ),
            },
        },
    }
    audit_path.write_text(json.dumps(audit, indent=2))

    print("wrote", train_path)
    print("wrote", val_main_path)
    print("wrote", val_eventful_path)
    print("wrote", final_test_path)
    print("wrote", audit_path)


if __name__ == "__main__":
    main()
