from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json
import random

from torch.utils.data import Dataset


@dataclass
class ClipSample:
    clip_id: str
    frame_paths: list[str]
    text_labels: list[str]
    metadata: dict[str, Any]


class STWMDataset(Dataset):
    """Clip dataset wrapper with mini-split-first loading behavior.

    Default behavior is to load `manifests/minisplits/stwm_week1_mini.json`
    when it can be resolved from `root`. If no manifest is found, the class
    falls back to lightweight directory discovery for quick smoke tests.
    """

    def __init__(
        self,
        root: str | Path,
        manifest: str | Path | None = None,
        limit: int | None = None,
        min_frames: int = 2,
        require_existing_paths: bool = True,
    ) -> None:
        self.root = Path(root)
        self.limit = limit
        self.min_frames = min_frames
        self.require_existing_paths = require_existing_paths
        self.manifest_path = self._resolve_manifest_path(manifest)

        if self.manifest_path is not None:
            self.samples = self._load_manifest(self.manifest_path)
        else:
            self.samples = self._discover_samples()

        self.samples = [sample for sample in self.samples if len(sample.frame_paths) >= self.min_frames]
        if self.limit is not None:
            self.samples = self.samples[: self.limit]
        if not self.samples:
            self.samples = [self._dummy_sample("synthetic-000")]

    def _resolve_manifest_path(self, manifest: str | Path | None) -> Path | None:
        if manifest:
            path = Path(manifest)
            return path if path.exists() else None

        candidates: list[Path] = []
        # Most common case: root=/.../stwm/data/external.
        if len(self.root.parents) >= 2:
            candidates.append(self.root.parents[1] / "manifests" / "minisplits" / "stwm_week1_mini.json")
        # root=/.../stwm
        candidates.append(self.root / "manifests" / "minisplits" / "stwm_week1_mini.json")
        # absolute fallback used in this workspace.
        candidates.append(Path("/home/chen034/workspace/stwm/manifests/minisplits/stwm_week1_mini.json"))

        for candidate in candidates:
            if candidate.exists():
                return candidate
        return None

    def _load_manifest(self, manifest: str | Path) -> list[ClipSample]:
        manifest_path = Path(manifest)
        data = json.loads(manifest_path.read_text())
        samples: list[ClipSample] = []

        for item in data:
            frame_paths = self._sanitize_paths(item.get("frame_paths", []))
            if not frame_paths:
                continue

            metadata = dict(item.get("metadata", {}))
            mask_paths = metadata.get("mask_paths")
            if isinstance(mask_paths, list):
                clean_masks = self._sanitize_paths(mask_paths)
                if clean_masks:
                    metadata["mask_paths"] = clean_masks
                else:
                    metadata.pop("mask_paths", None)

            metadata.setdefault("dataset", metadata.get("dataset", "unknown"))
            metadata["manifest_path"] = str(manifest_path)
            metadata["num_frames"] = len(frame_paths)
            metadata["num_masks"] = len(metadata.get("mask_paths", []))

            text_labels = item.get("text_labels")
            if not isinstance(text_labels, list) or not text_labels:
                text_labels = self._default_labels(metadata["dataset"])

            clip_id = str(item.get("clip_id") or Path(frame_paths[0]).parent.name)
            samples.append(
                ClipSample(
                    clip_id=clip_id,
                    frame_paths=frame_paths,
                    text_labels=[str(label) for label in text_labels],
                    metadata=metadata,
                )
            )

        return samples

    def _sanitize_paths(self, paths: list[Any]) -> list[str]:
        out: list[str] = []
        for path in paths:
            path_str = str(path)
            if not path_str:
                continue
            if self.require_existing_paths and not Path(path_str).exists():
                continue
            out.append(path_str)
        return out

    def _default_labels(self, dataset: str) -> list[str]:
        name = dataset.lower()
        if name == "vspw":
            return ["scene", "object"]
        if name == "vipseg":
            return ["thing", "stuff", "object"]
        if name == "burst":
            return ["object", "open-world"]
        if name == "visor":
            return ["hand", "active object"]
        return ["object", "scene"]

    def _discover_samples(self) -> list[ClipSample]:
        samples: list[ClipSample] = []
        if not self.root.exists():
            return [self._dummy_sample("missing-root")]

        for clip_dir in sorted([p for p in self.root.iterdir() if p.is_dir()]):
            frame_paths = [
                str(p)
                for p in sorted(clip_dir.glob("*.jpg"))
                + sorted(clip_dir.glob("*.jpeg"))
                + sorted(clip_dir.glob("*.png"))
                if not p.name.startswith("._")
            ]
            if len(frame_paths) < self.min_frames:
                continue

            samples.append(
                ClipSample(
                    clip_id=clip_dir.name,
                    frame_paths=frame_paths,
                    text_labels=["object", "scene"],
                    metadata={"dataset": "discovered", "source_dir": str(clip_dir), "num_frames": len(frame_paths)},
                )
            )

        return samples

    def _dummy_sample(self, clip_id: str) -> ClipSample:
        frame_paths = [f"{clip_id}/frame_{idx:05d}.jpg" for idx in range(8)]
        labels = random.sample(["object", "hand", "cup", "bowl", "scene"], k=3)
        return ClipSample(
            clip_id=clip_id,
            frame_paths=frame_paths,
            text_labels=labels,
            metadata={"dataset": "synthetic", "synthetic": True, "num_frames": len(frame_paths)},
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> ClipSample:
        return self.samples[index]
