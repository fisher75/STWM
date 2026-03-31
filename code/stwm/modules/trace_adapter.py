from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import hashlib

import numpy as np
import torch
from PIL import Image


@dataclass
class TraceSummary:
    centers: torch.Tensor
    velocities: torch.Tensor
    visibility: torch.Tensor
    metadata: dict[str, Any]


class TraceAdapter:
  """Trace summary adapter using available frame and mask signals.

  This adapter intentionally stays lightweight for week-1/2 smoke testing:
  when mask paths are provided it computes per-frame foreground centroids,
  otherwise it falls back to image-center tracks.
  """

  def __init__(
    self,
    feature_dim: int = 4,
    cache_dir: str | Path = "/home/chen034/workspace/stwm/data/cache/trace_summaries",
    use_cache: bool = True,
  ) -> None:
    self.feature_dim = feature_dim
    self.cache_dir = Path(cache_dir)
    self.use_cache = use_cache

  def encode(
    self,
    frame_paths: list[str],
    metadata: dict[str, Any] | None = None,
    clip_id: str | None = None,
  ) -> TraceSummary:
    if not frame_paths:
      empty = torch.zeros(1, 2)
      return TraceSummary(
        centers=empty,
        velocities=torch.zeros_like(empty),
        visibility=torch.zeros(1, 1),
        metadata={"num_frames": 0, "adapter": "trace_mask_center_v1", "empty_input": True},
      )

    metadata = metadata or {}
    target_label_id = metadata.get("target_label_id")
    try:
      target_label_id = int(target_label_id) if target_label_id is not None else None
    except (TypeError, ValueError):
      target_label_id = None
    sample_key = self._sample_key(
      clip_id or Path(frame_paths[0]).stem,
      frame_paths,
      target_label_id=target_label_id,
    )
    cache_path = self.cache_dir / f"{sample_key}.npz"
    if self.use_cache and cache_path.exists():
      return self._load_from_cache(cache_path)

    mask_paths = metadata.get("mask_paths") if isinstance(metadata.get("mask_paths"), list) else []
    centers_list: list[tuple[float, float]] = []
    visibility_list: list[float] = []

    for idx, frame_path in enumerate(frame_paths):
      center: tuple[float, float]
      visible: float

      if idx < len(mask_paths):
        center, visible = self._center_from_mask(Path(mask_paths[idx]), target_label_id=target_label_id)
        if visible < 0.5:
          fallback_center, _ = self._center_from_frame(Path(frame_path))
          if centers_list:
            center = centers_list[-1]
          else:
            center = fallback_center
      else:
        center, visible = self._center_from_frame(Path(frame_path))

      centers_list.append(center)
      visibility_list.append(visible)

    centers = torch.tensor(np.asarray(centers_list), dtype=torch.float32)
    velocities = torch.zeros_like(centers)
    velocities[1:] = centers[1:] - centers[:-1]
    visibility = torch.tensor(np.asarray(visibility_list), dtype=torch.float32).unsqueeze(-1)

    summary = TraceSummary(
      centers=centers,
      velocities=velocities,
      visibility=visibility,
      metadata={
        "num_frames": len(frame_paths),
        "adapter": "trace_mask_center_v1",
        "source": "mask" if mask_paths else "frame_center",
        "visible_ratio": float(visibility.mean().item()),
        "dataset": metadata.get("dataset", "unknown"),
        "target_label_id": target_label_id,
      },
    )

    if self.use_cache:
      self._save_to_cache(cache_path, summary)
      summary.metadata["cache_path"] = str(cache_path)
    return summary

  def _sample_key(self, clip_id: str, frame_paths: list[str], target_label_id: int | None = None) -> str:
    base = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in clip_id)
    base = base[:64] if base else "clip"
    digest = hashlib.sha1(
      "|".join([
        frame_paths[0],
        frame_paths[-1],
        str(len(frame_paths)),
        str(target_label_id) if target_label_id is not None else "none",
      ]).encode("utf-8")
    ).hexdigest()[:12]
    return f"{base}_{digest}"

  def _center_from_mask(self, mask_path: Path, target_label_id: int | None = None) -> tuple[tuple[float, float], float]:
    if not mask_path.exists():
      return (0.5, 0.5), 0.0
    mask = np.array(Image.open(mask_path))
    if target_label_id is None:
      foreground = mask > 0
    else:
      foreground = mask == int(target_label_id)

    # Fallback for clips where selected label disappears in this frame.
    if target_label_id is not None and not np.any(foreground):
      foreground = mask > 0
      if not np.any(foreground):
        return (0.5, 0.5), 0.0
      ys, xs = np.nonzero(foreground)
      h, w = mask.shape[:2]
      cx = float(xs.mean() / max(1, w - 1))
      cy = float(ys.mean() / max(1, h - 1))
      return (cx, cy), 0.0

    if not np.any(foreground):
      return (0.5, 0.5), 0.0

    ys, xs = np.nonzero(foreground)
    h, w = mask.shape[:2]
    cx = float(xs.mean() / max(1, w - 1))
    cy = float(ys.mean() / max(1, h - 1))
    return (cx, cy), 1.0

  def _center_from_frame(self, frame_path: Path) -> tuple[tuple[float, float], float]:
    if not frame_path.exists():
      return (0.5, 0.5), 0.0
    with Image.open(frame_path) as img:
      width, height = img.size
    # Keep fallback center stable while still using real frame dimensions.
    cx = 0.5 if width > 1 else 0.0
    cy = 0.5 if height > 1 else 0.0
    return (cx, cy), 1.0

  def _save_to_cache(self, cache_path: Path, summary: TraceSummary) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
      cache_path,
      centers=summary.centers.detach().cpu().numpy(),
      velocities=summary.velocities.detach().cpu().numpy(),
      visibility=summary.visibility.detach().cpu().numpy(),
    )

  def _load_from_cache(self, cache_path: Path) -> TraceSummary:
    loaded = np.load(cache_path)
    return TraceSummary(
      centers=torch.from_numpy(loaded["centers"]).float(),
      velocities=torch.from_numpy(loaded["velocities"]).float(),
      visibility=torch.from_numpy(loaded["visibility"]).float(),
      metadata={"adapter": "trace_mask_center_v1", "source": "cache", "cache_path": str(cache_path)},
    )
