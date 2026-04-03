from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import hashlib
import json
import os
import shutil
import time
import warnings
import zipfile

import fcntl

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
    cache_version: str = "trace_mask_center_v2",
  ) -> None:
    self.feature_dim = feature_dim
    self.cache_dir = Path(cache_dir)
    self.use_cache = use_cache
    self.cache_version = str(cache_version)
    self.frontend_hash = self._compute_frontend_hash()

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

    manifest_hash = self._manifest_hash(metadata)
    cache_meta_expected = {
      "cache_version": self.cache_version,
      "manifest_hash": manifest_hash,
      "frontend_hash": self.frontend_hash,
    }

    sample_key = self._sample_key(
      clip_id or Path(frame_paths[0]).stem,
      frame_paths,
      target_label_id=target_label_id,
      manifest_hash=manifest_hash,
      frontend_hash=self.frontend_hash,
    )
    cache_path = self.cache_dir / f"{sample_key}.npz"
    lock_path = cache_path.with_suffix(cache_path.suffix + ".lock")

    cache_rebuild_reason = ""
    cache_quarantine_path = ""

    if self.use_cache and cache_path.exists():
      try:
        return self._load_from_cache(cache_path, expected_metadata=cache_meta_expected)
      except Exception as exc:
        if not self._is_recoverable_cache_error(exc):
          raise
        cache_rebuild_reason = f"{type(exc).__name__}: {exc}"

    if self.use_cache:
      with self._exclusive_lock(lock_path):
        if cache_path.exists():
          try:
            return self._load_from_cache(cache_path, expected_metadata=cache_meta_expected)
          except Exception as exc:
            if not self._is_recoverable_cache_error(exc):
              raise
            if not cache_rebuild_reason:
              cache_rebuild_reason = f"{type(exc).__name__}: {exc}"
            quarantined = self._quarantine_cache_file(cache_path, exc)
            cache_quarantine_path = str(quarantined) if quarantined is not None else ""

        summary = self._build_summary(frame_paths=frame_paths, metadata=metadata, target_label_id=target_label_id)
        self._save_to_cache(
          cache_path,
          summary,
          sample_key=sample_key,
          cache_metadata=cache_meta_expected,
          target_label_id=target_label_id,
        )
        summary.metadata["cache_path"] = str(cache_path)
        if cache_rebuild_reason:
          summary.metadata["cache_rebuilt"] = True
          summary.metadata["cache_rebuild_reason"] = cache_rebuild_reason
          if cache_quarantine_path:
            summary.metadata["cache_quarantine_path"] = cache_quarantine_path
          warnings.warn(
            (
              f"[trace-cache] bad cache detected at {cache_path}; "
              f"rebuilding from source. reason={cache_rebuild_reason}"
            ),
            RuntimeWarning,
          )
        return summary

    return self._build_summary(frame_paths=frame_paths, metadata=metadata, target_label_id=target_label_id)

  def _build_summary(
    self,
    *,
    frame_paths: list[str],
    metadata: dict[str, Any],
    target_label_id: int | None,
  ) -> TraceSummary:
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

    return TraceSummary(
      centers=centers,
      velocities=velocities,
      visibility=visibility,
      metadata={
        "num_frames": len(frame_paths),
        "adapter": self.cache_version,
        "source": "mask" if mask_paths else "frame_center",
        "visible_ratio": float(visibility.mean().item()),
        "dataset": metadata.get("dataset", "unknown"),
        "target_label_id": target_label_id,
        "cache_version": self.cache_version,
        "manifest_hash": self._manifest_hash(metadata),
        "frontend_hash": self.frontend_hash,
      },
    )

  def _sample_key(
    self,
    clip_id: str,
    frame_paths: list[str],
    target_label_id: int | None = None,
    manifest_hash: str = "",
    frontend_hash: str = "",
  ) -> str:
    base = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in clip_id)
    base = base[:64] if base else "clip"
    digest = hashlib.sha1(
      "|".join([
        frame_paths[0],
        frame_paths[-1],
        str(len(frame_paths)),
        str(target_label_id) if target_label_id is not None else "none",
        manifest_hash if manifest_hash else "none",
        frontend_hash if frontend_hash else "none",
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

  def _save_to_cache(
    self,
    cache_path: Path,
    summary: TraceSummary,
    *,
    sample_key: str,
    cache_metadata: dict[str, Any],
    target_label_id: int | None,
  ) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload_meta = dict(cache_metadata)
    payload_meta.update(
      {
        "adapter": self.cache_version,
        "feature_dim": int(self.feature_dim),
        "sample_key": str(sample_key),
        "target_label_id": int(target_label_id) if target_label_id is not None else None,
      }
    )
    metadata_json = json.dumps(payload_meta, sort_keys=True, separators=(",", ":"))

    temp_path = cache_path.parent / f"{cache_path.name}.tmp.{os.getpid()}.{int(time.time() * 1000)}.npz"
    try:
      np.savez_compressed(
        temp_path,
        centers=summary.centers.detach().cpu().numpy(),
        velocities=summary.velocities.detach().cpu().numpy(),
        visibility=summary.visibility.detach().cpu().numpy(),
        cache_metadata_json=np.asarray(metadata_json),
      )
      with temp_path.open("rb") as temp_fp:
        os.fsync(temp_fp.fileno())
      os.replace(temp_path, cache_path)
      self._fsync_directory(cache_path.parent)
    finally:
      if temp_path.exists():
        try:
          temp_path.unlink()
        except OSError:
          pass

  def _load_from_cache(self, cache_path: Path, expected_metadata: dict[str, Any]) -> TraceSummary:
    with np.load(cache_path, allow_pickle=False) as loaded:
      if "cache_metadata_json" not in loaded.files:
        raise ValueError("missing cache metadata in trace cache")
      cache_meta = self._decode_cache_metadata(loaded["cache_metadata_json"])
      self._validate_cache_metadata(expected_metadata, cache_meta)
      centers = torch.from_numpy(loaded["centers"]).float()
      velocities = torch.from_numpy(loaded["velocities"]).float()
      visibility = torch.from_numpy(loaded["visibility"]).float()

    return TraceSummary(
      centers=centers,
      velocities=velocities,
      visibility=visibility,
      metadata={
        "adapter": str(cache_meta.get("adapter", self.cache_version)),
        "source": "cache",
        "cache_path": str(cache_path),
        "cache_version": str(cache_meta.get("cache_version", "")),
        "manifest_hash": str(cache_meta.get("manifest_hash", "")),
        "frontend_hash": str(cache_meta.get("frontend_hash", "")),
      },
    )

  def _decode_cache_metadata(self, encoded: np.ndarray | Any) -> dict[str, Any]:
    raw: str
    if isinstance(encoded, np.ndarray):
      if encoded.ndim == 0:
        raw = str(encoded.item())
      else:
        raw = str(encoded.tolist())
    else:
      raw = str(encoded)
    try:
      data = json.loads(raw)
    except json.JSONDecodeError as exc:
      raise ValueError(f"invalid cache metadata json: {exc}") from exc
    if not isinstance(data, dict):
      raise ValueError("invalid cache metadata payload type")
    return data

  def _validate_cache_metadata(self, expected: dict[str, Any], cached: dict[str, Any]) -> None:
    for required_key in ("cache_version", "frontend_hash"):
      if required_key not in cached:
        raise ValueError(f"missing cache metadata field: {required_key}")

    for key, expected_value in expected.items():
      if expected_value in (None, ""):
        continue
      cached_value = cached.get(key)
      if str(cached_value) != str(expected_value):
        raise ValueError(
          f"cache metadata mismatch for {key}: expected={expected_value}, got={cached_value}"
        )

  def _manifest_hash(self, metadata: dict[str, Any]) -> str:
    value = metadata.get("manifest_hash")
    if value is None:
      return ""
    return str(value).strip()

  def _compute_frontend_hash(self) -> str:
    digest = hashlib.sha1(f"{self.cache_version}|feature_dim={self.feature_dim}".encode("utf-8")).hexdigest()
    return digest[:16]

  @contextmanager
  def _exclusive_lock(self, lock_path: Path):
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as fp:
      fcntl.flock(fp.fileno(), fcntl.LOCK_EX)
      try:
        yield
      finally:
        fcntl.flock(fp.fileno(), fcntl.LOCK_UN)

  def _quarantine_cache_file(self, cache_path: Path, exc: Exception) -> Path | None:
    if not cache_path.exists():
      return None
    quarantine_dir = self.cache_dir / "quarantine"
    quarantine_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    digest = hashlib.sha1(f"{cache_path.name}|{type(exc).__name__}|{exc}".encode("utf-8")).hexdigest()[:8]
    destination = quarantine_dir / f"{cache_path.stem}.bad_{stamp}_{digest}{cache_path.suffix}"
    try:
      shutil.move(str(cache_path), str(destination))
      return destination
    except Exception as move_exc:
      warnings.warn(
        (
          f"[trace-cache] failed to quarantine bad cache {cache_path}: {move_exc}; "
          "continuing with in-place overwrite"
        ),
        RuntimeWarning,
      )
      return None

  def _is_recoverable_cache_error(self, exc: Exception) -> bool:
    if isinstance(exc, (zipfile.BadZipFile, EOFError, OSError, ValueError, KeyError)):
      return True
    message = str(exc).lower()
    keywords = ("zip", "eof", "metadata", "corrupt", "crc", "decode")
    return any(word in message for word in keywords)

  def _fsync_directory(self, path: Path) -> None:
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
      flags |= os.O_DIRECTORY
    try:
      dir_fd = os.open(str(path), flags)
    except OSError:
      return
    try:
      os.fsync(dir_fd)
    finally:
      os.close(dir_fd)
