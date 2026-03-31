from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import hashlib

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


@dataclass
class SemanticSummary:
    class_scores: torch.Tensor
    text_embeddings: torch.Tensor
    metadata: dict[str, Any]


class SemanticAdapter:
  """Semantic adapter with deterministic text embeddings and mask-aware priors.

  It does not replace OV2VSS/YOLO-World yet, but connects the pipeline to
  real clip metadata and provides stable, cacheable semantics for week-2
  prototype debugging.
  """

  def __init__(
    self,
    num_classes: int = 16,
    text_dim: int = 32,
    cache_dir: str | Path = "/home/chen034/workspace/stwm/data/cache/semantic_summaries",
    use_cache: bool = True,
  ) -> None:
    self.num_classes = num_classes
    self.text_dim = text_dim
    self.cache_dir = Path(cache_dir)
    self.use_cache = use_cache

  def encode(
    self,
    text_labels: list[str],
    num_steps: int,
    metadata: dict[str, Any] | None = None,
    clip_id: str | None = None,
  ) -> SemanticSummary:
    labels = self._normalize_labels(text_labels)
    metadata = metadata or {}
    sample_key = self._sample_key(clip_id or "clip", labels, num_steps)
    cache_path = self.cache_dir / f"{sample_key}.pt"

    if self.use_cache and cache_path.exists():
      return self._load_from_cache(cache_path)

    objectness, source = self._build_objectness_signal(metadata, num_steps)
    num_labels = len(labels)

    class_scores = torch.full((num_steps, num_labels, self.num_classes), 1e-3, dtype=torch.float32)
    for idx, label in enumerate(labels):
      class_idx = self._class_index(label)
      signal = self._label_signal(label, objectness)
      class_scores[:, idx, class_idx] = signal

    class_scores = class_scores / class_scores.sum(dim=-1, keepdim=True)

    base_embeddings = torch.stack([self._text_embedding(label) for label in labels], dim=0)
    text_embeddings = base_embeddings.unsqueeze(0).repeat(num_steps, 1, 1)

    summary = SemanticSummary(
      class_scores=class_scores,
      text_embeddings=text_embeddings,
      metadata={
        "labels": labels,
        "adapter": "semantic_proxy_v1",
        "objectness_source": source,
        "mean_objectness": float(objectness.mean().item()) if num_steps > 0 else 0.0,
        "dataset": metadata.get("dataset", "unknown"),
      },
    )

    if self.use_cache:
      self._save_to_cache(cache_path, summary)
      summary.metadata["cache_path"] = str(cache_path)
    return summary

  def _normalize_labels(self, labels: list[str]) -> list[str]:
    clean = [str(label).strip() for label in labels if str(label).strip()]
    if not clean:
      clean = ["object"]
    # Keep order while deduplicating.
    return list(dict.fromkeys(clean))

  def _sample_key(self, clip_id: str, labels: list[str], num_steps: int) -> str:
    base = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in clip_id)[:64]
    digest = hashlib.sha1(f"{labels}|{num_steps}".encode("utf-8")).hexdigest()[:12]
    return f"{base or 'clip'}_{digest}"

  def _class_index(self, label: str) -> int:
    digest = hashlib.sha1(label.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % self.num_classes

  def _text_embedding(self, label: str) -> torch.Tensor:
    seed = int(hashlib.sha1(label.encode("utf-8")).hexdigest()[:8], 16)
    generator = torch.Generator().manual_seed(seed)
    vec = torch.randn(self.text_dim, generator=generator)
    return F.normalize(vec, dim=0)

  def _label_signal(self, label: str, objectness: torch.Tensor) -> torch.Tensor:
    label_lower = label.lower()
    if any(tag in label_lower for tag in ("scene", "stuff", "background")):
      return 0.2 + 0.8 * (1.0 - objectness)
    if any(tag in label_lower for tag in ("object", "thing", "hand", "active", "open-world")):
      return 0.2 + 0.8 * objectness
    return torch.full_like(objectness, 0.5)

  def _build_objectness_signal(self, metadata: dict[str, Any], num_steps: int) -> tuple[torch.Tensor, str]:
    if num_steps <= 0:
      return torch.zeros(0, dtype=torch.float32), "empty"

    mask_paths = metadata.get("mask_paths")
    if not isinstance(mask_paths, list) or not mask_paths:
      return torch.full((num_steps,), 0.5, dtype=torch.float32), "constant_0.5"

    ratios = torch.zeros(num_steps, dtype=torch.float32)
    target_label_id = metadata.get("target_label_id")
    try:
      target_label_id = int(target_label_id) if target_label_id is not None else None
    except (TypeError, ValueError):
      target_label_id = None

    for idx in range(num_steps):
      if idx >= len(mask_paths):
        ratios[idx] = ratios[idx - 1] if idx > 0 else 0.5
        continue

      mask_path = Path(mask_paths[idx])
      if not mask_path.exists():
        ratios[idx] = ratios[idx - 1] if idx > 0 else 0.5
        continue

      mask = np.array(Image.open(mask_path))
      if target_label_id is None:
        foreground_ratio = float((mask > 0).mean())
      else:
        foreground_ratio = float((mask == int(target_label_id)).mean())
      ratios[idx] = min(max(foreground_ratio * 2.0, 0.0), 1.0)

    if target_label_id is not None:
      return ratios, "target_label_ratio"
    return ratios, "mask_foreground_ratio"

  def _save_to_cache(self, cache_path: Path, summary: SemanticSummary) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
      "class_scores": summary.class_scores.detach().cpu(),
      "text_embeddings": summary.text_embeddings.detach().cpu(),
      "metadata": summary.metadata,
    }
    torch.save(payload, cache_path)

  def _load_from_cache(self, cache_path: Path) -> SemanticSummary:
    payload = torch.load(cache_path, map_location="cpu")
    metadata = dict(payload.get("metadata", {}))
    metadata["source"] = "cache"
    metadata["cache_path"] = str(cache_path)
    return SemanticSummary(
      class_scores=payload["class_scores"].float(),
      text_embeddings=payload["text_embeddings"].float(),
      metadata=metadata,
    )
