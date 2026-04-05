from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import hashlib
import importlib.util
import json
import os
import shutil
import time
import warnings

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from stwm.modules.semantic_adapter import SemanticAdapter, SemanticSummary


_CACHE_VERSION = "semantic_teacher_v2"


@dataclass
class _ClipBackendState:
    initialized: bool = False
    available: bool = False
    reason: str = ""


class SemanticAdapterTeacherV2:
    """Teacher-grounded semantic adapter with ordered backend fallback.

    Priority:
      1) open-vocab grounding teacher (stub gate via env)
      2) CLIP visual mask-crop teacher
      3) proxy fallback (deterministic) and optional strict stop
    """

    def __init__(
        self,
        num_classes: int = 16,
        text_dim: int = 32,
        cache_dir: str | Path = "/home/chen034/workspace/stwm/data/cache/semantic_teacher_summaries_v2",
        use_cache: bool = True,
        strict_teacher: bool = False,
        capability_report_path: str = "",
        clip_model_name: str = "ViT-B/32",
    ) -> None:
        self.num_classes = int(num_classes)
        self.text_dim = int(text_dim)
        self.cache_dir = Path(cache_dir)
        self.use_cache = bool(use_cache)
        self.strict_teacher = bool(strict_teacher)
        self.capability_report_path = str(capability_report_path).strip()
        self.clip_model_name = str(clip_model_name)

        self._proxy = SemanticAdapter(
            num_classes=self.num_classes,
            text_dim=self.text_dim,
            cache_dir=self.cache_dir / "proxy_fallback",
            use_cache=self.use_cache,
        )
        self._clip_state = _ClipBackendState()
        self._clip_module = None
        self._clip_model = None
        self._clip_preprocess = None
        self._clip_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def encode(
        self,
        text_labels: list[str],
        num_steps: int,
        metadata: dict[str, Any] | None = None,
        clip_id: str | None = None,
    ) -> SemanticSummary:
        labels = self._normalize_labels(text_labels)
        metadata = dict(metadata or {})
        sample_key = self._sample_key(clip_id or "clip", labels, num_steps, metadata)
        cache_path = self.cache_dir / f"{sample_key}.pt"

        cache_rebuild_reason = ""
        quarantine_path = ""
        if self.use_cache and cache_path.exists():
            try:
                return self._load_from_cache(cache_path)
            except Exception as exc:
                if not self._is_recoverable_cache_error(exc):
                    raise
                cache_rebuild_reason = f"{type(exc).__name__}: {exc}"
                quarantined = self._quarantine_cache_file(cache_path, exc)
                quarantine_path = str(quarantined) if quarantined is not None else ""
                warnings.warn(
                    (
                        f"[semantic-teacher-cache] bad cache detected at {cache_path}; "
                        f"rebuilding from source. reason={cache_rebuild_reason}"
                    ),
                    RuntimeWarning,
                )

        summary = self._encode_uncached(
            text_labels=labels,
            num_steps=int(num_steps),
            metadata=metadata,
            clip_id=str(clip_id or "clip"),
        )

        if cache_rebuild_reason:
            summary.metadata["cache_rebuilt"] = True
            summary.metadata["cache_rebuild_reason"] = cache_rebuild_reason
            if quarantine_path:
                summary.metadata["cache_quarantine_path"] = quarantine_path

        if self.use_cache:
            self._save_to_cache(cache_path, summary)
            summary.metadata["cache_path"] = str(cache_path)
        return summary

    def cache_path_for_sample(self, text_labels: list[str], num_steps: int, clip_id: str | None = None) -> Path:
        labels = self._normalize_labels(text_labels)
        sample_key = self._sample_key(clip_id or "clip", labels, num_steps, {})
        return self.cache_dir / f"{sample_key}.pt"

    def is_cache_error_recoverable(self, exc: Exception) -> bool:
        return self._is_recoverable_cache_error(exc)

    def _encode_uncached(
        self,
        *,
        text_labels: list[str],
        num_steps: int,
        metadata: dict[str, Any],
        clip_id: str,
    ) -> SemanticSummary:
        num_steps = max(0, int(num_steps))
        objectness, objectness_source = self._build_objectness_signal(metadata, num_steps)
        manifest_hash = str(metadata.get("manifest_hash", ""))
        frontend_hash = self._frontend_hash(clip_id=clip_id, metadata=metadata, num_steps=num_steps)

        backend_attempts: list[dict[str, str]] = []

        if self._ov_teacher_available():
            backend_attempts.append({"backend": "ov_teacher", "status": "not_implemented"})

        clip_summary, clip_reason = self._try_encode_with_clip(
            text_labels=text_labels,
            num_steps=num_steps,
            metadata=metadata,
            objectness=objectness,
        )
        if clip_summary is not None:
            clip_summary.metadata.update(
                {
                    "adapter": "semantic_teacher_v2",
                    "teacher_backend": "clip_mask_crop_v1",
                    "objectness_source": objectness_source,
                    "cache_version": _CACHE_VERSION,
                    "manifest_hash": manifest_hash,
                    "frontend_hash": frontend_hash,
                    "strict_teacher": bool(self.strict_teacher),
                }
            )
            return clip_summary
        backend_attempts.append({"backend": "clip_mask_crop_v1", "status": clip_reason})

        if self.strict_teacher:
            self._write_capability_gap_report(
                {
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "clip_id": str(clip_id),
                    "num_steps": int(num_steps),
                    "text_labels": [str(x) for x in text_labels],
                    "strict_teacher": True,
                    "required_priority": [
                        "open_vocab_grounding_teacher",
                        "clip_siglip_open_clip_mask_crop",
                    ],
                    "attempts": backend_attempts,
                    "message": "No teacher backend available; fallback disabled by strict mode.",
                }
            )
            raise RuntimeError("semantic teacher capability gap: no teacher backend available")

        fallback = self._proxy.encode(
            text_labels=text_labels,
            num_steps=int(num_steps),
            metadata=metadata,
            clip_id=clip_id,
        )
        fallback.metadata.update(
            {
                "adapter": "semantic_teacher_v2",
                "teacher_backend": "fallback_proxy_v1",
                "cache_version": _CACHE_VERSION,
                "manifest_hash": manifest_hash,
                "frontend_hash": frontend_hash,
                "strict_teacher": bool(self.strict_teacher),
                "backend_attempts": backend_attempts,
            }
        )
        return fallback

    def _ov_teacher_available(self) -> bool:
        return str(os.getenv("STWM_ENABLE_OV_TEACHER", "")).strip() in {"1", "true", "TRUE"}

    def _try_encode_with_clip(
        self,
        *,
        text_labels: list[str],
        num_steps: int,
        metadata: dict[str, Any],
        objectness: torch.Tensor,
    ) -> tuple[SemanticSummary | None, str]:
        ok, reason = self._ensure_clip_backend()
        if not ok:
            return None, f"clip_unavailable:{reason}"

        frame_paths = metadata.get("frame_paths") if isinstance(metadata.get("frame_paths"), list) else []
        mask_paths = metadata.get("mask_paths") if isinstance(metadata.get("mask_paths"), list) else []
        if not frame_paths:
            return None, "missing_frame_paths"

        try:
            import clip as clip_mod  # type: ignore
        except Exception as exc:  # pragma: no cover - guarded by _ensure_clip_backend
            return None, f"clip_import_error:{type(exc).__name__}:{exc}"

        labels = self._normalize_labels(text_labels)
        text_tokens = clip_mod.tokenize(labels).to(self._clip_device)

        with torch.inference_mode():
            text_features = self._clip_model.encode_text(text_tokens).float()
            text_features = F.normalize(text_features, dim=-1)

        projected_text = self._project_clip_embeddings(text_features.cpu())
        class_indices = [self._class_index(label) for label in labels]

        class_scores_frames: list[torch.Tensor] = []
        for t in range(int(num_steps)):
            frame_idx = min(max(0, t), max(0, len(frame_paths) - 1))
            frame_path = str(frame_paths[frame_idx])
            mask_path = str(mask_paths[frame_idx]) if frame_idx < len(mask_paths) else ""
            img = self._load_teacher_crop(frame_path=frame_path, mask_path=mask_path, metadata=metadata)
            if img is None:
                if class_scores_frames:
                    class_scores_frames.append(class_scores_frames[-1].clone())
                    continue
                return None, f"frame_unreadable:{frame_path}"

            try:
                image_tensor = self._clip_preprocess(img).unsqueeze(0).to(self._clip_device)
            except Exception as exc:
                return None, f"clip_preprocess_failed:{type(exc).__name__}:{exc}"

            with torch.inference_mode():
                image_features = self._clip_model.encode_image(image_tensor).float()
                image_features = F.normalize(image_features, dim=-1)
                label_logits = torch.matmul(image_features, text_features.transpose(0, 1))[0]
                label_probs = torch.softmax(label_logits, dim=-1).cpu()

            cls_scores = torch.full((len(labels), self.num_classes), 1e-4, dtype=torch.float32)
            for li, class_idx in enumerate(class_indices):
                prior = float(label_probs[li].item())
                bias = float(objectness[t].item())
                cls_scores[li, class_idx] += max(1e-4, prior * (0.4 + 0.6 * bias))
            cls_scores = cls_scores / cls_scores.sum(dim=-1, keepdim=True).clamp(min=1e-6)
            class_scores_frames.append(cls_scores)

        if not class_scores_frames:
            return None, "no_valid_frames"

        class_scores = torch.stack(class_scores_frames, dim=0)
        text_embeddings = projected_text.unsqueeze(0).repeat(class_scores.shape[0], 1, 1)

        summary = SemanticSummary(
            class_scores=class_scores.float(),
            text_embeddings=text_embeddings.float(),
            metadata={
                "labels": labels,
                "clip_model": self.clip_model_name,
                "clip_device": str(self._clip_device),
                "teacher_backend": "clip_mask_crop_v1",
                "mean_objectness": float(objectness.mean().item()) if objectness.numel() else 0.0,
            },
        )
        return summary, "ok"

    def _ensure_clip_backend(self) -> tuple[bool, str]:
        if self._clip_state.initialized:
            return bool(self._clip_state.available), str(self._clip_state.reason)

        self._clip_state.initialized = True
        if importlib.util.find_spec("clip") is None:
            self._clip_state.available = False
            self._clip_state.reason = "module_not_found"
            return False, self._clip_state.reason

        try:
            import clip as clip_mod  # type: ignore

            model, preprocess = clip_mod.load(self.clip_model_name, device=str(self._clip_device), jit=False)
            model.eval()
            self._clip_module = clip_mod
            self._clip_model = model
            self._clip_preprocess = preprocess
            self._clip_state.available = True
            self._clip_state.reason = "ok"
            return True, self._clip_state.reason
        except Exception as exc:
            self._clip_state.available = False
            self._clip_state.reason = f"load_failed:{type(exc).__name__}:{exc}"
            return False, self._clip_state.reason

    def _project_clip_embeddings(self, vecs: torch.Tensor) -> torch.Tensor:
        if vecs.ndim != 2:
            raise ValueError(f"clip text embeddings must be 2D, got {tuple(vecs.shape)}")
        in_dim = int(vecs.shape[-1])
        out_dim = int(self.text_dim)
        if in_dim == out_dim:
            return F.normalize(vecs.float(), dim=-1)
        if in_dim > out_dim:
            return F.normalize(vecs[:, :out_dim].float(), dim=-1)
        pad = torch.zeros(vecs.shape[0], out_dim - in_dim, dtype=vecs.dtype)
        return F.normalize(torch.cat([vecs, pad], dim=-1).float(), dim=-1)

    def _load_teacher_crop(self, *, frame_path: str, mask_path: str, metadata: dict[str, Any]) -> Image.Image | None:
        p = Path(frame_path)
        if not p.exists():
            return None
        try:
            image = Image.open(p).convert("RGB")
        except Exception:
            return None

        mp = Path(mask_path) if mask_path else None
        if mp is None or not mp.exists():
            return image

        try:
            mask = np.array(Image.open(mp))
        except Exception:
            return image

        if mask.ndim == 3:
            mask = mask[..., 0]

        target_label_id = metadata.get("target_label_id")
        try:
            target_label_id = int(target_label_id) if target_label_id is not None else None
        except (TypeError, ValueError):
            target_label_id = None

        if target_label_id is None:
            fg = mask > 0
        else:
            fg = mask == int(target_label_id)
            if not fg.any():
                fg = mask > 0
        if not fg.any():
            return image

        ys, xs = np.nonzero(fg)
        y0 = int(max(0, ys.min() - 2))
        y1 = int(min(mask.shape[0], ys.max() + 3))
        x0 = int(max(0, xs.min() - 2))
        x1 = int(min(mask.shape[1], xs.max() + 3))
        if y1 <= y0 or x1 <= x0:
            return image
        return image.crop((x0, y0, x1, y1))

    def _normalize_labels(self, labels: list[str]) -> list[str]:
        clean = [str(label).strip() for label in labels if str(label).strip()]
        if not clean:
            clean = ["object"]
        return list(dict.fromkeys(clean))

    def _sample_key(self, clip_id: str, labels: list[str], num_steps: int, metadata: dict[str, Any]) -> str:
        base = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in clip_id)[:64]
        digest_src = {
            "labels": labels,
            "num_steps": int(num_steps),
            "manifest_hash": str(metadata.get("manifest_hash", "")),
            "frontend_hash": self._frontend_hash(clip_id=clip_id, metadata=metadata, num_steps=num_steps),
            "version": _CACHE_VERSION,
        }
        digest = hashlib.sha1(json.dumps(digest_src, sort_keys=True).encode("utf-8")).hexdigest()[:12]
        return f"{base or 'clip'}_{digest}"

    def _class_index(self, label: str) -> int:
        digest = hashlib.sha1(label.encode("utf-8")).hexdigest()
        return int(digest[:8], 16) % self.num_classes

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
            mask_path = Path(str(mask_paths[idx]))
            if not mask_path.exists():
                ratios[idx] = ratios[idx - 1] if idx > 0 else 0.5
                continue
            mask = np.array(Image.open(mask_path))
            if mask.ndim == 3:
                mask = mask[..., 0]
            if target_label_id is None:
                fg = float((mask > 0).mean())
            else:
                fg = float((mask == int(target_label_id)).mean())
            ratios[idx] = min(max(fg * 2.0, 0.0), 1.0)

        if target_label_id is not None:
            return ratios, "target_label_ratio"
        return ratios, "mask_foreground_ratio"

    def _frontend_hash(self, *, clip_id: str, metadata: dict[str, Any], num_steps: int) -> str:
        frame_paths = metadata.get("frame_paths") if isinstance(metadata.get("frame_paths"), list) else []
        mask_paths = metadata.get("mask_paths") if isinstance(metadata.get("mask_paths"), list) else []
        payload = {
            "clip_id": str(clip_id),
            "num_steps": int(num_steps),
            "frame_paths": [str(x) for x in frame_paths[: int(num_steps)]],
            "mask_paths": [str(x) for x in mask_paths[: int(num_steps)]],
        }
        return hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()

    def _write_capability_gap_report(self, payload: dict[str, Any]) -> None:
        path_raw = str(self.capability_report_path).strip()
        if not path_raw:
            return
        path = Path(path_raw)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2))

    def _save_to_cache(self, cache_path: Path, summary: SemanticSummary) -> None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "class_scores": summary.class_scores.detach().cpu(),
            "text_embeddings": summary.text_embeddings.detach().cpu(),
            "metadata": summary.metadata,
        }
        temp_path = cache_path.parent / f"{cache_path.name}.tmp.{os.getpid()}.{int(time.time() * 1000)}"
        try:
            torch.save(payload, temp_path)
            os.replace(temp_path, cache_path)
        finally:
            if temp_path.exists():
                try:
                    temp_path.unlink()
                except OSError:
                    pass

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
                    f"[semantic-teacher-cache] failed to quarantine bad cache {cache_path}: {move_exc}; "
                    "continuing with in-place overwrite"
                ),
                RuntimeWarning,
            )
            return None

    def _is_recoverable_cache_error(self, exc: Exception) -> bool:
        message = str(exc).lower()
        exc_name = type(exc).__name__.lower()
        keywords = (
            "zip archive",
            "torch.jit.load",
            "weights_only",
            "pickle",
            "unpick",
            "missing required cache",
            "unsupported cache payload",
        )
        if any(key in message for key in keywords):
            return True
        if "unpick" in exc_name:
            return True
        return False

    def _load_from_cache(self, cache_path: Path) -> SemanticSummary:
        try:
            payload = torch.load(cache_path, map_location="cpu", weights_only=False)
        except TypeError:
            payload = torch.load(cache_path, map_location="cpu")

        if not isinstance(payload, dict):
            raise ValueError(f"unsupported cache payload type: {type(payload)}")
        if "class_scores" not in payload or "text_embeddings" not in payload:
            raise ValueError("missing required cache fields: class_scores/text_embeddings")

        metadata_raw = payload.get("metadata", {})
        metadata = dict(metadata_raw) if isinstance(metadata_raw, dict) else {}
        metadata["source"] = "cache"
        metadata["cache_path"] = str(cache_path)
        return SemanticSummary(
            class_scores=payload["class_scores"].float(),
            text_embeddings=payload["text_embeddings"].float(),
            metadata=metadata,
        )
