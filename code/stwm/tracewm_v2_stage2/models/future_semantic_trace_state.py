from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class FutureSemanticTraceState:
    """Tensor contract for STWM future semantic trajectory field outputs.

    Required tensors use [B, H, K, ...], where B is batch size, H future
    horizon, and K trace/entity slots. Optional multi-hypothesis tensors use
    [B, M, H, K, ...].
    """

    future_trace_coord: torch.Tensor
    future_visibility_logit: torch.Tensor
    future_semantic_embedding: torch.Tensor
    future_identity_embedding: torch.Tensor
    future_uncertainty: torch.Tensor
    future_semantic_logits: torch.Tensor | None = None
    future_extent_box: torch.Tensor | None = None
    future_hypothesis_logits: torch.Tensor | None = None
    future_hypothesis_trace_coord: torch.Tensor | None = None

    def validate(self, *, strict: bool = True) -> dict[str, Any]:
        errors: list[str] = []
        coord_shape = tuple(self.future_trace_coord.shape)
        if len(coord_shape) != 4 or coord_shape[-1] not in (2, 3):
            errors.append(f"future_trace_coord must be [B,H,K,2 or 3], got {coord_shape}")
            bsz = horizon = slots = None
        else:
            bsz, horizon, slots, _ = coord_shape

        def _expect(name: str, tensor: torch.Tensor, suffix: tuple[int | None, ...]) -> None:
            if bsz is None:
                return
            expected_prefix = (bsz, horizon, slots)
            actual = tuple(tensor.shape)
            if actual[:3] != expected_prefix:
                errors.append(f"{name} must start with {expected_prefix}, got {actual}")
            if suffix and len(actual) != 3 + len(suffix):
                errors.append(f"{name} rank mismatch, got {actual}")
                return
            for idx, value in enumerate(suffix):
                if value is not None and actual[3 + idx] != value:
                    errors.append(f"{name} suffix mismatch at {idx}: expected {value}, got {actual}")

        _expect("future_visibility_logit", self.future_visibility_logit, ())
        _expect("future_semantic_embedding", self.future_semantic_embedding, (None,))
        _expect("future_identity_embedding", self.future_identity_embedding, (None,))
        _expect("future_uncertainty", self.future_uncertainty, ())
        if self.future_semantic_logits is not None:
            _expect("future_semantic_logits", self.future_semantic_logits, (None,))
        if self.future_extent_box is not None:
            _expect("future_extent_box", self.future_extent_box, (4,))
        if self.future_hypothesis_logits is not None:
            actual = tuple(self.future_hypothesis_logits.shape)
            if bsz is not None and (len(actual) != 2 or actual[0] != bsz):
                errors.append(f"future_hypothesis_logits must be [B,M], got {actual}")
        if self.future_hypothesis_trace_coord is not None:
            actual = tuple(self.future_hypothesis_trace_coord.shape)
            if bsz is not None and (len(actual) != 5 or actual[0] != bsz or actual[2:4] != (horizon, slots) or actual[-1] not in (2, 3)):
                errors.append(f"future_hypothesis_trace_coord must be [B,M,H,K,2 or 3], got {actual}")

        ok = not errors
        if strict and not ok:
            raise ValueError("; ".join(errors))
        return {"valid": ok, "errors": errors, "shapes": self.shape_dict()}

    def shape_dict(self) -> dict[str, tuple[int, ...] | None]:
        return {
            "future_trace_coord": tuple(self.future_trace_coord.shape),
            "future_visibility_logit": tuple(self.future_visibility_logit.shape),
            "future_semantic_embedding": tuple(self.future_semantic_embedding.shape),
            "future_semantic_logits": tuple(self.future_semantic_logits.shape) if self.future_semantic_logits is not None else None,
            "future_identity_embedding": tuple(self.future_identity_embedding.shape),
            "future_extent_box": tuple(self.future_extent_box.shape) if self.future_extent_box is not None else None,
            "future_uncertainty": tuple(self.future_uncertainty.shape),
            "future_hypothesis_logits": tuple(self.future_hypothesis_logits.shape) if self.future_hypothesis_logits is not None else None,
            "future_hypothesis_trace_coord": tuple(self.future_hypothesis_trace_coord.shape) if self.future_hypothesis_trace_coord is not None else None,
        }

    def as_tensor_dict(self) -> dict[str, torch.Tensor]:
        out = {
            "future_trace_coord": self.future_trace_coord,
            "future_visibility_logit": self.future_visibility_logit,
            "future_semantic_embedding": self.future_semantic_embedding,
            "future_identity_embedding": self.future_identity_embedding,
            "future_uncertainty": self.future_uncertainty,
        }
        if self.future_semantic_logits is not None:
            out["future_semantic_logits"] = self.future_semantic_logits
        if self.future_extent_box is not None:
            out["future_extent_box"] = self.future_extent_box
        if self.future_hypothesis_logits is not None:
            out["future_hypothesis_logits"] = self.future_hypothesis_logits
        if self.future_hypothesis_trace_coord is not None:
            out["future_hypothesis_trace_coord"] = self.future_hypothesis_trace_coord
        return out
