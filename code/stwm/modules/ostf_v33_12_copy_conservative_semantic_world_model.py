from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from stwm.modules.ostf_v33_11_identity_preserving_copy_residual_semantic_world_model import (
    IdentityPreservingCopyResidualSemanticWorldModelV3311,
)


class CopyConservativeSemanticWorldModelV3312(IdentityPreservingCopyResidualSemanticWorldModelV3311):
    """Identity-preserving copy-residual semantic head with conservative updates.

    V33.12 keeps V30 frozen and keeps the V33.11 identity path frozen/distilled.
    The semantic output defaults to the observed-copy prior and only lets the
    residual distribution override copy when the learned change gate clears a
    calibrated threshold.
    """

    def __init__(
        self,
        v30_checkpoint_path: str | Path,
        *,
        prototype_centers: torch.Tensor,
        teacher_embedding_dim: int = 512,
        identity_dim: int = 64,
        identity_teacher_checkpoint: str | Path | None = None,
        copy_logit_strength: float = 10.0,
        gate_threshold: float = 0.10,
        residual_update_budget: float = 0.35,
        freeze_identity_path: bool = True,
    ) -> None:
        super().__init__(
            v30_checkpoint_path,
            prototype_centers=prototype_centers,
            teacher_embedding_dim=teacher_embedding_dim,
            identity_dim=identity_dim,
            identity_teacher_checkpoint=None,
            copy_logit_strength=copy_logit_strength,
            freeze_identity_path=freeze_identity_path,
            no_stable_margin=False,
            no_gate_focal=False,
        )
        self.identity_teacher_checkpoint_loaded = False
        self.identity_teacher_missing_keys = []
        self.identity_teacher_unexpected_keys = []
        self.identity_teacher_shape_skipped_keys: list[str] = []
        if identity_teacher_checkpoint is not None and Path(identity_teacher_checkpoint).exists():
            ck: Any = torch.load(identity_teacher_checkpoint, map_location="cpu")
            state = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
            current = self.state_dict()
            compatible = {}
            for key, value in state.items():
                if key in current and tuple(current[key].shape) == tuple(value.shape):
                    compatible[key] = value
                else:
                    self.identity_teacher_shape_skipped_keys.append(key)
            incompatible = self.load_state_dict(compatible, strict=False)
            self.identity_teacher_missing_keys = list(incompatible.missing_keys)
            self.identity_teacher_unexpected_keys = list(incompatible.unexpected_keys)
            self.identity_teacher_checkpoint_loaded = True
        self.gate_threshold = float(gate_threshold)
        self.residual_update_budget = float(residual_update_budget)

    def forward(self, **kwargs: torch.Tensor) -> dict[str, torch.Tensor]:
        out = super().forward(**kwargs)
        raw_gate = out["semantic_change_gate"]
        effective_gate = ((raw_gate - self.gate_threshold) / max(1.0 - self.gate_threshold, 1e-6)).clamp(0.0, 1.0)
        if self.training and self.residual_update_budget > 0:
            # Keep the differentiable gate but bound updates around the requested
            # budget so stable copy states are the default behavior.
            effective_gate = torch.minimum(effective_gate, torch.full_like(effective_gate, self.residual_update_budget))
        copy_probs = F.softmax(out["copy_prior_semantic_logits"], dim=-1)
        residual_probs = F.softmax(out["semantic_residual_logits"], dim=-1)
        final_probs = (1.0 - effective_gate[..., None]) * copy_probs + effective_gate[..., None] * residual_probs
        out["semantic_change_gate_raw"] = raw_gate
        out["semantic_change_gate"] = effective_gate
        out["final_semantic_proto_logits"] = torch.log(final_probs.clamp_min(1e-8))
        out["future_semantic_proto_logits"] = out["final_semantic_proto_logits"]
        return out
