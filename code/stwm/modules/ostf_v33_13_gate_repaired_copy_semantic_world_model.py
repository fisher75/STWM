from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from stwm.modules.ostf_v33_11_identity_preserving_copy_residual_semantic_world_model import (
    IdentityPreservingCopyResidualSemanticWorldModelV3311,
)


class GateRepairedCopySemanticWorldModelV3313(IdentityPreservingCopyResidualSemanticWorldModelV3311):
    """Copy-residual semantic field head with explicit gate/logit semantics.

    The V33.12 implementation exposed a field named `semantic_change_gate_raw`
    that was already sigmoid-normalized and then reused it as if it were logits.
    V33.13 fixes the contract:
    - semantic_change_logits_raw: pre-sigmoid logits
    - semantic_change_prob: sigmoid(logits)
    - semantic_effective_gate: deterministic copy/residual mixing gate
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
            self.load_state_dict(compatible, strict=False)
            self.identity_teacher_checkpoint_loaded = True
        self.gate_threshold = float(gate_threshold)

    def effective_gate_from_prob(self, prob: torch.Tensor) -> torch.Tensor:
        denom = max(1.0 - self.gate_threshold, 1e-6)
        return ((prob - self.gate_threshold) / denom).clamp(0.0, 1.0)

    def forward(self, **kwargs: torch.Tensor) -> dict[str, torch.Tensor]:
        out = super().forward(**kwargs)
        logits_raw = out["semantic_change_logits"]
        prob = torch.sigmoid(logits_raw)
        effective_gate = self.effective_gate_from_prob(prob)
        copy_probs = F.softmax(out["copy_prior_semantic_logits"], dim=-1)
        residual_probs = F.softmax(out["semantic_residual_logits"], dim=-1)
        final_probs = (1.0 - effective_gate[..., None]) * copy_probs + effective_gate[..., None] * residual_probs
        out["semantic_change_logits_raw"] = logits_raw
        out["semantic_change_prob"] = prob
        out["semantic_effective_gate"] = effective_gate
        # Backward-compatible aliases used by older eval code.
        out["semantic_change_logits"] = logits_raw
        out["semantic_change_gate"] = effective_gate
        out["final_semantic_proto_logits"] = torch.log(final_probs.clamp_min(1e-8))
        out["future_semantic_proto_logits"] = out["final_semantic_proto_logits"]
        return out
