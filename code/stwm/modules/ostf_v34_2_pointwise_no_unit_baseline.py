from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

from stwm.modules.ostf_v33_2_visual_semantic_identity_world_model import VisualSemanticIdentityWorldModelV332


class PointwiseNoUnitBaselineV342(VisualSemanticIdentityWorldModelV332):
    def __init__(self, v30_checkpoint_path: str | Path, *, teacher_embedding_dim: int = 768, identity_dim: int = 64) -> None:
        super().__init__(v30_checkpoint_path, teacher_embedding_dim=teacher_embedding_dim, identity_dim=identity_dim, use_observed_instance_context=False)
        for p in self.v30.parameters():
            p.requires_grad_(False)

    @property
    def v30_backbone_frozen(self) -> bool:
        return all(not p.requires_grad for p in self.v30.parameters())

    def forward(
        self,
        *,
        obs_points: torch.Tensor,
        obs_vis: torch.Tensor,
        obs_conf: torch.Tensor,
        obs_semantic_measurements: torch.Tensor,
        obs_semantic_measurement_mask: torch.Tensor,
        semantic_id: torch.Tensor | None = None,
        **_: object,
    ) -> dict[str, torch.Tensor]:
        out = super().forward(
            obs_points=obs_points,
            obs_vis=obs_vis,
            obs_conf=obs_conf,
            obs_teacher_embedding=obs_semantic_measurements,
            obs_teacher_available_mask=obs_semantic_measurement_mask,
            semantic_id=semantic_id,
        )
        out["future_semantic_belief"] = F.normalize(out.pop("semantic_embedding_pred"), dim=-1)
        out["future_identity_belief"] = out.pop("same_instance_logits")
        out["semantic_uncertainty"] = F.softplus(out.pop("semantic_proto_uncertainty"))
        out["point_to_unit_assignment"] = torch.empty((*out["point_pred"].shape[:2], 0), device=out["point_pred"].device)
        out["teacher_as_method"] = False
        out["trace_conditioned_semantic_units_active"] = False
        out["outputs_future_trace_field"] = True
        out["outputs_future_semantic_field"] = True
        return out
