from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from stwm.modules.ostf_v33_2_visual_semantic_identity_world_model import VisualSemanticIdentityWorldModelV332


class StructuredSemanticIdentityWorldModelV333(VisualSemanticIdentityWorldModelV332):
    """Frozen V30 trajectory model with identity retrieval and structured semantic prototype logits."""

    def __init__(
        self,
        v30_checkpoint_path: str | Path,
        *,
        prototype_centers: torch.Tensor,
        teacher_embedding_dim: int = 512,
        identity_dim: int = 64,
        use_observed_instance_context: bool = False,
    ) -> None:
        super().__init__(
            v30_checkpoint_path,
            teacher_embedding_dim=teacher_embedding_dim,
            identity_dim=identity_dim,
            use_observed_instance_context=use_observed_instance_context,
        )
        centers = prototype_centers.float()
        centers = centers / centers.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        self.register_buffer("prototype_centers", centers)
        hidden = int(self.v30.cfg.hidden_dim)
        self.future_semantic_proto_head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, int(centers.shape[0])),
        )

    @property
    def prototype_vocab_size(self) -> int:
        return int(self.prototype_centers.shape[0])

    def forward(
        self,
        *,
        obs_points: torch.Tensor,
        obs_vis: torch.Tensor,
        obs_conf: torch.Tensor,
        obs_teacher_embedding: torch.Tensor,
        obs_teacher_available_mask: torch.Tensor,
        semantic_id: torch.Tensor | None = None,
        point_to_instance_id: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        out = super().forward(
            obs_points=obs_points,
            obs_vis=obs_vis,
            obs_conf=obs_conf,
            obs_teacher_embedding=obs_teacher_embedding,
            obs_teacher_available_mask=obs_teacher_available_mask,
            semantic_id=semantic_id,
            point_to_instance_id=point_to_instance_id,
        )
        # Recompute the hidden trunk deterministically so semantic proto logits are structured outputs.
        with torch.no_grad():
            features, _ = self._v30_features(obs_points=obs_points, obs_vis=obs_vis, obs_conf=obs_conf, semantic_id=semantic_id)
            v30_out = self.v30(obs_points=obs_points, obs_vis=obs_vis, obs_conf=obs_conf, semantic_id=semantic_id)
        point_token = features["point_token"].detach()
        step_hidden = features["step_hidden"].detach()
        point_pred = v30_out["point_pred"].detach()
        rel_pred = ((point_pred - features["last_visible"][:, :, None, :]) / features["spread"].squeeze(2)[:, :, None, :]).clamp(-16.0, 16.0)
        b, m, h, _ = point_pred.shape
        visual_token = self._visual_context(obs_teacher_embedding, obs_teacher_available_mask)
        chunks = [
            point_token[:, :, None, :].expand(-1, -1, h, -1),
            step_hidden[:, None, :, :].expand(-1, m, -1, -1),
            visual_token[:, :, None, :].expand(-1, -1, h, -1),
            rel_pred,
            v30_out["visibility_logits"].detach().sigmoid()[:, :, :, None],
            obs_vis.float()[:, :, -1:, None].expand(-1, -1, h, -1),
        ]
        if self.use_observed_instance_context:
            if point_to_instance_id is None:
                point_to_instance_id = torch.full((b, m), -1, device=point_pred.device, dtype=torch.long)
            chunks.append(self.instance_embed(point_to_instance_id.clamp_min(0) % self.instance_embed.num_embeddings)[:, :, None, :].expand(-1, -1, h, -1))
        hidden = self.identity_trunk(torch.cat(chunks, dim=-1))
        proto_logits = self.future_semantic_proto_head(hidden)
        out["future_semantic_proto_logits"] = proto_logits
        out["semantic_proto_uncertainty"] = self.semantic_proto_uncertainty_head(hidden).squeeze(-1)
        # Keep embedding prediction only as an auxiliary diagnostic, not the primary semantic field.
        out["semantic_embedding_pred_aux"] = out.pop("semantic_embedding_pred")
        return out
