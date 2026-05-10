from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

from stwm.modules.ostf_v33_3_structured_semantic_identity_world_model import StructuredSemanticIdentityWorldModelV333


class IdentityBeliefWorldModelV337(StructuredSemanticIdentityWorldModelV333):
    """Frozen V30 trajectory model with calibrated identity-belief logits.

    The trajectory field remains the frozen V30 M128 backbone. V33.7 only adds
    observed identity anchors, future embedding similarity logits, and fused
    same-instance belief calibration.
    """

    def __init__(
        self,
        v30_checkpoint_path: str | Path,
        *,
        prototype_centers: torch.Tensor,
        teacher_embedding_dim: int = 512,
        identity_dim: int = 64,
        use_observed_instance_context: bool = False,
        disable_embedding_similarity_logits: bool = False,
        disable_fused_logits: bool = False,
    ) -> None:
        super().__init__(
            v30_checkpoint_path,
            prototype_centers=prototype_centers,
            teacher_embedding_dim=teacher_embedding_dim,
            identity_dim=identity_dim,
            use_observed_instance_context=use_observed_instance_context,
        )
        hidden = int(self.v30.cfg.hidden_dim)
        self.observed_identity_anchor_head = nn.Sequential(
            nn.LayerNorm(hidden * 2 + 2),
            nn.Linear(hidden * 2 + 2, hidden),
            nn.GELU(),
            nn.Linear(hidden, identity_dim),
        )
        self.embedding_logit_scale = nn.Parameter(torch.tensor(6.0))
        self.embedding_logit_bias = nn.Parameter(torch.tensor(0.0))
        self.fusion_alpha = nn.Parameter(torch.tensor(1.0))
        self.fusion_beta = nn.Parameter(torch.tensor(1.0))
        self.fusion_bias = nn.Parameter(torch.tensor(0.0))
        self.disable_embedding_similarity_logits = bool(disable_embedding_similarity_logits)
        self.disable_fused_logits = bool(disable_fused_logits)

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
        same_head = self.same_instance_head(hidden).squeeze(-1)
        identity_embedding = self.identity_embedding_head(hidden)

        obs_anchor_input = torch.cat(
            [
                point_token,
                visual_token,
                obs_vis.float().mean(dim=-1, keepdim=True),
                obs_conf.float().mean(dim=-1, keepdim=True),
            ],
            dim=-1,
        )
        observed_anchor = self.observed_identity_anchor_head(obs_anchor_input)
        sim = F.cosine_similarity(identity_embedding, observed_anchor[:, :, None, :], dim=-1)
        embedding_logits = sim * self.embedding_logit_scale.clamp(0.1, 20.0) + self.embedding_logit_bias
        if self.disable_embedding_similarity_logits:
            embedding_logits = torch.zeros_like(embedding_logits)
        fused = self.fusion_alpha * same_head + self.fusion_beta * embedding_logits + self.fusion_bias
        if self.disable_fused_logits:
            fused = same_head

        proto_logits = self.future_semantic_proto_head(hidden)
        visibility_logits = v30_out["visibility_logits"].detach() + self.visibility_delta_head(hidden).squeeze(-1)
        return {
            "point_hypotheses": v30_out["point_hypotheses"].detach(),
            "point_pred": point_pred,
            "top1_point_pred": v30_out["top1_point_pred"].detach(),
            "visibility_logits": visibility_logits,
            "frozen_v30_visibility_logits": v30_out["visibility_logits"].detach(),
            "same_instance_logits": same_head,
            "embedding_similarity_logits": embedding_logits,
            "fused_same_instance_logits": fused,
            "identity_embedding": identity_embedding,
            "observed_identity_anchor": observed_anchor,
            "identity_uncertainty": self.identity_uncertainty_head(hidden).squeeze(-1),
            "future_semantic_proto_logits": proto_logits,
            "semantic_proto_uncertainty": self.semantic_proto_uncertainty_head(hidden).squeeze(-1),
            "semantic_logits": v30_out.get("semantic_logits", torch.empty((*same_head.shape, 0), device=point_pred.device)).detach(),
        }
