from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from stwm.modules.ostf_v33_integrated_semantic_identity_world_model import IntegratedSemanticIdentityWorldModelV331


class VisualSemanticIdentityWorldModelV332(IntegratedSemanticIdentityWorldModelV331):
    """Frozen V30 trajectory model with observed visual semantic context and future prototype heads."""

    def __init__(
        self,
        v30_checkpoint_path: str | Path,
        *,
        teacher_embedding_dim: int = 512,
        identity_dim: int = 64,
        use_observed_instance_context: bool = False,
    ) -> None:
        super().__init__(
            v30_checkpoint_path,
            identity_dim=identity_dim,
            use_observed_instance_context=use_observed_instance_context,
        )
        hidden = int(self.v30.cfg.hidden_dim)
        point_dim = int(self.v30.cfg.point_dim)
        self.teacher_embedding_dim = int(teacher_embedding_dim)
        self.visual_encoder = nn.Sequential(
            nn.LayerNorm(self.teacher_embedding_dim),
            nn.Linear(self.teacher_embedding_dim, hidden),
            nn.GELU(),
        )
        context_dim = hidden * 3 + point_dim + 2 + (16 if self.use_observed_instance_context else 0)
        self.identity_trunk = nn.Sequential(
            nn.LayerNorm(context_dim),
            nn.Linear(context_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        self.same_instance_head = nn.Linear(hidden, 1)
        self.identity_embedding_head = nn.Linear(hidden, identity_dim)
        self.identity_uncertainty_head = nn.Linear(hidden, 1)
        self.visibility_delta_head = nn.Linear(hidden, 1)
        self.semantic_embedding_head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, self.teacher_embedding_dim))
        self.semantic_proto_uncertainty_head = nn.Linear(hidden, 1)

    def _visual_context(self, obs_teacher_embedding: torch.Tensor, obs_teacher_available_mask: torch.Tensor) -> torch.Tensor:
        emb = torch.nan_to_num(obs_teacher_embedding.float(), nan=0.0, posinf=0.0, neginf=0.0)
        mask = obs_teacher_available_mask.float()
        denom = mask.sum(dim=-1, keepdim=True).clamp_min(1.0)
        pooled = (emb * mask[..., None]).sum(dim=2) / denom
        return self.visual_encoder(pooled)

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
            v30_out = self.v30(obs_points=obs_points, obs_vis=obs_vis, obs_conf=obs_conf, semantic_id=semantic_id)
            features, _ = self._v30_features(obs_points=obs_points, obs_vis=obs_vis, obs_conf=obs_conf, semantic_id=semantic_id)
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
        sem = self.semantic_embedding_head(hidden)
        sem = sem / sem.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        return {
            "point_hypotheses": v30_out["point_hypotheses"].detach(),
            "point_pred": point_pred,
            "top1_point_pred": v30_out["top1_point_pred"].detach(),
            "visibility_logits": v30_out["visibility_logits"].detach() + self.visibility_delta_head(hidden).squeeze(-1),
            "frozen_v30_visibility_logits": v30_out["visibility_logits"].detach(),
            "same_instance_logits": self.same_instance_head(hidden).squeeze(-1),
            "identity_embedding": self.identity_embedding_head(hidden),
            "identity_uncertainty": self.identity_uncertainty_head(hidden).squeeze(-1),
            "semantic_embedding_pred": sem,
            "semantic_proto_uncertainty": self.semantic_proto_uncertainty_head(hidden).squeeze(-1),
            "semantic_logits": v30_out.get("semantic_logits", torch.empty((*point_pred.shape[:3], 0), device=point_pred.device)).detach(),
        }
