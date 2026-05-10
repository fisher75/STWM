from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

from stwm.modules.ostf_v33_3_structured_semantic_identity_world_model import StructuredSemanticIdentityWorldModelV333


class CopyResidualSemanticWorldModelV3310(StructuredSemanticIdentityWorldModelV333):
    """Frozen V30 trajectory/identity head with copy-aware residual semantic prototype logits."""

    def __init__(
        self,
        v30_checkpoint_path: str | Path,
        *,
        prototype_centers: torch.Tensor,
        teacher_embedding_dim: int = 512,
        identity_dim: int = 64,
        copy_logit_strength: float = 7.0,
        no_copy_prior: bool = False,
        no_change_gate: bool = False,
    ) -> None:
        super().__init__(
            v30_checkpoint_path,
            prototype_centers=prototype_centers,
            teacher_embedding_dim=teacher_embedding_dim,
            identity_dim=identity_dim,
            use_observed_instance_context=False,
        )
        hidden = int(self.v30.cfg.hidden_dim)
        self.semantic_change_head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 1))
        self.semantic_uncertainty_head = nn.Linear(hidden, 1)
        self.copy_logit_strength = float(copy_logit_strength)
        self.no_copy_prior = bool(no_copy_prior)
        self.no_change_gate = bool(no_change_gate)

    def _copy_logits(self, copy_ids: torch.Tensor, k: int) -> torch.Tensor:
        valid = copy_ids >= 0
        ids = copy_ids.clamp_min(0).clamp_max(k - 1)
        logits = torch.full((*copy_ids.shape, k), -self.copy_logit_strength, device=copy_ids.device, dtype=torch.float32)
        logits.scatter_(-1, ids[..., None], self.copy_logit_strength)
        uniform = torch.zeros_like(logits)
        return torch.where(valid[..., None], logits, uniform)

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
        copy_semantic_prototype_id: torch.Tensor | None = None,
        last_observed_semantic_prototype_id: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        # Replicate the frozen trunk construction so the change gate uses the same future hidden state.
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
        hidden = self.identity_trunk(torch.cat(chunks, dim=-1))
        residual_logits = self.future_semantic_proto_head(hidden)
        k = int(residual_logits.shape[-1])
        if copy_semantic_prototype_id is None:
            if last_observed_semantic_prototype_id is None:
                last_observed_semantic_prototype_id = torch.full((b, m), -1, device=obs_points.device, dtype=torch.long)
            copy_semantic_prototype_id = last_observed_semantic_prototype_id[:, :, None].expand(b, m, h)
        copy_logits = self._copy_logits(copy_semantic_prototype_id.long(), k)
        change_logits = self.semantic_change_head(hidden).squeeze(-1)
        if self.no_copy_prior:
            final_logits = residual_logits
            gate = torch.ones_like(change_logits)
            copy_logits = torch.zeros_like(copy_logits)
        else:
            if self.no_change_gate:
                gate = torch.ones_like(change_logits) * 0.5
            else:
                gate = torch.sigmoid(change_logits)
            copy_probs = F.softmax(copy_logits, dim=-1)
            residual_probs = F.softmax(residual_logits, dim=-1)
            final_probs = (1.0 - gate[..., None]) * copy_probs + gate[..., None] * residual_probs
            final_logits = torch.log(final_probs.clamp_min(1e-8))
        visibility_logits = v30_out["visibility_logits"].detach() + self.visibility_delta_head(hidden).squeeze(-1)
        return {
            "point_hypotheses": v30_out["point_hypotheses"].detach(),
            "point_pred": point_pred,
            "top1_point_pred": v30_out["top1_point_pred"].detach(),
            "visibility_logits": visibility_logits,
            "frozen_v30_visibility_logits": v30_out["visibility_logits"].detach(),
            "same_instance_logits": self.same_instance_head(hidden).squeeze(-1),
            "identity_embedding": self.identity_embedding_head(hidden),
            "identity_uncertainty": self.identity_uncertainty_head(hidden).squeeze(-1),
            "copy_prior_semantic_logits": copy_logits,
            "semantic_residual_logits": residual_logits,
            "semantic_change_logits": change_logits,
            "semantic_change_gate": gate,
            "final_semantic_proto_logits": final_logits,
            "future_semantic_proto_logits": final_logits,
            "semantic_uncertainty": self.semantic_uncertainty_head(hidden).squeeze(-1),
            "semantic_proto_uncertainty": self.semantic_proto_uncertainty_head(hidden).squeeze(-1),
            "semantic_logits": v30_out.get("semantic_logits", torch.empty((*point_pred.shape[:3], 0), device=point_pred.device)).detach(),
        }
