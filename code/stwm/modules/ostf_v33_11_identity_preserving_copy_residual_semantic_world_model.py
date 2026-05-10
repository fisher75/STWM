from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from stwm.modules.ostf_v33_10_copy_residual_semantic_world_model import CopyResidualSemanticWorldModelV3310


class IdentityPreservingCopyResidualSemanticWorldModelV3311(CopyResidualSemanticWorldModelV3310):
    """Copy-residual semantic head with frozen V30 and frozen/distilled identity path.

    V33.11 intentionally stops sharing semantic gradients through the identity
    trunk by default. The hidden field used by semantic heads is produced by the
    V33.9 identity path, but detached before semantic optimization unless the
    explicit no-identity-freeze ablation is requested.
    """

    def __init__(
        self,
        v30_checkpoint_path: str | Path,
        *,
        prototype_centers: torch.Tensor,
        teacher_embedding_dim: int = 512,
        identity_dim: int = 64,
        identity_teacher_checkpoint: str | Path | None = None,
        copy_logit_strength: float = 9.0,
        freeze_identity_path: bool = True,
        no_stable_margin: bool = False,
        no_gate_focal: bool = False,
    ) -> None:
        super().__init__(
            v30_checkpoint_path,
            prototype_centers=prototype_centers,
            teacher_embedding_dim=teacher_embedding_dim,
            identity_dim=identity_dim,
            copy_logit_strength=copy_logit_strength,
            no_copy_prior=False,
            no_change_gate=False,
        )
        self.freeze_identity_path = bool(freeze_identity_path)
        self.no_stable_margin = bool(no_stable_margin)
        self.no_gate_focal = bool(no_gate_focal)
        self.identity_teacher_checkpoint_loaded = False
        self.identity_teacher_missing_keys: list[str] = []
        self.identity_teacher_unexpected_keys: list[str] = []
        if identity_teacher_checkpoint is not None and Path(identity_teacher_checkpoint).exists():
            ck = torch.load(identity_teacher_checkpoint, map_location="cpu")
            state = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
            incompatible = self.load_state_dict(state, strict=False)
            allowed_missing = {"semantic_change_head", "semantic_uncertainty_head"}
            self.identity_teacher_missing_keys = list(incompatible.missing_keys)
            self.identity_teacher_unexpected_keys = list(incompatible.unexpected_keys)
            self.identity_teacher_checkpoint_loaded = True
        if self.freeze_identity_path:
            for module in [
                self.visual_encoder,
                self.identity_trunk,
                self.same_instance_head,
                self.identity_embedding_head,
                self.identity_uncertainty_head,
                self.visibility_delta_head,
                self.semantic_embedding_head,
                self.instance_embed,
            ]:
                for p in module.parameters():
                    p.requires_grad_(False)
        # V30 is always frozen by parent class; semantic heads remain trainable.
        for p in self.v30.parameters():
            p.requires_grad_(False)

    @property
    def identity_path_frozen_or_distilled(self) -> bool:
        return self.freeze_identity_path or self.identity_teacher_checkpoint_loaded

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
        identity_hidden = hidden
        semantic_hidden = hidden.detach() if self.freeze_identity_path else hidden
        residual_logits = self.future_semantic_proto_head(semantic_hidden)
        k = int(residual_logits.shape[-1])
        if copy_semantic_prototype_id is None:
            if last_observed_semantic_prototype_id is None:
                last_observed_semantic_prototype_id = torch.full((b, m), -1, device=obs_points.device, dtype=torch.long)
            copy_semantic_prototype_id = last_observed_semantic_prototype_id[:, :, None].expand(b, m, h)
        copy_logits = self._copy_logits(copy_semantic_prototype_id.long(), k)
        change_logits = self.semantic_change_head(semantic_hidden).squeeze(-1)
        gate = torch.sigmoid(change_logits)
        copy_probs = F.softmax(copy_logits, dim=-1)
        residual_probs = F.softmax(residual_logits, dim=-1)
        final_probs = (1.0 - gate[..., None]) * copy_probs + gate[..., None] * residual_probs
        final_logits = torch.log(final_probs.clamp_min(1e-8))
        return {
            "point_hypotheses": v30_out["point_hypotheses"].detach(),
            "point_pred": point_pred,
            "top1_point_pred": v30_out["top1_point_pred"].detach(),
            "visibility_logits": v30_out["visibility_logits"].detach() + self.visibility_delta_head(identity_hidden).squeeze(-1),
            "frozen_v30_visibility_logits": v30_out["visibility_logits"].detach(),
            "same_instance_logits": self.same_instance_head(identity_hidden).squeeze(-1),
            "identity_embedding": self.identity_embedding_head(identity_hidden),
            "identity_uncertainty": self.identity_uncertainty_head(identity_hidden).squeeze(-1),
            "copy_prior_semantic_logits": copy_logits,
            "semantic_residual_logits": residual_logits,
            "semantic_change_logits": change_logits,
            "semantic_change_gate": gate,
            "final_semantic_proto_logits": final_logits,
            "future_semantic_proto_logits": final_logits,
            "semantic_uncertainty": self.semantic_uncertainty_head(semantic_hidden).squeeze(-1),
            "semantic_proto_uncertainty": self.semantic_proto_uncertainty_head(semantic_hidden).squeeze(-1),
            "identity_teacher_same_instance_logits": self.same_instance_head(identity_hidden).squeeze(-1).detach(),
            "semantic_logits": v30_out.get("semantic_logits", torch.empty((*point_pred.shape[:3], 0), device=point_pred.device)).detach(),
        }

