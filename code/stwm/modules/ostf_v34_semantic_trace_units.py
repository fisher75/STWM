from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

from stwm.modules.ostf_v33_2_visual_semantic_identity_world_model import VisualSemanticIdentityWorldModelV332


class SemanticTraceUnitTokenizer(nn.Module):
    def __init__(self, hidden_dim: int, semantic_dim: int, units: int = 16) -> None:
        super().__init__()
        self.units = int(units)
        self.assign = nn.Sequential(nn.LayerNorm(hidden_dim + semantic_dim), nn.Linear(hidden_dim + semantic_dim, units))
        self.sem_proj = nn.Sequential(nn.LayerNorm(semantic_dim), nn.Linear(semantic_dim, hidden_dim), nn.GELU())

    def forward(self, point_token: torch.Tensor, obs_sem: torch.Tensor, obs_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mask = obs_mask.float()
        pooled = (torch.nan_to_num(obs_sem.float()) * mask[..., None]).sum(dim=2) / mask.sum(dim=2, keepdim=True).clamp_min(1.0)
        logits = self.assign(torch.cat([point_token, pooled], dim=-1))
        assign = torch.softmax(logits, dim=-1)
        sem_token = self.sem_proj(pooled)
        denom = assign.sum(dim=1).clamp_min(1e-6)
        unit_sem = torch.einsum("bmu,bmd->bud", assign, sem_token) / denom[..., None]
        return assign, unit_sem


class FactorizedTraceSemanticState(nn.Module):
    def __init__(self, hidden_dim: int, units: int, identity_dim: int) -> None:
        super().__init__()
        self.z_dyn = nn.Linear(hidden_dim, hidden_dim)
        self.z_sem = nn.Linear(hidden_dim, hidden_dim)
        self.identity_key = nn.Linear(hidden_dim, identity_dim)
        self.confidence = nn.Linear(hidden_dim, 1)

    def forward(self, unit_sem: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "z_dyn": self.z_dyn(unit_sem),
            "z_sem": self.z_sem(unit_sem),
            "identity_key": self.identity_key(unit_sem),
            "unit_confidence": torch.sigmoid(self.confidence(unit_sem)).squeeze(-1),
        }


class TraceSemanticHandshake(nn.Module):
    def __init__(self, hidden_dim: int, semantic_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(hidden_dim * 3 + 3), nn.Linear(hidden_dim * 3 + 3, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim), nn.GELU())

    def forward(self, point_token: torch.Tensor, step_hidden: torch.Tensor, unit_point_sem: torch.Tensor, rel_pred: torch.Tensor, vis_prob: torch.Tensor) -> torch.Tensor:
        b, m, h, _ = rel_pred.shape
        chunks = [
            point_token[:, :, None, :].expand(-1, -1, h, -1),
            step_hidden[:, None, :, :].expand(-1, m, -1, -1),
            unit_point_sem[:, :, None, :].expand(-1, -1, h, -1),
            rel_pred,
            vis_prob[:, :, :, None],
        ]
        return self.net(torch.cat(chunks, dim=-1))


class SemanticTraceUnitRollout(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim))

    def forward(self, field_hidden: torch.Tensor) -> torch.Tensor:
        return self.net(field_hidden)


class TraceConditionedBeliefReadout(nn.Module):
    def __init__(self, hidden_dim: int, semantic_dim: int, identity_dim: int) -> None:
        super().__init__()
        self.semantic = nn.Linear(hidden_dim, semantic_dim)
        self.identity = nn.Linear(hidden_dim, 1)
        self.identity_embed = nn.Linear(hidden_dim, identity_dim)
        self.uncertainty = nn.Linear(hidden_dim, 1)

    def forward(self, hidden: torch.Tensor) -> dict[str, torch.Tensor]:
        sem = self.semantic(hidden)
        sem = sem / sem.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        emb = self.identity_embed(hidden)
        emb = emb / emb.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        return {
            "future_semantic_belief": sem,
            "future_identity_belief": self.identity(hidden).squeeze(-1),
            "identity_embedding": emb,
            "semantic_uncertainty": F.softplus(self.uncertainty(hidden)).squeeze(-1),
        }


class SemanticTraceUnitsWorldModelV34(VisualSemanticIdentityWorldModelV332):
    def __init__(
        self,
        v30_checkpoint_path: str | Path,
        *,
        teacher_embedding_dim: int = 768,
        identity_dim: int = 64,
        units: int = 16,
    ) -> None:
        super().__init__(v30_checkpoint_path, teacher_embedding_dim=teacher_embedding_dim, identity_dim=identity_dim, use_observed_instance_context=False)
        hidden = int(self.v30.cfg.hidden_dim)
        self.tokenizer = SemanticTraceUnitTokenizer(hidden, teacher_embedding_dim, units)
        self.factorized_state = FactorizedTraceSemanticState(hidden, units, identity_dim)
        self.handshake = TraceSemanticHandshake(hidden, teacher_embedding_dim)
        self.rollout = SemanticTraceUnitRollout(hidden)
        self.readout = TraceConditionedBeliefReadout(hidden, teacher_embedding_dim, identity_dim)
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
    ) -> dict[str, torch.Tensor]:
        with torch.no_grad():
            features, _ = self._v30_features(obs_points=obs_points, obs_vis=obs_vis, obs_conf=obs_conf, semantic_id=semantic_id)
            v30_out = self.v30(obs_points=obs_points, obs_vis=obs_vis, obs_conf=obs_conf, semantic_id=semantic_id)
        point_token = features["point_token"].detach()
        step_hidden = features["step_hidden"].detach()
        point_pred = v30_out["point_pred"].detach()
        rel_pred = ((point_pred - features["last_visible"][:, :, None, :]) / features["spread"].squeeze(2)[:, :, None, :]).clamp(-16.0, 16.0)
        assign, unit_sem = self.tokenizer(point_token, obs_semantic_measurements, obs_semantic_measurement_mask)
        state = self.factorized_state(unit_sem)
        unit_point_sem = torch.einsum("bmu,bud->bmd", assign, state["z_sem"])
        vis_prob = torch.sigmoid(v30_out["visibility_logits"].detach())
        hidden = self.rollout(self.handshake(point_token, step_hidden, unit_point_sem, rel_pred, vis_prob))
        readout = self.readout(hidden)
        readout.update(
            {
                "point_hypotheses": v30_out["point_hypotheses"].detach(),
                "point_pred": point_pred,
                "top1_point_pred": v30_out["top1_point_pred"].detach(),
                "visibility_logits": v30_out["visibility_logits"].detach(),
                "point_to_unit_assignment": assign,
                "trace_units": state,
                "outputs_future_trace_field": True,
                "outputs_future_semantic_field": True,
                "teacher_as_method": False,
                "trace_conditioned_semantic_units_active": True,
            }
        )
        return readout
