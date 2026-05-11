from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

from stwm.modules.ostf_v34_semantic_trace_units import (
    SemanticTraceUnitsWorldModelV34,
    SemanticTraceUnitTokenizer,
    FactorizedTraceSemanticState,
)


class TemporalUnitRolloutV341(nn.Module):
    """Unit-level recurrent rollout; z_dyn is the recurrent state."""

    def __init__(self, hidden_dim: int, units: int, horizon: int = 32) -> None:
        super().__init__()
        self.horizon = int(horizon)
        self.time_embed = nn.Embedding(self.horizon, hidden_dim)
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.sem_update = nn.Sequential(nn.LayerNorm(hidden_dim * 2), nn.Linear(hidden_dim * 2, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim))

    def forward(self, z_dyn: torch.Tensor, z_sem: torch.Tensor, horizon: int) -> tuple[torch.Tensor, torch.Tensor]:
        b, u, d = z_dyn.shape
        h_state = z_dyn.reshape(b * u, d)
        sem = z_sem.reshape(b * u, d)
        dyn_steps = []
        sem_steps = []
        for step in range(horizon):
            t = self.time_embed.weight[min(step, self.horizon - 1)].view(1, d).expand(b * u, -1)
            h_state = self.gru(sem + t, h_state)
            h_norm = self.norm(h_state)
            sem_h = sem + self.sem_update(torch.cat([sem, h_norm], dim=-1))
            dyn_steps.append(h_norm.view(b, u, d))
            sem_steps.append(sem_h.view(b, u, d))
        return torch.stack(dyn_steps, dim=2), torch.stack(sem_steps, dim=2)


class UnitConditionedReadoutV341(nn.Module):
    def __init__(self, hidden_dim: int, semantic_dim: int, identity_dim: int) -> None:
        super().__init__()
        self.semantic = nn.Sequential(nn.LayerNorm(hidden_dim * 2), nn.Linear(hidden_dim * 2, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, semantic_dim))
        self.identity = nn.Sequential(nn.LayerNorm(hidden_dim + identity_dim), nn.Linear(hidden_dim + identity_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1))
        self.identity_embed = nn.Sequential(nn.LayerNorm(hidden_dim + identity_dim), nn.Linear(hidden_dim + identity_dim, identity_dim))
        self.uncertainty = nn.Sequential(nn.LayerNorm(hidden_dim + 1), nn.Linear(hidden_dim + 1, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1))

    def forward(
        self,
        field_hidden: torch.Tensor,
        point_unit_sem: torch.Tensor,
        point_identity_key: torch.Tensor,
        point_unit_confidence: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        sem = self.semantic(torch.cat([field_hidden, point_unit_sem], dim=-1))
        sem = F.normalize(sem, dim=-1)
        identity_in = torch.cat([field_hidden, point_identity_key], dim=-1)
        emb = F.normalize(self.identity_embed(identity_in), dim=-1)
        unc = F.softplus(self.uncertainty(torch.cat([field_hidden, point_unit_confidence[..., None]], dim=-1))).squeeze(-1)
        return {
            "future_semantic_belief": sem,
            "future_identity_belief": self.identity(identity_in).squeeze(-1),
            "identity_embedding": emb,
            "semantic_uncertainty": unc,
        }


class IdentityBoundSemanticTraceUnitsV341(SemanticTraceUnitsWorldModelV34):
    """Frozen-V30 model where semantic trace units are identity-bound and load-bearing."""

    def __init__(
        self,
        v30_checkpoint_path: str | Path,
        *,
        teacher_embedding_dim: int = 768,
        identity_dim: int = 64,
        units: int = 16,
        horizon: int = 32,
    ) -> None:
        super().__init__(v30_checkpoint_path, teacher_embedding_dim=teacher_embedding_dim, identity_dim=identity_dim, units=units)
        hidden = int(self.v30.cfg.hidden_dim)
        self.tokenizer = SemanticTraceUnitTokenizer(hidden, teacher_embedding_dim, units)
        self.factorized_state = FactorizedTraceSemanticState(hidden, units, identity_dim)
        self.unit_rollout = TemporalUnitRolloutV341(hidden, units, horizon=horizon)
        self.field_fuse = nn.Sequential(
            nn.LayerNorm(hidden * 5 + 3),
            nn.Linear(hidden * 5 + 3, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        self.readout = UnitConditionedReadoutV341(hidden, teacher_embedding_dim, identity_dim)
        self.obs_reconstruct = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, teacher_embedding_dim))
        self.confidence_gate = nn.Sequential(nn.LayerNorm(hidden + 1), nn.Linear(hidden + 1, hidden), nn.GELU(), nn.Linear(hidden, 1))
        for p in self.v30.parameters():
            p.requires_grad_(False)

    def forward(
        self,
        *,
        obs_points: torch.Tensor,
        obs_vis: torch.Tensor,
        obs_conf: torch.Tensor,
        obs_semantic_measurements: torch.Tensor,
        obs_semantic_measurement_mask: torch.Tensor,
        semantic_id: torch.Tensor | None = None,
        intervention: str | None = None,
    ) -> dict[str, torch.Tensor]:
        with torch.no_grad():
            features, _ = self._v30_features(obs_points=obs_points, obs_vis=obs_vis, obs_conf=obs_conf, semantic_id=semantic_id)
            v30_out = self.v30(obs_points=obs_points, obs_vis=obs_vis, obs_conf=obs_conf, semantic_id=semantic_id)
        point_token = features["point_token"].detach()
        step_hidden = features["step_hidden"].detach()
        point_pred = v30_out["point_pred"].detach()
        rel_pred = ((point_pred - features["last_visible"][:, :, None, :]) / features["spread"].squeeze(2)[:, :, None, :]).clamp(-16.0, 16.0)
        b, m, h, _ = point_pred.shape
        sem_obs = obs_semantic_measurements
        sem_mask = obs_semantic_measurement_mask
        if intervention == "zero_observed_semantic_measurements":
            sem_obs = torch.zeros_like(sem_obs)
            sem_mask = torch.zeros_like(sem_mask)
        elif intervention == "shuffle_observed_semantic_measurements_across_points":
            idx = torch.randperm(m, device=sem_obs.device)
            sem_obs = sem_obs[:, idx]
            sem_mask = sem_mask[:, idx]
        elif intervention == "shuffle_observed_semantic_measurements_across_samples" and b > 1:
            idx = torch.randperm(b, device=sem_obs.device)
            sem_obs = sem_obs[idx]
            sem_mask = sem_mask[idx]

        assign, unit_sem = self.tokenizer(point_token, sem_obs, sem_mask)
        if intervention == "uniform_unit_assignment":
            assign = torch.full_like(assign, 1.0 / assign.shape[-1])
        elif intervention == "permute_unit_assignment":
            perm = torch.randperm(assign.shape[-1], device=assign.device)
            assign = assign[..., perm]

        state = self.factorized_state(unit_sem)
        z_dyn = state["z_dyn"]
        z_sem = state["z_sem"]
        identity_key = state["identity_key"]
        unit_conf = state["unit_confidence"]
        if intervention in {"drop_z_sem", "drop_unit_semantics"}:
            z_sem = torch.zeros_like(z_sem)
        if intervention == "drop_unit_semantics":
            z_dyn = torch.zeros_like(z_dyn)
            identity_key = torch.zeros_like(identity_key)
            unit_conf = torch.zeros_like(unit_conf)
        elif intervention == "randomize_unit_semantics":
            z_sem = torch.randn_like(z_sem)
            unit_conf = torch.rand_like(unit_conf)

        unit_dyn_h, unit_sem_h = self.unit_rollout(z_dyn, z_sem, h)
        point_unit_dyn_h = torch.einsum("bmu,buhd->bmhd", assign, unit_dyn_h)
        point_unit_sem_h = torch.einsum("bmu,buhd->bmhd", assign, unit_sem_h)
        point_identity_key = torch.einsum("bmu,bud->bmd", assign, identity_key)[:, :, None, :].expand(-1, -1, h, -1)
        point_unit_confidence = torch.einsum("bmu,bu->bm", assign, unit_conf)[:, :, None].expand(-1, -1, h)
        vis_prob = torch.sigmoid(v30_out["visibility_logits"].detach())
        gate = torch.sigmoid(self.confidence_gate(torch.cat([point_unit_dyn_h, point_unit_confidence[..., None]], dim=-1)))
        field_hidden = self.field_fuse(
            torch.cat(
                [
                    point_token[:, :, None, :].expand(-1, -1, h, -1),
                    step_hidden[:, None, :, :].expand(-1, m, -1, -1),
                    point_unit_dyn_h,
                    point_unit_sem_h * gate,
                    point_unit_sem_h,
                    rel_pred,
                    vis_prob[:, :, :, None],
                ],
                dim=-1,
            )
        )
        out = self.readout(field_hidden, point_unit_sem_h, point_identity_key, point_unit_confidence)
        obs_recon = F.normalize(self.obs_reconstruct(torch.einsum("bmu,bud->bmd", assign, z_sem)), dim=-1)
        out.update(
            {
                "point_hypotheses": v30_out["point_hypotheses"].detach(),
                "point_pred": point_pred,
                "top1_point_pred": v30_out["top1_point_pred"].detach(),
                "visibility_logits": v30_out["visibility_logits"].detach(),
                "point_to_unit_assignment": assign,
                "unit_state": {"unit_dyn_h": unit_dyn_h, "unit_sem_h": unit_sem_h},
                "z_dyn": z_dyn,
                "z_sem": z_sem,
                "identity_key": identity_key,
                "unit_confidence": unit_conf,
                "observed_semantic_reconstruction": obs_recon,
                "unit_load_bearing_diagnostics": {
                    "semantic_gate": gate.squeeze(-1),
                    "point_unit_confidence": point_unit_confidence,
                    "point_unit_dyn_norm": point_unit_dyn_h.norm(dim=-1),
                    "point_unit_sem_norm": point_unit_sem_h.norm(dim=-1),
                },
                "outputs_future_trace_field": True,
                "outputs_future_semantic_field": True,
                "teacher_as_method": False,
                "trace_conditioned_semantic_units_active": True,
            }
        )
        return out
