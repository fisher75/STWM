from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

from stwm.modules.ostf_v34_semantic_trace_units import SemanticTraceUnitsWorldModelV34


def _last_visible(points: torch.Tensor, vis: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    b, m, t, _ = points.shape
    idx_base = torch.arange(t, device=points.device).view(1, 1, t).expand(b, m, -1)
    valid_idx = torch.where(vis.bool(), idx_base, torch.full_like(idx_base, -1))
    last_idx = valid_idx.max(dim=2).values.clamp_min(0)
    gather = last_idx[:, :, None, None].expand(-1, -1, 1, 2)
    last = points.gather(2, gather).squeeze(2)
    first_idx = torch.where(vis.bool(), idx_base, torch.full_like(idx_base, t)).min(dim=2).values.clamp_max(t - 1)
    first = points.gather(2, first_idx[:, :, None, None].expand(-1, -1, 1, 2)).squeeze(2)
    vel = last - first
    return last, vel


class DualSourceTraceSemanticTokenizer(nn.Module):
    def __init__(self, hidden_dim: int, semantic_dim: int, units: int = 16) -> None:
        super().__init__()
        self.units = int(units)
        self.trace_geom = nn.Sequential(nn.LayerNorm(6), nn.Linear(6, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim))
        self.trace_proj = nn.Sequential(nn.LayerNorm(hidden_dim * 2), nn.Linear(hidden_dim * 2, hidden_dim), nn.GELU())
        self.semantic_proj = nn.Sequential(nn.LayerNorm(semantic_dim), nn.Linear(semantic_dim, hidden_dim), nn.GELU())
        self.trace_assign = nn.Linear(hidden_dim, units)
        self.semantic_assign = nn.Linear(hidden_dim, units)
        self.joint_assign = nn.Sequential(nn.LayerNorm(hidden_dim * 2), nn.Linear(hidden_dim * 2, units))
        self.confidence = nn.Sequential(nn.LayerNorm(hidden_dim + 2), nn.Linear(hidden_dim + 2, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1))

    def forward(
        self,
        point_token: torch.Tensor,
        obs_points: torch.Tensor,
        obs_vis: torch.Tensor,
        obs_conf: torch.Tensor,
        obs_sem: torch.Tensor,
        obs_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        sem_mask = obs_mask.float()
        sem_pooled = (torch.nan_to_num(obs_sem.float()) * sem_mask[..., None]).sum(dim=2) / sem_mask.sum(dim=2, keepdim=True).clamp_min(1.0)
        sem_feat = self.semantic_proj(sem_pooled)
        last, vel = _last_visible(obs_points.float(), obs_vis.bool())
        vis_ratio = obs_vis.float().mean(dim=2, keepdim=True)
        conf_mean = obs_conf.float().mean(dim=2, keepdim=True)
        geom = torch.cat([last, vel, vis_ratio, conf_mean], dim=-1)
        trace_feat = self.trace_proj(torch.cat([point_token, self.trace_geom(geom)], dim=-1))
        logits = self.trace_assign(trace_feat) + self.semantic_assign(sem_feat) + self.joint_assign(torch.cat([trace_feat, sem_feat], dim=-1))
        assign = torch.softmax(logits, dim=-1)
        denom = assign.sum(dim=1).clamp_min(1e-6)
        trace_unit = torch.einsum("bmu,bmd->bud", assign, trace_feat) / denom[..., None]
        sem_unit = torch.einsum("bmu,bmd->bud", assign, sem_feat) / denom[..., None]
        sem_coverage = sem_mask.mean(dim=2)
        point_conf = self.confidence(torch.cat([sem_feat, sem_coverage[..., None], conf_mean], dim=-1)).squeeze(-1).sigmoid()
        unit_conf = torch.einsum("bmu,bm->bu", assign, point_conf) / denom
        return {
            "point_to_unit_assignment": assign,
            "trace_point_features": trace_feat,
            "semantic_point_features": sem_feat,
            "trace_unit_features": trace_unit,
            "semantic_unit_features": sem_unit,
            "point_measurement_confidence": point_conf,
            "unit_confidence": unit_conf.clamp(0.0, 1.0),
        }


class DualSourceFactorizedState(nn.Module):
    def __init__(self, hidden_dim: int, identity_dim: int) -> None:
        super().__init__()
        self.dyn = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim))
        self.sem = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim))
        self.identity = nn.Sequential(nn.LayerNorm(hidden_dim * 2), nn.Linear(hidden_dim * 2, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, identity_dim))

    def forward(self, trace_unit_features: torch.Tensor, semantic_unit_features: torch.Tensor) -> dict[str, torch.Tensor]:
        z_dyn = self.dyn(trace_unit_features)
        z_sem = self.sem(semantic_unit_features)
        identity_key = F.normalize(self.identity(torch.cat([z_dyn, z_sem], dim=-1)), dim=-1)
        return {"z_dyn": z_dyn, "z_sem": z_sem, "identity_key": identity_key}


class TemporalUnitRolloutV342(nn.Module):
    def __init__(self, hidden_dim: int, horizon: int = 32) -> None:
        super().__init__()
        self.time_embed = nn.Embedding(horizon, hidden_dim)
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)
        self.sem_update = nn.Sequential(nn.LayerNorm(hidden_dim * 2), nn.Linear(hidden_dim * 2, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim))
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, z_dyn: torch.Tensor, z_sem: torch.Tensor, horizon: int) -> tuple[torch.Tensor, torch.Tensor]:
        b, u, d = z_dyn.shape
        dyn = z_dyn.reshape(b * u, d)
        sem0 = z_sem.reshape(b * u, d)
        dyn_steps = []
        sem_steps = []
        for h in range(horizon):
            inp = sem0 + self.time_embed.weight[min(h, self.time_embed.num_embeddings - 1)].view(1, d)
            dyn = self.gru(inp, dyn)
            dyn_h = self.norm(dyn)
            sem_h = sem0 + self.sem_update(torch.cat([sem0, dyn_h], dim=-1))
            dyn_steps.append(dyn_h.view(b, u, d))
            sem_steps.append(sem_h.view(b, u, d))
        return torch.stack(dyn_steps, dim=2), torch.stack(sem_steps, dim=2)


class TraceSemanticHandshakeV342(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(hidden_dim * 5 + 3), nn.Linear(hidden_dim * 5 + 3, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim), nn.GELU())

    def forward(
        self,
        point_token: torch.Tensor,
        step_hidden: torch.Tensor,
        point_dyn: torch.Tensor,
        point_sem: torch.Tensor,
        point_identity_trace: torch.Tensor,
        rel_pred: torch.Tensor,
        vis_prob: torch.Tensor,
    ) -> torch.Tensor:
        b, m, h, _ = rel_pred.shape
        return self.net(
            torch.cat(
                [
                    point_token[:, :, None, :].expand(-1, -1, h, -1),
                    step_hidden[:, None, :, :].expand(-1, m, -1, -1),
                    point_dyn,
                    point_sem,
                    point_identity_trace,
                    rel_pred,
                    vis_prob[:, :, :, None],
                ],
                dim=-1,
            )
        )


class TraceConditionedBeliefReadoutV342(nn.Module):
    def __init__(self, hidden_dim: int, semantic_dim: int, identity_dim: int) -> None:
        super().__init__()
        self.semantic = nn.Sequential(nn.LayerNorm(hidden_dim * 3), nn.Linear(hidden_dim * 3, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, semantic_dim))
        self.identity = nn.Sequential(nn.LayerNorm(hidden_dim + identity_dim), nn.Linear(hidden_dim + identity_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1))
        self.identity_embed = nn.Sequential(nn.LayerNorm(hidden_dim + identity_dim), nn.Linear(hidden_dim + identity_dim, identity_dim))
        self.uncertainty = nn.Sequential(nn.LayerNorm(hidden_dim + 1), nn.Linear(hidden_dim + 1, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1))

    def forward(self, hidden: torch.Tensor, point_dyn: torch.Tensor, point_sem: torch.Tensor, point_identity_key: torch.Tensor, point_unit_conf: torch.Tensor) -> dict[str, torch.Tensor]:
        sem = F.normalize(self.semantic(torch.cat([hidden, point_dyn, point_sem], dim=-1)), dim=-1)
        ident_in = torch.cat([hidden, point_identity_key], dim=-1)
        emb = F.normalize(self.identity_embed(ident_in), dim=-1)
        unc = F.softplus(self.uncertainty(torch.cat([hidden, point_unit_conf[..., None]], dim=-1))).squeeze(-1)
        return {
            "future_semantic_belief": sem,
            "future_identity_belief": self.identity(ident_in).squeeze(-1),
            "identity_embedding": emb,
            "semantic_uncertainty": unc,
        }


class DualSourceSemanticTraceUnitsV342(SemanticTraceUnitsWorldModelV34):
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
        self.tokenizer = DualSourceTraceSemanticTokenizer(hidden, teacher_embedding_dim, units)
        self.factorized_state = DualSourceFactorizedState(hidden, identity_dim)
        self.unit_rollout = TemporalUnitRolloutV342(hidden, horizon=horizon)
        self.identity_to_hidden = nn.Linear(identity_dim, hidden)
        self.handshake = TraceSemanticHandshakeV342(hidden)
        self.readout = TraceConditionedBeliefReadoutV342(hidden, teacher_embedding_dim, identity_dim)
        self.obs_reconstruct = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, teacher_embedding_dim))
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
        tok = self.tokenizer(point_token, obs_points, obs_vis, obs_conf, sem_obs, sem_mask)
        assign = tok["point_to_unit_assignment"]
        if intervention == "uniform_unit_assignment":
            assign = torch.full_like(assign, 1.0 / assign.shape[-1])
        elif intervention == "permute_unit_assignment":
            assign = assign[..., torch.randperm(assign.shape[-1], device=assign.device)]
        state = self.factorized_state(tok["trace_unit_features"], tok["semantic_unit_features"])
        z_dyn = state["z_dyn"]
        z_sem = state["z_sem"]
        identity_key = state["identity_key"]
        unit_conf = tok["unit_confidence"]
        if intervention == "drop_z_dyn":
            z_dyn = torch.zeros_like(z_dyn)
        elif intervention == "drop_z_sem":
            z_sem = torch.zeros_like(z_sem)
        elif intervention == "drop_identity_key":
            identity_key = torch.zeros_like(identity_key)
        elif intervention == "drop_unit_confidence":
            unit_conf = torch.zeros_like(unit_conf)
        elif intervention == "randomize_units":
            z_dyn = torch.randn_like(z_dyn)
            z_sem = torch.randn_like(z_sem)
            identity_key = F.normalize(torch.randn_like(identity_key), dim=-1)
        unit_dyn_h, unit_sem_h = self.unit_rollout(z_dyn, z_sem, h)
        point_dyn = torch.einsum("bmu,buhd->bmhd", assign, unit_dyn_h)
        point_sem = torch.einsum("bmu,buhd->bmhd", assign, unit_sem_h)
        point_identity_key = torch.einsum("bmu,bud->bmd", assign, identity_key)[:, :, None, :].expand(-1, -1, h, -1)
        point_identity_trace = self.identity_to_hidden(point_identity_key)
        point_unit_conf = torch.einsum("bmu,bu->bm", assign, unit_conf)[:, :, None].expand(-1, -1, h)
        hidden = self.handshake(point_token, step_hidden, point_dyn, point_sem, point_identity_trace, rel_pred, torch.sigmoid(v30_out["visibility_logits"].detach()))
        out = self.readout(hidden, point_dyn, point_sem, point_identity_key, point_unit_conf)
        obs_recon = F.normalize(self.obs_reconstruct(torch.einsum("bmu,bud->bmd", assign, z_sem)), dim=-1)
        out.update(
            {
                "point_hypotheses": v30_out["point_hypotheses"].detach(),
                "point_pred": point_pred,
                "top1_point_pred": v30_out["top1_point_pred"].detach(),
                "visibility_logits": v30_out["visibility_logits"].detach(),
                "point_to_unit_assignment": assign,
                "trace_unit_features": tok["trace_unit_features"],
                "semantic_unit_features": tok["semantic_unit_features"],
                "unit_state": {"unit_dyn_h": unit_dyn_h, "unit_sem_h": unit_sem_h},
                "z_dyn": z_dyn,
                "z_sem": z_sem,
                "identity_key": identity_key,
                "unit_confidence": unit_conf,
                "observed_semantic_reconstruction": obs_recon,
                "unit_load_bearing_diagnostics": {
                    "point_unit_confidence": point_unit_conf,
                    "point_dyn_norm": point_dyn.norm(dim=-1),
                    "point_sem_norm": point_sem.norm(dim=-1),
                },
                "outputs_future_trace_field": True,
                "outputs_future_semantic_field": True,
                "teacher_as_method": False,
                "trace_conditioned_semantic_units_active": True,
            }
        )
        return out
