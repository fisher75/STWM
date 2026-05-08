from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class OSTFExternalGTConfigV30:
    obs_len: int = 8
    horizon: int = 32
    point_dim: int = 2
    hidden_dim: int = 192
    point_token_dim: int = 96
    num_layers: int = 3
    num_heads: int = 6
    learned_modes: int = 4
    damped_gamma: float = 0.0
    use_semantic: bool = True
    semantic_buckets: int = 8192
    semantic_dim: int = 32
    predict_logvar: bool = True
    point_dropout: float = 0.0
    density_aware_pooling: str = "none"


def _last_visible(obs_points: torch.Tensor, obs_vis: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    b, m, t, d = obs_points.shape
    idx = torch.arange(t, device=obs_points.device).view(1, 1, t)
    masked = torch.where(obs_vis, idx, torch.full_like(idx, -1))
    last_idx = masked.max(dim=-1).values.clamp_min(0)
    gather = last_idx[..., None, None].expand(b, m, 1, d)
    last = obs_points.gather(2, gather).squeeze(2)
    has_vis = obs_vis.any(dim=-1)
    fallback = obs_points[:, :, -1]
    return torch.where(has_vis[..., None], last, fallback), last_idx


def _visible_velocity(obs_points: torch.Tensor, obs_vis: torch.Tensor, last_idx: torch.Tensor) -> torch.Tensor:
    b, m, t, d = obs_points.shape
    idx = torch.arange(t, device=obs_points.device).view(1, 1, t)
    before_last = idx < last_idx[..., None]
    masked_prev = torch.where(obs_vis & before_last, idx, torch.full_like(idx, -1))
    prev_idx = masked_prev.max(dim=-1).values
    safe_prev = prev_idx.clamp_min(0)
    last_gather = last_idx[..., None, None].expand(b, m, 1, d)
    prev_gather = safe_prev[..., None, None].expand(b, m, 1, d)
    last = obs_points.gather(2, last_gather).squeeze(2)
    prev = obs_points.gather(2, prev_gather).squeeze(2)
    dt = (last_idx - safe_prev).clamp_min(1).float()[..., None]
    vel = (last - prev) / dt
    return torch.where((prev_idx >= 0)[..., None], vel, torch.zeros_like(vel))


class OSTFExternalGTWorldModelV30(nn.Module):
    def __init__(self, cfg: OSTFExternalGTConfigV30) -> None:
        super().__init__()
        self.cfg = cfg
        in_dim = cfg.obs_len * (cfg.point_dim + 2) + 8
        self.point_encoder = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, cfg.point_token_dim),
            nn.GELU(),
            nn.Linear(cfg.point_token_dim, cfg.hidden_dim),
            nn.GELU(),
        )
        self.semantic_embed = nn.Embedding(cfg.semantic_buckets, cfg.semantic_dim)
        self.object_encoder = nn.Sequential(
            nn.LayerNorm(cfg.hidden_dim + cfg.semantic_dim),
            nn.Linear(cfg.hidden_dim + cfg.semantic_dim, cfg.hidden_dim),
            nn.GELU(),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.hidden_dim,
            nhead=cfg.num_heads,
            dim_feedforward=cfg.hidden_dim * 4,
            batch_first=True,
            dropout=0.0,
            activation="gelu",
        )
        self.rollout = nn.TransformerEncoder(layer, num_layers=cfg.num_layers)
        self.time_embed = nn.Parameter(torch.randn(cfg.horizon, cfg.hidden_dim) * 0.02)
        self.mode_embed = nn.Parameter(torch.randn(cfg.learned_modes, cfg.hidden_dim) * 0.02)
        self.residual_head = nn.Sequential(
            nn.LayerNorm(cfg.hidden_dim + cfg.point_dim),
            nn.Linear(cfg.hidden_dim + cfg.point_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.hidden_dim, cfg.point_dim),
        )
        self.mode_logits_head = nn.Sequential(nn.LayerNorm(cfg.hidden_dim), nn.Linear(cfg.hidden_dim, 5 + cfg.learned_modes))
        self.visibility_head = nn.Sequential(
            nn.LayerNorm(cfg.hidden_dim + 2),
            nn.Linear(cfg.hidden_dim + 2, cfg.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(cfg.hidden_dim // 2, 1),
        )
        self.semantic_head = nn.Sequential(nn.LayerNorm(cfg.hidden_dim), nn.Linear(cfg.hidden_dim, 32))
        self.logvar_head = nn.Sequential(nn.LayerNorm(cfg.hidden_dim), nn.Linear(cfg.hidden_dim, 1))
        self.residual_scale = nn.Parameter(torch.tensor(-4.0))
        nn.init.zeros_(self.residual_head[-1].weight)
        nn.init.zeros_(self.residual_head[-1].bias)

    def prior_modes(self, obs_points: torch.Tensor, obs_vis: torch.Tensor) -> dict[str, torch.Tensor]:
        obs_points = torch.nan_to_num(obs_points.float(), nan=0.0, posinf=1e6, neginf=-1e6)
        obs_vis = obs_vis.bool()
        h = self.cfg.horizon
        t = torch.arange(1, h + 1, device=obs_points.device, dtype=obs_points.dtype).view(1, 1, h, 1)
        last_observed = obs_points[:, :, -1]
        last_visible, last_idx = _last_visible(obs_points, obs_vis)
        vel_visible = _visible_velocity(obs_points, obs_vis, last_idx)
        vel = obs_points[:, :, -1] - obs_points[:, :, -2]
        median = last_visible.median(dim=1, keepdim=True).values
        layout = last_visible - median
        return {
            "last_visible_copy": last_visible[:, :, None, :].expand(-1, -1, h, -1),
            "last_observed_copy": last_observed[:, :, None, :].expand(-1, -1, h, -1),
            "visibility_aware_cv": last_visible[:, :, None, :] + vel_visible[:, :, None, :] * t,
            "visibility_aware_damped": last_visible[:, :, None, :] + float(self.cfg.damped_gamma) * vel_visible[:, :, None, :] * t,
            "median_object_anchor_copy": (median + layout)[:, :, None, :].expand(-1, -1, h, -1),
            "fixed_affine": last_observed[:, :, None, :] + 0.25 * vel[:, :, None, :] * t,
        }

    def forward(
        self,
        *,
        obs_points: torch.Tensor,
        obs_vis: torch.Tensor,
        obs_conf: torch.Tensor,
        semantic_id: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        obs_points = torch.nan_to_num(obs_points.float(), nan=0.0, posinf=1e6, neginf=-1e6)
        obs_vis_f = obs_vis.float()
        obs_conf = torch.nan_to_num(obs_conf.float(), nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
        priors = self.prior_modes(obs_points, obs_vis)
        last_visible = priors["last_visible_copy"][:, :, 0]
        center = last_visible.median(dim=1, keepdim=True).values
        spread = (last_visible - center).abs().flatten(1).median(dim=1).values.view(-1, 1, 1, 1).clamp_min(16.0)
        norm_obs = (obs_points - center[:, :, None, :]) / spread
        velocity = torch.zeros_like(norm_obs)
        velocity[:, :, 1:] = norm_obs[:, :, 1:] - norm_obs[:, :, :-1]
        log_scale = torch.log(spread.squeeze(2).clamp_min(1.0)).expand(-1, obs_points.shape[1], -1)
        anchor_feat = torch.cat(
            [
                ((last_visible - center) / spread.squeeze(2)),
                velocity[:, :, -1],
                obs_vis_f.mean(dim=-1, keepdim=True),
                obs_conf.mean(dim=-1, keepdim=True),
                obs_vis_f[:, :, -1:],
                log_scale,
            ],
            dim=-1,
        )
        flat = torch.cat([norm_obs, obs_vis_f[..., None], obs_conf[..., None]], dim=-1).flatten(2)
        point_token = self.point_encoder(torch.cat([flat, anchor_feat], dim=-1))
        valid_point = obs_vis.any(dim=-1).float()
        if self.training and self.cfg.point_dropout > 0:
            keep = (torch.rand_like(valid_point) >= float(self.cfg.point_dropout)).float()
            keep = torch.maximum(keep, valid_point * 0.0 + (keep.sum(dim=1, keepdim=True) <= 0).float())
            valid_point = valid_point * keep
            point_token = point_token * keep[..., None]
        if self.cfg.density_aware_pooling in {"valid_weighted", "local_attention"}:
            denom = valid_point.sum(dim=1, keepdim=True).clamp_min(1.0)
            pooled = (point_token * valid_point[..., None]).sum(dim=1) / denom
        else:
            pooled = point_token.mean(dim=1)
        if semantic_id is None:
            semantic_id = torch.full((obs_points.shape[0],), -1, device=obs_points.device, dtype=torch.long)
        sem_idx = semantic_id.clamp_min(0) % self.cfg.semantic_buckets
        sem = self.semantic_embed(sem_idx)
        if not self.cfg.use_semantic:
            sem = torch.zeros_like(sem)
        obj = self.object_encoder(torch.cat([pooled, sem], dim=-1))
        step_tokens = obj[:, None, :] + self.time_embed[None, :, :]
        step_hidden = self.rollout(step_tokens)
        k_prior = ["last_visible_copy", "last_observed_copy", "visibility_aware_cv", "visibility_aware_damped", "median_object_anchor_copy"]
        prior_stack = torch.stack([priors[name] for name in k_prior], dim=3)
        base = priors["last_visible_copy"]
        learned = []
        rel = ((last_visible - center) / spread.squeeze(2)).clamp(-12.0, 12.0)
        scale = torch.sigmoid(self.residual_scale) * spread.squeeze(2)
        for mode in range(self.cfg.learned_modes):
            hidden = step_hidden[:, None, :, :] + point_token[:, :, None, :] + self.mode_embed[None, None, mode, :]
            inp = torch.cat([hidden, rel[:, :, None, :].expand(-1, -1, self.cfg.horizon, -1)], dim=-1)
            delta = self.residual_head(inp) * scale[:, :, None, :]
            learned.append(base + delta)
        learned_stack = torch.stack(learned, dim=3) if learned else prior_stack[:, :, :, :0]
        point_hypotheses = torch.cat([prior_stack, learned_stack], dim=3)
        logits = self.mode_logits_head(obj)
        weights = torch.softmax(logits, dim=-1)
        weighted = (point_hypotheses * weights[:, None, None, :, None]).sum(dim=3)
        top_idx = logits.argmax(dim=-1)
        gather = top_idx[:, None, None, None, None].expand(-1, obs_points.shape[1], self.cfg.horizon, 1, self.cfg.point_dim)
        top1 = point_hypotheses.gather(3, gather).squeeze(3)
        vis_in = torch.cat([step_hidden[:, None, :, :].expand(-1, obs_points.shape[1], -1, -1), obs_vis_f[:, :, -1:, None].expand(-1, -1, self.cfg.horizon, -1), obs_conf[:, :, -1:, None].expand(-1, -1, self.cfg.horizon, -1)], dim=-1)
        visibility_logits = self.visibility_head(vis_in).squeeze(-1)
        semantic_logits = self.semantic_head(step_hidden)[:, None, :, :].expand(-1, obs_points.shape[1], -1, -1)
        out = {
            "point_hypotheses": point_hypotheses,
            "hypothesis_logits": logits,
            "point_pred": weighted,
            "top1_point_pred": top1,
            "visibility_logits": visibility_logits,
            "semantic_logits": semantic_logits,
            "prior_names": k_prior,
            "num_prior_modes": torch.tensor(len(k_prior), device=obs_points.device),
            "point_encoder_activation_norm": point_token.detach().norm(dim=-1).mean(),
            "point_valid_ratio": valid_point.detach().mean(),
            "actual_m_points": torch.tensor(obs_points.shape[1], device=obs_points.device),
        }
        if self.cfg.predict_logvar:
            out["mode_logvar"] = self.logvar_head(step_hidden).squeeze(-1)
        return out
