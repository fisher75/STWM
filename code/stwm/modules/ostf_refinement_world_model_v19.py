from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class OSTFRefinementConfig:
    obs_len: int = 8
    horizon: int = 8
    point_dim: int = 192
    hidden_dim: int = 384
    num_layers: int = 4
    num_heads: int = 8
    refinement_layers: int = 2
    semantic_dim: int = 10
    prototype_count: int = 32
    use_semantic_memory: bool = True
    use_dense_points: bool = True
    use_refinement_transformer: bool = True
    use_learnable_residual_scale: bool = True
    use_affine_prior: bool = True
    use_cv_prior: bool = True


class PointTraceEncoder(nn.Module):
    def __init__(self, obs_len: int, point_dim: int) -> None:
        super().__init__()
        in_dim = obs_len * 7
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, point_dim),
            nn.GELU(),
            nn.Linear(point_dim, point_dim),
            nn.GELU(),
        )

    def forward(self, obs_points: torch.Tensor, obs_vis: torch.Tensor, rel_xy: torch.Tensor) -> torch.Tensor:
        vel = torch.zeros_like(obs_points)
        if obs_points.shape[2] > 1:
            vel[:, :, 1:] = obs_points[:, :, 1:] - obs_points[:, :, :-1]
        rel = rel_xy[:, :, None, :].expand(-1, -1, obs_points.shape[2], -1)
        feat = torch.cat([obs_points, vel, obs_vis.float()[..., None], rel], dim=-1)
        return self.net(feat.reshape(obs_points.shape[0], obs_points.shape[1], -1))


class AttentionPool(nn.Module):
    def __init__(self, dim: int, heads: int) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.zeros(1, 1, dim))
        self.attn = nn.MultiheadAttention(dim, num_heads=heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        q = self.query.expand(tokens.shape[0], -1, -1)
        pooled, _ = self.attn(q, tokens, tokens, need_weights=False)
        return self.norm(pooled[:, 0])


class StepRefinementBlock(nn.Module):
    def __init__(self, dim: int, heads: int, layers: int) -> None:
        super().__init__()
        enc_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=dim * 4,
            batch_first=True,
            dropout=0.0,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=layers)

    def forward(self, point_tokens: torch.Tensor, future_token: torch.Tensor) -> torch.Tensor:
        seq = torch.cat([future_token[:, None, :], point_tokens], dim=1)
        refined = self.encoder(seq)
        return refined[:, 1:, :]


class OSTFRefinementWorldModel(nn.Module):
    def __init__(self, cfg: OSTFRefinementConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.point_encoder = PointTraceEncoder(cfg.obs_len, cfg.point_dim)
        self.point_to_hidden = nn.Linear(cfg.point_dim, cfg.hidden_dim)
        self.pool = AttentionPool(cfg.point_dim, heads=max(cfg.num_heads // 2, 1))
        self.anchor_proj = nn.Sequential(
            nn.LayerNorm(cfg.obs_len * 4),
            nn.Linear(cfg.obs_len * 4, cfg.hidden_dim),
            nn.GELU(),
        )
        self.semantic_proj = nn.Sequential(
            nn.LayerNorm(cfg.semantic_dim),
            nn.Linear(cfg.semantic_dim, cfg.hidden_dim),
            nn.GELU(),
        )
        self.fuse = nn.Sequential(
            nn.LayerNorm(cfg.point_dim + cfg.hidden_dim + cfg.hidden_dim),
            nn.Linear(cfg.point_dim + cfg.hidden_dim + cfg.hidden_dim, cfg.hidden_dim),
            nn.GELU(),
        )
        rollout_layer = nn.TransformerEncoderLayer(
            d_model=cfg.hidden_dim,
            nhead=cfg.num_heads,
            dim_feedforward=cfg.hidden_dim * 4,
            batch_first=True,
            dropout=0.0,
            activation="gelu",
        )
        self.rollout = nn.TransformerEncoder(rollout_layer, num_layers=cfg.num_layers)
        self.time_embed = nn.Parameter(torch.randn(cfg.horizon, cfg.hidden_dim) * 0.02)
        self.prior_mix_head = nn.Linear(cfg.hidden_dim, 2)
        self.affine_delta_head = nn.Sequential(
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.hidden_dim, 6),
        )
        self.anchor_res_head = nn.Sequential(
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.hidden_dim, 2),
        )
        self.semantic_base_head = nn.Sequential(
            nn.LayerNorm(cfg.semantic_dim),
            nn.Linear(cfg.semantic_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.hidden_dim, cfg.prototype_count),
        )
        self.semantic_delta_head = nn.Sequential(
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.hidden_dim, cfg.prototype_count),
        )
        self.refiner = StepRefinementBlock(cfg.hidden_dim, cfg.num_heads, cfg.refinement_layers)
        self.simple_refine = nn.Sequential(
            nn.LayerNorm(cfg.hidden_dim * 2),
            nn.Linear(cfg.hidden_dim * 2, cfg.hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.GELU(),
        )
        self.delta_head = nn.Sequential(
            nn.LayerNorm(cfg.hidden_dim + 2),
            nn.Linear(cfg.hidden_dim + 2, cfg.hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.hidden_dim, 2),
        )
        self.scale_head = nn.Sequential(
            nn.LayerNorm(cfg.hidden_dim + 2),
            nn.Linear(cfg.hidden_dim + 2, cfg.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(cfg.hidden_dim // 2, 1),
        )
        self.visibility_head = nn.Sequential(
            nn.LayerNorm(cfg.hidden_dim + 1),
            nn.Linear(cfg.hidden_dim + 1, cfg.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(cfg.hidden_dim // 2, 1),
        )
        self.global_residual_scale = nn.Parameter(torch.tensor(0.0))

    @staticmethod
    def _estimate_affine(obs_points: torch.Tensor, obs_vis: torch.Tensor, anchor_obs: torch.Tensor) -> torch.Tensor:
        x = (obs_points[:, :, -2] - anchor_obs[:, None, -2]).float()
        y = (obs_points[:, :, -1] - anchor_obs[:, None, -1]).float()
        valid = (obs_vis[:, :, -2] & obs_vis[:, :, -1]).float().unsqueeze(-1)
        eye = torch.eye(2, device=obs_points.device, dtype=torch.float32).unsqueeze(0)
        xtwx = torch.einsum("bmi,bmj->bij", x * valid, x).float() + 1e-4 * eye
        xtwy = torch.einsum("bmi,bmj->bij", x * valid, y).float()
        return torch.linalg.solve(xtwx, xtwy).to(obs_points.dtype)

    def _cv_prior(self, obs_points: torch.Tensor, anchor_obs: torch.Tensor, anchor_obs_vel: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        last_point = obs_points[:, :, -1]
        point_vel = obs_points[:, :, -1] - obs_points[:, :, -2]
        anchor_last = anchor_obs[:, -1]
        anchor_vel = anchor_obs_vel[:, -1]
        t = torch.arange(1, self.cfg.horizon + 1, device=obs_points.device, dtype=obs_points.dtype)
        point_cv = last_point[:, :, None, :] + point_vel[:, :, None, :] * t[None, None, :, None]
        anchor_cv = anchor_last[:, None, :] + anchor_vel[:, None, :] * t[None, :, None]
        return point_cv, anchor_cv

    def _affine_prior(
        self,
        obs_points: torch.Tensor,
        obs_vis: torch.Tensor,
        anchor_obs: torch.Tensor,
        anchor_cv: torch.Tensor,
        future_tokens: torch.Tensor,
    ) -> torch.Tensor:
        b = obs_points.shape[0]
        rel_last = obs_points[:, :, -1] - anchor_obs[:, None, -1]
        affine = self._estimate_affine(obs_points, obs_vis, anchor_obs)
        rel = rel_last
        out = []
        eye = torch.eye(2, device=obs_points.device, dtype=obs_points.dtype).unsqueeze(0)
        delta = 0.05 * self.affine_delta_head(future_tokens)
        for t in range(self.cfg.horizon):
            rel = torch.einsum("bmd,bdk->bmk", rel, affine.transpose(1, 2))
            d = delta[:, t]
            mat = eye + d[:, :4].reshape(b, 2, 2)
            trans = d[:, 4:]
            rel_t = torch.einsum("bmd,bdk->bmk", rel, mat.transpose(1, 2))
            out.append(rel_t + anchor_cv[:, t, None, :] + trans[:, None, :])
        return torch.stack(out, dim=2)

    def forward(
        self,
        *,
        obs_points: torch.Tensor,
        obs_vis: torch.Tensor,
        rel_xy: torch.Tensor,
        anchor_obs: torch.Tensor,
        anchor_obs_vel: torch.Tensor,
        semantic_feat: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if self.cfg.use_dense_points:
            point_tokens = self.point_encoder(obs_points, obs_vis, rel_xy)
        else:
            point_tokens = torch.zeros(
                obs_points.shape[0],
                obs_points.shape[1],
                self.cfg.point_dim,
                device=obs_points.device,
                dtype=obs_points.dtype,
            )
        point_summary = self.pool(point_tokens)
        anchor_feat = torch.cat([anchor_obs, anchor_obs_vel], dim=-1).reshape(obs_points.shape[0], -1)
        anchor_token = self.anchor_proj(anchor_feat)
        semantic_token = self.semantic_proj(semantic_feat) if self.cfg.use_semantic_memory else torch.zeros_like(anchor_token)
        object_token = self.fuse(torch.cat([point_summary, anchor_token, semantic_token], dim=-1))
        future_tokens = self.rollout(self.time_embed[None].expand(obs_points.shape[0], -1, -1) + object_token[:, None, :])

        point_cv, anchor_cv = self._cv_prior(obs_points, anchor_obs, anchor_obs_vel)
        point_affine = self._affine_prior(obs_points, obs_vis, anchor_obs, anchor_cv, future_tokens)
        prior_mix = torch.softmax(self.prior_mix_head(future_tokens), dim=-1)
        prior_cv = point_cv if self.cfg.use_cv_prior else torch.zeros_like(point_cv)
        prior_aff = point_affine if self.cfg.use_affine_prior else torch.zeros_like(point_affine)
        denom = prior_mix[..., 0:1] * float(self.cfg.use_cv_prior) + prior_mix[..., 1:2] * float(self.cfg.use_affine_prior)
        denom = denom.clamp_min(1e-6)
        point_prior = (
            prior_mix[..., 0:1, None] * prior_cv.transpose(1, 2) + prior_mix[..., 1:2, None] * prior_aff.transpose(1, 2)
        ) / denom[..., None]
        point_prior = point_prior.transpose(1, 2).contiguous()

        point_hidden = self.point_to_hidden(point_tokens)
        refined_steps = []
        for t in range(self.cfg.horizon):
            future_t = future_tokens[:, t]
            if self.cfg.use_refinement_transformer:
                refined = self.refiner(point_hidden, future_t)
            else:
                future_expand = future_t[:, None, :].expand(-1, point_hidden.shape[1], -1)
                refined = self.simple_refine(torch.cat([point_hidden, future_expand], dim=-1))
            refined_steps.append(refined)
        refined_tokens = torch.stack(refined_steps, dim=2)

        rel_expand = rel_xy[:, :, None, :].expand(-1, -1, self.cfg.horizon, -1)
        delta = self.delta_head(torch.cat([refined_tokens, rel_expand], dim=-1))
        if self.cfg.use_learnable_residual_scale:
            scale_logits = self.scale_head(torch.cat([refined_tokens, rel_expand], dim=-1)) + self.global_residual_scale
            residual_scale = 0.02 + 0.48 * torch.sigmoid(scale_logits)
        else:
            residual_scale = torch.full_like(delta[..., :1], 0.10)
        point_pred = point_prior + residual_scale * delta

        anchor_pred = point_pred.mean(dim=1) + 0.02 * torch.tanh(self.anchor_res_head(future_tokens))
        visibility_logits = self.visibility_head(
            torch.cat(
                [
                    refined_tokens,
                    obs_vis[:, :, -1:, None].float().expand(-1, -1, self.cfg.horizon, -1),
                ],
                dim=-1,
            )
        ).squeeze(-1)
        sem_base = self.semantic_base_head(semantic_feat if self.cfg.use_semantic_memory else torch.zeros_like(semantic_feat))
        semantic_logits = sem_base[:, None, :] + self.semantic_delta_head(future_tokens)
        return {
            "point_prior": point_prior,
            "point_pred": point_pred,
            "anchor_pred": anchor_pred,
            "visibility_logits": visibility_logits,
            "semantic_logits": semantic_logits,
            "prior_mix": prior_mix,
            "point_cv": point_cv,
            "point_affine": point_affine,
            "delta": delta,
            "residual_scale": residual_scale,
        }
