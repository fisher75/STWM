from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class OSTFPhysicsResidualConfig:
    obs_len: int = 8
    horizon: int = 8
    point_dim: int = 128
    hidden_dim: int = 256
    num_layers: int = 4
    num_heads: int = 8
    semantic_dim: int = 10
    prototype_count: int = 32
    dct_basis: int = 4
    use_semantic_memory: bool = True
    use_dense_points: bool = True
    use_residual_decoder: bool = True
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
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.zeros(1, 1, dim))
        self.attn = nn.MultiheadAttention(dim, num_heads=max(dim // 64, 1), batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        q = self.query.expand(tokens.shape[0], -1, -1)
        pooled, _ = self.attn(q, tokens, tokens, need_weights=False)
        return self.norm(pooled[:, 0])


class OSTFPhysicsResidualWorldModel(nn.Module):
    def __init__(self, cfg: OSTFPhysicsResidualConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.point_encoder = PointTraceEncoder(cfg.obs_len, cfg.point_dim)
        self.pool = AttentionPool(cfg.point_dim)
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
        residual_in = cfg.point_dim + cfg.hidden_dim + 2
        self.residual_coeff_head = nn.Sequential(
            nn.Linear(residual_in, cfg.hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.hidden_dim, cfg.dct_basis * 2),
        )
        self.residual_gain_head = nn.Sequential(
            nn.Linear(cfg.hidden_dim + cfg.point_dim, cfg.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(cfg.hidden_dim // 2, 1),
        )
        self.visibility_head = nn.Sequential(
            nn.Linear(cfg.hidden_dim + cfg.point_dim + 1, cfg.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(cfg.hidden_dim // 2, 1),
        )
        basis = self._build_dct_basis(cfg.horizon, cfg.dct_basis)
        self.register_buffer("dct_basis_matrix", basis, persistent=False)

    @staticmethod
    def _build_dct_basis(horizon: int, k: int) -> torch.Tensor:
        t = torch.arange(float(horizon)).unsqueeze(1)
        idx = torch.arange(float(k)).unsqueeze(0)
        basis = torch.cos(torch.pi * (t + 0.5) * idx / float(horizon))
        basis[:, 0] = 1.0
        basis = basis / basis.norm(dim=0, keepdim=True).clamp_min(1e-6)
        return basis.to(torch.float32)

    @staticmethod
    def _estimate_affine(obs_points: torch.Tensor, obs_vis: torch.Tensor, anchor_obs: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type=obs_points.device.type, enabled=False):
            x = (obs_points[:, :, -2] - anchor_obs[:, None, -2]).float()
            y = (obs_points[:, :, -1] - anchor_obs[:, None, -1]).float()
            valid = (obs_vis[:, :, -2] & obs_vis[:, :, -1]).float().unsqueeze(-1).to(x.dtype)
            xtwx = torch.einsum("bmi,bmj->bij", x * valid, x).float() + 1e-4 * torch.eye(2, device=obs_points.device, dtype=torch.float32).unsqueeze(0)
            xtwy = torch.einsum("bmi,bmj->bij", x * valid, y).float()
            a = torch.linalg.solve(xtwx, xtwy)
        return a.to(obs_points.dtype)

    def _cv_prior(self, obs_points: torch.Tensor, anchor_obs: torch.Tensor, anchor_obs_vel: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b, m, _, _ = obs_points.shape
        h = self.cfg.horizon
        last_point = obs_points[:, :, -1]
        point_vel = obs_points[:, :, -1] - obs_points[:, :, -2]
        anchor_last = anchor_obs[:, -1]
        anchor_vel = anchor_obs_vel[:, -1]
        t = torch.arange(1, h + 1, device=obs_points.device, dtype=obs_points.dtype)
        point_cv = last_point[:, :, None, :] + point_vel[:, :, None, :] * t[None, None, :, None]
        anchor_cv = anchor_last[:, None, :] + anchor_vel[:, None, :] * t[None, :, None]
        return point_cv, anchor_cv

    def _affine_prior(self, obs_points: torch.Tensor, obs_vis: torch.Tensor, anchor_obs: torch.Tensor, anchor_cv: torch.Tensor, future_tokens: torch.Tensor) -> torch.Tensor:
        b, m, _, _ = obs_points.shape
        rel_last = obs_points[:, :, -1] - anchor_obs[:, None, -1]
        a = self._estimate_affine(obs_points, obs_vis, anchor_obs)
        rel = rel_last
        out = []
        delta = 0.05 * self.affine_delta_head(future_tokens)
        eye = torch.eye(2, device=obs_points.device, dtype=obs_points.dtype).unsqueeze(0)
        for t in range(self.cfg.horizon):
            rel = torch.einsum("bmd,bdk->bmk", rel, a.transpose(1, 2))
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
            point_tokens = torch.zeros(obs_points.shape[0], obs_points.shape[1], self.cfg.point_dim, device=obs_points.device, dtype=obs_points.dtype)
        point_summary = self.pool(point_tokens)
        anchor_feat = torch.cat([anchor_obs, anchor_obs_vel], dim=-1).reshape(obs_points.shape[0], -1)
        anchor_token = self.anchor_proj(anchor_feat)
        semantic_token = self.semantic_proj(semantic_feat) if self.cfg.use_semantic_memory else torch.zeros_like(anchor_token)
        object_token = self.fuse(torch.cat([point_summary, anchor_token, semantic_token], dim=-1))
        future_tokens = self.rollout(self.time_embed[None].expand(obs_points.shape[0], -1, -1) + object_token[:, None])

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

        if self.cfg.use_residual_decoder:
            per_point = torch.cat(
                [
                    point_tokens,
                    object_token[:, None, :].expand(-1, point_tokens.shape[1], -1),
                    rel_xy,
                ],
                dim=-1,
            )
            coeff = self.residual_coeff_head(per_point).reshape(obs_points.shape[0], obs_points.shape[1], self.cfg.dct_basis, 2)
            residual = torch.einsum("hk,bmkc->bmhc", self.dct_basis_matrix.to(coeff), coeff)
            gain = torch.sigmoid(
                self.residual_gain_head(
                    torch.cat(
                        [
                            future_tokens[:, None, :, :].expand(-1, point_tokens.shape[1], -1, -1),
                            point_tokens[:, :, None, :].expand(-1, -1, self.cfg.horizon, -1),
                        ],
                        dim=-1,
                    )
                )
            )
            point_pred = point_prior + 0.10 * gain * residual
        else:
            residual = torch.zeros_like(point_prior)
            gain = torch.zeros_like(point_prior[..., :1])
            point_pred = point_prior

        anchor_pred = point_pred.mean(dim=1) + 0.02 * torch.tanh(self.anchor_res_head(future_tokens))
        vis_in = torch.cat(
            [
                future_tokens[:, None, :, :].expand(-1, point_tokens.shape[1], -1, -1),
                point_tokens[:, :, None, :].expand(-1, -1, self.cfg.horizon, -1),
                obs_vis[:, :, -1:, None].float().expand(-1, -1, self.cfg.horizon, -1),
            ],
            dim=-1,
        )
        visibility_logits = self.visibility_head(vis_in).squeeze(-1)
        sem_base = self.semantic_base_head(semantic_feat if self.cfg.use_semantic_memory else torch.zeros_like(semantic_feat))
        semantic_logits = sem_base[:, None, :] + self.semantic_delta_head(future_tokens)
        return {
            "point_prior": point_prior,
            "point_pred": point_pred,
            "anchor_pred": anchor_pred,
            "visibility_logits": visibility_logits,
            "semantic_logits": semantic_logits,
            "residual": residual,
            "prior_mix": prior_mix,
            "point_cv": point_cv,
            "point_affine": point_affine,
        }
