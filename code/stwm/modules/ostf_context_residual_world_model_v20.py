from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class OSTFContextResidualConfig:
    obs_len: int = 8
    horizon: int = 8
    point_dim: int = 192
    hidden_dim: int = 384
    num_layers: int = 4
    num_heads: int = 8
    refinement_layers: int = 2
    semantic_dim: int = 10
    prototype_count: int = 32
    num_hypotheses: int = 3
    use_context: bool = True
    use_dense_points: bool = True
    use_multi_hypothesis: bool = True
    use_semantic_memory: bool = True
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


class ContextTokenEncoder(nn.Module):
    def __init__(self, hidden_dim: int, semantic_dim: int) -> None:
        super().__init__()
        self.crop_proj = nn.Linear(semantic_dim, hidden_dim)
        self.box_proj = nn.Linear(14, hidden_dim)
        self.neighbor_proj = nn.Linear(10, hidden_dim)
        self.global_proj = nn.Linear(8, hidden_dim)
        self.semantic_proj = nn.Linear(semantic_dim, hidden_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=max(hidden_dim // 64, 1),
            dim_feedforward=hidden_dim * 4,
            batch_first=True,
            dropout=0.0,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        *,
        crop_feat: torch.Tensor,
        box_feat: torch.Tensor,
        neighbor_feat: torch.Tensor,
        global_feat: torch.Tensor,
        semantic_feat: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = torch.stack(
            [
                self.crop_proj(crop_feat),
                self.box_proj(box_feat),
                self.neighbor_proj(neighbor_feat),
                self.global_proj(global_feat),
                self.semantic_proj(semantic_feat),
            ],
            dim=1,
        )
        encoded = self.encoder(tokens)
        return encoded, self.norm(encoded.mean(dim=1))


class StepContextRefiner(nn.Module):
    def __init__(self, dim: int, heads: int, layers: int) -> None:
        super().__init__()
        enc = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=dim * 4,
            batch_first=True,
            dropout=0.0,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc, num_layers=layers)

    def forward(self, point_tokens: torch.Tensor, future_token: torch.Tensor, context_tokens: torch.Tensor) -> torch.Tensor:
        future = future_token[:, None, :]
        seq = torch.cat([future, context_tokens, point_tokens], dim=1)
        out = self.encoder(seq)
        return out[:, 1 + context_tokens.shape[1] :, :]


class OSTFContextResidualWorldModel(nn.Module):
    def __init__(self, cfg: OSTFContextResidualConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.point_encoder = PointTraceEncoder(cfg.obs_len, cfg.point_dim)
        self.point_to_hidden = nn.Linear(cfg.point_dim, cfg.hidden_dim)
        self.context_encoder = ContextTokenEncoder(cfg.hidden_dim, cfg.semantic_dim)
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
        self.object_fuse = nn.Sequential(
            nn.LayerNorm(cfg.hidden_dim * 4),
            nn.Linear(cfg.hidden_dim * 4, cfg.hidden_dim),
            nn.GELU(),
        )
        rollout = nn.TransformerEncoderLayer(
            d_model=cfg.hidden_dim,
            nhead=cfg.num_heads,
            dim_feedforward=cfg.hidden_dim * 4,
            batch_first=True,
            dropout=0.0,
            activation="gelu",
        )
        self.rollout = nn.TransformerEncoder(rollout, num_layers=cfg.num_layers)
        self.time_embed = nn.Parameter(torch.randn(cfg.horizon, cfg.hidden_dim) * 0.02)
        self.prior_mix_head = nn.Linear(cfg.hidden_dim, 1)
        self.affine_delta_head = nn.Sequential(
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.hidden_dim, 6),
        )
        self.refiner = StepContextRefiner(cfg.hidden_dim, cfg.num_heads, cfg.refinement_layers)
        self.delta_head = nn.Sequential(
            nn.LayerNorm(cfg.hidden_dim + 2),
            nn.Linear(cfg.hidden_dim + 2, cfg.hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.hidden_dim, cfg.num_hypotheses * 2),
        )
        self.anchor_res_head = nn.Sequential(
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.hidden_dim, 2),
        )
        self.hypothesis_score_head = nn.Sequential(
            nn.LayerNorm(cfg.hidden_dim * 2),
            nn.Linear(cfg.hidden_dim * 2, cfg.hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.hidden_dim, cfg.num_hypotheses),
        )
        self.visibility_head = nn.Sequential(
            nn.LayerNorm(cfg.hidden_dim + 1),
            nn.Linear(cfg.hidden_dim + 1, cfg.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(cfg.hidden_dim // 2, 1),
        )
        self.semantic_head = nn.Sequential(
            nn.LayerNorm(cfg.hidden_dim * 2),
            nn.Linear(cfg.hidden_dim * 2, cfg.hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.hidden_dim, cfg.prototype_count),
        )
        self.global_residual_scale = nn.Parameter(torch.tensor(0.0))
        self._init_parameters()

    def _init_parameters(self) -> None:
        # Start from a conservative physics-prior regime rather than a large random residual.
        nn.init.zeros_(self.prior_mix_head.weight)
        with torch.no_grad():
            self.prior_mix_head.bias.fill_(-6.0)
        nn.init.zeros_(self.affine_delta_head[-1].weight)
        nn.init.zeros_(self.affine_delta_head[-1].bias)
        nn.init.zeros_(self.delta_head[-1].weight)
        nn.init.zeros_(self.delta_head[-1].bias)
        nn.init.zeros_(self.anchor_res_head[-1].weight)
        nn.init.zeros_(self.anchor_res_head[-1].bias)
        nn.init.zeros_(self.hypothesis_score_head[-1].weight)
        nn.init.zeros_(self.hypothesis_score_head[-1].bias)
        with torch.no_grad():
            self.global_residual_scale.fill_(-5.0)

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
        crop_feat: torch.Tensor,
        box_feat: torch.Tensor,
        neighbor_feat: torch.Tensor,
        global_feat: torch.Tensor,
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
        point_hidden = self.point_to_hidden(point_tokens)
        anchor_token = self.anchor_proj(torch.cat([anchor_obs, anchor_obs_vel], dim=-1).reshape(obs_points.shape[0], -1))
        semantic_token = self.semantic_proj(semantic_feat if self.cfg.use_semantic_memory else torch.zeros_like(semantic_feat))
        if self.cfg.use_context:
            context_tokens, context_summary = self.context_encoder(
                crop_feat=crop_feat,
                box_feat=box_feat,
                neighbor_feat=neighbor_feat,
                global_feat=global_feat,
                semantic_feat=semantic_feat,
            )
        else:
            context_tokens = torch.zeros(obs_points.shape[0], 5, self.cfg.hidden_dim, device=obs_points.device, dtype=obs_points.dtype)
            context_summary = torch.zeros(obs_points.shape[0], self.cfg.hidden_dim, device=obs_points.device, dtype=obs_points.dtype)
        point_summary = point_hidden.mean(dim=1)
        object_token = self.object_fuse(torch.cat([point_summary, anchor_token, semantic_token, context_summary], dim=-1))
        future_tokens = self.rollout(self.time_embed[None].expand(obs_points.shape[0], -1, -1) + object_token[:, None, :])

        point_cv, anchor_cv = self._cv_prior(obs_points, anchor_obs, anchor_obs_vel)
        point_affine = self._affine_prior(obs_points, obs_vis, anchor_obs, anchor_cv, future_tokens)
        prior_cv = point_cv if self.cfg.use_cv_prior else torch.zeros_like(point_cv)
        prior_aff = point_affine if self.cfg.use_affine_prior else torch.zeros_like(point_affine)
        if self.cfg.use_cv_prior and self.cfg.use_affine_prior:
            alpha = torch.sigmoid(self.prior_mix_head(future_tokens)).clamp(0.0, 1.0)
            point_prior = prior_cv + alpha[:, None] * (prior_aff - prior_cv)
            prior_mix = torch.cat([1.0 - alpha, alpha], dim=-1)
        elif self.cfg.use_cv_prior:
            point_prior = prior_cv
            prior_mix = torch.cat(
                [
                    torch.ones(future_tokens.shape[0], self.cfg.horizon, 1, device=future_tokens.device, dtype=future_tokens.dtype),
                    torch.zeros(future_tokens.shape[0], self.cfg.horizon, 1, device=future_tokens.device, dtype=future_tokens.dtype),
                ],
                dim=-1,
            )
        else:
            point_prior = prior_aff
            prior_mix = torch.cat(
                [
                    torch.zeros(future_tokens.shape[0], self.cfg.horizon, 1, device=future_tokens.device, dtype=future_tokens.dtype),
                    torch.ones(future_tokens.shape[0], self.cfg.horizon, 1, device=future_tokens.device, dtype=future_tokens.dtype),
                ],
                dim=-1,
            )

        refined_steps = []
        for t in range(self.cfg.horizon):
            refined_steps.append(self.refiner(point_hidden, future_tokens[:, t], context_tokens))
        refined_tokens = torch.stack(refined_steps, dim=2)
        rel_expand = rel_xy[:, :, None, :].expand(-1, -1, self.cfg.horizon, -1)
        delta = 0.25 * torch.tanh(self.delta_head(torch.cat([refined_tokens, rel_expand], dim=-1)))
        delta = delta.view(obs_points.shape[0], obs_points.shape[1], self.cfg.horizon, self.cfg.num_hypotheses, 2)
        residual_scale = 0.005 + 0.145 * torch.sigmoid(self.global_residual_scale)
        point_hyp = point_prior[:, :, :, None, :] + residual_scale * delta

        hyp_context = torch.cat([future_tokens.mean(dim=1), context_summary], dim=-1)
        hyp_logits = self.hypothesis_score_head(hyp_context)
        if not self.cfg.use_multi_hypothesis:
            hyp_logits = hyp_logits.new_zeros(hyp_logits.shape[0], 1)
            point_hyp = point_hyp[:, :, :, :1, :]
        hyp_weight = torch.softmax(hyp_logits, dim=-1)
        point_pred = (point_hyp * hyp_weight[:, None, None, :, None]).sum(dim=3)

        anchor_pred = point_pred.mean(dim=1) + 0.02 * torch.tanh(self.anchor_res_head(future_tokens))
        visibility_logits = self.visibility_head(
            torch.cat([refined_tokens, obs_vis[:, :, -1:, None].float().expand(-1, -1, self.cfg.horizon, -1)], dim=-1)
        ).squeeze(-1)
        semantic_logits = self.semantic_head(torch.cat([future_tokens, context_summary[:, None, :].expand(-1, self.cfg.horizon, -1)], dim=-1))
        return {
            "point_prior": point_prior,
            "point_hypotheses": point_hyp,
            "point_pred": point_pred,
            "anchor_pred": anchor_pred,
            "visibility_logits": visibility_logits,
            "semantic_logits": semantic_logits,
            "hypothesis_logits": hyp_logits,
            "prior_mix": prior_mix,
            "point_cv": point_cv,
            "point_affine": point_affine,
            "delta": delta,
            "residual_scale": torch.as_tensor(residual_scale, device=obs_points.device, dtype=obs_points.dtype),
        }
