from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from stwm.modules.ostf_traceanything_world_model_v26 import (
    ContextEncoderV26,
    PointTraceEncoderV26,
    SemanticIdentityEncoderV26,
    StepContextRefinerV26,
)


@dataclass
class OSTFLastObservedResidualConfigV28:
    obs_len: int = 8
    horizon: int = 32
    point_dim: int = 224
    hidden_dim: int = 384
    num_layers: int = 4
    num_heads: int = 8
    refinement_layers: int = 2
    semantic_dim: int = 10
    tusb_dim: int = 8
    prototype_count: int = 32
    num_hypotheses: int = 6
    semantic_id_buckets: int = 8192
    semantic_id_dim: int = 64
    damped_gamma: float = 0.0
    use_dense_points: bool = True
    use_semantic_memory: bool = True
    use_context: bool = True
    use_residual_modes: bool = True
    use_damped_prior: bool = True
    use_cv_prior: bool = True
    predict_variance: bool = False


class OSTFLastObservedResidualWorldModelV28(nn.Module):
    def __init__(self, cfg: OSTFLastObservedResidualConfigV28) -> None:
        super().__init__()
        self.cfg = cfg
        self.point_encoder = PointTraceEncoderV26(cfg.obs_len, cfg.point_dim)
        self.point_to_hidden = nn.Linear(cfg.point_dim, cfg.hidden_dim)
        self.semantic_encoder = SemanticIdentityEncoderV26(
            cfg.semantic_dim,
            cfg.tusb_dim,
            cfg.hidden_dim,
            cfg.semantic_id_buckets,
            cfg.semantic_id_dim,
        )
        self.context_encoder = ContextEncoderV26(cfg.hidden_dim, cfg.semantic_dim, cfg.tusb_dim)
        self.anchor_proj = nn.Sequential(
            nn.LayerNorm(cfg.obs_len * 4),
            nn.Linear(cfg.obs_len * 4, cfg.hidden_dim),
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
        self.mode_embed = nn.Parameter(torch.randn(cfg.num_hypotheses, cfg.hidden_dim) * 0.02)
        self.refiner = StepContextRefinerV26(cfg.hidden_dim, cfg.num_heads, cfg.refinement_layers)
        self.delta_head = nn.Sequential(
            nn.LayerNorm(cfg.hidden_dim + 2),
            nn.Linear(cfg.hidden_dim + 2, cfg.hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.hidden_dim, 2),
        )
        self.anchor_res_head = nn.Sequential(
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.hidden_dim, 2),
        )
        self.hypothesis_score_head = nn.Sequential(
            nn.LayerNorm(cfg.hidden_dim * 3),
            nn.Linear(cfg.hidden_dim * 3, cfg.hidden_dim),
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
        self.logvar_head = nn.Sequential(
            nn.LayerNorm(cfg.hidden_dim),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(cfg.hidden_dim // 2, 1),
        )
        self.global_residual_scale = nn.Parameter(torch.tensor(-5.0))
        self._init_parameters()

    def _init_parameters(self) -> None:
        with torch.no_grad():
            self.global_residual_scale.fill_(-5.0)
        nn.init.zeros_(self.delta_head[-1].weight)
        nn.init.zeros_(self.delta_head[-1].bias)
        nn.init.zeros_(self.anchor_res_head[-1].weight)
        nn.init.zeros_(self.anchor_res_head[-1].bias)
        nn.init.zeros_(self.hypothesis_score_head[-1].weight)
        nn.init.zeros_(self.hypothesis_score_head[-1].bias)

    def _sanitize(
        self,
        obs_points: torch.Tensor,
        obs_vis: torch.Tensor,
        obs_conf: torch.Tensor,
        rel_xy: torch.Tensor,
        anchor_obs: torch.Tensor,
        anchor_obs_vel: torch.Tensor,
        semantic_feat: torch.Tensor,
        box_feat: torch.Tensor,
        neighbor_feat: torch.Tensor,
        global_feat: torch.Tensor,
        tusb_token: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        return (
            torch.nan_to_num(obs_points, nan=0.0, posinf=1.5, neginf=-0.5),
            torch.nan_to_num(obs_vis.float(), nan=0.0, posinf=1.0, neginf=0.0) > 0.5,
            torch.nan_to_num(obs_conf, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0),
            torch.nan_to_num(rel_xy, nan=0.0, posinf=2.0, neginf=-2.0),
            torch.nan_to_num(anchor_obs, nan=0.0, posinf=1.5, neginf=-0.5),
            torch.nan_to_num(anchor_obs_vel, nan=0.0, posinf=1.0, neginf=-1.0),
            torch.nan_to_num(semantic_feat, nan=0.0, posinf=4.0, neginf=-4.0),
            torch.nan_to_num(box_feat, nan=0.0, posinf=4.0, neginf=-4.0),
            torch.nan_to_num(neighbor_feat, nan=0.0, posinf=4.0, neginf=-4.0),
            torch.nan_to_num(global_feat, nan=0.0, posinf=4.0, neginf=-4.0),
            torch.nan_to_num(tusb_token, nan=0.0, posinf=4.0, neginf=-4.0),
        )

    def _prior_modes(self, obs_points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        last_point = obs_points[:, :, -1]
        velocity = obs_points[:, :, -1] - obs_points[:, :, -2]
        t = torch.arange(1, self.cfg.horizon + 1, device=obs_points.device, dtype=obs_points.dtype)
        last = last_point[:, :, None, :].expand(-1, -1, self.cfg.horizon, -1)
        damped = last + float(self.cfg.damped_gamma) * velocity[:, :, None, :] * t[None, None, :, None]
        cv = last + velocity[:, :, None, :] * t[None, None, :, None]
        modes = [last]
        if self.cfg.use_damped_prior:
            modes.append(damped)
        if self.cfg.use_cv_prior:
            modes.append(cv)
        return torch.stack(modes, dim=3), last

    def forward(
        self,
        *,
        obs_points: torch.Tensor,
        obs_vis: torch.Tensor,
        obs_conf: torch.Tensor,
        rel_xy: torch.Tensor,
        anchor_obs: torch.Tensor,
        anchor_obs_vel: torch.Tensor,
        semantic_feat: torch.Tensor,
        semantic_id: torch.Tensor,
        box_feat: torch.Tensor,
        neighbor_feat: torch.Tensor,
        global_feat: torch.Tensor,
        tusb_token: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        (
            obs_points,
            obs_vis,
            obs_conf,
            rel_xy,
            anchor_obs,
            anchor_obs_vel,
            semantic_feat,
            box_feat,
            neighbor_feat,
            global_feat,
            tusb_token,
        ) = self._sanitize(
            obs_points,
            obs_vis,
            obs_conf,
            rel_xy,
            anchor_obs,
            anchor_obs_vel,
            semantic_feat,
            box_feat,
            neighbor_feat,
            global_feat,
            tusb_token,
        )
        if self.cfg.use_dense_points:
            point_tokens = self.point_encoder(obs_points, obs_vis, obs_conf, rel_xy)
        else:
            point_tokens = torch.zeros(obs_points.shape[0], obs_points.shape[1], self.cfg.point_dim, device=obs_points.device, dtype=obs_points.dtype)
        point_hidden = torch.nan_to_num(self.point_to_hidden(point_tokens), nan=0.0, posinf=8.0, neginf=-8.0)
        anchor_token = torch.nan_to_num(
            self.anchor_proj(torch.cat([anchor_obs, anchor_obs_vel], dim=-1).reshape(obs_points.shape[0], -1)),
            nan=0.0,
            posinf=8.0,
            neginf=-8.0,
        )
        semantic_input = semantic_feat if self.cfg.use_semantic_memory else torch.zeros_like(semantic_feat)
        semantic_token = torch.nan_to_num(self.semantic_encoder(semantic_input, semantic_id, tusb_token), nan=0.0, posinf=8.0, neginf=-8.0)
        if self.cfg.use_context:
            context_tokens, context_summary = self.context_encoder(
                crop_feat=semantic_input,
                box_feat=box_feat,
                neighbor_feat=neighbor_feat,
                global_feat=global_feat,
                tusb_token=tusb_token,
            )
        else:
            context_tokens = torch.zeros(obs_points.shape[0], 5, self.cfg.hidden_dim, device=obs_points.device, dtype=obs_points.dtype)
            context_summary = torch.zeros(obs_points.shape[0], self.cfg.hidden_dim, device=obs_points.device, dtype=obs_points.dtype)
        context_tokens = torch.nan_to_num(context_tokens, nan=0.0, posinf=8.0, neginf=-8.0)
        context_summary = torch.nan_to_num(context_summary, nan=0.0, posinf=8.0, neginf=-8.0)
        point_summary = torch.nan_to_num(point_hidden.mean(dim=1), nan=0.0, posinf=8.0, neginf=-8.0)
        object_token = torch.nan_to_num(
            self.object_fuse(torch.cat([point_summary, anchor_token, semantic_token, context_summary], dim=-1)),
            nan=0.0,
            posinf=8.0,
            neginf=-8.0,
        )
        future_tokens = torch.nan_to_num(
            self.rollout(self.time_embed[None].expand(obs_points.shape[0], -1, -1) + object_token[:, None, :]),
            nan=0.0,
            posinf=8.0,
            neginf=-8.0,
        )

        prior_modes, last_prior = self._prior_modes(obs_points)
        num_prior = prior_modes.shape[3]
        num_hyp = self.cfg.num_hypotheses
        if not self.cfg.use_residual_modes:
            num_hyp = num_prior
        rel_expand = rel_xy[:, :, None, None, :].expand(-1, -1, self.cfg.horizon, max(num_hyp, 1), -1)
        mode_embed = self.mode_embed[: max(num_hyp - num_prior, 0)]
        learned_modes = []
        if self.cfg.use_residual_modes and num_hyp > num_prior:
            refined_steps = [self.refiner(point_hidden, future_tokens[:, t], context_tokens) for t in range(self.cfg.horizon)]
            refined_tokens = torch.nan_to_num(torch.stack(refined_steps, dim=2), nan=0.0, posinf=8.0, neginf=-8.0)
            mode_tokens = refined_tokens[:, :, :, None, :] + mode_embed[None, None, None, :, :]
            delta_in = torch.cat([mode_tokens, rel_expand[:, :, :, : mode_embed.shape[0]]], dim=-1)
            delta = torch.tanh(torch.nan_to_num(self.delta_head(delta_in), nan=0.0, posinf=4.0, neginf=-4.0))
            residual_scale = 0.0025 + 0.0975 * torch.sigmoid(self.global_residual_scale)
            learned_modes = [last_prior[:, :, :, None, :] + residual_scale * delta]
        else:
            refined_tokens = future_tokens[:, None].expand(-1, obs_points.shape[1], -1, -1)
            delta = prior_modes.new_zeros(prior_modes.shape[0], prior_modes.shape[1], prior_modes.shape[2], 1, 2)
            residual_scale = torch.sigmoid(self.global_residual_scale) * 0.0

        point_hyp = torch.cat([prior_modes] + learned_modes, dim=3)
        pooled = torch.nan_to_num(refined_tokens.mean(dim=1), nan=0.0, posinf=8.0, neginf=-8.0)
        hyp_logits_in = torch.cat([pooled.mean(dim=1), object_token, context_summary], dim=-1)
        hypothesis_logits = torch.nan_to_num(self.hypothesis_score_head(hyp_logits_in), nan=0.0, posinf=8.0, neginf=-8.0)
        hypothesis_logits = hypothesis_logits[:, : point_hyp.shape[3]]
        # Bias mode selection toward the strongest prior unless training evidence moves it.
        if point_hyp.shape[3] > 0:
            bias = torch.zeros_like(hypothesis_logits)
            bias[:, 0] = 1.0
            hypothesis_logits = hypothesis_logits + bias
        hyp_prob = torch.softmax(hypothesis_logits, dim=-1)
        top1_idx = hypothesis_logits.argmax(dim=-1)
        gather_top1 = top1_idx[:, None, None, None, None].expand(-1, point_hyp.shape[1], point_hyp.shape[2], 1, point_hyp.shape[4])
        top1_point = point_hyp.gather(3, gather_top1).squeeze(3)
        weighted_point = (point_hyp * hyp_prob[:, None, None, :, None]).sum(dim=3)

        anchor_last = anchor_obs[:, -1]
        anchor_prior = anchor_last[:, None, :].expand(-1, self.cfg.horizon, -1)
        anchor_res = 0.05 * torch.tanh(torch.nan_to_num(self.anchor_res_head(future_tokens), nan=0.0, posinf=4.0, neginf=-4.0))
        anchor_pred = anchor_prior + anchor_res
        vis_in = torch.cat([refined_tokens, last_prior.norm(dim=-1, keepdim=True)], dim=-1)
        visibility_logits = torch.nan_to_num(self.visibility_head(vis_in).squeeze(-1), nan=0.0, posinf=8.0, neginf=-8.0)
        sem_context = torch.cat([future_tokens, semantic_token[:, None, :].expand(-1, self.cfg.horizon, -1)], dim=-1)
        semantic_logits = torch.nan_to_num(self.semantic_head(sem_context), nan=0.0, posinf=8.0, neginf=-8.0)
        mode_logvar = None
        if self.cfg.predict_variance:
            embed = self.mode_embed[: point_hyp.shape[3]]
            mode_logvar = torch.nan_to_num(
                self.logvar_head(future_tokens[:, :, None, :] + embed[None, None, :, :]).squeeze(-1),
                nan=0.0,
                posinf=2.0,
                neginf=-6.0,
            ).clamp(-6.0, 2.0)
        return {
            "point_hypotheses": point_hyp,
            "point_pred": weighted_point,
            "top1_point_pred": top1_point,
            "anchor_pred": anchor_pred,
            "visibility_logits": visibility_logits,
            "semantic_logits": semantic_logits,
            "hypothesis_logits": hypothesis_logits,
            "top1_mode_idx": top1_idx,
            "delta": delta,
            "residual_scale": residual_scale.expand(obs_points.shape[0]).to(dtype=obs_points.dtype),
            "physics_mix_alpha": torch.zeros(obs_points.shape[0], self.cfg.horizon, device=obs_points.device, dtype=obs_points.dtype),
            "mode_logvar": mode_logvar,
            "num_prior_modes": torch.full((obs_points.shape[0],), float(num_prior), device=obs_points.device, dtype=obs_points.dtype),
        }
