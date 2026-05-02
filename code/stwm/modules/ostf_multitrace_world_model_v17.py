from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class OSTFMultiTraceWorldModelConfig:
    obs_len: int = 8
    horizon: int = 8
    point_dim: int = 128
    hidden_dim: int = 256
    num_layers: int = 4
    num_heads: int = 8
    semantic_dim: int = 10
    prototype_count: int = 32
    use_semantic_memory: bool = True
    use_dense_point_input: bool = True
    use_point_residual_decoder: bool = True
    use_semantic_unit_compression: bool = True


class PointTraceEncoder(nn.Module):
    def __init__(self, obs_len: int, point_dim: int) -> None:
        super().__init__()
        self.obs_len = int(obs_len)
        in_dim = self.obs_len * 7
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


class PointMemoryTransformer(nn.Module):
    """Dense point baseline without semantic unit compression."""

    def __init__(self, cfg: OSTFMultiTraceWorldModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.point_encoder = PointTraceEncoder(cfg.obs_len, cfg.point_dim)
        self.time_embed = nn.Parameter(torch.randn(cfg.horizon, cfg.hidden_dim) * 0.02)
        self.anchor_proj = nn.Linear(cfg.obs_len * 4, cfg.hidden_dim)
        self.query_proj = nn.Linear(cfg.point_dim, cfg.hidden_dim)
        self.attn = nn.MultiheadAttention(cfg.hidden_dim, cfg.num_heads, batch_first=True)
        self.anchor_head = nn.Linear(cfg.hidden_dim, 2)
        self.residual_head = nn.Sequential(
            nn.Linear(cfg.hidden_dim + cfg.point_dim + 2, cfg.hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.hidden_dim, 2),
        )
        self.visibility_head = nn.Sequential(
            nn.Linear(cfg.hidden_dim + cfg.point_dim, cfg.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(cfg.hidden_dim // 2, 1),
        )
        self.semantic_head = nn.Linear(cfg.hidden_dim, cfg.prototype_count)

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
        del semantic_feat
        point_tokens = self.point_encoder(obs_points, obs_vis, rel_xy)
        anchor_feat = torch.cat([anchor_obs, anchor_obs_vel], dim=-1).reshape(anchor_obs.shape[0], -1)
        time_queries = self.time_embed[None].expand(obs_points.shape[0], -1, -1) + self.anchor_proj(anchor_feat)[:, None]
        memory = self.query_proj(point_tokens)
        future_tokens, _ = self.attn(time_queries, memory, memory, need_weights=False)
        last_anchor = anchor_obs[:, -1]
        anchor_delta = 0.10 * torch.tanh(self.anchor_head(future_tokens))
        anchor_pred = last_anchor[:, None] + torch.cumsum(anchor_delta, dim=1)
        per_point = torch.cat(
            [
                future_tokens[:, :, None, :].expand(-1, -1, point_tokens.shape[1], -1),
                point_tokens[:, None, :, :].expand(-1, future_tokens.shape[1], -1, -1),
                rel_xy[:, None, :, :].expand(-1, future_tokens.shape[1], -1, -1),
            ],
            dim=-1,
        )
        residual = 0.10 * torch.tanh(self.residual_head(per_point))
        pred_points = anchor_pred[:, :, None, :] + residual
        vis_logits = self.visibility_head(per_point[..., :-2]).squeeze(-1)
        sem_logits = self.semantic_head(future_tokens)
        return {
            "anchor_pred": anchor_pred,
            "point_pred": pred_points.permute(0, 2, 1, 3).contiguous(),
            "visibility_logits": vis_logits.permute(0, 2, 1).contiguous(),
            "semantic_logits": sem_logits,
        }


class OSTFMultiTraceWorldModel(nn.Module):
    def __init__(self, cfg: OSTFMultiTraceWorldModelConfig) -> None:
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
        fused_dim = cfg.point_dim + cfg.hidden_dim + cfg.hidden_dim
        self.object_fuse = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Linear(fused_dim, cfg.hidden_dim),
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
        self.future_rollout = nn.TransformerEncoder(layer, num_layers=cfg.num_layers)
        self.time_embed = nn.Parameter(torch.randn(cfg.horizon, cfg.hidden_dim) * 0.02)
        self.anchor_head = nn.Sequential(
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.hidden_dim, 2),
        )
        self.point_residual_head = nn.Sequential(
            nn.Linear(cfg.hidden_dim + cfg.point_dim + 2, cfg.hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.hidden_dim, 2),
        )
        self.visibility_head = nn.Sequential(
            nn.Linear(cfg.hidden_dim + cfg.point_dim + 2, cfg.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(cfg.hidden_dim // 2, 1),
        )
        self.semantic_head = nn.Sequential(
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.hidden_dim, cfg.prototype_count),
        )

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
        if not self.cfg.use_dense_point_input:
            point_tokens = torch.zeros(
                obs_points.shape[0],
                obs_points.shape[1],
                self.cfg.point_dim,
                device=obs_points.device,
                dtype=obs_points.dtype,
            )
        else:
            point_tokens = self.point_encoder(obs_points, obs_vis, rel_xy)
        if self.cfg.use_semantic_unit_compression:
            point_summary = self.pool(point_tokens)
        else:
            point_summary = point_tokens.mean(dim=1)
        anchor_feat = torch.cat([anchor_obs, anchor_obs_vel], dim=-1).reshape(anchor_obs.shape[0], -1)
        anchor_token = self.anchor_proj(anchor_feat)
        if self.cfg.use_semantic_memory:
            semantic_token = self.semantic_proj(semantic_feat)
        else:
            semantic_token = torch.zeros_like(anchor_token)
        object_token = self.object_fuse(torch.cat([point_summary, anchor_token, semantic_token], dim=-1))
        future_tokens = self.time_embed[None].expand(obs_points.shape[0], -1, -1) + object_token[:, None]
        future_tokens = self.future_rollout(future_tokens)
        last_anchor = anchor_obs[:, -1]
        anchor_delta = 0.10 * torch.tanh(self.anchor_head(future_tokens))
        anchor_pred = last_anchor[:, None] + torch.cumsum(anchor_delta, dim=1)
        if self.cfg.use_point_residual_decoder:
            per_point = torch.cat(
                [
                    future_tokens[:, :, None, :].expand(-1, -1, point_tokens.shape[1], -1),
                    point_tokens[:, None, :, :].expand(-1, future_tokens.shape[1], -1, -1),
                    rel_xy[:, None, :, :].expand(-1, future_tokens.shape[1], -1, -1),
                ],
                dim=-1,
            )
            residual = 0.10 * torch.tanh(self.point_residual_head(per_point))
            vis_logits = self.visibility_head(per_point).squeeze(-1)
            point_pred = anchor_pred[:, :, None, :] + residual
        else:
            point_pred = anchor_pred[:, :, None, :].expand(-1, -1, obs_points.shape[1], -1)
            vis_logits = torch.zeros(
                obs_points.shape[0], future_tokens.shape[1], obs_points.shape[1], device=obs_points.device, dtype=obs_points.dtype
            )
        sem_logits = self.semantic_head(future_tokens)
        return {
            "anchor_pred": anchor_pred,
            "point_pred": point_pred.permute(0, 2, 1, 3).contiguous(),
            "visibility_logits": vis_logits.permute(0, 2, 1).contiguous(),
            "semantic_logits": sem_logits,
        }
