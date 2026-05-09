from __future__ import annotations

import torch
from torch import nn


class OSTFSemanticIdentityHeadV33(nn.Module):
    """Head-only V33 smoke module; it does not modify trajectory predictions."""

    def __init__(self, obs_len: int = 8, hidden_dim: int = 128, instance_buckets: int = 4096) -> None:
        super().__init__()
        self.obs_len = int(obs_len)
        self.instance_embed = nn.Embedding(instance_buckets, 16)
        in_dim = self.obs_len * 4 + 16 + 4
        self.encoder = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.valid_head = nn.Linear(hidden_dim, 1)
        self.same_instance_head = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        *,
        obs_points: torch.Tensor,
        obs_vis: torch.Tensor,
        obs_conf: torch.Tensor,
        point_to_instance_id: torch.Tensor,
        horizon: int,
    ) -> dict[str, torch.Tensor]:
        points = torch.nan_to_num(obs_points.float(), nan=0.0, posinf=1e6, neginf=-1e6)
        vis = obs_vis.float()
        conf = torch.nan_to_num(obs_conf.float(), nan=0.0, posinf=1.0, neginf=0.0).clamp(0, 1)
        center = points[:, :, -1:].detach()
        norm = points - center
        vel = torch.zeros_like(norm)
        vel[:, :, 1:] = norm[:, :, 1:] - norm[:, :, :-1]
        iid = point_to_instance_id.clamp_min(0) % self.instance_embed.num_embeddings
        emb = self.instance_embed(iid)
        feat = torch.cat(
            [
                norm.flatten(2),
                vel.flatten(2),
                vis.mean(dim=-1, keepdim=True),
                conf.mean(dim=-1, keepdim=True),
                vis[:, :, -1:],
                conf[:, :, -1:],
                emb,
            ],
            dim=-1,
        )
        token = self.encoder(feat)
        valid = self.valid_head(token).squeeze(-1)[:, :, None].expand(-1, -1, horizon)
        same = self.same_instance_head(token).squeeze(-1)[:, :, None].expand(-1, -1, horizon)
        return {
            "point_persistence_logits": valid,
            "same_instance_logits": same,
            "semantic_logits": torch.empty((*valid.shape, 0), device=valid.device, dtype=valid.dtype),
        }
