from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class OSTFTraceAnythingConfig:
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
    use_context: bool = True
    use_dense_points: bool = True
    use_semantic_memory: bool = True
    use_physics_prior: bool = True
    use_multimodal: bool = True
    predict_variance: bool = False


class PointTraceEncoderV26(nn.Module):
    def __init__(self, obs_len: int, point_dim: int) -> None:
        super().__init__()
        in_dim = obs_len * 10
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, point_dim),
            nn.GELU(),
            nn.Linear(point_dim, point_dim),
            nn.GELU(),
        )

    def forward(
        self,
        obs_points: torch.Tensor,
        obs_vis: torch.Tensor,
        obs_conf: torch.Tensor,
        rel_xy: torch.Tensor,
    ) -> torch.Tensor:
        vel = torch.zeros_like(obs_points)
        acc = torch.zeros_like(obs_points)
        if obs_points.shape[2] > 1:
            vel[:, :, 1:] = obs_points[:, :, 1:] - obs_points[:, :, :-1]
        if obs_points.shape[2] > 2:
            acc[:, :, 2:] = vel[:, :, 2:] - vel[:, :, 1:-1]
        rel = rel_xy[:, :, None, :].expand(-1, -1, obs_points.shape[2], -1)
        feat = torch.cat([obs_points, vel, acc, obs_vis.float()[..., None], obs_conf[..., None], rel], dim=-1)
        return self.net(feat.reshape(obs_points.shape[0], obs_points.shape[1], -1))


class SemanticIdentityEncoderV26(nn.Module):
    def __init__(self, semantic_dim: int, tusb_dim: int, hidden_dim: int, buckets: int, id_dim: int) -> None:
        super().__init__()
        self.id_embed = nn.Embedding(buckets, id_dim)
        self.net = nn.Sequential(
            nn.LayerNorm(semantic_dim + tusb_dim + id_dim),
            nn.Linear(semantic_dim + tusb_dim + id_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

    def forward(self, semantic_feat: torch.Tensor, semantic_id: torch.Tensor, tusb_token: torch.Tensor) -> torch.Tensor:
        sid = self.id_embed(semantic_id)
        return self.net(torch.cat([semantic_feat, tusb_token, sid], dim=-1))


class ContextEncoderV26(nn.Module):
    def __init__(self, hidden_dim: int, semantic_dim: int, tusb_dim: int) -> None:
        super().__init__()
        self.crop_proj = nn.Linear(semantic_dim, hidden_dim)
        self.box_proj = nn.Linear(14, hidden_dim)
        self.neighbor_proj = nn.Linear(10, hidden_dim)
        self.global_proj = nn.Linear(8, hidden_dim)
        self.tusb_proj = nn.Linear(tusb_dim, hidden_dim)
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
        tusb_token: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = torch.stack(
            [
                self.crop_proj(crop_feat),
                self.box_proj(box_feat),
                self.neighbor_proj(neighbor_feat),
                self.global_proj(global_feat),
                self.tusb_proj(tusb_token),
            ],
            dim=1,
        )
        encoded = self.encoder(tokens)
        return encoded, self.norm(encoded.mean(dim=1))


class StepContextRefinerV26(nn.Module):
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


class OSTFTraceAnythingWorldModelV26(nn.Module):
    def __init__(self, cfg: OSTFTraceAnythingConfig) -> None:
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
        self.prior_mix_head = nn.Linear(cfg.hidden_dim, 1)
        self.affine_delta_head = nn.Sequential(
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.hidden_dim, 6),
        )
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
        self.global_residual_scale = nn.Parameter(torch.tensor(-4.0))
        self._init_parameters()

    def _init_parameters(self) -> None:
        nn.init.zeros_(self.prior_mix_head.weight)
        with torch.no_grad():
            self.prior_mix_head.bias.fill_(-4.0)
            self.global_residual_scale.fill_(-4.0)
        nn.init.zeros_(self.affine_delta_head[-1].weight)
        nn.init.zeros_(self.affine_delta_head[-1].bias)
        nn.init.zeros_(self.delta_head[-1].weight)
        nn.init.zeros_(self.delta_head[-1].bias)
        nn.init.zeros_(self.anchor_res_head[-1].weight)
        nn.init.zeros_(self.anchor_res_head[-1].bias)
        nn.init.zeros_(self.hypothesis_score_head[-1].weight)
        nn.init.zeros_(self.hypothesis_score_head[-1].bias)

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
        obs_points = torch.nan_to_num(obs_points, nan=0.0, posinf=1.5, neginf=-0.5)
        obs_vis = torch.nan_to_num(obs_vis.float(), nan=0.0, posinf=1.0, neginf=0.0) > 0.5
        obs_conf = torch.nan_to_num(obs_conf, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
        rel_xy = torch.nan_to_num(rel_xy, nan=0.0, posinf=2.0, neginf=-2.0)
        anchor_obs = torch.nan_to_num(anchor_obs, nan=0.0, posinf=1.5, neginf=-0.5)
        anchor_obs_vel = torch.nan_to_num(anchor_obs_vel, nan=0.0, posinf=1.0, neginf=-1.0)
        semantic_feat = torch.nan_to_num(semantic_feat, nan=0.0, posinf=4.0, neginf=-4.0)
        box_feat = torch.nan_to_num(box_feat, nan=0.0, posinf=4.0, neginf=-4.0)
        neighbor_feat = torch.nan_to_num(neighbor_feat, nan=0.0, posinf=4.0, neginf=-4.0)
        global_feat = torch.nan_to_num(global_feat, nan=0.0, posinf=4.0, neginf=-4.0)
        tusb_token = torch.nan_to_num(tusb_token, nan=0.0, posinf=4.0, neginf=-4.0)

        if self.cfg.use_dense_points:
            point_tokens = self.point_encoder(obs_points, obs_vis, obs_conf, rel_xy)
        else:
            point_tokens = torch.zeros(
                obs_points.shape[0],
                obs_points.shape[1],
                self.cfg.point_dim,
                device=obs_points.device,
                dtype=obs_points.dtype,
            )
        point_hidden = torch.nan_to_num(self.point_to_hidden(point_tokens), nan=0.0, posinf=8.0, neginf=-8.0)
        anchor_token = self.anchor_proj(torch.cat([anchor_obs, anchor_obs_vel], dim=-1).reshape(obs_points.shape[0], -1))
        semantic_input = semantic_feat if self.cfg.use_semantic_memory else torch.zeros_like(semantic_feat)
        semantic_token = torch.nan_to_num(
            self.semantic_encoder(semantic_input, semantic_id, tusb_token),
            nan=0.0,
            posinf=8.0,
            neginf=-8.0,
        )
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
        anchor_token = torch.nan_to_num(anchor_token, nan=0.0, posinf=8.0, neginf=-8.0)
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

        point_cv, anchor_cv = self._cv_prior(obs_points, anchor_obs, anchor_obs_vel)
        point_affine = self._affine_prior(obs_points, obs_vis, anchor_obs, anchor_cv, future_tokens)
        if self.cfg.use_physics_prior:
            alpha = torch.sigmoid(torch.nan_to_num(self.prior_mix_head(future_tokens), nan=0.0, posinf=8.0, neginf=-8.0)).clamp(0.0, 1.0)
            learned_prior = point_cv + alpha[:, None] * (point_affine - point_cv)
        else:
            learned_prior = torch.zeros_like(point_cv)
            alpha = torch.zeros(future_tokens.shape[0], self.cfg.horizon, 1, device=future_tokens.device, dtype=future_tokens.dtype)

        refined_steps = [self.refiner(point_hidden, future_tokens[:, t], context_tokens) for t in range(self.cfg.horizon)]
        refined_tokens = torch.nan_to_num(torch.stack(refined_steps, dim=2), nan=0.0, posinf=8.0, neginf=-8.0)
        num_hyp = self.cfg.num_hypotheses if self.cfg.use_multimodal else 1
        rel_expand = rel_xy[:, :, None, None, :].expand(-1, -1, self.cfg.horizon, num_hyp, -1)
        mode_embed = self.mode_embed[:num_hyp]
        mode_tokens = torch.nan_to_num(refined_tokens[:, :, :, None, :] + mode_embed[None, None, None, :, :], nan=0.0, posinf=8.0, neginf=-8.0)
        delta = torch.tanh(torch.nan_to_num(self.delta_head(torch.cat([mode_tokens, rel_expand], dim=-1)), nan=0.0, posinf=4.0, neginf=-4.0))
        residual_scale = 0.01 + 0.19 * torch.sigmoid(self.global_residual_scale)
        point_hyp = learned_prior[:, :, :, None, :] + residual_scale * delta
        if self.cfg.use_physics_prior and num_hyp >= 1:
            point_hyp = point_hyp.clone()
            point_hyp[:, :, :, 0, :] = point_cv

        anchor_res = 0.10 * torch.tanh(torch.nan_to_num(self.anchor_res_head(future_tokens), nan=0.0, posinf=4.0, neginf=-4.0))
        anchor_pred = anchor_cv + anchor_res if self.cfg.use_physics_prior else anchor_res
        pooled = torch.nan_to_num(refined_tokens.mean(dim=1), nan=0.0, posinf=8.0, neginf=-8.0)
        hyp_logits_in = torch.cat([pooled.mean(dim=1), object_token, context_summary], dim=-1)
        hypothesis_logits = torch.nan_to_num(self.hypothesis_score_head(hyp_logits_in), nan=0.0, posinf=8.0, neginf=-8.0)
        if not self.cfg.use_multimodal:
            hypothesis_logits = torch.zeros(obs_points.shape[0], 1, device=obs_points.device, dtype=obs_points.dtype)
        hyp_prob = torch.softmax(hypothesis_logits, dim=-1)
        top1_idx = hypothesis_logits.argmax(dim=-1)
        gather_top1 = top1_idx[:, None, None, None, None].expand(-1, point_hyp.shape[1], point_hyp.shape[2], 1, point_hyp.shape[4])
        top1_point = point_hyp.gather(3, gather_top1).squeeze(3)
        weighted_point = (point_hyp * hyp_prob[:, None, None, :, None]).sum(dim=3)
        vis_in = torch.cat(
            [
                refined_tokens,
                learned_prior.norm(dim=-1, keepdim=True),
            ],
            dim=-1,
        )
        visibility_logits = torch.nan_to_num(self.visibility_head(vis_in).squeeze(-1), nan=0.0, posinf=8.0, neginf=-8.0)
        sem_context = torch.cat([future_tokens, semantic_token[:, None, :].expand(-1, self.cfg.horizon, -1)], dim=-1)
        semantic_logits = torch.nan_to_num(self.semantic_head(sem_context), nan=0.0, posinf=8.0, neginf=-8.0)
        mode_logvar = None
        if self.cfg.predict_variance:
            mode_logvar = torch.nan_to_num(
                self.logvar_head(future_tokens[:, :, None, :] + mode_embed[None, None, :, :]).squeeze(-1),
                nan=0.0,
                posinf=2.0,
                neginf=-6.0,
            )
            mode_logvar = mode_logvar.clamp(-6.0, 2.0)
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
            "physics_mix_alpha": alpha.squeeze(-1),
            "mode_logvar": mode_logvar,
        }
