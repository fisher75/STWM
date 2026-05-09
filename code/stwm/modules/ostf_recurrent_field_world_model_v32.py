from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class OSTFRecurrentFieldConfigV32:
    obs_len: int = 8
    horizon: int = 32
    point_dim: int = 2
    hidden_dim: int = 192
    point_token_dim: int = 96
    field_layers: int = 2
    num_heads: int = 6
    learned_modes: int = 4
    damped_gamma: float = 0.0
    use_semantic: bool = True
    semantic_buckets: int = 8192
    semantic_dim: int = 192
    predict_logvar: bool = True
    point_dropout: float = 0.0
    field_attention_mode: str = "full"
    induced_tokens: int = 16
    use_global_motion_prior: bool = True
    disable_global_motion_prior: bool = False
    disable_field_interaction: bool = False
    disable_semantic_context: bool = False
    detach_recurrent_position: bool = False


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


class InducedFieldBlockV32(nn.Module):
    """O(M*K) field mixer for dense point sets.

    M512 recurrent full attention across every future step is expensive. This
    block lets learned inducing tokens summarize point interactions without
    collapsing the rollout state into a single object token.
    """

    def __init__(self, hidden_dim: int, num_heads: int, induced_tokens: int) -> None:
        super().__init__()
        self.inducing = nn.Parameter(torch.randn(1, induced_tokens, hidden_dim) * 0.02)
        self.to_induced = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True, dropout=0.0)
        self.to_points = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True, dropout=0.0)
        self.point_ff = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.induced_ff = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.norm_points = nn.LayerNorm(hidden_dim)
        self.norm_induced = nn.LayerNorm(hidden_dim)
        self.last_attention_entropy: torch.Tensor | None = None

    def forward(self, point_tokens: torch.Tensor, context_tokens: torch.Tensor) -> torch.Tensor:
        b = point_tokens.shape[0]
        induced = self.inducing.expand(b, -1, -1)
        kv = torch.cat([context_tokens, point_tokens], dim=1)
        induced_attn, weights = self.to_induced(induced, kv, kv, need_weights=True)
        induced = self.norm_induced(induced + induced_attn)
        induced = induced + self.induced_ff(induced)
        point_attn, weights2 = self.to_points(point_tokens, torch.cat([context_tokens, induced], dim=1), torch.cat([context_tokens, induced], dim=1), need_weights=True)
        point_tokens = self.norm_points(point_tokens + point_attn)
        point_tokens = point_tokens + self.point_ff(point_tokens)
        probs = weights2.clamp_min(1e-8)
        self.last_attention_entropy = (-(probs * probs.log()).sum(dim=-1).mean()).detach()
        return point_tokens


class FullFieldBlockV32(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, layers: int) -> None:
        super().__init__()
        if layers <= 0:
            self.encoder: nn.Module = nn.Identity()
        else:
            layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                batch_first=True,
                dropout=0.0,
                activation="gelu",
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.last_attention_entropy: torch.Tensor | None = None

    def forward(self, point_tokens: torch.Tensor, context_tokens: torch.Tensor) -> torch.Tensor:
        tokens = torch.cat([context_tokens, point_tokens], dim=1)
        out = self.encoder(tokens)
        return out[:, context_tokens.shape[1] :]


class OSTFRecurrentFieldWorldModelV32(nn.Module):
    """Recurrent field dynamics for object-dense trajectory forecasting.

    Unlike V30, the M point tokens are never pooled into the rollout state.
    Unlike V31, future tokens are updated step-by-step, with point positions
    fed back into the next field dynamics step.
    """

    def __init__(self, cfg: OSTFRecurrentFieldConfigV32) -> None:
        super().__init__()
        self.cfg = cfg
        point_in_dim = cfg.obs_len * (cfg.point_dim + 2) + 10
        self.point_encoder = nn.Sequential(
            nn.LayerNorm(point_in_dim),
            nn.Linear(point_in_dim, cfg.point_token_dim),
            nn.GELU(),
            nn.Linear(cfg.point_token_dim, cfg.hidden_dim),
            nn.GELU(),
        )
        self.semantic_embed = nn.Embedding(cfg.semantic_buckets, cfg.semantic_dim)
        self.semantic_proj = nn.Sequential(nn.LayerNorm(cfg.semantic_dim), nn.Linear(cfg.semantic_dim, cfg.hidden_dim), nn.GELU())
        self.global_token = nn.Parameter(torch.randn(1, 1, cfg.hidden_dim) * 0.02)
        self.time_embed = nn.Parameter(torch.randn(cfg.horizon, cfg.hidden_dim) * 0.02)
        self.pos_vel_embed = nn.Sequential(
            nn.LayerNorm(4),
            nn.Linear(4, cfg.hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
        )
        self.global_motion_head = nn.Sequential(
            nn.LayerNorm(cfg.hidden_dim),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.hidden_dim, cfg.point_dim),
        )
        self.global_update = nn.Sequential(
            nn.LayerNorm(cfg.hidden_dim * 2),
            nn.Linear(cfg.hidden_dim * 2, cfg.hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
        )
        self.semantic_film = nn.Sequential(nn.LayerNorm(cfg.hidden_dim), nn.Linear(cfg.hidden_dim, cfg.hidden_dim * 2))
        self.context_proj = nn.Sequential(nn.LayerNorm(cfg.hidden_dim), nn.Linear(cfg.hidden_dim, cfg.hidden_dim), nn.GELU())
        if cfg.disable_field_interaction:
            self.field_block: nn.Module = nn.Identity()
        elif cfg.field_attention_mode == "induced":
            self.field_block = InducedFieldBlockV32(cfg.hidden_dim, cfg.num_heads, cfg.induced_tokens)
        else:
            self.field_block = FullFieldBlockV32(cfg.hidden_dim, cfg.num_heads, cfg.field_layers)
        self.point_update = nn.Sequential(
            nn.LayerNorm(cfg.hidden_dim * 2),
            nn.Linear(cfg.hidden_dim * 2, cfg.hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
        )
        self.mode_embed = nn.Parameter(torch.randn(cfg.learned_modes, cfg.hidden_dim) * 0.02)
        self.residual_head = nn.Sequential(
            nn.LayerNorm(cfg.hidden_dim + cfg.point_dim + 2),
            nn.Linear(cfg.hidden_dim + cfg.point_dim + 2, cfg.hidden_dim),
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
        nn.init.zeros_(self.global_motion_head[-1].weight)
        nn.init.zeros_(self.global_motion_head[-1].bias)

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

    def _encode_points(
        self,
        obs_points: torch.Tensor,
        obs_vis: torch.Tensor,
        obs_conf: torch.Tensor,
        priors: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        last_visible = priors["last_visible_copy"][:, :, 0]
        center = last_visible.median(dim=1, keepdim=True).values
        spread = (last_visible - center).abs().flatten(1).median(dim=1).values.view(-1, 1, 1, 1).clamp_min(16.0)
        norm_obs = (obs_points - center[:, :, None, :]) / spread
        obs_vis_f = obs_vis.float()
        obs_conf = torch.nan_to_num(obs_conf.float(), nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
        velocity = torch.zeros_like(norm_obs)
        velocity[:, :, 1:] = norm_obs[:, :, 1:] - norm_obs[:, :, :-1]
        acceleration = torch.zeros_like(norm_obs)
        acceleration[:, :, 2:] = velocity[:, :, 2:] - velocity[:, :, 1:-1]
        log_scale = torch.log(spread.squeeze(2).clamp_min(1.0)).expand(-1, obs_points.shape[1], -1)
        anchor_feat = torch.cat(
            [
                ((last_visible - center) / spread.squeeze(2)),
                velocity[:, :, -1],
                acceleration[:, :, -1],
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
            keep = torch.where(keep.sum(dim=1, keepdim=True) <= 0, torch.ones_like(keep), keep)
            valid_point = valid_point * keep
            point_token = point_token * keep[..., None]
        return point_token, valid_point, last_visible, center, spread, velocity[:, :, -1] * spread.squeeze(2)

    def _mix_field(self, z: torch.Tensor, global_ctx: torch.Tensor, sem: torch.Tensor) -> torch.Tensor:
        if self.cfg.disable_field_interaction:
            return z
        context = torch.cat([global_ctx[:, None, :], sem[:, None, :]], dim=1)
        return self.field_block(z, context)

    def forward(
        self,
        *,
        obs_points: torch.Tensor,
        obs_vis: torch.Tensor,
        obs_conf: torch.Tensor,
        semantic_id: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        obs_points = torch.nan_to_num(obs_points.float(), nan=0.0, posinf=1e6, neginf=-1e6)
        obs_vis = obs_vis.bool()
        priors = self.prior_modes(obs_points, obs_vis)
        z, valid_point, last_visible, center, spread, init_vel = self._encode_points(obs_points, obs_vis, obs_conf, priors)
        b, m, _ = z.shape
        if semantic_id is None:
            semantic_id = torch.full((b,), -1, device=obs_points.device, dtype=torch.long)
        sem_idx = semantic_id.clamp_min(0) % self.cfg.semantic_buckets
        sem = self.semantic_proj(self.semantic_embed(sem_idx))
        if not self.cfg.use_semantic or self.cfg.disable_semantic_context:
            sem = torch.zeros_like(sem)
        gamma_beta = self.semantic_film(sem)
        gamma, beta = gamma_beta.chunk(2, dim=-1)
        valid = valid_point.float()
        denom = valid.sum(dim=1, keepdim=True).clamp_min(1.0)
        global_ctx = self.global_token.expand(b, -1, -1).squeeze(1) + (z * valid[..., None]).sum(dim=1) / denom
        global_ctx = self.context_proj(global_ctx)
        current_pos = last_visible
        current_vel = init_vel
        base = priors["last_visible_copy"]
        rel_layout = ((last_visible - center) / spread.squeeze(2)).clamp(-12.0, 12.0)
        scale = torch.sigmoid(self.residual_scale) * spread.squeeze(2)
        prior_names = ["last_visible_copy", "last_observed_copy", "visibility_aware_cv", "visibility_aware_damped", "median_object_anchor_copy"]
        step_modes: list[torch.Tensor] = []
        step_logits: list[torch.Tensor] = []
        step_vis: list[torch.Tensor] = []
        step_sem: list[torch.Tensor] = []
        step_logvar: list[torch.Tensor] = []
        state_norms: list[torch.Tensor] = []
        delta_norms: list[torch.Tensor] = []
        global_norms: list[torch.Tensor] = []
        prev_pred = current_pos
        for h in range(self.cfg.horizon):
            pos_rel = ((current_pos - center.squeeze(1)[:, None, :]) / spread.squeeze(2)).clamp(-12.0, 12.0)
            vel_rel = (current_vel / spread.squeeze(2)).clamp(-12.0, 12.0)
            step_input = z + self.pos_vel_embed(torch.cat([pos_rel, vel_rel], dim=-1)) + self.time_embed[h][None, None, :]
            step_input = step_input * (1.0 + 0.1 * torch.tanh(gamma[:, None, :])) + 0.1 * beta[:, None, :]
            z_mix = self._mix_field(step_input, global_ctx, sem)
            z_next = z + self.point_update(torch.cat([z, z_mix], dim=-1))
            g_delta = (z_next * valid[..., None]).sum(dim=1) / denom
            global_ctx = global_ctx + self.global_update(torch.cat([global_ctx, g_delta], dim=-1))
            if self.cfg.use_global_motion_prior and not self.cfg.disable_global_motion_prior:
                global_step = self.global_motion_head(global_ctx)[:, None, :] * spread.squeeze(2)
            else:
                global_step = torch.zeros_like(current_pos)
            priors_h = torch.stack([priors[name][:, :, h] for name in prior_names], dim=2)
            learned = []
            for mode in range(self.cfg.learned_modes):
                hidden = z_next + self.mode_embed[None, None, mode, :]
                inp = torch.cat([hidden, rel_layout, vel_rel], dim=-1)
                delta = self.residual_head(inp) * scale
                learned.append(base[:, :, h] + global_step + delta)
            learned_stack = torch.stack(learned, dim=2) if learned else priors_h[:, :, :0]
            modes_h = torch.cat([priors_h, learned_stack], dim=2)
            logits_h = self.mode_logits_head(z_next)
            weights_h = torch.softmax(logits_h, dim=-1)
            pred_h = (modes_h * weights_h[..., None]).sum(dim=2)
            vis_in = torch.cat(
                [
                    z_next,
                    obs_vis.float()[:, :, -1:],
                    torch.nan_to_num(obs_conf.float(), nan=0.0, posinf=1.0, neginf=0.0)[:, :, -1:],
                ],
                dim=-1,
            )
            step_modes.append(modes_h)
            step_logits.append(logits_h)
            step_vis.append(self.visibility_head(vis_in).squeeze(-1))
            step_sem.append(self.semantic_head(z_next))
            step_logvar.append(self.logvar_head(z_next).squeeze(-1))
            state_norms.append(z_next.detach().norm(dim=-1).mean())
            delta_norms.append((z_next - z).detach().norm(dim=-1).mean())
            global_norms.append(global_step.detach().norm(dim=-1).mean())
            current_vel = pred_h - prev_pred
            current_pos = pred_h.detach() if self.cfg.detach_recurrent_position else pred_h
            prev_pred = current_pos
            z = z_next
        point_hypotheses = torch.stack(step_modes, dim=2)
        hypothesis_logits = torch.stack(step_logits, dim=2)
        weights = torch.softmax(hypothesis_logits, dim=-1)
        weighted = (point_hypotheses * weights[..., None]).sum(dim=3)
        top_idx = hypothesis_logits.argmax(dim=-1)
        gather = top_idx[..., None, None].expand(-1, -1, -1, 1, self.cfg.point_dim)
        top1 = point_hypotheses.gather(3, gather).squeeze(3)
        visibility_logits = torch.stack(step_vis, dim=2)
        semantic_logits = torch.stack(step_sem, dim=2)
        out = {
            "point_hypotheses": point_hypotheses,
            "hypothesis_logits": hypothesis_logits,
            "point_pred": weighted,
            "top1_point_pred": top1,
            "visibility_logits": visibility_logits,
            "semantic_logits": semantic_logits,
            "prior_names": prior_names,
            "num_prior_modes": torch.tensor(len(prior_names), device=obs_points.device),
            "point_encoder_activation_norm": z.detach().norm(dim=-1).mean(),
            "field_state_norm_per_step": torch.stack(state_norms),
            "recurrent_delta_norm_per_step": torch.stack(delta_norms),
            "global_motion_norm_per_step": torch.stack(global_norms),
            "field_state_norm": torch.stack(state_norms).mean(),
            "recurrent_delta_norm": torch.stack(delta_norms).mean(),
            "global_motion_norm": torch.stack(global_norms).mean(),
            "point_valid_ratio": valid_point.detach().mean(),
            "actual_m_points": torch.tensor(m, device=obs_points.device),
            "main_rollout_state_is_field": torch.tensor(True, device=obs_points.device),
            "recurrent_loop_steps": torch.tensor(self.cfg.horizon, device=obs_points.device),
            "global_motion_prior_active": torch.tensor(
                bool(self.cfg.use_global_motion_prior and not self.cfg.disable_global_motion_prior),
                device=obs_points.device,
            ),
            "disable_field_interaction": torch.tensor(bool(self.cfg.disable_field_interaction), device=obs_points.device),
            "field_attention_mode_induced": torch.tensor(self.cfg.field_attention_mode == "induced", device=obs_points.device),
        }
        entropy = getattr(self.field_block, "last_attention_entropy", None)
        if entropy is not None:
            out["point_interaction_attention_entropy"] = entropy
        if self.cfg.predict_logvar:
            out["mode_logvar"] = torch.stack(step_logvar, dim=2)
        return out
