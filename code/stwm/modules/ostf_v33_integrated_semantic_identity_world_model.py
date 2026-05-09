from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from stwm.modules.ostf_external_gt_world_model_v30 import OSTFExternalGTConfigV30, OSTFExternalGTWorldModelV30


def build_v30_from_checkpoint(checkpoint_path: str | Path, *, map_location: str | torch.device = "cpu") -> tuple[OSTFExternalGTWorldModelV30, dict[str, Any]]:
    ckpt = torch.load(checkpoint_path, map_location=map_location)
    args = ckpt.get("args", {}) if isinstance(ckpt, dict) else {}
    cfg = OSTFExternalGTConfigV30(
        horizon=int(args.get("horizon", 32)),
        point_dim=int(args.get("point_dim", 2)),
        hidden_dim=int(args.get("hidden_dim", 192)),
        point_token_dim=max(64, int(args.get("hidden_dim", 192)) // 2),
        num_layers=int(args.get("layers", 3)),
        num_heads=int(args.get("heads", 6)),
        learned_modes=int(args.get("learned_modes", 4)),
        damped_gamma=float(args.get("damped_gamma", 0.0)),
        use_semantic=not bool(args.get("wo_semantic", False)),
        point_dropout=float(args.get("point_dropout", 0.0)),
        density_aware_pooling=str(args.get("density_aware_pooling", "mean")),
        density_inducing_tokens=int(args.get("density_inducing_tokens", 16)),
        density_motion_topk=int(args.get("density_motion_topk", 128)),
        density_token_dropout=float(args.get("density_token_dropout", 0.0)),
    )
    model = OSTFExternalGTWorldModelV30(cfg)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    incompatible = model.load_state_dict(state, strict=False)
    bad_missing = [k for k in incompatible.missing_keys if not k.startswith("density_pooler.")]
    if bad_missing or incompatible.unexpected_keys:
        raise RuntimeError(f"Unsafe V30 checkpoint load mismatch: missing={bad_missing}, unexpected={incompatible.unexpected_keys}")
    model.v33_1_checkpoint_missing_keys = list(incompatible.missing_keys)  # type: ignore[attr-defined]
    model.v33_1_checkpoint_unexpected_keys = list(incompatible.unexpected_keys)  # type: ignore[attr-defined]
    return model, args


class IntegratedSemanticIdentityWorldModelV331(nn.Module):
    """Frozen V30 trajectory backbone plus horizon-dependent identity heads."""

    def __init__(
        self,
        v30_checkpoint_path: str | Path,
        *,
        identity_dim: int = 64,
        use_observed_instance_context: bool = False,
        instance_buckets: int = 8192,
    ) -> None:
        super().__init__()
        self.v30, self.v30_args = build_v30_from_checkpoint(v30_checkpoint_path)
        for p in self.v30.parameters():
            p.requires_grad_(False)
        self.v30.eval()
        self.use_observed_instance_context = bool(use_observed_instance_context)
        hidden = int(self.v30.cfg.hidden_dim)
        point_dim = int(self.v30.cfg.point_dim)
        self.instance_embed = nn.Embedding(instance_buckets, 16)
        context_dim = hidden * 2 + point_dim + 2 + (16 if self.use_observed_instance_context else 0)
        self.identity_trunk = nn.Sequential(
            nn.LayerNorm(context_dim),
            nn.Linear(context_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        self.same_instance_head = nn.Linear(hidden, 1)
        self.identity_embedding_head = nn.Linear(hidden, identity_dim)
        self.identity_uncertainty_head = nn.Linear(hidden, 1)
        self.visibility_delta_head = nn.Linear(hidden, 1)

    @property
    def v30_backbone_frozen(self) -> bool:
        return not any(p.requires_grad for p in self.v30.parameters())

    def _v30_features(
        self,
        *,
        obs_points: torch.Tensor,
        obs_vis: torch.Tensor,
        obs_conf: torch.Tensor,
        semantic_id: torch.Tensor | None,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        # This mirrors the frozen V30 forward path and exposes point_token/step_hidden.
        v30 = self.v30
        obs_points = torch.nan_to_num(obs_points.float(), nan=0.0, posinf=1e6, neginf=-1e6)
        obs_vis_f = obs_vis.float()
        obs_conf = torch.nan_to_num(obs_conf.float(), nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
        priors = v30.prior_modes(obs_points, obs_vis)
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
        point_token = v30.point_encoder(torch.cat([flat, anchor_feat], dim=-1))
        valid_point = obs_vis.any(dim=-1).float()
        pooled, _ = v30.density_pooler(point_token, valid_point, obs_points, obs_vis)
        if semantic_id is None:
            semantic_id = torch.full((obs_points.shape[0],), -1, device=obs_points.device, dtype=torch.long)
        sem_idx = semantic_id.clamp_min(0) % v30.cfg.semantic_buckets
        sem = v30.semantic_embed(sem_idx)
        if not v30.cfg.use_semantic:
            sem = torch.zeros_like(sem)
        obj = v30.object_encoder(torch.cat([pooled, sem], dim=-1))
        step_tokens = obj[:, None, :] + v30.time_embed[None, :, :]
        step_hidden = v30.rollout(step_tokens)
        return {
            "point_token": point_token,
            "step_hidden": step_hidden,
            "last_visible": last_visible,
            "center": center,
            "spread": spread,
        }, priors

    def forward(
        self,
        *,
        obs_points: torch.Tensor,
        obs_vis: torch.Tensor,
        obs_conf: torch.Tensor,
        semantic_id: torch.Tensor | None = None,
        point_to_instance_id: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        with torch.no_grad():
            v30_out = self.v30(obs_points=obs_points, obs_vis=obs_vis, obs_conf=obs_conf, semantic_id=semantic_id)
            features, _ = self._v30_features(obs_points=obs_points, obs_vis=obs_vis, obs_conf=obs_conf, semantic_id=semantic_id)
        point_token = features["point_token"].detach()
        step_hidden = features["step_hidden"].detach()
        point_pred = v30_out["point_pred"].detach()
        rel_pred = ((point_pred - features["last_visible"][:, :, None, :]) / features["spread"].squeeze(2)[:, :, None, :]).clamp(-16.0, 16.0)
        b, m, h, _ = point_pred.shape
        chunks = [
            point_token[:, :, None, :].expand(-1, -1, h, -1),
            step_hidden[:, None, :, :].expand(-1, m, -1, -1),
            rel_pred,
            v30_out["visibility_logits"].detach().sigmoid()[:, :, :, None],
            obs_vis.float()[:, :, -1:, None].expand(-1, -1, h, -1),
        ]
        if self.use_observed_instance_context:
            if point_to_instance_id is None:
                point_to_instance_id = torch.full((b, m), -1, device=point_pred.device, dtype=torch.long)
            iid = point_to_instance_id.clamp_min(0) % self.instance_embed.num_embeddings
            chunks.append(self.instance_embed(iid)[:, :, None, :].expand(-1, -1, h, -1))
        hidden = self.identity_trunk(torch.cat(chunks, dim=-1))
        same_instance_logits = self.same_instance_head(hidden).squeeze(-1)
        visibility_logits = v30_out["visibility_logits"].detach() + self.visibility_delta_head(hidden).squeeze(-1)
        identity_embedding = self.identity_embedding_head(hidden)
        identity_uncertainty = self.identity_uncertainty_head(hidden).squeeze(-1)
        return {
            "point_hypotheses": v30_out["point_hypotheses"].detach(),
            "point_pred": point_pred,
            "top1_point_pred": v30_out["top1_point_pred"].detach(),
            "visibility_logits": visibility_logits,
            "frozen_v30_visibility_logits": v30_out["visibility_logits"].detach(),
            "same_instance_logits": same_instance_logits,
            "identity_embedding": identity_embedding,
            "identity_uncertainty": identity_uncertainty,
            "semantic_logits": v30_out.get("semantic_logits", torch.empty((*same_instance_logits.shape, 0), device=point_pred.device)).detach(),
        }
