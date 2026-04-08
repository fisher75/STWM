from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
from torch import nn


@dataclass
class TraceCausalTransformerConfig:
    state_dim: int = 8
    d_model: int = 1152
    num_layers: int = 13
    num_heads: int = 16
    ff_mult: int = 4
    dropout: float = 0.1
    max_time: int = 64
    max_tokens: int = 256
    use_endpoint_head: bool = True


def build_tracewm_v2_config(preset: str = "prototype_220m") -> TraceCausalTransformerConfig:
    p = str(preset).strip().lower()
    if p in {"debug", "debug_small", "tiny"}:
        return TraceCausalTransformerConfig(
            state_dim=8,
            d_model=256,
            num_layers=4,
            num_heads=8,
            ff_mult=4,
            dropout=0.1,
            max_time=64,
            max_tokens=128,
            use_endpoint_head=True,
        )
    if p in {"base", "medium"}:
        return TraceCausalTransformerConfig(
            state_dim=8,
            d_model=768,
            num_layers=10,
            num_heads=12,
            ff_mult=4,
            dropout=0.1,
            max_time=64,
            max_tokens=192,
            use_endpoint_head=True,
        )
    return TraceCausalTransformerConfig()


def estimate_parameter_count(cfg: TraceCausalTransformerConfig) -> int:
    h = int(cfg.d_model)
    l = int(cfg.num_layers)
    ff = int(cfg.ff_mult) * h

    attn_block = (4 * h * h) + (4 * h)
    ff_block = (h * ff) + ff + (ff * h) + h
    ln_block = 4 * h
    transformer = l * (attn_block + ff_block + ln_block)

    embeddings = (cfg.state_dim * h + h) + (cfg.max_time * h) + (cfg.max_tokens * h)
    heads = (h * 2 + 2) + (h * 1 + 1) + (h * 2 + 2) + (h * 2 + 2) + (h * 1 + 1)
    endpoint = (h * 2 + 2) if cfg.use_endpoint_head else 0
    return int(transformer + embeddings + heads + endpoint)


class TraceCausalTransformer(nn.Module):
    def __init__(self, config: TraceCausalTransformerConfig) -> None:
        super().__init__()
        if config.d_model % config.num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")

        self.config = config
        self.input_proj = nn.Linear(config.state_dim, config.d_model)
        self.time_embed = nn.Embedding(config.max_time, config.d_model)
        self.token_embed = nn.Embedding(config.max_tokens, config.d_model)

        layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.num_heads,
            dim_feedforward=config.d_model * config.ff_mult,
            dropout=config.dropout,
            batch_first=True,
            activation="gelu",
        )
        self.backbone = nn.TransformerEncoder(layer, num_layers=config.num_layers)
        self.norm = nn.LayerNorm(config.d_model)

        self.coord_head = nn.Linear(config.d_model, 2)
        self.z_head = nn.Linear(config.d_model, 1)
        self.visibility_head = nn.Linear(config.d_model, 1)
        self.residual_head = nn.Linear(config.d_model, 2)
        self.velocity_head = nn.Linear(config.d_model, 2)
        self.endpoint_head = nn.Linear(config.d_model, 2) if config.use_endpoint_head else None

    def _causal_mask(self, t_len: int, k_len: int, device: torch.device) -> torch.Tensor:
        # Flattened token index: s = t * K + k; future mask only blocks future timesteps.
        time_ids = torch.arange(t_len, device=device).repeat_interleave(k_len)
        return (time_ids.unsqueeze(0) > time_ids.unsqueeze(1)).to(torch.bool)

    def forward(self, state_tokens: torch.Tensor, token_mask: torch.Tensor | None = None) -> Dict[str, torch.Tensor]:
        if state_tokens.ndim != 4:
            raise ValueError(f"expected [B,T,K,D], got {tuple(state_tokens.shape)}")

        bsz, t_len, k_len, d = state_tokens.shape
        if d != self.config.state_dim:
            raise ValueError(f"state_dim mismatch: got {d}, expected {self.config.state_dim}")
        if t_len > self.config.max_time:
            raise ValueError(f"time length {t_len} exceeds max_time={self.config.max_time}")
        if k_len > self.config.max_tokens:
            raise ValueError(f"token length {k_len} exceeds max_tokens={self.config.max_tokens}")

        x = self.input_proj(state_tokens)

        t_ids = torch.arange(t_len, device=x.device).view(1, t_len, 1)
        k_ids = torch.arange(k_len, device=x.device).view(1, 1, k_len)
        x = x + self.time_embed(t_ids) + self.token_embed(k_ids)

        x = x.reshape(bsz, t_len * k_len, self.config.d_model)

        key_padding_mask = None
        if token_mask is not None:
            if token_mask.ndim != 2 or token_mask.shape != (bsz, k_len):
                raise ValueError(f"token_mask shape mismatch: got {tuple(token_mask.shape)}, expected {(bsz, k_len)}")
            expanded = token_mask[:, None, :].expand(bsz, t_len, k_len).reshape(bsz, t_len * k_len)
            key_padding_mask = ~expanded.bool()

        hidden = self.backbone(
            x,
            mask=self._causal_mask(t_len=t_len, k_len=k_len, device=x.device),
            src_key_padding_mask=key_padding_mask,
        )
        hidden = self.norm(hidden)
        hidden = hidden.reshape(bsz, t_len, k_len, self.config.d_model)

        coord = self.coord_head(hidden)
        z = self.z_head(hidden)
        vis_logit = self.visibility_head(hidden)
        residual = self.residual_head(hidden)
        velocity = self.velocity_head(hidden)

        pred_state = torch.cat(
            [
                coord,
                z,
                torch.sigmoid(vis_logit),
                velocity,
                residual,
            ],
            dim=-1,
        )

        out: Dict[str, torch.Tensor] = {
            "coord": coord,
            "z": z,
            "vis_logit": vis_logit,
            "residual": residual,
            "velocity": velocity,
            "pred_state": pred_state,
            "hidden": hidden,
        }

        if self.endpoint_head is not None:
            out["endpoint"] = self.endpoint_head(hidden[:, -1])

        return out

    def parameter_count(self) -> int:
        return int(sum(p.numel() for p in self.parameters()))
