from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
from torch import nn

from stwm.tracewm_v2.constants import STATE_DIM


@dataclass
class GRUTraceBaselineConfig:
    state_mode: str = "multitoken"
    max_tokens: int = 64
    state_dim: int = STATE_DIM
    hidden_dim: int = 384
    num_layers: int = 2
    dropout: float = 0.1
    use_endpoint_head: bool = True


class GRUTraceBaseline(nn.Module):
    def __init__(self, config: GRUTraceBaselineConfig) -> None:
        super().__init__()
        self.config = config

        state_mode = str(config.state_mode).strip().lower()
        if state_mode not in {"legacy_mean5d", "multitoken"}:
            raise ValueError(f"unsupported state_mode={config.state_mode}")
        self.state_mode = state_mode

        if self.state_mode == "legacy_mean5d":
            input_dim = 5
        else:
            input_dim = int(config.max_tokens * config.state_dim)

        gru_dropout = float(config.dropout) if int(config.num_layers) > 1 else 0.0
        self.gru = nn.GRU(
            input_size=int(input_dim),
            hidden_size=int(config.hidden_dim),
            num_layers=int(config.num_layers),
            batch_first=True,
            dropout=gru_dropout,
        )

        self.coord_head = nn.Linear(config.hidden_dim, config.max_tokens * 2)
        self.z_head = nn.Linear(config.hidden_dim, config.max_tokens * 1)
        self.visibility_head = nn.Linear(config.hidden_dim, config.max_tokens * 1)
        self.residual_head = nn.Linear(config.hidden_dim, config.max_tokens * 2)
        self.velocity_head = nn.Linear(config.hidden_dim, config.max_tokens * 2)
        self.endpoint_head = nn.Linear(config.hidden_dim, config.max_tokens * 2) if bool(config.use_endpoint_head) else None

    def _prepare_input(self, state_tokens: torch.Tensor, token_mask: torch.Tensor | None) -> torch.Tensor:
        if state_tokens.ndim != 4:
            raise ValueError(f"expected [B,T,K,D], got {tuple(state_tokens.shape)}")

        bsz, t_len, k_len, d = state_tokens.shape
        if d != int(self.config.state_dim):
            raise ValueError(f"state_dim mismatch: got {d}, expected {self.config.state_dim}")
        if k_len > int(self.config.max_tokens):
            raise ValueError(f"token length {k_len} exceeds max_tokens={self.config.max_tokens}")

        if self.state_mode == "legacy_mean5d":
            x5 = state_tokens[..., :5]
            if token_mask is None:
                return x5.mean(dim=2)

            if token_mask.ndim != 2 or token_mask.shape != (bsz, k_len):
                raise ValueError(f"token_mask shape mismatch: got {tuple(token_mask.shape)}, expected {(bsz, k_len)}")
            mask = token_mask[:, None, :, None].to(dtype=state_tokens.dtype)
            denom = mask.sum(dim=2).clamp_min(1.0)
            return (x5 * mask).sum(dim=2) / denom

        if k_len < int(self.config.max_tokens):
            pad = torch.zeros(
                (bsz, t_len, int(self.config.max_tokens - k_len), d),
                dtype=state_tokens.dtype,
                device=state_tokens.device,
            )
            padded = torch.cat([state_tokens, pad], dim=2)
        else:
            padded = state_tokens
        return padded.reshape(bsz, t_len, int(self.config.max_tokens * d))

    def forward(self, state_tokens: torch.Tensor, token_mask: torch.Tensor | None = None) -> Dict[str, torch.Tensor]:
        bsz, t_len, k_len, _ = state_tokens.shape
        x = self._prepare_input(state_tokens, token_mask)

        hidden_seq, _ = self.gru(x)

        coord = self.coord_head(hidden_seq).reshape(bsz, t_len, self.config.max_tokens, 2)[:, :, :k_len]
        z = self.z_head(hidden_seq).reshape(bsz, t_len, self.config.max_tokens, 1)[:, :, :k_len]
        vis_logit = self.visibility_head(hidden_seq).reshape(bsz, t_len, self.config.max_tokens, 1)[:, :, :k_len]
        residual = self.residual_head(hidden_seq).reshape(bsz, t_len, self.config.max_tokens, 2)[:, :, :k_len]
        velocity = self.velocity_head(hidden_seq).reshape(bsz, t_len, self.config.max_tokens, 2)[:, :, :k_len]

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
            "hidden": hidden_seq,
        }

        if self.endpoint_head is not None:
            endpoint = self.endpoint_head(hidden_seq[:, -1]).reshape(bsz, self.config.max_tokens, 2)[:, :k_len]
            out["endpoint"] = endpoint

        return out

    def parameter_count(self) -> int:
        return int(sum(p.numel() for p in self.parameters()))
