from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class RetrievalMemoryStateV42:
    keys: torch.Tensor
    values: torch.Tensor
    valid_mask: torch.Tensor


class RetrievalReconnectMemoryV42(nn.Module):
    """Single retrieval/reconnect memory for STWM V4.2.

    This module intentionally keeps one compact memory path that supports:
    reappearance reconnect, same-category disambiguation, and short-horizon
    query persistence.
    """

    def __init__(
        self,
        token_dim: int,
        *,
        memory_slots: int = 32,
        momentum: float = 0.95,
        temperature: float = 0.07,
    ) -> None:
        super().__init__()
        self.token_dim = int(token_dim)
        self.memory_slots = int(memory_slots)
        self.momentum = float(momentum)
        self.temperature = float(temperature)

        self.key_proj = nn.Linear(self.token_dim, self.token_dim)
        self.val_proj = nn.Linear(self.token_dim, self.token_dim)
        self.gate_proj = nn.Sequential(
            nn.Linear(self.token_dim, self.token_dim),
            nn.GELU(),
            nn.Linear(self.token_dim, 1),
        )
        self.fuse_norm = nn.LayerNorm(self.token_dim)

    def init_state(
        self,
        batch_size: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> RetrievalMemoryStateV42:
        keys = torch.zeros(batch_size, self.memory_slots, self.token_dim, device=device, dtype=dtype)
        values = torch.zeros_like(keys)
        valid_mask = torch.zeros(batch_size, self.memory_slots, device=device, dtype=torch.bool)
        return RetrievalMemoryStateV42(keys=keys, values=values, valid_mask=valid_mask)

    def forward(
        self,
        tokens: torch.Tensor,
        *,
        memory_state: RetrievalMemoryStateV42 | None = None,
        update_memory: bool = True,
    ) -> tuple[torch.Tensor, RetrievalMemoryStateV42, dict[str, Any]]:
        if tokens.ndim != 3:
            raise ValueError("tokens must be [B, N, D]")
        bsz, _, token_dim = tokens.shape
        if token_dim != self.token_dim:
            raise ValueError(f"token_dim mismatch: expected {self.token_dim}, got {token_dim}")

        if memory_state is None:
            memory_state = self.init_state(
                batch_size=bsz,
                device=tokens.device,
                dtype=tokens.dtype,
            )

        q = F.normalize(self.key_proj(tokens), dim=-1)
        k = F.normalize(memory_state.keys, dim=-1)

        logits = torch.einsum("bnd,bmd->bnm", q, k) / max(1e-4, self.temperature)
        invalid = ~memory_state.valid_mask.unsqueeze(1)
        logits = logits.masked_fill(invalid, -1e4)

        attn = torch.softmax(logits, dim=-1)
        retrieved = torch.einsum("bnm,bmd->bnd", attn, memory_state.values)

        gate = torch.sigmoid(self.gate_proj(tokens))
        fused = self.fuse_norm(tokens + gate * self.val_proj(retrieved))

        next_state = memory_state
        if update_memory:
            with torch.no_grad():
                pooled = fused.detach().mean(dim=1)
                pooled_key = F.normalize(self.key_proj(pooled), dim=-1)
                pooled_val = self.val_proj(pooled)

                new_keys = torch.cat([pooled_key.unsqueeze(1), memory_state.keys[:, :-1, :]], dim=1)
                new_values = torch.cat([pooled_val.unsqueeze(1), memory_state.values[:, :-1, :]], dim=1)
                new_valid = torch.cat(
                    [torch.ones(bsz, 1, device=tokens.device, dtype=torch.bool), memory_state.valid_mask[:, :-1]],
                    dim=1,
                )

                # Momentum blend on existing valid slots to reduce memory churn.
                blended_values = torch.where(
                    memory_state.valid_mask.unsqueeze(-1),
                    self.momentum * new_values + (1.0 - self.momentum) * memory_state.values,
                    new_values,
                )
                next_state = RetrievalMemoryStateV42(
                    keys=new_keys.detach(),
                    values=blended_values.detach(),
                    valid_mask=new_valid.detach(),
                )

        attn_entropy = -torch.sum(attn * torch.log(attn.clamp(min=1e-8)), dim=-1)
        memory_diag = {
            "memory_gate_mean": float(gate.mean().detach().cpu()),
            "memory_gate_std": float(gate.std(unbiased=False).detach().cpu()),
            "retrieval_entropy": float(attn_entropy.mean().detach().cpu()),
            "valid_slots_mean": float(memory_state.valid_mask.float().mean().detach().cpu()),
        }

        return fused, next_state, memory_diag
