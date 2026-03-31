from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn


@dataclass
class StateTokenizerV42Output:
    state_tokens: torch.Tensor
    token_time_attention: torch.Tensor
    objectness: torch.Tensor
    diagnostics: dict[str, Any]


class ObjectBiasedStateTokenizerV42(nn.Module):
    """Object-biased learned tokenizer for STWM V4.2.

    The tokenizer keeps dense trajectory field as the primary substrate, then
    biases token assignment toward object-like regions using weak priors.
    Final tokens remain fully learnable and do not require teacher objects
    during inference.
    """

    def __init__(
        self,
        trace_dim: int,
        semantic_dim: int,
        prior_dim: int,
        *,
        hidden_size: int = 1024,
        num_tokens: int = 16,
        objectness_bias_scale: float = 1.5,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.trace_dim = int(trace_dim)
        self.semantic_dim = int(semantic_dim)
        self.prior_dim = int(prior_dim)
        self.hidden_size = int(hidden_size)
        self.num_tokens = int(num_tokens)
        self.objectness_bias_scale = float(objectness_bias_scale)

        input_dim = self.trace_dim + self.semantic_dim + self.prior_dim
        self.input_proj = nn.Linear(input_dim, self.hidden_size)
        self.input_norm = nn.LayerNorm(self.hidden_size)
        self.input_mlp = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_size, self.hidden_size),
        )
        self.objectness_head = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.GELU(),
            nn.Linear(self.hidden_size // 2, 1),
        )

        self.token_queries = nn.Parameter(torch.randn(self.num_tokens, self.hidden_size) * 0.02)
        self.output_norm = nn.LayerNorm(self.hidden_size)

    def forward(
        self,
        trace_features: torch.Tensor,
        semantic_features: torch.Tensor,
        *,
        prior_features: torch.Tensor | None = None,
        teacher_objectness: torch.Tensor | None = None,
    ) -> StateTokenizerV42Output:
        if trace_features.ndim != 3 or semantic_features.ndim != 3:
            raise ValueError("trace_features and semantic_features must be [B, T, D]")
        if trace_features.shape[:2] != semantic_features.shape[:2]:
            raise ValueError("trace/semantic sequence shapes must match on B and T")

        bsz, seq_len, _ = trace_features.shape
        device = trace_features.device

        if prior_features is None:
            prior_features = torch.zeros(
                bsz,
                seq_len,
                self.prior_dim,
                dtype=trace_features.dtype,
                device=device,
            )
        elif prior_features.ndim != 3 or prior_features.shape[:2] != trace_features.shape[:2]:
            raise ValueError("prior_features must be [B, T, D_prior] matching trace shape")

        fused_input = torch.cat([trace_features, semantic_features, prior_features], dim=-1)
        seq_hidden = self.input_proj(fused_input)
        seq_hidden = self.input_norm(seq_hidden + self.input_mlp(seq_hidden))

        objectness_logits = self.objectness_head(seq_hidden).squeeze(-1)
        if teacher_objectness is not None:
            if teacher_objectness.ndim != 2 or teacher_objectness.shape != objectness_logits.shape:
                raise ValueError("teacher_objectness must be [B, T]")
            teacher_bias = (teacher_objectness.clamp(0.0, 1.0) - 0.5) * self.objectness_bias_scale
            objectness_logits = objectness_logits + teacher_bias

        objectness = torch.sigmoid(objectness_logits)

        query = self.token_queries.unsqueeze(0).expand(bsz, -1, -1)
        scale = float(self.hidden_size) ** -0.5
        attn_logits = torch.einsum("bnh,bth->bnt", query, seq_hidden) * scale
        attn_logits = attn_logits + self.objectness_bias_scale * objectness.unsqueeze(1)

        token_time_attention = torch.softmax(attn_logits, dim=-1)
        token_values = torch.einsum("bnt,bth->bnh", token_time_attention, seq_hidden)
        state_tokens = self.output_norm(query + token_values)

        attn_entropy = -torch.sum(
            token_time_attention * torch.log(token_time_attention.clamp(min=1e-8)), dim=-1
        )
        norm = torch.log(torch.tensor(float(max(2, seq_len)), device=device, dtype=attn_entropy.dtype))
        attn_entropy_norm = (attn_entropy / norm).mean()

        token_usage = token_time_attention.mean(dim=1)
        token_usage_entropy = -torch.sum(token_usage * torch.log(token_usage.clamp(min=1e-8)), dim=-1)
        token_usage_entropy = token_usage_entropy / norm

        top_frame_index = torch.argmax(token_time_attention.mean(dim=1), dim=-1)

        diagnostics: dict[str, Any] = {
            "objectness_mean": float(objectness.mean().detach().cpu()),
            "objectness_std": float(objectness.std(unbiased=False).detach().cpu()),
            "assignment_entropy": float(attn_entropy_norm.detach().cpu()),
            "token_usage_entropy": float(token_usage_entropy.mean().detach().cpu()),
            "top_frame_index": [int(x) for x in top_frame_index.detach().cpu().tolist()],
            "num_tokens": int(self.num_tokens),
            "seq_len": int(seq_len),
        }

        return StateTokenizerV42Output(
            state_tokens=state_tokens,
            token_time_attention=token_time_attention,
            objectness=objectness,
            diagnostics=diagnostics,
        )
