from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterator, Tuple

import torch
from torch import nn


@dataclass
class TraceUnitBroadcastConfig:
    hidden_dim: int = 1152
    unit_dim: int = 384
    residual_weight: float = 0.35
    stopgrad_semantic: bool = False


class TraceUnitBroadcast(nn.Module):
    def __init__(self, cfg: TraceUnitBroadcastConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.dyn_proj = nn.Linear(int(cfg.unit_dim), int(cfg.hidden_dim), bias=False)
        self.sem_proj = nn.Linear(int(cfg.unit_dim), int(cfg.hidden_dim), bias=True)
        self.norm = nn.LayerNorm(int(cfg.hidden_dim))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        old_weight_key = prefix + "proj.weight"
        old_bias_key = prefix + "proj.bias"
        dyn_weight_key = prefix + "dyn_proj.weight"
        sem_weight_key = prefix + "sem_proj.weight"
        sem_bias_key = prefix + "sem_proj.bias"
        if old_weight_key in state_dict and dyn_weight_key not in state_dict and sem_weight_key not in state_dict:
            old_weight = state_dict[old_weight_key]
            unit_dim = int(self.cfg.unit_dim)
            state_dict[dyn_weight_key] = old_weight[:, :unit_dim].clone()
            state_dict[sem_weight_key] = old_weight[:, unit_dim:].clone()
        if old_bias_key in state_dict and sem_bias_key not in state_dict:
            state_dict[sem_bias_key] = state_dict[old_bias_key].clone()
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

    def semantic_named_parameters(self) -> Iterator[Tuple[str, nn.Parameter]]:
        for name, param in self.sem_proj.named_parameters():
            yield f"sem_proj.{name}", param

    def semantic_parameters(self) -> Iterator[nn.Parameter]:
        for _, param in self.semantic_named_parameters():
            yield param

    def dynamic_named_parameters(self) -> Iterator[Tuple[str, nn.Parameter]]:
        for name, param in self.dyn_proj.named_parameters():
            yield f"dyn_proj.{name}", param

    def dynamic_parameters(self) -> Iterator[nn.Parameter]:
        for _, param in self.dynamic_named_parameters():
            yield param

    def mixed_named_parameters(self) -> Iterator[Tuple[str, nn.Parameter]]:
        for name, param in self.norm.named_parameters():
            yield f"norm.{name}", param

    def forward(
        self,
        *,
        backbone_hidden: torch.Tensor,
        assignment: torch.Tensor,
        z_dyn: torch.Tensor,
        z_sem: torch.Tensor,
    ) -> Dict[str, torch.Tensor | Dict[str, float]]:
        sem = z_sem.detach() if bool(self.cfg.stopgrad_semantic) else z_sem
        unit_residual = self.dyn_proj(z_dyn) + self.sem_proj(sem)
        token_residual = torch.einsum("btkm,btmd->btkd", assignment, unit_residual)
        enhanced = self.norm(backbone_hidden + float(self.cfg.residual_weight) * token_residual)
        metrics = {
            "broadcast_residual_norm_mean": float(torch.linalg.norm(token_residual, dim=-1).detach().mean().cpu().item()),
        }
        return {
            "enhanced_hidden": enhanced,
            "token_residual": token_residual,
            "metrics": metrics,
        }
