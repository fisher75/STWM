from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterator, Tuple

import torch
from torch import nn


@dataclass
class TraceUnitFactorizedStateConfig:
    unit_dim: int = 384
    dyn_update: str = "gru"
    sem_update: str = "gated_ema"
    sem_alpha_min: float = 0.02
    sem_alpha_max: float = 0.12


class TraceUnitFactorizedState(nn.Module):
    def __init__(self, cfg: TraceUnitFactorizedStateConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.unit_proj = nn.Linear(int(cfg.unit_dim), int(cfg.unit_dim))
        self.dyn_gru = nn.GRU(int(cfg.unit_dim), int(cfg.unit_dim), batch_first=True)
        self.sem_proj = nn.Linear(int(cfg.unit_dim), int(cfg.unit_dim))
        self.sem_gate = nn.Linear(int(cfg.unit_dim), 1)
        self.norm_dyn = nn.LayerNorm(int(cfg.unit_dim))
        self.norm_sem = nn.LayerNorm(int(cfg.unit_dim))

    def semantic_named_parameters(self) -> Iterator[Tuple[str, nn.Parameter]]:
        for prefix, module in [
            ("sem_proj", self.sem_proj),
            ("sem_gate", self.sem_gate),
            ("norm_sem", self.norm_sem),
        ]:
            for name, param in module.named_parameters():
                yield f"{prefix}.{name}", param

    def semantic_parameters(self) -> Iterator[nn.Parameter]:
        for _, param in self.semantic_named_parameters():
            yield param

    def dynamic_named_parameters(self) -> Iterator[Tuple[str, nn.Parameter]]:
        for prefix, module in [
            ("dyn_gru", self.dyn_gru),
            ("norm_dyn", self.norm_dyn),
        ]:
            for name, param in module.named_parameters():
                yield f"{prefix}.{name}", param

    def dynamic_parameters(self) -> Iterator[nn.Parameter]:
        for _, param in self.dynamic_named_parameters():
            yield param

    def mixed_named_parameters(self) -> Iterator[Tuple[str, nn.Parameter]]:
        for name, param in self.unit_proj.named_parameters():
            yield f"unit_proj.{name}", param

    def forward(self, *, token_features: torch.Tensor, assignment: torch.Tensor) -> Dict[str, torch.Tensor | Dict[str, float]]:
        if token_features.shape[:3] != assignment.shape[:3]:
            raise ValueError(
                f"token/assignment mismatch: token={tuple(token_features.shape)} assignment={tuple(assignment.shape)}"
            )
        unit_inputs = torch.einsum("btkm,btkd->btmd", assignment, token_features)
        denom = assignment.sum(dim=2, keepdim=False)[..., None].clamp_min(1e-6)
        unit_inputs = self.unit_proj(unit_inputs / denom)
        unit_presence = (assignment.sum(dim=2) > 1e-3)

        bsz, t_len, unit_count, unit_dim = unit_inputs.shape
        dyn_in = unit_inputs.permute(0, 2, 1, 3).reshape(bsz * unit_count, t_len, unit_dim)
        dyn_out, _ = self.dyn_gru(dyn_in)
        z_dyn = self.norm_dyn(dyn_out).reshape(bsz, unit_count, t_len, unit_dim).permute(0, 2, 1, 3)

        sem_in = self.norm_sem(self.sem_proj(unit_inputs))
        alpha_gate = torch.sigmoid(self.sem_gate(sem_in))
        alpha = float(self.cfg.sem_alpha_min) + alpha_gate * float(max(self.cfg.sem_alpha_max - self.cfg.sem_alpha_min, 0.0))
        sem_states = []
        prev = sem_in[:, 0]
        sem_states.append(prev)
        for step in range(1, t_len):
            a = alpha[:, step]
            candidate = sem_in[:, step]
            updated = (1.0 - a) * prev + a * candidate
            present = unit_presence[:, step][..., None]
            prev = torch.where(present, updated, prev)
            sem_states.append(prev)
        z_sem = torch.stack(sem_states, dim=1)

        dyn_delta = torch.linalg.norm(z_dyn[:, 1:] - z_dyn[:, :-1], dim=-1)
        sem_delta = torch.linalg.norm(z_sem[:, 1:] - z_sem[:, :-1], dim=-1)
        valid_delta = unit_presence[:, 1:] & unit_presence[:, :-1]
        denom_delta = valid_delta.float().sum().clamp_min(1.0)
        z_dyn_drift = float((dyn_delta * valid_delta.float()).sum().detach().cpu().item() / float(denom_delta.detach().cpu().item()))
        z_sem_drift = float((sem_delta * valid_delta.float()).sum().detach().cpu().item() / float(denom_delta.detach().cpu().item()))
        sem_cos = torch.nn.functional.cosine_similarity(z_sem[:, 1:], z_sem[:, :-1], dim=-1)
        sem_stability = float((sem_cos * valid_delta.float()).sum().detach().cpu().item() / float(denom_delta.detach().cpu().item()))
        metrics = {
            "z_dyn_drift_mean": z_dyn_drift,
            "z_sem_drift_mean": z_sem_drift,
            "z_sem_to_z_dyn_drift_ratio": float(z_sem_drift / max(z_dyn_drift, 1e-6)),
            "unit_semantic_stability_over_time": sem_stability,
            "mean_sem_alpha": float(alpha.detach().mean().cpu().item()),
        }
        return {
            "unit_inputs": unit_inputs,
            "unit_presence": unit_presence,
            "z_dyn": z_dyn,
            "z_sem": z_sem,
            "alpha": alpha,
            "metrics": metrics,
        }
