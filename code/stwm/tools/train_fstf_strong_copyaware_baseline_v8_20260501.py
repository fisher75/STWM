#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from stwm.tracewm_v2_stage2.trainers.train_tracewm_stage2_smalltrain import _to_device
from stwm.tracewm_v2_stage2.utils.future_semantic_prototype_targets import (
    load_future_semantic_prototype_target_cache,
    prototype_tensors_for_batch,
    semantic_change_tensors,
)
from stwm.tools.run_semantic_memory_transition_residual_tiny_overfit_20260428 import _load_observed, _observed_for_batch
from stwm.tools.train_fstf_same_output_baseline_v7_20260501 import batch_slot_count, load_batches, proto_loss_and_metrics, write_json


LEARNED_BASELINES = {
    "copy_residual_mlp",
    "copy_residual_transformer",
    "copy_gated_residual_no_trace",
    "copy_gated_residual_trace_only",
    "copy_gated_residual_plain_trace_semantic",
}


def apply_process_title() -> None:
    title = os.environ.get("STWM_PROC_TITLE", "python") or "python"
    try:
        import setproctitle  # type: ignore

        setproctitle.setproctitle(title)
    except Exception:
        pass


def copy_logits(obs_dist: torch.Tensor, horizon: int) -> torch.Tensor:
    return torch.log(obs_dist.clamp_min(1e-6))[:, None].expand(-1, int(horizon), -1, -1)


def baseline_metadata(name: str) -> dict[str, Any]:
    common = {
        "baseline": name,
        "baseline_type": "same_output_fstf",
        "evidence_level": "controlled_ablation" if name not in {"copy_semantic_memory_baseline", "oracle_change_gate_upper_bound"} else ("trivial_lower_bound" if name == "copy_semantic_memory_baseline" else "oracle_upper_bound"),
        "official_repo_used": False,
        "official_repo_url_or_path": "",
        "official_commit_hash": "",
        "official_checkpoint_used": False,
        "pretrained_checkpoint_path": "",
        "output_contract_matched": True,
        "uses_future_candidate_measurement": False,
        "should_appear_in_external_boundary_table": False,
    }
    if name == "oracle_change_gate_upper_bound":
        return {
            **common,
            "baseline_name": name,
            "baseline_family": "oracle_upper_bound",
            "allowed_table_placement": "oracle_only",
            "should_appear_in_main_fstf_table": False,
            "allowed_claim": "Upper bound using GT changed/stable information; not a fair learned baseline.",
            "forbidden_claim": "Do not compare as a fair baseline.",
        }
    if name == "copy_semantic_memory_baseline":
        return {
            **common,
            "baseline_name": name,
            "baseline_family": "copy_lower_bound",
            "allowed_table_placement": "main_fstf_table",
            "should_appear_in_main_fstf_table": True,
            "allowed_claim": "Strong semantic persistence lower bound.",
            "forbidden_claim": "Do not describe as learned dynamics.",
        }
    return {
        **common,
        "baseline_name": name,
        "baseline_family": "copy_aware_controlled_baseline",
        "allowed_table_placement": "main_fstf_table",
        "should_appear_in_main_fstf_table": True,
        "allowed_claim": "Copy-aware same-output learned baseline under the STWM-FSTF protocol.",
        "forbidden_claim": "Do not describe as STWM/TUSB or external official baseline.",
    }


class CopyAwareFSTFBaseline(nn.Module):
    def __init__(
        self,
        *,
        baseline: str,
        prototype_count: int,
        trace_dim: int,
        horizon: int,
        slot_count: int,
        d_model: int = 256,
        layers: int = 3,
        heads: int = 4,
        residual_scale: float = 0.25,
    ) -> None:
        super().__init__()
        if baseline not in LEARNED_BASELINES:
            raise ValueError(f"unknown learned baseline: {baseline}")
        self.baseline = baseline
        self.prototype_count = int(prototype_count)
        self.horizon = int(horizon)
        self.slot_count = int(slot_count)
        self.d_model = int(d_model)
        self.residual_scale = float(residual_scale)
        self.trace_proj = nn.Linear(int(trace_dim), d_model)
        self.semantic_proj = nn.Linear(int(prototype_count), d_model)
        self.slot_pos = nn.Embedding(int(slot_count), d_model)
        self.horizon_pos = nn.Embedding(int(horizon), d_model)
        self.mlp = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
        )
        enc = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=int(heads),
            dim_feedforward=d_model * 4,
            dropout=0.05,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc, num_layers=int(layers))
        self.norm = nn.LayerNorm(d_model)
        self.residual_head = nn.Linear(d_model, int(prototype_count))
        self.gate_head = nn.Linear(d_model, 1)

    def _trace_features(self, batch: dict[str, Any]) -> torch.Tensor:
        obs = batch["obs_state"].float()
        valid = batch.get("obs_valid", torch.ones(obs.shape[:3], dtype=torch.bool, device=obs.device)).float()
        denom = valid.sum(dim=1).clamp_min(1.0).unsqueeze(-1)
        return (obs * valid.unsqueeze(-1)).sum(dim=1) / denom

    def token_features(self, batch: dict[str, Any], obs_dist: torch.Tensor) -> torch.Tensor:
        bsz, k, _ = obs_dist.shape
        feat = torch.zeros((bsz, k, self.d_model), device=obs_dist.device)
        if self.baseline in {"copy_residual_mlp", "copy_residual_transformer", "copy_gated_residual_trace_only", "copy_gated_residual_plain_trace_semantic"}:
            feat = feat + self.trace_proj(self._trace_features(batch)[:, :k])
        if self.baseline in {"copy_residual_mlp", "copy_residual_transformer", "copy_gated_residual_no_trace", "copy_gated_residual_plain_trace_semantic"}:
            feat = feat + self.semantic_proj(obs_dist[:, :k])
        slots = torch.arange(k, device=obs_dist.device).clamp_max(self.slot_count - 1)
        return feat + self.slot_pos(slots)[None]

    def forward(self, batch: dict[str, Any], obs_dist: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        bsz, k, _ = obs_dist.shape
        h = self.horizon
        hids = torch.arange(h, device=obs_dist.device)
        feat = self.token_features(batch, obs_dist)
        tokens = feat[:, None, :, :].expand(-1, h, -1, -1) + self.horizon_pos(hids)[None, :, None, :]
        tokens = tokens.reshape(bsz, h * k, self.d_model)
        if self.baseline == "copy_residual_mlp":
            out = self.mlp(tokens)
        else:
            out = self.encoder(tokens)
        out = self.norm(out).reshape(bsz, h, k, self.d_model)
        residual = self.residual_head(out)
        base = copy_logits(obs_dist, h)
        if self.baseline in {"copy_residual_mlp", "copy_residual_transformer"}:
            return base + self.residual_scale * residual, None
        gate_logit = self.gate_head(out).squeeze(-1)
        gate = torch.sigmoid(gate_logit)
        return base + gate[..., None] * self.residual_scale * residual, gate_logit


def make_model(baseline: str, prototype_count: int, horizon: int, batch: dict[str, Any], d_model: int, layers: int, heads: int, residual_scale: float) -> CopyAwareFSTFBaseline:
    return CopyAwareFSTFBaseline(
        baseline=baseline,
        prototype_count=prototype_count,
        trace_dim=int(batch["obs_state"].shape[-1]),
        horizon=horizon,
        slot_count=batch_slot_count(batch),
        d_model=d_model,
        layers=layers,
        heads=heads,
        residual_scale=residual_scale,
    )


def stable_copy_kl(logits: torch.Tensor, obs_dist: torch.Tensor, stable_mask: torch.Tensor) -> torch.Tensor:
    if not bool(stable_mask.any().item()):
        return logits.sum() * 0.0
    target = obs_dist[:, None].expand(-1, logits.shape[1], -1, -1)
    logp = F.log_softmax(logits, dim=-1)
    return F.kl_div(logp[stable_mask], target[stable_mask], reduction="batchmean")


def evaluate_metrics(model: CopyAwareFSTFBaseline, batches: list[dict[str, Any]], future_cache: Any, obs_data: dict[str, np.ndarray], device: torch.device, horizon: int) -> dict[str, float]:
    sums = {"top5": 0.0, "changed": 0.0, "stable": 0.0, "copy_changed": 0.0, "copy_stable": 0.0}
    counts = {"overall": 0, "changed": 0, "stable": 0}
    model.eval()
    with torch.no_grad():
        for batch_cpu in batches:
            batch = _to_device(batch_cpu, device, non_blocking=False)
            target, _dist, future_mask, _ = prototype_tensors_for_batch(future_cache, batch, horizon=horizon, slot_count=batch_slot_count(batch), device=device)
            obs_target, obs_dist, obs_mask = _observed_for_batch(obs_data, batch, device)
            change_target, change_mask, *_ = semantic_change_tensors(
                future_proto_target=target,
                future_proto_mask=future_mask,
                observed_proto_target=obs_target,
                observed_proto_mask=obs_mask,
            )
            logits, _gate = model(batch, obs_dist)
            copy = copy_logits(obs_dist, horizon)
            for name, mask in [("overall", change_mask), ("changed", change_mask & change_target), ("stable", change_mask & (~change_target))]:
                if not bool(mask.any().item()):
                    continue
                _loss, m = proto_loss_and_metrics(logits, target, mask)
                _closs, cm = proto_loss_and_metrics(copy, target, mask)
                n = int(m["valid_count"])
                counts[name] += n
                if name == "overall":
                    sums["top5"] += float(m["proto_top5"]) * n
                elif name == "changed":
                    sums["changed"] += float(m["proto_top5"]) * n
                    sums["copy_changed"] += float(cm["proto_top5"]) * n
                else:
                    sums["stable"] += float(m["proto_top5"]) * n
                    sums["copy_stable"] += float(cm["proto_top5"]) * n
    return {
        "proto_top5": sums["top5"] / max(counts["overall"], 1),
        "changed_subset_top5": sums["changed"] / max(counts["changed"], 1),
        "copy_changed_subset_top5": sums["copy_changed"] / max(counts["changed"], 1),
        "changed_subset_gain_over_copy": sums["changed"] / max(counts["changed"], 1) - sums["copy_changed"] / max(counts["changed"], 1),
        "stable_subset_top5": sums["stable"] / max(counts["stable"], 1),
        "copy_stable_subset_top5": sums["copy_stable"] / max(counts["stable"], 1),
        "stable_preservation_drop": sums["copy_stable"] / max(counts["stable"], 1) - sums["stable"] / max(counts["stable"], 1),
        "valid_count": counts["overall"],
        "changed_subset_count": counts["changed"],
        "stable_subset_count": counts["stable"],
    }


def main() -> None:
    apply_process_title()
    p = argparse.ArgumentParser()
    p.add_argument("--baseline", required=True, choices=sorted(LEARNED_BASELINES))
    p.add_argument("--prototype-count", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train-cache-report", default="reports/stwm_mixed_fullscale_v2_materialization_train_20260428.json")
    p.add_argument("--val-cache-report", default="reports/stwm_mixed_fullscale_v2_materialization_val_20260428.json")
    p.add_argument("--observed-report", default="reports/stwm_mixed_observed_semantic_prototype_targets_v2_20260428.json")
    p.add_argument("--future-cache-report", default="reports/stwm_fullscale_semantic_trace_prototype_targets_c32_v1_20260428.json")
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--layers", type=int, default=3)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--residual-scale", type=float, default=0.25)
    p.add_argument("--device", default="cuda")
    p.add_argument("--checkpoint-output", required=True)
    p.add_argument("--summary-output", required=True)
    p.add_argument("--progress-every", type=int, default=100)
    args = p.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    train_batches, train_report = load_batches(Path(args.train_cache_report))
    val_batches, val_report = load_batches(Path(args.val_cache_report))
    future_cache = load_future_semantic_prototype_target_cache(Path(args.future_cache_report))
    obs_data = _load_observed(Path(args.observed_report), args.prototype_count)
    first = _to_device(train_batches[0], device, non_blocking=False)
    target0, _d, _m, _i = prototype_tensors_for_batch(future_cache, first, horizon=8, slot_count=batch_slot_count(first), device=device)
    horizon = int(target0.shape[1])
    model = make_model(args.baseline, args.prototype_count, horizon, first, args.d_model, args.layers, args.heads, args.residual_scale).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    history: list[dict[str, float]] = []
    finite = 0
    start = time.time()
    print(f"[fstf-v8-train] start baseline={args.baseline} seed={args.seed} steps={args.steps} device={device} cuda_visible={os.environ.get('CUDA_VISIBLE_DEVICES','')}", flush=True)
    for step in range(1, args.steps + 1):
        batch = _to_device(train_batches[(step - 1) % len(train_batches)], device, non_blocking=False)
        target, _dist, future_mask, _ = prototype_tensors_for_batch(future_cache, batch, horizon=horizon, slot_count=batch_slot_count(batch), device=device)
        obs_target, obs_dist, obs_mask = _observed_for_batch(obs_data, batch, device)
        change_target, change_mask, *_ = semantic_change_tensors(
            future_proto_target=target,
            future_proto_mask=future_mask,
            observed_proto_target=obs_target,
            observed_proto_mask=obs_mask,
        )
        logits, gate_logit = model(batch, obs_dist)
        overall_loss, met = proto_loss_and_metrics(logits, target, change_mask)
        changed_mask = change_mask & change_target
        stable_mask = change_mask & (~change_target)
        changed_loss, _ = proto_loss_and_metrics(logits, target, changed_mask)
        loss = overall_loss + 2.5 * changed_loss + 0.25 * stable_copy_kl(logits, obs_dist, stable_mask)
        if gate_logit is not None and bool(change_mask.any().item()):
            pos = change_target[change_mask].float()
            pos_rate = float(pos.mean().detach().cpu().item()) if pos.numel() else 0.5
            weight = torch.where(pos > 0.5, torch.full_like(pos, max(1.0, (1.0 - pos_rate) / max(pos_rate, 1e-4))), torch.ones_like(pos))
            loss = loss + 0.5 * F.binary_cross_entropy_with_logits(gate_logit[change_mask], pos, weight=weight)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        finite += int(torch.isfinite(loss).detach().cpu().item())
        if step == 1 or step % args.progress_every == 0 or step == args.steps:
            rec = {"step": float(step), "loss": float(loss.detach().cpu().item()), "top5": float(met.get("proto_top5", 0.0))}
            history.append(rec)
            print(f"[fstf-v8-train] step={step} loss={rec['loss']:.6f} top5={rec['top5']:.4f}", flush=True)
    val_metrics = evaluate_metrics(model, val_batches, future_cache, obs_data, device, horizon)
    ckpt = Path(args.checkpoint_output)
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "config": {"baseline": args.baseline, "prototype_count": args.prototype_count, "horizon": horizon, "slot_count": batch_slot_count(first), "trace_dim": int(first["obs_state"].shape[-1]), "d_model": args.d_model, "layers": args.layers, "heads": args.heads, "residual_scale": args.residual_scale}, "seed": args.seed}, ckpt)
    summary = {
        "audit_name": "stwm_fstf_strong_copyaware_baseline_v8_train",
        **baseline_metadata(args.baseline),
        "prototype_count": args.prototype_count,
        "seed": args.seed,
        "steps": args.steps,
        "lr": args.lr,
        "parameter_count": int(sum(p.numel() for p in model.parameters())),
        "train_batch_count": len(train_batches),
        "val_batch_count": len(val_batches),
        "train_report": train_report,
        "val_report": val_report,
        "checkpoint_path": str(ckpt),
        "checkpoint_mtime": ckpt.stat().st_mtime,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "loss_finite_ratio": finite / max(args.steps, 1),
        "history": history,
        "val_metrics": val_metrics,
        "free_rollout_path": "copy_aware_baseline_forward_observed_inputs_only",
        "candidate_scorer_used": False,
        "future_candidate_leakage": False,
        "teacher_forced_path_used": False,
        "elapsed_seconds": time.time() - start,
    }
    write_json(args.summary_output, summary)
    print(f"[fstf-v8-train] done checkpoint={ckpt} summary={args.summary_output}", flush=True)


if __name__ == "__main__":
    main()
