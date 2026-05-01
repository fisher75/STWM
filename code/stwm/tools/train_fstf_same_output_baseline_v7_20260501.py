#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
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
from stwm.tracewm_v2_stage2.utils.future_semantic_feature_targets import stage2_item_key
from stwm.tracewm_v2_stage2.utils.future_semantic_prototype_targets import (
    load_future_semantic_prototype_target_cache,
    prototype_tensors_for_batch,
    semantic_change_tensors,
)
from stwm.tools.run_semantic_memory_transition_residual_tiny_overfit_20260428 import _load_observed, _observed_for_batch


BASELINES = {
    "trace_only_ar_transformer",
    "semantic_only_memory_transition",
    "trace_semantic_transformer",
    "slotformer_like_trace_unit_dynamics",
    "dino_wm_like_latent_dynamics_proxy",
}


def baseline_officiality_metadata(baseline: str) -> dict[str, Any]:
    common = {
        "baseline": baseline,
        "official_repo_used": False,
        "official_repo_url_or_path": "",
        "official_commit_hash": "",
        "official_checkpoint_used": False,
        "pretrained_checkpoint_path": "",
        "output_contract_matched": True,
        "uses_future_candidate_measurement": False,
    }
    if baseline == "trace_only_ar_transformer":
        return {
            **common,
            "baseline_name": "trace_only_ar_transformer",
            "baseline_family": "trace_dynamics_controlled_ablation",
            "evidence_level": "controlled_ablation",
            "adaptation_changes": "Native STWM-FSTF same-output trace-only AR Transformer; no official external repo.",
            "allowed_table_placement": "main_fstf_table",
            "allowed_claim": "Controlled trace-only same-output ablation under the STWM-FSTF protocol.",
            "forbidden_claim": "Do not present as an official external trajectory/world-model implementation.",
        }
    if baseline == "semantic_only_memory_transition":
        return {
            **common,
            "baseline_name": "semantic_only_memory_transition",
            "baseline_family": "semantic_memory_controlled_ablation",
            "evidence_level": "controlled_ablation",
            "adaptation_changes": "Native STWM-FSTF same-output semantic-only transition; no trace rollout hidden input.",
            "allowed_table_placement": "main_fstf_table",
            "allowed_claim": "Controlled semantic-only same-output ablation under the STWM-FSTF protocol.",
            "forbidden_claim": "Do not present as evidence against official object-centric dynamics methods.",
        }
    if baseline == "trace_semantic_transformer":
        return {
            **common,
            "baseline_name": "plain_trace_semantic_transformer",
            "baseline_family": "plain_trace_semantic_controlled_ablation",
            "evidence_level": "controlled_ablation",
            "adaptation_changes": "Native plain Transformer over trace rollout features plus observed semantic memory; no copy-gated residual structure.",
            "allowed_table_placement": "main_fstf_table",
            "allowed_claim": "Controlled same-output plain trace+semantic Transformer comparison.",
            "forbidden_claim": "Do not describe as an official SlotFormer/DINO-WM baseline.",
        }
    if baseline == "slotformer_like_trace_unit_dynamics":
        return {
            **common,
            "baseline_name": "slot_ar_trace_unit_proxy",
            "baseline_family": "object_slot_ar_proxy",
            "evidence_level": "proxy_inspired",
            "adaptation_changes": "Native autoregressive trace-unit slot proxy inspired by object-slot dynamics; official SlotFormer repo/code/checkpoint not used.",
            "allowed_table_placement": "appendix_proxy_table",
            "allowed_claim": "STWM can be compared against a controlled slot-AR trace-unit proxy.",
            "forbidden_claim": "Cannot claim beating official SlotFormer or say SlotFormer baseline completed.",
        }
    if baseline == "dino_wm_like_latent_dynamics_proxy":
        return {
            **common,
            "baseline_name": "dinov2_latent_dynamics_proxy",
            "baseline_family": "pretrained_feature_dynamics_proxy",
            "evidence_level": "proxy_inspired",
            "adaptation_changes": "Native latent dynamics proxy using existing frozen semantic/visual cache; official DINO-WM repo/code/checkpoint/feature protocol not used.",
            "allowed_table_placement": "appendix_proxy_table",
            "allowed_claim": "STWM can be compared against a DINOv2/pretrained-feature latent dynamics proxy.",
            "forbidden_claim": "Cannot claim beating official DINO-WM or say DINO-WM baseline completed.",
        }
    raise ValueError(f"unknown baseline: {baseline}")


def apply_process_title() -> None:
    title = os.environ.get("STWM_PROC_TITLE", "python") or "python"
    mode = os.environ.get("STWM_PROC_TITLE_MODE", "generic")
    if str(mode).lower() == "off":
        return
    try:
        import setproctitle  # type: ignore

        setproctitle.setproctitle(title)
    except Exception:
        pass


def write_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_batches(report_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    cache_path = Path(str(report["batch_cache_path"]))
    if not cache_path.is_absolute():
        cache_path = report_path.parent.parent / cache_path
    cache = torch.load(cache_path, map_location="cpu")
    return list(cache["batches"]), report


def batch_slot_count(batch: dict[str, Any]) -> int:
    token_mask = batch.get("token_mask")
    if isinstance(token_mask, torch.Tensor) and token_mask.ndim == 2:
        return int(token_mask.shape[1])
    return int(batch["obs_state"].shape[2])


class FSTFSameOutputBaseline(nn.Module):
    def __init__(
        self,
        *,
        baseline: str,
        prototype_count: int,
        trace_dim: int,
        horizon: int,
        slot_count: int,
        d_model: int = 192,
        layers: int = 2,
        heads: int = 4,
        latent_dim: int = 512,
    ) -> None:
        super().__init__()
        if baseline not in BASELINES:
            raise ValueError(f"unknown baseline: {baseline}")
        self.baseline = baseline
        self.prototype_count = int(prototype_count)
        self.horizon = int(horizon)
        self.slot_count = int(slot_count)
        self.d_model = int(d_model)
        self.trace_proj = nn.Linear(int(trace_dim), d_model)
        self.semantic_proj = nn.Linear(int(prototype_count), d_model)
        self.latent_proj = nn.Linear(int(latent_dim), d_model)
        self.slot_pos = nn.Embedding(int(slot_count), d_model)
        self.horizon_pos = nn.Embedding(int(horizon), d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=int(heads),
            dim_feedforward=d_model * 4,
            dropout=0.05,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        if baseline == "semantic_only_memory_transition":
            layers = max(1, int(layers))
        elif baseline == "slotformer_like_trace_unit_dynamics":
            layers = max(3, int(layers))
        elif baseline == "dino_wm_like_latent_dynamics_proxy":
            layers = max(2, int(layers))
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=int(layers))
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, int(prototype_count))

    def _trace_features(self, batch: dict[str, Any]) -> torch.Tensor:
        obs = batch["obs_state"].float()
        valid = batch.get("obs_valid", torch.ones(obs.shape[:3], dtype=torch.bool, device=obs.device)).float()
        denom = valid.sum(dim=1, keepdim=False).clamp_min(1.0).unsqueeze(-1)
        return (obs * valid.unsqueeze(-1)).sum(dim=1) / denom

    def _latent_features(self, batch: dict[str, Any]) -> torch.Tensor:
        latent = batch.get("semantic_teacher_prior")
        if isinstance(latent, torch.Tensor) and latent.ndim == 3:
            return latent.float()
        device = batch["obs_state"].device
        bsz = int(batch["obs_state"].shape[0])
        return torch.zeros((bsz, self.slot_count, 512), device=device, dtype=torch.float32)

    def forward(self, batch: dict[str, Any], obs_dist: torch.Tensor) -> torch.Tensor:
        bsz = int(batch["obs_state"].shape[0])
        k = int(obs_dist.shape[1])
        h = self.horizon
        slot_ids = torch.arange(k, device=obs_dist.device).clamp_max(self.slot_count - 1)
        h_ids = torch.arange(h, device=obs_dist.device)

        feat = torch.zeros((bsz, k, self.d_model), device=obs_dist.device)
        if self.baseline in {"trace_only_ar_transformer", "trace_semantic_transformer", "slotformer_like_trace_unit_dynamics"}:
            feat = feat + self.trace_proj(self._trace_features(batch)[:, :k])
        if self.baseline in {"semantic_only_memory_transition", "trace_semantic_transformer", "slotformer_like_trace_unit_dynamics"}:
            feat = feat + self.semantic_proj(obs_dist[:, :k])
        if self.baseline == "dino_wm_like_latent_dynamics_proxy":
            feat = feat + self.latent_proj(self._latent_features(batch)[:, :k])
        feat = feat + self.slot_pos(slot_ids)[None]

        tokens = feat[:, None, :, :].expand(-1, h, -1, -1) + self.horizon_pos(h_ids)[None, :, None, :]
        tokens = tokens.reshape(bsz, h * k, self.d_model)
        out = self.encoder(tokens)
        out = self.norm(out).reshape(bsz, h, k, self.d_model)
        return self.head(out)


def make_model_from_batch(
    *,
    baseline: str,
    prototype_count: int,
    horizon: int,
    batch: dict[str, Any],
    d_model: int,
    layers: int,
    heads: int,
) -> FSTFSameOutputBaseline:
    return FSTFSameOutputBaseline(
        baseline=baseline,
        prototype_count=int(prototype_count),
        trace_dim=int(batch["obs_state"].shape[-1]),
        horizon=int(horizon),
        slot_count=batch_slot_count(batch),
        d_model=int(d_model),
        layers=int(layers),
        heads=int(heads),
    )


def proto_loss_and_metrics(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
    valid = mask.bool() & (target >= 0)
    valid_count = int(valid.sum().detach().cpu().item())
    if valid_count <= 0:
        z = logits.sum() * 0.0
        return z, {"valid_count": 0, "proto_ce": 0.0, "proto_accuracy": 0.0, "proto_top5": 0.0}
    flat_logits = logits[valid]
    flat_target = target[valid]
    loss = F.cross_entropy(flat_logits, flat_target)
    pred = flat_logits.argmax(dim=-1)
    topk = min(5, int(flat_logits.shape[-1]))
    top = flat_logits.topk(k=topk, dim=-1).indices
    return loss, {
        "valid_count": valid_count,
        "proto_ce": float(loss.detach().cpu().item()),
        "proto_accuracy": float((pred == flat_target).float().mean().detach().cpu().item()),
        "proto_top5": float((top == flat_target[:, None]).any(dim=-1).float().mean().detach().cpu().item()),
    }


def evaluate_batches(
    *,
    model: FSTFSameOutputBaseline,
    batches: list[dict[str, Any]],
    future_cache: Any,
    obs_data: dict[str, np.ndarray],
    device: torch.device,
    horizon: int,
    max_batches: int = 0,
) -> dict[str, float]:
    model.eval()
    sums = {k: 0.0 for k in ["top1", "top5", "ce", "stable_top5", "changed_top5"]}
    counts = {"overall": 0, "stable": 0, "changed": 0}
    trace_errors: list[float] = []
    with torch.no_grad():
        for i, batch_cpu in enumerate(batches):
            if max_batches and i >= int(max_batches):
                break
            batch = _to_device(batch_cpu, device, non_blocking=False)
            target, _dist, future_mask, _info = prototype_tensors_for_batch(
                future_cache,
                batch,
                horizon=int(horizon),
                slot_count=batch_slot_count(batch),
                device=device,
            )
            obs_target, obs_dist, obs_mask = _observed_for_batch(obs_data, batch, device)
            change_target, change_mask, _event_target, _event_mask, _change_info = semantic_change_tensors(
                future_proto_target=target,
                future_proto_mask=future_mask,
                observed_proto_target=obs_target,
                observed_proto_mask=obs_mask,
            )
            logits = model(batch, obs_dist)
            for name, mask in (
                ("overall", change_mask),
                ("stable", change_mask & (~change_target)),
                ("changed", change_mask & change_target),
            ):
                _loss, metrics = proto_loss_and_metrics(logits, target, mask)
                n = int(metrics["valid_count"])
                if n <= 0:
                    continue
                counts[name] += n
                if name == "overall":
                    sums["top1"] += metrics["proto_accuracy"] * n
                    sums["top5"] += metrics["proto_top5"] * n
                    sums["ce"] += metrics["proto_ce"] * n
                elif name == "stable":
                    sums["stable_top5"] += metrics["proto_top5"] * n
                else:
                    sums["changed_top5"] += metrics["proto_top5"] * n
            fut_valid = batch.get("fut_valid")
            if isinstance(fut_valid, torch.Tensor) and bool(fut_valid.any().item()):
                pred_coord = batch["obs_state"][:, -1:, :, :2].expand(-1, int(horizon), -1, -1)
                target_coord = batch["fut_state"][:, : int(horizon), :, :2]
                valid = fut_valid[:, : int(horizon)].bool()
                err = torch.sqrt(((pred_coord - target_coord) ** 2).sum(dim=-1).clamp_min(1e-12))
                trace_errors.append(float(err[valid].mean().detach().cpu().item()))
    return {
        "proto_top1": float(sums["top1"] / max(counts["overall"], 1)),
        "proto_top5": float(sums["top5"] / max(counts["overall"], 1)),
        "proto_ce": float(sums["ce"] / max(counts["overall"], 1)),
        "stable_subset_top5": float(sums["stable_top5"] / max(counts["stable"], 1)),
        "changed_subset_top5": float(sums["changed_top5"] / max(counts["changed"], 1)),
        "valid_count": int(counts["overall"]),
        "stable_subset_count": int(counts["stable"]),
        "changed_subset_count": int(counts["changed"]),
        "future_trace_coord_error": float(np.mean(trace_errors)) if trace_errors else 0.0,
    }


def main() -> None:
    apply_process_title()
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True, choices=sorted(BASELINES))
    parser.add_argument("--prototype-count", type=int, default=32)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--train-cache-report", default="reports/stwm_mixed_fullscale_v2_materialization_train_20260428.json")
    parser.add_argument("--val-cache-report", default="reports/stwm_mixed_fullscale_v2_materialization_val_20260428.json")
    parser.add_argument("--observed-report", default="reports/stwm_mixed_observed_semantic_prototype_targets_v2_20260428.json")
    parser.add_argument("--future-cache-report", default="reports/stwm_fullscale_semantic_trace_prototype_targets_c32_v1_20260428.json")
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--d-model", type=int, default=192)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--checkpoint-output", required=True)
    parser.add_argument("--summary-output", required=True)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--torch-num-threads", type=int, default=int(os.environ.get("STWM_TORCH_NUM_THREADS", "16")))
    args = parser.parse_args()

    if int(args.torch_num_threads) > 0:
        torch.set_num_threads(int(args.torch_num_threads))
        torch.set_num_interop_threads(max(1, min(4, int(args.torch_num_threads))))
    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))

    device = torch.device("cuda" if str(args.device) == "cuda" and torch.cuda.is_available() else "cpu")
    train_batches, train_report = load_batches(Path(args.train_cache_report))
    val_batches, val_report = load_batches(Path(args.val_cache_report))
    future_cache = load_future_semantic_prototype_target_cache(Path(args.future_cache_report))
    obs_data = _load_observed(Path(args.observed_report), int(args.prototype_count))
    first = _to_device(train_batches[0], device, non_blocking=False)
    horizon = int(future_cache.proto_target.shape[1])
    model = make_model_from_batch(
        baseline=str(args.baseline),
        prototype_count=int(args.prototype_count),
        horizon=horizon,
        batch=first,
        d_model=int(args.d_model),
        layers=int(args.layers),
        heads=int(args.heads),
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=1e-4)
    start = time.time()
    loss_finite = 0
    history: list[dict[str, float]] = []
    print(
        f"[fstf-baseline-train] start baseline={args.baseline} seed={args.seed} steps={args.steps} device={device} cuda_visible={os.environ.get('CUDA_VISIBLE_DEVICES','')}",
        flush=True,
    )
    for step in range(1, int(args.steps) + 1):
        batch_cpu = train_batches[(step - 1) % len(train_batches)]
        batch = _to_device(batch_cpu, device, non_blocking=False)
        target, _dist, future_mask, _info = prototype_tensors_for_batch(
            future_cache,
            batch,
            horizon=horizon,
            slot_count=batch_slot_count(batch),
            device=device,
        )
        obs_target, obs_dist, obs_mask = _observed_for_batch(obs_data, batch, device)
        _change_target, change_mask, _event_target, _event_mask, _change_info = semantic_change_tensors(
            future_proto_target=target,
            future_proto_mask=future_mask,
            observed_proto_target=obs_target,
            observed_proto_mask=obs_mask,
        )
        logits = model(batch, obs_dist)
        loss, metrics = proto_loss_and_metrics(logits, target, change_mask)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        finite = bool(torch.isfinite(loss).detach().cpu().item())
        loss_finite += int(finite)
        if step == 1 or step % int(args.progress_every) == 0 or step == int(args.steps):
            row = {"step": float(step), "loss": float(loss.detach().cpu().item()), "top5": float(metrics["proto_top5"])}
            history.append(row)
            print(f"[fstf-baseline-train] step={step} loss={row['loss']:.6f} top5={row['top5']:.4f}", flush=True)
    val_metrics = evaluate_batches(
        model=model,
        batches=val_batches,
        future_cache=future_cache,
        obs_data=obs_data,
        device=device,
        horizon=horizon,
    )
    ckpt_path = Path(args.checkpoint_output)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": {
                "baseline": str(args.baseline),
                "prototype_count": int(args.prototype_count),
                "horizon": horizon,
                "slot_count": int(first["token_mask"].shape[1]),
                "trace_dim": int(first["obs_state"].shape[-1]),
                "d_model": int(args.d_model),
                "layers": int(args.layers),
                "heads": int(args.heads),
            },
            "seed": int(args.seed),
            "train_report": train_report,
            "val_report": val_report,
        },
        ckpt_path,
    )
    summary = {
        "audit_name": "stwm_fstf_same_output_baseline_v7_train",
        **baseline_officiality_metadata(str(args.baseline)),
        "baseline": str(args.baseline),
        "prototype_count": int(args.prototype_count),
        "seed": int(args.seed),
        "steps": int(args.steps),
        "lr": float(args.lr),
        "device": str(device),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "checkpoint_path": str(ckpt_path),
        "checkpoint_mtime": ckpt_path.stat().st_mtime,
        "train_cache_report": str(args.train_cache_report),
        "val_cache_report": str(args.val_cache_report),
        "train_batch_count": len(train_batches),
        "val_batch_count": len(val_batches),
        "loss_finite_ratio": float(loss_finite / max(int(args.steps), 1)),
        "history": history,
        "val_metrics": val_metrics,
        "candidate_scorer_used": False,
        "future_candidate_leakage": False,
        "teacher_forced_path_used": False,
        "free_rollout_path": "baseline_forward_observed_inputs_only",
        "elapsed_seconds": float(time.time() - start),
    }
    write_json(Path(args.summary_output), summary)
    print(f"[fstf-baseline-train] done checkpoint={ckpt_path} summary={args.summary_output}", flush=True)


if __name__ == "__main__":
    main()
