#!/usr/bin/env python3
from __future__ import annotations

from argparse import ArgumentParser
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple
import json
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from stwm.tracewm_v2.models.causal_trace_transformer import (
    TraceCausalTransformer,
    build_tracewm_v2_config,
)
from stwm.tracewm_v2.tools.run_stage1_v2_scientific_revalidation import _load_runtime_config
from stwm.tracewm_v2_stage2.datasets.stage2_semantic_dataset import (
    Stage2SemanticDataset,
    Stage2SemanticDatasetConfig,
    stage2_semantic_collate_fn,
)
from stwm.tracewm_v2_stage2.models.semantic_encoder import SemanticEncoder, SemanticEncoderConfig
from stwm.tracewm_v2_stage2.models.semantic_fusion import SemanticFusion, SemanticFusionConfig


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args() -> Any:
    p = ArgumentParser(description="Stage2 bootstrap trainer (smoke only)")
    p.add_argument("--contract-path", default="/home/chen034/workspace/data/_manifests/stage1_v2_trace_cache_contract_20260408.json")
    p.add_argument("--recommended-runtime-json", default="/home/chen034/workspace/stwm/reports/stage1_v2_recommended_runtime_20260408.json")
    p.add_argument("--use-recommended-runtime", action="store_true")

    p.add_argument(
        "--stage1-backbone-checkpoint",
        default="/home/chen034/workspace/stwm/outputs/checkpoints/stage1_v2_longtrain_220m_mainline_20260408/best.pt",
    )
    p.add_argument("--stage1-model-preset", default="prototype_220m")

    p.add_argument("--dataset-names", nargs="*", default=["pointodyssey", "kubric"])
    p.add_argument("--train-split", default="train")
    p.add_argument("--val-split", default="val")
    p.add_argument("--obs-len", type=int, default=8)
    p.add_argument("--fut-len", type=int, default=8)
    p.add_argument("--max-tokens", type=int, default=64)
    p.add_argument("--max-samples-train", type=int, default=8)
    p.add_argument("--max-samples-val", type=int, default=4)
    p.add_argument("--semantic-patch-radius", type=int, default=12)

    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--smoke-steps", type=int, default=4)

    p.add_argument("--semantic-hidden-dim", type=int, default=256)
    p.add_argument("--semantic-embed-dim", type=int, default=256)

    p.add_argument("--run-name", default="stage2_bootstrap_smoke_20260408")
    p.add_argument("--smoke-json", default="/home/chen034/workspace/stwm/reports/stage2_bootstrap_smoke_20260408.json")
    p.add_argument("--results-md", default="/home/chen034/workspace/stwm/docs/STAGE2_BOOTSTRAP_RESULTS_20260408.md")
    p.add_argument("--seed", type=int, default=20260408)
    return p.parse_args()


def _safe_load_checkpoint(path: str | Path, device: torch.device) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"checkpoint not found: {p}")
    try:
        payload = torch.load(p, map_location=device, weights_only=False)
    except TypeError:
        payload = torch.load(p, map_location=device)
    if not isinstance(payload, dict):
        raise RuntimeError(f"unsupported checkpoint payload type: {type(payload)}")
    return payload


def _load_frozen_stage1_backbone(args: Any, device: torch.device) -> Tuple[TraceCausalTransformer, Dict[str, Any]]:
    ckpt = _safe_load_checkpoint(args.stage1_backbone_checkpoint, device=device)
    cfg_payload = ckpt.get("config", {}) if isinstance(ckpt.get("config", {}), dict) else {}
    preset = str(cfg_payload.get("model_preset", args.stage1_model_preset))
    cfg = build_tracewm_v2_config(preset)

    model = TraceCausalTransformer(cfg).to(device)
    state_dict = ckpt.get("model_state_dict") if isinstance(ckpt.get("model_state_dict"), dict) else None
    if state_dict is None:
        state_dict = {k: v for k, v in ckpt.items() if isinstance(v, torch.Tensor)}

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if unexpected:
        raise RuntimeError(f"unexpected stage1 checkpoint keys: {unexpected[:8]}")
    if len(missing) > 16:
        raise RuntimeError(f"too many missing stage1 checkpoint keys: {len(missing)}")

    for p in model.parameters():
        p.requires_grad = False
    model.eval()

    meta = {
        "checkpoint_path": str(args.stage1_backbone_checkpoint),
        "model_preset": preset,
        "missing_keys": list(missing),
        "unexpected_keys": list(unexpected),
        "parameter_count": int(sum(x.numel() for x in model.parameters())),
        "trainable_parameter_count": int(sum(x.numel() for x in model.parameters() if x.requires_grad)),
    }
    return model, meta


def _to_device(batch: Dict[str, Any], device: torch.device, non_blocking: bool) -> Dict[str, Any]:
    out = dict(batch)
    for k in [
        "obs_state",
        "fut_state",
        "obs_valid",
        "fut_valid",
        "token_mask",
        "semantic_features",
        "semantic_mask",
    ]:
        out[k] = batch[k].to(device, non_blocking=non_blocking)
    return out


def _prepare_shifted(batch: Dict[str, Any]) -> torch.Tensor:
    full_state = torch.cat([batch["obs_state"], batch["fut_state"]], dim=1)
    shifted = torch.zeros_like(full_state)
    shifted[:, 0] = full_state[:, 0]
    shifted[:, 1:] = full_state[:, :-1]
    return shifted


def _masked_coord_loss(pred_coord: torch.Tensor, target_coord: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    l2 = ((pred_coord - target_coord) ** 2).sum(dim=-1)
    mask_f = valid_mask.float()
    denom = mask_f.sum().clamp_min(1.0)
    return (l2 * mask_f).sum() / denom


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_md(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ans = payload.get("answers", {}) if isinstance(payload.get("answers", {}), dict) else {}
    boundary = payload.get("freeze_trainable_boundary", {}) if isinstance(payload.get("freeze_trainable_boundary", {}), dict) else {}
    lines = [
        "# Stage2 Bootstrap Smoke Results",
        "",
        f"- generated_at_utc: {payload.get('generated_at_utc', '')}",
        f"- run_name: {payload.get('run_name', '')}",
        f"- stage2_bootstrap_ready: {payload.get('bootstrap_ready', False)}",
        f"- next_step_choice: {payload.get('next_step_choice', '')}",
        "",
        "## Required Answers",
        f"- stage1_frozen_backbone_loadable: {ans.get('stage1_frozen_backbone_loadable', False)}",
        f"- semantic_branch_accepts_inputs: {ans.get('semantic_branch_accepts_inputs', False)}",
        f"- freeze_trainable_boundary_working: {ans.get('freeze_trainable_boundary_working', False)}",
        f"- stage2_bootstrap_ready: {ans.get('stage2_bootstrap_ready', False)}",
        "",
        "## Freeze Boundary",
        f"- stage1_trainable_parameter_count: {boundary.get('stage1_trainable_parameter_count', 0)}",
        f"- semantic_trainable_parameter_count: {boundary.get('semantic_trainable_parameter_count', 0)}",
        f"- stage1_grad_detected_after_backward: {boundary.get('stage1_grad_detected_after_backward', False)}",
        f"- semantic_grad_norm: {boundary.get('semantic_grad_norm', 0.0)}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    set_seed(int(args.seed))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    runtime_meta: Dict[str, Any] = {
        "source": "manual",
        "num_workers": 0,
        "pin_memory": False,
        "persistent_workers": False,
        "prefetch_factor": 2,
        "single_gpu_only": True,
    }
    if bool(args.use_recommended_runtime):
        rt = _load_runtime_config(args.recommended_runtime_json)
        runtime_meta = {
            "source": "recommended_runtime_json",
            "path": str(args.recommended_runtime_json),
            "num_workers": int(rt.num_workers),
            "pin_memory": bool(rt.pin_memory),
            "persistent_workers": bool(rt.persistent_workers),
            "prefetch_factor": int(rt.prefetch_factor),
            "single_gpu_only": bool(rt.single_gpu_only),
            "selected_gpu_id_runtime_json": int(rt.selected_gpu_id),
        }

    train_cfg = Stage2SemanticDatasetConfig(
        dataset_names=[str(x) for x in args.dataset_names],
        split=str(args.train_split),
        contract_path=str(args.contract_path),
        obs_len=int(args.obs_len),
        fut_len=int(args.fut_len),
        max_tokens=int(args.max_tokens),
        max_samples_per_dataset=int(args.max_samples_train),
        semantic_patch_radius=int(args.semantic_patch_radius),
    )
    val_cfg = Stage2SemanticDatasetConfig(
        dataset_names=[str(x) for x in args.dataset_names],
        split=str(args.val_split),
        contract_path=str(args.contract_path),
        obs_len=int(args.obs_len),
        fut_len=int(args.fut_len),
        max_tokens=int(args.max_tokens),
        max_samples_per_dataset=int(args.max_samples_val),
        semantic_patch_radius=int(args.semantic_patch_radius),
    )

    train_ds = Stage2SemanticDataset(train_cfg)
    val_ds = Stage2SemanticDataset(val_cfg)

    num_workers = int(runtime_meta.get("num_workers", 0))
    pin_memory = bool(runtime_meta.get("pin_memory", False))
    persistent_workers = bool(runtime_meta.get("persistent_workers", False))
    prefetch_factor = int(runtime_meta.get("prefetch_factor", 2))

    train_kwargs: Dict[str, Any] = {
        "dataset": train_ds,
        "batch_size": int(args.batch_size),
        "shuffle": True,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "collate_fn": stage2_semantic_collate_fn,
    }
    if num_workers > 0:
        train_kwargs["persistent_workers"] = persistent_workers
        train_kwargs["prefetch_factor"] = prefetch_factor

    train_loader = DataLoader(**train_kwargs)
    val_loader = DataLoader(
        val_ds,
        batch_size=max(1, int(args.batch_size)),
        shuffle=False,
        num_workers=0,
        pin_memory=pin_memory,
        collate_fn=stage2_semantic_collate_fn,
    )

    stage1_loaded = False
    stage1_error = ""
    stage1_meta: Dict[str, Any] = {}
    try:
        stage1_model, stage1_meta = _load_frozen_stage1_backbone(args=args, device=device)
        stage1_loaded = True
    except Exception as exc:
        stage1_model = None
        stage1_error = f"{type(exc).__name__}: {exc}"

    semantic_encoder = SemanticEncoder(
        SemanticEncoderConfig(
            input_dim=10,
            hidden_dim=int(args.semantic_hidden_dim),
            output_dim=int(args.semantic_embed_dim),
            dropout=0.1,
        )
    ).to(device)

    fusion_hidden_dim = int(build_tracewm_v2_config(str(args.stage1_model_preset)).d_model)
    if stage1_loaded and stage1_model is not None:
        fusion_hidden_dim = int(stage1_model.config.d_model)

    semantic_fusion = SemanticFusion(
        SemanticFusionConfig(
            hidden_dim=fusion_hidden_dim,
            semantic_dim=int(args.semantic_embed_dim),
            dropout=0.1,
        )
    ).to(device)
    readout_head = torch.nn.Linear(fusion_hidden_dim, 2).to(device)

    trainable_modules = {
        "semantic_encoder": semantic_encoder,
        "semantic_fusion": semantic_fusion,
        "readout_head": readout_head,
    }
    trainable_params: List[torch.nn.Parameter] = []
    for module in trainable_modules.values():
        trainable_params.extend([p for p in module.parameters() if p.requires_grad])

    optimizer = torch.optim.AdamW(trainable_params, lr=float(args.lr), weight_decay=float(args.weight_decay))

    smoke_losses: List[float] = []
    semantic_input_nonempty = False
    semantic_forward_success = False
    stage1_grad_detected = False
    semantic_grad_norm = 0.0
    gate_mean = 0.0

    if stage1_loaded and stage1_model is not None:
        train_iter = iter(train_loader)
        for _step in range(max(int(args.smoke_steps), 1)):
            try:
                raw_batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                raw_batch = next(train_iter)

            batch = _to_device(raw_batch, device=device, non_blocking=bool(pin_memory and device.type == "cuda"))
            shifted = _prepare_shifted(batch)

            token_mask = batch["token_mask"]
            semantic_mask = (batch["semantic_mask"] & token_mask).to(torch.bool)
            semantic_input_nonempty = bool(semantic_input_nonempty or semantic_mask.any().item())

            with torch.no_grad():
                stage1_out = stage1_model(shifted, token_mask=token_mask)
            hidden = stage1_out["hidden"]

            semantic_encoded = semantic_encoder(batch["semantic_features"])
            fused_hidden, aux = semantic_fusion(hidden, semantic_encoded, token_mask=token_mask)
            gate_mean = float(aux.get("gate_mean", 0.0))
            semantic_forward_success = True

            pred_coord = readout_head(fused_hidden[:, int(args.obs_len) :])
            target_coord = batch["fut_state"][..., 0:2]
            valid_mask = batch["fut_valid"] & token_mask[:, None, :]

            loss = _masked_coord_loss(pred_coord=pred_coord, target_coord=target_coord, valid_mask=valid_mask)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            stage1_grad_detected = False
            for p in stage1_model.parameters():
                if p.grad is not None and float(p.grad.detach().abs().sum().item()) > 0.0:
                    stage1_grad_detected = True
                    break

            semantic_grad_sq = 0.0
            for p in trainable_params:
                if p.grad is None:
                    continue
                g = p.grad.detach()
                semantic_grad_sq += float((g * g).sum().item())
            semantic_grad_norm = float(np.sqrt(max(semantic_grad_sq, 0.0)))

            optimizer.step()
            smoke_losses.append(float(loss.detach().cpu().item()))

    eval_loss = None
    if stage1_loaded and stage1_model is not None:
        try:
            val_batch_raw = next(iter(val_loader))
            val_batch = _to_device(val_batch_raw, device=device, non_blocking=bool(pin_memory and device.type == "cuda"))
            val_shifted = _prepare_shifted(val_batch)
            with torch.no_grad():
                val_stage1 = stage1_model(val_shifted, token_mask=val_batch["token_mask"])
                val_sem = semantic_encoder(val_batch["semantic_features"])
                val_fused, _ = semantic_fusion(val_stage1["hidden"], val_sem, token_mask=val_batch["token_mask"])
                val_pred = readout_head(val_fused[:, int(args.obs_len) :])
                val_target = val_batch["fut_state"][..., 0:2]
                val_mask = val_batch["fut_valid"] & val_batch["token_mask"][:, None, :]
                eval_loss = float(_masked_coord_loss(val_pred, val_target, val_mask).detach().cpu().item())
        except Exception:
            eval_loss = None

    boundary_ok = bool((not stage1_grad_detected) and semantic_grad_norm > 0.0)
    bootstrap_ready = bool(stage1_loaded and semantic_input_nonempty and semantic_forward_success and boundary_ok)
    next_step_choice = "start_stage2_small_train" if bootstrap_ready else "refine_stage2_bootstrap"

    payload = {
        "generated_at_utc": now_iso(),
        "run_name": str(args.run_name),
        "runtime": runtime_meta,
        "stage1_backbone": {
            "load_success": bool(stage1_loaded),
            "load_error": str(stage1_error),
            **stage1_meta,
        },
        "semantic_branch": {
            "semantic_input_nonempty": bool(semantic_input_nonempty),
            "semantic_forward_success": bool(semantic_forward_success),
            "semantic_feature_dim": 10,
            "fusion_gate_mean": float(gate_mean),
        },
        "freeze_trainable_boundary": {
            "stage1_trainable_parameter_count": int(stage1_meta.get("trainable_parameter_count", 0)),
            "semantic_trainable_parameter_count": int(sum(p.numel() for p in trainable_params)),
            "stage1_grad_detected_after_backward": bool(stage1_grad_detected),
            "semantic_grad_norm": float(semantic_grad_norm),
            "boundary_ok": bool(boundary_ok),
        },
        "smoke_train": {
            "steps_requested": int(args.smoke_steps),
            "steps_completed": int(len(smoke_losses)),
            "loss_history": smoke_losses,
            "last_loss": float(smoke_losses[-1]) if smoke_losses else None,
        },
        "smoke_eval": {
            "val_loss": eval_loss,
        },
        "answers": {
            "stage1_frozen_backbone_loadable": bool(stage1_loaded),
            "semantic_branch_accepts_inputs": bool(semantic_input_nonempty and semantic_forward_success),
            "freeze_trainable_boundary_working": bool(boundary_ok),
            "stage2_bootstrap_ready": bool(bootstrap_ready),
        },
        "bootstrap_ready": bool(bootstrap_ready),
        "allowed_next_step_choice": [
            "start_stage2_small_train",
            "refine_stage2_bootstrap",
        ],
        "next_step_choice": str(next_step_choice),
    }

    smoke_json = Path(args.smoke_json)
    results_md = Path(args.results_md)
    _write_json(smoke_json, payload)
    _write_md(results_md, payload)

    print(f"[stage2-bootstrap] smoke_json={smoke_json}")
    print(f"[stage2-bootstrap] results_md={results_md}")
    print(f"[stage2-bootstrap] bootstrap_ready={payload['bootstrap_ready']}")
    print(f"[stage2-bootstrap] next_step_choice={payload['next_step_choice']}")


if __name__ == "__main__":
    main()
