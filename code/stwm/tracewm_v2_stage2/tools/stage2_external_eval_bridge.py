#!/usr/bin/env python3
from __future__ import annotations

from argparse import ArgumentParser
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List
import json

import numpy as np
import torch
from torch.utils.data import DataLoader

from stwm.tracewm_v2_stage2.datasets.stage2_semantic_dataset import (
    Stage2SemanticDataset,
    Stage2SemanticDatasetConfig,
    stage2_semantic_collate_fn,
)
from stwm.tracewm_v2_stage2.models.semantic_encoder import SemanticEncoder, SemanticEncoderConfig
from stwm.tracewm_v2_stage2.models.semantic_fusion import SemanticFusion, SemanticFusionConfig
from stwm.tracewm_v2_stage2.trainers import train_tracewm_stage2_smalltrain as stage2_trainer


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class _Stage1LoadArgs:
    stage1_backbone_checkpoint: str
    stage1_model_preset: str


def parse_args() -> Any:
    p = ArgumentParser(description="Stage2 external evaluation bridge for frozen core mainline")
    p.add_argument(
        "--core-mainline-final-json",
        default="/home/chen034/workspace/stwm/reports/stage2_core_mainline_train_final_20260408.json",
    )
    p.add_argument(
        "--core-mainline-raw-json",
        default="/home/chen034/workspace/stwm/reports/stage2_core_mainline_train_raw_20260408.json",
    )
    p.add_argument(
        "--stage2-contract-path",
        default="/home/chen034/workspace/stwm/reports/stage2_bootstrap_data_contract_20260408.json",
    )
    p.add_argument(
        "--checkpoint-under-test",
        default="/home/chen034/workspace/stwm/outputs/checkpoints/stage2_core_mainline_train_20260408/best.pt",
    )
    p.add_argument(
        "--secondary-checkpoint",
        default="/home/chen034/workspace/stwm/outputs/checkpoints/stage2_core_mainline_train_20260408/latest.pt",
    )
    p.add_argument(
        "--bridge-json",
        default="/home/chen034/workspace/stwm/reports/stage2_external_eval_bridge_20260408.json",
    )
    p.add_argument(
        "--results-md",
        default="/home/chen034/workspace/stwm/docs/STAGE2_EXTERNAL_EVAL_BRIDGE_RESULTS_20260408.md",
    )
    p.add_argument(
        "--tap-style-payload-npz",
        default="/home/chen034/workspace/stwm/reports/stage2_external_eval_bridge_tap_style_payload_20260408.npz",
    )
    p.add_argument(
        "--tap-style-secondary-payload-npz",
        default="/home/chen034/workspace/stwm/reports/stage2_external_eval_bridge_tap_style_payload_latest_20260408.npz",
    )
    p.add_argument("--max-eval-batches", type=int, default=8)
    return p.parse_args()


def _read_json(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"json not found: {p}")
    payload = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"json payload must be dict: {p}")
    return payload


def _write_json(path: str | Path, payload: Dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def _f(x: Any, default: float = 1e9) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def _i(x: Any, default: int) -> int:
    try:
        return int(x)
    except Exception:
        return int(default)


def _norm_dataset(name: str) -> str:
    return str(name).strip().lower()


def _norm_dataset_list(names: List[Any]) -> List[str]:
    return [_norm_dataset(str(x)) for x in names]


def _validate_fixed_facts(final_payload: Dict[str, Any]) -> None:
    if str(final_payload.get("current_mainline_semantic_source", "")) != "crop_visual_encoder":
        raise RuntimeError("current_mainline_semantic_source must be crop_visual_encoder")
    if not bool(final_payload.get("frozen_boundary_kept_correct", False)):
        raise RuntimeError("frozen_boundary_kept_correct must be true")
    if str(final_payload.get("next_step_choice", "")) != "freeze_stage2_core_mainline":
        raise RuntimeError("next_step_choice must be freeze_stage2_core_mainline before external bridge")


def _validate_core_only_eval_binding(eval_datasets: List[str]) -> None:
    ds = set(_norm_dataset_list(eval_datasets))
    if ds != {"vspw", "vipseg"}:
        raise RuntimeError(f"external-eval bridge must stay core-only (vspw+vipseg), got={sorted(ds)}")
    forbidden = {"burst", "tao", "visor"}
    if ds & forbidden:
        raise RuntimeError(f"forbidden dataset found in eval binding: {sorted(ds & forbidden)}")


def _validate_contract_exclusions(contract: Dict[str, Any]) -> None:
    excluded = contract.get("excluded_datasets", []) if isinstance(contract.get("excluded_datasets", []), list) else []
    emap = {str(x.get("dataset_name", "")).upper(): x for x in excluded if isinstance(x, dict)}
    if not bool(emap.get("TAO", {}).get("not_in_current_bootstrap", False)):
        raise RuntimeError("TAO must remain excluded")
    if not bool(emap.get("VISOR", {}).get("not_in_current_bootstrap", False)):
        raise RuntimeError("VISOR must remain excluded")


def _load_stage2_modules(
    *,
    checkpoint_path: str,
    raw_payload: Dict[str, Any],
    device: torch.device,
    semantic_source_mainline: str,
) -> Dict[str, Any]:
    ckpt = stage2_trainer._safe_load_checkpoint(checkpoint_path, device=device)
    ckpt_args = ckpt.get("args", {}) if isinstance(ckpt.get("args", {}), dict) else {}

    stage1_meta = raw_payload.get("stage1_backbone", {}) if isinstance(raw_payload.get("stage1_backbone", {}), dict) else {}
    stage1_ckpt = str(stage1_meta.get("checkpoint_path", ckpt_args.get("stage1_backbone_checkpoint", "")))
    stage1_preset = str(stage1_meta.get("model_preset", ckpt_args.get("stage1_model_preset", "prototype_220m")))
    if not stage1_ckpt:
        raise RuntimeError("stage1 backbone checkpoint path missing for external eval bridge")

    stage1_load_args = _Stage1LoadArgs(
        stage1_backbone_checkpoint=stage1_ckpt,
        stage1_model_preset=stage1_preset,
    )
    stage1_model, stage1_load_meta = stage2_trainer._load_frozen_stage1_backbone(args=stage1_load_args, device=device)

    semantic_hidden_dim = _i(ckpt_args.get("semantic_hidden_dim", 256), 256)
    semantic_embed_dim = _i(ckpt_args.get("semantic_embed_dim", 256), 256)
    legacy_source = str(ckpt_args.get("legacy_semantic_source", "hand_crafted_stats"))

    semantic_encoder = SemanticEncoder(
        SemanticEncoderConfig(
            input_dim=10,
            hidden_dim=semantic_hidden_dim,
            output_dim=semantic_embed_dim,
            dropout=0.1,
            mainline_source=str(semantic_source_mainline),
            legacy_source=legacy_source,
        )
    ).to(device)
    fusion_hidden_dim = int(stage1_model.config.d_model)
    semantic_fusion = SemanticFusion(
        SemanticFusionConfig(
            hidden_dim=fusion_hidden_dim,
            semantic_dim=semantic_embed_dim,
            dropout=0.1,
        )
    ).to(device)
    readout_head = torch.nn.Linear(fusion_hidden_dim, 2).to(device)

    semantic_encoder.load_state_dict(ckpt.get("semantic_encoder_state_dict", {}), strict=True)
    semantic_fusion.load_state_dict(ckpt.get("semantic_fusion_state_dict", {}), strict=True)
    readout_head.load_state_dict(ckpt.get("readout_head_state_dict", {}), strict=True)

    semantic_encoder.eval()
    semantic_fusion.eval()
    readout_head.eval()

    return {
        "checkpoint_payload": ckpt,
        "checkpoint_args": ckpt_args,
        "stage1_model": stage1_model,
        "stage1_load_meta": stage1_load_meta,
        "semantic_encoder": semantic_encoder,
        "semantic_fusion": semantic_fusion,
        "readout_head": readout_head,
        "legacy_semantic_source": legacy_source,
    }


def _build_eval_loader(
    *,
    dataset_names: List[str],
    checkpoint_args: Dict[str, Any],
    stage2_contract_path: str,
    semantic_source_mainline: str,
    device: torch.device,
) -> DataLoader:
    cfg = Stage2SemanticDatasetConfig(
        dataset_names=[str(x) for x in dataset_names],
        split=str(checkpoint_args.get("val_split", "val")),
        contract_path=str(stage2_contract_path),
        obs_len=_i(checkpoint_args.get("obs_len", 8), 8),
        fut_len=_i(checkpoint_args.get("fut_len", 8), 8),
        max_tokens=_i(checkpoint_args.get("max_tokens", 64), 64),
        max_samples_per_dataset=_i(checkpoint_args.get("max_samples_val", 24), 24),
        semantic_patch_radius=_i(checkpoint_args.get("semantic_patch_radius", 12), 12),
        semantic_crop_size=_i(checkpoint_args.get("semantic_crop_size", 64), 64),
        semantic_source_mainline=str(semantic_source_mainline),
    )
    ds = Stage2SemanticDataset(cfg)
    loader = DataLoader(
        ds,
        batch_size=max(1, _i(checkpoint_args.get("batch_size", 2), 2)),
        shuffle=False,
        num_workers=0,
        pin_memory=bool(device.type == "cuda"),
        collate_fn=stage2_semantic_collate_fn,
    )
    return loader


def _collect_tap_style_payload(
    *,
    stage1_model: Any,
    semantic_encoder: Any,
    semantic_fusion: Any,
    readout_head: Any,
    loader: DataLoader,
    device: torch.device,
    semantic_source_mainline: str,
    obs_len: int,
    fut_len: int,
    max_eval_batches: int,
    payload_npz_path: str,
) -> Dict[str, Any]:
    pred_tracks: List[np.ndarray] = []
    gt_tracks: List[np.ndarray] = []
    valid_masks: List[np.ndarray] = []

    with torch.no_grad():
        for bi, raw_batch in enumerate(loader):
            if int(max_eval_batches) > 0 and bi >= int(max_eval_batches):
                break
            batch = stage2_trainer._to_device(raw_batch, device=device, non_blocking=bool(device.type == "cuda"))
            out = stage2_trainer._free_rollout_predict(
                stage1_model=stage1_model,
                semantic_encoder=semantic_encoder,
                semantic_fusion=semantic_fusion,
                readout_head=readout_head,
                batch=batch,
                obs_len=int(obs_len),
                fut_len=int(fut_len),
                semantic_source_mainline=str(semantic_source_mainline),
            )

            pred_tracks.append(out["pred_coord"].detach().cpu().numpy().astype(np.float32))
            gt_tracks.append(out["target_coord"].detach().cpu().numpy().astype(np.float32))
            valid_masks.append(out["valid_mask"].detach().cpu().numpy().astype(np.bool_))

    if not pred_tracks:
        raise RuntimeError("tap-style payload collection got zero eval batches")

    pred_arr = np.concatenate(pred_tracks, axis=0)
    gt_arr = np.concatenate(gt_tracks, axis=0)
    valid_arr = np.concatenate(valid_masks, axis=0)
    query_points_2d = gt_arr[:, 0, :, :]

    l2 = np.sqrt(np.sum((pred_arr - gt_arr) ** 2, axis=-1))
    mask_f = valid_arr.astype(np.float32)
    mean_l2 = float(np.sum(l2 * mask_f) / max(float(np.sum(mask_f)), 1.0))

    endpoint_l2 = np.sqrt(np.sum((pred_arr[:, -1] - gt_arr[:, -1]) ** 2, axis=-1))
    endpoint_mask = valid_arr[:, -1].astype(np.float32)
    endpoint_metric = float(np.sum(endpoint_l2 * endpoint_mask) / max(float(np.sum(endpoint_mask)), 1.0))

    payload_path = Path(payload_npz_path)
    payload_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        payload_path,
        predicted_tracks_2d=pred_arr,
        gt_tracks_2d=gt_arr,
        visibility_mask=valid_arr,
        occluded_mask=np.logical_not(valid_arr),
        query_points_2d=query_points_2d,
    )

    return {
        "payload_npz": str(payload_path),
        "payload_shapes": {
            "predicted_tracks_2d": list(pred_arr.shape),
            "gt_tracks_2d": list(gt_arr.shape),
            "visibility_mask": list(valid_arr.shape),
            "query_points_2d": list(query_points_2d.shape),
        },
        "tap_style_bridge_metrics": {
            "free_rollout_coord_mean_l2": float(mean_l2),
            "free_rollout_endpoint_l2": float(endpoint_metric),
            "frames_evaluated": int(pred_arr.shape[1]),
            "tokens_per_frame": int(pred_arr.shape[2]),
            "samples": int(pred_arr.shape[0]),
        },
    }


def _evaluate_checkpoint(
    *,
    checkpoint_path: str,
    payload_npz_path: str,
    raw_payload: Dict[str, Any],
    stage2_contract_path: str,
    eval_datasets: List[str],
    semantic_source_mainline: str,
    max_eval_batches: int,
    device: torch.device,
) -> Dict[str, Any]:
    modules = _load_stage2_modules(
        checkpoint_path=checkpoint_path,
        raw_payload=raw_payload,
        device=device,
        semantic_source_mainline=semantic_source_mainline,
    )
    ckpt_args = modules["checkpoint_args"]

    loader = _build_eval_loader(
        dataset_names=eval_datasets,
        checkpoint_args=ckpt_args,
        stage2_contract_path=stage2_contract_path,
        semantic_source_mainline=semantic_source_mainline,
        device=device,
    )

    internal_metrics = stage2_trainer._evaluate(
        stage1_model=modules["stage1_model"],
        semantic_encoder=modules["semantic_encoder"],
        semantic_fusion=modules["semantic_fusion"],
        readout_head=modules["readout_head"],
        loader=loader,
        device=device,
        pin_memory=bool(device.type == "cuda"),
        obs_len=_i(ckpt_args.get("obs_len", 8), 8),
        fut_len=_i(ckpt_args.get("fut_len", 8), 8),
        max_batches=int(max_eval_batches),
        semantic_source_mainline=str(semantic_source_mainline),
    )

    tap_payload = _collect_tap_style_payload(
        stage1_model=modules["stage1_model"],
        semantic_encoder=modules["semantic_encoder"],
        semantic_fusion=modules["semantic_fusion"],
        readout_head=modules["readout_head"],
        loader=loader,
        device=device,
        semantic_source_mainline=str(semantic_source_mainline),
        obs_len=_i(ckpt_args.get("obs_len", 8), 8),
        fut_len=_i(ckpt_args.get("fut_len", 8), 8),
        max_eval_batches=int(max_eval_batches),
        payload_npz_path=str(payload_npz_path),
    )

    tap_style_eval = {
        "status": "implemented_and_run",
        "compatible_external_eval_payload": True,
        "bridge_protocol": "stage2_tap_style_proxy_v1",
        "official_tapvid_evaluator_connected": False,
        "payload": tap_payload,
        "metrics": tap_payload.get("tap_style_bridge_metrics", {}),
    }

    tap3d_style_eval = {
        "status": "not_yet_implemented",
        "compatible_external_eval_payload": False,
        "blocking_reason": "stage2 core state currently exposes 2D track targets only; no verified 3D target alignment is available in bridge path",
        "missing_component": [
            "tap3d_aligned_ground_truth_trajectories",
            "camera_geometry_projection_or_lift_module",
            "official_tapvid3d_metric_adapter",
        ],
    }

    return {
        "checkpoint_path": str(checkpoint_path),
        "internal_metrics_reference": {
            "teacher_forced_coord_loss": _f(internal_metrics.get("teacher_forced_coord_loss"), 1e9),
            "free_rollout_coord_mean_l2": _f(internal_metrics.get("free_rollout_coord_mean_l2"), 1e9),
            "free_rollout_endpoint_l2": _f(internal_metrics.get("free_rollout_endpoint_l2"), 1e9),
            "total_loss_reference": _f(internal_metrics.get("total_loss_reference"), 1e9),
        },
        "tap_style_eval": tap_style_eval,
        "tap3d_style_eval": tap3d_style_eval,
        "legacy_semantic_source": str(modules["legacy_semantic_source"]),
        "eval_batches": int(max_eval_batches),
    }


def _write_md(path: str | Path, payload: Dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Stage2 External Eval Bridge Results",
        "",
        f"- generated_at_utc: {payload.get('generated_at_utc', '')}",
        f"- current_stage2_mainline_checkpoint: {payload.get('current_stage2_mainline_checkpoint', '')}",
        f"- current_mainline_semantic_source: {payload.get('current_mainline_semantic_source', '')}",
        f"- frozen_boundary_kept_correct: {bool(payload.get('frozen_boundary_kept_correct', False))}",
        f"- external_eval_connected: {bool(payload.get('external_eval_connected', False))}",
        f"- tap_style_eval_status: {payload.get('tap_style_eval_status', '')}",
        f"- tap3d_style_eval_status: {payload.get('tap3d_style_eval_status', '')}",
        f"- readiness: {payload.get('readiness', '')}",
        f"- next_step_choice: {payload.get('next_step_choice', '')}",
        "",
        "## Eval Binding",
        f"- datasets_bound_for_eval: {payload.get('datasets_bound_for_eval', [])}",
        "",
    ]

    primary = payload.get("primary_checkpoint_eval", {}) if isinstance(payload.get("primary_checkpoint_eval", {}), dict) else {}
    tap_style = primary.get("tap_style_eval", {}) if isinstance(primary.get("tap_style_eval", {}), dict) else {}
    tap3d = primary.get("tap3d_style_eval", {}) if isinstance(primary.get("tap3d_style_eval", {}), dict) else {}
    tap_style_metrics = tap_style.get("metrics", {}) if isinstance(tap_style.get("metrics", {}), dict) else {}

    lines.extend(
        [
            "## Primary Checkpoint Eval",
            f"- checkpoint_under_test: {primary.get('checkpoint_path', '')}",
            f"- tap_style_eval_status: {tap_style.get('status', '')}",
            f"- tap3d_style_eval_status: {tap3d.get('status', '')}",
            f"- tap_style_payload_npz: {((tap_style.get('payload', {}) if isinstance(tap_style.get('payload', {}), dict) else {}).get('payload_npz', ''))}",
            f"- tap_style_free_rollout_endpoint_l2: {float(tap_style_metrics.get('free_rollout_endpoint_l2', 1e9)):.6f}",
            f"- tap_style_free_rollout_coord_mean_l2: {float(tap_style_metrics.get('free_rollout_coord_mean_l2', 1e9)):.6f}",
            f"- tap3d_blocking_reason: {tap3d.get('blocking_reason', '')}",
        ]
    )

    p.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()

    final_payload = _read_json(args.core_mainline_final_json)
    raw_payload = _read_json(args.core_mainline_raw_json)
    contract_payload = _read_json(args.stage2_contract_path)

    _validate_fixed_facts(final_payload)
    _validate_contract_exclusions(contract_payload)

    eval_datasets = final_payload.get("datasets_bound_for_eval", [])
    if not isinstance(eval_datasets, list):
        eval_datasets = []
    eval_datasets = [str(x) for x in eval_datasets]
    _validate_core_only_eval_binding(eval_datasets)

    mainline_source = str(final_payload.get("current_mainline_semantic_source", ""))
    if mainline_source != "crop_visual_encoder":
        raise RuntimeError("external eval bridge requires crop_visual_encoder mainline")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    primary = _evaluate_checkpoint(
        checkpoint_path=str(args.checkpoint_under_test),
        payload_npz_path=str(args.tap_style_payload_npz),
        raw_payload=raw_payload,
        stage2_contract_path=str(args.stage2_contract_path),
        eval_datasets=eval_datasets,
        semantic_source_mainline=mainline_source,
        max_eval_batches=int(args.max_eval_batches),
        device=device,
    )

    secondary: Dict[str, Any] | None = None
    secondary_path = Path(str(args.secondary_checkpoint)).expanduser()
    if secondary_path.exists() and str(secondary_path) != str(Path(str(args.checkpoint_under_test)).expanduser()):
        secondary = _evaluate_checkpoint(
            checkpoint_path=str(secondary_path),
            payload_npz_path=str(args.tap_style_secondary_payload_npz),
            raw_payload=raw_payload,
            stage2_contract_path=str(args.stage2_contract_path),
            eval_datasets=eval_datasets,
            semantic_source_mainline=mainline_source,
            max_eval_batches=int(args.max_eval_batches),
            device=device,
        )

    tap_style_eval_status = str((primary.get("tap_style_eval", {}) if isinstance(primary.get("tap_style_eval", {}), dict) else {}).get("status", "not_yet_implemented"))
    tap3d_style_eval_status = str((primary.get("tap3d_style_eval", {}) if isinstance(primary.get("tap3d_style_eval", {}), dict) else {}).get("status", "not_yet_implemented"))

    external_eval_connected = bool(tap_style_eval_status == "implemented_and_run")
    if tap_style_eval_status == "implemented_and_run" and tap3d_style_eval_status == "implemented_and_run":
        readiness = "paper_eval_ready"
        next_step_choice = "finalize_stage2_mainline_and_prepare_paper_results"
    elif external_eval_connected:
        readiness = "training_ready_but_eval_gap_remains"
        next_step_choice = "do_one_targeted_external_eval_fix"
    else:
        readiness = "training_ready_but_eval_gap_remains"
        next_step_choice = "revisit_stage2_eval_inputs"

    bridge_payload = {
        "generated_at_utc": now_iso(),
        "round": "stage2_external_eval_bridge_20260408",
        "external_eval_protocol_version": "stage2_external_eval_bridge_20260408_v1",
        "checkpoint_under_test": str(args.checkpoint_under_test),
        "current_stage2_mainline_checkpoint": str(args.checkpoint_under_test),
        "secondary_checkpoint_reference": str(secondary_path) if secondary is not None else "",
        "datasets_bound_for_eval": eval_datasets,
        "current_mainline_semantic_source": mainline_source,
        "frozen_boundary_kept_correct": bool(final_payload.get("frozen_boundary_kept_correct", False)),
        "tap_style_eval_status": tap_style_eval_status,
        "tap3d_style_eval_status": tap3d_style_eval_status,
        "external_eval_connected": bool(external_eval_connected),
        "readiness": str(readiness),
        "allowed_next_step_choice": [
            "finalize_stage2_mainline_and_prepare_paper_results",
            "do_one_targeted_external_eval_fix",
            "revisit_stage2_eval_inputs",
        ],
        "next_step_choice": str(next_step_choice),
        "primary_checkpoint_eval": primary,
        "secondary_checkpoint_eval": secondary,
    }

    _write_json(args.bridge_json, bridge_payload)
    _write_md(args.results_md, bridge_payload)

    print(f"[stage2-external-eval-bridge] bridge_json={args.bridge_json}")
    print(f"[stage2-external-eval-bridge] results_md={args.results_md}")
    print(f"[stage2-external-eval-bridge] checkpoint_under_test={bridge_payload['checkpoint_under_test']}")
    print(f"[stage2-external-eval-bridge] tap_style_eval_status={bridge_payload['tap_style_eval_status']}")
    print(f"[stage2-external-eval-bridge] tap3d_style_eval_status={bridge_payload['tap3d_style_eval_status']}")
    print(f"[stage2-external-eval-bridge] readiness={bridge_payload['readiness']}")
    print(f"[stage2-external-eval-bridge] next_step_choice={bridge_payload['next_step_choice']}")


if __name__ == "__main__":
    main()
