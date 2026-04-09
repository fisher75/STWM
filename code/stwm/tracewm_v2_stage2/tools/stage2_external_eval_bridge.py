#!/usr/bin/env python3
from __future__ import annotations

from argparse import ArgumentParser
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List
import json
import os
import subprocess

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
from stwm.tracewm_v2_stage2.tools.export_stage2_tap_payload import export_proxy_payload_to_tapvid
from stwm.tracewm_v2_stage2.tools.run_stage2_tap_eval import (
    DEFAULT_TAPNET_PYTHON,
    run_official_tapvid_eval,
)
from stwm.tracewm_v2_stage2.trainers import train_tracewm_stage2_smalltrain as stage2_trainer


TAP_STYLE_ALLOWED = {
    "fully_implemented_and_run",
    "partially_bridged",
    "proxy_only",
    "not_yet_implemented",
}
TAP3D_ALLOWED = {
    "fully_implemented_and_run",
    "partially_bridged",
    "not_yet_implemented",
}
READINESS_ALLOWED = {
    "paper_eval_ready",
    "training_ready_but_eval_gap_remains",
    "eval_not_ready",
}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _env_or_default(env_name: str, default: str) -> str:
    val = str(os.environ.get(env_name, "")).strip()
    if val:
        return val
    return str(default)


@dataclass
class _Stage1LoadArgs:
    stage1_backbone_checkpoint: str
    stage1_model_preset: str


def parse_args() -> Any:
    p = ArgumentParser(description="Stage2 external evaluation completion round for the frozen core mainline")
    p.add_argument(
        "--core-mainline-final-json",
        default=_env_or_default("TRACEWM_STAGE2_CORE_MAINLINE_FINAL_JSON", "/home/chen034/workspace/stwm/reports/stage2_core_mainline_train_final_20260408.json"),
    )
    p.add_argument(
        "--core-mainline-raw-json",
        default=_env_or_default("TRACEWM_STAGE2_CORE_MAINLINE_RAW_JSON", "/home/chen034/workspace/stwm/reports/stage2_core_mainline_train_raw_20260408.json"),
    )
    p.add_argument(
        "--stage2-contract-path",
        default=_env_or_default("TRACEWM_STAGE2_CONTRACT_JSON", "/home/chen034/workspace/stwm/reports/stage2_bootstrap_data_contract_20260408.json"),
    )
    p.add_argument(
        "--checkpoint-under-test",
        default=_env_or_default("TRACEWM_STAGE2_MAINLINE_BEST_CKPT", "/home/chen034/workspace/stwm/outputs/checkpoints/stage2_core_mainline_train_20260408/best.pt"),
    )
    p.add_argument(
        "--secondary-checkpoint",
        default=_env_or_default("TRACEWM_STAGE2_MAINLINE_LATEST_CKPT", "/home/chen034/workspace/stwm/outputs/checkpoints/stage2_core_mainline_train_20260408/latest.pt"),
    )
    p.add_argument(
        "--completion-json",
        default=_env_or_default("TRACEWM_STAGE2_EXTERNAL_EVAL_COMPLETION_JSON", "/home/chen034/workspace/stwm/reports/stage2_external_eval_completion_20260408.json"),
    )
    p.add_argument(
        "--results-md",
        default=_env_or_default("TRACEWM_STAGE2_EXTERNAL_EVAL_COMPLETION_MD", "/home/chen034/workspace/stwm/docs/STAGE2_EXTERNAL_EVAL_COMPLETION_RESULTS_20260408.md"),
    )
    p.add_argument(
        "--tap-style-proxy-payload-npz",
        default=_env_or_default("TRACEWM_STAGE2_TAP_PROXY_PAYLOAD", "/home/chen034/workspace/stwm/reports/stage2_external_eval_completion_tap_style_proxy_payload_20260408.npz"),
    )
    p.add_argument(
        "--tap-style-secondary-proxy-payload-npz",
        default=_env_or_default("TRACEWM_STAGE2_TAP_PROXY_PAYLOAD_LATEST", "/home/chen034/workspace/stwm/reports/stage2_external_eval_completion_tap_style_proxy_payload_latest_20260408.npz"),
    )
    p.add_argument(
        "--tap-style-official-payload-npz",
        default=_env_or_default("TRACEWM_STAGE2_TAP_OFFICIAL_PAYLOAD", "/home/chen034/workspace/stwm/reports/stage2_external_eval_completion_tap_style_official_payload_20260408.npz"),
    )
    p.add_argument(
        "--tap-style-secondary-official-payload-npz",
        default=_env_or_default("TRACEWM_STAGE2_TAP_OFFICIAL_PAYLOAD_LATEST", "/home/chen034/workspace/stwm/reports/stage2_external_eval_completion_tap_style_official_payload_latest_20260408.npz"),
    )
    p.add_argument(
        "--tap-style-export-report-json",
        default=_env_or_default("TRACEWM_STAGE2_TAP_EXPORT_JSON", "/home/chen034/workspace/stwm/reports/stage2_external_eval_completion_tap_style_export_20260408.json"),
    )
    p.add_argument(
        "--tap-style-secondary-export-report-json",
        default=_env_or_default("TRACEWM_STAGE2_TAP_EXPORT_JSON_LATEST", "/home/chen034/workspace/stwm/reports/stage2_external_eval_completion_tap_style_export_latest_20260408.json"),
    )
    p.add_argument(
        "--tap-style-official-eval-json",
        default=_env_or_default("TRACEWM_STAGE2_TAP_EVAL_JSON", "/home/chen034/workspace/stwm/reports/stage2_external_eval_completion_tap_style_eval_20260408.json"),
    )
    p.add_argument(
        "--tap-style-secondary-official-eval-json",
        default=_env_or_default("TRACEWM_STAGE2_TAP_EVAL_JSON_LATEST", "/home/chen034/workspace/stwm/reports/stage2_external_eval_completion_tap_style_eval_latest_20260408.json"),
    )
    p.add_argument("--tapnet-python", default=_env_or_default("TRACEWM_TAPNET_PYTHON", DEFAULT_TAPNET_PYTHON))
    p.add_argument("--tap-query-mode", default="first")
    p.add_argument("--tap-raster-resolution", type=int, default=256)
    p.add_argument("--tap3d-dataset-root", default=_env_or_default("TRACEWM_TAPVID3D_DATASET_ROOT", "/home/chen034/workspace/data/tapvid3d/minival_dataset"))
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


def _tail(txt: str, limit: int = 4000) -> str:
    s = str(txt)
    if len(s) <= limit:
        return s
    return s[-limit:]


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
        raise RuntimeError("next_step_choice must be freeze_stage2_core_mainline before external eval completion")


def _validate_core_only_eval_binding(eval_datasets: List[str]) -> None:
    ds = set(_norm_dataset_list(eval_datasets))
    if ds != {"vspw", "vipseg"}:
        raise RuntimeError(f"external eval completion must stay core-only (vspw+vipseg), got={sorted(ds)}")
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
        raise RuntimeError("stage1 backbone checkpoint path missing for external eval completion")

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


def _collect_tap_style_proxy_payload(
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
        raise RuntimeError("tap-style proxy payload collection got zero eval batches")

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
        "coordinate_space": "normalized_xy_in_stage2_state",
        "gt_occluded_points": int((~valid_arr).sum()),
        "total_points": int(valid_arr.size),
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
    proxy_payload_npz_path: str,
    official_payload_npz_path: str,
    export_report_json_path: str,
    official_eval_json_path: str,
    raw_payload: Dict[str, Any],
    stage2_contract_path: str,
    eval_datasets: List[str],
    semantic_source_mainline: str,
    max_eval_batches: int,
    tapnet_python: str,
    tap_query_mode: str,
    tap_raster_resolution: int,
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

    proxy_payload = _collect_tap_style_proxy_payload(
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
        payload_npz_path=str(proxy_payload_npz_path),
    )

    official_payload_export = export_proxy_payload_to_tapvid(
        proxy_payload_npz=proxy_payload["payload_npz"],
        output_npz=official_payload_npz_path,
        output_report_json=export_report_json_path,
        raster_resolution=int(tap_raster_resolution),
        pred_occlusion_mode="all_visible",
        query_time_index=0,
    )

    official_eval = run_official_tapvid_eval(
        tap_payload_npz=official_payload_npz_path,
        output_json=official_eval_json_path,
        tapnet_python=str(tapnet_python),
        query_mode=str(tap_query_mode),
    )

    exact_blocking_reasons: List[str] = []
    exact_missing_components: List[str] = []
    proxy_connected = True
    official_connected = bool(official_eval.get("official_tapvid_evaluator_connected", False))
    official_evaluator_invoked = bool(official_eval.get("official_evaluator_invoked", False))
    official_payload_export = official_payload_export if isinstance(official_payload_export, dict) else {}
    benchmark_native_full_tap_episode = bool(official_payload_export.get("benchmark_native_full_tap_episode", False))
    query_time_matches_official_task = bool(official_payload_export.get("query_time_matches_official_task", False))
    pred_visibility_from_model_output = bool(official_payload_export.get("predicted_visibility_is_model_output", False))
    dataset_binding_is_official_tap_dataset_family = False
    official_task_faithfully_instantiated = bool(
        benchmark_native_full_tap_episode
        and query_time_matches_official_task
        and pred_visibility_from_model_output
        and dataset_binding_is_official_tap_dataset_family
    )

    if official_evaluator_invoked and official_task_faithfully_instantiated:
        tap_style_status = "fully_implemented_and_run"
    elif official_evaluator_invoked or official_connected:
        tap_style_status = "partially_bridged"
    elif proxy_connected:
        tap_style_status = "proxy_only"
    else:
        tap_style_status = "not_yet_implemented"

    if not official_evaluator_invoked:
        exact_blocking_reasons.append(str(official_eval.get("exact_blocking_reason", "official TAP-Vid evaluator could not be invoked")))
        exact_missing_components.append(str(official_eval.get("exact_missing_component", "working official tapvid runtime hook")))
    if official_evaluator_invoked and not official_task_faithfully_instantiated:
        exact_blocking_reasons.extend(
            [
                "current frozen stage2 bridge exports future-only core-eval trajectories rather than benchmark-native full TAP-Vid episodes",
                "current frozen stage2 mainline does not expose a predicted occlusion head; pred_occluded is evaluator-side all-visible adapter output",
                "current evaluation binding remains VSPW+VIPSeg core-only rather than the official TAP-Vid dataset family",
            ]
        )
        exact_missing_components.extend(
            [
                "benchmark-native TAP query sampler over full observed video horizon",
                "model-produced occlusion predictions for TAP-style metrics",
                "official TAP-Vid dataset binding for the frozen stage2 core mainline",
            ]
        )

    if tap_style_status not in TAP_STYLE_ALLOWED:
        raise RuntimeError(f"invalid tap_style_status={tap_style_status}")

    tap_style_eval = {
        "status": tap_style_status,
        "proxy_bridge_connected": bool(proxy_connected),
        "official_evaluator_invoked": bool(official_evaluator_invoked),
        "official_tapvid_evaluator_connected": bool(official_connected),
        "official_task_faithfully_instantiated": bool(official_task_faithfully_instantiated),
        "benchmark_native_full_tap_episode": bool(benchmark_native_full_tap_episode),
        "query_time_matches_official_task": bool(query_time_matches_official_task),
        "pred_visibility_from_model_output": bool(pred_visibility_from_model_output),
        "dataset_binding_is_official_tap_dataset_family": bool(dataset_binding_is_official_tap_dataset_family),
        "bridge_protocol": "stage2_tap_style_completion_v1",
        "proxy_payload": proxy_payload,
        "official_payload_export": official_payload_export,
        "official_eval": official_eval,
        "exact_blocking_reason": exact_blocking_reasons[0] if exact_blocking_reasons else "",
        "exact_blocking_reasons": exact_blocking_reasons,
        "exact_missing_component": exact_missing_components[0] if exact_missing_components else "",
        "exact_missing_components": exact_missing_components,
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
        "legacy_semantic_source": str(modules["legacy_semantic_source"]),
        "eval_batches": int(max_eval_batches),
    }


def _probe_official_tapvid3d_runtime(tapnet_python: str) -> Dict[str, Any]:
    probe_code = r"""
import json
from tapnet.tapvid3d.evaluation import metrics as tap3d_metrics
print(json.dumps({
    "official_tapvid3d_metric_importable": True,
    "tapvid3d_module_path": str(tap3d_metrics.__file__),
}, ensure_ascii=True))
"""
    proc = subprocess.run([str(tapnet_python), "-c", probe_code], text=True, capture_output=True)
    result: Dict[str, Any] = {
        "tapnet_python": str(tapnet_python),
        "returncode": int(proc.returncode),
        "stdout_tail": _tail(proc.stdout),
        "stderr_tail": _tail(proc.stderr),
        "official_tapvid3d_metric_importable": False,
        "tapvid3d_module_path": "",
    }
    if proc.returncode == 0:
        parsed = json.loads(str(proc.stdout).strip())
        if isinstance(parsed, dict):
            result.update(parsed)
    else:
        result["exact_blocking_reason"] = "official TAPVid-3D metric import probe failed"
    return result


def _scan_tapvid3d_dataset(root: str | Path) -> Dict[str, Any]:
    dataset_root = Path(root)
    sources = {}
    total_npz = 0
    for name in ["adt", "pstudio", "drivetrack"]:
        sub = dataset_root / name
        count = len(sorted(sub.glob("*.npz"))) if sub.exists() else 0
        sources[name] = count
        total_npz += count
    return {
        "dataset_root": str(dataset_root),
        "dataset_root_exists": bool(dataset_root.exists()),
        "npz_count_by_source": sources,
        "total_npz_files": int(total_npz),
    }


def _build_tap3d_completion(
    *,
    tapnet_python: str,
    tap3d_dataset_root: str,
) -> Dict[str, Any]:
    runtime_probe = _probe_official_tapvid3d_runtime(str(tapnet_python))
    dataset_probe = _scan_tapvid3d_dataset(str(tap3d_dataset_root))

    aligned_3d_gt_for_current_binding = False
    camera_geometry_or_lifting_available = False
    verified_exporter_to_tracks_xyz_visibility = False
    missing_components = [
        "tap3d_aligned_ground_truth_for_current_core_only_vspw_vipseg_eval_binding",
        "camera_geometry_projection_or_2d_to_3d_lifting_path_for_stage2_bridge",
        "stage2_rollout_exporter_to_official_tapvid3d_tracks_xyz_visibility_format",
    ]
    exact_blocking_reasons = [
        "current frozen stage2 external eval binding is fixed to VSPW+VIPSeg, which does not provide TAPVid-3D aligned XYZ ground truth for the checkpoint under test",
        "current stage2 dataset/bridge path does not export intrinsics, extrinsics, projection, or lifting utilities needed to convert 2D rollout states into camera-consistent 3D trajectories",
        "current evaluator-side completion round does not yet include a verified adapter that emits official TAPVid-3D prediction files with tracks_XYZ and visibility for the frozen stage2 checkpoint",
    ]

    if aligned_3d_gt_for_current_binding and camera_geometry_or_lifting_available and verified_exporter_to_tracks_xyz_visibility:
        status = "partially_bridged"
    else:
        status = "not_yet_implemented"

    if status not in TAP3D_ALLOWED:
        raise RuntimeError(f"invalid tap3d status={status}")

    return {
        "status": status,
        "official_tapvid3d_evaluator_connected": False,
        "runtime_probe": runtime_probe,
        "reference_dataset_probe": dataset_probe,
        "aligned_3d_gt_for_current_binding": bool(aligned_3d_gt_for_current_binding),
        "camera_geometry_projection_or_lifting_path_available": bool(camera_geometry_or_lifting_available),
        "verified_exporter_to_tracks_xyz_visibility": bool(verified_exporter_to_tracks_xyz_visibility),
        "current_stage2_checkpoint_payload_has_3d_tracks": False,
        "current_stage2_checkpoint_payload_has_camera_geometry": False,
        "exact_blocking_reason": exact_blocking_reasons[0],
        "exact_blocking_reasons": exact_blocking_reasons,
        "missing_component": missing_components[0],
        "missing_component_list": missing_components,
    }


def _safe_metric(eval_payload: Dict[str, Any], key: str, default: float = 1e9) -> float:
    tap_style = eval_payload.get("tap_style_eval", {}) if isinstance(eval_payload.get("tap_style_eval", {}), dict) else {}
    official_eval = tap_style.get("official_eval", {}) if isinstance(tap_style.get("official_eval", {}), dict) else {}
    metric_means = official_eval.get("metric_means", {}) if isinstance(official_eval.get("metric_means", {}), dict) else {}
    return _f(metric_means.get(key), default)


def _compare_primary_secondary(primary: Dict[str, Any], secondary: Dict[str, Any] | None) -> Dict[str, Any] | None:
    if not isinstance(secondary, dict):
        return None
    primary_internal = primary.get("internal_metrics_reference", {}) if isinstance(primary.get("internal_metrics_reference", {}), dict) else {}
    secondary_internal = secondary.get("internal_metrics_reference", {}) if isinstance(secondary.get("internal_metrics_reference", {}), dict) else {}
    return {
        "best_checkpoint_path": str(primary.get("checkpoint_path", "")),
        "latest_checkpoint_path": str(secondary.get("checkpoint_path", "")),
        "teacher_forced_coord_loss_best": _f(primary_internal.get("teacher_forced_coord_loss"), 1e9),
        "teacher_forced_coord_loss_latest": _f(secondary_internal.get("teacher_forced_coord_loss"), 1e9),
        "free_rollout_coord_mean_l2_best": _f(primary_internal.get("free_rollout_coord_mean_l2"), 1e9),
        "free_rollout_coord_mean_l2_latest": _f(secondary_internal.get("free_rollout_coord_mean_l2"), 1e9),
        "tapvid_average_jaccard_best": _safe_metric(primary, "average_jaccard"),
        "tapvid_average_jaccard_latest": _safe_metric(secondary, "average_jaccard"),
        "tapvid_average_pts_within_thresh_best": _safe_metric(primary, "average_pts_within_thresh"),
        "tapvid_average_pts_within_thresh_latest": _safe_metric(secondary, "average_pts_within_thresh"),
    }


def _write_md(path: str | Path, payload: Dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    primary = payload.get("primary_checkpoint_eval", {}) if isinstance(payload.get("primary_checkpoint_eval", {}), dict) else {}
    secondary = payload.get("secondary_checkpoint_eval", {}) if isinstance(payload.get("secondary_checkpoint_eval", {}), dict) else {}
    primary_tap = primary.get("tap_style_eval", {}) if isinstance(primary.get("tap_style_eval", {}), dict) else {}
    primary_tap_official = primary_tap.get("official_eval", {}) if isinstance(primary_tap.get("official_eval", {}), dict) else {}
    primary_metric_means = primary_tap_official.get("metric_means", {}) if isinstance(primary_tap_official.get("metric_means", {}), dict) else {}
    tap3d = payload.get("tap3d_completion", {}) if isinstance(payload.get("tap3d_completion", {}), dict) else {}
    compare = payload.get("best_vs_latest_reference", {}) if isinstance(payload.get("best_vs_latest_reference", {}), dict) else {}

    lines = [
        "# Stage2 External Eval Completion Results",
        "",
        "> This document is completion-round status only. The frozen Stage2 mainline was not retrained in this round.",
        "",
        "## Locked Facts",
        f"- current_stage2_mainline_checkpoint: {payload.get('current_stage2_mainline_checkpoint', '')}",
        f"- secondary_checkpoint_reference: {payload.get('secondary_checkpoint_reference', '')}",
        f"- datasets_bound_for_eval: {payload.get('datasets_bound_for_eval', [])}",
        f"- current_mainline_semantic_source: {payload.get('current_mainline_semantic_source', '')}",
        f"- frozen_boundary_kept_correct: {bool(payload.get('frozen_boundary_kept_correct', False))}",
        "",
        "## Completion Status",
        f"- tap_style_eval_status: {payload.get('tap_style_eval_status', '')}",
        f"- tap_style_proxy_bridge_connected: {bool(payload.get('tap_style_proxy_bridge_connected', False))}",
        f"- official_evaluator_invoked: {bool(payload.get('official_evaluator_invoked', False))}",
        f"- official_tapvid_evaluator_connected: {bool(payload.get('official_tapvid_evaluator_connected', False))}",
        f"- official_task_faithfully_instantiated: {bool(payload.get('official_task_faithfully_instantiated', False))}",
        f"- tap3d_style_eval_status: {payload.get('tap3d_style_eval_status', '')}",
        f"- external_eval_readiness: {payload.get('external_eval_readiness', '')}",
        f"- next_step_choice: {payload.get('next_step_choice', '')}",
        "",
        "## TAP-Style Primary Result",
        f"- primary_checkpoint_path: {primary.get('checkpoint_path', '')}",
        f"- proxy_payload_npz: {((primary_tap.get('proxy_payload', {}) if isinstance(primary_tap.get('proxy_payload', {}), dict) else {}).get('payload_npz', ''))}",
        f"- official_payload_npz: {((primary_tap.get('official_payload_export', {}) if isinstance(primary_tap.get('official_payload_export', {}), dict) else {}).get('output_tap_payload_npz', ''))}",
        f"- average_jaccard: {float(primary_metric_means.get('average_jaccard', 1e9)):.6f}",
        f"- average_pts_within_thresh: {float(primary_metric_means.get('average_pts_within_thresh', 1e9)):.6f}",
        f"- occlusion_accuracy: {float(primary_metric_means.get('occlusion_accuracy', 1e9)):.6f}",
        "",
        "## TAP-Style Remaining Gaps",
    ]
    for item in primary_tap.get("exact_blocking_reasons", []):
        lines.append(f"- {item}")

    lines.extend(
        [
            "",
            "## TAP3D Remaining Gaps",
            f"- tap3d_status: {tap3d.get('status', '')}",
        ]
    )
    for item in tap3d.get("exact_blocking_reasons", []):
        lines.append(f"- {item}")

    if compare:
        lines.extend(
            [
                "",
                "## Best vs Latest Reference",
                f"- free_rollout_coord_mean_l2_best: {float(compare.get('free_rollout_coord_mean_l2_best', 1e9)):.6f}",
                f"- free_rollout_coord_mean_l2_latest: {float(compare.get('free_rollout_coord_mean_l2_latest', 1e9)):.6f}",
                f"- tapvid_average_jaccard_best: {float(compare.get('tapvid_average_jaccard_best', 1e9)):.6f}",
                f"- tapvid_average_jaccard_latest: {float(compare.get('tapvid_average_jaccard_latest', 1e9)):.6f}",
            ]
        )

    lines.extend(
        [
            "",
            "## Mandatory Answers",
            f"1. current mainline checkpoint is still `best.pt`: {str(payload.get('current_stage2_mainline_checkpoint', '')).endswith('/best.pt')}",
            f"2. TAP-style is currently: `{payload.get('tap_style_eval_status', '')}`",
            f"3. official TAP evaluator connected: {bool(payload.get('official_tapvid_evaluator_connected', False))}",
            f"4. TAP3D-style progressed to: `{payload.get('tap3d_style_eval_status', '')}`",
            f"5. project readiness is: `{payload.get('external_eval_readiness', '')}`",
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
        raise RuntimeError("external eval completion requires crop_visual_encoder mainline")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    primary = _evaluate_checkpoint(
        checkpoint_path=str(args.checkpoint_under_test),
        proxy_payload_npz_path=str(args.tap_style_proxy_payload_npz),
        official_payload_npz_path=str(args.tap_style_official_payload_npz),
        export_report_json_path=str(args.tap_style_export_report_json),
        official_eval_json_path=str(args.tap_style_official_eval_json),
        raw_payload=raw_payload,
        stage2_contract_path=str(args.stage2_contract_path),
        eval_datasets=eval_datasets,
        semantic_source_mainline=mainline_source,
        max_eval_batches=int(args.max_eval_batches),
        tapnet_python=str(args.tapnet_python),
        tap_query_mode=str(args.tap_query_mode),
        tap_raster_resolution=int(args.tap_raster_resolution),
        device=device,
    )

    secondary: Dict[str, Any] | None = None
    secondary_path = Path(str(args.secondary_checkpoint)).expanduser()
    if secondary_path.exists() and str(secondary_path) != str(Path(str(args.checkpoint_under_test)).expanduser()):
        secondary = _evaluate_checkpoint(
            checkpoint_path=str(secondary_path),
            proxy_payload_npz_path=str(args.tap_style_secondary_proxy_payload_npz),
            official_payload_npz_path=str(args.tap_style_secondary_official_payload_npz),
            export_report_json_path=str(args.tap_style_secondary_export_report_json),
            official_eval_json_path=str(args.tap_style_secondary_official_eval_json),
            raw_payload=raw_payload,
            stage2_contract_path=str(args.stage2_contract_path),
            eval_datasets=eval_datasets,
            semantic_source_mainline=mainline_source,
            max_eval_batches=int(args.max_eval_batches),
            tapnet_python=str(args.tapnet_python),
            tap_query_mode=str(args.tap_query_mode),
            tap_raster_resolution=int(args.tap_raster_resolution),
            device=device,
        )

    tap_style_eval = primary.get("tap_style_eval", {}) if isinstance(primary.get("tap_style_eval", {}), dict) else {}
    tap_style_eval_status = str(tap_style_eval.get("status", "not_yet_implemented"))
    tap_style_proxy_bridge_connected = bool(tap_style_eval.get("proxy_bridge_connected", False))
    official_evaluator_invoked = bool(tap_style_eval.get("official_evaluator_invoked", False))
    official_tapvid_evaluator_connected = bool(tap_style_eval.get("official_tapvid_evaluator_connected", False))
    official_task_faithfully_instantiated = bool(tap_style_eval.get("official_task_faithfully_instantiated", False))

    tap3d_completion = _build_tap3d_completion(
        tapnet_python=str(args.tapnet_python),
        tap3d_dataset_root=str(args.tap3d_dataset_root),
    )
    tap3d_style_eval_status = str(tap3d_completion.get("status", "not_yet_implemented"))

    exact_blocking_reasons: List[str] = []
    exact_blocking_reasons.extend(tap_style_eval.get("exact_blocking_reasons", []) if isinstance(tap_style_eval.get("exact_blocking_reasons", []), list) else [])
    exact_blocking_reasons.extend(tap3d_completion.get("exact_blocking_reasons", []) if isinstance(tap3d_completion.get("exact_blocking_reasons", []), list) else [])

    deduped_blockers: List[str] = []
    seen = set()
    for item in exact_blocking_reasons:
        txt = str(item).strip()
        if txt and txt not in seen:
            deduped_blockers.append(txt)
            seen.add(txt)

    if tap_style_eval_status == "fully_implemented_and_run" and tap3d_style_eval_status == "fully_implemented_and_run":
        external_eval_readiness = "paper_eval_ready"
        next_step_choice = "finalize_stage2_mainline_and_prepare_paper_results"
    elif tap_style_proxy_bridge_connected or official_evaluator_invoked or official_tapvid_evaluator_connected:
        external_eval_readiness = "training_ready_but_eval_gap_remains"
        next_step_choice = "do_one_targeted_external_eval_fix"
    else:
        external_eval_readiness = "eval_not_ready"
        next_step_choice = "do_one_targeted_external_eval_fix"

    if external_eval_readiness not in READINESS_ALLOWED:
        raise RuntimeError(f"invalid readiness={external_eval_readiness}")

    best_vs_latest_reference = _compare_primary_secondary(primary, secondary)

    completion_payload = {
        "generated_at_utc": now_iso(),
        "round": "stage2_external_eval_completion_20260408",
        "external_eval_protocol_version": "stage2_external_eval_completion_20260408_v1",
        "current_stage2_mainline_checkpoint": str(args.checkpoint_under_test),
        "secondary_checkpoint_reference": str(secondary_path) if secondary is not None else "",
        "datasets_bound_for_eval": eval_datasets,
        "current_mainline_semantic_source": mainline_source,
        "frozen_boundary_kept_correct": bool(final_payload.get("frozen_boundary_kept_correct", False)),
        "tap_style_eval_status": tap_style_eval_status,
        "tap_style_proxy_bridge_connected": tap_style_proxy_bridge_connected,
        "official_evaluator_invoked": official_evaluator_invoked,
        "official_tapvid_evaluator_connected": official_tapvid_evaluator_connected,
        "official_task_faithfully_instantiated": official_task_faithfully_instantiated,
        "tap3d_style_eval_status": tap3d_style_eval_status,
        "tap3d_missing_components": tap3d_completion.get("missing_component_list", []),
        "external_eval_readiness": external_eval_readiness,
        "exact_blocking_reasons": deduped_blockers,
        "next_step_choice": next_step_choice,
        "primary_checkpoint_eval": primary,
        "secondary_checkpoint_eval": secondary,
        "tap3d_completion": tap3d_completion,
        "best_vs_latest_reference": best_vs_latest_reference,
    }

    _write_json(args.completion_json, completion_payload)
    _write_md(args.results_md, completion_payload)

    print(f"[stage2-external-eval-completion] completion_json={args.completion_json}")
    print(f"[stage2-external-eval-completion] results_md={args.results_md}")
    print(f"[stage2-external-eval-completion] current_stage2_mainline_checkpoint={completion_payload['current_stage2_mainline_checkpoint']}")
    print(f"[stage2-external-eval-completion] tap_style_eval_status={completion_payload['tap_style_eval_status']}")
    print(f"[stage2-external-eval-completion] official_evaluator_invoked={completion_payload['official_evaluator_invoked']}")
    print(f"[stage2-external-eval-completion] official_tapvid_evaluator_connected={completion_payload['official_tapvid_evaluator_connected']}")
    print(f"[stage2-external-eval-completion] official_task_faithfully_instantiated={completion_payload['official_task_faithfully_instantiated']}")
    print(f"[stage2-external-eval-completion] tap3d_style_eval_status={completion_payload['tap3d_style_eval_status']}")
    print(f"[stage2-external-eval-completion] external_eval_readiness={completion_payload['external_eval_readiness']}")
    print(f"[stage2-external-eval-completion] next_step_choice={completion_payload['next_step_choice']}")


if __name__ == "__main__":
    main()
