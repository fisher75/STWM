#!/usr/bin/env python3
from __future__ import annotations

from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List
import gc
import itertools
import json
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from stwm.infra.gpu_lease import acquire_lease, release_lease
from stwm.infra.gpu_selector import select_single_gpu
from stwm.tracewm_v2_stage2.datasets.stage2_semantic_dataset import Stage2SemanticDataset, Stage2SemanticDatasetConfig, stage2_semantic_collate_fn
from stwm.tracewm_v2_stage2.models.semantic_encoder import SemanticEncoder, SemanticEncoderConfig
from stwm.tracewm_v2_stage2.models.semantic_fusion import SemanticFusion, SemanticFusionConfig
from stwm.tracewm_v2_stage2.trainers import train_tracewm_stage2_smalltrain as trainer


def _repo_root() -> Path:
    for candidate in [Path("/raid/chen034/workspace/stwm"), Path("/home/chen034/workspace/stwm")]:
        if candidate.exists():
            return candidate
    raise RuntimeError("unable to resolve STWM repo root")


ROOT = _repo_root()


def now_iso() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


def read_json(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {}
    payload = json.loads(p.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def write_json(path: str | Path, payload: Dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


@dataclass
class RuntimeChoice:
    num_workers: int
    pin_memory: bool
    persistent_workers: bool
    prefetch_factor: int


def _select_device(args: Any) -> tuple[torch.device, Dict[str, Any]]:
    if not torch.cuda.is_available():
        return torch.device("cpu"), {"mode": "cpu_only", "selected_gpu_id": -1, "lease_id": ""}
    selector = select_single_gpu(
        required_mem_gb=float(args.required_mem_gb),
        safety_margin_gb=float(args.safety_margin_gb),
        sample_count=3,
        interval_sec=0.5,
        lease_path=str(args.lease_path),
    )
    gpu_id = int(selector.get("selected_gpu_id", -1))
    if gpu_id < 0:
        return torch.device("cpu"), {"mode": "auto_cpu_no_fit", "selected_gpu_id": -1, "lease_id": "", "selector_payload": selector}
    lease = acquire_lease(
        gpu_id=gpu_id,
        owner="stage2_runtime_benchmark_20260416",
        ttl_seconds=3 * 3600,
        lease_path=str(args.lease_path),
        allow_shared=True,
    )
    return torch.device(f"cuda:{gpu_id}"), {
        "mode": "cuda_shared",
        "selected_gpu_id": int(gpu_id),
        "lease_id": str(lease.get("lease_id", "")),
        "selector_payload": selector,
    }


def _load_best_checkpoint_path(args: Any) -> Path:
    diag = read_json(args.final_pack_diagnosis)
    run_name = str(diag.get("current_best_overall_run_name", "stage2_calonly_topk1_seed123_wave1_20260413"))
    ckpt = ROOT / "outputs/checkpoints" / run_name / "best.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"best checkpoint missing: {ckpt}")
    return ckpt


def _load_modules(ckpt_path: Path, device: torch.device) -> Dict[str, Any]:
    try:
        payload = torch.load(ckpt_path, map_location=device, weights_only=False)
    except TypeError:
        payload = torch.load(ckpt_path, map_location=device)
    ckpt_args = payload.get("args", {}) if isinstance(payload.get("args", {}), dict) else {}
    stage1_path = Path(str(ckpt_args.get("stage1_backbone_checkpoint", ROOT / "outputs/checkpoints/stage1_v2_longtrain_220m_mainline_20260408/best.pt")))
    stage1_args = type("A", (), {
        "stage1_backbone_checkpoint": str(stage1_path),
        "stage1_model_preset": str(ckpt_args.get("stage1_model_preset", "prototype_220m")),
        "stage1_partial_unfreeze_mode": "none",
        "stage1_partial_unfreeze_layer_count": 1,
    })()
    stage1_model, _ = trainer._load_frozen_stage1_backbone(stage1_args, device=device)
    hidden_dim = int(stage1_model.config.d_model)
    semantic_encoder = SemanticEncoder(
        SemanticEncoderConfig(
            input_dim=10,
            hidden_dim=int(ckpt_args.get("semantic_hidden_dim", 256)),
            output_dim=int(ckpt_args.get("semantic_embed_dim", 256)),
            dropout=0.1,
            mainline_source=str(ckpt_args.get("semantic_source_mainline", "crop_visual_encoder")),
            legacy_source=str(ckpt_args.get("legacy_semantic_source", "hand_crafted_stats")),
            local_temporal_window=int(ckpt_args.get("local_temporal_window", 1)),
            local_temporal_fuse_weight=float(ckpt_args.get("local_temporal_fuse_weight", 0.5)),
        )
    ).to(device)
    semantic_fusion = SemanticFusion(
        trainer.SemanticFusionConfig(hidden_dim=hidden_dim, semantic_dim=int(ckpt_args.get("semantic_embed_dim", 256)), dropout=0.1)
    ).to(device)
    readout_head = torch.nn.Linear(hidden_dim, 2).to(device)
    semantic_encoder.load_state_dict(payload.get("semantic_encoder_state_dict", {}), strict=False)
    semantic_fusion.load_state_dict(payload.get("semantic_fusion_state_dict", {}), strict=False)
    readout_head.load_state_dict(payload.get("readout_head_state_dict", {}), strict=False)
    stage1_model.eval()
    semantic_encoder.eval()
    semantic_fusion.eval()
    readout_head.eval()
    return {
        "stage1_model": stage1_model,
        "semantic_encoder": semantic_encoder,
        "semantic_fusion": semantic_fusion,
        "readout_head": readout_head,
        "semantic_source_mainline": str(ckpt_args.get("semantic_source_mainline", "crop_visual_encoder")),
    }


def _benchmark_choice(
    *,
    dataset: Stage2SemanticDataset,
    choice: RuntimeChoice,
    modules: Dict[str, Any],
    device: torch.device,
    batch_size: int,
    measured_batches: int,
) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {
        "dataset": dataset,
        "batch_size": int(batch_size),
        "shuffle": True,
        "num_workers": int(choice.num_workers),
        "pin_memory": bool(choice.pin_memory),
        "collate_fn": stage2_semantic_collate_fn,
    }
    if int(choice.num_workers) > 0:
        kwargs["persistent_workers"] = bool(choice.persistent_workers)
        kwargs["prefetch_factor"] = int(choice.prefetch_factor)
    loader = DataLoader(**kwargs)
    it = iter(loader)
    samples = 0
    fetch_sec = 0.0
    compute_sec = 0.0
    start_cpu = time.process_time()
    start_wall = time.perf_counter()
    with torch.no_grad():
        for _ in range(int(measured_batches)):
            t0 = time.perf_counter()
            try:
                raw_batch = next(it)
            except StopIteration:
                it = iter(loader)
                raw_batch = next(it)
            t1 = time.perf_counter()
            batch = trainer._to_device(raw_batch, device=device, non_blocking=bool(choice.pin_memory and device.type == "cuda"))
            _ = trainer._teacher_forced_predict(
                stage1_model=modules["stage1_model"],
                semantic_encoder=modules["semantic_encoder"],
                semantic_fusion=modules["semantic_fusion"],
                readout_head=modules["readout_head"],
                batch=batch,
                obs_len=8,
                semantic_source_mainline=modules["semantic_source_mainline"],
                allow_stage1_grad=False,
            )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            t2 = time.perf_counter()
            fetch_sec += t1 - t0
            compute_sec += t2 - t1
            samples += int(raw_batch["batch_size"])
    wall = max(time.perf_counter() - start_wall, 1e-6)
    cpu = max(time.process_time() - start_cpu, 0.0)
    try:
        del loader, it, raw_batch, batch
    except Exception:
        pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        "num_workers": int(choice.num_workers),
        "pin_memory": bool(choice.pin_memory),
        "persistent_workers": bool(choice.persistent_workers),
        "prefetch_factor": int(choice.prefetch_factor),
        "samples": int(samples),
        "wall_sec": float(wall),
        "fetch_sec": float(fetch_sec),
        "compute_sec": float(compute_sec),
        "samples_per_sec": float(samples / wall),
        "gpu_util_proxy": float(compute_sec / wall),
        "data_wait_ratio": float(fetch_sec / wall),
        "host_cpu_load": float(cpu / wall),
    }


def parse_args() -> Any:
    p = ArgumentParser(description="Benchmark Stage2 runtime/data pipeline configs")
    p.add_argument("--contract-json", default=str(ROOT / "reports/stage2_bootstrap_data_contract_20260408.json"))
    p.add_argument("--final-pack-diagnosis", default=str(ROOT / "reports/stage2_calibration_only_final_pack_diagnosis_20260414.json"))
    p.add_argument("--output-json", default=str(ROOT / "reports/stage2_runtime_pipeline_benchmark_20260416.json"))
    p.add_argument("--recommended-runtime-json", default=str(ROOT / "configs/recommended_stage2_runtime_20260416.json"))
    p.add_argument("--dataset-names", nargs="*", default=["vspw", "vipseg"])
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-samples-per-dataset", type=int, default=128)
    p.add_argument("--measured-batches", type=int, default=16)
    p.add_argument("--predecode-cache-path", default="")
    p.add_argument("--lease-path", default=str(ROOT / "reports/stage1_v2_gpu_lease_20260408.json"))
    p.add_argument("--required-mem-gb", type=float, default=24.0)
    p.add_argument("--safety-margin-gb", type=float, default=4.0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device, device_info = _select_device(args)
    lease_id = str(device_info.get("lease_id", ""))
    ckpt = _load_best_checkpoint_path(args)
    dataset = Stage2SemanticDataset(
        Stage2SemanticDatasetConfig(
            dataset_names=[str(x) for x in args.dataset_names],
            split="train",
            contract_path=str(args.contract_json),
            obs_len=8,
            fut_len=8,
            max_tokens=64,
            max_samples_per_dataset=int(args.max_samples_per_dataset),
            semantic_patch_radius=12,
            semantic_crop_size=64,
            semantic_source_mainline="crop_visual_encoder",
            semantic_temporal_window=1,
            predecode_cache_path=str(args.predecode_cache_path),
        )
    )
    modules = _load_modules(ckpt, device=device)
    choices: List[RuntimeChoice] = []
    for num_workers, pin_memory, persistent_workers, prefetch_factor in itertools.product([0, 2, 4, 8, 12], [False, True], [False, True], [2, 4]):
        if num_workers == 0 and persistent_workers:
            continue
        choices.append(RuntimeChoice(num_workers, pin_memory, persistent_workers, prefetch_factor))
    rows = []
    for choice in choices:
        rows.append(
            _benchmark_choice(
                dataset=dataset,
                choice=choice,
                modules=modules,
                device=device,
                batch_size=int(args.batch_size),
                measured_batches=int(args.measured_batches),
            )
        )
    rows = sorted(rows, key=lambda row: (-float(row["samples_per_sec"]), -float(row["gpu_util_proxy"]), float(row["data_wait_ratio"])))
    best = rows[0] if rows else {}
    baseline = next((r for r in rows if int(r["num_workers"]) == 0 and not bool(r["pin_memory"]) and not bool(r["persistent_workers"]) and int(r["prefetch_factor"]) == 2), {})
    improvement = float(best.get("samples_per_sec", 0.0)) / max(float(baseline.get("samples_per_sec", 1e-6)), 1e-6) if best and baseline else 1.0
    runtime_bottleneck_relieved = bool(best and baseline and improvement > 1.15 and float(best.get("data_wait_ratio", 1.0)) < float(baseline.get("data_wait_ratio", 1.0)))
    report = {
        "generated_at_utc": now_iso(),
        "selected_device": str(device),
        "device_info": device_info,
        "dataset_names": [str(x) for x in args.dataset_names],
        "batch_size": int(args.batch_size),
        "max_samples_per_dataset": int(args.max_samples_per_dataset),
        "measured_batches": int(args.measured_batches),
        "default_runtime": {
            "num_workers": 0,
            "pin_memory": False,
            "persistent_workers": False,
            "prefetch_factor": 2,
        },
        "rows": rows,
        "best_runtime_config": best,
        "baseline_runtime_config": baseline,
        "samples_per_sec_improvement_vs_default": float(improvement),
        "runtime_bottleneck_relieved": bool(runtime_bottleneck_relieved),
        "gpu_util_proxy_definition": "compute_time / total_step_time over measured batches",
        "host_cpu_load_definition": "process_cpu_time / wall_time over measured batches",
    }
    write_json(args.output_json, report)
    recommended = {
        "generated_at_utc": now_iso(),
        "mode": "stage2_runtime_pipeline_optimized",
        "selected_gpu_policy": {
            "mode": "shared_gpu_selector",
            "selected_gpu_id": int(device_info.get("selected_gpu_id", -1)),
        },
        "recommended_num_workers": int(best.get("num_workers", 8) or 8),
        "recommended_pin_memory": bool(best.get("pin_memory", True)),
        "recommended_persistent_workers": bool(best.get("persistent_workers", True)),
        "recommended_prefetch_factor": int(best.get("prefetch_factor", 4) or 4),
        "required_mem_gb": float(args.required_mem_gb),
        "safety_margin_gb": float(args.safety_margin_gb),
        "single_gpu_only": True,
        "notes": [
            "20260416 runtime benchmark on calibration-only mainline checkpoint",
            "samples_per_sec used as primary runtime ranking metric",
            f"samples_per_sec_improvement_vs_default={report['samples_per_sec_improvement_vs_default']:.3f}",
        ],
    }
    write_json(args.recommended_runtime_json, recommended)
    print(json.dumps(report, ensure_ascii=True, indent=2))
    if lease_id:
        try:
            release_lease(lease_id=lease_id, lease_path=str(args.lease_path))
        except Exception:
            pass


if __name__ == "__main__":
    main()
