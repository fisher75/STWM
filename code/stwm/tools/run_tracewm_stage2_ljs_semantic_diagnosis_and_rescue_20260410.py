#!/usr/bin/env python3
from __future__ import annotations

from argparse import ArgumentParser
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Tuple
import json
import os
import subprocess
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from stwm.infra.gpu_lease import acquire_lease, release_lease
from stwm.infra.gpu_selector import select_single_gpu
from stwm.tracewm_v2_stage2.datasets.stage2_semantic_dataset import (
    Stage2SemanticDataset,
    Stage2SemanticDatasetConfig,
    stage2_semantic_collate_fn,
)
from stwm.tracewm_v2_stage2.models.semantic_encoder import SemanticEncoder, SemanticEncoderConfig
from stwm.tracewm_v2_stage2.models.semantic_fusion import SemanticFusion, SemanticFusionConfig
from stwm.tracewm_v2_stage2.trainers import train_tracewm_stage2_smalltrain as stage2_trainer


WORK_ROOT = Path("/home/chen034/workspace/stwm")
SESSION = "tracewm_stage2_ljs_aligned_semantic_diagnosis_and_rescue_20260410"
METRIC_KEYS = ["free_rollout_endpoint_l2", "free_rollout_coord_mean_l2", "teacher_forced_coord_loss"]
HARD_SUBSET_LIMIT = 24
HARD_SUBSET_CORE_SCAN_WINDOW = 60
HARD_SUBSET_BURST_SCAN_WINDOW = 40
CACHE_LIMIT_PER_DATASET_SPLIT = 64


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def _write_jsonl(path: str | Path, rows: Iterable[Dict[str, Any]]) -> int:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with p.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")
            count += 1
    return count


def _write_md(path: str | Path, lines: List[str]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _f(x: Any, default: float = 1e9) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def _mean_std(values: List[float]) -> Dict[str, Any]:
    vals = [float(x) for x in values if np.isfinite(float(x))]
    if not vals:
        return {"mean": None, "std": None, "count": 0}
    return {
        "mean": float(np.mean(vals)),
        "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
        "count": int(len(vals)),
    }


def _metrics_from_best_json(path: str | Path) -> Dict[str, float]:
    payload = _read_json(path)
    best = payload.get("best_checkpoint_metric", {}) if isinstance(payload.get("best_checkpoint_metric", {}), dict) else {}
    metrics = best.get("metrics", {}) if isinstance(best.get("metrics", {}), dict) else payload
    return {key: _f(metrics.get(key), 1e9) for key in METRIC_KEYS}


def _run_record(run_name: str, family: str, seed: int) -> Dict[str, Any]:
    return {
        "run_name": run_name,
        "family": family,
        "seed": int(seed),
        "final_json": str(WORK_ROOT / "reports" / f"{run_name}_final.json"),
        "best_checkpoint": str(WORK_ROOT / "outputs" / "checkpoints" / run_name / "best.pt"),
    }


def _stage2_runs() -> List[Dict[str, Any]]:
    return [
        _run_record("stage2_fullscale_core_cropenc_seed42_20260409", "cropenc", 42),
        _run_record("stage2_fullscale_core_cropenc_seed123_20260409", "cropenc", 123),
        _run_record("stage2_fullscale_core_cropenc_seed456_20260409", "cropenc", 456),
        _run_record("stage2_fullscale_core_cropenc_seed789_wave2_20260409", "cropenc", 789),
        _run_record("stage2_fullscale_core_legacysem_seed42_20260409", "legacysem", 42),
        _run_record("stage2_fullscale_core_legacysem_seed123_wave2_20260409", "legacysem", 123),
        _run_record("stage2_fullscale_core_legacysem_seed456_wave2_20260409", "legacysem", 456),
        _run_record("stage2_fullscale_coreplusburst_cropenc_seed42_20260409", "coreplusburst", 42),
        _run_record("stage2_fullscale_coreplusburst_cropenc_seed123_wave2_20260409", "coreplusburst", 123),
        _run_record("stage2_fullscale_coreplusburst_cropenc_seed456_wave2_20260409", "coreplusburst", 456),
    ]


def _make_dataset(dataset_names: List[str], split: str, contract_path: str, max_samples: int = -1, semantic_source: str = "crop_visual_encoder") -> Stage2SemanticDataset:
    return Stage2SemanticDataset(
        Stage2SemanticDatasetConfig(
            dataset_names=list(dataset_names),
            split=str(split),
            contract_path=str(contract_path),
            obs_len=8,
            fut_len=8,
            max_tokens=64,
            max_samples_per_dataset=int(max_samples),
            semantic_crop_size=64,
            semantic_source_mainline=str(semantic_source),
        )
    )


def _loader(dataset: Any, batch_size: int = 8) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        collate_fn=stage2_semantic_collate_fn,
    )


def _load_stage1_model(device: torch.device, stage1_ckpt: str) -> Any:
    args = SimpleNamespace(stage1_backbone_checkpoint=str(stage1_ckpt), stage1_model_preset="prototype_220m")
    model, meta = stage2_trainer._load_frozen_stage1_backbone(args=args, device=device)
    return model, meta


def _prepare_shifted(full_state: torch.Tensor) -> torch.Tensor:
    shifted = torch.zeros_like(full_state)
    shifted[:, 0] = full_state[:, 0]
    shifted[:, 1:] = full_state[:, :-1]
    return shifted


def _evaluate_stage1_baseline(stage1_model: Any, loader: DataLoader, device: torch.device) -> Dict[str, Any]:
    obs_len = 8
    fut_len = 8
    tf_sse = tf_count = free_l2_sum = free_l2_count = endpoint_sum = endpoint_count = 0.0
    batch_count = 0
    stage1_model.eval()
    with torch.no_grad():
        for raw_batch in loader:
            batch = stage2_trainer._to_device(raw_batch, device=device, non_blocking=False)
            token_mask = batch["token_mask"]
            full_state = torch.cat([batch["obs_state"], batch["fut_state"]], dim=1)
            shifted = _prepare_shifted(full_state)
            tf_out = stage1_model(shifted, token_mask=token_mask)
            tf_pred = tf_out["pred_state"][:, obs_len : obs_len + fut_len, :, 0:2]
            target = batch["fut_state"][..., 0:2]
            valid = batch["fut_valid"] & token_mask[:, None, :]
            tf_sq = ((tf_pred - target) ** 2).sum(dim=-1)
            tf_sse += float((tf_sq * valid.float()).sum().item())
            tf_count += float(valid.float().sum().item())

            obs_state = batch["obs_state"]
            bsz, _, k_len, d_state = obs_state.shape
            state_seq = torch.zeros((bsz, obs_len + fut_len, k_len, d_state), device=device, dtype=obs_state.dtype)
            state_seq[:, :obs_len] = obs_state
            for step in range(fut_len):
                shifted_roll = _prepare_shifted(state_seq)
                out = stage1_model(shifted_roll, token_mask=token_mask)
                idx = obs_len + step
                state_seq[:, idx : idx + 1] = out["pred_state"][:, idx : idx + 1].detach()
            free_pred = state_seq[:, obs_len:, :, 0:2]
            free_l2 = torch.sqrt(((free_pred - target) ** 2).sum(dim=-1).clamp_min(1e-12))
            free_l2_sum += float((free_l2 * valid.float()).sum().item())
            free_l2_count += float(valid.float().sum().item())
            endpoint = torch.sqrt(((free_pred[:, -1] - target[:, -1]) ** 2).sum(dim=-1).clamp_min(1e-12))
            endpoint_valid = valid[:, -1].float()
            endpoint_sum += float((endpoint * endpoint_valid).sum().item())
            endpoint_count += float(endpoint_valid.sum().item())
            batch_count += 1
    return {
        "teacher_forced_coord_loss": float(tf_sse / max(tf_count, 1.0)),
        "free_rollout_coord_mean_l2": float(free_l2_sum / max(free_l2_count, 1.0)),
        "free_rollout_endpoint_l2": float(endpoint_sum / max(endpoint_count, 1.0)),
        "eval_batches": int(batch_count),
    }


def _load_stage2_modules(checkpoint_path: str, device: torch.device, stage1_model: Any) -> Tuple[Any, Any, Any, str]:
    ckpt = torch.load(Path(checkpoint_path), map_location=device, weights_only=False)
    args = ckpt.get("args", {}) if isinstance(ckpt.get("args", {}), dict) else {}
    sem_dim = int(args.get("semantic_embed_dim", 256) or 256)
    hidden_dim = int(args.get("semantic_hidden_dim", 256) or 256)
    source = str(args.get("semantic_source_mainline", "crop_visual_encoder"))
    semantic_encoder = SemanticEncoder(
        SemanticEncoderConfig(
            input_dim=10,
            hidden_dim=hidden_dim,
            output_dim=sem_dim,
            dropout=0.1,
            mainline_source=source,
            legacy_source=str(args.get("legacy_semantic_source", "hand_crafted_stats")),
        )
    ).to(device)
    semantic_fusion = SemanticFusion(
        SemanticFusionConfig(
            hidden_dim=int(stage1_model.config.d_model),
            semantic_dim=sem_dim,
            dropout=0.1,
        )
    ).to(device)
    readout_head = torch.nn.Linear(int(stage1_model.config.d_model), 2).to(device)
    semantic_encoder.load_state_dict(ckpt["semantic_encoder_state_dict"])
    semantic_fusion.load_state_dict(ckpt["semantic_fusion_state_dict"])
    readout_head.load_state_dict(ckpt["readout_head_state_dict"])
    return semantic_encoder, semantic_fusion, readout_head, source


def _evaluate_loaded_stage2(
    *,
    loaded: Tuple[Any, Any, Any, str],
    stage1_model: Any,
    dataset: Any,
    device: torch.device,
) -> Dict[str, Any]:
    semantic_encoder, semantic_fusion, readout_head, source = loaded
    return stage2_trainer._evaluate(
        stage1_model=stage1_model,
        semantic_encoder=semantic_encoder,
        semantic_fusion=semantic_fusion,
        readout_head=readout_head,
        loader=_loader(dataset, batch_size=8),
        device=device,
        pin_memory=False,
        obs_len=8,
        fut_len=8,
        max_batches=-1,
        semantic_source_mainline=source,
    )


def _score_core_sample(sample: Dict[str, Any]) -> Dict[str, float]:
    state = torch.cat([sample["obs_state"], sample["fut_state"]], dim=0)[:, 0].numpy()
    coords = state[:, 0:2]
    area = np.clip(state[:, 6] * state[:, 7], 0.0, 1.0)
    diffs = np.sqrt(((coords[1:] - coords[:-1]) ** 2).sum(axis=-1))
    sem = sample["semantic_features"][0].numpy()
    center_dist = np.sqrt(((coords - 0.5) ** 2).sum(axis=-1))
    return {
        "small_area": float(np.mean(area)),
        "motion": float(np.sum(diffs)),
        "area_range": float(np.max(area) - np.min(area)),
        "color_std_mean": float(np.mean(sem[3:6])),
        "center_interaction": float(np.sum(diffs) + max(0.0, 0.75 - float(np.mean(center_dist)))),
        "frame_count": float(sample.get("meta", {}).get("frame_count_total", 0)),
    }


def _top_items(scored: List[Dict[str, Any]], score_key: str, reverse: bool, limit: int = HARD_SUBSET_LIMIT) -> List[Dict[str, Any]]:
    return sorted(scored, key=lambda x: float(x.get(score_key, 0.0)), reverse=reverse)[: int(limit)]


def build_hard_subsets(contract_path: str) -> Dict[str, Any]:
    print("[ljs-diagnosis] building hard subsets", flush=True)
    core_ds = _make_dataset(["vspw", "vipseg"], "val", contract_path)
    burst_ds = _make_dataset(["burst"], "val", contract_path)
    scored_core: List[Dict[str, Any]] = []
    for idx in range(min(len(core_ds), HARD_SUBSET_CORE_SCAN_WINDOW)):
        if idx > 0 and idx % 50 == 0:
            print(f"[ljs-diagnosis] hard subset core scan {idx}/{min(len(core_ds), HARD_SUBSET_CORE_SCAN_WINDOW)}", flush=True)
        sample = core_ds[idx]
        meta = dict(sample.get("meta", {}))
        scores = _score_core_sample(sample)
        scored_core.append({"dataset_index": idx, "dataset": meta.get("dataset", ""), "clip_id": meta.get("clip_id", ""), **scores})
    scored_burst: List[Dict[str, Any]] = []
    for idx in range(min(len(burst_ds), HARD_SUBSET_BURST_SCAN_WINDOW)):
        if idx > 0 and idx % 50 == 0:
            print(f"[ljs-diagnosis] hard subset burst scan {idx}/{min(len(burst_ds), HARD_SUBSET_BURST_SCAN_WINDOW)}", flush=True)
        sample = burst_ds[idx]
        meta = dict(sample.get("meta", {}))
        scores = _score_core_sample(sample)
        scored_burst.append({"dataset_index": idx, "dataset": meta.get("dataset", ""), "clip_id": meta.get("clip_id", ""), **scores})

    specs = {
        "occlusion_reappearance": {
            "dataset_scope": "core_val",
            "build_rule": "top area_range clips, using sampled mask-derived box area range as occlusion/reappearance proxy",
            "exact_source_fields_used": ["obs_state[...,6:8]", "fut_state[...,6:8]", "mask_paths via dataset box extraction"],
            "items": _top_items(scored_core, "area_range", True),
        },
        "crossing_or_interaction_ambiguity": {
            "dataset_scope": "core_val",
            "build_rule": "top center_interaction score, using high trajectory motion near image center as ambiguity proxy",
            "exact_source_fields_used": ["obs_state[...,0:2]", "fut_state[...,0:2]"],
            "items": _top_items(scored_core, "center_interaction", True),
        },
        "small_object_or_low_area": {
            "dataset_scope": "core_val",
            "build_rule": "lowest average normalized mask/box area",
            "exact_source_fields_used": ["obs_state[...,6:8]", "fut_state[...,6:8]"],
            "items": _top_items(scored_core, "small_area", False),
        },
        "appearance_change_or_semantic_shift": {
            "dataset_scope": "core_val",
            "build_rule": "top color_std_mean + area_range proxy over object/mask crop statistics",
            "exact_source_fields_used": ["semantic_features[3:6]", "obs_state[...,6:8]", "fut_state[...,6:8]"],
            "items": _top_items([{**x, "appearance_shift": float(x["color_std_mean"] + x["area_range"])} for x in scored_core], "appearance_shift", True),
        },
        "burst_persistence_stress": {
            "dataset_scope": "burst_val_optional_extension",
            "build_rule": "top frame_count BURST validation clips as optional persistence/long-gap stress panel",
            "exact_source_fields_used": ["meta.frame_count_total", "BURST val frame paths"],
            "items": _top_items(scored_burst, "frame_count", True),
        },
    }
    payload = {
        "generated_at_utc": now_iso(),
        "manifest_type": "stage2_semantic_hard_subsets_20260410",
        "core_dataset_binding": ["VSPW", "VIPSeg"],
        "burst_usage": "optional_persistence_stress_only_not_main_training",
        "scan_window": {
            "core_val_first_n": int(min(len(core_ds), HARD_SUBSET_CORE_SCAN_WINDOW)),
            "burst_val_first_n": int(min(len(burst_ds), HARD_SUBSET_BURST_SCAN_WINDOW)),
            "selection_limit_per_subset": int(HARD_SUBSET_LIMIT),
            "reason": "bounded deterministic audit window to keep this diagnosis round tractable while preserving reproducibility",
        },
        "subsets": {},
    }
    for name, spec in specs.items():
        counts: Dict[str, int] = {}
        for item in spec["items"]:
            counts[str(item.get("dataset", ""))] = counts.get(str(item.get("dataset", "")), 0) + 1
        payload["subsets"][name] = {
            **spec,
            "clip_count": int(len(spec["items"])),
            "per_dataset_count": counts,
        }
    return payload


def _aggregate_family(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key in METRIC_KEYS:
        out[key] = _mean_std([_f(r.get("metrics", {}).get(key), 1e9) for r in rows])
    return out


def _rank_families(family_metrics: Dict[str, Dict[str, Any]]) -> List[str]:
    return sorted(family_metrics, key=lambda k: float(family_metrics[k]["free_rollout_endpoint_l2"]["mean"]))


def _evaluate_stage2_family_on_subset(
    *,
    family: str,
    runs: List[Dict[str, Any]],
    dataset: Any,
    device: torch.device,
    stage1_model: Any,
    loaded_modules: Dict[str, Tuple[Any, Any, Any, str]],
) -> Dict[str, Any]:
    rows = []
    for run in runs:
        if run["family"] != family:
            continue
        metrics = _evaluate_loaded_stage2(
            loaded=loaded_modules[run["run_name"]],
            stage1_model=stage1_model,
            dataset=dataset,
            device=device,
        )
        rows.append({"run_name": run["run_name"], "seed": run["seed"], "metrics": {k: _f(metrics.get(k), 1e9) for k in METRIC_KEYS}})
    return {"runs": rows, "aggregate": _aggregate_family(rows)}


def _full_validation_panel(stage1_metrics: Dict[str, Any]) -> Dict[str, Any]:
    runs = _stage2_runs()
    rows = []
    for run in runs:
        metrics = _metrics_from_best_json(run["final_json"])
        rows.append({**run, "metrics": metrics})
    family = {}
    for fam in ["cropenc", "legacysem", "coreplusburst"]:
        family[fam] = _aggregate_family([r for r in rows if r["family"] == fam])
    return {
        "stage1_frozen_baseline": {k: _f(stage1_metrics.get(k), 1e9) for k in METRIC_KEYS},
        "stage2_runs": rows,
        "family_aggregates": family,
        "relative_ranking_by_endpoint_l2": _rank_families(family),
        "source": "stage2 fullscale Wave1/Wave2 final-json best_checkpoint_metric plus freshly evaluated Stage1 baseline",
    }


def _win_loss(crop: float, other: float) -> str:
    if crop < other:
        return "win"
    if crop > other:
        return "loss"
    return "tie"


def build_diagnosis(
    *,
    stage1_metrics: Dict[str, Any],
    subset_manifest: Dict[str, Any],
    device: torch.device,
    contract_path: str,
    stage1_ckpt: str,
) -> Dict[str, Any]:
    print("[ljs-diagnosis] building full and hard-subset diagnosis", flush=True)
    runs = _stage2_runs()
    full_panel = _full_validation_panel(stage1_metrics)
    core_ds = _make_dataset(["vspw", "vipseg"], "val", contract_path)
    burst_ds = _make_dataset(["burst"], "val", contract_path)
    subset_panels: Dict[str, Any] = {}
    stage1_model, _ = _load_stage1_model(device=device, stage1_ckpt=stage1_ckpt)
    loaded_modules = {
        run["run_name"]: _load_stage2_modules(run["best_checkpoint"], device, stage1_model)
        for run in runs
    }

    for subset_name, subset in subset_manifest.get("subsets", {}).items():
        print(f"[ljs-diagnosis] evaluating subset={subset_name}", flush=True)
        items = subset.get("items", []) if isinstance(subset.get("items", []), list) else []
        indices = [int(x["dataset_index"]) for x in items if isinstance(x, dict) and "dataset_index" in x]
        base_ds = burst_ds if subset.get("dataset_scope") == "burst_val_optional_extension" else core_ds
        eval_ds = Subset(base_ds, indices)
        stage1_subset_metrics = _evaluate_stage1_baseline(stage1_model, _loader(eval_ds, batch_size=8), device=device)
        family_panels = {
            fam: _evaluate_stage2_family_on_subset(
                family=fam,
                runs=runs,
                dataset=eval_ds,
                device=device,
                stage1_model=stage1_model,
                loaded_modules=loaded_modules,
            )
            for fam in ["cropenc", "legacysem", "coreplusburst"]
        }
        subset_panels[subset_name] = {
            "clip_count": int(len(indices)),
            "dataset_scope": subset.get("dataset_scope", ""),
            "stage1_frozen_baseline": {k: _f(stage1_subset_metrics.get(k), 1e9) for k in METRIC_KEYS},
            "families": family_panels,
            "relative_ranking_by_endpoint_l2": _rank_families({k: v["aggregate"] for k, v in family_panels.items()}),
        }

    crop_full = full_panel["family_aggregates"]["cropenc"]["free_rollout_endpoint_l2"]["mean"]
    legacy_full = full_panel["family_aggregates"]["legacysem"]["free_rollout_endpoint_l2"]["mean"]
    burst_full = full_panel["family_aggregates"]["coreplusburst"]["free_rollout_endpoint_l2"]["mean"]
    crop_tf = full_panel["family_aggregates"]["cropenc"]["teacher_forced_coord_loss"]["mean"]
    burst_tf = full_panel["family_aggregates"]["coreplusburst"]["teacher_forced_coord_loss"]["mean"]

    semantic_subset_names = [
        "occlusion_reappearance",
        "crossing_or_interaction_ambiguity",
        "small_object_or_low_area",
        "appearance_change_or_semantic_shift",
    ]
    hard_outcomes = []
    for name in semantic_subset_names:
        panel = subset_panels.get(name, {})
        families = panel.get("families", {}) if isinstance(panel.get("families", {}), dict) else {}
        crop_ep = families.get("cropenc", {}).get("aggregate", {}).get("free_rollout_endpoint_l2", {}).get("mean")
        leg_ep = families.get("legacysem", {}).get("aggregate", {}).get("free_rollout_endpoint_l2", {}).get("mean")
        if crop_ep is not None and leg_ep is not None:
            hard_outcomes.append(_win_loss(float(crop_ep), float(leg_ep)))
    hard_wins = sum(1 for x in hard_outcomes if x == "win")

    stage1_endpoint = _f(stage1_metrics.get("free_rollout_endpoint_l2"), 1e9)
    crop_overall_endpoint = float(crop_full)
    stage2_overall_better_than_stage1 = bool(crop_overall_endpoint < stage1_endpoint)
    cropenc_overall_better_than_legacysem = bool(float(crop_full) < float(legacy_full))
    cropenc_hard_better_than_legacysem = bool(hard_wins >= 3)
    coreplusburst_only_improves_teacher_forced = bool(float(burst_tf) < float(crop_tf) and float(burst_full) >= float(crop_full))

    if cropenc_overall_better_than_legacysem and cropenc_hard_better_than_legacysem and stage2_overall_better_than_stage1:
        conclusion = "keep_cropenc_mainline"
    elif cropenc_hard_better_than_legacysem:
        conclusion = "keep_cropenc_only_for_semantic_hard_regimes"
    elif not cropenc_overall_better_than_legacysem:
        conclusion = "keep_legacysem_as_stronger_current_baseline"
    else:
        conclusion = "redesign_stage2_semantic_objective_before_more_training"
    if (not cropenc_overall_better_than_legacysem) and (not cropenc_hard_better_than_legacysem):
        conclusion = "redesign_stage2_semantic_objective_before_more_training"

    phase_b_triggered = bool(
        (not cropenc_overall_better_than_legacysem)
        or (not cropenc_hard_better_than_legacysem)
        or (not stage2_overall_better_than_stage1)
    )

    return {
        "generated_at_utc": now_iso(),
        "ljs_alignment": {
            "main_task": "future trace / future state generation",
            "full_video_reconstruction_is_main_task": False,
            "teacher_as_mainline_semantic_source": False,
            "stage1_frozen": True,
        },
        "full_validation_panel": full_panel,
        "hard_subset_panels": subset_panels,
        "win_loss": {
            "cropenc_vs_legacysem_semantic_hard_endpoint": hard_outcomes,
            "cropenc_semantic_hard_win_count": int(hard_wins),
            "cropenc_semantic_hard_loss_count": int(sum(1 for x in hard_outcomes if x == "loss")),
            "cropenc_vs_legacysem_full_endpoint": _win_loss(float(crop_full), float(legacy_full)),
            "cropenc_vs_stage1_full_endpoint": _win_loss(crop_overall_endpoint, stage1_endpoint),
        },
        "scientific_judgment": {
            "stage2_overall_better_than_stage1_frozen_baseline": bool(stage2_overall_better_than_stage1),
            "cropenc_overall_better_than_legacysem": bool(cropenc_overall_better_than_legacysem),
            "cropenc_semantic_hard_better_than_legacysem": bool(cropenc_hard_better_than_legacysem),
            "coreplusburst_only_improves_teacher_forced_fitting": bool(coreplusburst_only_improves_teacher_forced),
            "phase_a_conclusion": conclusion,
            "phase_b_triggered": bool(phase_b_triggered),
        },
    }


def _write_baseline_docs(payload: Dict[str, Any], json_path: str, md_path: str) -> None:
    _write_json(json_path, payload)
    lines = [
        "# Stage2 Stage1 Frozen Baseline Eval",
        "",
        f"- generated_at_utc: {payload.get('generated_at_utc', '')}",
        f"- eval_binding: {payload.get('eval_binding', [])}",
        f"- free_rollout_endpoint_l2: {payload['metrics']['free_rollout_endpoint_l2']:.8f}",
        f"- free_rollout_coord_mean_l2: {payload['metrics']['free_rollout_coord_mean_l2']:.8f}",
        f"- teacher_forced_coord_loss: {payload['metrics']['teacher_forced_coord_loss']:.8f}",
    ]
    _write_md(md_path, lines)


def _write_subset_docs(payload: Dict[str, Any], manifest_path: str, report_path: str, md_path: str) -> None:
    _write_json(manifest_path, payload)
    _write_json(report_path, {k: v for k, v in payload.items() if k != "subsets"} | {"subsets": {k: {kk: vv for kk, vv in v.items() if kk != "items"} for k, v in payload.get("subsets", {}).items()}})
    lines = ["# Stage2 Semantic Hard Subsets", "", f"- generated_at_utc: {payload.get('generated_at_utc', '')}"]
    for name, subset in payload.get("subsets", {}).items():
        lines.extend([
            "",
            f"## {name}",
            f"- build_rule: {subset.get('build_rule', '')}",
            f"- clip_count: {subset.get('clip_count', 0)}",
            f"- per_dataset_count: {subset.get('per_dataset_count', {})}",
            f"- exact_source_fields_used: {subset.get('exact_source_fields_used', [])}",
        ])
    _write_md(md_path, lines)


def _write_diagnosis_docs(payload: Dict[str, Any], json_path: str, md_path: str) -> None:
    _write_json(json_path, payload)
    judge = payload["scientific_judgment"]
    full = payload["full_validation_panel"]["family_aggregates"]
    lines = [
        "# Stage2 Semantic Value Diagnosis",
        "",
        f"- generated_at_utc: {payload.get('generated_at_utc', '')}",
        f"- stage2_overall_better_than_stage1_frozen_baseline: {judge['stage2_overall_better_than_stage1_frozen_baseline']}",
        f"- cropenc_overall_better_than_legacysem: {judge['cropenc_overall_better_than_legacysem']}",
        f"- cropenc_semantic_hard_better_than_legacysem: {judge['cropenc_semantic_hard_better_than_legacysem']}",
        f"- coreplusburst_only_improves_teacher_forced_fitting: {judge['coreplusburst_only_improves_teacher_forced_fitting']}",
        f"- phase_a_conclusion: {judge['phase_a_conclusion']}",
        f"- phase_b_triggered: {judge['phase_b_triggered']}",
        "",
        "## Full Validation Endpoint L2",
    ]
    for fam in ["cropenc", "legacysem", "coreplusburst"]:
        metric = full[fam]["free_rollout_endpoint_l2"]
        lines.append(f"- {fam}: mean={metric['mean']:.8f}, std={metric['std']:.8f}, count={metric['count']}")
    _write_md(md_path, lines)


def _bootstrap_preflight() -> Dict[str, Any]:
    backends = []
    try:
        __import__("open_clip")
        backends.append({"name": "open_clip", "available": True})
    except Exception as exc:
        backends.append({"name": "open_clip", "available": False, "reason": str(exc)})
    try:
        __import__("transformers")
        local_hf = bool(list((WORK_ROOT / "models/hf_cache").glob("**/*siglip*")) or list((WORK_ROOT / "models/hf_cache").glob("**/*clip*")))
        backends.append({"name": "transformers_siglip_or_clip_local", "available": bool(local_hf), "reason": "" if local_hf else "transformers is installed but no local SigLIP/CLIP cache found"})
    except Exception as exc:
        backends.append({"name": "transformers_siglip_or_clip_local", "available": False, "reason": str(exc)})
    sam_exists = bool((WORK_ROOT / "models/checkpoints/sam2/sam2.1_hiera_large.pt").exists())
    backends.append({"name": "sam2_local_features_candidate", "available": sam_exists, "chosen": False, "reason": "available checkpoint but no verified Stage2 crop-feature interface in current trainer"})
    chosen = "crop_stats_pseudo_target_cache"
    return {
        "generated_at_utc": now_iso(),
        "backend_policy": "bootstrap/pseudo-label/cache only; never mainline semantic token source",
        "available_bootstrap_backends": backends,
        "chosen_bootstrap_backend": chosen,
        "feature_dim": 10,
        "target_format": "jsonl rows keyed by DATASET::clip_id with 10-d crop-stat pseudo target",
        "cache_feasibility_on_vspw_vipseg_train_val": "full core train/val is feasible but slow in this live round; using bounded required-subset cache first",
        "teacher_as_mainline_semantic_source": False,
    }


def _write_preflight_docs(payload: Dict[str, Any], json_path: str, md_path: str) -> None:
    _write_json(json_path, payload)
    lines = [
        "# Stage2 Semantic Bootstrap Preflight",
        "",
        f"- generated_at_utc: {payload.get('generated_at_utc', '')}",
        f"- chosen_bootstrap_backend: {payload.get('chosen_bootstrap_backend', '')}",
        f"- target_format: {payload.get('target_format', '')}",
        f"- teacher_as_mainline_semantic_source: {payload.get('teacher_as_mainline_semantic_source', False)}",
        f"- cache_feasibility_on_vspw_vipseg_train_val: {payload.get('cache_feasibility_on_vspw_vipseg_train_val', '')}",
    ]
    _write_md(md_path, lines)


def _cache_rows(contract_path: str) -> Iterable[Dict[str, Any]]:
    for split in ["train", "val"]:
        ds = _make_dataset(["vspw", "vipseg"], split, contract_path, max_samples=CACHE_LIMIT_PER_DATASET_SPLIT)
        for idx in range(len(ds)):
            sample = ds[idx]
            meta = sample.get("meta", {})
            yield {
                "dataset": str(meta.get("dataset", "")),
                "clip_id": str(meta.get("clip_id", "")),
                "split": split,
                "dataset_index": int(idx),
                "backend": "crop_stats_pseudo_target_cache",
                "feature_target": [float(x) for x in sample["semantic_features"][0].tolist()],
            }


def build_bootstrap_cache(contract_path: str) -> Dict[str, Any]:
    cache_path = WORK_ROOT / "data/processed/stage2_semantic_bootstrap_cache_20260410/core_trainval_pseudo_targets.jsonl"
    count = _write_jsonl(cache_path, _cache_rows(contract_path))
    counts: Dict[str, Dict[str, int]] = {"train": {}, "val": {}}
    with cache_path.open("r", encoding="utf-8") as f:
        for raw in f:
            item = json.loads(raw)
            split = str(item.get("split", ""))
            ds = str(item.get("dataset", ""))
            counts.setdefault(split, {})[ds] = counts.setdefault(split, {}).get(ds, 0) + 1
    return {
        "generated_at_utc": now_iso(),
        "cache_path": str(cache_path),
        "backend": "crop_stats_pseudo_target_cache",
        "coverage_mode": "required_subset_cache",
        "requested_limit_per_dataset_split": int(CACHE_LIMIT_PER_DATASET_SPLIT),
        "teacher_as_mainline_semantic_source": False,
        "total_covered_samples": int(count),
        "per_dataset_coverage": counts,
        "missing_or_failed_items_count": 0,
    }


def _write_cache_docs(payload: Dict[str, Any], json_path: str, md_path: str) -> None:
    _write_json(json_path, payload)
    lines = [
        "# Stage2 Semantic Bootstrap Cache Build",
        "",
        f"- generated_at_utc: {payload.get('generated_at_utc', '')}",
        f"- cache_path: {payload.get('cache_path', '')}",
        f"- backend: {payload.get('backend', '')}",
        f"- coverage_mode: {payload.get('coverage_mode', '')}",
        f"- requested_limit_per_dataset_split: {payload.get('requested_limit_per_dataset_split', 0)}",
        f"- total_covered_samples: {payload.get('total_covered_samples', 0)}",
        f"- per_dataset_coverage: {payload.get('per_dataset_coverage', {})}",
        f"- missing_or_failed_items_count: {payload.get('missing_or_failed_items_count', 0)}",
        f"- teacher_as_mainline_semantic_source: {payload.get('teacher_as_mainline_semantic_source', False)}",
    ]
    _write_md(md_path, lines)


def _select_eval_gpu(lease_path: str) -> Dict[str, Any]:
    selector = select_single_gpu(required_mem_gb=40.0, safety_margin_gb=8.0, sample_count=3, interval_sec=0.5, lease_path=lease_path)
    gpu_id = int(selector.get("selected_gpu_id", -1))
    if gpu_id < 0:
        return {"selected_gpu_id": -1, "lease_id": ""}
    lease = acquire_lease(gpu_id=gpu_id, owner="stage2_ljs_semantic_diagnosis_eval", ttl_seconds=8 * 3600, lease_path=lease_path)
    return {"selected_gpu_id": gpu_id, "lease_id": str(lease.get("lease_id", ""))}


def _release_eval_gpu(meta: Dict[str, Any], lease_path: str) -> None:
    lease_id = str(meta.get("lease_id", ""))
    if lease_id:
        try:
            release_lease(lease_id=lease_id, lease_path=lease_path)
        except Exception:
            pass


def _launch_and_wait_wave0(args: Any) -> Dict[str, Any]:
    launcher = WORK_ROOT / "code/stwm/tools/run_tracewm_stage2_semantic_rescue_wave0_20260410.py"
    subprocess.run([args.python_bin, str(launcher), "--mode", "launch", "--tmux-session", str(args.tmux_session)], check=True, cwd=str(WORK_ROOT))
    summary_path = WORK_ROOT / "reports/stage2_semantic_rescue_wave0_summary_20260410.json"
    deadline = time.time() + float(args.wave0_timeout_seconds)
    last = _read_json(summary_path)
    while time.time() < deadline:
        subprocess.run([args.python_bin, str(launcher), "--mode", "summarize", "--tmux-session", str(args.tmux_session)], check=True, cwd=str(WORK_ROOT))
        last = _read_json(summary_path)
        status = str(last.get("wave0_status", ""))
        if status.startswith("0_running_"):
            return last
        time.sleep(60)
    last["timed_out_waiting_for_completion"] = True
    _write_json(summary_path, last)
    return last


def _wave0_summary_is_terminal(summary: Dict[str, Any]) -> bool:
    rows = summary.get("runs", []) if isinstance(summary.get("runs", []), list) else []
    if not rows:
        return False
    return all(str(row.get("status", "")).lower() in {"completed", "failed"} for row in rows if isinstance(row, dict))


def _rescue_positive_signal(wave0_summary: Dict[str, Any], diagnosis: Dict[str, Any]) -> Dict[str, Any]:
    crop_endpoint = diagnosis["full_validation_panel"]["family_aggregates"]["cropenc"]["free_rollout_endpoint_l2"]["mean"]
    anchor_endpoint = 1e9
    completed = []
    for row in wave0_summary.get("runs", []) if isinstance(wave0_summary.get("runs", []), list) else []:
        if row.get("status") != "completed":
            continue
        best_metrics = row.get("best_checkpoint_metric", {}).get("metrics", {}) if isinstance(row.get("best_checkpoint_metric", {}), dict) else {}
        latest_metrics = row.get("latest_checkpoint_metric", {}).get("metrics", {}) if isinstance(row.get("latest_checkpoint_metric", {}), dict) else {}
        anchor_endpoint = min(anchor_endpoint, _f(best_metrics.get("free_rollout_endpoint_l2"), 1e9))
        completed.append(
            {
                "run_name": row.get("run_name", ""),
                "latest_endpoint": _f(latest_metrics.get("free_rollout_endpoint_l2"), 1e9),
                "best_endpoint": _f(best_metrics.get("free_rollout_endpoint_l2"), 1e9),
            }
        )
    best_latest = min([x["latest_endpoint"] for x in completed], default=1e9)
    return {
        "semantic_rescue_at_least_not_worse_than_current_cropenc_baseline": bool(best_latest <= float(crop_endpoint)),
        "semantic_rescue_latest_beats_warm_start_anchor": bool(best_latest < float(anchor_endpoint)),
        "best_wave0_latest_endpoint_l2": float(best_latest),
        "warm_start_anchor_best_endpoint_l2": float(anchor_endpoint),
        "current_cropenc_fullscale_mean_endpoint_l2": float(crop_endpoint),
        "completed_wave0_count": int(len(completed)),
        "completed_wave0_latest_rows": completed,
    }


def parse_args() -> Any:
    p = ArgumentParser(description="LJS-aligned Stage2 semantic diagnosis and rescue prep")
    p.add_argument("--work-root", default=str(WORK_ROOT))
    p.add_argument("--python-bin", default="/home/chen034/miniconda3/envs/stwm/bin/python")
    p.add_argument("--tmux-session", default=SESSION)
    p.add_argument("--contract-path", default=str(WORK_ROOT / "reports/stage2_bootstrap_data_contract_20260408.json"))
    p.add_argument("--stage1-best-ckpt", default=str(WORK_ROOT / "outputs/checkpoints/stage1_v2_longtrain_220m_mainline_20260408/best.pt"))
    p.add_argument("--lease-path", default=str(WORK_ROOT / "reports/stage1_v2_gpu_lease_20260408.json"))
    p.add_argument("--wave0-timeout-seconds", type=int, default=14400)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    paths = {
        "baseline_json": WORK_ROOT / "reports/stage2_stage1_frozen_baseline_eval_20260410.json",
        "baseline_md": WORK_ROOT / "docs/STAGE2_STAGE1_FROZEN_BASELINE_EVAL_20260410.md",
        "subset_manifest": WORK_ROOT / "manifests/protocol_v2/stage2_semantic_hard_subsets_20260410.json",
        "subset_report": WORK_ROOT / "reports/stage2_semantic_hard_subset_build_20260410.json",
        "subset_md": WORK_ROOT / "docs/STAGE2_SEMANTIC_HARD_SUBSETS_20260410.md",
        "diagnosis_json": WORK_ROOT / "reports/stage2_semantic_value_diagnosis_20260410.json",
        "diagnosis_md": WORK_ROOT / "docs/STAGE2_SEMANTIC_VALUE_DIAGNOSIS_20260410.md",
        "preflight_json": WORK_ROOT / "reports/stage2_semantic_bootstrap_preflight_20260410.json",
        "preflight_md": WORK_ROOT / "docs/STAGE2_SEMANTIC_BOOTSTRAP_PREFLIGHT_20260410.md",
        "cache_json": WORK_ROOT / "reports/stage2_semantic_bootstrap_cache_build_20260410.json",
        "cache_md": WORK_ROOT / "docs/STAGE2_SEMANTIC_BOOTSTRAP_CACHE_BUILD_20260410.md",
    }

    eval_gpu = _select_eval_gpu(str(args.lease_path))
    if int(eval_gpu.get("selected_gpu_id", -1)) >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(eval_gpu["selected_gpu_id"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        if Path(paths["baseline_json"]).exists():
            print("[ljs-diagnosis] reusing existing Stage1 frozen baseline eval", flush=True)
            baseline_payload = _read_json(paths["baseline_json"])
            stage1_metrics = dict(baseline_payload.get("metrics", {}))
        else:
            print("[ljs-diagnosis] evaluating Stage1 frozen baseline", flush=True)
            core_val = _make_dataset(["vspw", "vipseg"], "val", str(args.contract_path))
            stage1_model, stage1_meta = _load_stage1_model(device=device, stage1_ckpt=str(args.stage1_best_ckpt))
            stage1_metrics = _evaluate_stage1_baseline(stage1_model, _loader(core_val, batch_size=8), device=device)
            baseline_payload = {
                "generated_at_utc": now_iso(),
                "eval_binding": ["VSPW", "VIPSeg"],
                "checkpoint": str(args.stage1_best_ckpt),
                "stage1_backbone": stage1_meta,
                "metrics": {k: _f(stage1_metrics.get(k), 1e9) for k in METRIC_KEYS},
                "eval_batches": int(stage1_metrics.get("eval_batches", 0)),
            }
            _write_baseline_docs(baseline_payload, str(paths["baseline_json"]), str(paths["baseline_md"]))

        if Path(paths["diagnosis_json"]).exists():
            print("[ljs-diagnosis] reusing existing semantic value diagnosis", flush=True)
            diagnosis = _read_json(paths["diagnosis_json"])
        else:
            subset_payload = build_hard_subsets(str(args.contract_path))
            _write_subset_docs(subset_payload, str(paths["subset_manifest"]), str(paths["subset_report"]), str(paths["subset_md"]))

            diagnosis = build_diagnosis(
                stage1_metrics=stage1_metrics,
                subset_manifest=subset_payload,
                device=device,
                contract_path=str(args.contract_path),
                stage1_ckpt=str(args.stage1_best_ckpt),
            )
            _write_diagnosis_docs(diagnosis, str(paths["diagnosis_json"]), str(paths["diagnosis_md"]))
    finally:
        _release_eval_gpu(eval_gpu, str(args.lease_path))

    diagnosis = _read_json(paths["diagnosis_json"])
    if bool(diagnosis["scientific_judgment"].get("phase_b_triggered", False)):
        if Path(paths["preflight_json"]).exists():
            print("[ljs-diagnosis] reusing existing semantic bootstrap preflight", flush=True)
        else:
            preflight = _bootstrap_preflight()
            _write_preflight_docs(preflight, str(paths["preflight_json"]), str(paths["preflight_md"]))

        cache_file = WORK_ROOT / "data/processed/stage2_semantic_bootstrap_cache_20260410/core_trainval_pseudo_targets.jsonl"
        if Path(paths["cache_json"]).exists() and cache_file.exists() and cache_file.stat().st_size > 0:
            print("[ljs-diagnosis] reusing existing semantic bootstrap cache", flush=True)
        else:
            cache = build_bootstrap_cache(str(args.contract_path))
            _write_cache_docs(cache, str(paths["cache_json"]), str(paths["cache_md"]))
        wave0_summary_path = WORK_ROOT / "reports/stage2_semantic_rescue_wave0_summary_20260410.json"
        if wave0_summary_path.exists():
            existing_wave0 = _read_json(wave0_summary_path)
            if _wave0_summary_is_terminal(existing_wave0):
                print("[ljs-diagnosis] reusing terminal semantic rescue wave0 summary", flush=True)
                wave0 = existing_wave0
            else:
                wave0 = _launch_and_wait_wave0(args)
        else:
            wave0 = _launch_and_wait_wave0(args)
        rescue_signal = _rescue_positive_signal(wave0, diagnosis)
        diagnosis["semantic_rescue_wave0"] = {
            "summary_path": str(WORK_ROOT / "reports/stage2_semantic_rescue_wave0_summary_20260410.json"),
            "positive_signal": rescue_signal,
        }
        if rescue_signal["semantic_rescue_at_least_not_worse_than_current_cropenc_baseline"]:
            next_step = "stage2_semantic_rescue_fullscale_wave1"
        else:
            next_step = "redesign_stage2_semantic_objective"
    else:
        diagnosis["semantic_rescue_wave0"] = {"summary_path": "", "positive_signal": {}}
        next_step = "start_wave3_mainline_optimization"

    diagnosis["next_step_choice"] = next_step
    _write_diagnosis_docs(diagnosis, str(paths["diagnosis_json"]), str(paths["diagnosis_md"]))
    print(json.dumps({"diagnosis": str(paths["diagnosis_json"]), "next_step_choice": next_step}, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
