#!/usr/bin/env python3
from __future__ import annotations

from argparse import ArgumentParser
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple
import json
import math
import time

import numpy as np
import torch

from stwm.tools.run_tracewm_stage2_ljs_semantic_diagnosis_and_rescue_20260410 import (
    _f,
    _load_stage1_model,
    _load_stage2_modules,
    _make_dataset,
    _read_json,
    _score_core_sample,
)
from stwm.tools.run_tracewm_stage2_semantic_objective_redesign_v1_20260410 import _write_json, _write_md
from stwm.tools.run_tracewm_stage2_semantic_objective_redesign_v2_20260410 import _release_lease_safe, _select_gpu
from stwm.tracewm_v2_stage2.trainers import train_tracewm_stage2_smalltrain as stage2_trainer

try:
    import matplotlib.pyplot as plt
except Exception as exc:  # pragma: no cover
    plt = None
    _MPL_ERROR = str(exc)
else:
    _MPL_ERROR = ""

try:
    from PIL import Image
except Exception as exc:  # pragma: no cover
    Image = None
    _PIL_ERROR = str(exc)
else:
    _PIL_ERROR = ""


WORK_ROOT = Path("/home/chen034/workspace/stwm")
OUTPUT_DIR = WORK_ROOT / "outputs" / "visualizations" / "stage1_stage2_qualitative_pack_v3_20260413"
STAGE1_REPORT = WORK_ROOT / "reports" / "stage1_qualitative_pack_v3_20260413.json"
STAGE2_REPORT = WORK_ROOT / "reports" / "stage2_qualitative_pack_v3_20260413.json"
DOC_PATH = WORK_ROOT / "docs" / "STAGE1_STAGE2_QUALITATIVE_PACK_V3_20260413.md"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _clip01(x: float) -> float:
    return float(min(max(float(x), 0.0), 1.0))


def _image_from_sample(sample: Dict[str, Any], size: int = 512) -> np.ndarray:
    blank = np.ones((size, size, 3), dtype=np.uint8) * 245
    if Image is None:
        return blank
    frame_path = str(sample.get("semantic_frame_path", "") or "")
    if not frame_path:
        return blank
    p = Path(frame_path)
    if not p.exists():
        return blank
    try:
        img = Image.open(p).convert("RGB").resize((size, size))
        return np.asarray(img, dtype=np.uint8)
    except Exception:
        return blank


def _coords_from_sample(sample: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    obs = sample["obs_state"][:, 0, 0:2].detach().cpu().numpy()
    fut = sample["fut_state"][:, 0, 0:2].detach().cpu().numpy()
    return obs, fut


def _overlay_traj(ax: Any, coords: np.ndarray, color: str, label: str, linestyle: str = "-", alpha: float = 1.0) -> None:
    xs = np.clip(coords[:, 0], 0.0, 1.0) * 511.0
    ys = np.clip(coords[:, 1], 0.0, 1.0) * 511.0
    ax.plot(xs, ys, color=color, linewidth=2.0, linestyle=linestyle, alpha=alpha, label=label)
    ax.scatter(xs[-1:], ys[-1:], color=color, s=24, alpha=alpha)


def _case_id(prefix: str, idx: int) -> str:
    return f"{prefix}_{idx:03d}"


def _single_batch(sample: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    batch = stage2_trainer.stage2_semantic_collate_fn([sample])
    return stage2_trainer._to_device(batch, device=device, non_blocking=False)


def _stage1_predict(stage1_model: Any, sample: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    batch = _single_batch(sample, device)
    token_mask = batch["token_mask"]
    obs_state = batch["obs_state"]
    fut_state = batch["fut_state"]
    obs_len = int(obs_state.shape[1])
    fut_len = int(fut_state.shape[1])
    bsz, _, k_len, d_state = obs_state.shape
    state_seq = torch.zeros((bsz, obs_len + fut_len, k_len, d_state), device=device, dtype=obs_state.dtype)
    state_seq[:, :obs_len] = obs_state
    for step in range(fut_len):
        shifted = torch.zeros_like(state_seq)
        shifted[:, 0] = state_seq[:, 0]
        shifted[:, 1:] = state_seq[:, :-1]
        out = stage1_model(shifted, token_mask=token_mask)
        idx = obs_len + step
        state_seq[:, idx : idx + 1] = out["pred_state"][:, idx : idx + 1].detach()
    pred = state_seq[:, obs_len:, :, 0:2][0, :, 0].detach().cpu().numpy()
    target = fut_state[0, :, 0, 0:2].detach().cpu().numpy()
    endpoint = float(np.linalg.norm(pred[-1] - target[-1]))
    mean_l2 = float(np.linalg.norm(pred - target, axis=-1).mean())
    return {
        "pred_future": pred,
        "target_future": target,
        "endpoint_l2": endpoint,
        "coord_mean_l2": mean_l2,
    }


def _stage2_predict(
    loaded: Tuple[Any, Any, Any, str],
    stage1_model: Any,
    sample: Dict[str, Any],
    device: torch.device,
) -> Dict[str, Any]:
    semantic_encoder, semantic_fusion, readout_head, source = loaded
    batch = _single_batch(sample, device)
    out = stage2_trainer._free_rollout_predict(
        stage1_model=stage1_model,
        semantic_encoder=semantic_encoder,
        semantic_fusion=semantic_fusion,
        readout_head=readout_head,
        batch=batch,
        obs_len=8,
        fut_len=8,
        semantic_source_mainline=source,
    )
    pred = out["pred_coord"][0, :, 0].detach().cpu().numpy()
    target = out["target_coord"][0, :, 0].detach().cpu().numpy()
    endpoint = float(np.linalg.norm(pred[-1] - target[-1]))
    mean_l2 = float(np.linalg.norm(pred - target, axis=-1).mean())
    return {
        "pred_future": pred,
        "target_future": target,
        "endpoint_l2": endpoint,
        "coord_mean_l2": mean_l2,
    }


def _render_stage1_case(sample: Dict[str, Any], pred: Dict[str, Any], out_path: Path, title: str, note: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if plt is None:
        return
    bg = _image_from_sample(sample)
    obs, fut = _coords_from_sample(sample)
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.imshow(bg)
    ax.axis("off")
    ax.set_title(title)
    _overlay_traj(ax, obs, "black", "obs", alpha=0.9)
    _overlay_traj(ax, np.vstack([obs[-1:], fut]), "#2ca02c", "gt_future", linestyle="--", alpha=0.9)
    _overlay_traj(ax, np.vstack([obs[-1:], pred["pred_future"]]), "#d62728", "stage1_pred", alpha=0.95)
    ax.legend(loc="lower left", fontsize=8)
    ax.text(
        8,
        20,
        f"endpoint_l2={pred['endpoint_l2']:.4f}\ncoord_mean_l2={pred['coord_mean_l2']:.4f}\n{note}",
        fontsize=8,
        color="black",
        bbox={"facecolor": "white", "alpha": 0.7, "pad": 4},
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def _render_stage2_case(
    sample: Dict[str, Any],
    predictions: Dict[str, Dict[str, Any]],
    out_path: Path,
    title: str,
    note: str,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if plt is None:
        return
    bg = _image_from_sample(sample)
    obs, fut = _coords_from_sample(sample)
    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    ax.imshow(bg)
    ax.axis("off")
    ax.set_title(title)
    _overlay_traj(ax, obs, "black", "obs", alpha=0.9)
    _overlay_traj(ax, np.vstack([obs[-1:], fut]), "#2ca02c", "gt_future", linestyle="--", alpha=0.95)
    colors = {
        "cropenc": "#1f77b4",
        "legacysem": "#ff7f0e",
        "alignment_only": "#d62728",
        "persistence_declared": "#9467bd",
    }
    for key in ["cropenc", "legacysem", "alignment_only", "persistence_declared"]:
        pred = predictions[key]
        _overlay_traj(ax, np.vstack([obs[-1:], pred["pred_future"]]), colors[key], key, alpha=0.95)
    ax.legend(loc="lower left", fontsize=8)
    text = "\n".join(
        [
            f"cropenc_ep={predictions['cropenc']['endpoint_l2']:.4f}",
            f"legacy_ep={predictions['legacysem']['endpoint_l2']:.4f}",
            f"alignonly_ep={predictions['alignment_only']['endpoint_l2']:.4f}",
            f"persist_ep={predictions['persistence_declared']['endpoint_l2']:.4f}",
            note,
        ]
    )
    ax.text(
        8,
        20,
        text,
        fontsize=8,
        color="black",
        bbox={"facecolor": "white", "alpha": 0.75, "pad": 4},
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def _best_family_run(family_prefixes: List[str]) -> str:
    best_path = ""
    best_endpoint = math.inf
    for prefix in family_prefixes:
        final_json = WORK_ROOT / "reports" / f"{prefix}_final.json"
        if not final_json.exists():
            continue
        payload = _read_json(final_json)
        metrics = payload.get("best_checkpoint_metric", {}).get("metrics", {}) if isinstance(payload.get("best_checkpoint_metric", {}), dict) else {}
        endpoint = _f(metrics.get("free_rollout_endpoint_l2"), math.inf)
        if endpoint < best_endpoint:
            best_endpoint = endpoint
            best_path = prefix
    if not best_path:
        raise RuntimeError(f"no final json found for family_prefixes={family_prefixes}")
    return best_path


def _wait_for_v7(args: Any) -> Dict[str, Any]:
    deadline = time.time() + float(args.wait_timeout_seconds)
    while time.time() < deadline:
        if Path(args.v7_diagnosis_report).exists():
            payload = _read_json(args.v7_diagnosis_report)
            if bool(payload.get("v7_runs_terminal", False)):
                return payload
        time.sleep(float(args.poll_seconds))
    raise TimeoutError(f"timed out waiting for v7 diagnosis: {args.v7_diagnosis_report}")


def _preferred_checkpoint_for_run(run_name: str) -> str:
    final_json = WORK_ROOT / "reports" / f"{run_name}_final.json"
    if not final_json.exists():
        return "best.pt"
    payload = _read_json(final_json)
    sidecar = payload.get("sidecar_checkpoint_selection", {}) if isinstance(payload.get("sidecar_checkpoint_selection", {}), dict) else {}
    sidecar_ckpt = WORK_ROOT / "outputs" / "checkpoints" / run_name / "best_semantic_hard.pt"
    if bool(sidecar.get("sidecar_truly_diverged", False)) and sidecar_ckpt.exists():
        return "best_semantic_hard.pt"
    return "best.pt"


def _acquire_eval_gpu(args: Any, owner: str) -> Dict[str, Any]:
    deadline = time.time() + float(args.wait_timeout_seconds)
    last_error = ""
    while time.time() < deadline:
        try:
            return _select_gpu(run_name=owner, lease_path=str(args.shared_lease_path), required_mem_gb=40.0)
        except Exception as exc:
            last_error = str(exc)
            time.sleep(float(args.poll_seconds))
    raise RuntimeError(f"gpu_acquire_timeout owner={owner} last_error={last_error}")


def _select_stage1_cases(core_ds: Any, stage1_model: Any, device: torch.device, scan_window: int) -> List[Dict[str, Any]]:
    scored: List[Dict[str, Any]] = []
    limit = min(int(scan_window), len(core_ds))
    for idx in range(limit):
        sample = core_ds[idx]
        pred = _stage1_predict(stage1_model, sample, device)
        meta = dict(sample.get("meta", {}))
        scores = _score_core_sample(sample)
        scored.append(
            {
                "dataset_index": int(idx),
                "dataset": str(meta.get("dataset", "")),
                "clip_id": str(meta.get("clip_id", "")),
                "stage1": pred,
                "motion": float(scores.get("motion", 0.0)),
                "endpoint_l2": float(pred["endpoint_l2"]),
                "coord_mean_l2": float(pred["coord_mean_l2"]),
            }
        )
    easy_rank = sorted(scored, key=lambda x: (x["endpoint_l2"], -x["motion"]))
    dynamic_rank = sorted(scored, key=lambda x: (x["motion"], -x["endpoint_l2"]), reverse=True)
    fail_rank = sorted(scored, key=lambda x: (x["endpoint_l2"], x["motion"]), reverse=True)

    selected: List[Dict[str, Any]] = []
    used_indices: set[int] = set()
    used_clip_keys: set[Tuple[str, str]] = set()

    def _pick(rows: List[Dict[str, Any]], bucket: str, why: str, limit_pick: int) -> int:
        picked = 0
        for row in rows:
            idx = int(row["dataset_index"])
            clip_key = (str(row["dataset"]), str(row["clip_id"]))
            if idx in used_indices or clip_key in used_clip_keys:
                continue
            payload = dict(row)
            payload["bucket"] = bucket
            payload["why_selected"] = why
            selected.append(payload)
            used_indices.add(idx)
            used_clip_keys.add(clip_key)
            picked += 1
            if picked >= limit_pick:
                break
        return picked

    bucket_specs = [
        ("easy_cases", easy_rank, "low endpoint error under frozen trace rollout"),
        ("dynamic_change_cases", dynamic_rank, "high motion / stronger future trace variation"),
        ("failure_boundary_cases", fail_rank, "highest endpoint error in bounded scan window"),
    ]
    for bucket, rank_rows, why in bucket_specs:
        target = 3
        picked = _pick(rank_rows, bucket, why, target)
        if picked >= target:
            continue
        if bucket == "easy_cases":
            fallback = sorted(scored, key=lambda x: (x["endpoint_l2"], -x["motion"]))
        elif bucket == "dynamic_change_cases":
            fallback = sorted(scored, key=lambda x: (x["motion"], -x["endpoint_l2"]), reverse=True)
        else:
            fallback = sorted(scored, key=lambda x: (x["endpoint_l2"], x["motion"]), reverse=True)
        _pick(fallback, bucket, f"{why}; diversity fallback fill", target - picked)
    return selected


def _subset_tag_map(manifest: Dict[str, Any], allowed: List[str]) -> Dict[int, List[str]]:
    tag_map: Dict[int, List[str]] = {}
    subsets = manifest.get("subsets", {}) if isinstance(manifest.get("subsets", {}), dict) else {}
    for subset_name in allowed:
        items = subsets.get(subset_name, {}).get("items", []) if isinstance(subsets.get(subset_name, {}), dict) else []
        for item in items if isinstance(items, list) else []:
            if not isinstance(item, dict) or "dataset_index" not in item:
                continue
            idx = int(item["dataset_index"])
            tag_map.setdefault(idx, []).append(subset_name)
    return tag_map


def _select_stage2_cases(
    core_ds: Any,
    stage1_model: Any,
    device: torch.device,
    crop_loaded: Tuple[Any, Any, Any, str],
    legacy_loaded: Tuple[Any, Any, Any, str],
    align_loaded: Tuple[Any, Any, Any, str],
    persist_loaded: Tuple[Any, Any, Any, str],
    manifest: Dict[str, Any],
    persistence_declared_but_inactive_global: bool,
) -> Dict[str, List[Dict[str, Any]]]:
    tag_map = _subset_tag_map(
        manifest,
        ["occlusion_reappearance", "crossing_or_interaction_ambiguity", "small_object_or_low_area", "appearance_change_or_semantic_shift"],
    )
    candidates: List[Dict[str, Any]] = []
    for idx, tags in tag_map.items():
        sample = core_ds[idx]
        crop = _stage2_predict(crop_loaded, stage1_model, sample, device)
        legacy = _stage2_predict(legacy_loaded, stage1_model, sample, device)
        align = _stage2_predict(align_loaded, stage1_model, sample, device)
        persist = _stage2_predict(persist_loaded, stage1_model, sample, device)
        meta = dict(sample.get("meta", {}))
        base_best = min(float(crop["endpoint_l2"]), float(legacy["endpoint_l2"]))
        align_margin = float(base_best - float(align["endpoint_l2"]))
        persist_margin = float(base_best - float(persist["endpoint_l2"]))
        persist_vs_align_margin = float(float(align["endpoint_l2"]) - float(persist["endpoint_l2"]))
        align_vs_persist_margin = float(-persist_vs_align_margin)
        entry = {
            "dataset_index": int(idx),
            "dataset": str(meta.get("dataset", "")),
            "clip_id": str(meta.get("clip_id", "")),
            "subset_tags": list(tags),
            "predictions": {
                "cropenc": crop,
                "legacysem": legacy,
                "alignment_only": align,
                "persistence_declared": persist,
            },
            "align_margin": float(align_margin),
            "persist_margin": float(persist_margin),
            "persist_vs_align_margin": float(persist_vs_align_margin),
            "align_vs_persist_margin": float(align_vs_persist_margin),
        }
        candidates.append(entry)

    def _extend_unique(base: List[Dict[str, Any]], pool: List[Dict[str, Any]], limit: int) -> None:
        used = {(str(x.get("dataset", "")), str(x.get("clip_id", ""))) for x in base}
        for row in pool:
            if len(base) >= limit:
                break
            key = (str(row.get("dataset", "")), str(row.get("clip_id", "")))
            if key in used:
                continue
            base.append(row)
            used.add(key)

    persistence_active_improved_pref = sorted(
        [x for x in candidates if float(x.get("persist_vs_align_margin", 0.0)) > 0.003 and float(x.get("persist_margin", 0.0)) > 0.0],
        key=lambda x: (x["persist_vs_align_margin"], x["persist_margin"], len(x["subset_tags"])),
        reverse=True,
    )
    alignment_only_wins_pref = sorted(
        [x for x in candidates if float(x.get("align_vs_persist_margin", 0.0)) > 0.003 and float(x.get("align_margin", 0.0)) > 0.0],
        key=lambda x: (x["align_vs_persist_margin"], x["align_margin"], len(x["subset_tags"])),
        reverse=True,
    )
    inactive_pref = sorted(
        [x for x in candidates if bool(persistence_declared_but_inactive_global) and float(x.get("align_vs_persist_margin", 0.0)) >= 0.0],
        key=lambda x: (x["align_vs_persist_margin"], len(x["subset_tags"])),
        reverse=True,
    )
    unresolved_pref = sorted(
        candidates,
        key=lambda x: (min(float(x.get("align_margin", 0.0)), float(x.get("persist_margin", 0.0))), -len(x["subset_tags"])),
    )

    groups: Dict[str, List[Dict[str, Any]]] = {
        "persistence_active_improved_cases": [],
        "alignment_only_wins_cases": [],
        "persistence_declared_but_inactive_cases": [],
        "unresolved_or_failure_cases": [],
    }
    _extend_unique(groups["persistence_active_improved_cases"], persistence_active_improved_pref, limit=4)
    _extend_unique(groups["alignment_only_wins_cases"], alignment_only_wins_pref, limit=4)
    _extend_unique(groups["persistence_declared_but_inactive_cases"], inactive_pref, limit=4)
    _extend_unique(groups["unresolved_or_failure_cases"], unresolved_pref, limit=4)
    return groups


def build_stage1_pack(args: Any, core_ds: Any, stage1_model: Any, device: torch.device) -> Dict[str, Any]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cases = _select_stage1_cases(core_ds, stage1_model, device, scan_window=int(args.stage1_scan_window))
    case_rows: List[Dict[str, Any]] = []
    for i, case in enumerate(cases):
        sample = core_ds[int(case["dataset_index"])]
        out_path = OUTPUT_DIR / "stage1" / f"{case['bucket']}_{_case_id('stage1', i)}.png"
        note = f"dataset={case['dataset']} clip={case['clip_id']}"
        _render_stage1_case(sample, case["stage1"], out_path, title=f"Stage1 {case['bucket']}", note=note)
        case_rows.append(
            {
                "case_id": _case_id("stage1", i),
                "bucket": str(case["bucket"]),
                "dataset_source": str(case["dataset"]),
                "clip_id": str(case["clip_id"]),
                "dataset_index": int(case["dataset_index"]),
                "why_selected": str(case["why_selected"]),
                "qualitative_interpretation": (
                    "stable future trace continuation"
                    if case["bucket"] == "easy_cases"
                    else ("large motion regime with visible trajectory change" if case["bucket"] == "dynamic_change_cases" else "clear failure or boundary behavior under frozen Stage1 rollout")
                ),
                "semantic_frame_path": str(sample.get("semantic_frame_path", "")),
                "render_path": str(out_path),
                "metrics": {
                    "free_rollout_endpoint_l2": float(case["endpoint_l2"]),
                    "free_rollout_coord_mean_l2": float(case["coord_mean_l2"]),
                },
            }
        )
    payload = {
        "generated_at_utc": now_iso(),
        "pack_type": "stage1_qualitative_pack_v3",
        "source_checkpoint": str(args.stage1_best_ckpt),
        "selection_policy": {
            "scan_window": int(args.stage1_scan_window),
            "groups": ["easy_cases", "dynamic_change_cases", "failure_boundary_cases"],
            "target_case_count_per_group": 3,
            "cross_group_dedup_key": ["dataset", "clip_id"],
        },
        "cases": case_rows,
        "notes": {
            "matplotlib_available": bool(plt is not None),
            "pil_available": bool(Image is not None),
            "matplotlib_error": _MPL_ERROR,
            "pil_error": _PIL_ERROR,
        },
    }
    _write_json(args.stage1_pack_report, payload)
    return payload


def build_stage2_pack(args: Any, core_ds: Any, stage1_model: Any, device: torch.device, v7_diag: Dict[str, Any]) -> Dict[str, Any]:
    manifest = _read_json(args.semantic_hard_manifest_path)
    success = v7_diag.get("success_criteria", {}) if isinstance(v7_diag.get("success_criteria", {}), dict) else {}
    align_run = str(success.get("best_alignment_only_run_name", ""))
    persist_run = str(success.get("best_persistence_run_name", ""))
    if not align_run or align_run == "none":
        raise RuntimeError("v7 diagnosis missing best alignment-only run")
    if not persist_run or persist_run == "none":
        raise RuntimeError("v7 diagnosis missing best persistence run")

    crop_run = _best_family_run(
        [
            "stage2_fullscale_core_cropenc_seed42_20260409",
            "stage2_fullscale_core_cropenc_seed123_20260409",
            "stage2_fullscale_core_cropenc_seed456_20260409",
            "stage2_fullscale_core_cropenc_seed789_wave2_20260409",
        ]
    )
    legacy_run = _best_family_run(
        [
            "stage2_fullscale_core_legacysem_seed42_20260409",
            "stage2_fullscale_core_legacysem_seed123_wave2_20260409",
            "stage2_fullscale_core_legacysem_seed456_wave2_20260409",
        ]
    )
    crop_loaded = _load_stage2_modules(str(WORK_ROOT / "outputs/checkpoints" / crop_run / "best.pt"), device, stage1_model)
    legacy_loaded = _load_stage2_modules(str(WORK_ROOT / "outputs/checkpoints" / legacy_run / "best.pt"), device, stage1_model)
    align_ckpt_name = _preferred_checkpoint_for_run(align_run)
    persist_ckpt_name = _preferred_checkpoint_for_run(persist_run)
    align_loaded = _load_stage2_modules(str(WORK_ROOT / "outputs/checkpoints" / align_run / align_ckpt_name), device, stage1_model)
    persist_loaded = _load_stage2_modules(str(WORK_ROOT / "outputs/checkpoints" / persist_run / persist_ckpt_name), device, stage1_model)

    groups = _select_stage2_cases(
        core_ds,
        stage1_model,
        device,
        crop_loaded,
        legacy_loaded,
        align_loaded,
        persist_loaded,
        manifest,
        persistence_declared_but_inactive_global=bool(success.get("persistence_declared_but_inactive_all", False)),
    )
    case_rows: List[Dict[str, Any]] = []
    group_meta = {
        "persistence_active_improved_cases": {
            "why": "persistence branch beats alignment-only and baseline envelope on semantic-hard clips",
            "interpretation": "persistence branch appears active and helpful on this clip",
        },
        "alignment_only_wins_cases": {
            "why": "alignment-only branch beats persistence-declared branch and baseline envelope",
            "interpretation": "calibration-only branch appears sufficient on this clip",
        },
        "persistence_declared_but_inactive_cases": {
            "why": "persistence was declared but did not show active gain pattern",
            "interpretation": "declared persistence likely inactive or non-contributive",
        },
        "unresolved_or_failure_cases": {
            "why": "neither branch clearly dominates under semantic-hard stress",
            "interpretation": "remaining ambiguity/failure case",
        },
    }

    for group_name, cases in groups.items():
        for i, case in enumerate(cases):
            sample = core_ds[int(case["dataset_index"])]
            out_path = OUTPUT_DIR / "stage2" / f"{group_name}_{_case_id('stage2', len(case_rows))}.png"
            note = (
                f"tags={','.join(case['subset_tags'])}\n"
                f"align_margin={float(case['align_margin']):+.4f}\n"
                f"persist_margin={float(case['persist_margin']):+.4f}\n"
                f"persist_vs_align={float(case['persist_vs_align_margin']):+.4f}"
            )
            _render_stage2_case(
                sample,
                case["predictions"],
                out_path,
                title=f"Stage2 {group_name}",
                note=note,
            )
            case_rows.append(
                {
                    "case_id": _case_id("stage2", len(case_rows)),
                    "group": group_name,
                    "dataset_source": str(case["dataset"]),
                    "clip_id": str(case["clip_id"]),
                    "dataset_index": int(case["dataset_index"]),
                    "subset_tags": list(case["subset_tags"]),
                    "why_selected": str(group_meta.get(group_name, {}).get("why", "taxonomy-driven selection")),
                    "qualitative_interpretation": str(group_meta.get(group_name, {}).get("interpretation", "taxonomy-driven interpretation")),
                    "semantic_frame_path": str(sample.get("semantic_frame_path", "")),
                    "render_path": str(out_path),
                    "cropenc_run": crop_run,
                    "legacysem_run": legacy_run,
                    "alignment_only_run": align_run,
                    "alignment_only_checkpoint_used": align_ckpt_name,
                    "persistence_declared_run": persist_run,
                    "persistence_declared_checkpoint_used": persist_ckpt_name,
                    "align_margin": float(case["align_margin"]),
                    "persist_margin": float(case["persist_margin"]),
                    "persist_vs_align_margin": float(case["persist_vs_align_margin"]),
                    "metrics": {
                        key: {
                            "free_rollout_endpoint_l2": float(pred["endpoint_l2"]),
                            "free_rollout_coord_mean_l2": float(pred["coord_mean_l2"]),
                        }
                        for key, pred in case["predictions"].items()
                    },
                }
            )
    payload = {
        "generated_at_utc": now_iso(),
        "pack_type": "stage2_qualitative_pack_v3",
        "selection_policy": {
            "base_panel": "semantic-hard subsets from protocol_v2 manifest",
            "taxonomy": [
                "persistence_active_improved_cases",
                "alignment_only_wins_cases",
                "persistence_declared_but_inactive_cases",
                "unresolved_or_failure_cases",
            ],
            "persistence_vs_alignment_delta_threshold": 0.003,
            "diversity_enforced": "unique (dataset, clip_id) per group",
            "target_case_count_per_group": 4,
        },
        "best_cropenc_run": crop_run,
        "best_legacysem_run": legacy_run,
        "best_alignment_only_run": align_run,
        "best_alignment_only_checkpoint_used": align_ckpt_name,
        "best_persistence_declared_run": persist_run,
        "best_persistence_declared_checkpoint_used": persist_ckpt_name,
        "persistence_declared_but_inactive_all": bool(success.get("persistence_declared_but_inactive_all", False)),
        "cases": case_rows,
    }
    _write_json(args.stage2_pack_report, payload)
    return payload


def write_doc(path: str | Path, stage1_pack: Dict[str, Any], stage2_pack: Dict[str, Any]) -> None:
    lines = [
        "# Stage1 / Stage2 Qualitative Pack V3",
        "",
        f"- generated_at_utc: {now_iso()}",
        f"- stage1_pack: {STAGE1_REPORT}",
        f"- stage2_pack: {STAGE2_REPORT}",
        f"- output_dir: {OUTPUT_DIR}",
        "",
        "## Stage1",
        "",
        "| case_id | bucket | dataset | clip_id | endpoint_l2 | render |",
        "|---|---|---|---|---:|---|",
    ]
    for case in stage1_pack.get("cases", []) if isinstance(stage1_pack.get("cases", []), list) else []:
        lines.append(
            f"| {case['case_id']} | {case['bucket']} | {case['dataset_source']} | {case['clip_id']} | {_f(case.get('metrics', {}).get('free_rollout_endpoint_l2'), 1e9):.4f} | {case['render_path']} |"
        )
    lines.extend(
        [
            "",
            "## Stage2",
            "",
            "| case_id | group | dataset | clip_id | tags | cropenc_ep | legacy_ep | align_ep | persist_ep | persist_vs_align | render |",
            "|---|---|---|---|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for case in stage2_pack.get("cases", []) if isinstance(stage2_pack.get("cases", []), list) else []:
        metrics = case.get("metrics", {}) if isinstance(case.get("metrics", {}), dict) else {}
        lines.append(
            f"| {case['case_id']} | {case['group']} | {case['dataset_source']} | {case['clip_id']} | {','.join(case.get('subset_tags', []))} | "
            f"{_f(metrics.get('cropenc', {}).get('free_rollout_endpoint_l2'), 1e9):.4f} | "
            f"{_f(metrics.get('legacysem', {}).get('free_rollout_endpoint_l2'), 1e9):.4f} | "
            f"{_f(metrics.get('alignment_only', {}).get('free_rollout_endpoint_l2'), 1e9):.4f} | "
            f"{_f(metrics.get('persistence_declared', {}).get('free_rollout_endpoint_l2'), 1e9):.4f} | "
            f"{float(case.get('persist_vs_align_margin', 0.0)):+.4f} | {case['render_path']} |"
        )
    _write_md(path, lines)


def parse_args() -> Any:
    p = ArgumentParser(description="Build Stage1/Stage2 qualitative packs v3 for manual inspection")
    p.add_argument("--work-root", default=str(WORK_ROOT))
    p.add_argument("--stage2-contract-json", default=str(WORK_ROOT / "reports/stage2_bootstrap_data_contract_20260408.json"))
    p.add_argument("--stage1-best-ckpt", default=str(WORK_ROOT / "outputs/checkpoints/stage1_v2_longtrain_220m_mainline_20260408/best.pt"))
    p.add_argument("--shared-lease-path", default=str(WORK_ROOT / "reports/stage1_v2_gpu_lease_20260408.json"))
    p.add_argument("--semantic-hard-manifest-path", default=str(WORK_ROOT / "manifests/protocol_v2/stage2_semantic_hard_subsets_20260410.json"))
    p.add_argument("--v7-diagnosis-report", default=str(WORK_ROOT / "reports/stage2_semantic_objective_redesign_v7_diagnosis_20260413.json"))
    p.add_argument("--stage1-pack-report", default=str(STAGE1_REPORT))
    p.add_argument("--stage2-pack-report", default=str(STAGE2_REPORT))
    p.add_argument("--qualitative-doc", default=str(DOC_PATH))
    p.add_argument("--stage1-scan-window", type=int, default=96)
    p.add_argument("--wait-timeout-seconds", type=int, default=21600)
    p.add_argument("--poll-seconds", type=int, default=60)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    lease_path = str(args.shared_lease_path)

    stage1_pack: Dict[str, Any]
    eval_gpu = _acquire_eval_gpu(args, owner="stage1_qualitative_pack_v3_20260413")
    try:
        import os

        os.environ["CUDA_VISIBLE_DEVICES"] = str(eval_gpu["selected_gpu_id"])
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[qual-pack] stage1 phase device={device} eval_gpu={eval_gpu}", flush=True)
        core_ds = _make_dataset(["vspw", "vipseg"], "val", str(args.stage2_contract_json), max_samples=-1)
        stage1_model, _ = _load_stage1_model(device=device, stage1_ckpt=str(args.stage1_best_ckpt))
        stage1_pack = build_stage1_pack(args, core_ds, stage1_model, device)
    finally:
        _release_lease_safe(str(eval_gpu.get("lease_id", "")), lease_path)

    v7_diag = _wait_for_v7(args)

    eval_gpu = _acquire_eval_gpu(args, owner="stage2_qualitative_pack_v3_20260413")
    try:
        import os

        os.environ["CUDA_VISIBLE_DEVICES"] = str(eval_gpu["selected_gpu_id"])
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[qual-pack] stage2 phase device={device} eval_gpu={eval_gpu}", flush=True)
        core_ds = _make_dataset(["vspw", "vipseg"], "val", str(args.stage2_contract_json), max_samples=-1)
        stage1_model, _ = _load_stage1_model(device=device, stage1_ckpt=str(args.stage1_best_ckpt))
        stage2_pack = build_stage2_pack(args, core_ds, stage1_model, device, v7_diag)
        write_doc(args.qualitative_doc, stage1_pack, stage2_pack)
        print(json.dumps({"stage1_pack": str(args.stage1_pack_report), "stage2_pack": str(args.stage2_pack_report), "doc": str(args.qualitative_doc)}, ensure_ascii=True, indent=2))
    finally:
        _release_lease_safe(str(eval_gpu.get("lease_id", "")), lease_path)


if __name__ == "__main__":
    main()
