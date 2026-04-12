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
OUTPUT_DIR = WORK_ROOT / "outputs" / "visualizations" / "stage1_stage2_qualitative_pack_20260411"
STAGE1_REPORT = WORK_ROOT / "reports" / "stage1_qualitative_pack_20260411.json"
STAGE2_REPORT = WORK_ROOT / "reports" / "stage2_qualitative_pack_20260411.json"
DOC_PATH = WORK_ROOT / "docs" / "STAGE1_STAGE2_QUALITATIVE_PACK_20260411.md"


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
        "best_rescue": "#d62728",
    }
    for key in ["cropenc", "legacysem", "best_rescue"]:
        pred = predictions[key]
        _overlay_traj(ax, np.vstack([obs[-1:], pred["pred_future"]]), colors[key], key, alpha=0.95)
    ax.legend(loc="lower left", fontsize=8)
    text = "\n".join(
        [
            f"cropenc_ep={predictions['cropenc']['endpoint_l2']:.4f}",
            f"legacy_ep={predictions['legacysem']['endpoint_l2']:.4f}",
            f"rescue_ep={predictions['best_rescue']['endpoint_l2']:.4f}",
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


def _wait_for_v5(args: Any) -> Dict[str, Any]:
    deadline = time.time() + float(args.wait_timeout_seconds)
    while time.time() < deadline:
        if Path(args.v5_diagnosis_report).exists():
            payload = _read_json(args.v5_diagnosis_report)
            if bool(payload.get("v5_runs_terminal", False)):
                return payload
        time.sleep(float(args.poll_seconds))
    raise TimeoutError(f"timed out waiting for v5 diagnosis: {args.v5_diagnosis_report}")


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
    easy = sorted(scored, key=lambda x: (x["endpoint_l2"], -x["motion"]))[:3]
    dynamic = sorted(scored, key=lambda x: (x["motion"], -x["endpoint_l2"]), reverse=True)[:3]
    fail = sorted(scored, key=lambda x: (x["endpoint_l2"], x["motion"]), reverse=True)[:3]
    out: List[Dict[str, Any]] = []
    for bucket, rows, why in [
        ("easy_cases", easy, "low endpoint error under frozen trace rollout"),
        ("dynamic_change_cases", dynamic, "high motion / stronger future trace variation"),
        ("failure_boundary_cases", fail, "highest endpoint error in bounded scan window"),
    ]:
        for row in rows:
            row = dict(row)
            row["bucket"] = bucket
            row["why_selected"] = why
            out.append(row)
    return out


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
    rescue_loaded: Tuple[Any, Any, Any, str],
    manifest: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    tag_map = _subset_tag_map(
        manifest,
        ["occlusion_reappearance", "crossing_or_interaction_ambiguity", "small_object_or_low_area", "appearance_change_or_semantic_shift"],
    )
    positives: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    for idx, tags in tag_map.items():
        sample = core_ds[idx]
        crop = _stage2_predict(crop_loaded, stage1_model, sample, device)
        legacy = _stage2_predict(legacy_loaded, stage1_model, sample, device)
        rescue = _stage2_predict(rescue_loaded, stage1_model, sample, device)
        meta = dict(sample.get("meta", {}))
        entry = {
            "dataset_index": int(idx),
            "dataset": str(meta.get("dataset", "")),
            "clip_id": str(meta.get("clip_id", "")),
            "subset_tags": list(tags),
            "predictions": {
                "cropenc": crop,
                "legacysem": legacy,
                "best_rescue": rescue,
            },
        }
        rescue_margin = min(crop["endpoint_l2"], legacy["endpoint_l2"]) - rescue["endpoint_l2"]
        failure_margin = rescue["endpoint_l2"] - min(crop["endpoint_l2"], legacy["endpoint_l2"])
        entry["rescue_margin"] = float(rescue_margin)
        entry["failure_margin"] = float(failure_margin)
        if rescue_margin > 0.005:
            positives.append(entry)
        if failure_margin > 0.005:
            failures.append(entry)
    positives = sorted(positives, key=lambda x: (x["rescue_margin"], len(x["subset_tags"])), reverse=True)[:4]
    failures = sorted(failures, key=lambda x: (x["failure_margin"], len(x["subset_tags"])), reverse=True)[:4]
    return positives, failures


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
        "pack_type": "stage1_qualitative_pack",
        "source_checkpoint": str(args.stage1_best_ckpt),
        "selection_policy": {
            "scan_window": int(args.stage1_scan_window),
            "groups": ["easy_cases", "dynamic_change_cases", "failure_boundary_cases"],
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


def build_stage2_pack(args: Any, core_ds: Any, stage1_model: Any, device: torch.device, v5_diag: Dict[str, Any]) -> Dict[str, Any]:
    manifest = _read_json(args.semantic_hard_manifest_path)
    rescue_run = str(v5_diag.get("success_criteria", {}).get("semantic_hard_best_run_name", "") or v5_diag.get("success_criteria", {}).get("overall_best_run_name", ""))
    if not rescue_run:
        raise RuntimeError("v5 diagnosis missing best rescue run")
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
    rescue_final = _read_json(WORK_ROOT / "reports" / f"{rescue_run}_final.json")
    rescue_sidecar = rescue_final.get("sidecar_checkpoint_selection", {}) if isinstance(rescue_final.get("sidecar_checkpoint_selection", {}), dict) else {}
    rescue_ckpt_name = "best_semantic_hard.pt" if bool(rescue_sidecar.get("sidecar_truly_diverged", False)) and (WORK_ROOT / "outputs/checkpoints" / rescue_run / "best_semantic_hard.pt").exists() else "best.pt"
    rescue_loaded = _load_stage2_modules(str(WORK_ROOT / "outputs/checkpoints" / rescue_run / rescue_ckpt_name), device, stage1_model)

    positives, failures = _select_stage2_cases(core_ds, stage1_model, device, crop_loaded, legacy_loaded, rescue_loaded, manifest)
    case_rows: List[Dict[str, Any]] = []
    for group_name, cases in [("positive_cases", positives), ("failure_cases", failures)]:
        for i, case in enumerate(cases):
            sample = core_ds[int(case["dataset_index"])]
            out_path = OUTPUT_DIR / "stage2" / f"{group_name}_{_case_id('stage2', len(case_rows))}.png"
            note = (
                f"tags={','.join(case['subset_tags'])}\n"
                f"rescue_margin={float(case['rescue_margin']):+.4f}\n"
                f"failure_margin={float(case['failure_margin']):+.4f}"
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
                    "why_selected": (
                        "best rescue beats both cropenc and legacysem on semantic-hard case"
                        if group_name == "positive_cases"
                        else "best rescue still loses against at least one baseline on semantic-hard case"
                    ),
                    "qualitative_interpretation": (
                        "semantic-hard positive case where rescue readout calibration helps future localization"
                        if group_name == "positive_cases"
                        else "semantic-hard failure case where rescue still does not stabilize future semantic discrimination"
                    ),
                    "semantic_frame_path": str(sample.get("semantic_frame_path", "")),
                    "render_path": str(out_path),
                    "cropenc_run": crop_run,
                    "legacysem_run": legacy_run,
                    "best_rescue_run": rescue_run,
                    "best_rescue_checkpoint_used": rescue_ckpt_name,
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
        "pack_type": "stage2_qualitative_pack",
        "selection_policy": {
            "base_panel": "semantic-hard subsets from protocol_v2 manifest",
            "positive_margin_threshold": 0.005,
            "failure_margin_threshold": 0.005,
        },
        "best_cropenc_run": crop_run,
        "best_legacysem_run": legacy_run,
        "best_rescue_run": rescue_run,
        "best_rescue_checkpoint_used": rescue_ckpt_name,
        "cases": case_rows,
    }
    _write_json(args.stage2_pack_report, payload)
    return payload


def write_doc(path: str | Path, stage1_pack: Dict[str, Any], stage2_pack: Dict[str, Any]) -> None:
    lines = [
        "# Stage1 / Stage2 Qualitative Pack",
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
            "| case_id | group | dataset | clip_id | tags | cropenc_ep | legacy_ep | rescue_ep | render |",
            "|---|---|---|---|---|---:|---:|---:|---|",
        ]
    )
    for case in stage2_pack.get("cases", []) if isinstance(stage2_pack.get("cases", []), list) else []:
        metrics = case.get("metrics", {}) if isinstance(case.get("metrics", {}), dict) else {}
        lines.append(
            f"| {case['case_id']} | {case['group']} | {case['dataset_source']} | {case['clip_id']} | {','.join(case.get('subset_tags', []))} | "
            f"{_f(metrics.get('cropenc', {}).get('free_rollout_endpoint_l2'), 1e9):.4f} | "
            f"{_f(metrics.get('legacysem', {}).get('free_rollout_endpoint_l2'), 1e9):.4f} | "
            f"{_f(metrics.get('best_rescue', {}).get('free_rollout_endpoint_l2'), 1e9):.4f} | {case['render_path']} |"
        )
    _write_md(path, lines)


def parse_args() -> Any:
    p = ArgumentParser(description="Build Stage1/Stage2 qualitative packs for manual inspection")
    p.add_argument("--work-root", default=str(WORK_ROOT))
    p.add_argument("--stage2-contract-json", default=str(WORK_ROOT / "reports/stage2_bootstrap_data_contract_20260408.json"))
    p.add_argument("--stage1-best-ckpt", default=str(WORK_ROOT / "outputs/checkpoints/stage1_v2_longtrain_220m_mainline_20260408/best.pt"))
    p.add_argument("--shared-lease-path", default=str(WORK_ROOT / "reports/stage1_v2_gpu_lease_20260408.json"))
    p.add_argument("--semantic-hard-manifest-path", default=str(WORK_ROOT / "manifests/protocol_v2/stage2_semantic_hard_subsets_20260410.json"))
    p.add_argument("--v5-diagnosis-report", default=str(WORK_ROOT / "reports/stage2_semantic_objective_redesign_v5_diagnosis_20260411.json"))
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
    eval_gpu = _acquire_eval_gpu(args, owner="stage1_qualitative_pack_20260411")
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

    v5_diag = _wait_for_v5(args)

    eval_gpu = _acquire_eval_gpu(args, owner="stage2_qualitative_pack_20260411")
    try:
        import os

        os.environ["CUDA_VISIBLE_DEVICES"] = str(eval_gpu["selected_gpu_id"])
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[qual-pack] stage2 phase device={device} eval_gpu={eval_gpu}", flush=True)
        core_ds = _make_dataset(["vspw", "vipseg"], "val", str(args.stage2_contract_json), max_samples=-1)
        stage1_model, _ = _load_stage1_model(device=device, stage1_ckpt=str(args.stage1_best_ckpt))
        stage2_pack = build_stage2_pack(args, core_ds, stage1_model, device, v5_diag)
        write_doc(args.qualitative_doc, stage1_pack, stage2_pack)
        print(json.dumps({"stage1_pack": str(args.stage1_pack_report), "stage2_pack": str(args.stage2_pack_report), "doc": str(args.qualitative_doc)}, ensure_ascii=True, indent=2))
    finally:
        _release_lease_safe(str(eval_gpu.get("lease_id", "")), lease_path)


if __name__ == "__main__":
    main()
