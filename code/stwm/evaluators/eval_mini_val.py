from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import json
import random
from typing import Any

import numpy as np
from PIL import Image
import torch

from stwm.datasets.stwm_dataset import STWMDataset
from stwm.models.stwm_1b import STWM1B, STWMConfig, load_model_config
from stwm.models.stwm_v4_2 import STWMV42, STWMV42Config, load_model_config_v4_2
from stwm.modules.semantic_adapter import SemanticAdapter
from stwm.modules.tokenizer import SemanticTrajectoryTokenizer
from stwm.modules.trace_adapter import TraceAdapter
from stwm.utils.week2_protocol import AblationConfig, ablation_from_args, build_tokens_for_sample


EVALUATOR_VERSION = "v2_4_detached_frozen"
SUPPORTED_PROTOCOL_VERSIONS = ["v1", "v2", "v2_1", "v2_2", "v2_3", EVALUATOR_VERSION]
PROTOCOL_VERSION_ALIAS = {
    EVALUATOR_VERSION: "v2_3",
}
STABLE_COMPARABLE_METRICS = [
    "query_localization_error",
    "query_top1_acc",
    "query_hit_rate",
    "identity_consistency",
    "identity_switch_rate",
    "occlusion_recovery_acc",
    "future_trajectory_l1",
    "future_mask_iou",
    "visibility_accuracy",
    "visibility_f1",
]


def _canonical_protocol_version(protocol_version: str) -> str:
    raw = str(protocol_version)
    return PROTOCOL_VERSION_ALIAS.get(raw, raw)


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Evaluate STWM mini-val protocol (week-2)")
    parser.add_argument("--data-root", default="/home/chen034/workspace/stwm/data/external")
    parser.add_argument("--manifest", default="/home/chen034/workspace/stwm/manifests/minisplits/stwm_week1_mini.json")
    parser.add_argument("--dataset", default="vspw")
    parser.add_argument("--max-clips", type=int, default=20)
    parser.add_argument("--obs-steps", type=int, default=8)
    parser.add_argument("--pred-steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--model-preset", default="prototype_220m")
    parser.add_argument("--preset-file", default="/home/chen034/workspace/stwm/code/stwm/configs/model_presets.json")
    parser.add_argument("--device", default="auto")

    parser.add_argument("--disable-semantics", action="store_true")
    parser.add_argument("--disable-trajectory", action="store_true")
    parser.add_argument("--disable-identity-memory", action="store_true")
    parser.add_argument("--identity-memory-dim", type=int, default=8)

    parser.add_argument("--run-name", default="full")
    parser.add_argument("--output", default="/home/chen034/workspace/stwm/outputs/training/week2_minival/full/eval/mini_val_summary_last.json")
    parser.add_argument("--cases-output-dir", default="")
    parser.add_argument("--save-case-limit", type=int, default=20)
    parser.add_argument("--protocol-version", default=EVALUATOR_VERSION, choices=SUPPORTED_PROTOCOL_VERSIONS)
    parser.add_argument("--query-candidates", type=int, default=5)
    parser.add_argument("--query-hit-radius", type=float, default=0.08)
    parser.add_argument("--query-topk", type=int, default=1)
    parser.add_argument("--identity-hit-radius", type=float, default=0.03)
    parser.add_argument("--occlusion-recovery-window", type=int, default=3)
    parser.add_argument("--query-hard-negative-jitter", type=float, default=0.03)
    parser.add_argument("--identity-target-overlap-min", type=float, default=0.02)
    parser.add_argument("--identity-other-overlap-min", type=float, default=0.02)
    parser.add_argument("--occlusion-min-disappear-frames", type=int, default=2)
    parser.add_argument("--query-near-negative-count", type=int, default=1)
    parser.add_argument("--identity-consistency-window", type=int, default=3)
    parser.add_argument("--query-min-plausible-same-class", type=int, default=2)
    parser.add_argument("--occlusion-reconnect-distance", type=float, default=0.18)
    parser.add_argument("--occlusion-reconnect-target-overlap-min", type=float, default=0.01)
    return parser


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def select_protocol_samples(
    data_root: str,
    manifest: str,
    dataset_name: str,
    max_clips: int,
    obs_steps: int,
    pred_steps: int,
) -> list[Any]:
    dataset = STWMDataset(data_root, manifest=manifest, limit=None)
    needed = obs_steps + pred_steps
    selected = []
    dataset_filter = str(dataset_name).strip().lower()
    use_all_datasets = dataset_filter in {"all", "*", "mixed"}
    for sample in dataset.samples:
        if not use_all_datasets and sample.metadata.get("dataset", "").lower() != dataset_filter:
            continue
        if len(sample.frame_paths) < needed:
            continue
        mask_paths = sample.metadata.get("mask_paths", [])
        if not isinstance(mask_paths, list) or len(mask_paths) < needed:
            continue
        selected.append(sample)

    selected.sort(key=lambda item: item.clip_id)
    if max_clips > 0:
        selected = selected[:max_clips]
    return selected


def _safe_div(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator / denominator)


def _read_mask(path: str) -> np.ndarray:
    arr = np.array(Image.open(path))
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr > 0


def _read_mask_labels(path: str) -> np.ndarray:
    arr = np.array(Image.open(path))
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr.astype(np.int32)


def _torch_load_compat(path: str, device: torch.device) -> Any:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def _read_mask_ratio(path: str, target_label_id: int | None) -> float:
    p = Path(path)
    if not p.exists():
        return 0.0
    arr = np.array(Image.open(p))
    if arr.ndim == 3:
        arr = arr[..., 0]
    if target_label_id is None:
        return float((arr > 0).mean())
    tgt = arr == int(target_label_id)
    if tgt.any():
        return float(tgt.mean())
    return float((arr > 0).mean())


def _build_v4_2_features_for_sample(
    sample: Any,
    *,
    trace_adapter: TraceAdapter,
    semantic_adapter: SemanticAdapter,
    device: torch.device,
    ablation: AblationConfig,
) -> dict[str, Any]:
    trace_summary = trace_adapter.encode(sample.frame_paths, metadata=sample.metadata, clip_id=sample.clip_id)
    semantic_summary = semantic_adapter.encode(
        sample.text_labels,
        len(sample.frame_paths),
        metadata=sample.metadata,
        clip_id=sample.clip_id,
    )

    seq_len = int(min(trace_summary.centers.shape[0], semantic_summary.class_scores.shape[0]))
    if seq_len <= 1:
        raise RuntimeError(f"clip {sample.clip_id} has insufficient sequence length")

    centers = trace_summary.centers[:seq_len].to(device=device, dtype=torch.float32)
    velocities = trace_summary.velocities[:seq_len].to(device=device, dtype=torch.float32)
    visibility = trace_summary.visibility[:seq_len].to(device=device, dtype=torch.float32)
    trace_features = torch.cat([centers, velocities, visibility], dim=-1)

    if bool(ablation.disable_trajectory):
        trace_features[:, 0:4] = 0.0

    sem_text = semantic_summary.text_embeddings[:seq_len].mean(dim=1).to(device=device, dtype=torch.float32)
    sem_scores = semantic_summary.class_scores[:seq_len].mean(dim=1).to(device=device, dtype=torch.float32)
    semantic_features = torch.cat([sem_text, sem_scores], dim=-1)
    if bool(ablation.disable_semantics):
        semantic_features = torch.zeros_like(semantic_features)

    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    mask_paths = metadata.get("mask_paths") if isinstance(metadata.get("mask_paths"), list) else []
    target_label_id = _target_label_id(sample)

    mask_ratios: list[float] = []
    for i in range(seq_len):
        if i < len(mask_paths):
            mask_ratios.append(_read_mask_ratio(str(mask_paths[i]), target_label_id=target_label_id))
        else:
            mask_ratios.append(mask_ratios[-1] if mask_ratios else 0.0)
    mask_ratio_t = torch.tensor(mask_ratios, device=device, dtype=torch.float32)

    speed = torch.norm(velocities, dim=-1)
    semantic_conf = sem_scores.max(dim=-1).values
    prior_features = torch.stack(
        [
            mask_ratio_t.clamp(0.0, 1.0),
            visibility[:, 0].clamp(0.0, 1.0),
            speed.clamp(0.0, 1.0),
            semantic_conf.clamp(0.0, 1.0),
        ],
        dim=-1,
    )
    teacher_objectness = (0.6 * mask_ratio_t + 0.4 * visibility[:, 0]).clamp(0.0, 1.0)

    return {
        "trace_summary": trace_summary,
        "semantic_summary": semantic_summary,
        "trace_features": trace_features.unsqueeze(0),
        "semantic_features": semantic_features.unsqueeze(0),
        "prior_features": prior_features.unsqueeze(0),
        "teacher_objectness": teacher_objectness.unsqueeze(0),
        "seq_len": seq_len,
    }


def _target_label_id(sample: Any) -> int | None:
    raw = sample.metadata.get("target_label_id") if isinstance(sample.metadata, dict) else None
    try:
        return int(raw) if raw is not None else None
    except (TypeError, ValueError):
        return None


def _label_binary(mask_labels: np.ndarray, label_id: int | None) -> np.ndarray:
    if label_id is None:
        return mask_labels > 0
    target = mask_labels == int(label_id)
    if target.any():
        return target
    return np.zeros_like(mask_labels, dtype=bool)


def _estimate_radius_norm(mask_paths: list[str], obs_steps: int) -> float:
    areas = []
    for path in mask_paths[:obs_steps]:
        if not Path(path).exists():
            continue
        mask = _read_mask(path)
        areas.append(float(mask.mean()))
    if not areas:
        return 0.08
    mean_area = max(1e-5, float(np.mean(areas)))
    radius = float(np.sqrt(mean_area / np.pi))
    return float(np.clip(radius, 0.02, 0.30))


def _circle_mask(height: int, width: int, center_x: float, center_y: float, radius_norm: float) -> np.ndarray:
    cx = float(np.clip(center_x, 0.0, 1.0)) * max(1, width - 1)
    cy = float(np.clip(center_y, 0.0, 1.0)) * max(1, height - 1)
    radius_px = max(1.0, float(radius_norm) * float(min(height, width)))

    yy, xx = np.ogrid[:height, :width]
    dist2 = (xx - cx) ** 2 + (yy - cy) ** 2
    return dist2 <= radius_px * radius_px


def _choose_query_label(labels: list[str]) -> str:
    if not labels:
        return "object"
    for label in labels:
        text = str(label).strip().lower()
        if text and text not in {"scene", "background", "stuff"}:
            return str(label)
    return str(labels[0])


def _query_seed(label: str, clip_id: str) -> int:
    token = f"{label}|{clip_id}"
    return sum(ord(ch) for ch in token) % 10_000_019


def _build_query_candidates(
    gt_center: torch.Tensor,
    seed: int,
    num_candidates: int,
    *,
    mask_labels: np.ndarray | None,
    target_label_id: int | None,
    hard_negative_jitter: float,
    near_negative_count: int,
    min_plausible_same_class: int,
) -> tuple[list[torch.Tensor], int, int]:
    # Candidate 0 is always the true target location.
    candidates = [gt_center.clone()]
    same_class_candidate_count = 1
    rng = np.random.default_rng(seed)
    min_same_class = max(1, int(min_plausible_same_class))
    n = max(2, int(num_candidates), min_same_class)

    def _add_same_class_from_xy(x: float, y: float) -> None:
        nonlocal same_class_candidate_count
        candidates.append(torch.tensor([x, y], dtype=gt_center.dtype))
        same_class_candidate_count += 1

    if mask_labels is not None:
        h, w = mask_labels.shape[:2]
        target_ys: np.ndarray | None = None
        target_xs: np.ndarray | None = None
        if target_label_id is not None:
            target_mask = mask_labels == int(target_label_id)
            if target_mask.any():
                ys, xs = np.nonzero(target_mask)
                target_ys = ys
                target_xs = xs
                pick_idx = int(seed % max(1, len(xs)))
                hx = float(xs[pick_idx] / max(1, w - 1))
                hy = float(ys[pick_idx] / max(1, h - 1))
                hard_same_class = torch.tensor([hx, hy], dtype=gt_center.dtype)
                if float(torch.norm(hard_same_class - gt_center, p=2).item()) < float(max(0.005, hard_negative_jitter * 0.5)):
                    angle = float(seed % 360) * np.pi / 180.0
                    offset = torch.tensor(
                        [np.cos(angle) * hard_negative_jitter, np.sin(angle) * hard_negative_jitter],
                        dtype=gt_center.dtype,
                    )
                    hard_same_class = torch.clamp(gt_center + offset, 0.0, 1.0)
                candidates.append(hard_same_class)
                same_class_candidate_count += 1

        # Guarantee at least K plausible same-class candidates (including the true target).
        while same_class_candidate_count < min_same_class and len(candidates) < n:
            if target_xs is not None and target_ys is not None and len(target_xs) > 0:
                ridx = int(rng.integers(0, len(target_xs)))
                sx = float(target_xs[ridx] / max(1, w - 1))
                sy = float(target_ys[ridx] / max(1, h - 1))
                _add_same_class_from_xy(sx, sy)
            else:
                noise = torch.tensor(rng.normal(0.0, hard_negative_jitter, size=2), dtype=gt_center.dtype)
                candidates.append(torch.clamp(gt_center + noise, 0.0, 1.0))
                same_class_candidate_count += 1

        # Add near-target negatives to reduce query over-easiness while keeping minimal changes.
        for near_idx in range(max(0, int(near_negative_count))):
            if len(candidates) >= n:
                break
            angle = float((seed + 97 * (near_idx + 1)) % 360) * np.pi / 180.0
            scale = float(hard_negative_jitter) * (1.0 + 0.35 * near_idx)
            offset = torch.tensor(
                [np.cos(angle) * scale, np.sin(angle) * scale],
                dtype=gt_center.dtype,
            )
            candidates.append(torch.clamp(gt_center + offset, 0.0, 1.0))

        other_labels = [int(x) for x in np.unique(mask_labels) if int(x) != 0 and (target_label_id is None or int(x) != int(target_label_id))]
        if other_labels:
            rng.shuffle(other_labels)
        for label_id in other_labels:
            label_mask = mask_labels == int(label_id)
            if not label_mask.any():
                continue
            ys, xs = np.nonzero(label_mask)
            cx = float(xs.mean() / max(1, w - 1))
            cy = float(ys.mean() / max(1, h - 1))
            candidates.append(torch.tensor([cx, cy], dtype=gt_center.dtype))
            if len(candidates) >= n:
                break

    # Ensure at least two candidates overall.
    if len(candidates) < 2:
        angle = float(seed % 360) * np.pi / 180.0
        offset = torch.tensor(
            [np.cos(angle) * hard_negative_jitter, np.sin(angle) * hard_negative_jitter],
            dtype=gt_center.dtype,
        )
        candidates.append(torch.clamp(gt_center + offset, 0.0, 1.0))

    while len(candidates) < n:
        noise = torch.tensor(rng.normal(0.0, hard_negative_jitter, size=2), dtype=gt_center.dtype)
        candidates.append(torch.clamp(gt_center + noise, 0.0, 1.0))

    return candidates, 0, same_class_candidate_count


def _query_retrieval(
    pred_center: torch.Tensor,
    gt_center: torch.Tensor,
    query_label: str,
    clip_id: str,
    frame_idx: int,
    num_candidates: int,
    topk: int,
    *,
    mask_labels: np.ndarray | None,
    target_label_id: int | None,
    hard_negative_jitter: float,
    near_negative_count: int,
    min_plausible_same_class: int,
) -> tuple[float, float, int, int]:
    seed = _query_seed(query_label, f"{clip_id}:{frame_idx}")
    candidates, gt_index, same_class_candidate_count = _build_query_candidates(
        gt_center,
        seed=seed,
        num_candidates=num_candidates,
        mask_labels=mask_labels,
        target_label_id=target_label_id,
        hard_negative_jitter=float(hard_negative_jitter),
        near_negative_count=int(near_negative_count),
        min_plausible_same_class=int(min_plausible_same_class),
    )
    dists = [float(torch.norm(pred_center - cand, p=2).item()) for cand in candidates]
    ranks = np.argsort(np.asarray(dists))
    k = max(1, min(int(topk), len(ranks)))
    hit = float(gt_index in ranks[:k])
    return hit, float(dists[gt_index]), int(same_class_candidate_count), int(len(candidates))


def _visibility_stats(pred_prob: torch.Tensor, gt_binary: torch.Tensor) -> tuple[float, float]:
    pred_binary = (pred_prob >= 0.5).to(dtype=torch.int32)
    gt_binary = gt_binary.to(dtype=torch.int32)

    acc = float((pred_binary == gt_binary).float().mean().item())
    tp = int(((pred_binary == 1) & (gt_binary == 1)).sum().item())
    fp = int(((pred_binary == 1) & (gt_binary == 0)).sum().item())
    fn = int(((pred_binary == 0) & (gt_binary == 1)).sum().item())
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2.0 * precision * recall, precision + recall)
    return acc, f1


def evaluate_model(
    model: Any,
    samples: list[Any],
    *,
    model_family: str,
    dataset_name: str,
    ablation: AblationConfig,
    obs_steps: int,
    pred_steps: int,
    device: torch.device,
    run_name: str,
    trace_adapter: TraceAdapter,
    semantic_adapter: SemanticAdapter,
    tokenizer: SemanticTrajectoryTokenizer,
    case_output_dir: Path | None = None,
    save_case_limit: int = 20,
    protocol_version: str = "v1",
    query_candidates: int = 5,
    query_hit_radius: float = 0.08,
    query_topk: int = 1,
    identity_hit_radius: float = 0.03,
    occlusion_recovery_window: int = 3,
    query_hard_negative_jitter: float = 0.03,
    identity_target_overlap_min: float = 0.02,
    identity_other_overlap_min: float = 0.02,
    occlusion_min_disappear_frames: int = 2,
    query_near_negative_count: int = 1,
    identity_consistency_window: int = 3,
    query_min_plausible_same_class: int = 2,
    occlusion_reconnect_distance: float = 0.18,
    occlusion_reconnect_target_overlap_min: float = 0.01,
) -> dict[str, Any]:
    model.eval()
    requested_protocol_version = str(protocol_version)
    protocol_version = _canonical_protocol_version(requested_protocol_version)

    per_clip: list[dict[str, Any]] = []
    case_paths: list[str] = []

    with torch.inference_mode():
        for index, sample in enumerate(samples):
            token_time_attention: torch.Tensor | None = None
            query_token_logits: torch.Tensor | None = None

            if model_family == "stwm_v4_2":
                feature_result = _build_v4_2_features_for_sample(
                    sample,
                    trace_adapter=trace_adapter,
                    semantic_adapter=semantic_adapter,
                    device=device,
                    ablation=ablation,
                )
                outputs = model(
                    trace_features=feature_result["trace_features"],
                    semantic_features=feature_result["semantic_features"],
                    prior_features=feature_result["prior_features"],
                    teacher_objectness=feature_result["teacher_objectness"],
                    memory_state=None,
                    use_memory=not bool(ablation.disable_identity_memory),
                    update_memory=False,
                )

                seq_len = int(feature_result["seq_len"])
                pred_centers = torch.sigmoid(outputs["trajectory"][0, :seq_len]).detach().cpu()
                gt_centers = feature_result["trace_summary"].centers[:seq_len].detach().cpu()
                pred_visibility_prob = torch.sigmoid(outputs["visibility"][0, :seq_len, 0]).detach().cpu()
                gt_visibility = feature_result["trace_summary"].visibility[:seq_len, 0].detach().cpu()

                token_time_attention = outputs["token_time_attention"][0].detach().cpu()
                query_token_logits = outputs["query_token_logits"][0].detach().cpu()
                semantic_token_energy = outputs["semantic_logits"][0].detach().cpu().norm(dim=-1)
                frame_semantic_energy = torch.einsum("n,nt->t", semantic_token_energy, token_time_attention)
            else:
                token_result = build_tokens_for_sample(
                    sample,
                    trace_adapter,
                    semantic_adapter,
                    tokenizer,
                    ablation,
                    device,
                )
                tokens = token_result.tokens
                outputs = model(tokens)

                pred_centers = torch.sigmoid(outputs["trajectory"][0]).detach().cpu()
                gt_centers = token_result.trace_summary.centers.detach().cpu()
                pred_visibility_prob = torch.sigmoid(outputs["visibility"][0, :, 0]).detach().cpu()
                gt_visibility = token_result.trace_summary.visibility[:, 0].detach().cpu()
                frame_semantic_energy = outputs["semantic"][0].detach().cpu().norm(dim=-1)

            seq_len = int(min(pred_centers.shape[0], gt_centers.shape[0]))
            start = int(min(obs_steps, seq_len))
            end = int(min(seq_len, obs_steps + pred_steps))
            if end <= start:
                continue

            future_pred = pred_centers[start:end]
            future_gt = gt_centers[start:end]
            frame_errors = (future_pred - future_gt).abs().mean(dim=-1)
            trajectory_error = float(frame_errors.mean().item())

            future_pred_visibility = pred_visibility_prob[start:end]
            future_gt_visibility = (gt_visibility[start:end] >= 0.5).float()
            visibility_accuracy, visibility_f1 = _visibility_stats(future_pred_visibility, future_gt_visibility)

            mask_paths = sample.metadata.get("mask_paths", []) if isinstance(sample.metadata.get("mask_paths"), list) else []
            target_label_id = _target_label_id(sample)
            radius_norm = _estimate_radius_norm(mask_paths, obs_steps=obs_steps)
            use_v2_family = protocol_version in {"v2", "v2_1", "v2_2", "v2_3"}
            mask_ious: list[float] = []
            point_on_target_flags: list[float] = []
            point_on_other_foreground_flags: list[float] = []
            identity_switch_flags: list[float] = []
            identity_target_overlap_values: list[float] = []
            identity_other_overlap_values: list[float] = []
            target_area_ratio_by_frame: list[float] = []
            for local_i, frame_idx in enumerate(range(start, end)):
                if frame_idx >= len(mask_paths):
                    continue
                mask_path = Path(mask_paths[frame_idx])
                if not mask_path.exists():
                    continue
                gt_labels = _read_mask_labels(str(mask_path))
                gt_mask = _label_binary(gt_labels, target_label_id)
                pred_mask = _circle_mask(
                    gt_mask.shape[0],
                    gt_mask.shape[1],
                    center_x=float(future_pred[local_i, 0].item()),
                    center_y=float(future_pred[local_i, 1].item()),
                    radius_norm=radius_norm,
                )
                inter = float(np.logical_and(gt_mask, pred_mask).sum())
                union = float(np.logical_or(gt_mask, pred_mask).sum())
                mask_ious.append(_safe_div(inter, union))

                px = int(np.clip(round(float(future_pred[local_i, 0].item()) * (gt_labels.shape[1] - 1)), 0, gt_labels.shape[1] - 1))
                py = int(np.clip(round(float(future_pred[local_i, 1].item()) * (gt_labels.shape[0] - 1)), 0, gt_labels.shape[0] - 1))
                sampled_label = int(gt_labels[py, px])
                if protocol_version in {"v2_1", "v2_2", "v2_3"}:
                    hit_disk = _circle_mask(
                        gt_mask.shape[0],
                        gt_mask.shape[1],
                        center_x=float(future_pred[local_i, 0].item()),
                        center_y=float(future_pred[local_i, 1].item()),
                        radius_norm=float(identity_hit_radius),
                    )
                    disk_area = float(max(1, int(hit_disk.sum())))
                    target_overlap = float(np.logical_and(gt_mask, hit_disk).sum()) / disk_area
                    if target_label_id is not None:
                        other_fg = np.logical_and(gt_labels > 0, gt_labels != int(target_label_id))
                        other_overlap = float(np.logical_and(other_fg, hit_disk).sum()) / disk_area
                    else:
                        other_overlap = 0.0

                    if protocol_version in {"v2_2", "v2_3"}:
                        target_hit = float(target_overlap >= float(identity_target_overlap_min))
                        other_fg_hit = float(other_overlap >= float(identity_other_overlap_min))
                        switch_flag = float((other_overlap >= float(identity_other_overlap_min)) and (target_overlap < float(identity_target_overlap_min)))
                    else:
                        target_hit = float(target_overlap > 0.0)
                        other_fg_hit = float(other_overlap > 0.0)
                        switch_flag = float(other_fg_hit)

                    identity_target_overlap_values.append(float(target_overlap))
                    identity_other_overlap_values.append(float(other_overlap))
                    identity_switch_flags.append(float(switch_flag))
                else:
                    target_hit = float(sampled_label == int(target_label_id)) if target_label_id is not None else float(sampled_label > 0)
                    other_fg_hit = float((sampled_label > 0) and (target_label_id is not None) and (sampled_label != int(target_label_id)))
                    identity_switch_flags.append(float(other_fg_hit))
                point_on_target_flags.append(target_hit)
                point_on_other_foreground_flags.append(other_fg_hit)
                target_area_ratio_by_frame.append(float(gt_mask.mean()))
            future_mask_iou = float(np.mean(mask_ious)) if mask_ious else 0.0

            if use_v2_family:
                if protocol_version == "v2_3" and point_on_target_flags:
                    win = max(1, int(identity_consistency_window))
                    switch_series: list[float] = []
                    consistency_series: list[float] = []
                    for i in range(len(point_on_target_flags)):
                        j = min(len(point_on_target_flags), i + win)
                        targ_avg = float(np.mean(point_on_target_flags[i:j]))
                        other_avg = float(np.mean(point_on_other_foreground_flags[i:j]))
                        switch_series.append(float((other_avg >= 0.5) and (targ_avg < 0.5)))
                        consistency_series.append(float(targ_avg >= 0.5))
                    identity_switch_rate = float(np.mean(switch_series)) if switch_series else 0.0
                    identity_consistency = float(np.mean(consistency_series)) if consistency_series else 1.0
                elif identity_switch_flags:
                    identity_switch_rate = float(np.mean(identity_switch_flags))
                    identity_consistency = float(np.mean(point_on_target_flags)) if point_on_target_flags else float(1.0 - identity_switch_rate)
                else:
                    identity_switch_rate = 0.0
                    identity_consistency = 1.0
            else:
                if end - start > 1:
                    pred_delta = torch.norm(pred_centers[start + 1:end] - pred_centers[start:end - 1], dim=-1)
                    gt_delta = torch.norm(gt_centers[start + 1:end] - gt_centers[start:end - 1], dim=-1)
                    id_switch_flags = ((pred_delta > 0.25) & (gt_delta < 0.10)).float()
                    identity_switch_rate = float(id_switch_flags.mean().item())
                    identity_consistency = float(1.0 - identity_switch_rate)
                else:
                    identity_switch_rate = 0.0
                    identity_consistency = 1.0

            recovery_indices: list[int] = []
            occlusion_recovery_acc = 0.0

            future_gt_vis_int = (gt_visibility[start:end] >= 0.5).to(dtype=torch.int32)
            for i in range(1, len(future_gt_vis_int)):
                if int(future_gt_vis_int[i - 1].item()) == 0 and int(future_gt_vis_int[i].item()) == 1:
                    recovery_indices.append(i)

            if use_v2_family and target_area_ratio_by_frame:
                area_flags = [1 if ratio > 1e-4 else 0 for ratio in target_area_ratio_by_frame]
                min_gap = max(1, int(occlusion_min_disappear_frames))

                area_recovery_indices: list[int] = []
                for i in range(1, len(area_flags)):
                    if area_flags[i] != 1:
                        continue
                    j = i - 1
                    gap = 0
                    while j >= 0 and area_flags[j] == 0:
                        gap += 1
                        j -= 1
                    had_visible_before = j >= 0 and area_flags[j] == 1
                    if had_visible_before and gap >= min_gap:
                        area_recovery_indices.append(i)

                if protocol_version in {"v2_2", "v2_3"} and area_recovery_indices:
                    recovery_indices = area_recovery_indices

                if recovery_indices:
                    window = max(1, int(occlusion_recovery_window))
                    recovered_ok = []
                    for i in recovery_indices:
                        start_i = int(i)
                        end_i = min(len(point_on_target_flags), start_i + window)
                        if protocol_version == "v2_3":
                            reconnect_flags = []
                            for j in range(start_i, end_i):
                                target_overlap = float(identity_target_overlap_values[j]) if j < len(identity_target_overlap_values) else 0.0
                                dist_ok = bool(j < len(frame_errors) and float(frame_errors[j].item()) <= float(occlusion_reconnect_distance))
                                overlap_ok = target_overlap >= float(occlusion_reconnect_target_overlap_min)
                                reconnect_flags.append(float(dist_ok or overlap_ok))
                            recovered_ok.append(float(any(float(v) >= 0.5 for v in reconnect_flags)))
                        else:
                            win = point_on_target_flags[start_i:end_i]
                            recovered_ok.append(float(any(float(v) >= 0.5 for v in win)))
                    if recovered_ok:
                        occlusion_recovery_acc = float(np.mean(recovered_ok))
            elif recovery_indices:
                recovered_ok = [float(frame_errors[i].item() < 0.20) for i in recovery_indices if i < len(frame_errors)]
                if recovered_ok:
                    occlusion_recovery_acc = float(np.mean(recovered_ok))

            query_label = _choose_query_label(sample.text_labels)
            visible_indices = [i for i in range(end - start) if int(future_gt_visibility[i].item()) == 1]
            query_top1_acc = 0.0
            query_hit_rate = 0.0
            query_same_class_candidates = 0
            query_total_candidates = 0
            if use_v2_family:
                semantic_energy = frame_semantic_energy[start:end]
                if token_time_attention is not None and query_token_logits is not None and token_time_attention.numel() > 0:
                    query_token_index = int(torch.argmax(query_token_logits).item())
                    query_frame_scores = token_time_attention[query_token_index, start:end]
                    if query_frame_scores.numel() > 0:
                        query_local_idx = int(torch.argmax(query_frame_scores).item())
                    else:
                        query_local_idx = 0
                elif len(semantic_energy) > 0:
                    query_local_idx = int(torch.argmax(semantic_energy).item())
                else:
                    query_local_idx = 0
                query_local_idx = int(np.clip(query_local_idx, 0, max(0, end - start - 1)))
                q_err = float(frame_errors[query_local_idx].item()) if len(frame_errors) > 0 else trajectory_error

                query_global_idx = start + query_local_idx
                if query_global_idx < len(mask_paths):
                    query_mask_labels = None
                    query_mask_path = Path(mask_paths[query_global_idx])
                    if query_mask_path.exists():
                        query_mask_labels = _read_mask_labels(str(query_mask_path))

                    # Use retrieval against distractor candidates to avoid being identical to trajectory mean.
                    hit, _, query_same_class_candidates, query_total_candidates = _query_retrieval(
                        pred_center=future_pred[query_local_idx],
                        gt_center=future_gt[query_local_idx],
                        query_label=query_label,
                        clip_id=sample.clip_id,
                        frame_idx=query_global_idx,
                        num_candidates=query_candidates,
                        topk=query_topk,
                        mask_labels=query_mask_labels,
                        target_label_id=target_label_id,
                        hard_negative_jitter=float(query_hard_negative_jitter),
                        near_negative_count=int(query_near_negative_count),
                        min_plausible_same_class=int(query_min_plausible_same_class),
                    )
                    query_top1_acc = hit

                    # Radius-based localization hit on selected query frame.
                    query_hit_rate = float(torch.norm(future_pred[query_local_idx] - future_gt[query_local_idx], p=2).item() <= float(query_hit_radius))
            else:
                if visible_indices:
                    q_err = float(np.mean([float(frame_errors[i].item()) for i in visible_indices]))
                else:
                    q_err = trajectory_error
                semantic_energy = frame_semantic_energy[start:end]
            case_payload = {
                "run_name": run_name,
                "clip_id": sample.clip_id,
                "query_label": query_label,
                "obs_steps": int(obs_steps),
                "pred_steps": int(pred_steps),
                "future_range": [int(start), int(end)],
                "frame_paths": sample.frame_paths,
                "mask_paths": mask_paths,
                "pred_centers": pred_centers.tolist(),
                "gt_centers": gt_centers.tolist(),
                "pred_visibility_prob": pred_visibility_prob.tolist(),
                "gt_visibility": gt_visibility.tolist(),
                "radius_norm": float(radius_norm),
                "trajectory_error_by_frame": [float(x.item()) for x in frame_errors],
                "mask_iou_by_frame": [float(x) for x in mask_ious],
                "semantic_energy_by_frame": [float(x.item()) for x in semantic_energy],
                "has_occlusion_recovery_event": bool(len(recovery_indices) > 0),
                "occlusion_recovery_indices_local": [int(x) for x in recovery_indices],
                "target_label_id": int(target_label_id) if target_label_id is not None else None,
                "point_on_target_rate": float(np.mean(point_on_target_flags)) if point_on_target_flags else 0.0,
                "point_on_other_foreground_rate": float(np.mean(point_on_other_foreground_flags)) if point_on_other_foreground_flags else 0.0,
                "identity_target_overlap_mean": float(np.mean(identity_target_overlap_values)) if identity_target_overlap_values else 0.0,
                "identity_other_overlap_mean": float(np.mean(identity_other_overlap_values)) if identity_other_overlap_values else 0.0,
                "query_top1_acc": float(query_top1_acc),
                "query_hit_rate": float(query_hit_rate),
                "query_same_class_candidates": int(query_same_class_candidates),
                "query_total_candidates": int(query_total_candidates),
            }

            case_file = ""
            if case_output_dir is not None and (save_case_limit < 0 or len(case_paths) < save_case_limit):
                case_output_dir.mkdir(parents=True, exist_ok=True)
                case_path = case_output_dir / f"{sample.clip_id}.json"
                case_path.write_text(json.dumps(case_payload, indent=2))
                case_file = str(case_path)
                case_paths.append(case_file)

            per_clip.append(
                {
                    "clip_id": sample.clip_id,
                    "future_mask_iou": future_mask_iou,
                    "future_trajectory_l1": trajectory_error,
                    "visibility_accuracy": visibility_accuracy,
                    "visibility_f1": visibility_f1,
                    "identity_consistency": identity_consistency,
                    "identity_switch_rate": identity_switch_rate,
                    "occlusion_recovery_acc": occlusion_recovery_acc,
                    "query_localization_error": q_err,
                    "query_top1_acc": float(query_top1_acc),
                    "query_hit_rate": float(query_hit_rate),
                    "query_same_class_candidates": int(query_same_class_candidates),
                    "query_total_candidates": int(query_total_candidates),
                    "identity_target_overlap_mean": float(np.mean(identity_target_overlap_values)) if identity_target_overlap_values else 0.0,
                    "identity_other_overlap_mean": float(np.mean(identity_other_overlap_values)) if identity_other_overlap_values else 0.0,
                    "case_file": case_file,
                }
            )

    def mean_metric(key: str) -> float:
        if not per_clip:
            return 0.0
        return float(np.mean([float(item[key]) for item in per_clip]))

    summary = {
        "evaluator_version": EVALUATOR_VERSION,
        "run_name": run_name,
        "num_clips": len(per_clip),
        "protocol": {
            "protocol_version": str(protocol_version),
            "requested_protocol_version": str(requested_protocol_version),
            "evaluator_version": EVALUATOR_VERSION,
            "obs_steps": int(obs_steps),
            "pred_steps": int(pred_steps),
            "dataset": str(dataset_name),
            "fixed_seed": True,
            "query_candidates": int(query_candidates),
            "query_topk": int(query_topk),
            "query_hit_radius": float(query_hit_radius),
            "identity_hit_radius": float(identity_hit_radius),
            "occlusion_recovery_window": int(occlusion_recovery_window),
            "query_hard_negative_jitter": float(query_hard_negative_jitter),
            "identity_target_overlap_min": float(identity_target_overlap_min),
            "identity_other_overlap_min": float(identity_other_overlap_min),
            "occlusion_min_disappear_frames": int(occlusion_min_disappear_frames),
            "query_near_negative_count": int(query_near_negative_count),
            "identity_consistency_window": int(identity_consistency_window),
            "query_min_plausible_same_class": int(query_min_plausible_same_class),
            "occlusion_reconnect_distance": float(occlusion_reconnect_distance),
            "occlusion_reconnect_target_overlap_min": float(occlusion_reconnect_target_overlap_min),
            "metrics": [
                "future_mask_iou",
                "future_trajectory_l1",
                "visibility_accuracy",
                "visibility_f1",
                "identity_consistency",
                "identity_switch_rate",
                "occlusion_recovery_acc",
                "query_localization_error",
                "query_top1_acc",
                "query_hit_rate",
            ],
            "stable_comparable_metrics": list(STABLE_COMPARABLE_METRICS),
        },
        "ablation": ablation.to_dict(),
        "metrics": {
            "future_mask_iou": mean_metric("future_mask_iou"),
            "future_trajectory_l1": mean_metric("future_trajectory_l1"),
            "visibility_accuracy": mean_metric("visibility_accuracy"),
            "visibility_f1": mean_metric("visibility_f1"),
            "identity_consistency": mean_metric("identity_consistency"),
            "identity_switch_rate": mean_metric("identity_switch_rate"),
            "occlusion_recovery_acc": mean_metric("occlusion_recovery_acc"),
            "query_localization_error": mean_metric("query_localization_error"),
            "query_top1_acc": mean_metric("query_top1_acc"),
            "query_hit_rate": mean_metric("query_hit_rate"),
        },
        "per_clip": per_clip,
        "case_files": case_paths,
    }
    return summary


def main() -> None:
    args = build_parser().parse_args()
    _set_seed(args.seed)

    device = resolve_device(args.device)
    ablation = ablation_from_args(args)

    samples = select_protocol_samples(
        data_root=args.data_root,
        manifest=args.manifest,
        dataset_name=args.dataset,
        max_clips=args.max_clips,
        obs_steps=args.obs_steps,
        pred_steps=args.pred_steps,
    )
    if not samples:
        raise RuntimeError("No eligible samples found for mini-val protocol")

    trace_adapter = TraceAdapter()
    semantic_adapter = SemanticAdapter()
    tokenizer = SemanticTrajectoryTokenizer()

    model_family = "stwm_1b"

    if args.checkpoint:
        checkpoint = _torch_load_compat(args.checkpoint, device=device)
        model_state = checkpoint.get("model_state") if isinstance(checkpoint, dict) else checkpoint
        model_config_payload = checkpoint.get("model_config") if isinstance(checkpoint, dict) else None

        if isinstance(model_config_payload, dict) and "trace_dim" in model_config_payload:
            model_family = "stwm_v4_2"
            model_config = STWMV42Config(**model_config_payload)
            model = STWMV42(model_config).to(device)
            model.load_state_dict(model_state, strict=True)
        else:
            if isinstance(model_config_payload, dict):
                model_config = STWMConfig(**model_config_payload)
            else:
                first_tokens = build_tokens_for_sample(
                    samples[0],
                    trace_adapter,
                    semantic_adapter,
                    tokenizer,
                    ablation,
                    device,
                )
                model_config = load_model_config(
                    preset=args.model_preset,
                    input_dim=first_tokens.tokens.shape[-1],
                    preset_path=args.preset_file,
                )
            model = STWM1B(model_config).to(device)
            model.load_state_dict(model_state, strict=True)
    else:
        if "v4_2" in str(args.model_preset):
            model_family = "stwm_v4_2"
            warm = _build_v4_2_features_for_sample(
                samples[0],
                trace_adapter=trace_adapter,
                semantic_adapter=semantic_adapter,
                device=device,
                ablation=ablation,
            )
            model_config = load_model_config_v4_2(
                preset=args.model_preset,
                trace_dim=int(warm["trace_features"].shape[-1]),
                semantic_dim=int(warm["semantic_features"].shape[-1]),
                prior_dim=int(warm["prior_features"].shape[-1]),
                preset_path=args.preset_file,
            )
            model = STWMV42(model_config).to(device)
        else:
            first_tokens = build_tokens_for_sample(
                samples[0],
                trace_adapter,
                semantic_adapter,
                tokenizer,
                ablation,
                device,
            )
            model_config = load_model_config(
                preset=args.model_preset,
                input_dim=first_tokens.tokens.shape[-1],
                preset_path=args.preset_file,
            )
            model = STWM1B(model_config).to(device)

    case_dir = Path(args.cases_output_dir) if args.cases_output_dir else None
    summary = evaluate_model(
        model,
        samples,
        model_family=model_family,
        dataset_name=str(args.dataset),
        ablation=ablation,
        obs_steps=args.obs_steps,
        pred_steps=args.pred_steps,
        device=device,
        run_name=args.run_name,
        trace_adapter=trace_adapter,
        semantic_adapter=semantic_adapter,
        tokenizer=tokenizer,
        case_output_dir=case_dir,
        save_case_limit=args.save_case_limit,
        protocol_version=args.protocol_version,
        query_candidates=int(args.query_candidates),
        query_hit_radius=float(args.query_hit_radius),
        query_topk=int(args.query_topk),
        identity_hit_radius=float(args.identity_hit_radius),
        occlusion_recovery_window=int(args.occlusion_recovery_window),
        query_hard_negative_jitter=float(args.query_hard_negative_jitter),
        identity_target_overlap_min=float(args.identity_target_overlap_min),
        identity_other_overlap_min=float(args.identity_other_overlap_min),
        occlusion_min_disappear_frames=int(args.occlusion_min_disappear_frames),
        query_near_negative_count=int(args.query_near_negative_count),
        identity_consistency_window=int(args.identity_consistency_window),
        query_min_plausible_same_class=int(args.query_min_plausible_same_class),
        occlusion_reconnect_distance=float(args.occlusion_reconnect_distance),
        occlusion_reconnect_target_overlap_min=float(args.occlusion_reconnect_target_overlap_min),
    )
    if model_family == "stwm_v4_2":
        summary["model_config"] = {
            "family": model_family,
            "trace_dim": int(model.config.trace_dim),
            "semantic_dim": int(model.config.semantic_dim),
            "prior_dim": int(model.config.prior_dim),
            "hidden_size": int(model.config.hidden_size),
            "seq_num_layers": int(model.config.seq_num_layers),
            "token_num_layers": int(model.config.token_num_layers),
            "num_heads": int(model.config.num_heads),
        }
    else:
        summary["model_config"] = {
            "family": model_family,
            "input_dim": int(model.config.input_dim),
            "hidden_size": int(model.config.hidden_size),
            "num_layers": int(model.config.num_layers),
            "num_heads": int(model.config.num_heads),
            "semantic_dim": int(model.config.semantic_dim),
        }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
