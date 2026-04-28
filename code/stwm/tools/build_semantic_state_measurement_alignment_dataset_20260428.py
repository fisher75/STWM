#!/usr/bin/env python3
from __future__ import annotations

from argparse import ArgumentParser
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import hashlib
import json
import math
import os
import sys

import torch


def _apply_process_title_normalization() -> None:
    mode = str(os.environ.get("STWM_PROC_TITLE_MODE", "generic")).strip().lower()
    if mode == "off":
        return
    title = str(os.environ.get("STWM_PROC_TITLE", "python")).strip() or "python"
    try:
        import setproctitle  # type: ignore

        setproctitle.setproctitle(title)
    except Exception:
        pass


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def write_doc(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# STWM Semantic-State Measurement Alignment Dataset V1",
        "",
        f"- record_count: `{payload.get('record_count')}`",
        f"- item_count: `{payload.get('item_count')}`",
        f"- dev_record_count: `{payload.get('split_counts', {}).get('dev')}`",
        f"- heldout_record_count: `{payload.get('split_counts', {}).get('heldout')}`",
        f"- semantic_embedding_dim: `{payload.get('feature_stats', {}).get('semantic_embedding_dim')}`",
        f"- identity_embedding_dim: `{payload.get('feature_stats', {}).get('identity_embedding_dim')}`",
        f"- candidate_crop_feature_dim: `{payload.get('feature_stats', {}).get('candidate_crop_feature_dim')}`",
        f"- future_candidate_used_as_input: `{payload.get('future_candidate_used_as_input')}`",
        f"- candidate_feature_used_for_rollout: `{payload.get('candidate_feature_used_for_rollout')}`",
        f"- full_model_forward_executed: `{payload.get('full_model_forward_executed')}`",
        f"- full_free_rollout_executed: `{payload.get('full_free_rollout_executed')}`",
        f"- stage2_val_fallback_used: `{payload.get('stage2_val_fallback_used')}`",
        f"- old_association_report_used: `{payload.get('old_association_report_used')}`",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n")


def _round_list(tensor: torch.Tensor, ndigits: int = 6) -> list[float]:
    flat = tensor.detach().float().cpu().view(-1).tolist()
    return [round(float(x), ndigits) for x in flat]


def _as_tensor(values: Any) -> torch.Tensor:
    if not isinstance(values, list) or not values:
        return torch.zeros((0,), dtype=torch.float32)
    return torch.tensor([float(x) for x in values], dtype=torch.float32)


def _cosine01(a: torch.Tensor, b: torch.Tensor) -> float:
    if a.numel() == 0 or b.numel() == 0:
        return 0.5
    dim = min(int(a.numel()), int(b.numel()))
    aa = a.view(-1)[:dim].float()
    bb = b.view(-1)[:dim].float()
    denom = aa.norm() * bb.norm()
    if float(denom.item()) <= 1e-12:
        return 0.5
    return float(((torch.dot(aa, bb) / denom).clamp(-1.0, 1.0) * 0.5 + 0.5).item())


def _cache_feature(cache: dict[str, Any], item_id: str, candidate_id: str) -> tuple[torch.Tensor, torch.Tensor, float | None, dict[str, Any]]:
    item_cache = (cache.get("features_by_item") or {}).get(str(item_id), {}) if isinstance(cache, dict) else {}
    observed = item_cache.get("observed_target_frozen_feature") if isinstance(item_cache, dict) else []
    candidate = ((item_cache.get("candidates") or {}).get(str(candidate_id), {}) if isinstance(item_cache, dict) else {})
    cand_vec = candidate.get("candidate_frozen_feature") if isinstance(candidate, dict) else []
    sim = candidate.get("target_candidate_frozen_similarity") if isinstance(candidate, dict) else None
    return _as_tensor(cand_vec), _as_tensor(observed), (float(sim) if isinstance(sim, (int, float)) else None), candidate if isinstance(candidate, dict) else {}


def _candidate_center(candidate: dict[str, Any], image_size: Any) -> tuple[float, float] | None:
    bbox = candidate.get("bbox") if isinstance(candidate, dict) else None
    if not isinstance(bbox, list) or len(bbox) != 4:
        return None
    width = height = None
    if isinstance(image_size, dict):
        width = image_size.get("width")
        height = image_size.get("height")
    elif isinstance(image_size, list) and len(image_size) >= 2:
        width, height = image_size[:2]
    try:
        w = max(float(width), 1.0)
        h = max(float(height), 1.0)
        x1, y1, x2, y2 = [float(x) for x in bbox]
    except Exception:
        return None
    return ((x1 + x2) * 0.5 / w, (y1 + y2) * 0.5 / h)


def _split_for_item(item_id: str) -> str:
    value = int(hashlib.sha256(str(item_id).encode("utf-8")).hexdigest(), 16) % 2
    return "dev" if value == 0 else "heldout"


def _mean(values: list[float]) -> float | None:
    return float(sum(values) / len(values)) if values else None


def _variance(values: list[float]) -> float | None:
    if not values:
        return None
    m = sum(values) / len(values)
    return float(sum((x - m) ** 2 for x in values) / len(values))


def build_dataset(
    *,
    repo_root: Path,
    checkpoint: Path,
    candidate_manifest: Path,
    output: Path,
    doc: Path,
    max_items: int,
    device_name: str,
    mode: str,
    candidate_measurement_feature_mode: str,
    frozen_measurement_cache: Path | None = None,
) -> dict[str, Any]:
    _apply_process_title_normalization()
    code_dir = repo_root / "code"
    if str(code_dir) not in sys.path:
        sys.path.insert(0, str(code_dir))

    from stwm.tools.export_future_semantic_trace_state_20260427 import _build_full_model_from_checkpoint
    from stwm.tools.external_hardcase_query_batch_builder_20260428 import (
        build_item_candidate_measurement_cache,
        build_query_batch_from_item,
        group_candidate_records,
    )

    device = torch.device(device_name if device_name != "cuda" or torch.cuda.is_available() else "cpu")
    manifest = load_json(candidate_manifest)
    frozen_cache: dict[str, Any] = {}
    if str(candidate_measurement_feature_mode) == "frozen_measurement_v7":
        if frozen_measurement_cache is None:
            raise ValueError("frozen_measurement_v7 requires --frozen-measurement-cache")
        frozen_cache = load_json(frozen_measurement_cache)
    records = manifest.get("records") if isinstance(manifest, dict) else []
    records = [x for x in records if isinstance(x, dict)]
    grouped_items = group_candidate_records(records)
    selected_items = grouped_items[: int(max_items)]

    full = _build_full_model_from_checkpoint(
        repo_root,
        checkpoint,
        device,
        int(max_items),
        enable_semantic_state_feedback=False,
        semantic_state_feedback_alpha=0.0,
    )
    trainer = full["trainer"]
    args = full["args"]
    full_model_forward_executed = True
    full_free_rollout_executed = str(mode) == "full_model_free_rollout"
    rows: list[dict[str, Any]] = []
    item_failures: list[dict[str, Any]] = []
    dim_stats: dict[str, int | None] = {
        "semantic_embedding_dim": None,
        "identity_embedding_dim": None,
        "candidate_crop_feature_dim": None,
        "observed_target_crop_feature_dim": None,
    }
    norm_stats: dict[str, list[float]] = defaultdict(list)
    split_counts: Counter[str] = Counter()
    label_counts: Counter[str] = Counter()
    subset_counts: dict[str, Counter[str]] = defaultdict(Counter)

    with torch.no_grad():
        for item in selected_items:
            item_id = str(item.get("item_id"))
            try:
                raw_batch, builder_meta = build_query_batch_from_item(
                    item,
                    obs_len=int(args.obs_len),
                    fut_len=int(args.fut_len),
                    crop_size=int(args.semantic_crop_size),
                    semantic_temporal_window=int(args.local_temporal_window),
                )
                batch = trainer._to_device(raw_batch, device=device, non_blocking=False)
                if str(mode) == "full_model_teacher_forced":
                    out = trainer._teacher_forced_predict(
                        stage1_model=full["stage1_model"],
                        semantic_encoder=full["semantic_encoder"],
                        semantic_fusion=full["semantic_fusion"],
                        readout_head=full["readout_head"],
                        future_semantic_state_head=full["future_semantic_state_head"],
                        semantic_state_feedback_adapter=full.get("semantic_state_feedback_adapter"),
                        semantic_state_feedback_enabled=False,
                        semantic_state_feedback_alpha=0.0,
                        semantic_state_feedback_stopgrad_state=True,
                        semantic_state_feedback_mode="readout_only",
                        structure_mode=str(full["structure_mode"]),
                        trace_unit_tokenizer=full["trace_unit_tokenizer"],
                        trace_unit_factorized_state=full["trace_unit_factorized_state"],
                        trace_unit_handshake=full["trace_unit_handshake"],
                        trace_unit_broadcast=full["trace_unit_broadcast"],
                        trace_unit_disable_instance_path=bool(args.trace_unit_disable_instance_path),
                        batch=batch,
                        obs_len=int(args.obs_len),
                        semantic_source_mainline=str(args.semantic_source_mainline),
                    )
                elif str(mode) == "full_model_free_rollout":
                    out = trainer._free_rollout_predict(
                        stage1_model=full["stage1_model"],
                        semantic_encoder=full["semantic_encoder"],
                        semantic_fusion=full["semantic_fusion"],
                        readout_head=full["readout_head"],
                        future_semantic_state_head=full["future_semantic_state_head"],
                        semantic_state_feedback_adapter=full.get("semantic_state_feedback_adapter"),
                        semantic_state_feedback_enabled=False,
                        semantic_state_feedback_alpha=0.0,
                        semantic_state_feedback_stopgrad_state=True,
                        semantic_state_feedback_mode="readout_only",
                        structure_mode=str(full["structure_mode"]),
                        trace_unit_tokenizer=full["trace_unit_tokenizer"],
                        trace_unit_factorized_state=full["trace_unit_factorized_state"],
                        trace_unit_handshake=full["trace_unit_handshake"],
                        trace_unit_broadcast=full["trace_unit_broadcast"],
                        trace_unit_disable_instance_path=bool(args.trace_unit_disable_instance_path),
                        batch=batch,
                        obs_len=int(args.obs_len),
                        fut_len=int(args.fut_len),
                        semantic_source_mainline=str(args.semantic_source_mainline),
                    )
                else:
                    raise ValueError(f"unsupported mode: {mode}")
                state = out.get("future_semantic_trace_state")
                if state is None:
                    raise RuntimeError("missing future_semantic_trace_state")
                validation = state.validate(strict=False)
                if not bool(validation.get("valid")):
                    raise RuntimeError("invalid future semantic trace state: " + "; ".join(validation.get("errors", [])))
                measurement_cache = build_item_candidate_measurement_cache(
                    item,
                    semantic_encoder=full["semantic_encoder"],
                    device=device,
                    crop_size=int(args.semantic_crop_size),
                    feature_mode="crop_encoder_feature"
                    if str(candidate_measurement_feature_mode) == "frozen_measurement_v7"
                    else str(candidate_measurement_feature_mode),
                )
                pred_coord = state.future_trace_coord[0, -1, 0, :2].detach().float().cpu()
                pred_sem = state.future_semantic_embedding[0, :, 0].detach().mean(dim=0).float().cpu()
                pred_id = state.future_identity_embedding[0, :, 0].detach().mean(dim=0).float().cpu()
                if dim_stats["semantic_embedding_dim"] is None:
                    dim_stats["semantic_embedding_dim"] = int(pred_sem.numel())
                    dim_stats["identity_embedding_dim"] = int(pred_id.numel())
                visibility_prob = float(torch.sigmoid(state.future_visibility_logit[0, :, 0]).mean().detach().cpu().item())
                if getattr(state, "future_reappearance_event_logit", None) is not None:
                    reappearance_prob = float(torch.sigmoid(state.future_reappearance_event_logit[0, 0]).detach().cpu().item())
                elif getattr(state, "future_reappearance_logit", None) is not None:
                    reappearance_prob = float(torch.sigmoid(state.future_reappearance_logit[0, :, 0]).mean().detach().cpu().item())
                else:
                    reappearance_prob = visibility_prob
                tags = item.get("subset_tags") if isinstance(item.get("subset_tags"), dict) else {}
                split = _split_for_item(item_id)
                candidates = item.get("future_candidates") if isinstance(item.get("future_candidates"), list) else []
                labels = [int(x) for x in item.get("candidate_labels", [])] if isinstance(item.get("candidate_labels"), list) else []
                for idx, cand in enumerate(candidates):
                    if not isinstance(cand, dict):
                        continue
                    cid = str(cand.get("candidate_id"))
                    feature = (measurement_cache.get("candidate_features", {}) or {}).get(cid, {})
                    cand_feat = _as_tensor(feature.get("feature_vector"))
                    obs_feat = _as_tensor(feature.get("observed_target_feature_vector"))
                    frozen_cand_feat = torch.zeros((0,), dtype=torch.float32)
                    frozen_obs_feat = torch.zeros((0,), dtype=torch.float32)
                    frozen_similarity = None
                    frozen_meta: dict[str, Any] = {}
                    if str(candidate_measurement_feature_mode) == "frozen_measurement_v7":
                        frozen_cand_feat, frozen_obs_feat, frozen_similarity, frozen_meta = _cache_feature(frozen_cache, item_id, cid)
                    if cand_feat.numel() == 0 and frozen_cand_feat.numel() == 0:
                        continue
                    if dim_stats["candidate_crop_feature_dim"] is None:
                        dim_stats["candidate_crop_feature_dim"] = int(cand_feat.numel())
                    if dim_stats["observed_target_crop_feature_dim"] is None:
                        dim_stats["observed_target_crop_feature_dim"] = int(obs_feat.numel())
                    if str(candidate_measurement_feature_mode) == "frozen_measurement_v7":
                        dim_stats["candidate_frozen_feature_dim"] = dim_stats.get("candidate_frozen_feature_dim") or int(frozen_cand_feat.numel())
                        dim_stats["observed_target_frozen_feature_dim"] = dim_stats.get("observed_target_frozen_feature_dim") or int(frozen_obs_feat.numel())
                    center = _candidate_center(cand, item.get("image_size"))
                    if center is None:
                        dist_score = 0.0
                    else:
                        dx = float(pred_coord[0].item()) - float(center[0])
                        dy = float(pred_coord[1].item()) - float(center[1])
                        dist_score = float(math.exp(-8.0 * math.sqrt(dx * dx + dy * dy)))
                    appearance_score = _cosine01(obs_feat, cand_feat)
                    appearance_score_frozen = (
                        float(frozen_similarity)
                        if isinstance(frozen_similarity, (int, float))
                        else _cosine01(frozen_obs_feat, frozen_cand_feat)
                        if frozen_obs_feat.numel() and frozen_cand_feat.numel()
                        else None
                    )
                    label = int(labels[idx]) if idx < len(labels) else 0
                    row = {
                        "item_id": item_id,
                        "candidate_id": cid,
                        "candidate_index": int(idx),
                        "split": split,
                        "label_same_identity": label,
                        "predicted_semantic_embedding": _round_list(pred_sem),
                        "predicted_identity_embedding": _round_list(pred_id),
                        "candidate_crop_feature": [round(float(x), 6) for x in cand_feat.view(-1).tolist()],
                        "observed_target_crop_feature": [round(float(x), 6) for x in obs_feat.view(-1).tolist()],
                        "candidate_frozen_feature": [round(float(x), 6) for x in frozen_cand_feat.view(-1).tolist()],
                        "observed_target_frozen_feature": [round(float(x), 6) for x in frozen_obs_feat.view(-1).tolist()],
                        "distance_score": round(float(dist_score), 8),
                        "appearance_score": round(float(appearance_score), 8),
                        "appearance_score_crop_encoder": round(float(appearance_score), 8),
                        "appearance_score_frozen": round(float(appearance_score_frozen), 8) if appearance_score_frozen is not None else None,
                        "target_candidate_frozen_similarity": round(float(appearance_score_frozen), 8) if appearance_score_frozen is not None else None,
                        "future_visibility_prob": round(float(visibility_prob), 8),
                        "future_reappearance_event_prob": round(float(reappearance_prob), 8),
                        "subset_tags": tags,
                        "candidate_feature_source": (
                            str(frozen_cache.get("selected_backbone") or "frozen_measurement_v7")
                            if str(candidate_measurement_feature_mode) == "frozen_measurement_v7"
                            else str(feature.get("feature_source") or measurement_cache.get("candidate_feature_source") or "unknown")
                        ),
                        "candidate_frozen_feature_available": bool(frozen_cand_feat.numel() > 0),
                        "observed_target_frozen_feature_available": bool(frozen_obs_feat.numel() > 0),
                        "frozen_feature_meta": frozen_meta,
                        "future_candidate_used_as_input": False,
                        "candidate_feature_used_for_rollout": False,
                        "candidate_feature_used_for_scoring": True,
                        "model_batch_meta": {
                            "future_target_leakage": bool(builder_meta.get("future_target_leakage")),
                            "semantic_feature_source": builder_meta.get("semantic_feature_source"),
                            "padded_obs_repeated": builder_meta.get("padded_obs_repeated"),
                        },
                    }
                    rows.append(row)
                    split_counts[split] += 1
                    label_counts["positive" if label else "negative"] += 1
                    for subset, flag in tags.items():
                        if bool(flag):
                            subset_counts[str(subset)]["positive" if label else "negative"] += 1
                    norm_stats["predicted_semantic_norm"].append(float(pred_sem.norm().item()))
                    norm_stats["predicted_identity_norm"].append(float(pred_id.norm().item()))
                    norm_stats["candidate_crop_feature_norm"].append(float(cand_feat.norm().item()))
                    norm_stats["observed_target_crop_feature_norm"].append(float(obs_feat.norm().item()))
                    norm_stats["appearance_score"].append(float(appearance_score))
                    if frozen_cand_feat.numel():
                        norm_stats["candidate_frozen_feature_norm"].append(float(frozen_cand_feat.norm().item()))
                    if frozen_obs_feat.numel():
                        norm_stats["observed_target_frozen_feature_norm"].append(float(frozen_obs_feat.norm().item()))
                    if appearance_score_frozen is not None:
                        norm_stats["appearance_score_frozen"].append(float(appearance_score_frozen))
            except Exception as exc:
                item_failures.append({"item_id": item_id, "failure_reason": repr(exc)})

    payload = {
        "generated_at_utc": now_iso(),
        "schema_version": "semantic_state_measurement_alignment_dataset_v7"
        if str(candidate_measurement_feature_mode) == "frozen_measurement_v7"
        else "semantic_state_measurement_alignment_dataset_v1",
        "repo_root": str(repo_root),
        "checkpoint_path": str(checkpoint),
        "candidate_manifest": str(candidate_manifest),
        "frozen_measurement_cache": str(frozen_measurement_cache) if frozen_measurement_cache else None,
        "candidate_measurement_feature_mode": str(candidate_measurement_feature_mode),
        "frozen_measurement_feature_available": bool(
            str(candidate_measurement_feature_mode) == "frozen_measurement_v7"
            and bool(frozen_cache.get("frozen_measurement_feature_available"))
        ),
        "selected_frozen_backbone": frozen_cache.get("selected_backbone") if isinstance(frozen_cache, dict) else None,
        "mode": str(mode),
        "full_model_forward_executed": bool(full_model_forward_executed),
        "full_free_rollout_executed": bool(full_free_rollout_executed),
        "stage2_val_fallback_used": False,
        "old_association_report_used": False,
        "future_candidate_used_as_input": False,
        "candidate_feature_used_for_rollout": False,
        "candidate_feature_used_for_scoring": True,
        "no_candidate_leakage_to_rollout": True,
        "split_rule": "sha256(item_id)%2; 0=dev, 1=heldout",
        "item_count": len(selected_items),
        "successful_item_count": len({r["item_id"] for r in rows}),
        "failed_item_count": len(item_failures),
        "item_failures": item_failures[:20],
        "record_count": len(rows),
        "split_counts": dict(split_counts),
        "label_counts": dict(label_counts),
        "subset_label_counts": {k: dict(v) for k, v in sorted(subset_counts.items())},
        "feature_stats": {
            **dim_stats,
            **{
                key + "_mean": _mean(vals)
                for key, vals in norm_stats.items()
            },
            **{
                key + "_variance": _variance(vals)
                for key, vals in norm_stats.items()
            },
        },
        "records": rows,
    }
    write_json(output, payload)
    write_doc(doc, payload)
    return payload


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--repo-root", default=os.environ.get("STWM_ROOT", "."))
    parser.add_argument("--checkpoint", default="outputs/checkpoints/stage2_tusb_v3p1_semantic_state_feedback_v1_20260427/latest.pt")
    parser.add_argument("--candidate-manifest", default="reports/stwm_external_hardcase_candidate_expanded_manifest_v2_20260428.json")
    parser.add_argument("--output", default="reports/stwm_semantic_state_measurement_alignment_dataset_v1_20260428.json")
    parser.add_argument("--doc", default="docs/STWM_SEMANTIC_STATE_MEASUREMENT_ALIGNMENT_DATASET_V1_20260428.md")
    parser.add_argument("--max-items", type=int, default=389)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--mode", choices=["full_model_teacher_forced", "full_model_free_rollout"], default="full_model_free_rollout")
    parser.add_argument("--candidate-measurement-feature-mode", default="crop_encoder_feature")
    parser.add_argument("--frozen-measurement-cache", default=None)
    args = parser.parse_args()
    build_dataset(
        repo_root=Path(args.repo_root).expanduser().resolve(),
        checkpoint=Path(args.checkpoint),
        candidate_manifest=Path(args.candidate_manifest),
        output=Path(args.output),
        doc=Path(args.doc),
        max_items=int(args.max_items),
        device_name=str(args.device),
        mode=str(args.mode),
        candidate_measurement_feature_mode=str(args.candidate_measurement_feature_mode),
        frozen_measurement_cache=Path(args.frozen_measurement_cache) if args.frozen_measurement_cache else None,
    )


if __name__ == "__main__":
    main()
