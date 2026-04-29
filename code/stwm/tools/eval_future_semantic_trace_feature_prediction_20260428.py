#!/usr/bin/env python3
from __future__ import annotations

from argparse import ArgumentParser
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import json
import os

import torch
import torch.nn.functional as F

from stwm.tools.export_future_semantic_trace_state_20260427 import _build_full_model_from_checkpoint
from stwm.tracewm_v2_stage2.utils.future_semantic_feature_targets import (
    load_future_semantic_feature_target_cache,
    target_tensors_for_batch,
)


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


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_doc(path: Path, payload: dict[str, Any]) -> None:
    overall = payload.get("overall", {}) if isinstance(payload.get("overall"), dict) else {}
    lines = [
        "# STWM Future Semantic Trace Feature Eval V1",
        "",
        f"- checkpoint: `{payload.get('checkpoint')}`",
        f"- target_cache: `{payload.get('target_cache')}`",
        f"- feature_backbone: `{payload.get('feature_backbone')}`",
        f"- valid_target_ratio: `{payload.get('valid_target_ratio')}`",
        f"- teacher_forced cosine_positive_mean: `{overall.get('teacher_forced', {}).get('cosine_positive_mean') if isinstance(overall.get('teacher_forced'), dict) else None}`",
        f"- free_rollout cosine_positive_mean: `{overall.get('free_rollout', {}).get('cosine_positive_mean') if isinstance(overall.get('free_rollout'), dict) else None}`",
        f"- free_rollout retrieval_top1: `{overall.get('free_rollout', {}).get('retrieval_top1') if isinstance(overall.get('free_rollout'), dict) else None}`",
        f"- free_rollout retrieval_AUROC: `{overall.get('free_rollout', {}).get('retrieval_AUROC') if isinstance(overall.get('free_rollout'), dict) else None}`",
        f"- no_candidate_leakage: `{payload.get('no_candidate_leakage')}`",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _average_precision(scores: list[float], labels: list[int]) -> float | None:
    positives = sum(1 for x in labels if int(x) == 1)
    if positives <= 0:
        return None
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    hit = 0
    acc = 0.0
    for rank, idx in enumerate(order, start=1):
        if int(labels[idx]) == 1:
            hit += 1
            acc += hit / rank
    return float(acc / positives)


def _auroc(scores: list[float], labels: list[int]) -> float | None:
    pos = [scores[i] for i, y in enumerate(labels) if int(y) == 1]
    neg = [scores[i] for i, y in enumerate(labels) if int(y) == 0]
    if not pos or not neg:
        return None
    wins = 0.0
    total = 0.0
    for p in pos:
        for n in neg:
            total += 1.0
            if p > n:
                wins += 1.0
            elif p == n:
                wins += 0.5
    return float(wins / max(total, 1.0))


def _mode_metrics(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> dict[str, Any]:
    if pred is None or target is None or mask is None or not bool(mask.any().item()):
        return {
            "valid_target_count": 0,
            "cosine_positive_mean": None,
            "positive_vs_negative_margin_mean": None,
            "retrieval_top1": None,
            "retrieval_MRR": None,
            "retrieval_AP": None,
            "retrieval_AUROC": None,
        }
    pred_n = F.normalize(pred.float(), dim=-1)
    tgt_n = F.normalize(target.float(), dim=-1)
    valid_idx = torch.nonzero(mask, as_tuple=False)
    pos_scores: list[float] = []
    margins: list[float] = []
    top1: list[float] = []
    rr: list[float] = []
    pair_scores: list[float] = []
    pair_labels: list[int] = []
    for idx in valid_idx:
        b, h, k = [int(x.item()) for x in idx]
        valid_targets = torch.nonzero(mask[b], as_tuple=False)
        if valid_targets.numel() <= 0:
            continue
        p = pred_n[b, h, k]
        candidates = tgt_n[b][mask[b]]
        labels = []
        scores = torch.mv(candidates, p)
        flat_positions = [(int(x[0].item()), int(x[1].item())) for x in valid_targets]
        positive_pos = None
        for j, (hh, kk) in enumerate(flat_positions):
            label = int(hh == h and kk == k)
            labels.append(label)
            if label:
                positive_pos = j
        if positive_pos is None:
            continue
        score_values = [float(x) for x in scores.detach().cpu().tolist()]
        pos_score = score_values[positive_pos]
        neg_scores = [s for j, s in enumerate(score_values) if j != positive_pos]
        pos_scores.append(pos_score)
        margins.append(pos_score - max(neg_scores) if neg_scores else 0.0)
        order = sorted(range(len(score_values)), key=lambda j: score_values[j], reverse=True)
        rank = order.index(positive_pos) + 1
        top1.append(1.0 if rank == 1 else 0.0)
        rr.append(1.0 / rank)
        pair_scores.extend(score_values)
        pair_labels.extend(labels)
    return {
        "valid_target_count": int(len(pos_scores)),
        "cosine_positive_mean": float(sum(pos_scores) / max(len(pos_scores), 1)) if pos_scores else None,
        "positive_vs_negative_margin_mean": float(sum(margins) / max(len(margins), 1)) if margins else None,
        "retrieval_top1": float(sum(top1) / max(len(top1), 1)) if top1 else None,
        "retrieval_MRR": float(sum(rr) / max(len(rr), 1)) if rr else None,
        "retrieval_AP": _average_precision(pair_scores, pair_labels),
        "retrieval_AUROC": _auroc(pair_scores, pair_labels),
    }


def _run_mode(full: dict[str, Any], cache: Any, mode: str, device: torch.device, max_items: int) -> dict[str, Any]:
    trainer = full["trainer"]
    args = full["args"]
    pred_blocks: list[torch.Tensor] = []
    target_blocks: list[torch.Tensor] = []
    mask_blocks: list[torch.Tensor] = []
    coord_l2: list[float] = []
    valid_items = 0
    with torch.no_grad():
        for item_idx, raw_batch in enumerate(full["loader"]):
            if item_idx >= int(max_items):
                break
            batch = trainer._to_device(raw_batch, device=device, non_blocking=False)
            common = dict(
                stage1_model=full["stage1_model"],
                semantic_encoder=full["semantic_encoder"],
                semantic_fusion=full["semantic_fusion"],
                readout_head=full["readout_head"],
                future_semantic_state_head=full["future_semantic_state_head"],
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
            if mode == "teacher_forced":
                out = trainer._teacher_forced_predict(**common)
            else:
                out = trainer._free_rollout_predict(fut_len=int(args.fut_len), **common)
            state = out.get("future_semantic_trace_state")
            if state is None or state.future_measurement_feature_pred is None:
                continue
            target, mask, _ = target_tensors_for_batch(
                cache,
                batch,
                horizon=int(state.future_trace_coord.shape[1]),
                slot_count=int(state.future_trace_coord.shape[2]),
                device=device,
            )
            if target is None or mask is None:
                continue
            pred_blocks.append(state.future_measurement_feature_pred.detach().cpu())
            target_blocks.append(target.detach().cpu())
            mask_blocks.append(mask.detach().cpu())
            l2 = torch.sqrt(((out["pred_coord"] - out["target_coord"]) ** 2).sum(dim=-1).clamp_min(1e-12))
            valid = out["valid_mask"].float()
            coord_l2.append(float((l2 * valid).sum().detach().cpu().item() / float(valid.sum().clamp_min(1.0).detach().cpu().item())))
            valid_items += 1
    if not pred_blocks:
        return {"valid_items": 0, "blocked_reason": "no future_measurement_feature_pred outputs matched target cache"}
    max_h = max(int(x.shape[1]) for x in pred_blocks)
    max_k = max(int(x.shape[2]) for x in pred_blocks)
    dim = max(int(x.shape[-1]) for x in pred_blocks)

    def _pad_feature_block(block: torch.Tensor) -> torch.Tensor:
        out = torch.zeros((int(block.shape[0]), max_h, max_k, dim), dtype=block.dtype)
        h = min(max_h, int(block.shape[1]))
        k = min(max_k, int(block.shape[2]))
        d = min(dim, int(block.shape[-1]))
        out[:, :h, :k, :d] = block[:, :h, :k, :d]
        return out

    def _pad_mask_block(block: torch.Tensor) -> torch.Tensor:
        out = torch.zeros((int(block.shape[0]), max_h, max_k), dtype=torch.bool)
        h = min(max_h, int(block.shape[1]))
        k = min(max_k, int(block.shape[2]))
        out[:, :h, :k] = block[:, :h, :k].to(dtype=torch.bool)
        return out

    pred = torch.cat([_pad_feature_block(x) for x in pred_blocks], dim=0)
    target = torch.cat([_pad_feature_block(x) for x in target_blocks], dim=0)
    mask = torch.cat([_pad_mask_block(x) for x in mask_blocks], dim=0)
    metrics = _mode_metrics(pred, target, mask)
    metrics.update(
        {
            "valid_items": int(valid_items),
            "valid_output_ratio": float(valid_items / max(int(max_items), 1)),
            "future_trace_coord_error": float(sum(coord_l2) / max(len(coord_l2), 1)) if coord_l2 else None,
            "target_valid_ratio": float(mask.float().mean().item()),
        }
    )
    return metrics


def evaluate(
    *,
    repo_root: Path,
    checkpoint: Path,
    target_cache_report: Path,
    output: Path,
    doc: Path,
    max_items: int,
    device_name: str,
) -> dict[str, Any]:
    _apply_process_title_normalization()
    device = torch.device(device_name if device_name == "cuda" and torch.cuda.is_available() else "cpu")
    cache = load_future_semantic_feature_target_cache(target_cache_report)
    full = _build_full_model_from_checkpoint(repo_root, checkpoint, device, int(max_items))
    tf = _run_mode(full, cache, "teacher_forced", device, int(max_items))
    fr = _run_mode(full, cache, "free_rollout", device, int(max_items))
    payload = {
        "generated_at_utc": now_iso(),
        "checkpoint": str(checkpoint),
        "target_cache": str(target_cache_report),
        "feature_backbone": cache.feature_backbone if cache is not None else "",
        "feature_dim": cache.feature_dim if cache is not None else 0,
        "valid_target_ratio": float(cache.mask.float().mean().item()) if cache is not None else 0.0,
        "no_candidate_leakage": bool(cache.no_candidate_leakage if cache is not None else True),
        "full_model_forward_executed": True,
        "full_free_rollout_executed": True,
        "overall": {
            "teacher_forced": tf,
            "free_rollout": fr,
            "free_rollout_vs_teacher_forced_gap_cosine": (
                None
                if tf.get("cosine_positive_mean") is None or fr.get("cosine_positive_mean") is None
                else float(fr["cosine_positive_mean"] - tf["cosine_positive_mean"])
            ),
        },
    }
    write_json(output, payload)
    write_doc(doc, payload)
    return payload


def main() -> None:
    p = ArgumentParser()
    p.add_argument("--repo-root", default=".")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--target-cache", default="reports/stwm_future_semantic_trace_feature_targets_v1_20260428.json")
    p.add_argument("--output", default="reports/stwm_future_semantic_trace_feature_eval_v1_20260428.json")
    p.add_argument("--doc", default="docs/STWM_FUTURE_SEMANTIC_TRACE_FEATURE_EVAL_V1_20260428.md")
    p.add_argument("--max-items", type=int, default=32)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    evaluate(
        repo_root=Path(args.repo_root).resolve(),
        checkpoint=Path(args.checkpoint),
        target_cache_report=Path(args.target_cache),
        output=Path(args.output),
        doc=Path(args.doc),
        max_items=int(args.max_items),
        device_name=str(args.device),
    )


if __name__ == "__main__":
    main()
