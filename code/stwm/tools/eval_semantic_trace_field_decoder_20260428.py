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
from stwm.tracewm_v2_stage2.utils.future_semantic_prototype_targets import (
    load_future_semantic_prototype_target_cache,
    prototype_tensors_for_batch,
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
    fr = overall.get("free_rollout", {}) if isinstance(overall.get("free_rollout"), dict) else {}
    lines = [
        "# STWM Semantic Trace Field Decoder V1 Eval",
        "",
        f"- checkpoint: `{payload.get('checkpoint')}`",
        f"- prototype_count: `{payload.get('prototype_count')}`",
        f"- target_valid_ratio: `{payload.get('target_valid_ratio')}`",
        f"- free_rollout proto_accuracy: `{fr.get('proto_accuracy')}`",
        f"- free_rollout proto_top5_accuracy: `{fr.get('proto_top5_accuracy')}`",
        f"- free_rollout masked_CE: `{fr.get('masked_CE')}`",
        f"- free_rollout future_trace_coord_error: `{fr.get('future_trace_coord_error')}`",
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


def _mode_metrics(full: dict[str, Any], cache: Any, mode: str, device: torch.device, max_items: int) -> dict[str, Any]:
    trainer = full["trainer"]
    args = full["args"]
    valid_items = 0
    ce_sum = 0.0
    ce_count = 0
    correct = 0
    top5_correct = 0
    target_hist: dict[int, int] = {}
    coord_l2: list[float] = []
    visibility_scores: list[float] = []
    visibility_labels: list[int] = []
    reappearance_scores: list[float] = []
    reappearance_labels: list[int] = []
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
            out = trainer._teacher_forced_predict(**common) if mode == "teacher_forced" else trainer._free_rollout_predict(fut_len=int(args.fut_len), **common)
            state = out.get("future_semantic_trace_state")
            if state is None or state.future_semantic_proto_logits is None:
                continue
            target, _dist, mask, _info = prototype_tensors_for_batch(
                cache,
                batch,
                horizon=int(state.future_trace_coord.shape[1]),
                slot_count=int(state.future_trace_coord.shape[2]),
                device=device,
            )
            if target is None or mask is None:
                continue
            logits = state.future_semantic_proto_logits
            valid = mask & (target >= 0)
            if bool(valid.any().item()):
                flat_logits = logits[valid]
                flat_target = target[valid]
                ce = F.cross_entropy(flat_logits, flat_target, reduction="sum")
                ce_sum += float(ce.detach().cpu().item())
                ce_count += int(flat_target.numel())
                pred = flat_logits.argmax(dim=-1)
                correct += int((pred == flat_target).sum().detach().cpu().item())
                topk = min(5, int(flat_logits.shape[-1]))
                top5_correct += int(
                    (flat_logits.topk(k=topk, dim=-1).indices == flat_target[:, None])
                    .any(dim=-1)
                    .sum()
                    .detach()
                    .cpu()
                    .item()
                )
                for cls in flat_target.detach().cpu().tolist():
                    target_hist[int(cls)] = target_hist.get(int(cls), 0) + 1
                vis_target = batch["fut_valid"][:, : target.shape[1], : target.shape[2]].to(device=device, dtype=torch.bool)
                vis_prob = torch.sigmoid(state.future_visibility_logit[valid]).detach().cpu().tolist()
                vis_label = vis_target[valid].detach().cpu().tolist()
                visibility_scores.extend(float(x) for x in vis_prob)
                visibility_labels.extend(int(bool(x)) for x in vis_label)
                if state.future_reappearance_logit is not None:
                    obs_valid = batch["obs_valid"][:, : int(args.obs_len), : target.shape[2]].to(device=device, dtype=torch.bool)
                    obs_seen_any = obs_valid.any(dim=1)
                    endpoint = obs_valid[:, -1] if obs_valid.shape[1] > 0 else torch.zeros_like(obs_seen_any)
                    obs_occluded = obs_seen_any & (~obs_valid.all(dim=1))
                    gate = ((~endpoint) | obs_occluded) & obs_seen_any
                    rep_target = vis_target & gate[:, None, :]
                    rep_prob = torch.sigmoid(state.future_reappearance_logit[valid]).detach().cpu().tolist()
                    rep_label = rep_target[valid].detach().cpu().tolist()
                    reappearance_scores.extend(float(x) for x in rep_prob)
                    reappearance_labels.extend(int(bool(x)) for x in rep_label)
            l2 = torch.sqrt(((out["pred_coord"] - out["target_coord"]) ** 2).sum(dim=-1).clamp_min(1e-12))
            coord_valid = out["valid_mask"].float()
            coord_l2.append(float((l2 * coord_valid).sum().detach().cpu().item() / float(coord_valid.sum().clamp_min(1.0).detach().cpu().item())))
            valid_items += 1
    if ce_count <= 0:
        return {"valid_items": int(valid_items), "blocked_reason": "no valid prototype targets matched this checkpoint/data split"}
    most_common = max(target_hist.values()) if target_hist else 0
    sorted_counts = sorted(target_hist.values(), reverse=True)
    freq_top5 = sum(sorted_counts[:5]) / max(ce_count, 1)
    return {
        "valid_items": int(valid_items),
        "valid_output_ratio": float(valid_items / max(int(max_items), 1)),
        "valid_target_count": int(ce_count),
        "masked_CE": float(ce_sum / max(ce_count, 1)),
        "proto_accuracy": float(correct / max(ce_count, 1)),
        "proto_top5_accuracy": float(top5_correct / max(ce_count, 1)),
        "frequency_baseline_top1": float(most_common / max(ce_count, 1)),
        "frequency_baseline_top5": float(freq_top5),
        "visibility_AP": _average_precision(visibility_scores, visibility_labels),
        "visibility_AUROC": _auroc(visibility_scores, visibility_labels),
        "reappearance_AP": _average_precision(reappearance_scores, reappearance_labels),
        "reappearance_AUROC": _auroc(reappearance_scores, reappearance_labels),
        "future_trace_coord_error": float(sum(coord_l2) / max(len(coord_l2), 1)) if coord_l2 else None,
        "target_valid_ratio": float(ce_count / max(valid_items, 1) / max(int(args.fut_len) * int(args.max_entities_per_sample), 1)),
    }


def evaluate(
    *,
    repo_root: Path,
    checkpoint: Path,
    prototype_target_cache: Path,
    output: Path,
    doc: Path,
    max_items: int,
    device_name: str,
) -> dict[str, Any]:
    _apply_process_title_normalization()
    device = torch.device(device_name if device_name == "cuda" and torch.cuda.is_available() else "cpu")
    cache = load_future_semantic_prototype_target_cache(prototype_target_cache)
    full = _build_full_model_from_checkpoint(repo_root, checkpoint, device, int(max_items))
    tf = _mode_metrics(full, cache, "teacher_forced", device, int(max_items))
    fr = _mode_metrics(full, cache, "free_rollout", device, int(max_items))
    payload = {
        "generated_at_utc": now_iso(),
        "checkpoint": str(checkpoint),
        "prototype_target_cache": str(prototype_target_cache),
        "prototype_count": int(cache.prototype_count if cache is not None else 0),
        "feature_backbone": cache.feature_backbone if cache is not None else "",
        "target_valid_ratio": float(cache.mask.float().mean().item()) if cache is not None else 0.0,
        "no_candidate_leakage": bool(cache.no_candidate_leakage if cache is not None else True),
        "full_model_forward_executed": True,
        "full_free_rollout_executed": True,
        "overall": {
            "teacher_forced": tf,
            "free_rollout": fr,
            "free_rollout_vs_teacher_forced_gap_proto_accuracy": (
                None
                if tf.get("proto_accuracy") is None or fr.get("proto_accuracy") is None
                else float(fr["proto_accuracy"] - tf["proto_accuracy"])
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
    p.add_argument("--prototype-target-cache", default="reports/stwm_future_semantic_trace_prototype_targets_v1_20260428.json")
    p.add_argument("--output", default="reports/stwm_semantic_trace_field_decoder_v1_eval_20260428.json")
    p.add_argument("--doc", default="docs/STWM_SEMANTIC_TRACE_FIELD_DECODER_V1_EVAL_20260428.md")
    p.add_argument("--max-items", type=int, default=32)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    evaluate(
        repo_root=Path(args.repo_root).resolve(),
        checkpoint=Path(args.checkpoint),
        prototype_target_cache=Path(args.prototype_target_cache),
        output=Path(args.output),
        doc=Path(args.doc),
        max_items=int(args.max_items),
        device_name=str(args.device),
    )


if __name__ == "__main__":
    main()
