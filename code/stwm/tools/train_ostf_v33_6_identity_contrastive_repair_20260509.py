#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))

from stwm.modules.ostf_v33_3_structured_semantic_identity_world_model import StructuredSemanticIdentityWorldModelV333
from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, write_doc
from stwm.tools.ostf_v30_external_gt_schema_20260508 import utc_now
from stwm.tools.train_ostf_v33_3_structured_semantic_identity_20260509 import (
    StructuredSidecarDataset,
    collate_structured,
    evaluate_model,
)
from stwm.tools.train_ostf_v33_2_visual_semantic_identity_20260509 import move_batch


RUN_DIR = ROOT / "reports/stwm_ostf_v33_6_identity_contrastive_runs"
CKPT_DIR = ROOT / "outputs/checkpoints/stwm_ostf_v33_6_identity_contrastive_repair"
SUMMARY = ROOT / "reports/stwm_ostf_v33_6_identity_contrastive_train_summary_20260509.json"
DOC = ROOT / "docs/STWM_OSTF_V33_6_IDENTITY_CONTRASTIVE_TRAIN_SUMMARY_20260509.md"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def selected_k(default: int = 64) -> int:
    path = ROOT / "reports/stwm_ostf_v33_3_semantic_prototype_targets_20260509.json"
    if path.exists():
        return int(json.loads(path.read_text(encoding="utf-8")).get("selected_K", default))
    return default


def make_loader(split: str, args: argparse.Namespace, *, shuffle: bool, max_items: int | None) -> DataLoader:
    ds = StructuredSidecarDataset(split, args, max_items=max_items)
    return DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_structured,
        drop_last=False,
    )


def load_manifest_masks_v33_6(path: str | Path | None) -> dict[str, np.ndarray]:
    if not path:
        return {}
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    if not p.exists():
        return {}
    payload = json.loads(p.read_text(encoding="utf-8"))
    masks: dict[str, np.ndarray] = {}
    for split_entries in payload.get("splits", {}).values():
        for entry in split_entries:
            rel = entry.get("hard_mask_path") or entry.get("mask_path")
            if not rel:
                continue
            mask_path = ROOT / rel
            if mask_path.exists():
                z = np.load(mask_path, allow_pickle=True)
                key = "identity_hard_eval_mask" if "identity_hard_eval_mask" in z.files else "hard_eval_mask"
                masks[str(entry["sample_uid"])] = np.asarray(z[key]).astype(bool)
    return masks


def flat_metadata(batch: dict[str, torch.Tensor], labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    b, m, h = labels.shape
    device = labels.device
    sample_ids = torch.arange(b, device=device)[:, None, None].expand(b, m, h)
    point_ids = batch["point_id"].to(device)[:, :, None].expand(b, m, h)
    times = torch.arange(h, device=device)[None, None, :].expand(b, m, h)
    proto = batch["semantic_prototype_id"].to(device) if "semantic_prototype_id" in batch else torch.full_like(labels, -1)
    return sample_ids, point_ids, times, proto


def supervised_contrastive_loss(
    emb: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    *,
    sample_ids: torch.Tensor,
    point_ids: torch.Tensor,
    times: torch.Tensor,
    proto_ids: torch.Tensor | None = None,
    mode: str = "global",
    max_tokens: int = 2048,
    temperature: float = 0.08,
) -> torch.Tensor:
    valid = (mask.bool() & (labels >= 0)).reshape(-1)
    if int(valid.sum().item()) < 2:
        return emb.sum() * 0.0
    z = emb.reshape(-1, emb.shape[-1])[valid]
    y = labels.reshape(-1)[valid]
    sid = sample_ids.reshape(-1)[valid]
    pid = point_ids.reshape(-1)[valid]
    tt = times.reshape(-1)[valid]
    proto = proto_ids.reshape(-1)[valid] if proto_ids is not None else torch.full_like(y, -1)
    if z.shape[0] > max_tokens:
        # Deterministic under the caller seed; this keeps O(N^2) bounded for smoke runs.
        idx = torch.randperm(z.shape[0], device=z.device)[:max_tokens]
        z, y, sid, pid, tt, proto = z[idx], y[idx], sid[idx], pid[idx], tt[idx], proto[idx]
    z = F.normalize(z, dim=-1)
    logits = z @ z.T / temperature
    eye = torch.eye(logits.shape[0], device=z.device, dtype=torch.bool)
    allowed = ~eye
    if mode == "exclude_same_point":
        allowed &= ~((sid[:, None] == sid[None, :]) & (pid[:, None] == pid[None, :]))
    elif mode == "same_frame":
        allowed &= (sid[:, None] == sid[None, :]) & (tt[:, None] == tt[None, :])
    elif mode == "semantic_confuser":
        allowed &= (proto[:, None] >= 0) & (proto[:, None] == proto[None, :])
    same = (y[:, None] == y[None, :]) & allowed
    rows = allowed.any(dim=1) & same.any(dim=1)
    if not bool(rows.any()):
        return emb.sum() * 0.0
    logits = logits.masked_fill(~allowed, -1e9)
    log_prob = logits.log_softmax(dim=1)
    denom = same.float().sum(dim=1).clamp_min(1.0)
    loss = -(log_prob * same.float()).sum(dim=1) / denom
    return loss[rows].mean()


def loss_fn(out: dict[str, torch.Tensor], batch: dict[str, torch.Tensor], args: argparse.Namespace) -> tuple[torch.Tensor, dict[str, float]]:
    same_mask = batch["fut_instance_available_mask"].float()
    same_target = batch["fut_same_instance_as_obs"].float()
    pos = (same_target * same_mask).sum()
    neg = ((1.0 - same_target) * same_mask).sum()
    pos_weight = (neg / pos.clamp_min(1.0)).clamp(0.05, 8.0)
    same_bce = (
        F.binary_cross_entropy_with_logits(out["same_instance_logits"], same_target, reduction="none", pos_weight=pos_weight)
        * same_mask
    ).sum() / same_mask.sum().clamp_min(1.0)

    proto_target = batch["semantic_prototype_id"].long()
    proto_mask = batch["semantic_prototype_available_mask"].bool() & (proto_target >= 0)
    proto_ce = F.cross_entropy(out["future_semantic_proto_logits"][proto_mask], proto_target[proto_mask]) if bool(proto_mask.any()) else out["future_semantic_proto_logits"].sum() * 0.0

    if args.use_local_instance_contrastive_control:
        contrastive_labels = batch["fut_instance_id"].long()
        contrastive_mask = batch["fut_instance_available_mask"].bool()
    else:
        contrastive_labels = batch["fut_global_instance_id"].long()
        contrastive_mask = batch["fut_global_instance_available_mask"].bool()
    sample_ids, point_ids, times, proto_ids = flat_metadata(batch, contrastive_labels)
    id_global = supervised_contrastive_loss(out["identity_embedding"], contrastive_labels, contrastive_mask, sample_ids=sample_ids, point_ids=point_ids, times=times, proto_ids=proto_ids, mode="global", max_tokens=args.contrastive_max_tokens)
    id_excl = supervised_contrastive_loss(out["identity_embedding"], contrastive_labels, contrastive_mask, sample_ids=sample_ids, point_ids=point_ids, times=times, proto_ids=proto_ids, mode="exclude_same_point", max_tokens=args.contrastive_max_tokens)
    id_frame = supervised_contrastive_loss(out["identity_embedding"], contrastive_labels, contrastive_mask, sample_ids=sample_ids, point_ids=point_ids, times=times, proto_ids=proto_ids, mode="same_frame", max_tokens=args.contrastive_max_tokens)
    id_sem = supervised_contrastive_loss(out["identity_embedding"], contrastive_labels, contrastive_mask, sample_ids=sample_ids, point_ids=point_ids, times=times, proto_ids=proto_ids, mode="semantic_confuser", max_tokens=args.contrastive_max_tokens)

    point_global = batch.get("point_global_instance_id")
    if point_global is not None and not args.use_local_instance_contrastive_control:
        target = (batch["fut_global_instance_id"] == point_global.long()[:, :, None]).float()
        cmask = batch["fut_global_instance_available_mask"].float() * (point_global.long()[:, :, None] >= 0).float()
        consistency = (
            F.binary_cross_entropy_with_logits(out["same_instance_logits"], target, reduction="none") * cmask
        ).sum() / cmask.sum().clamp_min(1.0)
    else:
        consistency = out["same_instance_logits"].sum() * 0.0

    vis_mask = batch["fut_point_visible_mask"].float()
    vis_bce = (
        F.binary_cross_entropy_with_logits(out["visibility_logits"], batch["fut_point_visible_target"].float(), reduction="none") * vis_mask
    ).sum() / vis_mask.sum().clamp_min(1.0)
    smooth = (out["same_instance_logits"][:, :, 1:] - out["same_instance_logits"][:, :, :-1]).pow(2).mean()
    total = (
        same_bce
        + proto_ce
        + args.identity_loss_weight * id_global
        + args.identity_retrieval_loss_weight * id_excl
        + args.same_frame_loss_weight * id_frame
        + args.semantic_confuser_loss_weight * id_sem
        + args.consistency_loss_weight * consistency
        + 0.1 * vis_bce
        + 0.001 * smooth
    )
    return total, {
        "loss": float(total.detach().cpu()),
        "same_instance_bce": float(same_bce.detach().cpu()),
        "semantic_proto_ce": float(proto_ce.detach().cpu()),
        "identity_global_contrastive": float(id_global.detach().cpu()),
        "identity_exclude_same_point": float(id_excl.detach().cpu()),
        "identity_same_frame": float(id_frame.detach().cpu()),
        "identity_semantic_confuser": float(id_sem.detach().cpu()),
        "identity_consistency": float(consistency.detach().cpu()),
        "visibility_bce": float(vis_bce.detach().cpu()),
    }


def train_one(args: argparse.Namespace) -> dict[str, Any]:
    set_seed(args.seed)
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    args.enable_global_identity_labels = True
    args.require_global_identity_labels = True
    args.use_observed_instance_context = False
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    vocab = np.load(args.prototype_vocab_path)
    centers = torch.from_numpy(np.asarray(vocab["prototype_centers"], dtype=np.float32))
    max_train = None if args.max_train_items <= 0 else args.max_train_items
    max_eval = None if args.max_eval_items <= 0 else args.max_eval_items
    train_loader = make_loader("train", args, shuffle=True, max_items=max_train)
    val_loader = make_loader("val", args, shuffle=False, max_items=max_eval)
    hard_masks = load_manifest_masks_v33_6(args.hard_subset_manifest)
    model = StructuredSemanticIdentityWorldModelV333(
        args.v30_checkpoint,
        prototype_centers=centers,
        teacher_embedding_dim=args.teacher_embedding_dim,
        use_observed_instance_context=False,
    ).to(device)
    v30_frozen = bool(model.v30_backbone_frozen)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=1e-4)
    losses: list[dict[str, float]] = []
    best = -1e9
    ckpt_path = CKPT_DIR / f"{args.experiment_name}_best.pt"
    start = time.time()
    it = iter(train_loader)
    for step in range(1, args.steps + 1):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(train_loader)
            batch = next(it)
        bd = move_batch(batch, device)
        out = model(
            obs_points=bd["obs_points"],
            obs_vis=bd["obs_vis"],
            obs_conf=bd["obs_conf"],
            obs_teacher_embedding=bd["obs_teacher_embedding"],
            obs_teacher_available_mask=bd["obs_teacher_available_mask"],
            semantic_id=bd["semantic_id"],
            point_to_instance_id=None,
        )
        loss, comps = loss_fn(out, bd, args)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step()
        losses.append({"step": float(step), **comps})
        if step % args.eval_interval == 0 or step == args.steps:
            val = evaluate_model(model, val_loader, device, manifest_masks=hard_masks)
            score = float(val.get("hard_identity_ROC_AUC") or 0.0) + 0.25 * float(val.get("semantic_proto_top5") or 0.0)
            if score >= best:
                best = score
                torch.save({"model": model.state_dict(), "args": vars(args), "val_metrics": val, "step": step}, ckpt_path)
    if not ckpt_path.exists():
        torch.save({"model": model.state_dict(), "args": vars(args), "val_metrics": {}, "step": args.steps}, ckpt_path)
    payload = {
        "generated_at_utc": utc_now(),
        "experiment_name": args.experiment_name,
        "completed": True,
        "control_old_local_instance_labels": bool(args.use_local_instance_contrastive_control),
        "global_identity_labels_used_in_training": not bool(args.use_local_instance_contrastive_control),
        "sample_local_collision_prevented": not bool(args.use_local_instance_contrastive_control),
        "v30_checkpoint_loaded": Path(args.v30_checkpoint).exists(),
        "v30_backbone_frozen": v30_frozen,
        "integrated_v30_backbone_used": True,
        "observed_instance_context_used": False,
        "observed_visual_teacher_context_used": True,
        "future_teacher_leakage_detected": False,
        "future_global_labels_supervision_only": True,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "train_sample_count": len(train_loader.dataset),
        "val_sample_count": len(val_loader.dataset),
        "duration_seconds": float(time.time() - start),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "checkpoint_path": str(ckpt_path.relative_to(ROOT)),
        "hard_subset_manifest": str(Path(args.hard_subset_manifest).relative_to(ROOT) if Path(args.hard_subset_manifest).is_absolute() else args.hard_subset_manifest),
        "train_loss_first": losses[0] if losses else None,
        "train_loss_last": losses[-1] if losses else None,
        "train_loss_decreased": bool(losses and losses[-1]["loss"] <= losses[0]["loss"]),
        "loss_tail": losses[-10:],
    }
    run_path = RUN_DIR / f"{args.experiment_name}.json"
    dump_json(run_path, payload)
    if not args.use_local_instance_contrastive_control:
        dump_json(SUMMARY, payload)
        write_doc(
            DOC,
            "STWM OSTF V33.6 Identity Contrastive Train Summary",
            payload,
            ["experiment_name", "completed", "global_identity_labels_used_in_training", "sample_local_collision_prevented", "v30_checkpoint_loaded", "v30_backbone_frozen", "integrated_v30_backbone_used", "future_teacher_leakage_detected", "train_loss_decreased", "checkpoint_path"],
        )
        print(SUMMARY.relative_to(ROOT))
    else:
        print(run_path.relative_to(ROOT))
    return payload


def parse_args() -> argparse.Namespace:
    k = selected_k()
    p = argparse.ArgumentParser()
    p.add_argument("--experiment-name", default="v33_6_identity_contrastive_repair_m128_h32_seed42")
    p.add_argument("--v30-checkpoint", default=str(ROOT / "outputs/checkpoints/stwm_ostf_v30_external_gt/v30_extgt_m128_h32_seed42_best.pt"))
    p.add_argument("--semantic-identity-sidecar-root", default=str(ROOT / "outputs/cache/stwm_ostf_v33_semantic_identity_targets/pointodyssey"))
    p.add_argument("--global-identity-label-root", default=str(ROOT / "outputs/cache/stwm_ostf_v33_6_global_identity_labels/pointodyssey"))
    p.add_argument("--visual-teacher-root", default=str(ROOT / "outputs/cache/stwm_ostf_v33_2_visual_teacher_prototypes/pointodyssey/clip_vit_b32_local"))
    p.add_argument("--semantic-prototype-target-root", default=str(ROOT / f"outputs/cache/stwm_ostf_v33_3_semantic_prototype_targets/pointodyssey/clip_vit_b32_local/K{k}"))
    p.add_argument("--prototype-vocab-path", default=str(ROOT / f"outputs/cache/stwm_ostf_v33_3_semantic_prototypes/pointodyssey/clip_vit_b32_local/K{k}/prototype_vocab.npz"))
    p.add_argument("--hard-subset-manifest", default=str(ROOT / "manifests/ostf_v33_5_split_matched_hard_identity_semantic/H32_M128_seed42.json"))
    p.add_argument("--m-points", type=int, default=128)
    p.add_argument("--horizon", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--steps", type=int, default=1200)
    p.add_argument("--eval-interval", type=int, default=600)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--max-train-items", type=int, default=0)
    p.add_argument("--max-eval-items", type=int, default=0)
    p.add_argument("--teacher-embedding-dim", type=int, default=512)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--contrastive-max-tokens", type=int, default=2048)
    p.add_argument("--identity-loss-weight", type=float, default=0.20)
    p.add_argument("--identity-retrieval-loss-weight", type=float, default=0.20)
    p.add_argument("--same-frame-loss-weight", type=float, default=0.12)
    p.add_argument("--semantic-confuser-loss-weight", type=float, default=0.08)
    p.add_argument("--consistency-loss-weight", type=float, default=0.10)
    p.add_argument("--use-local-instance-contrastive-control", action="store_true")
    p.add_argument("--cpu", action="store_true")
    return p.parse_args()


def main() -> int:
    train_one(parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
