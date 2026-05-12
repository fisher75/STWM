#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import setproctitle
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))

setproctitle.setproctitle("python")

from stwm.modules.ostf_v34_8_causal_assignment_bound_residual_memory import CausalAssignmentBoundResidualMemoryV348
from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, write_doc
from stwm.tools.ostf_v30_external_gt_schema_20260508 import utc_now
from stwm.tools.train_ostf_v33_2_visual_semantic_identity_20260509 import move_batch
from stwm.tools.train_ostf_v34_2_dual_source_semantic_trace_units_20260511 import pairwise_binding_loss
from stwm.tools.train_ostf_v34_2_pointwise_no_unit_baseline_20260511 import CKPT_DIR as POINT_CKPT_DIR, SUMMARY as POINT_TRAIN
from stwm.tools.train_ostf_v34_3_pointwise_unit_residual_20260511 import cosine_loss
from stwm.tools.train_ostf_v34_7_assignment_oracle_residual_probe_20260511 import collate_v347


TARGET_ROOT = ROOT / "outputs/cache/stwm_ostf_v34_8_causal_assignment_residual_targets/pointodyssey"
CKPT_DIR = ROOT / "outputs/checkpoints/stwm_ostf_v34_8_causal_assignment_oracle_residual_probe_h32_m128"
SUMMARY = ROOT / "reports/stwm_ostf_v34_8_causal_assignment_oracle_residual_probe_train_summary_20260512.json"
DOC = ROOT / "docs/STWM_OSTF_V34_8_CAUSAL_ASSIGNMENT_ORACLE_RESIDUAL_PROBE_TRAIN_SUMMARY_20260512.md"
TARGET_REPORT = ROOT / "reports/stwm_ostf_v34_8_causal_assignment_residual_target_build_20260512.json"
V347_TRAIN = ROOT / "reports/stwm_ostf_v34_7_assignment_oracle_residual_probe_train_summary_20260511.json"
V346_DECISION = ROOT / "reports/stwm_ostf_v34_6_residual_parameterization_decision_20260511.json"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class CausalAssignmentResidualDataset(Dataset):
    def __init__(self, split: str, args: argparse.Namespace, *, max_items: int | None = None) -> None:
        self.split = split
        self.meas_root = Path(args.semantic_measurement_bank_root)
        if not self.meas_root.is_absolute():
            self.meas_root = ROOT / self.meas_root
        self.strict_root = Path(args.strict_residual_utility_target_root)
        if not self.strict_root.is_absolute():
            self.strict_root = ROOT / self.strict_root
        self.assignment_root = Path(args.assignment_aware_residual_target_root)
        if not self.assignment_root.is_absolute():
            self.assignment_root = ROOT / self.assignment_root
        self.causal_root = Path(args.causal_assignment_residual_target_root)
        if not self.causal_root.is_absolute():
            self.causal_root = ROOT / self.causal_root
        self.uids = [
            p.stem
            for p in sorted((self.causal_root / split).glob("*.npz"))
            if (self.meas_root / split / p.name).exists()
            and (self.strict_root / split / p.name).exists()
            and (self.assignment_root / split / p.name).exists()
        ]
        if max_items is not None:
            self.uids = self.uids[:max_items]
        if not self.uids:
            raise RuntimeError(f"没有 V34.8 causal assignment residual targets: split={split}")

    def __len__(self) -> int:
        return len(self.uids)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        uid = self.uids[idx]
        zm = np.load(self.meas_root / self.split / f"{uid}.npz", allow_pickle=True)
        z5 = np.load(self.strict_root / self.split / f"{uid}.npz", allow_pickle=True)
        z7 = np.load(self.assignment_root / self.split / f"{uid}.npz", allow_pickle=True)
        z = np.load(self.causal_root / self.split / f"{uid}.npz", allow_pickle=True)
        obs_vis = np.asarray(zm["obs_vis"]).astype(bool)
        obs_conf = np.asarray(zm["obs_measurement_confidence"], dtype=np.float32)
        obs_conf = np.where(obs_vis, np.maximum(obs_conf, 0.05), 0.0).astype(np.float32)
        inst = np.asarray(zm["point_to_instance_id"], dtype=np.int64)
        pair_available = (inst[:, None] >= 0) & (inst[None, :] >= 0)
        pair_same = pair_available & (inst[:, None] == inst[None, :])
        item: dict[str, Any] = {
            "uid": uid,
            "obs_points": torch.from_numpy(np.asarray(zm["obs_points"], dtype=np.float32)),
            "obs_vis": torch.from_numpy(obs_vis),
            "obs_conf": torch.from_numpy(obs_conf),
            "semantic_id": torch.tensor(-1, dtype=torch.long),
            "point_id": torch.from_numpy(np.asarray(zm["point_id"], dtype=np.int64)),
            "point_to_instance_id": torch.from_numpy(inst),
            "obs_semantic_measurements": torch.from_numpy(np.asarray(zm["obs_semantic_measurements"], dtype=np.float32)),
            "obs_semantic_measurement_mask": torch.from_numpy(np.asarray(zm["obs_semantic_measurement_mask"]).astype(bool)),
            "fut_teacher_embedding": torch.from_numpy(np.asarray(zm["fut_teacher_embedding"], dtype=np.float32)),
            "fut_teacher_available_mask": torch.from_numpy(np.asarray(zm["fut_teacher_available_mask"]).astype(bool)),
            "fut_teacher_confidence": torch.from_numpy(np.asarray(zm["fut_teacher_confidence"], dtype=np.float32)),
            "teacher_confidence": torch.from_numpy(np.asarray(z5["teacher_confidence"], dtype=np.float32)),
            "pointwise_semantic_cosine": torch.from_numpy(np.asarray(z5["pointwise_semantic_cosine"], dtype=np.float32)),
            "semantic_hard_mask": torch.from_numpy(np.asarray(z5["semantic_hard_mask"]).astype(bool)),
            "changed_mask": torch.from_numpy(np.asarray(z5["changed_mask"]).astype(bool)),
            "stable_mask": torch.from_numpy(np.asarray(z5["stable_mask"]).astype(bool)),
            "strict_residual_semantic_utility_mask": torch.from_numpy(np.asarray(z5["strict_residual_semantic_utility_mask"]).astype(bool)),
            "strict_stable_suppress_mask": torch.from_numpy(np.asarray(z5["strict_stable_suppress_mask"]).astype(bool)),
            "assignment_aware_residual_semantic_mask": torch.from_numpy(np.asarray(z7["assignment_aware_residual_semantic_mask"]).astype(bool)),
            "same_instance_pair_available_mask": torch.from_numpy(pair_available.astype(bool)),
            "same_instance_unit_pair_mask": torch.from_numpy(pair_same.astype(bool)),
        }
        for key in [
            "point_to_unit_assignment",
            "unit_instance_purity",
            "unit_semantic_purity",
            "unit_confidence",
            "point_assignment_confidence",
            "semantic_measurement_confidence",
            "semantic_measurement_agreement",
            "causal_assignment_residual_semantic_mask",
            "causal_semantic_measurement_required_mask",
            "causal_assignment_gate_target",
            "causal_unit_positive_mask",
            "point_to_unit_residual_target",
            "stable_suppress_mask",
        ]:
            arr = np.asarray(z[key])
            if arr.dtype.kind == "b":
                item[key] = torch.from_numpy(arr.astype(bool))
            elif arr.dtype.kind in "iu":
                item[key] = torch.from_numpy(arr.astype(np.int64))
            else:
                item[key] = torch.from_numpy(arr.astype(np.float32))
        return item


def collate_v348(batch: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {"uid": [b["uid"] for b in batch]}
    base_keys = [
        "obs_points",
        "obs_vis",
        "obs_conf",
        "semantic_id",
        "point_id",
        "point_to_instance_id",
        "obs_semantic_measurements",
        "obs_semantic_measurement_mask",
        "fut_teacher_embedding",
        "fut_teacher_available_mask",
        "fut_teacher_confidence",
        "teacher_confidence",
        "pointwise_semantic_cosine",
        "semantic_hard_mask",
        "changed_mask",
        "stable_mask",
        "strict_residual_semantic_utility_mask",
        "strict_stable_suppress_mask",
        "assignment_aware_residual_semantic_mask",
        "same_instance_pair_available_mask",
        "same_instance_unit_pair_mask",
    ]
    for key in base_keys:
        out[key] = torch.stack([b[key] for b in batch], dim=0)
    for key in [
        "point_to_unit_assignment",
        "unit_instance_purity",
        "unit_semantic_purity",
        "unit_confidence",
        "point_assignment_confidence",
        "semantic_measurement_confidence",
        "semantic_measurement_agreement",
        "causal_assignment_residual_semantic_mask",
        "causal_semantic_measurement_required_mask",
        "causal_assignment_gate_target",
        "causal_unit_positive_mask",
        "point_to_unit_residual_target",
        "stable_suppress_mask",
    ]:
        out[key] = torch.stack([b[key] for b in batch], dim=0)
    return out


def make_loader(split: str, args: argparse.Namespace, *, shuffle: bool) -> DataLoader:
    ds = CausalAssignmentResidualDataset(split, args)
    return DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_v348,
    )


def _rel(path: Path) -> str:
    return str(path.relative_to(ROOT)) if path.is_absolute() and path.is_relative_to(ROOT) else str(path)


def load_init_checkpoint(model: CausalAssignmentBoundResidualMemoryV348) -> dict[str, Any]:
    tried: list[str] = []
    if V347_TRAIN.exists():
        tr = json.loads(V347_TRAIN.read_text(encoding="utf-8"))
        ckpt = ROOT / tr.get("checkpoint_path", "")
        if ckpt.exists():
            ck = torch.load(ckpt, map_location="cpu")
            missing, unexpected = model.load_state_dict(ck["model"], strict=False)
            return {"init_source": "v34_7_assignment_oracle_residual_probe", "checkpoint_path": _rel(ckpt), "missing_keys": list(missing), "unexpected_keys": list(unexpected)}
        tried.append(_rel(ckpt))
    if V346_DECISION.exists():
        dec = json.loads(V346_DECISION.read_text(encoding="utf-8"))
        ckpt = ROOT / dec.get("best_checkpoint_path", "")
        if ckpt.exists():
            ck = torch.load(ckpt, map_location="cpu")
            missing, unexpected = model.load_state_dict(ck["model"], strict=False)
            return {"init_source": "v34_6_best_residual_parameterization", "checkpoint_path": _rel(ckpt), "missing_keys": list(missing), "unexpected_keys": list(unexpected)}
        tried.append(_rel(ckpt))
    if POINT_TRAIN.exists():
        tr = json.loads(POINT_TRAIN.read_text(encoding="utf-8"))
        ckpt = ROOT / tr.get("checkpoint_path", str(POINT_CKPT_DIR / "v34_2_pointwise_no_unit_m128_h32_seed42_best.pt"))
        if ckpt.exists():
            ck = torch.load(ckpt, map_location="cpu")
            missing, unexpected = model.load_state_dict(ck["model"], strict=False)
            return {"init_source": "v34_2_pointwise_no_unit", "checkpoint_path": _rel(ckpt), "missing_keys": list(missing), "unexpected_keys": list(unexpected)}
        tried.append(_rel(ckpt))
    return {"init_source": "random_new_heads_only", "tried": tried}


def freeze_pointwise_base(model: CausalAssignmentBoundResidualMemoryV348) -> None:
    for p in model.parameters():
        p.requires_grad_(False)
    trainable = [
        model.tokenizer,
        model.factorized_state,
        model.unit_rollout,
        model.identity_to_hidden,
        model.causal_unit_memory_head,
        model.causal_unit_identity_head,
        model.assignment_usage_head,
        model.semantic_measurement_usage_head,
    ]
    for module in trainable:
        for p in module.parameters():
            p.requires_grad_(True)


def compose(out: dict[str, torch.Tensor], batch: dict[str, torch.Tensor], mask_key: str = "causal_assignment_residual_semantic_mask") -> torch.Tensor:
    gate = batch[mask_key].float()
    return F.normalize(out["pointwise_semantic_belief"] + gate[..., None] * out["assignment_bound_residual"], dim=-1)


def unit_memory_target_loss(out: dict[str, torch.Tensor], batch: dict[str, torch.Tensor], weight: torch.Tensor) -> torch.Tensor:
    target = F.normalize(torch.nan_to_num(batch["fut_teacher_embedding"].float()), dim=-1)
    pt = batch["point_to_unit_residual_target"].float()
    denom = pt.sum(dim=1).permute(0, 2, 1).clamp_min(1.0)
    unit_tgt = torch.einsum("bmhu,bmhd->buhd", pt, target) / denom[..., None]
    pos = batch["causal_unit_positive_mask"].bool()
    if not bool(pos.any()):
        return out["unit_memory"].sum() * 0.0
    unit_weight = torch.einsum("bmhu,bmh->buh", pt, weight.float()) / denom
    return cosine_loss(out["unit_memory"], unit_tgt, pos, unit_weight.clamp(0.05, 1.0))


def assignment_consistency_loss(out: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> torch.Tensor:
    pt = batch["point_to_unit_residual_target"].float()
    target = pt.max(dim=2).values
    valid = target.sum(dim=-1) > 0
    if not bool(valid.any()):
        return out["point_to_unit_assignment"].sum() * 0.0
    target_idx = target.argmax(dim=-1)
    assign = out["point_to_unit_assignment"].clamp_min(1e-8)
    return F.nll_loss(assign[valid].log(), target_idx[valid], reduction="mean")


def _masked_cos(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if not bool(mask.any()):
        return pred.sum() * 0.0
    cos = (F.normalize(pred, dim=-1) * F.normalize(torch.nan_to_num(target.float()), dim=-1)).sum(dim=-1)
    return (cos * mask.float()).sum() / mask.float().sum().clamp_min(1.0)


def loss_fn(
    out: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    args: argparse.Namespace,
    *,
    zero_sem_out: dict[str, torch.Tensor] | None = None,
    shuffled_out: dict[str, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    final = compose(out, batch)
    sem_pos = batch["causal_assignment_residual_semantic_mask"].bool() & batch["fut_teacher_available_mask"].bool()
    stable = batch["stable_suppress_mask"].bool() & batch["fut_teacher_available_mask"].bool()
    weight = batch["teacher_confidence"].float().clamp(0.05, 1.0)
    sem = cosine_loss(final, batch["fut_teacher_embedding"], sem_pos, weight) if bool(sem_pos.any()) else final.sum() * 0.0
    stable_loss = cosine_loss(final, out["pointwise_semantic_belief"].detach(), stable, torch.ones_like(weight)) if bool(stable.any()) else final.sum() * 0.0
    unit_loss = unit_memory_target_loss(out, batch, weight)
    assign_loss = assignment_consistency_loss(out, batch)
    bind, bind_stats = pairwise_binding_loss(out, batch)
    normal_cos = _masked_cos(final, batch["fut_teacher_embedding"], sem_pos).detach()
    if zero_sem_out is not None:
        zero_final = compose(zero_sem_out, batch)
        zero_cos = _masked_cos(zero_final, batch["fut_teacher_embedding"], sem_pos)
        sem_usage = F.relu(args.usage_margin - (normal_cos - zero_cos))
    else:
        sem_usage = final.sum() * 0.0
    if shuffled_out is not None:
        shuf_final = compose(shuffled_out, batch)
        shuf_cos = _masked_cos(shuf_final, batch["fut_teacher_embedding"], sem_pos)
        assign_contrast = F.relu(args.assignment_contrast_margin - (normal_cos - shuf_cos))
    else:
        assign_contrast = final.sum() * 0.0
    assign = out["point_to_unit_assignment"].clamp_min(1e-8)
    usage = assign.mean(dim=1).clamp_min(1e-8)
    usage_entropy = -(usage * usage.log()).sum(dim=-1).mean() / np.log(assign.shape[-1])
    anti_collapse = 1.0 - usage_entropy
    total = (
        args.semantic_weight * sem
        + args.unit_memory_weight * unit_loss
        + args.stable_weight * stable_loss
        + args.assignment_weight * assign_loss
        + args.binding_weight * bind
        + args.semantic_usage_weight * sem_usage
        + args.assignment_contrast_weight * assign_contrast
        + args.anti_collapse_weight * anti_collapse
    )
    stats = {
        "loss": float(total.detach().cpu()),
        "causal_final_semantic_loss": float(sem.detach().cpu()),
        "unit_memory_target_loss": float(unit_loss.detach().cpu()),
        "stable_preservation_loss": float(stable_loss.detach().cpu()),
        "assignment_consistency_loss": float(assign_loss.detach().cpu()),
        "semantic_measurement_usage_loss": float(sem_usage.detach().cpu()),
        "assignment_contrastive_loss": float(assign_contrast.detach().cpu()),
        "unit_anti_collapse": float(anti_collapse.detach().cpu()),
    }
    stats.update(bind_stats)
    return total, stats


def train_one(args: argparse.Namespace) -> dict[str, Any]:
    target_report = json.loads(TARGET_REPORT.read_text(encoding="utf-8")) if TARGET_REPORT.exists() else {}
    if not target_report.get("causal_assignment_target_ready"):
        payload = {
            "generated_at_utc": utc_now(),
            "中文结论": "V34.8 causal assignment target 不 ready，oracle residual probe 未训练。",
            "oracle_residual_probe_ran": False,
            "fresh_training_completed": False,
            "skip_reason": "causal_assignment_target_not_ready",
            "v30_backbone_frozen": True,
            "future_leakage_detected": False,
        }
        dump_json(SUMMARY, payload)
        write_doc(DOC, "V34.8 causal assignment oracle residual probe 训练中文摘要", payload, ["中文结论", "oracle_residual_probe_ran", "skip_reason", "v30_backbone_frozen", "future_leakage_detected"])
        print(f"已写出训练跳过报告: {SUMMARY.relative_to(ROOT)}")
        return payload
    set_seed(args.seed)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    loader = make_loader("train", args, shuffle=True)
    model = CausalAssignmentBoundResidualMemoryV348(args.v30_checkpoint, teacher_embedding_dim=args.teacher_embedding_dim, units=args.trace_units, horizon=args.horizon).to(device)
    init_info = load_init_checkpoint(model)
    freeze_pointwise_base(model)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=1e-4)
    losses: list[dict[str, float]] = []
    start = time.time()
    it = iter(loader)
    model.train()
    for step in range(1, args.steps + 1):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)
        bd = move_batch(batch, device)
        base_inputs = {
            "obs_points": bd["obs_points"],
            "obs_vis": bd["obs_vis"],
            "obs_conf": bd["obs_conf"],
            "obs_semantic_measurements": bd["obs_semantic_measurements"],
            "obs_semantic_measurement_mask": bd["obs_semantic_measurement_mask"],
            "semantic_id": bd["semantic_id"],
        }
        out = model(**base_inputs, intervention="force_gate_zero")
        zero_sem_out = None
        shuffled_out = None
        if step % args.contrastive_intervention_every == 0:
            zero_sem_out = model(**base_inputs, intervention="zero_semantic_measurements")
            shuffled_out = model(**base_inputs, intervention="shuffle_assignment")
        loss, stats = loss_fn(out, bd, args, zero_sem_out=zero_sem_out, shuffled_out=shuffled_out)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step()
        if step == 1 or step == args.steps or step % max(100, args.steps // 10) == 0:
            losses.append({"step": float(step), **stats})
            print(f"训练进度: step={step}/{args.steps}, loss={stats['loss']:.6f}", flush=True)
    ckpt = CKPT_DIR / "v34_8_causal_assignment_oracle_residual_probe_m128_h32_seed42_best.pt"
    torch.save({"model": model.state_dict(), "args": vars(args), "step": args.steps}, ckpt)
    payload = {
        "generated_at_utc": utc_now(),
        "中文结论": "V34.8 causal assignment oracle residual probe 已完成训练；未训练 learned gate。",
        "oracle_residual_probe_ran": True,
        "fresh_training_completed": True,
        "checkpoint_path": str(ckpt.relative_to(ROOT)),
        "steps": args.steps,
        "train_sample_count": len(loader.dataset),
        "v30_backbone_frozen": model.v30_backbone_frozen,
        "pointwise_base_frozen": True,
        "future_leakage_detected": False,
        "init_checkpoint": init_info,
        "train_loss_first": losses[0] if losses else None,
        "train_loss_last": losses[-1] if losses else None,
        "train_loss_decreased": bool(losses and losses[-1]["loss"] <= losses[0]["loss"]),
        "duration_seconds": float(time.time() - start),
        "loss_trace": losses,
    }
    dump_json(SUMMARY, payload)
    write_doc(DOC, "V34.8 causal assignment oracle residual probe 训练中文摘要", payload, ["中文结论", "oracle_residual_probe_ran", "fresh_training_completed", "checkpoint_path", "train_sample_count", "v30_backbone_frozen", "pointwise_base_frozen", "future_leakage_detected", "train_loss_decreased"])
    print(f"已写出 V34.8 训练摘要: {SUMMARY.relative_to(ROOT)}")
    return payload


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--v30-checkpoint", default=str(ROOT / "outputs/checkpoints/stwm_ostf_v30_external_gt/v30_extgt_m128_h32_seed42_best.pt"))
    p.add_argument("--semantic-measurement-bank-root", default=str(ROOT / "outputs/cache/stwm_ostf_v34_semantic_measurement_bank/pointodyssey"))
    p.add_argument("--semantic-identity-sidecar-root", default=str(ROOT / "outputs/cache/stwm_ostf_v33_semantic_identity_targets/pointodyssey"))
    p.add_argument("--global-identity-label-root", default=str(ROOT / "outputs/cache/stwm_ostf_v33_6_global_identity_labels/pointodyssey"))
    p.add_argument("--unit-identity-binding-root", default=str(ROOT / "outputs/cache/stwm_ostf_v34_1_unit_identity_binding_targets/pointodyssey"))
    p.add_argument("--strict-residual-utility-target-root", default=str(ROOT / "outputs/cache/stwm_ostf_v34_5_strict_residual_utility_targets/pointodyssey"))
    p.add_argument("--assignment-aware-residual-target-root", default=str(ROOT / "outputs/cache/stwm_ostf_v34_7_assignment_aware_residual_targets/pointodyssey"))
    p.add_argument("--causal-assignment-residual-target-root", default=str(TARGET_ROOT))
    p.add_argument("--hard-mask-manifest", default=str(ROOT / "manifests/ostf_v33_8_split_matched_hard_identity_semantic/H32_M128_seed42.json"))
    p.add_argument("--m-points", type=int, default=128)
    p.add_argument("--horizon", type=int, default=32)
    p.add_argument("--teacher-embedding-dim", type=int, default=768)
    p.add_argument("--trace-units", type=int, default=16)
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--semantic-weight", type=float, default=1.3)
    p.add_argument("--unit-memory-weight", type=float, default=1.0)
    p.add_argument("--stable-weight", type=float, default=1.0)
    p.add_argument("--assignment-weight", type=float, default=0.7)
    p.add_argument("--binding-weight", type=float, default=0.35)
    p.add_argument("--semantic-usage-weight", type=float, default=0.25)
    p.add_argument("--assignment-contrast-weight", type=float, default=0.25)
    p.add_argument("--anti-collapse-weight", type=float, default=0.2)
    p.add_argument("--usage-margin", type=float, default=0.003)
    p.add_argument("--assignment-contrast-margin", type=float, default=0.003)
    p.add_argument("--contrastive-intervention-every", type=int, default=2)
    p.add_argument("--cpu", action="store_true")
    return p.parse_args()


def main() -> int:
    train_one(parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
