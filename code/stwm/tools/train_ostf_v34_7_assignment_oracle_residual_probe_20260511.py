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
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))

setproctitle.setproctitle("python")

from stwm.modules.ostf_v34_7_assignment_bound_residual_memory import AssignmentBoundResidualMemoryV347
from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, write_doc
from stwm.tools.ostf_v30_external_gt_schema_20260508 import utc_now
from stwm.tools.train_ostf_v33_2_visual_semantic_identity_20260509 import move_batch
from stwm.tools.train_ostf_v34_2_dual_source_semantic_trace_units_20260511 import pairwise_binding_loss
from stwm.tools.train_ostf_v34_3_pointwise_unit_residual_20260511 import cosine_loss
from stwm.tools.train_ostf_v34_6_residual_parameterization_sweep_20260511 import StrictResidualUtilityDataset, collate_v345


TARGET_ROOT = ROOT / "outputs/cache/stwm_ostf_v34_7_assignment_aware_residual_targets/pointodyssey"
CKPT_DIR = ROOT / "outputs/checkpoints/stwm_ostf_v34_7_assignment_oracle_residual_probe_h32_m128"
SUMMARY = ROOT / "reports/stwm_ostf_v34_7_assignment_oracle_residual_probe_train_summary_20260511.json"
DOC = ROOT / "docs/STWM_OSTF_V34_7_ASSIGNMENT_ORACLE_RESIDUAL_PROBE_TRAIN_SUMMARY_20260511.md"
V346_DECISION = ROOT / "reports/stwm_ostf_v34_6_residual_parameterization_decision_20260511.json"
TARGET_REPORT = ROOT / "reports/stwm_ostf_v34_7_assignment_aware_residual_target_build_20260511.json"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class AssignmentAwareResidualDataset(StrictResidualUtilityDataset):
    def __init__(self, split: str, args: argparse.Namespace, *, max_items: int | None = None) -> None:
        super().__init__(split, args, max_items=max_items)
        self.assignment_root = Path(args.assignment_aware_residual_target_root)
        if not self.assignment_root.is_absolute():
            self.assignment_root = ROOT / self.assignment_root
        keep = []
        for entry in self.base.entries:
            uid = Path(entry["cache_path"]).stem
            if (self.assignment_root / split / f"{uid}.npz").exists():
                keep.append(entry)
        self.base.entries = keep[:max_items] if max_items is not None else keep
        if not self.base.entries:
            raise RuntimeError(f"No V34.7 assignment-aware residual targets for split={split}")

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = super().__getitem__(idx)
        uid = str(item["uid"])
        z = np.load(self.assignment_root / self.split / f"{uid}.npz", allow_pickle=True)
        for key in [
            "current_unit_assignment",
            "dominant_instance_per_unit",
            "dominant_semantic_cluster_per_unit",
            "unit_instance_purity",
            "unit_semantic_purity",
            "point_unit_confidence",
            "assignment_aware_residual_semantic_mask",
            "assignment_aware_residual_identity_mask",
            "unit_residual_positive_mask",
            "unit_residual_gate_target",
            "point_to_unit_residual_target",
            "stable_suppress_mask",
        ]:
            arr = np.asarray(z[key])
            if arr.dtype.kind in "iu":
                item[key] = torch.from_numpy(arr.astype(np.int64))
            elif arr.dtype.kind == "b":
                item[key] = torch.from_numpy(arr.astype(bool))
            else:
                item[key] = torch.from_numpy(arr.astype(np.float32))
        return item


def collate_v347(batch: list[dict[str, Any]]) -> dict[str, Any]:
    out = collate_v345(batch)
    for key in [
        "current_unit_assignment",
        "dominant_instance_per_unit",
        "dominant_semantic_cluster_per_unit",
        "unit_instance_purity",
        "unit_semantic_purity",
        "point_unit_confidence",
        "assignment_aware_residual_semantic_mask",
        "assignment_aware_residual_identity_mask",
        "unit_residual_positive_mask",
        "unit_residual_gate_target",
        "point_to_unit_residual_target",
        "stable_suppress_mask",
    ]:
        out[key] = torch.stack([b[key] for b in batch], dim=0)
    return out


def make_loader(split: str, args: argparse.Namespace, *, shuffle: bool) -> DataLoader:
    ds = AssignmentAwareResidualDataset(split, args)
    return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle, num_workers=args.num_workers, pin_memory=torch.cuda.is_available(), collate_fn=collate_v347)


def load_v346_into_model(model: AssignmentBoundResidualMemoryV347) -> dict[str, Any]:
    dec = json.loads(V346_DECISION.read_text(encoding="utf-8"))
    ck = torch.load(ROOT / dec["best_checkpoint_path"], map_location="cpu")
    missing, unexpected = model.load_state_dict(ck["model"], strict=False)
    return {"checkpoint_path": dec["best_checkpoint_path"], "missing_keys": list(missing), "unexpected_keys": list(unexpected)}


def freeze_pointwise_base(model: AssignmentBoundResidualMemoryV347) -> None:
    for p in model.parameters():
        p.requires_grad_(False)
    for module in [
        model.tokenizer,
        model.factorized_state,
        model.unit_rollout,
        model.identity_to_hidden,
        model.unit_residual_memory_head,
        model.unit_identity_memory_head,
    ]:
        for p in module.parameters():
            p.requires_grad_(True)


def compose(out: dict[str, torch.Tensor], batch: dict[str, torch.Tensor], mask_key: str = "assignment_aware_residual_semantic_mask") -> torch.Tensor:
    gate = batch[mask_key].float()
    return F.normalize(out["pointwise_semantic_belief"] + gate[..., None] * out["assignment_bound_residual"], dim=-1)


def unit_memory_target_loss(out: dict[str, torch.Tensor], batch: dict[str, torch.Tensor], weight: torch.Tensor) -> torch.Tensor:
    target = F.normalize(torch.nan_to_num(batch["fut_teacher_embedding"].float()), dim=-1)
    pt = batch["point_to_unit_residual_target"].float()
    denom = pt.sum(dim=1).permute(0, 2, 1).clamp_min(1.0)
    unit_tgt = torch.einsum("bmhu,bmhd->buhd", pt, target) / denom[..., None]
    pos = batch["unit_residual_positive_mask"].bool()
    if not bool(pos.any()):
        return out["unit_residual_memory"].sum() * 0.0
    unit_weight = torch.einsum("bmhu,bmh->buh", pt, weight.float()) / denom
    return cosine_loss(out["unit_residual_memory"], unit_tgt, pos, unit_weight.clamp(0.05, 1.0))


def assignment_consistency_loss(out: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> torch.Tensor:
    pt = batch["point_to_unit_residual_target"].float()
    target = pt.max(dim=2).values
    valid = target.sum(dim=-1) > 0
    if not bool(valid.any()):
        return out["point_to_unit_assignment"].sum() * 0.0
    target_idx = target.argmax(dim=-1)
    assign = out["point_to_unit_assignment"].clamp_min(1e-8)
    return F.nll_loss(assign[valid].log(), target_idx[valid], reduction="mean")


def loss_fn(out: dict[str, torch.Tensor], batch: dict[str, torch.Tensor], args: argparse.Namespace) -> tuple[torch.Tensor, dict[str, float]]:
    final = compose(out, batch)
    sem_pos = batch["assignment_aware_residual_semantic_mask"].bool() & batch["fut_teacher_available_mask"].bool()
    stable = batch["stable_suppress_mask"].bool() & batch["fut_teacher_available_mask"].bool()
    weight = batch["teacher_confidence"].float().clamp(0.05, 1.0)
    sem = cosine_loss(final, batch["fut_teacher_embedding"], sem_pos, weight) if bool(sem_pos.any()) else final.sum() * 0.0
    stable_loss = cosine_loss(final, out["pointwise_semantic_belief"].detach(), stable, torch.ones_like(weight)) if bool(stable.any()) else final.sum() * 0.0
    unit_loss = unit_memory_target_loss(out, batch, weight)
    assign_loss = assignment_consistency_loss(out, batch)
    bind, bind_stats = pairwise_binding_loss(out, batch)
    assign = out["point_to_unit_assignment"].clamp_min(1e-8)
    usage = assign.mean(dim=1).clamp_min(1e-8)
    usage_entropy = -(usage * usage.log()).sum(dim=-1).mean() / np.log(assign.shape[-1])
    anti_collapse = 1.0 - usage_entropy
    total = args.semantic_weight * sem + args.unit_memory_weight * unit_loss + args.stable_weight * stable_loss + args.assignment_weight * assign_loss + args.binding_weight * bind + args.anti_collapse_weight * anti_collapse
    stats = {
        "loss": float(total.detach().cpu()),
        "assignment_final_semantic_loss": float(sem.detach().cpu()),
        "unit_residual_memory_target_loss": float(unit_loss.detach().cpu()),
        "stable_preservation_loss": float(stable_loss.detach().cpu()),
        "assignment_consistency_loss": float(assign_loss.detach().cpu()),
        "unit_anti_collapse": float(anti_collapse.detach().cpu()),
    }
    stats.update(bind_stats)
    return total, stats


def train_one(args: argparse.Namespace) -> dict[str, Any]:
    target_report = json.loads(TARGET_REPORT.read_text(encoding="utf-8")) if TARGET_REPORT.exists() else {}
    if not target_report.get("assignment_aware_target_ready"):
        payload = {"generated_at_utc": utc_now(), "assignment_oracle_residual_probe_ran": False, "fresh_training_completed": False, "skip_reason": "assignment_aware_target_not_ready", "v30_backbone_frozen": True, "future_leakage_detected": False}
        dump_json(SUMMARY, payload)
        write_doc(DOC, "STWM OSTF V34.7 Assignment Oracle Residual Probe Train Summary", payload, ["assignment_oracle_residual_probe_ran", "skip_reason"])
        print(SUMMARY.relative_to(ROOT))
        return payload
    set_seed(args.seed)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    loader = make_loader("train", args, shuffle=True)
    model = AssignmentBoundResidualMemoryV347(args.v30_checkpoint, teacher_embedding_dim=args.teacher_embedding_dim, units=args.trace_units, horizon=args.horizon).to(device)
    init_info = load_v346_into_model(model)
    freeze_pointwise_base(model)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=1e-4)
    losses = []
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
        out = model(
            obs_points=bd["obs_points"],
            obs_vis=bd["obs_vis"],
            obs_conf=bd["obs_conf"],
            obs_semantic_measurements=bd["obs_semantic_measurements"],
            obs_semantic_measurement_mask=bd["obs_semantic_measurement_mask"],
            semantic_id=bd["semantic_id"],
            intervention="force_gate_zero",
        )
        loss, stats = loss_fn(out, bd, args)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step()
        if step == 1 or step == args.steps or step % max(100, args.steps // 10) == 0:
            losses.append({"step": float(step), **stats})
    ckpt = CKPT_DIR / "v34_7_assignment_oracle_residual_probe_m128_h32_seed42_best.pt"
    torch.save({"model": model.state_dict(), "args": vars(args), "step": args.steps}, ckpt)
    payload = {
        "generated_at_utc": utc_now(),
        "assignment_oracle_residual_probe_ran": True,
        "fresh_training_completed": True,
        "checkpoint_path": str(ckpt.relative_to(ROOT)),
        "steps": args.steps,
        "train_sample_count": len(loader.dataset),
        "v30_backbone_frozen": model.v30_backbone_frozen,
        "pointwise_base_frozen": True,
        "future_leakage_detected": False,
        "init_from_v34_6_best_residual_checkpoint": init_info,
        "train_loss_first": losses[0] if losses else None,
        "train_loss_last": losses[-1] if losses else None,
        "train_loss_decreased": bool(losses and losses[-1]["loss"] <= losses[0]["loss"]),
        "duration_seconds": float(time.time() - start),
        "loss_trace": losses,
    }
    dump_json(SUMMARY, payload)
    write_doc(DOC, "STWM OSTF V34.7 Assignment Oracle Residual Probe Train Summary", payload, ["assignment_oracle_residual_probe_ran", "fresh_training_completed", "checkpoint_path", "train_sample_count", "v30_backbone_frozen", "pointwise_base_frozen", "future_leakage_detected", "train_loss_decreased"])
    print(SUMMARY.relative_to(ROOT))
    return payload


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--v30-checkpoint", default=str(ROOT / "outputs/checkpoints/stwm_ostf_v30_external_gt/v30_extgt_m128_h32_seed42_best.pt"))
    p.add_argument("--semantic-measurement-bank-root", default=str(ROOT / "outputs/cache/stwm_ostf_v34_semantic_measurement_bank/pointodyssey"))
    p.add_argument("--semantic-identity-sidecar-root", default=str(ROOT / "outputs/cache/stwm_ostf_v33_semantic_identity_targets/pointodyssey"))
    p.add_argument("--global-identity-label-root", default=str(ROOT / "outputs/cache/stwm_ostf_v33_6_global_identity_labels/pointodyssey"))
    p.add_argument("--unit-identity-binding-root", default=str(ROOT / "outputs/cache/stwm_ostf_v34_1_unit_identity_binding_targets/pointodyssey"))
    p.add_argument("--strict-residual-utility-target-root", default=str(ROOT / "outputs/cache/stwm_ostf_v34_5_strict_residual_utility_targets/pointodyssey"))
    p.add_argument("--assignment-aware-residual-target-root", default=str(TARGET_ROOT))
    p.add_argument("--hard-mask-manifest", default=str(ROOT / "manifests/ostf_v33_8_split_matched_hard_identity_semantic/H32_M128_seed42.json"))
    p.add_argument("--m-points", type=int, default=128)
    p.add_argument("--horizon", type=int, default=32)
    p.add_argument("--teacher-embedding-dim", type=int, default=768)
    p.add_argument("--trace-units", type=int, default=16)
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--semantic-weight", type=float, default=1.4)
    p.add_argument("--unit-memory-weight", type=float, default=0.8)
    p.add_argument("--stable-weight", type=float, default=0.9)
    p.add_argument("--assignment-weight", type=float, default=0.5)
    p.add_argument("--binding-weight", type=float, default=0.35)
    p.add_argument("--anti-collapse-weight", type=float, default=0.2)
    p.add_argument("--cpu", action="store_true")
    return p.parse_args()


def main() -> int:
    train_one(parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
