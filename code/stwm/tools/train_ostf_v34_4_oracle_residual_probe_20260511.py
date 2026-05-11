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
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))

from stwm.modules.ostf_v34_3_pointwise_unit_residual_world_model import PointwiseUnitResidualWorldModelV343
from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, write_doc
from stwm.tools.ostf_v30_external_gt_schema_20260508 import utc_now
from stwm.tools.train_ostf_v33_2_visual_semantic_identity_20260509 import move_batch
from stwm.tools.train_ostf_v34_1_identity_bound_semantic_trace_units_20260511 import BINDING_ROOT, V341Dataset, collate_v341
from stwm.tools.train_ostf_v34_2_dual_source_semantic_trace_units_20260511 import pairwise_binding_loss
from stwm.tools.train_ostf_v34_2_pointwise_no_unit_baseline_20260511 import CKPT_DIR as POINT_CKPT_DIR, SUMMARY as POINT_TRAIN
from stwm.tools.train_ostf_v34_3_pointwise_unit_residual_20260511 import cosine_loss
from stwm.tools.train_ostf_v34_semantic_trace_units_20260510 import GLOBAL_ROOT, IDENTITY_ROOT, MASK_ROOT, MEAS_ROOT, contrastive_loss


UTILITY_ROOT = ROOT / "outputs/cache/stwm_ostf_v34_4_residual_utility_targets/pointodyssey"
CKPT_DIR = ROOT / "outputs/checkpoints/stwm_ostf_v34_4_oracle_residual_probe_h32_m128"
SUMMARY = ROOT / "reports/stwm_ostf_v34_4_oracle_residual_probe_train_summary_20260511.json"
DOC = ROOT / "docs/STWM_OSTF_V34_4_ORACLE_RESIDUAL_PROBE_TRAIN_SUMMARY_20260511.md"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class ResidualUtilityDataset(V341Dataset):
    def __init__(self, split: str, args: argparse.Namespace, *, max_items: int | None = None) -> None:
        super().__init__(split, args, max_items=max_items)
        self.utility_root = Path(args.residual_utility_target_root)
        if not self.utility_root.is_absolute():
            self.utility_root = ROOT / self.utility_root
        keep = []
        for entry in self.base.entries:
            uid = Path(entry["cache_path"]).stem
            if (self.utility_root / split / f"{uid}.npz").exists():
                keep.append(entry)
        self.base.entries = keep[:max_items] if max_items is not None else keep
        if not self.base.entries:
            raise RuntimeError(f"No V34.4 residual utility targets for split={split}")

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = super().__getitem__(idx)
        uid = str(item["uid"])
        z = np.load(self.utility_root / self.split / f"{uid}.npz", allow_pickle=True)
        for key in [
            "pointwise_semantic_cosine",
            "pointwise_identity_error",
            "semantic_target_confidence",
            "semantic_hard_mask",
            "changed_mask",
            "stable_mask",
            "stable_suppress_mask",
            "identity_hard_mask",
            "residual_semantic_utility_mask",
            "residual_identity_utility_mask",
            "residual_gate_target",
            "residual_gate_available_mask",
        ]:
            arr = np.asarray(z[key])
            item[key] = torch.from_numpy(arr.astype(np.float32 if arr.dtype.kind == "f" else bool))
        return item


def collate_v344(batch: list[dict[str, Any]]) -> dict[str, Any]:
    out = collate_v341(batch)
    for key in [
        "pointwise_semantic_cosine",
        "pointwise_identity_error",
        "semantic_target_confidence",
        "semantic_hard_mask",
        "changed_mask",
        "stable_mask",
        "stable_suppress_mask",
        "identity_hard_mask",
        "residual_semantic_utility_mask",
        "residual_identity_utility_mask",
        "residual_gate_target",
        "residual_gate_available_mask",
    ]:
        out[key] = torch.stack([b[key] for b in batch], dim=0)
    return out


def make_loader(split: str, args: argparse.Namespace, *, shuffle: bool) -> DataLoader:
    ds = ResidualUtilityDataset(split, args)
    return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle, num_workers=args.num_workers, pin_memory=torch.cuda.is_available(), collate_fn=collate_v344)


def load_pointwise_into_residual(model: PointwiseUnitResidualWorldModelV343) -> None:
    tr = json.loads(POINT_TRAIN.read_text(encoding="utf-8"))
    ckpt = ROOT / tr.get("checkpoint_path", str(POINT_CKPT_DIR / "v34_2_pointwise_no_unit_m128_h32_seed42_best.pt"))
    ck = torch.load(ckpt, map_location="cpu")
    model.load_state_dict(ck["model"], strict=False)


def freeze_pointwise_base(model: PointwiseUnitResidualWorldModelV343) -> None:
    for p in model.parameters():
        p.requires_grad_(False)
    for module in [
        model.tokenizer,
        model.factorized_state,
        model.unit_rollout,
        model.identity_to_hidden,
        model.unit_memory,
        model.semantic_residual_head,
        model.identity_residual_head,
        model.semantic_uncertainty_head,
    ]:
        for p in module.parameters():
            p.requires_grad_(True)


def oracle_outputs(out: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    sem_gate = batch["residual_semantic_utility_mask"].float()
    id_gate = batch["residual_identity_utility_mask"].float()
    final_sem = F.normalize(out["pointwise_semantic_belief"] + sem_gate[..., None] * out["unit_semantic_residual"], dim=-1)
    final_id = out["pointwise_identity_belief"] + id_gate * out["unit_identity_residual"]
    return final_sem, final_id


def loss_fn(out: dict[str, torch.Tensor], batch: dict[str, torch.Tensor], args: argparse.Namespace) -> tuple[torch.Tensor, dict[str, float]]:
    final_sem, final_id = oracle_outputs(out, batch)
    sem_pos = batch["residual_semantic_utility_mask"].bool() & batch["fut_teacher_available_mask"].bool()
    stable = batch["stable_suppress_mask"].bool() & batch["fut_teacher_available_mask"].bool()
    sem_weight = batch["semantic_target_confidence"].float().clamp(0.05, 1.0)
    sem_loss = cosine_loss(final_sem, batch["fut_teacher_embedding"], sem_pos, sem_weight) if bool(sem_pos.any()) else final_sem.sum() * 0.0
    residual_direct = cosine_loss(out["unit_semantic_residual"], batch["fut_teacher_embedding"], sem_pos, sem_weight) if bool(sem_pos.any()) else final_sem.sum() * 0.0
    stable_loss = cosine_loss(final_sem, out["pointwise_semantic_belief"].detach(), stable, torch.ones_like(sem_weight)) if bool(stable.any()) else final_sem.sum() * 0.0
    id_pos = batch["residual_identity_utility_mask"].bool() & batch["fut_instance_available_mask"].bool()
    same = batch["fut_same_instance_as_obs"].float()
    id_bce = F.binary_cross_entropy_with_logits(final_id, same, reduction="none")
    id_loss = (id_bce * id_pos.float()).sum() / id_pos.float().sum().clamp_min(1.0) if bool(id_pos.any()) else final_id.sum() * 0.0
    contr = contrastive_loss(out["identity_embedding"], batch["fut_global_instance_id"].long(), batch["fut_global_instance_available_mask"].bool())
    bind, bind_stats = pairwise_binding_loss(out, batch)
    assign = out["point_to_unit_assignment"].clamp_min(1e-8)
    usage = assign.mean(dim=1).clamp_min(1e-8)
    usage_entropy = -(usage * usage.log()).sum(dim=-1).mean() / np.log(assign.shape[-1])
    anti_collapse = 1.0 - usage_entropy
    total = (
        args.semantic_utility_weight * sem_loss
        + 0.4 * residual_direct
        + args.stable_preservation_weight * stable_loss
        + 0.8 * id_loss
        + 0.15 * contr
        + 0.4 * bind
        + 0.2 * anti_collapse
    )
    stats = {
        "loss": float(total.detach().cpu()),
        "residual_semantic_utility_loss": float(sem_loss.detach().cpu()),
        "direct_residual_content_loss": float(residual_direct.detach().cpu()),
        "stable_pointwise_preservation_loss": float(stable_loss.detach().cpu()),
        "residual_identity_utility_loss": float(id_loss.detach().cpu()),
        "identity_contrastive": float(contr.detach().cpu()),
        "unit_anti_collapse": float(anti_collapse.detach().cpu()),
    }
    stats.update(bind_stats)
    return total, stats


def train_one(args: argparse.Namespace) -> dict[str, Any]:
    set_seed(args.seed)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    loader = make_loader("train", args, shuffle=True)
    model = PointwiseUnitResidualWorldModelV343(args.v30_checkpoint, teacher_embedding_dim=args.teacher_embedding_dim, units=args.trace_units, horizon=args.horizon).to(device)
    load_pointwise_into_residual(model)
    freeze_pointwise_base(model)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=1e-4)
    losses = []
    start = time.time()
    it = iter(loader)
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
        losses.append({"step": float(step), **stats})
    ckpt = CKPT_DIR / "v34_4_oracle_residual_probe_m128_h32_seed42_best.pt"
    torch.save({"model": model.state_dict(), "args": vars(args), "step": args.steps}, ckpt)
    payload = {
        "generated_at_utc": utc_now(),
        "oracle_residual_probe_ran": True,
        "fresh_training_completed": True,
        "checkpoint_path": str(ckpt.relative_to(ROOT)),
        "steps": args.steps,
        "batch_size": args.batch_size,
        "train_sample_count": len(loader.dataset),
        "v30_backbone_frozen": model.v30_backbone_frozen,
        "pointwise_base_frozen": True,
        "oracle_gate_training": True,
        "future_leakage_detected": False,
        "train_loss_first": losses[0] if losses else None,
        "train_loss_last": losses[-1] if losses else None,
        "train_loss_decreased": bool(losses and losses[-1]["loss"] <= losses[0]["loss"]),
        "duration_seconds": float(time.time() - start),
        "loss_tail": losses[-10:],
    }
    dump_json(SUMMARY, payload)
    write_doc(DOC, "STWM OSTF V34.4 Oracle Residual Probe Train Summary", payload, ["oracle_residual_probe_ran", "fresh_training_completed", "checkpoint_path", "train_sample_count", "v30_backbone_frozen", "pointwise_base_frozen", "oracle_gate_training", "future_leakage_detected", "train_loss_decreased"])
    print(SUMMARY.relative_to(ROOT))
    return payload


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--v30-checkpoint", default=str(ROOT / "outputs/checkpoints/stwm_ostf_v30_external_gt/v30_extgt_m128_h32_seed42_best.pt"))
    p.add_argument("--semantic-measurement-bank-root", default=str(MEAS_ROOT))
    p.add_argument("--semantic-identity-sidecar-root", default=str(IDENTITY_ROOT))
    p.add_argument("--global-identity-label-root", default=str(GLOBAL_ROOT))
    p.add_argument("--unit-identity-binding-root", default=str(BINDING_ROOT))
    p.add_argument("--residual-utility-target-root", default=str(UTILITY_ROOT))
    p.add_argument("--hard-mask-manifest", default=str(MASK_ROOT / "H32_M128_seed42.json"))
    p.add_argument("--m-points", type=int, default=128)
    p.add_argument("--horizon", type=int, default=32)
    p.add_argument("--teacher-embedding-dim", type=int, default=768)
    p.add_argument("--trace-units", type=int, default=16)
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--semantic-utility-weight", type=float, default=1.5)
    p.add_argument("--stable-preservation-weight", type=float, default=0.8)
    p.add_argument("--cpu", action="store_true")
    return p.parse_args()


def main() -> int:
    train_one(parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
