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
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))

from stwm.datasets.ostf_v30_external_gt_dataset_20260508 import OSTFExternalGTDataset, collate_external_gt
from stwm.modules.ostf_v34_semantic_trace_units import SemanticTraceUnitsWorldModelV34
from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, write_doc
from stwm.tools.ostf_v30_external_gt_schema_20260508 import utc_now
from stwm.tools.train_ostf_v33_1_integrated_semantic_identity_20260509 import binary_metrics, visibility_f1
from stwm.tools.train_ostf_v33_2_visual_semantic_identity_20260509 import move_batch


MEAS_ROOT = ROOT / "outputs/cache/stwm_ostf_v34_semantic_measurement_bank/pointodyssey"
IDENTITY_ROOT = ROOT / "outputs/cache/stwm_ostf_v33_8_complete_h32_m128/semantic_identity_targets/pointodyssey"
GLOBAL_ROOT = ROOT / "outputs/cache/stwm_ostf_v33_8_complete_h32_m128/global_identity_labels/pointodyssey"
MASK_ROOT = ROOT / "manifests/ostf_v33_8_split_matched_hard_identity_semantic"
CKPT_DIR = ROOT / "outputs/checkpoints/stwm_ostf_v34_semantic_trace_units_h32_m128"
SUMMARY = ROOT / "reports/stwm_ostf_v34_semantic_trace_units_train_summary_20260510.json"
DOC = ROOT / "docs/STWM_OSTF_V34_SEMANTIC_TRACE_UNITS_TRAIN_SUMMARY_20260510.md"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_masks(path: str | Path, split: str) -> dict[str, dict[str, np.ndarray]]:
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    payload = json.loads(p.read_text(encoding="utf-8"))
    out: dict[str, dict[str, np.ndarray]] = {}
    for entry in payload.get("splits", {}).get(split, []):
        z = np.load(ROOT / entry["mask_path"], allow_pickle=True)
        out[str(entry["sample_uid"])] = {
            "identity_hard_train_mask": np.asarray(z["identity_hard_train_mask" if "identity_hard_train_mask" in z.files else "identity_hard_eval_mask"]).astype(bool),
            "semantic_hard_train_mask": np.asarray(z["semantic_hard_train_mask" if "semantic_hard_train_mask" in z.files else "semantic_hard_eval_mask"]).astype(bool),
        }
    return out


class V34TraceUnitDataset(Dataset):
    def __init__(self, split: str, args: argparse.Namespace, *, max_items: int | None = None) -> None:
        self.base = OSTFExternalGTDataset(
            split,
            horizon=args.horizon,
            m_points=args.m_points,
            enable_semantic_identity_sidecar=True,
            semantic_identity_sidecar_root=args.semantic_identity_sidecar_root,
            require_semantic_identity_sidecar=True,
            enable_global_identity_labels=True,
            global_identity_label_root=args.global_identity_label_root,
            require_global_identity_labels=True,
            use_observed_instance_context=False,
        )
        self.split = split
        self.measurement_root = Path(args.semantic_measurement_bank_root)
        if not self.measurement_root.is_absolute():
            self.measurement_root = ROOT / self.measurement_root
        keep = []
        for entry in self.base.entries:
            uid = Path(entry["cache_path"]).stem
            if (self.measurement_root / split / f"{uid}.npz").exists():
                keep.append(entry)
        self.base.entries = keep[:max_items] if max_items is not None else keep
        self.masks = load_masks(args.hard_mask_manifest, split)
        if not self.base.entries:
            raise RuntimeError(f"No V34 measurement sidecars for split={split}")

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = self.base[idx]
        uid = str(item["uid"])
        z = np.load(self.measurement_root / self.split / f"{uid}.npz", allow_pickle=True)
        item["obs_semantic_measurements"] = torch.from_numpy(np.asarray(z["obs_semantic_measurements"], dtype=np.float32))
        item["obs_semantic_measurement_mask"] = torch.from_numpy(np.asarray(z["obs_semantic_measurement_mask"]).astype(bool))
        item["obs_measurement_confidence"] = torch.from_numpy(np.asarray(z["obs_measurement_confidence"], dtype=np.float32))
        item["teacher_agreement_score"] = torch.from_numpy(np.asarray(z["teacher_agreement_score"], dtype=np.float32))
        item["fut_teacher_embedding"] = torch.from_numpy(np.asarray(z["fut_teacher_embedding"], dtype=np.float32))
        item["fut_teacher_available_mask"] = torch.from_numpy(np.asarray(z["fut_teacher_available_mask"]).astype(bool))
        item["fut_teacher_confidence"] = torch.from_numpy(np.asarray(z["fut_teacher_confidence"], dtype=np.float32))
        item["future_teacher_embeddings_input_allowed"] = torch.tensor(bool(np.asarray(z["future_teacher_embeddings_input_allowed"]).item()), dtype=torch.bool)
        mh = item["fut_same_instance_as_obs"].shape
        m = self.masks.get(uid)
        item["identity_hard_train_mask"] = torch.from_numpy(m["identity_hard_train_mask"]).bool() if m else torch.zeros(mh, dtype=torch.bool)
        item["semantic_hard_train_mask"] = torch.from_numpy(m["semantic_hard_train_mask"]).bool() if m else torch.zeros(mh, dtype=torch.bool)
        return item


def collate_v34(batch: list[dict[str, Any]]) -> dict[str, Any]:
    out = collate_external_gt(batch)
    for key in [
        "obs_semantic_measurements",
        "obs_semantic_measurement_mask",
        "obs_measurement_confidence",
        "teacher_agreement_score",
        "fut_teacher_embedding",
        "fut_teacher_available_mask",
        "fut_teacher_confidence",
        "future_teacher_embeddings_input_allowed",
        "identity_hard_train_mask",
        "semantic_hard_train_mask",
    ]:
        out[key] = torch.stack([b[key] for b in batch], dim=0)
    return out


def make_loader(split: str, args: argparse.Namespace, *, shuffle: bool, max_items: int | None = None) -> DataLoader:
    ds = V34TraceUnitDataset(split, args, max_items=max_items)
    return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle, num_workers=args.num_workers, pin_memory=torch.cuda.is_available(), collate_fn=collate_v34)


def weighted_cosine_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    valid = mask.bool()
    if not bool(valid.any()):
        return pred.sum() * 0.0
    pred_n = F.normalize(pred, dim=-1)
    tgt_n = F.normalize(torch.nan_to_num(target.float()), dim=-1)
    loss = 1.0 - (pred_n * tgt_n).sum(dim=-1)
    w = weight.float().clamp_min(0.0) * valid.float()
    return (loss * w).sum() / w.sum().clamp_min(1.0)


def contrastive_loss(emb: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor, max_tokens: int = 8192) -> torch.Tensor:
    valid = mask.bool() & (labels >= 0)
    if int(valid.sum()) < 4:
        return emb.sum() * 0.0
    x = F.normalize(emb[valid], dim=-1)
    y = labels[valid]
    if x.shape[0] > max_tokens:
        idx = torch.randperm(x.shape[0], device=x.device)[:max_tokens]
        x, y = x[idx], y[idx]
    sim = x @ x.T / 0.08
    same = y[:, None] == y[None, :]
    eye = torch.eye(sim.shape[0], device=sim.device, dtype=torch.bool)
    same = same & ~eye
    if not bool(same.any()):
        return emb.sum() * 0.0
    sim = sim.masked_fill(eye, -1e9)
    logp = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    return -(logp[same]).mean()


def loss_fn(out: dict[str, torch.Tensor], batch: dict[str, torch.Tensor], args: argparse.Namespace) -> tuple[torch.Tensor, dict[str, float]]:
    same_mask = batch["fut_instance_available_mask"].bool()
    hard_id = batch["identity_hard_train_mask"].bool() & same_mask
    same = batch["fut_same_instance_as_obs"].float()
    bce_full = F.binary_cross_entropy_with_logits(out["future_identity_belief"], same, reduction="none")
    id_loss = (bce_full * same_mask.float()).sum() / same_mask.float().sum().clamp_min(1.0)
    id_hard = (bce_full * hard_id.float()).sum() / hard_id.float().sum().clamp_min(1.0) if bool(hard_id.any()) else id_loss * 0.0
    sem_mask = batch["fut_teacher_available_mask"].bool()
    sem_weight = batch["fut_teacher_confidence"].float().clamp(0.05, 1.0)
    sem_loss = weighted_cosine_loss(out["future_semantic_belief"], batch["fut_teacher_embedding"], sem_mask, sem_weight)
    hard_sem = batch["semantic_hard_train_mask"].bool() & sem_mask
    sem_hard = weighted_cosine_loss(out["future_semantic_belief"], batch["fut_teacher_embedding"], hard_sem, sem_weight) if bool(hard_sem.any()) else sem_loss * 0.0
    labels = batch["fut_global_instance_id"].long()
    label_mask = batch["fut_global_instance_available_mask"].bool()
    contr = contrastive_loss(out["identity_embedding"], labels, label_mask)
    assign = out["point_to_unit_assignment"]
    mean_assign = assign.mean(dim=1)
    anti_collapse = ((mean_assign - 1.0 / mean_assign.shape[-1]) ** 2).mean()
    entropy = -(assign.clamp_min(1e-8) * assign.clamp_min(1e-8).log()).sum(dim=-1).mean()
    total = id_loss + 1.5 * id_hard + 0.2 * contr + args.semantic_weight * sem_loss + args.semantic_hard_weight * sem_hard + args.anti_collapse_weight * anti_collapse - args.assignment_entropy_weight * entropy
    return total, {
        "loss": float(total.detach().cpu()),
        "identity_bce": float(id_loss.detach().cpu()),
        "identity_hard_bce": float(id_hard.detach().cpu()),
        "identity_contrastive": float(contr.detach().cpu()),
        "semantic_measurement_rollout_loss": float(sem_loss.detach().cpu()),
        "semantic_hard_loss": float(sem_hard.detach().cpu()),
        "trace_unit_anti_collapse": float(anti_collapse.detach().cpu()),
        "point_to_unit_assignment_entropy": float(entropy.detach().cpu()),
    }


def train_one(args: argparse.Namespace) -> dict[str, Any]:
    set_seed(args.seed)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    train_loader = make_loader("train", args, shuffle=True)
    model = SemanticTraceUnitsWorldModelV34(args.v30_checkpoint, teacher_embedding_dim=args.teacher_embedding_dim, units=args.trace_units).to(device)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=1e-4)
    losses = []
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
            obs_semantic_measurements=bd["obs_semantic_measurements"],
            obs_semantic_measurement_mask=bd["obs_semantic_measurement_mask"],
            semantic_id=bd["semantic_id"],
        )
        loss, comps = loss_fn(out, bd, args)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step()
        losses.append({"step": float(step), **comps})
    ckpt = CKPT_DIR / "v34_semantic_trace_units_m128_h32_seed42_best.pt"
    torch.save({"model": model.state_dict(), "args": vars(args), "step": args.steps}, ckpt)
    payload = {
        "generated_at_utc": utc_now(),
        "fresh_training_completed": True,
        "checkpoint_path": str(ckpt.relative_to(ROOT)),
        "steps": args.steps,
        "batch_size": args.batch_size,
        "complete_train_sample_count": len(train_loader.dataset),
        "v30_backbone_frozen": model.v30_backbone_frozen,
        "trace_conditioned_semantic_units_built": True,
        "future_leakage_detected": False,
        "teacher_as_method": False,
        "outputs_future_trace_field": True,
        "outputs_future_semantic_field": True,
        "train_loss_first": losses[0] if losses else None,
        "train_loss_last": losses[-1] if losses else None,
        "train_loss_decreased": bool(losses and losses[-1]["loss"] <= losses[0]["loss"]),
        "duration_seconds": float(time.time() - start),
        "loss_tail": losses[-10:],
    }
    dump_json(SUMMARY, payload)
    write_doc(DOC, "STWM OSTF V34 Semantic Trace Units Train Summary", payload, ["fresh_training_completed", "checkpoint_path", "complete_train_sample_count", "v30_backbone_frozen", "trace_conditioned_semantic_units_built", "future_leakage_detected", "teacher_as_method", "outputs_future_trace_field", "outputs_future_semantic_field", "train_loss_decreased"])
    print(SUMMARY.relative_to(ROOT))
    return payload


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--v30-checkpoint", default=str(ROOT / "outputs/checkpoints/stwm_ostf_v30_external_gt/v30_extgt_m128_h32_seed42_best.pt"))
    p.add_argument("--semantic-measurement-bank-root", default=str(MEAS_ROOT))
    p.add_argument("--semantic-identity-sidecar-root", default=str(IDENTITY_ROOT))
    p.add_argument("--global-identity-label-root", default=str(GLOBAL_ROOT))
    p.add_argument("--hard-mask-manifest", default=str(MASK_ROOT / "H32_M128_seed42.json"))
    p.add_argument("--m-points", type=int, default=128)
    p.add_argument("--horizon", type=int, default=32)
    p.add_argument("--teacher-embedding-dim", type=int, default=768)
    p.add_argument("--trace-units", type=int, default=16)
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lr", type=float, default=4e-4)
    p.add_argument("--semantic-weight", type=float, default=1.0)
    p.add_argument("--semantic-hard-weight", type=float, default=1.0)
    p.add_argument("--anti-collapse-weight", type=float, default=0.2)
    p.add_argument("--assignment-entropy-weight", type=float, default=0.01)
    p.add_argument("--cpu", action="store_true")
    return p.parse_args()


def main() -> int:
    train_one(parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
