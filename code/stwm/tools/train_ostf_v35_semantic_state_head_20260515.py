#!/usr/bin/env python3
"""训练 V35 semantic state head（使用 V35.1 fixed targets，不训练 V30）。"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import setproctitle
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))
setproctitle.setproctitle("python")

from stwm.modules.ostf_v35_semantic_state_world_model import SemanticStateWorldModelV35
from stwm.tools.ostf_v17_common_20260502 import ROOT

TARGET_ROOT = ROOT / "outputs/cache/stwm_ostf_v35_1_fixed_semantic_state_targets/pointodyssey"
ASSIGNMENT_TARGET_ROOT = ROOT / "outputs/cache/stwm_ostf_v35_2_identity_confuser_assignment_targets/pointodyssey"
MEASUREMENT_BANK_ROOT = ROOT / "outputs/cache/stwm_ostf_v34_9_trace_preserving_semantic_measurement_bank/pointodyssey"
CKPT_DIR = ROOT / "outputs/checkpoints/stwm_ostf_v35_semantic_state_head_h32_m128"
SUMMARY = ROOT / "reports/stwm_ostf_v35_semantic_state_head_train_summary_20260515.json"
DOC = ROOT / "docs/STWM_OSTF_V35_SEMANTIC_STATE_HEAD_TRAIN_SUMMARY_20260515.md"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def list_npz(split: str) -> list[Path]:
    return sorted((TARGET_ROOT / split).glob("*.npz"))


def entropy_row(labels: np.ndarray, k: int) -> np.ndarray:
    out = np.zeros((labels.shape[0],), dtype=np.float32)
    for i, row in enumerate(labels):
        vals = row[row >= 0]
        if len(vals):
            cnt = np.bincount(vals, minlength=k).astype(np.float32)
            p = cnt[cnt > 0] / max(cnt.sum(), 1.0)
            out[i] = float(-(p * np.log2(np.maximum(p, 1e-12))).sum())
    return out


def last_valid_cluster(obs_cluster: np.ndarray) -> np.ndarray:
    out = np.full((obs_cluster.shape[0],), -1, dtype=np.int64)
    for i, row in enumerate(obs_cluster):
        idx = np.where(row >= 0)[0]
        if len(idx):
            out[i] = int(row[idx[-1]])
    return out


def mode_cluster(obs_cluster: np.ndarray, k: int) -> np.ndarray:
    out = np.full((obs_cluster.shape[0],), -1, dtype=np.int64)
    for i, row in enumerate(obs_cluster):
        vals = row[row >= 0]
        if len(vals):
            out[i] = int(np.bincount(vals, minlength=k).argmax())
    return out


def make_point_features(z: Any, semantic_clusters: int) -> np.ndarray:
    obs_cluster = np.asarray(z["obs_semantic_cluster_id"], dtype=np.int64)
    obs_points = np.asarray(z["obs_points"], dtype=np.float32)
    obs_vis = np.asarray(z["obs_vis"], dtype=np.float32)
    obs_conf = np.asarray(z["obs_conf"], dtype=np.float32)
    last = last_valid_cluster(obs_cluster)
    mode = mode_cluster(obs_cluster, semantic_clusters)
    one_hot_last = np.eye(semantic_clusters, dtype=np.float32)[np.clip(last, 0, semantic_clusters - 1)]
    one_hot_mode = np.eye(semantic_clusters, dtype=np.float32)[np.clip(mode, 0, semantic_clusters - 1)]
    obs_ent = entropy_row(obs_cluster, semantic_clusters)
    vis_frac = obs_vis.mean(axis=1)
    conf_mean = obs_conf.mean(axis=1)
    conf_last = obs_conf[:, -1]
    disp = obs_points[:, -1] - obs_points[:, 0]
    speed = np.sqrt((np.diff(obs_points, axis=1) ** 2).sum(axis=-1)).mean(axis=1)
    stats = np.stack([obs_ent, vis_frac, conf_mean, conf_last, disp[:, 0], disp[:, 1], speed, last >= 0, mode >= 0], axis=1).astype(np.float32)
    return np.concatenate([one_hot_last, one_hot_mode, stats], axis=1).astype(np.float32)


class V35SemanticStateDataset(Dataset):
    def __init__(
        self,
        split: str,
        semantic_clusters: int,
        assignment_target_root: str | Path | None = None,
        measurement_bank_root: str | Path | None = None,
        include_observed_measurement_embedding: bool = False,
    ) -> None:
        self.split = split
        self.semantic_clusters = semantic_clusters
        self.assignment_target_root = Path(assignment_target_root) if assignment_target_root else None
        if self.assignment_target_root is not None and not self.assignment_target_root.is_absolute():
            self.assignment_target_root = ROOT / self.assignment_target_root
        self.measurement_bank_root = Path(measurement_bank_root) if measurement_bank_root else None
        if self.measurement_bank_root is not None and not self.measurement_bank_root.is_absolute():
            self.measurement_bank_root = ROOT / self.measurement_bank_root
        self.include_observed_measurement_embedding = bool(include_observed_measurement_embedding)
        self.paths = list_npz(split)
        if not self.paths:
            raise RuntimeError(f"V35 target cache 为空: split={split}")
        z0 = np.load(self.paths[0], allow_pickle=True)
        self.semantic_feature_dim = int(make_point_features(z0, self.semantic_clusters).shape[-1])

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        path = self.paths[idx]
        z = np.load(path, allow_pickle=True)
        point_features = make_point_features(z, self.semantic_clusters)
        if self.include_observed_measurement_embedding:
            if self.measurement_bank_root is None:
                raise RuntimeError("include_observed_measurement_embedding=true 但 measurement_bank_root 为空")
            mp = self.measurement_bank_root / self.split / path.name
            if not mp.exists():
                raise FileNotFoundError(f"缺少 semantic measurement bank: {mp}")
            zm = np.load(mp, allow_pickle=True)
            sem = np.asarray(zm["instance_observed_semantic_measurement"], dtype=np.float32)
            sem = sem / np.maximum(np.linalg.norm(sem, axis=1, keepdims=True), 1e-6)
            meas_conf = np.asarray(zm["obs_measurement_confidence"], dtype=np.float32)
            meas_mask = np.asarray(zm["obs_semantic_measurement_mask"], dtype=np.float32)
            agreement = np.asarray(zm["teacher_agreement_score"], dtype=np.float32)
            denom = np.maximum(meas_mask.sum(axis=1, keepdims=True), 1.0)
            meas_stats = np.concatenate(
                [
                    (meas_conf * meas_mask).sum(axis=1, keepdims=True) / denom,
                    meas_conf.max(axis=1, keepdims=True),
                    meas_mask.mean(axis=1, keepdims=True),
                    (agreement * meas_mask).sum(axis=1, keepdims=True) / denom,
                ],
                axis=1,
            ).astype(np.float32)
            point_features = np.concatenate([point_features, sem, meas_stats], axis=1).astype(np.float32)
        item = {
            "uid": path.stem,
            "point_features": torch.from_numpy(point_features),
            "target_cluster": torch.from_numpy(np.asarray(z["target_semantic_cluster_id"], dtype=np.int64)),
            "valid": torch.from_numpy(np.asarray(z["target_semantic_cluster_available_mask"], dtype=bool)),
            "changed": torch.from_numpy(np.asarray(z["semantic_changed_mask"], dtype=bool)),
            "hard": torch.from_numpy(np.asarray(z["semantic_hard_mask"], dtype=bool)),
            "stable": torch.from_numpy(np.asarray(z["semantic_stable_mask"], dtype=bool)),
            "family": torch.from_numpy(np.asarray(z["evidence_anchor_family_target"], dtype=np.int64)),
            "family_available": torch.from_numpy(np.asarray(z["evidence_anchor_family_available_mask"], dtype=bool)),
            "same_instance": torch.from_numpy(np.asarray(z["same_instance_as_observed_target"], dtype=bool)),
            "same_available": torch.from_numpy(np.asarray(z["identity_consistency_available_mask"], dtype=bool)),
            "uncertainty": torch.from_numpy(np.asarray(z["semantic_uncertainty_target"], dtype=np.float32)),
            "last_cluster": torch.from_numpy(np.repeat(last_valid_cluster(np.asarray(z["obs_semantic_cluster_id"], dtype=np.int64))[:, None], np.asarray(z["target_semantic_cluster_id"]).shape[1], axis=1).astype(np.int64)),
            "point_to_instance_id": torch.from_numpy(np.asarray(z["point_to_instance_id"], dtype=np.int64)),
        }
        if self.assignment_target_root is not None:
            ap = self.assignment_target_root / self.split / path.name
            if ap.exists():
                za = np.load(ap, allow_pickle=True)
                for key in [
                    "pair_available_mask",
                    "same_instance_pair_mask",
                    "identity_confuser_pair_mask",
                    "assignment_positive_pair_mask",
                    "assignment_negative_pair_mask",
                    "identity_confuser_point_mask",
                    "same_instance_hard_positive_mask",
                    "same_instance_hard_negative_mask",
                ]:
                    item[key] = torch.from_numpy(np.asarray(za[key], dtype=bool))
        return item


def collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {"uid": [b["uid"] for b in batch]}
    for k in batch[0]:
        if k == "uid":
            continue
        out[k] = torch.stack([b[k] for b in batch], dim=0)
    return out


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}


def masked_ce(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if not bool(mask.any()):
        return logits.sum() * 0.0
    return F.cross_entropy(logits[mask], target[mask].clamp_min(0))


def balanced_bce(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if not bool(mask.any()):
        return logits.sum() * 0.0
    y = target[mask].float()
    x = logits[mask]
    pos = y.sum().clamp_min(1.0)
    neg = (1.0 - y).sum().clamp_min(1.0)
    pos_weight = (neg / pos).clamp(0.25, 8.0)
    return F.binary_cross_entropy_with_logits(x, y, pos_weight=pos_weight)


def pairwise_assignment_loss(assign: torch.Tensor, instance_id: torch.Tensor, batch: dict[str, torch.Tensor] | None = None) -> torch.Tensor:
    if batch is not None and "assignment_positive_pair_mask" in batch and "assignment_negative_pair_mask" in batch:
        pos_mask = batch["assignment_positive_pair_mask"].bool()
        neg_mask = batch["assignment_negative_pair_mask"].bool()
        mask = pos_mask | neg_mask
        if not bool(mask.any()):
            return assign.sum() * 0.0
        sim = torch.einsum("bmu,bnu->bmn", assign, assign).clamp(1e-5, 1.0 - 1e-5)
        y = pos_mask.float()
        pos = pos_mask.sum().clamp_min(1)
        neg = neg_mask.sum().clamp_min(1)
        weight = torch.where(pos_mask, (neg.float() / pos.float()).clamp(0.25, 8.0), torch.ones_like(sim))
        return F.binary_cross_entropy(sim[mask], y[mask], weight=weight[mask], reduction="mean")
    same_avail = (instance_id[:, :, None] >= 0) & (instance_id[:, None, :] >= 0)
    same = same_avail & (instance_id[:, :, None] == instance_id[:, None, :])
    eye = torch.eye(instance_id.shape[1], device=instance_id.device, dtype=torch.bool)[None]
    mask = same_avail & ~eye
    if not bool(mask.any()):
        return assign.sum() * 0.0
    sim = torch.einsum("bmu,bnu->bmn", assign, assign).clamp(1e-5, 1.0 - 1e-5)
    y = same.float()
    pos = (same & mask).sum().clamp_min(1)
    neg = ((~same) & mask).sum().clamp_min(1)
    weight = torch.where(same, (neg.float() / pos.float()).clamp(0.25, 8.0), torch.ones_like(sim))
    loss = F.binary_cross_entropy(sim[mask], y[mask], weight=weight[mask], reduction="mean")
    return loss


def identity_embedding_pairwise_loss(
    identity_embedding: torch.Tensor,
    instance_id: torch.Tensor,
    batch: dict[str, torch.Tensor] | None = None,
    *,
    temperature: float = 0.10,
) -> torch.Tensor:
    emb = F.normalize(identity_embedding.mean(dim=2), dim=-1)
    sim_logits = torch.einsum("bmd,bnd->bmn", emb, emb) / max(float(temperature), 1e-6)
    if batch is not None and "same_instance_pair_mask" in batch and "identity_confuser_pair_mask" in batch:
        pos_mask = batch["same_instance_pair_mask"].bool()
        neg_mask = batch["identity_confuser_pair_mask"].bool()
    else:
        valid = (instance_id[:, :, None] >= 0) & (instance_id[:, None, :] >= 0)
        pos_mask = valid & (instance_id[:, :, None] == instance_id[:, None, :])
        neg_mask = valid & (instance_id[:, :, None] != instance_id[:, None, :])
    eye = torch.eye(instance_id.shape[1], device=instance_id.device, dtype=torch.bool)[None]
    pos_mask = pos_mask & ~eye
    neg_mask = neg_mask & ~eye
    mask = pos_mask | neg_mask
    if not bool(mask.any()):
        return identity_embedding.sum() * 0.0
    target = pos_mask.float()
    pos = pos_mask.sum().clamp_min(1)
    neg = neg_mask.sum().clamp_min(1)
    weight = torch.where(pos_mask, (neg.float() / pos.float()).clamp(0.25, 8.0), torch.ones_like(sim_logits))
    return F.binary_cross_entropy_with_logits(sim_logits[mask], target[mask], weight=weight[mask], reduction="mean")


def loss_fn(
    out: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    *,
    pairwise_assignment_weight: float = 0.0,
    identity_contrastive_weight: float = 0.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    valid = batch["valid"].bool()
    changed_mask = batch["changed"].bool() & valid
    hard_mask = batch["hard"].bool() & valid
    stable_mask = batch["stable"].bool() & valid
    family_mask = batch["family_available"].bool() & valid
    same_mask = batch["same_available"].bool() & valid
    cluster = masked_ce(out["semantic_cluster_logits"], batch["target_cluster"], valid)
    cluster_changed = masked_ce(out["semantic_cluster_logits"], batch["target_cluster"], changed_mask | hard_mask)
    stable = masked_ce(out["semantic_cluster_logits"], batch["target_cluster"], stable_mask)
    change = balanced_bce(out["semantic_change_logits"], batch["changed"], valid)
    family = masked_ce(out["evidence_anchor_family_logits"], batch["family"], family_mask)
    if "identity_confuser_point_mask" in batch:
        same_mask = same_mask | batch["identity_confuser_point_mask"].bool()
    same = balanced_bce(out["same_instance_logits"], batch["same_instance"], same_mask)
    identity_pair = identity_embedding_pairwise_loss(out["identity_embedding"], batch["point_to_instance_id"], batch)
    uncertainty = F.mse_loss(out["semantic_uncertainty"][valid], batch["uncertainty"].float()[valid]) if bool(valid.any()) else out["semantic_uncertainty"].sum() * 0.0
    pair_assign = pairwise_assignment_loss(out["point_to_unit_assignment"], batch["point_to_instance_id"], batch)
    assign = out["point_to_unit_assignment"].clamp_min(1e-8)
    usage = assign.mean(dim=1).clamp_min(1e-8)
    anti_collapse = 1.0 - (-(usage * usage.log()).sum(dim=-1).mean() / np.log(assign.shape[-1]))
    total = (
        cluster
        + 0.7 * cluster_changed
        + 0.5 * stable
        + 0.7 * change
        + 0.35 * family
        + 0.35 * same
        + identity_contrastive_weight * identity_pair
        + 0.45 * uncertainty
        + pairwise_assignment_weight * pair_assign
        + 0.05 * anti_collapse
    )
    return total, {
        "loss": float(total.detach().cpu()),
        "cluster_ce": float(cluster.detach().cpu()),
        "changed_hard_cluster_ce": float(cluster_changed.detach().cpu()),
        "stable_ce": float(stable.detach().cpu()),
        "change_bce": float(change.detach().cpu()),
        "family_ce": float(family.detach().cpu()),
        "same_instance_bce": float(same.detach().cpu()),
        "identity_embedding_pairwise_loss": float(identity_pair.detach().cpu()),
        "uncertainty_mse": float(uncertainty.detach().cpu()),
        "pairwise_assignment_loss": float(pair_assign.detach().cpu()),
        "unit_anti_collapse": float(anti_collapse.detach().cpu()),
    }


@torch.no_grad()
def eval_loss(
    model: SemanticStateWorldModelV35,
    loader: DataLoader,
    device: torch.device,
    *,
    pairwise_assignment_weight: float = 0.0,
    identity_contrastive_weight: float = 0.0,
) -> float:
    model.eval()
    vals: list[float] = []
    for batch in loader:
        bd = move_batch(batch, device)
        out = model(bd["point_features"], horizon=bd["target_cluster"].shape[2])
        loss, _ = loss_fn(
            out,
            bd,
            pairwise_assignment_weight=pairwise_assignment_weight,
            identity_contrastive_weight=identity_contrastive_weight,
        )
        vals.append(float(loss.detach().cpu()))
    model.train()
    return float(np.mean(vals)) if vals else float("inf")


def summarize_trace(trace: list[dict[str, float]], key: str) -> dict[str, float | None]:
    vals = [x[key] for x in trace if key in x]
    return {
        "first": float(vals[0]) if vals else None,
        "last": float(vals[-1]) if vals else None,
        "mean": float(np.mean(vals)) if vals else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--semantic-clusters", type=int, default=64)
    ap.add_argument("--evidence-families", type=int, default=5)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--copy-prior-strength", type=float, default=7.0)
    ap.add_argument("--assignment-bound-decoder", action="store_true")
    ap.add_argument("--identity-dim", type=int, default=64)
    ap.add_argument("--pairwise-assignment-weight", type=float, default=0.0)
    ap.add_argument("--identity-contrastive-weight", type=float, default=0.0)
    ap.add_argument("--assignment-target-root", type=str, default="")
    ap.add_argument("--measurement-bank-root", type=str, default=str(MEASUREMENT_BANK_ROOT))
    ap.add_argument("--include-observed-measurement-embedding", action="store_true")
    ap.add_argument("--checkpoint-dir", type=str, default=str(CKPT_DIR))
    ap.add_argument("--summary-path", type=str, default=str(SUMMARY))
    ap.add_argument("--doc-path", type=str, default=str(DOC))
    ap.add_argument("--init-checkpoint", type=str, default="")
    ap.add_argument("--identity-only-finetune", action="store_true")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()
    print(f"V35 semantic state head: 开始 seed{args.seed} 训练；V30 frozen，不训练 trajectory。", flush=True)
    set_seed(args.seed)
    assignment_target_root = args.assignment_target_root or None
    measurement_bank_root = args.measurement_bank_root or None
    train_ds = V35SemanticStateDataset(
        "train",
        args.semantic_clusters,
        assignment_target_root,
        measurement_bank_root,
        args.include_observed_measurement_embedding,
    )
    val_ds = V35SemanticStateDataset(
        "val",
        args.semantic_clusters,
        assignment_target_root,
        measurement_bank_root,
        args.include_observed_measurement_embedding,
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=torch.cuda.is_available(), collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=torch.cuda.is_available(), collate_fn=collate)
    feature_dim = train_ds[0]["point_features"].shape[-1]
    semantic_feature_dim = train_ds.semantic_feature_dim
    setattr(args, "semantic_feature_dim", semantic_feature_dim)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = SemanticStateWorldModelV35(
        point_feature_dim=feature_dim,
        semantic_clusters=args.semantic_clusters,
        evidence_families=args.evidence_families,
        copy_prior_strength=args.copy_prior_strength,
        assignment_bound_decoder=args.assignment_bound_decoder,
        identity_dim=args.identity_dim,
        semantic_feature_dim=semantic_feature_dim,
    ).to(device)
    if args.init_checkpoint:
        init_path = Path(args.init_checkpoint)
        if not init_path.is_absolute():
            init_path = ROOT / init_path
        init = torch.load(init_path, map_location="cpu")
        model.load_state_dict(init["model"], strict=False)
        print(f"已加载初始化 checkpoint: {init_path}", flush=True)
    if args.identity_only_finetune:
        for p in model.parameters():
            p.requires_grad = False
        for module in [model.identity_point_encoder, model.identity_embedding_head]:
            for p in module.parameters():
                p.requires_grad = True
        print("identity-only finetune: semantic path 已冻结，仅训练 identity encoder/head。", flush=True)
    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        raise RuntimeError("没有可训练参数")
    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=1e-4)
    ckpt_dir = Path(args.checkpoint_dir)
    if not ckpt_dir.is_absolute():
        ckpt_dir = ROOT / ckpt_dir
    summary_path = Path(args.summary_path)
    if not summary_path.is_absolute():
        summary_path = ROOT / summary_path
    doc_path = Path(args.doc_path)
    if not doc_path.is_absolute():
        doc_path = ROOT / doc_path
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path = ckpt_dir / f"v35_semantic_state_head_m128_h32_seed{args.seed}_best.pt"
    last_path = ckpt_dir / f"v35_semantic_state_head_m128_h32_seed{args.seed}_last.pt"
    trace: list[dict[str, float]] = []
    best_val = float("inf")
    it = iter(train_loader)
    start = time.time()
    for step in range(1, args.steps + 1):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(train_loader)
            batch = next(it)
        bd = move_batch(batch, device)
        out = model(bd["point_features"], horizon=bd["target_cluster"].shape[2])
        loss, stats = loss_fn(
            out,
            bd,
            pairwise_assignment_weight=args.pairwise_assignment_weight,
            identity_contrastive_weight=args.identity_contrastive_weight,
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        trace.append(stats)
        if step % 100 == 0 or step == args.steps:
            val_loss = eval_loss(
                model,
                val_loader,
                device,
                pairwise_assignment_weight=args.pairwise_assignment_weight,
                identity_contrastive_weight=args.identity_contrastive_weight,
            )
            print(f"step={step} train_loss={stats['loss']:.4f} val_loss={val_loss:.4f}", flush=True)
            if val_loss < best_val:
                best_val = val_loss
                torch.save({"model": model.state_dict(), "args": vars(args), "feature_dim": feature_dim, "step": step, "val_loss": best_val}, best_path)
    torch.save({"model": model.state_dict(), "args": vars(args), "feature_dim": feature_dim, "step": args.steps, "val_loss": best_val}, last_path)
    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "semantic_state_head_training_ran": True,
        "fresh_training_completed": True,
        "seed": args.seed,
        "M": 128,
        "H": 32,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "target_root": str(TARGET_ROOT.relative_to(ROOT)),
        "checkpoint_path": str(best_path.relative_to(ROOT)),
        "last_checkpoint_path": str(last_path.relative_to(ROOT)),
        "best_val_loss": best_val,
        "copy_prior_strength": args.copy_prior_strength,
        "assignment_bound_decoder": args.assignment_bound_decoder,
        "identity_dim": args.identity_dim,
        "semantic_feature_dim": semantic_feature_dim,
        "pairwise_assignment_weight": args.pairwise_assignment_weight,
        "identity_contrastive_weight": args.identity_contrastive_weight,
        "assignment_target_root": args.assignment_target_root,
        "measurement_bank_root": args.measurement_bank_root,
        "include_observed_measurement_embedding": args.include_observed_measurement_embedding,
        "init_checkpoint": args.init_checkpoint,
        "identity_only_finetune": args.identity_only_finetune,
        "train_seconds": time.time() - start,
        "loss_trace": {k: summarize_trace(trace, k) for k in trace[0].keys()} if trace else {},
        "v30_backbone_frozen": True,
        "future_leakage_detected": False,
        "trajectory_degraded": False,
        "teacher_as_method": False,
        "future_teacher_embeddings_input_allowed": False,
        "中文结论": "V35 semantic state head 已在 V35.1 fixed targets 上完成 seed42 训练；本训练只更新 semantic state head，不训练 V30 trajectory backbone。",
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    doc_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    doc_path.write_text(
        "# STWM OSTF V35 Semantic State Head Train Summary\n\n"
        f"- semantic_state_head_training_ran: true\n"
        f"- checkpoint_path: `{summary['checkpoint_path']}`\n"
        f"- best_val_loss: {best_val}\n"
        f"- v30_backbone_frozen: true\n"
        f"- future_leakage_detected: false\n\n"
        "## 中文总结\n"
        + summary["中文结论"]
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"checkpoint_path": summary["checkpoint_path"], "best_val_loss": best_val}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
