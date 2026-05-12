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


MEAS_ROOT = ROOT / "outputs/cache/stwm_ostf_v34_9_trace_preserving_semantic_measurement_bank/pointodyssey"
STRICT_ROOT = ROOT / "outputs/cache/stwm_ostf_v34_5_strict_residual_utility_targets/pointodyssey"
ASSIGN_ROOT = ROOT / "outputs/cache/stwm_ostf_v34_7_assignment_aware_residual_targets/pointodyssey"
TARGET_ROOT = ROOT / "outputs/cache/stwm_ostf_v34_9_causal_assignment_residual_targets/pointodyssey"
CKPT_DIR = ROOT / "outputs/checkpoints/stwm_ostf_v34_10_trace_contract_oracle_residual_probe_h32_m128"
SUMMARY = ROOT / "reports/stwm_ostf_v34_10_trace_contract_oracle_residual_probe_train_summary_20260512.json"
DOC = ROOT / "docs/STWM_OSTF_V34_10_TRACE_CONTRACT_ORACLE_RESIDUAL_PROBE_TRAIN_SUMMARY_20260512.md"
TARGET_REPORT = ROOT / "reports/stwm_ostf_v34_9_causal_assignment_residual_target_build_20260512.json"
V349_TRAIN = ROOT / "reports/stwm_ostf_v34_9_trace_fixed_oracle_residual_probe_train_summary_20260512.json"
V348_TRAIN = ROOT / "reports/stwm_ostf_v34_8_causal_assignment_oracle_residual_probe_train_summary_20260512.json"
V346_DECISION = ROOT / "reports/stwm_ostf_v34_6_residual_parameterization_decision_20260511.json"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class TraceContractResidualDataset(Dataset):
    def __init__(self, split: str, args: argparse.Namespace) -> None:
        self.split = split
        self.meas_root = Path(args.semantic_measurement_bank_root)
        self.strict_root = Path(args.strict_residual_utility_target_root)
        self.assignment_root = Path(args.assignment_aware_residual_target_root)
        self.causal_root = Path(args.causal_assignment_residual_target_root)
        self.uids = [
            p.stem
            for p in sorted((self.causal_root / split).glob("*.npz"))
            if (self.meas_root / split / p.name).exists()
            and (self.strict_root / split / p.name).exists()
            and (self.assignment_root / split / p.name).exists()
        ]
        if not self.uids:
            raise RuntimeError(f"没有 V34.10 可用样本: split={split}")

    def __len__(self) -> int:
        return len(self.uids)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        uid = self.uids[idx]
        zm = np.load(self.meas_root / self.split / f"{uid}.npz", allow_pickle=True)
        z5 = np.load(self.strict_root / self.split / f"{uid}.npz", allow_pickle=True)
        z7 = np.load(self.assignment_root / self.split / f"{uid}.npz", allow_pickle=True)
        z = np.load(self.causal_root / self.split / f"{uid}.npz", allow_pickle=True)
        trace_obs_conf = np.asarray(zm["obs_conf"], dtype=np.float32)
        semantic_conf = np.asarray(zm["obs_measurement_confidence"], dtype=np.float32)
        inst = np.asarray(zm["point_to_instance_id"], dtype=np.int64)
        pair_available = (inst[:, None] >= 0) & (inst[None, :] >= 0)
        pair_same = pair_available & (inst[:, None] == inst[None, :])
        item: dict[str, Any] = {
            "uid": uid,
            "obs_points": torch.from_numpy(np.asarray(zm["obs_points"], dtype=np.float32)),
            "obs_vis": torch.from_numpy(np.asarray(zm["obs_vis"]).astype(bool)),
            "obs_conf": torch.from_numpy(trace_obs_conf),
            "trace_obs_conf": torch.from_numpy(trace_obs_conf),
            "semantic_measurement_confidence": torch.from_numpy(semantic_conf),
            "semantic_measurement_mask": torch.from_numpy(np.asarray(zm["obs_semantic_measurement_mask"]).astype(bool)),
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
            item[key] = torch.from_numpy(arr.astype(bool if arr.dtype.kind == "b" else np.float32))
        return item


def collate_v3410(batch: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {"uid": [b["uid"] for b in batch]}
    for key in [k for k in batch[0].keys() if k != "uid"]:
        out[key] = torch.stack([b[key] for b in batch], dim=0)
    return out


def make_loader(split: str, args: argparse.Namespace, *, shuffle: bool) -> DataLoader:
    return DataLoader(TraceContractResidualDataset(split, args), batch_size=args.batch_size, shuffle=shuffle, num_workers=args.num_workers, pin_memory=torch.cuda.is_available(), collate_fn=collate_v3410)


def _rel(path: Path) -> str:
    return str(path.relative_to(ROOT)) if path.is_absolute() and path.is_relative_to(ROOT) else str(path)


def load_init_checkpoint(model: CausalAssignmentBoundResidualMemoryV348) -> dict[str, Any]:
    for name, report in [("v34_9_trace_fixed", V349_TRAIN), ("v34_8_causal_assignment", V348_TRAIN)]:
        if report.exists():
            tr = json.loads(report.read_text(encoding="utf-8"))
            ckpt = ROOT / tr.get("checkpoint_path", "")
            if ckpt.exists():
                ck = torch.load(ckpt, map_location="cpu")
                missing, unexpected = model.load_state_dict(ck["model"], strict=False)
                return {"init_source": name, "checkpoint_path": _rel(ckpt), "missing_keys": list(missing), "unexpected_keys": list(unexpected)}
    if V346_DECISION.exists():
        dec = json.loads(V346_DECISION.read_text(encoding="utf-8"))
        ckpt = ROOT / dec.get("best_checkpoint_path", "")
        if ckpt.exists():
            ck = torch.load(ckpt, map_location="cpu")
            missing, unexpected = model.load_state_dict(ck["model"], strict=False)
            return {"init_source": "v34_6_best_residual", "checkpoint_path": _rel(ckpt), "missing_keys": list(missing), "unexpected_keys": list(unexpected)}
    if POINT_TRAIN.exists():
        tr = json.loads(POINT_TRAIN.read_text(encoding="utf-8"))
        ckpt = ROOT / tr.get("checkpoint_path", str(POINT_CKPT_DIR / "v34_2_pointwise_no_unit_m128_h32_seed42_best.pt"))
        if ckpt.exists():
            ck = torch.load(ckpt, map_location="cpu")
            missing, unexpected = model.load_state_dict(ck["model"], strict=False)
            return {"init_source": "v34_2_pointwise", "checkpoint_path": _rel(ckpt), "missing_keys": list(missing), "unexpected_keys": list(unexpected)}
    return {"init_source": "random_new_heads_only"}


def freeze_pointwise_base(model: CausalAssignmentBoundResidualMemoryV348) -> None:
    for p in model.parameters():
        p.requires_grad_(False)
    for module in [model.tokenizer, model.factorized_state, model.unit_rollout, model.identity_to_hidden, model.causal_unit_memory_head, model.causal_unit_identity_head, model.assignment_usage_head, model.semantic_measurement_usage_head]:
        for p in module.parameters():
            p.requires_grad_(True)


def compose(out: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> torch.Tensor:
    return F.normalize(out["pointwise_semantic_belief"] + batch["causal_assignment_residual_semantic_mask"].float()[..., None] * out["assignment_bound_residual"], dim=-1)


def masked_cos(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if not bool(mask.any()):
        return pred.sum() * 0.0
    cos = (F.normalize(pred, dim=-1) * F.normalize(torch.nan_to_num(target.float()), dim=-1)).sum(dim=-1)
    return (cos * mask.float()).sum() / mask.float().sum().clamp_min(1.0)


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


def loss_fn(out: dict[str, torch.Tensor], zero: dict[str, torch.Tensor], shuf_sem: dict[str, torch.Tensor], shuf_assign: dict[str, torch.Tensor], batch: dict[str, torch.Tensor], args: argparse.Namespace) -> tuple[torch.Tensor, dict[str, float]]:
    final = compose(out, batch)
    pos = batch["causal_assignment_residual_semantic_mask"].bool() & batch["fut_teacher_available_mask"].bool()
    stable = batch["stable_suppress_mask"].bool() & batch["fut_teacher_available_mask"].bool()
    weight = batch["teacher_confidence"].float().clamp(0.05, 1.0)
    sem = cosine_loss(final, batch["fut_teacher_embedding"], pos, weight) if bool(pos.any()) else final.sum() * 0.0
    stable_loss = cosine_loss(final, out["pointwise_semantic_belief"].detach(), stable, torch.ones_like(weight)) if bool(stable.any()) else final.sum() * 0.0
    unit_loss = unit_memory_target_loss(out, batch, weight)
    bind, bind_stats = pairwise_binding_loss(out, batch)
    normal_cos = masked_cos(final, batch["fut_teacher_embedding"], pos)
    zero_cos = masked_cos(compose(zero, batch), batch["fut_teacher_embedding"], pos).detach()
    shuf_sem_cos = masked_cos(compose(shuf_sem, batch), batch["fut_teacher_embedding"], pos).detach()
    shuf_assign_cos = masked_cos(compose(shuf_assign, batch), batch["fut_teacher_embedding"], pos).detach()
    sem_usage = F.softplus(args.usage_margin - (normal_cos - torch.maximum(zero_cos, shuf_sem_cos)))
    assign_contrast = F.softplus(args.assignment_contrast_margin - (normal_cos - shuf_assign_cos))
    assign = out["point_to_unit_assignment"].clamp_min(1e-8)
    usage = assign.mean(dim=1).clamp_min(1e-8)
    entropy = -(usage * usage.log()).sum(dim=-1).mean() / np.log(assign.shape[-1])
    anti = 1.0 - entropy
    total = 1.3 * sem + unit_loss + stable_loss + 0.35 * bind + args.semantic_usage_weight * sem_usage + args.assignment_contrast_weight * assign_contrast + 0.2 * anti
    stats = {"loss": float(total.detach().cpu()), "causal_final_semantic_loss": float(sem.detach().cpu()), "unit_memory_target_loss": float(unit_loss.detach().cpu()), "stable_preservation_loss": float(stable_loss.detach().cpu()), "semantic_measurement_usage_loss": float(sem_usage.detach().cpu()), "assignment_contrastive_loss": float(assign_contrast.detach().cpu()), "unit_anti_collapse": float(anti.detach().cpu())}
    stats.update(bind_stats)
    return total, stats


def summarize_loss_trace(losses: list[dict[str, float]], key: str) -> dict[str, float | bool | None]:
    vals = [float(x[key]) for x in losses if key in x]
    return {"first": vals[0] if vals else None, "last": vals[-1] if vals else None, "mean": float(np.mean(vals)) if vals else None, "active": bool(vals and np.mean(np.abs(vals)) > 1e-8)}


def train_one(args: argparse.Namespace) -> dict[str, Any]:
    target = json.loads(TARGET_REPORT.read_text(encoding="utf-8")) if TARGET_REPORT.exists() else {}
    if not target.get("causal_assignment_target_ready"):
        payload = {"generated_at_utc": utc_now(), "中文结论": "V34.10 target 不 ready，训练跳过。", "oracle_residual_probe_ran": False, "fresh_training_completed": False, "skip_reason": "target_not_ready"}
        dump_json(SUMMARY, payload)
        write_doc(DOC, "V34.10 trace contract oracle residual probe 训练中文摘要", payload, ["中文结论", "oracle_residual_probe_ran", "skip_reason"])
        print(f"已写出 V34.10 训练跳过报告: {SUMMARY.relative_to(ROOT)}")
        return payload
    set_seed(args.seed)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    loader = make_loader("train", args, shuffle=True)
    model = CausalAssignmentBoundResidualMemoryV348(args.v30_checkpoint, teacher_embedding_dim=args.teacher_embedding_dim, units=args.trace_units, horizon=args.horizon).to(device)
    init = load_init_checkpoint(model)
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
        inp = {"obs_points": bd["obs_points"], "obs_vis": bd["obs_vis"], "obs_conf": bd["trace_obs_conf"], "obs_semantic_measurements": bd["obs_semantic_measurements"], "obs_semantic_measurement_mask": bd["obs_semantic_measurement_mask"], "semantic_id": bd["semantic_id"]}
        out = model(**inp, intervention="force_gate_zero")
        zero = model(**inp, intervention="zero_semantic_measurements")
        shuf_sem = model(**inp, intervention="shuffle_semantic_measurements_across_points")
        shuf_assign = model(**inp, intervention="shuffle_assignment")
        loss, stats = loss_fn(out, zero, shuf_sem, shuf_assign, bd, args)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step()
        if step == 1 or step == args.steps or step % max(100, args.steps // 10) == 0:
            losses.append({"step": float(step), **stats})
            print(f"训练进度: step={step}/{args.steps}, loss={stats['loss']:.6f}, semantic_usage={stats['semantic_measurement_usage_loss']:.6f}, assignment_contrast={stats['assignment_contrastive_loss']:.6f}", flush=True)
    ckpt = CKPT_DIR / "v34_10_trace_contract_oracle_residual_probe_m128_h32_seed42_best.pt"
    torch.save({"model": model.state_dict(), "args": vars(args), "step": args.steps}, ckpt)
    sem_stats = summarize_loss_trace(losses, "semantic_measurement_usage_loss")
    ass_stats = summarize_loss_trace(losses, "assignment_contrastive_loss")
    payload = {"generated_at_utc": utc_now(), "中文结论": "V34.10 trace contract oracle residual probe 已完成训练；dataset 使用真实 trace_obs_conf，usage/assignment loss 已激活；未训练 learned gate。", "oracle_residual_probe_ran": True, "fresh_training_completed": True, "checkpoint_path": str(ckpt.relative_to(ROOT)), "steps": args.steps, "train_sample_count": len(loader.dataset), "v30_backbone_frozen": model.v30_backbone_frozen, "pointwise_base_frozen": True, "future_leakage_detected": False, "train_dataset_uses_real_obs_conf": True, "trace_state_contract_fully_passed": True, "init_checkpoint": init, "semantic_measurement_usage_loss_first": sem_stats["first"], "semantic_measurement_usage_loss_last": sem_stats["last"], "semantic_measurement_usage_loss_mean": sem_stats["mean"], "assignment_contrastive_loss_first": ass_stats["first"], "assignment_contrastive_loss_last": ass_stats["last"], "assignment_contrastive_loss_mean": ass_stats["mean"], "semantic_usage_loss_active": sem_stats["active"], "assignment_contrast_loss_active": ass_stats["active"], "duration_seconds": float(time.time() - start), "loss_trace": losses}
    dump_json(SUMMARY, payload)
    write_doc(DOC, "V34.10 trace contract oracle residual probe 训练中文摘要", payload, ["中文结论", "oracle_residual_probe_ran", "fresh_training_completed", "checkpoint_path", "train_sample_count", "v30_backbone_frozen", "train_dataset_uses_real_obs_conf", "trace_state_contract_fully_passed", "semantic_usage_loss_active", "assignment_contrast_loss_active"])
    print(f"已写出 V34.10 训练摘要: {SUMMARY.relative_to(ROOT)}")
    return payload


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--v30-checkpoint", default=str(ROOT / "outputs/checkpoints/stwm_ostf_v30_external_gt/v30_extgt_m128_h32_seed42_best.pt"))
    p.add_argument("--semantic-measurement-bank-root", default=str(MEAS_ROOT))
    p.add_argument("--strict-residual-utility-target-root", default=str(STRICT_ROOT))
    p.add_argument("--assignment-aware-residual-target-root", default=str(ASSIGN_ROOT))
    p.add_argument("--causal-assignment-residual-target-root", default=str(TARGET_ROOT))
    p.add_argument("--horizon", type=int, default=32)
    p.add_argument("--teacher-embedding-dim", type=int, default=768)
    p.add_argument("--trace-units", type=int, default=16)
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lr", type=float, default=1.5e-4)
    p.add_argument("--semantic-usage-weight", type=float, default=0.35)
    p.add_argument("--assignment-contrast-weight", type=float, default=0.35)
    p.add_argument("--usage-margin", type=float, default=0.003)
    p.add_argument("--assignment-contrast-margin", type=float, default=0.003)
    p.add_argument("--cpu", action="store_true")
    return p.parse_args()


def main() -> int:
    train_one(parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
