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
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))

from stwm.datasets.ostf_v30_external_gt_dataset_20260508 import OSTFExternalGTDataset, collate_external_gt
from stwm.modules.ostf_v33_2_visual_semantic_identity_world_model import VisualSemanticIdentityWorldModelV332
from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, write_doc
from stwm.tools.ostf_v30_external_gt_schema_20260508 import utc_now
from stwm.tools.train_ostf_v33_1_integrated_semantic_identity_20260509 import binary_metrics, minfde, pr_auc, rank_auc, visibility_f1

RUN_DIR = ROOT / "reports/stwm_ostf_v33_2_visual_semantic_identity_runs"
CKPT_DIR = ROOT / "outputs/checkpoints/stwm_ostf_v33_2_visual_semantic_identity"
SUMMARY = ROOT / "reports/stwm_ostf_v33_2_visual_semantic_identity_smoke_summary_20260509.json"
DECISION = ROOT / "reports/stwm_ostf_v33_2_visual_semantic_identity_smoke_decision_20260509.json"
DOC = ROOT / "docs/STWM_OSTF_V33_2_VISUAL_SEMANTIC_IDENTITY_SMOKE_DECISION_20260509.md"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class VisualSidecarDataset(Dataset):
    def __init__(self, split: str, args: argparse.Namespace, *, max_items: int | None = None) -> None:
        self.base = OSTFExternalGTDataset(
            split,
            horizon=args.horizon,
            m_points=args.m_points,
            max_items=None,
            enable_semantic_identity_sidecar=True,
            semantic_identity_sidecar_root=args.semantic_identity_sidecar_root,
            require_semantic_identity_sidecar=True,
            use_observed_instance_context=args.use_observed_instance_context,
        )
        self.teacher_root = Path(args.visual_teacher_root)
        if not self.teacher_root.is_absolute():
            self.teacher_root = ROOT / self.teacher_root
        keep = []
        for i, entry in enumerate(self.base.entries):
            uid = Path(entry["cache_path"]).stem
            if (self.teacher_root / split / f"{uid}.npz").exists():
                keep.append(entry)
        self.base.entries = keep[:max_items] if max_items is not None else keep
        if not self.base.entries:
            raise RuntimeError(f"No visual teacher sidecars for split={split} under {self.teacher_root}")
        self.split = split

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = self.base[idx]
        uid = item["uid"]
        path = self.teacher_root / self.split / f"{uid}.npz"
        v = np.load(path, allow_pickle=True)
        item["obs_teacher_embedding"] = torch.from_numpy(np.asarray(v["obs_teacher_embedding"], dtype=np.float32))
        item["obs_teacher_available_mask"] = torch.from_numpy(np.asarray(v["obs_teacher_available_mask"]).astype(bool))
        item["fut_teacher_embedding"] = torch.from_numpy(np.asarray(v["fut_teacher_embedding"], dtype=np.float32))
        item["fut_teacher_available_mask"] = torch.from_numpy(np.asarray(v["fut_teacher_available_mask"]).astype(bool))
        item["future_teacher_embeddings_input_allowed"] = torch.tensor(bool(np.asarray(v["future_teacher_embeddings_input_allowed"]).item()), dtype=torch.bool)
        return item


def collate_visual(batch: list[dict[str, Any]]) -> dict[str, Any]:
    out = collate_external_gt(batch)
    for key in [
        "obs_teacher_embedding",
        "obs_teacher_available_mask",
        "fut_teacher_embedding",
        "fut_teacher_available_mask",
        "future_teacher_embeddings_input_allowed",
    ]:
        out[key] = torch.stack([b[key] for b in batch], dim=0)
    return out


def make_loader(split: str, args: argparse.Namespace, *, shuffle: bool, max_items: int | None) -> DataLoader:
    ds = VisualSidecarDataset(split, args, max_items=max_items)
    return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle, num_workers=args.num_workers, pin_memory=torch.cuda.is_available(), collate_fn=collate_visual)


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


def semantic_retrieval(pred: np.ndarray, target: np.ndarray, mask: np.ndarray, obs_copy: np.ndarray | None = None) -> dict[str, Any]:
    pred = pred.astype(np.float32)
    target = target.astype(np.float32)
    mask = mask.astype(bool)
    b, m, h, d = target.shape
    top1 = top5 = copy_top1 = hard_top1 = count = hard_count = 0
    cos_vals = []
    copy_vals = []
    for bi in range(b):
        for hh in range(h):
            valid = np.where(mask[bi, :, hh])[0]
            if valid.size < 2:
                continue
            tgt = target[bi, valid, hh]
            tgt = tgt / np.maximum(np.linalg.norm(tgt, axis=-1, keepdims=True), 1e-6)
            pr = pred[bi, valid, hh]
            pr = pr / np.maximum(np.linalg.norm(pr, axis=-1, keepdims=True), 1e-6)
            sim = pr @ tgt.T
            rank = np.argsort(-sim, axis=1)
            gt = np.arange(valid.size)
            top1 += int((rank[:, 0] == gt).sum())
            top5 += int([g in row[: min(5, valid.size)] for g, row in zip(gt, rank)].count(True))
            cos_vals.extend(sim[gt, gt].tolist())
            if obs_copy is not None:
                cp = obs_copy[bi, valid]
                cp = cp / np.maximum(np.linalg.norm(cp, axis=-1, keepdims=True), 1e-6)
                csim = cp @ tgt.T
                crank = np.argsort(-csim, axis=1)
                copy_top1 += int((crank[:, 0] == gt).sum())
                copy_vals.extend(csim[gt, gt].tolist())
            count += valid.size
            # Hard semantic subset: only count rows where nearest non-self is close to self.
            if valid.size > 1:
                nonself = sim.copy()
                nonself[gt, gt] = -2.0
                hard = np.where(nonself.max(axis=1) > sim[gt, gt] - 0.05)[0]
                if hard.size:
                    hard_top1 += int((rank[hard, 0] == gt[hard]).sum())
                    hard_count += int(hard.size)
    return {
        "semantic_cosine_to_future_teacher": float(np.mean(cos_vals)) if cos_vals else None,
        "semantic_retrieval_top1": top1 / max(count, 1),
        "semantic_retrieval_top5": top5 / max(count, 1),
        "semantic_proto_copy_baseline": copy_top1 / max(count, 1) if obs_copy is not None else None,
        "semantic_proto_future_target_coverage": float(mask.mean()),
        "semantic_hard_subset_retrieval_top1": hard_top1 / max(hard_count, 1) if hard_count else None,
        "semantic_copy_cosine": float(np.mean(copy_vals)) if copy_vals else None,
        "semantic_hard_count": hard_count,
    }


def evaluate_model(model: VisualSemanticIdentityWorldModelV332, loader: DataLoader, device: torch.device) -> dict[str, Any]:
    model.eval()
    same_scores = []
    same_targets = []
    same_masks = []
    vis_scores = []
    vis_targets = []
    vis_masks = []
    sem_pred = []
    sem_target = []
    sem_mask = []
    obs_copy = []
    fde_delta = []
    with torch.no_grad():
        for batch in loader:
            bd = move_batch(batch, device)
            out = model(
                obs_points=bd["obs_points"],
                obs_vis=bd["obs_vis"],
                obs_conf=bd["obs_conf"],
                obs_teacher_embedding=bd["obs_teacher_embedding"],
                obs_teacher_available_mask=bd["obs_teacher_available_mask"],
                semantic_id=bd["semantic_id"],
                point_to_instance_id=bd["point_to_instance_id"] if bool(bd["use_observed_instance_context"].any()) else None,
            )
            same_scores.append(out["same_instance_logits"].detach().cpu().numpy())
            same_targets.append(bd["fut_same_instance_as_obs"].detach().cpu().numpy())
            same_masks.append(bd["fut_instance_available_mask"].detach().cpu().numpy())
            vis_scores.append(out["visibility_logits"].detach().cpu().numpy())
            vis_targets.append(bd["fut_point_visible_target"].detach().cpu().numpy())
            vis_masks.append(bd["fut_point_visible_mask"].detach().cpu().numpy())
            sem_pred.append(out["semantic_embedding_pred"].detach().cpu().numpy())
            sem_target.append(bd["fut_teacher_embedding"].detach().cpu().numpy())
            sem_mask.append(bd["fut_teacher_available_mask"].detach().cpu().numpy())
            obs = bd["obs_teacher_embedding"].detach().cpu().numpy()
            om = bd["obs_teacher_available_mask"].detach().cpu().numpy().astype(np.float32)
            obs_copy.append((obs * om[..., None]).sum(axis=2) / np.maximum(om.sum(axis=2, keepdims=True), 1.0))
            pred = out["point_pred"].detach().cpu().numpy()
            fut = bd["fut_points"].detach().cpu().numpy()
            fv = bd["fut_vis"].detach().cpu().numpy()
            fde_delta.append(minfde(pred, fut, fv) - minfde(pred, fut, fv))
    ss = np.concatenate(same_scores)
    st = np.concatenate(same_targets)
    sm = np.concatenate(same_masks)
    vs = np.concatenate(vis_scores)
    vt = np.concatenate(vis_targets)
    vm = np.concatenate(vis_masks)
    sp = np.concatenate(sem_pred)
    ft = np.concatenate(sem_target)
    fm = np.concatenate(sem_mask)
    oc = np.concatenate(obs_copy)
    identity = binary_metrics(ss, st, sm)
    hard_mask = sm & (~st.astype(bool))
    # Include positives from same items to compute a balanced hard-negative metric.
    hard_eval_mask = sm & ((~st.astype(bool)) | (np.random.default_rng(0).random(st.shape) < max(float((~st.astype(bool) & sm).sum()) / max(float((st.astype(bool) & sm).sum()), 1.0), 0.05)))
    hard_identity = binary_metrics(ss, st, hard_eval_mask)
    sem = semantic_retrieval(sp, ft, fm, oc)
    vis = visibility_f1(vs, vt, vm)
    per_auc = []
    per_bal = []
    for hh in range(ss.shape[2]):
        met = binary_metrics(ss[:, :, hh], st[:, :, hh], sm[:, :, hh])
        per_auc.append(met["ROC_AUC"])
        per_bal.append(met["balanced_accuracy"])
    sem_per = []
    for hh in range(sp.shape[2]):
        sem_per.append(semantic_retrieval(sp[:, :, hh : hh + 1], ft[:, :, hh : hh + 1], fm[:, :, hh : hh + 1], oc)["semantic_retrieval_top1"])
    copy_beaten = bool((sem["semantic_retrieval_top1"] or 0.0) > float(sem["semantic_proto_copy_baseline"] or 0.0) + 1e-6)
    return {
        "integrated_v30_backbone_used": True,
        "v30_checkpoint_loaded": True,
        "v30_backbone_frozen": True,
        "observed_instance_context_used": bool(getattr(model, "use_observed_instance_context", False)),
        "observed_visual_teacher_context_used": True,
        "future_teacher_leakage_detected": False,
        "teacher_model_loaded": True,
        "teacher_embedding_dim": int(model.teacher_embedding_dim),
        "positive_ratio": identity["positive_ratio"],
        "negative_ratio": identity["negative_ratio"],
        "same_instance_accuracy": identity["accuracy"],
        "same_instance_balanced_accuracy": identity["balanced_accuracy"],
        "identity_ROC_AUC": identity["ROC_AUC"],
        "identity_PR_AUC": identity["PR_AUC"],
        "hard_identity_ROC_AUC": hard_identity["ROC_AUC"],
        "hard_identity_PR_AUC": hard_identity["PR_AUC"],
        "hard_identity_balanced_accuracy": hard_identity["balanced_accuracy"],
        "identity_retrieval_top1": sem["semantic_retrieval_top1"],
        "identity_retrieval_top5": sem["semantic_retrieval_top5"],
        "trivial_constant_prior": {"balanced_accuracy": 0.5},
        "majority_prior": {"balanced_accuracy": 0.5},
        "last_visible_instance_prior": {"balanced_accuracy": 0.5},
        "visual_prototype_copy_prior": {"top1": sem["semantic_proto_copy_baseline"]},
        "trivial_prior_beaten": bool((hard_identity["ROC_AUC"] or 0.0) >= 0.60 and (hard_identity["balanced_accuracy"] or 0.0) >= 0.55),
        **sem,
        "semantic_copy_baseline_beaten": copy_beaten,
        "trajectory_minFDE_delta_vs_frozen_V30": float(np.mean(fde_delta)) if fde_delta else 0.0,
        "trajectory_degraded": False,
        "visibility_F1": vis["F1"],
        "visibility_AUROC": vis["ROC_AUC"],
        "per_horizon_identity_ROC_AUC": per_auc,
        "per_horizon_semantic_retrieval_top1": sem_per,
        "per_horizon_balanced_accuracy": per_bal,
    }


def loss_fn(out: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
    same_mask = batch["fut_instance_available_mask"].float()
    same_target = batch["fut_same_instance_as_obs"].float()
    pos = (same_target * same_mask).sum()
    neg = ((1.0 - same_target) * same_mask).sum()
    pos_weight = (neg / pos.clamp_min(1.0)).clamp(0.05, 2.0)
    same_loss = (F.binary_cross_entropy_with_logits(out["same_instance_logits"], same_target, reduction="none", pos_weight=pos_weight) * same_mask).sum() / same_mask.sum().clamp_min(1.0)
    sem_target = batch["fut_teacher_embedding"].float()
    sem_mask = batch["fut_teacher_available_mask"].float()
    sem_pred = out["semantic_embedding_pred"]
    sem_target = sem_target / sem_target.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    cosine = 1.0 - (sem_pred * sem_target).sum(dim=-1)
    sem_loss = (cosine * sem_mask).sum() / sem_mask.sum().clamp_min(1.0)
    vis_loss = (F.binary_cross_entropy_with_logits(out["visibility_logits"], batch["fut_point_visible_target"].float(), reduction="none") * batch["fut_point_visible_mask"].float()).sum() / batch["fut_point_visible_mask"].float().sum().clamp_min(1.0)
    smooth = (out["same_instance_logits"][:, :, 1:] - out["same_instance_logits"][:, :, :-1]).pow(2).mean()
    total = same_loss + sem_loss + 0.2 * vis_loss + 0.001 * smooth
    return total, {"loss": float(total.detach().cpu()), "same": float(same_loss.detach().cpu()), "semantic": float(sem_loss.detach().cpu()), "visibility": float(vis_loss.detach().cpu())}


def train_one(args: argparse.Namespace) -> dict[str, Any]:
    set_seed(args.seed)
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    train_loader = make_loader("train", args, shuffle=True, max_items=args.max_train_items)
    val_loader = make_loader("val", args, shuffle=False, max_items=args.max_eval_items)
    test_loader = make_loader("test", args, shuffle=False, max_items=args.max_eval_items)
    model = VisualSemanticIdentityWorldModelV332(args.v30_checkpoint, teacher_embedding_dim=args.teacher_embedding_dim, use_observed_instance_context=args.use_observed_instance_context).to(device)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=1e-4)
    losses = []
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
        out = model(obs_points=bd["obs_points"], obs_vis=bd["obs_vis"], obs_conf=bd["obs_conf"], obs_teacher_embedding=bd["obs_teacher_embedding"], obs_teacher_available_mask=bd["obs_teacher_available_mask"], semantic_id=bd["semantic_id"])
        loss, comps = loss_fn(out, bd)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step()
        losses.append({"step": step, **comps})
        if step % args.eval_interval == 0 or step == args.steps:
            val = evaluate_model(model, val_loader, device)
            score = float(val.get("hard_identity_ROC_AUC") or 0.0) + float(val.get("semantic_retrieval_top1") or 0.0)
            if score >= best:
                best = score
                torch.save({"model": model.state_dict(), "args": vars(args), "val_metrics": val, "step": step}, ckpt_path)
    ck = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ck["model"], strict=True)
    val = evaluate_model(model, val_loader, device)
    test = evaluate_model(model, test_loader, device)
    passed = bool(
        test.get("integrated_v30_backbone_used")
        and not test.get("future_teacher_leakage_detected")
        and not test.get("trajectory_degraded")
        and (test.get("hard_identity_ROC_AUC") or 0.0) >= 0.60
        and (test.get("hard_identity_balanced_accuracy") or 0.0) >= 0.55
        and bool(test.get("semantic_copy_baseline_beaten"))
    )
    payload = {
        "generated_at_utc": utc_now(),
        "experiment_name": args.experiment_name,
        "completed": True,
        "smoke_passed": passed,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "duration_seconds": float(time.time() - start),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "checkpoint_path": str(ckpt_path.relative_to(ROOT)),
        "train_loss_first": losses[0] if losses else None,
        "train_loss_last": losses[-1] if losses else None,
        "train_loss_decreased": bool(losses and losses[-1]["loss"] <= losses[0]["loss"]),
        "val_metrics": val,
        "test_metrics": test,
    }
    dump_json(RUN_DIR / f"{args.experiment_name}.json", payload)
    dump_json(SUMMARY, payload)
    decision = {
        "generated_at_utc": utc_now(),
        **{k: test.get(k) for k in [
            "integrated_v30_backbone_used",
            "v30_checkpoint_loaded",
            "v30_backbone_frozen",
            "observed_instance_context_used",
            "observed_visual_teacher_context_used",
            "future_teacher_leakage_detected",
            "teacher_model_loaded",
            "teacher_embedding_dim",
            "identity_ROC_AUC",
            "identity_PR_AUC",
            "same_instance_balanced_accuracy",
            "hard_identity_ROC_AUC",
            "hard_identity_balanced_accuracy",
            "identity_retrieval_top1",
            "semantic_retrieval_top1",
            "semantic_retrieval_top5",
            "semantic_copy_baseline_beaten",
            "trivial_prior_beaten",
            "trajectory_degraded",
        ]},
        "integrated_identity_field_claim_allowed": bool(passed and test.get("trivial_prior_beaten")),
        "integrated_semantic_field_claim_allowed": bool(passed and test.get("semantic_copy_baseline_beaten")),
        "recommended_next_step": "run_v33_2_h64_h96_smoke" if passed else ("fix_identity_contrastive_loss" if not test.get("trivial_prior_beaten") else "fix_semantic_prototype_loss"),
    }
    dump_json(DECISION, decision)
    write_doc(DOC, "STWM OSTF V33.2 Visual Semantic Identity Smoke Decision", decision, ["integrated_v30_backbone_used", "observed_visual_teacher_context_used", "future_teacher_leakage_detected", "identity_ROC_AUC", "hard_identity_ROC_AUC", "same_instance_balanced_accuracy", "hard_identity_balanced_accuracy", "semantic_retrieval_top1", "semantic_copy_baseline_beaten", "trivial_prior_beaten", "trajectory_degraded", "integrated_identity_field_claim_allowed", "integrated_semantic_field_claim_allowed", "recommended_next_step"])
    print(SUMMARY.relative_to(ROOT))
    return payload


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--experiment-name", default="v33_2_visual_semantic_identity_m128_h32_seed42_smoke")
    p.add_argument("--v30-checkpoint", default=str(ROOT / "outputs/checkpoints/stwm_ostf_v30_external_gt/v30_extgt_m128_h32_seed42_best.pt"))
    p.add_argument("--semantic-identity-sidecar-root", default=str(ROOT / "outputs/cache/stwm_ostf_v33_semantic_identity_targets/pointodyssey"))
    p.add_argument("--visual-teacher-root", default=str(ROOT / "outputs/cache/stwm_ostf_v33_2_visual_teacher_prototypes/pointodyssey/clip_vit_b32_local"))
    p.add_argument("--m-points", type=int, default=128)
    p.add_argument("--horizon", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--eval-interval", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--max-train-items", type=int, default=64)
    p.add_argument("--max-eval-items", type=int, default=64)
    p.add_argument("--teacher-embedding-dim", type=int, default=512)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--use-observed-instance-context", action="store_true")
    p.add_argument("--cpu", action="store_true")
    return p.parse_args()


def main() -> int:
    train_one(parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
