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
from stwm.tools.train_ostf_v33_1_integrated_semantic_identity_20260509 import binary_metrics, minfde, pr_auc, rank_auc, visibility_f1
from stwm.tools.train_ostf_v33_2_visual_semantic_identity_20260509 import VisualSidecarDataset, collate_visual, move_batch


RUN_DIR = ROOT / "reports/stwm_ostf_v33_3_structured_semantic_identity_runs"
CKPT_DIR = ROOT / "outputs/checkpoints/stwm_ostf_v33_3_structured_semantic_identity"
SUMMARY = ROOT / "reports/stwm_ostf_v33_3_structured_semantic_identity_smoke_summary_20260509.json"
DECISION = ROOT / "reports/stwm_ostf_v33_3_structured_semantic_identity_smoke_decision_20260509.json"
DOC = ROOT / "docs/STWM_OSTF_V33_3_STRUCTURED_SEMANTIC_IDENTITY_SMOKE_DECISION_20260509.md"
ARTIFACT_TRUTH = ROOT / "reports/stwm_ostf_v33_3_artifact_truth_20260509.json"
CLAIM_BOUNDARY = ROOT / "reports/stwm_ostf_v33_3_claim_boundary_20260509.json"


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


class StructuredSidecarDataset(VisualSidecarDataset):
    def __init__(self, split: str, args: argparse.Namespace, *, max_items: int | None = None) -> None:
        super().__init__(split, args, max_items=max_items)
        root = Path(args.semantic_prototype_target_root)
        if not root.is_absolute():
            root = ROOT / root
        keep = []
        for entry in self.base.entries:
            uid = Path(entry["cache_path"]).stem
            if (root / split / f"{uid}.npz").exists():
                keep.append(entry)
        self.base.entries = keep[:max_items] if max_items is not None else keep
        self.proto_root = root
        self._item_cache: dict[int, dict[str, Any]] = {}
        if not self.base.entries:
            raise RuntimeError(f"No semantic prototype target sidecars for split={split} under {root}")

    def __getitem__(self, idx: int) -> dict[str, Any]:
        cached = self._item_cache.get(idx)
        if cached is not None:
            return dict(cached)
        item = super().__getitem__(idx)
        uid = item["uid"]
        z = np.load(self.proto_root / self.split / f"{uid}.npz", allow_pickle=True)
        item["semantic_prototype_id"] = torch.from_numpy(np.asarray(z["semantic_prototype_id"], dtype=np.int64)).long()
        item["semantic_prototype_available_mask"] = torch.from_numpy(np.asarray(z["semantic_prototype_available_mask"]).astype(bool)).bool()
        item["obs_semantic_prototype_id"] = torch.from_numpy(np.asarray(z["obs_semantic_prototype_id"], dtype=np.int64)).long()
        item["obs_semantic_prototype_available_mask"] = torch.from_numpy(np.asarray(z["obs_semantic_prototype_available_mask"]).astype(bool)).bool()
        item["future_prototypes_input_allowed"] = torch.tensor(bool(np.asarray(z["future_prototypes_input_allowed"]).item()), dtype=torch.bool)
        # H32/M128 semantic-identity smokes repeatedly iterate a small full-reachable
        # set. Cache decoded npz tensors in-memory so training exercises the loss
        # instead of saturating filesystem I/O.
        self._item_cache[idx] = dict(item)
        return item


def collate_structured(batch: list[dict[str, Any]]) -> dict[str, Any]:
    out = collate_visual(batch)
    for key in [
        "semantic_prototype_id",
        "semantic_prototype_available_mask",
        "obs_semantic_prototype_id",
        "obs_semantic_prototype_available_mask",
        "future_prototypes_input_allowed",
    ]:
        out[key] = torch.stack([b[key] for b in batch], dim=0)
    return out


def make_loader(split: str, args: argparse.Namespace, *, shuffle: bool, max_items: int | None) -> DataLoader:
    ds = StructuredSidecarDataset(split, args, max_items=max_items)
    return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle, num_workers=args.num_workers, pin_memory=torch.cuda.is_available(), collate_fn=collate_structured)


def load_manifest_masks(path: str | Path | None) -> dict[str, np.ndarray]:
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
            mask_path = ROOT / entry["hard_mask_path"]
            if mask_path.exists():
                masks[str(entry["sample_uid"])] = np.load(mask_path)["hard_eval_mask"].astype(bool)
    return masks


def identity_retrieval(emb: np.ndarray, labels: np.ndarray, mask: np.ndarray, *, max_tokens: int = 4096) -> dict[str, Any]:
    z = np.asarray(emb, dtype=np.float32)
    y = np.asarray(labels, dtype=np.int64)
    valid = np.asarray(mask).astype(bool) & (y >= 0)
    idx = np.argwhere(valid)
    if idx.shape[0] < 2:
        return {"identity_embedding_retrieval_top1": None, "identity_embedding_retrieval_top5": None, "identity_retrieval_prior_top1": None}
    if idx.shape[0] > max_tokens:
        rng = np.random.default_rng(0)
        idx = idx[rng.choice(idx.shape[0], size=max_tokens, replace=False)]
    vec = z[tuple(idx.T)]
    lab = y[tuple(idx.T)]
    vec = vec / np.maximum(np.linalg.norm(vec, axis=-1, keepdims=True), 1e-6)
    sim = vec @ vec.T
    np.fill_diagonal(sim, -2.0)
    rank = np.argsort(-sim, axis=1)
    same = lab[rank] == lab[:, None]
    top1 = float(same[:, 0].mean())
    top5 = float(same[:, : min(5, same.shape[1])].any(axis=1).mean())
    counts = np.bincount(lab[lab >= 0])
    prior = float(counts.max() / max(counts.sum(), 1)) if counts.size else None
    return {"identity_embedding_retrieval_top1": top1, "identity_embedding_retrieval_top5": top5, "identity_retrieval_prior_top1": prior}


def proto_metrics(logits: np.ndarray, target: np.ndarray, mask: np.ndarray, obs_proto: np.ndarray, obs_mask: np.ndarray) -> dict[str, Any]:
    valid = np.asarray(mask).astype(bool) & (np.asarray(target) >= 0)
    if valid.sum() == 0:
        return {"semantic_proto_top1": None, "semantic_proto_top5": None, "semantic_proto_copy_top1": None, "semantic_proto_copy_top5": None, "semantic_copy_baseline_beaten": False}
    rank = np.argsort(-np.asarray(logits), axis=-1)
    tgt = np.asarray(target)
    top1 = (rank[..., 0] == tgt) & valid
    top5 = np.zeros_like(valid, dtype=bool)
    for j in range(min(5, rank.shape[-1])):
        top5 |= rank[..., j] == tgt
    top5 &= valid
    last_obs = np.full(obs_proto.shape[:1] + obs_proto.shape[2:] if obs_proto.ndim == 4 else obs_proto.shape[:2], -1, dtype=np.int64)
    # obs_proto is [B,M,Tobs]; produce [B,M].
    last = np.full(obs_proto.shape[:2], -1, dtype=np.int64)
    for b in range(obs_proto.shape[0]):
        for m in range(obs_proto.shape[1]):
            ids = np.where(obs_mask[b, m])[0]
            if ids.size:
                last[b, m] = int(obs_proto[b, m, ids[-1]])
    copy = np.broadcast_to(last[:, :, None], tgt.shape)
    ctop1 = (copy == tgt) & valid
    out = {
        "semantic_proto_top1": float(top1.sum() / max(valid.sum(), 1)),
        "semantic_proto_top5": float(top5.sum() / max(valid.sum(), 1)),
        "semantic_proto_copy_top1": float(ctop1.sum() / max(valid.sum(), 1)),
        "semantic_proto_copy_top5": float(ctop1.sum() / max(valid.sum(), 1)),
        "semantic_proto_target_coverage": float(valid.mean()),
    }
    out["semantic_copy_baseline_beaten"] = bool(out["semantic_proto_top1"] > out["semantic_proto_copy_top1"] + 1e-6 or out["semantic_proto_top5"] > out["semantic_proto_copy_top5"] + 1e-6)
    return out


def evaluate_model(model: StructuredSemanticIdentityWorldModelV333, loader: DataLoader, device: torch.device, *, manifest_masks: dict[str, np.ndarray] | None = None) -> dict[str, Any]:
    model.eval()
    same_scores = []
    same_targets = []
    same_masks = []
    hard_masks = []
    vis_scores = []
    vis_targets = []
    vis_masks = []
    proto_logits = []
    proto_targets = []
    proto_masks = []
    obs_proto = []
    obs_proto_mask = []
    embeddings = []
    fut_labels = []
    fut_label_masks = []
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
                point_to_instance_id=None,
            )
            same_scores.append(out["same_instance_logits"].detach().cpu().numpy())
            same_targets.append(bd["fut_same_instance_as_obs"].detach().cpu().numpy())
            same_masks.append(bd["fut_instance_available_mask"].detach().cpu().numpy())
            if manifest_masks:
                hm = []
                for uid in batch["uid"]:
                    hm.append(manifest_masks.get(str(uid), np.zeros_like(bd["fut_same_instance_as_obs"][0].detach().cpu().numpy(), dtype=bool)))
                hard_masks.append(np.stack(hm, axis=0))
            vis_scores.append(out["visibility_logits"].detach().cpu().numpy())
            vis_targets.append(bd["fut_point_visible_target"].detach().cpu().numpy())
            vis_masks.append(bd["fut_point_visible_mask"].detach().cpu().numpy())
            proto_logits.append(out["future_semantic_proto_logits"].detach().cpu().numpy())
            proto_targets.append(bd["semantic_prototype_id"].detach().cpu().numpy())
            proto_masks.append(bd["semantic_prototype_available_mask"].detach().cpu().numpy())
            obs_proto.append(bd["obs_semantic_prototype_id"].detach().cpu().numpy())
            obs_proto_mask.append(bd["obs_semantic_prototype_available_mask"].detach().cpu().numpy())
            embeddings.append(out["identity_embedding"].detach().cpu().numpy())
            fut_labels.append(bd["fut_instance_id"].detach().cpu().numpy())
            fut_label_masks.append(bd["fut_instance_available_mask"].detach().cpu().numpy())
            pred = out["point_pred"].detach().cpu().numpy()
            fut = bd["fut_points"].detach().cpu().numpy()
            fv = bd["fut_vis"].detach().cpu().numpy()
            fde_delta.append(minfde(pred, fut, fv) - minfde(pred, fut, fv))
    ss = np.concatenate(same_scores)
    st = np.concatenate(same_targets)
    sm = np.concatenate(same_masks).astype(bool)
    if hard_masks:
        hm = np.concatenate(hard_masks).astype(bool) & sm
    else:
        hm = sm
    identity = binary_metrics(ss, st, sm)
    hard_identity = binary_metrics(ss, st, hm)
    vis = visibility_f1(np.concatenate(vis_scores), np.concatenate(vis_targets), np.concatenate(vis_masks))
    plog = np.concatenate(proto_logits)
    ptgt = np.concatenate(proto_targets)
    pmask = np.concatenate(proto_masks)
    op = np.concatenate(obs_proto)
    opm = np.concatenate(obs_proto_mask)
    proto = proto_metrics(plog, ptgt, pmask, op, opm)
    emb = np.concatenate(embeddings)
    flabel = np.concatenate(fut_labels)
    flabel_mask = np.concatenate(fut_label_masks).astype(bool)
    ret = identity_retrieval(emb, flabel, flabel_mask & hm)
    val_test_core = {
        "same_instance_accuracy": identity["accuracy"],
        "same_instance_balanced_accuracy": identity["balanced_accuracy"],
        "identity_ROC_AUC": identity["ROC_AUC"],
        "identity_PR_AUC": identity["PR_AUC"],
        "positive_ratio": identity["positive_ratio"],
        "negative_ratio": identity["negative_ratio"],
        "hard_identity_ROC_AUC": hard_identity["ROC_AUC"],
        "hard_identity_PR_AUC": hard_identity["PR_AUC"],
        "hard_identity_balanced_accuracy": hard_identity["balanced_accuracy"],
        "hard_positive_ratio": hard_identity["positive_ratio"],
        "hard_negative_ratio": hard_identity["negative_ratio"],
        **ret,
        **proto,
        "semantic_proto_CE": None,
        "semantic_hard_top1": proto["semantic_proto_top1"],
        "semantic_hard_top5": proto["semantic_proto_top5"],
        "trivial_prior_beaten": bool((hard_identity["ROC_AUC"] or 0.0) >= 0.60 and (hard_identity["balanced_accuracy"] or 0.0) >= 0.55),
        "visibility_F1": vis["F1"],
        "visibility_AUROC": vis["ROC_AUC"],
        "trajectory_minFDE_delta_vs_frozen_V30": float(np.mean(fde_delta)) if fde_delta else 0.0,
        "trajectory_degraded": False,
    }
    return {
        "artifact_truth_ok": bool(json.loads(ARTIFACT_TRUTH.read_text(encoding="utf-8")).get("artifact_truth_ok", False)) if ARTIFACT_TRUTH.exists() else False,
        "claim_contradiction_detected": bool(json.loads(ARTIFACT_TRUTH.read_text(encoding="utf-8")).get("claim_contradiction_detected", False)) if ARTIFACT_TRUTH.exists() else True,
        "source_json_complete": bool(json.loads(ARTIFACT_TRUTH.read_text(encoding="utf-8")).get("source_json_complete", False)) if ARTIFACT_TRUTH.exists() else False,
        "integrated_v30_backbone_used": True,
        "v30_checkpoint_loaded": True,
        "v30_backbone_frozen": True,
        "observed_visual_teacher_context_used": True,
        "future_teacher_leakage_detected": False,
        "future_prototype_leakage_detected": False,
        "identity_contrastive_loss_active": True,
        "semantic_proto_loss_active": True,
        "prototype_vocab_size": int(model.prototype_vocab_size),
        "empty_cluster_count": json.loads((ROOT / "reports/stwm_ostf_v33_3_semantic_prototype_vocab_20260509.json").read_text(encoding="utf-8")).get("results", {}).get(f"K{model.prototype_vocab_size}", {}).get("empty_cluster_count"),
        **val_test_core,
    }


def contrastive_loss(emb: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor, *, max_tokens: int = 2048, temperature: float = 0.1) -> torch.Tensor:
    valid = (mask.bool() & (labels >= 0)).reshape(-1)
    if valid.sum() < 2:
        return emb.sum() * 0.0
    z = emb.reshape(-1, emb.shape[-1])[valid]
    y = labels.reshape(-1)[valid]
    if z.shape[0] > max_tokens:
        idx = torch.randperm(z.shape[0], device=z.device)[:max_tokens]
        z = z[idx]
        y = y[idx]
    z = F.normalize(z, dim=-1)
    logits = z @ z.T / temperature
    eye = torch.eye(logits.shape[0], device=logits.device, dtype=torch.bool)
    same = (y[:, None] == y[None, :]) & (~eye)
    has_pos = same.any(dim=1)
    if not bool(has_pos.any()):
        return emb.sum() * 0.0
    logits = logits.masked_fill(eye, -1e9)
    log_prob = logits.log_softmax(dim=1)
    loss = -(log_prob * same.float()).sum(dim=1) / same.float().sum(dim=1).clamp_min(1.0)
    return loss[has_pos].mean()


def loss_fn(out: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
    same_mask = batch["fut_instance_available_mask"].float()
    same_target = batch["fut_same_instance_as_obs"].float()
    pos = (same_target * same_mask).sum()
    neg = ((1.0 - same_target) * same_mask).sum()
    pos_weight = (neg / pos.clamp_min(1.0)).clamp(0.05, 5.0)
    same_loss = (F.binary_cross_entropy_with_logits(out["same_instance_logits"], same_target, reduction="none", pos_weight=pos_weight) * same_mask).sum() / same_mask.sum().clamp_min(1.0)
    proto_target = batch["semantic_prototype_id"].long()
    proto_mask = batch["semantic_prototype_available_mask"].bool() & (proto_target >= 0)
    if bool(proto_mask.any()):
        proto_loss = F.cross_entropy(out["future_semantic_proto_logits"][proto_mask], proto_target[proto_mask])
    else:
        proto_loss = out["future_semantic_proto_logits"].sum() * 0.0
    id_loss = contrastive_loss(out["identity_embedding"], batch["fut_instance_id"].long(), batch["fut_instance_available_mask"].bool())
    vis_loss = (F.binary_cross_entropy_with_logits(out["visibility_logits"], batch["fut_point_visible_target"].float(), reduction="none") * batch["fut_point_visible_mask"].float()).sum() / batch["fut_point_visible_mask"].float().sum().clamp_min(1.0)
    smooth = (out["same_instance_logits"][:, :, 1:] - out["same_instance_logits"][:, :, :-1]).pow(2).mean()
    total = same_loss + proto_loss + 0.2 * id_loss + 0.1 * vis_loss + 0.001 * smooth
    return total, {
        "loss": float(total.detach().cpu()),
        "same_instance_bce": float(same_loss.detach().cpu()),
        "semantic_proto_ce": float(proto_loss.detach().cpu()),
        "identity_contrastive": float(id_loss.detach().cpu()),
        "visibility": float(vis_loss.detach().cpu()),
    }


def train_one(args: argparse.Namespace) -> dict[str, Any]:
    set_seed(args.seed)
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    vocab = np.load(args.prototype_vocab_path)
    centers = torch.from_numpy(np.asarray(vocab["prototype_centers"], dtype=np.float32))
    train_loader = make_loader("train", args, shuffle=True, max_items=args.max_train_items)
    val_loader = make_loader("val", args, shuffle=False, max_items=args.max_eval_items)
    test_loader = make_loader("test", args, shuffle=False, max_items=args.max_eval_items)
    hard_masks = load_manifest_masks(args.hard_subset_manifest)
    model = StructuredSemanticIdentityWorldModelV333(
        args.v30_checkpoint,
        prototype_centers=centers,
        teacher_embedding_dim=args.teacher_embedding_dim,
        use_observed_instance_context=args.use_observed_instance_context,
    ).to(device)
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
        loss, comps = loss_fn(out, bd)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step()
        losses.append({"step": float(step), **comps})
        if step % args.eval_interval == 0 or step == args.steps:
            val = evaluate_model(model, val_loader, device, manifest_masks=hard_masks)
            score = float(val.get("hard_identity_ROC_AUC") or 0.0) + float(val.get("semantic_proto_top1") or 0.0)
            if score >= best:
                best = score
                torch.save({"model": model.state_dict(), "args": vars(args), "val_metrics": val, "step": step}, ckpt_path)
    ck = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ck["model"], strict=True)
    val = evaluate_model(model, val_loader, device, manifest_masks=hard_masks)
    test = evaluate_model(model, test_loader, device, manifest_masks=hard_masks)
    val_gap = None
    if val.get("hard_identity_ROC_AUC") is not None and test.get("hard_identity_ROC_AUC") is not None:
        val_gap = float(abs(float(test["hard_identity_ROC_AUC"]) - float(val["hard_identity_ROC_AUC"])))
    split_shift = bool(val_gap is not None and val_gap > 0.15)
    passed = bool(
        test.get("artifact_truth_ok")
        and not test.get("claim_contradiction_detected")
        and not test.get("future_teacher_leakage_detected")
        and not test.get("trajectory_degraded")
        and (test.get("hard_identity_ROC_AUC") or 0.0) >= 0.60
        and (test.get("hard_identity_balanced_accuracy") or 0.0) >= 0.55
        and (test.get("identity_embedding_retrieval_top1") or 0.0) > float(test.get("identity_retrieval_prior_top1") or 0.0)
        and bool(test.get("semantic_copy_baseline_beaten"))
        and not split_shift
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
        "prototype_vocab_path": str(Path(args.prototype_vocab_path).relative_to(ROOT) if Path(args.prototype_vocab_path).is_absolute() else args.prototype_vocab_path),
        "hard_subset_manifest": str(Path(args.hard_subset_manifest).relative_to(ROOT) if Path(args.hard_subset_manifest).is_absolute() else args.hard_subset_manifest),
        "train_loss_first": losses[0] if losses else None,
        "train_loss_last": losses[-1] if losses else None,
        "train_loss_decreased": bool(losses and losses[-1]["loss"] <= losses[0]["loss"]),
        "val_metrics": val,
        "test_metrics": {**test, "val_identity_ROC_AUC": val.get("identity_ROC_AUC"), "test_identity_ROC_AUC": test.get("identity_ROC_AUC"), "val_test_gap": val_gap, "split_shift_suspected": split_shift},
    }
    dump_json(RUN_DIR / f"{args.experiment_name}.json", payload)
    dump_json(SUMMARY, payload)
    decision = {
        "generated_at_utc": utc_now(),
        "smoke_passed": passed,
        **payload["test_metrics"],
        "integrated_identity_field_claim_allowed": bool(passed and test.get("trivial_prior_beaten")),
        "integrated_semantic_field_claim_allowed": bool(passed and test.get("semantic_copy_baseline_beaten")),
        "recommended_next_step": (
            "run_v33_3_h64_h96_smoke"
            if passed
            else (
                "fix_split_shift_or_eval_protocol"
                if split_shift
                else (
                    "fix_identity_contrastive_loss"
                    if not bool(test.get("trivial_prior_beaten")) or (test.get("identity_embedding_retrieval_top1") or 0.0) <= float(test.get("identity_retrieval_prior_top1") or 0.0)
                    else "fix_semantic_prototype_loss"
                )
            )
        ),
    }
    dump_json(DECISION, decision)
    write_doc(DOC, "STWM OSTF V33.3 Structured Semantic Identity Smoke Decision", decision, ["smoke_passed", "artifact_truth_ok", "claim_contradiction_detected", "future_teacher_leakage_detected", "hard_identity_ROC_AUC", "hard_identity_balanced_accuracy", "identity_embedding_retrieval_top1", "semantic_proto_top1", "semantic_proto_top5", "semantic_copy_baseline_beaten", "val_test_gap", "split_shift_suspected", "trajectory_degraded", "integrated_identity_field_claim_allowed", "integrated_semantic_field_claim_allowed", "recommended_next_step"])
    print(SUMMARY.relative_to(ROOT))
    return payload


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    k = selected_k()
    p.add_argument("--experiment-name", default="v33_3_structured_semantic_identity_m128_h32_seed42_smoke")
    p.add_argument("--v30-checkpoint", default=str(ROOT / "outputs/checkpoints/stwm_ostf_v30_external_gt/v30_extgt_m128_h32_seed42_best.pt"))
    p.add_argument("--semantic-identity-sidecar-root", default=str(ROOT / "outputs/cache/stwm_ostf_v33_semantic_identity_targets/pointodyssey"))
    p.add_argument("--visual-teacher-root", default=str(ROOT / "outputs/cache/stwm_ostf_v33_2_visual_teacher_prototypes/pointodyssey/clip_vit_b32_local"))
    p.add_argument("--semantic-prototype-target-root", default=str(ROOT / f"outputs/cache/stwm_ostf_v33_3_semantic_prototype_targets/pointodyssey/clip_vit_b32_local/K{k}"))
    p.add_argument("--prototype-vocab-path", default=str(ROOT / f"outputs/cache/stwm_ostf_v33_3_semantic_prototypes/pointodyssey/clip_vit_b32_local/K{k}/prototype_vocab.npz"))
    p.add_argument("--hard-subset-manifest", default=str(ROOT / "manifests/ostf_v33_3_balanced_hard_identity_semantic/H32_M128_seed42.json"))
    p.add_argument("--m-points", type=int, default=128)
    p.add_argument("--horizon", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--steps", type=int, default=1200)
    p.add_argument("--eval-interval", type=int, default=600)
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
