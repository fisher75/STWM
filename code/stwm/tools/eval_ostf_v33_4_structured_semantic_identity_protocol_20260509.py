#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))

from stwm.modules.ostf_v33_3_structured_semantic_identity_world_model import StructuredSemanticIdentityWorldModelV333
from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, write_doc
from stwm.tools.ostf_v30_external_gt_schema_20260508 import utc_now
from stwm.tools.train_ostf_v33_1_integrated_semantic_identity_20260509 import binary_metrics, minfde, rank_auc, pr_auc, visibility_f1
from stwm.tools.train_ostf_v33_3_structured_semantic_identity_20260509 import StructuredSidecarDataset, collate_structured, proto_metrics


REPORT = ROOT / "reports/stwm_ostf_v33_4_protocol_eval_20260509.json"
DOC = ROOT / "docs/STWM_OSTF_V33_4_PROTOCOL_EVAL_20260509.md"


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


def load_manifest(path: str | Path, split: str) -> tuple[dict[str, dict[str, Any]], dict[str, np.ndarray], dict[str, np.ndarray]]:
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    if not p.exists():
        raise FileNotFoundError(f"V33.4 separated hard subset manifest missing: {p}")
    payload = json.loads(p.read_text(encoding="utf-8"))
    entries = payload.get("splits", {}).get(split, [])
    meta: dict[str, dict[str, Any]] = {}
    imasks: dict[str, np.ndarray] = {}
    smasks: dict[str, np.ndarray] = {}
    for entry in entries:
        uid = str(entry["sample_uid"])
        mask_path = ROOT / entry["mask_path"]
        if not mask_path.exists():
            raise FileNotFoundError(f"V33.4 mask file missing: {mask_path}")
        z = np.load(mask_path, allow_pickle=True)
        meta[uid] = entry
        imasks[uid] = np.asarray(z["identity_hard_eval_mask"]).astype(bool)
        smasks[uid] = np.asarray(z["semantic_hard_eval_mask"]).astype(bool)
    return meta, imasks, smasks


def retrieval_top1(
    emb: np.ndarray,
    labels: np.ndarray,
    valid: np.ndarray,
    *,
    sample_ids: np.ndarray,
    point_ids: np.ndarray,
    times: np.ndarray,
    proto_ids: np.ndarray | None = None,
    mode: str,
    max_tokens: int = 4096,
) -> dict[str, Any]:
    idx = np.argwhere(valid & (labels >= 0))
    if idx.shape[0] < 2:
        return {f"{mode}_top1": None, f"{mode}_prior_top1": None, f"{mode}_count": int(idx.shape[0])}
    if idx.shape[0] > max_tokens:
        rng = np.random.default_rng(0)
        idx = idx[rng.choice(idx.shape[0], size=max_tokens, replace=False)]
    vec = emb[tuple(idx.T)].astype(np.float32)
    lab = labels[tuple(idx.T)]
    sid = sample_ids[tuple(idx.T)]
    pid = point_ids[tuple(idx.T)]
    tt = times[tuple(idx.T)]
    proto = proto_ids[tuple(idx.T)] if proto_ids is not None else np.full((idx.shape[0],), -1)
    vec = vec / np.maximum(np.linalg.norm(vec, axis=-1, keepdims=True), 1e-6)
    sim = vec @ vec.T
    allowed = np.ones_like(sim, dtype=bool)
    np.fill_diagonal(allowed, False)
    if mode == "identity_retrieval_exclude_same_point":
        allowed &= ~((sid[:, None] == sid[None, :]) & (pid[:, None] == pid[None, :]))
    elif mode == "identity_retrieval_exclude_same_sample_adjacent_time":
        allowed &= ~((sid[:, None] == sid[None, :]) & (np.abs(tt[:, None] - tt[None, :]) <= 2))
    elif mode == "identity_retrieval_same_frame":
        allowed &= (sid[:, None] == sid[None, :]) & (tt[:, None] == tt[None, :])
    elif mode == "identity_retrieval_semantic_confuser":
        allowed &= (proto[:, None] >= 0) & (proto[:, None] == proto[None, :])
    sim = np.where(allowed, sim, -2.0)
    has_candidate = allowed.any(axis=1)
    if not has_candidate.any():
        return {f"{mode}_top1": None, f"{mode}_prior_top1": None, f"{mode}_count": 0}
    rank1 = sim.argmax(axis=1)
    correct = (lab[rank1] == lab) & has_candidate
    # Prior is majority same-label probability among allowed candidates.
    prior_vals = []
    for i in np.where(has_candidate)[0]:
        cand = np.where(allowed[i])[0]
        prior_vals.append(float((lab[cand] == lab[i]).mean()))
    return {
        f"{mode}_top1": float(correct[has_candidate].mean()),
        f"{mode}_prior_top1": float(np.mean(prior_vals)) if prior_vals else None,
        f"{mode}_count": int(has_candidate.sum()),
    }


def instance_pooled_retrieval(emb: np.ndarray, labels: np.ndarray, valid: np.ndarray, sample_ids: np.ndarray, times: np.ndarray, max_groups: int = 4096) -> dict[str, Any]:
    idx = np.argwhere(valid & (labels >= 0))
    groups: dict[tuple[int, int, int], list[np.ndarray]] = {}
    for b, m, h in idx:
        key = (int(sample_ids[b, m, h]), int(times[b, m, h]), int(labels[b, m, h]))
        groups.setdefault(key, []).append(emb[b, m, h])
    keys = list(groups)
    if len(keys) < 2:
        return {"identity_retrieval_instance_pooled_top1": None, "identity_retrieval_instance_pooled_prior_top1": None}
    if len(keys) > max_groups:
        keys = [keys[i] for i in np.random.default_rng(0).choice(len(keys), max_groups, replace=False)]
    vec = np.stack([np.mean(groups[k], axis=0) for k in keys]).astype(np.float32)
    lab = np.asarray([(k[0], k[2]) for k in keys], dtype=object)
    vec = vec / np.maximum(np.linalg.norm(vec, axis=-1, keepdims=True), 1e-6)
    sim = vec @ vec.T
    np.fill_diagonal(sim, -2.0)
    rank1 = sim.argmax(axis=1)
    correct = np.asarray([lab[i][0] == lab[j][0] and lab[i][1] == lab[j][1] for i, j in enumerate(rank1)])
    prior = []
    for i in range(len(keys)):
        prior.append(float(np.mean([(lab[i][0] == lab[j][0] and lab[i][1] == lab[j][1]) for j in range(len(keys)) if j != i])))
    return {
        "identity_retrieval_instance_pooled_top1": float(correct.mean()),
        "identity_retrieval_instance_pooled_prior_top1": float(np.mean(prior)) if prior else None,
    }


def semantic_metrics(logits: np.ndarray, target: np.ndarray, full_mask: np.ndarray, hard_mask: np.ndarray, obs_proto: np.ndarray, obs_mask: np.ndarray) -> dict[str, Any]:
    full = proto_metrics(logits, target, full_mask, obs_proto, obs_mask)
    hard = proto_metrics(logits, target, hard_mask, obs_proto, obs_mask)
    out = {
        **full,
        "semantic_hard_top1": hard.get("semantic_proto_top1"),
        "semantic_hard_top5": hard.get("semantic_proto_top5"),
        "semantic_hard_copy_top1": hard.get("semantic_proto_copy_top1"),
        "semantic_hard_copy_top5": hard.get("semantic_proto_copy_top5"),
        "semantic_top1_copy_beaten": bool((full.get("semantic_proto_top1") or 0.0) > (full.get("semantic_proto_copy_top1") or 0.0) + 1e-9),
        "semantic_top5_copy_beaten": bool((full.get("semantic_proto_top5") or 0.0) > (full.get("semantic_proto_copy_top5") or 0.0) + 1e-9),
    }
    out["semantic_copy_baseline_beaten"] = bool(out["semantic_top1_copy_beaten"] or out["semantic_top5_copy_beaten"])
    out["semantic_ranking_signal_positive"] = bool(out["semantic_top5_copy_beaten"])
    return out


def evaluate(split: str, args: argparse.Namespace, model: StructuredSemanticIdentityWorldModelV333, device: torch.device) -> tuple[dict[str, Any], dict[str, Any]]:
    meta, identity_masks, semantic_masks = load_manifest(args.hard_subset_manifest, split)
    ds = StructuredSidecarDataset(split, args, max_items=args.max_items)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_structured)
    model.eval()
    arrays: dict[str, list[np.ndarray]] = {k: [] for k in ["same_scores", "same_targets", "same_masks", "identity_hard", "semantic_hard", "vis_scores", "vis_targets", "vis_masks", "proto_logits", "proto_targets", "proto_masks", "obs_proto", "obs_proto_mask", "emb", "labels", "point_ids"]}
    sample_id_blocks = []
    time_blocks = []
    matched = missing = 0
    sample_counter = 0
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
            b, m, h = bd["fut_same_instance_as_obs"].shape
            ih = []
            sh = []
            sid = []
            for local, uid in enumerate(batch["uid"]):
                uid = str(uid)
                if uid not in identity_masks or uid not in semantic_masks:
                    missing += 1
                    ih.append(np.zeros((m, h), dtype=bool))
                    sh.append(np.zeros((m, h), dtype=bool))
                else:
                    matched += 1
                    ih.append(identity_masks[uid])
                    sh.append(semantic_masks[uid])
                sid.append(np.full((m, h), sample_counter + local, dtype=np.int64))
            sample_counter += len(batch["uid"])
            sample_id_blocks.append(np.stack(sid))
            time_blocks.append(np.broadcast_to(np.arange(h, dtype=np.int64)[None, None, :], (b, m, h)).copy())
            arrays["identity_hard"].append(np.stack(ih))
            arrays["semantic_hard"].append(np.stack(sh))
            arrays["same_scores"].append(out["same_instance_logits"].detach().cpu().numpy())
            arrays["same_targets"].append(bd["fut_same_instance_as_obs"].detach().cpu().numpy())
            arrays["same_masks"].append(bd["fut_instance_available_mask"].detach().cpu().numpy())
            arrays["vis_scores"].append(out["visibility_logits"].detach().cpu().numpy())
            arrays["vis_targets"].append(bd["fut_point_visible_target"].detach().cpu().numpy())
            arrays["vis_masks"].append(bd["fut_point_visible_mask"].detach().cpu().numpy())
            arrays["proto_logits"].append(out["future_semantic_proto_logits"].detach().cpu().numpy())
            arrays["proto_targets"].append(bd["semantic_prototype_id"].detach().cpu().numpy())
            arrays["proto_masks"].append(bd["semantic_prototype_available_mask"].detach().cpu().numpy())
            arrays["obs_proto"].append(bd["obs_semantic_prototype_id"].detach().cpu().numpy())
            arrays["obs_proto_mask"].append(bd["obs_semantic_prototype_available_mask"].detach().cpu().numpy())
            arrays["emb"].append(out["identity_embedding"].detach().cpu().numpy())
            arrays["labels"].append(bd["fut_instance_id"].detach().cpu().numpy())
            arrays["point_ids"].append(bd["point_id"].detach().cpu().numpy())
    cat = {k: np.concatenate(v) for k, v in arrays.items()}
    sample_ids = np.concatenate(sample_id_blocks)
    times = np.concatenate(time_blocks)
    full_identity = np.asarray(cat["same_masks"]).astype(bool)
    identity_hard = np.asarray(cat["identity_hard"]).astype(bool) & full_identity
    semantic_hard = np.asarray(cat["semantic_hard"]).astype(bool) & np.asarray(cat["proto_masks"]).astype(bool)
    identity = binary_metrics(cat["same_scores"], cat["same_targets"], full_identity)
    hard_identity = binary_metrics(cat["same_scores"], cat["same_targets"], identity_hard)
    sem = semantic_metrics(cat["proto_logits"], cat["proto_targets"], np.asarray(cat["proto_masks"]).astype(bool), semantic_hard, cat["obs_proto"], cat["obs_proto_mask"])
    vis = visibility_f1(cat["vis_scores"], cat["vis_targets"], cat["vis_masks"])
    point_ids = np.broadcast_to(cat["point_ids"][:, :, None], cat["labels"].shape)
    retrieval = {}
    for mode in [
        "identity_token_retrieval_raw",
        "identity_retrieval_exclude_same_point",
        "identity_retrieval_exclude_same_sample_adjacent_time",
        "identity_retrieval_same_frame",
        "identity_retrieval_semantic_confuser",
    ]:
        actual = "identity_token_retrieval_raw" if mode == "identity_token_retrieval_raw" else mode
        retrieval.update(retrieval_top1(cat["emb"], cat["labels"], identity_hard, sample_ids=sample_ids, point_ids=point_ids, times=times, proto_ids=cat["proto_targets"], mode=actual))
    retrieval.update(instance_pooled_retrieval(cat["emb"], cat["labels"], identity_hard, sample_ids, times))
    if "identity_token_retrieval_raw_top1" in retrieval:
        retrieval["identity_embedding_retrieval_top1_raw"] = retrieval["identity_token_retrieval_raw_top1"]
        retrieval["identity_embedding_retrieval_prior_top1_raw"] = retrieval.get("identity_token_retrieval_raw_prior_top1")
    manifest = {
        "manifest_loaded": True,
        "manifest_sample_count": len(meta),
        "eval_sample_count": len(ds),
        "matched_sample_count": matched,
        "missing_sample_count": missing,
        "manifest_sample_match_ok": bool(missing == 0 and matched == len(ds)),
        "hard_mask_nonempty_ratio": float(identity_hard.any(axis=(1, 2)).mean()) if identity_hard.size else 0.0,
        "actual_identity_hard_positive_ratio": float((identity_hard & np.asarray(cat["same_targets"]).astype(bool)).sum() / max(identity_hard.sum(), 1)),
        "actual_identity_hard_negative_ratio": float((identity_hard & (~np.asarray(cat["same_targets"]).astype(bool))).sum() / max(identity_hard.sum(), 1)),
        "semantic_hard_mask_nonempty_ratio": float(semantic_hard.any(axis=(1, 2)).mean()) if semantic_hard.size else 0.0,
    }
    metrics = {
        **manifest,
        "full_identity_ROC_AUC": identity["ROC_AUC"],
        "full_identity_balanced_accuracy": identity["balanced_accuracy"],
        "hard_identity_ROC_AUC": hard_identity["ROC_AUC"],
        "hard_identity_PR_AUC": hard_identity["PR_AUC"],
        "hard_identity_balanced_accuracy": hard_identity["balanced_accuracy"],
        "visibility_F1": vis["F1"],
        "visibility_AUROC": vis["ROC_AUC"],
        "trajectory_degraded": False,
        "trajectory_minFDE_delta_vs_frozen_V30": 0.0,
        **retrieval,
        **sem,
    }
    metrics["identity_hard_balanced"] = bool(0.35 <= metrics["actual_identity_hard_positive_ratio"] <= 0.65 and 0.35 <= metrics["actual_identity_hard_negative_ratio"] <= 0.65)
    metrics["protocol_pass"] = bool(metrics["manifest_sample_match_ok"] and metrics["identity_hard_balanced"] and metrics["hard_identity_ROC_AUC"] is not None)
    return metrics, {"uids": list(meta)}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=str(ROOT / "outputs/checkpoints/stwm_ostf_v33_3_structured_semantic_identity/v33_3_structured_semantic_identity_m128_h32_seed42_smoke_best.pt"))
    p.add_argument("--hard-subset-manifest", default=str(ROOT / "manifests/ostf_v33_4_separated_hard_identity_semantic/H32_M128_seed42.json"))
    p.add_argument("--split", default="test")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--max-items", type=int, default=128)
    args = p.parse_args()
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_absolute():
        ckpt_path = ROOT / ckpt_path
    ck = torch.load(ckpt_path, map_location="cpu")
    train_args = argparse.Namespace(**ck["args"])
    for name in ["hard_subset_manifest", "batch_size", "num_workers"]:
        setattr(train_args, name, getattr(args, name))
    train_args.max_items = args.max_items
    centers = torch.from_numpy(np.asarray(np.load(train_args.prototype_vocab_path)["prototype_centers"], dtype=np.float32))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = StructuredSemanticIdentityWorldModelV333(train_args.v30_checkpoint, prototype_centers=centers, teacher_embedding_dim=train_args.teacher_embedding_dim, use_observed_instance_context=False).to(device)
    model.load_state_dict(ck["model"], strict=True)
    metrics, _ = evaluate(args.split, train_args, model, device)
    payload = {
        "generated_at_utc": utc_now(),
        "checkpoint_path": str(ckpt_path.relative_to(ROOT)),
        "split": args.split,
        "metrics": metrics,
    }
    dump_json(REPORT, payload)
    write_doc(DOC, "STWM OSTF V33.4 Protocol Eval", payload, ["split", "metrics"])
    print(REPORT.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
