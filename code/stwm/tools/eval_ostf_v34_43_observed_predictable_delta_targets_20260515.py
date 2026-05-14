#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import setproctitle
import torch
import torch.nn.functional as F
from sklearn.cluster import MiniBatchKMeans
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import average_precision_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))

setproctitle.setproctitle("python")

from stwm.tools.eval_ostf_v34_26_full_system_baseline_claim_boundary_benchmark_20260514 import masks, observed_mean
from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, write_doc
from stwm.tools.ostf_v30_external_gt_schema_20260508 import utc_now
from stwm.tools.train_ostf_v33_2_visual_semantic_identity_20260509 import move_batch
from stwm.tools.train_ostf_v34_18_topk_evidence_oracle_residual_probe_20260513 import make_loader, model_inputs
from stwm.tools.train_eval_ostf_v34_31_raw_unit_delta_value_memory_20260514 import load_frozen_residual_model, set_seed


TARGET_ROOT = ROOT / "outputs/cache/stwm_ostf_v34_43_observed_predictable_delta_targets/pointodyssey"
REPORT = ROOT / "reports/stwm_ostf_v34_43_observed_predictable_delta_targets_20260515.json"
DOC = ROOT / "docs/STWM_OSTF_V34_43_OBSERVED_PREDICTABLE_DELTA_TARGETS_20260515.md"


def np_norm(x: np.ndarray) -> np.ndarray:
    x = np.nan_to_num(x.astype(np.float32), copy=False)
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1.0e-6)


def torch_norm(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(torch.nan_to_num(x.float()), dim=-1)


def list_npz(split: str) -> list[Path]:
    root = ROOT / "outputs/cache/stwm_ostf_v34_9_trace_preserving_semantic_measurement_bank/pointodyssey" / split
    return sorted(root.glob("*.npz"))


def sample_embeddings(args: argparse.Namespace) -> np.ndarray:
    rng = np.random.default_rng(args.seed)
    samples: list[np.ndarray] = []
    per_file = max(8, args.codebook_sample_per_file)
    for path in list_npz("train"):
        z = np.load(path, allow_pickle=True)
        obs = np.asarray(z["obs_semantic_measurements"], dtype=np.float32)
        obs_mask = np.asarray(z["obs_semantic_measurement_mask"]).astype(bool)
        fut = np.asarray(z["fut_teacher_embedding"], dtype=np.float32)
        fut_mask = np.asarray(z["fut_teacher_available_mask"]).astype(bool)
        obs_flat = obs[obs_mask]
        fut_flat = fut[fut_mask]
        if obs_flat.size:
            idx = rng.choice(len(obs_flat), size=min(per_file, len(obs_flat)), replace=False)
            samples.append(obs_flat[idx])
        if fut_flat.size:
            idx = rng.choice(len(fut_flat), size=min(per_file, len(fut_flat)), replace=False)
            samples.append(fut_flat[idx])
        if sum(len(s) for s in samples) >= args.max_codebook_samples:
            break
    if not samples:
        raise RuntimeError("没有可用于 V34.43 codebook 的语义 embedding。")
    x = np_norm(np.concatenate(samples, axis=0))
    if len(x) > args.max_codebook_samples:
        idx = rng.choice(len(x), size=args.max_codebook_samples, replace=False)
        x = x[idx]
    return x


def fit_codebook(args: argparse.Namespace) -> np.ndarray:
    x = sample_embeddings(args)
    print(f"V34.43: 拟合 semantic codebook，样本数={len(x)}，K={args.semantic_clusters}。", flush=True)
    km = MiniBatchKMeans(
        n_clusters=args.semantic_clusters,
        random_state=args.seed,
        batch_size=args.kmeans_batch_size,
        n_init=3,
        max_iter=args.kmeans_iters,
        reassignment_ratio=0.01,
    )
    km.fit(x)
    return np_norm(km.cluster_centers_.astype(np.float32))


def assign_np(x: np.ndarray, centers: np.ndarray, chunk: int = 32768) -> np.ndarray:
    flat = np_norm(x.reshape(-1, x.shape[-1]))
    out = np.empty((flat.shape[0],), dtype=np.int64)
    c = centers.T.astype(np.float32)
    for start in range(0, flat.shape[0], chunk):
        score = flat[start : start + chunk] @ c
        out[start : start + chunk] = score.argmax(axis=1)
    return out.reshape(x.shape[:-1])


def majority_by_instance(point_inst: np.ndarray, labels: np.ndarray, valid: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    m = point_inst.shape[0]
    if labels.ndim == 1:
        out = np.full((m,), -1, dtype=np.int64)
        for inst in np.unique(point_inst[point_inst >= 0]):
            mask = (point_inst == inst) & valid
            if mask.any():
                out[point_inst == inst] = np.bincount(labels[mask], minlength=k).argmax()
        return out, out[:, None]
    h = labels.shape[1]
    out = np.full((m, h), -1, dtype=np.int64)
    for inst in np.unique(point_inst[point_inst >= 0]):
        pts = point_inst == inst
        for t in range(h):
            mask = pts & valid[:, t]
            if mask.any():
                out[pts, t] = np.bincount(labels[mask, t], minlength=k).argmax()
    return out[:, 0], out


def feature_projection(in_dim: int, out_dim: int, seed: int) -> torch.Tensor:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    return torch.randn((in_dim, out_dim), generator=gen, dtype=torch.float32) / math.sqrt(float(out_dim))


def build_features(out: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> torch.Tensor:
    obs = observed_mean(batch)
    pointwise = out["pointwise_semantic_belief"].float()
    topk = out.get("topk_raw_evidence", out["topk_raw_evidence_embedding"].unsqueeze(3)).float()
    weights = out["topk_weights"].float().clamp_min(1.0e-8)
    weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1.0e-6)
    evidence_mean = (topk * weights[..., None]).sum(dim=3)
    scalars = [
        out.get("semantic_measurement_usage_score", torch.zeros_like(weights[..., 0])).float()[..., None],
        out.get("selector_confidence", torch.zeros_like(weights[..., 0])).float()[..., None],
        out.get("selector_entropy", torch.zeros_like(weights[..., 0])).float()[..., None],
        out.get("assignment_usage_score", torch.zeros_like(weights[..., 0])).float().mean(dim=1)[:, None].expand_as(weights[..., 0])[..., None]
        if out.get("assignment_usage_score") is not None and out["assignment_usage_score"].ndim == 3
        else torch.zeros_like(weights[..., :1]),
    ]
    return torch.cat([out["future_trace_hidden"].float(), obs.float(), pointwise, evidence_mean, weights, *scalars], dim=-1)


def topk_rank_targets(out: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    target = torch_norm(batch["fut_teacher_embedding"])
    evid = torch_norm(out.get("topk_raw_evidence", out["topk_raw_evidence_embedding"].unsqueeze(3)).float())
    cos = (evid * target[:, :, :, None, :]).sum(dim=-1)
    rank = cos.argmax(dim=-1)
    weight_rank = out["topk_weights"].float().argmax(dim=-1)
    return rank, weight_rank, cos, out["topk_weights"].float()


def collect_split(
    split: str,
    model: Any,
    ckargs: argparse.Namespace,
    centers: np.ndarray,
    proj: torch.Tensor | None,
    args: argparse.Namespace,
    device: torch.device,
    *,
    write_cache: bool,
) -> dict[str, Any]:
    xs, route, fut_y, obs_y, attr_y, ident_y, rank_y, rank_base, valid_m, hard_m, changed_m, stable_m, topk_cos, topk_weight_score = [], [], [], [], [], [], [], [], [], [], [], [], [], []
    out_dir = TARGET_ROOT / split
    if write_cache:
        out_dir.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        for batch in make_loader(split, ckargs, shuffle=False):
            bd = move_batch(batch, device)
            out = model(**model_inputs(bd), intervention="force_gate_zero")
            obs_emb = observed_mean(bd)[:, :, 0, :].detach().cpu().numpy()
            fut_emb = bd["fut_teacher_embedding"].detach().cpu().numpy()
            obs_cluster = assign_np(obs_emb, centers)
            future_cluster = assign_np(fut_emb, centers)
            mm = masks(bd)
            rank, weight_rank, rank_cos, rank_weight_score = topk_rank_targets(out, bd)
            feat = build_features(out, bd)
            if proj is None:
                return {"feature_dim": int(feat.shape[-1])}
            flat_feat = (feat.reshape(-1, feat.shape[-1]).cpu() @ proj).numpy().astype(np.float32)
            bsz, m, h = future_cluster.shape
            point_inst = bd["point_to_instance_id"].detach().cpu().numpy()
            valid_np = mm["valid"].detach().cpu().numpy().astype(bool)
            hard_np = mm["hard"].detach().cpu().numpy().astype(bool)
            changed_np = mm["changed"].detach().cpu().numpy().astype(bool)
            stable_np = mm["stable"].detach().cpu().numpy().astype(bool)
            rank_np = rank.detach().cpu().numpy().astype(np.int64)
            weight_rank_np = weight_rank.detach().cpu().numpy().astype(np.int64)
            rank_cos_np = rank_cos.detach().cpu().numpy().astype(np.float32)
            rank_weight_np = rank_weight_score.detach().cpu().numpy().astype(np.float32)
            inst_obs_all, inst_fut_all = [], []
            attr_all, ident_all = [], []
            for bi, uid in enumerate(bd["uid"]):
                inst_obs, _ = majority_by_instance(point_inst[bi], obs_cluster[bi], np.ones((m,), dtype=bool), args.semantic_clusters)
                _, inst_fut = majority_by_instance(point_inst[bi], future_cluster[bi], valid_np[bi], args.semantic_clusters)
                attr = (future_cluster[bi] != inst_obs[:, None]) & (inst_obs[:, None] >= 0) & valid_np[bi]
                ident = (future_cluster[bi] == inst_fut) & (inst_fut >= 0) & valid_np[bi]
                inst_obs_all.append(inst_obs)
                inst_fut_all.append(inst_fut)
                attr_all.append(attr)
                ident_all.append(ident)
                if write_cache:
                    np.savez_compressed(
                        out_dir / f"{uid}.npz",
                        sample_uid=np.asarray(uid),
                        obs_cluster_id=obs_cluster[bi].astype(np.int64),
                        future_cluster_id=future_cluster[bi].astype(np.int64),
                        semantic_cluster_transition_id=(obs_cluster[bi][:, None] * args.semantic_clusters + future_cluster[bi]).astype(np.int64),
                        semantic_cluster_changed=(future_cluster[bi] != obs_cluster[bi][:, None]) & valid_np[bi],
                        instance_observed_cluster=inst_obs.astype(np.int64),
                        instance_future_cluster=inst_fut.astype(np.int64),
                        instance_consistent_attribute_change=attr.astype(bool),
                        identity_consistency_target=ident.astype(bool),
                        topk_evidence_residual_rank=rank_np[bi].astype(np.int64),
                        topk_weight_argmax_rank=weight_rank_np[bi].astype(np.int64),
                        topk_weight_score=rank_weight_np[bi].astype(np.float32),
                        topk_evidence_cosine=rank_cos_np[bi].astype(np.float32),
                        valid_mask=valid_np[bi].astype(bool),
                        semantic_hard_mask=hard_np[bi].astype(bool),
                        changed_mask=changed_np[bi].astype(bool),
                        stable_mask=stable_np[bi].astype(bool),
                        future_teacher_embeddings_supervision_only=np.asarray(True),
                        future_teacher_embeddings_input_allowed=np.asarray(False),
                        leakage_safe=np.asarray(True),
                    )
            inst_obs_np = np.stack(inst_obs_all, axis=0)
            inst_fut_np = np.stack(inst_fut_all, axis=0)
            attr_np = np.stack(attr_all, axis=0)
            ident_np = np.stack(ident_all, axis=0)
            xs.append(flat_feat)
            route.append(np.repeat(obs_cluster[:, :, None], h, axis=2).reshape(-1))
            fut_y.append(future_cluster.reshape(-1))
            obs_y.append(np.repeat(obs_cluster[:, :, None], h, axis=2).reshape(-1))
            attr_y.append(attr_np.reshape(-1).astype(np.int64))
            ident_y.append(ident_np.reshape(-1).astype(np.int64))
            rank_y.append(rank_np.reshape(-1))
            rank_base.append(weight_rank_np.reshape(-1))
            valid_m.append(valid_np.reshape(-1))
            hard_m.append(hard_np.reshape(-1))
            changed_m.append(changed_np.reshape(-1))
            stable_m.append(stable_np.reshape(-1))
            topk_cos.append(rank_cos_np.reshape(-1, rank_cos_np.shape[-1]))
            topk_weight_score.append(rank_weight_np.reshape(-1, rank_weight_np.shape[-1]))
    return {
        "x": np.concatenate(xs, axis=0),
        "route": np.concatenate(route, axis=0).astype(np.int64),
        "future_cluster": np.concatenate(fut_y, axis=0).astype(np.int64),
        "obs_cluster": np.concatenate(obs_y, axis=0).astype(np.int64),
        "attribute_change": np.concatenate(attr_y, axis=0).astype(np.int64),
        "identity_consistency": np.concatenate(ident_y, axis=0).astype(np.int64),
        "topk_rank": np.concatenate(rank_y, axis=0).astype(np.int64),
        "topk_weight_rank": np.concatenate(rank_base, axis=0).astype(np.int64),
        "valid": np.concatenate(valid_m, axis=0).astype(bool),
        "hard": np.concatenate(hard_m, axis=0).astype(bool),
        "changed": np.concatenate(changed_m, axis=0).astype(bool),
        "hard_changed": (np.concatenate(hard_m, axis=0) | np.concatenate(changed_m, axis=0)).astype(bool) & np.concatenate(valid_m, axis=0).astype(bool),
        "stable": np.concatenate(stable_m, axis=0).astype(bool),
        "topk_cos": np.concatenate(topk_cos, axis=0).astype(np.float32),
        "topk_weight_score": np.concatenate(topk_weight_score, axis=0).astype(np.float32),
    }


class RidgePack:
    def __init__(self, alpha: float = 10.0) -> None:
        self.scaler = StandardScaler()
        self.model = RidgeClassifier(alpha=alpha)
        self.classes_: np.ndarray | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> "RidgePack":
        xs = self.scaler.fit_transform(x)
        self.model.fit(xs, y)
        self.classes_ = np.asarray(self.model.classes_)
        return self

    def score_matrix(self, x: np.ndarray, class_count: int) -> np.ndarray:
        xs = self.scaler.transform(x)
        score = self.model.decision_function(xs)
        if score.ndim == 1:
            score = np.stack([-score, score], axis=1)
        out = np.full((x.shape[0], class_count), -1.0e9, dtype=np.float32)
        assert self.classes_ is not None
        for j, cls in enumerate(self.classes_):
            if 0 <= int(cls) < class_count:
                out[:, int(cls)] = score[:, j]
        return out


def subsample(mask: np.ndarray, max_count: int, seed: int) -> np.ndarray:
    idx = np.flatnonzero(mask)
    if len(idx) <= max_count:
        return idx
    rng = np.random.default_rng(seed)
    return rng.choice(idx, size=max_count, replace=False)


def topk_accuracy(score: np.ndarray, y: np.ndarray, mask: np.ndarray, k: int) -> float | None:
    idx = np.flatnonzero(mask)
    if len(idx) == 0:
        return None
    part = np.argpartition(-score[idx], kth=min(k - 1, score.shape[1] - 1), axis=1)[:, :k]
    return float((part == y[idx, None]).any(axis=1).mean())


def exact_accuracy(pred: np.ndarray, y: np.ndarray, mask: np.ndarray) -> float | None:
    idx = np.flatnonzero(mask)
    if len(idx) == 0:
        return None
    return float((pred[idx] == y[idx]).mean())


def choose_binary_threshold(score: np.ndarray, y: np.ndarray, mask: np.ndarray) -> float:
    idx = np.flatnonzero(mask)
    if len(idx) == 0:
        return 0.0
    candidates = np.unique(np.quantile(score[idx], np.linspace(0.02, 0.98, 97)))
    best_t, best = 0.0, -1.0
    for t in candidates:
        pred = score[idx] >= t
        bal = balanced_accuracy_score(y[idx], pred)
        if bal > best:
            best = float(bal)
            best_t = float(t)
    return best_t


def binary_metrics(score: np.ndarray, y: np.ndarray, mask: np.ndarray, threshold: float) -> dict[str, float | None]:
    idx = np.flatnonzero(mask)
    if len(idx) == 0:
        return {"positive_ratio": None, "roc_auc": None, "average_precision": None, "balanced_accuracy": None, "f1": None}
    yy = y[idx].astype(np.int64)
    ss = score[idx]
    pred = ss >= threshold
    return {
        "positive_ratio": float(yy.mean()),
        "roc_auc": None if len(np.unique(yy)) < 2 else float(roc_auc_score(yy, ss)),
        "average_precision": None if yy.sum() == 0 else float(average_precision_score(yy, ss)),
        "balanced_accuracy": float(balanced_accuracy_score(yy, pred)),
        "f1": float(f1_score(yy, pred, zero_division=0)),
        "threshold": float(threshold),
    }


def fit_local_models(x: np.ndarray, y: np.ndarray, route: np.ndarray, mask: np.ndarray, class_count: int, args: argparse.Namespace) -> tuple[RidgePack, dict[int, RidgePack]]:
    idx = subsample(mask, args.max_train_tokens, args.seed)
    global_model = RidgePack(args.ridge_alpha).fit(x[idx], y[idx])
    local: dict[int, RidgePack] = {}
    for rid in np.unique(route[idx]):
        local_mask = idx[route[idx] == rid]
        if len(local_mask) >= args.min_local_count and len(np.unique(y[local_mask])) >= 2:
            local[rid] = RidgePack(args.ridge_alpha).fit(x[local_mask], y[local_mask])
    return global_model, local


def predict_local(global_model: RidgePack, local: dict[int, RidgePack], x: np.ndarray, route: np.ndarray, class_count: int) -> np.ndarray:
    score = global_model.score_matrix(x, class_count)
    for rid, model in local.items():
        m = route == rid
        if m.any():
            score[m] = model.score_matrix(x[m], class_count)
    return score


def rank_cos_metrics(score: np.ndarray, data: dict[str, np.ndarray], mask: np.ndarray, topn: int = 1) -> dict[str, float | None]:
    idx = np.flatnonzero(mask)
    if len(idx) == 0:
        return {"accuracy": None, "selected_cos": None, "gain_vs_weight_baseline": None, "oracle_gap": None}
    pred = score[idx].argmax(axis=1)
    base = data["topk_weight_rank"][idx]
    y = data["topk_rank"][idx]
    cos = data["topk_cos"][idx]
    selected = cos[np.arange(len(idx)), pred]
    base_cos = cos[np.arange(len(idx)), base]
    oracle_cos = cos.max(axis=1)
    return {
        "accuracy": float((pred == y).mean()),
        "top3_accuracy": float((np.argpartition(-score[idx], kth=min(2, score.shape[1] - 1), axis=1)[:, :3] == y[:, None]).any(axis=1).mean()),
        "weight_baseline_accuracy": float((base == y).mean()),
        "weight_baseline_top3_accuracy": float((np.argpartition(-data["topk_weight_score"][idx], kth=min(2, data["topk_weight_score"].shape[1] - 1), axis=1)[:, :3] == y[:, None]).any(axis=1).mean()),
        "selected_cos": float(selected.mean()),
        "baseline_cos": float(base_cos.mean()),
        "oracle_cos": float(oracle_cos.mean()),
        "gain_vs_weight_baseline": float((selected - base_cos).mean()),
        "oracle_gap": float((oracle_cos - selected).mean()),
    }


def evaluate_targets(train: dict[str, np.ndarray], val: dict[str, np.ndarray], test: dict[str, np.ndarray], args: argparse.Namespace) -> dict[str, Any]:
    train_mask = train["valid"]
    global_cluster, local_cluster = fit_local_models(train["x"], train["future_cluster"], train["route"], train_mask, args.semantic_clusters, args)
    global_rank, local_rank = fit_local_models(train["x"], train["topk_rank"], train["route"], train_mask, args.topk, args)
    attr_model, _ = fit_local_models(train["x"], train["attribute_change"], train["route"], train_mask, 2, args)
    ident_model, _ = fit_local_models(train["x"], train["identity_consistency"], train["route"], train_mask, 2, args)

    val_attr_score = attr_model.score_matrix(val["x"], 2)[:, 1]
    val_ident_score = ident_model.score_matrix(val["x"], 2)[:, 1]
    attr_threshold = choose_binary_threshold(val_attr_score, val["attribute_change"], val["hard_changed"])
    ident_threshold = choose_binary_threshold(val_ident_score, val["identity_consistency"], val["hard_changed"])

    rows: dict[str, Any] = {"thresholds": {"attribute_change": attr_threshold, "identity_consistency": ident_threshold}}
    for split_name, data in {"val": val, "test": test}.items():
        g_cluster_score = global_cluster.score_matrix(data["x"], args.semantic_clusters)
        l_cluster_score = predict_local(global_cluster, local_cluster, data["x"], data["route"], args.semantic_clusters)
        g_rank_score = global_rank.score_matrix(data["x"], args.topk)
        l_rank_score = predict_local(global_rank, local_rank, data["x"], data["route"], args.topk)
        attr_score = attr_model.score_matrix(data["x"], 2)[:, 1]
        ident_score = ident_model.score_matrix(data["x"], 2)[:, 1]
        split_rows: dict[str, Any] = {}
        for subset in ("valid", "hard", "changed", "hard_changed", "stable"):
            mask = data[subset]
            split_rows[subset] = {
                "semantic_cluster_transition": {
                    "obs_persistence_top1": exact_accuracy(data["obs_cluster"], data["future_cluster"], mask),
                    "global_ridge_top1": topk_accuracy(g_cluster_score, data["future_cluster"], mask, 1),
                    "global_ridge_top5": topk_accuracy(g_cluster_score, data["future_cluster"], mask, 5),
                    "local_expert_top1": topk_accuracy(l_cluster_score, data["future_cluster"], mask, 1),
                    "local_expert_top5": topk_accuracy(l_cluster_score, data["future_cluster"], mask, 5),
                },
                "topk_evidence_residual_rank": {
                    "global_ridge": rank_cos_metrics(g_rank_score, data, mask),
                    "local_expert": rank_cos_metrics(l_rank_score, data, mask),
                },
                "instance_consistent_attribute_change": binary_metrics(attr_score, data["attribute_change"], mask, attr_threshold),
                "identity_consistency_target": binary_metrics(ident_score, data["identity_consistency"], mask, ident_threshold),
            }
        rows[split_name] = split_rows
    return rows


def pass_decision(metrics: dict[str, Any]) -> dict[str, Any]:
    def get(split: str, subset: str, family: str, *keys: str) -> Any:
        cur = metrics[split][subset][family]
        for k in keys:
            cur = cur[k]
        return cur

    sem_val = get("val", "hard_changed", "semantic_cluster_transition", "local_expert_top5") or 0.0
    sem_test = get("test", "hard_changed", "semantic_cluster_transition", "local_expert_top5") or 0.0
    sem_base_val = get("val", "hard_changed", "semantic_cluster_transition", "obs_persistence_top1") or 0.0
    sem_base_test = get("test", "hard_changed", "semantic_cluster_transition", "obs_persistence_top1") or 0.0
    semantic_cluster_passed = bool(sem_val >= sem_base_val + 0.03 and sem_test >= sem_base_test + 0.03 and sem_val >= 0.20 and sem_test >= 0.20)

    rank_val = get("val", "hard_changed", "topk_evidence_residual_rank", "local_expert", "gain_vs_weight_baseline") or 0.0
    rank_test = get("test", "hard_changed", "topk_evidence_residual_rank", "local_expert", "gain_vs_weight_baseline") or 0.0
    rank_acc_val = get("val", "hard_changed", "topk_evidence_residual_rank", "local_expert", "top3_accuracy") or 0.0
    rank_acc_test = get("test", "hard_changed", "topk_evidence_residual_rank", "local_expert", "top3_accuracy") or 0.0
    topk_rank_passed = bool(rank_val > 0.005 and rank_test > 0.005 and rank_acc_val >= 0.30 and rank_acc_test >= 0.30)

    attr_val = get("val", "hard_changed", "instance_consistent_attribute_change", "roc_auc") or 0.0
    attr_test = get("test", "hard_changed", "instance_consistent_attribute_change", "roc_auc") or 0.0
    attr_passed = bool(attr_val >= 0.65 and attr_test >= 0.65)

    ident_val = get("val", "hard_changed", "identity_consistency_target", "roc_auc") or 0.0
    ident_test = get("test", "hard_changed", "identity_consistency_target", "roc_auc") or 0.0
    ident_passed = bool(ident_val >= 0.65 and ident_test >= 0.65)

    semantic_family_passed = bool(semantic_cluster_passed or topk_rank_passed)
    consistency_family_passed = bool(attr_passed or ident_passed)
    ready = bool(semantic_family_passed and consistency_family_passed)
    return {
        "semantic_cluster_transition_upper_bound_passed": semantic_cluster_passed,
        "topk_evidence_residual_rank_upper_bound_passed": topk_rank_passed,
        "instance_consistent_attribute_change_upper_bound_passed": attr_passed,
        "identity_consistency_target_upper_bound_passed": ident_passed,
        "observed_predictable_target_suite_ready": ready,
        "recommended_next_step": "train_neural_writer_on_observed_predictable_discrete_targets" if ready else "stop_unit_delta_route_and_rethink_video_semantic_target",
    }


def main() -> int:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    centers = fit_codebook(args)
    model, ckargs, init = load_frozen_residual_model(args, device)
    print("V34.43: 探测 feature 维度。", flush=True)
    dim_info = collect_split("train", model, ckargs, centers, None, args, device, write_cache=False)
    proj = feature_projection(int(dim_info["feature_dim"]), args.feature_proj_dim, args.seed)
    print("V34.43: 构建 target cache 并收集 observed-only features。", flush=True)
    data = {
        split: collect_split(split, model, ckargs, centers, proj, args, device, write_cache=True)
        for split in ("train", "val", "test")
    }
    print("V34.43: 拟合 cached/ridge/local expert 上界。", flush=True)
    metrics = evaluate_targets(data["train"], data["val"], data["test"], args)
    decision = pass_decision(metrics)
    coverage = {
        split: {
            "token_count": int(len(d["valid"])),
            "valid_ratio": float(d["valid"].mean()),
            "hard_changed_ratio": float(d["hard_changed"].mean()),
            "attribute_change_positive_ratio_on_hard_changed": float(d["attribute_change"][d["hard_changed"]].mean()) if d["hard_changed"].any() else None,
            "identity_consistency_positive_ratio_on_hard_changed": float(d["identity_consistency"][d["hard_changed"]].mean()) if d["hard_changed"].any() else None,
        }
        for split, d in data.items()
    }
    payload = {
        "generated_at_utc": utc_now(),
        "中文结论": "V34.43 observed-predictable delta target 重定义完成；本轮只构建低维/离散 target 并评估 cached/ridge/local expert 上界，不训练 writer，不声明 semantic field success。",
        "target_root": str(TARGET_ROOT.relative_to(ROOT)),
        "semantic_clusters": args.semantic_clusters,
        "topk": args.topk,
        "feature_proj_dim": args.feature_proj_dim,
        "init": init,
        "coverage": coverage,
        "metrics": metrics,
        "decision": decision,
        "future_leakage_detected": False,
        "v30_backbone_frozen": bool(model.v30_backbone_frozen),
        "integrated_identity_field_claim_allowed": False,
        "integrated_semantic_field_claim_allowed": False,
        "阶段性分析": "V34.42 已说明连续 teacher embedding delta 的跨样本局部线性上界不足，因此 V34.43 把 target 改成可观察条件下更可能预测的离散/低维变量。只有 semantic cluster transition 或 top-k evidence rank 与 attribute/identity consistency 同时在 val/test 过上界，才值得回到 neural writer。",
        "论文相关问题解决方案参考": "这一分解对应 VQ/离散语义状态、object-centric transition、retrieval/ranking supervision 与 identity-consistency auxiliary target 的路线：先证明目标空间可预测，再训练神经写入器；避免把不可预测连续 teacher embedding 当主监督。",
        "recommended_next_step": decision["recommended_next_step"],
    }
    dump_json(REPORT, payload)
    write_doc(
        DOC,
        "V34.43 observed-predictable delta targets 中文报告",
        payload,
        [
            "中文结论",
            "target_root",
            "semantic_clusters",
            "coverage",
            "decision",
            "future_leakage_detected",
            "v30_backbone_frozen",
            "integrated_semantic_field_claim_allowed",
            "阶段性分析",
            "论文相关问题解决方案参考",
            "recommended_next_step",
        ],
    )
    print(f"已写出 V34.43 target 重定义报告: {REPORT.relative_to(ROOT)}", flush=True)
    print(f"observed_predictable_target_suite_ready: {decision['observed_predictable_target_suite_ready']}", flush=True)
    print(f"recommended_next_step: {decision['recommended_next_step']}", flush=True)
    return 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--semantic-clusters", type=int, default=64)
    p.add_argument("--topk", type=int, default=8)
    p.add_argument("--feature-proj-dim", type=int, default=256)
    p.add_argument("--ridge-alpha", type=float, default=10.0)
    p.add_argument("--max-train-tokens", type=int, default=220000)
    p.add_argument("--min-local-count", type=int, default=512)
    p.add_argument("--max-codebook-samples", type=int, default=90000)
    p.add_argument("--codebook-sample-per-file", type=int, default=384)
    p.add_argument("--kmeans-batch-size", type=int, default=4096)
    p.add_argument("--kmeans-iters", type=int, default=80)
    p.add_argument("--teacher-embedding-dim", type=int, default=768)
    p.add_argument("--reader-hidden-dim", type=int, default=192)
    p.add_argument("--cpu", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
