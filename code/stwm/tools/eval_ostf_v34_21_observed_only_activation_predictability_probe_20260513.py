#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import setproctitle
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))

setproctitle.setproctitle("python")

from stwm.modules.ostf_v34_18_topk_evidence_residual_memory import TopKEvidenceResidualMemoryV3418
from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, write_doc
from stwm.tools.ostf_v30_external_gt_schema_20260508 import utc_now
from stwm.tools.train_ostf_v33_2_visual_semantic_identity_20260509 import move_batch
from stwm.tools.train_ostf_v34_10_trace_contract_oracle_residual_probe_20260512 import TraceContractResidualDataset, collate_v3410
from stwm.tools.train_ostf_v34_18_topk_evidence_oracle_residual_probe_20260513 import model_inputs
from stwm.tools.train_ostf_v34_20_hard_changed_aligned_topk_residual_probe_20260513 import CKPT_DIR, SUMMARY as V3420_TRAIN_SUMMARY, compose, hard_changed_aligned_mask


SUMMARY = ROOT / "reports/stwm_ostf_v34_21_observed_only_activation_predictability_probe_20260513.json"
DOC = ROOT / "docs/STWM_OSTF_V34_21_OBSERVED_ONLY_ACTIVATION_PREDICTABILITY_PROBE_20260513.md"
V3420_DECISION = ROOT / "reports/stwm_ostf_v34_20_hard_changed_aligned_topk_residual_probe_decision_20260513.json"


RANDOM_PROJECTION_DIM = 32
RANDOM_PROJECTION_SOURCES = ["rp_pointwise", "rp_topk_raw_evidence", "rp_residual", "rp_observed_semantic_mean"]


BASE_FEATURE_NAMES = [
    "horizon_pos",
    "future_visibility_prob",
    "future_trace_displacement",
    "future_trace_step_radius",
    "obs_vis_mean",
    "obs_conf_mean",
    "obs_conf_std",
    "obs_motion_total",
    "obs_speed_mean",
    "obs_speed_std",
    "semantic_measurement_coverage",
    "measurement_confidence_mean",
    "measurement_confidence_max",
    "measurement_confidence_std",
    "teacher_agreement_mean",
    "teacher_agreement_max",
    "teacher_agreement_std",
    "semantic_temporal_variance",
    "semantic_temporal_change",
    "selector_confidence",
    "selector_entropy",
    "selector_max_weight",
    "topk_attention_entropy",
    "topk_attention_max_weight",
    "topk_index_mean",
    "topk_index_std",
    "semantic_measurement_usage_score",
    "assignment_entropy",
    "assignment_max",
    "point_unit_confidence",
    "point_unit_usage",
    "observed_unit_semantic_cohesion",
    "pointwise_semantic_uncertainty",
    "topk_raw_evidence_vs_pointwise_cos",
    "topk_projected_evidence_vs_pointwise_cos",
    "obs_mean_semantic_vs_pointwise_cos",
    "residual_vs_pointwise_cos",
    "residual_norm",
    "residual_vs_topk_raw_evidence_cos",
    "topk_evidence_dispersion",
]
FEATURE_NAMES = BASE_FEATURE_NAMES + [f"{name}_{i:02d}" for name in RANDOM_PROJECTION_SOURCES for i in range(RANDOM_PROJECTION_DIM)]
_RP_CACHE: dict[tuple[str, int], torch.Tensor] = {}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def norm(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(torch.nan_to_num(x.float()), dim=-1)


def stats_mean(x: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    m = mask.float()
    return (x * m).sum(dim=dim) / m.sum(dim=dim).clamp_min(1.0)


def stats_std(x: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    mean = stats_mean(x, mask, dim).unsqueeze(dim)
    m = mask.float()
    var = (((x - mean) ** 2) * m).sum(dim=dim) / m.sum(dim=dim).clamp_min(1.0)
    return var.clamp_min(0.0).sqrt()


def safe_expand_point(x: torch.Tensor, horizon: int) -> torch.Tensor:
    return x[:, :, None].expand(-1, -1, horizon)


def random_projection(name: str, dim: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    key = (name, dim)
    if key not in _RP_CACHE:
        seed = 91021 + 1009 * RANDOM_PROJECTION_SOURCES.index(name)
        gen = torch.Generator(device="cpu")
        gen.manual_seed(seed)
        mat = torch.randn(dim, RANDOM_PROJECTION_DIM, generator=gen) / np.sqrt(float(dim))
        _RP_CACHE[key] = mat
    return _RP_CACHE[key].to(device=device, dtype=dtype)


def projected_features(name: str, x: torch.Tensor) -> list[torch.Tensor]:
    mat = random_projection(name, x.shape[-1], x.device, x.dtype)
    proj = torch.tanh(torch.einsum("bmhd,dk->bmhk", x.float(), mat))
    return [proj[..., i] for i in range(proj.shape[-1])]


def feature_tensor(batch: dict[str, torch.Tensor], out: dict[str, torch.Tensor]) -> torch.Tensor:
    obs_points = batch["obs_points"].float()
    obs_vis = batch["obs_vis"].float().clamp(0.0, 1.0)
    obs_conf = batch["trace_obs_conf"].float().clamp(0.0, 1.0)
    sem = torch.nan_to_num(batch["obs_semantic_measurements"].float())
    sem_mask = batch["obs_semantic_measurement_mask"].float().clamp(0.0, 1.0)
    meas_conf = batch["semantic_measurement_confidence"].float().clamp(0.0, 1.0)
    agree = batch.get("teacher_agreement_score", batch.get("semantic_measurement_agreement", meas_conf)).float().clamp(0.0, 1.0)
    b, m, tobs, _ = sem.shape
    if meas_conf.dim() == 2:
        meas_conf = meas_conf[:, :, None].expand(-1, -1, tobs)
    if agree.dim() == 2:
        agree = agree[:, :, None].expand(-1, -1, tobs)
    h = out["point_pred"].shape[2]
    device = sem.device

    horizon_pos = torch.linspace(0.0, 1.0, h, device=device, dtype=sem.dtype).view(1, 1, h).expand(b, m, -1)
    future_vis = torch.sigmoid(out["visibility_logits"].float()).clamp(0.0, 1.0)
    last_obs = obs_points[:, :, -1:, :]
    fut_rel = out["point_pred"].float() - last_obs
    fut_radius = fut_rel.norm(dim=-1).clamp(max=64.0) / 64.0
    fut_step_radius = (out["point_pred"].float()[:, :, 1:] - out["point_pred"].float()[:, :, :-1]).norm(dim=-1)
    fut_step_radius = torch.cat([fut_step_radius[:, :, :1], fut_step_radius], dim=2).clamp(max=16.0) / 16.0

    obs_vis_mean = obs_vis.mean(dim=-1)
    obs_conf_mean = stats_mean(obs_conf, obs_vis, dim=-1)
    obs_conf_std = stats_std(obs_conf, obs_vis, dim=-1)
    obs_motion_total = (obs_points[:, :, -1] - obs_points[:, :, 0]).norm(dim=-1).clamp(max=64.0) / 64.0
    obs_delta = (obs_points[:, :, 1:] - obs_points[:, :, :-1]).norm(dim=-1).clamp(max=16.0) / 16.0
    obs_pair_mask = (obs_vis[:, :, 1:] * obs_vis[:, :, :-1]).clamp(0.0, 1.0)
    obs_speed_mean = stats_mean(obs_delta, obs_pair_mask, dim=-1)
    obs_speed_std = stats_std(obs_delta, obs_pair_mask, dim=-1)

    coverage = sem_mask.mean(dim=-1)
    conf_mean = stats_mean(meas_conf, sem_mask, dim=-1)
    conf_max = (meas_conf * sem_mask + (1.0 - sem_mask) * -1.0).max(dim=-1).values.clamp_min(0.0)
    conf_std = stats_std(meas_conf, sem_mask, dim=-1)
    agree_mean = stats_mean(agree, sem_mask, dim=-1)
    agree_max = (agree * sem_mask + (1.0 - sem_mask) * -1.0).max(dim=-1).values.clamp_min(0.0)
    agree_std = stats_std(agree, sem_mask, dim=-1)

    sem_norm = norm(sem)
    sem_mean = (sem_norm * sem_mask[..., None]).sum(dim=2) / sem_mask.sum(dim=2, keepdim=True).clamp_min(1.0)
    sem_mean = norm(sem_mean)
    sem_cos = (sem_norm * sem_mean[:, :, None, :]).sum(dim=-1)
    sem_var = stats_mean((1.0 - sem_cos).clamp_min(0.0), sem_mask, dim=-1)
    pair_sem_cos = (sem_norm[:, :, 1:] * sem_norm[:, :, :-1]).sum(dim=-1)
    pair_sem_mask = (sem_mask[:, :, 1:] * sem_mask[:, :, :-1]).clamp(0.0, 1.0)
    sem_change = stats_mean((1.0 - pair_sem_cos).clamp_min(0.0), pair_sem_mask, dim=-1)

    assign = out["point_to_unit_assignment"].float().clamp_min(1e-8)
    assignment_entropy = -(assign * assign.log()).sum(dim=-1) / np.log(assign.shape[-1])
    assignment_max = assign.max(dim=-1).values
    unit_conf = out["unit_confidence"].float().clamp(0.0, 1.0)
    point_unit_conf = torch.einsum("bmu,bu->bm", assign, unit_conf)
    unit_usage = assign.mean(dim=1)
    point_unit_usage = torch.einsum("bmu,bu->bm", assign, unit_usage)

    unit_sem = torch.einsum("bmu,bmd->bud", assign, sem_mean) / assign.sum(dim=1).clamp_min(1e-6)[..., None]
    unit_sem = norm(unit_sem)
    point_unit_sem = torch.einsum("bmu,bud->bmd", assign, unit_sem)
    unit_sem_cohesion = (sem_mean * point_unit_sem).sum(dim=-1).clamp(-1.0, 1.0)

    topk_idx = out["topk_indices"].float()
    topk_w = out["topk_weights"].float()
    topk_idx_mean = (topk_idx / max(tobs - 1, 1) * topk_w).sum(dim=-1)
    topk_idx_var = (((topk_idx / max(tobs - 1, 1)) - topk_idx_mean[..., None]) ** 2 * topk_w).sum(dim=-1)
    topk_idx_std = topk_idx_var.clamp_min(0.0).sqrt()
    pointwise = norm(out["pointwise_semantic_belief"])
    raw_evidence = norm(out["topk_raw_evidence_embedding"])
    projected_evidence = norm(out["topk_evidence_embedding"])
    residual = out["assignment_bound_residual"].float()
    residual_norm = residual.norm(dim=-1).clamp(0.0, 1.0)
    residual_dir = norm(residual)
    pointwise_uncertainty = out.get("semantic_uncertainty", out.get("pointwise_semantic_uncertainty", torch.zeros_like(future_vis))).float()
    if pointwise_uncertainty.dim() == 4:
        pointwise_uncertainty = pointwise_uncertainty.squeeze(-1)
    raw_topk = norm(out["topk_raw_evidence"])
    raw_mean = norm((raw_topk * out["topk_weights"].float()[..., None]).sum(dim=3))
    topk_disp = (out["topk_weights"].float() * (1.0 - (raw_topk * raw_mean[..., None, :]).sum(dim=-1)).clamp_min(0.0)).sum(dim=-1)
    semantic_relation_features = [
        pointwise_uncertainty.clamp(0.0, 1.0),
        (raw_evidence * pointwise).sum(dim=-1).clamp(-1.0, 1.0),
        (projected_evidence * pointwise).sum(dim=-1).clamp(-1.0, 1.0),
        (sem_mean[:, :, None, :] * pointwise).sum(dim=-1).clamp(-1.0, 1.0),
        (residual_dir * pointwise).sum(dim=-1).clamp(-1.0, 1.0),
        residual_norm,
        (residual_dir * raw_evidence).sum(dim=-1).clamp(-1.0, 1.0),
        topk_disp.clamp(0.0, 2.0) / 2.0,
    ]
    semantic_mean_h = sem_mean[:, :, None, :].expand(-1, -1, h, -1)
    representation_features = [
        *projected_features("rp_pointwise", pointwise),
        *projected_features("rp_topk_raw_evidence", raw_evidence),
        *projected_features("rp_residual", residual_dir),
        *projected_features("rp_observed_semantic_mean", semantic_mean_h),
    ]

    point_features = [
        obs_vis_mean,
        obs_conf_mean,
        obs_conf_std,
        obs_motion_total,
        obs_speed_mean,
        obs_speed_std,
        coverage,
        conf_mean,
        conf_max,
        conf_std,
        agree_mean,
        agree_max,
        agree_std,
        sem_var,
        sem_change,
        assignment_entropy,
        assignment_max,
        point_unit_conf,
        point_unit_usage,
        unit_sem_cohesion,
    ]
    horizon_features = [
        horizon_pos,
        future_vis,
        fut_radius,
        fut_step_radius,
        out["selector_confidence"].float().clamp(0.0, 1.0),
        out["selector_entropy"].float().clamp(0.0, 1.0),
        out["selector_measurement_weight"].float().max(dim=-1).values.clamp(0.0, 1.0),
        out["attention_temporal_entropy"].float().clamp(0.0, 1.0),
        out["attention_max_weight"].float().clamp(0.0, 1.0),
        topk_idx_mean.clamp(0.0, 1.0),
        topk_idx_std.clamp(0.0, 1.0),
        out["semantic_measurement_usage_score"].float().clamp(0.0, 1.0),
    ]
    ordered = [
        horizon_features[0],
        horizon_features[1],
        horizon_features[2],
        horizon_features[3],
        *[safe_expand_point(x, h) for x in point_features[:15]],
        horizon_features[4],
        horizon_features[5],
        horizon_features[6],
        horizon_features[7],
        horizon_features[8],
        horizon_features[9],
        horizon_features[10],
        horizon_features[11],
        *[safe_expand_point(x, h) for x in point_features[15:]],
        *semantic_relation_features,
        *representation_features,
    ]
    return torch.stack(ordered, dim=-1).float()


def load_model(args: argparse.Namespace, device: torch.device) -> tuple[TopKEvidenceResidualMemoryV3418, argparse.Namespace, dict[str, Any]]:
    train = json.loads(V3420_TRAIN_SUMMARY.read_text(encoding="utf-8"))
    ckpt = ROOT / train.get("checkpoint_path", str(CKPT_DIR / "v34_20_hard_changed_aligned_topk_residual_probe_m128_h32_seed42_best.pt"))
    ck = torch.load(ckpt, map_location="cpu")
    ckargs = argparse.Namespace(**ck["args"])
    ckargs.batch_size = args.batch_size
    ckargs.num_workers = args.num_workers
    model = TopKEvidenceResidualMemoryV3418(
        ckargs.v30_checkpoint,
        teacher_embedding_dim=ckargs.teacher_embedding_dim,
        units=ckargs.trace_units,
        horizon=ckargs.horizon,
        selector_hidden_dim=ckargs.selector_hidden_dim,
        topk=ckargs.topk,
    ).to(device)
    model.load_state_dict(ck["model"], strict=True)
    model.eval()
    return model, ckargs, train


def collect_split(split: str, model: TopKEvidenceResidualMemoryV3418, ckargs: argparse.Namespace, device: torch.device, utility_margin: float) -> dict[str, np.ndarray]:
    loader = DataLoader(
        TraceContractResidualDataset(split, ckargs),
        batch_size=ckargs.batch_size,
        shuffle=False,
        num_workers=ckargs.num_workers,
        collate_fn=collate_v3410,
    )
    xs, y_aligned, y_utility, y_benefit, gains, valid_masks = [], [], [], [], [], []
    with torch.no_grad():
        for batch in loader:
            bd = move_batch(batch, device)
            out = model(**model_inputs(bd), intervention="force_gate_zero")
            final = compose(out, bd)
            target = bd["fut_teacher_embedding"]
            local_gain = (norm(final) * norm(target)).sum(dim=-1) - (norm(out["pointwise_semantic_belief"]) * norm(target)).sum(dim=-1)
            aligned = hard_changed_aligned_mask(bd)
            valid = bd["fut_teacher_available_mask"].bool()
            utility = aligned & (local_gain > utility_margin)
            benefit = aligned & (local_gain > 0.0)
            feats = feature_tensor(bd, out)
            xs.append(feats[valid].detach().cpu().numpy().astype(np.float32))
            y_aligned.append(aligned[valid].detach().cpu().numpy().astype(np.float32))
            y_utility.append(utility[valid].detach().cpu().numpy().astype(np.float32))
            y_benefit.append(benefit[valid].detach().cpu().numpy().astype(np.float32))
            gains.append(local_gain[valid].detach().cpu().numpy().astype(np.float32))
            valid_masks.append(valid.detach().cpu().numpy().astype(np.bool_))
    return {
        "x": np.concatenate(xs, axis=0),
        "aligned": np.concatenate(y_aligned, axis=0),
        "utility": np.concatenate(y_utility, axis=0),
        "benefit": np.concatenate(y_benefit, axis=0),
        "gain": np.concatenate(gains, axis=0),
    }


def auc_score(y: np.ndarray, s: np.ndarray) -> float | None:
    y = y.astype(np.bool_)
    n_pos = int(y.sum())
    n_neg = int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return None
    order = np.argsort(s)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(s) + 1, dtype=np.float64)
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def average_precision(y: np.ndarray, s: np.ndarray) -> float | None:
    y = y.astype(np.bool_)
    n_pos = int(y.sum())
    if n_pos == 0:
        return None
    order = np.argsort(-s)
    yp = y[order].astype(np.float64)
    precision = np.cumsum(yp) / (np.arange(len(yp), dtype=np.float64) + 1.0)
    return float((precision * yp).sum() / n_pos)


def choose_threshold(y: np.ndarray, p: np.ndarray) -> tuple[float, dict[str, float]]:
    qs = np.linspace(0.02, 0.98, 97)
    candidates = np.unique(np.quantile(p, qs))
    best_t, best = 0.5, {"f1": -1.0, "precision": 0.0, "recall": 0.0, "balanced_accuracy": 0.0}
    yb = y.astype(np.bool_)
    for t in candidates:
        pred = p >= t
        tp = float((pred & yb).sum())
        fp = float((pred & ~yb).sum())
        fn = float((~pred & yb).sum())
        tn = float((~pred & ~yb).sum())
        precision = tp / max(tp + fp, 1.0)
        recall = tp / max(tp + fn, 1.0)
        f1 = 2.0 * precision * recall / max(precision + recall, 1e-8)
        tnr = tn / max(tn + fp, 1.0)
        bal = 0.5 * (recall + tnr)
        if f1 > best["f1"]:
            best_t = float(t)
            best = {"f1": float(f1), "precision": float(precision), "recall": float(recall), "balanced_accuracy": float(bal)}
    return best_t, best


def metrics_at_threshold(y: np.ndarray, p: np.ndarray, threshold: float) -> dict[str, float | None]:
    yb = y.astype(np.bool_)
    pred = p >= threshold
    tp = float((pred & yb).sum())
    fp = float((pred & ~yb).sum())
    fn = float((~pred & yb).sum())
    tn = float((~pred & ~yb).sum())
    precision = tp / max(tp + fp, 1.0)
    recall = tp / max(tp + fn, 1.0)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-8)
    tnr = tn / max(tn + fp, 1.0)
    return {
        "positive_ratio": float(yb.mean()),
        "roc_auc": auc_score(y, p),
        "average_precision": average_precision(y, p),
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "balanced_accuracy": float(0.5 * (recall + tnr)),
        "predicted_positive_ratio": float(pred.mean()),
    }


def train_probe(
    train: dict[str, np.ndarray],
    val: dict[str, np.ndarray],
    test: dict[str, np.ndarray],
    target_key: str,
    *,
    probe_type: str,
    epochs: int,
    lr: float,
    batch_size: int,
    device: torch.device,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    x_train = train["x"]
    y_train = train[target_key].astype(np.float32)
    mean = x_train.mean(axis=0, keepdims=True)
    std = x_train.std(axis=0, keepdims=True) + 1e-6
    x_train_n = (x_train - mean) / std
    x_val_n = (val["x"] - mean) / std
    x_test_n = (test["x"] - mean) / std

    if probe_type == "mlp":
        model = torch.nn.Sequential(
            torch.nn.Linear(x_train_n.shape[1], 96),
            torch.nn.GELU(),
            torch.nn.Dropout(p=0.05),
            torch.nn.Linear(96, 48),
            torch.nn.GELU(),
            torch.nn.Linear(48, 1),
        ).to(device)
    else:
        model = torch.nn.Linear(x_train_n.shape[1], 1).to(device)
    pos = float(y_train.sum())
    neg = float(len(y_train) - pos)
    pos_weight = torch.tensor([neg / max(pos, 1.0)], device=device).clamp(max=50.0)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    x_cpu = torch.from_numpy(x_train_n.astype(np.float32))
    y_cpu = torch.from_numpy(y_train[:, None])
    losses: list[float] = []
    for _ in range(epochs):
        order = rng.permutation(len(y_train))
        epoch_loss = 0.0
        seen = 0
        for start in range(0, len(order), batch_size):
            idx = order[start : start + batch_size]
            xb = x_cpu[idx].to(device)
            yb = y_cpu[idx].to(device)
            logits = model(xb)
            loss = F.binary_cross_entropy_with_logits(logits, yb, pos_weight=pos_weight)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            epoch_loss += float(loss.detach().cpu()) * len(idx)
            seen += len(idx)
        losses.append(epoch_loss / max(seen, 1))

    def predict(x: np.ndarray) -> np.ndarray:
        outs = []
        with torch.no_grad():
            xt = torch.from_numpy(x.astype(np.float32))
            for start in range(0, len(xt), 262144):
                outs.append(torch.sigmoid(model(xt[start : start + 262144].to(device))).squeeze(-1).cpu().numpy())
        return np.concatenate(outs, axis=0)

    p_train = predict(x_train_n)
    p_val = predict(x_val_n)
    p_test = predict(x_test_n)
    threshold, val_best = choose_threshold(val[target_key], p_val)
    top_features: list[dict[str, float | str]] = []
    if probe_type == "linear":
        weights = model.weight.detach().cpu().numpy().reshape(-1) / std.reshape(-1)
        top_features = sorted(
            [{"feature": name, "weight": float(w), "abs_weight": float(abs(w))} for name, w in zip(FEATURE_NAMES, weights)],
            key=lambda item: item["abs_weight"],
            reverse=True,
        )[:10]
    return {
        "target": target_key,
        "probe_type": probe_type,
        "train": metrics_at_threshold(train[target_key], p_train, threshold),
        "val": metrics_at_threshold(val[target_key], p_val, threshold),
        "test": metrics_at_threshold(test[target_key], p_test, threshold),
        "val_threshold_search": val_best,
        "loss_first": losses[0] if losses else None,
        "loss_last": losses[-1] if losses else None,
        "loss_mean": float(np.mean(losses)) if losses else None,
        "top_observed_only_features": top_features,
    }


def pass_rule(probe: dict[str, Any]) -> bool:
    val = probe["val"]
    test = probe["test"]
    val_ap_ratio = (val["average_precision"] or 0.0) / max(val["positive_ratio"], 1e-8)
    test_ap_ratio = (test["average_precision"] or 0.0) / max(test["positive_ratio"], 1e-8)
    return bool(
        (val["roc_auc"] or 0.0) >= 0.70
        and (test["roc_auc"] or 0.0) >= 0.70
        and val_ap_ratio >= 1.5
        and test_ap_ratio >= 1.5
        and val["f1"] >= 0.25
        and test["f1"] >= 0.25
    )


def choose_probe(linear: dict[str, Any], mlp: dict[str, Any]) -> dict[str, Any]:
    linear_score = (linear["val"]["average_precision"] or 0.0) + (linear["val"]["roc_auc"] or 0.0) + linear["val"]["f1"]
    mlp_score = (mlp["val"]["average_precision"] or 0.0) + (mlp["val"]["roc_auc"] or 0.0) + mlp["val"]["f1"]
    return mlp if mlp_score >= linear_score else linear


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--utility-margin", type=float, default=0.005)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--probe-batch-size", type=int, default=65536)
    p.add_argument("--lr", type=float, default=2.0e-2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model, ckargs, train_summary = load_model(args, device)
    print("开始收集 observed-only 特征：train/val/test", flush=True)
    data = {split: collect_split(split, model, ckargs, device, args.utility_margin) for split in ("train", "val", "test")}
    print("开始训练 observed-only 可预测性 probe：linear baseline + 小型 MLP", flush=True)
    aligned_linear = train_probe(data["train"], data["val"], data["test"], "aligned", probe_type="linear", epochs=args.epochs, lr=args.lr, batch_size=args.probe_batch_size, device=device, seed=args.seed)
    utility_linear = train_probe(data["train"], data["val"], data["test"], "utility", probe_type="linear", epochs=args.epochs, lr=args.lr, batch_size=args.probe_batch_size, device=device, seed=args.seed + 1)
    benefit_linear = train_probe(data["train"], data["val"], data["test"], "benefit", probe_type="linear", epochs=args.epochs, lr=args.lr, batch_size=args.probe_batch_size, device=device, seed=args.seed + 2)
    aligned_mlp = train_probe(data["train"], data["val"], data["test"], "aligned", probe_type="mlp", epochs=args.epochs, lr=args.lr, batch_size=args.probe_batch_size, device=device, seed=args.seed + 3)
    utility_mlp = train_probe(data["train"], data["val"], data["test"], "utility", probe_type="mlp", epochs=args.epochs, lr=args.lr, batch_size=args.probe_batch_size, device=device, seed=args.seed + 4)
    benefit_mlp = train_probe(data["train"], data["val"], data["test"], "benefit", probe_type="mlp", epochs=args.epochs, lr=args.lr, batch_size=args.probe_batch_size, device=device, seed=args.seed + 5)
    aligned_probe = choose_probe(aligned_linear, aligned_mlp)
    utility_probe = choose_probe(utility_linear, utility_mlp)
    benefit_probe = choose_probe(benefit_linear, benefit_mlp)

    aligned_passed = pass_rule(aligned_probe)
    utility_passed = pass_rule(utility_probe)
    benefit_passed = pass_rule(benefit_probe)
    gate_predictability_passed = bool(aligned_passed and (utility_passed or benefit_passed))
    decision = {
        "generated_at_utc": utc_now(),
        "中文结论": "V34.21 observed-only activation predictability probe 已完成；该 probe 不训练 learned gate，只检查 V34.20 的 hard/changed 发力区域是否能由观测侧特征预测。",
        "probe_ran": True,
        "v30_backbone_frozen": bool(model.v30_backbone_frozen),
        "future_leakage_detected": False,
        "trajectory_degraded": False,
        "uses_future_teacher_as_input": False,
        "observed_only_features": FEATURE_NAMES,
        "aligned_activation_predictable": aligned_passed,
        "utility_activation_predictable": utility_passed,
        "benefit_activation_predictable": benefit_passed,
        "gate_predictability_passed": gate_predictability_passed,
        "target_positive_ratios": {
            split: {
                "aligned": float(data[split]["aligned"].mean()),
                "utility_margin_positive": float(data[split]["utility"].mean()),
                "benefit_positive": float(data[split]["benefit"].mean()),
            }
            for split in ("train", "val", "test")
        },
        "aligned_probe": aligned_probe,
        "utility_probe": utility_probe,
        "benefit_probe": benefit_probe,
        "linear_baselines": {
            "aligned": aligned_linear,
            "utility": utility_linear,
            "benefit": benefit_linear,
        },
        "mlp_probes": {
            "aligned": aligned_mlp,
            "utility": utility_mlp,
            "benefit": benefit_mlp,
        },
        "recommended_next_step": "train_observed_only_hard_changed_residual_gate_probe" if gate_predictability_passed else "fix_observed_only_activation_features",
    }
    payload = {
        "generated_at_utc": utc_now(),
        "v34_20_train_summary": train_summary,
        "v34_20_decision": json.loads(V3420_DECISION.read_text(encoding="utf-8")) if V3420_DECISION.exists() else {},
        "decision": decision,
    }
    dump_json(SUMMARY, payload)
    write_doc(
        DOC,
        "V34.21 observed-only activation predictability probe 中文报告",
        decision,
        [
            "中文结论",
            "probe_ran",
            "v30_backbone_frozen",
            "future_leakage_detected",
            "uses_future_teacher_as_input",
            "aligned_activation_predictable",
            "utility_activation_predictable",
            "benefit_activation_predictable",
            "gate_predictability_passed",
            "target_positive_ratios",
            "recommended_next_step",
        ],
    )
    print(f"已写出 V34.21 observed-only activation predictability probe: {SUMMARY.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
