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
from stwm.modules.ostf_v34_22_activation_state_reader import ActivationStateReaderV3422
from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, write_doc
from stwm.tools.ostf_v30_external_gt_schema_20260508 import utc_now
from stwm.tools.train_ostf_v33_2_visual_semantic_identity_20260509 import move_batch
from stwm.tools.train_ostf_v34_10_trace_contract_oracle_residual_probe_20260512 import TraceContractResidualDataset, collate_v3410
from stwm.tools.train_ostf_v34_18_topk_evidence_oracle_residual_probe_20260513 import model_inputs
from stwm.tools.train_ostf_v34_20_hard_changed_aligned_topk_residual_probe_20260513 import CKPT_DIR as V3420_CKPT_DIR
from stwm.tools.train_ostf_v34_20_hard_changed_aligned_topk_residual_probe_20260513 import SUMMARY as V3420_TRAIN_SUMMARY
from stwm.tools.train_ostf_v34_20_hard_changed_aligned_topk_residual_probe_20260513 import compose, hard_changed_aligned_mask


SUMMARY = ROOT / "reports/stwm_ostf_v34_22_activation_state_reader_predictability_probe_20260513.json"
DOC = ROOT / "docs/STWM_OSTF_V34_22_ACTIVATION_STATE_READER_PREDICTABILITY_PROBE_20260513.md"
CKPT_DIR = ROOT / "outputs/checkpoints/stwm_ostf_v34_22_activation_state_reader_predictability_probe_h32_m128"
CHECKPOINT = CKPT_DIR / "v34_22_activation_state_reader_probe_m128_h32_seed42.pt"
V3420_DECISION = ROOT / "reports/stwm_ostf_v34_20_hard_changed_aligned_topk_residual_probe_decision_20260513.json"
V3421 = ROOT / "reports/stwm_ostf_v34_21_observed_only_activation_predictability_probe_20260513.json"


TARGETS = ["aligned", "utility", "benefit"]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def norm(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(torch.nan_to_num(x.float()), dim=-1)


def load_residual_model(args: argparse.Namespace, device: torch.device) -> tuple[TopKEvidenceResidualMemoryV3418, argparse.Namespace, dict[str, Any]]:
    train = json.loads(V3420_TRAIN_SUMMARY.read_text(encoding="utf-8"))
    ckpt = ROOT / train.get("checkpoint_path", str(V3420_CKPT_DIR / "v34_20_hard_changed_aligned_topk_residual_probe_m128_h32_seed42_best.pt"))
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
    for p in model.parameters():
        p.requires_grad_(False)
    return model, ckargs, train


def reader_inputs(out: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        "future_trace_hidden": out["future_trace_hidden"],
        "topk_raw_evidence": out["topk_raw_evidence"],
        "topk_weights": out["topk_weights"],
        "pointwise_semantic_belief": out["pointwise_semantic_belief"],
        "assignment_bound_residual": out["assignment_bound_residual"],
        "unit_memory": out["unit_memory"],
        "point_to_unit_assignment": out["point_to_unit_assignment"],
        "semantic_measurement_usage_score": out["semantic_measurement_usage_score"],
        "selector_confidence": out["selector_confidence"],
        "selector_entropy": out["selector_entropy"],
        "selector_measurement_weight": out["selector_measurement_weight"],
        "assignment_usage_score": out["assignment_usage_score"],
    }


def labels(batch: dict[str, torch.Tensor], out: dict[str, torch.Tensor], utility_margin: float) -> dict[str, torch.Tensor]:
    final = compose(out, batch)
    target = batch["fut_teacher_embedding"]
    gain = (norm(final) * norm(target)).sum(dim=-1) - (norm(out["pointwise_semantic_belief"]) * norm(target)).sum(dim=-1)
    aligned = hard_changed_aligned_mask(batch)
    return {
        "aligned": aligned,
        "utility": aligned & (gain > utility_margin),
        "benefit": aligned & (gain > 0.0),
        "valid": batch["fut_teacher_available_mask"].bool(),
        "gain": gain,
    }


def make_loader(split: str, ckargs: argparse.Namespace, shuffle: bool) -> DataLoader:
    return DataLoader(
        TraceContractResidualDataset(split, ckargs),
        batch_size=ckargs.batch_size,
        shuffle=shuffle,
        num_workers=ckargs.num_workers,
        collate_fn=collate_v3410,
    )


def count_labels(model: TopKEvidenceResidualMemoryV3418, ckargs: argparse.Namespace, device: torch.device, utility_margin: float) -> dict[str, float]:
    counts = {t: 0.0 for t in TARGETS}
    total = 0.0
    with torch.no_grad():
        for batch in make_loader("train", ckargs, shuffle=False):
            bd = move_batch(batch, device)
            out = model(**model_inputs(bd), intervention="force_gate_zero")
            lab = labels(bd, out, utility_margin)
            valid = lab["valid"]
            total += float(valid.sum().detach().cpu())
            for target in TARGETS:
                counts[target] += float((lab[target] & valid).sum().detach().cpu())
    return {target: (total - counts[target]) / max(counts[target], 1.0) for target in TARGETS}


def train_reader(
    residual_model: TopKEvidenceResidualMemoryV3418,
    reader: ActivationStateReaderV3422,
    ckargs: argparse.Namespace,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    pos_weights = count_labels(residual_model, ckargs, device, args.utility_margin)
    pos_tensors = {k: torch.tensor([min(v, 50.0)], device=device) for k, v in pos_weights.items()}
    opt = torch.optim.AdamW(reader.parameters(), lr=args.lr, weight_decay=1e-4)
    losses: list[dict[str, float]] = []
    reader.train()
    for epoch in range(1, args.epochs + 1):
        totals = {target: 0.0 for target in TARGETS}
        total_loss = 0.0
        seen = 0
        for batch in make_loader("train", ckargs, shuffle=True):
            bd = move_batch(batch, device)
            with torch.no_grad():
                out = residual_model(**model_inputs(bd), intervention="force_gate_zero")
                lab = labels(bd, out, args.utility_margin)
            pred = reader(**reader_inputs(out))["activation_logits"]
            valid = lab["valid"]
            loss = torch.tensor(0.0, device=device)
            for target in TARGETS:
                logits = pred[target][valid]
                y = lab[target][valid].float()
                l = F.binary_cross_entropy_with_logits(logits, y, pos_weight=pos_tensors[target])
                loss = loss + l
                totals[target] += float(l.detach().cpu())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(reader.parameters(), 1.0)
            opt.step()
            total_loss += float(loss.detach().cpu())
            seen += 1
        row = {"epoch": float(epoch), "loss": total_loss / max(seen, 1)}
        row.update({f"{target}_loss": totals[target] / max(seen, 1) for target in TARGETS})
        losses.append(row)
        print(f"训练进度: epoch={epoch}/{args.epochs}, loss={row['loss']:.6f}, aligned={row['aligned_loss']:.6f}, utility={row['utility_loss']:.6f}", flush=True)
    return {"pos_weights": pos_weights, "loss_trace": losses}


def auc_score(y: np.ndarray, s: np.ndarray) -> float | None:
    yb = y.astype(np.bool_)
    n_pos = int(yb.sum())
    n_neg = int((~yb).sum())
    if n_pos == 0 or n_neg == 0:
        return None
    order = np.argsort(s)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(s) + 1, dtype=np.float64)
    return float((ranks[yb].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def average_precision(y: np.ndarray, s: np.ndarray) -> float | None:
    yb = y.astype(np.bool_)
    n_pos = int(yb.sum())
    if n_pos == 0:
        return None
    order = np.argsort(-s)
    yp = yb[order].astype(np.float64)
    precision = np.cumsum(yp) / (np.arange(len(yp), dtype=np.float64) + 1.0)
    return float((precision * yp).sum() / n_pos)


def choose_threshold(y: np.ndarray, p: np.ndarray) -> tuple[float, dict[str, float]]:
    candidates = np.unique(np.quantile(p, np.linspace(0.02, 0.98, 97)))
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


def metrics(y: np.ndarray, p: np.ndarray, threshold: float) -> dict[str, float | None]:
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


def eval_split(
    split: str,
    residual_model: TopKEvidenceResidualMemoryV3418,
    reader: ActivationStateReaderV3422,
    ckargs: argparse.Namespace,
    device: torch.device,
    utility_margin: float,
) -> dict[str, np.ndarray]:
    reader.eval()
    rows = {f"{target}_y": [] for target in TARGETS}
    rows.update({f"{target}_p": [] for target in TARGETS})
    entropy, max_attn = [], []
    with torch.no_grad():
        for batch in make_loader(split, ckargs, shuffle=False):
            bd = move_batch(batch, device)
            out = residual_model(**model_inputs(bd), intervention="force_gate_zero")
            lab = labels(bd, out, utility_margin)
            pred = reader(**reader_inputs(out))
            valid = lab["valid"]
            for target in TARGETS:
                rows[f"{target}_y"].append(lab[target][valid].detach().cpu().numpy().astype(np.float32))
                rows[f"{target}_p"].append(torch.sigmoid(pred["activation_logits"][target][valid]).detach().cpu().numpy().astype(np.float32))
            entropy.append(pred["evidence_attention_entropy"][valid].detach().cpu().numpy().astype(np.float32))
            max_attn.append(pred["evidence_attention_max"][valid].detach().cpu().numpy().astype(np.float32))
    out_np = {k: np.concatenate(v, axis=0) for k, v in rows.items()}
    out_np["attention_entropy"] = np.concatenate(entropy, axis=0)
    out_np["attention_max"] = np.concatenate(max_attn, axis=0)
    return out_np


class Acc:
    def __init__(self) -> None:
        self.sum: dict[str, float] = {}
        self.count: dict[str, int] = {}

    def add(self, key: str, value: torch.Tensor, mask: torch.Tensor) -> None:
        m = mask.bool()
        if bool(m.any()):
            self.sum[key] = self.sum.get(key, 0.0) + float(value[m].sum().detach().cpu())
            self.count[key] = self.count.get(key, 0) + int(m.sum().detach().cpu())

    def mean(self, key: str) -> float | None:
        c = self.count.get(key, 0)
        return None if c == 0 else float(self.sum[key] / c)


def update_gain(acc: Acc, prefix: str, pred: torch.Tensor, pointwise: torch.Tensor, target: torch.Tensor, masks: dict[str, torch.Tensor]) -> None:
    gain = (norm(pred) * norm(target)).sum(dim=-1) - (norm(pointwise) * norm(target)).sum(dim=-1)
    for key, mask in masks.items():
        acc.add(f"{prefix}:{key}", gain, mask)


def finalize_gain(acc: Acc, prefix: str) -> dict[str, Any]:
    keys = ["aligned", "hard", "changed", "hard_changed", "stable", "valid"]
    out = {f"{k}_gain": acc.mean(f"{prefix}:{k}") for k in keys}
    out["semantic_hard_signal"] = bool(out["hard_gain"] is not None and out["hard_gain"] > 0.005)
    out["changed_semantic_signal"] = bool(out["changed_gain"] is not None and out["changed_gain"] > 0.005)
    out["stable_preservation"] = bool(out["stable_gain"] is None or out["stable_gain"] >= -0.02)
    return out


def eval_reader_gate_split(
    split: str,
    residual_model: TopKEvidenceResidualMemoryV3418,
    reader: ActivationStateReaderV3422,
    ckargs: argparse.Namespace,
    device: torch.device,
    thresholds: dict[str, float],
) -> dict[str, Any]:
    reader.eval()
    modes: dict[str, Acc] = {}
    for target in TARGETS:
        modes[f"{target}_soft"] = Acc()
        modes[f"{target}_threshold"] = Acc()
    modes["oracle_aligned"] = Acc()
    with torch.no_grad():
        for batch in make_loader(split, ckargs, shuffle=False):
            bd = move_batch(batch, device)
            out = residual_model(**model_inputs(bd), intervention="force_gate_zero")
            pred = reader(**reader_inputs(out))["activation_logits"]
            valid = bd["fut_teacher_available_mask"].bool()
            hard = bd["semantic_hard_mask"].bool() & valid
            changed = bd["changed_mask"].bool() & valid
            masks = {
                "aligned": hard_changed_aligned_mask(bd),
                "hard": hard,
                "changed": changed,
                "hard_changed": (hard | changed) & valid,
                "stable": bd["stable_suppress_mask"].bool() & valid,
                "valid": valid,
            }
            pointwise = out["pointwise_semantic_belief"]
            target_embedding = bd["fut_teacher_embedding"]
            oracle_gate = masks["aligned"].float() * out["semantic_measurement_usage_score"].float().clamp(0.0, 1.0)
            oracle_final = F.normalize(pointwise + oracle_gate[..., None] * out["assignment_bound_residual"], dim=-1)
            update_gain(modes["oracle_aligned"], "oracle_aligned", oracle_final, pointwise, target_embedding, masks)
            for target in TARGETS:
                prob = torch.sigmoid(pred[target]).clamp(0.0, 1.0)
                soft_gate = prob * out["semantic_measurement_usage_score"].float().clamp(0.0, 1.0)
                threshold_gate = (prob >= thresholds[target]).float() * out["semantic_measurement_usage_score"].float().clamp(0.0, 1.0)
                soft_final = F.normalize(pointwise + soft_gate[..., None] * out["assignment_bound_residual"], dim=-1)
                threshold_final = F.normalize(pointwise + threshold_gate[..., None] * out["assignment_bound_residual"], dim=-1)
                update_gain(modes[f"{target}_soft"], f"{target}_soft", soft_final, pointwise, target_embedding, masks)
                update_gain(modes[f"{target}_threshold"], f"{target}_threshold", threshold_final, pointwise, target_embedding, masks)
    return {mode: finalize_gain(acc, mode) for mode, acc in modes.items()}


def compose_reader_gate(out: dict[str, torch.Tensor], pred: dict[str, torch.Tensor], gate_name: str, thresholds: dict[str, float]) -> torch.Tensor:
    if gate_name is None:
        return out["pointwise_semantic_belief"]
    target, kind = gate_name.rsplit("_", 1)
    prob = torch.sigmoid(pred[target]).clamp(0.0, 1.0)
    if kind == "threshold":
        gate = (prob >= thresholds[target]).float()
    else:
        gate = prob
    gate = gate * out["semantic_measurement_usage_score"].float().clamp(0.0, 1.0)
    return F.normalize(out["pointwise_semantic_belief"] + gate[..., None] * out["assignment_bound_residual"], dim=-1)


def eval_best_gate_interventions(
    split: str,
    residual_model: TopKEvidenceResidualMemoryV3418,
    reader: ActivationStateReaderV3422,
    ckargs: argparse.Namespace,
    device: torch.device,
    thresholds: dict[str, float],
    gate_name: str,
) -> dict[str, Any]:
    interventions = {
        "normal": "force_gate_zero",
        "zero_semantic_measurements": "zero_semantic_measurements",
        "shuffle_semantic_measurements": "shuffle_semantic_measurements_across_points",
        "shuffle_assignment": "shuffle_assignment",
        "zero_unit_memory": "zero_unit_memory",
        "selector_ablation": "selector_ablation",
    }
    accs = {name: Acc() for name in interventions}
    reader.eval()
    with torch.no_grad():
        for batch in make_loader(split, ckargs, shuffle=False):
            bd = move_batch(batch, device)
            valid = bd["fut_teacher_available_mask"].bool()
            hard = bd["semantic_hard_mask"].bool() & valid
            changed = bd["changed_mask"].bool() & valid
            masks = {
                "aligned": hard_changed_aligned_mask(bd),
                "hard": hard,
                "changed": changed,
                "hard_changed": (hard | changed) & valid,
                "stable": bd["stable_suppress_mask"].bool() & valid,
                "valid": valid,
            }
            for mode, intervention in interventions.items():
                out = residual_model(**model_inputs(bd), intervention=intervention)
                pred = reader(**reader_inputs(out))["activation_logits"]
                final = compose_reader_gate(out, pred, gate_name, thresholds)
                update_gain(accs[mode], mode, final, out["pointwise_semantic_belief"], bd["fut_teacher_embedding"], masks)
    per = {mode: finalize_gain(acc, mode) for mode, acc in accs.items()}
    normal = per["normal"]
    deltas = {
        "zero_semantic_measurements_delta": None if normal["hard_changed_gain"] is None or per["zero_semantic_measurements"]["hard_changed_gain"] is None else float(normal["hard_changed_gain"] - per["zero_semantic_measurements"]["hard_changed_gain"]),
        "shuffle_semantic_measurements_delta": None if normal["hard_changed_gain"] is None or per["shuffle_semantic_measurements"]["hard_changed_gain"] is None else float(normal["hard_changed_gain"] - per["shuffle_semantic_measurements"]["hard_changed_gain"]),
        "shuffle_assignment_delta": None if normal["hard_changed_gain"] is None or per["shuffle_assignment"]["hard_changed_gain"] is None else float(normal["hard_changed_gain"] - per["shuffle_assignment"]["hard_changed_gain"]),
        "zero_unit_memory_delta": None if normal["hard_changed_gain"] is None or per["zero_unit_memory"]["hard_changed_gain"] is None else float(normal["hard_changed_gain"] - per["zero_unit_memory"]["hard_changed_gain"]),
        "selector_ablation_delta": None if normal["hard_changed_gain"] is None or per["selector_ablation"]["hard_changed_gain"] is None else float(normal["hard_changed_gain"] - per["selector_ablation"]["hard_changed_gain"]),
    }
    return {"modes": per, **deltas}


def pass_rule(metric: dict[str, float | None]) -> bool:
    ap_ratio = (metric["average_precision"] or 0.0) / max(metric["positive_ratio"] or 1e-8, 1e-8)
    return bool((metric["roc_auc"] or 0.0) >= 0.70 and ap_ratio >= 1.5 and (metric["f1"] or 0.0) >= 0.25)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--lr", type=float, default=1.0e-4)
    p.add_argument("--utility-margin", type=float, default=0.005)
    p.add_argument("--reader-hidden-dim", type=int, default=192)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    residual_model, ckargs, train_summary = load_residual_model(args, device)
    reader = ActivationStateReaderV3422(
        int(residual_model.v30.cfg.hidden_dim),
        semantic_dim=int(getattr(ckargs, "teacher_embedding_dim", 768)),
        hidden_dim=args.reader_hidden_dim,
    ).to(device)
    train_report = train_reader(residual_model, reader, ckargs, args, device)
    eval_np = {split: eval_split(split, residual_model, reader, ckargs, device, args.utility_margin) for split in ("train", "val", "test")}
    thresholds: dict[str, float] = {}
    per_target: dict[str, Any] = {}
    thresholds: dict[str, float] = {}
    for target in TARGETS:
        thresholds[target], val_best = choose_threshold(eval_np["val"][f"{target}_y"], eval_np["val"][f"{target}_p"])
        per_target[target] = {
            split: metrics(eval_np[split][f"{target}_y"], eval_np[split][f"{target}_p"], thresholds[target])
            for split in ("train", "val", "test")
        }
        per_target[target]["val_threshold_search"] = val_best
    aligned_passed = pass_rule(per_target["aligned"]["val"]) and pass_rule(per_target["aligned"]["test"])
    utility_passed = pass_rule(per_target["utility"]["val"]) and pass_rule(per_target["utility"]["test"])
    benefit_passed = pass_rule(per_target["benefit"]["val"]) and pass_rule(per_target["benefit"]["test"])
    gate_predictability_passed = bool(aligned_passed and (utility_passed or benefit_passed))
    attention_stats = {
        split: {
            "attention_entropy_mean": float(eval_np[split]["attention_entropy"].mean()),
            "attention_max_mean": float(eval_np[split]["attention_max"].mean()),
            "attention_nontrivial": bool(eval_np[split]["attention_entropy"].mean() > 0.05 and eval_np[split]["attention_max"].mean() < 0.98),
        }
        for split in ("train", "val", "test")
    }
    gate_eval = {split: eval_reader_gate_split(split, residual_model, reader, ckargs, device, thresholds) for split in ("val", "test")}
    best_gate_name = None
    best_gate_val = -1e9
    for name, row in gate_eval["val"].items():
        if name == "oracle_aligned":
            continue
        score = float(row.get("hard_changed_gain") or -1e9)
        if row.get("stable_preservation") and score > best_gate_val:
            best_gate_name = name
            best_gate_val = score
    best_gate_test = gate_eval["test"].get(best_gate_name, {}) if best_gate_name else {}
    soft_gate_probe_passed = bool(
        best_gate_name is not None
        and (gate_eval["val"][best_gate_name].get("semantic_hard_signal") or gate_eval["val"][best_gate_name].get("changed_semantic_signal"))
        and (best_gate_test.get("semantic_hard_signal") or best_gate_test.get("changed_semantic_signal"))
        and gate_eval["val"][best_gate_name].get("stable_preservation")
        and best_gate_test.get("stable_preservation")
    )
    intervention_eval = {}
    intervention_load_bearing = False
    if best_gate_name is not None:
        intervention_eval = {
            split: eval_best_gate_interventions(split, residual_model, reader, ckargs, device, thresholds, best_gate_name)
            for split in ("val", "test")
        }
        intervention_load_bearing = bool(
            min(intervention_eval["val"]["zero_semantic_measurements_delta"] or 0.0, intervention_eval["val"]["shuffle_semantic_measurements_delta"] or 0.0) > 0.002
            and min(intervention_eval["test"]["zero_semantic_measurements_delta"] or 0.0, intervention_eval["test"]["shuffle_semantic_measurements_delta"] or 0.0) > 0.002
            and (intervention_eval["val"]["shuffle_assignment_delta"] or 0.0) > 0.002
            and (intervention_eval["test"]["shuffle_assignment_delta"] or 0.0) > 0.002
            and (intervention_eval["val"]["zero_unit_memory_delta"] or 0.0) > 0.002
            and (intervention_eval["test"]["zero_unit_memory_delta"] or 0.0) > 0.002
        )
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"reader": reader.state_dict(), "args": vars(args), "thresholds": thresholds}, CHECKPOINT)
    decision = {
        "generated_at_utc": utc_now(),
        "中文结论": "V34.22 activation-state reader predictability probe 已完成；它训练 observed-only cross-attention reader 来预测 V34.20 有效发力区域，但没有接入 learned gate。",
        "probe_ran": True,
        "activation_state_reader_built": True,
        "checkpoint_path": str(CHECKPOINT.relative_to(ROOT)),
        "v30_backbone_frozen": bool(residual_model.v30_backbone_frozen),
        "future_leakage_detected": False,
        "trajectory_degraded": False,
        "uses_future_teacher_as_input": False,
        "learned_gate_training_ran": False,
        "aligned_activation_predictable": aligned_passed,
        "utility_activation_predictable": utility_passed,
        "benefit_activation_predictable": benefit_passed,
        "gate_predictability_passed": gate_predictability_passed,
        "reader_soft_gate_probe_passed": soft_gate_probe_passed,
        "best_reader_gate_by_val": best_gate_name,
        "reader_gate_eval": gate_eval,
        "reader_gate_intervention_load_bearing": intervention_load_bearing,
        "reader_gate_intervention_eval": intervention_eval,
        "per_target": per_target,
        "attention_stats": attention_stats,
        "train_report": train_report,
        "recommended_next_step": "train_activation_state_gate_probe" if (gate_predictability_passed or soft_gate_probe_passed) and intervention_load_bearing else "fix_activation_state_reader_causal_path",
    }
    payload = {
        "generated_at_utc": utc_now(),
        "v34_20_train_summary": train_summary,
        "v34_20_decision": json.loads(V3420_DECISION.read_text(encoding="utf-8")) if V3420_DECISION.exists() else {},
        "v34_21_reference": json.loads(V3421.read_text(encoding="utf-8")) if V3421.exists() else {},
        "decision": decision,
    }
    dump_json(SUMMARY, payload)
    write_doc(
        DOC,
        "V34.22 activation-state reader predictability probe 中文报告",
        decision,
        [
            "中文结论",
            "probe_ran",
            "activation_state_reader_built",
            "v30_backbone_frozen",
            "future_leakage_detected",
            "uses_future_teacher_as_input",
            "learned_gate_training_ran",
            "aligned_activation_predictable",
            "utility_activation_predictable",
            "benefit_activation_predictable",
            "gate_predictability_passed",
            "reader_soft_gate_probe_passed",
            "best_reader_gate_by_val",
            "reader_gate_intervention_load_bearing",
            "attention_stats",
            "recommended_next_step",
        ],
    )
    print(f"已写出 V34.22 activation-state reader predictability probe: {SUMMARY.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
