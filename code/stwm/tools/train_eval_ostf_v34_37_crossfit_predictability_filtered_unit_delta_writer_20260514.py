#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import setproctitle
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))

setproctitle.setproctitle("python")

import stwm.tools.train_eval_ostf_v34_36_predictability_filtered_unit_delta_writer_20260514 as v36
from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, write_doc
from stwm.tools.ostf_v30_external_gt_schema_20260508 import utc_now
from stwm.tools.train_ostf_v33_2_visual_semantic_identity_20260509 import move_batch
from stwm.tools.train_ostf_v34_18_topk_evidence_oracle_residual_probe_20260513 import make_loader, model_inputs
from stwm.tools.train_ostf_v34_20_hard_changed_aligned_topk_residual_probe_20260513 import hard_changed_aligned_mask
from stwm.tools.train_eval_ostf_v34_31_raw_unit_delta_value_memory_20260514 import load_frozen_residual_model, set_seed
from stwm.tools.train_eval_ostf_v34_33_oracle_unit_delta_value_decoder_20260514 import load_target_batch, top1
from stwm.tools.eval_ostf_v34_35_unit_delta_generalization_audit_20260514 import fit_ridge, predict, unit_features


TARGET_ROOT = ROOT / "outputs/cache/stwm_ostf_v34_37_crossfit_predictability_filtered_unit_delta_targets/pointodyssey"
TARGET_REPORT = ROOT / "reports/stwm_ostf_v34_37_crossfit_predictability_filtered_unit_delta_target_build_20260514.json"
TARGET_DOC = ROOT / "docs/STWM_OSTF_V34_37_CROSSFIT_PREDICTABILITY_FILTERED_UNIT_DELTA_TARGET_BUILD_20260514.md"
CKPT_DIR = ROOT / "outputs/checkpoints/stwm_ostf_v34_37_crossfit_predictability_filtered_unit_delta_writer_h32_m128"
SUMMARY = ROOT / "reports/stwm_ostf_v34_37_crossfit_predictability_filtered_unit_delta_writer_summary_20260514.json"
DECISION = ROOT / "reports/stwm_ostf_v34_37_crossfit_predictability_filtered_unit_delta_writer_decision_20260514.json"
DOC = ROOT / "docs/STWM_OSTF_V34_37_CROSSFIT_PREDICTABILITY_FILTERED_UNIT_DELTA_WRITER_SUMMARY_20260514.md"
DECISION_DOC = ROOT / "docs/STWM_OSTF_V34_37_CROSSFIT_PREDICTABILITY_FILTERED_UNIT_DELTA_WRITER_DECISION_20260514.md"


def patch_v36_paths() -> None:
    v36.TARGET_ROOT = TARGET_ROOT
    v36.CKPT_DIR = CKPT_DIR
    v36.SUMMARY = SUMMARY
    v36.DECISION = DECISION
    v36.DOC = DOC
    v36.DECISION_DOC = DECISION_DOC


def collect_records(split: str, model: Any, ckargs: argparse.Namespace, args: argparse.Namespace, device: torch.device) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with torch.no_grad():
        for batch in make_loader(split, ckargs, shuffle=False):
            bd = move_batch(batch, device)
            out = model(**model_inputs(bd), intervention="force_gate_zero")
            assign = top1(out["point_to_unit_assignment"].float())
            feat = unit_features(out, bd, assign).detach().cpu()
            target = load_target_batch(split, bd["uid"], device, args.target_kind)
            oracle = target["oracle_unit_delta"].detach().cpu()
            active = target["oracle_unit_delta_active"].detach().cpu().bool()
            hard = hard_changed_aligned_mask(bd).detach().cpu().bool()
            assign_cpu = assign.detach().cpu()
            for i, uid in enumerate(bd["uid"]):
                records.append({
                    "uid": str(uid),
                    "feat": feat[i],
                    "oracle": oracle[i],
                    "active": active[i],
                    "assign": assign_cpu[i],
                    "hard": hard[i],
                })
    return records


def flatten_records(records: list[dict[str, Any]]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x = torch.cat([r["feat"].flatten(0, 1) for r in records], dim=0)
    y = torch.cat([r["oracle"].flatten(0, 1) for r in records], dim=0)
    a = torch.cat([r["active"].flatten(0, 1) for r in records], dim=0).bool()
    return x, y, a


def save_records(split: str, records: list[dict[str, Any]], preds: list[torch.Tensor], args: argparse.Namespace) -> dict[str, Any]:
    (TARGET_ROOT / split).mkdir(parents=True, exist_ok=True)
    total_active = 0
    total_filtered = 0
    total_point_pos = 0
    total_point_valid = 0
    cos_values = []
    for rec, pred in zip(records, preds):
        oracle = rec["oracle"]
        active = rec["active"]
        assign = rec["assign"]
        hard = rec["hard"]
        pred_norm = pred.norm(dim=-1)
        oracle_norm = oracle.norm(dim=-1)
        cos = (F.normalize(pred, dim=-1) * F.normalize(oracle, dim=-1)).sum(dim=-1)
        confidence = ((cos - args.predictability_cos_threshold) / max(1.0e-6, 1.0 - args.predictability_cos_threshold)).clamp(0.0, 1.0)
        confidence = confidence.pow(args.predictability_confidence_power)
        filtered_active = active & (confidence > 0.0) & (pred_norm > args.min_predicted_delta_norm) & (oracle_norm > args.min_oracle_delta_norm)
        filtered_delta = oracle * confidence[..., None] * filtered_active[..., None].float()
        point_mask = (torch.einsum("mu,uh->mh", assign, filtered_active.float()) > 0.5) & hard
        total_active += int(active.sum().item())
        total_filtered += int(filtered_active.sum().item())
        total_point_pos += int(point_mask.sum().item())
        total_point_valid += int(hard.sum().item())
        if bool(active.any()):
            cos_values.append(cos[active])
        np.savez_compressed(
            TARGET_ROOT / split / f"{rec['uid']}.npz",
            uid=rec["uid"],
            predictability_filtered_unit_delta=filtered_delta.numpy().astype(np.float32),
            predictability_filtered_active=filtered_active.numpy().astype(bool),
            predictability_score=confidence.numpy().astype(np.float32),
            ridge_pred_unit_delta=pred.numpy().astype(np.float32),
            original_oracle_unit_delta=oracle.numpy().astype(np.float32),
            original_oracle_active=active.numpy().astype(bool),
            point_predictable_mask=point_mask.numpy().astype(bool),
        )
    cos_cat = torch.cat(cos_values) if cos_values else torch.zeros(1)
    return {
        "original_active_count": total_active,
        "filtered_active_count": total_filtered,
        "filtered_active_ratio_vs_original": float(total_filtered / max(total_active, 1)),
        "point_predictable_ratio_vs_hard_changed": float(total_point_pos / max(total_point_valid, 1)),
        "direction_cosine_mean_on_original_active": float(cos_cat.mean().item()),
        "direction_cosine_p50_on_original_active": float(cos_cat.median().item()),
    }


def build_crossfit_targets(model: Any, ckargs: argparse.Namespace, args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    if (TARGET_ROOT / "train").exists() and not args.rebuild_targets:
        return {"target_built": True, "target_root": str(TARGET_ROOT.relative_to(ROOT)), "reused_existing": True}
    print("V34.37: 收集 train records 并执行 out-of-fold predictability filtering...", flush=True)
    train_records = collect_records("train", model, ckargs, args, device)
    n = len(train_records)
    folds = max(2, int(args.crossfit_folds))
    train_preds: list[torch.Tensor | None] = [None] * n
    for fold in range(folds):
        held = [i for i in range(n) if i % folds == fold]
        kept = [i for i in range(n) if i % folds != fold]
        x, y, active = flatten_records([train_records[i] for i in kept])
        ridge = fit_ridge(x, y, active, args.ridge_lambda)
        for i in held:
            feat = train_records[i]["feat"]
            u, h, f = feat.shape
            train_preds[i] = predict(ridge, feat.flatten(0, 1)).reshape(u, h, -1)
        print(f"V34.37 crossfit fold {fold + 1}/{folds} 完成", flush=True)
    train_stats = save_records("train", train_records, [p for p in train_preds if p is not None], args)

    print("V34.37: 用全 train ridge 预测 val/test filtered targets...", flush=True)
    train_x, train_y, train_active = flatten_records(train_records)
    full_ridge = fit_ridge(train_x, train_y, train_active, args.ridge_lambda)
    split_stats = {"train": train_stats}
    for split in ("val", "test"):
        records = collect_records(split, model, ckargs, args, device)
        preds = []
        for rec in records:
            u, h, f = rec["feat"].shape
            preds.append(predict(full_ridge, rec["feat"].flatten(0, 1)).reshape(u, h, -1))
        split_stats[split] = save_records(split, records, preds, args)
    payload = {
        "generated_at_utc": utc_now(),
        "中文结论": "V34.37 cross-fitted predictability-filtered targets 已构建；train split 使用 out-of-fold ridge，避免 in-sample predictability 过宽。",
        "target_built": True,
        "target_root": str(TARGET_ROOT.relative_to(ROOT)),
        "reused_existing": False,
        "crossfit_folds": folds,
        "ridge_lambda": args.ridge_lambda,
        "predictability_cos_threshold": args.predictability_cos_threshold,
        "split_stats": split_stats,
    }
    dump_json(TARGET_REPORT, payload)
    write_doc(TARGET_DOC, "V34.37 crossfit predictability target build 中文报告", payload, ["中文结论", "target_built", "crossfit_folds", "ridge_lambda", "predictability_cos_threshold", "split_stats"])
    return payload


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lr", type=float, default=8.0e-5)
    p.add_argument("--teacher-embedding-dim", type=int, default=768)
    p.add_argument("--value-hidden-dim", type=int, default=256)
    p.add_argument("--reader-hidden-dim", type=int, default=192)
    p.add_argument("--max-delta-magnitude", type=float, default=2.5)
    p.add_argument("--target-kind", choices=["top1"], default="top1")
    p.add_argument("--ridge-lambda", type=float, default=100.0)
    p.add_argument("--predictability-cos-threshold", type=float, default=0.15)
    p.add_argument("--predictability-confidence-power", type=float, default=1.0)
    p.add_argument("--min-predicted-delta-norm", type=float, default=0.01)
    p.add_argument("--min-oracle-delta-norm", type=float, default=0.01)
    p.add_argument("--train-residual-scale", type=float, default=1.0)
    p.add_argument("--eval-scales", type=float, nargs="+", default=[0.25, 0.5, 1.0, 2.0])
    p.add_argument("--target-supervision-weight", type=float, default=3.0)
    p.add_argument("--final-target-weight", type=float, default=0.8)
    p.add_argument("--anchor-gain-weight", type=float, default=0.8)
    p.add_argument("--assignment-contrast-weight", type=float, default=1.5)
    p.add_argument("--unit-contrast-weight", type=float, default=1.0)
    p.add_argument("--stable-suppress-weight", type=float, default=0.05)
    p.add_argument("--anchor-gain-margin", type=float, default=0.006)
    p.add_argument("--assignment-margin", type=float, default=0.006)
    p.add_argument("--unit-margin", type=float, default=0.006)
    p.add_argument("--rebuild-targets", action="store_true")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--crossfit-folds", type=int, default=4)
    return p.parse_args()


def main() -> int:
    patch_v36_paths()
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model, ckargs, _ = load_frozen_residual_model(args, device)
    target_report = build_crossfit_targets(model, ckargs, args, device)
    head, train_summary = v36.train_one(model, ckargs, args, device)
    v36.evaluate(model, ckargs, head, train_summary, target_report, args, device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
