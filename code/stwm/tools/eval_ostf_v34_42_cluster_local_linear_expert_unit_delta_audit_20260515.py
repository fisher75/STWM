#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import setproctitle
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))

setproctitle.setproctitle("python")

import stwm.tools.train_eval_ostf_v34_36_predictability_filtered_unit_delta_writer_20260514 as v36
from stwm.tools.eval_ostf_v34_26_full_system_baseline_claim_boundary_benchmark_20260514 import masks, observed_mean
from stwm.tools.eval_ostf_v34_27_evidence_anchored_full_system_benchmark_20260514 import Acc, finalize_method, update_method
from stwm.tools.eval_ostf_v34_35_unit_delta_generalization_audit_20260514 import fit_ridge, predict, unit_features
from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, write_doc
from stwm.tools.ostf_v30_external_gt_schema_20260508 import utc_now
from stwm.tools.train_ostf_v33_2_visual_semantic_identity_20260509 import move_batch
from stwm.tools.train_ostf_v34_18_topk_evidence_oracle_residual_probe_20260513 import make_loader, model_inputs
from stwm.tools.train_eval_ostf_v34_31_raw_unit_delta_value_memory_20260514 import compose, load_frozen_residual_model, read_unit_delta, set_seed
from stwm.tools.train_eval_ostf_v34_33_oracle_unit_delta_value_decoder_20260514 import top1


TARGET_ROOT = ROOT / "outputs/cache/stwm_ostf_v34_39_prototype_blended_unit_delta_targets/pointodyssey"
PROTO_TARGET_ROOT = ROOT / "outputs/cache/stwm_ostf_v34_38_cluster_regularized_unit_delta_targets/pointodyssey"
REPORT = ROOT / "reports/stwm_ostf_v34_42_cluster_local_linear_expert_unit_delta_audit_20260515.json"
DOC = ROOT / "docs/STWM_OSTF_V34_42_CLUSTER_LOCAL_LINEAR_EXPERT_UNIT_DELTA_AUDIT_20260515.md"


def load_target_and_label(split: str, uids: list[str], device: torch.device) -> dict[str, torch.Tensor]:
    deltas, active, point_masks, labels = [], [], [], []
    for uid in uids:
        t = np.load(TARGET_ROOT / split / f"{uid}.npz", allow_pickle=True)
        p = np.load(PROTO_TARGET_ROOT / split / f"{uid}.npz", allow_pickle=True)
        deltas.append(torch.from_numpy(np.asarray(t["predictability_filtered_unit_delta"], dtype=np.float32)))
        active.append(torch.from_numpy(np.asarray(t["predictability_filtered_active"]).astype(bool)))
        point_masks.append(torch.from_numpy(np.asarray(t["point_predictable_mask"]).astype(bool)))
        labels.append(torch.from_numpy(np.asarray(p["prototype_id"], dtype=np.int64)).clamp_min(0))
    return {
        "delta": torch.stack(deltas, dim=0).to(device),
        "active": torch.stack(active, dim=0).to(device),
        "point_mask": torch.stack(point_masks, dim=0).to(device),
        "label": torch.stack(labels, dim=0).to(device),
    }


def random_projection(in_dim: int, out_dim: int, seed: int) -> torch.Tensor:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    return torch.randn((in_dim, out_dim), generator=gen, dtype=torch.float32) / math.sqrt(float(out_dim))


def collect(split: str, model: Any, ckargs: argparse.Namespace, proj: torch.Tensor | None, args: argparse.Namespace, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    xs, ys, active, labels = [], [], [], []
    with torch.no_grad():
        for batch in make_loader(split, ckargs, shuffle=False):
            bd = move_batch(batch, device)
            out = model(**model_inputs(bd), intervention="force_gate_zero")
            assign = top1(out["point_to_unit_assignment"].float())
            target = load_target_and_label(split, bd["uid"], device)
            feat = unit_features(out, bd, assign).detach().flatten(0, 2).cpu()
            if proj is not None:
                feat = feat @ proj
            xs.append(feat)
            ys.append(target["delta"].detach().flatten(0, 2).cpu())
            active.append(target["active"].detach().flatten(0, 2).cpu())
            labels.append(target["label"].detach().flatten(0, 2).cpu())
    return torch.cat(xs, dim=0), torch.cat(ys, dim=0), torch.cat(active, dim=0).bool(), torch.cat(labels, dim=0).long()


def fit_experts(x: torch.Tensor, y: torch.Tensor, active: torch.Tensor, label: torch.Tensor, args: argparse.Namespace) -> dict[str, Any]:
    global_ridge = fit_ridge(x, y, active, args.ridge_lambda)
    experts: dict[int, dict[str, torch.Tensor]] = {}
    counts: dict[int, int] = {}
    for pid in range(args.prototype_count):
        mask = active & (label == pid)
        counts[pid] = int(mask.sum().item())
        if counts[pid] >= args.min_cluster_count:
            experts[pid] = fit_ridge(x, y, mask, args.ridge_lambda)
    return {"global": global_ridge, "experts": experts, "counts": counts}


def predict_experts(experts: dict[str, Any], x: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
    out = torch.zeros((x.shape[0], experts["global"]["weight"].shape[1]), dtype=torch.float32)
    global_pred = predict(experts["global"], x)
    out[:] = global_pred
    for pid, model in experts["experts"].items():
        mask = label == int(pid)
        if bool(mask.any()):
            out[mask] = predict(model, x[mask])
    return out


def eval_compose(split: str, model: Any, ckargs: argparse.Namespace, experts: dict[str, Any], proj: torch.Tensor, args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    acc_by_scale = {float(s): Acc() for s in args.eval_scales}
    with torch.no_grad():
        for batch in make_loader(split, ckargs, shuffle=False):
            bd = move_batch(batch, device)
            out = model(**model_inputs(bd), intervention="force_gate_zero")
            assign = top1(out["point_to_unit_assignment"].float())
            target = load_target_and_label(split, bd["uid"], device)
            feat = unit_features(out, bd, assign)
            b, u, h, f = feat.shape
            pred = predict_experts(experts, feat.reshape(-1, f).cpu() @ proj, target["label"].flatten(0, 2).cpu()).to(device)
            unit_delta = pred.reshape(b, u, h, -1) * target["active"][..., None].float()
            anchor = observed_mean(bd)
            point_delta = read_unit_delta(assign, unit_delta)
            gate = target["point_mask"].float()
            mm = masks(bd)
            pointwise = out["pointwise_semantic_belief"]
            fut = bd["fut_teacher_embedding"]
            for scale in args.eval_scales:
                method = f"v34_42_cluster_local_linear_expert_scale_{scale:g}"
                final = compose(anchor, point_delta, gate, float(scale))
                update_method(acc_by_scale[float(scale)], "copy_mean_observed", anchor, pointwise=pointwise, target=fut, mm=mm)
                update_method(acc_by_scale[float(scale)], method, final, pointwise=pointwise, target=fut, mm=mm, anchor=anchor, gate=gate)
    rows = {}
    for scale, acc in acc_by_scale.items():
        names = sorted({key.split(":")[0] for key in acc.sum.keys()})
        rows[str(scale)] = {name: finalize_method(acc, name) for name in names}
    return rows


def main() -> int:
    args = parse_args()
    set_seed(args.seed)
    v36.TARGET_ROOT = TARGET_ROOT
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model, ckargs, _ = load_frozen_residual_model(args, device)
    print("V34.42: 收集 train unit features / blended unit_delta / prototype labels...", flush=True)
    # First pass discovers feature dimension cheaply.
    first_batch = next(iter(make_loader("train", ckargs, shuffle=False)))
    bd = move_batch(first_batch, device)
    with torch.no_grad():
        out = model(**model_inputs(bd), intervention="force_gate_zero")
        assign = top1(out["point_to_unit_assignment"].float())
        feat_dim = int(unit_features(out, bd, assign).shape[-1])
    proj = random_projection(feat_dim, args.feature_proj_dim, args.seed)
    train_x, train_y, train_active, train_label = collect("train", model, ckargs, proj, args, device)
    print("V34.42: 拟合 prototype-conditioned local linear experts...", flush=True)
    experts = fit_experts(train_x, train_y, train_active, train_label, args)
    split_stats = {}
    for split in ("train", "val", "test"):
        x, y, active, label = collect(split, model, ckargs, proj, args, device)
        pred = predict_experts(experts, x, label)
        if bool(active.any()):
            cos = (torch.nn.functional.normalize(pred[active], dim=-1) * torch.nn.functional.normalize(y[active], dim=-1)).sum(dim=-1)
            split_stats[split] = {
                "active_count": int(active.sum().item()),
                "direction_cosine_mean": float(cos.mean().item()),
                "direction_cosine_p50": float(cos.median().item()),
            }
        else:
            split_stats[split] = {"active_count": 0, "direction_cosine_mean": None, "direction_cosine_p50": None}
    print("V34.42: 评估 local expert compose 上界...", flush=True)
    compose_rows = {split: eval_compose(split, model, ckargs, experts, proj, args, device) for split in ("val", "test")}
    selected_scale = max(
        args.eval_scales,
        key=lambda s: float(compose_rows["val"][str(float(s))][f"v34_42_cluster_local_linear_expert_scale_{s:g}"]["hard_changed_gain_vs_anchor"] or -1.0e9),
    )
    method = f"v34_42_cluster_local_linear_expert_scale_{selected_scale:g}"
    val_m = compose_rows["val"][str(float(selected_scale))][method]
    test_m = compose_rows["test"][str(float(selected_scale))][method]
    expert_upper_bound_passed = bool(
        (val_m["hard_changed_gain_vs_anchor"] or -1.0) > 0.002
        and (test_m["hard_changed_gain_vs_anchor"] or -1.0) > 0.002
    )
    payload = {
        "generated_at_utc": utc_now(),
        "中文结论": "V34.42 cluster-local linear expert audit 完成；用 prototype/mode 选择局部线性 correction expert，检验固定 codebook 失败后是否仍存在跨样本共享的可预测连续 correction。",
        "feature_proj_dim": args.feature_proj_dim,
        "prototype_count": args.prototype_count,
        "min_cluster_count": args.min_cluster_count,
        "expert_count_fit": len(experts["experts"]),
        "cluster_count_min": min(experts["counts"].values()),
        "cluster_count_max": max(experts["counts"].values()),
        "target_predictability": split_stats,
        "selected_scale_by_val": float(selected_scale),
        "selected_metrics": {"val": val_m, "test": test_m},
        "compose_scale_sweep": compose_rows,
        "expert_upper_bound_passed": expert_upper_bound_passed,
        "recommended_next_step": "train_cluster_conditioned_neural_expert_writer" if expert_upper_bound_passed else "fix_observed_predictable_delta_targets_or_stop_unit_delta_route",
        "阶段性分析": "若 local linear expert 也不能在 val/test 产生正 anchor gain，则说明当前 observed-only features 与 prototype labels 对未来 delta 的可预测性仍不足，继续加神经 writer 或 gate 都会跑偏。",
        "论文相关问题解决方案参考": "该轮对应 Mixture-of-Experts / local linear experts：不是把 prototype 当固定输出，而是把 prototype 当 routing context，每个 mode 学共享的线性 residual function。",
    }
    dump_json(REPORT, payload)
    write_doc(
        DOC,
        "V34.42 cluster-local linear expert unit_delta audit 中文报告",
        payload,
        ["中文结论", "feature_proj_dim", "expert_count_fit", "target_predictability", "selected_scale_by_val", "selected_metrics", "expert_upper_bound_passed", "recommended_next_step", "阶段性分析", "论文相关问题解决方案参考"],
    )
    print(f"已写出 V34.42 cluster-local linear expert audit 报告: {REPORT.relative_to(ROOT)}", flush=True)
    print(f"expert_upper_bound_passed: {expert_upper_bound_passed}", flush=True)
    print(f"recommended_next_step: {payload['recommended_next_step']}", flush=True)
    return 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--teacher-embedding-dim", type=int, default=768)
    p.add_argument("--reader-hidden-dim", type=int, default=192)
    p.add_argument("--target-kind", choices=["top1"], default="top1")
    p.add_argument("--prototype-count", type=int, default=64)
    p.add_argument("--feature-proj-dim", type=int, default=256)
    p.add_argument("--min-cluster-count", type=int, default=32)
    p.add_argument("--ridge-lambda", type=float, default=50.0)
    p.add_argument("--eval-scales", type=float, nargs="+", default=[0.25, 0.5, 1.0, 2.0])
    p.add_argument("--cpu", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
