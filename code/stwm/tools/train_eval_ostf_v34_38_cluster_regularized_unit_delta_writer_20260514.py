#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
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
from stwm.tools.train_eval_ostf_v34_31_raw_unit_delta_value_memory_20260514 import load_frozen_residual_model, set_seed


BASE_TARGET_ROOT = ROOT / "outputs/cache/stwm_ostf_v34_37_crossfit_predictability_filtered_unit_delta_targets/pointodyssey"
TARGET_ROOT = ROOT / "outputs/cache/stwm_ostf_v34_38_cluster_regularized_unit_delta_targets/pointodyssey"
TARGET_REPORT = ROOT / "reports/stwm_ostf_v34_38_cluster_regularized_unit_delta_target_build_20260514.json"
TARGET_DOC = ROOT / "docs/STWM_OSTF_V34_38_CLUSTER_REGULARIZED_UNIT_DELTA_TARGET_BUILD_20260514.md"
CKPT_DIR = ROOT / "outputs/checkpoints/stwm_ostf_v34_38_cluster_regularized_unit_delta_writer_h32_m128"
SUMMARY = ROOT / "reports/stwm_ostf_v34_38_cluster_regularized_unit_delta_writer_summary_20260514.json"
DECISION = ROOT / "reports/stwm_ostf_v34_38_cluster_regularized_unit_delta_writer_decision_20260514.json"
DOC = ROOT / "docs/STWM_OSTF_V34_38_CLUSTER_REGULARIZED_UNIT_DELTA_WRITER_SUMMARY_20260514.md"
DECISION_DOC = ROOT / "docs/STWM_OSTF_V34_38_CLUSTER_REGULARIZED_UNIT_DELTA_WRITER_DECISION_20260514.md"


def patch_v36_paths() -> None:
    v36.TARGET_ROOT = TARGET_ROOT
    v36.CKPT_DIR = CKPT_DIR
    v36.SUMMARY = SUMMARY
    v36.DECISION = DECISION
    v36.DOC = DOC
    v36.DECISION_DOC = DECISION_DOC


def load_train_deltas() -> torch.Tensor:
    chunks = []
    for path in sorted((BASE_TARGET_ROOT / "train").glob("*.npz")):
        z = np.load(path, allow_pickle=True)
        delta = torch.from_numpy(np.asarray(z["predictability_filtered_unit_delta"], dtype=np.float32))
        active = torch.from_numpy(np.asarray(z["predictability_filtered_active"]).astype(bool))
        if bool(active.any()):
            chunks.append(delta[active])
    if not chunks:
        raise RuntimeError("没有可用于聚类的 V34.37 filtered unit_delta active target")
    return torch.cat(chunks, dim=0)


def kmeans_directions(delta: torch.Tensor, k: int, iters: int, seed: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    x = F.normalize(delta.float(), dim=-1)
    k = min(int(k), x.shape[0])
    idx = torch.randperm(x.shape[0], generator=gen)[:k]
    cent_dir = x[idx].clone()
    labels = torch.zeros(x.shape[0], dtype=torch.long)
    for _ in range(int(iters)):
        score = x @ cent_dir.T
        labels = score.argmax(dim=-1)
        new_cent = []
        for c in range(k):
            mask = labels == c
            if bool(mask.any()):
                new_cent.append(F.normalize(x[mask].mean(dim=0), dim=0))
            else:
                new_cent.append(x[torch.randint(0, x.shape[0], (1,), generator=gen).item()])
        cent_dir = torch.stack(new_cent, dim=0)
    proto = []
    counts = []
    for c in range(k):
        mask = labels == c
        counts.append(int(mask.sum().item()))
        if bool(mask.any()):
            proto.append(delta[mask].mean(dim=0))
        else:
            proto.append(delta[torch.randint(0, delta.shape[0], (1,), generator=gen).item()])
    return torch.stack(proto, dim=0), cent_dir, torch.tensor(counts, dtype=torch.long)


def smooth_split(split: str, prototypes: torch.Tensor, prototype_dirs: torch.Tensor) -> dict[str, Any]:
    out_dir = TARGET_ROOT / split
    out_dir.mkdir(parents=True, exist_ok=True)
    total_active = 0
    total_smoothed = 0
    total_point_pos = 0
    total_point_valid = 0
    sims = []
    for path in sorted((BASE_TARGET_ROOT / split).glob("*.npz")):
        z = np.load(path, allow_pickle=True)
        old_delta = torch.from_numpy(np.asarray(z["predictability_filtered_unit_delta"], dtype=np.float32))
        old_active = torch.from_numpy(np.asarray(z["predictability_filtered_active"]).astype(bool))
        old_point_mask = torch.from_numpy(np.asarray(z["point_predictable_mask"]).astype(bool))
        old_score = torch.from_numpy(np.asarray(z["predictability_score"], dtype=np.float32))
        flat = old_delta.flatten(0, 1)
        active_flat = old_active.flatten(0, 1)
        proto_delta = torch.zeros_like(flat)
        proto_id = torch.full((flat.shape[0],), -1, dtype=torch.long)
        proto_sim = torch.zeros((flat.shape[0],), dtype=torch.float32)
        if bool(active_flat.any()):
            x = F.normalize(flat[active_flat], dim=-1)
            score = x @ prototype_dirs.T
            label = score.argmax(dim=-1)
            sim = score.max(dim=-1).values
            proto_delta[active_flat] = prototypes[label]
            proto_id[active_flat] = label
            proto_sim[active_flat] = sim
            sims.append(sim)
        proto_delta = proto_delta.reshape_as(old_delta)
        proto_id = proto_id.reshape(old_active.shape)
        proto_sim = proto_sim.reshape(old_active.shape)
        # Keep the V34.37 confidence gate, but replace noisy values with shared prototype payloads.
        smoothed_delta = proto_delta * old_score[..., None].clamp(0.0, 1.0) * old_active[..., None].float()
        smoothed_active = old_active & (proto_sim > 0.0)
        total_active += int(old_active.sum().item())
        total_smoothed += int(smoothed_active.sum().item())
        total_point_pos += int(old_point_mask.sum().item())
        total_point_valid += int(old_point_mask.numel())
        np.savez_compressed(
            out_dir / path.name,
            uid=str(z["uid"]) if "uid" in z else path.stem,
            predictability_filtered_unit_delta=smoothed_delta.numpy().astype(np.float32),
            predictability_filtered_active=smoothed_active.numpy().astype(bool),
            predictability_score=old_score.numpy().astype(np.float32),
            prototype_id=proto_id.numpy().astype(np.int32),
            prototype_similarity=proto_sim.numpy().astype(np.float32),
            prototype_smoothed=True,
            original_predictability_filtered_unit_delta=old_delta.numpy().astype(np.float32),
            original_predictability_filtered_active=old_active.numpy().astype(bool),
            point_predictable_mask=old_point_mask.numpy().astype(bool),
        )
    sim_cat = torch.cat(sims) if sims else torch.zeros(1)
    return {
        "base_active_count": total_active,
        "smoothed_active_count": total_smoothed,
        "smoothed_active_ratio_vs_base": float(total_smoothed / max(total_active, 1)),
        "point_positive_count": total_point_pos,
        "point_positive_ratio_all_tokens": float(total_point_pos / max(total_point_valid, 1)),
        "prototype_similarity_mean": float(sim_cat.mean().item()),
        "prototype_similarity_p50": float(sim_cat.median().item()),
    }


def build_targets(args: argparse.Namespace) -> dict[str, Any]:
    if (TARGET_ROOT / "train").exists() and not args.rebuild_targets:
        return {"target_built": True, "target_root": str(TARGET_ROOT.relative_to(ROOT)), "reused_existing": True}
    print("V34.38: 从 V34.37 filtered unit_delta 构建跨样本 prototype/centroid target...", flush=True)
    TARGET_ROOT.mkdir(parents=True, exist_ok=True)
    train_delta = load_train_deltas()
    prototypes, prototype_dirs, counts = kmeans_directions(train_delta, args.prototype_count, args.kmeans_iters, args.seed)
    split_stats = {split: smooth_split(split, prototypes, prototype_dirs) for split in ("train", "val", "test")}
    payload = {
        "generated_at_utc": utc_now(),
        "中文结论": "V34.38 cluster-regularized unit_delta targets 已构建；把 V34.37 filtered delta 替换为跨样本共享 prototype payload，减少 sample-specific correction 噪声。",
        "target_built": True,
        "target_root": str(TARGET_ROOT.relative_to(ROOT)),
        "base_target_root": str(BASE_TARGET_ROOT.relative_to(ROOT)),
        "reused_existing": False,
        "prototype_count": int(prototypes.shape[0]),
        "kmeans_iters": args.kmeans_iters,
        "prototype_count_nonempty": int((counts > 0).sum().item()),
        "prototype_count_min": int(counts.min().item()),
        "prototype_count_max": int(counts.max().item()),
        "split_stats": split_stats,
    }
    dump_json(TARGET_REPORT, payload)
    write_doc(
        TARGET_DOC,
        "V34.38 cluster-regularized unit_delta target build 中文报告",
        payload,
        ["中文结论", "target_built", "target_root", "base_target_root", "prototype_count", "prototype_count_nonempty", "split_stats"],
    )
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
    p.add_argument("--prototype-count", type=int, default=64)
    p.add_argument("--kmeans-iters", type=int, default=25)
    p.add_argument("--rebuild-targets", action="store_true")
    p.add_argument("--cpu", action="store_true")
    return p.parse_args()


def update_written_reports(target_report: dict[str, Any]) -> None:
    if SUMMARY.exists():
        payload = json.loads(SUMMARY.read_text(encoding="utf-8"))
        payload["target_report"] = target_report
        payload["阶段性分析"] = "V34.38 将 V34.37 crossfit-filtered unit_delta 进一步聚类为跨样本共享 prototype correction，目标是降低 sample-specific delta 噪声并提升 writer 泛化。"
        payload["论文相关问题解决方案参考"] = "该轮对应 vector quantization / prototype distillation 思路：把连续 teacher residual 转成共享 correction codebook，使 observed-only writer 学到可复用模式，而不是记忆样本噪声。"
        payload["最佳下一步方案"] = payload.get("decision", {}).get("recommended_next_step")
        dump_json(SUMMARY, payload)
    if DECISION.exists():
        decision = json.loads(DECISION.read_text(encoding="utf-8"))
        decision["中文结论"] = "V34.38 cluster-regularized unit_delta writer 完成；本轮不训练 gate、不跑 M512，只验证 prototype-smoothed correction target 是否能提升 learned writer 泛化。"
        decision["cluster_regularized_targets_built"] = True
        decision["target_report"] = target_report
        if decision.get("probe_passed"):
            decision["recommended_next_step"] = "return_to_full_system_benchmark"
        elif decision.get("unit_residual_improves_evidence_anchor"):
            decision["recommended_next_step"] = "train_observed_predictability_activation"
        else:
            decision["recommended_next_step"] = "fix_cluster_regularized_targets_or_writer_generalization"
        dump_json(DECISION, decision)
        write_doc(
            DECISION_DOC,
            "V34.38 cluster-regularized unit_delta writer 决策中文报告",
            decision,
            ["中文结论", "cluster_regularized_targets_built", "probe_passed", "selected_config_by_val", "beats_copy_topk_baseline", "unit_residual_improves_evidence_anchor", "assignment_load_bearing_on_system", "unit_memory_load_bearing_on_system", "semantic_hard_signal", "changed_semantic_signal", "stable_preservation", "recommended_next_step"],
        )


def main() -> int:
    patch_v36_paths()
    args = parse_args()
    set_seed(args.seed)
    target_report = build_targets(args)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model, ckargs, _ = load_frozen_residual_model(args, device)
    head, train_summary = v36.train_one(model, ckargs, args, device)
    v36.evaluate(model, ckargs, head, train_summary, target_report, args, device)
    update_written_reports(target_report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
