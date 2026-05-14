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

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))

setproctitle.setproctitle("python")

import stwm.tools.train_eval_ostf_v34_36_predictability_filtered_unit_delta_writer_20260514 as v36
from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, write_doc
from stwm.tools.ostf_v30_external_gt_schema_20260508 import utc_now
from stwm.tools.train_eval_ostf_v34_31_raw_unit_delta_value_memory_20260514 import load_frozen_residual_model, set_seed


BASE_TARGET_ROOT = ROOT / "outputs/cache/stwm_ostf_v34_37_crossfit_predictability_filtered_unit_delta_targets/pointodyssey"
PROTO_TARGET_ROOT = ROOT / "outputs/cache/stwm_ostf_v34_38_cluster_regularized_unit_delta_targets/pointodyssey"
TARGET_ROOT = ROOT / "outputs/cache/stwm_ostf_v34_39_prototype_blended_unit_delta_targets/pointodyssey"
TARGET_REPORT = ROOT / "reports/stwm_ostf_v34_39_prototype_blended_unit_delta_target_build_20260514.json"
TARGET_DOC = ROOT / "docs/STWM_OSTF_V34_39_PROTOTYPE_BLENDED_UNIT_DELTA_TARGET_BUILD_20260514.md"
CKPT_DIR = ROOT / "outputs/checkpoints/stwm_ostf_v34_39_prototype_blended_unit_delta_writer_h32_m128"
SUMMARY = ROOT / "reports/stwm_ostf_v34_39_prototype_blended_unit_delta_writer_summary_20260514.json"
DECISION = ROOT / "reports/stwm_ostf_v34_39_prototype_blended_unit_delta_writer_decision_20260514.json"
DOC = ROOT / "docs/STWM_OSTF_V34_39_PROTOTYPE_BLENDED_UNIT_DELTA_WRITER_SUMMARY_20260514.md"
DECISION_DOC = ROOT / "docs/STWM_OSTF_V34_39_PROTOTYPE_BLENDED_UNIT_DELTA_WRITER_DECISION_20260514.md"


def patch_v36_paths() -> None:
    v36.TARGET_ROOT = TARGET_ROOT
    v36.CKPT_DIR = CKPT_DIR
    v36.SUMMARY = SUMMARY
    v36.DECISION = DECISION
    v36.DOC = DOC
    v36.DECISION_DOC = DECISION_DOC


def build_split(split: str, alpha: float) -> dict[str, Any]:
    out_dir = TARGET_ROOT / split
    out_dir.mkdir(parents=True, exist_ok=True)
    active_total = 0
    point_total = 0
    point_valid = 0
    norms = []
    proto_norms = []
    base_norms = []
    for base_path in sorted((BASE_TARGET_ROOT / split).glob("*.npz")):
        proto_path = PROTO_TARGET_ROOT / split / base_path.name
        if not proto_path.exists():
            raise FileNotFoundError(f"缺少 prototype target: {proto_path}")
        base = np.load(base_path, allow_pickle=True)
        proto = np.load(proto_path, allow_pickle=True)
        base_delta = np.asarray(base["predictability_filtered_unit_delta"], dtype=np.float32)
        proto_delta = np.asarray(proto["predictability_filtered_unit_delta"], dtype=np.float32)
        base_active = np.asarray(base["predictability_filtered_active"]).astype(bool)
        proto_active = np.asarray(proto["predictability_filtered_active"]).astype(bool)
        point_mask = np.asarray(base["point_predictable_mask"]).astype(bool)
        score = np.asarray(base["predictability_score"], dtype=np.float32) if "predictability_score" in base else np.ones(base_active.shape, dtype=np.float32)
        blended = alpha * base_delta + (1.0 - alpha) * proto_delta
        active = base_active & (np.linalg.norm(blended, axis=-1) > 1.0e-8)
        active_total += int(active.sum())
        point_total += int(point_mask.sum())
        point_valid += int(point_mask.size)
        if active.any():
            norms.append(np.linalg.norm(blended[active], axis=-1))
            base_norms.append(np.linalg.norm(base_delta[active], axis=-1))
            proto_norms.append(np.linalg.norm(proto_delta[active], axis=-1))
        np.savez_compressed(
            out_dir / base_path.name,
            uid=str(base["uid"]) if "uid" in base else base_path.stem,
            predictability_filtered_unit_delta=blended.astype(np.float32),
            predictability_filtered_active=active.astype(bool),
            predictability_score=score.astype(np.float32),
            point_predictable_mask=point_mask.astype(bool),
            blend_alpha=np.float32(alpha),
            base_active=base_active.astype(bool),
            prototype_active=proto_active.astype(bool),
            base_target_root=str(BASE_TARGET_ROOT.relative_to(ROOT)),
            prototype_target_root=str(PROTO_TARGET_ROOT.relative_to(ROOT)),
        )
    norm_cat = np.concatenate(norms) if norms else np.zeros((1,), dtype=np.float32)
    base_norm_cat = np.concatenate(base_norms) if base_norms else np.zeros((1,), dtype=np.float32)
    proto_norm_cat = np.concatenate(proto_norms) if proto_norms else np.zeros((1,), dtype=np.float32)
    return {
        "active_count": active_total,
        "point_positive_count": point_total,
        "point_positive_ratio_all_tokens": float(point_total / max(point_valid, 1)),
        "blended_delta_norm_mean": float(norm_cat.mean()),
        "base_delta_norm_mean_on_active": float(base_norm_cat.mean()),
        "prototype_delta_norm_mean_on_active": float(proto_norm_cat.mean()),
    }


def build_targets(args: argparse.Namespace) -> dict[str, Any]:
    if (TARGET_ROOT / "train").exists() and not args.rebuild_targets:
        return {
            "target_built": True,
            "target_root": str(TARGET_ROOT.relative_to(ROOT)),
            "reused_existing": True,
            "blend_alpha": args.blend_alpha,
            "cached_sweep_reference": "reports/stwm_ostf_v34_39_prototype_blend_target_sweep_20260514.json",
        }
    if not (BASE_TARGET_ROOT / "train").exists():
        raise FileNotFoundError(f"缺少 V34.37 base target root: {BASE_TARGET_ROOT}")
    if not (PROTO_TARGET_ROOT / "train").exists():
        raise FileNotFoundError(f"缺少 V34.38 prototype target root: {PROTO_TARGET_ROOT}")
    print(f"V34.39: 构建 prototype-blended unit_delta target，alpha={args.blend_alpha:.2f}", flush=True)
    TARGET_ROOT.mkdir(parents=True, exist_ok=True)
    split_stats = {split: build_split(split, args.blend_alpha) for split in ("train", "val", "test")}
    payload = {
        "generated_at_utc": utc_now(),
        "中文结论": "V34.39 prototype-blended unit_delta target 已构建；不是硬替换 centroid，而是在 V34.37 crossfit target 上加入轻度 prototype smoothing，保留 cached upper bound 的同时降低 sample-specific 噪声。",
        "target_built": True,
        "target_root": str(TARGET_ROOT.relative_to(ROOT)),
        "base_target_root": str(BASE_TARGET_ROOT.relative_to(ROOT)),
        "prototype_target_root": str(PROTO_TARGET_ROOT.relative_to(ROOT)),
        "cached_sweep_reference": "reports/stwm_ostf_v34_39_prototype_blend_target_sweep_20260514.json",
        "blend_alpha": args.blend_alpha,
        "reused_existing": False,
        "split_stats": split_stats,
    }
    dump_json(TARGET_REPORT, payload)
    write_doc(
        TARGET_DOC,
        "V34.39 prototype-blended unit_delta target build 中文报告",
        payload,
        ["中文结论", "target_built", "target_root", "base_target_root", "prototype_target_root", "blend_alpha", "split_stats"],
    )
    return payload


def rename_checkpoint(train_summary: dict[str, Any], seed: int) -> dict[str, Any]:
    old_rel = train_summary.get("checkpoint_path")
    if not old_rel:
        return train_summary
    old_path = ROOT / old_rel
    new_path = CKPT_DIR / f"v34_39_prototype_blended_unit_delta_writer_m128_h32_seed{seed}.pt"
    if old_path.exists() and old_path != new_path:
        new_path.parent.mkdir(parents=True, exist_ok=True)
        old_path.replace(new_path)
    train_summary["checkpoint_path"] = str(new_path.relative_to(ROOT))
    return train_summary


def update_written_reports(target_report: dict[str, Any]) -> None:
    if SUMMARY.exists():
        payload = json.loads(SUMMARY.read_text(encoding="utf-8"))
        payload["target_report"] = target_report
        payload["中文结论"] = "V34.39 prototype-blended writer 训练/评估完成；该轮验证轻度 prototype smoothing 后的可预测 correction 是否能被 learned writer 泛化。"
        payload["阶段性分析"] = "V34.38 证明硬 centroid 会破坏 cached target 上界；V34.39 改为 blend，不把 prototype 当答案，而把它作为跨样本共享先验，测试 writer 是否能从更平滑的 target 中获得 val/test 正增益。"
        payload["论文相关问题解决方案参考"] = "该轮参考 prototype distillation / vector-quantized residual memory / mixture-of-experts 的思想：保留连续 target 的信息量，同时引入共享 correction 模式，避免每个样本的细碎 delta 直接监督 writer。"
        payload["最佳下一步方案"] = payload.get("decision", {}).get("recommended_next_step")
        dump_json(SUMMARY, payload)
    if DECISION.exists():
        decision = json.loads(DECISION.read_text(encoding="utf-8"))
        decision["中文结论"] = "V34.39 prototype-blended unit_delta writer 完成；不训练 gate、不跑 M512，只验证 learned writer 是否能在 copy/top-k evidence anchor 上泛化出正 correction。"
        decision["prototype_blended_targets_built"] = True
        decision["blend_alpha"] = target_report.get("blend_alpha")
        decision["target_report"] = target_report
        # v36 的推荐语偏旧，这里按当前 claim boundary 改写。
        if decision.get("probe_passed"):
            decision["recommended_next_step"] = "return_to_full_system_benchmark"
        elif decision.get("unit_residual_improves_evidence_anchor"):
            decision["recommended_next_step"] = "rerun_writer_seed123_or_refine_assignment_contrast"
        else:
            decision["recommended_next_step"] = "fix_writer_generalization_or_target_predictability"
        dump_json(DECISION, decision)
        write_doc(
            DECISION_DOC,
            "V34.39 prototype-blended unit_delta writer 决策中文报告",
            decision,
            [
                "中文结论",
                "prototype_blended_targets_built",
                "blend_alpha",
                "probe_passed",
                "selected_config_by_val",
                "beats_copy_topk_baseline",
                "unit_residual_improves_evidence_anchor",
                "assignment_load_bearing_on_system",
                "unit_memory_load_bearing_on_system",
                "semantic_hard_signal",
                "changed_semantic_signal",
                "stable_preservation",
                "recommended_next_step",
            ],
        )


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
    p.add_argument("--blend-alpha", type=float, default=0.9)
    p.add_argument("--rebuild-targets", action="store_true")
    p.add_argument("--cpu", action="store_true")
    return p.parse_args()


def main() -> int:
    patch_v36_paths()
    args = parse_args()
    set_seed(args.seed)
    target_report = build_targets(args)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model, ckargs, _ = load_frozen_residual_model(args, device)
    head, train_summary = v36.train_one(model, ckargs, args, device)
    train_summary = rename_checkpoint(train_summary, args.seed)
    v36.evaluate(model, ckargs, head, train_summary, target_report, args, device)
    update_written_reports(target_report)
    print(f"已写出 V34.39 prototype-blended writer 决策报告: {DECISION.relative_to(ROOT)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
