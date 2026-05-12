#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))

from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, write_doc
from stwm.tools.ostf_v30_external_gt_schema_20260508 import utc_now


V345_ROOT = ROOT / "outputs/cache/stwm_ostf_v34_5_strict_residual_utility_targets/pointodyssey"
V347_ROOT = ROOT / "outputs/cache/stwm_ostf_v34_7_assignment_aware_residual_targets/pointodyssey"
MEAS_ROOT = ROOT / "outputs/cache/stwm_ostf_v34_semantic_measurement_bank/pointodyssey"
OUT_ROOT = ROOT / "outputs/cache/stwm_ostf_v34_8_causal_assignment_residual_targets/pointodyssey"
REPORT = ROOT / "reports/stwm_ostf_v34_8_causal_assignment_residual_target_build_20260512.json"
DOC = ROOT / "docs/STWM_OSTF_V34_8_CAUSAL_ASSIGNMENT_RESIDUAL_TARGET_BUILD_20260512.md"


def q(v: np.ndarray, quantile: float, default: float) -> float:
    v = np.asarray(v, dtype=np.float32)
    v = v[np.isfinite(v)]
    return float(np.quantile(v, quantile)) if v.size else default


def thresholds_for_split(split: str, args: argparse.Namespace) -> dict[str, float]:
    point_conf, inst_pur, sem_pur, meas_conf, teacher_conf, err = [], [], [], [], [], []
    for p in sorted((V347_ROOT / split).glob("*.npz")):
        uid = p.stem
        z7 = np.load(p, allow_pickle=True)
        z5 = np.load(V345_ROOT / split / f"{uid}.npz", allow_pickle=True)
        zm = np.load(MEAS_ROOT / split / f"{uid}.npz", allow_pickle=True)
        strict = np.asarray(z5["strict_residual_semantic_utility_mask"]).astype(bool)
        candidate = strict & (np.asarray(z5["semantic_hard_mask"]).astype(bool) | np.asarray(z5["changed_mask"]).astype(bool))
        if not candidate.any():
            continue
        assign = np.asarray(z7["current_unit_assignment"], dtype=np.float32)
        point_unit = assign.argmax(axis=-1)
        point_conf_arr = np.asarray(z7["point_unit_confidence"], dtype=np.float32)
        unit_inst = np.asarray(z7["unit_instance_purity"], dtype=np.float32)[point_unit]
        unit_sem = np.asarray(z7["unit_semantic_purity"], dtype=np.float32)[point_unit]
        obs_conf = np.asarray(zm["obs_measurement_confidence"], dtype=np.float32)
        obs_mask = np.asarray(zm["obs_semantic_measurement_mask"]).astype(bool)
        meas = (obs_conf * obs_mask).sum(axis=1) / np.maximum(obs_mask.sum(axis=1), 1)
        fut_conf = np.asarray(zm["fut_teacher_confidence"], dtype=np.float32)
        sem_err = 1.0 - np.asarray(z5["pointwise_semantic_cosine"], dtype=np.float32)
        point_conf.append(np.broadcast_to(point_conf_arr[:, None], strict.shape)[candidate])
        inst_pur.append(np.broadcast_to(unit_inst[:, None], strict.shape)[candidate])
        sem_pur.append(np.broadcast_to(unit_sem[:, None], strict.shape)[candidate])
        meas_conf.append(np.broadcast_to(meas[:, None], strict.shape)[candidate])
        teacher_conf.append(fut_conf[candidate])
        err.append(sem_err[candidate])
    return {
        "point_assignment_confidence": max(q(np.concatenate(point_conf) if point_conf else np.array([]), args.point_conf_quantile, 0.20), args.min_point_assignment_confidence),
        "unit_instance_purity": max(q(np.concatenate(inst_pur) if inst_pur else np.array([]), args.instance_purity_quantile, 0.62), args.min_unit_instance_purity),
        "unit_semantic_purity": max(q(np.concatenate(sem_pur) if sem_pur else np.array([]), args.semantic_purity_quantile, 0.58), args.min_unit_semantic_purity),
        "semantic_measurement_confidence": max(q(np.concatenate(meas_conf) if meas_conf else np.array([]), args.measurement_conf_quantile, 0.55), args.min_semantic_measurement_confidence),
        "teacher_confidence": max(q(np.concatenate(teacher_conf) if teacher_conf else np.array([]), args.teacher_conf_quantile, 0.55), args.min_teacher_confidence),
        "pointwise_error": q(np.concatenate(err) if err else np.array([]), args.error_quantile, 0.35),
    }


def process_split(split: str, args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, float]]:
    th = thresholds_for_split(split, args)
    out_dir = OUT_ROOT / split
    out_dir.mkdir(parents=True, exist_ok=True)
    totals = {"samples": 0, "valid": 0, "strict": 0, "v347": 0, "point_pos": 0, "unit_den": 0, "unit_pos": 0, "stable": 0, "meas_req": 0}
    hist_values = []
    for p in sorted((V347_ROOT / split).glob("*.npz")):
        uid = p.stem
        z7 = np.load(p, allow_pickle=True)
        z5 = np.load(V345_ROOT / split / f"{uid}.npz", allow_pickle=True)
        zm = np.load(MEAS_ROOT / split / f"{uid}.npz", allow_pickle=True)
        assign = np.asarray(z7["current_unit_assignment"], dtype=np.float32)
        point_unit = assign.argmax(axis=-1)
        point_conf = np.asarray(z7["point_unit_confidence"], dtype=np.float32)
        unit_inst = np.asarray(z7["unit_instance_purity"], dtype=np.float32)
        unit_sem = np.asarray(z7["unit_semantic_purity"], dtype=np.float32)
        unit_conf = assign.mean(axis=0).astype(np.float32)
        obs_conf = np.asarray(zm["obs_measurement_confidence"], dtype=np.float32)
        obs_mask = np.asarray(zm["obs_semantic_measurement_mask"]).astype(bool)
        meas = (obs_conf * obs_mask).sum(axis=1) / np.maximum(obs_mask.sum(axis=1), 1)
        agreement = np.asarray(zm["teacher_agreement_score"], dtype=np.float32)
        agreement_p = (agreement * obs_mask).sum(axis=1) / np.maximum(obs_mask.sum(axis=1), 1)
        strict = np.asarray(z5["strict_residual_semantic_utility_mask"]).astype(bool)
        v347 = np.asarray(z7["assignment_aware_residual_semantic_mask"]).astype(bool)
        stable = np.asarray(z7["stable_suppress_mask"]).astype(bool)
        valid = np.asarray(zm["fut_teacher_available_mask"]).astype(bool)
        sem_hard = np.asarray(z5["semantic_hard_mask"]).astype(bool)
        changed = np.asarray(z5["changed_mask"]).astype(bool)
        fut_conf = np.asarray(zm["fut_teacher_confidence"], dtype=np.float32)
        err = 1.0 - np.asarray(z5["pointwise_semantic_cosine"], dtype=np.float32)
        point_ok = (
            (point_conf >= th["point_assignment_confidence"])
            & (unit_inst[point_unit] >= th["unit_instance_purity"])
            & (unit_sem[point_unit] >= th["unit_semantic_purity"])
            & (meas >= th["semantic_measurement_confidence"])
        )
        base = strict & valid & (sem_hard | changed) & v347
        causal = (
            base
            & point_ok[:, None]
            & (fut_conf >= th["teacher_confidence"])
            & (err >= th["pointwise_error"])
        )
        max_pos = int(args.max_point_positive_ratio * max(int(valid.sum()), 1))
        if causal.sum() > max_pos:
            score = np.where(causal, err + 0.1 * fut_conf + 0.05 * meas[:, None], -np.inf).reshape(-1)
            keep = max(1, max_pos)
            idx = np.argpartition(-score, min(keep, score.size - 1))[:keep]
            filt = np.zeros(score.shape, dtype=bool)
            filt[idx] = np.isfinite(score[idx])
            causal = filt.reshape(causal.shape)
        point_to_unit = np.zeros((*causal.shape, assign.shape[1]), dtype=np.float32)
        for mi, ui in enumerate(point_unit):
            point_to_unit[mi, causal[mi], int(ui)] = 1.0
        unit_score = np.einsum("mu,mh->uh", assign, causal.astype(np.float32))
        unit_mass = np.maximum(assign.sum(axis=0)[:, None], 1e-6)
        unit_pos = (unit_score / unit_mass) > args.min_unit_positive_fraction
        meas_req = causal.copy()
        np.savez_compressed(
            out_dir / f"{uid}.npz",
            sample_uid=str(uid),
            point_id=np.asarray(z7["point_id"]).astype(np.int64),
            point_to_unit_assignment=assign,
            unit_instance_purity=unit_inst,
            unit_semantic_purity=unit_sem,
            unit_confidence=unit_conf,
            point_assignment_confidence=point_conf,
            semantic_measurement_confidence=obs_conf.astype(np.float32),
            semantic_measurement_agreement=agreement_p.astype(np.float32),
            strict_residual_semantic_utility_mask=strict,
            causal_assignment_residual_semantic_mask=causal,
            causal_semantic_measurement_required_mask=meas_req,
            causal_assignment_gate_target=causal.astype(np.float32),
            causal_unit_positive_mask=unit_pos,
            point_to_unit_residual_target=point_to_unit,
            stable_suppress_mask=stable,
            semantic_hard_mask=sem_hard,
            changed_mask=changed,
            leakage_safe=True,
        )
        totals["samples"] += 1
        totals["valid"] += int(valid.sum())
        totals["strict"] += int(strict.sum())
        totals["v347"] += int(v347.sum())
        totals["point_pos"] += int(causal.sum())
        totals["unit_den"] += int(unit_pos.size)
        totals["unit_pos"] += int(unit_pos.sum())
        totals["stable"] += int(stable.sum())
        totals["meas_req"] += int(meas_req.sum())
        if valid.sum() > 0:
            hist_values.append(float(causal.sum() / valid.sum()))
    summary = {
        "sample_count": totals["samples"],
        "valid_semantic_count": totals["valid"],
        "strict_positive_count": totals["strict"],
        "v34_7_assignment_positive_count": totals["v347"],
        "causal_point_positive_count": totals["point_pos"],
        "causal_unit_positive_count": totals["unit_pos"],
        "stable_suppress_count": totals["stable"],
        "point_positive_ratio": float(totals["point_pos"] / max(totals["valid"], 1)),
        "unit_positive_ratio": float(totals["unit_pos"] / max(totals["unit_den"], 1)),
        "semantic_measurement_required_ratio": float(totals["meas_req"] / max(totals["valid"], 1)),
        "coverage_loss_vs_v34_7": float(1.0 - totals["point_pos"] / max(totals["v347"], 1)),
        "coverage_loss_vs_v34_5_strict": float(1.0 - totals["point_pos"] / max(totals["strict"], 1)),
        "target_ratio_histogram": np.histogram(np.asarray(hist_values, dtype=np.float32), bins=[0, 0.02, 0.05, 0.10, 0.20, 1.0])[0].tolist() if hist_values else [0, 0, 0, 0, 0],
    }
    return summary, th


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--point-conf-quantile", type=float, default=0.65)
    p.add_argument("--instance-purity-quantile", type=float, default=0.60)
    p.add_argument("--semantic-purity-quantile", type=float, default=0.60)
    p.add_argument("--measurement-conf-quantile", type=float, default=0.70)
    p.add_argument("--teacher-conf-quantile", type=float, default=0.60)
    p.add_argument("--error-quantile", type=float, default=0.70)
    p.add_argument("--min-point-assignment-confidence", type=float, default=0.12)
    p.add_argument("--min-unit-instance-purity", type=float, default=0.62)
    p.add_argument("--min-unit-semantic-purity", type=float, default=0.58)
    p.add_argument("--min-semantic-measurement-confidence", type=float, default=0.55)
    p.add_argument("--min-teacher-confidence", type=float, default=0.55)
    p.add_argument("--max-point-positive-ratio", type=float, default=0.20)
    p.add_argument("--min-unit-positive-fraction", type=float, default=0.025)
    args = p.parse_args()
    split_summaries, thresholds, blockers = {}, {}, {}
    for split in ("train", "val", "test"):
        summary, th = process_split(split, args)
        split_summaries[split] = summary
        thresholds[split] = th
        if summary["causal_point_positive_count"] <= 0:
            blockers[split] = "没有 causal assignment residual 正样本"
        elif summary["point_positive_ratio"] > args.max_point_positive_ratio + 1e-6:
            blockers[split] = "causal 正样本比例超过 20% 上限"
        elif summary["stable_suppress_count"] <= 0:
            blockers[split] = "缺少 stable suppress negative"
    ready = not blockers
    payload = {
        "generated_at_utc": utc_now(),
        "中文结论": "已构建更严格的 causal assignment targets；正样本需要 assignment、unit purity 和 observed semantic measurement 同时满足阈值。",
        "output_root": str(OUT_ROOT.relative_to(ROOT)),
        "causal_assignment_targets_built": True,
        "causal_assignment_target_ready": ready,
        "point_positive_ratio_by_split": {k: v["point_positive_ratio"] for k, v in split_summaries.items()},
        "unit_positive_ratio_by_split": {k: v["unit_positive_ratio"] for k, v in split_summaries.items()},
        "semantic_measurement_required_ratio_by_split": {k: v["semantic_measurement_required_ratio"] for k, v in split_summaries.items()},
        "coverage_loss_vs_v34_7": {k: v["coverage_loss_vs_v34_7"] for k, v in split_summaries.items()},
        "coverage_loss_vs_v34_5_strict": {k: v["coverage_loss_vs_v34_5_strict"] for k, v in split_summaries.items()},
        "thresholds_by_split": thresholds,
        "target_ratio_histogram_by_split": {k: v["target_ratio_histogram"] for k, v in split_summaries.items()},
        "split_summaries": split_summaries,
        "leakage_safe": True,
        "future_teacher_embedding_input_allowed": False,
        "exact_blockers": blockers,
    }
    dump_json(REPORT, payload)
    write_doc(DOC, "V34.8 causal assignment residual target 构建中文报告", payload, ["中文结论", "causal_assignment_targets_built", "causal_assignment_target_ready", "point_positive_ratio_by_split", "unit_positive_ratio_by_split", "semantic_measurement_required_ratio_by_split", "coverage_loss_vs_v34_7", "coverage_loss_vs_v34_5_strict", "thresholds_by_split", "exact_blockers"])
    print(f"已写出 causal assignment target 报告: {REPORT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
