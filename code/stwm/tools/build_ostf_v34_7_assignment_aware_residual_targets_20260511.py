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
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))

setproctitle.setproctitle("python")

from stwm.modules.ostf_v34_3_pointwise_unit_residual_world_model import PointwiseUnitResidualWorldModelV343
from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, write_doc
from stwm.tools.ostf_v30_external_gt_schema_20260508 import utc_now
from stwm.tools.train_ostf_v33_2_visual_semantic_identity_20260509 import move_batch
from stwm.tools.train_ostf_v34_6_residual_parameterization_sweep_20260511 import StrictResidualUtilityDataset, collate_v345


OUT_ROOT = ROOT / "outputs/cache/stwm_ostf_v34_7_assignment_aware_residual_targets/pointodyssey"
REPORT = ROOT / "reports/stwm_ostf_v34_7_assignment_aware_residual_target_build_20260511.json"
DOC = ROOT / "docs/STWM_OSTF_V34_7_ASSIGNMENT_AWARE_RESIDUAL_TARGET_BUILD_20260511.md"
V346_DECISION = ROOT / "reports/stwm_ostf_v34_6_residual_parameterization_decision_20260511.json"


def load_best_model(args: argparse.Namespace, device: torch.device) -> tuple[PointwiseUnitResidualWorldModelV343, argparse.Namespace]:
    decision = json.loads(V346_DECISION.read_text(encoding="utf-8"))
    ck = torch.load(ROOT / decision["best_checkpoint_path"], map_location="cpu")
    ckargs = argparse.Namespace(**ck["args"])
    ckargs.batch_size = args.batch_size
    ckargs.num_workers = args.num_workers
    model = PointwiseUnitResidualWorldModelV343(
        ckargs.v30_checkpoint,
        teacher_embedding_dim=ckargs.teacher_embedding_dim,
        units=ckargs.trace_units,
        horizon=ckargs.horizon,
    ).to(device)
    model.load_state_dict(ck["model"], strict=True)
    model.eval()
    return model, ckargs


def weighted_purity(assign: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    m, u = assign.shape
    dom = np.full((u,), -1, dtype=np.int64)
    pur = np.zeros((u,), dtype=np.float32)
    for ui in range(u):
        w = assign[:, ui]
        total = float(w.sum())
        if total <= 1e-8:
            continue
        scores: dict[int, float] = {}
        for mi in range(m):
            lab = int(labels[mi])
            if lab < 0:
                continue
            scores[lab] = scores.get(lab, 0.0) + float(w[mi])
        if scores:
            best_lab, best_w = max(scores.items(), key=lambda kv: kv[1])
            dom[ui] = int(best_lab)
            pur[ui] = float(best_w / max(total, 1e-8))
    return dom, pur


def semantic_clusters(obs_sem: np.ndarray, obs_mask: np.ndarray) -> np.ndarray:
    valid = obs_mask.astype(np.float32)
    pooled = (np.nan_to_num(obs_sem).astype(np.float32) * valid[..., None]).sum(axis=1) / np.maximum(valid.sum(axis=1, keepdims=True), 1.0)
    norm = np.linalg.norm(pooled, axis=-1)
    clusters = np.where(norm > 1e-6, np.argmax(np.abs(pooled), axis=-1), -1)
    return clusters.astype(np.int64)


def process_split(split: str, args: argparse.Namespace, model: PointwiseUnitResidualWorldModelV343, ckargs: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    ds = StrictResidualUtilityDataset(split, ckargs)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_v345)
    out_dir = OUT_ROOT / split
    out_dir.mkdir(parents=True, exist_ok=True)
    totals = {"samples": 0, "strict": 0, "point_pos": 0, "unit_valid": 0, "unit_pos": 0, "stable": 0}
    with torch.no_grad():
        for batch in loader:
            bd = move_batch(batch, device)
            out = model(
                obs_points=bd["obs_points"],
                obs_vis=bd["obs_vis"],
                obs_conf=bd["obs_conf"],
                obs_semantic_measurements=bd["obs_semantic_measurements"],
                obs_semantic_measurement_mask=bd["obs_semantic_measurement_mask"],
                semantic_id=bd["semantic_id"],
                intervention="force_gate_zero",
            )
            assign_b = out["point_to_unit_assignment"].detach().cpu().numpy()
            unit_conf_b = out["unit_confidence"].detach().cpu().numpy()
            for bi, uid in enumerate(batch["uid"]):
                assign = assign_b[bi].astype(np.float32)
                point_unit = assign.argmax(axis=-1)
                point_conf = assign.max(axis=-1).astype(np.float32)
                inst = bd["point_to_instance_id"][bi].detach().cpu().numpy().astype(np.int64)
                obs_sem = bd["obs_semantic_measurements"][bi].detach().cpu().numpy()
                obs_mask = bd["obs_semantic_measurement_mask"][bi].detach().cpu().numpy().astype(bool)
                sem_cluster = semantic_clusters(obs_sem, obs_mask)
                dom_inst, inst_purity = weighted_purity(assign, inst)
                dom_sem, sem_purity = weighted_purity(assign, sem_cluster)
                unit_ok = ((inst_purity >= args.min_unit_instance_purity) | (sem_purity >= args.min_unit_semantic_purity)) & (unit_conf_b[bi] >= args.min_unit_confidence)
                point_ok = unit_ok[point_unit] & (point_conf >= args.min_point_unit_confidence)
                strict = bd["strict_residual_semantic_utility_mask"][bi].detach().cpu().numpy().astype(bool)
                stable = bd["strict_stable_suppress_mask"][bi].detach().cpu().numpy().astype(bool)
                identity_hard = bd["identity_hard_train_mask"][bi].detach().cpu().numpy().astype(bool)
                point_pos = strict & point_ok[:, None]
                identity_pos = identity_hard & point_ok[:, None]
                m, h = strict.shape
                u = assign.shape[1]
                unit_pos_score = np.einsum("mu,mh->uh", assign, point_pos.astype(np.float32))
                unit_mass = np.maximum(assign.sum(axis=0)[:, None], 1e-6)
                unit_residual_positive = (unit_pos_score / unit_mass) > args.min_unit_positive_fraction
                point_to_unit_target = np.zeros((m, h, u), dtype=np.float32)
                for mi in range(m):
                    ui = int(point_unit[mi])
                    point_to_unit_target[mi, point_pos[mi], ui] = 1.0
                np.savez_compressed(
                    out_dir / f"{uid}.npz",
                    sample_uid=str(uid),
                    point_id=bd["point_id"][bi].detach().cpu().numpy().astype(np.int64),
                    current_unit_assignment=assign,
                    dominant_instance_per_unit=dom_inst,
                    dominant_semantic_cluster_per_unit=dom_sem,
                    unit_instance_purity=inst_purity,
                    unit_semantic_purity=sem_purity,
                    point_unit_confidence=point_conf,
                    strict_residual_semantic_utility_mask=strict,
                    assignment_aware_residual_semantic_mask=point_pos,
                    assignment_aware_residual_identity_mask=identity_pos,
                    unit_residual_positive_mask=unit_residual_positive,
                    unit_residual_gate_target=unit_residual_positive.astype(np.float32),
                    point_to_unit_residual_target=point_to_unit_target,
                    stable_suppress_mask=stable,
                    leakage_safe=True,
                )
                totals["samples"] += 1
                totals["strict"] += int(strict.sum())
                totals["point_pos"] += int(point_pos.sum())
                totals["unit_valid"] += int(unit_ok.size * h)
                totals["unit_pos"] += int(unit_residual_positive.sum())
                totals["stable"] += int(stable.sum())
    return {
        "sample_count": totals["samples"],
        "strict_positive_count": totals["strict"],
        "assignment_aware_point_positive_count": totals["point_pos"],
        "assignment_aware_unit_positive_count": totals["unit_pos"],
        "stable_suppress_count": totals["stable"],
        "point_positive_ratio": float(totals["point_pos"] / max(totals["strict"], 1)),
        "unit_positive_ratio": float(totals["unit_pos"] / max(totals["unit_valid"], 1)),
        "coverage_loss_vs_v34_5_strict_targets": float(1.0 - totals["point_pos"] / max(totals["strict"], 1)),
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--min-unit-instance-purity", type=float, default=0.60)
    p.add_argument("--min-unit-semantic-purity", type=float, default=0.55)
    p.add_argument("--min-unit-confidence", type=float, default=0.05)
    p.add_argument("--min-point-unit-confidence", type=float, default=0.10)
    p.add_argument("--min-unit-positive-fraction", type=float, default=0.03)
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model, ckargs = load_best_model(args, device)
    split_summaries = {split: process_split(split, args, model, ckargs, device) for split in ("train", "val", "test")}
    blockers = {
        split: "no_assignment_aware_positives"
        for split, s in split_summaries.items()
        if s["assignment_aware_point_positive_count"] <= 0 or s["assignment_aware_unit_positive_count"] <= 0
    }
    ready = not blockers
    payload = {
        "generated_at_utc": utc_now(),
        "output_root": str(OUT_ROOT.relative_to(ROOT)),
        "assignment_aware_targets_built": True,
        "assignment_aware_target_ready": ready,
        "point_positive_ratio_by_split": {k: v["point_positive_ratio"] for k, v in split_summaries.items()},
        "unit_positive_ratio_by_split": {k: v["unit_positive_ratio"] for k, v in split_summaries.items()},
        "unit_purity_thresholds": {
            "min_unit_instance_purity": args.min_unit_instance_purity,
            "min_unit_semantic_purity": args.min_unit_semantic_purity,
            "min_unit_confidence": args.min_unit_confidence,
            "min_point_unit_confidence": args.min_point_unit_confidence,
        },
        "coverage_loss_vs_v34_5_strict_targets": {k: v["coverage_loss_vs_v34_5_strict_targets"] for k, v in split_summaries.items()},
        "split_summaries": split_summaries,
        "leakage_safe": True,
        "exact_blockers": blockers,
    }
    dump_json(REPORT, payload)
    write_doc(DOC, "STWM OSTF V34.7 Assignment-Aware Residual Target Build", payload, ["assignment_aware_targets_built", "assignment_aware_target_ready", "point_positive_ratio_by_split", "unit_positive_ratio_by_split", "unit_purity_thresholds", "coverage_loss_vs_v34_5_strict_targets", "exact_blockers"])
    print(REPORT.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
