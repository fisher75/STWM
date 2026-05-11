#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))

from stwm.modules.ostf_v34_2_pointwise_no_unit_baseline import PointwiseNoUnitBaselineV342
from stwm.tools.eval_ostf_v34_1_identity_bound_semantic_trace_units_20260511 import _norm
from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, write_doc
from stwm.tools.ostf_v30_external_gt_schema_20260508 import utc_now
from stwm.tools.train_ostf_v33_2_visual_semantic_identity_20260509 import move_batch
from stwm.tools.train_ostf_v34_1_identity_bound_semantic_trace_units_20260511 import make_loader
from stwm.tools.train_ostf_v34_2_pointwise_no_unit_baseline_20260511 import CKPT_DIR as POINT_CKPT_DIR, SUMMARY as POINT_TRAIN
from stwm.tools.train_ostf_v34_semantic_trace_units_20260510 import GLOBAL_ROOT, IDENTITY_ROOT, MASK_ROOT, MEAS_ROOT


OUT_ROOT = ROOT / "outputs/cache/stwm_ostf_v34_4_residual_utility_targets/pointodyssey"
REPORT = ROOT / "reports/stwm_ostf_v34_4_residual_utility_target_build_20260511.json"
DOC = ROOT / "docs/STWM_OSTF_V34_4_RESIDUAL_UTILITY_TARGET_BUILD_20260511.md"


def load_pointwise(args: argparse.Namespace, device: torch.device) -> tuple[PointwiseNoUnitBaselineV342, argparse.Namespace]:
    tr = json.loads(POINT_TRAIN.read_text(encoding="utf-8"))
    ckpt = ROOT / tr.get("checkpoint_path", str(POINT_CKPT_DIR / "v34_2_pointwise_no_unit_m128_h32_seed42_best.pt"))
    ck = torch.load(ckpt, map_location="cpu")
    ckargs = argparse.Namespace(**ck["args"])
    ckargs.batch_size = 1
    ckargs.num_workers = args.num_workers
    model = PointwiseNoUnitBaselineV342(ckargs.v30_checkpoint, teacher_embedding_dim=ckargs.teacher_embedding_dim).to(device)
    model.load_state_dict(ck["model"], strict=True)
    model.eval()
    return model, ckargs


def copy_cosine(obs: np.ndarray, obs_mask: np.ndarray, target: np.ndarray) -> np.ndarray:
    last = np.zeros((obs.shape[0], target.shape[-1]), dtype=np.float32)
    for mi in range(obs.shape[0]):
        idx = np.where(obs_mask[mi].astype(bool))[0]
        if idx.size:
            last[mi] = obs[mi, idx[-1]]
    copy = np.broadcast_to(last[:, None, :], target.shape)
    return (_norm(copy) * _norm(target)).sum(axis=-1)


def process_split(split: str, args: argparse.Namespace, model: PointwiseNoUnitBaselineV342, ckargs: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    loader = make_loader(split, ckargs, shuffle=False)
    out_dir = OUT_ROOT / split
    out_dir.mkdir(parents=True, exist_ok=True)
    totals = {
        "samples": 0,
        "valid_sem": 0,
        "sem_pos": 0,
        "id_valid": 0,
        "id_pos": 0,
        "stable": 0,
        "stable_suppress": 0,
        "gate_available": 0,
        "gate_pos": 0,
        "teacher_low_conf": 0,
    }
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
            )
            uid = str(batch["uid"][0])
            pred = out["future_semantic_belief"][0].detach().cpu().numpy()
            tgt = bd["fut_teacher_embedding"][0].detach().cpu().numpy()
            sem_mask = bd["fut_teacher_available_mask"][0].detach().cpu().numpy().astype(bool)
            conf = bd["fut_teacher_confidence"][0].detach().cpu().numpy().astype(np.float32)
            pointwise_cos = (_norm(pred) * _norm(tgt)).sum(axis=-1)
            error = 1.0 - pointwise_cos
            copy_cos = copy_cosine(
                bd["obs_semantic_measurements"][0].detach().cpu().numpy(),
                bd["obs_semantic_measurement_mask"][0].detach().cpu().numpy(),
                tgt,
            )
            stable = sem_mask & (copy_cos >= args.stable_copy_cosine_threshold)
            changed = sem_mask & (copy_cos < args.changed_copy_cosine_threshold)
            hard_sem = sem_mask & bd["semantic_hard_train_mask"][0].detach().cpu().numpy().astype(bool)
            high_conf = conf >= args.teacher_confidence_threshold
            sem_pos = sem_mask & high_conf & (error >= args.pointwise_error_threshold) & (hard_sem | changed)
            stable_suppress = stable & high_conf & (error <= args.stable_good_error_threshold)
            same = bd["fut_same_instance_as_obs"][0].detach().cpu().numpy().astype(np.float32)
            id_avail = bd["fut_instance_available_mask"][0].detach().cpu().numpy().astype(bool) & bd["fut_global_instance_available_mask"][0].detach().cpu().numpy().astype(bool)
            id_hard = bd["identity_hard_train_mask"][0].detach().cpu().numpy().astype(bool) & id_avail
            id_score = torch.sigmoid(out["future_identity_belief"][0]).detach().cpu().numpy().astype(np.float32)
            id_error = np.abs(id_score - same)
            id_margin = np.abs(id_score - 0.5)
            id_pos = id_hard & ((id_error >= args.identity_error_threshold) | (id_margin <= args.identity_low_margin_threshold))
            gate_target = (sem_pos | id_pos).astype(np.float32)
            gate_available = (sem_mask | id_avail) & (sem_pos | id_pos | stable_suppress)
            np.savez_compressed(
                out_dir / f"{uid}.npz",
                sample_uid=uid,
                point_id=bd["point_id"][0].detach().cpu().numpy(),
                pointwise_semantic_cosine=pointwise_cos.astype(np.float32),
                pointwise_identity_error=id_error.astype(np.float32),
                semantic_target_confidence=conf.astype(np.float32),
                semantic_hard_mask=hard_sem.astype(bool),
                changed_mask=changed.astype(bool),
                stable_mask=stable.astype(bool),
                stable_suppress_mask=stable_suppress.astype(bool),
                identity_hard_mask=id_hard.astype(bool),
                residual_semantic_utility_mask=sem_pos.astype(bool),
                residual_identity_utility_mask=id_pos.astype(bool),
                residual_gate_target=gate_target.astype(np.float32),
                residual_gate_available_mask=gate_available.astype(bool),
                leakage_safe=True,
                future_labels_supervision_only=True,
            )
            totals["samples"] += 1
            totals["valid_sem"] += int(sem_mask.sum())
            totals["sem_pos"] += int(sem_pos.sum())
            totals["id_valid"] += int(id_avail.sum())
            totals["id_pos"] += int(id_pos.sum())
            totals["stable"] += int(stable.sum())
            totals["stable_suppress"] += int(stable_suppress.sum())
            totals["gate_available"] += int(gate_available.sum())
            totals["gate_pos"] += int(gate_target[gate_available].sum()) if gate_available.any() else 0
            totals["teacher_low_conf"] += int((sem_mask & ~high_conf).sum())
    return {
        "sample_count": totals["samples"],
        "residual_semantic_positive_count": totals["sem_pos"],
        "residual_identity_positive_count": totals["id_pos"],
        "stable_suppress_count": totals["stable_suppress"],
        "valid_semantic_count": totals["valid_sem"],
        "valid_identity_count": totals["id_valid"],
        "residual_semantic_positive_ratio": float(totals["sem_pos"] / max(totals["valid_sem"], 1)),
        "residual_identity_positive_ratio": float(totals["id_pos"] / max(totals["id_valid"], 1)),
        "stable_suppress_ratio": float(totals["stable_suppress"] / max(totals["stable"], 1)),
        "gate_positive_ratio": float(totals["gate_pos"] / max(totals["gate_available"], 1)),
        "low_confidence_teacher_count": totals["teacher_low_conf"],
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--semantic-measurement-bank-root", default=str(MEAS_ROOT))
    p.add_argument("--semantic-identity-sidecar-root", default=str(IDENTITY_ROOT))
    p.add_argument("--global-identity-label-root", default=str(GLOBAL_ROOT))
    p.add_argument("--unit-identity-binding-root", default=str(ROOT / "outputs/cache/stwm_ostf_v34_1_unit_identity_binding_targets/pointodyssey"))
    p.add_argument("--hard-mask-manifest", default=str(MASK_ROOT / "H32_M128_seed42.json"))
    p.add_argument("--m-points", type=int, default=128)
    p.add_argument("--horizon", type=int, default=32)
    p.add_argument("--teacher-embedding-dim", type=int, default=768)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--teacher-confidence-threshold", type=float, default=0.60)
    p.add_argument("--pointwise-error-threshold", type=float, default=0.35)
    p.add_argument("--stable-good-error-threshold", type=float, default=0.20)
    p.add_argument("--stable-copy-cosine-threshold", type=float, default=0.80)
    p.add_argument("--changed-copy-cosine-threshold", type=float, default=0.65)
    p.add_argument("--identity-error-threshold", type=float, default=0.35)
    p.add_argument("--identity-low-margin-threshold", type=float, default=0.12)
    p.add_argument("--cpu", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model, ckargs = load_pointwise(args, device)
    summaries = {split: process_split(split, args, model, ckargs, device) for split in ("train", "val", "test")}
    ready = all(v["residual_semantic_positive_count"] > 0 and v["stable_suppress_count"] > 0 for v in summaries.values())
    payload = {
        "generated_at_utc": utc_now(),
        "output_root": str(OUT_ROOT.relative_to(ROOT)),
        "teacher_confidence_threshold": args.teacher_confidence_threshold,
        "pointwise_error_threshold": args.pointwise_error_threshold,
        "stable_good_error_threshold": args.stable_good_error_threshold,
        "residual_semantic_positive_ratio_by_split": {k: v["residual_semantic_positive_ratio"] for k, v in summaries.items()},
        "residual_identity_positive_ratio_by_split": {k: v["residual_identity_positive_ratio"] for k, v in summaries.items()},
        "stable_suppress_ratio_by_split": {k: v["stable_suppress_ratio"] for k, v in summaries.items()},
        "gate_positive_ratio_by_split": {k: v["gate_positive_ratio"] for k, v in summaries.items()},
        "split_summaries": summaries,
        "residual_utility_target_ready": bool(ready),
        "exact_blocker": "none" if ready else "residual utility positives or stable suppress negatives are empty under confidence/error thresholds",
        "leakage_safe": True,
    }
    dump_json(REPORT, payload)
    write_doc(
        DOC,
        "STWM OSTF V34.4 Residual Utility Target Build",
        payload,
        [
            "output_root",
            "residual_semantic_positive_ratio_by_split",
            "residual_identity_positive_ratio_by_split",
            "stable_suppress_ratio_by_split",
            "teacher_confidence_threshold",
            "pointwise_error_threshold",
            "residual_utility_target_ready",
            "exact_blocker",
            "leakage_safe",
        ],
    )
    print(REPORT.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
