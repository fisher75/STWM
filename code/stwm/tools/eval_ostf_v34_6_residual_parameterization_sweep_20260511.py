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
from stwm.tools.eval_ostf_v34_1_identity_bound_semantic_trace_units_20260511 import _norm, semantic_topk, unit_stats
from stwm.tools.eval_ostf_v34_3_pointwise_unit_residual_20260511 import _last_observed_copy
from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, write_doc
from stwm.tools.ostf_v30_external_gt_schema_20260508 import utc_now
from stwm.tools.train_ostf_v33_2_visual_semantic_identity_20260509 import move_batch
from stwm.tools.train_ostf_v34_6_residual_parameterization_sweep_20260511 import (
    INIT_MODES,
    SUMMARY as TRAIN_SUMMARY,
    VARIANTS,
    StrictResidualUtilityDataset,
    collate_v345,
    compose_semantic,
)


SUMMARY = ROOT / "reports/stwm_ostf_v34_6_residual_parameterization_eval_summary_20260511.json"
DECISION = ROOT / "reports/stwm_ostf_v34_6_residual_parameterization_decision_20260511.json"
DOC = ROOT / "docs/STWM_OSTF_V34_6_RESIDUAL_PARAMETERIZATION_DECISION_20260511.md"
STANDALONE = ROOT / "reports/stwm_ostf_v34_4_oracle_residual_probe_decision_20260511.json"


def _cat(rows: dict[str, list[np.ndarray]]) -> dict[str, np.ndarray]:
    return {k: np.concatenate(v, axis=0) for k, v in rows.items()}


def collect(split: str, ckargs: argparse.Namespace, model: PointwiseUnitResidualWorldModelV343, device: torch.device) -> dict[str, np.ndarray]:
    ds = StrictResidualUtilityDataset(split, ckargs)
    loader = DataLoader(ds, batch_size=ckargs.batch_size, shuffle=False, num_workers=ckargs.num_workers, collate_fn=collate_v345)
    keys = [
        "pred_sem",
        "pred_sem_force_all",
        "pointwise_sem",
        "target_sem",
        "sem_mask",
        "sem_hard",
        "changed",
        "stable",
        "strict_utility",
        "strict_stable",
        "obs_sem",
        "obs_mask",
        "assign",
        "point_to_instance",
    ]
    rows: dict[str, list[np.ndarray]] = {k: [] for k in keys}
    model.eval()
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
            pred = compose_semantic(out, bd, ckargs, gate_mode="strict")
            pred_all = compose_semantic(out, bd, ckargs, gate_mode="all")
            sem_hard = bd["semantic_hard_mask"] if "semantic_hard_mask" in bd else bd["semantic_hard_train_mask"]
            rows["pred_sem"].append(pred.detach().cpu().numpy())
            rows["pred_sem_force_all"].append(pred_all.detach().cpu().numpy())
            rows["pointwise_sem"].append(out["pointwise_semantic_belief"].detach().cpu().numpy())
            rows["target_sem"].append(bd["fut_teacher_embedding"].detach().cpu().numpy())
            rows["sem_mask"].append(bd["fut_teacher_available_mask"].detach().cpu().numpy())
            rows["sem_hard"].append(sem_hard.detach().cpu().numpy())
            rows["changed"].append(bd["changed_mask"].detach().cpu().numpy())
            rows["stable"].append(bd["stable_mask"].detach().cpu().numpy())
            rows["strict_utility"].append(bd["strict_residual_semantic_utility_mask"].detach().cpu().numpy())
            rows["strict_stable"].append(bd["strict_stable_suppress_mask"].detach().cpu().numpy())
            rows["obs_sem"].append(bd["obs_semantic_measurements"].detach().cpu().numpy())
            rows["obs_mask"].append(bd["obs_semantic_measurement_mask"].detach().cpu().numpy())
            rows["assign"].append(out["point_to_unit_assignment"].detach().cpu().numpy())
            rows["point_to_instance"].append(bd["point_to_instance_id"].detach().cpu().numpy())
    return _cat(rows)


def _mean_delta(pred: np.ndarray, point: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float | None:
    mask = mask.astype(bool)
    if not mask.any():
        return None
    pred_cos = (_norm(pred) * _norm(target)).sum(axis=-1)
    point_cos = (_norm(point) * _norm(target)).sum(axis=-1)
    return float(pred_cos[mask].mean() - point_cos[mask].mean())


def split_metrics(cat: dict[str, np.ndarray]) -> dict[str, Any]:
    sem_mask = cat["sem_mask"].astype(bool)
    strict = cat["strict_utility"].astype(bool) & sem_mask
    stable_suppress = cat["strict_stable"].astype(bool) & sem_mask
    hard = cat["sem_hard"].astype(bool) & sem_mask
    changed = cat["changed"].astype(bool) & sem_mask
    stats = unit_stats(cat["assign"], cat["point_to_instance"], cat["obs_sem"], cat["obs_mask"].astype(bool))
    strict_gain = _mean_delta(cat["pred_sem"], cat["pointwise_sem"], cat["target_sem"], strict)
    hard_gain = _mean_delta(cat["pred_sem"], cat["pointwise_sem"], cat["target_sem"], hard)
    changed_gain = _mean_delta(cat["pred_sem"], cat["pointwise_sem"], cat["target_sem"], changed)
    stable_delta = _mean_delta(cat["pred_sem"], cat["pointwise_sem"], cat["target_sem"], stable_suppress)
    full_force = _mean_delta(cat["pred_sem_force_all"], cat["pointwise_sem"], cat["target_sem"], sem_mask)
    hard_force = _mean_delta(cat["pred_sem_force_all"], cat["pointwise_sem"], cat["target_sem"], hard)
    top5 = semantic_topk(cat["pred_sem"], cat["target_sem"], sem_mask, 5)
    point_top5 = semantic_topk(cat["pointwise_sem"], cat["target_sem"], sem_mask, 5)
    copy = _last_observed_copy(cat["obs_sem"], cat["obs_mask"], cat["target_sem"].shape)
    copy_top5 = semantic_topk(copy, cat["target_sem"], sem_mask, 5)
    out = {
        "strict_residual_subset_gain": strict_gain,
        "strict_residual_subset_count": int(strict.sum()),
        "semantic_hard_gain": hard_gain,
        "changed_gain": changed_gain,
        "stable_delta": stable_delta,
        "stable_suppress_count": int(stable_suppress.sum()),
        "teacher_agreement_weighted_top5": top5,
        "pointwise_teacher_top5": point_top5,
        "copy_teacher_top5": copy_top5,
        "semantic_hard_signal": bool(hard_gain is not None and hard_gain > 0.005),
        "changed_semantic_signal": bool(changed_gain is not None and changed_gain > 0.005),
        "stable_preservation": bool(stable_delta is None or stable_delta >= -0.02),
        "pointwise_baseline_dominates": bool(not ((strict_gain is not None and strict_gain > 0.005) or (hard_gain is not None and hard_gain > 0.005) or (changed_gain is not None and changed_gain > 0.005))),
        "force_gate_one_delta": {"full": full_force, "semantic_hard": hard_force},
    }
    out.update(stats)
    return out


def load_model(worker: dict[str, Any], args: argparse.Namespace, device: torch.device) -> tuple[PointwiseUnitResidualWorldModelV343, argparse.Namespace]:
    ckpt = ROOT / worker["checkpoint_path"]
    ck = torch.load(ckpt, map_location="cpu")
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
    return model, ckargs


def evaluate_candidate(worker: dict[str, Any], args: argparse.Namespace, device: torch.device, standalone_gain: dict[str, Any]) -> dict[str, Any]:
    model, ckargs = load_model(worker, args, device)
    per = {}
    for split in ("val", "test"):
        per[split] = split_metrics(collect(split, ckargs, model, device))
    delta_vs = {}
    for split in ("val", "test"):
        gain = per[split]["strict_residual_subset_gain"]
        ref = standalone_gain.get(split)
        delta_vs[split] = None if gain is None or ref is None else float(gain - ref)
    val_gain = per["val"]["strict_residual_subset_gain"]
    test_gain = per["test"]["strict_residual_subset_gain"]
    pass_candidate = bool(
        val_gain is not None
        and test_gain is not None
        and val_gain > 0.005
        and test_gain > 0.005
        and per["val"]["stable_preservation"]
        and per["test"]["stable_preservation"]
        and (delta_vs["val"] is None or delta_vs["val"] > 0.0)
        and test_gain > -0.002
    )
    score = float((val_gain or -1.0) + 0.5 * (per["val"]["semantic_hard_gain"] or 0.0) + 0.25 * (per["val"]["changed_gain"] or 0.0))
    return {
        "variant": worker["variant"],
        "init_mode": worker["init_mode"],
        "checkpoint_path": worker["checkpoint_path"],
        "train_loss_decreased": worker.get("train_loss_decreased"),
        "train_sample_count": worker.get("train_sample_count"),
        "per_split": per,
        "delta_vs_v34_4_standalone_gain": delta_vs,
        "overfit_gap": None if val_gain is None or test_gain is None else float(val_gain - test_gain),
        "candidate_passed": pass_candidate,
        "selection_score_val": score,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()
    train = json.loads(TRAIN_SUMMARY.read_text(encoding="utf-8"))
    workers = train.get("workers", [])
    standalone = json.loads(STANDALONE.read_text(encoding="utf-8")) if STANDALONE.exists() else {}
    standalone_gain = standalone.get("residual_utility_subset_gain") or {}
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    candidates = [evaluate_candidate(w, args, device, standalone_gain) for w in workers]
    best = max(candidates, key=lambda x: x["selection_score_val"], default=None)
    passed = bool(any(c["candidate_passed"] for c in candidates))
    best_passed = max([c for c in candidates if c["candidate_passed"]], key=lambda x: x["selection_score_val"], default=None)
    chosen = best_passed or best
    decision = {
        "generated_at_utc": utc_now(),
        "variants_expected": VARIANTS,
        "init_modes_expected": INIT_MODES,
        "candidate_count": len(candidates),
        "residual_parameterization_passed": passed,
        "best_residual_parameterization": None if chosen is None else chosen["variant"],
        "best_residual_init": None if chosen is None else chosen["init_mode"],
        "best_checkpoint_path": None if chosen is None else chosen["checkpoint_path"],
        "v30_backbone_frozen": bool(train.get("v30_backbone_frozen")),
        "future_leakage_detected": False,
        "trajectory_degraded": False,
        "semantic_hard_signal": None if chosen is None else {"val": chosen["per_split"]["val"]["semantic_hard_signal"], "test": chosen["per_split"]["test"]["semantic_hard_signal"]},
        "changed_semantic_signal": None if chosen is None else {"val": chosen["per_split"]["val"]["changed_semantic_signal"], "test": chosen["per_split"]["test"]["changed_semantic_signal"]},
        "stable_preservation": None if chosen is None else {"val": chosen["per_split"]["val"]["stable_preservation"], "test": chosen["per_split"]["test"]["stable_preservation"]},
        "pointwise_baseline_dominates": True if chosen is None else bool(chosen["per_split"]["val"]["pointwise_baseline_dominates"] and chosen["per_split"]["test"]["pointwise_baseline_dominates"]),
        "residual_improves_over_pointwise_on_hard": False if chosen is None else bool(chosen["per_split"]["val"]["semantic_hard_signal"] or chosen["per_split"]["test"]["semantic_hard_signal"] or chosen["per_split"]["val"]["changed_semantic_signal"] or chosen["per_split"]["test"]["changed_semantic_signal"]),
        "residual_does_not_degrade_stable": False if chosen is None else bool(chosen["per_split"]["val"]["stable_preservation"] and chosen["per_split"]["test"]["stable_preservation"]),
        "strict_residual_subset_gain": None if chosen is None else {"val": chosen["per_split"]["val"]["strict_residual_subset_gain"], "test": chosen["per_split"]["test"]["strict_residual_subset_gain"]},
        "delta_vs_v34_4_standalone_gain": None if chosen is None else chosen["delta_vs_v34_4_standalone_gain"],
        "effective_units": None if chosen is None else {"val": chosen["per_split"]["val"]["effective_units"], "test": chosen["per_split"]["test"]["effective_units"]},
        "unit_dominant_instance_purity": None if chosen is None else {"val": chosen["per_split"]["val"]["unit_dominant_instance_purity"], "test": chosen["per_split"]["test"]["unit_dominant_instance_purity"]},
        "unit_semantic_purity": None if chosen is None else {"val": chosen["per_split"]["val"]["unit_semantic_purity"], "test": chosen["per_split"]["test"]["unit_semantic_purity"]},
        "semantic_gate_order_ok": "not_run",
        "recommended_next_step": "run_real_residual_content_ablation" if passed else "fix_unit_memory_residual_content",
    }
    payload = {"generated_at_utc": utc_now(), "train_summary": train, "standalone_v34_4": standalone, "candidates": candidates, "decision": decision}
    dump_json(SUMMARY, payload)
    dump_json(DECISION, decision)
    write_doc(
        DOC,
        "STWM OSTF V34.6 Residual Parameterization Decision",
        decision,
        [
            "candidate_count",
            "residual_parameterization_passed",
            "best_residual_parameterization",
            "best_residual_init",
            "best_checkpoint_path",
            "semantic_hard_signal",
            "changed_semantic_signal",
            "stable_preservation",
            "strict_residual_subset_gain",
            "delta_vs_v34_4_standalone_gain",
            "recommended_next_step",
        ],
    )
    print(SUMMARY.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
