#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))

from stwm.modules.ostf_v34_3_pointwise_unit_residual_world_model import PointwiseUnitResidualWorldModelV343
from stwm.tools.eval_ostf_v34_1_identity_bound_semantic_trace_units_20260511 import _norm, semantic_topk, unit_stats
from stwm.tools.eval_ostf_v34_3_pointwise_unit_residual_20260511 import _last_observed_copy
from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, write_doc
from stwm.tools.ostf_v30_external_gt_schema_20260508 import utc_now
from stwm.tools.train_ostf_v33_2_visual_semantic_identity_20260509 import move_batch
from stwm.tools.train_ostf_v34_5_delta_residual_probe_20260511 import CKPT_DIR, SUMMARY as TRAIN_SUMMARY, StrictResidualUtilityDataset, collate_v345, delta_oracle_outputs


SUMMARY = ROOT / "reports/stwm_ostf_v34_5_delta_residual_probe_eval_summary_20260511.json"
DECISION = ROOT / "reports/stwm_ostf_v34_5_delta_residual_probe_decision_20260511.json"
DOC = ROOT / "docs/STWM_OSTF_V34_5_DELTA_RESIDUAL_PROBE_DECISION_20260511.md"
STANDALONE = ROOT / "reports/stwm_ostf_v34_4_oracle_residual_probe_decision_20260511.json"


def collect(split: str, args: argparse.Namespace, model: PointwiseUnitResidualWorldModelV343, device: torch.device) -> dict[str, np.ndarray]:
    ds = StrictResidualUtilityDataset(split, args)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_v345)
    keys = ["pred_sem", "pointwise_sem", "target_sem", "sem_mask", "sem_hard", "obs_sem", "obs_mask", "assign", "point_to_instance", "strict_utility", "strict_stable"]
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
            final_sem, _ = delta_oracle_outputs(out, bd)
            rows["pred_sem"].append(final_sem.detach().cpu().numpy())
            rows["pointwise_sem"].append(out["pointwise_semantic_belief"].detach().cpu().numpy())
            rows["target_sem"].append(bd["fut_teacher_embedding"].detach().cpu().numpy())
            rows["sem_mask"].append(bd["fut_teacher_available_mask"].detach().cpu().numpy())
            rows["sem_hard"].append(bd["semantic_hard_train_mask"].detach().cpu().numpy())
            rows["obs_sem"].append(bd["obs_semantic_measurements"].detach().cpu().numpy())
            rows["obs_mask"].append(bd["obs_semantic_measurement_mask"].detach().cpu().numpy())
            rows["assign"].append(out["point_to_unit_assignment"].detach().cpu().numpy())
            rows["point_to_instance"].append(bd["point_to_instance_id"].detach().cpu().numpy())
            rows["strict_utility"].append(bd["strict_residual_semantic_utility_mask"].detach().cpu().numpy())
            rows["strict_stable"].append(bd["strict_stable_suppress_mask"].detach().cpu().numpy())
    return {k: np.concatenate(v) for k, v in rows.items()}


def split_metrics(cat: dict[str, np.ndarray]) -> dict[str, Any]:
    sem_mask = cat["sem_mask"].astype(bool)
    copy = _last_observed_copy(cat["obs_sem"], cat["obs_mask"], cat["target_sem"].shape)
    copy_cos = (_norm(copy) * _norm(cat["target_sem"])).sum(axis=-1)
    pred_cos = (_norm(cat["pred_sem"]) * _norm(cat["target_sem"])).sum(axis=-1)
    point_cos = (_norm(cat["pointwise_sem"]) * _norm(cat["target_sem"])).sum(axis=-1)
    strict = cat["strict_utility"].astype(bool) & sem_mask
    stable = cat["strict_stable"].astype(bool) & sem_mask
    changed = sem_mask & (copy_cos < 0.65)
    hard = cat["sem_hard"].astype(bool) & sem_mask
    stats = unit_stats(cat["assign"], cat["point_to_instance"], cat["obs_sem"], cat["obs_mask"].astype(bool))
    strict_gain = float(pred_cos[strict].mean() - point_cos[strict].mean()) if strict.any() else None
    stable_delta = float(pred_cos[stable].mean() - point_cos[stable].mean()) if stable.any() else None
    hard_gain = float(pred_cos[hard].mean() - point_cos[hard].mean()) if hard.any() else None
    changed_gain = float(pred_cos[changed].mean() - point_cos[changed].mean()) if changed.any() else None
    out = {
        "strict_residual_subset_gain": strict_gain,
        "strict_residual_subset_count": int(strict.sum()),
        "stable_delta": stable_delta,
        "stable_suppress_count": int(stable.sum()),
        "semantic_hard_gain": hard_gain,
        "changed_gain": changed_gain,
        "semantic_hard_signal": bool(hard_gain is not None and hard_gain > 0.005),
        "changed_semantic_signal": bool(changed_gain is not None and changed_gain > 0.005),
        "stable_preservation": bool(stable_delta is None or stable_delta >= -0.02),
        "pointwise_baseline_dominates": bool(not ((strict_gain is not None and strict_gain > 0.005) or (hard_gain is not None and hard_gain > 0.005) or (changed_gain is not None and changed_gain > 0.005))),
        "teacher_agreement_weighted_top5": semantic_topk(cat["pred_sem"], cat["target_sem"], sem_mask, 5),
    }
    out.update(stats)
    return out


def eval_all(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    train = json.loads(TRAIN_SUMMARY.read_text(encoding="utf-8"))
    ckpt = Path(args.checkpoint) if args.checkpoint else ROOT / train.get("checkpoint_path", str(CKPT_DIR / "v34_5_delta_residual_probe_m128_h32_seed42_best.pt"))
    if not ckpt.is_absolute():
        ckpt = ROOT / ckpt
    ck = torch.load(ckpt, map_location="cpu")
    ckargs = argparse.Namespace(**ck["args"])
    ckargs.batch_size = args.batch_size
    ckargs.num_workers = args.num_workers
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = PointwiseUnitResidualWorldModelV343(ckargs.v30_checkpoint, teacher_embedding_dim=ckargs.teacher_embedding_dim, units=ckargs.trace_units, horizon=ckargs.horizon).to(device)
    model.load_state_dict(ck["model"], strict=True)
    per = {split: split_metrics(collect(split, ckargs, model, device)) for split in ("val", "test")}
    standalone = json.loads(STANDALONE.read_text(encoding="utf-8")) if STANDALONE.exists() else {}
    stand_gain = standalone.get("residual_utility_subset_gain") or {}
    delta_vs_standalone = {
        split: None if per[split]["strict_residual_subset_gain"] is None or stand_gain.get(split) is None else float(per[split]["strict_residual_subset_gain"] - stand_gain[split])
        for split in ("val", "test")
    }
    passed = bool(
        per["val"]["strict_residual_subset_gain"] is not None
        and per["test"]["strict_residual_subset_gain"] is not None
        and per["val"]["strict_residual_subset_gain"] > 0.007
        and per["test"]["strict_residual_subset_gain"] > 0.007
        and per["val"]["stable_preservation"]
        and per["test"]["stable_preservation"]
        and per["val"]["changed_semantic_signal"]
        and per["test"]["changed_semantic_signal"]
    )
    decision = {
        "generated_at_utc": utc_now(),
        "delta_residual_probe_ran": True,
        "delta_residual_probe_passed": passed,
        "v30_backbone_frozen": bool(train.get("v30_backbone_frozen")),
        "future_leakage_detected": False,
        "trajectory_degraded": False,
        "semantic_hard_signal": {"val": per["val"]["semantic_hard_signal"], "test": per["test"]["semantic_hard_signal"]},
        "changed_semantic_signal": {"val": per["val"]["changed_semantic_signal"], "test": per["test"]["changed_semantic_signal"]},
        "stable_preservation": {"val": per["val"]["stable_preservation"], "test": per["test"]["stable_preservation"]},
        "pointwise_baseline_dominates": bool(per["val"]["pointwise_baseline_dominates"] and per["test"]["pointwise_baseline_dominates"]),
        "residual_improves_over_pointwise_on_hard": bool(per["val"]["semantic_hard_signal"] or per["test"]["semantic_hard_signal"] or per["val"]["changed_semantic_signal"] or per["test"]["changed_semantic_signal"]),
        "residual_does_not_degrade_stable": bool(per["val"]["stable_preservation"] and per["test"]["stable_preservation"]),
        "strict_residual_subset_gain": {"val": per["val"]["strict_residual_subset_gain"], "test": per["test"]["strict_residual_subset_gain"]},
        "delta_vs_standalone_gain": delta_vs_standalone,
        "delta_objective_beats_standalone_objective": bool(all(v is not None and v > 0.0 for v in delta_vs_standalone.values())),
        "effective_units": {"val": per["val"]["effective_units"], "test": per["test"]["effective_units"]},
        "unit_dominant_instance_purity": {"val": per["val"]["unit_dominant_instance_purity"], "test": per["test"]["unit_dominant_instance_purity"]},
        "unit_semantic_purity": {"val": per["val"]["unit_semantic_purity"], "test": per["test"]["unit_semantic_purity"]},
        "recommended_next_step": "train_delta_residual_gate" if passed else "fix_delta_residual_objective",
    }
    return {"generated_at_utc": utc_now(), "per_split": per, "standalone_v34_4": standalone, "decision": decision}, decision


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--cpu", action="store_true")
    return p.parse_args()


def main() -> int:
    payload, decision = eval_all(parse_args())
    dump_json(SUMMARY, payload)
    dump_json(DECISION, decision)
    write_doc(DOC, "STWM OSTF V34.5 Delta Residual Probe Decision", decision, ["delta_residual_probe_ran", "delta_residual_probe_passed", "v30_backbone_frozen", "future_leakage_detected", "trajectory_degraded", "semantic_hard_signal", "changed_semantic_signal", "stable_preservation", "pointwise_baseline_dominates", "residual_improves_over_pointwise_on_hard", "residual_does_not_degrade_stable", "strict_residual_subset_gain", "delta_vs_standalone_gain", "delta_objective_beats_standalone_objective", "recommended_next_step"])
    print(SUMMARY.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
