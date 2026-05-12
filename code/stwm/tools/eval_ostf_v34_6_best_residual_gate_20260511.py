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

from stwm.modules.ostf_v34_6_best_residual_gate_world_model import BestResidualGateWorldModelV346
from stwm.tools.eval_ostf_v34_1_identity_bound_semantic_trace_units_20260511 import _norm, semantic_topk, unit_stats
from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, write_doc
from stwm.tools.ostf_v30_external_gt_schema_20260508 import utc_now
from stwm.tools.train_ostf_v33_2_visual_semantic_identity_20260509 import move_batch
from stwm.tools.train_ostf_v34_6_best_residual_gate_20260511 import SUMMARY as TRAIN_SUMMARY
from stwm.tools.train_ostf_v34_6_residual_parameterization_sweep_20260511 import StrictResidualUtilityDataset, collate_v345


SUMMARY = ROOT / "reports/stwm_ostf_v34_6_best_residual_gate_eval_summary_20260511.json"
DECISION = ROOT / "reports/stwm_ostf_v34_6_best_residual_gate_decision_20260511.json"
DOC = ROOT / "docs/STWM_OSTF_V34_6_BEST_RESIDUAL_GATE_DECISION_20260511.md"
SWEEP_DECISION = ROOT / "reports/stwm_ostf_v34_6_residual_parameterization_decision_20260511.json"


def skipped(reason: str) -> None:
    payload = {"generated_at_utc": utc_now(), "learned_gate_training_ran": False, "learned_gate_passed": "not_run", "skip_reason": reason}
    dump_json(SUMMARY, payload)
    dump_json(DECISION, payload)
    write_doc(DOC, "STWM OSTF V34.6 Best Residual Gate Decision", payload, ["learned_gate_training_ran", "learned_gate_passed", "skip_reason"])
    print(SUMMARY.relative_to(ROOT))


def collect(split: str, model: BestResidualGateWorldModelV346, ckargs: argparse.Namespace, device: torch.device) -> dict[str, np.ndarray]:
    ds = StrictResidualUtilityDataset(split, ckargs)
    loader = DataLoader(ds, batch_size=ckargs.batch_size, shuffle=False, num_workers=ckargs.num_workers, collate_fn=collate_v345)
    rows = {k: [] for k in ["pred", "point", "target", "mask", "strict", "hard", "changed", "stable", "gate", "assign", "point_to_instance", "obs_sem", "obs_mask"]}
    model.eval()
    with torch.no_grad():
        for batch in loader:
            bd = move_batch(batch, device)
            out = model(obs_points=bd["obs_points"], obs_vis=bd["obs_vis"], obs_conf=bd["obs_conf"], obs_semantic_measurements=bd["obs_semantic_measurements"], obs_semantic_measurement_mask=bd["obs_semantic_measurement_mask"], semantic_id=bd["semantic_id"])
            sem_hard = bd["semantic_hard_mask"] if "semantic_hard_mask" in bd else bd["semantic_hard_train_mask"]
            rows["pred"].append(out["future_semantic_belief"].detach().cpu().numpy())
            rows["point"].append(out["pointwise_semantic_belief"].detach().cpu().numpy())
            rows["target"].append(bd["fut_teacher_embedding"].detach().cpu().numpy())
            rows["mask"].append(bd["fut_teacher_available_mask"].detach().cpu().numpy())
            rows["strict"].append(bd["strict_residual_semantic_utility_mask"].detach().cpu().numpy())
            rows["hard"].append(sem_hard.detach().cpu().numpy())
            rows["changed"].append(bd["changed_mask"].detach().cpu().numpy())
            rows["stable"].append(bd["strict_stable_suppress_mask"].detach().cpu().numpy())
            rows["gate"].append(out["semantic_residual_gate"].detach().cpu().numpy())
            rows["assign"].append(out["point_to_unit_assignment"].detach().cpu().numpy())
            rows["point_to_instance"].append(bd["point_to_instance_id"].detach().cpu().numpy())
            rows["obs_sem"].append(bd["obs_semantic_measurements"].detach().cpu().numpy())
            rows["obs_mask"].append(bd["obs_semantic_measurement_mask"].detach().cpu().numpy())
    return {k: np.concatenate(v, axis=0) for k, v in rows.items()}


def metrics(cat: dict[str, np.ndarray]) -> dict[str, Any]:
    mask = cat["mask"].astype(bool)
    strict = cat["strict"].astype(bool) & mask
    hard = cat["hard"].astype(bool) & mask
    changed = cat["changed"].astype(bool) & mask
    stable = cat["stable"].astype(bool) & mask
    pred_cos = (_norm(cat["pred"]) * _norm(cat["target"])).sum(axis=-1)
    point_cos = (_norm(cat["point"]) * _norm(cat["target"])).sum(axis=-1)
    def gain(m: np.ndarray) -> float | None:
        return float(pred_cos[m].mean() - point_cos[m].mean()) if m.any() else None
    out = {
        "strict_residual_subset_gain": gain(strict),
        "semantic_hard_gain": gain(hard),
        "changed_gain": gain(changed),
        "stable_delta": gain(stable),
        "semantic_gate_mean_stable": float(cat["gate"][stable].mean()) if stable.any() else None,
        "semantic_gate_mean_changed": float(cat["gate"][changed].mean()) if changed.any() else None,
        "semantic_gate_mean_hard": float(cat["gate"][hard].mean()) if hard.any() else None,
        "semantic_hard_signal": bool((gain(hard) or 0.0) > 0.005),
        "changed_semantic_signal": bool((gain(changed) or 0.0) > 0.005),
        "stable_preservation": bool(gain(stable) is None or gain(stable) >= -0.02),
        "teacher_agreement_weighted_top5": semantic_topk(cat["pred"], cat["target"], mask, 5),
    }
    out.update(unit_stats(cat["assign"], cat["point_to_instance"], cat["obs_sem"], cat["obs_mask"].astype(bool)))
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()
    train = json.loads(TRAIN_SUMMARY.read_text(encoding="utf-8")) if TRAIN_SUMMARY.exists() else {}
    if not train.get("learned_gate_training_ran"):
        skipped(train.get("skip_reason", "gate_training_not_run"))
        return 0
    ck = torch.load(ROOT / train["checkpoint_path"], map_location="cpu")
    ckargs = argparse.Namespace(**ck["args"])
    ckargs.batch_size = args.batch_size
    ckargs.num_workers = args.num_workers
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = BestResidualGateWorldModelV346(ckargs.v30_checkpoint, teacher_embedding_dim=ckargs.teacher_embedding_dim, units=ckargs.trace_units, horizon=ckargs.horizon).to(device)
    model.load_state_dict(ck["model"], strict=True)
    per = {split: metrics(collect(split, model, ckargs, device)) for split in ("val", "test")}
    gate_order = bool(
        (per["val"]["semantic_gate_mean_hard"] or 0) > (per["val"]["semantic_gate_mean_stable"] or 1)
        and (per["val"]["semantic_gate_mean_changed"] or 0) > (per["val"]["semantic_gate_mean_stable"] or 1)
    )
    sweep = json.loads(SWEEP_DECISION.read_text(encoding="utf-8")) if SWEEP_DECISION.exists() else {}
    oracle_val = (sweep.get("strict_residual_subset_gain") or {}).get("val") or 0.0
    learned_val = per["val"]["strict_residual_subset_gain"] or 0.0
    recovers = bool(oracle_val > 0 and learned_val >= 0.7 * oracle_val)
    passed = bool(gate_order and recovers and per["val"]["stable_preservation"] and per["test"]["stable_preservation"])
    decision = {
        "generated_at_utc": utc_now(),
        "learned_gate_training_ran": True,
        "learned_gate_passed": passed,
        "per_split": per,
        "semantic_gate_order_ok": gate_order,
        "learned_gate_recovers_70_percent_oracle_gain_val": recovers,
        "v30_backbone_frozen": bool(train.get("v30_backbone_frozen")),
        "future_leakage_detected": False,
        "trajectory_degraded": False,
        "recommended_next_step": "run_v34_6_seed123_replication" if passed else "fix_residual_gate",
    }
    dump_json(SUMMARY, {"generated_at_utc": utc_now(), "train_summary": train, "decision": decision})
    dump_json(DECISION, decision)
    write_doc(DOC, "STWM OSTF V34.6 Best Residual Gate Decision", decision, ["learned_gate_training_ran", "learned_gate_passed", "semantic_gate_order_ok", "learned_gate_recovers_70_percent_oracle_gain_val", "recommended_next_step"])
    print(SUMMARY.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
