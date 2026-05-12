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
from stwm.tools.eval_ostf_v34_1_identity_bound_semantic_trace_units_20260511 import _norm, semantic_topk
from stwm.tools.eval_ostf_v34_6_residual_parameterization_sweep_20260511 import DECISION as SWEEP_DECISION
from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, write_doc
from stwm.tools.ostf_v30_external_gt_schema_20260508 import utc_now
from stwm.tools.train_ostf_v33_2_visual_semantic_identity_20260509 import move_batch
from stwm.tools.train_ostf_v34_6_residual_parameterization_sweep_20260511 import StrictResidualUtilityDataset, collate_v345, compose_semantic


OUT = ROOT / "reports/stwm_ostf_v34_6_real_residual_content_ablation_20260511.json"
DOC = ROOT / "docs/STWM_OSTF_V34_6_REAL_RESIDUAL_CONTENT_ABLATION_20260511.md"


MODES = [
    "normal",
    "random_unit_residual",
    "residual_without_unit_memory",
    "residual_with_shuffled_unit_assignment",
    "residual_with_shuffled_semantic_measurements",
    "residual_with_zero_semantic_measurements",
    "residual_with_pointwise_only_base",
    "oracle_target_upper_bound",
]


def load_model(args: argparse.Namespace, device: torch.device) -> tuple[PointwiseUnitResidualWorldModelV343 | None, argparse.Namespace | None, dict[str, Any]]:
    decision = json.loads(SWEEP_DECISION.read_text(encoding="utf-8")) if SWEEP_DECISION.exists() else {}
    ckpt_rel = decision.get("best_checkpoint_path")
    if not ckpt_rel:
        return None, None, decision
    ckpt = ROOT / ckpt_rel
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
    model.eval()
    return model, ckargs, decision


def final_for_mode(mode: str, out: dict[str, torch.Tensor], bd: dict[str, torch.Tensor], ckargs: argparse.Namespace) -> torch.Tensor:
    if mode == "oracle_target_upper_bound":
        return torch.nn.functional.normalize(torch.nan_to_num(bd["fut_teacher_embedding"].float()), dim=-1)
    if mode == "residual_with_pointwise_only_base":
        return out["pointwise_semantic_belief"]
    if mode == "random_unit_residual":
        hacked = dict(out)
        hacked["unit_semantic_residual"] = torch.nn.functional.normalize(torch.randn_like(out["unit_semantic_residual"]), dim=-1)
        return compose_semantic(hacked, bd, ckargs, gate_mode="strict")
    if mode == "residual_without_unit_memory":
        hacked = dict(out)
        hacked["unit_semantic_residual"] = torch.zeros_like(out["unit_semantic_residual"])
        return compose_semantic(hacked, bd, ckargs, gate_mode="strict")
    return compose_semantic(out, bd, ckargs, gate_mode="strict")


def collect(split: str, model: PointwiseUnitResidualWorldModelV343, ckargs: argparse.Namespace, device: torch.device) -> dict[str, dict[str, list[np.ndarray]]]:
    ds = StrictResidualUtilityDataset(split, ckargs)
    loader = DataLoader(ds, batch_size=ckargs.batch_size, shuffle=False, num_workers=ckargs.num_workers, collate_fn=collate_v345)
    rows: dict[str, dict[str, list[np.ndarray]]] = {m: {"pred": [], "point": [], "target": [], "mask": [], "strict": [], "hard": [], "changed": [], "stable": []} for m in MODES}
    with torch.no_grad():
        for batch in loader:
            bd = move_batch(batch, device)
            sem_hard = bd["semantic_hard_mask"] if "semantic_hard_mask" in bd else bd["semantic_hard_train_mask"]
            mode_outs = {}
            base_inputs = {
                "obs_points": bd["obs_points"],
                "obs_vis": bd["obs_vis"],
                "obs_conf": bd["obs_conf"],
                "obs_semantic_measurements": bd["obs_semantic_measurements"],
                "obs_semantic_measurement_mask": bd["obs_semantic_measurement_mask"],
                "semantic_id": bd["semantic_id"],
            }
            mode_outs["normal"] = model(**base_inputs, intervention="force_gate_zero")
            mode_outs["residual_without_unit_memory"] = model(**base_inputs, intervention="zero_unit_residual")
            mode_outs["residual_with_shuffled_unit_assignment"] = model(**base_inputs, intervention="permute_unit_assignment")
            mode_outs["residual_with_zero_semantic_measurements"] = model(**base_inputs, intervention="zero_observed_semantic_measurements")
            shuffled_inputs = dict(base_inputs)
            shuffled_inputs["obs_semantic_measurements"] = torch.roll(bd["obs_semantic_measurements"], shifts=1, dims=1)
            shuffled_inputs["obs_semantic_measurement_mask"] = torch.roll(bd["obs_semantic_measurement_mask"], shifts=1, dims=1)
            mode_outs["residual_with_shuffled_semantic_measurements"] = model(**shuffled_inputs, intervention="force_gate_zero")
            mode_outs["random_unit_residual"] = mode_outs["normal"]
            mode_outs["residual_with_pointwise_only_base"] = mode_outs["normal"]
            mode_outs["oracle_target_upper_bound"] = mode_outs["normal"]
            for mode in MODES:
                pred = final_for_mode(mode, mode_outs[mode], bd, ckargs)
                rows[mode]["pred"].append(pred.detach().cpu().numpy())
                rows[mode]["point"].append(mode_outs["normal"]["pointwise_semantic_belief"].detach().cpu().numpy())
                rows[mode]["target"].append(bd["fut_teacher_embedding"].detach().cpu().numpy())
                rows[mode]["mask"].append(bd["fut_teacher_available_mask"].detach().cpu().numpy())
                rows[mode]["strict"].append(bd["strict_residual_semantic_utility_mask"].detach().cpu().numpy())
                rows[mode]["hard"].append(sem_hard.detach().cpu().numpy())
                rows[mode]["changed"].append(bd["changed_mask"].detach().cpu().numpy())
                rows[mode]["stable"].append(bd["strict_stable_suppress_mask"].detach().cpu().numpy())
    return rows


def metrics_from_rows(rows: dict[str, list[np.ndarray]]) -> dict[str, Any]:
    cat = {k: np.concatenate(v, axis=0) for k, v in rows.items()}
    mask = cat["mask"].astype(bool)
    strict = cat["strict"].astype(bool) & mask
    hard = cat["hard"].astype(bool) & mask
    changed = cat["changed"].astype(bool) & mask
    stable = cat["stable"].astype(bool) & mask
    pred_cos = (_norm(cat["pred"]) * _norm(cat["target"])).sum(axis=-1)
    point_cos = (_norm(cat["point"]) * _norm(cat["target"])).sum(axis=-1)
    def gain(m: np.ndarray) -> float | None:
        return float(pred_cos[m].mean() - point_cos[m].mean()) if m.any() else None
    return {
        "strict_residual_subset_gain": gain(strict),
        "semantic_hard_gain": gain(hard),
        "changed_gain": gain(changed),
        "stable_delta": gain(stable),
        "teacher_agreement_weighted_top5": semantic_topk(cat["pred"], cat["target"], mask, 5),
    }


def split_eval(split: str, model: PointwiseUnitResidualWorldModelV343, ckargs: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    rows = collect(split, model, ckargs, device)
    return {mode: metrics_from_rows(mode_rows) for mode, mode_rows in rows.items()}


def delta(a: float | None, b: float | None) -> float | None:
    return None if a is None or b is None else float(a - b)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model, ckargs, decision = load_model(args, device)
    if model is None or ckargs is None:
        payload = {
            "generated_at_utc": utc_now(),
            "real_residual_content_ablation_done": False,
            "exact_blockers": ["missing_best_residual_checkpoint"],
            "unit_memory_load_bearing_on_residual": "not_run",
            "semantic_measurements_load_bearing_on_residual": "not_run",
            "assignment_load_bearing_on_residual": "not_run",
            "best_variant_not_random_artifact": "not_run",
        }
        dump_json(OUT, payload)
        write_doc(DOC, "STWM OSTF V34.6 Real Residual Content Ablation", payload, ["real_residual_content_ablation_done", "exact_blockers"])
        print(OUT.relative_to(ROOT))
        return 0
    per = {split: split_eval(split, model, ckargs, device) for split in ("val", "test")}
    normal = {s: per[s]["normal"] for s in ("val", "test")}
    no_mem = {s: per[s]["residual_without_unit_memory"] for s in ("val", "test")}
    shuffle_assign = {s: per[s]["residual_with_shuffled_unit_assignment"] for s in ("val", "test")}
    zero_sem = {s: per[s]["residual_with_zero_semantic_measurements"] for s in ("val", "test")}
    shuffle_sem = {s: per[s]["residual_with_shuffled_semantic_measurements"] for s in ("val", "test")}
    random_res = {s: per[s]["random_unit_residual"] for s in ("val", "test")}
    unit_delta = {s: delta(normal[s]["strict_residual_subset_gain"], no_mem[s]["strict_residual_subset_gain"]) for s in ("val", "test")}
    assign_delta = {s: delta(normal[s]["strict_residual_subset_gain"], shuffle_assign[s]["strict_residual_subset_gain"]) for s in ("val", "test")}
    sem_delta = {s: delta(normal[s]["strict_residual_subset_gain"], max(zero_sem[s]["strict_residual_subset_gain"] or -9, shuffle_sem[s]["strict_residual_subset_gain"] or -9)) for s in ("val", "test")}
    random_delta = {s: delta(normal[s]["strict_residual_subset_gain"], random_res[s]["strict_residual_subset_gain"]) for s in ("val", "test")}
    unit_lb = bool(all(v is not None and v > 0.002 for v in unit_delta.values()))
    sem_lb = bool(all(v is not None and v > 0.002 for v in sem_delta.values()))
    assign_lb = bool(all(v is not None and v > 0.002 for v in assign_delta.values()))
    not_random = bool(all(v is not None and v > 0.002 for v in random_delta.values()))
    payload = {
        "generated_at_utc": utc_now(),
        "real_residual_content_ablation_done": True,
        "best_residual_parameterization": decision.get("best_residual_parameterization"),
        "best_residual_init": decision.get("best_residual_init"),
        "per_split": per,
        "unit_memory_load_bearing_on_residual": unit_lb,
        "semantic_measurements_load_bearing_on_residual": sem_lb,
        "assignment_load_bearing_on_residual": assign_lb,
        "best_variant_not_random_artifact": not_random,
        "strict_residual_subset_gain_delta": {
            "unit_memory_vs_no_memory": unit_delta,
            "semantic_measurement_vs_zero_or_shuffle": sem_delta,
            "assignment_vs_shuffled": assign_delta,
            "normal_vs_random": random_delta,
        },
        "semantic_hard_gain_delta": {
            s: delta(normal[s]["semantic_hard_gain"], no_mem[s]["semantic_hard_gain"]) for s in ("val", "test")
        },
        "changed_gain_delta": {
            s: delta(normal[s]["changed_gain"], no_mem[s]["changed_gain"]) for s in ("val", "test")
        },
        "stable_delta": {s: normal[s]["stable_delta"] for s in ("val", "test")},
    }
    dump_json(OUT, payload)
    write_doc(DOC, "STWM OSTF V34.6 Real Residual Content Ablation", payload, ["real_residual_content_ablation_done", "best_residual_parameterization", "best_residual_init", "unit_memory_load_bearing_on_residual", "semantic_measurements_load_bearing_on_residual", "assignment_load_bearing_on_residual", "best_variant_not_random_artifact", "strict_residual_subset_gain_delta"])
    print(OUT.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
