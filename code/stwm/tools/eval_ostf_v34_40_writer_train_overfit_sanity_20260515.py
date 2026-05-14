#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import setproctitle
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))

setproctitle.setproctitle("python")

import stwm.tools.train_eval_ostf_v34_36_predictability_filtered_unit_delta_writer_20260514 as v36
from stwm.tools.eval_ostf_v34_26_full_system_baseline_claim_boundary_benchmark_20260514 import load_v3425_readers
from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, write_doc
from stwm.tools.ostf_v30_external_gt_schema_20260508 import utc_now
from stwm.tools.train_eval_ostf_v34_31_raw_unit_delta_value_memory_20260514 import best_copy_topk, load_frozen_residual_model
from stwm.tools.train_eval_ostf_v34_40_prototype_conditioned_mixture_unit_delta_writer_20260515 import (
    TARGET_ROOT,
    build_codebook,
    load_trained_head,
)


REPORT = ROOT / "reports/stwm_ostf_v34_40_writer_train_overfit_sanity_20260515.json"
DOC = ROOT / "docs/STWM_OSTF_V34_40_WRITER_TRAIN_OVERFIT_SANITY_20260515.md"


def main() -> int:
    args = parse_args()
    v36.TARGET_ROOT = TARGET_ROOT
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    codebook, codebook_report = build_codebook(args)
    model, ckargs, _ = load_frozen_residual_model(args, device)
    head, _summary = load_trained_head(model, codebook, codebook_report, args, device)
    readers = load_v3425_readers(args, model, device)
    cache = v36.eval_split("train", model, ckargs, head, readers, args, device)
    rows = []
    method = "v34_36_predictable_oracle_mask_predictability_filtered_writer"
    for scale in args.eval_scales:
        m = cache["predictable_oracle_mask"][float(scale)]["methods"][method]
        rows.append(
            {
                "scale": float(scale),
                "stable": m["stable_preservation"],
                "train_gain_anchor": m["hard_changed_gain_vs_anchor"],
                "train_gain_pointwise": m["hard_changed_gain_vs_pointwise"],
            }
        )
    selected = max(rows, key=lambda r: float(r["train_gain_anchor"] or -1.0e9))
    pack = cache["predictable_oracle_mask"][float(selected["scale"])]
    metrics = pack["methods"][method]
    delta = pack["intervention_delta"]
    base = best_copy_topk(pack)
    train_overfits_prototype_mixture = bool(
        (metrics["hard_changed_gain_vs_anchor"] or -1.0) > 0.002
        and (delta["shuffle_assignment_delta"] or 0.0) > 0.002
        and (delta["zero_unit_memory_delta"] or 0.0) > 0.002
    )
    payload = {
        "generated_at_utc": utc_now(),
        "中文结论": "V34.40 writer train-overfit sanity 完成；检查 prototype-conditioned mixture writer 是否至少能在训练集形成 assignment-bound 正 correction。",
        "selected_train_config": selected,
        "train_overfits_prototype_mixture": train_overfits_prototype_mixture,
        "train_metrics": metrics,
        "train_intervention_delta": delta,
        "best_copy_topk_baseline_train": base,
        "recommended_next_step": "fix_prototype_mixture_capacity_or_assignment_binding" if not train_overfits_prototype_mixture else "fix_prototype_mixture_generalization",
    }
    dump_json(REPORT, payload)
    write_doc(
        DOC,
        "V34.40 writer train-overfit sanity 中文报告",
        payload,
        ["中文结论", "selected_train_config", "train_overfits_prototype_mixture", "train_intervention_delta", "recommended_next_step"],
    )
    print(f"已写出 V34.40 writer train-overfit sanity 报告: {REPORT.relative_to(ROOT)}", flush=True)
    print(f"train_overfits_prototype_mixture: {train_overfits_prototype_mixture}", flush=True)
    print(f"recommended_next_step: {payload['recommended_next_step']}", flush=True)
    return 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--teacher-embedding-dim", type=int, default=768)
    p.add_argument("--value-hidden-dim", type=int, default=256)
    p.add_argument("--reader-hidden-dim", type=int, default=192)
    p.add_argument("--prototype-count", type=int, default=64)
    p.add_argument("--prototype-temperature", type=float, default=0.7)
    p.add_argument("--max-residual-magnitude", type=float, default=0.35)
    p.add_argument("--target-kind", choices=["top1"], default="top1")
    p.add_argument("--eval-scales", type=float, nargs="+", default=[0.25, 0.5, 1.0, 2.0])
    p.add_argument("--cpu", action="store_true")
    p.set_defaults(learnable_codebook=False)
    return p.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
