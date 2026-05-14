#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import setproctitle
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))

setproctitle.setproctitle("python")

from stwm.modules.ostf_v34_34_cross_attention_unit_delta_writer import CrossAttentionUnitDeltaWriterV3434
from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, write_doc
from stwm.tools.ostf_v30_external_gt_schema_20260508 import utc_now
from stwm.tools.train_eval_ostf_v34_31_raw_unit_delta_value_memory_20260514 import best_copy_topk, load_frozen_residual_model
from stwm.tools.train_eval_ostf_v34_34_cross_attention_unit_delta_value_writer_20260514 import (
    CKPT_DIR,
    TensorUnitDeltaWriter,
    choose_best,
    eval_sweep_split,
)


REPORT = ROOT / "reports/stwm_ostf_v34_34_train_overfit_sanity_20260514.json"
DOC = ROOT / "docs/STWM_OSTF_V34_34_TRAIN_OVERFIT_SANITY_20260514.md"


def main() -> int:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model, ckargs, _ = load_frozen_residual_model(args, device)
    writer = CrossAttentionUnitDeltaWriterV3434(
        int(model.v30.cfg.hidden_dim),
        args.teacher_embedding_dim,
        args.value_hidden_dim,
        max_delta_magnitude=args.max_delta_magnitude,
    ).to(device)
    head = TensorUnitDeltaWriter(writer, args.target_kind).to(device)
    ckpt = CKPT_DIR / f"v34_34_cross_attention_unit_delta_value_writer_m128_h32_seed{args.seed}_{args.target_kind}.pt"
    ck = torch.load(ckpt, map_location=device)
    head.load_state_dict(ck["head"], strict=True)
    head.eval()
    readers = load_readers(args, model, device)
    split_cache = eval_sweep_split("train", model, ckargs, head, readers, args, device)
    rows = []
    for gate_mode in ("sparse_gate", "oracle_mask"):
        for scale in args.eval_scales:
            method = f"v34_34_{gate_mode}_cross_attention_unit_delta_writer"
            m = split_cache[gate_mode][float(scale)]["methods"][method]
            rows.append({
                "gate_mode": gate_mode,
                "scale": float(scale),
                "stable": m["stable_preservation"],
                "train_gain_anchor": m["hard_changed_gain_vs_anchor"],
                "train_gain_pointwise": m["hard_changed_gain_vs_pointwise"],
            })
    selected = choose_best([{"val_gain_anchor": r["train_gain_anchor"], "val_gain_pointwise": r["train_gain_pointwise"], **r} for r in rows if r["gate_mode"] == "sparse_gate"])
    gate_mode = selected["gate_mode"]
    scale = float(selected["scale"])
    method = f"v34_34_{gate_mode}_cross_attention_unit_delta_writer"
    metrics = split_cache[gate_mode][scale]["methods"][method]
    delta = split_cache[gate_mode][scale]["intervention_delta"]
    base = best_copy_topk(split_cache[gate_mode][scale])
    train_overfits_anchor = bool((metrics["hard_changed_gain_vs_anchor"] or -1.0) > 0.002)
    train_beats_copy_topk = bool((metrics["hard_changed_gain_vs_pointwise"] or -1.0) > float(base["hard_changed_gain_vs_pointwise"] or 0.0) + 0.002)
    payload = {
        "generated_at_utc": utc_now(),
        "中文结论": "V34.34 train-overfit sanity 完成；直接评估训练 split，判断 cross-attention value writer 是否至少能在训练集超过 evidence anchor。",
        "checkpoint_path": str(ckpt.relative_to(ROOT)),
        "selected_train_config": selected,
        "train_overfits_anchor": train_overfits_anchor,
        "train_beats_copy_topk": train_beats_copy_topk,
        "train_metrics": metrics,
        "train_intervention_delta": delta,
        "best_copy_topk_baseline_train": base,
        "阶段性分析": "若 train_overfits_anchor=false，当前 writer 即使在训练样本上也没有学成有益 unit_delta，问题偏向 value decoder capacity/监督分解；若 train=true 但 val/test=false，问题偏向 observed-only future delta 泛化。",
        "recommended_next_step": "fix_value_decoder_capacity" if not train_overfits_anchor else "fix_value_decoder_generalization",
    }
    dump_json(REPORT, payload)
    write_doc(
        DOC,
        "V34.34 train-overfit sanity 中文报告",
        payload,
        ["中文结论", "selected_train_config", "train_overfits_anchor", "train_beats_copy_topk", "train_intervention_delta", "recommended_next_step"],
    )
    print(f"已写出 V34.34 train-overfit sanity 报告: {REPORT.relative_to(ROOT)}", flush=True)
    print(f"train_overfits_anchor: {train_overfits_anchor}", flush=True)
    print(f"recommended_next_step: {payload['recommended_next_step']}", flush=True)
    return 0


def load_readers(args: argparse.Namespace, model: Any, device: torch.device) -> dict[str, dict[str, Any]]:
    from stwm.tools.eval_ostf_v34_26_full_system_baseline_claim_boundary_benchmark_20260514 import load_v3425_readers

    return load_v3425_readers(args, model, device)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--teacher-embedding-dim", type=int, default=768)
    p.add_argument("--value-hidden-dim", type=int, default=256)
    p.add_argument("--reader-hidden-dim", type=int, default=192)
    p.add_argument("--max-delta-magnitude", type=float, default=2.5)
    p.add_argument("--target-kind", choices=["soft", "top1"], default="top1")
    p.add_argument("--eval-scales", type=float, nargs="+", default=[0.25, 0.5, 1.0, 1.5, 2.0, 3.0])
    p.add_argument("--cpu", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
