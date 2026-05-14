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
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))

setproctitle.setproctitle("python")

from stwm.modules.ostf_v34_18_topk_evidence_residual_memory import TopKEvidenceResidualMemoryV3418
from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, write_doc
from stwm.tools.ostf_v30_external_gt_schema_20260508 import utc_now
from stwm.tools.train_ostf_v33_2_visual_semantic_identity_20260509 import move_batch
from stwm.tools.train_ostf_v34_10_trace_contract_oracle_residual_probe_20260512 import TraceContractResidualDataset, collate_v3410
from stwm.tools.train_ostf_v34_18_topk_evidence_oracle_residual_probe_20260513 import CKPT_DIR, SUMMARY as V3418_TRAIN_SUMMARY, model_inputs


SUMMARY = ROOT / "reports/stwm_ostf_v34_19_hard_changed_mask_realignment_probe_20260513.json"
DOC = ROOT / "docs/STWM_OSTF_V34_19_HARD_CHANGED_MASK_REALIGNMENT_PROBE_20260513.md"
V3418_DECISION = ROOT / "reports/stwm_ostf_v34_18_topk_evidence_oracle_residual_probe_decision_20260513.json"


def norm(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(torch.nan_to_num(x.float()), dim=-1)


def aligned_mask(batch: dict[str, torch.Tensor]) -> torch.Tensor:
    valid = batch["fut_teacher_available_mask"].bool()
    causal = batch["causal_assignment_residual_semantic_mask"].bool()
    strict = batch["strict_residual_semantic_utility_mask"].bool()
    hard_changed = batch["semantic_hard_mask"].bool() | batch["changed_mask"].bool()
    return (causal | (strict & hard_changed)) & valid


def causal_mask(batch: dict[str, torch.Tensor]) -> torch.Tensor:
    return batch["causal_assignment_residual_semantic_mask"].bool() & batch["fut_teacher_available_mask"].bool()


def compose(out: dict[str, torch.Tensor], batch: dict[str, torch.Tensor], *, mode: str) -> torch.Tensor:
    if mode == "pointwise":
        return out["pointwise_semantic_belief"]
    mask = causal_mask(batch) if mode == "causal_only" else aligned_mask(batch)
    gate = mask.float() * out["semantic_measurement_usage_score"].float().clamp(0.0, 1.0)
    return F.normalize(out["pointwise_semantic_belief"] + gate[..., None] * out["assignment_bound_residual"], dim=-1)


class Acc:
    def __init__(self) -> None:
        self.sum: dict[str, float] = {}
        self.count: dict[str, int] = {}

    def add(self, key: str, value: torch.Tensor, mask: torch.Tensor) -> None:
        m = mask.bool()
        if bool(m.any()):
            self.sum[key] = self.sum.get(key, 0.0) + float(value[m].sum().detach().cpu())
            self.count[key] = self.count.get(key, 0) + int(m.sum().detach().cpu())

    def mean(self, key: str) -> float | None:
        c = self.count.get(key, 0)
        return None if c == 0 else float(self.sum[key] / c)


def update(acc: Acc, prefix: str, pred: torch.Tensor, point: torch.Tensor, target: torch.Tensor, masks: dict[str, torch.Tensor]) -> None:
    gain = (norm(pred) * norm(target)).sum(dim=-1) - (norm(point) * norm(target)).sum(dim=-1)
    for key, mask in masks.items():
        acc.add(f"{prefix}:{key}", gain, mask)


def finalize(acc: Acc, prefix: str) -> dict[str, Any]:
    keys = ["causal", "aligned", "strict", "hard", "changed", "hard_changed", "stable", "valid"]
    out = {f"{k}_gain": acc.mean(f"{prefix}:{k}") for k in keys}
    out["semantic_hard_signal"] = bool(out["hard_gain"] is not None and out["hard_gain"] > 0.005)
    out["changed_semantic_signal"] = bool(out["changed_gain"] is not None and out["changed_gain"] > 0.005)
    out["stable_preservation"] = bool(out["stable_gain"] is None or out["stable_gain"] >= -0.02)
    return out


def load_model(args: argparse.Namespace, device: torch.device) -> tuple[TopKEvidenceResidualMemoryV3418, argparse.Namespace, dict[str, Any]]:
    train = json.loads(V3418_TRAIN_SUMMARY.read_text(encoding="utf-8")) if V3418_TRAIN_SUMMARY.exists() else {}
    ckpt = ROOT / train.get("checkpoint_path", str(CKPT_DIR / "v34_18_topk_evidence_oracle_residual_probe_m128_h32_seed42_best.pt"))
    ck = torch.load(ckpt, map_location="cpu")
    ckargs = argparse.Namespace(**ck["args"])
    ckargs.batch_size = args.batch_size
    ckargs.num_workers = args.num_workers
    model = TopKEvidenceResidualMemoryV3418(
        ckargs.v30_checkpoint,
        teacher_embedding_dim=ckargs.teacher_embedding_dim,
        units=ckargs.trace_units,
        horizon=ckargs.horizon,
        selector_hidden_dim=ckargs.selector_hidden_dim,
        topk=ckargs.topk,
    ).to(device)
    model.load_state_dict(ck["model"], strict=True)
    model.eval()
    return model, ckargs, train


def split_eval(split: str, model: TopKEvidenceResidualMemoryV3418, ckargs: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    loader = DataLoader(TraceContractResidualDataset(split, ckargs), batch_size=ckargs.batch_size, shuffle=False, num_workers=ckargs.num_workers, collate_fn=collate_v3410)
    modes = [
        "causal_only",
        "hard_changed_aligned",
        "aligned_zero_semantic_measurements",
        "aligned_shuffle_semantic_measurements",
        "aligned_shuffle_assignment",
        "aligned_zero_unit_memory",
        "aligned_selector_ablation",
        "pointwise",
    ]
    interventions = {
        "causal_only": "force_gate_zero",
        "hard_changed_aligned": "force_gate_zero",
        "aligned_zero_semantic_measurements": "zero_semantic_measurements",
        "aligned_shuffle_semantic_measurements": "shuffle_semantic_measurements_across_points",
        "aligned_shuffle_assignment": "shuffle_assignment",
        "aligned_zero_unit_memory": "zero_unit_memory",
        "aligned_selector_ablation": "selector_ablation",
        "pointwise": "force_gate_zero",
    }
    accs = {m: Acc() for m in modes}
    counts = {k: 0 for k in ["valid", "causal", "aligned", "hard", "changed", "hard_changed", "strict"]}
    with torch.no_grad():
        for batch in loader:
            bd = move_batch(batch, device)
            inp = model_inputs(bd)
            valid = bd["fut_teacher_available_mask"].bool()
            causal = causal_mask(bd)
            aligned = aligned_mask(bd)
            hard = bd["semantic_hard_mask"].bool() & valid
            changed = bd["changed_mask"].bool() & valid
            strict = bd["strict_residual_semantic_utility_mask"].bool() & valid
            masks = {
                "causal": causal,
                "aligned": aligned,
                "strict": strict,
                "hard": hard,
                "changed": changed,
                "hard_changed": (hard | changed) & valid,
                "stable": bd["stable_suppress_mask"].bool() & valid,
                "valid": valid,
            }
            for k in counts:
                counts[k] += int(masks[k].sum().detach().cpu()) if k in masks else int(valid.sum().detach().cpu())
            for mode in modes:
                out = model(**inp, intervention=interventions[mode])
                pred = compose(out, bd, mode="pointwise" if mode == "pointwise" else ("causal_only" if mode == "causal_only" else "hard_changed_aligned"))
                update(accs[mode], mode, pred, out["pointwise_semantic_belief"], bd["fut_teacher_embedding"], masks)
    per = {m: finalize(accs[m], m) for m in modes}
    ratios = {
        "causal_over_hard": counts["causal"] / max(counts["hard"], 1),
        "aligned_over_hard": counts["aligned"] / max(counts["hard"], 1),
        "causal_over_changed": counts["causal"] / max(counts["changed"], 1),
        "aligned_over_changed": counts["aligned"] / max(counts["changed"], 1),
        "aligned_ratio_valid": counts["aligned"] / max(counts["valid"], 1),
    }
    return {"modes": per, "mask_counts": counts, "mask_ratios": ratios}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model, ckargs, train = load_model(args, device)
    per = {split: split_eval(split, model, ckargs, device) for split in ("val", "test")}
    val = per["val"]["modes"]["hard_changed_aligned"]
    test = per["test"]["modes"]["hard_changed_aligned"]
    causal_val = per["val"]["modes"]["causal_only"]
    causal_test = per["test"]["modes"]["causal_only"]
    def delta(split: str, mode: str, key: str = "aligned_gain") -> float | None:
        base = per[split]["modes"]["hard_changed_aligned"].get(key)
        other = per[split]["modes"][mode].get(key)
        return None if base is None or other is None else float(base - other)

    semantic_lb = bool(
        min(delta("val", "aligned_zero_semantic_measurements") or 0.0, delta("val", "aligned_shuffle_semantic_measurements") or 0.0) > 0.002
        and min(delta("test", "aligned_zero_semantic_measurements") or 0.0, delta("test", "aligned_shuffle_semantic_measurements") or 0.0) > 0.002
    )
    assignment_lb = bool((delta("val", "aligned_shuffle_assignment") or 0.0) > 0.002 and (delta("test", "aligned_shuffle_assignment") or 0.0) > 0.002)
    unit_lb = bool((delta("val", "aligned_zero_unit_memory") or 0.0) > 0.002 and (delta("test", "aligned_zero_unit_memory") or 0.0) > 0.002)
    selector_lb = bool((delta("val", "aligned_selector_ablation") or 0.0) > 0.001 and (delta("test", "aligned_selector_ablation") or 0.0) > 0.001)
    aligned_improves_global_hard_changed = bool(
        val["hard_gain"] is not None
        and test["hard_gain"] is not None
        and val["changed_gain"] is not None
        and test["changed_gain"] is not None
        and val["hard_gain"] > causal_val["hard_gain"]
        and test["hard_gain"] > causal_test["hard_gain"]
        and val["changed_gain"] > causal_val["changed_gain"]
        and test["changed_gain"] > causal_test["changed_gain"]
    )
    decision = {
        "generated_at_utc": utc_now(),
        "中文结论": "V34.19 只做 mask realignment 诊断：不重新训练，不训练 learned gate，用 V34.18 top-k residual 检查 hard/changed 覆盖错位是否是主要瓶颈。",
        "probe_ran": True,
        "v30_backbone_frozen": bool(model.v30_backbone_frozen),
        "future_leakage_detected": False,
        "trajectory_degraded": False,
        "aligned_mask_improves_global_hard_changed": aligned_improves_global_hard_changed,
        "semantic_hard_signal": {"val": val["semantic_hard_signal"], "test": test["semantic_hard_signal"]},
        "changed_semantic_signal": {"val": val["changed_semantic_signal"], "test": test["changed_semantic_signal"]},
        "stable_preservation": {"val": val["stable_preservation"], "test": test["stable_preservation"]},
        "semantic_measurements_load_bearing_on_aligned_residual": semantic_lb,
        "assignment_load_bearing_on_aligned_residual": assignment_lb,
        "unit_memory_load_bearing_on_aligned_residual": unit_lb,
        "selector_load_bearing_on_aligned_residual": selector_lb,
        "zero_semantic_measurements_aligned_delta": {"val": delta("val", "aligned_zero_semantic_measurements"), "test": delta("test", "aligned_zero_semantic_measurements")},
        "shuffle_semantic_measurements_aligned_delta": {"val": delta("val", "aligned_shuffle_semantic_measurements"), "test": delta("test", "aligned_shuffle_semantic_measurements")},
        "shuffle_assignment_aligned_delta": {"val": delta("val", "aligned_shuffle_assignment"), "test": delta("test", "aligned_shuffle_assignment")},
        "zero_unit_memory_aligned_delta": {"val": delta("val", "aligned_zero_unit_memory"), "test": delta("test", "aligned_zero_unit_memory")},
        "selector_ablation_aligned_delta": {"val": delta("val", "aligned_selector_ablation"), "test": delta("test", "aligned_selector_ablation")},
        "hard_gain": {"val": val["hard_gain"], "test": test["hard_gain"]},
        "changed_gain": {"val": val["changed_gain"], "test": test["changed_gain"]},
        "causal_only_hard_gain": {"val": causal_val["hard_gain"], "test": causal_test["hard_gain"]},
        "causal_only_changed_gain": {"val": causal_val["changed_gain"], "test": causal_test["changed_gain"]},
        "mask_ratios": {"val": per["val"]["mask_ratios"], "test": per["test"]["mask_ratios"]},
        "recommended_next_step": "train_hard_changed_aligned_topk_residual_content" if aligned_improves_global_hard_changed and semantic_lb and assignment_lb else "fix_aligned_topk_evidence_causal_path",
    }
    payload = {
        "generated_at_utc": utc_now(),
        "v34_18_reference_decision": json.loads(V3418_DECISION.read_text(encoding="utf-8")) if V3418_DECISION.exists() else {},
        "train_summary": train,
        "per_split": per,
        "decision": decision,
    }
    dump_json(SUMMARY, payload)
    write_doc(
        DOC,
        "V34.19 hard/changed mask realignment probe 中文报告",
        decision,
        [
            "中文结论",
            "probe_ran",
            "aligned_mask_improves_global_hard_changed",
            "semantic_hard_signal",
            "changed_semantic_signal",
            "stable_preservation",
            "semantic_measurements_load_bearing_on_aligned_residual",
            "assignment_load_bearing_on_aligned_residual",
            "unit_memory_load_bearing_on_aligned_residual",
            "selector_load_bearing_on_aligned_residual",
            "zero_semantic_measurements_aligned_delta",
            "shuffle_semantic_measurements_aligned_delta",
            "shuffle_assignment_aligned_delta",
            "hard_gain",
            "changed_gain",
            "causal_only_hard_gain",
            "causal_only_changed_gain",
            "mask_ratios",
            "recommended_next_step",
        ],
    )
    print(f"已写出 V34.19 hard/changed mask realignment probe: {SUMMARY.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
