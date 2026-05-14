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

from stwm.modules.ostf_v34_8_causal_assignment_bound_residual_memory import CausalAssignmentBoundResidualMemoryV348
from stwm.modules.ostf_v34_14_horizon_conditioned_measurement_selector import HorizonConditionedMeasurementSelectorV3414
from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, write_doc
from stwm.tools.ostf_v30_external_gt_schema_20260508 import utc_now
from stwm.tools.train_ostf_v33_2_visual_semantic_identity_20260509 import move_batch
from stwm.tools.train_ostf_v34_10_trace_contract_oracle_residual_probe_20260512 import TraceContractResidualDataset, collate_v3410
from stwm.tools.train_ostf_v34_14_horizon_conditioned_measurement_selector_20260513 import selector_inputs
from stwm.tools.train_ostf_v34_15_horizon_timestep_supervised_selector_20260513 import SUMMARY as TRAIN_SUMMARY, oracle_timestep_labels
from stwm.tools.eval_ostf_v34_14_horizon_conditioned_measurement_selector_20260513 import Acc, mean_pool, last_pool, max_conf_pool, weighted_pool, repeat_h, cos_future


EVAL = ROOT / "reports/stwm_ostf_v34_15_horizon_timestep_supervised_selector_eval_summary_20260513.json"
DECISION = ROOT / "reports/stwm_ostf_v34_15_horizon_timestep_supervised_selector_decision_20260513.json"
DOC = ROOT / "docs/STWM_OSTF_V34_15_HORIZON_TIMESTEP_SUPERVISED_SELECTOR_DECISION_20260513.md"
V3414_DECISION = ROOT / "reports/stwm_ostf_v34_14_horizon_conditioned_measurement_selector_decision_20260513.json"


def load_model(device: torch.device) -> tuple[HorizonConditionedMeasurementSelectorV3414 | None, CausalAssignmentBoundResidualMemoryV348 | None, argparse.Namespace | None, dict[str, Any]]:
    train = json.loads(TRAIN_SUMMARY.read_text(encoding="utf-8")) if TRAIN_SUMMARY.exists() else {}
    ckpt_rel = train.get("checkpoint_path")
    if not ckpt_rel:
        return None, None, None, train
    ckpt = ROOT / ckpt_rel
    if not ckpt.exists():
        return None, None, None, train
    ck = torch.load(ckpt, map_location="cpu")
    ckargs = argparse.Namespace(**ck["args"])
    base_model = CausalAssignmentBoundResidualMemoryV348(ckargs.v30_checkpoint, teacher_embedding_dim=ckargs.teacher_embedding_dim, units=ckargs.trace_units, horizon=ckargs.horizon).to(device)
    for p in base_model.parameters():
        p.requires_grad_(False)
    base_model.eval()
    selector = HorizonConditionedMeasurementSelectorV3414(int(ck["trace_hidden_dim"]), ckargs.teacher_embedding_dim, ckargs.hidden_dim).to(device)
    selector.load_state_dict(ck["model"], strict=True)
    selector.eval()
    return selector, base_model, ckargs, train


def make_loader(split: str, args: argparse.Namespace) -> DataLoader:
    return DataLoader(TraceContractResidualDataset(split, args), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_v3410)


def eval_split(split: str, selector: HorizonConditionedMeasurementSelectorV3414, base_model: CausalAssignmentBoundResidualMemoryV348, ckargs: argparse.Namespace, args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    ckargs.batch_size = args.batch_size
    ckargs.num_workers = args.num_workers
    loader = make_loader(split, ckargs)
    acc = Acc()
    entropy_vals: list[float] = []
    maxw_vals: list[float] = []
    top1_vals: list[float] = []
    with torch.no_grad():
        for batch in loader:
            bd = move_batch(batch, device)
            out = selector(**selector_inputs(selector, base_model, bd))
            obs = torch.nan_to_num(bd["obs_semantic_measurements"].float())
            mask = bd["obs_semantic_measurement_mask"].bool()
            conf = bd["semantic_measurement_confidence"].float()
            agree = bd.get("teacher_agreement_score", bd.get("semantic_measurement_agreement", bd["semantic_measurement_confidence"])).float()
            if conf.dim() == 2:
                conf = conf[:, :, None].expand_as(mask.float())
            if agree.dim() == 2:
                agree = agree[:, :, None].expand_as(mask.float())
            h = bd["fut_teacher_embedding"].shape[2]
            variants = {
                "timestep_supervised_selector": out["selected_evidence"],
                "fixed_mean": repeat_h(F.normalize(mean_pool(obs, mask), dim=-1), h),
                "fixed_last": repeat_h(F.normalize(last_pool(obs, mask), dim=-1), h),
                "fixed_max_confidence": repeat_h(F.normalize(max_conf_pool(obs, mask, conf), dim=-1), h),
                "fixed_agreement_weighted": repeat_h(F.normalize(weighted_pool(obs, mask, conf, agree), dim=-1), h),
            }
            variants["random_shuffled"] = variants["fixed_mean"][:, torch.randperm(variants["fixed_mean"].shape[1], device=device)]
            cos = {k: cos_future(v, bd["fut_teacher_embedding"]) for k, v in variants.items()}
            obs_cos = torch.einsum("bmtd,bmhd->bmht", F.normalize(obs, dim=-1), F.normalize(torch.nan_to_num(bd["fut_teacher_embedding"].float()), dim=-1))
            obs_cos = obs_cos.masked_fill(~mask[:, :, None, :], -1e4)
            cos["oracle_timestep_best"] = obs_cos.max(dim=-1).values
            stack = torch.stack([cos[k] for k in ["fixed_mean", "fixed_last", "fixed_max_confidence", "fixed_agreement_weighted"]], dim=0)
            cos["oracle_fixed_variant_best"] = stack.max(dim=0).values
            cos["pointwise_base"] = bd["pointwise_semantic_cosine"].float()
            labels, _, available = oracle_timestep_labels(bd)
            pred = out["measurement_weight"].argmax(dim=-1)
            valid = bd["fut_teacher_available_mask"].bool() & available
            masks = {
                "valid": valid,
                "hard": bd["semantic_hard_mask"].bool() & valid,
                "changed": bd["changed_mask"].bool() & valid,
                "strict": bd["strict_residual_semantic_utility_mask"].bool() & valid,
                "causal": bd["causal_assignment_residual_semantic_mask"].bool() & valid,
            }
            for name, val in cos.items():
                for subset, smask in masks.items():
                    acc.add(f"{name}:{subset}", val, smask)
            hit = (pred == labels).float()
            for subset, smask in masks.items():
                acc.add(f"top1:{subset}", hit, smask)
                acc.add(f"selector_minus_random:{subset}", cos["timestep_supervised_selector"] - cos["random_shuffled"], smask)
                acc.add(f"selector_minus_pointwise:{subset}", cos["timestep_supervised_selector"] - cos["pointwise_base"], smask)
                acc.add(f"selector_minus_fixed_oracle:{subset}", cos["timestep_supervised_selector"] - cos["oracle_fixed_variant_best"], smask)
                acc.add(f"oracle_fixed_minus_selector:{subset}", cos["oracle_fixed_variant_best"] - cos["timestep_supervised_selector"], smask)
                acc.add(f"oracle_timestep_minus_selector:{subset}", cos["oracle_timestep_best"] - cos["timestep_supervised_selector"], smask)
            entropy_vals.append(float(out["selector_entropy"].mean().detach().cpu()))
            maxw_vals.append(float(out["selector_max_weight"].mean().detach().cpu()))
            top1_vals.append(float(hit[valid].mean().detach().cpu()) if bool(valid.any()) else 0.0)
    deltas = {
        "selector_minus_random_valid": acc.mean("selector_minus_random:valid"),
        "selector_minus_pointwise_hard": acc.mean("selector_minus_pointwise:hard"),
        "selector_minus_pointwise_changed": acc.mean("selector_minus_pointwise:changed"),
        "selector_minus_pointwise_causal": acc.mean("selector_minus_pointwise:causal"),
        "selector_minus_fixed_oracle_hard": acc.mean("selector_minus_fixed_oracle:hard"),
        "oracle_gap_to_selector_hard": acc.mean("oracle_fixed_minus_selector:hard"),
        "oracle_gap_to_selector_changed": acc.mean("oracle_fixed_minus_selector:changed"),
        "oracle_timestep_gap_to_selector_hard": acc.mean("oracle_timestep_minus_selector:hard"),
        "oracle_timestep_gap_to_selector_changed": acc.mean("oracle_timestep_minus_selector:changed"),
        "oracle_timestep_top1_hard": acc.mean("top1:hard"),
        "oracle_timestep_top1_changed": acc.mean("top1:changed"),
    }
    return {
        "sample_count": len(loader.dataset),
        "variant_cosine_by_subset": {name: {subset: acc.mean(f"{name}:{subset}") for subset in ["valid", "hard", "changed", "strict", "causal"]} for name in ["timestep_supervised_selector", "fixed_mean", "fixed_last", "fixed_max_confidence", "fixed_agreement_weighted", "pointwise_base", "random_shuffled", "oracle_fixed_variant_best", "oracle_timestep_best"]},
        "selector_deltas": deltas,
        "selector_entropy": float(np.mean(entropy_vals)) if entropy_vals else None,
        "selector_max_weight": float(np.mean(maxw_vals)) if maxw_vals else None,
        "oracle_timestep_top1": float(np.mean(top1_vals)) if top1_vals else None,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    selector, base_model, ckargs, train = load_model(device)
    if selector is None or base_model is None or ckargs is None:
        decision = {"generated_at_utc": utc_now(), "中文结论": "V34.15 selector 未找到 checkpoint，评估跳过。", "measurement_selector_nonoracle_passed": False, "recommended_next_step": "fix_nonoracle_measurement_selector"}
        dump_json(EVAL, {"generated_at_utc": utc_now(), "train_summary": train, "decision": decision})
        dump_json(DECISION, decision)
        write_doc(DOC, "V34.15 selector 决策中文报告", decision, ["中文结论", "measurement_selector_nonoracle_passed", "recommended_next_step"])
        print(f"已写出 V34.15 selector 评估跳过报告: {DECISION.relative_to(ROOT)}")
        return 0
    per = {split: eval_split(split, selector, base_model, ckargs, args, device) for split in ("val", "test")}
    val, test = per["val"]["selector_deltas"], per["test"]["selector_deltas"]
    beats_random = {"val": bool((val["selector_minus_random_valid"] or 0.0) > 0.01), "test": bool((test["selector_minus_random_valid"] or 0.0) > 0.01)}
    beats_hard = {"val": bool((val["selector_minus_pointwise_hard"] or 0.0) > 0.002), "test": bool((test["selector_minus_pointwise_hard"] or 0.0) > 0.002)}
    beats_changed = {"val": bool((val["selector_minus_pointwise_changed"] or 0.0) > 0.002), "test": bool((test["selector_minus_pointwise_changed"] or 0.0) > 0.002)}
    hard_gap = {"val": val["oracle_gap_to_selector_hard"], "test": test["oracle_gap_to_selector_hard"]}
    changed_gap = {"val": val["oracle_gap_to_selector_changed"], "test": test["oracle_gap_to_selector_changed"]}
    max_gap = max(float(hard_gap["val"] or 0.0), float(hard_gap["test"] or 0.0), float(changed_gap["val"] or 0.0), float(changed_gap["test"] or 0.0))
    v3414 = json.loads(V3414_DECISION.read_text(encoding="utf-8")) if V3414_DECISION.exists() else {}
    v3414_gap = max(float(v3414.get("oracle_gap_to_selector_hard", {}).get("val") or 999), float(v3414.get("oracle_gap_to_selector_hard", {}).get("test") or 999), float(v3414.get("oracle_gap_to_selector_changed", {}).get("val") or 999), float(v3414.get("oracle_gap_to_selector_changed", {}).get("test") or 999))
    passed = bool(all(beats_random.values()) and all(beats_hard.values()) and all(beats_changed.values()) and max_gap <= 0.08)
    decision = {
        "generated_at_utc": utc_now(),
        "中文结论": "V34.15 timestep-supervised selector 已评估；该轮测试用 future target 只生成训练监督的 observed timestep label，forward 仍不读 future target。",
        "horizon_timestep_supervised_selector_built": True,
        "selector_was_trained": True,
        "measurement_selector_nonoracle_passed": passed,
        "selector_beats_random": beats_random,
        "selector_beats_pointwise_on_hard": beats_hard,
        "selector_beats_pointwise_on_changed": beats_changed,
        "selector_beats_v34_14_on_oracle_gap": bool(max_gap < v3414_gap),
        "oracle_gap_to_selector_hard": hard_gap,
        "oracle_gap_to_selector_changed": changed_gap,
        "oracle_timestep_gap_to_selector_hard": {"val": val["oracle_timestep_gap_to_selector_hard"], "test": test["oracle_timestep_gap_to_selector_hard"]},
        "oracle_timestep_gap_to_selector_changed": {"val": val["oracle_timestep_gap_to_selector_changed"], "test": test["oracle_timestep_gap_to_selector_changed"]},
        "oracle_timestep_top1_hard": {"val": val["oracle_timestep_top1_hard"], "test": test["oracle_timestep_top1_hard"]},
        "oracle_timestep_top1_changed": {"val": val["oracle_timestep_top1_changed"], "test": test["oracle_timestep_top1_changed"]},
        "selector_entropy": {"val": per["val"]["selector_entropy"], "test": per["test"]["selector_entropy"]},
        "selector_max_weight": {"val": per["val"]["selector_max_weight"], "test": per["test"]["selector_max_weight"]},
        "measurement_weight_shape": "B,M,H,Tobs",
        "selected_evidence_shape": "B,M,H,D",
        "future_teacher_embedding_input_allowed": False,
        "v30_backbone_frozen": True,
        "recommended_next_step": "run_selector_conditioned_oracle_probe" if passed else "fix_nonoracle_measurement_selector",
    }
    payload = {"generated_at_utc": utc_now(), "train_summary": train, "v34_14_reference": v3414, "per_split": per, "decision": decision}
    dump_json(EVAL, payload)
    dump_json(DECISION, decision)
    write_doc(DOC, "V34.15 timestep-supervised selector 决策中文报告", decision, ["中文结论", "horizon_timestep_supervised_selector_built", "selector_was_trained", "measurement_selector_nonoracle_passed", "selector_beats_random", "selector_beats_pointwise_on_hard", "selector_beats_pointwise_on_changed", "selector_beats_v34_14_on_oracle_gap", "oracle_gap_to_selector_hard", "oracle_gap_to_selector_changed", "oracle_timestep_top1_hard", "oracle_timestep_top1_changed", "selector_entropy", "selector_max_weight", "recommended_next_step"])
    print(f"已写出 V34.15 selector 评估摘要: {EVAL.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
