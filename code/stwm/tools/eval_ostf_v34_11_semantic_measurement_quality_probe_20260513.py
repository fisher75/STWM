#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))

from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, write_doc
from stwm.tools.ostf_v30_external_gt_schema_20260508 import utc_now


MEAS_ROOT = ROOT / "outputs/cache/stwm_ostf_v34_9_trace_preserving_semantic_measurement_bank/pointodyssey"
TARGET_ROOT = ROOT / "outputs/cache/stwm_ostf_v34_9_causal_assignment_residual_targets/pointodyssey"
STRICT_ROOT = ROOT / "outputs/cache/stwm_ostf_v34_5_strict_residual_utility_targets/pointodyssey"
REPORT = ROOT / "reports/stwm_ostf_v34_11_semantic_measurement_quality_probe_20260513.json"
DOC = ROOT / "docs/STWM_OSTF_V34_11_SEMANTIC_MEASUREMENT_QUALITY_PROBE_20260513.md"


def _norm(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = np.nan_to_num(x.astype(np.float32), copy=False)
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), eps)


def _cos_point_future(point_vec: np.ndarray, fut: np.ndarray) -> np.ndarray:
    return (_norm(point_vec)[:, None, :] * _norm(fut)).sum(axis=-1)


def _safe_mean(vals: list[float]) -> float | None:
    return None if not vals else float(np.mean(vals))


class Acc:
    def __init__(self) -> None:
        self.sum: dict[str, float] = {}
        self.count: dict[str, int] = {}

    def add(self, key: str, values: np.ndarray, mask: np.ndarray) -> None:
        m = mask.astype(bool)
        if not m.any():
            return
        self.sum[key] = self.sum.get(key, 0.0) + float(values[m].sum())
        self.count[key] = self.count.get(key, 0) + int(m.sum())

    def mean(self, key: str) -> float | None:
        c = self.count.get(key, 0)
        return None if c == 0 else float(self.sum[key] / c)


def last_measurement(obs: np.ndarray, mask: np.ndarray) -> np.ndarray:
    m, t, d = obs.shape
    out = np.zeros((m, d), dtype=np.float32)
    for i in range(m):
        idx = np.where(mask[i])[0]
        if idx.size:
            out[i] = obs[i, idx[-1]]
    return out


def mean_measurement(obs: np.ndarray, mask: np.ndarray) -> np.ndarray:
    den = mask.sum(axis=1, keepdims=True).clip(min=1)
    return (obs * mask[..., None]).sum(axis=1) / den


def max_conf_measurement(obs: np.ndarray, mask: np.ndarray, conf: np.ndarray) -> np.ndarray:
    m, _, d = obs.shape
    score = np.where(mask, conf, -1.0)
    idx = score.argmax(axis=1)
    out = obs[np.arange(m), idx].astype(np.float32)
    out[score.max(axis=1) < 0] = 0.0
    return out


def weighted_measurement(obs: np.ndarray, mask: np.ndarray, conf: np.ndarray, agree: np.ndarray) -> np.ndarray:
    w = mask.astype(np.float32) * np.maximum(conf, 0.0) * np.maximum(agree, 0.0)
    den = w.sum(axis=1, keepdims=True).clip(min=1e-6)
    return (obs * w[..., None]).sum(axis=1) / den


def instance_pool(point_vec: np.ndarray, inst: np.ndarray) -> np.ndarray:
    out = point_vec.copy()
    for iid in np.unique(inst):
        if iid < 0:
            continue
        pts = inst == iid
        out[pts] = point_vec[pts].mean(axis=0, keepdims=True)
    return out


def unit_pool(point_vec: np.ndarray, assign: np.ndarray) -> np.ndarray:
    den = assign.sum(axis=0).clip(min=1e-6)
    unit = np.einsum("mu,md->ud", assign, point_vec) / den[:, None]
    return np.einsum("mu,ud->md", assign, unit)


def temporal_stability(obs: np.ndarray, mask: np.ndarray) -> float | None:
    vals: list[float] = []
    obs_n = _norm(obs)
    for i in range(obs.shape[0]):
        idx = np.where(mask[i])[0]
        if idx.size >= 2:
            vals.extend((obs_n[i, idx[:-1]] * obs_n[i, idx[1:]]).sum(axis=-1).tolist())
    return _safe_mean(vals)


def instance_consistency(point_vec: np.ndarray, inst: np.ndarray) -> float | None:
    vals: list[float] = []
    vec = _norm(point_vec)
    for iid in np.unique(inst):
        if iid < 0:
            continue
        pts = np.where(inst == iid)[0]
        if pts.size >= 2:
            center = _norm(vec[pts].mean(axis=0, keepdims=True))[0]
            vals.extend((vec[pts] * center[None]).sum(axis=-1).tolist())
    return _safe_mean(vals)


def split_eval(split: str, args: argparse.Namespace) -> dict[str, Any]:
    rng = np.random.default_rng(args.seed + (0 if split == "val" else 1000))
    files = sorted((MEAS_ROOT / split).glob("*.npz"))
    acc = Acc()
    stability_vals: list[float] = []
    inst_vals: list[float] = []
    margin_vals: list[float] = []
    sample_count = 0
    for mp in files:
        tp = TARGET_ROOT / split / mp.name
        sp = STRICT_ROOT / split / mp.name
        if not tp.exists() or not sp.exists():
            continue
        z = np.load(mp, allow_pickle=True)
        t = np.load(tp, allow_pickle=True)
        s = np.load(sp, allow_pickle=True)
        obs = np.asarray(z["obs_semantic_measurements"], dtype=np.float32)
        mask_obs = np.asarray(z["obs_semantic_measurement_mask"]).astype(bool)
        conf = np.asarray(z["obs_measurement_confidence"], dtype=np.float32)
        agree = np.asarray(z.get("teacher_agreement_score", conf), dtype=np.float32)
        fut = np.asarray(z["fut_teacher_embedding"], dtype=np.float32)
        valid = np.asarray(z["fut_teacher_available_mask"]).astype(bool)
        inst = np.asarray(z["point_to_instance_id"], dtype=np.int64)
        hard = np.asarray(s["semantic_hard_mask"]).astype(bool) & valid
        changed = np.asarray(s["changed_mask"]).astype(bool) & valid
        strict = np.asarray(s["strict_residual_semantic_utility_mask"]).astype(bool) & valid
        stable = np.asarray(s["stable_mask"]).astype(bool) & valid
        assign = np.asarray(t["point_to_unit_assignment"], dtype=np.float32)
        pointwise = np.asarray(s["pointwise_semantic_cosine"], dtype=np.float32)
        mean_vec = mean_measurement(obs, mask_obs)
        variants = {
            "last_observed": last_measurement(obs, mask_obs),
            "mean_observed": mean_vec,
            "max_confidence_observed": max_conf_measurement(obs, mask_obs, conf),
            "instance_pooled_observed": instance_pool(mean_vec, inst),
            "unit_pooled_observed": unit_pool(mean_vec, assign),
            "teacher_agreement_weighted_observed": weighted_measurement(obs, mask_obs, conf, agree),
        }
        perm = rng.permutation(mean_vec.shape[0])
        variants["random_shuffled_measurement"] = mean_vec[perm]
        masks = {"valid": valid, "semantic_hard": hard, "changed": changed, "strict_residual": strict, "stable": stable}
        for name, vec in variants.items():
            cos = _cos_point_future(vec, fut)
            for subset, subset_mask in masks.items():
                acc.add(f"{name}:{subset}", cos, subset_mask)
        for subset, subset_mask in masks.items():
            acc.add(f"pointwise_base:{subset}", pointwise, subset_mask)
        best_cos = np.maximum.reduce([_cos_point_future(v, fut) for k, v in variants.items() if k != "random_shuffled_measurement"])
        rand_cos = _cos_point_future(variants["random_shuffled_measurement"], fut)
        acc.add("best_measurement_minus_random:valid", best_cos - rand_cos, valid)
        acc.add("best_measurement_minus_pointwise:semantic_hard", best_cos - pointwise, hard)
        acc.add("best_measurement_minus_pointwise:changed", best_cos - pointwise, changed)
        acc.add("best_measurement_minus_pointwise:strict_residual", best_cos - pointwise, strict)
        stability = temporal_stability(obs, mask_obs)
        if stability is not None:
            stability_vals.append(stability)
        inst_cons = instance_consistency(mean_vec, inst)
        if inst_cons is not None:
            inst_vals.append(inst_cons)
        if valid.any():
            margin_vals.append(float((best_cos - rand_cos)[valid].mean()))
        sample_count += 1
    subsets = ["valid", "semantic_hard", "changed", "strict_residual", "stable"]
    variants_out = {}
    for name in [
        "last_observed",
        "mean_observed",
        "max_confidence_observed",
        "instance_pooled_observed",
        "unit_pooled_observed",
        "teacher_agreement_weighted_observed",
        "random_shuffled_measurement",
        "pointwise_base",
    ]:
        variants_out[name] = {subset: acc.mean(f"{name}:{subset}") for subset in subsets}
    best_valid = max(v["valid"] for k, v in variants_out.items() if k not in {"random_shuffled_measurement", "pointwise_base"} and v["valid"] is not None)
    random_valid = variants_out["random_shuffled_measurement"]["valid"]
    hard_delta = acc.mean("best_measurement_minus_pointwise:semantic_hard")
    changed_delta = acc.mean("best_measurement_minus_pointwise:changed")
    return {
        "sample_count": sample_count,
        "variant_cosine_by_subset": variants_out,
        "measurement_future_alignment_by_subset": {
            "best_measurement_valid": best_valid,
            "random_valid": random_valid,
            "pointwise_valid": variants_out["pointwise_base"]["valid"],
            "best_minus_random_valid": acc.mean("best_measurement_minus_random:valid"),
            "best_minus_pointwise_semantic_hard": hard_delta,
            "best_minus_pointwise_changed": changed_delta,
            "best_minus_pointwise_strict_residual": acc.mean("best_measurement_minus_pointwise:strict_residual"),
        },
        "measurement_temporal_stability": _safe_mean(stability_vals),
        "measurement_instance_consistency": _safe_mean(inst_vals),
        "measurement_discriminative_margin": _safe_mean(margin_vals),
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    per = {split: split_eval(split, args) for split in ("val", "test")}
    val_align = per["val"]["measurement_future_alignment_by_subset"]
    test_align = per["test"]["measurement_future_alignment_by_subset"]
    beats_random = bool((val_align["best_minus_random_valid"] or 0.0) > 0.01 and (test_align["best_minus_random_valid"] or 0.0) > 0.01)
    beats_hard = bool((val_align["best_minus_pointwise_semantic_hard"] or 0.0) > 0.002 and (test_align["best_minus_pointwise_semantic_hard"] or 0.0) > 0.002)
    beats_changed = bool((val_align["best_minus_pointwise_changed"] or 0.0) > 0.002 and (test_align["best_minus_pointwise_changed"] or 0.0) > 0.002)
    passed = bool(beats_random and beats_hard and beats_changed)
    payload = {
        "generated_at_utc": utc_now(),
        "中文结论": "V34.11 semantic measurement quality probe 已完成；该 probe 只评估 observed measurement 与 future supervision 的关系，不训练主模型。",
        "semantic_measurement_quality_probe_done": True,
        "semantic_measurement_quality_passed": passed,
        "measurement_beats_random": beats_random,
        "measurement_beats_pointwise_on_hard": beats_hard,
        "measurement_beats_pointwise_on_changed": beats_changed,
        "measurement_future_alignment_by_subset": {split: per[split]["measurement_future_alignment_by_subset"] for split in per},
        "measurement_temporal_stability": {split: per[split]["measurement_temporal_stability"] for split in per},
        "measurement_instance_consistency": {split: per[split]["measurement_instance_consistency"] for split in per},
        "measurement_discriminative_margin": {split: per[split]["measurement_discriminative_margin"] for split in per},
        "per_split": per,
        "recommended_interpretation": "measurement 若不赢 hard/changed 的 pointwise base，优先修 measurement bank；若只赢 random，可谨慎跑 local usage probe 但不能训练 learned gate。",
    }
    dump_json(REPORT, payload)
    write_doc(
        DOC,
        "V34.11 semantic measurement quality probe 中文报告",
        payload,
        [
            "中文结论",
            "semantic_measurement_quality_probe_done",
            "semantic_measurement_quality_passed",
            "measurement_beats_random",
            "measurement_beats_pointwise_on_hard",
            "measurement_beats_pointwise_on_changed",
            "measurement_future_alignment_by_subset",
            "measurement_temporal_stability",
            "measurement_instance_consistency",
            "measurement_discriminative_margin",
            "recommended_interpretation",
        ],
    )
    print(f"已写出 V34.11 semantic measurement quality probe: {REPORT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
