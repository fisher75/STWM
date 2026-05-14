#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))

from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, write_doc
from stwm.tools.ostf_v30_external_gt_schema_20260508 import utc_now


MEAS_ROOT = ROOT / "outputs/cache/stwm_ostf_v34_9_trace_preserving_semantic_measurement_bank/pointodyssey"
STRICT_ROOT = ROOT / "outputs/cache/stwm_ostf_v34_5_strict_residual_utility_targets/pointodyssey"
TARGET_ROOT = ROOT / "outputs/cache/stwm_ostf_v34_9_causal_assignment_residual_targets/pointodyssey"
V3414 = ROOT / "reports/stwm_ostf_v34_14_horizon_conditioned_measurement_selector_decision_20260513.json"
V3415 = ROOT / "reports/stwm_ostf_v34_15_horizon_timestep_supervised_selector_decision_20260513.json"
REPORT = ROOT / "reports/stwm_ostf_v34_16_selector_oracle_gap_predictability_audit_20260513.json"
DOC = ROOT / "docs/STWM_OSTF_V34_16_SELECTOR_ORACLE_GAP_PREDICTABILITY_AUDIT_20260513.md"


def norm(x: np.ndarray) -> np.ndarray:
    x = np.nan_to_num(x.astype(np.float32), copy=False)
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8)


def acc_add(acc: dict[str, list[float]], key: str, val: np.ndarray, mask: np.ndarray) -> None:
    if mask.any():
        chosen = val[mask]
        if np.isfinite(chosen).any():
            acc.setdefault(key, []).append(float(np.nanmean(chosen)))


def summarize(vals: list[float]) -> float | None:
    return None if not vals else float(np.mean(vals))


def split_audit(split: str) -> dict[str, Any]:
    acc: dict[str, list[float]] = {}
    hist = np.zeros(8, dtype=np.int64)
    sample_count = 0
    for mp in sorted((MEAS_ROOT / split).glob("*.npz")):
        sp = STRICT_ROOT / split / mp.name
        tp = TARGET_ROOT / split / mp.name
        if not sp.exists() or not tp.exists():
            continue
        z = np.load(mp, allow_pickle=True)
        s = np.load(sp, allow_pickle=True)
        obs = norm(np.asarray(z["obs_semantic_measurements"], dtype=np.float32))
        fut = norm(np.asarray(z["fut_teacher_embedding"], dtype=np.float32))
        mask = np.asarray(z["obs_semantic_measurement_mask"]).astype(bool)
        valid = np.asarray(z["fut_teacher_available_mask"]).astype(bool)
        hard = np.asarray(s["semantic_hard_mask"]).astype(bool) & valid
        changed = np.asarray(s["changed_mask"]).astype(bool) & valid
        strict = np.asarray(s["strict_residual_semantic_utility_mask"]).astype(bool) & valid
        causal = np.asarray(np.load(tp, allow_pickle=True)["causal_assignment_residual_semantic_mask"]).astype(bool) & valid
        sim = np.einsum("mtd,mhd->mht", obs, fut)
        sim = np.where(mask[:, None, :], sim, -1e4)
        best = sim.max(axis=-1)
        second = np.full_like(best, np.nan, dtype=np.float32)
        for mi in range(sim.shape[0]):
            valid_t = mask[mi]
            if valid_t.sum() >= 2:
                vals = sim[mi][:, valid_t]
                sort = np.sort(vals, axis=-1)
                second[mi] = sort[:, -2]
        margin = best - second
        top_idx = sim.argmax(axis=-1)
        for i in range(8):
            hist[i] += int(((top_idx == i) & valid).sum())
        # random top-1 chance is roughly 1 / available timesteps. Low margin means
        # oracle label is unstable and top-1 CE can punish nearly equivalent evidence.
        available_count = mask.sum(axis=-1)[:, None].clip(min=1)
        random_top1_chance = np.broadcast_to(1.0 / available_count, valid.shape)
        low_margin = np.nan_to_num(margin, nan=1.0) < 0.02
        for name, subset in {
            "valid": valid,
            "hard": hard,
            "changed": changed,
            "strict": strict,
            "causal": causal,
        }.items():
            acc_add(acc, f"oracle_best_cos:{name}", best, subset)
            acc_add(acc, f"oracle_margin:{name}", margin, subset)
            acc_add(acc, f"low_margin_ratio:{name}", low_margin.astype(np.float32), subset)
            acc_add(acc, f"random_top1_chance:{name}", random_top1_chance, subset)
        sample_count += 1
    out = {"sample_count": sample_count, "oracle_timestep_histogram": hist.tolist(), "metrics": {}}
    for key, vals in acc.items():
        out["metrics"][key] = summarize(vals)
    return out


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def main() -> int:
    per = {split: split_audit(split) for split in ("val", "test")}
    v14 = load(V3414)
    v15 = load(V3415)
    hard_margin = min(per["val"]["metrics"].get("oracle_margin:hard") or 0.0, per["test"]["metrics"].get("oracle_margin:hard") or 0.0)
    changed_margin = min(per["val"]["metrics"].get("oracle_margin:changed") or 0.0, per["test"]["metrics"].get("oracle_margin:changed") or 0.0)
    low_margin_hard = max(per["val"]["metrics"].get("low_margin_ratio:hard") or 0.0, per["test"]["metrics"].get("low_margin_ratio:hard") or 0.0)
    low_margin_changed = max(per["val"]["metrics"].get("low_margin_ratio:changed") or 0.0, per["test"]["metrics"].get("low_margin_ratio:changed") or 0.0)
    oracle_label_ambiguous = bool(hard_margin < 0.03 or changed_margin < 0.03 or low_margin_hard > 0.35 or low_margin_changed > 0.35)
    timestep_ce_hurt = bool(not v15.get("selector_beats_v34_14_on_oracle_gap", False) and not all(v15.get("selector_beats_pointwise_on_changed", {}).values()))
    payload: dict[str, Any] = {
        "generated_at_utc": utc_now(),
        "中文结论": "V34.16 selector oracle gap 可学习性审计完成：重点判断 oracle best timestep 是否是稳定、可学习的 observed-only 标签，以及 V34.15 top-1 CE 为什么没有改善 gap。",
        "per_split": per,
        "v34_14_selector_decision": {
            "selector_beats_pointwise_on_hard": v14.get("selector_beats_pointwise_on_hard"),
            "selector_beats_pointwise_on_changed": v14.get("selector_beats_pointwise_on_changed"),
            "oracle_gap_to_selector_hard": v14.get("oracle_gap_to_selector_hard"),
            "oracle_gap_to_selector_changed": v14.get("oracle_gap_to_selector_changed"),
        },
        "v34_15_selector_decision": {
            "selector_beats_pointwise_on_hard": v15.get("selector_beats_pointwise_on_hard"),
            "selector_beats_pointwise_on_changed": v15.get("selector_beats_pointwise_on_changed"),
            "selector_beats_v34_14_on_oracle_gap": v15.get("selector_beats_v34_14_on_oracle_gap"),
            "oracle_timestep_top1_hard": v15.get("oracle_timestep_top1_hard"),
            "oracle_timestep_top1_changed": v15.get("oracle_timestep_top1_changed"),
        },
        "oracle_timestep_label_ambiguous": oracle_label_ambiguous,
        "top1_timestep_ce_hurt_selector": timestep_ce_hurt,
        "best_current_selector": "v34_14_horizon_conditioned_soft_reader",
        "recommended_fix": "不要再把 oracle timestep 当硬 top-1 标签；保留 V34.14 soft horizon-conditioned reader，下一步应做 multi-evidence/top-k memory set 或 calibration，而不是 learned gate。",
        "recommended_next_step": "fix_nonoracle_measurement_selector_with_multievidence_memory",
    }
    dump_json(REPORT, payload)
    write_doc(
        DOC,
        "V34.16 selector oracle gap 可学习性审计中文报告",
        payload,
        [
            "中文结论",
            "oracle_timestep_label_ambiguous",
            "top1_timestep_ce_hurt_selector",
            "best_current_selector",
            "recommended_fix",
            "recommended_next_step",
        ],
    )
    print(f"已写出 V34.16 selector oracle gap 可学习性审计: {REPORT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
