#!/usr/bin/env python3
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _dump(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_doc(path: Path, payload: dict[str, Any]) -> None:
    lines = ["# STWM-FSTF Trace Density Valid Units Audit V13", ""]
    for key in [
        "trace_density_scaling_positive",
        "dense_trace_field_claim_allowed",
        "recommended_wording",
        "K16_actually_adds_semantic_supervision",
        "K32_actually_adds_semantic_supervision",
    ]:
        lines.append(f"- {key}: `{payload.get(key)}`")
    lines.append("")
    for k, row in payload.get("by_K", {}).items():
        lines.append(f"## {k}")
        for field in [
            "total_slots",
            "valid_observed_memory_slots",
            "valid_future_target_slots",
            "valid_joint_supervised_slots",
            "valid_changed_slots",
            "valid_stable_slots",
            "new_valid_slots_added_vs_K8",
            "invalid_slot_ratio",
        ]:
            lines.append(f"- {field}: `{row.get(field)}`")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _report_for_k(k: int) -> Path:
    if k == 8:
        return Path("reports/stwm_fullscale_semantic_trace_prototype_targets_c32_v1_20260428.json")
    return Path(f"reports/stwm_fstf_trace_density_k{k}_prototype_targets_c32_v12_20260502.json")


def _future_path(k: int) -> Path:
    report = _load_json(_report_for_k(k))
    return Path(report["target_cache_path"])


def _observed_path() -> Path:
    report = _load_json(Path("reports/stwm_mixed_observed_semantic_prototype_targets_v2_20260428.json"))
    by_count = report.get("target_cache_paths_by_prototype_count", {})
    return Path(by_count.get("32") or report["target_cache_path"])


def _norm_dataset_name(x: Any) -> str:
    text = str(x)
    if text.lower() == "vipseg":
        return "VIPSeg"
    if text.lower() == "vspw":
        return "VSPW"
    return text


def _audit_k(k: int, obs: np.lib.npyio.NpzFile, k8_joint: int | None = None) -> dict[str, Any]:
    fut = np.load(_future_path(k), allow_pickle=True)
    item_keys = [str(x) for x in fut["item_keys"].tolist()]
    datasets = [_norm_dataset_name(x) for x in fut["datasets"].tolist()] if "datasets" in fut.files else ["unknown"] * len(item_keys)
    target = fut["future_semantic_proto_target"]
    target_mask = fut["target_mask"].astype(bool)
    obs_keys = [str(x) for x in obs["item_keys"].tolist()]
    obs_index = {key: idx for idx, key in enumerate(obs_keys)}
    obs_mask_all = obs["observed_semantic_proto_mask"].astype(bool)
    obs_target_all = obs["observed_semantic_proto_target"]
    n, h, slots = target_mask.shape
    obs_mask = np.zeros((n, slots), dtype=bool)
    obs_target = np.full((n, slots), -1, dtype=np.int64)
    for i, key in enumerate(item_keys):
        j = obs_index.get(key)
        if j is None:
            continue
        take = min(slots, obs_mask_all.shape[1])
        obs_mask[i, :take] = obs_mask_all[j, :take]
        obs_target[i, :take] = obs_target_all[j, :take]
    obs_exp = np.broadcast_to(obs_mask[:, None, :], target_mask.shape)
    obs_target_exp = np.broadcast_to(obs_target[:, None, :], target_mask.shape)
    valid_future = target_mask & (target >= 0)
    valid_joint = valid_future & obs_exp & (obs_target_exp >= 0)
    changed = valid_joint & (target != obs_target_exp)
    stable = valid_joint & (target == obs_target_exp)
    total_slots = int(n * h * slots)
    per_dataset: dict[str, dict[str, Any]] = {}
    for ds in sorted(set(datasets)):
        idx = np.asarray([x == ds for x in datasets], dtype=bool)
        ds_total = int(idx.sum() * h * slots)
        per_dataset[ds] = {
            "item_count": int(idx.sum()),
            "total_slots": ds_total,
            "valid_observed_memory_slots": int(np.broadcast_to(obs_mask[idx, None, :], (idx.sum(), h, slots)).sum()) if idx.any() else 0,
            "valid_future_target_slots": int(valid_future[idx].sum()),
            "valid_joint_supervised_slots": int(valid_joint[idx].sum()),
            "valid_changed_slots": int(changed[idx].sum()),
            "valid_stable_slots": int(stable[idx].sum()),
            "future_target_coverage": float(valid_future[idx].sum() / max(ds_total, 1)),
            "joint_supervision_coverage": float(valid_joint[idx].sum() / max(ds_total, 1)),
        }
    joint_count = int(valid_joint.sum())
    return {
        "K": int(k),
        "future_cache_path": str(_future_path(k)),
        "total_items": int(n),
        "horizon": int(h),
        "slot_count": int(slots),
        "total_slots": total_slots,
        "valid_observed_memory_slots": int(obs_exp.sum()),
        "valid_future_target_slots": int(valid_future.sum()),
        "valid_joint_supervised_slots": joint_count,
        "valid_changed_slots": int(changed.sum()),
        "valid_stable_slots": int(stable.sum()),
        "new_valid_slots_added_vs_K8": None if k8_joint is None else int(joint_count - k8_joint),
        "invalid_slot_ratio": float(1.0 - joint_count / max(total_slots, 1)),
        "future_target_coverage": float(valid_future.sum() / max(total_slots, 1)),
        "joint_supervision_coverage": float(joint_count / max(total_slots, 1)),
        "per_dataset": per_dataset,
    }


def main() -> int:
    obs_path = _observed_path()
    obs = np.load(obs_path, allow_pickle=True)
    by_k: dict[str, Any] = {}
    k8 = _audit_k(8, obs, None)
    by_k["K8"] = k8
    for k in [16, 32]:
        by_k[f"K{k}"] = _audit_k(k, obs, int(k8["valid_joint_supervised_slots"]))
    k16_adds = bool(by_k["K16"]["new_valid_slots_added_vs_K8"] and by_k["K16"]["new_valid_slots_added_vs_K8"] > 0)
    k32_adds = bool(by_k["K32"]["new_valid_slots_added_vs_K8"] and by_k["K32"]["new_valid_slots_added_vs_K8"] > 0)
    k16_valid_ratio = float(by_k["K16"]["joint_supervision_coverage"])
    k32_valid_ratio = float(by_k["K32"]["joint_supervision_coverage"])
    trace_density_positive: str | bool = True if (k16_adds and k32_adds and k32_valid_ratio >= 0.5 * float(k8["joint_supervision_coverage"])) else "weak_or_inconclusive"
    dense_allowed = bool(trace_density_positive is True and k32_valid_ratio >= 0.5 * float(k8["joint_supervision_coverage"]))
    payload = {
        "audit_name": "stwm_fstf_trace_density_valid_units_audit_v13",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "observed_cache_path": str(obs_path),
        "by_K": by_k,
        "K16_actually_adds_semantic_supervision": k16_adds,
        "K32_actually_adds_semantic_supervision": k32_adds,
        "trace_density_scaling_positive": trace_density_positive,
        "dense_trace_field_claim_allowed": dense_allowed,
        "recommended_wording": "semantic trace-unit field" if not dense_allowed else "semi-dense semantic trace field under valid-unit audit",
        "density_claim_note": "K16/K32 are valid scaling experiments, but dense-field wording requires additional valid semantic units rather than mostly invalid added slots.",
    }
    _dump(Path("reports/stwm_fstf_trace_density_valid_units_audit_v13_20260502.json"), payload)
    _write_doc(Path("docs/STWM_FSTF_TRACE_DENSITY_VALID_UNITS_AUDIT_V13_20260502.md"), payload)
    print("reports/stwm_fstf_trace_density_valid_units_audit_v13_20260502.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
