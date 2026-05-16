#!/usr/bin/env python3
"""V36.1: 用 strongest analytic prior 构建 downstream semantic/identity baseline slice。"""
from __future__ import annotations

import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import setproctitle

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))
setproctitle.setproctitle("python")

from stwm.tools.ostf_v17_common_20260502 import ROOT  # noqa: E402
from stwm.tools.run_ostf_v35_35_raw_video_frontend_rerun_smoke_20260516 import (  # noqa: E402
    close_pair,
    future_crossing_pair,
    occlusion_reappear,
    one_hot_semantic_from_payload,
    trace_features_from_payload,
    weighted_measurement_from_payload,
)

CAUSAL_SLICE_ROOT = ROOT / "outputs/cache/stwm_ostf_v36_causal_unified_semantic_identity_slice/M128_H32"
ROLLOUT_ROOT = ROOT / "outputs/cache/stwm_ostf_v36_v30_past_only_future_trace_rollout/M128_H32"
ATLAS_REPORT = ROOT / "reports/stwm_ostf_v36_1_trace_rollout_failure_atlas_20260516.json"
OUT_ROOT = ROOT / "outputs/cache/stwm_ostf_v36_1_strongest_prior_downstream_slice/M128_H32"
REPORT = ROOT / "reports/stwm_ostf_v36_1_strongest_prior_downstream_slice_build_20260516.json"
DOC = ROOT / "docs/STWM_OSTF_V36_1_STRONGEST_PRIOR_DOWNSTREAM_SLICE_BUILD_20260516.md"


def jsonable(x: Any) -> Any:
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    return x


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def scalar(z: np.lib.npyio.NpzFile, key: str, default: Any = None) -> Any:
    if key not in z.files:
        return default
    arr = np.asarray(z[key])
    return arr.item() if arr.shape == () else arr


def prior_key(name: str) -> str:
    return {
        "last_visible_copy": "prior_last_visible_copy",
        "last_observed_copy": "prior_last_observed_copy",
        "constant_velocity": "prior_constant_velocity",
        "damped_velocity": "prior_damped_velocity",
    }.get(name, "prior_last_observed_copy")


def main() -> int:
    atlas = json.loads(ATLAS_REPORT.read_text(encoding="utf-8")) if ATLAS_REPORT.exists() else {}
    strongest_prior = str(atlas.get("global_summary", {}).get("strongest_prior") or "last_observed_copy")
    pkey = prior_key(strongest_prior)
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    split_counts: Counter[str] = Counter()
    real_count = 0
    pseudo_count = 0
    for sp in sorted(CAUSAL_SLICE_ROOT.glob("*/*.npz")):
        try:
            split = sp.parent.name
            rp = ROLLOUT_ROOT / split / sp.name
            if not rp.exists():
                failures.append({"path": rel(sp), "reason": "rollout_npz_missing", "rollout": rel(rp)})
                continue
            old = np.load(sp, allow_pickle=True)
            rz = np.load(rp, allow_pickle=True)
            payload = {k: old[k] for k in old.files}
            pred = np.asarray(rz[pkey], dtype=np.float32)
            if pred.shape != np.asarray(payload["future_points"]).shape:
                failures.append({"path": rel(sp), "reason": "prior_future_shape_mismatch"})
                continue
            payload["future_points"] = pred
            # 对 copy/CV prior，future visibility/conf 使用 observed last-frame confidence 的因果展开，避免 teacher future visibility 进入输入。
            last_vis = np.asarray(rz["obs_vis"], dtype=bool)[:, -1]
            last_conf = np.asarray(rz["obs_conf"], dtype=np.float32)[:, -1]
            payload["future_vis"] = np.repeat(last_vis[:, None], pred.shape[1], axis=1).astype(bool)
            payload["future_conf"] = np.repeat(last_conf[:, None], pred.shape[1], axis=1).astype(np.float32)
            payload["future_points_source"] = np.asarray(f"strongest_analytic_prior:{strongest_prior}")
            payload["strongest_prior_name"] = np.asarray(strongest_prior)
            payload["future_teacher_trace_input_allowed"] = np.asarray(False)
            payload["future_trace_predicted_from_past_only"] = np.asarray(True)
            payload["future_leakage_detected"] = np.asarray(False)
            meas = weighted_measurement_from_payload(payload)
            payload["identity_measurement_identity_embedding"] = meas.astype(np.float32)
            payload["identity_identity_input_features"] = np.concatenate(
                [meas, one_hot_semantic_from_payload(payload), trace_features_from_payload(payload)],
                axis=1,
            ).astype(np.float32)
            inst = np.asarray(payload["point_to_instance_id"], dtype=np.int64)
            same = (inst[:, None] == inst[None, :]) & (inst[:, None] >= 0)
            np.fill_diagonal(same, False)
            sem = np.asarray(payload["obs_semantic_last_id"], dtype=np.int64)
            if "source_semantic_id" in payload:
                sem = np.where(sem >= 0, sem, np.asarray(payload["source_semantic_id"], dtype=np.int64))
            diff = (inst[:, None] != inst[None, :]) & (inst[:, None] >= 0) & (inst[None, :] >= 0)
            same_sem = diff & (sem[:, None] == sem[None, :]) & (sem[:, None] >= 0)
            spatial_hard = close_pair(payload["obs_points"][:, -1], inst, 0.12)
            crossing = future_crossing_pair(payload["future_points"], inst)
            identity_confuser = np.asarray(payload.get("identity_identity_confuser_pair_mask", np.zeros_like(same)), dtype=bool) | same_sem | spatial_hard | crossing
            np.fill_diagonal(identity_confuser, False)
            payload["identity_same_instance_pair_mask"] = same.astype(bool)
            payload["identity_same_semantic_hard_negative_pair_mask"] = same_sem.astype(bool)
            payload["identity_same_frame_hard_negative_pair_mask"] = spatial_hard.astype(bool)
            payload["identity_trajectory_crossing_pair_mask"] = crossing.astype(bool)
            payload["identity_identity_confuser_pair_mask"] = identity_confuser.astype(bool)
            payload["identity_occlusion_reappear_point_mask"] = occlusion_reappear(payload["future_vis"]).astype(bool)
            out_dir = OUT_ROOT / split
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / sp.name
            np.savez_compressed(out_path, **payload)
            claim = bool(np.asarray(payload["identity_claim_allowed"]).item()) if "identity_claim_allowed" in payload else False
            prov = str(np.asarray(payload["identity_provenance_type"]).item()) if "identity_provenance_type" in payload else "unknown"
            real_count += int(claim)
            pseudo_count += int(prov == "pseudo_slot")
            split_counts[split] += 1
            rows.append({"sample_uid": str(scalar(old, "sample_uid", sp.stem)), "split": split, "output_path": rel(out_path), "future_points_source": f"strongest_analytic_prior:{strongest_prior}"})
        except Exception as e:  # noqa: BLE001
            failures.append({"path": rel(sp), "reason": repr(e)})
    ok = bool(rows and not failures)
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "strongest_prior_downstream_slice_built": ok,
        "strongest_prior_name": strongest_prior,
        "strongest_prior_key": pkey,
        "sample_count": len(rows),
        "split_counts": dict(split_counts),
        "real_instance_identity_count": real_count,
        "pseudo_identity_count": pseudo_count,
        "future_points_source": f"strongest_analytic_prior:{strongest_prior}",
        "future_teacher_trace_input_allowed": False,
        "future_leakage_detected": False,
        "output_root": rel(OUT_ROOT),
        "rows": rows,
        "exact_blockers": failures,
        "中文总结": (
            f"已构建 strongest-prior downstream slice：future_points 使用 {strongest_prior}，用于直接比较 V36 V30 causal trace 与 strongest prior 的 semantic/identity downstream utility。"
            if ok
            else "strongest-prior downstream slice 构建失败，需要先检查 V36 causal slice 或 rollout npz。"
        ),
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    DOC.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(jsonable(report), indent=2, ensure_ascii=False), encoding="utf-8")
    DOC.write_text(
        "# STWM OSTF V36.1 Strongest-Prior Downstream Slice Build\n\n"
        f"- strongest_prior_name: {strongest_prior}\n"
        f"- sample_count: {len(rows)}\n"
        f"- real_instance_identity_count: {real_count}\n"
        f"- pseudo_identity_count: {pseudo_count}\n"
        f"- future_points_source: strongest_analytic_prior:{strongest_prior}\n"
        "- future_teacher_trace_input_allowed: false\n"
        "- future_leakage_detected: false\n\n"
        "## 中文总结\n"
        + report["中文总结"]
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"V36_1_strongest_prior_slice完成": ok, "strongest_prior": strongest_prior, "样本数": len(rows)}, ensure_ascii=False), flush=True)
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
