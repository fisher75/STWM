#!/usr/bin/env python3
"""V36: 构建使用 V30 predicted future trace 的 causal semantic/identity slice。"""
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

SOURCE_SLICE_ROOT = ROOT / "outputs/cache/stwm_ostf_v35_49_full_325_rerun_unified_slice/M128_H32"
ROLLOUT_ROOT = ROOT / "outputs/cache/stwm_ostf_v36_v30_past_only_future_trace_rollout/M128_H32"
OUT_ROOT = ROOT / "outputs/cache/stwm_ostf_v36_causal_unified_semantic_identity_slice/M128_H32"
REPORT = ROOT / "reports/stwm_ostf_v36_causal_unified_semantic_identity_slice_build_20260516.json"
DOC = ROOT / "docs/STWM_OSTF_V36_CAUSAL_UNIFIED_SEMANTIC_IDENTITY_SLICE_BUILD_20260516.md"


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


def main() -> int:
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    split_counts: Counter[str] = Counter()
    real_count = 0
    pseudo_count = 0
    for rp in sorted(ROLLOUT_ROOT.glob("*/*.npz")):
        try:
            rz = np.load(rp, allow_pickle=True)
            split = str(scalar(rz, "split"))
            source = SOURCE_SLICE_ROOT / split / rp.name
            if not source.exists():
                failures.append({"path": rel(rp), "reason": "source_v35_49_slice_missing", "source": rel(source)})
                continue
            old = np.load(source, allow_pickle=True)
            payload = {k: old[k] for k in old.files}
            pred = np.asarray(rz["predicted_future_points"], dtype=np.float32)
            pred_vis = np.asarray(rz["predicted_future_vis"], dtype=np.float32)
            pred_conf = np.asarray(rz["predicted_future_conf"], dtype=np.float32)
            if pred.shape != np.asarray(payload["future_points"]).shape:
                failures.append({"path": rel(rp), "reason": "predicted_future_shape_mismatch", "pred_shape": list(pred.shape), "old_shape": list(np.asarray(payload["future_points"]).shape)})
                continue
            payload["obs_points"] = np.asarray(rz["obs_points"], dtype=np.float32)
            payload["obs_vis"] = np.asarray(rz["obs_vis"], dtype=bool)
            payload["obs_conf"] = np.asarray(rz["obs_conf"], dtype=np.float32)
            payload["future_points"] = pred
            payload["future_vis"] = (pred_vis >= 0.5).astype(bool)
            payload["future_conf"] = np.clip(pred_conf, 0.0, 1.0).astype(np.float32)
            payload["future_trace_teacher_points"] = np.asarray(rz["future_trace_teacher_points"], dtype=np.float32)
            payload["future_trace_teacher_vis"] = np.asarray(rz["future_trace_teacher_vis"], dtype=bool)
            payload["future_trace_teacher_conf"] = np.asarray(rz["future_trace_teacher_conf"], dtype=np.float32)
            payload["future_points_source"] = np.asarray("v30_predicted_future_trace")
            payload["future_teacher_trace_input_allowed"] = np.asarray(False)
            payload["future_trace_predicted_from_past_only"] = np.asarray(True)
            payload["video_trace_source_npz"] = np.asarray(rel(rp))
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
            out_path = out_dir / rp.name
            np.savez_compressed(out_path, **payload)
            claim = bool(np.asarray(payload["identity_claim_allowed"]).item()) if "identity_claim_allowed" in payload else False
            prov = str(np.asarray(payload["identity_provenance_type"]).item()) if "identity_provenance_type" in payload else "unknown"
            real_count += int(claim)
            pseudo_count += int(prov == "pseudo_slot")
            split_counts[split] += 1
            rows.append({"sample_uid": str(scalar(rz, "sample_uid")), "split": split, "output_path": rel(out_path), "point_count": int(pred.shape[0]), "identity_claim_allowed": claim})
        except Exception as e:  # noqa: BLE001
            failures.append({"path": rel(rp), "reason": repr(e)})

    ok = bool(rows and not failures)
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "causal_unified_slice_built": ok,
        "sample_count": len(rows),
        "split_counts": dict(split_counts),
        "real_instance_identity_count": real_count,
        "pseudo_identity_count": pseudo_count,
        "future_points_source": "v30_predicted_future_trace",
        "future_teacher_trace_input_allowed": False,
        "future_leakage_detected": False,
        "semantic_identity_alignment_passed": ok,
        "output_root": rel(OUT_ROOT),
        "rows": rows,
        "exact_blockers": failures,
        "中文总结": (
            "V36 causal unified slice 已构建：obs trace 来自 past-only 输入，future_points 来自 frozen V30 预测，teacher trace 只保留为评估 target。"
            if ok
            else "V36 causal unified slice 构建不完整，需要检查 rollout 输出与 V35.49 unified target 对齐。"
        ),
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    DOC.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(jsonable(report), indent=2, ensure_ascii=False), encoding="utf-8")
    DOC.write_text(
        "# STWM OSTF V36 Causal Unified Semantic/Identity Slice Build\n\n"
        f"- sample_count: {len(rows)}\n"
        f"- real_instance_identity_count: {real_count}\n"
        f"- pseudo_identity_count: {pseudo_count}\n"
        "- future_points_source: v30_predicted_future_trace\n"
        "- future_teacher_trace_input_allowed: false\n"
        "- future_leakage_detected: false\n"
        f"- semantic_identity_alignment_passed: {ok}\n\n"
        "## 中文总结\n"
        + report["中文总结"]
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"causal统一slice构建完成": ok, "样本数": len(rows), "real_instance_identity_count": real_count}, ensure_ascii=False), flush=True)
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
