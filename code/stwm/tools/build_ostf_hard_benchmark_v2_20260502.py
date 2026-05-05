#!/usr/bin/env python3
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from stwm.tools.ostf_v17_common_20260502 import ROOT, load_v16_samples, dump_json, write_doc
from stwm.tools.ostf_v20_common_20260502 import context_features_for_sample, hard_subset_flags
from stwm.tools.run_cotracker_object_dense_teacher_v15c_20260502 import _frame_sequence


REPORT_PATH = ROOT / "reports/stwm_ostf_hard_benchmark_v2_20260502.json"
DOC_PATH = ROOT / "docs/STWM_OSTF_HARD_BENCHMARK_V2_20260502.md"


def _scalar(x: Any) -> Any:
    arr = np.asarray(x)
    return arr.item() if arr.shape == () else arr.reshape(-1)[0]


def _horizon_status(pre_path: Path, horizon: int, obs_len: int = 8) -> dict[str, Any]:
    z = np.load(pre_path, allow_pickle=True)
    anchor = Path(str(_scalar(z["semantic_frame_path"])))
    frames, query_frame, err = _frame_sequence(anchor, total=obs_len + horizon, preferred_query_frame=obs_len - 1)
    return {
        "feasible": bool(frames is not None and err is None),
        "query_frame": int(query_frame) if query_frame is not None else None,
        "reason": err,
        "frame_count": len(frames) if frames is not None else 0,
        "raw_frame_available": bool(anchor.exists()),
    }


def main() -> int:
    rows = load_v16_samples("M128_H8")
    samples = rows["train"] + rows["val"] + rows["test"]
    context_records: list[dict[str, Any]] = []
    for sample in samples:
        feats = context_features_for_sample(sample)
        context_records.append(
            {
                "item_key": sample.item_key,
                "object_id": int(sample.object_id),
                "dataset": sample.dataset,
                "split": sample.split,
                **feats,
            }
        )
    flags = hard_subset_flags(context_records)
    per_item: dict[str, dict[str, Any]] = {}
    for idx, record in enumerate(context_records):
        item_key = record["item_key"]
        row = per_item.setdefault(
            item_key,
            {
                "item_key": item_key,
                "clip_id": item_key.split("::", 1)[1],
                "dataset": record["dataset"],
                "split": record["split"],
                "reason_tags": set(),
                "object_count": 0,
            },
        )
        row["object_count"] += 1
        for name, arr in flags.items():
            if bool(arr[idx]):
                row["reason_tags"].add(name)
    for item_key, row in per_item.items():
        sample = next(s for s in samples if s.item_key == item_key)
        cache = np.load(ROOT / sample.source_cache_path, allow_pickle=True)
        pre_path = Path(str(_scalar(cache["predecode_path"])))
        pre = np.load(pre_path, allow_pickle=True)
        row["raw_frame_path"] = str(_scalar(pre["semantic_frame_path"]))
        row["raw_frame_available"] = bool(Path(row["raw_frame_path"]).exists())
        row["semantic_instance_available"] = bool("semantic_instance_id_map" in pre.files)
        row["instance_track_boxes_available"] = bool("entity_boxes_over_time" in pre.files)
        row["horizon_status"] = {
            f"H{h}": _horizon_status(pre_path, h) for h in [8, 16, 32, 64]
        }
        row["reason_tags"] = sorted(row["reason_tags"])

    subsets = defaultdict(list)
    for row in per_item.values():
        for tag in row["reason_tags"]:
            subsets[tag].append(row["item_key"])

    h_blockers = {f"H{h}": Counter() for h in [32, 64]}
    for row in per_item.values():
        for h in [32, 64]:
            st = row["horizon_status"][f"H{h}"]
            if not st["feasible"]:
                h_blockers[f"H{h}"][str(st["reason"] or "unknown")] += 1

    payload = {
        "audit_name": "stwm_ostf_hard_benchmark_v2",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_combo": "V16_M128_H8_real_cotracker_teacher_cache",
        "item_count": len(per_item),
        "per_dataset_counts": dict(Counter(row["dataset"] for row in per_item.values())),
        "per_split_counts": dict(Counter(row["split"] for row in per_item.values())),
        "subset_counts": {k: len(v) for k, v in subsets.items()},
        "subsets": {k: sorted(v) for k, v in subsets.items()},
        "per_item": {k: {**v, "reason_tags": list(v["reason_tags"])} for k, v in sorted(per_item.items())},
        "horizon_feasibility_summary": {
            f"H{h}": {
                "feasible_item_count": int(sum(1 for row in per_item.values() if row["horizon_status"][f"H{h}"]["feasible"])),
                "failed_item_count": int(sum(1 for row in per_item.values() if not row["horizon_status"][f"H{h}"]["feasible"])),
                "top_blockers": dict(h_blockers.get(f"H{h}", {})),
            }
            for h in [8, 16, 32, 64]
        },
        "h32_h64_feasible": bool(
            sum(1 for row in per_item.values() if row["horizon_status"]["H32"]["feasible"]) > 0
            and sum(1 for row in per_item.values() if row["horizon_status"]["H64"]["feasible"]) > 0
        ),
        "hard_benchmark_ready": bool(
            len(subsets.get("top20_cv_hard", [])) > 0
            and len(subsets.get("occlusion_hard", [])) > 0
            and len(subsets.get("nonlinear_hard", [])) > 0
            and len(subsets.get("interaction_hard", [])) > 0
        ),
        "notes": {
            "cv_hard_source": "existing CoTracker-teacher CV proxies from V20 context feature pipeline",
            "semantic_instance_and_raw_frame_availability_audited": True,
            "H32_H64_blockers_reflect_contiguous_window_feasibility_from_current_semantic_frame_anchor_policy": True,
        },
    }
    dump_json(REPORT_PATH, payload)
    write_doc(
        DOC_PATH,
        "STWM OSTF Hard Benchmark V2",
        payload,
        ["item_count", "per_dataset_counts", "subset_counts", "horizon_feasibility_summary", "h32_h64_feasible", "hard_benchmark_ready"],
    )
    print(REPORT_PATH.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
