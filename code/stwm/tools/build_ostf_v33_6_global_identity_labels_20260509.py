#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))

from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, write_doc
from stwm.tools.ostf_v30_external_gt_schema_20260508 import utc_now


SRC = ROOT / "outputs/cache/stwm_ostf_v33_semantic_identity_targets/pointodyssey"
OUT = ROOT / "outputs/cache/stwm_ostf_v33_6_global_identity_labels/pointodyssey"
REPORT = ROOT / "reports/stwm_ostf_v33_6_global_identity_label_build_20260509.json"
DOC = ROOT / "docs/STWM_OSTF_V33_6_GLOBAL_IDENTITY_LABEL_BUILD_20260509.md"


def main() -> int:
    key_to_id: dict[str, int] = {}
    collision_count = 0
    processed = 0
    valid = 0
    total = 0
    by_split = {}
    for split in ("train", "val", "test"):
        out_dir = OUT / split
        out_dir.mkdir(parents=True, exist_ok=True)
        split_count = 0
        for path in sorted((SRC / split).glob("*_M128_H32.npz")):
            z = np.load(path, allow_pickle=True)
            uid = str(np.asarray(z["sample_uid"]).item())
            dataset = str(np.asarray(z["dataset"]).item()) if "dataset" in z.files else "pointodyssey"
            fut = np.asarray(z["fut_instance_id"], dtype=np.int64)
            obs = np.asarray(z["obs_instance_id"], dtype=np.int64)
            point = np.asarray(z["point_to_instance_id"], dtype=np.int64)
            fmask = np.asarray(z["fut_instance_available_mask"]).astype(bool) & (fut >= 0)
            out_fut = np.full_like(fut, -1, dtype=np.int64)
            out_obs = np.full_like(obs, -1, dtype=np.int64)
            out_point = np.full_like(point, -1, dtype=np.int64)
            sample_map: dict[str, int] = {}
            for lab in np.unique(fut[fmask]):
                key = f"{dataset}|{split}|{uid}|instance_{int(lab)}"
                if key not in key_to_id:
                    key_to_id[key] = len(key_to_id)
                sample_map[key] = key_to_id[key]
                out_fut[fut == lab] = key_to_id[key]
                out_obs[obs == lab] = key_to_id[key]
                out_point[point == lab] = key_to_id[key]
            np.savez_compressed(
                out_dir / f"{uid}.npz",
                sample_uid=np.asarray(uid),
                dataset=np.asarray(dataset),
                split=np.asarray(split),
                fut_global_instance_id=out_fut,
                fut_global_instance_available_mask=fmask.astype(bool),
                obs_global_instance_id=out_obs,
                point_global_instance_id=out_point,
                global_identity_key_map=np.asarray(json.dumps(sample_map, sort_keys=True)),
                identity_label_namespace=np.asarray("sample_uid_instance"),
                leakage_safe=np.asarray(True),
                future_global_labels_supervision_only=np.asarray(True),
            )
            processed += 1
            split_count += 1
            total += int(fut.size)
            valid += int(fmask.sum())
        by_split[split] = {"samples_processed": split_count}
    payload = {
        "generated_at_utc": utc_now(),
        "global_identity_labels_built": True,
        "samples_processed": processed,
        "coverage": float(valid / max(total, 1)),
        "global_label_collision_count": collision_count,
        "global_identity_count": len(key_to_id),
        "label_namespace": "sample_uid_instance",
        "leakage_safe": True,
        "future_global_labels_supervision_only": True,
        "by_split": by_split,
    }
    dump_json(REPORT, payload)
    write_doc(DOC, "STWM OSTF V33.6 Global Identity Label Build", payload, ["global_identity_labels_built", "samples_processed", "coverage", "global_label_collision_count", "label_namespace", "leakage_safe"])
    print(REPORT.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
