#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))

from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, write_doc
from stwm.tools.ostf_v30_external_gt_schema_20260508 import utc_now
from stwm.tools.ostf_v33_11_common_20260510 import (
    BASELINE_NAMES,
    BASELINE_ROOT,
    PROTO_ROOT,
    SUBSETS,
    baseline_arrays_for_sample,
    load_manifest_mask,
    topk_from_scores,
    train_global_distribution,
    V33_11_MASK_ROOT,
)


REPORT = ROOT / "reports/stwm_ostf_v33_11_semantic_baseline_bank_20260510.json"
DOC = ROOT / "docs/STWM_OSTF_V33_11_SEMANTIC_BASELINE_BANK_20260510.md"


def main() -> int:
    k = 32
    train_global = train_global_distribution(k)
    metrics: dict[str, dict[str, list[float]]] = {split: {f"{name}_{subset}_top{kk}": [] for name in BASELINE_NAMES for subset in SUBSETS for kk in (1, 5)} for split in ("train", "val", "test")}
    blockers: list[str] = []
    for split in ("train", "val", "test"):
        (BASELINE_ROOT / split).mkdir(parents=True, exist_ok=True)
        for path in sorted((PROTO_ROOT / split).glob("*.npz")):
            z = np.load(path, allow_pickle=True)
            uid = str(np.asarray(z["sample_uid"]).item())
            target = np.asarray(z["semantic_prototype_id"], dtype=np.int64)
            mask = np.asarray(z["semantic_prototype_available_mask"]).astype(bool)
            obs = np.asarray(z["obs_semantic_prototype_id"], dtype=np.int64)
            obs_mask = np.asarray(z["obs_semantic_prototype_available_mask"]).astype(bool)
            arrays = baseline_arrays_for_sample(z, train_global)
            copy = arrays["copy_semantic_prototype_id"]
            stable = mask & (copy == target) & (copy >= 0)
            changed = mask & (copy != target) & (copy >= 0)
            semantic_hard = load_manifest_mask(V33_11_MASK_ROOT, 42, split, uid, target.shape, "semantic_hard_eval_mask") & mask
            subsets = {"global": mask, "stable": stable, "changed": changed, "semantic_hard": semantic_hard}
            save_payload = {
                "sample_uid": np.asarray(uid),
                "dataset": np.asarray(str(np.asarray(z["dataset"]).item()) if "dataset" in z.files else "pointodyssey"),
                "split": np.asarray(split),
                "baseline_names": np.asarray(BASELINE_NAMES),
                "baseline_available_mask": mask,
                "stable_mask": stable,
                "changed_mask": changed,
                "leakage_safe": np.asarray(True),
                "future_teacher_input_allowed": np.asarray(False),
            }
            for name in BASELINE_NAMES:
                dist = arrays[name].astype(np.float32)
                save_payload[f"{name}_distribution"] = dist
                logits = np.log(dist.clip(1e-8, 1.0))
                for subset, smask in subsets.items():
                    for kk in (1, 5):
                        val = topk_from_scores(logits, target, smask, kk)
                        if val is not None:
                            metrics[split][f"{name}_{subset}_top{kk}"].append(val)
            np.savez_compressed(BASELINE_ROOT / split / f"{uid}.npz", **save_payload)
    agg: dict[str, dict[str, Any]] = {}
    for split, rows in metrics.items():
        agg[split] = {k2: (float(np.mean(v)) if v else None) for k2, v in rows.items()}
    selected: dict[str, str] = {}
    for subset in SUBSETS:
        best_name = "last_observed_copy"
        best_val = -1.0
        for name in BASELINE_NAMES:
            val = agg["val"].get(f"{name}_{subset}_top5")
            if val is not None and float(val) > best_val:
                best_val = float(val)
                best_name = name
        selected[subset] = best_name
    payload = {
        "generated_at_utc": utc_now(),
        "baseline_bank_ready": not blockers,
        "output_root": str(BASELINE_ROOT.relative_to(ROOT)),
        "baseline_names": BASELINE_NAMES,
        "baseline_metrics_by_split": agg,
        "strongest_baseline_by_subset_selected_on_val": selected,
        "global_strongest_baseline": selected["global"],
        "stable_strongest_baseline": selected["stable"],
        "changed_strongest_baseline": selected["changed"],
        "semantic_hard_strongest_baseline": selected["semantic_hard"],
        "val_selection_only": True,
        "exact_blockers": blockers,
    }
    dump_json(REPORT, payload)
    write_doc(DOC, "STWM OSTF V33.11 Semantic Baseline Bank", payload, ["baseline_bank_ready", "output_root", "strongest_baseline_by_subset_selected_on_val", "global_strongest_baseline", "stable_strongest_baseline", "changed_strongest_baseline", "semantic_hard_strongest_baseline", "val_selection_only", "exact_blockers"])
    print(REPORT.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
