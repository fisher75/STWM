#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))

from stwm.tools.ostf_lastobs_v28_common_20260502 import (
    ROOT,
    add_v28_flags_to_item_rows,
    build_v28_rows,
    choose_visibility_aware_gamma_on_val,
    predict_last_visible_copy,
    predict_median_object_anchor_copy,
    predict_visibility_aware_cv,
    predict_visibility_aware_damped_velocity,
    v28_subset_aggregate,
    visibility_logits_last_visible,
)
from stwm.tools.ostf_traceanything_metrics_v26_20260502 import (
    aggregate_item_rows_v26,
    multimodal_item_scores_v26,
    paired_bootstrap_from_rows_v26,
)
from stwm.tools.ostf_v17_common_20260502 import dump_json, write_doc
from stwm.tools.ostf_v27_prior_utils_20260502 import observed_memory_logits


RUN_DIR = ROOT / "reports/stwm_ostf_v28_runs"
REPORT_PATH = ROOT / "reports/stwm_ostf_v28_stronger_prior_eval_20260507.json"
BOOT_PATH = ROOT / "reports/stwm_ostf_v28_stronger_prior_bootstrap_20260507.json"
DOC_PATH = ROOT / "docs/STWM_OSTF_V28_STRONGER_PRIOR_EVAL_20260507.md"


def _load_runs() -> dict[str, dict[str, Any]]:
    out = {}
    for path in sorted(RUN_DIR.glob("*.json")):
        x = json.loads(path.read_text(encoding="utf-8"))
        out[x["experiment_name"]] = x
    return out


def _rows_for_pred(samples: list[Any], proto_centers: np.ndarray, pred: np.ndarray) -> list[dict[str, Any]]:
    rows = multimodal_item_scores_v26(
        samples,
        point_modes=pred[:, :, :, None, :],
        mode_logits=np.zeros((len(samples), 1), dtype=np.float32),
        top1_point_pred=pred,
        weighted_point_pred=pred,
        pred_vis_logits=visibility_logits_last_visible(samples),
        pred_proto_logits=observed_memory_logits(samples, proto_centers, proto_count=32),
        pred_logvar=None,
        cv_mode_index=0,
    )
    return add_v28_flags_to_item_rows(rows, samples)


def _prior_suite(combo: str) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    split_rows, proto_centers, _, _ = build_v28_rows(combo, seed=42)
    train_rows, val_rows, test_rows = split_rows["train"], split_rows["val"], split_rows["test"]
    gamma, gamma_scores = choose_visibility_aware_gamma_on_val(val_rows, proto_centers)
    preds = {
        "last_visible_copy": predict_last_visible_copy(test_rows),
        "visibility_aware_damped_velocity": predict_visibility_aware_damped_velocity(test_rows, gamma),
        "visibility_aware_cv": predict_visibility_aware_cv(test_rows),
        "median_object_anchor_copy": predict_median_object_anchor_copy(test_rows),
    }
    item_rows = {name: _rows_for_pred(test_rows, proto_centers, pred) for name, pred in preds.items()}
    summary = {
        "combo": combo,
        "val_selected_visibility_aware_gamma": gamma,
        "val_gamma_scores": gamma_scores,
        "priors": {
            name: {
                "test_metrics": aggregate_item_rows_v26(rows),
                "test_subset_metrics": v28_subset_aggregate(rows),
                "test_metrics_by_dataset": {ds: aggregate_item_rows_v26(rows, dataset=ds) for ds in sorted({s.dataset for s in test_rows})},
            }
            for name, rows in item_rows.items()
        },
    }
    return summary, item_rows


def _boot(bootstrap: dict[str, Any], key: str, model_rows: list[dict[str, Any]], prior_rows: list[dict[str, Any]], subset_key: str | None) -> None:
    bootstrap[f"{key}_minfde"] = paired_bootstrap_from_rows_v26(model_rows, prior_rows, metric="minFDE_K_px", higher_better=False, subset_key=subset_key)
    bootstrap[f"{key}_miss32"] = paired_bootstrap_from_rows_v26(model_rows, prior_rows, metric="MissRate_32px", higher_better=False, subset_key=subset_key)


def main() -> int:
    runs = _load_runs()
    combos = {
        "M128_H32": "v28_lastobs_m128_h32_seed42",
        "M512_H32": "v28_lastobs_m512_h32_seed42",
        "M128_H64": "v28_lastobs_m128_h64_seed42",
    }
    eval_payload: dict[str, Any] = {
        "audit_name": "stwm_ostf_v28_stronger_prior_eval",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "combos": {},
    }
    bootstrap: dict[str, Any] = {
        "audit_name": "stwm_ostf_v28_stronger_prior_bootstrap",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    for combo, run_name in combos.items():
        if run_name not in runs:
            eval_payload["combos"][combo] = {"missing_model_run": run_name}
            continue
        summary, prior_rows = _prior_suite(combo)
        eval_payload["combos"][combo] = summary
        model_rows = runs[run_name]["item_scores"]
        for prior_name, rows in prior_rows.items():
            _boot(bootstrap, f"{combo}_{prior_name}_all", model_rows, rows, None)
            _boot(bootstrap, f"{combo}_{prior_name}_last_observed_hard", model_rows, rows, "last_observed_hard_top20")
    dump_json(REPORT_PATH, eval_payload)
    dump_json(BOOT_PATH, bootstrap)
    write_doc(
        DOC_PATH,
        "STWM OSTF V28 Stronger Prior Eval",
        {
            "generated_at_utc": eval_payload["generated_at_utc"],
            "combos_evaluated": list(eval_payload["combos"].keys()),
            "stronger_priors": ["last_visible_copy", "visibility_aware_damped_velocity", "visibility_aware_cv", "median_object_anchor_copy"],
            "bootstrap_path": str(BOOT_PATH.relative_to(ROOT)),
        },
        ["combos_evaluated", "stronger_priors", "bootstrap_path"],
    )
    print(REPORT_PATH.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
