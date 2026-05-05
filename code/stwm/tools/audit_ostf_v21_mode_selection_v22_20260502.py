#!/usr/bin/env python3
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))

from stwm.tools.ostf_multimodal_metrics_v22 import aggregate_rows_v22, calibration_summary, expected_vs_oracle_bootstrap, multimodal_item_scores_v22
from stwm.tools.ostf_v17_common_20260502 import ROOT, batch_from_samples, dump_json, load_json, write_doc
from stwm.tools.ostf_v18_common_20260502 import analytic_constant_velocity_predict
from stwm.tools.ostf_v20_common_20260502 import hard_subset_flags, load_context_cache, load_combo_rows, sample_key
from stwm.tools.train_ostf_multimodal_v21_20260502 import _build_model as _build_v21_model


REPORT_PATH = ROOT / "reports/stwm_ostf_v21_mode_selection_audit_v22_20260502.json"
DOC_PATH = ROOT / "docs/STWM_OSTF_V21_MODE_SELECTION_AUDIT_V22_20260502.md"


def _subset_flags(samples: list[Any], ctx_map: dict[tuple[str, int], dict[str, Any]]) -> dict[str, np.ndarray]:
    records = []
    for s in samples:
        c = ctx_map[sample_key(s)]
        records.append(
            {
                "cv_point_l1_proxy": c["cv_point_l1_proxy"],
                "curvature_proxy": c["curvature_proxy"],
                "occlusion_ratio": c["occlusion_ratio"],
                "interaction_proxy": c["interaction_proxy"],
            }
        )
    return hard_subset_flags(records)


def _context_batch(sample_rows: list[Any], ctx_map: dict[tuple[str, int], dict[str, Any]], device: torch.device) -> dict[str, torch.Tensor]:
    crop = []
    box = []
    neigh = []
    glob = []
    for s in sample_rows:
        c = ctx_map[sample_key(s)]
        crop.append(c["crop_feat"])
        box.append(c["box_feat"])
        neigh.append(c["neighbor_feat"])
        glob.append(c["global_feat"])
    return {
        "crop_feat": torch.tensor(np.stack(crop), device=device, dtype=torch.float32),
        "box_feat": torch.tensor(np.stack(box), device=device, dtype=torch.float32),
        "neighbor_feat": torch.tensor(np.stack(neigh), device=device, dtype=torch.float32),
        "global_feat": torch.tensor(np.stack(glob), device=device, dtype=torch.float32),
    }


def main() -> int:
    decision = load_json(ROOT / "reports/stwm_ostf_v21_decision_20260502.json")
    if not decision:
        raise SystemExit("Missing V21 decision report")
    best_name = str(decision["best_variant_name"])
    run_report = load_json(ROOT / f"reports/stwm_ostf_v21_runs/{best_name}.json")
    combo = str(run_report["source_combo"])
    ckpt = torch.load(ROOT / run_report["best_checkpoint_path"], map_location="cpu", weights_only=False)
    model = _build_v21_model(run_report["model_kind"], int(run_report["horizon"]))
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    device = torch.device("cpu")

    rows, _ = load_combo_rows(combo, seed=int(run_report.get("seed", 42)))
    test_samples = rows["test"]
    ctx_map = load_context_cache(ROOT / run_report["context_cache_path"])
    subset_flags = _subset_flags(test_samples, ctx_map)

    point_modes = []
    mode_logits = []
    point_pred = []
    top1_pred = []
    vis_logits = []
    sem_logits = []
    bs = 8 if "m128" in combo.lower() else 4
    with torch.no_grad():
        for start in range(0, len(test_samples), bs):
            batch_rows = test_samples[start : start + bs]
            batch = batch_from_samples(batch_rows, device)
            batch_ctx = _context_batch(batch_rows, ctx_map, device)
            out = model(
                obs_points=batch["obs_points"],
                obs_vis=batch["obs_vis"],
                rel_xy=batch["rel_xy"],
                anchor_obs=batch["anchor_obs"],
                anchor_obs_vel=batch["anchor_obs_vel"],
                semantic_feat=batch["semantic_feat"],
                crop_feat=batch_ctx["crop_feat"],
                box_feat=batch_ctx["box_feat"],
                neighbor_feat=batch_ctx["neighbor_feat"],
                global_feat=batch_ctx["global_feat"],
            )
            pm = out["point_hypotheses"].cpu().numpy()
            logits = out["hypothesis_logits"].cpu().numpy()
            point_modes.append(pm)
            mode_logits.append(logits)
            point_pred.append(out["point_pred"].cpu().numpy())
            vis_logits.append(out["visibility_logits"].cpu().numpy())
            sem_logits.append(out["semantic_logits"].cpu().numpy())
            top_idx = logits.argmax(axis=-1)
            gather = np.take_along_axis(pm, top_idx[:, None, None, None, None], axis=3).squeeze(3)
            top1_pred.append(gather)
    point_modes_np = np.concatenate(point_modes, axis=0)
    mode_logits_np = np.concatenate(mode_logits, axis=0)
    point_pred_np = np.concatenate(point_pred, axis=0)
    top1_pred_np = np.concatenate(top1_pred, axis=0)
    vis_logits_np = np.concatenate(vis_logits, axis=0)
    sem_logits_np = np.concatenate(sem_logits, axis=0)
    rows_v22 = multimodal_item_scores_v22(
        test_samples,
        point_modes=point_modes_np,
        mode_logits=mode_logits_np,
        point_pred=point_pred_np,
        top1_pred=top1_pred_np,
        pred_vis_logits=vis_logits_np,
        pred_proto_logits=sem_logits_np,
        subset_flags=subset_flags,
        cv_mode_index=0,
    )

    cv_points, cv_vis, _ = analytic_constant_velocity_predict(test_samples, proto_count=32, semantic_mode="observed_memory")
    cv_rows = multimodal_item_scores_v22(
        test_samples,
        point_modes=cv_points[:, :, :, None, :],
        mode_logits=np.zeros((cv_points.shape[0], 1), dtype=np.float32),
        point_pred=cv_points,
        top1_pred=cv_points,
        pred_vis_logits=cv_vis,
        pred_proto_logits=None,
        subset_flags=subset_flags,
        cv_mode_index=0,
    )

    oracle_best_dist = np.bincount(np.asarray([int(r["best_mode_idx_FDE"]) for r in rows_v22], dtype=np.int64), minlength=mode_logits_np.shape[1]).tolist()
    top1_dist = np.bincount(np.asarray([int(r["top1_mode_idx"]) for r in rows_v22], dtype=np.int64), minlength=mode_logits_np.shape[1]).tolist()

    payload = {
        "audit_name": "stwm_ostf_v21_mode_selection_audit_v22",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "best_variant_name": best_name,
        "point_hypotheses_exist": True,
        "deterministic_eval_used_weighted_average_point_pred": True,
        "oracle_best_mode_index_distribution": oracle_best_dist,
        "predicted_top1_mode_index_distribution": top1_dist,
        "top1_mode_accuracy_against_oracle_best": calibration_summary(rows_v22)["top1_mode_accuracy"],
        "top1_selected_mode_metrics": aggregate_rows_v22(rows_v22),
        "top1_selected_mode_hard_subset_metrics": aggregate_rows_v22(rows_v22, subset_key="top20_cv_hard"),
        "weighted_average_metrics": {
            "weighted_point_L1_px": aggregate_rows_v22(rows_v22)["weighted_point_l1_px"],
            "weighted_endpoint_error_px": aggregate_rows_v22(rows_v22)["weighted_endpoint_error_px"],
        },
        "expected_FDE_all_average": aggregate_rows_v22(rows_v22)["expected_FDE_px"],
        "expected_FDE_hard_subset": aggregate_rows_v22(rows_v22, subset_key="top20_cv_hard")["expected_FDE_px"],
        "mode_calibration": calibration_summary(rows_v22),
        "top1_vs_cv_all_endpoint_delta": aggregate_rows_v22(cv_rows)["top1_endpoint_error_px"] - aggregate_rows_v22(rows_v22)["top1_endpoint_error_px"],
        "top1_vs_cv_hard_endpoint_delta": aggregate_rows_v22(cv_rows, subset_key="top20_cv_hard")["top1_endpoint_error_px"] - aggregate_rows_v22(rows_v22, subset_key="top20_cv_hard")["top1_endpoint_error_px"],
        "best_of_k_vs_weighted_average": bool(load_json(ROOT / "reports/stwm_ostf_multimodal_eval_gap_v21_20260502.json").get("best_of_K_beats_weighted_average", False)),
        "best_of_k_vs_expected_fde_bootstrap": expected_vs_oracle_bootstrap(rows_v22),
        "best_of_k_vs_expected_fde_bootstrap_hard": expected_vs_oracle_bootstrap(rows_v22, subset_key="top20_cv_hard"),
        "whether_mode_logits_are_calibrated": bool(calibration_summary(rows_v22)["ece_top1_mode"] is not None and calibration_summary(rows_v22)["ece_top1_mode"] < 0.20),
        "whether_best_of_k_gain_is_usable_without_oracle": bool(
            calibration_summary(rows_v22)["top1_mode_accuracy"] > 0.30
            and aggregate_rows_v22(rows_v22)["top1_endpoint_error_px"] <= aggregate_rows_v22(cv_rows)["top1_endpoint_error_px"] * 1.10
        ),
        "nll_available": calibration_summary(rows_v22)["mode_nll_mean"] is not None,
    }
    dump_json(REPORT_PATH, payload)
    write_doc(
        DOC_PATH,
        "STWM OSTF V21 Mode Selection Audit V22",
        payload,
        [
            "best_variant_name",
            "top1_mode_accuracy_against_oracle_best",
            "expected_FDE_all_average",
            "expected_FDE_hard_subset",
            "whether_mode_logits_are_calibrated",
            "whether_best_of_k_gain_is_usable_without_oracle",
            "nll_available",
        ],
    )
    print(REPORT_PATH.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
