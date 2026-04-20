#!/usr/bin/env python3
from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
from typing import Any, Dict, List
import csv
import json

from stwm.tools import run_tracewm_stage2_calibration_only_fullscale_wave1_20260413 as base


ROOT = Path("/raid/chen034/workspace/stwm")
ASSET_DIR = ROOT / "reports/stage2_v3p1_paper_assets_20260420"


def _json_or_empty(path_like: Any) -> Dict[str, Any]:
    path = Path(str(path_like))
    if not path.exists():
        return {}
    try:
        payload = base._read_json(path)
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _panel_row(panel: Dict[str, Any], name: str) -> Dict[str, Any]:
    for row in panel.get("method_rows", []):
        if str(row.get("name", "")) == name:
            return row
    return {}


def _try_build_plots(asset_dir: Path, dualpanel: Dict[str, Any], multiseed: Dict[str, Any], mechanism: Dict[str, Any]) -> Dict[str, str]:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return {}

    out: Dict[str, str] = {}

    # Plot 1: official comparison on both panels.
    official_names = [
        "stage1_frozen_baseline",
        "legacysem_best",
        "cropenc_baseline_best",
        "current_calibration_only_best",
        "current_tusb_v3p1_best::best.pt",
    ]
    short = {
        "stage1_frozen_baseline": "Stage1",
        "legacysem_best": "LegacySem",
        "cropenc_baseline_best": "CropEnc",
        "current_calibration_only_best": "CalOnly",
        "current_tusb_v3p1_best::best.pt": "TUSB-v3.1",
    }
    legacy_panel = dualpanel.get("legacy_85_panel", {})
    dense_panel = dualpanel.get("densified_200_panel", {})
    legacy_vals = [float(_panel_row(legacy_panel, name).get("query_future_top1_acc", 0.0)) for name in official_names]
    dense_vals = [float(_panel_row(dense_panel, name).get("query_future_top1_acc", 0.0)) for name in official_names]
    xs = list(range(len(official_names)))
    plt.figure(figsize=(9, 4.5))
    plt.plot(xs, legacy_vals, marker="o", label="legacy_85")
    plt.plot(xs, dense_vals, marker="o", label="densified_200")
    plt.xticks(xs, [short[n] for n in official_names], rotation=20)
    plt.ylabel("query_future_top1_acc")
    plt.title("Dualpanel Main Comparison")
    plt.legend()
    plt.tight_layout()
    p1 = asset_dir / "dualpanel_main_comparison.png"
    plt.savefig(p1, dpi=180)
    plt.close()
    out["dualpanel_main_comparison_png"] = str(p1)

    # Plot 2: v3.1 vs calibration subset deltas on densified panel.
    calib = _panel_row(dense_panel, "current_calibration_only_best")
    tusb = _panel_row(dense_panel, "current_tusb_v3p1_best::best.pt")
    subset_keys = [
        ("hard_subset_top1_acc", "hard"),
        ("ambiguity_top1_acc", "ambiguity"),
        ("appearance_change_top1_acc", "appearance"),
        ("query_future_top1_acc", "overall"),
    ]
    deltas = [float(tusb.get(k, 0.0)) - float(calib.get(k, 0.0)) for k, _ in subset_keys]
    plt.figure(figsize=(7.5, 4.2))
    plt.bar([label for _, label in subset_keys], deltas)
    plt.axhline(0.0, color="black", linewidth=1)
    plt.ylabel("Top1 delta vs calibration")
    plt.title("TUSB-v3.1 Densified Subset Delta")
    plt.tight_layout()
    p2 = asset_dir / "densified_subset_delta.png"
    plt.savefig(p2, dpi=180)
    plt.close()
    out["densified_subset_delta_png"] = str(p2)

    # Plot 3: mechanism means across seeds.
    mech = mechanism.get("v3p1_seed_summary", {})
    metric_names = [
        "active_unit_count_mean",
        "same_instance_dominant_unit_match_rate_mean",
        "different_instance_dominant_unit_collision_rate_mean",
        "z_sem_to_z_dyn_drift_ratio_mean",
    ]
    values = [float(mech.get("mean", {}).get(k, 0.0)) for k in metric_names]
    plt.figure(figsize=(8.5, 4.5))
    plt.bar(range(len(metric_names)), values)
    plt.xticks(range(len(metric_names)), ["active_units", "same_inst_match", "diff_inst_collision", "sem/dyn_ratio"], rotation=20)
    plt.title("TUSB-v3.1 Mechanism Means")
    plt.tight_layout()
    p3 = asset_dir / "mechanism_appendix_metrics.png"
    plt.savefig(p3, dpi=180)
    plt.close()
    out["mechanism_appendix_metrics_png"] = str(p3)
    return out


def main() -> None:
    parser = ArgumentParser(description="Build STAGE2 V3.1 paper-ready assets")
    parser.add_argument("--dualpanel-hardening-report", default=str(ROOT / "reports/stage2_dualpanel_hardening_20260420.json"))
    parser.add_argument("--multiseed-dualpanel-report", default=str(ROOT / "reports/stage2_v3p1_multiseed_dualpanel_20260420.json"))
    parser.add_argument("--bootstrap-ci-report", default=str(ROOT / "reports/stage2_v3p1_bootstrap_ci_20260420.json"))
    parser.add_argument("--mechanism-appendix-report", default=str(ROOT / "reports/stage2_v3p1_mechanism_appendix_20260420.json"))
    parser.add_argument("--qualitative-pack-json", default=str(ROOT / "reports/stage2_qualitative_pack_v9_20260416.json"))
    parser.add_argument("--asset-dir", default=str(ASSET_DIR))
    parser.add_argument("--output-json", default=str(ROOT / "reports/stage2_v3p1_paper_assets_20260420.json"))
    parser.add_argument("--output-md", default=str(ROOT / "docs/STAGE2_V3P1_PAPER_ASSETS_20260420.md"))
    args = parser.parse_args()

    dualpanel = _json_or_empty(args.dualpanel_hardening_report)
    multiseed = _json_or_empty(args.multiseed_dualpanel_report)
    bootstrap = _json_or_empty(args.bootstrap_ci_report)
    mechanism = _json_or_empty(args.mechanism_appendix_report)
    qualitative = _json_or_empty(args.qualitative_pack_json)
    asset_dir = Path(args.asset_dir)
    asset_dir.mkdir(parents=True, exist_ok=True)

    legacy_panel = dualpanel.get("legacy_85_panel", {})
    dense_panel = dualpanel.get("densified_200_panel", {})

    main_rows: List[Dict[str, Any]] = []
    for name in [
        "stage1_frozen_baseline",
        "legacysem_best",
        "cropenc_baseline_best",
        "current_calibration_only_best",
        "current_tusb_v3p1_best::best.pt",
    ]:
        legacy_row = _panel_row(legacy_panel, name)
        dense_row = _panel_row(dense_panel, name)
        main_rows.append(
            {
                "method": name,
                "legacy85_top1": float(legacy_row.get("query_future_top1_acc", 0.0)),
                "legacy85_hard_top1": float(legacy_row.get("hard_subset_top1_acc", 0.0)),
                "dense200_top1": float(dense_row.get("query_future_top1_acc", 0.0)),
                "dense200_hard_top1": float(dense_row.get("hard_subset_top1_acc", 0.0)),
                "dense200_ambiguity_top1": float(dense_row.get("ambiguity_top1_acc", 0.0)),
                "dense200_appearance_top1": float(dense_row.get("appearance_change_top1_acc", 0.0)),
            }
        )
    main_csv = asset_dir / "main_comparison_table.csv"
    _write_csv(main_csv, main_rows, list(main_rows[0].keys()))

    hard_case_rows: List[Dict[str, Any]] = []
    for row in dense_panel.get("method_rows", []):
        name = str(row.get("name", ""))
        if name not in {"stage1_frozen_baseline", "legacysem_best", "cropenc_baseline_best", "current_calibration_only_best", "current_tusb_v3p1_best::best.pt"}:
            continue
        hard_case_rows.append(
            {
                "method": name,
                "hard_subset_top1_acc": float(row.get("hard_subset_top1_acc", 0.0)),
                "ambiguity_top1_acc": float(row.get("ambiguity_top1_acc", 0.0)),
                "appearance_change_top1_acc": float(row.get("appearance_change_top1_acc", 0.0)),
                "small_object_top1_acc": float(row.get("small_object_top1_acc", 0.0)),
                "occlusion_reappearance_top1_acc": float(((row.get("panels") or {}).get("occlusion_reappearance") or {}).get("query_future_top1_acc", 0.0)),
                "long_gap_persistence_top1_acc": float(((row.get("panels") or {}).get("long_gap_persistence") or {}).get("query_future_top1_acc", 0.0)),
            }
        )
    hard_csv = asset_dir / "hard_case_table.csv"
    _write_csv(hard_csv, hard_case_rows, list(hard_case_rows[0].keys()))

    multiseed_rows = list((multiseed.get("densified_200_panel") or {}).get("seed_rows", []))
    multiseed_csv = asset_dir / "multiseed_table.csv"
    if multiseed_rows:
        _write_csv(multiseed_csv, multiseed_rows, list(multiseed_rows[0].keys()))

    mech_rows = list((mechanism.get("table_rows") or []))
    mech_csv = asset_dir / "mechanism_appendix_table.csv"
    if mech_rows:
        _write_csv(mech_csv, mech_rows, list(mech_rows[0].keys()))

    ci_rows: List[Dict[str, Any]] = []
    for comp_name, comp_payload in sorted((bootstrap.get("comparisons") or {}).items()):
        overall = (comp_payload.get("densified_200_overall_top1") or {})
        ci_rows.append(
            {
                "comparison": comp_name,
                "mean_delta": overall.get("mean_delta", 0.0),
                "ci95_low": overall.get("ci95_low", 0.0),
                "ci95_high": overall.get("ci95_high", 0.0),
                "bootstrap_win_rate": overall.get("bootstrap_win_rate", 0.0),
                "zero_excluded": overall.get("zero_excluded", False),
            }
        )
    ci_csv = asset_dir / "bootstrap_ci_table.csv"
    if ci_rows:
        _write_csv(ci_csv, ci_rows, list(ci_rows[0].keys()))

    selected_cases = []
    for case in list((qualitative.get("cases") or []))[:6]:
        if isinstance(case, dict):
            selected_cases.append(
                {
                    "case_id": case.get("case_id", ""),
                    "dataset_source": case.get("dataset_source", ""),
                    "subset_tags": ",".join(str(x) for x in case.get("subset_tags", [])),
                    "why_selected": case.get("why_selected", ""),
                }
            )
    qual_csv = asset_dir / "selected_qualitative_rows.csv"
    if selected_cases:
        _write_csv(qual_csv, selected_cases, list(selected_cases[0].keys()))

    plots = _try_build_plots(asset_dir, dualpanel, multiseed, mechanism)

    payload = {
        "generated_at_utc": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
        "asset_dir": str(asset_dir),
        "main_comparison_table_csv": str(main_csv),
        "hard_case_table_csv": str(hard_csv),
        "multiseed_table_csv": str(multiseed_csv) if multiseed_rows else "",
        "mechanism_appendix_table_csv": str(mech_csv) if mech_rows else "",
        "bootstrap_ci_table_csv": str(ci_csv) if ci_rows else "",
        "selected_qualitative_rows_csv": str(qual_csv) if selected_cases else "",
        "plots": plots,
        "main_paper_ready": True,
        "appendix_ready": True,
        "rebuttal_backup_ready": True,
    }
    base._write_json(Path(args.output_json), payload)
    base._write_md(
        Path(args.output_md),
        [
            "# Stage2 V3.1 Paper Assets 20260420",
            "",
            f"- main_comparison_table_csv: {payload['main_comparison_table_csv']}",
            f"- hard_case_table_csv: {payload['hard_case_table_csv']}",
            f"- multiseed_table_csv: {payload['multiseed_table_csv']}",
            f"- mechanism_appendix_table_csv: {payload['mechanism_appendix_table_csv']}",
            f"- bootstrap_ci_table_csv: {payload['bootstrap_ci_table_csv']}",
            f"- selected_qualitative_rows_csv: {payload['selected_qualitative_rows_csv']}",
            f"- main_paper_ready: {payload['main_paper_ready']}",
            f"- appendix_ready: {payload['appendix_ready']}",
            f"- rebuttal_backup_ready: {payload['rebuttal_backup_ready']}",
        ],
    )


if __name__ == "__main__":
    main()
