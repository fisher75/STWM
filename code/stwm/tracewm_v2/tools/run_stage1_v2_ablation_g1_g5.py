#!/usr/bin/env python3
from __future__ import annotations

from argparse import ArgumentParser
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List
import json
import subprocess


DATE_TAG = "20260408"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_args() -> Any:
    parser = ArgumentParser(description="Run Stage1 v2 ablation grid G1-G5")
    parser.add_argument("--python-bin", default="/home/chen034/miniconda3/envs/stwm/bin/python")
    parser.add_argument("--work-root", default="/home/chen034/workspace/stwm")
    parser.add_argument("--contract-path", default="/home/chen034/workspace/data/_manifests/stage1_v2_trace_cache_contract_20260408.json")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--steps-per-epoch", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-samples-per-dataset", type=int, default=128)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260408)
    parser.add_argument("--report-json", default=f"/home/chen034/workspace/stwm/reports/tracewm_stage1_v2_g1_g5_{DATE_TAG}.json")
    parser.add_argument("--report-md", default=f"/home/chen034/workspace/stwm/docs/TRACEWM_STAGE1_V2_G1_G5_{DATE_TAG}.md")
    return parser.parse_args()


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    work_root = Path(args.work_root)

    trainer = work_root / "code/stwm/tracewm_v2/trainers/train_tracewm_stage1_v2.py"
    out_root = work_root / "outputs/training/tracewm_stage1_v2_ablations_20260408"
    out_root.mkdir(parents=True, exist_ok=True)

    report_json = Path(args.report_json)
    report_md = Path(args.report_md)
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_md.parent.mkdir(parents=True, exist_ok=True)

    ablations: List[Dict[str, Any]] = [
        {
            "name": "G1",
            "note": "coord-only baseline on multi-token state",
            "extra_flags": [],
        },
        {
            "name": "G2",
            "note": "G1 + visibility supervision",
            "extra_flags": ["--enable-visibility"],
        },
        {
            "name": "G3",
            "note": "G2 + residual supervision",
            "extra_flags": ["--enable-visibility", "--enable-residual"],
        },
        {
            "name": "G4",
            "note": "G3 + velocity supervision",
            "extra_flags": ["--enable-visibility", "--enable-residual", "--enable-velocity"],
        },
        {
            "name": "G5",
            "note": "G4 + endpoint supervision",
            "extra_flags": [
                "--enable-visibility",
                "--enable-residual",
                "--enable-velocity",
                "--enable-endpoint",
            ],
        },
    ]

    runs: List[Dict[str, Any]] = []
    for i, lane in enumerate(ablations):
        tag = lane["name"].lower()
        summary_json = work_root / "reports" / f"tracewm_stage1_v2_{tag}_summary_{DATE_TAG}.json"
        results_md = work_root / "docs" / f"TRACEWM_STAGE1_V2_{lane['name']}_{DATE_TAG}.md"
        output_dir = out_root / tag

        cmd = [
            str(args.python_bin),
            str(trainer),
            "--contract-path",
            str(args.contract_path),
            "--output-dir",
            str(output_dir),
            "--summary-json",
            str(summary_json),
            "--results-md",
            str(results_md),
            "--ablation-tag",
            tag,
            "--model-preset",
            "debug_small",
            "--epochs",
            str(args.epochs),
            "--steps-per-epoch",
            str(args.steps_per_epoch),
            "--batch-size",
            str(args.batch_size),
            "--max-samples-per-dataset",
            str(args.max_samples_per_dataset),
            "--max-tokens",
            str(args.max_tokens),
            "--seed",
            str(int(args.seed) + i),
        ]
        cmd.extend([str(x) for x in lane["extra_flags"]])

        proc = subprocess.run(cmd, cwd=str(work_root), text=True)

        run_info: Dict[str, Any] = {
            "name": lane["name"],
            "note": lane["note"],
            "return_code": int(proc.returncode),
            "summary_json": str(summary_json),
            "results_md": str(results_md),
        }

        if proc.returncode == 0 and summary_json.exists():
            payload = _load_json(summary_json)
            run_info["final_metrics"] = payload.get("final_metrics", {})
            run_info["parameter_count"] = payload.get("model", {}).get("parameter_count", 0)
            run_info["estimated_parameter_count"] = payload.get("model", {}).get("estimated_parameter_count", 0)
        else:
            run_info["final_metrics"] = {}

        runs.append(run_info)

    successful = [r for r in runs if r.get("return_code") == 0 and isinstance(r.get("final_metrics"), dict)]

    best = None
    if successful:
        best = min(successful, key=lambda x: float(x.get("final_metrics", {}).get("total_loss", 1e9)))

    payload = {
        "generated_at_utc": now_iso(),
        "contract_path": str(args.contract_path),
        "runs": runs,
        "mainline_recommendation": {
            "selected": best.get("name") if best else "none",
            "selection_metric": "min_total_loss",
            "selected_total_loss": float(best.get("final_metrics", {}).get("total_loss", 0.0)) if best else None,
        },
    }

    report_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# TRACEWM Stage1 v2 G1-G5 Ablation",
        "",
        f"- generated_at_utc: {payload['generated_at_utc']}",
        f"- selected_mainline: {payload['mainline_recommendation']['selected']}",
        f"- selection_metric: {payload['mainline_recommendation']['selection_metric']}",
        "",
        "| lane | return_code | total_loss | coord_loss | vis_loss | residual_loss | velocity_loss | endpoint_loss |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for run in runs:
        fm = run.get("final_metrics", {}) if isinstance(run.get("final_metrics"), dict) else {}
        lines.append(
            "| {name} | {code} | {total:.6f} | {coord:.6f} | {vis:.6f} | {res:.6f} | {vel:.6f} | {endp:.6f} |".format(
                name=run.get("name", "-"),
                code=int(run.get("return_code", -1)),
                total=float(fm.get("total_loss", 0.0)),
                coord=float(fm.get("coord_loss", 0.0)),
                vis=float(fm.get("visibility_loss", 0.0)),
                res=float(fm.get("residual_loss", 0.0)),
                vel=float(fm.get("velocity_loss", 0.0)),
                endp=float(fm.get("endpoint_loss", 0.0)),
            )
        )

    report_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"[g1-g5] report_json={report_json}")
    print(f"[g1-g5] report_md={report_md}")
    print(f"[g1-g5] selected={payload['mainline_recommendation']['selected']}")


if __name__ == "__main__":
    main()
