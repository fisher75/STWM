from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import json
from typing import Any


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Summarize STWM V4.2 seed42 mini-val runs")
    parser.add_argument("--runs-root", default="/home/chen034/workspace/stwm/outputs/training/stwm_v4_2_minival_seed42")
    parser.add_argument("--runs", default="full_v4_2,wo_semantics_v4_2,wo_identity_v4_2")
    parser.add_argument("--summary-name", default="mini_val_summary.json")
    parser.add_argument("--output-json", default="/home/chen034/workspace/stwm/outputs/training/stwm_v4_2_minival_seed42/comparison_seed42.json")
    parser.add_argument("--output-md", default="/home/chen034/workspace/stwm/outputs/training/stwm_v4_2_minival_seed42/comparison_seed42.md")
    return parser


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _load_log_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def _first_last(rows: list[dict[str, Any]], keys: list[str]) -> dict[str, dict[str, float]]:
    if not rows:
        return {"first": {}, "last": {}}
    first = {k: float(rows[0].get(k, 0.0)) for k in keys}
    last = {k: float(rows[-1].get(k, 0.0)) for k in keys}
    return {"first": first, "last": last}


def main() -> None:
    args = build_parser().parse_args()
    runs_root = Path(args.runs_root)
    runs = [x.strip() for x in str(args.runs).split(",") if x.strip()]

    compare: dict[str, Any] = {
        "runs_root": str(runs_root),
        "runs": runs,
        "summary_name": str(args.summary_name),
        "runs_data": {},
        "delta_vs_full": {},
    }

    keys = [
        "total_loss",
        "trajectory_loss",
        "trajectory_l1",
        "semantic_loss",
        "reid_loss",
        "query_loss",
        "query_localization_error",
        "query_traj_gap",
        "assignment_entropy",
        "token_usage_entropy",
        "memory_gate_mean",
        "reconnect_success",
        "reconnect_min_error",
    ]

    for run in runs:
        run_dir = runs_root / run
        summary = _load_json(run_dir / str(args.summary_name))
        rows = _load_log_rows(run_dir / "train_log.jsonl")
        first_last = _first_last(rows, keys)

        compare["runs_data"][run] = {
            "summary": summary,
            "first_last": first_last,
            "num_rows": len(rows),
        }

    if "full_v4_2" in compare["runs_data"]:
        full = compare["runs_data"]["full_v4_2"]["summary"]
        full_loss = full.get("average_losses", {})
        full_diag = full.get("diagnostics", {})
        for run in runs:
            if run == "full_v4_2":
                continue
            d = compare["runs_data"][run]["summary"]
            run_loss = d.get("average_losses", {})
            run_diag = d.get("diagnostics", {})
            compare["delta_vs_full"][run] = {
                "trajectory_l1": float(run_loss.get("trajectory_l1", 0.0)) - float(full_loss.get("trajectory_l1", 0.0)),
                "query_localization_error": float(run_loss.get("query_localization_error", 0.0)) - float(full_loss.get("query_localization_error", 0.0)),
                "semantic_loss": float(run_loss.get("semantic", 0.0)) - float(full_loss.get("semantic", 0.0)),
                "reid_loss": float(run_loss.get("reid", 0.0)) - float(full_loss.get("reid", 0.0)),
                "memory_gate_mean": float(run_diag.get("memory_gate_mean", 0.0)) - float(full_diag.get("memory_gate_mean", 0.0)),
                "reconnect_success_rate": float(run_diag.get("reconnect_success_rate", 0.0)) - float(full_diag.get("reconnect_success_rate", 0.0)),
            }

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(compare, indent=2))

    lines: list[str] = []
    lines.append("# STWM V4.2 Mini-Val Seed42 Comparison")
    lines.append("")
    lines.append(f"Runs root: `{runs_root}`")
    lines.append("")
    lines.append("## Average Loss/Metric Table")
    lines.append("")
    lines.append("| run | total | traj_l1 | query_loc_err | semantic | reid | query | memory_gate | reconnect_success |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for run in runs:
        s = compare["runs_data"][run]["summary"]
        loss = s.get("average_losses", {})
        diag = s.get("diagnostics", {})
        lines.append(
            "| {} | {:.6f} | {:.6f} | {:.6f} | {:.6f} | {:.6f} | {:.6f} | {:.6f} | {:.6f} |".format(
                run,
                float(loss.get("total", 0.0)),
                float(loss.get("trajectory_l1", 0.0)),
                float(loss.get("query_localization_error", 0.0)),
                float(loss.get("semantic", 0.0)),
                float(loss.get("reid", 0.0)),
                float(loss.get("query", 0.0)),
                float(diag.get("memory_gate_mean", 0.0)),
                float(diag.get("reconnect_success_rate", 0.0)),
            )
        )

    lines.append("")
    lines.append("## First/Last Trend")
    lines.append("")
    for run in runs:
        fl = compare["runs_data"][run]["first_last"]
        lines.append(f"- {run}")
        lines.append(
            "  - first total/traj_l1/query_err: {:.6f} / {:.6f} / {:.6f}".format(
                float(fl["first"].get("total_loss", 0.0)),
                float(fl["first"].get("trajectory_l1", 0.0)),
                float(fl["first"].get("query_localization_error", 0.0)),
            )
        )
        lines.append(
            "  - last total/traj_l1/query_err: {:.6f} / {:.6f} / {:.6f}".format(
                float(fl["last"].get("total_loss", 0.0)),
                float(fl["last"].get("trajectory_l1", 0.0)),
                float(fl["last"].get("query_localization_error", 0.0)),
            )
        )

    lines.append("")
    lines.append("## Risk Flags")
    lines.append("")
    for run in runs:
        risk = compare["runs_data"][run]["summary"].get("risk_flags", {})
        lines.append(f"- {run}: {json.dumps(risk, ensure_ascii=True)}")

    if compare.get("delta_vs_full"):
        lines.append("")
        lines.append("## Delta vs full_v4_2")
        lines.append("")
        lines.append("| run | d_traj_l1 | d_query_err | d_sem | d_reid | d_memory_gate | d_reconnect_success |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for run, d in compare["delta_vs_full"].items():
            lines.append(
                "| {} | {:.6f} | {:.6f} | {:.6f} | {:.6f} | {:.6f} | {:.6f} |".format(
                    run,
                    float(d.get("trajectory_l1", 0.0)),
                    float(d.get("query_localization_error", 0.0)),
                    float(d.get("semantic_loss", 0.0)),
                    float(d.get("reid_loss", 0.0)),
                    float(d.get("memory_gate_mean", 0.0)),
                    float(d.get("reconnect_success_rate", 0.0)),
                )
            )

    output_md = Path(args.output_md)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text("\n".join(lines) + "\n")

    print(json.dumps({"output_json": str(output_json), "output_md": str(output_md), "runs": runs}, indent=2))


if __name__ == "__main__":
    main()
