from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import json
from typing import Any


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Summarize STWM V4.2 identity rescue round")
    parser.add_argument("--out-root", default="/home/chen034/workspace/stwm/outputs/training/stwm_v4_2_identity_rescue_round")
    parser.add_argument("--variants", default="control_resume_base,resume_eventful_mix,resume_eventful_hardquery_mix")
    parser.add_argument("--seeds", default="42,123")
    parser.add_argument("--output-json", default="/home/chen034/workspace/stwm/reports/stwm_v4_2_identity_rescue_round_v1.json")
    parser.add_argument("--output-md", default="/home/chen034/workspace/stwm/reports/stwm_v4_2_identity_rescue_round_v1.md")
    return parser


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _get_metric_mean(comp: dict[str, Any], run: str, metric: str) -> float:
    return float(comp["aggregate"][run][metric]["mean"])


def _sign_counts(delta_per_seed: dict[str, dict[str, float]]) -> dict[str, int]:
    traj = 0
    query = 0
    reconnect = 0
    for _, vals in delta_per_seed.items():
        d_traj = float(vals.get("trajectory_l1", 0.0))
        d_query = float(vals.get("query_localization_error", 0.0))
        d_reconnect = float(vals.get("reconnect_success_rate", 0.0))
        if d_traj > 0.0:
            traj += 1
        if d_query > 0.0:
            query += 1
        if d_reconnect < 0.0:
            reconnect += 1
    return {
        "full_better_count_traj": traj,
        "full_better_count_query": query,
        "full_better_count_reconnect_success": reconnect,
    }


def main() -> None:
    args = build_parser().parse_args()
    out_root = Path(args.out_root)
    variants = [x.strip() for x in str(args.variants).split(",") if x.strip()]
    seeds = [x.strip() for x in str(args.seeds).split(",") if x.strip()]

    per_variant: dict[str, Any] = {}

    for variant in variants:
        base_comp = _load(out_root / "eval" / "base" / variant / "comparison_base.json")
        eventful_comp = _load(out_root / "eval" / "eventful" / variant / "comparison_eventful.json")
        hard_comp = _load(out_root / "eval" / "hard_query" / variant / "comparison_hard_query.json")

        eventful_bucket = _load(out_root / "eval" / "eventful" / variant / "occlusion_reconnect_eventful.json")
        hard_decoupling = _load(out_root / "eval" / "hard_query" / variant / "query_decoupling_hard_query.json")

        delta_eventful = eventful_comp.get("delta_vs_full", {}).get("wo_identity_v4_2", {})
        delta_hard = hard_comp.get("delta_vs_full", {}).get("wo_identity_v4_2", {})

        per_seed_delta_eventful = eventful_comp.get("pairwise_delta_per_seed", {}).get("wo_identity_v4_2", {})
        per_seed_delta_hard = hard_comp.get("pairwise_delta_per_seed", {}).get("wo_identity_v4_2", {})

        eventful_sign_counts = _sign_counts(per_seed_delta_eventful)
        hard_sign_counts = _sign_counts(per_seed_delta_hard)

        hard_cmp_vs_full = hard_decoupling.get("comparison_vs_full", {}).get("wo_identity_v4_2", {})
        hard_decoupling_delta = hard_cmp_vs_full.get("aggregate_delta", {})

        per_variant[variant] = {
            "base": {
                "full": {
                    "trajectory_l1": _get_metric_mean(base_comp, "full_v4_2", "trajectory_l1"),
                    "query_localization_error": _get_metric_mean(base_comp, "full_v4_2", "query_localization_error"),
                    "reconnect_success_rate": _get_metric_mean(base_comp, "full_v4_2", "reconnect_success_rate"),
                },
                "wo_identity": {
                    "trajectory_l1": _get_metric_mean(base_comp, "wo_identity_v4_2", "trajectory_l1"),
                    "query_localization_error": _get_metric_mean(base_comp, "wo_identity_v4_2", "query_localization_error"),
                    "reconnect_success_rate": _get_metric_mean(base_comp, "wo_identity_v4_2", "reconnect_success_rate"),
                },
            },
            "eventful": {
                "full": {
                    "trajectory_l1": _get_metric_mean(eventful_comp, "full_v4_2", "trajectory_l1"),
                    "query_localization_error": _get_metric_mean(eventful_comp, "full_v4_2", "query_localization_error"),
                    "reconnect_success_rate": _get_metric_mean(eventful_comp, "full_v4_2", "reconnect_success_rate"),
                    "reappearance_event_ratio": _get_metric_mean(eventful_comp, "full_v4_2", "reappearance_event_ratio"),
                },
                "wo_identity": {
                    "trajectory_l1": _get_metric_mean(eventful_comp, "wo_identity_v4_2", "trajectory_l1"),
                    "query_localization_error": _get_metric_mean(eventful_comp, "wo_identity_v4_2", "query_localization_error"),
                    "reconnect_success_rate": _get_metric_mean(eventful_comp, "wo_identity_v4_2", "reconnect_success_rate"),
                    "reappearance_event_ratio": _get_metric_mean(eventful_comp, "wo_identity_v4_2", "reappearance_event_ratio"),
                },
                "delta_wo_identity_minus_full": {
                    "trajectory_l1_mean": float(delta_eventful.get("trajectory_l1", {}).get("mean", 0.0)),
                    "query_localization_error_mean": float(delta_eventful.get("query_localization_error", {}).get("mean", 0.0)),
                    "reconnect_success_rate_mean": float(delta_eventful.get("reconnect_success_rate", {}).get("mean", 0.0)),
                    "reappearance_event_ratio_mean": float(delta_eventful.get("reappearance_event_ratio", {}).get("mean", 0.0)),
                },
                "sign_consistency": eventful_sign_counts,
                "bucket_report": {
                    "total_event_rows_full": int(eventful_bucket["aggregate"]["full_v4_2"]["total_event_rows"]),
                    "total_event_rows_wo_identity": int(eventful_bucket["aggregate"]["wo_identity_v4_2"]["total_event_rows"]),
                    "sufficient_for_reconnect_claim": bool(eventful_bucket["statistical_power"].get("sufficient_for_reconnect_claim", False)),
                    "delta_reconnect_success_rate_mean": float(
                        eventful_bucket["comparison_vs_full"]["wo_identity_v4_2"]["aggregate_delta"]["delta_reconnect_success_rate"]["mean"]
                    ),
                },
            },
            "hard_query": {
                "full": {
                    "trajectory_l1": _get_metric_mean(hard_comp, "full_v4_2", "trajectory_l1"),
                    "query_localization_error": _get_metric_mean(hard_comp, "full_v4_2", "query_localization_error"),
                    "reconnect_success_rate": _get_metric_mean(hard_comp, "full_v4_2", "reconnect_success_rate"),
                },
                "wo_identity": {
                    "trajectory_l1": _get_metric_mean(hard_comp, "wo_identity_v4_2", "trajectory_l1"),
                    "query_localization_error": _get_metric_mean(hard_comp, "wo_identity_v4_2", "query_localization_error"),
                    "reconnect_success_rate": _get_metric_mean(hard_comp, "wo_identity_v4_2", "reconnect_success_rate"),
                },
                "delta_wo_identity_minus_full": {
                    "trajectory_l1_mean": float(delta_hard.get("trajectory_l1", {}).get("mean", 0.0)),
                    "query_localization_error_mean": float(delta_hard.get("query_localization_error", {}).get("mean", 0.0)),
                    "reconnect_success_rate_mean": float(delta_hard.get("reconnect_success_rate", {}).get("mean", 0.0)),
                },
                "sign_consistency": hard_sign_counts,
                "decoupling_delta_vs_full": {
                    "delta_corr_abs_mean": float(hard_decoupling_delta.get("delta_corr_abs", {}).get("mean", 0.0)),
                    "delta_close_ratio_mean": float(hard_decoupling_delta.get("delta_close_ratio", {}).get("mean", 0.0)),
                    "delta_decoupling_score_mean": float(hard_decoupling_delta.get("delta_decoupling_score", {}).get("mean", 0.0)),
                },
            },
        }

    control = per_variant.get("control_resume_base")
    amplification: dict[str, Any] = {}
    if control is not None:
        base_gap = abs(float(control["eventful"]["delta_wo_identity_minus_full"]["reconnect_success_rate_mean"]))
        for variant in variants:
            if variant == "control_resume_base":
                continue
            current_gap = abs(float(per_variant[variant]["eventful"]["delta_wo_identity_minus_full"]["reconnect_success_rate_mean"]))
            amplification[variant] = {
                "reconnect_gap_abs_control": base_gap,
                "reconnect_gap_abs_variant": current_gap,
                "reconnect_gap_amplified": bool(current_gap > base_gap + 1e-12),
            }

    out = {
        "out_root": str(out_root),
        "variants": variants,
        "seeds": seeds,
        "per_variant": per_variant,
        "amplification_vs_control": amplification,
    }

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(out, indent=2))

    lines: list[str] = []
    lines.append("# STWM V4.2 Identity Rescue Round Summary")
    lines.append("")
    lines.append(f"Out root: `{out_root}`")
    lines.append(f"Seeds: `{', '.join(seeds)}`")
    lines.append("")
    lines.append("## Variant Comparison (Eventful Delta, wo_identity - full)")
    lines.append("")
    lines.append("| variant | d_traj | d_query | d_reconnect_success | full_better_reconnect_count | full_better_traj_count | full_better_query_count |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for variant in variants:
        ev = per_variant[variant]["eventful"]
        d = ev["delta_wo_identity_minus_full"]
        c = ev["sign_consistency"]
        lines.append(
            "| {} | {:+.6f} | {:+.6f} | {:+.6f} | {} | {} | {} |".format(
                variant,
                float(d["trajectory_l1_mean"]),
                float(d["query_localization_error_mean"]),
                float(d["reconnect_success_rate_mean"]),
                int(c["full_better_count_reconnect_success"]),
                int(c["full_better_count_traj"]),
                int(c["full_better_count_query"]),
            )
        )

    lines.append("")
    lines.append("## Hard-Query Decoupling Delta (wo_identity - full)")
    lines.append("")
    lines.append("| variant | d_corr_abs | d_close_ratio | d_decoupling_score |")
    lines.append("|---|---:|---:|---:|")
    for variant in variants:
        d = per_variant[variant]["hard_query"]["decoupling_delta_vs_full"]
        lines.append(
            "| {} | {:+.6f} | {:+.6f} | {:+.6f} |".format(
                variant,
                float(d["delta_corr_abs_mean"]),
                float(d["delta_close_ratio_mean"]),
                float(d["delta_decoupling_score_mean"]),
            )
        )

    if amplification:
        lines.append("")
        lines.append("## Amplification vs Control (Reconnect Gap)")
        lines.append("")
        lines.append("| variant | abs_gap_control | abs_gap_variant | amplified |")
        lines.append("|---|---:|---:|---|")
        for variant, row in amplification.items():
            lines.append(
                "| {} | {:.6f} | {:.6f} | {} |".format(
                    variant,
                    float(row["reconnect_gap_abs_control"]),
                    float(row["reconnect_gap_abs_variant"]),
                    bool(row["reconnect_gap_amplified"]),
                )
            )

    output_md = Path(args.output_md)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text("\n".join(lines) + "\n")

    print(json.dumps({"output_json": str(output_json), "output_md": str(output_md), "variants": variants}, indent=2))


if __name__ == "__main__":
    main()
