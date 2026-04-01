from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import json
from typing import Any


BASE_METRICS = [
    "trajectory_l1",
    "query_localization_error",
    "semantic_loss",
    "reid_loss",
    "query_traj_gap",
    "memory_gate_mean",
    "reconnect_success_rate",
]

STATE_METRICS = [
    "trajectory_l1",
    "query_localization_error",
    "query_traj_gap",
    "reconnect_success_rate",
]

LOWER_BETTER = {
    "trajectory_l1",
    "query_localization_error",
    "semantic_loss",
    "reid_loss",
    "query_traj_gap",
}


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Compare STWM V4.2 220M vs 1B and produce 3B go/no-go answers")
    parser.add_argument("--base-220m", default="/home/chen034/workspace/stwm/outputs/training/stwm_v4_2_minival_multiseed/comparison_multiseed.json")
    parser.add_argument("--base-1b", default="/home/chen034/workspace/stwm/outputs/training/stwm_v4_2_1b_minival_multiseed/comparison_multiseed.json")
    parser.add_argument("--state-220m", default="/home/chen034/workspace/stwm/outputs/training/stwm_v4_2_state_identifiability/comparison_state_identifiability.json")
    parser.add_argument("--state-1b", default="/home/chen034/workspace/stwm/outputs/training/stwm_v4_2_1b_state_identifiability/comparison_state_identifiability.json")
    parser.add_argument("--decoupling-220m", default="/home/chen034/workspace/stwm/reports/stwm_v4_2_state_identifiability_decoupling_v1.json")
    parser.add_argument("--decoupling-1b", default="/home/chen034/workspace/stwm/reports/stwm_v4_2_1b_state_identifiability_decoupling_v1.json")
    parser.add_argument("--output-json", default="/home/chen034/workspace/stwm/reports/stwm_v4_2_220m_vs_1b.json")
    parser.add_argument("--output-md", default="/home/chen034/workspace/stwm/docs/STWM_V4_2_3B_GO_NO_GO.md")
    return parser


def _load(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return json.loads(path.read_text())


def _agg_metric(payload: dict[str, Any], run: str, metric: str) -> float:
    return float(payload.get("aggregate", {}).get(run, {}).get(metric, {}).get("mean", 0.0))


def _state_agg_metric(payload: dict[str, Any], run: str, metric: str) -> float:
    return float(payload.get("aggregate", {}).get(run, {}).get("overall", {}).get(metric, {}).get("mean", 0.0))


def _is_better(metric: str, delta_1b_minus_220m: float) -> bool:
    if metric in LOWER_BETTER:
        return delta_1b_minus_220m < 0.0
    return delta_1b_minus_220m > 0.0


def _fmt_signed(v: float) -> str:
    return f"{v:+.6f}"


def _pick_stability_count(base_1b: dict[str, Any], run: str, metric: str) -> int:
    return int(base_1b.get("pairwise_stability_count", {}).get(run, {}).get(metric, 0))


def _pick_state_sign_count(state_1b: dict[str, Any], run: str, metric: str) -> int:
    return int(state_1b.get("sign_consistency", {}).get(run, {}).get("overall", {}).get(metric, 0))


def main() -> None:
    args = build_parser().parse_args()

    base_220m = _load(Path(args.base_220m))
    base_1b = _load(Path(args.base_1b))
    state_220m = _load(Path(args.state_220m))
    state_1b = _load(Path(args.state_1b))
    dec_220m = _load(Path(args.decoupling_220m))
    dec_1b = _load(Path(args.decoupling_1b))

    base_delta: dict[str, float] = {}
    for metric in BASE_METRICS:
        v_220m = _agg_metric(base_220m, "full_v4_2", metric)
        v_1b = _agg_metric(base_1b, "full_v4_2", metric)
        base_delta[metric] = float(v_1b - v_220m)

    state_delta: dict[str, float] = {}
    for metric in STATE_METRICS:
        v_220m = _state_agg_metric(state_220m, "full_v4_2", metric)
        v_1b = _state_agg_metric(state_1b, "full_v4_2", metric)
        state_delta[metric] = float(v_1b - v_220m)

    dec_220m_score = float(dec_220m.get("aggregate", {}).get("full_v4_2", {}).get("decoupling_score", {}).get("mean", 0.0))
    dec_1b_score = float(dec_1b.get("aggregate", {}).get("full_v4_2", {}).get("decoupling_score", {}).get("mean", 0.0))
    dec_220m_proxy = int(dec_220m.get("aggregate", {}).get("full_v4_2", {}).get("proxy_like_count", 0))
    dec_1b_proxy = int(dec_1b.get("aggregate", {}).get("full_v4_2", {}).get("proxy_like_count", 0))

    q1_sem_stability = _pick_stability_count(base_1b, "wo_semantics_v4_2", "query_localization_error") >= 2 and _pick_stability_count(base_1b, "wo_semantics_v4_2", "trajectory_l1") >= 2
    q2_objbias_stability = _pick_stability_count(base_1b, "wo_object_bias_v4_2", "query_localization_error") >= 2 and _pick_stability_count(base_1b, "wo_object_bias_v4_2", "trajectory_l1") >= 2
    q3_state_ident = _pick_state_sign_count(state_1b, "wo_semantics_v4_2", "query_localization_error") >= 2 and _pick_state_sign_count(state_1b, "wo_object_bias_v4_2", "query_localization_error") >= 2
    q4_harder_decoupling = (dec_1b_score - dec_220m_score) > 0.0 and dec_1b_proxy == 0

    q5_scale_gain = (
        _is_better("trajectory_l1", base_delta["trajectory_l1"])
        and _is_better("query_localization_error", base_delta["query_localization_error"])
        and _is_better("trajectory_l1", state_delta["trajectory_l1"])
        and _is_better("query_localization_error", state_delta["query_localization_error"])
    )

    go_no_go = bool(q1_sem_stability and q2_objbias_stability and q3_state_ident and q4_harder_decoupling and q5_scale_gain)

    questions = [
        {
            "id": "Q1",
            "question": "1B上，full 相比 wo_semantics 的主效应是否跨seed稳定？",
            "answer": "YES" if q1_sem_stability else "NO",
            "evidence": {
                "full_better_count_query": _pick_stability_count(base_1b, "wo_semantics_v4_2", "query_localization_error"),
                "full_better_count_traj": _pick_stability_count(base_1b, "wo_semantics_v4_2", "trajectory_l1"),
            },
        },
        {
            "id": "Q2",
            "question": "1B上，full 相比 wo_object_bias 的主效应是否跨seed稳定？",
            "answer": "YES" if q2_objbias_stability else "NO",
            "evidence": {
                "full_better_count_query": _pick_stability_count(base_1b, "wo_object_bias_v4_2", "query_localization_error"),
                "full_better_count_traj": _pick_stability_count(base_1b, "wo_object_bias_v4_2", "trajectory_l1"),
            },
        },
        {
            "id": "Q3",
            "question": "在 state-identifiability harder protocol 上，full 是否仍优于两组消融？",
            "answer": "YES" if q3_state_ident else "NO",
            "evidence": {
                "full_better_count_vs_wo_semantics_query": _pick_state_sign_count(state_1b, "wo_semantics_v4_2", "query_localization_error"),
                "full_better_count_vs_wo_object_bias_query": _pick_state_sign_count(state_1b, "wo_object_bias_v4_2", "query_localization_error"),
            },
        },
        {
            "id": "Q4",
            "question": "harder-protocol decoupling 是否优于220M且保持非proxy-like？",
            "answer": "YES" if q4_harder_decoupling else "NO",
            "evidence": {
                "decoupling_score_220m": dec_220m_score,
                "decoupling_score_1b": dec_1b_score,
                "delta_1b_minus_220m": float(dec_1b_score - dec_220m_score),
                "proxy_like_count_220m": dec_220m_proxy,
                "proxy_like_count_1b": dec_1b_proxy,
            },
        },
        {
            "id": "Q5",
            "question": "是否满足进入3B训练的 go/no-go 门槛？",
            "answer": "GO" if go_no_go else "NO-GO",
            "evidence": {
                "base_delta_trajectory_l1": base_delta["trajectory_l1"],
                "base_delta_query_localization_error": base_delta["query_localization_error"],
                "state_delta_trajectory_l1": state_delta["trajectory_l1"],
                "state_delta_query_localization_error": state_delta["query_localization_error"],
                "all_gates": {
                    "q1_sem_stability": q1_sem_stability,
                    "q2_objbias_stability": q2_objbias_stability,
                    "q3_state_ident": q3_state_ident,
                    "q4_harder_decoupling": q4_harder_decoupling,
                    "q5_scale_gain": q5_scale_gain,
                },
            },
        },
    ]

    out = {
        "inputs": {
            "base_220m": str(args.base_220m),
            "base_1b": str(args.base_1b),
            "state_220m": str(args.state_220m),
            "state_1b": str(args.state_1b),
            "decoupling_220m": str(args.decoupling_220m),
            "decoupling_1b": str(args.decoupling_1b),
        },
        "base_full_delta_1b_minus_220m": base_delta,
        "state_full_delta_1b_minus_220m": state_delta,
        "decoupling_full": {
            "score_220m": dec_220m_score,
            "score_1b": dec_1b_score,
            "delta_1b_minus_220m": float(dec_1b_score - dec_220m_score),
            "proxy_like_count_220m": dec_220m_proxy,
            "proxy_like_count_1b": dec_1b_proxy,
        },
        "questions": questions,
        "go_no_go": {
            "ready_for_3b": go_no_go,
            "decision": "GO" if go_no_go else "NO-GO",
        },
    }

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(out, indent=2, ensure_ascii=False))

    lines: list[str] = []
    lines.append("# STWM V4.2 3B Go/No-Go (220M vs 1B)")
    lines.append("")
    lines.append("## Full Run Delta (1B - 220M)")
    lines.append("")
    lines.append("### Base Protocol")
    lines.append("")
    lines.append("| metric | delta_1b_minus_220m | expected_direction |")
    lines.append("|---|---:|---|")
    for metric in BASE_METRICS:
        direction = "lower_better" if metric in LOWER_BETTER else "higher_better"
        lines.append(f"| {metric} | {_fmt_signed(base_delta[metric])} | {direction} |")

    lines.append("")
    lines.append("### State-Identifiability Protocol")
    lines.append("")
    lines.append("| metric | delta_1b_minus_220m | expected_direction |")
    lines.append("|---|---:|---|")
    for metric in STATE_METRICS:
        direction = "lower_better" if metric in LOWER_BETTER else "higher_better"
        lines.append(f"| {metric} | {_fmt_signed(state_delta[metric])} | {direction} |")

    lines.append("")
    lines.append("### Harder-Protocol Decoupling")
    lines.append("")
    lines.append(f"- decoupling_score_220m: {dec_220m_score:.6f}")
    lines.append(f"- decoupling_score_1b: {dec_1b_score:.6f}")
    lines.append(f"- delta_1b_minus_220m: {_fmt_signed(dec_1b_score - dec_220m_score)}")
    lines.append(f"- proxy_like_count_220m: {dec_220m_proxy}")
    lines.append(f"- proxy_like_count_1b: {dec_1b_proxy}")

    lines.append("")
    lines.append("## Final 5 Questions")
    lines.append("")
    for q in questions:
        lines.append(f"- {q['id']}: {q['question']}")
        lines.append(f"  - answer: {q['answer']}")
        lines.append(f"  - evidence: {json.dumps(q['evidence'], ensure_ascii=False)}")

    lines.append("")
    lines.append("## 3B Decision")
    lines.append("")
    lines.append(f"- decision: {'GO' if go_no_go else 'NO-GO'}")
    lines.append(f"- ready_for_3b: {str(go_no_go).lower()}")

    output_md = Path(args.output_md)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps({"output_json": str(output_json), "output_md": str(output_md), "decision": "GO" if go_no_go else "NO-GO"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
