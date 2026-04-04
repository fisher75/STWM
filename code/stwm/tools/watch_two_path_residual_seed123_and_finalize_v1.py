from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import json
import math
import re
import time
from typing import Any


STEP_RE = re.compile(r"step_(\d+)\.json$")


def build_parser() -> ArgumentParser:
    p = ArgumentParser(description="Watch two_path seed123 run then write final decision report")
    p.add_argument(
        "--queue-status-dir",
        default=(
            "/home/chen034/workspace/stwm/outputs/queue/"
            "stwm_protocol_v2_frontend_default_v1/d1_train/status"
        ),
    )
    p.add_argument(
        "--submit-json",
        default=(
            "/home/chen034/workspace/stwm/reports/"
            "stwm_two_path_residual_seed123_submit_v1.json"
        ),
    )
    p.add_argument("--poll-seconds", type=int, default=120)
    p.add_argument("--timeout-seconds", type=int, default=0)
    p.add_argument(
        "--out-report",
        default=(
            "/home/chen034/workspace/stwm/reports/"
            "stwm_two_path_residual_seed123_final_decision_v1.json"
        ),
    )
    p.add_argument(
        "--out-doc",
        default=(
            "/home/chen034/workspace/stwm/docs/"
            "STWM_TWO_PATH_RESIDUAL_SEED123_FINAL_DECISION_V1.md"
        ),
    )
    return p


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text())
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _safe_float(v: Any) -> float | None:
    try:
        x = float(v)
    except Exception:
        return None
    if math.isnan(x) or math.isinf(x):
        return None
    return x


def _official_key(metrics: dict[str, Any] | None) -> tuple[float, float, float] | None:
    if not isinstance(metrics, dict):
        return None
    qloc = _safe_float(metrics.get("query_localization_error"))
    qtop = _safe_float(metrics.get("query_top1_acc"))
    l1 = _safe_float(metrics.get("future_trajectory_l1"))
    if qloc is None or qtop is None or l1 is None:
        return None
    return (float(qloc), -float(qtop), float(l1))


def _cmp(lhs_metrics: dict[str, Any] | None, rhs_metrics: dict[str, Any] | None) -> dict[str, Any]:
    lk = _official_key(lhs_metrics)
    rk = _official_key(rhs_metrics)

    out: dict[str, Any] = {
        "lhs_official_key": list(lk) if lk is not None else None,
        "rhs_official_key": list(rk) if rk is not None else None,
        "lhs_beats_rhs_official": None,
        "lhs_ties_rhs_official": None,
        "delta_query_localization_error": None,
        "delta_query_top1_acc": None,
        "delta_future_trajectory_l1": None,
    }

    if lk is not None and rk is not None:
        out["lhs_beats_rhs_official"] = bool(lk < rk)
        out["lhs_ties_rhs_official"] = bool(lk == rk)

    if isinstance(lhs_metrics, dict) and isinstance(rhs_metrics, dict):
        l_qloc = _safe_float(lhs_metrics.get("query_localization_error"))
        l_qtop = _safe_float(lhs_metrics.get("query_top1_acc"))
        l_l1 = _safe_float(lhs_metrics.get("future_trajectory_l1"))
        r_qloc = _safe_float(rhs_metrics.get("query_localization_error"))
        r_qtop = _safe_float(rhs_metrics.get("query_top1_acc"))
        r_l1 = _safe_float(rhs_metrics.get("future_trajectory_l1"))
        if l_qloc is not None and r_qloc is not None:
            out["delta_query_localization_error"] = float(l_qloc - r_qloc)
        if l_qtop is not None and r_qtop is not None:
            out["delta_query_top1_acc"] = float(l_qtop - r_qtop)
        if l_l1 is not None and r_l1 is not None:
            out["delta_future_trajectory_l1"] = float(l_l1 - r_l1)
    return out


def _find_latest_status(status_dir: Path, run_name: str) -> Path | None:
    cands = sorted(status_dir.glob(f"*_{run_name}.status.json"))
    return cands[-1] if cands else None


def _count_train_rows_and_step(train_log: Path) -> tuple[int, int]:
    if not train_log.exists():
        return 0, -1
    rows = 0
    max_step = -1
    with train_log.open("r") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            rows += 1
            try:
                obj = json.loads(s)
                step = int(obj.get("step", -1))
                if step > max_step:
                    max_step = step
            except Exception:
                continue
    return rows, max_step


def _find_sidecar(run_dir: Path) -> Path | None:
    p = run_dir / "checkpoints" / "best_protocol_main_selection.json"
    if p.exists():
        return p
    cands = sorted(run_dir.glob("**/*selection*.json"))
    return cands[-1] if cands else None


def _find_eval_summary(run_dir: Path, sidecar: dict[str, Any] | None) -> Path | None:
    if isinstance(sidecar, dict):
        ev = sidecar.get("eval_summary")
        if isinstance(ev, str) and ev.strip():
            p = Path(ev)
            if p.exists():
                return p
    cands = sorted((run_dir / "checkpoints" / "protocol_eval").glob("protocol_val_main_step_*.json"))
    return cands[-1] if cands else None


def _selected_step(sidecar: dict[str, Any] | None, eval_path: Path | None) -> int | None:
    if isinstance(sidecar, dict):
        v = sidecar.get("selected_step")
        if isinstance(v, int):
            return v
        try:
            if v is not None:
                return int(v)
        except Exception:
            pass
    if eval_path is not None:
        m = STEP_RE.search(eval_path.name)
        if m:
            return int(m.group(1))
    return None


def _extract_metrics(sidecar: dict[str, Any] | None, eval_obj: dict[str, Any] | None) -> dict[str, float | None]:
    sm = sidecar.get("metrics", {}) if isinstance(sidecar, dict) else {}
    em = eval_obj.get("metrics", {}) if isinstance(eval_obj, dict) else {}
    if not isinstance(sm, dict):
        sm = {}
    if not isinstance(em, dict):
        em = {}

    def g(k: str) -> float | None:
        if k in sm:
            return _safe_float(sm.get(k))
        if k in em:
            return _safe_float(em.get(k))
        return None

    return {
        "query_localization_error": g("query_localization_error"),
        "query_top1_acc": g("query_top1_acc"),
        "future_trajectory_l1": g("future_trajectory_l1"),
        "future_mask_iou": g("future_mask_iou"),
        "identity_consistency": g("identity_consistency"),
        "identity_switch_rate": g("identity_switch_rate"),
    }


def _collect_run(run_name: str, status_dir: Path, run_dir: Path) -> dict[str, Any]:
    status_path = _find_latest_status(status_dir, run_name)
    status = _load_json(status_path) if status_path is not None else None

    train_log = run_dir / "train_log.jsonl"
    rows, max_step = _count_train_rows_and_step(train_log)

    sidecar_path = _find_sidecar(run_dir)
    sidecar = _load_json(sidecar_path) if sidecar_path is not None else None
    eval_path = _find_eval_summary(run_dir, sidecar)
    eval_obj = _load_json(eval_path) if eval_path is not None else None

    info: dict[str, Any] = {
        "run_name": run_name,
        "state": str(status.get("state", "missing")) if isinstance(status, dict) else "missing",
        "job_id": str(status.get("job_id", "")) if isinstance(status, dict) else "",
        "status_file": str(status_path) if status_path is not None else "",
        "main_log": str(status.get("main_log", "")) if isinstance(status, dict) else "",
        "output_dir": str(run_dir),
        "train_log": str(train_log),
        "train_log_rows": int(rows),
        "train_max_step": int(max_step),
        "selection_sidecar": str(sidecar_path) if sidecar_path is not None else "",
        "eval_summary": str(eval_path) if eval_path is not None else "",
        "selected_best_step": _selected_step(sidecar, eval_path),
        "metrics": _extract_metrics(sidecar, eval_obj),
    }
    return info


def _terminal_state(s: str) -> bool:
    return str(s).lower() in {"done", "failed"}


def _fmt(v: Any) -> str:
    x = _safe_float(v)
    if x is None:
        return "na"
    return f"{x:.6f}"


def main() -> None:
    args = build_parser().parse_args()
    status_dir = Path(args.queue_status_dir)
    submit_json_path = Path(args.submit_json)
    poll_seconds = max(15, int(args.poll_seconds))
    timeout_seconds = max(0, int(args.timeout_seconds))

    submit = _load_json(submit_json_path)
    if not isinstance(submit, dict):
        raise SystemExit(f"submit json missing: {submit_json_path}")

    two_path_run = str(submit.get("run_name", "two_path_residual_seed123_challenge_v1"))
    two_path_out_dir = Path(str(submit.get("output_dir", "")).strip())
    if not str(two_path_out_dir):
        two_path_out_dir = Path(
            "/home/chen034/workspace/stwm/outputs/training/"
            "stwm_v4_2_220m_protocol_frozen_frontend_replication_seed123_v1/seed_123/"
            "two_path_residual_seed123_challenge_v1"
        )

    seed123_root = Path(
        "/home/chen034/workspace/stwm/outputs/training/"
        "stwm_v4_2_220m_protocol_frozen_frontend_replication_seed123_v1/seed_123"
    )

    refs = {
        "baseline": ("full_v4_2_seed123_fixed_nowarm_lambda1_rerun_v2", seed123_root / "full_v4_2_seed123_fixed_nowarm_lambda1_rerun_v2"),
        "alpha050": ("full_v4_2_seed123_objbias_alpha050_replacement_v1", seed123_root / "full_v4_2_seed123_objbias_alpha050_replacement_v1"),
        "gated": ("full_v4_2_seed123_objbias_gated_replacement_challenge_v1", seed123_root / "full_v4_2_seed123_objbias_gated_replacement_challenge_v1"),
        "wo_object_bias": ("wo_object_bias_v4_2_seed123_control_v1", seed123_root / "wo_object_bias_v4_2_seed123_control_v1"),
    }

    start_ts = time.time()
    while True:
        two_info = _collect_run(two_path_run, status_dir=status_dir, run_dir=two_path_out_dir)
        print(
            "[two-path-seed123-watch] "
            f"state={two_info.get('state')} step={two_info.get('train_max_step')} "
            f"rows={two_info.get('train_log_rows')} sidecar={bool(two_info.get('selection_sidecar'))}",
            flush=True,
        )

        if _terminal_state(str(two_info.get("state", ""))):
            break

        if timeout_seconds > 0 and (time.time() - start_ts) >= timeout_seconds:
            break

        time.sleep(poll_seconds)

    two = _collect_run(two_path_run, status_dir=status_dir, run_dir=two_path_out_dir)
    baseline = _collect_run(refs["baseline"][0], status_dir=status_dir, run_dir=refs["baseline"][1])
    alpha = _collect_run(refs["alpha050"][0], status_dir=status_dir, run_dir=refs["alpha050"][1])
    gated = _collect_run(refs["gated"][0], status_dir=status_dir, run_dir=refs["gated"][1])
    wo = _collect_run(refs["wo_object_bias"][0], status_dir=status_dir, run_dir=refs["wo_object_bias"][1])

    cmp_vs_baseline = _cmp(two.get("metrics"), baseline.get("metrics"))
    cmp_vs_alpha = _cmp(two.get("metrics"), alpha.get("metrics"))
    cmp_vs_gated = _cmp(two.get("metrics"), gated.get("metrics"))
    cmp_vs_wo = _cmp(two.get("metrics"), wo.get("metrics"))

    two_ok = bool(
        str(two.get("state", "")).lower() == "done"
        and int(two.get("train_max_step", -1)) >= 2000
        and bool(two.get("selection_sidecar"))
        and bool(two.get("eval_summary"))
    )

    beats_baseline = bool(cmp_vs_baseline.get("lhs_beats_rhs_official") is True)
    beats_alpha = bool(cmp_vs_alpha.get("lhs_beats_rhs_official") is True)
    beats_gated = bool(cmp_vs_gated.get("lhs_beats_rhs_official") is True)
    beats_wo = bool(cmp_vs_wo.get("lhs_beats_rhs_official") is True)

    ties_alpha = bool(cmp_vs_alpha.get("lhs_ties_rhs_official") is True)
    ties_gated = bool(cmp_vs_gated.get("lhs_ties_rhs_official") is True)

    # Strict candidate gate for promotion candidacy.
    enough_for_promotion_candidate = bool(
        two_ok
        and beats_baseline
        and (beats_alpha or ties_alpha)
        and (beats_gated or ties_gated)
        and beats_wo
    )

    report: dict[str, Any] = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "submit_json": str(submit_json_path),
        "run": {
            "two_path_residual": two,
            "baseline": baseline,
            "alpha050": alpha,
            "gated": gated,
            "wo_object_bias": wo,
        },
        "official_comparisons": {
            "two_vs_baseline": cmp_vs_baseline,
            "two_vs_alpha050": cmp_vs_alpha,
            "two_vs_gated": cmp_vs_gated,
            "two_vs_wo_object_bias": cmp_vs_wo,
        },
        "required_answers": {
            "two_path_wins_current_full_baseline": beats_baseline,
            "two_path_wins_alpha050": beats_alpha,
            "two_path_wins_gated": beats_gated,
            "two_path_vs_wo_object_bias": {
                "wins_official": beats_wo,
                "delta_query_localization_error": cmp_vs_wo.get("delta_query_localization_error"),
                "delta_query_top1_acc": cmp_vs_wo.get("delta_query_top1_acc"),
                "delta_future_trajectory_l1": cmp_vs_wo.get("delta_future_trajectory_l1"),
            },
            "enough_as_new_object_mainline_promotion_candidate": enough_for_promotion_candidate,
        },
    }

    out_report = Path(args.out_report)
    out_doc = Path(args.out_doc)
    out_report.parent.mkdir(parents=True, exist_ok=True)
    out_doc.parent.mkdir(parents=True, exist_ok=True)
    out_report.write_text(json.dumps(report, indent=2, ensure_ascii=False))

    lines: list[str] = []
    lines.append("# STWM Two Path Residual Seed123 Final Decision V1")
    lines.append("")
    lines.append(f"Generated: {report['generated_at']}")
    lines.append(f"Submit JSON: {report['submit_json']}")
    lines.append("")
    lines.append("## Completion")
    lines.append("")
    lines.append(f"- two_path_state: {two.get('state')}")
    lines.append(f"- two_path_step: {two.get('train_max_step')}")
    lines.append(f"- two_path_artifacts_ok: {two_ok}")
    lines.append("")
    lines.append("## Official Comparisons")
    lines.append("")
    lines.append("| comparison | two_path_beats_official | tie | d_qloc | d_qtop1 | d_l1 |")
    lines.append("|---|---|---|---:|---:|---:|")

    def add_row(name: str, cmp_obj: dict[str, Any]) -> None:
        lines.append(
            "| "
            f"{name} | {cmp_obj.get('lhs_beats_rhs_official')} | {cmp_obj.get('lhs_ties_rhs_official')} | "
            f"{_fmt(cmp_obj.get('delta_query_localization_error'))} | "
            f"{_fmt(cmp_obj.get('delta_query_top1_acc'))} | "
            f"{_fmt(cmp_obj.get('delta_future_trajectory_l1'))} |"
        )

    add_row("two vs current full baseline", cmp_vs_baseline)
    add_row("two vs alpha050", cmp_vs_alpha)
    add_row("two vs gated", cmp_vs_gated)
    add_row("two vs wo_object_bias", cmp_vs_wo)
    lines.append("")

    lines.append("## Required Answers")
    lines.append("")
    lines.append(f"1) two_path wins current full baseline: {beats_baseline}")
    lines.append(f"2) two_path wins alpha050: {beats_alpha}")
    lines.append(f"3) two_path wins gated: {beats_gated}")
    lines.append(
        "4) two_path vs wo_object_bias: "
        f"wins={beats_wo}, d_qloc={_fmt(cmp_vs_wo.get('delta_query_localization_error'))}, "
        f"d_qtop1={_fmt(cmp_vs_wo.get('delta_query_top1_acc'))}, "
        f"d_l1={_fmt(cmp_vs_wo.get('delta_future_trajectory_l1'))}"
    )
    lines.append(
        "5) enough as new object mainline promotion candidate: "
        f"{enough_for_promotion_candidate}"
    )
    lines.append("")

    out_doc.write_text("\n".join(lines) + "\n")

    print(str(out_report))
    print(str(out_doc))


if __name__ == "__main__":
    main()
