from __future__ import annotations

from pathlib import Path
import json
import math
import re
import time
from typing import Any


REPO_ROOT = Path("/home/chen034/workspace/stwm")
STATUS_DIR = REPO_ROOT / "outputs/queue/stwm_protocol_v2_frontend_default_v1/d1_train/status"
RUN_ROOT = REPO_ROOT / "outputs/training/stwm_v4_2_220m_protocol_frozen_frontend_replication_seed123_v1/seed_123"

BLINDBOX_PATH = REPO_ROOT / "reports/stwm_seed42_objdiag_blindbox_readonly_v1.json"

OUT_JSON = REPO_ROOT / "reports/stwm_gated_challenge_seed123_final_decision_v1.json"
OUT_DOC = REPO_ROOT / "docs/STWM_GATED_CHALLENGE_SEED123_FINAL_DECISION_V1.md"
OUT_PLAN = REPO_ROOT / "docs/STWM_POST_GATED_LOCKED_PLAN_V1.md"

BG_STATUS = REPO_ROOT / "outputs/background_jobs/stwm_gated_challenge_seed123_finalizer_v1.status.json"

RUNS = {
    "gated_challenge": "full_v4_2_seed123_objbias_gated_replacement_challenge_v1",
    "baseline": "full_v4_2_seed123_fixed_nowarm_lambda1_rerun_v2",
    "alpha050": "full_v4_2_seed123_objbias_alpha050_replacement_v1",
    "wo_object_bias": "wo_object_bias_v4_2_seed123_control_v1",
}

STEP_RE = re.compile(r"step_(\d+)\.json$")


def now_ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def write_bg_status(stage: str, payload: dict[str, Any]) -> None:
    BG_STATUS.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "stage": stage,
        "update_ts": now_ts(),
        **payload,
    }
    BG_STATUS.write_text(json.dumps(data, indent=2, ensure_ascii=False))


def safe_float(v: Any) -> float | None:
    try:
        x = float(v)
    except Exception:
        return None
    if math.isnan(x) or math.isinf(x):
        return None
    return x


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        obj = json.loads(path.read_text())
    except Exception:
        return None
    return obj if isinstance(obj, dict) else None


def latest_status_path(run_name: str) -> Path | None:
    cands = sorted(STATUS_DIR.glob(f"*_{run_name}.status.json"))
    return cands[-1] if cands else None


def train_max_step(train_log: Path) -> int:
    if not train_log.exists():
        return -1
    mx = -1
    with train_log.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                j = json.loads(s)
                step = int(j.get("step", -1))
            except Exception:
                continue
            if step > mx:
                mx = step
    return mx


def find_sidecar(run_dir: Path) -> Path | None:
    p = run_dir / "checkpoints/best_protocol_main_selection.json"
    if p.exists():
        return p
    cands = sorted(run_dir.glob("**/*selection*.json"))
    return cands[-1] if cands else None


def find_best_ckpt(run_dir: Path) -> Path | None:
    p = run_dir / "checkpoints/best_protocol_main.pt"
    if p.exists():
        return p
    cands = sorted(run_dir.glob("**/best_protocol_main.pt"))
    return cands[-1] if cands else None


def find_eval_summary(run_dir: Path, sidecar: dict[str, Any] | None) -> Path | None:
    if isinstance(sidecar, dict):
        ev = sidecar.get("eval_summary")
        if isinstance(ev, str) and ev.strip():
            p = Path(ev)
            if p.exists():
                return p
    cands = sorted((run_dir / "checkpoints/protocol_eval").glob("protocol_val_main_step_*.json"))
    return cands[-1] if cands else None


def selected_step(sidecar: dict[str, Any] | None, eval_path: Path | None) -> int | None:
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


def extract_metrics(sidecar: dict[str, Any] | None, eval_obj: dict[str, Any] | None) -> dict[str, float | None]:
    sm = sidecar.get("metrics", {}) if isinstance(sidecar, dict) else {}
    em = eval_obj.get("metrics", {}) if isinstance(eval_obj, dict) else {}
    if not isinstance(sm, dict):
        sm = {}
    if not isinstance(em, dict):
        em = {}

    def g(k: str) -> float | None:
        if k in sm:
            return safe_float(sm.get(k))
        if k in em:
            return safe_float(em.get(k))
        return None

    return {
        "query_localization_error": g("query_localization_error"),
        "query_top1_acc": g("query_top1_acc"),
        "future_trajectory_l1": g("future_trajectory_l1"),
        "future_mask_iou": g("future_mask_iou"),
        "identity_consistency": g("identity_consistency"),
        "identity_switch_rate": g("identity_switch_rate"),
    }


def official_key(metrics: dict[str, float | None]) -> tuple[float, float, float] | None:
    qloc = metrics.get("query_localization_error")
    qtop = metrics.get("query_top1_acc")
    l1 = metrics.get("future_trajectory_l1")
    if qloc is None or qtop is None or l1 is None:
        return None
    return (float(qloc), -float(qtop), float(l1))


def collect_run(run_name: str) -> dict[str, Any]:
    status_path = latest_status_path(run_name)
    status = load_json(status_path) if status_path is not None else None
    run_dir = RUN_ROOT / run_name
    train_log = run_dir / "train_log.jsonl"
    sidecar_path = find_sidecar(run_dir)
    sidecar = load_json(sidecar_path) if sidecar_path is not None else None
    eval_path = find_eval_summary(run_dir, sidecar)
    eval_obj = load_json(eval_path) if eval_path is not None else None
    best_ckpt = find_best_ckpt(run_dir)

    metrics = extract_metrics(sidecar, eval_obj)

    info: dict[str, Any] = {
        "run_name": run_name,
        "state": str(status.get("state", "missing")) if isinstance(status, dict) else "missing",
        "job_id": str(status.get("job_id", "")) if isinstance(status, dict) else "",
        "status_file": str(status_path) if status_path is not None else "",
        "main_log": str(status.get("main_log", "")) if isinstance(status, dict) else "",
        "output_dir": str(run_dir),
        "train_log": str(train_log),
        "train_max_step": train_max_step(train_log),
        "best_protocol_main": str(best_ckpt) if best_ckpt is not None else "",
        "selection_sidecar": str(sidecar_path) if sidecar_path is not None else "",
        "eval_summary": str(eval_path) if eval_path is not None else "",
        "selected_best_step": selected_step(sidecar, eval_path),
        "metrics": metrics,
    }
    return info


def compare(gated: dict[str, Any], other: dict[str, Any]) -> dict[str, Any]:
    gm = gated["metrics"]
    om = other["metrics"]
    gk = official_key(gm)
    ok = official_key(om)
    return {
        "gated_beats_official": bool(gk is not None and ok is not None and gk < ok),
        "gated_key": gk,
        "other_key": ok,
        "delta_query_localization_error": (
            None
            if gm.get("query_localization_error") is None or om.get("query_localization_error") is None
            else float(gm["query_localization_error"] - om["query_localization_error"])
        ),
        "delta_query_top1_acc": (
            None
            if gm.get("query_top1_acc") is None or om.get("query_top1_acc") is None
            else float(gm["query_top1_acc"] - om["query_top1_acc"])
        ),
        "delta_future_trajectory_l1": (
            None
            if gm.get("future_trajectory_l1") is None or om.get("future_trajectory_l1") is None
            else float(gm["future_trajectory_l1"] - om["future_trajectory_l1"])
        ),
    }


def write_outputs(report: dict[str, Any]) -> None:
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_DOC.parent.mkdir(parents=True, exist_ok=True)
    OUT_PLAN.parent.mkdir(parents=True, exist_ok=True)

    OUT_JSON.write_text(json.dumps(report, indent=2, ensure_ascii=False))

    lines: list[str] = []
    lines.append("# STWM Gated Challenge Seed123 Final Decision V1")
    lines.append("")
    lines.append(f"Generated: {report['generated_at']}")
    lines.append("")
    lines.append("## A. Completion Verification")
    lines.append("")
    lines.append(f"- challenge_complete: {report['completion']['challenge_complete']}")
    if report['completion']['anomalies']:
        lines.append(f"- anomalies: {', '.join(report['completion']['anomalies'])}")
    lines.append("")

    lines.append("## B. Seed123 Metrics")
    lines.append("")
    lines.append("| run | selected_best_step | qloc | qtop1 | l1 | iou | id_consistency | id_switch_rate |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for key in ("gated_challenge", "baseline", "alpha050", "wo_object_bias"):
        r = report['runs'][key]
        m = r['metrics']
        lines.append(
            "| "
            f"{r['run_name']} | {r.get('selected_best_step')} | {m.get('query_localization_error')} | {m.get('query_top1_acc')} | "
            f"{m.get('future_trajectory_l1')} | {m.get('future_mask_iou')} | {m.get('identity_consistency')} | {m.get('identity_switch_rate')} |"
        )
    lines.append("")

    lines.append("## C. Cross-Seed Judgment")
    lines.append("")
    lines.append(f"- seed42_gated_beats_wo_object_bias_in_blindbox: {report['cross_seed']['seed42_gated_beats_wo_object_bias_in_blindbox']}")
    lines.append(f"- seed123_gated_beats_baseline: {report['cross_seed']['seed123_gated_beats_baseline']}")
    lines.append(f"- seed123_gated_beats_wo_object_bias: {report['cross_seed']['seed123_gated_beats_wo_object_bias']}")
    lines.append(f"- gated_signal_cross_seed_stable: {report['cross_seed']['gated_signal_cross_seed_stable']}")
    lines.append("")

    lines.append("## D. Locked Next Mainline")
    lines.append("")
    lines.append(f"- chosen_conclusion: {report['locked_mainline']['conclusion']}")
    lines.append(f"- recommendation: {report['locked_mainline']['recommendation']}")
    lines.append("")
    OUT_DOC.write_text("\n".join(lines) + "\n")

    plan: list[str] = []
    plan.append("# STWM Post Gated Locked Plan V1")
    plan.append("")
    plan.append(f"Generated: {report['generated_at']}")
    plan.append(f"Locked conclusion: {report['locked_mainline']['conclusion']}")
    plan.append("")
    plan.append("Unique next mainline:")
    plan.append(f"- {report['locked_mainline']['recommendation']}")
    plan.append("")
    OUT_PLAN.write_text("\n".join(plan) + "\n")


def main() -> None:
    write_bg_status(
        "started",
        {
            "runs": RUNS,
            "outputs": {
                "report_json": str(OUT_JSON),
                "report_doc": str(OUT_DOC),
                "plan_doc": str(OUT_PLAN),
            },
        },
    )

    runs = {k: collect_run(v) for k, v in RUNS.items()}
    gated = runs["gated_challenge"]

    checks = {
        "state_done": gated["state"].lower() == "done",
        "step_reached": int(gated["train_max_step"]) >= 2000,
        "best_ckpt_exists": bool(gated["best_protocol_main"] and Path(gated["best_protocol_main"]).exists()),
        "sidecar_exists": bool(gated["selection_sidecar"] and Path(gated["selection_sidecar"]).exists()),
        "eval_exists": bool(gated["eval_summary"] and Path(gated["eval_summary"]).exists()),
    }
    anomalies = [k for k, v in checks.items() if not v]

    if anomalies:
        report = {
            "generated_at": now_ts(),
            "completion": {
                "challenge_complete": False,
                "anomalies": anomalies,
            },
            "runs": runs,
            "official": {},
            "cross_seed": {},
            "locked_mainline": {
                "conclusion": "aborted_due_to_challenge_artifact_anomaly",
                "recommendation": "fix challenge artifact integrity first; do not draw mainline conclusion",
            },
        }
        write_outputs(report)
        write_bg_status(
            "done",
            {
                "challenge_complete": False,
                "anomalies": anomalies,
                "report_json": str(OUT_JSON),
                "report_doc": str(OUT_DOC),
                "plan_doc": str(OUT_PLAN),
            },
        )
        return

    cmp_baseline = compare(gated, runs["baseline"])
    cmp_alpha = compare(gated, runs["alpha050"])
    cmp_wo = compare(gated, runs["wo_object_bias"])

    blindbox = load_json(BLINDBOX_PATH) or {}
    bb_runs = blindbox.get("runs", {}) if isinstance(blindbox.get("runs", {}), dict) else {}
    bb_gated = bb_runs.get("gated", {}) if isinstance(bb_runs.get("gated", {}), dict) else {}
    seed42_gated_beats_wo = bool(bb_gated.get("beats_ref_official", False))

    seed123_gated_beats_baseline = bool(cmp_baseline["gated_beats_official"])
    seed123_gated_beats_alpha = bool(cmp_alpha["gated_beats_official"])
    seed123_gated_beats_wo = bool(cmp_wo["gated_beats_official"])

    gated_signal_cross_seed_stable = bool(seed42_gated_beats_wo and seed123_gated_beats_baseline)

    if seed123_gated_beats_baseline and seed123_gated_beats_alpha:
        conclusion = "conclusion_1_switch_mainline_to_gated_clean_matrix_challenge"
        recommendation = "switch mainline to gated clean matrix challenge"
    elif seed123_gated_beats_baseline and (not seed123_gated_beats_alpha or not seed123_gated_beats_wo):
        conclusion = "conclusion_2_minimal_fancy_upgrade_design"
        recommendation = "do not set gated as mainline yet; design minimal fancy upgrade scheme"
    else:
        conclusion = "conclusion_3_stop_gated_mainline_turn_to_systematic_combo_design"
        recommendation = "stop gated mainline and move to delayed-only/two-path-only/combined systematic design"

    qloc_gap = cmp_wo.get("delta_query_localization_error")
    qtop_gap = cmp_wo.get("delta_query_top1_acc")
    l1_gap = cmp_wo.get("delta_future_trajectory_l1")

    report = {
        "generated_at": now_ts(),
        "completion": {
            "challenge_complete": True,
            "anomalies": [],
            "checks": checks,
        },
        "runs": runs,
        "official": {
            "gated_vs_baseline": cmp_baseline,
            "gated_vs_alpha050": cmp_alpha,
            "gated_vs_wo_object_bias": cmp_wo,
        },
        "cross_seed": {
            "seed42_blindbox_report": str(BLINDBOX_PATH),
            "seed42_gated_beats_wo_object_bias_in_blindbox": seed42_gated_beats_wo,
            "seed123_gated_beats_baseline": seed123_gated_beats_baseline,
            "seed123_gated_beats_alpha050": seed123_gated_beats_alpha,
            "seed123_gated_beats_wo_object_bias": seed123_gated_beats_wo,
            "gated_signal_cross_seed_stable": gated_signal_cross_seed_stable,
            "gated_vs_wo_object_bias_gap": {
                "query_localization_error_delta": qloc_gap,
                "query_top1_acc_delta": qtop_gap,
                "future_trajectory_l1_delta": l1_gap,
            },
        },
        "locked_mainline": {
            "conclusion": conclusion,
            "recommendation": recommendation,
        },
    }

    write_outputs(report)
    write_bg_status(
        "done",
        {
            "challenge_complete": True,
            "report_json": str(OUT_JSON),
            "report_doc": str(OUT_DOC),
            "plan_doc": str(OUT_PLAN),
            "locked_mainline": report["locked_mainline"],
        },
    )


if __name__ == "__main__":
    main()
