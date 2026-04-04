from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import csv
import json
import math
import os
import re
import statistics
import subprocess
import time
from typing import Any


ROLE_ORDER = [
    "baseline_full",
    "replacement_alpha050",
    "wo_semantics_control",
    "wo_object_bias_control",
]

ROLE_LABEL = {
    "baseline_full": "current full rerun baseline",
    "replacement_alpha050": "alpha050 replacement",
    "wo_semantics_control": "wo_semantics control",
    "wo_object_bias_control": "wo_object_bias control",
}

CANONICAL_NAMES = {
    "baseline_full": "full_v4_2_seed42_fixed_nowarm_lambda1_rerun_v2",
    "replacement_alpha050": "full_v4_2_seed42_objbias_alpha050_replacement_v1",
    "wo_semantics_control": "wo_semantics_v4_2_seed42_control_v1",
    "wo_object_bias_control": "wo_object_bias_v4_2_seed42_control_v1",
}

SEED123_NAMES = {
    "baseline_full": "full_v4_2_seed123_fixed_nowarm_lambda1_rerun_v2",
    "replacement_alpha050": "full_v4_2_seed123_objbias_alpha050_replacement_v1",
    "wo_semantics_control": "wo_semantics_v4_2_seed123_control_v1",
    "wo_object_bias_control": "wo_object_bias_v4_2_seed123_control_v1",
}

CACHE_RE = re.compile(
    r"cache miss|cache rebuild|frontend[_ -]?cache.*(error|fail|corrupt|rebuild)|missing shard|checksum",
    re.IGNORECASE,
)
NAN_RE = re.compile(r"\bnan\b|\binf\b|overflow|underflow", re.IGNORECASE)
INSTABILITY_RE = re.compile(
    r"traceback|runtimeerror|assertionerror|floatingpointerror|cuda error|illegal memory access",
    re.IGNORECASE,
)
STEP_RE = re.compile(r"step_(\d+)\.json$")
STEPS_ARG_RE = re.compile(r"--steps\s+([0-9]+)")


def now_ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        obj = json.loads(path.read_text())
    except Exception:
        return None
    if isinstance(obj, dict):
        return obj
    return None


def safe_float(v: Any) -> float | None:
    try:
        x = float(v)
    except Exception:
        return None
    if math.isnan(x) or math.isinf(x):
        return None
    return x


def latest_file(glob_iter: list[Path]) -> Path | None:
    if not glob_iter:
        return None
    return sorted(glob_iter, key=lambda p: p.stat().st_mtime)[-1]


def latest_status_path(status_dir: Path, run_name: str) -> Path | None:
    return latest_file(list(status_dir.glob(f"*_{run_name}.status.json")))


def parse_submit_tsv(tsv_path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    if not tsv_path.exists():
        return rows
    with tsv_path.open("r", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            rows.append({k: str(v or "") for k, v in row.items()})
    return rows


def latest_seed42_submit_tsv(report_root: Path) -> Path | None:
    return latest_file(list(report_root.glob("stwm_replacement_clean_matrix_seed42_submit_v1_*.tsv")))


def run_name_from_status_file(status_file: Path) -> str:
    stem = status_file.name.replace(".status.json", "")
    parts = stem.split("_", 1)
    if len(parts) == 2:
        return parts[1]
    return stem


def role_match(role: str, run_name: str) -> bool:
    n = run_name.lower()
    if role == "baseline_full":
        return (
            "full_v4_2" in n
            and "seed42" in n
            and "objbias" not in n
            and ("rerun" in n or "fixed_nowarm" in n)
        )
    if role == "replacement_alpha050":
        return "seed42" in n and "objbias" in n and ("alpha050" in n or "alpha50" in n)
    if role == "wo_semantics_control":
        return "seed42" in n and "wo_semantics" in n
    if role == "wo_object_bias_control":
        return "seed42" in n and "wo_object_bias" in n
    return False


def resolve_seed42_runs(status_dir: Path, report_root: Path) -> tuple[dict[str, str], dict[str, dict[str, str]], str]:
    by_name: dict[str, dict[str, str]] = {}
    source = "canonical_only"

    tsv = latest_seed42_submit_tsv(report_root)
    if tsv is not None:
        source = str(tsv)
        for row in parse_submit_tsv(tsv):
            name = row.get("run_name", "").strip()
            if name:
                by_name[name] = row

    for sp in status_dir.glob("*seed42*.status.json"):
        name = run_name_from_status_file(sp)
        if name and name not in by_name:
            by_name[name] = {
                "run_name": name,
                "status_file": str(sp),
                "main_log": "",
                "output_dir": "",
                "job_id": "",
            }

    resolved: dict[str, str] = {}
    names = sorted(by_name.keys())
    for role in ROLE_ORDER:
        cands = [n for n in names if role_match(role, n)]
        if cands:
            cands.sort(
                key=lambda n: (
                    latest_status_path(status_dir, n).stat().st_mtime if latest_status_path(status_dir, n) else -1.0,
                    n,
                )
            )
            resolved[role] = cands[-1]
        else:
            resolved[role] = CANONICAL_NAMES[role]

    return resolved, by_name, source


def parse_steps_from_main_log(main_log: Path) -> int | None:
    if not main_log.exists():
        return None
    try:
        with main_log.open("r", encoding="utf-8", errors="ignore") as f:
            for _ in range(400):
                line = f.readline()
                if not line:
                    break
                m = STEPS_ARG_RE.search(line)
                if m:
                    return int(m.group(1))
    except Exception:
        return None
    return None


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    p = max(0.0, min(1.0, float(p)))
    idx = (len(s) - 1) * p
    lo = int(math.floor(idx))
    hi = int(math.ceil(idx))
    if lo == hi:
        return s[lo]
    frac = idx - lo
    return s[lo] + (s[hi] - s[lo]) * frac


def read_train_log(train_log: Path) -> dict[str, Any]:
    step_times: list[float] = []
    data_times: list[float] = []
    wait_ratios: list[float] = []
    rows = 0
    max_step = -1
    nan_rows = 0
    bad_json_rows = 0

    if not train_log.exists():
        return {
            "exists": False,
            "rows": 0,
            "max_step": -1,
            "step_times": [],
            "data_times": [],
            "wait_ratios": [],
            "nan_rows": 0,
            "bad_json_rows": 0,
        }

    with train_log.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            text = line.strip()
            if not text:
                continue
            rows += 1
            try:
                obj = json.loads(text)
            except Exception:
                bad_json_rows += 1
                continue

            step = obj.get("step")
            try:
                step_i = int(step)
                if step_i > max_step:
                    max_step = step_i
            except Exception:
                pass

            st = obj.get("step_time_s")
            dt = obj.get("data_time_s")
            wr = obj.get("data_wait_ratio")

            finite_ok = True
            for v in (st, dt, wr):
                try:
                    fv = float(v)
                    if math.isnan(fv) or math.isinf(fv):
                        finite_ok = False
                        break
                except Exception:
                    continue
            if not finite_ok:
                nan_rows += 1

            stf = safe_float(st)
            dtf = safe_float(dt)
            wrf = safe_float(wr)
            if stf is not None:
                step_times.append(stf)
            if dtf is not None:
                data_times.append(dtf)
            if wrf is not None:
                wait_ratios.append(wrf)

    return {
        "exists": True,
        "rows": rows,
        "max_step": max_step,
        "step_times": step_times,
        "data_times": data_times,
        "wait_ratios": wait_ratios,
        "nan_rows": nan_rows,
        "bad_json_rows": bad_json_rows,
    }


def timing_summary(step_times: list[float], data_times: list[float], wait_ratios: list[float]) -> dict[str, float | None]:
    def mean_or_none(xs: list[float]) -> float | None:
        return statistics.fmean(xs) if xs else None

    recent_step = step_times[-500:] if len(step_times) > 500 else list(step_times)
    recent_wait = wait_ratios[-500:] if len(wait_ratios) > 500 else list(wait_ratios)

    return {
        "full_mean_step_time_s": mean_or_none(step_times),
        "full_mean_data_time_s": mean_or_none(data_times),
        "full_mean_data_wait_ratio": mean_or_none(wait_ratios),
        "recent500_mean_step_time_s": mean_or_none(recent_step),
        "recent500_mean_data_wait_ratio": mean_or_none(recent_wait),
        "step_time_p50_s": percentile(step_times, 0.50),
        "step_time_p95_s": percentile(step_times, 0.95),
    }


def scan_log_signals(main_log: Path) -> dict[str, Any]:
    out = {
        "cache_issue_count": 0,
        "nan_issue_count": 0,
        "instability_issue_count": 0,
        "cache_examples": [],
        "nan_examples": [],
        "instability_examples": [],
    }
    if not main_log.exists() or not main_log.is_file():
        return out

    with main_log.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if CACHE_RE.search(line):
                out["cache_issue_count"] += 1
                if len(out["cache_examples"]) < 5:
                    out["cache_examples"].append(line.strip())
            if NAN_RE.search(line):
                out["nan_issue_count"] += 1
                if len(out["nan_examples"]) < 5:
                    out["nan_examples"].append(line.strip())
            if INSTABILITY_RE.search(line):
                out["instability_issue_count"] += 1
                if len(out["instability_examples"]) < 5:
                    out["instability_examples"].append(line.strip())
    return out


def find_best_checkpoint(output_dir: Path) -> Path | None:
    p = output_dir / "checkpoints" / "best_protocol_main.pt"
    if p.exists():
        return p
    cands = sorted(output_dir.glob("**/best_protocol_main.pt"))
    return cands[-1] if cands else None


def find_selection_sidecar(output_dir: Path) -> Path | None:
    p = output_dir / "checkpoints" / "best_protocol_main_selection.json"
    if p.exists():
        return p
    cands = sorted(output_dir.glob("**/*selection*.json"))
    return cands[-1] if cands else None


def infer_selected_step(sidecar: dict[str, Any] | None, eval_summary: Path | None) -> int | None:
    if sidecar is not None:
        v = sidecar.get("selected_step")
        if isinstance(v, int):
            return v
        try:
            return int(v)
        except Exception:
            pass
    if eval_summary is not None:
        m = STEP_RE.search(eval_summary.name)
        if m:
            return int(m.group(1))
    return None


def extract_metrics(sidecar: dict[str, Any] | None, eval_summary_obj: dict[str, Any] | None) -> dict[str, float | None]:
    sm = sidecar.get("metrics", {}) if isinstance(sidecar, dict) else {}
    if not isinstance(sm, dict):
        sm = {}
    em = eval_summary_obj.get("metrics", {}) if isinstance(eval_summary_obj, dict) else {}
    if not isinstance(em, dict):
        em = {}

    def g(*keys: str) -> float | None:
        for k in keys:
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
    qtop1 = metrics.get("query_top1_acc")
    l1 = metrics.get("future_trajectory_l1")
    if qloc is None or qtop1 is None or l1 is None:
        return None
    return (float(qloc), -float(qtop1), float(l1))


def collect_run(
    role: str,
    run_name: str,
    status_dir: Path,
    default_run_root: Path,
    by_name_row: dict[str, dict[str, str]],
) -> dict[str, Any]:
    row = by_name_row.get(run_name, {})
    status_path = latest_status_path(status_dir, run_name)
    status = load_json(status_path) if status_path is not None else None

    out_dir_text = row.get("output_dir", "").strip()
    output_dir = Path(out_dir_text) if out_dir_text else (default_run_root / run_name)

    main_log: Path | None = None
    if status is not None:
        st_main_log = status.get("main_log")
        if isinstance(st_main_log, str) and st_main_log.strip():
            main_log = Path(st_main_log)
        else:
            main_log = Path(row.get("main_log", "")) if row.get("main_log", "").strip() else None
    else:
        main_log = Path(row.get("main_log", "")) if row.get("main_log", "").strip() else None

    train_log = output_dir / "train_log.jsonl"
    train = read_train_log(train_log)

    sidecar_path = find_selection_sidecar(output_dir)
    sidecar = load_json(sidecar_path) if sidecar_path is not None else None

    eval_summary_path: Path | None = None
    if isinstance(sidecar, dict):
        ev = sidecar.get("eval_summary")
        if isinstance(ev, str) and ev.strip():
            p = Path(ev)
            if p.exists():
                eval_summary_path = p

    if eval_summary_path is None:
        cands = sorted((output_dir / "checkpoints" / "protocol_eval").glob("protocol_val_main_step_*.json"))
        if cands:
            eval_summary_path = cands[-1]

    eval_summary = load_json(eval_summary_path) if eval_summary_path is not None else None
    best_ckpt = find_best_checkpoint(output_dir)

    planned_steps = None
    if main_log is not None:
        planned_steps = parse_steps_from_main_log(main_log)
    if planned_steps is None:
        planned_steps = 2000

    metrics = extract_metrics(sidecar, eval_summary)

    checks = {
        "state_done": bool(status is not None and str(status.get("state", "")).lower() == "done"),
        "train_log_exists": bool(train.get("exists", False)),
        "train_final_step_reached": bool(int(train.get("max_step", -1)) >= int(planned_steps)),
        "best_protocol_main_exists": bool(best_ckpt is not None and best_ckpt.exists()),
        "selection_sidecar_exists": bool(sidecar_path is not None and sidecar_path.exists()),
        "eval_summary_exists": bool(eval_summary_path is not None and eval_summary_path.exists()),
    }

    anomalies: list[str] = []
    if not checks["state_done"]:
        anomalies.append("state_not_done")
    if not checks["train_log_exists"]:
        anomalies.append("train_log_missing")
    if not checks["train_final_step_reached"]:
        anomalies.append("train_final_step_not_reached")
    if not checks["best_protocol_main_exists"]:
        anomalies.append("best_protocol_main_missing")
    if not checks["selection_sidecar_exists"]:
        anomalies.append("selection_sidecar_missing")
    if not checks["eval_summary_exists"]:
        anomalies.append("eval_summary_missing")

    timing = timing_summary(
        train.get("step_times", []),
        train.get("data_times", []),
        train.get("wait_ratios", []),
    )

    signals = scan_log_signals(main_log) if main_log is not None else {
        "cache_issue_count": 0,
        "nan_issue_count": 0,
        "instability_issue_count": 0,
        "cache_examples": [],
        "nan_examples": [],
        "instability_examples": [],
    }

    info: dict[str, Any] = {
        "role": role,
        "role_label": ROLE_LABEL[role],
        "run_name": run_name,
        "job_id": str(status.get("job_id", "")) if isinstance(status, dict) else str(row.get("job_id", "")),
        "state": str(status.get("state", "missing")) if isinstance(status, dict) else "missing",
        "status_file": str(status_path) if status_path is not None else "",
        "main_log": str(main_log) if main_log is not None else "",
        "output_dir": str(output_dir),
        "planned_steps": int(planned_steps),
        "train_log": str(train_log),
        "train_log_rows": int(train.get("rows", 0)),
        "train_log_max_step": int(train.get("max_step", -1)),
        "selected_best_step": infer_selected_step(sidecar, eval_summary_path),
        "best_protocol_main": str(best_ckpt) if best_ckpt is not None else "",
        "selection_sidecar": str(sidecar_path) if sidecar_path is not None else "",
        "eval_summary": str(eval_summary_path) if eval_summary_path is not None else "",
        "metrics": metrics,
        "timing": timing,
        "nan_rows": int(train.get("nan_rows", 0)),
        "bad_json_rows": int(train.get("bad_json_rows", 0)),
        "log_signals": signals,
        "checks": checks,
        "anomalies": anomalies,
    }
    return info


def compare_pair(rep: dict[str, Any], other: dict[str, Any]) -> dict[str, Any]:
    rm = rep.get("metrics", {})
    om = other.get("metrics", {})
    rk_r = official_key(rm)
    rk_o = official_key(om)

    qloc_r = safe_float(rm.get("query_localization_error"))
    qloc_o = safe_float(om.get("query_localization_error"))
    qtop_r = safe_float(rm.get("query_top1_acc"))
    qtop_o = safe_float(om.get("query_top1_acc"))
    l1_r = safe_float(rm.get("future_trajectory_l1"))
    l1_o = safe_float(om.get("future_trajectory_l1"))

    return {
        "replacement_better_official": bool(rk_r is not None and rk_o is not None and rk_r < rk_o),
        "official_key_replacement": rk_r,
        "official_key_other": rk_o,
        "query_localization_error_delta": (None if qloc_r is None or qloc_o is None else float(qloc_r - qloc_o)),
        "query_top1_acc_delta": (None if qtop_r is None or qtop_o is None else float(qtop_r - qtop_o)),
        "future_trajectory_l1_delta": (None if l1_r is None or l1_o is None else float(l1_r - l1_o)),
    }


def collect_seed123_existing(status_dir: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seed123_root = Path(
        "/home/chen034/workspace/stwm/outputs/training/"
        "stwm_v4_2_220m_protocol_frozen_frontend_replication_seed123_v1/seed_123"
    )
    for role in ROLE_ORDER:
        run_name = SEED123_NAMES[role]
        sp = latest_status_path(status_dir, run_name)
        st = load_json(sp) if sp is not None else None
        if st is None:
            return []
        state = str(st.get("state", "")).lower()
        if state not in {"queued", "waiting_for_gpu", "running", "done"}:
            return []
        out.append(
            {
                "role": role,
                "run_name": run_name,
                "job_id": str(st.get("job_id", "")),
                "state": state,
                "status_file": str(sp),
                "main_log": str(st.get("main_log", "")),
                "output_dir": str(seed123_root / run_name),
            }
        )
    return out


def parse_seed123_submit_tsv_from_stdout(stdout: str) -> Path | None:
    for line in stdout.splitlines():
        if "submissions=" in line:
            path = line.split("submissions=", 1)[1].strip()
            if path:
                p = Path(path)
                if p.exists():
                    return p
    return None


def run_seed123_enqueue(repo_root: Path, preferred_gpu: str) -> tuple[list[dict[str, str]], str, str]:
    script = repo_root / "scripts" / "enqueue_stwm_protocol_v2_replication_clean_matrix_seed123_v1.sh"
    env = os.environ.copy()
    if preferred_gpu:
        env["STWM_D1_PREFERRED_GPU_ALL"] = preferred_gpu
    env["STWM_D1_SEED"] = "123"

    out = subprocess.check_output(["bash", str(script)], text=True, env=env, cwd=str(repo_root))
    tsv = parse_seed123_submit_tsv_from_stdout(out)
    rows = parse_submit_tsv(tsv) if tsv is not None else []
    return rows, (str(tsv) if tsv is not None else ""), out


def write_seed123_launch_doc(
    out_doc: Path,
    gate: dict[str, Any],
    launched_rows: list[dict[str, Any]],
    reused_existing: bool,
    submit_tsv: str,
) -> None:
    lines: list[str] = []
    lines.append("# STWM Seed123 Replication Launch V1")
    lines.append("")
    lines.append(f"Generated: {now_ts()}")
    lines.append(f"Gate passed: {gate.get('gate_pass')}")
    lines.append(f"Launch mode: {'reuse_existing' if reused_existing else 'new_submit'}")
    if submit_tsv:
        lines.append(f"Submit TSV: {submit_tsv}")
    lines.append("")
    lines.append("## Jobs")
    lines.append("")
    lines.append("| run_name | job_id | state | status_file | main_log | output_dir |")
    lines.append("|---|---|---|---|---|---|")
    for r in launched_rows:
        lines.append(
            "| "
            f"{r.get('run_name','')} | {r.get('job_id','')} | {r.get('state','')} | "
            f"{r.get('status_file','')} | {r.get('main_log','')} | {r.get('output_dir','')} |"
        )
    lines.append("")

    out_doc.parent.mkdir(parents=True, exist_ok=True)
    out_doc.write_text("\n".join(lines) + "\n")


def write_decision_doc(path: Path, report: dict[str, Any]) -> None:
    lines: list[str] = []
    lines.append("# STWM Replacement Clean Matrix Seed42 Final Decision V1")
    lines.append("")
    lines.append(f"Generated: {report['generated_at']}")
    lines.append(f"Run resolution source: {report['run_resolution_source']}")
    lines.append("")

    lines.append("## A. Completion Verification")
    lines.append("")
    lines.append(f"all_done_and_complete: {report['completion']['all_runs_complete']}")
    if report["completion"]["failures"]:
        lines.append("failures:")
        for x in report["completion"]["failures"]:
            lines.append(f"- {x}")
    lines.append("")

    lines.append("## B. Official Result Comparison")
    lines.append("")
    lines.append("| role | run_name | selected_best_step | qloc | qtop1 | l1 | mask_iou | id_consistency | id_switch_rate | eval_summary | selection_sidecar |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|")
    for item in report["runs"]:
        m = item.get("metrics", {})
        lines.append(
            "| "
            f"{item.get('role_label','')} | {item.get('run_name','')} | {item.get('selected_best_step')} | "
            f"{m.get('query_localization_error')} | {m.get('query_top1_acc')} | {m.get('future_trajectory_l1')} | "
            f"{m.get('future_mask_iou')} | {m.get('identity_consistency')} | {m.get('identity_switch_rate')} | "
            f"{item.get('eval_summary','')} | {item.get('selection_sidecar','')} |"
        )
    lines.append("")

    lines.append("official_answers:")
    oa = report["official_answers"]
    lines.append(f"- replacement_beats_baseline_official: {oa['replacement_beats_baseline_official']}")
    lines.append(f"- replacement_improvement_not_noise: {oa['replacement_improvement_not_noise']}")
    lines.append(f"- best_full_still_loses_to_wo_object_bias: {oa['best_full_still_loses_to_wo_object_bias']}")
    lines.append(f"- supports_alpha050_as_new_default_full: {oa['supports_alpha050_as_new_default_full']}")
    lines.append("")

    lines.append("## C. Efficiency And Stability")
    lines.append("")
    lines.append("| run_name | full_mean_step_time_s | full_mean_data_time_s | full_mean_data_wait_ratio | recent500_mean_step_time_s | recent500_mean_data_wait_ratio | p50_step_time_s | p95_step_time_s |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for item in report["runs"]:
        t = item.get("timing", {})
        lines.append(
            "| "
            f"{item.get('run_name','')} | {t.get('full_mean_step_time_s')} | {t.get('full_mean_data_time_s')} | "
            f"{t.get('full_mean_data_wait_ratio')} | {t.get('recent500_mean_step_time_s')} | {t.get('recent500_mean_data_wait_ratio')} | "
            f"{t.get('step_time_p50_s')} | {t.get('step_time_p95_s')} |"
        )
    lines.append("")
    lines.append("stability_answers:")
    sa = report["stability_answers"]
    lines.append(f"- frontend_cache_stable_and_fast: {sa['frontend_cache_stable_and_fast']}")
    lines.append(f"- cache_miss_rebuild_corruption_signal: {sa['cache_miss_rebuild_corruption_signal']}")
    lines.append(f"- nan_or_instability_signal: {sa['nan_or_instability_signal']}")
    lines.append(f"- replacement_slower_than_baseline: {sa['replacement_slower_than_baseline']}")
    lines.append(f"- replacement_more_unstable_than_baseline: {sa['replacement_more_unstable_than_baseline']}")
    lines.append("")

    lines.append("## D. Seed123 Promotion Gate")
    lines.append("")
    gate = report["promotion_gate"]
    lines.append(f"gate_pass: {gate['gate_pass']}")
    lines.append("conditions:")
    for k, v in gate["conditions"].items():
        lines.append(f"- {k}: {v}")
    lines.append(f"seed123_launched: {gate['seed123_launched']}")
    lines.append(f"seed123_reused_existing: {gate['seed123_reused_existing']}")
    if gate.get("seed123_submit_tsv"):
        lines.append(f"seed123_submit_tsv: {gate['seed123_submit_tsv']}")
    lines.append("")

    if gate.get("seed123_jobs"):
        lines.append("| seed123_run | job_id | state | status_file | main_log | output_dir |")
        lines.append("|---|---|---|---|---|---|")
        for r in gate["seed123_jobs"]:
            lines.append(
                "| "
                f"{r.get('run_name','')} | {r.get('job_id','')} | {r.get('state','')} | "
                f"{r.get('status_file','')} | {r.get('main_log','')} | {r.get('output_dir','')} |"
            )
        lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def build_parser() -> ArgumentParser:
    p = ArgumentParser(description="Finalize seed42 replacement matrix and gate seed123 replication")
    p.add_argument(
        "--repo-root",
        default="/home/chen034/workspace/stwm",
    )
    p.add_argument(
        "--queue-status-dir",
        default="/home/chen034/workspace/stwm/outputs/queue/stwm_protocol_v2_frontend_default_v1/d1_train/status",
    )
    p.add_argument(
        "--seed42-run-root",
        default="/home/chen034/workspace/stwm/outputs/training/stwm_v4_2_220m_protocol_frozen_frontend_replacement_seed42_v1/seed_42",
    )
    p.add_argument(
        "--report-json",
        default="/home/chen034/workspace/stwm/reports/stwm_replacement_clean_matrix_seed42_final_decision_v1.json",
    )
    p.add_argument(
        "--report-doc",
        default="/home/chen034/workspace/stwm/docs/STWM_REPLACEMENT_CLEAN_MATRIX_SEED42_FINAL_DECISION_V1.md",
    )
    p.add_argument(
        "--seed123-launch-doc",
        default="/home/chen034/workspace/stwm/docs/STWM_SEED123_REPLICATION_LAUNCH_V1.md",
    )
    p.add_argument(
        "--seed123-preferred-gpu",
        default="3",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()

    repo_root = Path(args.repo_root)
    status_dir = Path(args.queue_status_dir)
    seed42_run_root = Path(args.seed42_run_root)
    report_root = repo_root / "reports"

    resolved_runs, by_name_row, resolution_source = resolve_seed42_runs(status_dir, report_root)

    run_infos: list[dict[str, Any]] = []
    by_role: dict[str, dict[str, Any]] = {}
    completion_failures: list[str] = []

    for role in ROLE_ORDER:
        run_name = resolved_runs[role]
        info = collect_run(
            role=role,
            run_name=run_name,
            status_dir=status_dir,
            default_run_root=seed42_run_root,
            by_name_row=by_name_row,
        )
        run_infos.append(info)
        by_role[role] = info
        if info["anomalies"]:
            completion_failures.append(f"{run_name}: {','.join(info['anomalies'])}")

    all_runs_complete = len(completion_failures) == 0

    replacement = by_role["replacement_alpha050"]
    baseline = by_role["baseline_full"]
    wo_sem = by_role["wo_semantics_control"]
    wo_obj = by_role["wo_object_bias_control"]

    cmp_baseline = compare_pair(replacement, baseline)
    cmp_wo_sem = compare_pair(replacement, wo_sem)
    cmp_wo_obj = compare_pair(replacement, wo_obj)

    qloc_r = safe_float(replacement["metrics"].get("query_localization_error"))
    qloc_b = safe_float(baseline["metrics"].get("query_localization_error"))
    qtop_r = safe_float(replacement["metrics"].get("query_top1_acc"))
    qtop_b = safe_float(baseline["metrics"].get("query_top1_acc"))
    l1_r = safe_float(replacement["metrics"].get("future_trajectory_l1"))
    l1_b = safe_float(baseline["metrics"].get("future_trajectory_l1"))

    qloc_improve = None if qloc_r is None or qloc_b is None else float(qloc_b - qloc_r)
    qloc_improve_pct = None if qloc_improve is None or qloc_b in (None, 0.0) else float(qloc_improve / qloc_b)
    qtop_delta = None if qtop_r is None or qtop_b is None else float(qtop_r - qtop_b)
    l1_delta = None if l1_r is None or l1_b is None else float(l1_r - l1_b)

    qloc_substantial = bool(
        qloc_improve is not None and qloc_b is not None and qloc_improve >= max(0.001, 0.15 * qloc_b)
    )
    qtop_non_degrade = bool(qtop_delta is not None and qtop_delta >= -0.001)
    l1_acceptable = bool(l1_delta is not None and l1_delta <= 0.0005)

    replacement_noise_check = bool(
        qloc_substantial
        and qtop_delta is not None
        and qtop_delta >= 0.005
        and l1_delta is not None
        and l1_delta <= -0.0005
    )

    full_candidates = [baseline, replacement]
    full_candidates_rankable = [x for x in full_candidates if official_key(x.get("metrics", {})) is not None]
    full_best = min(full_candidates_rankable, key=lambda x: official_key(x.get("metrics", {}))) if full_candidates_rankable else baseline

    best_full_loses_to_wo_obj = False
    k_best = official_key(full_best.get("metrics", {}))
    k_wo = official_key(wo_obj.get("metrics", {}))
    if k_best is not None and k_wo is not None:
        best_full_loses_to_wo_obj = not (k_best < k_wo)

    all_cache_issues = sum(int(x.get("log_signals", {}).get("cache_issue_count", 0)) for x in run_infos)
    all_nan_issues = sum(int(x.get("log_signals", {}).get("nan_issue_count", 0)) for x in run_infos)
    all_instability = sum(int(x.get("log_signals", {}).get("instability_issue_count", 0)) for x in run_infos)
    all_nan_rows = sum(int(x.get("nan_rows", 0)) for x in run_infos)

    avg_recent_wait = statistics.fmean(
        [
            float(x.get("timing", {}).get("recent500_mean_data_wait_ratio"))
            for x in run_infos
            if x.get("timing", {}).get("recent500_mean_data_wait_ratio") is not None
        ]
    ) if any(x.get("timing", {}).get("recent500_mean_data_wait_ratio") is not None for x in run_infos) else None

    frontend_cache_stable_and_fast = bool(
        all_cache_issues == 0
        and all_nan_issues == 0
        and all_instability == 0
        and all_nan_rows == 0
        and avg_recent_wait is not None
        and avg_recent_wait < 0.30
    )

    rep_step_mean = replacement.get("timing", {}).get("full_mean_step_time_s")
    base_step_mean = baseline.get("timing", {}).get("full_mean_step_time_s")
    replacement_slower_than_baseline = bool(
        rep_step_mean is not None and base_step_mean is not None and float(rep_step_mean) > 1.05 * float(base_step_mean)
    )

    rep_issues = (
        int(replacement.get("log_signals", {}).get("cache_issue_count", 0))
        + int(replacement.get("log_signals", {}).get("nan_issue_count", 0))
        + int(replacement.get("log_signals", {}).get("instability_issue_count", 0))
        + int(replacement.get("nan_rows", 0))
    )
    base_issues = (
        int(baseline.get("log_signals", {}).get("cache_issue_count", 0))
        + int(baseline.get("log_signals", {}).get("nan_issue_count", 0))
        + int(baseline.get("log_signals", {}).get("instability_issue_count", 0))
        + int(baseline.get("nan_rows", 0))
    )
    replacement_more_unstable_than_baseline = rep_issues > base_issues

    replacement_beats_baseline_official = bool(cmp_baseline.get("replacement_better_official", False))

    gate_conditions = {
        "official_rule_better_than_baseline": replacement_beats_baseline_official,
        "query_localization_error_substantial_improvement": qloc_substantial,
        "query_top1_acc_non_degrade": qtop_non_degrade,
        "future_trajectory_l1_acceptable": l1_acceptable,
        "no_new_stability_issues": bool(
            all_cache_issues == 0 and all_nan_issues == 0 and all_instability == 0 and all_nan_rows == 0 and not replacement_more_unstable_than_baseline
        ),
    }

    gate_pass = all_runs_complete and all(gate_conditions.values())

    seed123_jobs: list[dict[str, Any]] = []
    seed123_launched = False
    seed123_reused_existing = False
    seed123_submit_tsv = ""
    seed123_submit_stdout = ""
    next_action = ""

    if gate_pass:
        existing = collect_seed123_existing(status_dir)
        if existing:
            seed123_jobs = existing
            seed123_launched = True
            seed123_reused_existing = True
        else:
            rows, submit_tsv, submit_out = run_seed123_enqueue(repo_root, str(args.seed123_preferred_gpu))
            seed123_submit_tsv = submit_tsv
            seed123_submit_stdout = submit_out
            seed123_launched = len(rows) > 0

            for row in rows:
                run_name = row.get("run_name", "")
                sp = row.get("status_file", "")
                state = "queued"
                if sp and Path(sp).exists():
                    st = load_json(Path(sp)) or {}
                    state = str(st.get("state", "queued"))
                seed123_jobs.append(
                    {
                        "run_name": run_name,
                        "job_id": row.get("job_id", ""),
                        "state": state,
                        "status_file": row.get("status_file", ""),
                        "main_log": row.get("main_log", ""),
                        "output_dir": row.get("output_dir", ""),
                    }
                )

        if seed123_launched:
            next_action = "keep detached watcher on seed123 matrix and compare seed123 official ranking after all four terminal states"
        else:
            next_action = "seed123 gate passed but launch inventory empty; check enqueue log and queue worker health before retry"
    else:
        next_action = "do not launch seed123; return to object-bias diagnostics with targeted alpha/timing/gating sweep and re-run seed42 clean matrix confirmation"

    official_answers = {
        "replacement_beats_baseline_official": replacement_beats_baseline_official,
        "replacement_improvement_not_noise": replacement_noise_check,
        "best_full_still_loses_to_wo_object_bias": best_full_loses_to_wo_obj,
        "supports_alpha050_as_new_default_full": bool(gate_pass),
    }

    stability_answers = {
        "frontend_cache_stable_and_fast": frontend_cache_stable_and_fast,
        "cache_miss_rebuild_corruption_signal": bool(all_cache_issues > 0),
        "nan_or_instability_signal": bool(all_nan_issues > 0 or all_instability > 0 or all_nan_rows > 0),
        "replacement_slower_than_baseline": replacement_slower_than_baseline,
        "replacement_more_unstable_than_baseline": replacement_more_unstable_than_baseline,
    }

    report: dict[str, Any] = {
        "generated_at": now_ts(),
        "run_resolution_source": resolution_source,
        "resolved_runs": resolved_runs,
        "completion": {
            "all_runs_complete": all_runs_complete,
            "failures": completion_failures,
        },
        "official_comparison": {
            "replacement_vs_baseline": cmp_baseline,
            "replacement_vs_wo_semantics": cmp_wo_sem,
            "replacement_vs_wo_object_bias": cmp_wo_obj,
            "qloc_improvement_abs": qloc_improve,
            "qloc_improvement_pct": qloc_improve_pct,
            "qtop1_delta": qtop_delta,
            "future_l1_delta": l1_delta,
        },
        "official_answers": official_answers,
        "stability_answers": stability_answers,
        "promotion_gate": {
            "gate_pass": gate_pass,
            "conditions": gate_conditions,
            "seed123_launched": seed123_launched,
            "seed123_reused_existing": seed123_reused_existing,
            "seed123_submit_tsv": seed123_submit_tsv,
            "seed123_submit_stdout": seed123_submit_stdout,
            "seed123_jobs": seed123_jobs,
        },
        "runs": run_infos,
        "final_conclusions": {
            "q1_all_four_success": all_runs_complete,
            "q2_replacement_beats_baseline": replacement_beats_baseline_official,
            "q3_replacement_new_default": bool(gate_pass),
            "q4_frontend_cache_stable_fast": frontend_cache_stable_and_fast,
            "q5_seed123_launched": seed123_launched,
            "q6_next_action": next_action,
        },
    }

    report_json = Path(args.report_json)
    report_doc = Path(args.report_doc)
    seed123_doc = Path(args.seed123_launch_doc)

    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    write_decision_doc(report_doc, report)

    if gate_pass and seed123_launched:
        write_seed123_launch_doc(
            seed123_doc,
            report["promotion_gate"],
            seed123_jobs,
            seed123_reused_existing,
            seed123_submit_tsv,
        )

    print(str(report_json))
    print(str(report_doc))
    if gate_pass and seed123_launched:
        print(str(seed123_doc))


if __name__ == "__main__":
    main()
