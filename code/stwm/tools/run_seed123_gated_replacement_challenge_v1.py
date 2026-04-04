from __future__ import annotations

from pathlib import Path
import json
import math
import re
import subprocess
import time
import traceback
from typing import Any


REPO_ROOT = Path("/home/chen034/workspace/stwm")
QUEUE_DIR = REPO_ROOT / "outputs/queue/stwm_protocol_v2_frontend_default_v1/d1_train"
STATUS_DIR = QUEUE_DIR / "status"

RUN_BASELINE = "full_v4_2_seed123_fixed_nowarm_lambda1_rerun_v2"
RUN_CHALLENGE = "full_v4_2_seed123_objbias_gated_replacement_challenge_v1"

OUT_ROOT = REPO_ROOT / "outputs/training/stwm_v4_2_220m_protocol_frozen_frontend_replication_seed123_v1/seed_123"

REPORT_JSON = REPO_ROOT / "reports/stwm_seed123_gated_replacement_challenge_final_decision_v1.json"
REPORT_DOC = REPO_ROOT / "docs/STWM_SEED123_GATED_REPLACEMENT_CHALLENGE_FINAL_DECISION_V1.md"

BG_STATUS = REPO_ROOT / "outputs/background_jobs/stwm_seed123_gated_replacement_challenge_v1.status.json"

STEP_RE = re.compile(r"step_(\d+)\.json$")


def now_ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def write_bg_status(stage: str, payload: dict[str, Any]) -> None:
    BG_STATUS.parent.mkdir(parents=True, exist_ok=True)
    obj = {
        "stage": stage,
        "update_ts": now_ts(),
        **payload,
    }
    BG_STATUS.write_text(json.dumps(obj, indent=2, ensure_ascii=False))


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


def latest_status_path(run_name: str) -> Path | None:
    cands = sorted(STATUS_DIR.glob(f"*_{run_name}.status.json"))
    return cands[-1] if cands else None


def parse_submit_output(text: str) -> dict[str, str]:
    out: dict[str, str] = {
        "job_id": "",
        "status_file": "",
        "main_log": "",
        "pid_file": "",
    }
    for line in text.splitlines():
        if line.startswith("  job_id:"):
            out["job_id"] = line.split(":", 1)[1].strip()
        elif line.startswith("  status_file:"):
            out["status_file"] = line.split(":", 1)[1].strip()
        elif line.startswith("  main_log:"):
            out["main_log"] = line.split(":", 1)[1].strip()
        elif line.startswith("  pid_file:"):
            out["pid_file"] = line.split(":", 1)[1].strip()
    return out


def submit_challenge() -> dict[str, str]:
    train_script = REPO_ROOT / "code/stwm/trainers/train_stwm_v4_2_real.py"
    train_manifest = REPO_ROOT / "manifests/protocol_v2/train_v2.json"
    protocol_main = REPO_ROOT / "manifests/protocol_v2/protocol_val_main_v1.json"
    protocol_eventful = REPO_ROOT / "manifests/protocol_v2/protocol_val_eventful_v1.json"
    preset_file = REPO_ROOT / "code/stwm/configs/model_presets_v4_2.json"
    data_root = REPO_ROOT / "data/external"
    cache_dir = REPO_ROOT / "data/cache/frontend_cache_protocol_v2_full_v1"
    cache_index = cache_dir / "index.json"
    out_dir = OUT_ROOT / RUN_CHALLENGE
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "env",
        f"PYTHONPATH={REPO_ROOT / 'code'}:{__import__('os').environ.get('PYTHONPATH', '')}",
        "conda",
        "run",
        "--no-capture-output",
        "-n",
        "stwm",
        "python",
        str(train_script),
        "--data-root",
        str(data_root),
        "--manifest",
        str(train_manifest),
        "--output-dir",
        str(out_dir),
        "--run-name",
        RUN_CHALLENGE,
        "--seed",
        "123",
        "--steps",
        "2000",
        "--target-epochs",
        "0",
        "--min-optimizer-steps",
        "0",
        "--max-optimizer-steps",
        "0",
        "--sample-limit",
        "0",
        "--model-preset",
        "prototype_220m_v4_2",
        "--preset-file",
        str(preset_file),
        "--use-teacher-priors",
        "--save-checkpoint",
        "--checkpoint-dir-name",
        "checkpoints",
        "--checkpoint-interval",
        "500",
        "--milestone-interval",
        "0",
        "--auto-resume",
        "--micro-batch-per-gpu",
        "2",
        "--grad-accum",
        "8",
        "--num-workers",
        "12",
        "--prefetch-factor",
        "2",
        "--persistent-workers",
        "--pin-memory",
        "--bf16",
        "--activation-checkpointing",
        "--lambda-traj",
        "1.0",
        "--lambda-vis",
        "0.25",
        "--lambda-sem",
        "0.5",
        "--lambda-reid",
        "0.25",
        "--lambda-query",
        "0.25",
        "--lambda-reconnect",
        "0.1",
        "--gradient-audit-interval",
        "0",
        "--protocol-eval-interval",
        "500",
        "--protocol-eval-manifest",
        str(protocol_main),
        "--protocol-eval-dataset",
        "all",
        "--protocol-eval-max-clips",
        "0",
        "--protocol-eval-seed",
        "123",
        "--protocol-eval-obs-steps",
        "8",
        "--protocol-eval-pred-steps",
        "8",
        "--protocol-eval-run-name",
        "protocol_val_main",
        "--protocol-diagnostics-manifest",
        str(protocol_eventful),
        "--protocol-diagnostics-dataset",
        "all",
        "--protocol-diagnostics-max-clips",
        "0",
        "--protocol-diagnostics-run-name",
        "protocol_val_eventful",
        "--protocol-version",
        "v2_4_detached_frozen",
        "--protocol-best-checkpoint-name",
        "best_protocol_main.pt",
        "--protocol-best-selection-name",
        "best_protocol_main_selection.json",
        "--data-mode",
        "frontend_cache",
        "--frontend-cache-dir",
        str(cache_dir),
        "--frontend-cache-index",
        str(cache_index),
        "--frontend-cache-max-shards-in-memory",
        "8",
        "--object-bias-gated",
        "--object-bias-gate-threshold",
        "0.5",
    ]

    submit_cmd = [
        "bash",
        str(REPO_ROOT / "scripts/protocol_v2_queue_submit.sh"),
        "--queue-dir",
        str(QUEUE_DIR),
        "--job-name",
        RUN_CHALLENGE,
        "--class-type",
        "B",
        "--workdir",
        str(REPO_ROOT),
        "--notes",
        "Seed123 gated replacement challenge v1 | data_mode=frontend_cache",
        "--resume-hint",
        "Resume with same output_dir and --auto-resume; compare with seed123 baseline by official rule",
        "--preferred-gpu",
        "3",
        "--",
        *cmd,
    ]

    output = subprocess.check_output(submit_cmd, text=True, cwd=str(REPO_ROOT))
    info = parse_submit_output(output)
    info["submit_output"] = output
    info["output_dir"] = str(out_dir)
    return info


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


def eval_summary_path(run_dir: Path, sidecar: dict[str, Any] | None) -> Path | None:
    if isinstance(sidecar, dict):
        ev = sidecar.get("eval_summary")
        if isinstance(ev, str) and ev.strip():
            p = Path(ev)
            if p.exists():
                return p
    cands = sorted((run_dir / "checkpoints/protocol_eval").glob("protocol_val_main_step_*.json"))
    return cands[-1] if cands else None


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


def official_key(metrics: dict[str, float | None]) -> tuple[float, float, float] | None:
    qloc = metrics.get("query_localization_error")
    qtop = metrics.get("query_top1_acc")
    l1 = metrics.get("future_trajectory_l1")
    if qloc is None or qtop is None or l1 is None:
        return None
    return (float(qloc), -float(qtop), float(l1))


def collect_run(run_name: str) -> dict[str, Any]:
    sp = latest_status_path(run_name)
    st = load_json(sp) if sp is not None else None
    run_dir = OUT_ROOT / run_name
    train_log = run_dir / "train_log.jsonl"
    sidecar_path = find_sidecar(run_dir)
    sidecar = load_json(sidecar_path) if sidecar_path is not None else None
    eval_path = eval_summary_path(run_dir, sidecar)
    eval_obj = load_json(eval_path) if eval_path is not None else None
    best_ckpt = find_best_ckpt(run_dir)
    metrics = extract_metrics(sidecar, eval_obj)

    info = {
        "run_name": run_name,
        "state": str(st.get("state", "missing")) if isinstance(st, dict) else "missing",
        "job_id": str(st.get("job_id", "")) if isinstance(st, dict) else "",
        "status_file": str(sp) if sp is not None else "",
        "main_log": str(st.get("main_log", "")) if isinstance(st, dict) else "",
        "output_dir": str(run_dir),
        "train_log": str(train_log),
        "train_max_step": train_max_step(train_log),
        "best_protocol_main": str(best_ckpt) if best_ckpt is not None else "",
        "selection_sidecar": str(sidecar_path) if sidecar_path is not None else "",
        "eval_summary": str(eval_path) if eval_path is not None else "",
        "selected_best_step": selected_step(sidecar, eval_path),
        "metrics": metrics,
    }
    info["checks"] = {
        "state_done": info["state"].lower() == "done",
        "step_reached": int(info["train_max_step"]) >= 2000,
        "best_ckpt_exists": bool(best_ckpt is not None and Path(info["best_protocol_main"]).exists()),
        "sidecar_exists": bool(sidecar_path is not None and Path(info["selection_sidecar"]).exists()),
        "eval_exists": bool(eval_path is not None and Path(info["eval_summary"]).exists()),
    }
    return info


def wait_until_terminal(status_file: Path) -> dict[str, Any]:
    while True:
        st = load_json(status_file) if status_file.exists() else None
        state = str(st.get("state", "missing")) if isinstance(st, dict) else "missing"
        write_bg_status(
            "waiting_challenge_terminal",
            {
                "challenge_state": state,
                "challenge_status_file": str(status_file),
                "challenge_job_id": str(st.get("job_id", "")) if isinstance(st, dict) else "",
            },
        )
        if state.lower() in {"done", "failed"}:
            return st if isinstance(st, dict) else {}
        time.sleep(120)


def write_report(report: dict[str, Any]) -> None:
    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    REPORT_DOC.parent.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(json.dumps(report, indent=2, ensure_ascii=False))

    b = report["baseline"]
    c = report["challenge"]
    lines: list[str] = []
    lines.append("# STWM Seed123 Gated Replacement Challenge Final Decision V1")
    lines.append("")
    lines.append(f"Generated: {report['generated_at']}")
    lines.append(f"Challenge run: {RUN_CHALLENGE}")
    lines.append(f"Baseline run: {RUN_BASELINE}")
    lines.append("")
    lines.append("## Verification")
    lines.append("")
    lines.append(f"- baseline_complete: {report['verification']['baseline_complete']}")
    lines.append(f"- challenge_complete: {report['verification']['challenge_complete']}")
    lines.append("")
    lines.append("## Official Rule Comparison")
    lines.append("")
    lines.append("| run | qloc | qtop1 | future_l1 | selected_step |")
    lines.append("|---|---:|---:|---:|---:|")
    lines.append(
        "| baseline | "
        f"{b['metrics'].get('query_localization_error')} | {b['metrics'].get('query_top1_acc')} | "
        f"{b['metrics'].get('future_trajectory_l1')} | {b.get('selected_best_step')} |"
    )
    lines.append(
        "| challenge | "
        f"{c['metrics'].get('query_localization_error')} | {c['metrics'].get('query_top1_acc')} | "
        f"{c['metrics'].get('future_trajectory_l1')} | {c.get('selected_best_step')} |"
    )
    lines.append("")
    lines.append(f"challenge_beats_baseline_official: {report['official_rule']['challenge_beats_baseline_official']}")
    lines.append(f"decision: {report['decision']['decision']}")
    lines.append(f"reason: {report['decision']['reason']}")
    lines.append("")
    REPORT_DOC.write_text("\n".join(lines) + "\n")


def main() -> None:
    write_bg_status(
        "started",
        {
            "baseline_run": RUN_BASELINE,
            "challenge_run": RUN_CHALLENGE,
            "expected_report_json": str(REPORT_JSON),
            "expected_report_doc": str(REPORT_DOC),
        },
    )

    existing_sp = latest_status_path(RUN_CHALLENGE)
    submit_info: dict[str, str]
    if existing_sp is not None:
        existing = load_json(existing_sp) or {}
        state = str(existing.get("state", "")).lower()
        if state in {"queued", "waiting_for_gpu", "running", "done"}:
            submit_info = {
                "job_id": str(existing.get("job_id", "")),
                "status_file": str(existing_sp),
                "main_log": str(existing.get("main_log", "")),
                "pid_file": str(existing.get("pid_file", "")),
                "submit_output": "reused_existing_status",
                "output_dir": str(OUT_ROOT / RUN_CHALLENGE),
            }
        else:
            submit_info = submit_challenge()
    else:
        submit_info = submit_challenge()

    write_bg_status("challenge_submitted", submit_info)

    status_file = Path(submit_info["status_file"]) if submit_info.get("status_file") else latest_status_path(RUN_CHALLENGE)
    if status_file is None:
        raise RuntimeError("challenge status_file missing after submission")

    wait_until_terminal(status_file)

    baseline = collect_run(RUN_BASELINE)
    challenge = collect_run(RUN_CHALLENGE)

    baseline_complete = all(bool(v) for v in baseline["checks"].values())
    challenge_complete = all(bool(v) for v in challenge["checks"].values())

    kb = official_key(baseline["metrics"])
    kc = official_key(challenge["metrics"])
    beats = bool(kc is not None and kb is not None and kc < kb)

    if not baseline_complete:
        decision = {
            "decision": "invalid_baseline_artifacts",
            "reason": "baseline artifacts are incomplete; cannot perform official rule comparison",
        }
    elif not challenge_complete:
        decision = {
            "decision": "challenge_failed_or_incomplete",
            "reason": "challenge run did not finish with complete artifacts",
        }
    elif beats:
        decision = {
            "decision": "gated_challenge_wins_vs_baseline",
            "reason": "challenge beats seed123 baseline under official selection rule",
        }
    else:
        decision = {
            "decision": "gated_challenge_not_superior",
            "reason": "challenge does not beat seed123 baseline under official selection rule",
        }

    report = {
        "generated_at": now_ts(),
        "submission": submit_info,
        "verification": {
            "baseline_complete": baseline_complete,
            "challenge_complete": challenge_complete,
        },
        "official_rule": {
            "baseline_key": kb,
            "challenge_key": kc,
            "challenge_beats_baseline_official": beats,
        },
        "baseline": baseline,
        "challenge": challenge,
        "decision": decision,
    }

    write_report(report)

    write_bg_status(
        "done",
        {
            "report_json": str(REPORT_JSON),
            "report_doc": str(REPORT_DOC),
            "decision": decision,
        },
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        write_bg_status(
            "error",
            {
                "error": str(exc),
                "traceback": traceback.format_exc(),
            },
        )
        raise
