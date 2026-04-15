#!/usr/bin/env python3
from __future__ import annotations

from argparse import ArgumentParser
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List
import json
import os
import shlex
import subprocess
import time

from stwm.tools import run_tracewm_stage2_calibration_only_wave2_20260414 as prev
from stwm.tools import run_tracewm_stage2_calibration_only_fullscale_wave1_20260413 as base

ROOT = Path("/home/chen034/workspace/stwm")
SESSION = "tracewm_stage2_final_evidence_closure_20260414"
LOG_PATH = ROOT / "logs/stage2_final_evidence_closure_20260414.log"
BOOTSTRAP_BACKEND = "local_clip_vit_b32_mask_crop_visual_teacher"
EXTRA_STEPS = 4000
BATCH_SIZE = 8
EVAL_INTERVAL = 500
SAVE_EVERY = 500
MAX_TRAIN = -1
MAX_VAL = -1


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def write_json(path: str | Path, payload: Dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def write_md(path: str | Path, lines: List[str]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _apply_process_title() -> None:
    base._apply_process_title_normalization(default_title="python")


def _spec_base(run_name: str, seed: int, track: str, ablation_name: str, window_name: str) -> Dict[str, Any]:
    return {
        "run_name": run_name,
        "track": track,
        "ablation_name": ablation_name,
        "family": "topk1" if track == "longconfirm" else "ablation",
        "seed": seed,
        "objective_family": "stage2_final_evidence_closure",
        "persistence_objective_declared": False,
        "semantic_rescue_mode": "v7alignonly",
        "semantic_rescue_weight": 0.00015,
        "confidence_gated_alignment_loss_weight": 1.0,
        "sparse_persistence_contrastive_loss_weight": 0.0,
        "confidence_gating_margin_threshold": 0.10,
        "confidence_gating_temperature": 0.05,
        "semantic_hard_score_threshold": 0.25,
        "aux_loss_delay_steps": 180,
        "aux_loss_ramp_steps": 360,
        "v6_gating_family": "hard_topk_query_gating_v2",
        "v6_topk_query_k": 1,
        "v6_capped_quantile": 0.85,
        "v6_max_affected_ratio": 0.15,
        "v6_gate_min_strength": 0.05,
        "v6_strict_max_pairs_per_sample": 0,
        "v6_hard_negative_cap": 0,
        "v6_pair_sampling_temperature": 0.35,
        "v6_guaranteed_min_pairs_per_sample": 0,
        "v6_two_level_pair_mining_enabled": False,
        "v6_relaxed_motion_threshold": 0.08,
        "v6_relaxed_area_jump_threshold": 0.06,
        "v6_relaxed_small_query_threshold": 0.20,
        "v6_relaxed_appearance_shift_threshold": 0.25,
        "v6_relaxed_center_interaction_threshold": 0.10,
        "window_name": window_name,
    }


def _run_specs() -> List[Dict[str, Any]]:
    specs = []
    specs.append({**_spec_base("stage2_calonly_noalign_seed42_ablate_20260414", 42, "ablation", "noalign", "fc_noalign42"), "objective_combo": "closure_noalign_seed42", "semantic_rescue_weight": 0.0, "confidence_gated_alignment_loss_weight": 0.0})
    specs.append({**_spec_base("stage2_calonly_noalign_seed456_ablate_20260414", 456, "ablation", "noalign", "fc_noalign456"), "objective_combo": "closure_noalign_seed456", "semantic_rescue_weight": 0.0, "confidence_gated_alignment_loss_weight": 0.0})
    dense = _spec_base("stage2_calonly_densegate_seed42_ablate_20260414", 42, "ablation", "densegate", "fc_dense42")
    dense.update({"objective_combo": "closure_densegate_seed42", "v6_gating_family": "capped_quantile_sparse_gating_v2", "v6_capped_quantile": 0.0, "v6_max_affected_ratio": 1.0})
    specs.append(dense)
    nodelay = _spec_base("stage2_calonly_nodelay_seed42_ablate_20260414", 42, "ablation", "nodelay", "fc_nodelay42")
    nodelay.update({"objective_combo": "closure_nodelay_seed42", "aux_loss_delay_steps": 0, "aux_loss_ramp_steps": 0})
    specs.append(nodelay)
    specs.append({**_spec_base("stage2_calonly_topk1_seed123_longconfirm_20260414", 123, "longconfirm", "none", "fc_long123"), "objective_combo": "closure_topk1_seed123_longconfirm", "family": "topk1"})
    specs.append({**_spec_base("stage2_calonly_topk1_seed654_longconfirm_20260414", 654, "longconfirm", "none", "fc_long654"), "objective_combo": "closure_topk1_seed654_longconfirm", "family": "topk1"})
    return specs


def _launch_meta_dir(args: Any) -> Path:
    return Path(args.work_root) / "reports/stage2_final_evidence_closure_runs_20260414"


def _resume_ckpt_for_spec(spec: Dict[str, Any]) -> Path:
    name = str(spec.get("run_name", ""))
    if "seed123_longconfirm" in name:
        return ROOT / "outputs/checkpoints/stage2_calonly_topk1_seed123_wave1_20260413/best.pt"
    if "seed654_longconfirm" in name:
        best = ROOT / "outputs/checkpoints/stage2_calonly_topk1_seed654_wave2_20260414/best.pt"
        return best if best.exists() else ROOT / "outputs/checkpoints/stage2_calonly_topk1_seed654_wave2_20260414/latest.pt"
    return base._resume_ckpt_for_seed(int(spec.get("seed", 42)))


def _tmux_window_command(args: Any, meta_json: Path, meta: Dict[str, Any]) -> str:
    run_name = str(meta.get("run_name", ""))
    log_path = str(meta.get("log_path", ""))
    pid_path = str(meta.get("worker_pid_file", ""))
    script_path = Path(args.work_root) / "code/stwm/tools/run_tracewm_stage2_final_evidence_closure_20260414.py"
    pythonpath_value = f"{args.work_root}/code:{os.environ.get('PYTHONPATH', '')}"
    proc_title = str(os.environ.get("STWM_PROC_TITLE", "python"))
    proc_title_mode = str(os.environ.get("STWM_PROC_TITLE_MODE", "generic"))
    cmd = (
        f"PYTHONPATH={shlex.quote(pythonpath_value)} "
        f"STWM_PROC_TITLE={shlex.quote(proc_title)} "
        f"STWM_PROC_TITLE_MODE={shlex.quote(proc_title_mode)} "
        f"nohup {shlex.quote(str(args.python_bin))} {shlex.quote(str(script_path))} --mode run-one --meta-json {shlex.quote(str(meta_json))} "
        f">> {shlex.quote(log_path)} 2>&1 < /dev/null & echo $! > {shlex.quote(pid_path)}; "
        f"while kill -0 \"$(cat {shlex.quote(pid_path)})\" 2>/dev/null; do sleep 30; done"
    )
    return "bash -lc " + shlex.quote(f"cd {shlex.quote(str(args.work_root))}; rm -f {shlex.quote(pid_path)}; {cmd}; printf '[%s] tmux_window_exit run_name={run_name} observed_child_exit\\n' \"$(date -Iseconds)\" >> {shlex.quote(log_path)}")


def _install_prev_hooks() -> None:
    prev._run_specs = _run_specs
    prev._launch_meta_dir = _launch_meta_dir
    prev._resume_ckpt_for_spec = _resume_ckpt_for_spec
    prev._tmux_window_command = _tmux_window_command


def _rows(args: Any) -> List[Dict[str, Any]]:
    _install_prev_hooks()
    return prev._collect_all_rows(args)


def _completed(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [r for r in rows if str(r.get("status", "")).lower() == "completed" and bool(r.get("scientific_result_valid", False))]


def _rank(row: Dict[str, Any]) -> tuple[float, float, float]:
    return base._metric_rank_tuple(row.get("best_checkpoint_metric", {}))


def _hard(row: Dict[str, Any]) -> float:
    block = row.get("semantic_hard_sidecar_metric", {}) if isinstance(row.get("semantic_hard_sidecar_metric", {}), dict) else {}
    return base._f(block.get("semantic_hard_sidecar_score"), _rank(row)[0])


def summarize(args: Any) -> Dict[str, Any]:
    rows = _rows(args)
    running = sum(str(r.get("status", "")).lower() == "running" for r in rows)
    completed = sum(str(r.get("status", "")).lower() == "completed" for r in rows)
    failed = sum(str(r.get("status", "")).lower() == "failed" for r in rows)
    completed_rows = _completed(rows)
    best_overall = min(completed_rows, key=_rank) if completed_rows else {}
    best_hard = min(completed_rows, key=_hard) if completed_rows else {}
    payload = {
        "generated_at_utc": now_iso(),
        "stage2_final_evidence_closure_status": f"{running}_running_{completed}_completed_{failed}_failed",
        "running_count": int(running),
        "completed_count": int(completed),
        "failed_count": int(failed),
        "all_runs_terminal": bool(rows and running == 0 and completed + failed == len(rows)),
        "run_rows": rows,
        "overall_best_run_name": str(best_overall.get("run_name", "none")) if best_overall else "none",
        "semantic_hard_best_run_name": str(best_hard.get("run_name", "none")) if best_hard else "none",
        "previous_final_pack_summary": str(args.final_pack_summary_report),
        "previous_final_pack_diagnosis": str(args.final_pack_diagnosis_report),
    }
    write_json(args.summary_report, payload)
    return payload


def _final_pack_row_map(args: Any) -> Dict[str, Dict[str, Any]]:
    final = read_json(args.final_pack_summary_report)
    rows = final.get("run_rows", []) if isinstance(final.get("run_rows", []), list) else []
    return {str(r.get("run_name", "")): r for r in rows if isinstance(r, dict)}


def _ref_for_seed(seed: int, final_rows: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    for row in final_rows.values():
        if int(row.get("seed", -1)) == int(seed) and "topk1" in str(row.get("run_name", "")):
            return row
    return {}


def _worse_than(row: Dict[str, Any], ref: Dict[str, Any]) -> bool:
    if not row or not ref:
        return False
    return bool(_rank(row)[0] > _rank(ref)[0] or _hard(row) > _hard(ref))


def _longrun_new_best(row: Dict[str, Any], ref: Dict[str, Any]) -> bool:
    if not row or not ref:
        return False
    row_step = int((row.get("best_checkpoint_metric", {}) if isinstance(row.get("best_checkpoint_metric", {}), dict) else {}).get("global_step", -1))
    ref_step = int((ref.get("best_checkpoint_metric", {}) if isinstance(ref.get("best_checkpoint_metric", {}), dict) else {}).get("global_step", -1))
    return bool(row_step > ref_step and _rank(row) < _rank(ref))


def cleanup_aux_probe(args: Any) -> Dict[str, Any]:
    payload = read_json(args.aux_probe_report)
    rows = payload.get("rows", []) if isinstance(payload.get("rows", []), list) else []
    vals = [float(r.get("average_jaccard", 0.0)) for r in rows if isinstance(r, dict) and str(r.get("probe_status", "")) == "completed"]
    saturation = bool(vals and sum(v >= 0.999 for v in vals) >= max(2, len(vals) // 2))
    payload.update({"generated_at_utc_cleanup": now_iso(), "adapter_probe_only": True, "paper_official_benchmark": False, "probe_saturation_detected": saturation, "not_suitable_as_main_ranking": saturation, "auxiliary_only_reason": "adapter/proxy probe; official task faithfully instantiated remains false or not used as main ranking"})
    write_json(args.aux_probe_report, payload)
    write_md(args.aux_probe_md, ["# Stage2 Aux External Probe Batch", "", "- adapter_probe_only: True", "- paper_official_benchmark: False", f"- probe_saturation_detected: {saturation}", f"- not_suitable_as_main_ranking: {saturation}", "- usage: auxiliary non-regression check only"])
    return payload


def run_utility_and_qual(args: Any) -> Dict[str, Any]:
    qcmd = [str(args.python_bin), str(ROOT / "code/stwm/tools/run_tracewm_stage2_future_query_utility_eval_20260414.py"), "--closure-summary", str(args.summary_report), "--closure-diagnosis", str(args.diagnosis_report)]
    subprocess.run(qcmd, cwd=str(args.work_root), check=False)
    aux = cleanup_aux_probe(args)
    vcmd = [str(args.python_bin), str(ROOT / "code/stwm/tools/run_tracewm_stage1_stage2_qualitative_pack_v6_20260414.py"), "--closure-summary", str(args.summary_report), "--closure-diagnosis", str(args.diagnosis_report)]
    subprocess.run(vcmd, cwd=str(args.work_root), check=False)
    return {"utility": read_json(args.utility_report), "aux": aux, "stage2_qual": read_json(args.stage2_qual_report)}


def diagnose(args: Any) -> Dict[str, Any]:
    summary = summarize(args)
    rows = summary.get("run_rows", []) if isinstance(summary.get("run_rows", []), list) else []
    final_diag = read_json(args.final_pack_diagnosis_report)
    final_rows = _final_pack_row_map(args)
    pending = not bool(summary.get("all_runs_terminal", False)) or int(summary.get("failed_count", 0)) > 0
    completed = _completed(rows)
    row_by_name = {str(r.get("run_name", "")): r for r in completed}
    if pending:
        payload = {"generated_at_utc": now_iso(), "status": "pending_training_completion", "mainline_still_calibration_only": True, "6_seed_support_still_valid": bool(final_diag.get("6_seed_support_present", False)), "current_stage2_ready_to_freeze": False, "next_step_choice": "run_one_more_targeted_ablation_fix", "run_rows_completed": len(completed), "run_rows_total": len(rows)}
        write_json(args.diagnosis_report, payload)
        write_md(args.results_md, ["# Stage2 Final Evidence Closure Results", "", f"- status: {payload['status']}", f"- completed: {len(completed)} / {len(rows)}", "- final diagnosis will update after all six closure runs finish."])
        return payload

    utility = read_json(args.utility_report)
    if not utility:
        run_utility_and_qual(args)
        utility = read_json(args.utility_report)
    qual = read_json(args.stage2_qual_report)
    aux = cleanup_aux_probe(args)

    noalign42 = row_by_name.get("stage2_calonly_noalign_seed42_ablate_20260414", {})
    noalign456 = row_by_name.get("stage2_calonly_noalign_seed456_ablate_20260414", {})
    dense42 = row_by_name.get("stage2_calonly_densegate_seed42_ablate_20260414", {})
    nodelay42 = row_by_name.get("stage2_calonly_nodelay_seed42_ablate_20260414", {})
    ref42 = _ref_for_seed(42, final_rows)
    ref456 = _ref_for_seed(456, final_rows)
    alignment_load = bool(_worse_than(noalign42, ref42) and _worse_than(noalign456, ref456))
    sparse_load = bool(_worse_than(dense42, ref42))
    delay_load = bool(_worse_than(nodelay42, ref42))
    mechanism_cross = bool(alignment_load and sparse_load and delay_load)

    long123 = row_by_name.get("stage2_calonly_topk1_seed123_longconfirm_20260414", {})
    long654 = row_by_name.get("stage2_calonly_topk1_seed654_longconfirm_20260414", {})
    ref123 = final_rows.get("stage2_calonly_topk1_seed123_wave1_20260413", {})
    ref654 = final_rows.get("stage2_calonly_topk1_seed654_wave2_20260414", {})
    long_new = bool(_longrun_new_best(long123, ref123) or _longrun_new_best(long654, ref654))
    utility_ok = bool(utility.get("future_query_utility_improved_vs_baselines", False))
    qual_ready = bool(qual.get("ready_for_human_figure_selection", False))
    aux_only = bool(aux.get("adapter_probe_only", True) and not aux.get("paper_official_benchmark", False))
    ready = bool(final_diag.get("calibration_only_is_final_stage2_mainline", False) and final_diag.get("6_seed_support_present", False) and mechanism_cross and utility_ok and qual_ready and aux_only)
    if ready:
        next_step = "freeze_stage2_calibration_only_mainline"
    elif not utility_ok:
        next_step = "run_one_more_targeted_query_utility_fix"
    elif not mechanism_cross:
        next_step = "run_one_more_targeted_ablation_fix"
    elif long_new:
        next_step = "calibration_only_longrun_wave2"
    else:
        next_step = "reconsider_stage2_design_only_if_utility_fails"
    payload = {"generated_at_utc": now_iso(), "status": "completed", "mainline_still_calibration_only": True, "6_seed_support_still_valid": bool(final_diag.get("6_seed_support_present", False)), "mechanism_ablation_cross_seed_support": mechanism_cross, "alignment_load_bearing": alignment_load, "sparse_gating_load_bearing": sparse_load, "delayed_schedule_load_bearing": delay_load, "longrun_produces_new_best": long_new, "future_query_utility_improved_vs_baselines": utility_ok, "qualitative_pack_ready_for_human_figure_selection": qual_ready, "aux_probe_is_only_auxiliary": aux_only, "current_stage2_ready_to_freeze": ready, "overall_best_run_name": summary.get("overall_best_run_name", "none"), "semantic_hard_best_run_name": summary.get("semantic_hard_best_run_name", "none"), "next_step_choice": next_step}
    write_json(args.diagnosis_report, payload)
    lines = ["# Stage2 Final Evidence Closure Results", "", *[f"- {k}: {payload[k]}" for k in ["status", "mainline_still_calibration_only", "6_seed_support_still_valid", "mechanism_ablation_cross_seed_support", "alignment_load_bearing", "sparse_gating_load_bearing", "delayed_schedule_load_bearing", "longrun_produces_new_best", "future_query_utility_improved_vs_baselines", "qualitative_pack_ready_for_human_figure_selection", "aux_probe_is_only_auxiliary", "current_stage2_ready_to_freeze", "next_step_choice"]], "", "| run_name | track | ablation | status | endpoint | hard_score |", "|---|---|---|---|---:|---:|", *[f"| {r.get('run_name','')} | {r.get('track','')} | {r.get('ablation_name','')} | {r.get('status','')} | {_rank(r)[0]:.6f} | {_hard(r):.6f} |" for r in rows]]
    write_md(args.results_md, lines)
    return payload


def launch(args: Any) -> Dict[str, Any]:
    _install_prev_hooks()
    prev.FULL_EXTRA_STEPS = EXTRA_STEPS
    prev.FULL_BATCH_SIZE = BATCH_SIZE
    prev.FULL_EVAL_INTERVAL = EVAL_INTERVAL
    prev.FULL_SAVE_EVERY = SAVE_EVERY
    prev.FULL_MAX_TRAIN_PER_DATASET = MAX_TRAIN
    prev.FULL_MAX_VAL_PER_DATASET = MAX_VAL
    result = prev.launch(args)
    return summarize(args)


def wait_for_completion(args: Any) -> Dict[str, Any]:
    deadline = time.time() + float(args.wait_timeout_seconds)
    last = summarize(args)
    while time.time() < deadline:
        if bool(last.get("all_runs_terminal", False)):
            return last
        time.sleep(float(args.poll_seconds))
        last = summarize(args)
    last["timed_out_waiting_for_completion"] = True
    write_json(args.summary_report, last)
    return last


def run_all(args: Any) -> Dict[str, Any]:
    launch(args)
    summary = wait_for_completion(args)
    if bool(summary.get("all_runs_terminal", False)) and int(summary.get("failed_count", 0)) == 0:
        diagnose(args)
    return {"summary": read_json(args.summary_report), "diagnosis": read_json(args.diagnosis_report), "utility": read_json(args.utility_report)}


def parse_args() -> Any:
    parser = ArgumentParser()
    parser.add_argument("--mode", default="all", choices=["all", "launch", "run-one", "summarize", "diagnose"])
    parser.add_argument("--meta-json", default="")
    parser.add_argument("--work-root", default=str(ROOT))
    parser.add_argument("--python-bin", default=base._python_bin_default())
    parser.add_argument("--tmux-session", default=SESSION)
    parser.add_argument("--stage2-contract-json", default=str(ROOT / "reports/stage2_bootstrap_data_contract_20260408.json"))
    parser.add_argument("--stage1-best-ckpt", default=str(ROOT / "outputs/checkpoints/stage1_v2_longtrain_220m_mainline_20260408/best.pt"))
    parser.add_argument("--shared-lease-path", default=str(ROOT / "reports/stage1_v2_gpu_lease_20260408.json"))
    parser.add_argument("--bootstrap-cache-jsonl", default=str(ROOT / "data/processed/stage2_real_bootstrap_cache_20260410/clip_vit_b32_core_trainval_required_subset.jsonl"))
    parser.add_argument("--semantic-hard-manifest-path", default=str(ROOT / "manifests/protocol_v2/stage2_semantic_hard_subsets_20260410.json"))
    parser.add_argument("--stage2-semantic-value-diagnosis-report", default=str(ROOT / "reports/stage2_semantic_value_diagnosis_20260410.json"))
    parser.add_argument("--v7-repaired-summary-report", default=str(ROOT / "reports/stage2_semantic_objective_redesign_v7_summary_20260413.json"))
    parser.add_argument("--v7-repaired-diagnosis-report", default=str(ROOT / "reports/stage2_semantic_objective_redesign_v7_diagnosis_20260413.json"))
    parser.add_argument("--wave1-summary-report", default=str(ROOT / "reports/stage2_calibration_only_fullscale_wave1_summary_20260413.json"))
    parser.add_argument("--wave1-diagnosis-report", default=str(ROOT / "reports/stage2_calibration_only_fullscale_wave1_diagnosis_20260413.json"))
    parser.add_argument("--process-title-report", default=str(ROOT / "reports/stage2_process_title_normalization_20260414.json"))
    parser.add_argument("--concurrent-runtime-report", default=str(ROOT / "reports/stage2_final_evidence_closure_runtime_20260414.json"))
    parser.add_argument("--launch-report", default=str(ROOT / "reports/stage2_final_evidence_closure_launch_20260414.json"))
    parser.add_argument("--summary-report", default=str(ROOT / "reports/stage2_final_evidence_closure_summary_20260414.json"))
    parser.add_argument("--diagnosis-report", default=str(ROOT / "reports/stage2_final_evidence_closure_diagnosis_20260414.json"))
    parser.add_argument("--results-md", default=str(ROOT / "docs/STAGE2_FINAL_EVIDENCE_CLOSURE_RESULTS_20260414.md"))
    parser.add_argument("--ablation-pack-report", default=str(ROOT / "reports/stage2_final_evidence_closure_ablation_pack_20260414.json"))
    parser.add_argument("--ablation-pack-md", default=str(ROOT / "docs/STAGE2_FINAL_EVIDENCE_CLOSURE_ABLATION_PACK_20260414.md"))
    parser.add_argument("--final-pack-summary-report", default=str(ROOT / "reports/stage2_calibration_only_final_pack_summary_20260414.json"))
    parser.add_argument("--final-pack-diagnosis-report", default=str(ROOT / "reports/stage2_calibration_only_final_pack_diagnosis_20260414.json"))
    parser.add_argument("--final-pack-md", default=str(ROOT / "docs/STAGE2_CALIBRATION_ONLY_FINAL_PACK_RESULTS_20260414.md"))
    parser.add_argument("--aux-probe-report", default=str(ROOT / "reports/stage2_aux_external_probe_batch_20260414.json"))
    parser.add_argument("--aux-probe-md", default=str(ROOT / "docs/STAGE2_AUX_EXTERNAL_PROBE_BATCH_20260414.md"))
    parser.add_argument("--aux-external-probe-report", default=str(ROOT / "reports/stage2_aux_external_probe_batch_20260414.json"))
    parser.add_argument("--aux-external-probe-md", default=str(ROOT / "docs/STAGE2_AUX_EXTERNAL_PROBE_BATCH_20260414.md"))
    parser.add_argument("--utility-report", default=str(ROOT / "reports/stage2_future_query_utility_eval_20260414.json"))
    parser.add_argument("--stage2-qual-report", default=str(ROOT / "reports/stage2_qualitative_pack_v6_20260414.json"))
    parser.add_argument("--reserve-idle-gpu-count", type=int, default=2)
    parser.add_argument("--gpu-acquire-timeout-seconds", type=int, default=28800)
    parser.add_argument("--gpu-acquire-retry-seconds", type=int, default=20)
    parser.add_argument("--wait-timeout-seconds", type=int, default=172800)
    parser.add_argument("--poll-seconds", type=int, default=120)
    return parser.parse_args()


def run_one(args: Any) -> None:
    _install_prev_hooks()
    prev.run_one(args)


def main() -> None:
    _apply_process_title()
    args = parse_args()
    if args.mode == "all":
        print(json.dumps(run_all(args), ensure_ascii=True, indent=2))
    elif args.mode == "launch":
        print(json.dumps(launch(args), ensure_ascii=True, indent=2))
    elif args.mode == "run-one":
        run_one(args)
    elif args.mode == "summarize":
        print(json.dumps(summarize(args), ensure_ascii=True, indent=2))
    elif args.mode == "diagnose":
        print(json.dumps(diagnose(args), ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
