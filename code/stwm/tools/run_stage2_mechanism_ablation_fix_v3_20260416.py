#!/usr/bin/env python3
from __future__ import annotations

from argparse import ArgumentParser
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple
import json
import subprocess
import time

from stwm.tools import run_stage2_mechanism_ablation_fix_20260415 as prevfix
from stwm.tools import run_tracewm_stage2_calibration_only_fullscale_wave1_20260413 as base


ROOT = prevfix.ROOT
SESSION = "tracewm_stage2_oral_hardening_20260416"
LOG_PATH = ROOT / "logs/stage2_oral_hardening_20260416.log"
BOOTSTRAP_BACKEND = "local_clip_vit_b32_mask_crop_visual_teacher"
DATE_TAG = "20260416"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _append_log(message: str) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as fh:
        fh.write(f"[{now_iso()}] {message}\n")


def _spec_base(run_name: str, seed: int, ablation_name: str, window_name: str) -> Dict[str, Any]:
    spec = prevfix._spec_base(run_name=run_name, seed=seed, ablation_name=ablation_name, window_name=window_name, reuse_source_run_name="")
    spec["objective_combo"] = f"oral_hardening_v3_{ablation_name}_seed{int(seed)}"
    spec["track"] = "ablation_fix_v3"
    spec["family"] = "calibration_only_ablation_fix_v3"
    spec["objective_family"] = "calibration_only_mechanism_ablation_fix_v3"
    return spec


def _run_specs() -> List[Dict[str, Any]]:
    specs: List[Dict[str, Any]] = []
    specs.append(_spec_base("stage2_calonly_noalign_seed654_ablate_fix_v3_20260416", 654, "noalign", "abv3_noal_654"))
    specs.append(_spec_base("stage2_calonly_densegate_seed456_ablate_fix_v3_20260416", 456, "densegate", "abv3_dens_456"))
    specs.append(_spec_base("stage2_calonly_densegate_seed654_ablate_fix_v3_20260416", 654, "densegate", "abv3_dens_654"))
    specs.append(_spec_base("stage2_calonly_nodelay_seed456_ablate_fix_v3_20260416", 456, "nodelay", "abv3_node_456"))
    specs.append(_spec_base("stage2_calonly_nodelay_seed654_ablate_fix_v3_20260416", 654, "nodelay", "abv3_node_654"))
    confirm_noalign = _spec_base("stage2_calonly_noalign_seed123_ablate_confirm_v3_20260416", 123, "noalign", "abv3_noal_123c")
    confirm_noalign["confirm_target_run_name"] = "stage2_calonly_noalign_seed123_ablate_fix_20260415"
    confirm_noalign["track"] = "ablation_confirm_v3"
    specs.append(confirm_noalign)
    confirm_nodelay = _spec_base("stage2_calonly_nodelay_seed42_ablate_confirm_v3_20260416", 42, "nodelay", "abv3_node_42c")
    confirm_nodelay["confirm_target_run_name"] = "stage2_calonly_nodelay_seed42_ablate_fix_20260415"
    confirm_nodelay["track"] = "ablation_confirm_v3"
    specs.append(confirm_nodelay)
    confirm_densegate = _spec_base("stage2_calonly_densegate_seed123_ablate_confirm_v3_20260416", 123, "densegate", "abv3_dens_123c")
    confirm_densegate["confirm_target_run_name"] = "stage2_calonly_densegate_seed123_ablate_fix_20260415"
    confirm_densegate["track"] = "ablation_confirm_v3"
    specs.append(confirm_densegate)
    for spec in specs:
        if spec["ablation_name"] == "noalign":
            spec["semantic_rescue_weight"] = 0.0
            spec["confidence_gated_alignment_loss_weight"] = 0.0
        elif spec["ablation_name"] == "densegate":
            spec["v6_gating_family"] = "capped_quantile_sparse_gating_v2"
            spec["v6_capped_quantile"] = 0.0
            spec["v6_max_affected_ratio"] = 1.0
        elif spec["ablation_name"] == "nodelay":
            spec["aux_loss_delay_steps"] = 0
            spec["aux_loss_ramp_steps"] = 0
    return specs


def _run_spec_map() -> Dict[str, Dict[str, Any]]:
    return {str(spec["run_name"]): spec for spec in _run_specs()}


def _meta_dir(args: Any) -> Path:
    return Path(args.work_root) / "reports/stage2_mechanism_ablation_fix_v3_runs_20260416"


def _shared_select_gpu_v3(run_name: str, lease_path: str, required_mem_gb: float = 24.0, safety_margin_gb: float = 4.0) -> Dict[str, Any]:
    # Match the current calibration-only/mainline shared-threshold policy.
    # The older helper filtered out every already-leased GPU even when those
    # leases were explicitly shared, which left these v3 reruns silently
    # retrying forever with `selected_gpu_id = -1`.
    selector = base._select_clean_gpu_for_calibration(
        run_name=str(run_name),
        lease_path=str(lease_path),
        required_mem_gb=float(required_mem_gb + safety_margin_gb),
        safety_margin_gb=0.0,
    )
    gpu_id = int(selector.get("selected_gpu_id", -1))
    lease_id = str(selector.get("lease_id", ""))
    if gpu_id < 0 or not lease_id:
        raise RuntimeError(f"no_threshold_gpu_candidate_for_{run_name}")
    return {
        "selected_gpu_id": int(gpu_id),
        "lease_id": str(lease_id),
        "selector_payload": selector,
    }


def _paths_for_run(args: Any, run_name: str) -> Dict[str, Path]:
    reports = Path(args.work_root) / "reports"
    return {
        "raw": reports / f"{run_name}_raw.json",
        "progress": reports / f"{run_name}_progress.json",
        "final": reports / f"{run_name}_final.json",
        "log": Path(args.work_root) / "logs" / f"{run_name}.log",
        "output_dir": Path(args.work_root) / "outputs/checkpoints" / run_name,
        "best": Path(args.work_root) / "outputs/checkpoints" / run_name / "best.pt",
        "latest": Path(args.work_root) / "outputs/checkpoints" / run_name / "latest.pt",
        "sidecar": Path(args.work_root) / "outputs/checkpoints" / run_name / "best_semantic_hard.pt",
    }


def _build_launch_meta(args: Any, spec: Dict[str, Any], ctx: Dict[str, Any]) -> Dict[str, Any]:
    meta = prevfix._build_launch_meta(args, spec, ctx)
    meta["objective_combo"] = str(spec["objective_combo"])
    meta["objective_family"] = str(spec["objective_family"])
    meta["track"] = str(spec.get("track", "ablation_fix_v3"))
    meta["predecode_cache_path"] = str(args.predecode_cache_path)
    meta["work_root"] = str(args.work_root)
    meta["worker_pid_file"] = str(Path(ctx["meta_dir"]) / f"{spec['run_name']}.pid")
    meta["meta_json"] = str(Path(ctx["meta_dir"]) / f"{spec['run_name']}_launch_meta.json")
    return meta


def _tmux_window_command(args: Any, meta_json: Path, meta: Dict[str, Any]) -> str:
    import os
    import shlex
    pid_path = str(meta["worker_pid_file"])
    log_path = str(meta["log_path"])
    pythonpath_value = f"{args.work_root}/code:{os.environ.get('PYTHONPATH', '')}"
    proc_title = str(os.environ.get("STWM_PROC_TITLE", "python"))
    proc_title_mode = str(os.environ.get("STWM_PROC_TITLE_MODE", "generic"))
    script_path = Path(args.work_root) / "code/stwm/tools/run_stage2_mechanism_ablation_fix_v3_20260416.py"
    cmd = (
        f"PYTHONPATH={shlex.quote(pythonpath_value)} "
        f"STWM_PROC_TITLE={shlex.quote(proc_title)} "
        f"STWM_PROC_TITLE_MODE={shlex.quote(proc_title_mode)} "
        f"nohup {shlex.quote(str(args.python_bin))} {shlex.quote(str(script_path))} --mode run-one --meta-json {shlex.quote(str(meta_json))} --work-root {shlex.quote(str(args.work_root))} "
        f">> {shlex.quote(log_path)} 2>&1 < /dev/null & echo $! > {shlex.quote(pid_path)}; "
        f"while kill -0 \"$(cat {shlex.quote(pid_path)})\" 2>/dev/null; do sleep 30; done"
    )
    return "bash -lc " + shlex.quote(
        f"cd {shlex.quote(str(args.work_root))}; rm -f {shlex.quote(pid_path)}; {cmd}; "
        f"printf '[%s] tmux_window_exit run_name={str(meta['run_name'])} observed_child_exit\\n' \"$(date -Iseconds)\" >> {shlex.quote(log_path)}"
    )


def _write_and_launch_meta(args: Any, meta: Dict[str, Any], existing_windows: set[str]) -> Dict[str, Any]:
    run_name = str(meta["run_name"])
    meta_json = Path(str(meta["meta_json"]))
    prevfix._write_json(meta_json, meta)
    cmd = _tmux_window_command(args=args, meta_json=meta_json, meta=meta)
    if str(meta["window_name"]) in existing_windows:
        subprocess.run(["tmux", "kill-window", "-t", f"{args.tmux_session}:{meta['window_name']}"], check=False)
        existing_windows.discard(str(meta["window_name"]))
    subprocess.run(["tmux", "new-window", "-t", str(args.tmux_session), "-n", str(meta["window_name"]), cmd], check=True)
    existing_windows.add(str(meta["window_name"]))
    return {
        "run_name": run_name,
        "mode": "real_train",
        "resume_from": str(meta["resume_from"]),
        "resume_global_step": int(meta["resume_global_step"]),
    }


def launch(args: Any) -> Dict[str, Any]:
    _append_log("launch_start_v3")
    ctx = prevfix._common_launch_context(args)
    actions: List[Dict[str, Any]] = []
    runs: List[Dict[str, Any]] = []
    for spec in _run_specs():
        run_name = str(spec["run_name"])
        actions.append(prevfix._reset_real_run_artifacts(args, run_name))
        meta = _build_launch_meta(args, spec, ctx)
        runs.append(_write_and_launch_meta(args, meta, ctx["existing_windows"]))
    payload = {
        "generated_at_utc": now_iso(),
        "tmux_session": str(args.tmux_session),
        "teacher_backend": BOOTSTRAP_BACKEND,
        "policy": "real-train only 20260416 oral hardening v3 mechanism closure runs; persistence disabled",
        "actions": actions,
        "runs": runs,
        "real_train_count": len(runs),
    }
    prevfix._write_json(args.launch_report, payload)
    return payload


def run_one(args: Any) -> None:
    prevfix._shared_select_gpu = _shared_select_gpu_v3
    prevfix.run_one(args)


def _collect_rows(args: Any) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    meta_dir = _meta_dir(args)
    for spec in _run_specs():
        run_name = str(spec["run_name"])
        meta_json = meta_dir / f"{run_name}_launch_meta.json"
        meta = prevfix._read_json(meta_json) if meta_json.exists() else {}
        paths = _paths_for_run(args, run_name)
        raw_payload = prevfix._read_json(paths["raw"])
        progress_payload = prevfix._read_json(paths["progress"])
        final_payload = prevfix._read_json(paths["final"])
        status_info = base._status_for(
            {**meta, "window_name": str(meta.get("window_name", spec.get("window_name", ""))), "progress_json": str(paths["progress"]), "final_json": str(paths["final"])},
            session_name=str(args.tmux_session),
        )
        resolved_status = str(status_info.get("status", "pending"))
        best_ckpt_exists = bool(paths["best"].exists())
        latest_ckpt_exists = bool(paths["latest"].exists())
        sidecar_exists = bool(paths["sidecar"].exists())
        raw_json_exists = bool(paths["raw"].exists())
        scientific_result_valid = base._scientific_artifact_valid(
            resolved_status=resolved_status,
            best_ckpt_exists=best_ckpt_exists,
            latest_ckpt_exists=latest_ckpt_exists,
            raw_json_exists=raw_json_exists,
        )
        best_block = base._best_block(final_payload, raw_payload, progress_payload)
        latest_block = base._latest_block(final_payload, raw_payload, progress_payload)
        sidecar_block = base._sidecar_block(final_payload, raw_payload, progress_payload)
        if not scientific_result_valid:
            best_block = {}
            latest_block = {}
            sidecar_block = {}
        branch = raw_payload.get("semantic_branch_summary", {}) if isinstance(raw_payload.get("semantic_branch_summary", {}), dict) else {}
        if not branch and isinstance(final_payload.get("semantic_branch_summary", {}), dict):
            branch = final_payload.get("semantic_branch_summary", {})
        if not branch and isinstance(progress_payload.get("semantic_branch_summary", {}), dict):
            branch = progress_payload.get("semantic_branch_summary", {})
        selected_gpu_id, lease_id = base._gpu_selection_from_payload(final_payload, progress_payload, meta)
        rows.append(
            {
                "run_name": run_name,
                "seed": int(spec["seed"]),
                "ablation_name": str(spec["ablation_name"]),
                "track": str(spec.get("track", "")),
                "confirm_target_run_name": str(spec.get("confirm_target_run_name", "")),
                "status": resolved_status,
                "global_step": int(progress_payload.get("global_step", best_block.get("global_step", -1))),
                "final_json_exists": bool(paths["final"].exists()),
                "best_ckpt_exists": best_ckpt_exists,
                "latest_ckpt_exists": latest_ckpt_exists,
                "sidecar_exists": sidecar_exists,
                "scientific_result_valid": bool(scientific_result_valid),
                "selected_gpu_id": int(selected_gpu_id),
                "lease_id": str(lease_id),
                "batch_size": int(meta.get("batch_size", prevfix.BATCH_SIZE)),
                "train_steps": int(meta.get("train_steps", 0)),
                "best_checkpoint_metric": best_block,
                "latest_checkpoint_metric": latest_block,
                "semantic_hard_sidecar_metric": sidecar_block,
                "actual_gate_positive_ratio_mean": float(branch.get("actual_gate_positive_ratio_mean", branch.get("eval_gate_mean", 1.0))) if scientific_result_valid and branch else None,
            }
        )
    return rows


def summarize(args: Any) -> Dict[str, Any]:
    rows = _collect_rows(args)
    running = sum(str(r.get("status", "")).lower() == "running" for r in rows)
    completed = sum(str(r.get("status", "")).lower() == "completed" for r in rows)
    failed = sum(str(r.get("status", "")).lower() == "failed" for r in rows)
    payload = {
        "generated_at_utc": now_iso(),
        "mechanism_ablation_fix_v3_status": f"{running}_running_{completed}_completed_{failed}_failed",
        "running_count": int(running),
        "completed_count": int(completed),
        "failed_count": int(failed),
        "all_runs_terminal": bool(rows and running == 0 and completed + failed == len(rows)),
        "run_rows": rows,
    }
    prevfix._write_json(args.summary_report, payload)
    return payload


def _reference_rows(args: Any, family: str) -> Dict[int, Dict[str, Any]]:
    out: Dict[int, Dict[str, Any]] = {}
    prev_summary = prevfix._read_json(args.prev_mechanism_fix_summary)
    for row in prev_summary.get("run_rows", []):
        if not isinstance(row, dict):
            continue
        if str(row.get("ablation_name", "")) != family:
            continue
        if str(row.get("status", "")).lower() != "completed":
            continue
        out[int(row.get("seed", -1))] = row
    return out


def diagnose(args: Any) -> Dict[str, Any]:
    summary = summarize(args)
    rows = summary.get("run_rows", []) if isinstance(summary.get("run_rows", []), list) else []
    mainline_summary = prevfix._read_json(args.final_pack_summary_report)
    mainline_rows = mainline_summary.get("run_rows", []) if isinstance(mainline_summary.get("run_rows", []), list) else []
    ref_by_seed = {
        int(r.get("seed", -1)): r
        for r in mainline_rows
        if isinstance(r, dict) and str(r.get("family", "")) == "topk1" and str(r.get("status", "")).lower() == "completed"
    }

    def _ep(row: Dict[str, Any]) -> float:
        return base._metric_rank_tuple(row.get("best_checkpoint_metric", {}))[0]

    def _hard(row: Dict[str, Any]) -> float:
        block = row.get("semantic_hard_sidecar_metric", {}) if isinstance(row.get("semantic_hard_sidecar_metric", {}), dict) else {}
        return base._f(block.get("semantic_hard_sidecar_score"), 1e9)

    new_by_family_seed = {
        (str(r.get("ablation_name", "")), int(r.get("seed", -1))): r
        for r in rows
    }

    family_payloads: Dict[str, Any] = {}
    family_flags: Dict[str, bool] = {}
    anomaly_resolution: Dict[str, Any] = {}
    for family, seeds in {"noalign": [42, 123, 456, 654], "densegate": [42, 123, 456, 654], "nodelay": [42, 123, 456, 654]}.items():
        old_refs = _reference_rows(args, family)
        combined_rows: List[Dict[str, Any]] = []
        for seed in seeds:
            row = new_by_family_seed.get((family, seed)) or old_refs.get(seed) or {}
            if row:
                combined_rows.append(row)
        judgments: List[Dict[str, Any]] = []
        anomalies: List[Dict[str, Any]] = []
        for row in combined_rows:
            seed = int(row.get("seed", -1))
            ref = ref_by_seed.get(seed, {})
            endpoint = float(_ep(row))
            hard = float(_hard(row))
            ref_endpoint = float(_ep(ref)) if ref else 1e9
            ref_hard = float(_hard(ref)) if ref else 1e9
            degraded = bool(ref and (endpoint > ref_endpoint or hard > ref_hard))
            anomaly = bool(ref and (endpoint < ref_endpoint or hard < ref_hard))
            judgment = {
                "run_name": str(row.get("run_name", "")),
                "seed": seed,
                "reference_run_name": str(ref.get("run_name", "none")) if ref else "none",
                "reference_endpoint_l2": ref_endpoint,
                "reference_semantic_hard_sidecar_score": ref_hard,
                "endpoint_l2": endpoint,
                "semantic_hard_sidecar_score": hard,
                "degraded_vs_reference": degraded,
                "anomaly_better_than_reference": anomaly,
                "source": "v2" if (family, seed) in new_by_family_seed else "v1",
            }
            judgments.append(judgment)
            if anomaly:
                anomalies.append(judgment)
        family_flag = bool(
            len(judgments) == 4
            and all(str((new_by_family_seed.get((family, j["seed"])) or old_refs.get(j["seed"], {})).get("status", "")).lower() == "completed" for j in judgments)
            and all(bool((new_by_family_seed.get((family, j["seed"])) or old_refs.get(j["seed"], {})).get("scientific_result_valid", False)) for j in judgments)
            and all(j["degraded_vs_reference"] and not j["anomaly_better_than_reference"] for j in judgments)
        )
        family_flags[family] = family_flag
        family_payloads[family] = {"judgments": judgments, "anomalies": anomalies, "cross_seed_support": family_flag}

    anomaly_resolution = {
        "stage2_calonly_noalign_seed123_ablate_confirm_v3_20260416": {
            "confirmed": any(j["seed"] == 123 and j["anomaly_better_than_reference"] for j in family_payloads["noalign"]["judgments"]),
            "scope": "seed123 noalign",
        },
        "stage2_calonly_nodelay_seed42_ablate_confirm_v3_20260416": {
            "confirmed": any(j["seed"] == 42 and j["anomaly_better_than_reference"] for j in family_payloads["nodelay"]["judgments"]),
            "scope": "seed42 nodelay",
        },
        "stage2_calonly_densegate_seed123_ablate_confirm_v3_20260416": {
            "confirmed": any(j["seed"] == 123 and j["anomaly_better_than_reference"] for j in family_payloads["densegate"]["judgments"]),
            "scope": "seed123 densegate",
        },
    }

    payload = {
        "generated_at_utc": now_iso(),
        "all_runs_terminal": bool(summary.get("all_runs_terminal", False)),
        "alignment_load_bearing_cross_seed": bool(family_flags.get("noalign", False)),
        "sparse_gating_load_bearing_cross_seed": bool(family_flags.get("densegate", False)),
        "delayed_schedule_load_bearing_cross_seed": bool(family_flags.get("nodelay", False)),
        "anomaly_confirmed_or_rejected": anomaly_resolution,
        "families": family_payloads,
    }
    prevfix._write_json(args.diagnosis_report, payload)
    prevfix._write_md(
        args.results_md,
        [
            "# Stage2 Mechanism Ablation Fix V3 20260416",
            "",
            f"- alignment_load_bearing_cross_seed: {payload['alignment_load_bearing_cross_seed']}",
            f"- sparse_gating_load_bearing_cross_seed: {payload['sparse_gating_load_bearing_cross_seed']}",
            f"- delayed_schedule_load_bearing_cross_seed: {payload['delayed_schedule_load_bearing_cross_seed']}",
            "",
            f"- anomaly_confirmed_or_rejected: {json.dumps(anomaly_resolution, ensure_ascii=True)}",
        ],
    )
    return payload


def run_all(args: Any) -> Dict[str, Any]:
    launch_payload = launch(args)
    deadline = time.time() + float(args.wait_timeout_seconds)
    while time.time() < deadline:
        summary = summarize(args)
        if bool(summary.get("all_runs_terminal", False)):
            diag = diagnose(args)
            return {"launch": launch_payload, "summary": summary, "diagnosis": diag}
        time.sleep(float(args.poll_seconds))
    diag = diagnose(args)
    return {"launch": launch_payload, "summary": summarize(args), "diagnosis": diag, "timeout": True}


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
    parser.add_argument("--runtime-json", default=str(ROOT / "configs/recommended_stage2_runtime_20260416.json"))
    parser.add_argument("--predecode-cache-path", default=str(ROOT / "data/processed/stage2_predecode_cache_20260416"))
    parser.add_argument("--final-pack-summary-report", default=str(ROOT / "reports/stage2_calibration_only_final_pack_summary_20260414.json"))
    parser.add_argument("--prev-mechanism-fix-summary", default=str(ROOT / "reports/stage2_mechanism_ablation_fix_v2_summary_20260416.json"))
    parser.add_argument("--launch-report", default=str(ROOT / "reports/stage2_mechanism_ablation_fix_v3_launch_20260416.json"))
    parser.add_argument("--summary-report", default=str(ROOT / "reports/stage2_mechanism_ablation_fix_v3_summary_20260416.json"))
    parser.add_argument("--diagnosis-report", default=str(ROOT / "reports/stage2_mechanism_ablation_fix_v3_diagnosis_20260416.json"))
    parser.add_argument("--results-md", default=str(ROOT / "docs/STAGE2_MECHANISM_ABLATION_FIX_V3_20260416.md"))
    parser.add_argument("--gpu-acquire-timeout-seconds", type=int, default=28800)
    parser.add_argument("--gpu-acquire-retry-seconds", type=int, default=20)
    parser.add_argument("--wait-timeout-seconds", type=int, default=172800)
    parser.add_argument("--poll-seconds", type=int, default=120)
    return parser.parse_args()


def main() -> None:
    prevfix._apply_process_title()
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
