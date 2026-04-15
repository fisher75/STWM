#!/usr/bin/env python3
from __future__ import annotations

from argparse import ArgumentParser
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple
import json
import os
import shlex
import shutil
import signal
import subprocess
import time

from stwm.tools import run_tracewm_stage2_calibration_only_fullscale_wave1_20260413 as base
from stwm.tools.run_tracewm_stage1_stage2_qualitative_pack_v3_20260413 import _best_family_run


WORK_ROOT = Path('/home/chen034/workspace/stwm')
SESSION = 'tracewm_stage2_calibration_only_finalization_pack_20260414'
LOG_PATH = WORK_ROOT / 'logs/stage2_calibration_only_finalization_pack_20260414.log'
DATE_TAG = '20260414'
FULL_EXTRA_STEPS = 4000
FULL_BATCH_SIZE = 8
FULL_EVAL_INTERVAL = 500
FULL_SAVE_EVERY = 500
FULL_EVAL_MAX_BATCHES = 0
FULL_MAX_TRAIN_PER_DATASET = -1
FULL_MAX_VAL_PER_DATASET = -1
CONCURRENT_RUNTIME_NUM_WORKERS = 8
CONCURRENT_RUNTIME_PIN_MEMORY = True
CONCURRENT_RUNTIME_PERSISTENT_WORKERS = True
CONCURRENT_RUNTIME_PREFETCH_FACTOR = 4
GPU_MIN_FREE_MEM_GB = 30.0
BOOTSTRAP_BACKEND = 'local_clip_vit_b32_mask_crop_visual_teacher'


now_iso = base.now_iso


def _apply_process_title_normalization(default_title: str = 'python') -> None:
    base._apply_process_title_normalization(default_title=default_title)


def _json_or_empty(path_like: Any) -> Dict[str, Any]:
    return base._json_or_empty(path_like)


def _select_exclusive_gpu(run_name: str, lease_path: str, required_mem_gb: float = GPU_MIN_FREE_MEM_GB, reserve_idle_gpu_count: int = 2) -> Dict[str, Any]:
    samples: List[Dict[str, Any]] = []
    for i in range(int(base.GPU_ALLOC_SAMPLE_COUNT)):
        samples.append(base.snapshot_gpu_telemetry(prefer_nvml=True))
        if i + 1 < int(base.GPU_ALLOC_SAMPLE_COUNT):
            time.sleep(float(base.GPU_ALLOC_INTERVAL_SEC))
    aggregated = base._aggregate_gpu_window(samples)
    active_leases = base.list_active_leases(lease_path=lease_path)
    leased_gpu_ids = {int(lease.get('gpu_id', -1)) for lease in active_leases if int(lease.get('gpu_id', -1)) >= 0}
    rows: List[Dict[str, Any]] = []
    candidates: List[Dict[str, Any]] = []
    for gpu_id in sorted(aggregated.keys()):
        row = dict(aggregated[gpu_id])
        enough_mem = bool(float(row.get('avg_free_mem_gb', 0.0)) >= float(required_mem_gb))
        leased = bool(int(gpu_id) in leased_gpu_ids)
        if not enough_mem:
            reason = 'filtered_insufficient_free_mem'
        elif leased:
            reason = 'filtered_existing_lease'
        else:
            reason = 'exclusive_candidate'
            candidates.append(row)
        row['selected'] = False
        row['selected_reason'] = reason
        row['leased'] = bool(leased)
        row['required_mem_gb'] = float(required_mem_gb)
        rows.append(row)
    candidates = sorted(candidates, key=lambda x: (-float(x.get('avg_free_mem_gb', 0.0)), float(x.get('avg_used_mem_gb', 0.0)), float(x.get('avg_gpu_util', 0.0)), int(x.get('active_compute_process_count', 0)), int(x.get('gpu_id', -1))))
    if len(candidates) <= int(reserve_idle_gpu_count):
        raise RuntimeError(f'not_enough_clean_gpus_for_exclusive_launch run={run_name} candidates={len(candidates)} reserve_idle_gpu_count={int(reserve_idle_gpu_count)}')
    selected_gpu_id = int(candidates[0]['gpu_id'])
    for row in rows:
        if int(row.get('gpu_id', -1)) == selected_gpu_id:
            row['selected'] = True
            row['selected_reason'] = 'best_rank_after_exclusive_filter'
        elif row.get('selected_reason') == 'exclusive_candidate':
            row['selected_reason'] = 'exclusive_candidate_not_top_rank'
    lease = base.acquire_lease(
        gpu_id=selected_gpu_id,
        owner=str(run_name),
        ttl_seconds=12 * 3600,
        lease_path=str(lease_path),
        allow_shared=False,
    )
    return {
        'selected_gpu_id': int(selected_gpu_id),
        'lease_id': str(lease.get('lease_id', '')),
        'selector_payload': {
            'generated_at_utc': now_iso(),
            'mode': 'exclusive_clean_gpu_selector_stage2_calibration_only_finalization_pack',
            'required_mem_gb': float(required_mem_gb),
            'reserve_idle_gpu_count': int(reserve_idle_gpu_count),
            'policy': {
                'allow_shared_gpu': False,
                'leave_idle_gpu_count': int(reserve_idle_gpu_count),
                'rank': ['avg_free_mem_gb desc', 'avg_used_mem_gb asc', 'avg_gpu_util asc', 'active_compute_process_count asc'],
            },
            'gpus': rows,
            'candidate_ranking': [int(x.get('gpu_id', -1)) for x in candidates],
        },
    }


def _run_specs() -> List[Dict[str, Any]]:
    mainline_specs: List[Dict[str, Any]] = []
    for seed in [789, 321, 654]:
        mainline_specs.append(
            {
                'run_name': f'stage2_calonly_topk1_seed{seed}_wave2_{DATE_TAG}',
                'track': 'mainline',
                'ablation_name': 'none',
                'family': 'topk1',
                'seed': seed,
                'objective_combo': 'calibration_only_topk1_wave2_fullscale',
                'objective_family': 'calibration_only_mainline',
                'persistence_objective_declared': False,
                'semantic_rescue_mode': 'v7alignonly',
                'semantic_rescue_weight': 0.00015,
                'confidence_gated_alignment_loss_weight': 1.0,
                'sparse_persistence_contrastive_loss_weight': 0.0,
                'confidence_gating_margin_threshold': 0.10,
                'confidence_gating_temperature': 0.05,
                'semantic_hard_score_threshold': 0.25,
                'aux_loss_delay_steps': 180,
                'aux_loss_ramp_steps': 360,
                'v6_gating_family': 'hard_topk_query_gating_v2',
                'v6_topk_query_k': 1,
                'v6_capped_quantile': 0.85,
                'v6_max_affected_ratio': 0.15,
                'v6_gate_min_strength': 0.05,
                'v6_strict_max_pairs_per_sample': 0,
                'v6_hard_negative_cap': 0,
                'v6_pair_sampling_temperature': 0.35,
                'v6_guaranteed_min_pairs_per_sample': 0,
                'v6_two_level_pair_mining_enabled': False,
                'v6_relaxed_motion_threshold': 0.08,
                'v6_relaxed_area_jump_threshold': 0.06,
                'v6_relaxed_small_query_threshold': 0.20,
                'v6_relaxed_appearance_shift_threshold': 0.25,
                'v6_relaxed_center_interaction_threshold': 0.10,
                'window_name': f'calw2_t1_s{seed}',
            }
        )
    ablations = [
        {
            'run_name': f'stage2_calonly_noalign_seed123_ablate_{DATE_TAG}',
            'track': 'ablation',
            'ablation_name': 'noalign',
            'family': 'ablation',
            'seed': 123,
            'objective_combo': 'calibration_only_noalign_ablation',
            'objective_family': 'calibration_only_mechanism_ablation',
            'persistence_objective_declared': False,
            'semantic_rescue_mode': 'v7alignonly',
            'semantic_rescue_weight': 0.0,
            'confidence_gated_alignment_loss_weight': 0.0,
            'sparse_persistence_contrastive_loss_weight': 0.0,
            'confidence_gating_margin_threshold': 0.10,
            'confidence_gating_temperature': 0.05,
            'semantic_hard_score_threshold': 0.25,
            'aux_loss_delay_steps': 180,
            'aux_loss_ramp_steps': 360,
            'v6_gating_family': 'hard_topk_query_gating_v2',
            'v6_topk_query_k': 1,
            'v6_capped_quantile': 0.85,
            'v6_max_affected_ratio': 0.15,
            'v6_gate_min_strength': 0.05,
            'v6_strict_max_pairs_per_sample': 0,
            'v6_hard_negative_cap': 0,
            'v6_pair_sampling_temperature': 0.35,
            'v6_guaranteed_min_pairs_per_sample': 0,
            'v6_two_level_pair_mining_enabled': False,
            'v6_relaxed_motion_threshold': 0.08,
            'v6_relaxed_area_jump_threshold': 0.06,
            'v6_relaxed_small_query_threshold': 0.20,
            'v6_relaxed_appearance_shift_threshold': 0.25,
            'v6_relaxed_center_interaction_threshold': 0.10,
            'window_name': 'abl_noalign',
        },
        {
            'run_name': f'stage2_calonly_densegate_seed123_ablate_{DATE_TAG}',
            'track': 'ablation',
            'ablation_name': 'densegate',
            'family': 'ablation',
            'seed': 123,
            'objective_combo': 'calibration_only_densegate_ablation',
            'objective_family': 'calibration_only_mechanism_ablation',
            'persistence_objective_declared': False,
            'semantic_rescue_mode': 'v7alignonly',
            'semantic_rescue_weight': 0.00015,
            'confidence_gated_alignment_loss_weight': 1.0,
            'sparse_persistence_contrastive_loss_weight': 0.0,
            'confidence_gating_margin_threshold': 0.10,
            'confidence_gating_temperature': 0.05,
            'semantic_hard_score_threshold': 0.25,
            'aux_loss_delay_steps': 180,
            'aux_loss_ramp_steps': 360,
            'v6_gating_family': 'capped_quantile_sparse_gating_v2',
            'v6_topk_query_k': 1,
            'v6_capped_quantile': 0.0,
            'v6_max_affected_ratio': 1.0,
            'v6_gate_min_strength': 0.05,
            'v6_strict_max_pairs_per_sample': 0,
            'v6_hard_negative_cap': 0,
            'v6_pair_sampling_temperature': 0.35,
            'v6_guaranteed_min_pairs_per_sample': 0,
            'v6_two_level_pair_mining_enabled': False,
            'v6_relaxed_motion_threshold': 0.08,
            'v6_relaxed_area_jump_threshold': 0.06,
            'v6_relaxed_small_query_threshold': 0.20,
            'v6_relaxed_appearance_shift_threshold': 0.25,
            'v6_relaxed_center_interaction_threshold': 0.10,
            'window_name': 'abl_dense',
        },
        {
            'run_name': f'stage2_calonly_nodelay_seed123_ablate_{DATE_TAG}',
            'track': 'ablation',
            'ablation_name': 'nodelay',
            'family': 'ablation',
            'seed': 123,
            'objective_combo': 'calibration_only_nodelay_ablation',
            'objective_family': 'calibration_only_mechanism_ablation',
            'persistence_objective_declared': False,
            'semantic_rescue_mode': 'v7alignonly',
            'semantic_rescue_weight': 0.00015,
            'confidence_gated_alignment_loss_weight': 1.0,
            'sparse_persistence_contrastive_loss_weight': 0.0,
            'confidence_gating_margin_threshold': 0.10,
            'confidence_gating_temperature': 0.05,
            'semantic_hard_score_threshold': 0.25,
            'aux_loss_delay_steps': 0,
            'aux_loss_ramp_steps': 0,
            'v6_gating_family': 'hard_topk_query_gating_v2',
            'v6_topk_query_k': 1,
            'v6_capped_quantile': 0.85,
            'v6_max_affected_ratio': 0.15,
            'v6_gate_min_strength': 0.05,
            'v6_strict_max_pairs_per_sample': 0,
            'v6_hard_negative_cap': 0,
            'v6_pair_sampling_temperature': 0.35,
            'v6_guaranteed_min_pairs_per_sample': 0,
            'v6_two_level_pair_mining_enabled': False,
            'v6_relaxed_motion_threshold': 0.08,
            'v6_relaxed_area_jump_threshold': 0.06,
            'v6_relaxed_small_query_threshold': 0.20,
            'v6_relaxed_appearance_shift_threshold': 0.25,
            'v6_relaxed_center_interaction_threshold': 0.10,
            'window_name': 'abl_nodelay',
        },
    ]
    return mainline_specs + ablations


def _launch_meta_dir(args: Any) -> Path:
    return Path(args.work_root) / 'reports' / 'stage2_calibration_only_wave2_runs_20260414'


def _launch_meta_by_run(args: Any) -> Dict[str, Dict[str, Any]]:
    meta_dir = _launch_meta_dir(args)
    rows: Dict[str, Dict[str, Any]] = {}
    if meta_dir.exists():
        for meta_json in sorted(meta_dir.glob('*_launch_meta.json')):
            meta = _json_or_empty(meta_json)
            run_name = str(meta.get('run_name', '')).strip()
            if run_name:
                rows[run_name] = meta
    if rows:
        return rows
    launch = _json_or_empty(args.launch_report)
    return {
        str(meta.get('run_name', '')): meta
        for meta in launch.get('runs', [])
        if isinstance(meta, dict) and str(meta.get('run_name', '')).strip()
    }


def _paths_for(args: Any, meta: Dict[str, Any], run_name: str) -> Dict[str, Path]:
    report_root = Path(args.work_root) / 'reports'
    ckpt_root = Path(args.work_root) / 'outputs' / 'checkpoints' / run_name
    return {
        'launch': _launch_meta_dir(args) / f'{run_name}_launch_meta.json',
        'progress': Path(str(meta.get('progress_json', report_root / f'{run_name}_progress.json'))),
        'final': Path(str(meta.get('final_json', report_root / f'{run_name}_final.json'))),
        'raw': Path(str(meta.get('raw_json', report_root / f'{run_name}_raw.json'))),
        'best': ckpt_root / 'best.pt',
        'latest': ckpt_root / 'latest.pt',
        'sidecar': ckpt_root / 'best_semantic_hard.pt',
    }


def _rotate_path_if_exists(path: Path) -> str:
    if not path.exists():
        return ''
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    rotated = path.with_name(f'{path.name}.failed_snapshot_{stamp}')
    path.rename(rotated)
    return str(rotated)


def _reset_run_artifacts(args: Any, meta: Dict[str, Any], run_name: str) -> Dict[str, Any]:
    paths = _paths_for(args, meta, run_name)
    output_dir = Path(str(meta.get('output_dir', paths['best'].parent)))
    rotated_logs: List[str] = []
    removed_paths: List[str] = []
    pid_file = Path(str(meta.get('worker_pid_file', '')))
    if str(pid_file) and pid_file.exists():
        try:
            pid = base._int_or_default(pid_file.read_text(encoding='utf-8').strip(), -1)
            if base._pid_alive(pid):
                try:
                    os.kill(pid, signal.SIGTERM)
                except Exception:
                    pass
        except Exception:
            pass
        try:
            pid_file.unlink()
            removed_paths.append(str(pid_file))
        except Exception:
            pass
    for key in ['progress', 'final', 'raw', 'launch']:
        target = paths[key]
        if target.exists():
            target.unlink()
            removed_paths.append(str(target))
    log_path = Path(str(meta.get('log_path', '')))
    if str(log_path):
        rotated = _rotate_path_if_exists(log_path)
        if rotated:
            rotated_logs.append(rotated)
    if output_dir.exists():
        shutil.rmtree(output_dir)
        removed_paths.append(str(output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    return {
        'run_name': str(run_name),
        'removed_paths': removed_paths,
        'rotated_logs': rotated_logs,
    }


def _tmux_window_command(args: Any, meta_json: Path, meta: Dict[str, Any]) -> str:
    run_name = str(meta.get('run_name', ''))
    log_path = str(meta.get('log_path', ''))
    pid_path = str(meta.get('worker_pid_file', ''))
    script_path = Path(args.work_root) / 'code/stwm/tools/run_tracewm_stage2_calibration_only_wave2_20260414.py'
    pythonpath_value = f"{args.work_root}/code:{os.environ.get('PYTHONPATH', '')}"
    proc_title = str(os.environ.get('STWM_PROC_TITLE', 'python'))
    proc_title_mode = str(os.environ.get('STWM_PROC_TITLE_MODE', 'generic'))
    cmd = (
        f'PYTHONPATH={shlex.quote(pythonpath_value)} '
        f'STWM_PROC_TITLE={shlex.quote(proc_title)} '
        f'STWM_PROC_TITLE_MODE={shlex.quote(proc_title_mode)} '
        f'nohup {shlex.quote(str(args.python_bin))} '
        f'{shlex.quote(str(script_path))} '
        f'--mode run-one --meta-json {shlex.quote(str(meta_json))} '
        f'>> {shlex.quote(log_path)} 2>&1 < /dev/null & '
        f'echo $! > {shlex.quote(pid_path)}; '
        f'while kill -0 "$(cat {shlex.quote(pid_path)})" 2>/dev/null; do sleep 30; done'
    )
    return (
        'bash -lc '
        + shlex.quote(
            f"cd {shlex.quote(str(args.work_root))}; "
            f"rm -f {shlex.quote(pid_path)}; "
            f"{cmd}; "
            f"printf '[%s] tmux_window_exit run_name={run_name} observed_child_exit\n' \"$(date -Iseconds)\" >> {shlex.quote(log_path)}"
        )
    )


def write_process_title_report(args: Any) -> Dict[str, Any]:
    payload = {
        'generated_at_utc': now_iso(),
        'scope': 'stage2_process_title_normalization',
        'supported_env': {
            'STWM_PROC_TITLE': 'python',
            'STWM_PROC_TITLE_MODE': 'generic',
        },
        'neutral_titles_allowed': ['python', 'python:train', 'python:eval'],
        'forbidden_patterns': ['stwm', 'tracewm', '/home/chen034/workspace/stwm'],
        'patched_entrypoints': [
            'code/stwm/tracewm_v2_stage2/trainers/train_tracewm_stage2_smalltrain.py',
            'code/stwm/tools/run_tracewm_stage2_calibration_only_wave2_20260414.py',
            'code/stwm/tools/run_tracewm_stage1_stage2_qualitative_pack_v5_20260414.py',
            'code/stwm/tracewm_v2_stage2/tools/stage2_external_eval_bridge.py',
        ],
        'runtime_policy': 'process title stays generic only; run_name remains in logs/reports/checkpoints',
    }
    base._write_json(args.process_title_report, payload)
    return payload


def write_runtime_report(args: Any) -> Dict[str, Any]:
    payload = {
        'generated_at_utc': now_iso(),
        'mode': 'stage2_calibration_only_wave2_runtime',
        'selected_gpu_policy': {
            'mode': 'strict_clean_gpu_selector',
            'required_mem_gb': float(GPU_MIN_FREE_MEM_GB),
            'reserve_idle_gpu_count': int(args.reserve_idle_gpu_count),
        },
        'recommended_num_workers': int(CONCURRENT_RUNTIME_NUM_WORKERS),
        'recommended_pin_memory': bool(CONCURRENT_RUNTIME_PIN_MEMORY),
        'recommended_persistent_workers': bool(CONCURRENT_RUNTIME_PERSISTENT_WORKERS),
        'recommended_prefetch_factor': int(CONCURRENT_RUNTIME_PREFETCH_FACTOR),
        'single_gpu_only': True,
        'notes': [
            '20260414 finalization pack: 3 wave2 fresh seeds + 3 mechanism ablations',
            'hard cap: 6 concurrent training tasks, keep 2 GPUs idle',
            'mainline family fixed to topk1 calibration-only; persistence disabled everywhere',
        ],
    }
    base._write_json(args.concurrent_runtime_report, payload)
    return payload


def _resume_ckpt_for_spec(spec: Dict[str, Any]) -> Path:
    seed = int(spec.get('seed', 42))
    return base._resume_ckpt_for_seed(seed)


def launch(args: Any) -> Dict[str, Any]:
    lease_cleanup = base._cleanup_stale_leases(str(args.shared_lease_path), allowed_prefixes=('stage2_calonly_', 'stage2_partialunfreeze_', 'stage2_aux_external_probe_batch_'))
    runtime_report = write_runtime_report(args)
    if subprocess.run(['tmux', 'has-session', '-t', str(args.tmux_session)], capture_output=True).returncode != 0:
        subprocess.run(['tmux', 'new-session', '-d', '-s', str(args.tmux_session), 'bash'], check=True)

    anchor_args = base._load_ckpt_args(base._resume_ckpt_for_seed(42))
    obs_len = int(anchor_args.get('obs_len', 8) or 8)
    fut_len = int(anchor_args.get('fut_len', 8) or 8)
    max_tokens = int(anchor_args.get('max_tokens', 64) or 64)
    crop_size = int(anchor_args.get('semantic_crop_size', 64) or 64)
    train_counts = base._dataset_counts(['vspw', 'vipseg'], 'train', args.stage2_contract_json, max_samples=FULL_MAX_TRAIN_PER_DATASET)
    val_counts = base._dataset_counts(['vspw', 'vipseg'], 'val', args.stage2_contract_json, max_samples=FULL_MAX_VAL_PER_DATASET)

    runs: List[Dict[str, Any]] = []
    cleanup_actions: List[Dict[str, Any]] = []
    meta_dir = _launch_meta_dir(args)
    meta_dir.mkdir(parents=True, exist_ok=True)
    existing_windows = set(base._tmux_windows(str(args.tmux_session)))
    for spec in _run_specs():
        run_name = str(spec['run_name'])
        resume_from = _resume_ckpt_for_spec(spec)
        resume_step = base._load_ckpt_step(resume_from)
        out_dir = Path(args.work_root) / 'outputs' / 'checkpoints' / run_name
        meta = {
            **spec,
            'selected_gpu_id': -1,
            'lease_id': '',
            'dataset_names': ['vspw', 'vipseg'],
            'obs_len': obs_len,
            'fut_len': fut_len,
            'max_tokens': max_tokens,
            'semantic_crop_size': crop_size,
            'semantic_source_mainline': 'crop_visual_encoder',
            'legacy_semantic_source': 'hand_crafted_stats',
            'batch_size': FULL_BATCH_SIZE,
            'resume_from': str(resume_from),
            'resume_global_step': int(resume_step),
            'additional_train_steps': FULL_EXTRA_STEPS,
            'train_steps': int(resume_step + FULL_EXTRA_STEPS),
            'eval_interval': FULL_EVAL_INTERVAL,
            'eval_max_batches': FULL_EVAL_MAX_BATCHES,
            'save_every_n_steps': FULL_SAVE_EVERY,
            'max_samples_train': FULL_MAX_TRAIN_PER_DATASET,
            'max_samples_val': FULL_MAX_VAL_PER_DATASET,
            'effective_train_sample_count_per_dataset': train_counts,
            'effective_val_sample_count_per_dataset': val_counts,
            'semantic_bootstrap_target_dim': 512,
            'semantic_hard_curriculum_weight': 0.0,
            'semantic_aux_subset_weighting_strength': 1.0,
            'output_dir': str(out_dir),
            'raw_json': str(Path(args.work_root) / 'reports' / f'{run_name}_raw.json'),
            'progress_json': str(Path(args.work_root) / 'reports' / f'{run_name}_progress.json'),
            'final_json': str(Path(args.work_root) / 'reports' / f'{run_name}_final.json'),
            'log_path': str(Path(args.work_root) / 'logs' / f'{run_name}.log'),
            'stage2_contract_json': str(args.stage2_contract_json),
            'stage1_runtime_json': str(args.concurrent_runtime_report),
            'stage1_best_ckpt': str(args.stage1_best_ckpt),
            'shared_lease_path': str(args.shared_lease_path),
            'bootstrap_cache_jsonl': str(args.bootstrap_cache_jsonl),
            'semantic_hard_manifest_path': str(args.semantic_hard_manifest_path),
            'work_root': str(args.work_root),
            'python_bin': str(args.python_bin),
            'worker_pid_file': str(meta_dir / f'{run_name}.pid'),
            'reserve_idle_gpu_count': int(args.reserve_idle_gpu_count),
            'gpu_acquire_timeout_seconds': int(args.gpu_acquire_timeout_seconds),
            'gpu_acquire_retry_seconds': int(args.gpu_acquire_retry_seconds),
        }
        meta_json = meta_dir / f'{run_name}_launch_meta.json'
        meta['meta_json'] = str(meta_json)
        cleanup_actions.append(_reset_run_artifacts(args=args, meta=meta, run_name=run_name))
        meta['selector_payload'] = {}
        base._write_json(meta_json, meta)
        runs.append(meta)
        cmd = _tmux_window_command(args=args, meta_json=meta_json, meta=meta)
        if str(meta['window_name']) not in existing_windows:
            subprocess.run(['tmux', 'new-window', '-t', str(args.tmux_session), '-n', str(meta['window_name']), cmd], check=True)
            existing_windows.add(str(meta['window_name']))

    payload = {
        'generated_at_utc': now_iso(),
        'mode': 'stage2_calibration_only_finalization_pack_launch',
        'tmux_session': str(args.tmux_session),
        'teacher_backend': BOOTSTRAP_BACKEND,
        'policy': 'wave2 topk1 calibration-only fresh seeds + mechanism ablations; persistence disabled; max 6 training tasks; reserve 2 idle GPUs',
        'lease_cleanup': lease_cleanup,
        'cleanup_actions': cleanup_actions,
        'concurrent_runtime_report': str(args.concurrent_runtime_report),
        'concurrent_runtime': runtime_report,
        'runs': runs,
    }
    base._write_json(args.launch_report, payload)
    return summarize(args)


def _collect_all_rows(args: Any) -> List[Dict[str, Any]]:
    meta_by_run = _launch_meta_by_run(args)
    run_rows: List[Dict[str, Any]] = []
    for spec in _run_specs():
        run_name = str(spec['run_name'])
        meta = meta_by_run.get(run_name, {})
        paths = _paths_for(args, meta, run_name)
        progress_payload = _json_or_empty(paths['progress'])
        final_payload = _json_or_empty(paths['final'])
        raw_payload = _json_or_empty(paths['raw'])
        status_info = base._status_for({**meta, 'window_name': str(meta.get('window_name', spec.get('window_name', ''))), 'progress_json': str(paths['progress']), 'final_json': str(paths['final'])}, session_name=str(args.tmux_session))
        resolved_status = str(status_info.get('status', 'launched')).lower()
        if resolved_status == 'failed' and not paths['final'].exists():
            base._write_json(
                paths['final'],
                {
                    'generated_at_utc': now_iso(),
                    'run_name': run_name,
                    'status': 'failed',
                    'message': str(status_info.get('salvage_reason', 'failed_without_final_artifact')),
                    'salvaged_from_progress': bool(paths['progress'].exists()),
                },
            )
            final_payload = _json_or_empty(paths['final'])
        best_ckpt_exists = bool(paths['best'].exists())
        latest_ckpt_exists = bool(paths['latest'].exists())
        sidecar_exists = bool(paths['sidecar'].exists())
        raw_json_exists = bool(paths['raw'].exists())
        scientific_result_valid = base._scientific_artifact_valid(
            resolved_status=resolved_status,
            best_ckpt_exists=best_ckpt_exists,
            latest_ckpt_exists=latest_ckpt_exists,
            raw_json_exists=raw_json_exists,
        )
        best_block = base._best_block(final_payload, raw_payload, progress_payload)
        latest_block = base._latest_block(final_payload, raw_payload, progress_payload)
        sidecar_block = base._sidecar_block(final_payload, raw_payload, progress_payload)
        branch = base._branch_block(final_payload, raw_payload, progress_payload)
        if not scientific_result_valid:
            best_block = {}
            latest_block = {}
            sidecar_block = {}
            branch = {}
        sidecar_sel = raw_payload.get('sidecar_checkpoint_selection', {}) if isinstance(raw_payload.get('sidecar_checkpoint_selection', {}), dict) else final_payload.get('sidecar_checkpoint_selection', {}) if isinstance(final_payload.get('sidecar_checkpoint_selection', {}), dict) else {}
        if not sidecar_sel and isinstance(progress_payload.get('sidecar_checkpoint_selection', {}), dict):
            sidecar_sel = progress_payload.get('sidecar_checkpoint_selection', {})
        global_step = base._int_or_default(progress_payload.get('global_step', best_block.get('global_step', -1)), -1)
        selected_gpu_id, lease_id = base._gpu_selection_from_payload(final_payload, progress_payload, meta)
        run_rows.append(
            {
                'run_name': run_name,
                'track': str(spec.get('track', 'mainline')),
                'ablation_name': str(spec.get('ablation_name', 'none')),
                'family': str(spec.get('family', 'topk1')),
                'seed': int(spec.get('seed', -1)),
                'status': resolved_status,
                'global_step': global_step,
                'final_json_exists': bool(paths['final'].exists()),
                'progress_json_exists': bool(paths['progress'].exists()),
                'raw_json_exists': raw_json_exists,
                'best_ckpt_exists': best_ckpt_exists,
                'latest_ckpt_exists': latest_ckpt_exists,
                'sidecar_exists': sidecar_exists,
                'scientific_result_valid': bool(scientific_result_valid),
                'selected_gpu_id': int(selected_gpu_id),
                'lease_id': str(lease_id),
                'batch_size': int(meta.get('batch_size', FULL_BATCH_SIZE)),
                'train_steps': int(meta.get('train_steps', 0)),
                'eval_interval': int(meta.get('eval_interval', 0)),
                'save_every_n_steps': int(meta.get('save_every_n_steps', 0)),
                'effective_train_sample_count_per_dataset': meta.get('effective_train_sample_count_per_dataset', {}),
                'effective_val_sample_count_per_dataset': meta.get('effective_val_sample_count_per_dataset', {}),
                'best_checkpoint_metric': best_block,
                'latest_checkpoint_metric': latest_block,
                'semantic_hard_sidecar_metric': sidecar_block,
                'actual_gate_positive_ratio_mean': (float(branch.get('actual_gate_positive_ratio_mean', branch.get('eval_gate_mean', 1.0))) if scientific_result_valid and isinstance(branch, dict) and branch else None),
                'valuable_pair_ratio_mean': (float(branch.get('valuable_pair_ratio_mean', branch.get('high_value_pair_ratio', 0.0))) if scientific_result_valid and isinstance(branch, dict) and branch else None),
                'same_checkpoint_selected': bool(sidecar_sel.get('same_checkpoint_selected', True)),
                'sidecar_truly_diverged': bool(sidecar_sel.get('sidecar_truly_diverged', False)),
                'gating_family': str(spec.get('v6_gating_family', '')),
                'aux_loss_delay_steps': int(spec.get('aux_loss_delay_steps', 0)),
                'aux_loss_ramp_steps': int(spec.get('aux_loss_ramp_steps', 0)),
                'persistence_objective_declared': False,
                'persistence_objective_effective': False,
            }
        )
    return run_rows


def _completed_valid(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [row for row in rows if str(row.get('status', '')).lower() == 'completed' and bool(row.get('scientific_result_valid', False))]


def _filter_rows(rows: List[Dict[str, Any]], track: str) -> List[Dict[str, Any]]:
    return [row for row in rows if str(row.get('track', '')) == track]


def _agg_payload(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    completed = _completed_valid(rows)
    return {
        'count': len(completed),
        'free_rollout_endpoint_l2': base._mean_std([base._metric_rank_tuple(row.get('best_checkpoint_metric', {}))[0] for row in completed]),
        'free_rollout_coord_mean_l2': base._mean_std([base._metric_rank_tuple(row.get('best_checkpoint_metric', {}))[1] for row in completed]),
        'teacher_forced_coord_loss': base._mean_std([base._metric_rank_tuple(row.get('best_checkpoint_metric', {}))[2] for row in completed]),
        'semantic_hard_sidecar_score': base._mean_std([base._f((row.get('semantic_hard_sidecar_metric', {}) if isinstance(row.get('semantic_hard_sidecar_metric', {}), dict) else {}).get('semantic_hard_sidecar_score'), 1e9) for row in completed]),
        'actual_gate_positive_ratio_mean': base._mean_std([float(row.get('actual_gate_positive_ratio_mean', 1.0)) for row in completed]),
    }


def _wave1_topk1_rows(args: Any) -> List[Dict[str, Any]]:
    payload = _json_or_empty(args.wave1_summary_report)
    rows = payload.get('run_rows', []) if isinstance(payload.get('run_rows', []), list) else []
    out = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        if str(row.get('family', '')) != 'topk1':
            continue
        if str(row.get('status', '')).lower() != 'completed':
            continue
        if not bool(row.get('scientific_result_valid', True)):
            continue
        out.append(row)
    return out


def _specific_run_row(summary_payload: Dict[str, Any], run_name: str) -> Dict[str, Any]:
    rows = summary_payload.get('run_rows', []) if isinstance(summary_payload.get('run_rows', []), list) else []
    for row in rows:
        if isinstance(row, dict) and str(row.get('run_name', '')) == str(run_name):
            return row
    return {}


def _write_ablation_pack(args: Any, all_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    rows = _filter_rows(all_rows, 'ablation')
    running = sum(int(str(r.get('status', '')).lower() == 'running') for r in rows)
    completed = sum(int(str(r.get('status', '')).lower() == 'completed') for r in rows)
    failed = sum(int(str(r.get('status', '')).lower() == 'failed') for r in rows)
    wave1_summary = _json_or_empty(args.wave1_summary_report)
    reference_run_name = 'stage2_calonly_topk1_seed123_wave1_20260413'
    reference_row = _specific_run_row(wave1_summary, reference_run_name)
    ref_ep = base._metric_rank_tuple(reference_row.get('best_checkpoint_metric', {}))[0] if reference_row else 1e9
    ref_hard = base._f((reference_row.get('semantic_hard_sidecar_metric', {}) if isinstance(reference_row.get('semantic_hard_sidecar_metric', {}), dict) else {}).get('semantic_hard_sidecar_score'), 1e9)
    ref_gate = float(reference_row.get('actual_gate_positive_ratio_mean', 1.0)) if reference_row else 1.0

    judgments: Dict[str, Any] = {}
    for ablation_name in ['noalign', 'densegate', 'nodelay']:
        row = next((r for r in rows if str(r.get('ablation_name', '')) == ablation_name), {})
        valid = bool(row) and str(row.get('status', '')).lower() == 'completed' and bool(row.get('scientific_result_valid', False))
        endpoint = base._metric_rank_tuple(row.get('best_checkpoint_metric', {}))[0] if valid else 1e9
        hard = base._f((row.get('semantic_hard_sidecar_metric', {}) if isinstance(row.get('semantic_hard_sidecar_metric', {}), dict) else {}).get('semantic_hard_sidecar_score'), 1e9) if valid else 1e9
        gate = float(row.get('actual_gate_positive_ratio_mean', 1.0)) if valid and row.get('actual_gate_positive_ratio_mean', None) is not None else 1.0
        judgments[ablation_name] = {
            'run_name': str(row.get('run_name', 'none')) if row else 'none',
            'completed': bool(valid),
            'endpoint_l2': float(endpoint),
            'semantic_hard_sidecar_score': float(hard),
            'gate_ratio': float(gate),
            'worse_than_reference': bool(valid and (endpoint > ref_ep or hard > ref_hard)),
            'expected_direction_observed': bool(
                valid and (
                    (ablation_name == 'densegate' and gate > max(0.50, ref_gate + 0.20) and (endpoint > ref_ep or hard > ref_hard))
                    or (ablation_name != 'densegate' and (endpoint > ref_ep or hard > ref_hard))
                )
            ),
        }

    all_completed = bool(rows and completed == len(rows) and failed == 0)
    load_bearing = bool(all_completed and all(judgments[name]['expected_direction_observed'] for name in ['noalign', 'densegate', 'nodelay']))
    payload = {
        'generated_at_utc': now_iso(),
        'ablation_pack_status': f'{running}_running_{completed}_completed_{failed}_failed',
        'running_count': running,
        'completed_count': completed,
        'failed_count': failed,
        'all_runs_terminal': bool(len(rows) > 0 and running == 0 and completed + failed == len(rows)),
        'reference_mainline_run_name': reference_run_name,
        'reference_best_checkpoint_metric': reference_row.get('best_checkpoint_metric', {}) if reference_row else {},
        'reference_semantic_hard_sidecar_metric': reference_row.get('semantic_hard_sidecar_metric', {}) if reference_row else {},
        'run_rows': rows,
        'judgments': judgments,
        'alignment_sparse_gating_delay_load_bearing': bool(load_bearing),
    }
    base._write_json(args.ablation_pack_report, payload)
    lines = [
        '# Stage2 Calibration-Only Ablation Pack',
        '',
        f'- generated_at_utc: {payload["generated_at_utc"]}',
        f'- ablation_pack_status: {payload["ablation_pack_status"]}',
        f'- reference_mainline_run_name: {reference_run_name}',
        f'- alignment_sparse_gating_delay_load_bearing: {payload["alignment_sparse_gating_delay_load_bearing"]}',
        '',
        '| ablation | run_name | completed | endpoint_l2 | hard_score | gate_ratio | expected_direction_observed |',
        '|---|---|---|---:|---:|---:|---|',
    ]
    for name in ['noalign', 'densegate', 'nodelay']:
        row = judgments[name]
        lines.append(
            f"| {name} | {row['run_name']} | {row['completed']} | {row['endpoint_l2']:.6f} | {row['semantic_hard_sidecar_score']:.6f} | {row['gate_ratio']:.4f} | {row['expected_direction_observed']} |"
        )
    base._write_md(args.ablation_pack_md, lines)
    return payload


def summarize(args: Any) -> Dict[str, Any]:
    all_rows = _collect_all_rows(args)
    rows = _filter_rows(all_rows, 'mainline')
    running = sum(int(str(r.get('status', '')).lower() == 'running') for r in rows)
    completed = sum(int(str(r.get('status', '')).lower() == 'completed') for r in rows)
    failed = sum(int(str(r.get('status', '')).lower() == 'failed') for r in rows)
    completed_rows = _completed_valid(rows)
    overall_best_run_name = 'none'
    if completed_rows:
        overall_best_run_name = str(min(completed_rows, key=base._summary_overall_rank).get('run_name', 'none'))
    payload = {
        'generated_at_utc': now_iso(),
        'stage2_calibration_only_wave2_status': f'{running}_running_{completed}_completed_{failed}_failed',
        'running_count': running,
        'completed_count': completed,
        'failed_count': failed,
        'all_runs_terminal': bool(len(rows) > 0 and running == 0 and completed + failed == len(rows)),
        'run_rows': rows,
        'overall_best_run_name': overall_best_run_name,
        'best_family': 'topk1',
        'family_aggregate': _agg_payload(rows),
        'teacher_backend': BOOTSTRAP_BACKEND,
        'next_step_choice_internal': '',
    }
    payload['next_step_choice_internal'] = (
        'ready_for_wave2_diagnosis'
        if payload['all_runs_terminal'] and payload['failed_count'] == 0 and payload['completed_count'] == len(rows)
        else ('fix_failed_runs' if payload['failed_count'] > 0 else 'continue_running')
    )
    base._write_json(args.summary_report, payload)
    _write_ablation_pack(args, all_rows)
    return payload


def wait_for_completion(args: Any) -> Dict[str, Any]:
    deadline = time.time() + float(args.wait_timeout_seconds)
    last = summarize(args)
    while time.time() < deadline:
        ablation = _json_or_empty(args.ablation_pack_report)
        if bool(last.get('all_runs_terminal', False)) and bool(ablation.get('all_runs_terminal', False)):
            return last
        time.sleep(float(args.poll_seconds))
        last = summarize(args)
    last['timed_out_waiting_for_completion'] = True
    base._write_json(args.summary_report, last)
    return last


def _wave1_diag(args: Any) -> Dict[str, Any]:
    return _json_or_empty(args.wave1_diagnosis_report)


def _v7_diag(args: Any) -> Dict[str, Any]:
    return _json_or_empty(args.v7_repaired_diagnosis_report)


def _v7_summary(args: Any) -> Dict[str, Any]:
    return _json_or_empty(args.v7_repaired_summary_report)


def _write_wave2_results_md(args: Any, summary: Dict[str, Any], diagnosis: Dict[str, Any]) -> None:
    lines = [
        '# Stage2 Calibration-Only Wave2 Results',
        '',
        f'- generated_at_utc: {now_iso()}',
        f"- stage2_calibration_only_wave2_status: {summary.get('stage2_calibration_only_wave2_status', 'unknown')}",
        f"- overall_best_run_name: {diagnosis.get('overall_best_run_name', 'none')}",
        f"- wave2_support_present: {diagnosis.get('wave2_support_present', None)}",
        f"- calibration_only_improved_vs_current_cropenc_baseline: {diagnosis.get('calibration_only_improved_vs_current_cropenc_baseline', None)}",
        f"- calibration_only_improved_vs_wave1_best: {diagnosis.get('calibration_only_improved_vs_wave1_best', None)}",
        f"- semantic_hard_signal_preserved: {diagnosis.get('semantic_hard_signal_preserved', None)}",
        '',
        '| run_name | seed | status | global_step | endpoint_l2 | hard_score | gate_ratio |',
        '|---|---:|---|---:|---:|---:|---:|',
    ]
    for row in summary.get('run_rows', []) if isinstance(summary.get('run_rows', []), list) else []:
        endpoint = base._metric_rank_tuple(row.get('best_checkpoint_metric', {}))[0] if bool(row.get('scientific_result_valid', False)) else None
        hard = base._f((row.get('semantic_hard_sidecar_metric', {}) if isinstance(row.get('semantic_hard_sidecar_metric', {}), dict) else {}).get('semantic_hard_sidecar_score'), 1e9) if bool(row.get('scientific_result_valid', False)) else None
        gate = row.get('actual_gate_positive_ratio_mean', None) if bool(row.get('scientific_result_valid', False)) else None
        lines.append(
            f"| {row.get('run_name', '')} | {int(row.get('seed', -1))} | {row.get('status', '')} | {int(row.get('global_step', -1))} | {(f'{endpoint:.6f}' if endpoint is not None else 'n/a')} | {(f'{hard:.6f}' if hard is not None else 'n/a')} | {(f'{float(gate):.4f}' if gate is not None else 'n/a')} |"
        )
    base._write_md(args.results_md, lines)


def _write_final_pack_summary(args: Any, wave2_summary: Dict[str, Any]) -> Dict[str, Any]:
    wave1_rows = _wave1_topk1_rows(args)
    wave2_rows = _completed_valid(wave2_summary.get('run_rows', []) if isinstance(wave2_summary.get('run_rows', []), list) else [])
    combined = list(wave1_rows) + list(wave2_rows)
    overall_best = min(combined, key=base._summary_overall_rank) if combined else {}
    hard_best = min(combined, key=base._summary_hard_rank) if combined else {}
    payload = {
        'generated_at_utc': now_iso(),
        'status': 'completed' if len(combined) == 6 else 'pending',
        'wave1_topk1_run_names': [str(r.get('run_name', '')) for r in wave1_rows],
        'wave2_topk1_run_names': [str(r.get('run_name', '')) for r in wave2_rows],
        'all_topk1_run_names': [str(r.get('run_name', '')) for r in combined],
        'six_seed_count': len(combined),
        'endpoint_mean_std': _agg_payload(combined)['free_rollout_endpoint_l2'],
        'coord_mean_mean_std': _agg_payload(combined)['free_rollout_coord_mean_l2'],
        'teacher_forced_mean_std': _agg_payload(combined)['teacher_forced_coord_loss'],
        'semantic_hard_sidecar_score_mean_std': _agg_payload(combined)['semantic_hard_sidecar_score'],
        'actual_gate_positive_ratio_mean_std': _agg_payload(combined)['actual_gate_positive_ratio_mean'],
        'current_best_overall_run_name': str(overall_best.get('run_name', 'none')) if overall_best else 'none',
        'current_best_semantic_hard_run_name': str(hard_best.get('run_name', 'none')) if hard_best else 'none',
        'run_rows': combined,
    }
    base._write_json(args.final_pack_summary_report, payload)
    lines = [
        '# Stage2 Calibration-Only Final Pack Results',
        '',
        f'- generated_at_utc: {payload["generated_at_utc"]}',
        f'- six_seed_count: {payload["six_seed_count"]}',
        f'- current_best_overall_run_name: {payload["current_best_overall_run_name"]}',
        f'- current_best_semantic_hard_run_name: {payload["current_best_semantic_hard_run_name"]}',
        f"- endpoint_mean_std: {payload['endpoint_mean_std']}",
        f"- coord_mean_mean_std: {payload['coord_mean_mean_std']}",
        f"- teacher_forced_mean_std: {payload['teacher_forced_mean_std']}",
        f"- semantic_hard_sidecar_score_mean_std: {payload['semantic_hard_sidecar_score_mean_std']}",
        f"- actual_gate_positive_ratio_mean_std: {payload['actual_gate_positive_ratio_mean_std']}",
        '',
        '| run_name | endpoint_l2 | hard_score | gate_ratio |',
        '|---|---:|---:|---:|',
    ]
    for row in combined:
        endpoint = base._metric_rank_tuple(row.get('best_checkpoint_metric', {}))[0]
        hard = base._f((row.get('semantic_hard_sidecar_metric', {}) if isinstance(row.get('semantic_hard_sidecar_metric', {}), dict) else {}).get('semantic_hard_sidecar_score'), 1e9)
        gate = float(row.get('actual_gate_positive_ratio_mean', 1.0))
        lines.append(f"| {row.get('run_name', '')} | {endpoint:.6f} | {hard:.6f} | {gate:.4f} |")
    base._write_md(args.final_pack_md, lines)
    return payload


def _write_final_pack_diagnosis(args: Any, final_summary: Dict[str, Any], ablation_payload: Dict[str, Any]) -> Dict[str, Any]:
    rows = final_summary.get('run_rows', []) if isinstance(final_summary.get('run_rows', []), list) else []
    crop_refs = _json_or_empty(args.stage2_semantic_value_diagnosis_report)
    crop_agg = crop_refs.get('full_validation_panel', {}).get('family_aggregates', {}).get('cropenc', {}) if isinstance(crop_refs.get('full_validation_panel', {}), dict) else {}
    crop_ep = base._f(crop_agg.get('free_rollout_endpoint_l2', {}).get('mean'), 1e9)
    crop_coord = base._f(crop_agg.get('free_rollout_coord_mean_l2', {}).get('mean'), 1e9)
    v7_diag = _v7_diag(args)
    v7_summary = _v7_summary(args)
    v7_best_name = str(v7_diag.get('overall_best_run_name', 'none'))
    v7_hard_name = str(v7_diag.get('semantic_hard_best_run_name', 'none'))
    v7_row_map = {str(r.get('run_name', '')): r for r in v7_summary.get('run_rows', []) if isinstance(r, dict)} if isinstance(v7_summary.get('run_rows', []), list) else {}
    v7_best_row = v7_row_map.get(v7_best_name, {})
    v7_hard_row = v7_row_map.get(v7_hard_name, {})
    v7_best_rank = base._metric_rank_tuple(v7_best_row.get('best_checkpoint_metric', {})) if v7_best_row else (1e9, 1e9, 1e9)
    v7_hard_score = base._f((v7_hard_row.get('semantic_hard_sidecar_metric', {}) if isinstance(v7_hard_row.get('semantic_hard_sidecar_metric', {}), dict) else {}).get('semantic_hard_sidecar_score'), 1e9)

    overall_best = min(rows, key=base._summary_overall_rank) if rows else {}
    hard_best = min(rows, key=base._summary_hard_rank) if rows else {}
    overall_rank = base._metric_rank_tuple(overall_best.get('best_checkpoint_metric', {})) if overall_best else (1e9, 1e9, 1e9)
    hard_score = base._f((hard_best.get('semantic_hard_sidecar_metric', {}) if isinstance(hard_best.get('semantic_hard_sidecar_metric', {}), dict) else {}).get('semantic_hard_sidecar_score'), 1e9) if hard_best else 1e9

    supportive_rows = [
        row for row in rows
        if base._metric_rank_tuple(row.get('best_checkpoint_metric', {}))[0] < crop_ep
        and float(row.get('actual_gate_positive_ratio_mean', 1.0)) < 0.30
    ]
    fresh_supportive = [row for row in supportive_rows if str(row.get('run_name', '')).endswith('_wave2_20260414')]
    six_seed_support_present = bool(len(rows) == 6 and len(supportive_rows) >= 4 and len(fresh_supportive) >= 2)
    improved_vs_cropenc = bool(_agg_payload(rows)['free_rollout_endpoint_l2']['mean'] < crop_ep and _agg_payload(rows)['free_rollout_coord_mean_l2']['mean'] <= crop_coord)
    improved_vs_v7 = bool(overall_rank < v7_best_rank)
    still_improved = bool(improved_vs_cropenc and improved_vs_v7)
    semantic_hard_signal_preserved = bool(hard_score <= (v7_hard_score * 1.02))
    mechanisms_load_bearing = bool(ablation_payload.get('alignment_sparse_gating_delay_load_bearing', False))
    final_mainline = bool(six_seed_support_present and improved_vs_cropenc and improved_vs_v7 and semantic_hard_signal_preserved and mechanisms_load_bearing)

    if final_mainline:
        next_step = 'freeze_stage2_calibration_only_mainline'
    elif six_seed_support_present and not mechanisms_load_bearing:
        next_step = 'redesign_calibration_family_small_fix'
    elif improved_vs_cropenc or improved_vs_v7:
        next_step = 'calibration_only_wave3_longrun'
    else:
        next_step = 'revisit_architecture_after_mainline_freeze'

    payload = {
        'generated_at_utc': now_iso(),
        'chosen_bootstrap_backend': BOOTSTRAP_BACKEND,
        '6_seed_support_present': bool(six_seed_support_present),
        'current_best_overall_run_name': str(overall_best.get('run_name', 'none')) if overall_best else 'none',
        'current_best_semantic_hard_run_name': str(hard_best.get('run_name', 'none')) if hard_best else 'none',
        'calibration_only_is_final_stage2_mainline': bool(final_mainline),
        'calibration_only_improved_vs_current_cropenc_baseline': bool(improved_vs_cropenc),
        'calibration_only_improved_vs_v7_alignment_only': bool(improved_vs_v7),
        'calibration_only_still_improved_after_adding_fresh_seeds': bool(still_improved),
        'mechanism_ablations_confirm_alignment_sparse_gating_delay_are_load_bearing': bool(mechanisms_load_bearing),
        'semantic_hard_signal_preserved': bool(semantic_hard_signal_preserved),
        'current_cropenc_baseline_anchor': {
            'source_report': str(args.stage2_semantic_value_diagnosis_report),
            'aggregate_mean_free_rollout_endpoint_l2': float(crop_ep),
            'aggregate_mean_free_rollout_coord_mean_l2': float(crop_coord),
        },
        'v7_alignment_only_anchor': {
            'source_report': str(args.v7_repaired_diagnosis_report),
            'run_name': v7_best_name,
            'best_checkpoint_metric': v7_best_row.get('best_checkpoint_metric', {}) if isinstance(v7_best_row, dict) else {},
            'semantic_hard_sidecar_metric': v7_hard_row.get('semantic_hard_sidecar_metric', {}) if isinstance(v7_hard_row, dict) else {},
        },
        'next_step_choice': next_step,
    }
    base._write_json(args.final_pack_diagnosis_report, payload)
    return payload


def _probe_targets(args: Any, wave2_summary: Dict[str, Any]) -> List[Dict[str, Any]]:
    v7_diag = _v7_diag(args)
    wave1_diag = _wave1_diag(args)
    wave2_rows = wave2_summary.get('run_rows', []) if isinstance(wave2_summary.get('run_rows', []), list) else []
    wave2_completed = _completed_valid(wave2_rows)
    wave2_best = min(wave2_completed, key=base._summary_overall_rank).get('run_name', 'none') if wave2_completed else 'none'
    crop_run = _best_family_run([
        'stage2_fullscale_core_cropenc_seed42_20260409',
        'stage2_fullscale_core_cropenc_seed123_20260409',
        'stage2_fullscale_core_cropenc_seed456_20260409',
    ])
    legacy_run = _best_family_run([
        'stage2_fullscale_core_legacysem_seed42_20260409',
        'stage2_fullscale_core_legacysem_seed123_wave2_20260409',
        'stage2_fullscale_core_legacysem_seed456_wave2_20260409',
    ])
    targets = [
        {'name': 'stage1_frozen_baseline', 'run_name': 'stage1_frozen_baseline', 'bridge_supported': False},
        {'name': 'legacysem_best', 'run_name': legacy_run, 'bridge_supported': True},
        {'name': 'cropenc_baseline_best', 'run_name': crop_run, 'bridge_supported': True},
        {'name': 'v7_alignment_only_best', 'run_name': str(v7_diag.get('overall_best_run_name', 'none')), 'bridge_supported': True},
        {'name': 'calibration_only_wave1_best', 'run_name': str(wave1_diag.get('overall_best_run_name', 'none')), 'bridge_supported': True},
    ]
    if wave2_best and wave2_best != 'none':
        targets.append({'name': 'calibration_only_wave2_best', 'run_name': str(wave2_best), 'bridge_supported': True})
    return targets


def _checkpoint_for_run(run_name: str) -> Path:
    ckpt_dir = WORK_ROOT / 'outputs' / 'checkpoints' / run_name
    best = ckpt_dir / 'best.pt'
    if best.exists():
        return best
    latest = ckpt_dir / 'latest.pt'
    return latest


def _run_aux_external_probe_batch(args: Any, wave2_summary: Dict[str, Any]) -> Dict[str, Any]:
    targets = _probe_targets(args, wave2_summary)
    out_dir = WORK_ROOT / 'reports' / 'stage2_aux_external_probe_batch_20260414_assets'
    out_dir.mkdir(parents=True, exist_ok=True)
    eval_gpu = _select_exclusive_gpu(
        run_name='stage2_aux_external_probe_batch_20260414',
        lease_path=str(args.shared_lease_path),
        required_mem_gb=40.0,
        reserve_idle_gpu_count=int(args.reserve_idle_gpu_count),
    )
    rows: List[Dict[str, Any]] = []
    lease_id = str(eval_gpu.get('lease_id', ''))
    gpu_id = int(eval_gpu.get('selected_gpu_id', -1))
    try:
        for target in targets:
            name = str(target['name'])
            run_name = str(target['run_name'])
            if not bool(target.get('bridge_supported', False)):
                rows.append({
                    'name': name,
                    'run_name': run_name,
                    'probe_status': 'unsupported',
                    'reason': 'current auxiliary bridge expects a Stage2 checkpoint payload with semantic modules; Stage1 frozen baseline is kept as internal anchor only',
                    'adapter_probe_only': True,
                    'paper_official_benchmark': False,
                })
                continue
            ckpt = _checkpoint_for_run(run_name)
            if not ckpt.exists():
                rows.append({
                    'name': name,
                    'run_name': run_name,
                    'probe_status': 'missing_checkpoint',
                    'checkpoint_path': str(ckpt),
                    'adapter_probe_only': True,
                    'paper_official_benchmark': False,
                })
                continue
            completion_json = out_dir / f'{name}_completion.json'
            results_md = out_dir / f'{name}_results.md'
            proxy_npz = out_dir / f'{name}_proxy.npz'
            proxy_latest_npz = out_dir / f'{name}_proxy_latest.npz'
            official_npz = out_dir / f'{name}_official.npz'
            official_latest_npz = out_dir / f'{name}_official_latest.npz'
            export_json = out_dir / f'{name}_export.json'
            export_latest_json = out_dir / f'{name}_export_latest.json'
            eval_json = out_dir / f'{name}_eval.json'
            eval_latest_json = out_dir / f'{name}_eval_latest.json'
            cmd = [
                str(args.python_bin),
                str(WORK_ROOT / 'code/stwm/tracewm_v2_stage2/tools/stage2_external_eval_bridge.py'),
                '--checkpoint-under-test', str(ckpt),
                '--secondary-checkpoint', str(ckpt),
                '--completion-json', str(completion_json),
                '--results-md', str(results_md),
                '--tap-style-proxy-payload-npz', str(proxy_npz),
                '--tap-style-secondary-proxy-payload-npz', str(proxy_latest_npz),
                '--tap-style-official-payload-npz', str(official_npz),
                '--tap-style-secondary-official-payload-npz', str(official_latest_npz),
                '--tap-style-export-report-json', str(export_json),
                '--tap-style-secondary-export-report-json', str(export_latest_json),
                '--tap-style-official-eval-json', str(eval_json),
                '--tap-style-secondary-official-eval-json', str(eval_latest_json),
                '--max-eval-batches', '4',
            ]
            env = os.environ.copy()
            env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
            env['STWM_PROC_TITLE'] = 'python:eval'
            env['STWM_PROC_TITLE_MODE'] = 'generic'
            env['PYTHONPATH'] = f"{args.work_root}/code:{env.get('PYTHONPATH', '')}"
            proc = subprocess.run(cmd, cwd=str(args.work_root), env=env, text=True, capture_output=True)
            if proc.returncode != 0 or not completion_json.exists():
                rows.append({
                    'name': name,
                    'run_name': run_name,
                    'probe_status': 'failed',
                    'checkpoint_path': str(ckpt),
                    'returncode': int(proc.returncode),
                    'stdout_tail': proc.stdout[-2000:],
                    'stderr_tail': proc.stderr[-2000:],
                    'adapter_probe_only': True,
                    'paper_official_benchmark': False,
                })
                continue
            payload = base._read_json(completion_json)
            primary = payload.get('primary_checkpoint_eval', {}) if isinstance(payload.get('primary_checkpoint_eval', {}), dict) else {}
            tap_style = primary.get('tap_style_eval', {}) if isinstance(primary.get('tap_style_eval', {}), dict) else {}
            official = tap_style.get('official_eval', {}) if isinstance(tap_style.get('official_eval', {}), dict) else {}
            metric_means = official.get('metric_means', {}) if isinstance(official.get('metric_means', {}), dict) else {}
            rows.append({
                'name': name,
                'run_name': run_name,
                'probe_status': 'completed',
                'checkpoint_path': str(ckpt),
                'tap_style_eval_status': str(payload.get('tap_style_eval_status', 'not_yet_implemented')),
                'official_evaluator_invoked': bool(payload.get('official_evaluator_invoked', False)),
                'official_task_faithfully_instantiated': bool(payload.get('official_task_faithfully_instantiated', False)),
                'adapter_probe_only': bool(payload.get('adapter_probe_only', True)),
                'paper_official_benchmark': bool(payload.get('paper_official_benchmark', False)),
                'average_jaccard': base._f(metric_means.get('average_jaccard'), 1e9),
                'average_pts_within_thresh': base._f(metric_means.get('average_pts_within_thresh'), 1e9),
                'occlusion_accuracy': base._f(metric_means.get('occlusion_accuracy'), 1e9),
                'results_md': str(results_md),
                'completion_json': str(completion_json),
            })
    finally:
        base._release_lease_safe(lease_id=lease_id, lease_path=str(args.shared_lease_path))

    completed_rows = [row for row in rows if str(row.get('probe_status', '')) == 'completed']
    cal_rows = [row for row in completed_rows if str(row.get('name', '')).startswith('calibration_only_')]
    best_cal = min(cal_rows, key=lambda r: (base._f(r.get('average_jaccard'), 1e9) * -1.0, base._f(r.get('average_pts_within_thresh'), 1e9) * -1.0, str(r.get('run_name', '')))) if cal_rows else {}
    crop_row = next((row for row in completed_rows if str(row.get('name', '')) == 'cropenc_baseline_best'), {})
    legacy_row = next((row for row in completed_rows if str(row.get('name', '')) == 'legacysem_best'), {})
    payload = {
        'generated_at_utc': now_iso(),
        'probe_scope': 'auxiliary adapter-based TAP-style probe only; not an official benchmark',
        'adapter_probe_only': True,
        'paper_official_benchmark': False,
        'rows': rows,
        'best_calibration_probe_target': str(best_cal.get('name', 'none')) if best_cal else 'none',
        'calibration_only_not_worse_than_cropenc_on_aux_probe': bool(best_cal and crop_row and base._f(best_cal.get('average_jaccard'), -1e9) >= base._f(crop_row.get('average_jaccard'), -1e9)),
        'calibration_only_not_worse_than_legacysem_on_aux_probe': bool(best_cal and legacy_row and base._f(best_cal.get('average_jaccard'), -1e9) >= base._f(legacy_row.get('average_jaccard'), -1e9)),
    }
    base._write_json(args.aux_external_probe_report, payload)
    base._write_md(
        args.aux_external_probe_md,
        [
            '# Stage2 Auxiliary External Probe Batch',
            '',
            '- scope: adapter-based TAP-style probe only; not official benchmark',
            f"- generated_at_utc: {payload['generated_at_utc']}",
            f"- best_calibration_probe_target: {payload['best_calibration_probe_target']}",
            f"- calibration_only_not_worse_than_cropenc_on_aux_probe: {payload['calibration_only_not_worse_than_cropenc_on_aux_probe']}",
            f"- calibration_only_not_worse_than_legacysem_on_aux_probe: {payload['calibration_only_not_worse_than_legacysem_on_aux_probe']}",
            '',
            '| name | run_name | probe_status | tap_style_eval_status | average_jaccard | avg_pts_within_thresh | adapter_probe_only |',
            '|---|---|---|---|---:|---:|---|',
            *[
                f"| {row.get('name', '')} | {row.get('run_name', '')} | {row.get('probe_status', '')} | {row.get('tap_style_eval_status', '')} | {base._f(row.get('average_jaccard'), 1e9):.6f} | {base._f(row.get('average_pts_within_thresh'), 1e9):.6f} | {row.get('adapter_probe_only', True)} |"
                for row in rows
            ],
        ],
    )
    return payload


def diagnose(args: Any) -> Dict[str, Any]:
    summary = summarize(args)
    ablation_payload = _json_or_empty(args.ablation_pack_report)
    rows = summary.get('run_rows', []) if isinstance(summary.get('run_rows', []), list) else []
    completed = _completed_valid(rows)
    crop_refs = _json_or_empty(args.stage2_semantic_value_diagnosis_report)
    crop_agg = crop_refs.get('full_validation_panel', {}).get('family_aggregates', {}).get('cropenc', {}) if isinstance(crop_refs.get('full_validation_panel', {}), dict) else {}
    crop_ep = base._f(crop_agg.get('free_rollout_endpoint_l2', {}).get('mean'), 1e9)
    crop_coord = base._f(crop_agg.get('free_rollout_coord_mean_l2', {}).get('mean'), 1e9)
    wave1_diag = _wave1_diag(args)
    wave1_best_name = str(wave1_diag.get('overall_best_run_name', 'none'))
    wave1_hard_name = str(wave1_diag.get('semantic_hard_best_run_name', 'none'))
    wave1_summary = _json_or_empty(args.wave1_summary_report)
    wave1_row_map = {str(r.get('run_name', '')): r for r in wave1_summary.get('run_rows', []) if isinstance(r, dict)} if isinstance(wave1_summary.get('run_rows', []), list) else {}
    wave1_best_row = wave1_row_map.get(wave1_best_name, {})
    wave1_hard_row = wave1_row_map.get(wave1_hard_name, {})
    wave1_best_rank = base._metric_rank_tuple(wave1_best_row.get('best_checkpoint_metric', {})) if wave1_best_row else (1e9, 1e9, 1e9)
    wave1_hard_score = base._f((wave1_hard_row.get('semantic_hard_sidecar_metric', {}) if isinstance(wave1_hard_row.get('semantic_hard_sidecar_metric', {}), dict) else {}).get('semantic_hard_sidecar_score'), 1e9)

    pending = not bool(summary.get('all_runs_terminal', False)) or not bool(ablation_payload.get('all_runs_terminal', False)) or int(summary.get('failed_count', 0)) > 0 or int(ablation_payload.get('failed_count', 0)) > 0
    if pending:
        payload = {
            'generated_at_utc': now_iso(),
            'status': 'pending_mainline_completion',
            'chosen_bootstrap_backend': BOOTSTRAP_BACKEND,
            'overall_best_run_name': str(summary.get('overall_best_run_name', 'none')),
            'wave2_support_present': None,
            'calibration_only_improved_vs_current_cropenc_baseline': None,
            'calibration_only_improved_vs_wave1_best': None,
            'true_new_best_not_warm_start_inherited': None,
            'semantic_hard_signal_preserved': None,
            'next_step_choice': 'pending_mainline_completion',
        }
        base._write_json(args.diagnosis_report, payload)
        _write_wave2_results_md(args, summary, payload)
        final_summary = _write_final_pack_summary(args, summary)
        _write_final_pack_diagnosis(args, final_summary, ablation_payload)
        return payload

    overall_best = min(completed, key=base._summary_overall_rank)
    hard_best = min(completed, key=base._summary_hard_rank)
    overall_rank = base._metric_rank_tuple(overall_best.get('best_checkpoint_metric', {}))
    hard_score = base._f((hard_best.get('semantic_hard_sidecar_metric', {}) if isinstance(hard_best.get('semantic_hard_sidecar_metric', {}), dict) else {}).get('semantic_hard_sidecar_score'), 1e9)
    warm_step = base._load_ckpt_step(_resume_ckpt_for_spec(overall_best))
    best_step = base._int_or_default((overall_best.get('best_checkpoint_metric', {}) if isinstance(overall_best.get('best_checkpoint_metric', {}), dict) else {}).get('global_step', -1), -1)
    true_new_best = bool(best_step > warm_step >= 0)
    improved_vs_cropenc = bool(overall_rank[0] < crop_ep and overall_rank[1] <= crop_coord)
    improved_vs_wave1 = bool(overall_rank < wave1_best_rank)
    semantic_hard_signal_preserved = bool(hard_score <= (wave1_hard_score * 1.02))
    supportive_rows = [row for row in completed if base._metric_rank_tuple(row.get('best_checkpoint_metric', {}))[0] < crop_ep and float(row.get('actual_gate_positive_ratio_mean', 1.0)) < 0.30]
    wave2_support_present = bool(len({int(row.get('seed', -1)) for row in supportive_rows}) >= 2)
    payload = {
        'generated_at_utc': now_iso(),
        'status': 'completed',
        'chosen_bootstrap_backend': BOOTSTRAP_BACKEND,
        'overall_best_run_name': str(overall_best.get('run_name', 'none')),
        'semantic_hard_best_run_name': str(hard_best.get('run_name', 'none')),
        'wave2_support_present': bool(wave2_support_present),
        'calibration_only_improved_vs_current_cropenc_baseline': bool(improved_vs_cropenc),
        'calibration_only_improved_vs_wave1_best': bool(improved_vs_wave1),
        'true_new_best_not_warm_start_inherited': bool(true_new_best),
        'semantic_hard_signal_preserved': bool(semantic_hard_signal_preserved),
        'next_step_choice': 'ready_for_final_pack_aggregation',
    }
    base._write_json(args.diagnosis_report, payload)
    _write_wave2_results_md(args, summary, payload)
    final_summary = _write_final_pack_summary(args, summary)
    _write_final_pack_diagnosis(args, final_summary, ablation_payload)
    return payload


def run_all(args: Any) -> Dict[str, Any]:
    write_process_title_report(args)
    launch(args)
    summary = wait_for_completion(args)
    ablation = _json_or_empty(args.ablation_pack_report)
    if bool(summary.get('all_runs_terminal', False)) and int(summary.get('failed_count', 0)) == 0 and bool(ablation.get('all_runs_terminal', False)) and int(ablation.get('failed_count', 0)) == 0:
        _run_aux_external_probe_batch(args, summary)
        qual_cmd = [
            str(args.python_bin),
            str(WORK_ROOT / 'code/stwm/tools/run_tracewm_stage1_stage2_qualitative_pack_v5_20260414.py'),
            '--wave2-summary-report', str(args.summary_report),
            '--wave2-diagnosis-report', str(args.diagnosis_report),
            '--ablation-pack-report', str(args.ablation_pack_report),
            '--final-pack-diagnosis-report', str(args.final_pack_diagnosis_report),
        ]
        subprocess.run(qual_cmd, cwd=str(args.work_root), check=False)
    summary = summarize(args)
    diagnosis = diagnose(args)
    final_summary = _json_or_empty(args.final_pack_summary_report)
    final_diagnosis = _json_or_empty(args.final_pack_diagnosis_report)
    return {
        'summary': summary,
        'diagnosis': diagnosis,
        'ablation_pack': _json_or_empty(args.ablation_pack_report),
        'final_pack_summary': final_summary,
        'final_pack_diagnosis': final_diagnosis,
        'aux_external_probe': _json_or_empty(args.aux_external_probe_report),
    }


def parse_args() -> Any:
    p = ArgumentParser(description='Stage2 calibration-only finalization pack launcher / worker')
    p.add_argument('--mode', default='all', choices=['all', 'launch', 'run-one', 'summarize', 'diagnose'])
    p.add_argument('--meta-json', default='')
    p.add_argument('--work-root', default=str(WORK_ROOT))
    p.add_argument('--python-bin', default=base._python_bin_default())
    p.add_argument('--tmux-session', default=SESSION)
    p.add_argument('--stage2-contract-json', default=str(WORK_ROOT / 'reports/stage2_bootstrap_data_contract_20260408.json'))
    p.add_argument('--stage1-best-ckpt', default=str(WORK_ROOT / 'outputs/checkpoints/stage1_v2_longtrain_220m_mainline_20260408/best.pt'))
    p.add_argument('--shared-lease-path', default=str(WORK_ROOT / 'reports/stage1_v2_gpu_lease_20260408.json'))
    p.add_argument('--bootstrap-cache-jsonl', default=str(WORK_ROOT / 'data/processed/stage2_real_bootstrap_cache_20260410/clip_vit_b32_core_trainval_required_subset.jsonl'))
    p.add_argument('--semantic-hard-manifest-path', default=str(WORK_ROOT / 'manifests/protocol_v2/stage2_semantic_hard_subsets_20260410.json'))
    p.add_argument('--stage2-semantic-value-diagnosis-report', default=str(WORK_ROOT / 'reports/stage2_semantic_value_diagnosis_20260410.json'))
    p.add_argument('--v7-repaired-summary-report', default=str(WORK_ROOT / 'reports/stage2_semantic_objective_redesign_v7_summary_20260413.json'))
    p.add_argument('--v7-repaired-diagnosis-report', default=str(WORK_ROOT / 'reports/stage2_semantic_objective_redesign_v7_diagnosis_20260413.json'))
    p.add_argument('--wave1-summary-report', default=str(WORK_ROOT / 'reports/stage2_calibration_only_fullscale_wave1_summary_20260413.json'))
    p.add_argument('--wave1-diagnosis-report', default=str(WORK_ROOT / 'reports/stage2_calibration_only_fullscale_wave1_diagnosis_20260413.json'))
    p.add_argument('--process-title-report', default=str(WORK_ROOT / 'reports/stage2_process_title_normalization_20260414.json'))
    p.add_argument('--concurrent-runtime-report', default=str(WORK_ROOT / 'reports/stage2_calibration_only_wave2_runtime_20260414.json'))
    p.add_argument('--launch-report', default=str(WORK_ROOT / 'reports/stage2_calibration_only_wave2_launch_20260414.json'))
    p.add_argument('--summary-report', default=str(WORK_ROOT / 'reports/stage2_calibration_only_wave2_summary_20260414.json'))
    p.add_argument('--diagnosis-report', default=str(WORK_ROOT / 'reports/stage2_calibration_only_wave2_diagnosis_20260414.json'))
    p.add_argument('--results-md', default=str(WORK_ROOT / 'docs/STAGE2_CALIBRATION_ONLY_WAVE2_RESULTS_20260414.md'))
    p.add_argument('--ablation-pack-report', default=str(WORK_ROOT / 'reports/stage2_calibration_only_ablation_pack_20260414.json'))
    p.add_argument('--ablation-pack-md', default=str(WORK_ROOT / 'docs/STAGE2_CALIBRATION_ONLY_ABLATION_PACK_20260414.md'))
    p.add_argument('--final-pack-summary-report', default=str(WORK_ROOT / 'reports/stage2_calibration_only_final_pack_summary_20260414.json'))
    p.add_argument('--final-pack-diagnosis-report', default=str(WORK_ROOT / 'reports/stage2_calibration_only_final_pack_diagnosis_20260414.json'))
    p.add_argument('--final-pack-md', default=str(WORK_ROOT / 'docs/STAGE2_CALIBRATION_ONLY_FINAL_PACK_RESULTS_20260414.md'))
    p.add_argument('--aux-external-probe-report', default=str(WORK_ROOT / 'reports/stage2_aux_external_probe_batch_20260414.json'))
    p.add_argument('--aux-external-probe-md', default=str(WORK_ROOT / 'docs/STAGE2_AUX_EXTERNAL_PROBE_BATCH_20260414.md'))
    p.add_argument('--reserve-idle-gpu-count', type=int, default=2)
    p.add_argument('--gpu-acquire-timeout-seconds', type=int, default=28800)
    p.add_argument('--gpu-acquire-retry-seconds', type=int, default=20)
    p.add_argument('--wait-timeout-seconds', type=int, default=172800)
    p.add_argument('--poll-seconds', type=int, default=120)
    return p.parse_args()



def run_one(args: Any) -> None:
    meta = base._read_json(args.meta_json)
    if int(meta.get('selected_gpu_id', -1)) < 0:
        deadline = time.time() + float(meta.get('gpu_acquire_timeout_seconds', 28800))
        retry_seconds = float(meta.get('gpu_acquire_retry_seconds', 20))
        last_error = ''
        while time.time() < deadline:
            try:
                gpu = _select_exclusive_gpu(
                    run_name=str(meta.get('run_name', '')),
                    lease_path=str(meta.get('shared_lease_path', '')),
                    required_mem_gb=float(GPU_MIN_FREE_MEM_GB),
                    reserve_idle_gpu_count=int(meta.get('reserve_idle_gpu_count', 2)),
                )
                meta['selected_gpu_id'] = int(gpu['selected_gpu_id'])
                meta['lease_id'] = str(gpu['lease_id'])
                meta['selector_payload'] = gpu.get('selector_payload', {})
                base._write_json(args.meta_json, meta)
                break
            except Exception as exc:
                last_error = str(exc)
                time.sleep(retry_seconds)
        if int(meta.get('selected_gpu_id', -1)) < 0:
            final_json = Path(str(meta.get('final_json', '')))
            payload = {
                'generated_at_utc': now_iso(),
                'run_name': str(meta.get('run_name', '')),
                'status': 'failed',
                'message': f'gpu_acquire_timeout_exclusive_selector last_error={last_error}',
            }
            base._write_json(final_json, payload)
            raise RuntimeError(payload['message'])
    base.run_one(args)


def main() -> None:
    _apply_process_title_normalization(default_title='python')
    args = parse_args()
    if args.mode == 'all':
        print(json.dumps(run_all(args), ensure_ascii=True, indent=2))
    elif args.mode == 'launch':
        print(json.dumps(launch(args), ensure_ascii=True, indent=2))
    elif args.mode == 'run-one':
        run_one(args)
    elif args.mode == 'summarize':
        print(json.dumps(summarize(args), ensure_ascii=True, indent=2))
    elif args.mode == 'diagnose':
        print(json.dumps(diagnose(args), ensure_ascii=True, indent=2))


if __name__ == '__main__':
    main()
