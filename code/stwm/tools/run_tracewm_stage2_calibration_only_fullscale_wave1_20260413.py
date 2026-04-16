#!/usr/bin/env python3
from __future__ import annotations

from argparse import ArgumentParser
from collections import defaultdict
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
import traceback

from stwm.tools.run_tracewm_stage2_ljs_semantic_diagnosis_and_rescue_20260410 import (
    _f,
    _mean_std,
)
from stwm.tools.run_tracewm_stage2_semantic_objective_redesign_v1_20260410 import (
    _dataset_counts,
    _load_ckpt_args,
    _load_ckpt_step,
    _read_json,
    _resume_ckpt_for_seed,
    _write_json,
    _write_md,
)
from stwm.tools.run_tracewm_stage2_semantic_objective_redesign_v2_20260410 import (
    _python_bin_default,
    _release_lease_safe,
    _select_gpu,
    _tmux_windows,
)
from stwm.tools.run_tracewm_stage2_semantic_objective_redesign_v7_20260413 import _gpu_headroom_ok
from stwm.infra.gpu_lease import acquire_lease, list_active_leases
from stwm.infra.gpu_telemetry import snapshot_gpu_telemetry


WORK_ROOT = Path('/home/chen034/workspace/stwm')
SESSION = 'tracewm_stage2_calibration_only_fullscale_wave1_20260413'
LOG_PATH = WORK_ROOT / 'logs/stage2_calibration_only_fullscale_wave1_20260413.log'
DATE_TAG = '20260413'
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
GPU_ALLOC_SAMPLE_COUNT = 2
GPU_ALLOC_INTERVAL_SEC = 0.25
GPU_MIN_FREE_MEM_GB = 30.0


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _apply_process_title_normalization(default_title: str = 'python') -> None:
    mode = str(os.environ.get('STWM_PROC_TITLE_MODE', 'generic')).strip().lower()
    if mode != 'generic':
        return
    title = str(os.environ.get('STWM_PROC_TITLE', default_title)).strip() or default_title
    lowered = title.lower()
    if 'stwm' in lowered or 'tracewm' in lowered or '/home/' in lowered:
        title = default_title
    try:
        import setproctitle  # type: ignore
        setproctitle.setproctitle(title)
    except Exception:
        pass


def _json_or_empty(path_like: Any) -> Dict[str, Any]:
    path = Path(str(path_like))
    if not str(path_like) or not path.exists():
        return {}
    try:
        payload = _read_json(path)
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _metric_payload(block: Any) -> Dict[str, Any]:
    if not isinstance(block, dict):
        return {}
    payload = block.get('metrics', {})
    return payload if isinstance(payload, dict) else {}


def _int_or_default(value: Any, default: int) -> int:
    try:
        if value is None:
            return int(default)
        return int(value)
    except Exception:
        return int(default)


def _metric_rank_tuple(block: Any) -> Tuple[float, float, float]:
    metrics = _metric_payload(block)
    return (
        _f(metrics.get('free_rollout_endpoint_l2'), 1e9),
        _f(metrics.get('free_rollout_coord_mean_l2'), 1e9),
        _f(metrics.get('teacher_forced_coord_loss'), 1e9),
    )


def _summary_overall_rank(row: Dict[str, Any]) -> Tuple[float, float, float, str]:
    return (*_metric_rank_tuple(row.get('best_checkpoint_metric', {})), str(row.get('run_name', '')))


def _summary_hard_rank(row: Dict[str, Any]) -> Tuple[float, float, float, float, str]:
    sidecar = row.get('semantic_hard_sidecar_metric', {}) if isinstance(row.get('semantic_hard_sidecar_metric', {}), dict) else {}
    return (
        _f(sidecar.get('semantic_hard_sidecar_score'), 1e9),
        *_metric_rank_tuple(row.get('best_checkpoint_metric', {})),
        str(row.get('run_name', '')),
    )


def _scientific_artifact_valid(
    resolved_status: str,
    best_ckpt_exists: bool,
    latest_ckpt_exists: bool,
    raw_json_exists: bool,
) -> bool:
    return bool(
        str(resolved_status).lower() == 'completed'
        and bool(best_ckpt_exists)
        and bool(latest_ckpt_exists)
        and bool(raw_json_exists)
    )


def _run_specs() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for family, gating_family, topk, quantile, cap_ratio in [
        ('topk1', 'hard_topk_query_gating_v2', 1, 0.85, 0.15),
        ('qcap15', 'capped_quantile_sparse_gating_v2', 1, 0.80, 0.15),
    ]:
        for seed in [42, 123, 456]:
            rows.append(
                {
                    'run_name': f'stage2_calonly_{family}_seed{seed}_wave1_{DATE_TAG}',
                    'family': family,
                    'seed': seed,
                    'objective_combo': f'calibration_only_{family}_alignment_fullscale_wave1',
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
                    'v6_gating_family': gating_family,
                    'v6_topk_query_k': topk,
                    'v6_capped_quantile': quantile,
                    'v6_max_affected_ratio': cap_ratio,
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
                    'window_name': f'cal_{family}_s{seed}',
                }
            )
    return rows


def _launch_meta_dir(args: Any) -> Path:
    return Path(args.work_root) / 'reports' / 'stage2_calibration_only_fullscale_wave1_runs_20260413'


def _launch_meta_by_run(args: Any) -> Dict[str, Dict[str, Any]]:
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
            pid = _int_or_default(pid_file.read_text(encoding='utf-8').strip(), -1)
            if _pid_alive(pid):
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
    script_path = Path(args.work_root) / 'code/stwm/tools/run_tracewm_stage2_calibration_only_fullscale_wave1_20260413.py'
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
        f'while kill -0 \"$(cat {shlex.quote(pid_path)})\" 2>/dev/null; do sleep 30; done'
    )
    return (
        "bash -lc "
        + shlex.quote(
            f"cd {shlex.quote(str(args.work_root))}; "
            f"rm -f {shlex.quote(pid_path)}; "
            f"{cmd}; "
            f"printf '[%s] tmux_window_exit run_name={run_name} observed_child_exit\\n' \"$(date -Iseconds)\" >> {shlex.quote(log_path)}"
        )
    )


def _status_for(meta: Dict[str, Any], session_name: str) -> Dict[str, Any]:
    final_path = Path(str(meta.get('final_json', '')))
    progress_path = Path(str(meta.get('progress_json', '')))
    if str(meta.get('window_name', '')) in _tmux_windows(session_name):
        detail = _json_or_empty(progress_path)
        return {'status': 'running', 'detail': detail}
    if final_path.exists():
        detail = _json_or_empty(final_path)
        status = str(detail.get('status', 'launched')).lower()
        if status in {'completed', 'failed'}:
            return {'status': status, 'detail': detail}
    if _run_activity_alive(meta):
        detail = _json_or_empty(progress_path)
        return {'status': 'running', 'detail': detail}
    detail = _json_or_empty(progress_path)
    if detail:
        progress_status = str(detail.get('status', 'launched')).lower()
        if progress_status in {'initialized', 'launched', 'running'}:
            return {'status': 'failed', 'detail': detail, 'salvage_reason': 'run_activity_missing_without_final'}
        return {'status': progress_status, 'detail': detail}
    return {'status': 'failed', 'detail': {}, 'salvage_reason': 'no_window_no_final_no_progress'}


def _best_block(final_payload: Dict[str, Any], raw_payload: Dict[str, Any], progress_payload: Dict[str, Any]) -> Dict[str, Any]:
    for payload in [final_payload, raw_payload, progress_payload]:
        block = payload.get('best_checkpoint_metric', {})
        if isinstance(block, dict) and block and isinstance(block.get('metrics', {}), dict):
            return block
        block = payload.get('best_metric_so_far', {})
        if isinstance(block, dict) and block and isinstance(block.get('metrics', {}), dict):
            return block
    return {}


def _latest_block(final_payload: Dict[str, Any], raw_payload: Dict[str, Any], progress_payload: Dict[str, Any]) -> Dict[str, Any]:
    for payload in [final_payload, raw_payload, progress_payload]:
        block = payload.get('latest_checkpoint_metric', {})
        if isinstance(block, dict) and block and isinstance(block.get('metrics', {}), dict):
            return block
        block = payload.get('latest_eval_metrics', {})
        if isinstance(block, dict) and block:
            if isinstance(block.get('metrics', {}), dict):
                return block
            if any(k in block for k in ['free_rollout_endpoint_l2', 'free_rollout_coord_mean_l2', 'teacher_forced_coord_loss']):
                return {
                    'global_step': _int_or_default(payload.get('global_step', -1), -1),
                    'metrics': block,
                }
    return {}


def _sidecar_block(final_payload: Dict[str, Any], raw_payload: Dict[str, Any], progress_payload: Dict[str, Any]) -> Dict[str, Any]:
    for payload in [final_payload, raw_payload, progress_payload]:
        block = payload.get('semantic_hard_sidecar_metric', {})
        if isinstance(block, dict) and block:
            return block
        block = payload.get('best_semantic_hard_metric', {})
        if isinstance(block, dict) and block:
            return block
    return {}


def _branch_block(final_payload: Dict[str, Any], raw_payload: Dict[str, Any], progress_payload: Dict[str, Any]) -> Dict[str, Any]:
    for payload in [final_payload, raw_payload]:
        block = payload.get('semantic_branch_metrics', {})
        if isinstance(block, dict) and block:
            return block
    progress_best = progress_payload.get('best_metric_so_far', {})
    if isinstance(progress_best, dict):
        metrics = progress_best.get('metrics', {})
        if isinstance(metrics, dict):
            branch = metrics.get('semantic_branch_metrics', {})
            if isinstance(branch, dict) and branch:
                return branch
    latest_metrics = progress_payload.get('latest_eval_metrics', {})
    if isinstance(latest_metrics, dict):
        branch = latest_metrics.get('semantic_branch_metrics', {})
        if isinstance(branch, dict) and branch:
            return branch
    return {}


def _gpu_selection_from_payload(final_payload: Dict[str, Any], progress_payload: Dict[str, Any], meta: Dict[str, Any]) -> Tuple[int, str]:
    final_gpu = _int_or_default(final_payload.get('selected_gpu_id', meta.get('selected_gpu_id', -1)), -1)
    final_lease = str(final_payload.get('lease_id', meta.get('lease_id', '')) or '')
    if final_gpu >= 0 or final_lease:
        return final_gpu, final_lease
    run_meta = progress_payload.get('run_metadata', {}) if isinstance(progress_payload.get('run_metadata', {}), dict) else {}
    gpu_sel = run_meta.get('gpu_selection', {}) if isinstance(run_meta.get('gpu_selection', {}), dict) else {}
    gpu_id = _int_or_default(gpu_sel.get('selected_gpu_id', meta.get('selected_gpu_id', -1)), -1)
    lease_id = str(gpu_sel.get('lease_id', meta.get('lease_id', '')) or '')
    return gpu_id, lease_id



def _pid_alive(pid: int) -> bool:
    if int(pid) <= 0:
        return False
    try:
        os.kill(int(pid), 0)
        return True
    except Exception:
        return False


def _process_args_lines() -> List[str]:
    proc = subprocess.run(['ps', '-eo', 'args='], text=True, capture_output=True)
    if proc.returncode != 0:
        return []
    return [str(line).strip() for line in proc.stdout.splitlines() if str(line).strip()]


def _run_activity_alive(meta: Dict[str, Any]) -> bool:
    run_name = str(meta.get('run_name', '')).strip()
    meta_json = str(meta.get('meta_json', '')).strip()
    pid_file = str(meta.get('worker_pid_file', '')).strip()
    if pid_file:
        try:
            pid_path = Path(pid_file)
            if pid_path.exists():
                pid = _int_or_default(pid_path.read_text(encoding='utf-8').strip(), -1)
                if _pid_alive(pid):
                    return True
        except Exception:
            pass
    if not run_name and not meta_json:
        return False
    for line in _process_args_lines():
        if 'rg ' in line:
            continue
        if meta_json and f'--mode run-one --meta-json {meta_json}' in line:
            return True
        if run_name and f'--run-name {run_name}' in line:
            return True
    return False


def _aggregate_gpu_window(samples: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    by_gpu: Dict[int, Dict[str, Any]] = defaultdict(
        lambda: {
            'name': '',
            'uuid': '',
            'gpu_util': [],
            'mem_util': [],
            'free_mem_gb': [],
            'used_mem_gb': [],
            'active_compute_process_count': [],
        }
    )
    for snap in samples:
        for row in snap.get('gpus', []) if isinstance(snap.get('gpus', []), list) else []:
            gpu_id = int(row.get('gpu_id', -1))
            if gpu_id < 0:
                continue
            acc = by_gpu[gpu_id]
            acc['name'] = str(row.get('name', ''))
            acc['uuid'] = str(row.get('uuid', ''))
            acc['gpu_util'].append(float(row.get('gpu_util', 0.0)))
            acc['mem_util'].append(float(row.get('mem_util', 0.0)))
            acc['free_mem_gb'].append(float(row.get('free_mem_gb', 0.0)))
            acc['used_mem_gb'].append(float(row.get('memory_used_gb', 0.0)))
            acc['active_compute_process_count'].append(float(row.get('active_compute_process_count', 0.0)))
    out: Dict[int, Dict[str, Any]] = {}
    for gpu_id, acc in by_gpu.items():
        def _avg(xs: List[float]) -> float:
            return float(sum(xs) / max(len(xs), 1))
        out[gpu_id] = {
            'gpu_id': int(gpu_id),
            'name': str(acc.get('name', '')),
            'uuid': str(acc.get('uuid', '')),
            'avg_gpu_util': _avg(list(acc['gpu_util'])),
            'avg_mem_util': _avg(list(acc['mem_util'])),
            'avg_free_mem_gb': _avg(list(acc['free_mem_gb'])),
            'avg_used_mem_gb': _avg(list(acc['used_mem_gb'])),
            'active_compute_process_count': int(round(_avg(list(acc['active_compute_process_count'])))),
            'sample_count': int(len(acc['gpu_util'])),
        }
    return out


def _select_clean_gpu_for_calibration(run_name: str, lease_path: str, required_mem_gb: float = GPU_MIN_FREE_MEM_GB, safety_margin_gb: float = 0.0) -> Dict[str, Any]:
    samples: List[Dict[str, Any]] = []
    for i in range(int(GPU_ALLOC_SAMPLE_COUNT)):
        samples.append(snapshot_gpu_telemetry(prefer_nvml=True))
        if i + 1 < int(GPU_ALLOC_SAMPLE_COUNT):
            time.sleep(float(GPU_ALLOC_INTERVAL_SEC))
    aggregated = _aggregate_gpu_window(samples)
    active_leases = list_active_leases(lease_path=lease_path)
    lease_count_by_gpu: Dict[int, int] = {}
    for lease in active_leases:
        gpu_id = int(lease.get('gpu_id', -1))
        if gpu_id < 0:
            continue
        lease_count_by_gpu[gpu_id] = int(lease_count_by_gpu.get(gpu_id, 0)) + 1
    rows: List[Dict[str, Any]] = []
    candidates: List[Dict[str, Any]] = []
    for gpu_id in sorted(aggregated.keys()):
        row = dict(aggregated[gpu_id])
        lease_count = int(lease_count_by_gpu.get(gpu_id, 0))
        enough_mem = bool(float(row.get('avg_free_mem_gb', 0.0)) >= float(required_mem_gb))
        if not enough_mem:
            reason = 'filtered_insufficient_free_mem'
        else:
            reason = 'threshold_candidate'
            candidates.append(row)
        row['leased'] = bool(lease_count > 0)
        row['lease_count'] = int(lease_count)
        row['enough_mem'] = bool(enough_mem)
        row['cleanish'] = None
        row['selected'] = False
        row['selected_reason'] = str(reason)
        row['required_mem_gb'] = float(required_mem_gb)
        row['safety_margin_gb'] = float(safety_margin_gb)
        rows.append(row)
    candidates = sorted(
        candidates,
        key=lambda x: (
            int(lease_count_by_gpu.get(int(x.get('gpu_id', -1)), 0)),
            -float(x.get('avg_free_mem_gb', 0.0)),
            float(x.get('avg_used_mem_gb', 0.0)),
            float(x.get('avg_gpu_util', 0.0)),
        ),
    )
    if not candidates:
        raise RuntimeError(
            f'no_threshold_gpu_candidate run={run_name} '
            f'(need_free_mem>={float(required_mem_gb):.1f}GB)'
        )
    selected_gpu_id = int(candidates[0]['gpu_id'])
    for row in rows:
        if int(row.get('gpu_id', -1)) == selected_gpu_id:
            row['selected'] = True
            row['selected_reason'] = 'best_rank_after_threshold_filter'
        elif row.get('selected_reason') == 'threshold_candidate':
            row['selected_reason'] = 'threshold_candidate_not_top_rank'
    lease = acquire_lease(
        gpu_id=selected_gpu_id,
        owner=str(run_name),
        ttl_seconds=12 * 3600,
        lease_path=str(lease_path),
        allow_shared=True,
    )
    return {
        'selected_gpu_id': int(selected_gpu_id),
        'lease_id': str(lease.get('lease_id', '')),
        'selector_payload': {
            'generated_at_utc': now_iso(),
            'mode': 'threshold_based_greedy_shared_gpu_selector_for_calibration_wave1',
            'required_mem_gb': float(required_mem_gb),
            'safety_margin_gb': float(safety_margin_gb),
            'policy': {
                'min_free_mem_gb': float(required_mem_gb),
                'allow_shared_gpu': True,
                'greedy_rank': ['lease_count asc', 'avg_free_mem_gb desc', 'avg_used_mem_gb asc', 'avg_gpu_util asc'],
            },
            'sample_count': int(GPU_ALLOC_SAMPLE_COUNT),
            'sample_interval_sec': float(GPU_ALLOC_INTERVAL_SEC),
            'gpus': rows,
            'candidate_ranking': [int(x.get('gpu_id', -1)) for x in candidates],
        },
    }


def _cleanup_stale_leases(lease_path: str, allowed_prefixes: Tuple[str, ...]) -> Dict[str, Any]:
    payload = _json_or_empty(lease_path)
    leases = payload.get('leases', []) if isinstance(payload.get('leases', []), list) else []
    kept = []
    removed = []
    for lease in leases:
        if not isinstance(lease, dict):
            continue
        owner = str(lease.get('owner', ''))
        pid = _int_or_default(lease.get('pid', -1), -1)
        if owner.startswith(allowed_prefixes) and not _pid_alive(pid):
            removed.append({'lease_id': str(lease.get('lease_id', '')), 'owner': owner, 'pid': pid, 'gpu_id': _int_or_default(lease.get('gpu_id', -1), -1)})
            continue
        kept.append(lease)
    out = {'leases': kept}
    _write_json(lease_path, out)
    return {'removed_count': len(removed), 'removed': removed, 'kept_count': len(kept)}


def write_concurrent_runtime_report(args: Any) -> Dict[str, Any]:
    base = _json_or_empty(args.stage1_runtime_json)
    selected_gpu = (((base.get('selected_gpu_policy', {}) if isinstance(base.get('selected_gpu_policy', {}), dict) else {}).get('selected_gpu_id', -1)))
    payload = {
        'generated_at_utc': now_iso(),
        'source': 'stage2_calibration_only_fullscale_wave1_concurrent_runtime',
        'based_on': str(args.stage1_runtime_json),
        'selected_gpu_policy': {
            'mode': 'threshold_shared_gpu',
            'selection_rule': ['avg_free_mem_gb highest', 'avg_used_mem_gb lowest', 'avg_gpu_util lowest', 'lease_count lowest'],
            'window': {'sample_count': int(GPU_ALLOC_SAMPLE_COUNT), 'sample_interval_sec': float(GPU_ALLOC_INTERVAL_SEC)},
            'memory_filter': {'required_mem_gb': float(GPU_MIN_FREE_MEM_GB), 'safety_margin_gb': 0.0},
            'selected_gpu_id': _int_or_default(selected_gpu, -1),
        },
        'required_mem_gb': float(GPU_MIN_FREE_MEM_GB),
        'safety_margin_gb': 0.0,
        'recommended_num_workers': int(CONCURRENT_RUNTIME_NUM_WORKERS),
        'recommended_pin_memory': bool(CONCURRENT_RUNTIME_PIN_MEMORY),
        'recommended_persistent_workers': bool(CONCURRENT_RUNTIME_PERSISTENT_WORKERS),
        'recommended_prefetch_factor': int(CONCURRENT_RUNTIME_PREFETCH_FACTOR),
        'single_gpu_only': True,
        'notes': [
            'wave1 threshold-based shared-GPU runtime override',
            'selector no longer waits for perfectly clean GPUs on 192GB B200 cards',
            'runs may share a GPU when free memory stays above threshold',
        ],
    }
    _write_json(args.concurrent_runtime_report, payload)
    return payload


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
            'code/stwm/tools/run_tracewm_stage2_calibration_only_fullscale_wave1_20260413.py',
            'code/stwm/tools/run_tracewm_stage2_partial_unfreeze_ablation_20260413.py',
            'code/stwm/tools/run_tracewm_stage1_stage2_qualitative_pack_v4_20260413.py',
        ],
        'runtime_policy': 'logs, reports, checkpoints keep full run_name; process title stays generic only',
    }
    _write_json(args.process_title_report, payload)
    return payload


def write_protocol_doc(args: Any) -> None:
    _write_md(
        args.results_md.replace('RESULTS', 'PROTOCOL'),
        [
            '# Stage2 Calibration-Only Fullscale Wave1 Protocol',
            '',
            f'- generated_at_utc: {now_iso()}',
            '- stage1_mutation_allowed: false in mainline calibration-only wave1',
            '- main_task: future trace / future state generation',
            '- teacher_as_mainline_semantic_source: false',
            '- calibration_only_definition: readout-side semantic alignment + sparse gating + delayed aux schedule + semantic-hard sidecar',
            '- persistence_mainline_allowed: false',
            '- calibration_families: topk1 / qcap15',
            '- fullscale_policy: full VSPW+VIPSeg train/val, no sample caps, no DDP retrofit, threshold-based shared-GPU greedy allocation',
        f'- concurrent_runtime_override: workers={CONCURRENT_RUNTIME_NUM_WORKERS}, pin_memory={CONCURRENT_RUNTIME_PIN_MEMORY}, persistent_workers={CONCURRENT_RUNTIME_PERSISTENT_WORKERS}, prefetch={CONCURRENT_RUNTIME_PREFETCH_FACTOR}',
            '- partial_unfreeze_branch: gated secondary ablation only after all 6 calibration runs complete',
            '- forbidden: Stage1 retraining; codec/VAE wave0; batch/lr sweep; external-eval expansion; persistence-as-mainline narrative',
        ],
    )


def launch(args: Any) -> Dict[str, Any]:
    lease_cleanup = _cleanup_stale_leases(str(args.shared_lease_path), allowed_prefixes=('stage2_calonly_', 'stage2_partialunfreeze_'))
    concurrent_runtime = write_concurrent_runtime_report(args)
    if subprocess.run(['tmux', 'has-session', '-t', str(args.tmux_session)], capture_output=True).returncode != 0:
        subprocess.run(['tmux', 'new-session', '-d', '-s', str(args.tmux_session), 'bash'], check=True)

    anchor_args = _load_ckpt_args(_resume_ckpt_for_seed(42))
    obs_len = int(anchor_args.get('obs_len', 8) or 8)
    fut_len = int(anchor_args.get('fut_len', 8) or 8)
    max_tokens = int(anchor_args.get('max_tokens', 64) or 64)
    crop_size = int(anchor_args.get('semantic_crop_size', 64) or 64)
    train_counts = _dataset_counts(['vspw', 'vipseg'], 'train', args.stage2_contract_json, max_samples=FULL_MAX_TRAIN_PER_DATASET)
    val_counts = _dataset_counts(['vspw', 'vipseg'], 'val', args.stage2_contract_json, max_samples=FULL_MAX_VAL_PER_DATASET)

    runs: List[Dict[str, Any]] = []
    cleanup_actions: List[Dict[str, Any]] = []
    meta_dir = _launch_meta_dir(args)
    meta_dir.mkdir(parents=True, exist_ok=True)
    existing_windows = set(_tmux_windows(str(args.tmux_session)))
    for spec in _run_specs():
        run_name = str(spec['run_name'])
        resume_from = _resume_ckpt_for_seed(int(spec['seed']))
        resume_step = _load_ckpt_step(resume_from)
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
        gpu = _select_clean_gpu_for_calibration(
            run_name=run_name,
            lease_path=str(args.shared_lease_path),
            required_mem_gb=float(GPU_MIN_FREE_MEM_GB),
            safety_margin_gb=0.0,
        )
        meta['selected_gpu_id'] = int(gpu['selected_gpu_id'])
        meta['lease_id'] = str(gpu['lease_id'])
        meta['selector_payload'] = gpu.get('selector_payload', {})
        _write_json(meta_json, meta)
        runs.append(meta)
        cmd = _tmux_window_command(args=args, meta_json=meta_json, meta=meta)
        if str(meta['window_name']) not in existing_windows:
            subprocess.run(['tmux', 'new-window', '-t', str(args.tmux_session), '-n', str(meta['window_name']), cmd], check=True)
            existing_windows.add(str(meta['window_name']))

    payload = {
        'generated_at_utc': now_iso(),
        'mode': 'stage2_calibration_only_fullscale_wave1_launch',
        'tmux_session': str(args.tmux_session),
        'teacher_backend': 'local_clip_vit_b32_mask_crop_visual_teacher',
        'policy': 'calibration-only mainline; persistence disabled; threshold-based shared-GPU greedy allocation; Stage1 remains frozen in mainline',
        'lease_cleanup': lease_cleanup,
        'cleanup_actions': cleanup_actions,
        'concurrent_runtime_report': str(args.concurrent_runtime_report),
        'concurrent_runtime': concurrent_runtime,
        'runs': runs,
    }
    _write_json(args.launch_report, payload)
    return summarize(args)


def run_one(args: Any) -> None:
    meta = _read_json(args.meta_json)
    lease_id = str(meta.get('lease_id', ''))
    lease_path = str(meta.get('shared_lease_path', ''))
    run_name = str(meta.get('run_name', ''))
    selected_gpu_id = int(meta.get('selected_gpu_id', -1))
    log_path = Path(str(meta.get('log_path', '')))
    if str(log_path):
        log_path.parent.mkdir(parents=True, exist_ok=True)
    final_json_path = Path(str(meta.get('final_json', '')))
    abort_written = False
    trainer_proc: subprocess.Popen[str] | None = None

    def _append_log(message: str) -> None:
        if not str(log_path):
            return
        with log_path.open('a', encoding='utf-8') as log_fh:
            log_fh.write(f"[{now_iso()}] {message}\n")
            log_fh.flush()

    def _write_abort_payload(message: str, *, signal_name: str = '', returncode: int | None = None) -> None:
        nonlocal abort_written
        if abort_written:
            return
        abort_written = True
        payload: Dict[str, Any] = {
            'generated_at_utc': now_iso(),
            'run_name': str(run_name),
            'status': 'failed',
            'selected_gpu_id': int(selected_gpu_id),
            'lease_id': str(lease_id),
            'message': str(message),
        }
        if signal_name:
            payload['signal_name'] = str(signal_name)
        if returncode is not None:
            payload['returncode'] = int(returncode)
        try:
            _write_json(final_json_path, payload)
        except Exception:
            pass

    def _signal_handler(signum: int, _frame: Any) -> None:
        sig_name = ''
        try:
            sig_name = signal.Signals(signum).name
        except Exception:
            sig_name = str(signum)
        _append_log(f'run_one_received_signal run_name={run_name} signal={sig_name}')
        nonlocal trainer_proc
        if trainer_proc is not None and trainer_proc.poll() is None:
            try:
                os.killpg(int(trainer_proc.pid), signal.SIGTERM)
                _append_log(
                    f'run_one_forwarded_signal_to_trainer_pg run_name={run_name} '
                    f'trainer_pid={int(trainer_proc.pid)} signal=SIGTERM'
                )
            except Exception as exc:
                _append_log(
                    f'run_one_failed_to_forward_signal run_name={run_name} '
                    f'trainer_pid={int(trainer_proc.pid)} error={exc!r}'
                )
        _write_abort_payload(f'run_one_terminated_by_signal_{sig_name}', signal_name=sig_name, returncode=128 + int(signum))
        raise SystemExit(128 + int(signum))

    # `run-one` is launched under `nohup`; keep SIGHUP ignored so session/window churn
    # does not abort the worker and its trainer child.
    try:
        signal.signal(signal.SIGHUP, signal.SIG_IGN)
        _append_log(f'run_one_ignoring_signal run_name={run_name} signal=SIGHUP')
    except Exception:
        pass

    for sig in [signal.SIGTERM, signal.SIGINT]:
        try:
            signal.signal(sig, _signal_handler)
        except Exception:
            pass

    if selected_gpu_id < 0:
        gpu = _select_clean_gpu_for_calibration(
            run_name=run_name,
            lease_path=lease_path,
            required_mem_gb=float(GPU_MIN_FREE_MEM_GB),
            safety_margin_gb=0.0,
        )
        selected_gpu_id = int(gpu['selected_gpu_id'])
        lease_id = str(gpu['lease_id'])
        meta['selected_gpu_id'] = int(selected_gpu_id)
        meta['lease_id'] = str(lease_id)
        meta['selector_payload'] = gpu.get('selector_payload', {})
        _write_json(args.meta_json, meta)
        _append_log(f"gpu_acquired_threshold run_name={run_name} selected_gpu_id={selected_gpu_id} lease_id={lease_id}")
    trainer = Path(str(meta['work_root'])) / 'code/stwm/tracewm_v2_stage2/trainers/train_tracewm_stage2_smalltrain.py'
    cmd = [
        str(meta['python_bin']), str(trainer),
        '--stage2-contract-path', str(meta['stage2_contract_json']),
        '--recommended-runtime-json', str(meta['stage1_runtime_json']),
        '--use-recommended-runtime',
        '--stage1-backbone-checkpoint', str(meta['stage1_best_ckpt']),
        '--dataset-names', 'vspw', 'vipseg',
        '--train-split', 'train',
        '--val-split', 'val',
        '--obs-len', str(meta['obs_len']),
        '--fut-len', str(meta['fut_len']),
        '--max-tokens', str(meta['max_tokens']),
        '--max-samples-train', str(meta['max_samples_train']),
        '--max-samples-val', str(meta['max_samples_val']),
        '--batch-size', str(meta['batch_size']),
        '--train-steps', str(meta['train_steps']),
        '--eval-interval', str(meta['eval_interval']),
        '--eval-max-batches', str(meta['eval_max_batches']),
        '--save-every-n-steps', str(meta['save_every_n_steps']),
        '--semantic-source-mainline', str(meta['semantic_source_mainline']),
        '--legacy-semantic-source', str(meta['legacy_semantic_source']),
        '--semantic-crop-size', str(meta['semantic_crop_size']),
        '--semantic-rescue-mode', str(meta['semantic_rescue_mode']),
        '--semantic-rescue-weight', str(meta['semantic_rescue_weight']),
        '--semantic-bootstrap-cache-path', str(meta['bootstrap_cache_jsonl']),
        '--semantic-bootstrap-target-dim', str(meta['semantic_bootstrap_target_dim']),
        '--semantic-hard-curriculum-weight', str(meta['semantic_hard_curriculum_weight']),
        '--semantic-aux-subset-weighting-strength', str(meta['semantic_aux_subset_weighting_strength']),
        '--confidence-gated-alignment-loss-weight', str(meta['confidence_gated_alignment_loss_weight']),
        '--sparse-persistence-contrastive-loss-weight', str(meta['sparse_persistence_contrastive_loss_weight']),
        '--confidence-gating-margin-threshold', str(meta['confidence_gating_margin_threshold']),
        '--confidence-gating-temperature', str(meta['confidence_gating_temperature']),
        '--semantic-hard-score-threshold', str(meta['semantic_hard_score_threshold']),
        '--aux-loss-delay-steps', str(meta['aux_loss_delay_steps']),
        '--aux-loss-ramp-steps', str(meta['aux_loss_ramp_steps']),
        '--v6-gating-family', str(meta['v6_gating_family']),
        '--v6-topk-query-k', str(meta['v6_topk_query_k']),
        '--v6-capped-quantile', str(meta['v6_capped_quantile']),
        '--v6-max-affected-ratio', str(meta['v6_max_affected_ratio']),
        '--v6-gate-min-strength', str(meta['v6_gate_min_strength']),
        '--v6-strict-max-pairs-per-sample', str(meta['v6_strict_max_pairs_per_sample']),
        '--v6-hard-negative-cap', str(meta['v6_hard_negative_cap']),
        '--v6-pair-sampling-temperature', str(meta['v6_pair_sampling_temperature']),
        '--v6-guaranteed-min-pairs-per-sample', str(meta['v6_guaranteed_min_pairs_per_sample']),
        '--v6-relaxed-motion-threshold', str(meta['v6_relaxed_motion_threshold']),
        '--v6-relaxed-area-jump-threshold', str(meta['v6_relaxed_area_jump_threshold']),
        '--v6-relaxed-small-query-threshold', str(meta['v6_relaxed_small_query_threshold']),
        '--v6-relaxed-appearance-shift-threshold', str(meta['v6_relaxed_appearance_shift_threshold']),
        '--v6-relaxed-center-interaction-threshold', str(meta['v6_relaxed_center_interaction_threshold']),
        '--semantic-hard-manifest-path', str(meta['semantic_hard_manifest_path']),
        '--resume-from', str(meta['resume_from']),
        '--skip-resume-optimizer',
        '--semantic-hard-sidecar-enabled',
        '--output-dir', str(meta['output_dir']),
        '--run-name', str(meta['run_name']),
        '--run-summary-json', str(meta['raw_json']),
        '--progress-json', str(meta['progress_json']),
        '--seed', str(meta['seed']),
    ]
    predecode_cache_path = str(meta.get('predecode_cache_path', '') or '')
    if predecode_cache_path:
        cmd.extend(['--predecode-cache-path', predecode_cache_path])
    if 'local_temporal_window' in meta:
        cmd.extend(['--local-temporal-window', str(meta['local_temporal_window'])])
    if 'local_temporal_fuse_weight' in meta:
        cmd.extend(['--local-temporal-fuse-weight', str(meta['local_temporal_fuse_weight'])])
    cmd.append('--v6-two-level-pair-mining-enabled' if bool(meta['v6_two_level_pair_mining_enabled']) else '--no-v6-two-level-pair-mining-enabled')
    try:
        proc_env = os.environ.copy()
        proc_env['CUDA_VISIBLE_DEVICES'] = str(selected_gpu_id)
        proc_env['STWM_PROC_TITLE'] = str(proc_env.get('STWM_PROC_TITLE', 'python'))
        proc_env['STWM_PROC_TITLE_MODE'] = str(proc_env.get('STWM_PROC_TITLE_MODE', 'generic'))
        proc_env['PYTHONUNBUFFERED'] = '1'
        proc_env['TRACEWM_STAGE1_V2_GPU_SELECTION_METADATA_JSON'] = json.dumps(
            {
                'selected_gpu_id': int(selected_gpu_id),
                'lease_id': str(lease_id),
                'owner': str(meta.get('run_name', '')),
                'mode': 'single_gpu_only',
            },
            ensure_ascii=True,
        )
        with log_path.open('w', encoding='utf-8') as log_fh:
            log_fh.write(f"[run-one] run_name={run_name} selected_gpu_id={selected_gpu_id} lease_id={lease_id}\n")
            if isinstance(meta.get('selector_payload', {}), dict):
                log_fh.write(json.dumps(meta['selector_payload'], ensure_ascii=True) + "\n")
            log_fh.write(
                f"[run-one] detached_trainer_policy run_name={run_name} "
                f"sighup=ignored trainer_start_new_session=true\n"
            )
            log_fh.flush()
            trainer_proc = subprocess.Popen(
                [cmd[0], '-u', *cmd[1:]],
                cwd=str(meta['work_root']),
                text=True,
                stdout=log_fh,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                env=proc_env,
                start_new_session=True,
            )
            _append_log(
                f'run_one_spawned_trainer run_name={run_name} '
                f'trainer_pid={int(trainer_proc.pid)} start_new_session=true'
            )
            returncode = trainer_proc.wait()
        trainer_proc = None
        proc = subprocess.CompletedProcess(args=[cmd[0], '-u', *cmd[1:]], returncode=returncode)
        if proc.returncode != 0:
            log_tail = log_path.read_text(encoding='utf-8', errors='ignore')[-4000:] if log_path.exists() else ''
            _write_abort_payload(f'trainer_failed_rc_{proc.returncode}', returncode=int(proc.returncode))
            final_payload = _json_or_empty(final_json_path)
            final_payload['log_tail'] = log_tail
            _write_json(final_json_path, final_payload)
            raise RuntimeError(f'trainer failed rc={proc.returncode}')
        raw = _read_json(meta['raw_json'])
        raw.update(
            {
                'generated_at_utc': now_iso(),
                'status': 'completed',
                'selected_gpu_id': int(selected_gpu_id),
                'lease_id': str(lease_id),
                'objective_combo': str(meta['objective_combo']),
                'objective_family': str(meta['objective_family']),
                'persistence_objective_declared': False,
                'persistence_objective_effective': False,
                'resume_global_step': int(meta['resume_global_step']),
                'teacher_backend': 'local_clip_vit_b32_mask_crop_visual_teacher',
            }
        )
        _write_json(meta['final_json'], raw)
        abort_written = True
    except Exception as exc:
        _write_abort_payload(str(exc))
        final_payload = _json_or_empty(final_json_path)
        final_payload['traceback'] = traceback.format_exc()
        _write_json(final_json_path, final_payload)
        raise
    finally:
        _release_lease_safe(lease_id=lease_id, lease_path=lease_path)


def summarize(args: Any) -> Dict[str, Any]:
    launch = _json_or_empty(args.launch_report)
    meta_by_run = _launch_meta_by_run(args)
    run_rows: List[Dict[str, Any]] = []
    running = completed = failed = 0
    for spec in _run_specs():
        run_name = str(spec['run_name'])
        meta = meta_by_run.get(run_name, {})
        paths = _paths_for(args, meta, run_name)
        progress_payload = _json_or_empty(paths['progress'])
        final_payload = _json_or_empty(paths['final'])
        raw_payload = _json_or_empty(paths['raw'])
        status_info = _status_for({**meta, 'window_name': str(meta.get('window_name', spec.get('window_name', ''))), 'progress_json': str(paths['progress']), 'final_json': str(paths['final'])}, session_name=str(args.tmux_session))
        resolved_status = str(status_info.get('status', 'launched')).lower()
        if resolved_status == 'failed' and not paths['final'].exists():
            _write_json(
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
        running += int(resolved_status == 'running')
        completed += int(resolved_status == 'completed')
        failed += int(resolved_status == 'failed')
        best_ckpt_exists = bool(paths['best'].exists())
        latest_ckpt_exists = bool(paths['latest'].exists())
        sidecar_exists = bool(paths['sidecar'].exists())
        raw_json_exists = bool(paths['raw'].exists())
        scientific_result_valid = _scientific_artifact_valid(
            resolved_status=resolved_status,
            best_ckpt_exists=best_ckpt_exists,
            latest_ckpt_exists=latest_ckpt_exists,
            raw_json_exists=raw_json_exists,
        )
        best_block = _best_block(final_payload, raw_payload, progress_payload)
        latest_block = _latest_block(final_payload, raw_payload, progress_payload)
        sidecar_block = _sidecar_block(final_payload, raw_payload, progress_payload)
        branch = _branch_block(final_payload, raw_payload, progress_payload)
        if not scientific_result_valid:
            best_block = {}
            latest_block = {}
            sidecar_block = {}
            branch = {}
        sidecar_sel = raw_payload.get('sidecar_checkpoint_selection', {}) if isinstance(raw_payload.get('sidecar_checkpoint_selection', {}), dict) else final_payload.get('sidecar_checkpoint_selection', {}) if isinstance(final_payload.get('sidecar_checkpoint_selection', {}), dict) else {}
        if not sidecar_sel and isinstance(progress_payload.get('sidecar_checkpoint_selection', {}), dict):
            sidecar_sel = progress_payload.get('sidecar_checkpoint_selection', {})
        global_step = _int_or_default(progress_payload.get('global_step', best_block.get('global_step', -1)), -1)
        selected_gpu_id, lease_id = _gpu_selection_from_payload(final_payload, progress_payload, meta)
        run_rows.append(
            {
                'run_name': run_name,
                'family': str(spec['family']),
                'seed': int(spec['seed']),
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
                'actual_gate_positive_ratio_mean': (
                    float(branch.get('actual_gate_positive_ratio_mean', branch.get('eval_gate_mean', 1.0)))
                    if scientific_result_valid and isinstance(branch, dict) and branch
                    else None
                ),
                'raw_quantile_ratio_mean': (
                    float(branch.get('raw_quantile_ratio_mean', 0.0))
                    if scientific_result_valid and isinstance(branch, dict) and branch
                    else None
                ),
                'capped_ratio_mean': (
                    float(branch.get('capped_ratio_mean', 0.0))
                    if scientific_result_valid and isinstance(branch, dict) and branch
                    else None
                ),
                'valuable_pair_ratio_mean': (
                    float(branch.get('valuable_pair_ratio_mean', branch.get('high_value_pair_ratio', 0.0)))
                    if scientific_result_valid and isinstance(branch, dict) and branch
                    else None
                ),
                'same_checkpoint_selected': bool(sidecar_sel.get('same_checkpoint_selected', True)),
                'sidecar_truly_diverged': bool(sidecar_sel.get('sidecar_truly_diverged', False)),
                'persistence_objective_declared': False,
                'persistence_objective_effective': False,
                'artifact_truth_note': (
                    'valid_completed_run'
                    if scientific_result_valid
                    else 'failed_or_incomplete_run_metrics_suppressed_to_avoid_warm_start_or_progress_residue_misread'
                ),
            }
        )
    completed_rows = [row for row in run_rows if row['status'] == 'completed']
    overall_best_run_name = 'none'
    semantic_hard_best_run_name = 'none'
    best_family = 'none'
    if completed_rows:
        overall_best = min(completed_rows, key=_summary_overall_rank)
        hard_best = min(completed_rows, key=_summary_hard_rank)
        overall_best_run_name = str(overall_best['run_name'])
        semantic_hard_best_run_name = str(hard_best['run_name'])
        best_family = str(overall_best['family'])

    family_aggregates: Dict[str, Any] = {}
    for family in ['topk1', 'qcap15']:
        family_rows = [row for row in completed_rows if str(row.get('family', '')) == family]
        family_aggregates[family] = {
            'count': len(family_rows),
            'free_rollout_endpoint_l2': _mean_std([_metric_rank_tuple(row.get('best_checkpoint_metric', {}))[0] for row in family_rows]),
            'free_rollout_coord_mean_l2': _mean_std([_metric_rank_tuple(row.get('best_checkpoint_metric', {}))[1] for row in family_rows]),
            'teacher_forced_coord_loss': _mean_std([_metric_rank_tuple(row.get('best_checkpoint_metric', {}))[2] for row in family_rows]),
            'semantic_hard_sidecar_score': _mean_std([_f((row.get('semantic_hard_sidecar_metric', {}) if isinstance(row.get('semantic_hard_sidecar_metric', {}), dict) else {}).get('semantic_hard_sidecar_score'), 1e9) for row in family_rows]),
            'actual_gate_positive_ratio_mean': _mean_std([float(row.get('actual_gate_positive_ratio_mean', 1.0)) for row in family_rows]),
        }

    payload = {
        'generated_at_utc': now_iso(),
        'calibration_only_wave1_status': f"{running}_running_{completed}_completed_{failed}_failed",
        'running_count': running,
        'completed_count': completed,
        'failed_count': failed,
        'all_runs_terminal': bool(len(run_rows) > 0 and running == 0 and completed + failed == len(run_rows)),
        'run_rows': run_rows,
        'runs': run_rows,
        'overall_best_run_name': overall_best_run_name,
        'semantic_hard_best_run_name': semantic_hard_best_run_name,
        'best_family': best_family,
        'family_aggregates': family_aggregates,
        'teacher_backend': 'local_clip_vit_b32_mask_crop_visual_teacher',
        'next_step_choice_internal': '',
    }
    payload['next_step_choice_internal'] = (
        'ready_for_calibration_only_diagnosis'
        if payload['all_runs_terminal'] and payload['failed_count'] == 0 and payload['completed_count'] == len(run_rows)
        else ('fix_failed_runs' if payload['failed_count'] > 0 else 'continue_running')
    )
    _write_json(args.summary_report, payload)
    return payload


def wait_for_completion(args: Any) -> Dict[str, Any]:
    deadline = time.time() + float(args.wait_timeout_seconds)
    last = summarize(args)
    while time.time() < deadline:
        if bool(last.get('all_runs_terminal', False)):
            return last
        time.sleep(float(args.poll_seconds))
        last = summarize(args)
    last['timed_out_waiting_for_completion'] = True
    _write_json(args.summary_report, last)
    return last


def _v7_row_map(summary_payload: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {str(row.get('run_name', '')): row for row in summary_payload.get('run_rows', []) if isinstance(row, dict)}


def diagnose(args: Any) -> Dict[str, Any]:
    summary = summarize(args)
    refs = _read_json(args.stage2_semantic_value_diagnosis_report)
    v7_diag = _json_or_empty(args.v7_repaired_diagnosis_report)
    v7_summary = _json_or_empty(args.v7_repaired_summary_report)
    partial = _json_or_empty(args.partial_unfreeze_report)

    rows = [row for row in summary.get('run_rows', []) if isinstance(row, dict)]
    completed = [row for row in rows if str(row.get('status', '')).lower() == 'completed']
    row_map = {str(row.get('run_name', '')): row for row in rows}

    cropenc_mean = refs.get('full_validation_panel', {}).get('family_aggregates', {}).get('cropenc', {}) if isinstance(refs.get('full_validation_panel', {}), dict) else {}
    crop_ep = _f(cropenc_mean.get('free_rollout_endpoint_l2', {}).get('mean'), 1e9)
    crop_coord = _f(cropenc_mean.get('free_rollout_coord_mean_l2', {}).get('mean'), 1e9)

    v7_best_name = str(v7_diag.get('overall_best_run_name', 'none'))
    v7_hard_name = str(v7_diag.get('semantic_hard_best_run_name', 'none'))
    v7_rows = _v7_row_map(v7_summary)
    v7_best_row = v7_rows.get(v7_best_name, {})
    v7_hard_row = v7_rows.get(v7_hard_name, {})
    v7_best_rank = _metric_rank_tuple(v7_best_row.get('best_checkpoint_metric', {})) if v7_best_row else (1e9, 1e9, 1e9)
    v7_hard_score = _f((v7_hard_row.get('semantic_hard_sidecar_metric', {}) if isinstance(v7_hard_row.get('semantic_hard_sidecar_metric', {}), dict) else {}).get('semantic_hard_sidecar_score'), 1e9)

    pending = not bool(summary.get('all_runs_terminal', False)) or int(summary.get('failed_count', 0)) > 0 or not completed
    if pending:
        payload = {
            'generated_at_utc': now_iso(),
            'status': 'pending_mainline_completion',
            'overall_best_run_name': str(summary.get('overall_best_run_name', 'none')),
            'semantic_hard_best_run_name': str(summary.get('semantic_hard_best_run_name', 'none')),
            'best_family': str(summary.get('best_family', 'none')),
            'stage2_calibration_only_is_true_mainline': None,
            'calibration_only_improved_vs_current_cropenc_baseline': None,
            'calibration_only_improved_vs_v7_alignment_only': None,
            'true_new_best_not_warm_start_inherited': None,
            'cross_seed_support_present': None,
            'semantic_hard_signal_preserved': None,
            'overall_best_and_semantic_hard_best_diverged': None,
            'partial_unfreeze_beats_frozen_calibration': None,
            'forgetting_or_instability_detected': None,
            'next_step_choice': 'pending_mainline_completion',
        }
        _write_json(args.diagnosis_report, payload)
        write_results_md(args, summary, payload)
        return payload

    overall_best = min(completed, key=_summary_overall_rank)
    hard_best = min(completed, key=_summary_hard_rank)
    overall_best_run_name = str(overall_best['run_name'])
    semantic_hard_best_run_name = str(hard_best['run_name'])
    best_family = str(overall_best['family'])
    overall_rank = _metric_rank_tuple(overall_best.get('best_checkpoint_metric', {}))
    hard_score = _f((hard_best.get('semantic_hard_sidecar_metric', {}) if isinstance(hard_best.get('semantic_hard_sidecar_metric', {}), dict) else {}).get('semantic_hard_sidecar_score'), 1e9)

    warm_step = _load_ckpt_step(_resume_ckpt_for_seed(int(overall_best.get('seed', 42))))
    best_step = _int_or_default((overall_best.get('best_checkpoint_metric', {}) if isinstance(overall_best.get('best_checkpoint_metric', {}), dict) else {}).get('global_step', -1), -1)
    true_new_best = bool(best_step > warm_step >= 0)
    improved_vs_cropenc = bool(overall_rank[0] < crop_ep and overall_rank[1] <= crop_coord)
    improved_vs_v7 = bool(overall_rank < v7_best_rank)
    semantic_hard_signal_preserved = bool(hard_score <= (v7_hard_score * 1.02))
    diverged = bool(overall_best_run_name != semantic_hard_best_run_name or bool(overall_best.get('sidecar_truly_diverged', False)))

    family_rows = [row for row in completed if str(row.get('family', '')) == best_family]
    supportive_rows = [
        row for row in family_rows
        if _metric_rank_tuple(row.get('best_checkpoint_metric', {}))[0] < crop_ep and float(row.get('actual_gate_positive_ratio_mean', 1.0)) < 0.30
    ]
    cross_seed_support = bool(len({int(row.get('seed', -1)) for row in supportive_rows if int(row.get('seed', -1)) >= 0}) >= 2)

    partial_ran = bool(partial)
    partial_beats = None
    forgetting_detected = None
    if partial_ran:
        partial_beats = bool(partial.get('partial_unfreeze_beats_frozen_calibration', False))
        forgetting_detected = bool(partial.get('forgetting_or_instability_detected', False))

    true_mainline = bool(
        improved_vs_cropenc
        and improved_vs_v7
        and true_new_best
        and cross_seed_support
        and semantic_hard_signal_preserved
    )
    if true_mainline:
        next_step = 'stage2_calibration_only_is_true_mainline'
    elif bool(partial_beats) and not bool(forgetting_detected):
        next_step = 'stage2_partial_unfreeze_branch_is_promising'
    elif improved_vs_cropenc or semantic_hard_signal_preserved:
        next_step = 'stage2_calibration_only_continue_wave2'
    else:
        next_step = 'redesign_stage2_semantic_objective_v8'

    payload = {
        'generated_at_utc': now_iso(),
        'status': 'completed',
        'chosen_bootstrap_backend': 'local_clip_vit_b32_mask_crop_visual_teacher',
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
        'overall_best_run_name': overall_best_run_name,
        'semantic_hard_best_run_name': semantic_hard_best_run_name,
        'best_family': best_family,
        'stage2_calibration_only_is_true_mainline': bool(true_mainline),
        'calibration_only_improved_vs_current_cropenc_baseline': bool(improved_vs_cropenc),
        'calibration_only_improved_vs_v7_alignment_only': bool(improved_vs_v7),
        'true_new_best_not_warm_start_inherited': bool(true_new_best),
        'cross_seed_support_present': bool(cross_seed_support),
        'semantic_hard_signal_preserved': bool(semantic_hard_signal_preserved),
        'overall_best_and_semantic_hard_best_diverged': bool(diverged),
        'partial_unfreeze_beats_frozen_calibration': partial_beats,
        'forgetting_or_instability_detected': forgetting_detected,
        'next_step_choice': next_step,
        'success_criteria': {
            'stage2_calibration_only_is_true_mainline': bool(true_mainline),
            'calibration_only_improved_vs_current_cropenc_baseline': bool(improved_vs_cropenc),
            'calibration_only_improved_vs_v7_alignment_only': bool(improved_vs_v7),
            'true_new_best_not_warm_start_inherited': bool(true_new_best),
            'cross_seed_support_present': bool(cross_seed_support),
            'semantic_hard_signal_preserved': bool(semantic_hard_signal_preserved),
            'overall_best_and_semantic_hard_best_diverged': bool(diverged),
            'partial_unfreeze_beats_frozen_calibration': partial_beats,
            'forgetting_or_instability_detected': forgetting_detected,
            'next_step_choice': next_step,
        },
    }
    _write_json(args.diagnosis_report, payload)
    write_results_md(args, summary, payload)
    return payload


def write_results_md(args: Any, summary: Dict[str, Any], diagnosis: Dict[str, Any]) -> None:
    lines = [
        '# Stage2 Calibration-Only Fullscale Wave1 Results',
        '',
        f"- generated_at_utc: {now_iso()}",
        f"- calibration_only_wave1_status: {summary.get('calibration_only_wave1_status', 'unknown')}",
        '- failed/incomplete runs: metrics suppressed when no valid completed raw+checkpoint artifact exists',
        f"- overall_best_run_name: {diagnosis.get('overall_best_run_name', 'none')}",
        f"- semantic_hard_best_run_name: {diagnosis.get('semantic_hard_best_run_name', 'none')}",
        f"- best_family: {diagnosis.get('best_family', summary.get('best_family', 'none'))}",
        f"- stage2_calibration_only_is_true_mainline: {diagnosis.get('stage2_calibration_only_is_true_mainline', None)}",
        f"- calibration_only_improved_vs_current_cropenc_baseline: {diagnosis.get('calibration_only_improved_vs_current_cropenc_baseline', None)}",
        f"- calibration_only_improved_vs_v7_alignment_only: {diagnosis.get('calibration_only_improved_vs_v7_alignment_only', None)}",
        f"- true_new_best_not_warm_start_inherited: {diagnosis.get('true_new_best_not_warm_start_inherited', None)}",
        f"- cross_seed_support_present: {diagnosis.get('cross_seed_support_present', None)}",
        f"- semantic_hard_signal_preserved: {diagnosis.get('semantic_hard_signal_preserved', None)}",
        f"- overall_best_and_semantic_hard_best_diverged: {diagnosis.get('overall_best_and_semantic_hard_best_diverged', None)}",
        f"- partial_unfreeze_beats_frozen_calibration: {diagnosis.get('partial_unfreeze_beats_frozen_calibration', None)}",
        f"- forgetting_or_instability_detected: {diagnosis.get('forgetting_or_instability_detected', None)}",
        f"- next_step_choice: {diagnosis.get('next_step_choice', 'none')}",
        '',
        '| run_name | family | seed | status | global_step | endpoint_l2 | hard_score | gate_ratio | sidecar_diverged |',
        '|---|---|---:|---|---:|---:|---:|---:|---|',
    ]
    for row in summary.get('run_rows', []) if isinstance(summary.get('run_rows', []), list) else []:
        if not isinstance(row, dict):
            continue
        scientific_valid = bool(row.get('scientific_result_valid', False))
        endpoint = _metric_rank_tuple(row.get('best_checkpoint_metric', {}))[0] if scientific_valid else None
        hard_score = (
            _f((row.get('semantic_hard_sidecar_metric', {}) if isinstance(row.get('semantic_hard_sidecar_metric', {}), dict) else {}).get('semantic_hard_sidecar_score'), 1e9)
            if scientific_valid else None
        )
        gate_ratio = row.get('actual_gate_positive_ratio_mean', None) if scientific_valid else None
        lines.append(
            f"| {row.get('run_name', '')} | {row.get('family', '')} | {int(row.get('seed', -1))} | {row.get('status', '')} | {int(row.get('global_step', -1))} | "
            f"{(f'{endpoint:.6f}' if endpoint is not None else 'n/a')} | "
            f"{(f'{hard_score:.6f}' if hard_score is not None else 'n/a')} | "
            f"{(f'{float(gate_ratio):.4f}' if gate_ratio is not None else 'n/a')} | {bool(row.get('sidecar_truly_diverged', False))} |"
        )
    _write_md(args.results_md, lines)


def run_all(args: Any) -> Dict[str, Any]:
    write_process_title_report(args)
    write_protocol_doc(args)
    launch(args)
    summary = wait_for_completion(args)
    if bool(summary.get('all_runs_terminal', False)) and int(summary.get('failed_count', 0)) == 0:
        partial_cmd = [
            str(args.python_bin),
            str(Path(args.work_root) / 'code/stwm/tools/run_tracewm_stage2_partial_unfreeze_ablation_20260413.py'),
            '--mode', 'run',
        ]
        subprocess.run(partial_cmd, cwd=str(args.work_root), check=False)
        qual_cmd = [
            str(args.python_bin),
            str(Path(args.work_root) / 'code/stwm/tools/run_tracewm_stage1_stage2_qualitative_pack_v4_20260413.py'),
        ]
        subprocess.run(qual_cmd, cwd=str(args.work_root), check=False)
    summary = summarize(args)
    diagnosis = diagnose(args)
    return {'summary': summary, 'diagnosis': diagnosis}


def parse_args() -> Any:
    p = ArgumentParser(description='Stage2 calibration-only fullscale wave1 launcher / worker')
    p.add_argument('--mode', default='all', choices=['all', 'launch', 'run-one', 'summarize', 'diagnose'])
    p.add_argument('--meta-json', default='')
    p.add_argument('--work-root', default=str(WORK_ROOT))
    p.add_argument('--python-bin', default=_python_bin_default())
    p.add_argument('--tmux-session', default=SESSION)
    p.add_argument('--stage2-contract-json', default=str(WORK_ROOT / 'reports/stage2_bootstrap_data_contract_20260408.json'))
    p.add_argument('--stage1-runtime-json', default=str(WORK_ROOT / 'reports/stage1_v2_recommended_runtime_20260408.json'))
    p.add_argument('--stage1-best-ckpt', default=str(WORK_ROOT / 'outputs/checkpoints/stage1_v2_longtrain_220m_mainline_20260408/best.pt'))
    p.add_argument('--shared-lease-path', default=str(WORK_ROOT / 'reports/stage1_v2_gpu_lease_20260408.json'))
    p.add_argument('--bootstrap-cache-jsonl', default=str(WORK_ROOT / 'data/processed/stage2_real_bootstrap_cache_20260410/clip_vit_b32_core_trainval_required_subset.jsonl'))
    p.add_argument('--semantic-hard-manifest-path', default=str(WORK_ROOT / 'manifests/protocol_v2/stage2_semantic_hard_subsets_20260410.json'))
    p.add_argument('--stage2-semantic-value-diagnosis-report', default=str(WORK_ROOT / 'reports/stage2_semantic_value_diagnosis_20260410.json'))
    p.add_argument('--v7-repaired-summary-report', default=str(WORK_ROOT / 'reports/stage2_semantic_objective_redesign_v7_summary_20260413.json'))
    p.add_argument('--v7-repaired-diagnosis-report', default=str(WORK_ROOT / 'reports/stage2_semantic_objective_redesign_v7_diagnosis_20260413.json'))
    p.add_argument('--process-title-report', default=str(WORK_ROOT / 'reports/stage2_process_title_normalization_20260413.json'))
    p.add_argument('--concurrent-runtime-report', default=str(WORK_ROOT / 'reports/stage2_calibration_only_fullscale_wave1_runtime_20260413.json'))
    p.add_argument('--launch-report', default=str(WORK_ROOT / 'reports/stage2_calibration_only_fullscale_wave1_launch_20260413.json'))
    p.add_argument('--summary-report', default=str(WORK_ROOT / 'reports/stage2_calibration_only_fullscale_wave1_summary_20260413.json'))
    p.add_argument('--diagnosis-report', default=str(WORK_ROOT / 'reports/stage2_calibration_only_fullscale_wave1_diagnosis_20260413.json'))
    p.add_argument('--results-md', default=str(WORK_ROOT / 'docs/STAGE2_CALIBRATION_ONLY_FULLSCALE_WAVE1_RESULTS_20260413.md'))
    p.add_argument('--partial-unfreeze-report', default=str(WORK_ROOT / 'reports/stage2_partial_unfreeze_ablation_20260413.json'))
    p.add_argument('--reserve-idle-gpu-count', type=int, default=2)
    p.add_argument('--gpu-acquire-timeout-seconds', type=int, default=28800)
    p.add_argument('--gpu-acquire-retry-seconds', type=int, default=20)
    p.add_argument('--wait-timeout-seconds', type=int, default=172800)
    p.add_argument('--poll-seconds', type=int, default=120)
    return p.parse_args()


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
