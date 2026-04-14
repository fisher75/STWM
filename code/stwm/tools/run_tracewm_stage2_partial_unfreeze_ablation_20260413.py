#!/usr/bin/env python3
from __future__ import annotations

from argparse import ArgumentParser
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple
import json
import os
import subprocess
import time
import traceback

from stwm.tools.run_tracewm_stage2_semantic_objective_redesign_v1_20260410 import (
    _read_json,
    _write_json,
    _write_md,
)
from stwm.tools.run_tracewm_stage2_semantic_objective_redesign_v2_20260410 import (
    _python_bin_default,
    _release_lease_safe,
)
from stwm.tools.run_tracewm_stage2_calibration_only_fullscale_wave1_20260413 import (
    _f,
    _gpu_headroom_ok,
    _metric_rank_tuple,
    _select_clean_gpu_for_calibration,
    _summary_overall_rank,
)

WORK_ROOT = Path('/home/chen034/workspace/stwm')
REPORT_PATH = WORK_ROOT / 'reports/stage2_partial_unfreeze_ablation_20260413.json'
DOC_PATH = WORK_ROOT / 'docs/STAGE2_PARTIAL_UNFREEZE_ABLATION_20260413.md'
RUN_EXTRA_STEPS = 800
RUN_EVAL_INTERVAL = 200
RUN_SAVE_EVERY = 200
RUN_BATCH_SIZE = 8


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_or_empty(path_like: Any) -> Dict[str, Any]:
    path = Path(str(path_like))
    if not str(path_like) or not path.exists():
        return {}
    try:
        payload = _read_json(path)
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _summary_row_map(summary_payload: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {str(row.get('run_name', '')): row for row in summary_payload.get('run_rows', []) if isinstance(row, dict)}


def _winning_family(summary_payload: Dict[str, Any]) -> str:
    family = str(summary_payload.get('best_family', 'none'))
    if family in {'topk1', 'qcap15'}:
        return family
    rows = [row for row in summary_payload.get('run_rows', []) if isinstance(row, dict) and str(row.get('status', '')).lower() == 'completed']
    if not rows:
        return 'none'
    return str(min(rows, key=_summary_overall_rank).get('family', 'none'))


def _base_run_name(family: str, seed: int) -> str:
    return f'stage2_calonly_{family}_seed{int(seed)}_wave1_20260413'


def _run_partial_job(meta: Dict[str, Any]) -> Dict[str, Any]:
    lease_id = ''
    lease_path = str(meta['shared_lease_path'])
    run_name = str(meta['run_name'])
    log_path = Path(str(meta['log_path']))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    selected_gpu_id = -1
    try:
        acquire_deadline = time.time() + float(meta.get('gpu_acquire_timeout_seconds', 7200))
        last_gpu_error = ''
        while True:
            if not _gpu_headroom_ok(lease_path, int(meta.get('reserve_idle_gpu_count', 2))):
                last_gpu_error = f"gpu_headroom_blocked reserve_idle_gpu_count={int(meta.get('reserve_idle_gpu_count', 2))}"
            else:
                try:
                    gpu = _select_clean_gpu_for_calibration(run_name=run_name, lease_path=lease_path, required_mem_gb=40.0, safety_margin_gb=8.0)
                    selected_gpu_id = int(gpu['selected_gpu_id'])
                    lease_id = str(gpu['lease_id'])
                    break
                except Exception as exc:
                    last_gpu_error = str(exc)
            if time.time() >= acquire_deadline:
                raise RuntimeError(f'gpu_acquire_timeout run={run_name} last_error={last_gpu_error}')
            time.sleep(float(meta.get('gpu_acquire_retry_seconds', 20)))

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
            '--max-samples-train', '-1',
            '--max-samples-val', '-1',
            '--batch-size', str(meta['batch_size']),
            '--train-steps', str(meta['train_steps']),
            '--eval-interval', str(meta['eval_interval']),
            '--eval-max-batches', '0',
            '--save-every-n-steps', str(meta['save_every_n_steps']),
            '--semantic-source-mainline', 'crop_visual_encoder',
            '--legacy-semantic-source', 'hand_crafted_stats',
            '--semantic-crop-size', str(meta['semantic_crop_size']),
            '--semantic-rescue-mode', 'v7alignonly',
            '--semantic-rescue-weight', '0.00015',
            '--semantic-bootstrap-cache-path', str(meta['bootstrap_cache_jsonl']),
            '--semantic-bootstrap-target-dim', '512',
            '--semantic-hard-curriculum-weight', '0.0',
            '--semantic-aux-subset-weighting-strength', '1.0',
            '--confidence-gated-alignment-loss-weight', '1.0',
            '--sparse-persistence-contrastive-loss-weight', '0.0',
            '--confidence-gating-margin-threshold', '0.10',
            '--confidence-gating-temperature', '0.05',
            '--semantic-hard-score-threshold', '0.25',
            '--aux-loss-delay-steps', '180',
            '--aux-loss-ramp-steps', '360',
            '--v6-gating-family', str(meta['v6_gating_family']),
            '--v6-topk-query-k', str(meta['v6_topk_query_k']),
            '--v6-capped-quantile', str(meta['v6_capped_quantile']),
            '--v6-max-affected-ratio', str(meta['v6_max_affected_ratio']),
            '--v6-gate-min-strength', '0.05',
            '--v6-strict-max-pairs-per-sample', '0',
            '--v6-hard-negative-cap', '0',
            '--v6-pair-sampling-temperature', '0.35',
            '--v6-guaranteed-min-pairs-per-sample', '0',
            '--no-v6-two-level-pair-mining-enabled',
            '--semantic-hard-manifest-path', str(meta['semantic_hard_manifest_path']),
            '--resume-from', str(meta['resume_from']),
            '--skip-resume-optimizer',
            '--semantic-hard-sidecar-enabled',
            '--stage1-partial-unfreeze-mode', 'topblock',
            '--stage1-partial-unfreeze-layer-count', '1',
            '--stage1-partial-unfreeze-lr-scale', '0.10',
            '--output-dir', str(meta['output_dir']),
            '--run-name', str(meta['run_name']),
            '--run-summary-json', str(meta['raw_json']),
            '--progress-json', str(meta['progress_json']),
            '--seed', str(meta['seed']),
        ]
        proc_env = os.environ.copy()
        proc_env['CUDA_VISIBLE_DEVICES'] = str(selected_gpu_id)
        proc_env['STWM_PROC_TITLE'] = str(proc_env.get('STWM_PROC_TITLE', 'python'))
        proc_env['STWM_PROC_TITLE_MODE'] = str(proc_env.get('STWM_PROC_TITLE_MODE', 'generic'))
        proc_env['TRACEWM_STAGE1_V2_GPU_SELECTION_METADATA_JSON'] = json.dumps(
            {
                'selected_gpu_id': int(selected_gpu_id),
                'lease_id': str(lease_id),
                'owner': run_name,
                'mode': 'single_gpu_only',
            },
            ensure_ascii=True,
        )
        with log_path.open('w', encoding='utf-8') as log_fh:
            proc = subprocess.run(cmd, cwd=str(meta['work_root']), text=True, stdout=log_fh, stderr=subprocess.STDOUT, env=proc_env)
        if proc.returncode != 0:
            raise RuntimeError(f'trainer failed rc={proc.returncode}')
        raw = _read_json(meta['raw_json'])
        raw.update({'generated_at_utc': now_iso(), 'status': 'completed', 'selected_gpu_id': selected_gpu_id, 'lease_id': lease_id})
        _write_json(meta['final_json'], raw)
        return raw
    except Exception as exc:
        payload = {
            'generated_at_utc': now_iso(),
            'run_name': run_name,
            'status': 'failed',
            'selected_gpu_id': selected_gpu_id,
            'lease_id': lease_id,
            'message': str(exc),
            'traceback': traceback.format_exc(),
        }
        _write_json(meta['final_json'], payload)
        return payload
    finally:
        _release_lease_safe(lease_id=lease_id, lease_path=lease_path)


def run(args: Any) -> Dict[str, Any]:
    summary = _json_or_empty(args.calibration_summary_report)
    if not summary or not bool(summary.get('all_runs_terminal', False)) or int(summary.get('failed_count', 0)) > 0:
        payload = {
            'generated_at_utc': now_iso(),
            'status': 'blocked_waiting_for_calibration_only_completion',
            'reason': 'calibration-only mainline not fully completed and clean yet',
            'partial_unfreeze_beats_frozen_calibration': None,
            'forgetting_or_instability_detected': None,
            'runs': [],
        }
        _write_json(args.partial_unfreeze_report, payload)
        _write_md(args.partial_unfreeze_doc, ['# Stage2 Partial-Unfreeze Ablation', '', f'- status: {payload["status"]}', f'- reason: {payload["reason"]}'])
        return payload

    family = _winning_family(summary)
    if family not in {'topk1', 'qcap15'}:
        payload = {
            'generated_at_utc': now_iso(),
            'status': 'blocked_missing_winning_family',
            'reason': 'calibration-only summary missing topk1/qcap15 winner',
            'partial_unfreeze_beats_frozen_calibration': None,
            'forgetting_or_instability_detected': None,
            'runs': [],
        }
        _write_json(args.partial_unfreeze_report, payload)
        return payload

    row_map = _summary_row_map(summary)
    launch_rows: List[Dict[str, Any]] = []
    completed_rows: List[Dict[str, Any]] = []
    for seed in [42, 123]:
        base_run_name = _base_run_name(family, seed)
        base_row = row_map.get(base_run_name, {})
        if not base_row or str(base_row.get('status', '')).lower() != 'completed':
            continue
        resume_from = str(WORK_ROOT / 'outputs/checkpoints' / base_run_name / 'best.pt')
        resume_step = int((base_row.get('best_checkpoint_metric', {}) if isinstance(base_row.get('best_checkpoint_metric', {}), dict) else {}).get('global_step', -1) or -1)
        meta = {
            'run_name': f'stage2_partialunfreeze_topblock_seed{seed}_20260413',
            'seed': seed,
            'resume_from': resume_from,
            'resume_global_step': resume_step,
            'train_steps': int(max(resume_step, 0) + RUN_EXTRA_STEPS),
            'eval_interval': RUN_EVAL_INTERVAL,
            'save_every_n_steps': RUN_SAVE_EVERY,
            'batch_size': RUN_BATCH_SIZE,
            'obs_len': 8,
            'fut_len': 8,
            'max_tokens': 64,
            'semantic_crop_size': 64,
            'v6_gating_family': 'hard_topk_query_gating_v2' if family == 'topk1' else 'capped_quantile_sparse_gating_v2',
            'v6_topk_query_k': 1,
            'v6_capped_quantile': 0.80 if family == 'qcap15' else 0.85,
            'v6_max_affected_ratio': 0.15,
            'bootstrap_cache_jsonl': str(args.bootstrap_cache_jsonl),
            'semantic_hard_manifest_path': str(args.semantic_hard_manifest_path),
            'shared_lease_path': str(args.shared_lease_path),
            'stage2_contract_json': str(args.stage2_contract_json),
            'stage1_runtime_json': str(args.stage1_runtime_json),
            'stage1_best_ckpt': str(args.stage1_best_ckpt),
            'python_bin': str(args.python_bin),
            'work_root': str(args.work_root),
            'gpu_acquire_timeout_seconds': int(args.gpu_acquire_timeout_seconds),
            'gpu_acquire_retry_seconds': int(args.gpu_acquire_retry_seconds),
            'reserve_idle_gpu_count': int(args.reserve_idle_gpu_count),
            'output_dir': str(WORK_ROOT / 'outputs/checkpoints' / f'stage2_partialunfreeze_topblock_seed{seed}_20260413'),
            'raw_json': str(WORK_ROOT / 'reports' / f'stage2_partialunfreeze_topblock_seed{seed}_20260413_raw.json'),
            'progress_json': str(WORK_ROOT / 'reports' / f'stage2_partialunfreeze_topblock_seed{seed}_20260413_progress.json'),
            'final_json': str(WORK_ROOT / 'reports' / f'stage2_partialunfreeze_topblock_seed{seed}_20260413_final.json'),
            'log_path': str(WORK_ROOT / 'logs' / f'stage2_partialunfreeze_topblock_seed{seed}_20260413.log'),
            'frozen_anchor_run_name': base_run_name,
            'frozen_anchor_best_checkpoint_metric': base_row.get('best_checkpoint_metric', {}),
        }
        launch_rows.append(meta)
        completed_rows.append(_run_partial_job(meta))

    frozen_best_row = min(
        [row for row in summary.get('run_rows', []) if isinstance(row, dict) and str(row.get('status', '')).lower() == 'completed'],
        key=_summary_overall_rank,
    )
    frozen_best_rank = _metric_rank_tuple(frozen_best_row.get('best_checkpoint_metric', {}))
    partial_final_rows: List[Dict[str, Any]] = []
    for meta in launch_rows:
        final_payload = _json_or_empty(meta['final_json'])
        raw_payload = _json_or_empty(meta['raw_json'])
        metrics_block = final_payload.get('best_checkpoint_metric', {}) if isinstance(final_payload.get('best_checkpoint_metric', {}), dict) else raw_payload.get('best_checkpoint_metric', {}) if isinstance(raw_payload.get('best_checkpoint_metric', {}), dict) else {}
        freeze_boundary = raw_payload.get('freeze_trainable_boundary', {}) if isinstance(raw_payload.get('freeze_trainable_boundary', {}), dict) else {}
        trainable_delta = int(freeze_boundary.get('stage1_trainable_parameter_count', 0))
        partial_final_rows.append(
            {
                'run_name': str(meta['run_name']),
                'seed': int(meta['seed']),
                'status': str(final_payload.get('status', raw_payload.get('status', 'unknown'))),
                'best_checkpoint_metric': metrics_block,
                'trainable_parameter_count_delta': trainable_delta,
                'frozen_anchor_run_name': str(meta['frozen_anchor_run_name']),
                'frozen_anchor_best_checkpoint_metric': meta['frozen_anchor_best_checkpoint_metric'],
                'boundary': freeze_boundary,
                'run_stable': bool(raw_payload.get('run_stable', False)),
            }
        )

    completed_partial = [row for row in partial_final_rows if str(row.get('status', '')).lower() == 'completed']
    best_partial = min(completed_partial, key=lambda row: _metric_rank_tuple(row.get('best_checkpoint_metric', {}))) if completed_partial else {}
    best_partial_rank = _metric_rank_tuple(best_partial.get('best_checkpoint_metric', {})) if best_partial else (1e9, 1e9, 1e9)
    partial_beats = bool(best_partial and best_partial_rank < frozen_best_rank)
    forgetting_detected = bool(any((not bool(row.get('run_stable', False))) or (_metric_rank_tuple(row.get('best_checkpoint_metric', {}))[0] > frozen_best_rank[0] * 1.05) for row in completed_partial))

    payload = {
        'generated_at_utc': now_iso(),
        'status': 'completed' if completed_partial else 'failed_or_empty',
        'winning_calibration_family': family,
        'frozen_calibration_best_run_name': str(frozen_best_row.get('run_name', 'none')),
        'frozen_calibration_best_checkpoint_metric': frozen_best_row.get('best_checkpoint_metric', {}),
        'runs': partial_final_rows,
        'partial_unfreeze_beats_frozen_calibration': bool(partial_beats),
        'forgetting_or_instability_detected': bool(forgetting_detected),
    }
    _write_json(args.partial_unfreeze_report, payload)
    lines = [
        '# Stage2 Partial-Unfreeze Ablation',
        '',
        f'- generated_at_utc: {payload["generated_at_utc"]}',
        f'- status: {payload["status"]}',
        f'- winning_calibration_family: {payload["winning_calibration_family"]}',
        f'- frozen_calibration_best_run_name: {payload["frozen_calibration_best_run_name"]}',
        f'- partial_unfreeze_beats_frozen_calibration: {payload["partial_unfreeze_beats_frozen_calibration"]}',
        f'- forgetting_or_instability_detected: {payload["forgetting_or_instability_detected"]}',
        '',
        '| run_name | seed | status | endpoint_l2 | trainable_parameter_count_delta | frozen_anchor_run_name |',
        '|---|---:|---|---:|---:|---|',
    ]
    for row in partial_final_rows:
        lines.append(
            f"| {row.get('run_name', '')} | {int(row.get('seed', -1))} | {row.get('status', '')} | {_metric_rank_tuple(row.get('best_checkpoint_metric', {}))[0]:.6f} | {int(row.get('trainable_parameter_count_delta', 0))} | {row.get('frozen_anchor_run_name', '')} |"
        )
    _write_md(args.partial_unfreeze_doc, lines)
    return payload


def parse_args() -> Any:
    p = ArgumentParser(description='Stage2 partial-unfreeze ablation runner')
    p.add_argument('--mode', default='run', choices=['run'])
    p.add_argument('--work-root', default=str(WORK_ROOT))
    p.add_argument('--python-bin', default=_python_bin_default())
    p.add_argument('--stage2-contract-json', default=str(WORK_ROOT / 'reports/stage2_bootstrap_data_contract_20260408.json'))
    p.add_argument('--stage1-runtime-json', default=str(WORK_ROOT / 'reports/stage1_v2_recommended_runtime_20260408.json'))
    p.add_argument('--stage1-best-ckpt', default=str(WORK_ROOT / 'outputs/checkpoints/stage1_v2_longtrain_220m_mainline_20260408/best.pt'))
    p.add_argument('--shared-lease-path', default=str(WORK_ROOT / 'reports/stage1_v2_gpu_lease_20260408.json'))
    p.add_argument('--bootstrap-cache-jsonl', default=str(WORK_ROOT / 'data/processed/stage2_real_bootstrap_cache_20260410/clip_vit_b32_core_trainval_required_subset.jsonl'))
    p.add_argument('--semantic-hard-manifest-path', default=str(WORK_ROOT / 'manifests/protocol_v2/stage2_semantic_hard_subsets_20260410.json'))
    p.add_argument('--calibration-summary-report', default=str(WORK_ROOT / 'reports/stage2_calibration_only_fullscale_wave1_summary_20260413.json'))
    p.add_argument('--partial-unfreeze-report', default=str(REPORT_PATH))
    p.add_argument('--partial-unfreeze-doc', default=str(DOC_PATH))
    p.add_argument('--reserve-idle-gpu-count', type=int, default=2)
    p.add_argument('--gpu-acquire-timeout-seconds', type=int, default=14400)
    p.add_argument('--gpu-acquire-retry-seconds', type=int, default=20)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    print(json.dumps(run(args), ensure_ascii=True, indent=2))


if __name__ == '__main__':
    main()
