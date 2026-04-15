#!/usr/bin/env python3
from __future__ import annotations

from argparse import ArgumentParser
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple
import json
import math
import os
import time

import torch

from stwm.tools.run_tracewm_stage2_semantic_objective_redesign_v1_20260410 import _write_json, _write_md
from stwm.tools.run_tracewm_stage2_semantic_objective_redesign_v2_20260410 import _release_lease_safe
from stwm.tools.run_tracewm_stage2_calibration_only_wave2_20260414 import _select_exclusive_gpu
from stwm.tools.run_tracewm_stage1_stage2_qualitative_pack_v3_20260413 import (
    _MPL_ERROR,
    _PIL_ERROR,
    _best_family_run,
    _case_id,
    _coords_from_sample,
    _image_from_sample,
    _overlay_traj,
    _render_stage1_case,
    _select_stage1_cases,
    _stage1_predict,
    _subset_tag_map,
    Image,
    plt,
)
from stwm.tools.run_tracewm_stage2_ljs_semantic_diagnosis_and_rescue_20260410 import (
    _f,
    _load_stage1_model,
    _load_stage2_modules,
    _make_dataset,
    _read_json,
)

WORK_ROOT = Path('/home/chen034/workspace/stwm')
OUTPUT_DIR = WORK_ROOT / 'outputs' / 'visualizations' / 'stage1_stage2_qualitative_pack_v5_20260414'
STAGE1_REPORT = WORK_ROOT / 'reports' / 'stage1_qualitative_pack_v5_20260414.json'
STAGE2_REPORT = WORK_ROOT / 'reports' / 'stage2_qualitative_pack_v5_20260414.json'
DOC_PATH = WORK_ROOT / 'docs' / 'STAGE1_STAGE2_QUALITATIVE_PACK_V5_20260414.md'


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _apply_process_title_normalization() -> None:
    mode = str(os.environ.get('STWM_PROC_TITLE_MODE', 'generic')).strip().lower()
    if mode != 'generic':
        return
    title = str(os.environ.get('STWM_PROC_TITLE', 'python:eval')).strip() or 'python:eval'
    lowered = title.lower()
    if 'stwm' in lowered or 'tracewm' in lowered or '/home/' in lowered:
        title = 'python:eval'
    try:
        import setproctitle  # type: ignore
        setproctitle.setproctitle(title)
    except Exception:
        pass


def _preferred_checkpoint_for_run(run_name: str) -> str:
    final_json = WORK_ROOT / 'reports' / f'{run_name}_final.json'
    if not final_json.exists():
        return 'best.pt'
    payload = _read_json(final_json)
    sidecar = payload.get('sidecar_checkpoint_selection', {}) if isinstance(payload.get('sidecar_checkpoint_selection', {}), dict) else {}
    sidecar_ckpt = WORK_ROOT / 'outputs' / 'checkpoints' / run_name / 'best_semantic_hard.pt'
    if bool(sidecar.get('sidecar_truly_diverged', False)) and sidecar_ckpt.exists():
        return 'best_semantic_hard.pt'
    return 'best.pt'


def _acquire_eval_gpu(args: Any, owner: str) -> Dict[str, Any]:
    deadline = time.time() + float(args.wait_timeout_seconds)
    last_error = ''
    while time.time() < deadline:
        try:
            return _select_exclusive_gpu(run_name=owner, lease_path=str(args.shared_lease_path), required_mem_gb=40.0, reserve_idle_gpu_count=2)
        except Exception as exc:
            last_error = str(exc)
            time.sleep(float(args.poll_seconds))
    raise RuntimeError(f'gpu_acquire_timeout owner={owner} last_error={last_error}')


def _wait_for_file(path_like: Any, timeout: int, poll: int) -> Dict[str, Any]:
    deadline = time.time() + float(timeout)
    target = Path(str(path_like))
    while time.time() < deadline:
        if target.exists():
            payload = _read_json(target)
            if isinstance(payload, dict):
                return payload
        time.sleep(float(poll))
    raise TimeoutError(f'timed out waiting for {target}')


def _load_stage1_with_optional_override(device: torch.device, stage1_ckpt: str, stage2_ckpt: str | None = None) -> Any:
    stage1_model, _ = _load_stage1_model(device=device, stage1_ckpt=str(stage1_ckpt))
    if stage2_ckpt:
        ckpt = torch.load(Path(stage2_ckpt), map_location=device, weights_only=False)
        stage1_state = ckpt.get('stage1_model_state_dict') if isinstance(ckpt.get('stage1_model_state_dict'), dict) else None
        if stage1_state is not None:
            stage1_model.load_state_dict(stage1_state, strict=False)
    return stage1_model


def _stage2_predict_bundle(stage1_model: Any, loaded: Tuple[Any, Any, Any, str], sample: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    from stwm.tools.run_tracewm_stage1_stage2_qualitative_pack_v3_20260413 import _stage2_predict
    return _stage2_predict(loaded, stage1_model, sample, device)


def _render_stage2_case_v4(sample: Dict[str, Any], predictions: Dict[str, Dict[str, Any]], out_path: Path, title: str, note: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if plt is None:
        return
    bg = _image_from_sample(sample)
    obs, fut = _coords_from_sample(sample)
    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    ax.imshow(bg)
    ax.axis('off')
    ax.set_title(title)
    _overlay_traj(ax, obs, 'black', 'obs', alpha=0.9)
    _overlay_traj(ax, torch.vstack([]).cpu().numpy() if False else __import__('numpy').vstack([obs[-1:], fut]), '#2ca02c', 'gt_future', linestyle='--', alpha=0.95)
    colors = {
        'cropenc_baseline': '#1f77b4',
        'legacysem_baseline': '#ff7f0e',
        'v7_alignment_only': '#d62728',
        'calibration_only_wave1': '#2ca02c',
        'calibration_only_wave2': '#17becf',
        'noalign_ablation': '#9467bd',
        'densegate_ablation': '#8c564b',
        'nodelay_ablation': '#e377c2',
    }
    lines = []
    import numpy as np
    ordered_keys = ['cropenc_baseline', 'legacysem_baseline', 'v7_alignment_only', 'calibration_only_wave1', 'calibration_only_wave2', 'noalign_ablation', 'densegate_ablation', 'nodelay_ablation']
    for key in ordered_keys:
        if key not in predictions:
            continue
        pred = predictions[key]
        _overlay_traj(ax, np.vstack([obs[-1:], pred['pred_future']]), colors.get(key, '#444444'), key, alpha=0.95)
        lines.append(f"{key}_ep={float(pred['endpoint_l2']):.4f}")
    lines.append(note)
    ax.legend(loc='lower left', fontsize=7)
    ax.text(8, 20, '\n'.join(lines), fontsize=8, color='black', bbox={'facecolor': 'white', 'alpha': 0.75, 'pad': 4})
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def build_stage1_pack(args: Any, core_ds: Any, stage1_model: Any, device: torch.device) -> Dict[str, Any]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cases = _select_stage1_cases(core_ds, stage1_model, device, scan_window=int(args.stage1_scan_window))
    case_rows: List[Dict[str, Any]] = []
    for i, case in enumerate(cases):
        sample = core_ds[int(case['dataset_index'])]
        out_path = OUTPUT_DIR / 'stage1' / f"{case['bucket']}_{_case_id('stage1v5', i)}.png"
        note = f"dataset={case['dataset']} clip={case['clip_id']}"
        _render_stage1_case(sample, case['stage1'], out_path, title=f"Stage1 {case['bucket']}", note=note)
        case_rows.append(
            {
                'case_id': _case_id('stage1v5', i),
                'bucket': str(case['bucket']),
                'dataset_source': str(case['dataset']),
                'clip_id': str(case['clip_id']),
                'dataset_index': int(case['dataset_index']),
                'subset_tags': [str(case['bucket'])],
                'why_selected': str(case['why_selected']),
                'qualitative_interpretation': (
                    'stable future trace continuation' if case['bucket'] == 'easy_cases' else (
                        'large motion regime with visible trajectory change' if case['bucket'] == 'dynamic_change_cases' else 'clear failure or boundary behavior under frozen Stage1 rollout'
                    )
                ),
                'semantic_frame_path': str(sample.get('semantic_frame_path', '')),
                'render_path': str(out_path),
                'metrics': {
                    'stage1_frozen': {
                        'free_rollout_endpoint_l2': float(case['endpoint_l2']),
                        'free_rollout_coord_mean_l2': float(case['coord_mean_l2']),
                    }
                },
            }
        )
    payload = {
        'generated_at_utc': now_iso(),
        'pack_type': 'stage1_qualitative_pack_v5',
        'source_checkpoint': str(args.stage1_best_ckpt),
        'selection_policy': {
            'scan_window': int(args.stage1_scan_window),
            'groups': ['easy_cases', 'dynamic_change_cases', 'failure_boundary_cases'],
            'target_case_count_per_group': 3,
            'cross_group_dedup_key': ['dataset', 'clip_id'],
            'paper_usefulness_goal': 'human figure shortlist ready',
        },
        'cases': case_rows,
        'notes': {
            'matplotlib_available': bool(plt is not None),
            'pil_available': bool(Image is not None),
            'matplotlib_error': _MPL_ERROR,
            'pil_error': _PIL_ERROR,
        },
    }
    _write_json(args.stage1_pack_report, payload)
    return payload


def _method_bundle(args: Any, device: torch.device) -> Dict[str, Dict[str, Any]]:
    wave1_diag = _wait_for_file(args.wave1_diagnosis_report, args.wait_timeout_seconds, args.poll_seconds)
    wave2_diag = _wait_for_file(args.wave2_diagnosis_report, args.wait_timeout_seconds, args.poll_seconds)
    wave2_summary = _wait_for_file(args.wave2_summary_report, args.wait_timeout_seconds, args.poll_seconds)
    v7_diag = _wait_for_file(args.v7_diagnosis_report, args.wait_timeout_seconds, args.poll_seconds)
    ablation_report = _wait_for_file(args.ablation_pack_report, args.wait_timeout_seconds, args.poll_seconds)

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
    v7_align_run = str(v7_diag.get('overall_best_run_name', 'none'))
    cal_wave1_run = str(wave1_diag.get('overall_best_run_name', 'none'))
    cal_wave2_run = str(wave2_diag.get('overall_best_run_name', wave2_summary.get('overall_best_run_name', 'none')))
    ablation_rows = [row for row in ablation_report.get('run_rows', []) if isinstance(row, dict) and str(row.get('status', '')).lower() == 'completed']
    ablation_by_name = {str(row.get('ablation_name', '')): row for row in ablation_rows}

    bundles: Dict[str, Dict[str, Any]] = {}
    base_stage1 = _load_stage1_with_optional_override(device, args.stage1_best_ckpt)
    for key, run_name in [
        ('cropenc_baseline', crop_run),
        ('legacysem_baseline', legacy_run),
        ('v7_alignment_only', v7_align_run),
        ('calibration_only_wave1', cal_wave1_run),
        ('calibration_only_wave2', cal_wave2_run),
    ]:
        if not run_name or run_name == 'none':
            continue
        ckpt_name = _preferred_checkpoint_for_run(run_name)
        ckpt_path = str(WORK_ROOT / 'outputs/checkpoints' / run_name / ckpt_name)
        if not Path(ckpt_path).exists():
            continue
        bundles[key] = {
            'run_name': run_name,
            'checkpoint_name': ckpt_name,
            'stage1_model': base_stage1,
            'loaded': _load_stage2_modules(ckpt_path, device, base_stage1),
        }
    for ablation_name, bundle_key in [('noalign', 'noalign_ablation'), ('densegate', 'densegate_ablation'), ('nodelay', 'nodelay_ablation')]:
        row = ablation_by_name.get(ablation_name, {})
        run_name = str(row.get('run_name', 'none'))
        if not run_name or run_name == 'none':
            continue
        ckpt_name = _preferred_checkpoint_for_run(run_name)
        ckpt_path = str(WORK_ROOT / 'outputs/checkpoints' / run_name / ckpt_name)
        if not Path(ckpt_path).exists():
            continue
        bundles[bundle_key] = {
            'run_name': run_name,
            'checkpoint_name': ckpt_name,
            'stage1_model': base_stage1,
            'loaded': _load_stage2_modules(ckpt_path, device, base_stage1),
        }
    return bundles


def _select_stage2_groups(core_ds: Any, bundles: Dict[str, Dict[str, Any]], manifest: Dict[str, Any], device: torch.device) -> Dict[str, List[Dict[str, Any]]]:
    tag_map = _subset_tag_map(manifest, ['occlusion_reappearance', 'crossing_or_interaction_ambiguity', 'small_object_or_low_area', 'appearance_change_or_semantic_shift'])
    rows: List[Dict[str, Any]] = []
    for idx, tags in tag_map.items():
        sample = core_ds[int(idx)]
        preds: Dict[str, Dict[str, Any]] = {}
        for key, bundle in bundles.items():
            preds[key] = _stage2_predict_bundle(bundle['stage1_model'], bundle['loaded'], sample, device)
        crop_ep = float(preds['cropenc_baseline']['endpoint_l2'])
        legacy_ep = float(preds['legacysem_baseline']['endpoint_l2'])
        v7_ep = float(preds['v7_alignment_only']['endpoint_l2'])
        wave1_ep = float(preds['calibration_only_wave1']['endpoint_l2'])
        wave2_ep = float(preds['calibration_only_wave2']['endpoint_l2']) if 'calibration_only_wave2' in preds else wave1_ep
        best_cal_key = 'calibration_only_wave2' if ('calibration_only_wave2' in preds and wave2_ep <= wave1_ep) else 'calibration_only_wave1'
        best_cal_ep = float(preds[best_cal_key]['endpoint_l2'])
        base_best = min(crop_ep, legacy_ep)
        entry = {
            'dataset_index': int(idx),
            'dataset': str(sample.get('meta', {}).get('dataset', '')),
            'clip_id': str(sample.get('meta', {}).get('clip_id', '')),
            'subset_tags': list(tags),
            'predictions': preds,
            'best_calibration_key': best_cal_key,
            'cal_margin_vs_base': float(base_best - best_cal_ep),
            'cal_margin_vs_v7': float(v7_ep - best_cal_ep),
            'legacy_margin_vs_cal': float(best_cal_ep - legacy_ep),
            'crop_margin_vs_cal': float(best_cal_ep - crop_ep),
            'wave2_margin_vs_wave1': float(wave1_ep - wave2_ep) if 'calibration_only_wave2' in preds else 0.0,
            'noalign_margin_vs_cal': float(preds['noalign_ablation']['endpoint_l2'] - best_cal_ep) if 'noalign_ablation' in preds else 0.0,
            'densegate_margin_vs_cal': float(preds['densegate_ablation']['endpoint_l2'] - best_cal_ep) if 'densegate_ablation' in preds else 0.0,
            'nodelay_margin_vs_cal': float(preds['nodelay_ablation']['endpoint_l2'] - best_cal_ep) if 'nodelay_ablation' in preds else 0.0,
        }
        rows.append(entry)

    def pick(pool: List[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        used = set()
        for row in pool:
            key = (str(row.get('dataset', '')), str(row.get('clip_id', '')))
            if key in used:
                continue
            out.append(row)
            used.add(key)
            if len(out) >= limit:
                break
        return out

    groups = {
        'calibration_only_positive_cases': pick(sorted([r for r in rows if r['cal_margin_vs_base'] > 0.002 and r['cal_margin_vs_v7'] > 0.0], key=lambda r: (r['cal_margin_vs_base'], r['cal_margin_vs_v7'], r['wave2_margin_vs_wave1'], len(r['subset_tags'])), reverse=True), 4),
        'legacysem_win_cases': pick(sorted([r for r in rows if r['legacy_margin_vs_cal'] < -0.001], key=lambda r: (abs(r['legacy_margin_vs_cal']), len(r['subset_tags'])), reverse=True), 4),
        'cropenc_win_cases': pick(sorted([r for r in rows if r['crop_margin_vs_cal'] < -0.001], key=lambda r: (abs(r['crop_margin_vs_cal']), len(r['subset_tags'])), reverse=True), 4),
        'noalign_failure_cases': pick(sorted([r for r in rows if 'noalign_ablation' in r['predictions'] and r['noalign_margin_vs_cal'] > 0.001], key=lambda r: (r['noalign_margin_vs_cal'], len(r['subset_tags'])), reverse=True), 4),
        'densegate_failure_cases': pick(sorted([r for r in rows if 'densegate_ablation' in r['predictions'] and r['densegate_margin_vs_cal'] > 0.001], key=lambda r: (r['densegate_margin_vs_cal'], len(r['subset_tags'])), reverse=True), 4),
        'nodelay_failure_cases': pick(sorted([r for r in rows if 'nodelay_ablation' in r['predictions'] and r['nodelay_margin_vs_cal'] > 0.001], key=lambda r: (r['nodelay_margin_vs_cal'], len(r['subset_tags'])), reverse=True), 4),
    }
    return groups


def build_stage2_pack(args: Any, core_ds: Any, device: torch.device) -> Dict[str, Any]:
    manifest = _read_json(args.semantic_hard_manifest_path)
    bundles = _method_bundle(args, device)
    groups = _select_stage2_groups(core_ds, bundles, manifest, device)
    case_rows: List[Dict[str, Any]] = []
    group_meta = {
        'calibration_only_positive_cases': ('calibration-only beats baseline envelope on semantic-hard clips', 'alignment/calibration branch appears genuinely helpful here'),
        'legacysem_win_cases': ('legacy semantics still wins on this clip', 'hand-crafted semantic stats remain stronger on this failure mode'),
        'cropenc_win_cases': ('plain cropenc baseline beats calibration-only here', 'semantic calibration is not uniformly better'),
        'noalign_failure_cases': ('removing alignment head hurts this clip', 'supports semantic alignment as a load-bearing mechanism'),
        'densegate_failure_cases': ('making the gate dense hurts this clip', 'supports sparse gating as a load-bearing mechanism'),
        'nodelay_failure_cases': ('removing delayed schedule hurts this clip', 'supports delayed aux schedule as a load-bearing mechanism'),
    }
    for group_name, cases in groups.items():
        for case in cases:
            sample = core_ds[int(case['dataset_index'])]
            out_path = OUTPUT_DIR / 'stage2' / f"{group_name}_{_case_id('stage2v5', len(case_rows))}.png"
            note = (
                f"tags={','.join(case['subset_tags'])}\n"
                f"best_cal={case['best_calibration_key']}\n"
                f"cal_vs_base={float(case['cal_margin_vs_base']):+.4f}\n"
                f"cal_vs_v7={float(case['cal_margin_vs_v7']):+.4f}\n"
                f"wave2_vs_wave1={float(case.get('wave2_margin_vs_wave1', 0.0)):+.4f}"
            )
            _render_stage2_case_v4(sample, case['predictions'], out_path, title=f"Stage2 {group_name}", note=note)
            metrics = {}
            for key, pred in case['predictions'].items():
                metrics[key] = {
                    'free_rollout_endpoint_l2': float(pred['endpoint_l2']),
                    'free_rollout_coord_mean_l2': float(pred['coord_mean_l2']),
                }
            why, interp = group_meta.get(group_name, ('taxonomy-driven selection', 'taxonomy-driven interpretation'))
            case_rows.append(
                {
                    'case_id': _case_id('stage2v5', len(case_rows)),
                    'group': group_name,
                    'dataset_source': str(case['dataset']),
                    'clip_id': str(case['clip_id']),
                    'dataset_index': int(case['dataset_index']),
                    'subset_tags': list(case['subset_tags']),
                    'why_selected': why,
                    'qualitative_interpretation': interp,
                    'best_calibration_key': str(case['best_calibration_key']),
                    'semantic_frame_path': str(sample.get('semantic_frame_path', '')),
                    'render_path': str(out_path),
                    'metrics': metrics,
                    'method_runs': {key: str(bundle.get('run_name', '')) for key, bundle in bundles.items()},
                }
            )
    payload = {
        'generated_at_utc': now_iso(),
        'pack_type': 'stage2_qualitative_pack_v5',
        'selection_policy': {
            'base_panel': 'semantic-hard subsets from protocol_v2 manifest',
            'taxonomy': [
                'calibration_only_positive_cases',
                'legacysem_win_cases',
                'cropenc_win_cases',
                'noalign_failure_cases',
                'densegate_failure_cases',
                'nodelay_failure_cases',
            ],
            'paper_usefulness_goal': 'direct human shortlist for figure selection and figure draft assembly',
            'target_case_count_per_group': 4,
        },
        'comparison_objects': list(bundles.keys()),
        'cases': case_rows,
    }
    _write_json(args.stage2_pack_report, payload)
    return payload


def write_doc(path: str | Path, stage1_pack: Dict[str, Any], stage2_pack: Dict[str, Any]) -> None:
    lines = [
        '# Stage1 / Stage2 Qualitative Pack V5',
        '',
        f'- generated_at_utc: {now_iso()}',
        f'- stage1_pack: {STAGE1_REPORT}',
        f'- stage2_pack: {STAGE2_REPORT}',
        f'- output_dir: {OUTPUT_DIR}',
        '',
        '## Stage1',
        '',
        '| case_id | bucket | dataset | clip_id | endpoint_l2 | render |',
        '|---|---|---|---|---:|---|',
    ]
    for case in stage1_pack.get('cases', []) if isinstance(stage1_pack.get('cases', []), list) else []:
        lines.append(
            f"| {case['case_id']} | {case['bucket']} | {case['dataset_source']} | {case['clip_id']} | {_f(case.get('metrics', {}).get('stage1_frozen', {}).get('free_rollout_endpoint_l2'), 1e9):.4f} | {case['render_path']} |"
        )
    lines.extend([
        '',
        '## Stage2',
        '',
        '| case_id | group | dataset | clip_id | tags | cropenc_ep | legacy_ep | v7_align_ep | cal_wave1_ep | cal_wave2_ep | noalign_ep | densegate_ep | nodelay_ep | render |',
        '|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|',
    ])
    for case in stage2_pack.get('cases', []) if isinstance(stage2_pack.get('cases', []), list) else []:
        metrics = case.get('metrics', {}) if isinstance(case.get('metrics', {}), dict) else {}
        lines.append(
            f"| {case['case_id']} | {case['group']} | {case['dataset_source']} | {case['clip_id']} | {','.join(case.get('subset_tags', []))} | "
            f"{_f(metrics.get('cropenc_baseline', {}).get('free_rollout_endpoint_l2'), 1e9):.4f} | "
            f"{_f(metrics.get('legacysem_baseline', {}).get('free_rollout_endpoint_l2'), 1e9):.4f} | "
            f"{_f(metrics.get('v7_alignment_only', {}).get('free_rollout_endpoint_l2'), 1e9):.4f} | "
            f"{_f(metrics.get('calibration_only_wave1', {}).get('free_rollout_endpoint_l2'), 1e9):.4f} | "
            f"{_f(metrics.get('calibration_only_wave2', {}).get('free_rollout_endpoint_l2'), 1e9):.4f} | "
            f"{_f(metrics.get('noalign_ablation', {}).get('free_rollout_endpoint_l2'), 1e9):.4f} | "
            f"{_f(metrics.get('densegate_ablation', {}).get('free_rollout_endpoint_l2'), 1e9):.4f} | "
            f"{_f(metrics.get('nodelay_ablation', {}).get('free_rollout_endpoint_l2'), 1e9):.4f} | {case['render_path']} |"
        )
    _write_md(path, lines)


def parse_args() -> Any:
    p = ArgumentParser(description='Build Stage1/Stage2 qualitative packs v5 for manual figure selection')
    p.add_argument('--work-root', default=str(WORK_ROOT))
    p.add_argument('--stage2-contract-json', default=str(WORK_ROOT / 'reports/stage2_bootstrap_data_contract_20260408.json'))
    p.add_argument('--stage1-best-ckpt', default=str(WORK_ROOT / 'outputs/checkpoints/stage1_v2_longtrain_220m_mainline_20260408/best.pt'))
    p.add_argument('--shared-lease-path', default=str(WORK_ROOT / 'reports/stage1_v2_gpu_lease_20260408.json'))
    p.add_argument('--semantic-hard-manifest-path', default=str(WORK_ROOT / 'manifests/protocol_v2/stage2_semantic_hard_subsets_20260410.json'))
    p.add_argument('--v7-diagnosis-report', default=str(WORK_ROOT / 'reports/stage2_semantic_objective_redesign_v7_diagnosis_20260413.json'))
    p.add_argument('--wave1-diagnosis-report', default=str(WORK_ROOT / 'reports/stage2_calibration_only_fullscale_wave1_diagnosis_20260413.json'))
    p.add_argument('--wave2-summary-report', default=str(WORK_ROOT / 'reports/stage2_calibration_only_wave2_summary_20260414.json'))
    p.add_argument('--wave2-diagnosis-report', default=str(WORK_ROOT / 'reports/stage2_calibration_only_wave2_diagnosis_20260414.json'))
    p.add_argument('--final-pack-diagnosis-report', default=str(WORK_ROOT / 'reports/stage2_calibration_only_final_pack_diagnosis_20260414.json'))
    p.add_argument('--ablation-pack-report', default=str(WORK_ROOT / 'reports/stage2_calibration_only_ablation_pack_20260414.json'))
    p.add_argument('--stage1-pack-report', default=str(STAGE1_REPORT))
    p.add_argument('--stage2-pack-report', default=str(STAGE2_REPORT))
    p.add_argument('--qualitative-doc', default=str(DOC_PATH))
    p.add_argument('--stage1-scan-window', type=int, default=96)
    p.add_argument('--wait-timeout-seconds', type=int, default=172800)
    p.add_argument('--poll-seconds', type=int, default=120)
    return p.parse_args()


def main() -> None:

    _apply_process_title_normalization()
    args = parse_args()
    lease_path = str(args.shared_lease_path)

    eval_gpu = _acquire_eval_gpu(args, owner='stage1_qualitative_pack_v5_20260413')
    try:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(eval_gpu['selected_gpu_id'])
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        core_ds = _make_dataset(['vspw', 'vipseg'], 'val', str(args.stage2_contract_json), max_samples=-1)
        stage1_model = _load_stage1_with_optional_override(device, args.stage1_best_ckpt)
        stage1_pack = build_stage1_pack(args, core_ds, stage1_model, device)
    finally:
        _release_lease_safe(str(eval_gpu.get('lease_id', '')), lease_path)

    eval_gpu = _acquire_eval_gpu(args, owner='stage2_qualitative_pack_v5_20260413')
    try:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(eval_gpu['selected_gpu_id'])
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        core_ds = _make_dataset(['vspw', 'vipseg'], 'val', str(args.stage2_contract_json), max_samples=-1)
        stage2_pack = build_stage2_pack(args, core_ds, device)
        write_doc(args.qualitative_doc, stage1_pack, stage2_pack)
        print(json.dumps({'stage1_pack': str(args.stage1_pack_report), 'stage2_pack': str(args.stage2_pack_report), 'doc': str(args.qualitative_doc)}, ensure_ascii=True, indent=2))
    finally:
        _release_lease_safe(str(eval_gpu.get('lease_id', '')), lease_path)


if __name__ == '__main__':
    main()
