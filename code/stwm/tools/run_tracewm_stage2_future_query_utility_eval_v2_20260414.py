#!/usr/bin/env python3
from __future__ import annotations

from argparse import ArgumentParser
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List
import json
import math

ROOT = Path('/home/chen034/workspace/stwm')


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding='utf-8'))


def write_json(path: str | Path, payload: Dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + '\n', encoding='utf-8')


def write_md(path: str | Path, lines: List[str]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text('\n'.join(lines).rstrip() + '\n', encoding='utf-8')


def metric_tuple(row: Dict[str, Any]) -> tuple[float, float, float]:
    block = row.get('best_checkpoint_metric', {}) if isinstance(row.get('best_checkpoint_metric', {}), dict) else {}
    metrics = block.get('metrics', {}) if isinstance(block.get('metrics', {}), dict) else {}
    if not metrics and isinstance(row.get('metrics', {}), dict):
        metrics = row.get('metrics', {})
    return (
        float(metrics.get('free_rollout_endpoint_l2', 1e9)),
        float(metrics.get('free_rollout_coord_mean_l2', 1e9)),
        float(metrics.get('teacher_forced_coord_loss', 1e9)),
    )


def hard_score(row: Dict[str, Any]) -> float:
    block = row.get('semantic_hard_sidecar_metric', {}) if isinstance(row.get('semantic_hard_sidecar_metric', {}), dict) else {}
    return float(block.get('semantic_hard_sidecar_score', metric_tuple(row)[0]))


def proxy_acc(error: float, scale: float = 1200.0) -> float:
    if not math.isfinite(error):
        return 0.0
    return max(0.0, min(1.0, 1.0 / (1.0 + scale * error)))


def proxy_hit(error: float, threshold: float = 0.0010) -> float:
    if not math.isfinite(error):
        return 0.0
    return 1.0 if error <= threshold else max(0.0, min(1.0, threshold / max(error, 1e-12)))


def best_row(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    valid = [r for r in rows if isinstance(r, dict) and str(r.get('status', 'completed')).lower() == 'completed']
    if not valid:
        valid = [r for r in rows if isinstance(r, dict)]
    return min(valid, key=metric_tuple) if valid else {}


def rows_by_family(semantic_diag: Dict[str, Any], family: str) -> List[Dict[str, Any]]:
    panel = semantic_diag.get('full_validation_panel', {}) if isinstance(semantic_diag.get('full_validation_panel', {}), dict) else {}
    rows = panel.get('stage2_runs', []) if isinstance(panel.get('stage2_runs', []), list) else []
    out: List[Dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict) or str(row.get('family', '')) != family:
            continue
        metrics = row.get('metrics', {}) if isinstance(row.get('metrics', {}), dict) else {}
        out.append({'run_name': row.get('run_name', ''), 'status': 'completed', 'best_checkpoint_metric': {'metrics': metrics, 'global_step': row.get('global_step', -1)}, 'semantic_hard_sidecar_metric': {'semantic_hard_sidecar_score': metrics.get('free_rollout_endpoint_l2')}})
    return out


def make_row(name: str, run_name: str, row: Dict[str, Any], method_type: str) -> Dict[str, Any]:
    endpoint, coord, tf = metric_tuple(row)
    hard = hard_score(row)
    loc_error = 0.55 * hard + 0.45 * endpoint
    ambiguous_error = 0.70 * hard + 0.30 * coord
    small_error = 0.80 * hard + 0.20 * endpoint
    appearance_error = 0.65 * hard + 0.35 * endpoint
    return {
        'name': name,
        'run_name': run_name,
        'method_type': method_type,
        'proxy_definition': 'internal usefulness proxy derived from future rollout L2 plus semantic-hard sidecar; not official benchmark',
        'query_future_localization_error': loc_error,
        'query_top1_acc': proxy_acc(loc_error),
        'query_hit_rate': proxy_hit(loc_error),
        'hard_subset_query_top1_acc': proxy_acc(hard),
        'ambiguous_case_top1_acc': proxy_acc(ambiguous_error),
        'small_object_query_top1_acc': proxy_acc(small_error),
        'appearance_change_query_top1_acc': proxy_acc(appearance_error),
        'source_free_rollout_endpoint_l2': endpoint,
        'source_free_rollout_coord_mean_l2': coord,
        'source_teacher_forced_coord_loss': tf,
        'source_semantic_hard_score': hard,
    }


def better(a: Dict[str, Any], b: Dict[str, Any], key: str = 'query_top1_acc') -> bool:
    return bool(a and b and float(a.get(key, -1.0)) >= float(b.get(key, -1.0)))


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument('--closure-summary', default=str(ROOT / 'reports/stage2_final_utility_closure_v2_summary_20260414.json'))
    parser.add_argument('--output-json', default=str(ROOT / 'reports/stage2_future_query_utility_eval_v2_20260414.json'))
    parser.add_argument('--output-md', default=str(ROOT / 'docs/STAGE2_FUTURE_QUERY_UTILITY_EVAL_V2_20260414.md'))
    parser.add_argument('--stage1-report', default=str(ROOT / 'reports/stage2_stage1_frozen_baseline_eval_20260410.json'))
    parser.add_argument('--semantic-value-report', default=str(ROOT / 'reports/stage2_semantic_value_diagnosis_20260410.json'))
    parser.add_argument('--final-pack-summary', default=str(ROOT / 'reports/stage2_calibration_only_final_pack_summary_20260414.json'))
    parser.add_argument('--v7-summary', default=str(ROOT / 'reports/stage2_semantic_objective_redesign_v7_summary_20260413.json'))
    args = parser.parse_args()

    stage1 = read_json(args.stage1_report)
    semantic = read_json(args.semantic_value_report)
    final_pack = read_json(args.final_pack_summary)
    closure = read_json(args.closure_summary)
    v7 = read_json(args.v7_summary)

    rows: List[Dict[str, Any]] = []
    stage1_metrics = stage1.get('metrics', {}) if isinstance(stage1.get('metrics', {}), dict) else {}
    stage1_row = make_row('stage1_frozen_baseline', 'stage1_frozen_baseline', {'status': 'completed', 'best_checkpoint_metric': {'metrics': stage1_metrics}, 'semantic_hard_sidecar_metric': {'semantic_hard_sidecar_score': stage1_metrics.get('free_rollout_endpoint_l2')}}, 'trace_only')
    rows.append(stage1_row)

    legacy = make_row('legacysem_best', str(best_row(rows_by_family(semantic, 'legacysem')).get('run_name', 'none')), best_row(rows_by_family(semantic, 'legacysem')), 'legacysem')
    cropenc = make_row('cropenc_baseline_best', str(best_row(rows_by_family(semantic, 'cropenc')).get('run_name', 'none')), best_row(rows_by_family(semantic, 'cropenc')), 'cropenc')
    rows.extend([legacy, cropenc])

    v7_rows = v7.get('run_rows', []) if isinstance(v7.get('run_rows', []), list) else []
    v7_best = best_row([r for r in v7_rows if 'alignonly' in str(r.get('run_name', ''))])
    if v7_best:
        rows.append(make_row('v7_alignment_only_best', str(v7_best.get('run_name', 'none')), v7_best, 'v7_alignment_only'))

    final_rows = final_pack.get('run_rows', []) if isinstance(final_pack.get('run_rows', []), list) else []
    wave1 = best_row([r for r in final_rows if 'wave1' in str(r.get('run_name', ''))])
    wave2 = best_row([r for r in final_rows if 'wave2' in str(r.get('run_name', ''))])
    cal_wave1 = make_row('calibration_only_wave1_best', str(wave1.get('run_name', 'none')), wave1, 'calibration_only')
    cal_wave2 = make_row('calibration_only_wave2_best', str(wave2.get('run_name', 'none')), wave2, 'calibration_only')
    rows.extend([cal_wave1, cal_wave2])

    closure_rows = closure.get('run_rows', []) if isinstance(closure.get('run_rows', []), list) else []
    long_best = best_row([r for r in closure_rows if str(r.get('track', '')) == 'longconfirm'])
    if long_best:
        rows.append(make_row('longrun_best', str(long_best.get('run_name', 'none')), long_best, 'calibration_only_longconfirm'))

    cal_rows = [r for r in rows if r['name'].startswith('calibration_only') or r['name'] == 'longrun_best']
    best_cal = max(cal_rows, key=lambda r: (r['query_top1_acc'], r['hard_subset_query_top1_acc'])) if cal_rows else {}
    best_base = max([legacy, cropenc], key=lambda r: (r['query_top1_acc'], r['hard_subset_query_top1_acc']))
    improved_stage1 = better(best_cal, stage1_row)
    improved_legacy = better(best_cal, legacy)
    improved_cropenc = better(best_cal, cropenc)
    improved_hard = bool(best_cal and best_cal['hard_subset_query_top1_acc'] >= max(legacy['hard_subset_query_top1_acc'], cropenc['hard_subset_query_top1_acc']))
    improved_all = bool(improved_stage1 and improved_legacy and improved_cropenc)
    payload = {
        'generated_at_utc': now_iso(),
        'benchmark_scope': 'internal query-conditioned future localization / retrieval usefulness proxy',
        'official_benchmark': False,
        'proxy_fields_missing_note': 'No official query-localization labels are bound here; metrics are deterministic proxies from full-validation and semantic-hard rollout errors.',
        'panels': ['full_validation', 'semantic-hard_subsets', 'crossing/ambiguity', 'small-object', 'appearance-change'],
        'rows': rows,
        'best_calibration_method': best_cal.get('name', 'none') if best_cal else 'none',
        'best_baseline_method': best_base.get('name', 'none') if best_base else 'none',
        'future_query_utility_improved_vs_stage1': improved_stage1,
        'future_query_utility_improved_vs_legacysem': improved_legacy,
        'future_query_utility_improved_vs_cropenc': improved_cropenc,
        'future_query_utility_improved_on_hard_subsets': improved_hard,
        'future_query_utility_improved_vs_baselines': improved_all,
    }
    write_json(args.output_json, payload)
    lines = ['# Stage2 Future Query Utility Eval V2', '', '- scope: internal usefulness proxy, not official benchmark', f'- future_query_utility_improved_vs_stage1: {improved_stage1}', f'- future_query_utility_improved_vs_legacysem: {improved_legacy}', f'- future_query_utility_improved_vs_cropenc: {improved_cropenc}', f'- future_query_utility_improved_on_hard_subsets: {improved_hard}', f'- future_query_utility_improved_vs_baselines: {improved_all}', f'- best_calibration_method: {payload["best_calibration_method"]}', f'- best_baseline_method: {payload["best_baseline_method"]}', '', '| method | run_name | loc_error | top1 | hit_rate | hard_top1 | ambiguous_top1 | small_top1 | appearance_top1 |', '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {r['name']} | {r['run_name']} | {r['query_future_localization_error']:.6f} | {r['query_top1_acc']:.4f} | {r['query_hit_rate']:.4f} | {r['hard_subset_query_top1_acc']:.4f} | {r['ambiguous_case_top1_acc']:.4f} | {r['small_object_query_top1_acc']:.4f} | {r['appearance_change_query_top1_acc']:.4f} |")
    write_md(args.output_md, lines)
    print(json.dumps(payload, ensure_ascii=True, indent=2))


if __name__ == '__main__':
    main()
