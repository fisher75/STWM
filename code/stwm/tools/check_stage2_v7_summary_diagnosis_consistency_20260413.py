#!/usr/bin/env python3
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

WORK_ROOT = Path('/home/chen034/workspace/stwm')
SUMMARY_PATH = WORK_ROOT / 'reports/stage2_semantic_objective_redesign_v7_summary_20260413.json'
DIAGNOSIS_PATH = WORK_ROOT / 'reports/stage2_semantic_objective_redesign_v7_diagnosis_20260413.json'
REPORT_PATH = WORK_ROOT / 'reports/stage2_semantic_objective_redesign_v7_consistency_check_20260413.json'


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}


def metric_from_row(row: Dict[str, Any], key: str, default: float = 1e9) -> float:
    block = row.get('best_checkpoint_metric', {}) if isinstance(row.get('best_checkpoint_metric', {}), dict) else {}
    metrics = block.get('metrics', {}) if isinstance(block.get('metrics', {}), dict) else {}
    try:
        return float(metrics.get(key, default))
    except Exception:
        return float(default)


def sidecar_score(row: Dict[str, Any]) -> float:
    sidecar = row.get('semantic_hard_sidecar_metric', {}) if isinstance(row.get('semantic_hard_sidecar_metric', {}), dict) else {}
    try:
        return float(sidecar.get('semantic_hard_sidecar_score', 1e9))
    except Exception:
        return 1e9


def main() -> None:
    summary = read_json(SUMMARY_PATH)
    diagnosis = read_json(DIAGNOSIS_PATH)
    run_rows = [row for row in summary.get('run_rows', []) if isinstance(row, dict)]
    run_map = {str(row.get('run_name', '')): row for row in run_rows}
    failing_rules: List[str] = []
    repaired_fields: List[str] = []

    running = int(summary.get('running_count', -1))
    completed = int(summary.get('completed_count', -1))
    failed = int(summary.get('failed_count', -1))
    if running + completed + failed != len(run_rows):
        failing_rules.append('summary_counts_do_not_match_run_rows')
    else:
        repaired_fields.append('summary_counts')

    required_diag_fields = [
        'true_new_best_not_warm_start_inherited',
        'actual_gate_positive_ratio_below_0_30',
        'semantic_hard_composite_improved_vs_v6',
        'cross_seed_support_present',
        'overall_best_run_name',
        'semantic_hard_best_run_name',
        'best_effective_persistence_run_name',
        'alignment_only_is_already_sufficient',
        'persistence_branch_actually_contributed',
        'persistence_declared_but_inactive_any',
        'persistence_declared_but_inactive_all',
        'next_step_choice',
    ]
    for key in required_diag_fields:
        if key not in diagnosis or diagnosis.get(key) is None:
            failing_rules.append(f'diagnosis_missing_{key}')
        else:
            repaired_fields.append(key)

    for key in ['overall_best_run_name', 'semantic_hard_best_run_name']:
        run_name = str(diagnosis.get(key, 'none'))
        if run_name == 'none' or run_name not in run_map:
            failing_rules.append(f'diagnosis_{key}_not_in_summary')

    eff_name = str(diagnosis.get('best_effective_persistence_run_name', 'none'))
    if eff_name != 'none' and eff_name not in run_map:
        failing_rules.append('diagnosis_best_effective_persistence_run_name_not_in_summary')

    if bool(diagnosis.get('persistence_branch_actually_contributed', False)) and eff_name == 'none':
        failing_rules.append('persistence_contributed_true_but_no_effective_persistence_run')

    if bool(diagnosis.get('actual_gate_positive_ratio_below_0_30', False)):
        if not run_rows or max(float(row.get('actual_gate_positive_ratio_mean', 1.0)) for row in run_rows) >= 0.30:
            failing_rules.append('gate_ratio_claim_true_but_summary_contains_ratio_ge_0_30')

    if bool(diagnosis.get('alignment_only_is_already_sufficient', False)):
        overall = run_map.get(str(diagnosis.get('overall_best_run_name', '')), {})
        if str(overall.get('family', '')) != 'alignonly':
            failing_rules.append('alignment_only_sufficient_true_but_overall_best_not_alignonly')

    if bool(diagnosis.get('semantic_hard_composite_improved_vs_v6', False)):
        semantic_best = run_map.get(str(diagnosis.get('semantic_hard_best_run_name', '')), {})
        v6_anchor = diagnosis.get('v6_best_objective_combo_anchor', {}) if isinstance(diagnosis.get('v6_best_objective_combo_anchor', {}), dict) else {}
        v6_hard = float(v6_anchor.get('best_v6_semantic_hard_composite_score', 1e9))
        if sidecar_score(semantic_best) >= v6_hard:
            failing_rules.append('semantic_hard_improved_vs_v6_true_but_score_not_better_than_v6_anchor')

    if bool(diagnosis.get('true_new_best_not_warm_start_inherited', False)):
        overall = run_map.get(str(diagnosis.get('overall_best_run_name', '')), {})
        best_step = int((overall.get('best_checkpoint_metric', {}) or {}).get('global_step', -1) or -1)
        warm = diagnosis.get('warm_start_anchor', {}) if isinstance(diagnosis.get('warm_start_anchor', {}), dict) else {}
        seed = str(overall.get('seed', ''))
        warm_step = int((warm.get(seed, {}) or {}).get('global_step', -1) or -1)
        if best_step <= warm_step:
            failing_rules.append('true_new_best_true_but_best_step_not_above_warm_start')

    report = {
        'generated_at_utc': now_iso(),
        'summary_path': str(SUMMARY_PATH),
        'diagnosis_path': str(DIAGNOSIS_PATH),
        'passed': len(failing_rules) == 0,
        'failing_rules': failing_rules,
        'repaired_fields': sorted(set(repaired_fields)),
        'summary_counts': {
            'running_count': running,
            'completed_count': completed,
            'failed_count': failed,
            'run_row_count': len(run_rows),
        },
    }
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=True, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, ensure_ascii=True, indent=2))


if __name__ == '__main__':
    main()
