from __future__ import annotations

from pathlib import Path
import json
import math
import time
from typing import Any


REPO_ROOT = Path('/home/chen034/workspace/stwm')

INPUT_DELAYED_REPORT = REPO_ROOT / 'reports/stwm_delayed_router_mainline_seed42_report_v1.json'
INPUT_REPLACEMENT_DECISION = REPO_ROOT / 'reports/stwm_replacement_clean_matrix_seed42_final_decision_v1.json'
INPUT_SEED42_GATED_REF = REPO_ROOT / 'reports/stwm_seed42_objdiag_blindbox_readonly_v1.json'
INPUT_SEED123_GATED_REF = REPO_ROOT / 'reports/stwm_gated_challenge_seed123_final_decision_v1.json'

OUT_JSON = REPO_ROOT / 'reports/stwm_two_path_residual_promotion_decision_v1.json'
OUT_DOC = REPO_ROOT / 'docs/STWM_TWO_PATH_RESIDUAL_PROMOTION_DECISION_V1.md'


def now_ts() -> str:
    return time.strftime('%Y-%m-%d %H:%M:%S')


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        obj = json.loads(path.read_text())
    except Exception:
        return None
    return obj if isinstance(obj, dict) else None


def safe_float(v: Any) -> float | None:
    try:
        x = float(v)
    except Exception:
        return None
    if math.isnan(x) or math.isinf(x):
        return None
    return x


def metric_triplet(metrics: dict[str, Any] | None) -> tuple[float, float, float] | None:
    if not isinstance(metrics, dict):
        return None
    qloc = safe_float(metrics.get('query_localization_error'))
    qtop1 = safe_float(metrics.get('query_top1_acc'))
    l1 = safe_float(metrics.get('future_trajectory_l1'))
    if qloc is None or qtop1 is None or l1 is None:
        return None
    return (float(qloc), float(qtop1), float(l1))


def official_key(metrics: dict[str, Any] | None) -> tuple[float, float, float] | None:
    t = metric_triplet(metrics)
    if t is None:
        return None
    qloc, qtop1, l1 = t
    return (qloc, -qtop1, l1)


def cmp_official(lhs: dict[str, Any] | None, rhs: dict[str, Any] | None) -> dict[str, Any]:
    lk = official_key(lhs)
    rk = official_key(rhs)
    out: dict[str, Any] = {
        'lhs_official_key': list(lk) if lk is not None else None,
        'rhs_official_key': list(rk) if rk is not None else None,
        'lhs_beats_rhs_official': None,
        'lhs_ties_rhs_official': None,
        'delta_query_localization_error': None,
        'delta_query_top1_acc': None,
        'delta_future_trajectory_l1': None,
    }
    if lk is not None and rk is not None:
        out['lhs_beats_rhs_official'] = bool(lk < rk)
        out['lhs_ties_rhs_official'] = bool(lk == rk)

    lt = metric_triplet(lhs)
    rt = metric_triplet(rhs)
    if lt is not None and rt is not None:
        out['delta_query_localization_error'] = float(lt[0] - rt[0])
        out['delta_query_top1_acc'] = float(lt[1] - rt[1])
        out['delta_future_trajectory_l1'] = float(lt[2] - rt[2])
    return out


def pick_run_by_name(runs: list[dict[str, Any]], run_name: str) -> dict[str, Any] | None:
    for r in runs:
        if str(r.get('run_name', '')) == run_name:
            return r
    return None


def pick_run_by_role(runs: list[dict[str, Any]], role: str) -> dict[str, Any] | None:
    for r in runs:
        if str(r.get('role', '')) == role:
            return r
    return None


def fmt(v: float | None) -> str:
    if v is None:
        return 'na'
    return f'{v:.6f}'


def main() -> None:
    delayed = load_json(INPUT_DELAYED_REPORT) or {}
    replacement = load_json(INPUT_REPLACEMENT_DECISION) or {}
    blindbox = load_json(INPUT_SEED42_GATED_REF) or {}
    seed123_gated = load_json(INPUT_SEED123_GATED_REF) or {}

    delayed_runs = delayed.get('runs', []) if isinstance(delayed.get('runs', []), list) else []
    replacement_runs = replacement.get('runs', []) if isinstance(replacement.get('runs', []), list) else []

    two_path = pick_run_by_name(delayed_runs, 'two_path_residual_seed42_challenge_v1')
    delayed_residual = pick_run_by_name(delayed_runs, 'delayed_residual_router_seed42_challenge_v1')
    delayed_only = pick_run_by_name(delayed_runs, 'delayed_only_seed42_challenge_v1')

    baseline_full = pick_run_by_role(replacement_runs, 'baseline_full')
    alpha050 = pick_run_by_role(replacement_runs, 'replacement_alpha050')
    wo_object_bias = pick_run_by_role(replacement_runs, 'wo_object_bias_control')

    bb_runs = blindbox.get('runs', {}) if isinstance(blindbox.get('runs', {}), dict) else {}
    bb_gated = bb_runs.get('gated', {}) if isinstance(bb_runs.get('gated', {}), dict) else {}
    bb_ref = bb_runs.get('ref', {}) if isinstance(bb_runs.get('ref', {}), dict) else {}

    seed123_gated_metrics = (
        (((seed123_gated.get('runs', {}) if isinstance(seed123_gated.get('runs', {}), dict) else {}).get('gated_challenge', {}))
         .get('metrics', {}))
        if isinstance(seed123_gated, dict)
        else {}
    )

    two_metrics = (two_path or {}).get('selection_metrics', {})
    delayed_residual_metrics = (delayed_residual or {}).get('selection_metrics', {})
    delayed_only_metrics = (delayed_only or {}).get('selection_metrics', {})
    baseline_metrics = (baseline_full or {}).get('metrics', {})
    alpha_metrics = (alpha050 or {}).get('metrics', {})
    wo_metrics = (wo_object_bias or {}).get('metrics', {})
    gated_seed42_metrics = bb_gated.get('metrics', {}) if isinstance(bb_gated.get('metrics', {}), dict) else {}

    cmp_two_vs_delayed_only = cmp_official(two_metrics, delayed_only_metrics)
    cmp_two_vs_delayed_residual = cmp_official(two_metrics, delayed_residual_metrics)
    cmp_two_vs_full = cmp_official(two_metrics, baseline_metrics)
    cmp_two_vs_alpha = cmp_official(two_metrics, alpha_metrics)
    cmp_two_vs_wo = cmp_official(two_metrics, wo_metrics)
    cmp_two_vs_gated_seed42_ref = cmp_official(two_metrics, gated_seed42_metrics)
    cmp_alpha_vs_wo = cmp_official(alpha_metrics, wo_metrics)
    cmp_gated_seed42_vs_wo_ref = cmp_official(gated_seed42_metrics, bb_ref.get('metrics', {}))

    two_best_within_delayed3 = bool(
        delayed.get('best_run_by_official_rule', '') == 'two_path_residual_seed42_challenge_v1'
    )

    beats_full = bool(cmp_two_vs_full.get('lhs_beats_rhs_official') is True)
    beats_alpha = bool(cmp_two_vs_alpha.get('lhs_beats_rhs_official') is True)
    ties_alpha = bool(cmp_two_vs_alpha.get('lhs_ties_rhs_official') is True)
    beats_wo = bool(cmp_two_vs_wo.get('lhs_beats_rhs_official') is True)

    # Gated seed42 reference is from blindbox objdiag track; keep as reference-only.
    seed42_gated_reference_available = bool(metric_triplet(gated_seed42_metrics) is not None)
    beats_seed42_gated_reference = bool(cmp_two_vs_gated_seed42_ref.get('lhs_beats_rhs_official') is True)

    gap_to_wo_two = {
        'query_localization_error_delta': cmp_two_vs_wo.get('delta_query_localization_error'),
        'query_top1_acc_delta': cmp_two_vs_wo.get('delta_query_top1_acc'),
        'future_trajectory_l1_delta': cmp_two_vs_wo.get('delta_future_trajectory_l1'),
    }
    gap_to_wo_alpha = {
        'query_localization_error_delta': cmp_alpha_vs_wo.get('delta_query_localization_error'),
        'query_top1_acc_delta': cmp_alpha_vs_wo.get('delta_query_top1_acc'),
        'future_trajectory_l1_delta': cmp_alpha_vs_wo.get('delta_future_trajectory_l1'),
    }
    gap_to_wo_gated_seed42_ref = {
        'query_localization_error_delta': cmp_gated_seed42_vs_wo_ref.get('delta_query_localization_error'),
        'query_top1_acc_delta': cmp_gated_seed42_vs_wo_ref.get('delta_query_top1_acc'),
        'future_trajectory_l1_delta': cmp_gated_seed42_vs_wo_ref.get('delta_future_trajectory_l1'),
    }

    closer_than_alpha = False
    if (
        gap_to_wo_two['query_localization_error_delta'] is not None
        and gap_to_wo_alpha['query_localization_error_delta'] is not None
    ):
        closer_than_alpha = abs(gap_to_wo_two['query_localization_error_delta']) < abs(
            gap_to_wo_alpha['query_localization_error_delta']
        )

    closer_than_gated_seed42_ref = False
    if (
        gap_to_wo_two['query_localization_error_delta'] is not None
        and gap_to_wo_gated_seed42_ref['query_localization_error_delta'] is not None
    ):
        closer_than_gated_seed42_ref = abs(gap_to_wo_two['query_localization_error_delta']) < abs(
            gap_to_wo_gated_seed42_ref['query_localization_error_delta']
        )

    if beats_full and beats_alpha and beats_wo:
        verdict_code = 'A'
        verdict_text = 'two_path_residual 已足够强，应该直接进入 seed123 replication'
    elif two_best_within_delayed3 and beats_full and (not beats_wo):
        verdict_code = 'B'
        verdict_text = 'two_path_residual 是当前最强新主线，但仍未足够 promotion；下一步只做最小 two_path_residual seed123 challenge'
    else:
        verdict_code = 'C'
        verdict_text = 'two_path_residual 也不够，才考虑更 fancy 的组合版'

    report: dict[str, Any] = {
        'generated_at': now_ts(),
        'inputs': {
            'delayed_router_seed42_report': str(INPUT_DELAYED_REPORT),
            'replacement_seed42_final_decision': str(INPUT_REPLACEMENT_DECISION),
            'seed42_gated_reference_blindbox': str(INPUT_SEED42_GATED_REF),
            'seed123_gated_reference_optional': str(INPUT_SEED123_GATED_REF),
        },
        'run_set': {
            'two_path_residual': (two_path or {}).get('run_name', 'two_path_residual_seed42_challenge_v1'),
            'delayed_residual_router': (delayed_residual or {}).get('run_name', 'delayed_residual_router_seed42_challenge_v1'),
            'delayed_only': (delayed_only or {}).get('run_name', 'delayed_only_seed42_challenge_v1'),
            'current_full_nowarm': (baseline_full or {}).get('run_name', 'full_v4_2_seed42_fixed_nowarm_lambda1_rerun_v2'),
            'alpha050_replacement': (alpha050 or {}).get('run_name', 'full_v4_2_seed42_objbias_alpha050_replacement_v1'),
            'wo_object_bias_seed42': (wo_object_bias or {}).get('run_name', 'wo_object_bias_v4_2_seed42_control_v1'),
            'gated_seed42_reference': bb_gated.get('run_name', ''),
            'gated_seed123_reference': 'full_v4_2_seed123_objbias_gated_replacement_challenge_v1',
        },
        'metrics_snapshot': {
            'two_path_residual': two_metrics,
            'delayed_residual_router': delayed_residual_metrics,
            'delayed_only': delayed_only_metrics,
            'current_full_nowarm': baseline_metrics,
            'alpha050_replacement': alpha_metrics,
            'wo_object_bias_seed42': wo_metrics,
            'gated_seed42_reference': gated_seed42_metrics,
            'gated_seed123_reference': seed123_gated_metrics,
        },
        'official_comparisons': {
            'two_vs_delayed_only': cmp_two_vs_delayed_only,
            'two_vs_delayed_residual_router': cmp_two_vs_delayed_residual,
            'two_vs_current_full_nowarm': cmp_two_vs_full,
            'two_vs_alpha050_replacement': cmp_two_vs_alpha,
            'two_vs_wo_object_bias_seed42': cmp_two_vs_wo,
            'two_vs_gated_seed42_reference': cmp_two_vs_gated_seed42_ref,
            'alpha050_vs_wo_object_bias_seed42': cmp_alpha_vs_wo,
            'gated_seed42_reference_vs_its_ref': cmp_gated_seed42_vs_wo_ref,
        },
        'required_answers': {
            'two_path_wins_current_full_nowarm_official': beats_full,
            'two_path_wins_alpha050_official': beats_alpha,
            'two_path_ties_alpha050_official': ties_alpha,
            'seed42_gated_reference_available': seed42_gated_reference_available,
            'two_path_wins_seed42_gated_reference_official': beats_seed42_gated_reference if seed42_gated_reference_available else None,
            'two_path_wins_wo_object_bias_official': beats_wo,
            'if_not_win_wo_is_gap_closer_than_alpha': bool((not beats_wo) and closer_than_alpha),
            'if_not_win_wo_is_gap_closer_than_seed42_gated_reference': bool((not beats_wo) and closer_than_gated_seed42_ref),
            'seed42_gated_reference_scope_note': 'reference_only_objdiag_seed42_not_same_matrix',
        },
        'gap_to_wo_object_bias': {
            'two_path': gap_to_wo_two,
            'alpha050': gap_to_wo_alpha,
            'seed42_gated_reference': gap_to_wo_gated_seed42_ref,
        },
        'promotion_verdict': {
            'verdict_code': verdict_code,
            'verdict_text': verdict_text,
            'recommendation': (
                'only_minimal_two_path_residual_seed123_challenge'
                if verdict_code == 'B'
                else 'as_verdict'
            ),
        },
        'five_line_summary': {
            'line1': 'two_path_residual is complete winner inside delayed-only/delayed+residual 3-run matrix',
            'line2': 'two_path_residual beats current full_nowarm and ties alpha050 (not strict win)',
            'line3': 'two_path_residual does not beat wo_object_bias on seed42 official rule',
            'line4': 'full seed123 replication not justified yet; minimal seed123 two_path challenge is justified',
            'line5': 'unique mainline recommendation: two_path_residual as new candidate line under strict non-promotion gate',
        },
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_DOC.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(report, indent=2, ensure_ascii=False))

    lines: list[str] = []
    lines.append('# STWM Two Path Residual Promotion Decision V1')
    lines.append('')
    lines.append(f'Generated: {report["generated_at"]}')
    lines.append('')
    lines.append('## A. Inputs')
    lines.append('')
    lines.append(f'- delayed router seed42 report: {INPUT_DELAYED_REPORT}')
    lines.append(f'- replacement seed42 decision: {INPUT_REPLACEMENT_DECISION}')
    lines.append(f'- seed42 gated reference (blindbox): {INPUT_SEED42_GATED_REF}')
    lines.append(f'- seed123 gated optional reference: {INPUT_SEED123_GATED_REF}')
    lines.append('')
    lines.append('## B. Official Rule Comparisons (qloc asc, qtop1 desc, l1 asc)')
    lines.append('')
    lines.append('| comparison | two_path_beats_official | tie | d_qloc | d_qtop1 | d_l1 |')
    lines.append('|---|---|---|---:|---:|---:|')

    def add_cmp_row(name: str, cmp_obj: dict[str, Any]) -> None:
        lines.append(
            '| '
            f"{name} | {cmp_obj.get('lhs_beats_rhs_official')} | {cmp_obj.get('lhs_ties_rhs_official')} | "
            f"{fmt(safe_float(cmp_obj.get('delta_query_localization_error')))} | "
            f"{fmt(safe_float(cmp_obj.get('delta_query_top1_acc')))} | "
            f"{fmt(safe_float(cmp_obj.get('delta_future_trajectory_l1')))} |"
        )

    add_cmp_row('two vs delayed_only', cmp_two_vs_delayed_only)
    add_cmp_row('two vs delayed_residual_router', cmp_two_vs_delayed_residual)
    add_cmp_row('two vs current_full_nowarm', cmp_two_vs_full)
    add_cmp_row('two vs alpha050_replacement', cmp_two_vs_alpha)
    add_cmp_row('two vs wo_object_bias_seed42', cmp_two_vs_wo)
    add_cmp_row('two vs seed42_gated_reference', cmp_two_vs_gated_seed42_ref)

    lines.append('')
    lines.append('## C. Required Answers')
    lines.append('')
    lines.append(f"1) two_path wins current full_nowarm: {beats_full}")
    lines.append(f"2) two_path wins alpha050 replacement: {beats_alpha} (tie={ties_alpha})")
    lines.append(
        '3) two_path wins seed42 gated best reference: '
        f"{(beats_seed42_gated_reference if seed42_gated_reference_available else 'na')} "
        '(seed42 gated here is reference-only blindbox track)'
    )
    lines.append(f"4) two_path wins wo_object_bias: {beats_wo}")
    lines.append(
        '5) if not win wo_object_bias, gap is closer than alpha/gated: '
        f"alpha={bool((not beats_wo) and closer_than_alpha)}, "
        f"seed42_gated_ref={bool((not beats_wo) and closer_than_gated_seed42_ref)}"
    )
    lines.append('')
    lines.append('## D. Unique Promotion Verdict')
    lines.append('')
    lines.append(f"- verdict: {verdict_code}")
    lines.append(f"- verdict_text: {verdict_text}")
    lines.append('')

    OUT_DOC.write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
