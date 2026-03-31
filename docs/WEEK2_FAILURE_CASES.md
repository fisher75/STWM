# Week2 V2.1 Failure Cases

Failure pack root:

- `outputs/visualizations/week2_figures_v2_1`

Failure manifest:

- `outputs/visualizations/week2_figures_v2_1/figure_manifest.json`

Base comparison manifest:

- `outputs/visualizations/week2_figures_v2_1/base_cases/figure_manifest.json`

## Panels Generated

1. `full_fail_wo_identity_worse`
   - full fails, and `wo_identity_memory` fails more.
   - 4 cases selected.

2. `full_success_wo_semantics_fail`
   - full succeeds, while `wo_semantics` degrades.
   - 4 cases selected.

3. `query_hard_success_failure`
   - hard-query cases split into success/failure patterns.
   - 4 cases selected.

## Current Selected Case IDs (failure manifest)

- `2183_9Jq-7HnWa_0`
- `747_ILU2occQNYQ`
- `107_tQA8kJXlTwc`
- `1061_hWl2HQh1MG8`
- `857_NQKL_UzQCd8`
- `1041_kIXALP9plU0`
- `476_mNnY0LY1Tq0`

## Example Failure Artifacts

- `outputs/visualizations/week2_figures_v2_1/full_fail_wo_identity_worse/2183_9Jq-7HnWa_0.png`
- `outputs/visualizations/week2_figures_v2_1/full_fail_wo_identity_worse/747_ILU2occQNYQ.png`
- `outputs/visualizations/week2_figures_v2_1/full_success_wo_semantics_fail/857_NQKL_UzQCd8.png`
- `outputs/visualizations/week2_figures_v2_1/query_hard_success_failure/1061_hWl2HQh1MG8.png`

## Interpretation Notes

- The package now includes failure-first evidence instead of success-only showcases.
- Query hard cases show mixed behavior, which is useful for reviewer-facing honesty.
- Identity failure panels still indicate high switch tendencies; this aligns with current V2.1 diagnosis.
