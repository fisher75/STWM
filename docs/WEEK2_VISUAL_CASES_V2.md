# Week2 Visual Cases V2

## Artifact Roots

- Figure manifest: `outputs/visualizations/week2_figures_v2/figure_manifest.json`
- Figure root: `outputs/visualizations/week2_figures_v2`
- Cases per type in this pack: `4`

## Current Figure Groups

1. `full_vs_wo_semantics_mask`
   - semantic ablation mask comparison
2. `full_vs_wo_trajectory`
   - trajectory ablation path comparison
3. `full_vs_wo_identity_memory`
   - identity-memory ablation visibility/path comparison
4. `query_conditioned_cases`
   - query-conditioned behavior snapshots

## Included Case IDs

- `1041_kIXALP9plU0`
- `1061_hWl2HQh1MG8`
- `107_tQA8kJXlTwc`
- `1125_5Eplp7nV12E`

## Most Useful 8 Figures for Drafting

- `outputs/visualizations/week2_figures_v2/full_vs_wo_semantics_mask/1041_kIXALP9plU0.png`
- `outputs/visualizations/week2_figures_v2/full_vs_wo_semantics_mask/1061_hWl2HQh1MG8.png`
- `outputs/visualizations/week2_figures_v2/full_vs_wo_trajectory/1041_kIXALP9plU0.png`
- `outputs/visualizations/week2_figures_v2/full_vs_wo_trajectory/1061_hWl2HQh1MG8.png`
- `outputs/visualizations/week2_figures_v2/full_vs_wo_identity_memory/1041_kIXALP9plU0.png`
- `outputs/visualizations/week2_figures_v2/full_vs_wo_identity_memory/1061_hWl2HQh1MG8.png`
- `outputs/visualizations/week2_figures_v2/query_conditioned_cases/1041_kIXALP9plU0.png`
- `outputs/visualizations/week2_figures_v2/query_conditioned_cases/1061_hWl2HQh1MG8.png`

## What V2 Figures Support

- The ablations now show clearer behavior differences on target-label-aware cases.
- Query-conditioned panel is now tied to query-specific metrics (`query_top1_acc`, `query_hit_rate`).
- Identity-memory panel can be aligned with stronger trajectory/query degradation in `wo_identity_memory`.

## Remaining Gaps

- Need explicit failure panels for high switch-rate/low consistency cases.
- Need per-case annotation with target label ID and occlusion events from case JSON.

## Immediate Next Visual Upgrade

1. Expand to 8-12 clips from V2 hard ranking top set.
2. Add one row per case showing target-label mask and predicted query frame.
3. Add a failure-only appendix panel for identity switches.
