# Week2 Visual Cases (Figure Pack Notes)

## Source

- Manifest: `outputs/visualizations/week2_figures/figure_manifest.json`
- Cases per type in this pack: `2`
- Current case IDs: `1041_kIXALP9plU0`, `1061_hWl2HQh1MG8`

## Figure Groups and Meaning

### 1) Semantic mask comparison

- Folder: `outputs/visualizations/week2_figures/full_vs_wo_semantics_mask`
- Files:
  - `1041_kIXALP9plU0.png`
  - `1061_hWl2HQh1MG8.png`
- What it demonstrates:
  - Visual impact of removing semantics on future mask quality and localization coherence.
  - Supports claim that semantic conditioning improves future state interpretation.

### 2) Trajectory comparison

- Folder: `outputs/visualizations/week2_figures/full_vs_wo_trajectory`
- Files:
  - `1041_kIXALP9plU0.png`
  - `1061_hWl2HQh1MG8.png`
- What it demonstrates:
  - Effect of disabling trajectory tokens on motion extrapolation quality.
  - Supports claim that trajectory channel is essential for predictive dynamics.

### 3) Identity recovery comparison

- Folder: `outputs/visualizations/week2_figures/full_vs_wo_identity_memory`
- Files:
  - `1041_kIXALP9plU0.png`
  - `1061_hWl2HQh1MG8.png`
- What it demonstrates:
  - Qualitative behavior difference when identity memory is removed.
  - Useful for discussing re-identification continuity and failure patterns.

### 4) Query-conditioned examples

- Folder: `outputs/visualizations/week2_figures/query_conditioned_cases`
- Files:
  - `1041_kIXALP9plU0.png`
  - `1061_hWl2HQh1MG8.png`
- What it demonstrates:
  - Text/query-conditioned localization behavior over predicted horizon.
  - Connects world-state forecasting with open-vocabulary query utility.

## Most Paper-Relevant Current Picks (8 images)

- `full_vs_wo_semantics_mask/1041_kIXALP9plU0.png`
- `full_vs_wo_semantics_mask/1061_hWl2HQh1MG8.png`
- `full_vs_wo_trajectory/1041_kIXALP9plU0.png`
- `full_vs_wo_trajectory/1061_hWl2HQh1MG8.png`
- `full_vs_wo_identity_memory/1041_kIXALP9plU0.png`
- `full_vs_wo_identity_memory/1061_hWl2HQh1MG8.png`
- `query_conditioned_cases/1041_kIXALP9plU0.png`
- `query_conditioned_cases/1061_hWl2HQh1MG8.png`

## What These Figures Support vs What They Do Not Yet Prove

Supported now:

- Semantics and trajectory ablations have visible effects consistent with metric drops.
- Query-conditioned outputs provide narrative bridge from forecasting to semantic retrieval.

Not yet fully proven by current pack:

- Strong identity-memory superiority in hard occlusion/re-entry scenarios.
- Robustness across broader case diversity (only 2 cases per type currently).

## Recommended Next Figure Iteration (Targeted, Not Full Rerun)

1. Expand each figure type from 2 to 6-10 clips from existing case JSON outputs.
2. Prioritize clips with occlusion and re-appearance events.
3. Add one failure-focused panel per figure type to make limitations explicit.
