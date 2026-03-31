# Week2 Mini-Val Protocol V2.1

## Goal

V2.1 keeps the same hard split as V2 and tightens evaluation realism before any model scaling.

Focus:

- identity hit: point hit -> small-radius region hit
- occlusion recovery: single-frame hit -> short-window recovery
- query candidates: harder distractors, including same-class hard negatives when possible

## Fixed Setup

- Data split: unchanged from V2
  - `manifests/minisplits/stwm_week2_minival_v2.json`
  - `manifests/minisplits/stwm_week2_minival_v2_val_clip_ids.json`
- Model: `prototype_220m`
- Obs/pred: `8/8`
- Steps: `80`
- Eval interval: `20`
- Save interval: `20`
- Train/val clip caps: `32/18`

## Multi-Seed Scope

- Required runs (all seeds): `full`, `wo_semantics`, `wo_identity_memory`
- Seeds: `42`, `123`, `456`
- Optional: `wo_trajectory` (added for seed 42 visualization compatibility)

Runner:

- `scripts/run_week2_minival_v2_1_multiseed.sh`

## V2.1 Metric Knobs

- `identity_hit_radius = 0.035`
- `occlusion_recovery_window = 3`
- `query_candidates = 6`
- `query_hit_radius = 0.08`
- `query_topk = 1`
- `query_hard_negative_jitter = 0.03`

## Metric Definitions (V2.1)

- `future_mask_iou`: predicted center disk vs target-label mask IoU.
- `future_trajectory_l1`: mean absolute center error across future horizon.
- `query_localization_error`: error at query-selected frame (semantic-energy argmax), not future-frame average.
- `query_top1_acc`: target selection hit among target + distractor candidates.
- `query_hit_rate`: distance-threshold hit on query-selected frame.
- `identity_consistency`: target-label hit ratio using identity disk.
- `identity_switch_rate`: non-target foreground hit ratio using identity disk.
- `occlusion_recovery_acc`: recovery hit within a short window after target re-appearance.

## Outputs

- Runs root: `outputs/training/week2_minival_v2_1`
- Multi-seed aggregate summary:
  - `reports/week2_minival_v2_1_multiseed_summary.json`
- Failure-focused figure pack:
  - `outputs/visualizations/week2_figures_v2_1`
- Base comparison pack (seed 42):
  - `outputs/visualizations/week2_figures_v2_1/base_cases`
