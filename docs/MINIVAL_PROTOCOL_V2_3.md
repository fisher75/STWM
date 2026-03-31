# Week2 Mini-Val Protocol V2.3

## Goal

V2.3 is the last minimal protocol hardening pass.

Scope is strictly evaluator-only, with fixed split/model/seeds, to decide whether:

1. semantic trajectory signal is stable enough for main claim
2. identity memory is strong enough to remain a primary contribution

## Fixed Setup

- Data split: unchanged hard split
  - `manifests/minisplits/stwm_week2_minival_v2.json`
  - `manifests/minisplits/stwm_week2_minival_v2_val_clip_ids.json`
- Model: `prototype_220m`
- Obs/pred: `8/8`
- Steps: `80`
- Eval interval: `20`
- Save interval: `20`
- Train/val caps: `32/18`
- Seeds: `42`, `123`, `456`
- Runs: `full`, `wo_semantics`, `wo_identity_memory`

Runner:

- `scripts/run_week2_minival_v2_3_multiseed.sh`

## V2.3 Minimal Changes (Evaluator Only)

1. Occlusion recovery -> reconnect window success
   - keep re-appearance event detection with minimum disappearance gap
   - score success by reconnect condition inside short window:
     - center-distance reconnect (`frame_error <= occlusion_reconnect_distance`) OR
     - target-overlap reconnect (`target_overlap >= occlusion_reconnect_target_overlap_min`)

2. Identity switch -> short-sequence consistency
   - replace single-frame switch interpretation by short window aggregation
   - window switch: non-target overlap dominates and target overlap does not dominate

3. Query slight hardening
   - keep hard negatives + near negatives
   - enforce minimum plausible same-class candidates (`query_min_plausible_same_class`)

## V2.3 Knobs

- `query_candidates = 8`
- `query_hit_radius = 0.08`
- `query_topk = 1`
- `query_hard_negative_jitter = 0.03`
- `query_near_negative_count = 3`
- `query_min_plausible_same_class = 2`
- `identity_hit_radius = 0.04`
- `identity_target_overlap_min = 0.02`
- `identity_other_overlap_min = 0.15`
- `identity_consistency_window = 3`
- `occlusion_recovery_window = 3`
- `occlusion_min_disappear_frames = 1`
- `occlusion_reconnect_distance = 0.18`
- `occlusion_reconnect_target_overlap_min = 0.01`

## Outputs

- Runs root:
  - `outputs/training/week2_minival_v2_3`
- Multi-seed summary:
  - `reports/week2_minival_v2_3_multiseed_summary.json`
- Paired analysis:
  - `reports/week2_minival_v2_3_paired_analysis.json`
- Failure-first figures:
  - `outputs/visualizations/week2_figures_v2_3`
  - `outputs/visualizations/week2_figures_v2_3/figure_manifest.json`

## Boundary

V2.3 is explicitly the final evaluator adjustment round.

- no V2.4/V2.5 continuation
- no model scaling decision based on unstable protocol evidence