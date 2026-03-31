# Week2 Mini-Val Protocol V2.2

## Goal

V2.2 keeps the V2/V2.1 hard split and applies only minimal evaluator-level changes to improve interpretability for identity and occlusion metrics.

Focus:

- keep query and trajectory decoupled
- reduce identity false switch inflation by thresholding target/other overlaps
- make occlusion recovery events stricter and less noisy

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

Runner:

- `scripts/run_week2_minival_v2_2_multiseed.sh`

## V2.2 Metric Knobs

- `query_candidates = 7`
- `query_hit_radius = 0.08`
- `query_topk = 1`
- `query_hard_negative_jitter = 0.03`
- `query_near_negative_count = 2`
- `identity_hit_radius = 0.04`
- `identity_target_overlap_min = 0.02`
- `identity_other_overlap_min = 0.15`
- `occlusion_recovery_window = 3`
- `occlusion_min_disappear_frames = 2`

## Minimal Logic Changes vs V2.1

1. Identity hit/switch thresholding
   - target hit only if `target_overlap >= identity_target_overlap_min`
   - non-target overlap considered only if `other_overlap >= identity_other_overlap_min`
   - switch counted only when non-target overlap passes threshold and target overlap fails threshold

2. Occlusion recovery gating
   - recovery event counted only after at least `occlusion_min_disappear_frames` consecutive invisible frames, then re-appearance
   - recovery success computed in a short window (`occlusion_recovery_window`)

3. Harder query candidates
   - keep same-class hard negative when possible
   - add near-target negatives with jitter (`query_near_negative_count`)

## Outputs

- Runs root: `outputs/training/week2_minival_v2_2`
- Multi-seed aggregate summary:
  - `reports/week2_minival_v2_2_multiseed_summary.json`
- Paired analysis (seed-level + clip-level bootstrap CI):
  - `reports/week2_minival_v2_2_paired_analysis.json`
- Failure-first figure pack:
  - `outputs/visualizations/week2_figures_v2_2`
  - `outputs/visualizations/week2_figures_v2_2/figure_manifest.json`

## Scope Boundary

V2.2 is an evaluator/protocol calibration step only.

- no model architecture change
- no data expansion
- no scale-up to 1B/3B