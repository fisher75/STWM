# Week2 Mini-Val Protocol V2

## Objective

V2 targets two known week2-v1 weaknesses:

1. Identity metrics were saturated and non-discriminative.
2. Query localization error collapsed to a trajectory proxy.

## Data Protocol

- Source manifest: `manifests/minisplits/stwm_week1_mini.json`
- V2 manifest builder: `code/stwm/tools/build_minival_v2_protocol.py`
- Built artifacts:
  - `manifests/minisplits/stwm_week2_minival_v2.json`
  - `manifests/minisplits/stwm_week2_minival_v2_val_clip_ids.json`
  - `reports/week2_minival_v2_hard_selection.json`

### Hard-Clip Selection

The builder ranks VSPW clips by target-label difficulty score using:

- re-appearance/disappearance events
- motion magnitude
- area variability
- observation/future presence shift

Validation uses top `18` ranked clips with valid target label IDs.

## Target Definition

For each selected VSPW clip, V2 writes `metadata.target_label_id` into the manifest.

- `TraceAdapter` uses this target label to compute center/visibility.
- `SemanticAdapter` uses target-label area ratio for objectness signal.

This replaces the v1 behavior that treated all non-zero semantic labels as one object.

## Training/Eval Settings (this run)

- Script: `scripts/run_week2_minival_v2.sh`
- Model: `prototype_220m`
- Steps: `80`
- Eval interval: `20`
- Save interval: `20`
- Seed: `42`
- Obs/pred: `8/8`
- Train/val caps: `32/18`
- Protocol version flag: `v2`

## V2 Metrics

Kept:

- `future_mask_iou`
- `future_trajectory_l1`
- `visibility_accuracy`
- `visibility_f1`

Reworked for V2:

- `identity_consistency`
  - now computed from predicted point hitting target-label pixels
- `identity_switch_rate`
  - now measured by predicted point landing on non-target foreground labels
- `occlusion_recovery_acc`
  - now driven by target-area disappearance/re-appearance events
- `query_localization_error`
  - now computed at query-selected frame (semantic-energy argmax), not visibility-mean trajectory error
- `query_top1_acc`
  - retrieval hit over target+distractor candidate locations
- `query_hit_rate`
  - radius-hit indicator on query-selected frame

## Protocol Health Check (V2)

What improved:

- Query metric is no longer identical to trajectory mean error.
- Identity metrics are no longer pinned at exactly `1.0/0.0`.

What still needs work:

- `occlusion_recovery_acc` remained `0.0` in all four runs (event/hit definition still too strict or target extraction still noisy).
- Query retrieval setup is still easy in early steps (`query_top1_acc` near 1.0), then degrades at step 80.
