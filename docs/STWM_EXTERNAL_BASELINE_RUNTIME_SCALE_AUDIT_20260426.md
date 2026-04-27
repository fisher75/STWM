# STWM External Baseline Runtime Scale Audit 20260426

- full eval 是否真跑了 389 items: `True`
- total_items: `389`
- per_subset_counts: `{'OOD_hard': 370, 'appearance_change': 161, 'crossing_ambiguity': 302, 'long_gap_persistence': 52, 'occlusion_reappearance': 127}`
- per_baseline_wall_time: `{'cutie': 282.189, 'sam2': 741.582, 'cotracker': 240.521}`
- per_baseline_runtime_per_item: `{'cutie': 0.7249, 'sam2': 1.8937, 'cotracker': 0.6176}`
- average_frame_count_per_item: `16.00`
- average_candidate_count: `14.58`
- reason_eval_fast: `short 16-frame local windows, 384 max-side resize, single future association target, no training/full-video benchmark`
- scale_statement: 389 items are a hard-case diagnostic set for external-baseline comparison on the STWM future identity/reacquisition protocol, not a full video benchmark.

```json
{
  "average_candidate_count": 14.580976863753213,
  "average_frame_count_per_item": 16,
  "created_at": "2026-04-27T08:30:02.409449+00:00",
  "full_eval_really_ran_389_items": true,
  "max_candidate_count": 52,
  "max_frame_count_per_item": 16,
  "median_candidate_count": 7,
  "median_frame_count_per_item": 16,
  "original_resolution_stats": {
    "available_items": 389,
    "height_max": 2160,
    "height_mean": 866.8277634961439,
    "height_min": 360,
    "width_max": 3840,
    "width_mean": 1476.2879177377893,
    "width_min": 480
  },
  "per_baseline_runtime_per_item": {
    "cotracker": 0.617604884318766,
    "cutie": 0.7249339331619538,
    "sam2": 1.893683033419023
  },
  "per_baseline_wall_time": {
    "cotracker": 240.521,
    "cutie": 282.189,
    "sam2": 741.582
  },
  "per_subset_counts": {
    "OOD_hard": 370,
    "appearance_change": 161,
    "crossing_ambiguity": 302,
    "long_gap_persistence": 52,
    "occlusion_reappearance": 127
  },
  "reason_eval_fast": [
    "389 items is a hard-case diagnostic manifest, not a full-video benchmark.",
    "Each item uses a short local window from observed prompt to future frame; average/median/max frame count are all 16.",
    "Frames are resized to max side 384 for adapter inference.",
    "Association is evaluated at one future frame by matching predicted mask/points to pre-materialized candidate masks/boxes.",
    "No training, no full-dataset video propagation, no multi-object exhaustive benchmark.",
    "CoTracker uses sparse point sampling with <=64 observed-target points rather than dense tracking.",
    "Runs used B200 GPUs and parallelized SAM2/CoTracker after Cutie finished."
  ],
  "resized_resolution_or_max_side": {
    "max_side": 384,
    "policy": "read_resize_frame downsizes each frame so max(width,height)<=max_side, preserving aspect ratio",
    "typical_1920x1200_resized_to": "384x240"
  },
  "scale_statement": "389 items are a hard-case diagnostic set for external-baseline comparison on the STWM future identity/reacquisition protocol, not a full video benchmark.",
  "total_items": 389
}
```
