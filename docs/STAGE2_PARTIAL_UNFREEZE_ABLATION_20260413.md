# Stage2 Partial-Unfreeze Ablation

- generated_at_utc: 2026-04-14T09:25:35.094631+00:00
- status: completed
- winning_calibration_family: topk1
- frozen_calibration_best_run_name: stage2_calonly_topk1_seed123_wave1_20260413
- partial_unfreeze_beats_frozen_calibration: False
- forgetting_or_instability_detected: True

| run_name | seed | status | endpoint_l2 | trainable_parameter_count_delta | frozen_anchor_run_name |
|---|---:|---|---:|---:|---|
| stage2_partialunfreeze_topblock_seed42_20260413 | 42 | completed | 0.000675 | 15942528 | stage2_calonly_topk1_seed42_wave1_20260413 |
| stage2_partialunfreeze_topblock_seed123_20260413 | 123 | completed | 0.000528 | 15942528 | stage2_calonly_topk1_seed123_wave1_20260413 |
