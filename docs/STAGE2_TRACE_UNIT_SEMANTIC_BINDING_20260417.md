# Stage2 Trace-Unit Semantic Binding 20260417

- generated_at_utc: 2026-04-17T14:52:46.270627+00:00
- tusb_status: 0_running_5_completed_0_failed
- best_tusb_run_name: stage2_tusb_lite_seed123_20260417
- protocol_v3_improved_vs_current_calonly: False
- hard_subsets_improved: False
- z_sem_slower_than_z_dyn: True
- assignment_sparse_and_interpretable: False
- instance_aware_real_signal_used: False
- next_step_choice: keep_tusb_lite_direction_but_refine_instance_path

| run_name | family | seed | status | endpoint_l2 | hard_score | assign_entropy | top2_ratio | active_units | z_dyn_drift | z_sem_drift |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| stage2_tusb_lite_seed123_20260417 | tusb_lite_main | 123 | completed | 0.000528 | 0.010404 | 0.0148 | 0.0206 | 1.32 | 0.4098 | 0.0024 |
| stage2_tusb_lite_seed42_20260417 | tusb_lite_main | 42 | completed | 0.000675 | 0.023221 | 0.0096 | 0.0150 | 1.04 | 0.4314 | 0.0020 |
| stage2_tusb_lite_seed456_20260417 | tusb_lite_main | 456 | completed | 0.000572 | 0.036936 | 0.0059 | 0.0099 | 1.04 | 0.4612 | 0.0017 |
| stage2_tusb_lite_no_slowsem_seed123_20260417 | tusb_lite_ablation | 123 | completed | 0.000528 | 0.015184 | 0.0119 | 0.0172 | 1.37 | 0.4175 | 0.0608 |
| stage2_tusb_lite_no_handshake_seed123_20260417 | tusb_lite_ablation | 123 | completed | 0.000528 | 0.010422 | 0.0137 | 0.0183 | 1.29 | 0.4078 | 0.0025 |

## Protocol V3 Comparison

| method | run_name | top1_acc | hit_rate | loc_error | mask_iou | hard_top1 | ambiguity_top1 | small_top1 | appearance_top1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| stage1_frozen_baseline | stage1_frozen_baseline | 0.1800 | 0.0200 | 0.223339 | 0.1805 | 0.1800 | 0.1781 | 0.1705 | 0.1825 |
| legacysem_best | stage2_fullscale_core_legacysem_seed456_wave2_20260409 | 0.1750 | 0.0200 | 0.216021 | 0.1754 | 0.1750 | 0.1644 | 0.1591 | 0.1971 |
| cropenc_baseline_best | stage2_fullscale_core_cropenc_seed456_20260409 | 0.1650 | 0.0200 | 0.215643 | 0.1654 | 0.1650 | 0.1507 | 0.1477 | 0.1898 |
| current_calibration_only_best | stage2_calonly_topk1_seed123_wave1_20260413 | 0.1650 | 0.0200 | 0.215659 | 0.1654 | 0.1650 | 0.1507 | 0.1477 | 0.1898 |
| tusb_lite_best | stage2_tusb_lite_seed123_20260417 | 0.1650 | 0.0200 | 0.215659 | 0.1654 | 0.1650 | 0.1507 | 0.1477 | 0.1898 |
| no_slowsem_ablation | stage2_tusb_lite_no_slowsem_seed123_20260417 | 0.1650 | 0.0200 | 0.215659 | 0.1654 | 0.1650 | 0.1507 | 0.1477 | 0.1898 |
| no_handshake_ablation | stage2_tusb_lite_no_handshake_seed123_20260417 | 0.1650 | 0.0200 | 0.215659 | 0.1654 | 0.1650 | 0.1507 | 0.1477 | 0.1898 |
