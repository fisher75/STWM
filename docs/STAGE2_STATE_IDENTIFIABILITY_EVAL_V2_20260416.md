# Stage2 State-Identifiability Eval V2 20260416

- scope: real future grounding with true instance identity / future mask continuity
- official_benchmark: False
- protocol_contribution: True
- protocol_item_count: 180
- selected_device: cpu
- state_identifiability_protocol_v2_success: True
- future_grounding_usefulness_improved_vs_baselines: False
- future_grounding_usefulness_improved_on_hard_subsets: False
- protocol_v2_statistically_more_discriminative_than_v1: True
- protocol_v2_discriminative_enough_for_top_tier: False

| method | run_name | top1_acc | hit_rate | loc_error | top1_mask_iou | hard_top1 | ambiguity_top1 | small_top1 | appearance_top1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| stage1_frozen_baseline | stage1_frozen_baseline | 0.1722 | 0.0222 | 0.222414 | 0.1727 | 0.1722 | 0.1667 | 0.1635 | 0.1890 |
| legacysem_best | stage2_fullscale_core_legacysem_seed456_wave2_20260409 | 0.1778 | 0.0111 | 0.215040 | 0.1781 | 0.1778 | 0.1667 | 0.1635 | 0.2047 |
| cropenc_baseline_best | stage2_fullscale_core_cropenc_seed456_20260409 | 0.1667 | 0.0111 | 0.214829 | 0.1670 | 0.1667 | 0.1508 | 0.1509 | 0.1969 |
| calibration_only_mainline_best | stage2_calonly_topk1_seed123_longconfirm_v2_20260414 | 0.1667 | 0.0111 | 0.214831 | 0.1670 | 0.1667 | 0.1508 | 0.1509 | 0.1969 |
| noalign_failure | stage2_calonly_noalign_seed42_ablate_fix_20260415 | 0.1722 | 0.0111 | 0.214842 | 0.1727 | 0.1722 | 0.1508 | 0.1572 | 0.1969 |
| densegate_failure | stage2_calonly_densegate_seed42_ablate_fix_20260415 | 0.1667 | 0.0111 | 0.214866 | 0.1670 | 0.1667 | 0.1508 | 0.1509 | 0.1969 |
| nodelay_failure | stage2_calonly_nodelay_seed42_ablate_fix_20260415 | 0.1667 | 0.0111 | 0.214864 | 0.1670 | 0.1667 | 0.1508 | 0.1509 | 0.1969 |

## Paired Bootstrap Comparisons

| comparator | top1_mean_diff | top1_ci95 | top1_win_rate | locerr_mean_diff | locerr_ci95 |
|---|---:|---|---:|---:|---|
| stage1_frozen_baseline | -0.0056 | [-0.0557, 0.0444] | 0.0611 | 0.007583 | [0.001221, 0.014055] |
| legacysem_best | -0.0111 | [-0.0333, 0.0111] | 0.0056 | 0.000209 | [-0.000832, 0.001248] |
| cropenc_baseline_best | 0.0000 | [0.0000, 0.0000] | 0.0000 | -0.000002 | [-0.000113, 0.000105] |
| noalign_failure | -0.0056 | [-0.0167, 0.0000] | 0.0000 | 0.000011 | [-0.000347, 0.000373] |
| densegate_failure | 0.0000 | [0.0000, 0.0000] | 0.0000 | 0.000035 | [-0.000224, 0.000285] |
| nodelay_failure | 0.0000 | [0.0000, 0.0000] | 0.0000 | 0.000033 | [-0.000225, 0.000279] |
