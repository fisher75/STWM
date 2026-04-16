# Stage2 State-Identifiability Eval V3 20260416

- scope: real future grounding with true instance identity / future mask continuity
- official_benchmark: False
- protocol_contribution: True
- protocol_item_count: 200
- selected_device: cuda:0
- state_identifiability_protocol_v3_success: True
- future_grounding_usefulness_improved_vs_baselines: False
- future_grounding_usefulness_improved_on_hard_subsets: False
- protocol_v3_statistically_more_discriminative_than_v2: True
- protocol_v3_discriminative_enough_for_top_tier: False

| method | run_name | top1_acc | hit_rate | loc_error | top1_mask_iou | hard_top1 | ambiguity_top1 | small_top1 | appearance_top1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| stage1_frozen_baseline | stage1_frozen_baseline | 0.1800 | 0.0200 | 0.223339 | 0.1805 | 0.1800 | 0.1781 | 0.1705 | 0.1825 |
| legacysem_best | stage2_fullscale_core_legacysem_seed456_wave2_20260409 | 0.1750 | 0.0200 | 0.216021 | 0.1754 | 0.1750 | 0.1644 | 0.1591 | 0.1971 |
| cropenc_baseline_best | stage2_fullscale_core_cropenc_seed456_20260409 | 0.1650 | 0.0200 | 0.215643 | 0.1654 | 0.1650 | 0.1507 | 0.1477 | 0.1898 |
| calibration_only_mainline_best | stage2_calonly_topk1_seed123_longconfirm_v2_20260414 | 0.1650 | 0.0200 | 0.215659 | 0.1654 | 0.1650 | 0.1507 | 0.1477 | 0.1898 |
| noalign_failure | stage2_calonly_noalign_seed654_ablate_fix_v2_20260416 | 0.1650 | 0.0200 | 0.215616 | 0.1654 | 0.1650 | 0.1507 | 0.1477 | 0.1898 |
| densegate_failure | stage2_calonly_densegate_seed654_ablate_fix_v2_20260416 | 0.1650 | 0.0150 | 0.215616 | 0.1653 | 0.1650 | 0.1507 | 0.1477 | 0.1898 |
| nodelay_failure | stage2_calonly_nodelay_seed654_ablate_fix_v2_20260416 | 0.1650 | 0.0150 | 0.215615 | 0.1653 | 0.1650 | 0.1507 | 0.1477 | 0.1898 |

## Paired Bootstrap Comparisons

| comparator | top1_mean_diff | top1_ci95 | top1_win_rate | locerr_mean_diff | locerr_ci95 |
|---|---:|---|---:|---:|---|
| stage1_frozen_baseline | -0.0150 | [-0.0650, 0.0350] | 0.0600 | 0.007680 | [0.001707, 0.013532] |
| legacysem_best | -0.0100 | [-0.0300, 0.0100] | 0.0050 | 0.000363 | [-0.000644, 0.001304] |
| cropenc_baseline_best | 0.0000 | [0.0000, 0.0000] | 0.0000 | -0.000015 | [-0.000117, 0.000085] |
| noalign_failure | 0.0000 | [0.0000, 0.0000] | 0.0000 | -0.000043 | [-0.000193, 0.000102] |
| densegate_failure | 0.0000 | [-0.0150, 0.0150] | 0.0050 | -0.000042 | [-0.000654, 0.000569] |
| nodelay_failure | 0.0000 | [-0.0150, 0.0150] | 0.0050 | -0.000044 | [-0.000654, 0.000568] |
