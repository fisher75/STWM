# Stage2 Future Query Utility Eval V2

- scope: internal usefulness proxy, not official benchmark
- future_query_utility_improved_vs_stage1: True
- future_query_utility_improved_vs_legacysem: True
- future_query_utility_improved_vs_cropenc: True
- future_query_utility_improved_on_hard_subsets: True
- future_query_utility_improved_vs_baselines: True
- best_calibration_method: calibration_only_wave1_best
- best_baseline_method: cropenc_baseline_best

| method | run_name | loc_error | top1 | hit_rate | hard_top1 | ambiguous_top1 | small_top1 | appearance_top1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| stage1_frozen_baseline | stage1_frozen_baseline | 0.086462 | 0.0095 | 0.0116 | 0.0095 | 0.0095 | 0.0095 | 0.0095 |
| legacysem_best | stage2_fullscale_core_legacysem_seed456_wave2_20260409 | 0.000633 | 0.5682 | 1.0000 | 0.5682 | 0.5710 | 0.5682 | 0.5682 |
| cropenc_baseline_best | stage2_fullscale_core_cropenc_seed456_20260409 | 0.000572 | 0.5930 | 1.0000 | 0.5930 | 0.5964 | 0.5930 | 0.5930 |
| v7_alignment_only_best | stage2_semobjv7_alignonly_topk1_seed123_20260413 | 0.000730 | 0.5329 | 1.0000 | 0.5539 | 0.5424 | 0.5444 | 0.5374 |
| calibration_only_wave1_best | stage2_calonly_topk1_seed123_wave1_20260413 | 0.000307 | 0.7311 | 1.0000 | 0.8695 | 0.7774 | 0.8020 | 0.7579 |
| calibration_only_wave2_best | stage2_calonly_topk1_seed654_wave2_20260414 | 0.000613 | 0.5764 | 1.0000 | 0.6629 | 0.6061 | 0.6214 | 0.5936 |
| longrun_best | stage2_calonly_topk1_seed123_longconfirm_v2_20260414 | 0.000431 | 0.6593 | 1.0000 | 0.7038 | 0.6776 | 0.6833 | 0.6687 |
