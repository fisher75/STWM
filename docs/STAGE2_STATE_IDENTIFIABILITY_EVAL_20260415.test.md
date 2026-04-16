# Stage2 State-Identifiability Eval 20260415

- scope: real future grounding with true instance identity / future mask continuity
- official_benchmark: False
- protocol_item_count: 29
- selected_device: cuda
- state_identifiability_protocol_success: True
- future_grounding_usefulness_improved_vs_stage1: False
- future_grounding_usefulness_improved_vs_legacysem: True
- future_grounding_usefulness_improved_vs_cropenc: True
- future_grounding_usefulness_improved_vs_baselines: False
- future_grounding_usefulness_improved_on_hard_subsets: False

| method | run_name | top1_acc | hit_rate | loc_error | top1_mask_iou | hard_top1 | ambiguity_top1 | small_top1 | appearance_top1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| stage1_frozen_baseline | stage1_frozen_baseline | 0.2069 | 0.0000 | 0.224565 | 0.2069 | 0.2069 | 0.1667 | 0.1905 | 0.1053 |
| legacysem_best | stage2_fullscale_core_legacysem_seed456_wave2_20260409 | 0.1724 | 0.0345 | 0.212797 | 0.1726 | 0.1724 | 0.1667 | 0.1429 | 0.1579 |
| cropenc_baseline_best | stage2_fullscale_core_cropenc_seed456_20260409 | 0.1724 | 0.0345 | 0.210373 | 0.1726 | 0.1724 | 0.1667 | 0.1429 | 0.1579 |
| calibration_only_mainline_best | stage2_calonly_topk1_seed123_longconfirm_v2_20260414 | 0.1724 | 0.0345 | 0.210389 | 0.1726 | 0.1724 | 0.1667 | 0.1429 | 0.1579 |
| noalign_failure | stage2_calonly_noalign_seed321_ablate_v2_20260414 | 0.1724 | 0.0345 | 0.210080 | 0.1726 | 0.1724 | 0.1667 | 0.1429 | 0.1579 |
| densegate_failure | stage2_calonly_densegate_seed789_ablate_v2_20260414 | 0.1724 | 0.0345 | 0.209135 | 0.1726 | 0.1724 | 0.1667 | 0.1429 | 0.1579 |
| nodelay_failure | stage2_calonly_nodelay_seed789_ablate_v2_20260414 | 0.1724 | 0.0345 | 0.209135 | 0.1726 | 0.1724 | 0.1667 | 0.1429 | 0.1579 |
