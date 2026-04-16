# Stage2 State-Identifiability Eval 20260415

- scope: real future grounding with true instance identity / future mask continuity
- official_benchmark: False
- protocol_item_count: 38
- selected_device: cuda
- state_identifiability_protocol_success: True
- future_grounding_usefulness_improved_vs_stage1: True
- future_grounding_usefulness_improved_vs_legacysem: True
- future_grounding_usefulness_improved_vs_cropenc: True
- future_grounding_usefulness_improved_vs_baselines: True
- future_grounding_usefulness_improved_on_hard_subsets: True

| method | run_name | top1_acc | hit_rate | loc_error | top1_mask_iou | hard_top1 | ambiguity_top1 | small_top1 | appearance_top1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| stage1_frozen_baseline | stage1_frozen_baseline | 0.1579 | 0.0000 | 0.218046 | 0.1579 | 0.1579 | 0.0968 | 0.1176 | 0.1111 |
| legacysem_best | stage2_fullscale_core_legacysem_seed456_wave2_20260409 | 0.1842 | 0.0263 | 0.199258 | 0.1843 | 0.1842 | 0.1613 | 0.1471 | 0.2222 |
| cropenc_baseline_best | stage2_fullscale_core_cropenc_seed456_20260409 | 0.1842 | 0.0263 | 0.198648 | 0.1843 | 0.1842 | 0.1613 | 0.1471 | 0.2222 |
| calibration_only_mainline_best | stage2_calonly_topk1_seed123_longconfirm_v2_20260414 | 0.1842 | 0.0263 | 0.198537 | 0.1843 | 0.1842 | 0.1613 | 0.1471 | 0.2222 |
| noalign_failure | stage2_calonly_noalign_seed321_ablate_v2_20260414 | 0.1842 | 0.0263 | 0.198644 | 0.1843 | 0.1842 | 0.1613 | 0.1471 | 0.2222 |
| densegate_failure | stage2_calonly_densegate_seed789_ablate_v2_20260414 | 0.1842 | 0.0263 | 0.198380 | 0.1843 | 0.1842 | 0.1613 | 0.1471 | 0.2222 |
| nodelay_failure | stage2_calonly_nodelay_seed789_ablate_v2_20260414 | 0.1842 | 0.0263 | 0.198380 | 0.1843 | 0.1842 | 0.1613 | 0.1471 | 0.2222 |
