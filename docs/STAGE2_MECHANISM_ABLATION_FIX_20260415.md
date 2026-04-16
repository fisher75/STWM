# Stage2 Mechanism Ablation Fix 20260415

- alignment_load_bearing_cross_seed: False
- sparse_gating_load_bearing_cross_seed: False
- delayed_schedule_load_bearing_cross_seed: False

| family | seed | run_name | degraded_vs_reference | anomaly_better_than_reference | reused_equivalent |
|---|---:|---|---|---|---|
| noalign | 42 | stage2_calonly_noalign_seed42_ablate_fix_20260415 | True | False | True |
| noalign | 123 | stage2_calonly_noalign_seed123_ablate_fix_20260415 | True | True | True |
| noalign | 456 | stage2_calonly_noalign_seed456_ablate_fix_20260415 | True | False | True |
| noalign | 654 | stage2_calonly_noalign_seed654_ablate_fix_20260415 | True | False | False |
| densegate | 42 | stage2_calonly_densegate_seed42_ablate_fix_20260415 | True | False | True |
| densegate | 123 | stage2_calonly_densegate_seed123_ablate_fix_20260415 | True | False | True |
| densegate | 456 | stage2_calonly_densegate_seed456_ablate_fix_20260415 | True | False | False |
| densegate | 654 | stage2_calonly_densegate_seed654_ablate_fix_20260415 | True | False | False |
| nodelay | 42 | stage2_calonly_nodelay_seed42_ablate_fix_20260415 | False | True | True |
| nodelay | 123 | stage2_calonly_nodelay_seed123_ablate_fix_20260415 | True | False | True |
| nodelay | 456 | stage2_calonly_nodelay_seed456_ablate_fix_20260415 | True | False | False |
| nodelay | 654 | stage2_calonly_nodelay_seed654_ablate_fix_20260415 | True | False | False |
