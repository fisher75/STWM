# Stage2 Semantic Objective Redesign V7 Results

- generated_at_utc: 2026-04-13T14:46:23.452452+00:00
- v7_runs_terminal: True
- running_count: 0
- completed_count: 8
- failed_count: 0
- overall_best_run_name: stage2_semobjv7_alignonly_topk1_seed123_20260413
- semantic_hard_best_run_name: stage2_semobjv7_alignpersist_topk1_seed123_20260413
- best_effective_persistence_run_name: none
- true_new_best_not_warm_start_inherited: False
- actual_gate_positive_ratio_below_0_30: True
- semantic_hard_composite_improved_vs_v6: False
- cross_seed_support_present: True
- alignment_only_is_already_sufficient: True
- persistence_branch_actually_contributed: False
- persistence_declared_but_inactive_any: True
- persistence_declared_but_inactive_all: True
- next_step_choice: alignment_only_is_true_mainline

## Final Read

- v7 的 8 个 run 真实终态: 8 completed / 0 failed / 0 running。
- overall best: `stage2_semobjv7_alignonly_topk1_seed123_20260413`。
- semantic-hard best: `stage2_semobjv7_alignpersist_topk1_seed123_20260413`。
- best effective persistence run: `none`。
- 当前修复后的 family verdict 更支持 calibration-only / alignment-only line，而不支持把 persistence-aware line 写成已被证实的主线。
- 若 semantic-hard best 来自 persistence-declared run，但 persistence telemetry 仍为 inactive，则该结果只能记为 sidecar probe，不得解释为 persistence 分支已实际起效。

## Run Table

| run_name | family | seed | status | global_step | best_endpoint_l2 | semantic_hard_composite | gate_ratio | valuable_pair_ratio | guaranteed_pair_count | declared_but_inactive |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---|
| stage2_semobjv7_alignonly_topk1_seed42_20260413 | alignonly | 42 | completed | 3360 | 0.00113695 | 0.00111109 | 0.1250 | 0.0000 | 0.0000 | False |
| stage2_semobjv7_alignonly_qcap15_seed42_20260413 | alignonly | 42 | completed | 3360 | 0.00113695 | 0.00111146 | 0.1250 | 0.0000 | 0.0000 | False |
| stage2_semobjv7_alignpersist_topk1_seed42_20260413 | alignpersist | 42 | completed | 3360 | 0.00113695 | 0.00111004 | 0.1250 | 0.0000 | 0.0000 | True |
| stage2_semobjv7_alignpersist_qcap15_seed42_20260413 | alignpersist | 42 | completed | 3360 | 0.00113695 | 0.00111054 | 0.1250 | 0.0000 | 0.0000 | True |
| stage2_semobjv7_alignonly_topk1_seed123_20260413 | alignonly | 123 | completed | 8360 | 0.00080269 | 0.00067122 | 0.1250 | 0.0000 | 0.0000 | False |
| stage2_semobjv7_alignonly_qcap15_seed123_20260413 | alignonly | 123 | completed | 8360 | 0.00080269 | 0.00067867 | 0.1250 | 0.0000 | 0.0000 | False |
| stage2_semobjv7_alignpersist_topk1_seed123_20260413 | alignpersist | 123 | completed | 8360 | 0.00080269 | 0.00066904 | 0.1250 | 0.0000 | 0.0000 | True |
| stage2_semobjv7_alignpersist_qcap15_seed123_20260413 | alignpersist | 123 | completed | 8360 | 0.00080269 | 0.00068548 | 0.1250 | 0.0000 | 0.0000 | True |
