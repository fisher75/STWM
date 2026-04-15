# Stage2 Final Utility Closure V2 Results

- status: completed
- mainline_still_calibration_only: True
- 6_seed_support_still_valid: True
- mechanism_ablation_cross_seed_support: False
- alignment_load_bearing: False
- sparse_gating_load_bearing: False
- delayed_schedule_load_bearing: False
- longrun_produces_new_best: True
- future_query_utility_improved_vs_baselines: True
- future_query_utility_improved_on_hard_subsets: True
- qualitative_pack_ready_for_human_figure_selection: True
- aux_probe_is_only_auxiliary: True
- current_stage2_ready_to_freeze: False
- next_step_choice: run_one_more_targeted_ablation_fix

| run_name | track | ablation | status | endpoint | hard_score |
|---|---|---|---|---:|---:|
| stage2_calonly_noalign_seed789_ablate_v2_20260414 | ablation | noalign | completed | 0.000604 | 0.000193 |
| stage2_calonly_noalign_seed321_ablate_v2_20260414 | ablation | noalign | completed | 0.000807 | 0.000390 |
| stage2_calonly_densegate_seed789_ablate_v2_20260414 | ablation | densegate | completed | 0.000987 | 0.000649 |
| stage2_calonly_densegate_seed321_ablate_v2_20260414 | ablation | densegate | completed | 0.000596 | 0.000163 |
| stage2_calonly_nodelay_seed789_ablate_v2_20260414 | ablation | nodelay | completed | 0.000987 | 0.000594 |
| stage2_calonly_nodelay_seed321_ablate_v2_20260414 | ablation | nodelay | completed | 0.000597 | 0.000163 |
| stage2_calonly_topk1_seed123_longconfirm_v2_20260414 | longconfirm | none | completed | 0.000528 | 0.000351 |
| stage2_calonly_topk1_seed321_longconfirm_v2_20260414 | longconfirm | none | completed | 0.000596 | 0.000163 |
