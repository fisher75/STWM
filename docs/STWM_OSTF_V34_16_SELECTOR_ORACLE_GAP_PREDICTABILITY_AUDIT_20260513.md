# V34.16 selector oracle gap 可学习性审计中文报告

- 中文结论: `V34.16 selector oracle gap 可学习性审计完成：重点判断 oracle best timestep 是否是稳定、可学习的 observed-only 标签，以及 V34.15 top-1 CE 为什么没有改善 gap。`
- oracle_timestep_label_ambiguous: `True`
- top1_timestep_ce_hurt_selector: `True`
- best_current_selector: `v34_14_horizon_conditioned_soft_reader`
- recommended_fix: `不要再把 oracle timestep 当硬 top-1 标签；保留 V34.14 soft horizon-conditioned reader，下一步应做 multi-evidence/top-k memory set 或 calibration，而不是 learned gate。`
- recommended_next_step: `fix_nonoracle_measurement_selector_with_multievidence_memory`
