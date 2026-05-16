# STWM OSTF V36.1 Trace Rollout Failure Atlas

- sample_count: 325
- global_v30_ADE_mean: 126.70885446086619
- global_strongest_prior: last_observed_copy
- global_strongest_prior_ADE_mean: 116.70810838118665
- global_v30_minus_strongest_prior_ADE: 10.000746079679544
- v30_sample_win_rate_vs_sample_strongest_prior: 0.1076923076923077
- fragile_category_count: 40
- robust_category_count: 1

## 高风险类别 Top
- dataset_vspw: sample_count=204, v30_minus_prior_ADE=32.31693443714404, strongest_prior=damped_velocity
- pseudo_identity_diagnostic: sample_count=204, v30_minus_prior_ADE=32.31693443714404, strongest_prior=damped_velocity
- pseudo_or_unknown_identity: sample_count=204, v30_minus_prior_ADE=32.31693443714404, strongest_prior=damped_velocity
- changed_absent: sample_count=8, v30_minus_prior_ADE=31.15795135498047, strongest_prior=damped_velocity
- hard_absent: sample_count=8, v30_minus_prior_ADE=31.15795135498047, strongest_prior=damped_velocity
- camera_motion_low: sample_count=109, v30_minus_prior_ADE=26.903993791396466, strongest_prior=damped_velocity
- motion_low: sample_count=109, v30_minus_prior_ADE=26.481541645636252, strongest_prior=damped_velocity
- low_motion: sample_count=162, v30_minus_prior_ADE=25.156145837571888, strongest_prior=damped_velocity
- split_test: sample_count=35, v30_minus_prior_ADE=24.975726539748052, strongest_prior=last_observed_copy
- split_val: sample_count=57, v30_minus_prior_ADE=24.46899924928492, strongest_prior=damped_velocity

## 中文总结
V36.1 failure atlas 显示 V30 causal trace rollout 在 full 325 上整体没有赢 strongest analytic prior；需要进一步看 downstream semantic/identity 是否仍优于 strongest-prior slice。
