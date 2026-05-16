# STWM OSTF V36.8 Release Bundle with Causal Claim Boundary

## 中文总结
V36.8 已打包 V36.3 causal benchmark、V36.6 occlusion identity repair 和 V36.7 claim table。当前允许 bounded M128/H32 full-325 causal video world model benchmark claim；仍不允许 full CVPR-scale complete system claim。

## 允许 claim
- `claim_v35_49_teacher_trace_upper_bound`: V35.49 可以作为 full 325 M128/H32 raw-video-derived teacher-trace upper-bound closure。
- `claim_v36_3_causal_m128_h32_full325_benchmark`: V36.3 允许 claim M128/H32 full-325 causal video world model benchmark：past-only trace input，经 selector trace、semantic state、real-instance identity retrieval 闭环通过。
- `claim_frozen_v30_selector_trace_no_harm_prior`: V36.2c 保守 copy-default selector 在 val/test 都不伤 copy，并且赢 strongest analytic prior。
- `claim_causal_semantic_state_field`: V36.3 在 causal predicted trace 上通过三 seed future semantic state / transition / uncertainty 评估。
- `claim_real_instance_pairwise_identity_field`: V36.3 在 real-instance subset 上 pairwise identity retrieval 三 seed 通过。
- `claim_causal_occlusion_reappear_identity_eval_contract_repaired`: V36.6 修复 occlusion/reappear identity target/eval contract 后，在 real-instance subset 的 teacher-vis-defined occlusion/reappear 点上三 seed 通过。

## 不允许 claim
- `claim_full_cvpr_scale_complete_system`: 不能 claim full CVPR-scale complete world model success。
- `claim_v34_continuous_teacher_delta_route`: 不能回到 V34 continuous teacher embedding delta writer/gate/prototype/local expert 路线或把它包装成 semantic field success。

## Release bundle 状态
- v36_release_bundle_ready: True
- selected_clip_count: 325
- causal_slice_counts: {'train': 233, 'val': 57, 'test': 35, 'all': 325}
- occlusion_override_counts: {'train': 233, 'val': 57, 'test': 35, 'all': 325}
- artifact_missing_count: 0
- 无缺失 required artifacts。

## Claim 边界
- V35.49 只能作为 teacher-trace upper-bound。
- V36.3 是 causal past-only M128/H32 full-325 benchmark。
- V36.6 的 occlusion/reappear target 来自 teacher future visibility，仅作为 eval/supervision target，不作为输入。
- full_cvpr_scale_claim_allowed: false

## 输出
- frozen_causal_claim_boundary_manifest: `reports/stwm_ostf_v36_8_frozen_causal_claim_boundary_manifest_20260516.json`
- non_paper_release_bundle_index: `reports/stwm_ostf_v36_8_non_paper_release_bundle_index_20260516.json`
- release_report: `reports/stwm_ostf_v36_8_release_bundle_with_causal_claim_boundary_20260516.json`
- log: `outputs/logs/stwm_ostf_v36_8_release_bundle_with_causal_claim_boundary_20260516.log`
- recommended_next_step: `independent_environment_replay_or_prepare_result_section_from_v36_bundle`
