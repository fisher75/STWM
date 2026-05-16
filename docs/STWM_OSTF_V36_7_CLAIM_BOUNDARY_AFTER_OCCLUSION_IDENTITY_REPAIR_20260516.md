# STWM OSTF V36.7 Claim Boundary After Occlusion Identity Repair

## 中文总结
V36.7 已把 V36.6 的 occlusion/reappear target/eval contract 修复写入 machine-checkable claim table。当前允许 bounded M128/H32 full-325 causal benchmark claim，并允许在 teacher-vis eval target contract 下报告 occlusion/reappear identity positive；仍不允许 full CVPR-scale complete system。

## 已允许 claim
- `claim_v35_49_teacher_trace_upper_bound`: V35.49 可以作为 full 325 M128/H32 raw-video-derived teacher-trace upper-bound closure。
- `claim_v36_3_causal_m128_h32_full325_benchmark`: V36.3 允许 claim M128/H32 full-325 causal video world model benchmark：past-only trace input，经 selector trace、semantic state、real-instance identity retrieval 闭环通过。
- `claim_frozen_v30_selector_trace_no_harm_prior`: V36.2c 保守 copy-default selector 在 val/test 都不伤 copy，并且赢 strongest analytic prior。
- `claim_causal_semantic_state_field`: V36.3 在 causal predicted trace 上通过三 seed future semantic state / transition / uncertainty 评估。
- `claim_real_instance_pairwise_identity_field`: V36.3 在 real-instance subset 上 pairwise identity retrieval 三 seed 通过。
- `claim_causal_occlusion_reappear_identity_eval_contract_repaired`: V36.6 修复 occlusion/reappear identity target/eval contract 后，在 real-instance subset 的 teacher-vis-defined occlusion/reappear 点上三 seed 通过。

## 仍不允许 claim
- `claim_full_cvpr_scale_complete_system`: 不能 claim full CVPR-scale complete world model success。
- `claim_v34_continuous_teacher_delta_route`: 不能回到 V34 continuous teacher embedding delta writer/gate/prototype/local expert 路线或把它包装成 semantic field success。

## 仍需披露的 reviewer 风险
- `teacher_trace_upper_bound_gap`: must_report。V35.49 仍只能作为 teacher-trace upper-bound；V36 causal predicted trace 相对 upper-bound 有 semantic gap，必须报告。
- `semantic_medium_margin`: reviewer_risk。semantic changed/hard/uncertainty 已过三 seed，但不是压倒性分数；需要保持 per-category failure atlas。
- `occlusion_target_contract_disclosure`: controlled_with_disclosure。必须披露 occlusion target 从 predicted future_vis 改为 teacher future vis eval-only target；不得让 teacher trace 进入模型输入。
- `full_cvpr_scale_not_allowed`: hard_boundary。当前仍只允许 M128/H32 full-325 causal benchmark claim，不允许 full CVPR-scale complete system。

## 关键指标
- val_occlusion_reappear_retrieval_top1: 0.9981999672721322
- test_occlusion_reappear_retrieval_top1: 0.9967759269210102
- val_occlusion_reappear_total: 2037.0
- test_occlusion_reappear_total: 1861.0
- future_teacher_trace_input_allowed: false
- future_trace_predicted_from_past_only: true
- full_cvpr_scale_claim_allowed: false

## 输出
- updated_claim_table: `reports/stwm_ostf_v36_7_machine_checkable_claim_table_20260516.json`
- decision_report: `reports/stwm_ostf_v36_7_claim_boundary_after_occlusion_identity_repair_20260516.json`
- recommended_next_step: `write_v36_release_bundle_with_causal_claim_boundary`
