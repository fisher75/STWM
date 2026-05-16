# STWM OSTF V36.4 Causal Claim Boundary and Packaging Audit

## 中文总结
V36.4 将 V35.49 改名为 teacher-trace upper-bound，并确认 V36.3 才是 causal past-only M128/H32 full-325 benchmark。当前允许 bounded M128/H32 causal video world model benchmark claim；不允许 full CVPR-scale claim。occlusion/reappear identity top1=0.0 是硬风险，必须进入下一轮 failure atlas/reviewer-risk audit。

## 允许 claim
- `claim_v35_49_teacher_trace_upper_bound`: V35.49 可以作为 full 325 M128/H32 raw-video-derived teacher-trace upper-bound closure。 证据: `reports/stwm_ostf_v36_v35_49_causal_trace_contract_audit_20260516.json`
- `claim_v36_3_causal_m128_h32_full325_benchmark`: V36.3 允许 claim M128/H32 full-325 causal video world model benchmark：past-only trace input，经 selector trace、semantic state、real-instance identity retrieval 闭环通过。 证据: `reports/stwm_ostf_v36_3_full_325_causal_benchmark_rerun_decision_20260516.json`
- `claim_frozen_v30_selector_trace_no_harm_prior`: V36.2c 保守 copy-default selector 在 val/test 都不伤 copy，并且赢 strongest analytic prior。 证据: `reports/stwm_ostf_v36_2c_conservative_copy_default_selector_rollout_20260516.json`
- `claim_causal_semantic_state_field`: V36.3 在 causal predicted trace 上通过三 seed future semantic state / transition / uncertainty 评估。 证据: `reports/stwm_ostf_v36_3_full_325_causal_benchmark_rerun_decision_20260516.json`
- `claim_real_instance_pairwise_identity_field`: V36.3 在 real-instance subset 上 pairwise identity retrieval 三 seed 通过。 证据: `reports/stwm_ostf_v36_3_full_325_causal_benchmark_rerun_decision_20260516.json`

## 不允许 claim
- `claim_occlusion_reappear_identity_solved`: 不能 claim occlusion/reappear identity 已解决。 当前值: `0.0`
- `claim_full_cvpr_scale_complete_system`: 不能 claim full CVPR-scale complete world model success。 当前值: `False`
- `claim_v34_continuous_teacher_delta_route`: 不能回到 V34 continuous teacher embedding delta writer/gate/prototype/local expert 路线或把它包装成 semantic field success。 当前值: `V35/V36 已改为 observed-predictable semantic state targets`

## 关键剩余风险
- `occlusion_reappear_identity`: overall identity retrieval 很高，但 occlusion/reappear top1=0.0；必须做专门 failure atlas 和修复路线。
- `semantic_medium_margin`: semantic state field 已过 gate，但 hard/changed 分数不是压倒性，需要 category atlas 解释成功/失败边界。
- `teacher_trace_upper_bound_gap`: 必须主动说明 V35.49 是 teacher-trace upper-bound，V36.3 是因果版本，两者存在 gap。
- `packaging_vs_claim`: claim 成立与 release 包完整是两件事；本审计将二者拆开。

## Artifact / Packaging 检查
- artifact_packaging_complete_for_v36_claim: True
- v36_2c_downstream_slice_npz_counts: {'train': 233, 'val': 57, 'test': 35, 'all': 325}
- missing_required_artifacts: 0
- `reports/stwm_ostf_v35_49_full_325_raw_video_closure_decision_20260516.json`: exists=True, kind=report_json, file_count=None
- `reports/stwm_ostf_v35_49_full_325_raw_video_closure_benchmark_eval_summary_20260516.json`: exists=True, kind=report_json, file_count=None
- `reports/stwm_ostf_v35_49_full_325_raw_video_closure_visualization_manifest_20260516.json`: exists=True, kind=visualization_manifest_json, file_count=None
- `reports/stwm_ostf_v35_49_full_325_per_category_failure_atlas_eval_20260516.json`: exists=True, kind=failure_atlas_json, file_count=None
- `outputs/cache/stwm_ostf_v35_49_full_325_raw_video_closure_manifest/manifest.json`: exists=True, kind=manifest_json, file_count=None
- `outputs/logs/stwm_ostf_v35_49_full_325_raw_video_frontend_rerun_20260516.log`: exists=True, kind=log, file_count=None
- `outputs/figures/stwm_ostf_v35_49_full_325_raw_video_closure`: exists=True, kind=figure_dir, file_count=12
- `reports/stwm_ostf_v36_v35_49_causal_trace_contract_audit_20260516.json`: exists=True, kind=report_json, file_count=None
- `reports/stwm_ostf_v36_past_only_observed_trace_input_build_20260516.json`: exists=True, kind=report_json, file_count=None
- `reports/stwm_ostf_v36_v30_past_only_future_trace_rollout_20260516.json`: exists=True, kind=report_json, file_count=None
- `reports/stwm_ostf_v36_causal_unified_semantic_identity_slice_build_20260516.json`: exists=True, kind=report_json, file_count=None
- `reports/stwm_ostf_v36_decision_20260516.json`: exists=True, kind=report_json, file_count=None
- `reports/stwm_ostf_v36_causal_past_only_world_model_visualization_manifest_20260516.json`: exists=True, kind=visualization_manifest_json, file_count=None
- `reports/stwm_ostf_v36_1_trace_rollout_failure_atlas_20260516.json`: exists=True, kind=report_json, file_count=None
- `reports/stwm_ostf_v36_1_strongest_prior_downstream_baseline_decision_20260516.json`: exists=True, kind=report_json, file_count=None
- `reports/stwm_ostf_v36_1_decision_20260516.json`: exists=True, kind=report_json, file_count=None
- `reports/stwm_ostf_v36_2c_conservative_copy_default_selector_rollout_20260516.json`: exists=True, kind=report_json, file_count=None
- `reports/stwm_ostf_v36_2c_conservative_selector_downstream_slice_build_20260516.json`: exists=True, kind=report_json, file_count=None
- `reports/stwm_ostf_v36_2c_conservative_selector_downstream_gate_decision_20260516.json`: exists=True, kind=report_json, file_count=None
- `outputs/cache/stwm_ostf_v36_2c_conservative_selector_downstream_slice/M128_H32`: exists=True, kind=cache_dir, file_count=325
- `reports/stwm_ostf_v36_3_full_325_causal_benchmark_rerun_eval_summary_20260516.json`: exists=True, kind=report_json, file_count=None
- `reports/stwm_ostf_v36_3_full_325_causal_benchmark_rerun_decision_20260516.json`: exists=True, kind=report_json, file_count=None

## Claim 边界
- V35.49 只能称为 teacher-trace upper-bound closure，因为 CoTracker/full-clip frontend 看见 future frames。
- V36.3 才能称为 causal past-only M128/H32 full-325 benchmark，因为 future trace 来自 past-only observed trace 的 selector rollout。
- 当前不能 claim full CVPR-scale complete system、H64/H96、M512/M1024、任意 horizon、full open-vocabulary dense segmentation 或 occlusion/reappear identity solved。

## 输出
- machine_checkable_claim_table: `reports/stwm_ostf_v36_4_machine_checkable_claim_table_20260516.json`
- audit_report: `reports/stwm_ostf_v36_4_causal_claim_boundary_and_packaging_audit_20260516.json`
- log: `outputs/logs/stwm_ostf_v36_4_causal_claim_boundary_and_packaging_audit_20260516.log`
- recommended_next_step: `run_v36_5_occlusion_reappear_identity_and_reviewer_risk_audit`
