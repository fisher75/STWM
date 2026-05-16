# STWM OSTF V35.51 External Comparison and Reviewer-Risk Audit

## 中文总结
V35.51 完成 reviewer-risk 与 baseline 对比审计：纯 trace、copy/persistence、semantic-only、identity-only 都不能单独构成完整系统；V35.49/V35.50 的新增证据是 full 325 raw-video closure 下的联合 trace+semantic state+real-instance identity retrieval 闭环。主要 reviewer 风险已被 artifact、atlas、pseudo exclusion 和 leakage audit 缓解，但 claim 必须保持 bounded M128/H32。

## Baseline / Component 对比
- pure_trace_v30_only: necessary_but_not_sufficient。V30 是闭环系统的可靠 future trace backbone，但纯 trace 只解决 future object-dense trace，不输出 semantic state 或 pairwise identity retrieval。
- copy_persistence_semantic_baseline: beaten_on_state_tasks_while_preserved_on_stable。copy/persistence 对 stable token 很强，V35 保留 stable preservation；但 changed/hard/uncertainty 是状态预测任务，V35.49 三 seed 通过，说明不是只靠 stable copy。
- semantic_only_v35_21: component_passed_but_not_complete_system_alone。V35.21 证明 semantic adapter 三 seed 可复现；V35.49 把它放入 raw-video full 325 closure，并与 identity retrieval 联合评估。
- identity_only_v35_29: component_passed_but_not_complete_system_alone。identity 从 same-instance pointwise BCE 改成 pairwise retrieval / contrastive field 后，真实 instance subset 在 full 325 closure 中三 seed 通过。
- pseudo_identity_diagnostic: excluded_from_claim_gate。pseudo identity 可帮助观察系统行为，但不进入 identity claim gate；真实 claim 只看 real-instance subset。
- v34_continuous_teacher_delta_route: not_system_contribution。V35 的贡献不是 continuous teacher embedding delta writer/gate/prototype/local expert；该路线已作为负结果被停止。

## Reviewer 风险矩阵
- frontend_is_just_old_cache: mitigated / high。full 325 从 raw frame paths / predecode 重跑 frontend，旧 trace cache 只用于 drift comparison。
- semantic_is_only_copy: mitigated_but_claim_limited / medium。stable copy 很强且被保留；但 changed/hard/uncertainty 三 seed 过，说明系统不只是输出 copy。
- identity_depends_on_pseudo_labels: mitigated / high。VSPW pseudo slot identity 明确 diagnostic-only；identity claim 只使用 real-instance subset。
- future_teacher_embedding_leakage: mitigated / high。V35.49/V35.50 均记录 future_leakage_detected=false，future teacher embedding 不作为 input。
- teacher_or_external_tool_is_method: mitigated_but_wording_sensitive / medium。teacher/DINO/CLIP/SAM2/CoTracker 只作为 frontend/measurement/supervision/source；STWM 方法主线是 trace→future trace→semantic state/identity retrieval。
- failure_cases_hidden: mitigated / medium。full 325 atlas_ready=true，high_risk_category_count=0；case-mined visualization 覆盖 changed/hard/identity 成败、occlusion/crossing/confuser/high motion。
- scale_overclaim: active_boundary / high。当前只允许 full 325 M128/H32 video-system benchmark claim，full_cvpr_scale_claim_allowed=false。

## 贡献边界
- 正向贡献: V35 的核心贡献边界是：video-derived dense trace / raw-video frontend rerun → frozen V30 M128 future trace → 可观测可预测 semantic state / transition / uncertainty field → real-instance pairwise identity retrieval field。
- 负向边界: 不是 V34 continuous teacher embedding delta writer/gate/prototype/local expert；不是 teacher/prototype-only；不是 open-vocabulary dense segmentation；不是任意尺度系统。

- reviewer_risk_audit_passed: True
- m128_h32_full_325_video_system_benchmark_claim_allowed: True
- full_cvpr_scale_claim_allowed: false
- recommended_next_step: prepare_v35_52_reproducibility_package_and_benchmark_card
