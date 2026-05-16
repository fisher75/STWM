# STWM OSTF V36.5 Causal Failure Atlas and Reviewer-Risk Audit

## 中文总结
V36.5 reviewer-risk audit 确认：V36.3 causal M128/H32 full-325 benchmark claim 可以保留，但不能升级为 full CVPR-scale。最明确的下一步是修 occlusion/reappear identity field，而不是继续扩大 claim 或训练 V34 路线。

## Reviewer 风险矩阵
- `future_trace_teacher_upper_bound_confusion`: controlled_by_claim_boundary。V35.49 已被明确改名为 teacher-trace upper-bound；V36.3 才是 causal past-only benchmark。
- `occlusion_reappear_identity_failure`: hard_risk。该项为 0.0，不能 claim occlusion/reappear identity solved；下一步应做 occlusion-aware identity memory/reassociation 修复。
- `semantic_medium_margin`: needs_failure_boundary_explanation。semantic state field 已过三 seed，但分数不是压倒性，需要以类别图谱解释 high-motion/VIPSeg/changed/hard 边界。
- `teacher_trace_upper_bound_gap`: must_report。必须报告 V36 causal predicted trace 相对 V35.49 teacher-trace upper-bound 的 gap；不能把 upper-bound 当因果结果。
- `pseudo_identity_overclaim`: controlled。VSPW pseudo identity 仍 diagnostic-only，不进入 identity claim gate。

## 类别级风险摘录
- test / dataset_vipseg / sample_count=15 / risk={'semantic_uncertainty': 0.5489189011960419}
- test / real_instance_identity / sample_count=15 / risk={'semantic_uncertainty': 0.5489189011960419}

## Claim 边界
- 允许：M128/H32 full-325 causal video world model benchmark。
- 不允许：full CVPR-scale complete system、H64/H96、M512/M1024、full open-vocabulary semantic segmentation、occlusion/reappear identity solved。
- V35.49 只能作为 teacher-trace upper-bound，不是 causal result。

## 输出
- causal_failure_atlas_eval: `reports/stwm_ostf_v36_5_causal_failure_atlas_eval_20260516.json`
- causal_failure_atlas_decision: `reports/stwm_ostf_v36_5_causal_failure_atlas_decision_20260516.json`
- reviewer_risk_audit: `reports/stwm_ostf_v36_5_reviewer_risk_audit_20260516.json`
- log: `outputs/logs/stwm_ostf_v36_5_causal_failure_atlas_20260516.log`
- recommended_next_step: `fix_occlusion_reappear_identity_field`
