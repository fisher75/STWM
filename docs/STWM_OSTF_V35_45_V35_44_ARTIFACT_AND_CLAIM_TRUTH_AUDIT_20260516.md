# STWM OSTF V35.45 V35.44 Artifact And Claim Truth Audit

- artifact_truth_audit_done: true
- v35_34_json_missing: False
- v35_38_json_missing: False
- v35_42_json_missing: False
- v35_43_json_missing: False
- v35_44_depends_on_missing_json: False
- artifact_packaging_fixed_required: False
- bounded_m128_h32_claim_only: True
- full_cvpr_scale_claim_allowed: false
- recommended_fix: 无需补齐；继续扩大 V35.45 raw-video closure benchmark

## 中文总结
本地 live repo 中 V35.34/V35.38/V35.42/V35.43 依赖 JSON 均存在；V35.44 是 bounded M128/H32 claim，不是 full CVPR-scale claim。
