# STWM OSTF V35.46 Per-Category Failure Atlas Decision

- per_category_failure_atlas_done: true
- atlas_ready: True
- category_count: 17
- high_risk_category_count: 4
- semantic_fragile_categories_test: 3
- identity_fragile_categories_test: 0
- m128_h32_larger_video_system_benchmark_claim_allowed: True
- full_cvpr_scale_claim_allowed: false
- recommended_next_step: run_v35_47_full_m128_h32_raw_video_closure_protocol_decision

## 中文总结
V35.46 完成类别级失败图谱。创新点没有跑偏：仍是 raw video/video-derived dense trace 到 future trace/semantic state/identity retrieval 的闭环分析，不训练新模型。当前应把 V35.46 作为 V35.47 full M128/H32 raw-video closure protocol decision 的输入，而不是直接宣称 full CVPR-scale success。

## Test split 高风险类别
- test / dataset_vipseg / sample_count=6 / risk_metrics={'semantic_changed': 0.5448331270994358}
- test / high_motion / sample_count=6 / risk_metrics={'semantic_hard': 0.5468473344002774}
- test / real_instance_identity / sample_count=6 / risk_metrics={'semantic_changed': 0.5448331270994358}

## Claim boundary
- V35.46 是 failure atlas，不是新模型训练，也不是 full CVPR-scale success。
- V35.45 的 bounded M128/H32 larger video system benchmark claim 仍可保留。
- full CVPR-scale 仍需要更大/完整 M128/H32 raw-video closure protocol 与更多真实 instance identity provenance。
