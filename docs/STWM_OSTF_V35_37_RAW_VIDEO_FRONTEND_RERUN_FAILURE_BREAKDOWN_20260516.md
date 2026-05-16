# STWM OSTF V35.37 Raw-Video Rerun Failure Breakdown

- v35_37_failure_breakdown_done: true
- raw_frontend_drift_ok: True
- split_counts: {'test': 1, 'train': 8, 'val': 3}
- eval_split_balance_issue_detected: True
- semantic_failure_detected: True
- identity_failure_detected: True
- m128_h32_video_system_benchmark_claim_allowed: false
- full_cvpr_scale_claim_allowed: false
- recommended_next_step: fix_raw_video_rerun_eval_split_balance_and_target_alignment

## 中文总结
V35.37 归因显示 raw-video frontend 重跑本身可复现，当前 blocker 是扩展 subset 的评估组成和 target/model 对齐：semantic 有 seed 级别 test 泛化失败，identity 主要卡在 val instance-pooled retrieval；不能把问题归因成 trace drift。

## 好消息/坏消息
- 好消息：rerun trace 与 cache 的 shape、frame path、visibility、confidence、motion drift 都对齐，raw frontend reproducibility 没有暴露硬错误。
- 坏消息：扩展 subset 只有 1 个 test clip、3 个 val clip，评估统计过小且类别分布偏，semantic seed123 与 identity instance-pooled val gate 暴露不稳。

## Claim Boundary
当前只能说 raw frontend 小规模复现链路可运行；不能 claim M128/H32 video system benchmark，更不能 claim full CVPR-scale complete system。
