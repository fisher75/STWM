# STWM OSTF V35.47 Full M128/H32 Raw-Video Closure Protocol Decision

- artifact_truth_audit_done: True
- artifact_packaging_fixed: True
- v35_45_bounded_claim_still_allowed: True
- v35_46_failure_atlas_ready: True
- full_cvpr_scale_claim_allowed: false
- selected_clip_count: 32
- semantic_fragile_category_count_test: 3
- identity_fragile_category_count_test: 0
- identity_real_instance_count: 16
- vipseg_changed_fragile: True
- high_motion_hard_fragile: True
- real_instance_semantic_changed_fragile: True
- next_scale_choice: run_100plus_stratified_m128_h32_raw_video_closure
- recommended_next_step: run_100plus_stratified_m128_h32_raw_video_closure

## 中文总结
V35.47 协议决策：不要直接 full 325。当前 32-clip bounded benchmark 成立，但 test fragile 类别仍存在且 sample_count 小，真实 instance identity 数量也只有 16。最合理下一步是 96-128 clip 的 100+ stratified M128/H32 raw-video closure，过采样 VIPSeg changed、高运动 hard、真实 instance semantic changed，并并行扩大真实 instance identity provenance。

## 关键依据
- V35.45 32-clip larger benchmark 通过，但不是 full scale。
- V35.46 test fragile 集中在 VIPSeg changed、高运动 hard、real-instance 子集 semantic changed，且每类 sample_count=6。
- real-instance identity 当前很稳，但 claim 样本数 16，低于下一阶段建议阈值 30。
- 因此应先跑 100+ stratified，而不是直接 full 325。
