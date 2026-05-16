# STWM OSTF V35.44 Raw-Video Closure Final Decision

- current_completed_version: V35.44
- raw_video_frontend_rerun_done: True
- raw_video_frontend_drift_ok: True
- semantic_three_seed_passed_on_eval_balanced_raw_rerun: True
- identity_label_provenance_fixed: True
- identity_three_seed_passed_on_real_instance_subset: True
- visualization_ready: True
- m128_h32_video_system_benchmark_claim_allowed: True
- full_cvpr_scale_claim_allowed: false
- recommended_next_step: run_larger_m128_h32_raw_video_closure_subset_with_identity_provenance_filter

## 中文总结
V35.44 完成阶段性闭环：raw frame paths 最小重跑 frontend、M128/H32 trace、semantic state adapter、pairwise identity retrieval 在真实 instance subset 上形成 bounded video system benchmark。创新点比 V34 路线更稳：语义目标从 continuous teacher delta 改成可观测语义状态，identity 从 pointwise BCE 改成 pairwise retrieval。但这还不是 full CVPR-scale claim：规模仍是 12-clip rerun subset，identity 需要明确排除 VSPW pseudo identity。

## 好消息
- V35.38 eval-balanced raw-video rerun trace 与 cache drift 为 0，frame path、visibility、confidence、motion 对齐。
- V35.38 semantic 三 seed 均通过，stable preservation 全 seed 通过。
- V35.42 修正 identity label provenance 后，VIPSeg real-instance subset identity 三 seed 通过。
- V35.43 真实 case-mined PNG 已生成。

## 坏消息 / Claim boundary
- VSPW identity target 在当前构造中是 pseudo slot/semantic-track group，不能作为真实 identity field claim。
- 当前 raw-video closure 规模仍小，不能 claim full CVPR-scale complete system。
- 还没有完整 325-clip raw frontend rerun，也没有更大 subset 的 per-category robust breakdown。
