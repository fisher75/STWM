# STWM OSTF V35.56 Claim Boundary Freeze and Release Bundle

## 中文总结
V35.56 已冻结 V35.55 的 benchmark claim boundary，并生成 non-paper release bundle 索引。当前允许 bounded full 325 M128/H32 raw-video closure video-system benchmark claim；仍不允许 full CVPR-scale、任意尺度或 open-vocabulary dense segmentation claim。

## 已冻结允许 claim
- full_m128_h32_raw_video_closure_video_system_benchmark: 允许 claim：full 325 M128/H32 raw-video closure video-system benchmark 通过。
- raw_video_frontend_rerun_not_old_trace_cache: 允许 claim：本轮从 raw frame paths / predecode 重新跑 frontend，旧 cache 只作 drift comparison。
- future_semantic_state_transition_field: 允许 claim：输出 future semantic state / transition / uncertainty field。
- pairwise_identity_retrieval_field_real_instance_subset: 允许 claim：真实 instance-labeled subset 上的 pairwise identity retrieval field。

## 已冻结不允许 claim
- full_cvpr_scale_complete_system: 不允许 claim：任意尺度/任意分辨率/任意 horizon/full open-vocabulary 的 CVPR-scale complete system 已完成。
- teacher_or_future_embedding_as_method: 不允许 claim 或实现：teacher / future teacher embedding 作为方法主输入。

## 当前顶会边界
- claim_boundary_freeze_ready: True
- selected_clip_count: 325
- real_instance_identity_count: 121
- m128_h32_full_325_video_system_benchmark_claim_allowed: True
- full_cvpr_scale_claim_allowed: false
- frozen_claim_boundary_manifest: reports/stwm_ostf_v35_56_frozen_claim_boundary_manifest_20260516.json
- non_paper_release_bundle_index: reports/stwm_ostf_v35_56_non_paper_release_bundle_index_20260516.json
- recommended_next_step: stop_and_return_to_claim_boundary_or_run_independent_environment_replay
