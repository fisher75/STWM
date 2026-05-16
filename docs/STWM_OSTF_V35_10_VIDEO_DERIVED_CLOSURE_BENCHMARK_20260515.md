# STWM OSTF V35.10 Video-Derived Closure Benchmark

- video_derived_closure_benchmark_ran: true
- sample_count: 6
- stable_top5: 0.9778740032740139
- copy_stable_top1: 1.0
- stable_preservation: False
- identity_retrieval_passed: False
- semantic_changed_signal: not_evaluable_on_stable_class_id_video_cache
- semantic_hard_signal: not_evaluable_on_stable_class_id_video_cache
- full_video_semantic_identity_field_claim_allowed: false
- recommended_next_step: build_video_derived_future_semantic_state_targets

## 中文总结
V35.10 完成 video-derived trace + observed CLIP measurement 的有限闭环评估；stable semantic 与 identity retrieval 可评估，但 changed/hard semantic target 仍缺失。
