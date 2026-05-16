# STWM OSTF V35.38 Raw Video Frontend Rerun Smoke

- raw_video_frontend_rerun_attempted: true
- selected_sample_count: 12
- rerun_success_count: 12
- cached_vs_rerun_drift_ok: True
- minimal_unified_slice_built: True
- joint_eval_ran: True
- semantic_smoke_passed_all_seeds: True
- identity_smoke_passed_all_seeds: False
- m128_h32_video_system_benchmark_claim_allowed: False
- full_cvpr_scale_claim_allowed: false
- recommended_next_step: fix_frontend_reproducibility_trace_quality_or_target_alignment

## 中文总结
V35.38 已完成 raw-video frontend rerun smoke，但还没有通过全部 smoke gate；当前不能把 M128/H32 video system claim 从 cache 闭环推进到 raw-video rerun 闭环。

## Claim boundary
本轮只证明 M128/H32 小规模 raw-video frontend rerun smoke；不代表 full CVPR-scale complete system。
