# STWM OSTF V35.35 Raw Video Frontend Rerun Smoke

- raw_video_frontend_rerun_attempted: true
- selected_sample_count: 4
- rerun_success_count: 4
- cached_vs_rerun_drift_ok: True
- minimal_unified_slice_built: True
- joint_eval_ran: True
- semantic_smoke_passed_all_seeds: True
- identity_smoke_passed_all_seeds: True
- m128_h32_video_system_benchmark_claim_allowed: True
- full_cvpr_scale_claim_allowed: false
- recommended_next_step: expand_m128_h32_raw_video_frontend_rerun_subset

## 中文总结
V35.35 raw-video frontend rerun smoke 通过：小规模 raw frame rerun trace 与缓存 trace 漂移可控，并且最小 unified slice 的 semantic/identity 联合评估没有崩。

## Claim boundary
本轮只证明 M128/H32 小规模 raw-video frontend rerun smoke；不代表 full CVPR-scale complete system。
