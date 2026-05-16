# STWM OSTF V35.34 Raw Video Frontend Reproducibility Harness

- frontend_reproducibility_harness_ready: True
- gpu_rerun_attempted: false
- cotracker_v16_script_exists: True
- cotracker_v35_20_script_exists: True
- frontend_scripts_have_setproctitle: True
- raw_first_frame_exists_ratio: 1.0000
- trace_source_exists_ratio: 1.0000
- trace_required_fields_nonzero_ratio: 1.0000
- full_cvpr_scale_claim_allowed: false
- recommended_next_step: run_small_raw_video_frontend_rerun_smoke_m128_h32_when_gpu_budget_allows

## 中文总结
V35.34 确认 raw-video frontend reproducibility harness 已经具备：raw frame、video-derived trace source、CoTracker 前端脚本、setproctitle 规范和 unified benchmark 重建/评估命令都可追溯。这让 V35 的 M128/H32 完整视频闭环从“cache 上成立”推进到“可复现协议已打包”。但本轮未重跑 GPU 前端，所以仍不能宣称 full-scale CVPR complete system；下一步若有 GPU 预算，应做小规模 raw-video frontend rerun smoke。
