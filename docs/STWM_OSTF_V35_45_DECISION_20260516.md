# STWM OSTF V35.45 Decision

- current_completed_version: V35.45
- artifact_packaging_fixed: True
- selected_clip_count: 32
- raw_frontend_rerun_done: True
- raw_frontend_rerun_success_rate: 1.0
- trace_drift_ok: True
- unified_slice_built: True
- semantic_three_seed_passed: True
- stable_preservation: True
- identity_real_instance_three_seed_passed: True
- identity_pseudo_targets_excluded_from_claim: True
- per_category_breakdown_ready: True
- visualization_ready: True
- v30_backbone_frozen: true
- future_leakage_detected: False
- trajectory_degraded: False
- m128_h32_larger_video_system_benchmark_claim_allowed: True
- full_cvpr_scale_claim_allowed: false
- recommended_next_step: run_v35_46_per_category_failure_atlas

## 中文总结
V35.45 完成扩大版 M128/H32 raw-video closure benchmark：artifact、32-clip subset、raw frontend rerun、unified semantic/identity slice、三 seed joint eval、case-mined visualization 均闭合。当前可以允许 bounded M128/H32 larger video system benchmark claim，但仍不能 claim full CVPR-scale complete system。下一步应进入 V35.46 per-category failure atlas，把成功/失败按 motion、occlusion、crossing、confuser、stable/changed/hard 更细拆开。

## 好消息
- V35.45 artifact truth audit 证明 V35.44 依赖 JSON 在 live repo 中存在，artifact packaging 当前已补齐。
- larger raw-video closure subset 扩到 32 clips，VSPW/VIPSeg 各 16，val/test/train 均有覆盖。
- raw-video frontend rerun 成功率 1.0，trace drift vs cache mean/max 均为 0，visibility agreement 为 1.0。
- larger rerun unified slice 构建成功，16 个真实 instance identity 样本用于 claim，16 个 pseudo identity 样本只做诊断。
- semantic 三 seed、stable preservation、real-instance identity 三 seed、per-category breakdown 与可视化均通过。

## 坏消息 / Claim boundary
- 这仍是 bounded M128/H32 larger subset，不是 full CVPR-scale complete benchmark。
- semantic field 当前仍应称 future semantic state / transition field，不是 open-vocabulary dense semantic segmentation field。
- identity field claim 仍依赖真实 instance-labeled subset；VSPW pseudo slot identity 已排除，后续需要扩大真实 instance provenance。
