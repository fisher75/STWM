# STWM OSTF V35.48 100+ Stratified Raw-Video Closure Subset Build

- selected_clip_count: 128
- target_clip_count: 128
- min_clip_count: 96
- dataset_counts: {'VIPSEG': 64, 'VSPW': 64}
- split_counts: {'train': 48, 'val': 45, 'test': 35}
- real_instance_identity_count: 64
- pseudo_identity_count: 64
- risk_vipseg_changed_count: 62
- risk_high_motion_hard_count: 59
- risk_real_instance_semantic_changed_count: 62
- exact_blockers: []

## 中文总结
V35.48 已构建 100+ stratified subset：128 clips，重点过采样 VIPSeg changed、高运动 hard、真实 instance semantic changed、occlusion、crossing、identity confuser。本阶段仍只做 M128/H32 raw-video closure，不训练新模型，不跑 H64/H96/M512/M1024。
