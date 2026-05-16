# STWM OSTF V35.48 100+ Stratified Raw-Video Closure Plan

- target_clip_count: 128
- minimum_clip_count: 96
- oversample: VIPSeg changed / high_motion hard / real_instance semantic_changed / occlusion / crossing / identity confuser
- VSPW/VIPSeg balanced: true
- train/val/test balanced: true
- real_instance_identity_target_minimum: 30
- real_instance_identity_target_ideal: 50+
- pseudo_identity_policy: diagnostic-only
- train_new_model: false
- run_h64_h96: false
- run_m512_m1024: false

## 中文总结
V35.48 应跑 96-128 clip stratified M128/H32 raw-video closure，不直接 full 325；重点过采样 VIPSeg changed、高运动 hard、真实 instance semantic changed，同时扩大真实 instance identity provenance。
