# STWM V35.20 VIPSeg-only CoTracker Boost

- processed_clip_count: 121
- vipseg_processed_split_counts: {'train': 85, 'val': 21, 'test': 15}
- skipped_existing_clip_count: 85
- failed_clip_count: 163
- cache_root: outputs/cache/stwm_real_teacher_object_dense_v16/M128_H32

## 中文总结
本轮只补 VIPSeg source-domain 的 M128/H32 trace cache，目的是修 VIPSeg→VSPW 的 source 覆盖不足；没有跑 H64/H96/M512，也没有训练 semantic adapter。
