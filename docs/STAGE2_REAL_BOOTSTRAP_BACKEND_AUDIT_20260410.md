# Stage2 Real Bootstrap Backend Audit

- generated_at_utc: 2026-04-10T15:47:42.067302+00:00
- chosen_bootstrap_backend: local_clip_vit_b32_mask_crop_visual_teacher
- chosen_backend_usable: True
- chosen_backend_feature_dim: 512
- teacher_as_mainline_semantic_source: False
- crop_stats_insufficient_reason: crop_stats_pseudo_target_cache is a 10-d handcrafted color/area/foreground-stat target. It is useful as a fallback sanity signal but is not a real visual semantic teacher representation.
- blocking_reason_if_no_real_backend: 

## Verified Usable Backends
- local_clip_vit_b32_mask_crop_visual_teacher: feature_dim=512, local_weight=/home/chen034/.cache/clip/ViT-B-32.pt
