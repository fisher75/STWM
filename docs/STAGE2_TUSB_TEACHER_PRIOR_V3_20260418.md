# Stage2 TUSB Teacher Prior V3 20260418

- chosen_teacher_prior_v3: clip_vit-b_16_temporal_weighted_masked_mean_v3
- feature_dim: 512
- cached_entry_count: 64
- newly_written_count: 64
- reused_existing_count: 0
- teacher_is_mainline_semantic_source: false

## Current Env Blocked Backends

- dinov2_like: backend_not_available_in_current_env
- siglip_like: backend_not_available_in_current_env

## Per Dataset Counts

- VIPSeg: 32
- VSPW: 32

## Selection Rationale

- DINOv2-like and SigLIP-like backends are not available in the current environment; ViT-B/16 is the strongest CLIP-family backbone that loaded successfully, and v3 uses temporal weighted masked aggregation instead of plain temporal mean.
