# Stage2 Semantic Objective Redesign V1 Protocol

- generated_at_utc: 2026-04-10T15:47:38.551794+00:00
- main_task: future trace / future state generation
- full_video_reconstruction_is_main_task: false
- stage1_status: frozen trace/state backbone
- stage2_mainline_semantic_source: trainable object-region/mask-crop semantic encoder
- teacher_usage: bootstrap / pseudo-label / cache / alignment target only
- teacher_as_mainline_semantic_source: false
- this_round_is_not: teacher mainline, full video reconstruction, paper framing, Stage1 rollback, DDP retrofit, batch/lr sweep

## What This Round Fixes
- semantic objective: add alignment and query/persistence auxiliary objectives without rewriting frozen Stage1 dynamics
- bootstrap target quality: replace crop-stats pseudo bootstrap with verified local visual teacher cache when available
- semantic-hard supervision: add bounded sample/loss reweighting for hard semantic/persistence clips
