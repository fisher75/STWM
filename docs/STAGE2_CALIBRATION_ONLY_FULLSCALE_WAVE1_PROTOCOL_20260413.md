# Stage2 Calibration-Only Fullscale Wave1 Protocol

- generated_at_utc: 2026-04-14T06:09:02.211302+00:00
- stage1_mutation_allowed: false in mainline calibration-only wave1
- main_task: future trace / future state generation
- teacher_as_mainline_semantic_source: false
- calibration_only_definition: readout-side semantic alignment + sparse gating + delayed aux schedule + semantic-hard sidecar
- persistence_mainline_allowed: false
- calibration_families: topk1 / qcap15
- fullscale_policy: full VSPW+VIPSeg train/val, no sample caps, no DDP retrofit, threshold-based shared-GPU greedy allocation
- concurrent_runtime_override: workers=8, pin_memory=True, persistent_workers=True, prefetch=4
- partial_unfreeze_branch: gated secondary ablation only after all 6 calibration runs complete
- forbidden: Stage1 retraining; codec/VAE wave0; batch/lr sweep; external-eval expansion; persistence-as-mainline narrative
