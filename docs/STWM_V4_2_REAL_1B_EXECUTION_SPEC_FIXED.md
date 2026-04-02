# STWM V4.2 Real 1B Execution Spec (Fixed)

## Objective

This spec freezes the real 1B execution policy for main evidence generation.

Main-evidence source:

- real 220M + real 1B only

## Fixed Runtime Configuration

Scale:

- `1b`

Model preset:

- `prototype_1b_v4_2`
- preset file: `code/stwm/configs/model_presets_v4_2_1b.json`

Precision and memory:

- `bf16 = true`
- `activation_checkpointing = true`

Batching:

- `micro_batch_per_gpu = 1`
- `grad_accum = 16`
- `effective_batch = 16`

Dataloader start values:

- `num_workers = 8`
- `prefetch_factor = 2`
- `persistent_workers = true`
- `pin_memory = true`

Checkpointing:

- retain `latest.pt` and `best.pt`
- `latest.pt` overwrite interval: every 100 optimizer steps
- sparse milestones disabled by default (`milestone_interval=0`)
- optional sparse milestone files only when explicitly enabled and disk budget permits

Resume:

- `auto_resume = true`
- resume source defaults to `checkpoints/latest.pt`

## Unified Hard Budget

For each real 1B main training run:

- `target_epochs = 3`
- `min_optimizer_steps = 5000`
- `max_optimizer_steps = 8000`
- resolved requirement: `5000` optimizer steps

Training set must remain:

- VSPW train + VIPSeg train full manifest

## Lane Strategy (Phase0/1 First)

Mandatory phase0/1 warmup policy:

1. Start with exactly 2 single-GPU lanes:
   - one lane for 1B
   - one lane for 220M
2. Collect utilization, step-time stability, IO wait behavior, CPU worker behavior, and disk/checkpoint behavior.
3. Expand to 4 lanes only if all scale-out checks pass.

If instability appears:

- reduce lane count first
- do not change fixed micro-batch/grad-accum policy

## Launch Interfaces

Single seed strict launcher:

- `scripts/run_stwm_v4_2_real_train_seed.sh --scale 1b --seed <seed> [out_root]`

Phase0/1 audit launcher:

- `scripts/run_stwm_v4_2_phase01_audit.sh`

Key environment knobs (fixed defaults expected):

- `STWM_V4_2_REAL_TARGET_EPOCHS=3`
- `STWM_V4_2_REAL_MIN_STEPS=5000`
- `STWM_V4_2_REAL_MAX_STEPS=8000`
- `STWM_V4_2_REAL_SAMPLE_LIMIT=0`

## Forbidden For Main Evidence

1. Positive sample limits.
2. Replacing true train split with mini/eval protocol sets.
3. Changing fixed precision/batch policy without explicit governance update.
4. Reporting lightweight/minival runs as main conclusion evidence.
