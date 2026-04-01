# STWM V4.2 Real 220M Budget Plan

## Scope

This document defines the fixed real-training budget for STWM V4.2 220M main evidence.

Frozen boundaries:

1. Main evidence must come from real 220M and real 1B only.
2. Training set must be true train split: VSPW train + VIPSeg train.
3. Base/state-identifiability/decoupling are evaluation protocols, not training-set replacements.
4. No ad-hoc batch, precision, or lane policy changes.

## Training Set Source Of Truth

Manifest builder:

- `code/stwm/tools/build_stwm_v4_2_real_train_manifest.py`

Default split files:

- `data/external/vspw/VSPW/train.txt`
- `data/external/vipseg/VIPSeg/train.txt`

Current manifest report target:

- `manifests/realsplits/stwm_v4_2_vspw_vipseg_train_v1.json`
- `outputs/audits/<phase01_run>/real_train_manifest_report.json`

Observed full-train manifest count:

- `sample_count = 5612`

## Fixed 220M Training Configuration

Required fixed config for real 220M:

- model preset: `prototype_220m_v4_2`
- precision: `bf16 = true`
- activation checkpointing: `true`
- micro batch per GPU: `2`
- grad accumulation: `8`
- effective batch: `2 * 8 = 16`
- dataloader start values: `num_workers=6`, `prefetch_factor=2`, `persistent_workers=true`, `pin_memory=true`

Reference launcher:

- `scripts/run_stwm_v4_2_real_train_seed.sh --scale 220m --seed <seed>`

## Hard Budget Rule (Unified With 1B)

Budget computation tool:

- `code/stwm/tools/compute_stwm_v4_2_real_budget.py`

Fixed rule:

- `target_epochs = 3`
- `min_optimizer_steps = 5000`
- `max_optimizer_steps = 8000`
- `resolved_steps = min(max(min_steps, steps_for_3_epochs), max_steps)`

With `sample_count=5612` and `effective_batch=16`:

- `steps_per_epoch = ceil(5612 / 16) = 351`
- `steps_for_3_epochs = 351 * 3 = 1053`
- `resolved_optimizer_steps = min(max(5000, 1053), 8000) = 5000`

Final required optimizer steps for each 220M real training run:

- `5000`

## Real 220M Matrix

Run set:

- `full_v4_2`
- `wo_semantics_v4_2`
- `wo_object_bias_v4_2`

Seed policy:

- mandatory first: `42,123`
- conditional third seed: `456` (only after stability/resource gate)

Minimum mandatory workload:

- `2 seeds * 3 runs = 6 run-seeds`
- each run-seed at `5000` optimizer steps under fixed config

## Compliance Checks

Per run summary (`mini_val_summary.json`) must show:

1. `budget.sample_count >= 5612` (no truncation)
2. `budget.effective_batch = 16`
3. `precision.bf16 = true`
4. `precision.activation_checkpointing = true`
5. `budget.resolved_optimizer_steps = 5000` for main training matrix

Operational guardrail:

- `STWM_V4_2_REAL_SAMPLE_LIMIT` default is `0` (full manifest); do not set positive sample limits for main-evidence training.
