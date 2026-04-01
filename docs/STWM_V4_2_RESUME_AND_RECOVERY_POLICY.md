# STWM V4.2 Resume And Recovery Policy

## Scope

This policy defines robust resume and recovery for STWM V4.2 real training runs.

Applicable launcher/trainer:

- `scripts/run_stwm_v4_2_real_train_seed.sh`
- `code/stwm/trainers/train_stwm_v4_2_real.py`

## Resume Rule

Default behavior:

1. if `--resume-checkpoint` is explicitly provided, use it
2. else if `--auto-resume` and `checkpoints/latest.pt` exists, resume from latest
3. otherwise start from step 0

Resume payload restores:

- model state
- optimizer state (train mode)
- step counter
- best loss metadata

## Required Verification (Phase0/1)

Resume verification protocol:

1. warmup run to `WARMUP_STEPS`
2. restart same run with target `WARMUP_STEPS + 2`
3. verify in summary:
   - `resume.start_step >= WARMUP_STEPS`
   - `budget.resolved_optimizer_steps >= WARMUP_STEPS + 2`

Lane check artifact:

- `lane*/resume_check.json`

`resume_verified` must be true in both lanes before scale-out approval.

## Failure Recovery Procedure

If a lane fails (OOM, preemption, transient IO error):

1. keep output directory and checkpoint directory unchanged
2. relaunch same scale/seed/run-name with `--auto-resume`
3. verify summary `resume.start_step` increased from previous attempt
4. confirm latest and best checkpoints still exist

If checkpoint file is missing or corrupted:

1. fall back to the newest valid checkpoint in same run directory
2. if no valid checkpoint is available, restart run from step 0
3. mark event in audit notes and keep failure logs

## Operational Constraints

1. Do not change run-name when recovering a run.
2. Do not redirect output directory during resume.
3. Do not disable auto-resume for production real runs.
4. Keep checkpoint interval unchanged inside the same run unless explicitly documented.
