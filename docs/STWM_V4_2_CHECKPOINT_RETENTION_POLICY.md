# STWM V4.2 Checkpoint Retention Policy

## Scope

This policy applies to STWM V4.2 real 220M and real 1B main-evidence training.

Trainer implementation:

- `code/stwm/trainers/train_stwm_v4_2_real.py`

## Retention Rule

Required retained files per run:

1. `checkpoints/latest.pt`
2. `checkpoints/best.pt`

Optional retained files:

- `checkpoints/milestone_step_*.pt` only when `milestone_interval > 0`

Default intervals:

- phase0/1 warmup: `checkpoint_interval=50`, `milestone_interval=0`
- main long run: `checkpoint_interval=100`, `milestone_interval=0`

Default retention string for main long run:

- `latest_every_100+best`

Milestone retention is disabled by default and must be explicitly enabled only
after disk budget review.

## Save Conditions

Checkpoint save is skipped if free disk falls below threshold:

- `min_free_disk_gb = 50`

Policy intent:

- avoid disk exhaustion and preserve latest+best survivability.

## Best Checkpoint Semantics

`best.pt` is updated when current `total_loss <= best_total_loss`.

`latest.pt` is overwritten at each save event.

If run ends and `best.pt` is missing while `latest.pt` exists, trainer backfills `best.pt` from `latest.pt`.

## Disk Budget Estimation

Trainer reports estimated checkpoint size and max retained budget:

- `estimated_checkpoint_each_gb`
- `estimated_max_retained_gb`

Milestone enablement must respect remaining disk capacity and should stay sparse.

## Compliance Requirements

Per run summary (`mini_val_summary.json`) must include:

1. checkpoint policy section with retention text
2. non-empty `latest` and `best` paths
3. `min_free_disk_gb` value

Phase0/1 audit (`phase1_audit_summary.json`) must report:

- `latest_exists = true`
- `best_exists = true`

for every lane before approving lane expansion.
