# STWM V4.2 Stop Report 2026-04-03

Date: 2026-04-03 12:47 +08

## 0) Freeze Goal and Boundary

Goal: safely freeze current STWM training queue with full recoverability and no evidence loss.

Boundary enforced:

- No artifact deletion.
- No checkpoint/log/output_dir deletion.
- No detached eval deletion.
- No forced SIGKILL for running jobs.

## 1) Pre-Stop Snapshot Reference

Read-only pre-stop snapshot is recorded in:

- docs/STWM_V4_2_TRAINING_STOP_SNAPSHOT_20260403.md

Snapshot captured lane-level running/pending, current steps, recent 100-step averages, PID/GPU mapping, and active run recoverability paths.

## 2) Graceful Stop Actions

### 2.1 Running training jobs stopped (graceful)

Two active training runs were gracefully interrupted by SIGINT/Ctrl+C path:

1. lane0 active run
   - run: stwm_v4_2_real_1b/seed_456/wo_semantics_v4_2_1b
   - last logged step: 2085
   - latest checkpoint: /home/chen034/workspace/stwm/outputs/training/stwm_v4_2_real_1b/seed_456/wo_semantics_v4_2_1b/checkpoints/latest.pt
   - best checkpoint: /home/chen034/workspace/stwm/outputs/training/stwm_v4_2_real_1b/seed_456/wo_semantics_v4_2_1b/checkpoints/best.pt
   - train log: /home/chen034/workspace/stwm/outputs/training/stwm_v4_2_real_1b/seed_456/wo_semantics_v4_2_1b/train_log.jsonl

2. lane1 active run
   - run: stwm_v4_2_real_220m/seed_456/wo_semantics_v4_2
   - last logged step: 3461
   - latest checkpoint: /home/chen034/workspace/stwm/outputs/training/stwm_v4_2_real_220m/seed_456/wo_semantics_v4_2/checkpoints/latest.pt
   - best checkpoint: /home/chen034/workspace/stwm/outputs/training/stwm_v4_2_real_220m/seed_456/wo_semantics_v4_2/checkpoints/best.pt
   - train log: /home/chen034/workspace/stwm/outputs/training/stwm_v4_2_real_220m/seed_456/wo_semantics_v4_2/train_log.jsonl

### 2.2 lane2 status and stop

- lane2 had worker activity but no active training PID for train_stwm_v4_2_real.py.
- lane2 log showed persistent waiting for candidate GPU set (candidate_gpus=6), with long no-suitable-GPU streak.
- lane2 worker was stopped (Ctrl+C), without fabricating a training stop.

## 3) Queue Worker Shutdown

Stopped worker/session scope:

- stwm_real_matrix_lane0
- stwm_real_matrix_lane1
- stwm_real_matrix_lane2

Additional safety shutdown also applied to legacy STWM queue tmux sessions so no auto-consumption continues.

Post-stop checks:

- active trainers count: 0
- active queue workers count: 0
- tmux worker server: not running

## 4) Pending Jobs Parked (Order Preserved)

Pending jobs were moved from lane pending folders into:

- /home/chen034/workspace/stwm/outputs/queue/stwm_v4_2_real_matrix/parked_20260403_1245/

Manifest:

- /home/chen034/workspace/stwm/outputs/queue/stwm_v4_2_real_matrix/parked_20260403_1245/PARKED_MANIFEST.tsv

Counts:

- lane0 parked: 4
- lane1 parked: 4
- lane2 parked: 3
- total parked: 11

Lane source and original order are recorded in PARKED_MANIFEST.tsv.

## 5) Completed Runs (Already Finished)

Detected completed runs with summary present:

1. /home/chen034/workspace/stwm/outputs/training/stwm_v4_2_real_1b/seed_123/full_v4_2_1b
2. /home/chen034/workspace/stwm/outputs/training/stwm_v4_2_real_1b/seed_123/wo_semantics_v4_2_1b
3. /home/chen034/workspace/stwm/outputs/training/stwm_v4_2_real_1b/seed_456/full_v4_2_1b
4. /home/chen034/workspace/stwm/outputs/training/stwm_v4_2_real_220m/seed_42/full_v4_2
5. /home/chen034/workspace/stwm/outputs/training/stwm_v4_2_real_220m/seed_42/wo_semantics_v4_2
6. /home/chen034/workspace/stwm/outputs/training/stwm_v4_2_real_220m/seed_456/full_v4_2

## 6) Resume vs Requeue Recommendation

### 6.1 Resume-eligible now

The following stopped runs can resume directly from latest checkpoint:

1. /home/chen034/workspace/stwm/outputs/training/stwm_v4_2_real_1b/seed_456/wo_semantics_v4_2_1b
2. /home/chen034/workspace/stwm/outputs/training/stwm_v4_2_real_220m/seed_456/wo_semantics_v4_2

### 6.2 Prefer requeue under new frozen protocol policy

Prefer requeue (rather than blindly continue old queue order) for:

1. all 11 parked pending jobs under parked_20260403_1245
2. unfinished pre-freeze exploratory runs that predate policy freeze and are intended for final claim evidence

## 7) Evidence Preservation Confirmation

Preserved without deletion:

- output_dir trees
- checkpoints (latest/best)
- train logs and summaries
- queue done/failed/running history files
- detached protocol eval outputs

This freeze is a safe stop, not a cleanup purge.
