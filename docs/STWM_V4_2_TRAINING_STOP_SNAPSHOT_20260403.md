# STWM V4.2 Training Freeze Snapshot (Pre-Stop)

Date: 2026-04-03  
Snapshot epoch: 1775191301  
Queue root: /home/chen034/workspace/stwm/outputs/queue/stwm_v4_2_real_matrix  
Policy: read-only snapshot before any stop action

## 1) Lane Status Overview

### lane0

- Running job file:
  - outputs/queue/stwm_v4_2_real_matrix/lane0/running/1775113282635_stwm_real_1b_seed456.job
- Running job name: stwm_real_1b_seed456
- Candidate GPUs (job): 0,1,3,7
- Current active training run:
  - output_dir: /home/chen034/workspace/stwm/outputs/training/stwm_v4_2_real_1b/seed_456/wo_semantics_v4_2_1b
  - PID: 310992
  - GPU mapping: GPU0 (uuid GPU-5f162168-f499-f75f-3a6a-48e3e9d09153)
- Current step: 2077
- Recent 100-step average step_time_s: 10.452216
- Recent 100-step average data_wait_ratio: 0.619892
- Pending jobs (in order):
  1. stwm_real_220m_seed123_full_resume
  2. stwm_real_1b_seed42_rerun_wo_semantics
  3. stwm_real_220m_seed123_wo_object_bias
  4. stwm_real_220m_seed456_wo_semantics_lowprio

### lane1

- Running job file:
  - outputs/queue/stwm_v4_2_real_matrix/lane1/running/1775113282931_stwm_real_220m_seed456.job
- Running job name: stwm_real_220m_seed456
- Candidate GPUs (job): 2,4,5,6
- Current active training run:
  - output_dir: /home/chen034/workspace/stwm/outputs/training/stwm_v4_2_real_220m/seed_456/wo_semantics_v4_2
  - PID: 3765141
  - GPU mapping: GPU5 (uuid GPU-60a40337-1114-ee46-3365-df237e338b35)
- Current step: 3453
- Recent 100-step average step_time_s: 8.875370
- Recent 100-step average data_wait_ratio: 0.683129
- Pending jobs (in order):
  1. stwm_real_220m_seed42_wo_object_bias_rerun_fresh
  2. stwm_real_1b_seed42_rerun_wo_object_bias
  3. stwm_real_1b_seed456_wo_semantics_lowprio
  4. stwm_real_220m_seed456_wo_object_bias_lowprio

### lane2

- Running job file:
  - outputs/queue/stwm_v4_2_real_matrix/lane2/running/1775134632820_stwm_real_1b_seed42_rerun_full_resume.job
- Running job name: stwm_real_1b_seed42_rerun_full_resume
- Candidate GPUs (job): 6
- Active training PID: none detected
- Worker state evidence:
  - logs/stwm_real_matrix_lane2.log tail repeatedly reports "no suitable GPU set yet"
  - elapsed wait observed > 37,000s
- Pending jobs (in order):
  1. stwm_real_1b_seed123_wo_object_bias
  2. stwm_real_220m_seed123_wo_semantics
  3. stwm_real_1b_seed456_wo_object_bias_lowprio

## 2) Active Runs and Recoverability Paths

### Active run A (lane0)

- output_dir:
  - /home/chen034/workspace/stwm/outputs/training/stwm_v4_2_real_1b/seed_456/wo_semantics_v4_2_1b
- latest checkpoint:
  - /home/chen034/workspace/stwm/outputs/training/stwm_v4_2_real_1b/seed_456/wo_semantics_v4_2_1b/checkpoints/latest.pt
  - exists: true
  - mtime epoch: 1775188390
- best checkpoint:
  - /home/chen034/workspace/stwm/outputs/training/stwm_v4_2_real_1b/seed_456/wo_semantics_v4_2_1b/checkpoints/best.pt
  - exists: true
  - mtime epoch: 1775188412
- train log:
  - /home/chen034/workspace/stwm/outputs/training/stwm_v4_2_real_1b/seed_456/wo_semantics_v4_2_1b/train_log.jsonl

### Active run B (lane1)

- output_dir:
  - /home/chen034/workspace/stwm/outputs/training/stwm_v4_2_real_220m/seed_456/wo_semantics_v4_2
- latest checkpoint:
  - /home/chen034/workspace/stwm/outputs/training/stwm_v4_2_real_220m/seed_456/wo_semantics_v4_2/checkpoints/latest.pt
  - exists: true
  - mtime epoch: 1775189947
- best checkpoint:
  - /home/chen034/workspace/stwm/outputs/training/stwm_v4_2_real_220m/seed_456/wo_semantics_v4_2/checkpoints/best.pt
  - exists: true
  - mtime epoch: 1775185111
- train log:
  - /home/chen034/workspace/stwm/outputs/training/stwm_v4_2_real_220m/seed_456/wo_semantics_v4_2/train_log.jsonl

## 3) Worker and Session Snapshot

- Matrix lane workers detected:
  - lane0: pids 4031660, 4031662
  - lane1: pids 4031697, 4031699
  - lane2: pids 2120149, 2120152
- Matrix tmux sessions detected:
  - stwm_real_matrix_lane0
  - stwm_real_matrix_lane1
  - stwm_real_matrix_lane2

## 4) Freeze Rationale Marker

At this snapshot time, detached evaluator compatibility and 8/8 detached protocol outputs are already complete. This queue state is therefore treated as pre-freeze exploratory execution context pending protocol/split/checkpoint-policy refactor.
