# STWM V4.2 Priority Reorder Safe Mode

## Purpose

This note records the immediate safe-degrade actions applied after the multi-lane `exit_code=137` incident pattern.

Execution time: 2026-04-02 20:12+08 (follow-up verification at ~20:16+08).

## 1) Why rollback to 2 active lanes now

Observed facts from queue evidence:

- Multiple jobs across lane0/lane1/lane2/lane3 ended at the same timestamp (`2026-04-02 18:45:53`) with `exit_code=137`.
- This synchronized failure pattern is inconsistent with independent single-run code bugs and consistent with external kill pressure.

Decision:

- Immediately cap active production lanes to 2 (lane0/lane1 only) until system-level kill root cause is fully validated.

## 2) Why free VRAM does not disprove 137 risk

- Shared multi-tenant GPU state means visible free VRAM at one moment is not equivalent to cgroup/system memory safety over time.
- `137` can be triggered by external kill paths (OOM killer, cgroup policy, scheduler kill, admin kill) even without clear Python traceback.
- Therefore, apparent memory headroom is insufficient as a safety guarantee for 4-lane full-concurrency continuation.

## 3) Why current seed456 full runs were preserved

To minimize disruption and follow safety constraints:

- lane0 running job `stwm_real_1b_seed456` was kept running.
- lane1 running job `stwm_real_220m_seed456` was kept running.
- No restart/kill was applied to these active jobs.

At the same time:

- lane2/lane3 worker tmux sessions were stopped (`stwm_real_matrix_lane2`, `stwm_real_matrix_lane3`).
- lane2/lane3 are now cold-standby and not participating in scheduling.

## 4) Priority reorder for 42/123 gap closure

All existing pending entries were backed up under:

- `outputs/queue/stwm_v4_2_real_matrix/reorder_backup_<timestamp>/`

Then pending was rebuilt for lane0/lane1 with gap-closure-first policy.

### lane0 pending order (1B track)

1. `stwm_real_1b_seed123_wo_semantics_resume`
2. `stwm_real_1b_seed42_rerun_full_resume`
3. `stwm_real_1b_seed123_wo_object_bias`
4. `stwm_real_1b_seed42_rerun_wo_semantics`
5. `stwm_real_1b_seed42_rerun_wo_object_bias`
6. `stwm_real_1b_seed456_wo_semantics_lowprio`
7. `stwm_real_1b_seed456_wo_object_bias_lowprio`

### lane1 pending order (220M track)

1. `stwm_real_220m_seed42_wo_object_bias_rerun_fresh`
2. `stwm_real_220m_seed123_full_resume`
3. `stwm_real_220m_seed123_wo_semantics`
4. `stwm_real_220m_seed123_wo_object_bias`
5. `stwm_real_220m_seed456_wo_semantics_lowprio`
6. `stwm_real_220m_seed456_wo_object_bias_lowprio`

### Resume vs fresh specifics

- Resume-targeted jobs were submitted as single-run commands via `STWM_V4_2_REAL_RUNS=<run_name>` and existing `--auto-resume` behavior.
- `220M seed42 / wo_object_bias` was marked fresh rerun:
  - previous partial directory was archived as `wo_object_bias_v4_2_pre_safe_mode_<timestamp>`
  - rerun job remains targeted to `wo_object_bias_v4_2` only.

### Strict anti-leapfrog guard for currently running seed456 full jobs

Because the currently running seed456 jobs were already launched as full matrix scripts, they would normally continue to ablation runs after full completes.

To enforce the new priority policy without interrupting current full training:

- lane0 guard watches for `done ... run=full_v4_2_1b` in current job log, then terminates that job process.
- lane1 guard watches for `done ... run=full_v4_2` in current job log, then terminates that job process.

Effect:

- current full run is allowed to finish,
- seed456 ablations do not leapfrog 42/123 gap-filling pending queue.

## 5) Safety mode state verification snapshot

At verification:

- tmux sessions present: lane0, lane1
- tmux sessions absent: lane2, lane3
- queue state:
  - lane0: `running=1`, `pending=7`
  - lane1: `running=1`, `pending=6`
  - lane2: `running=0`, `pending=0`
  - lane3: `running=0`, `pending=0`

This confirms active-lane operation has been reduced to lane0/lane1 while preserving current running jobs.
