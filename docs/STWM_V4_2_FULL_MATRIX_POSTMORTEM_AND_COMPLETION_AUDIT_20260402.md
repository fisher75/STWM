# STWM V4.2 Full Matrix Postmortem and Completion Audit (2026-04-02)

## 0) Scope and Snapshot

- Audit time: 2026-04-02 19:55 +08
- Audit mode: non-invasive, read-only (queue metadata, logs, train outputs)
- Queue root: `outputs/queue/stwm_v4_2_real_matrix`
- Goal: reconstruct 4-lane full timeline and verify matrix completion status by experiment unit (not only by lane).

Current queue snapshot at audit time:

- lane0: pending=0, running=1, done=0, failed=2 (running `stwm_real_1b_seed456`)
- lane1: pending=0, running=1, done=0, failed=1 (running `stwm_real_220m_seed456`)
- lane2: pending=0, running=0, done=0, failed=1
- lane3: pending=0, running=0, done=0, failed=1

---

## 1) 4-Lane Full Timeline Reconstruction

### lane0 timeline

1. 2026-04-01 17:44:45: worker started.
2. 2026-04-01 17:45:05: started `stwm_real_1b_seed42`.
3. 2026-04-01 18:58:19: `stwm_real_1b_seed42` failed (exit=1).
4. 2026-04-01 18:58:19: started `stwm_real_1b_seed123`.
5. 2026-04-02 18:45:53: `stwm_real_1b_seed123` failed (exit=137).
6. 2026-04-02 18:45:53: auto-switched to `stwm_real_1b_seed456` (currently running).

Key run transitions from worker log:

- seed123: `full_v4_2_1b` START -> DONE
- seed123: `wo_semantics_v4_2_1b` START (no DONE)

Interpretation:

- lane0 did a failure-driven switch, not a clean matrix closure.

### lane1 timeline

1. 2026-04-01 17:44:45: worker started.
2. 2026-04-01 17:45:05: started `stwm_real_220m_seed42`.
3. 2026-04-02 18:45:53: `stwm_real_220m_seed42` failed (exit=137).
4. 2026-04-02 18:45:53: auto-switched to `stwm_real_220m_seed456` (currently running).

Key run transitions from worker log:

- seed42: `full_v4_2` START -> DONE
- seed42: `wo_semantics_v4_2` START -> DONE
- seed42: `wo_object_bias_v4_2` START (no DONE)

Interpretation:

- lane1 also switched due failure, not because all runs were finished.

### lane2 timeline

1. 2026-04-02 14:14:59: worker started.
2. 2026-04-02 14:15:19: started `stwm_real_1b_seed42_rerun`.
3. 2026-04-02 18:45:53: `stwm_real_1b_seed42_rerun` failed (exit=137).
4. no pending remained; lane2 became idle.

Key run transitions:

- seed42_rerun: `full_v4_2_1b` START (no DONE)

### lane3 timeline

1. 2026-04-02 14:24:58: worker started.
2. 2026-04-02 14:24:58: started `stwm_real_220m_seed123`.
3. 2026-04-02 18:45:53: `stwm_real_220m_seed123` failed (exit=137).
4. no pending remained; lane3 became idle.

Key run transitions:

- seed123: `full_v4_2` START (no DONE)

---

## 2) Completion Audit by Experiment Unit (Matrix View)

Legend:

- DONE: completed with summary artifact and completion evidence.
- RUNNING: currently active process.
- FAILED: interrupted with job failure.
- NOT_STARTED: no run directory / no run evidence.
- PARTIAL/INCOMPLETE: run started but not completed.

Columns:

- `last_step`: from `train_log.jsonl` latest row
- `summary`: `mini_val_summary.json` exists or not
- `ckpt`: checkpoint resumability evidence (`latest.pt`/`best.pt`)

| Experiment Unit | Run | Status | last_step | summary | ckpt | End mode / evidence |
|---|---|---|---:|---|---|---|
| 1B seed123 | full_v4_2_1b | DONE | 5000 | yes | best+latest+milestones | worker log shows START->DONE |
| 1B seed123 | wo_semantics_v4_2_1b | FAILED / PARTIAL | 3462 | no | best+latest | started, then parent job exit=137 |
| 1B seed123 | wo_object_bias_v4_2_1b | NOT_STARTED | NA | no | none | no run directory |
| 220M seed42 | full_v4_2 | DONE | 5000 | yes | best+latest+milestones | worker log shows START->DONE |
| 220M seed42 | wo_semantics_v4_2 | DONE | 5000 | yes | best+latest+milestones | worker log shows START->DONE |
| 220M seed42 | wo_object_bias_v4_2 | FAILED / PARTIAL | 424 | no | none (no latest/best) | started, parent job exit=137 |
| 1B seed42_rerun | full_v4_2_1b | FAILED / PARTIAL | 1669 | no | best+latest | started, parent job exit=137 |
| 1B seed42_rerun | wo_semantics_v4_2_1b | NOT_STARTED | NA | no | none | no run directory |
| 1B seed42_rerun | wo_object_bias_v4_2_1b | NOT_STARTED | NA | no | none | no run directory |
| 220M seed123 | full_v4_2 | FAILED / PARTIAL | 2286 | no | best+latest | started, parent job exit=137 |
| 220M seed123 | wo_semantics_v4_2 | NOT_STARTED | NA | no | none | no run directory |
| 220M seed123 | wo_object_bias_v4_2 | NOT_STARTED | NA | no | none | no run directory |
| 1B seed456 | full_v4_2_1b | RUNNING | 455 (snapshot) | no | best+latest (in-progress) | currently running on lane0 |
| 1B seed456 | wo_semantics_v4_2_1b | NOT_STARTED | NA | no | none | not entered yet |
| 1B seed456 | wo_object_bias_v4_2_1b | NOT_STARTED | NA | no | none | not entered yet |
| 220M seed456 | full_v4_2 | RUNNING | 496 (snapshot) | no | best+latest (in-progress) | currently running on lane1 |
| 220M seed456 | wo_semantics_v4_2 | NOT_STARTED | NA | no | none | not entered yet |
| 220M seed456 | wo_object_bias_v4_2 | NOT_STARTED | NA | no | none | not entered yet |

---

## 3) Critical Answer: Did lane0/lane1 old tasks truly finish before seed456 takeover?

### lane0 previous `1B seed123`

- `full_v4_2_1b` did finish (DONE evidence present).
- `wo_semantics_v4_2_1b` did NOT finish (last_step=3462, summary missing, no DONE marker).
- `wo_object_bias_v4_2_1b` never started.

Conclusion:

- seed456 started on lane0 after a failure (exit=137), not after full matrix closure for seed123.

### lane1 previous `220M seed42`

- `full_v4_2` finished.
- `wo_semantics_v4_2` finished.
- `wo_object_bias_v4_2` did NOT finish (last_step=424, summary missing, no DONE marker).

Conclusion:

- seed456 started on lane1 after a failure (exit=137), while matrix for seed42 remained incomplete.

---

## 4) Failure Root-Cause Classification

### Classification table

| Item | Evidence | Classification |
|---|---|---|
| lane0 initial `1b seed42` (exit=1) | queue shows exit=1; current worker log lacks traceback context | Category 4 (insufficient in current snapshot), historically consistent with prior semantic-cache bug path |
| `1B seed123 wo_semantics` partial | job failed exit=137 at 2026-04-02 18:45:53 | Category 2 (external kill / SIGKILL-like) |
| `220M seed42 wo_object_bias` partial | job failed exit=137 at same timestamp | Category 2 |
| `1B seed42_rerun full` partial | job failed exit=137 at same timestamp | Category 2 |
| `220M seed123 full` partial | job failed exit=137 at same timestamp | Category 2 |

Additional hard kill evidence from per-run launcher logs:

- `logs/stwm_v4_2_real_1b_seed123_wo_semantics_v4_2_1b.log` line 1 contains `3191762 Killed python ...`
- `logs/stwm_v4_2_real_1b_seed42_full_v4_2_1b.log` line 1 contains `2067146 Killed python ...`
- `logs/stwm_v4_2_real_220m_seed123_full_v4_2.log` line 1 contains `2154978 Killed python ...`
- `logs/stwm_v4_2_real_220m_seed42_wo_object_bias_v4_2.log` line 1 contains `4144676 Killed python ...`

### lane2/lane3 requested root causes

- lane2 (`1B seed42_rerun`): failed with exit=137; last_step=1669; checkpoint latest/best exists -> resume feasible.
- lane3 (`220M seed123`): failed with exit=137; last_step=2286; checkpoint latest/best exists -> resume feasible.

Notes:

- four lanes (old lane0/lane1 jobs + lane2 + lane3) share the same fail timestamp 2026-04-02 18:45:53 for exit=137 events, indicating external termination pressure rather than independent code exceptions.

---

## 5) Missing Matrix and Minimal Backfill Plan

### What is still missing / incomplete

1. 1B seed123:
   - wo_semantics_v4_2_1b incomplete
   - wo_object_bias_v4_2_1b not started
2. 220M seed42:
   - wo_object_bias_v4_2 incomplete
3. 1B seed42_rerun:
   - full_v4_2_1b incomplete
   - wo_semantics_v4_2_1b not started
   - wo_object_bias_v4_2_1b not started
4. 220M seed123:
   - full_v4_2 incomplete
   - wo_semantics_v4_2 not started
   - wo_object_bias_v4_2 not started
5. seed456:
   - both scales are currently in full stage RUNNING (not yet completed).

### Minimal manual backfill list (excluding currently running seed456 full jobs)

1. `1B seed123`:
   - resume `wo_semantics_v4_2_1b`
   - then run `wo_object_bias_v4_2_1b`
   - resume feasible: YES (latest checkpoint exists)
2. `220M seed42`:
   - rerun `wo_object_bias_v4_2`
   - resume feasible: NO (no checkpoint file present at failure step 424)
3. `1B seed42_rerun`:
   - resume `full_v4_2_1b`
   - then run `wo_semantics_v4_2_1b` and `wo_object_bias_v4_2_1b`
   - resume feasible: YES
4. `220M seed123`:
   - resume `full_v4_2`
   - then run `wo_semantics_v4_2` and `wo_object_bias_v4_2`
   - resume feasible: YES

---

## 6) Direct Answers to Mandatory Questions

1. lane0 previous `1B seed123` completed what?
   - DONE: `full_v4_2_1b`
   - NOT DONE: `wo_semantics_v4_2_1b` (partial), `wo_object_bias_v4_2_1b` (not started)

2. lane1 previous `220M seed42` completed what?
   - DONE: `full_v4_2`, `wo_semantics_v4_2`
   - NOT DONE: `wo_object_bias_v4_2` (partial)

3. Why did lane2 and lane3 fail?
   - both failed with exit=137 at 2026-04-02 18:45:53; classification is external kill/SIGKILL-like pressure.

4. Did seed456 start before previous matrix was fully closed?
   - YES. seed456 on lane0/lane1 was auto-started immediately after failure of incomplete previous jobs.

5. What is still missing in the matrix now?
   - all incomplete units listed in Section 5 (seed123 1B ablations, seed42 220M wo_object_bias, seed42_rerun 1B chain, seed123 220M chain).

6. What is the minimal next backfill list?
   - exactly the four items in Section 5 minimal manual backfill list.
