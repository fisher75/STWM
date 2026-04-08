# TRACEWM Stage1-v2 220M Long-Train Continuation Protocol (2026-04-08)

## 1. Frozen Facts

1. Stage1-v2 220M backbone has already won in mainline freeze:
   - `final_stage1_backbone_decision = freeze_220m_as_stage1_backbone`
2. Current 5000-step long-train has completed with:
   - `best primary metric = 0.2102444279531349`
   - trend from 4000 to 5000 remains improving on primary endpoint metric.
3. The current best next step is not architecture change; it is continuation of long-train to 10000 steps.
4. The only engineering gap to fix first in this round:
   - launcher must truly reuse `gpu_selector.py + gpu_lease.py`
   - old fallback based mostly on free memory is not sufficient for shared-cluster policy.

## 2. Scope Boundary (Hard Freeze)

This round does not modify:
- state definition
- backbone family
- loss family
- Stage2
- WAN / MotionCrafter
- new data
- joint training
- new architecture/loss/optimizer search

## 3. Launcher GPU Selection Policy

`scripts/run_tracewm_stage1_v2_220m_longtrain_20260408.sh` must:
- use unified selector + lease (`select_single_gpu` + `acquire_lease`/`release_lease`),
- filter by `free_mem >= required_mem + safety_margin`,
- rank candidates by lowest short-window average GPU utilization first,
- use lease to avoid card collision,
- remain single-GPU-only.

Run metadata must include at least:
- `selected_gpu_id`
- `avg_gpu_util`
- `avg_mem_util`
- `free_mem_gb`
- `lease_id`
- `fallback_reason` (if recommended GPU not selected)

## 4. Fixed Continuation Run (Exactly 1)

Only one run is allowed:
- `stage1_v2_longtrain_220m_mainline_continue_10000`

Must resume from:
- `outputs/checkpoints/stage1_v2_longtrain_220m_mainline_20260408/latest.pt`

Target:
- `optimizer_steps_target = 10000`

Keep fixed:
- frozen 220M mainline recipe
- single-GPU runtime mode
- data contract
- loss family
- ranking protocol

## 5. Save/Eval Cadence (Unchanged)

- `save_every_n_steps = 1000`
- `eval_interval = 1000`
- maintain `best.pt` and `latest.pt`
- required new step checkpoints:
  - `step_0006000.pt`
  - `step_0007000.pt`
  - `step_0008000.pt`
  - `step_0009000.pt`
  - `step_0010000.pt`

## 6. Fixed Evaluation and Best Policy

Keep recording:
- `teacher_forced_coord_loss`
- `free_rollout_coord_mean_l2`
- `free_rollout_endpoint_l2`
- TAP-Vid eval
- TAPVid-3D limited eval
- `parameter_count`

`best.pt` update must use lexicographic ranking:
- primary: `free_rollout_endpoint_l2`
- secondary: `free_rollout_coord_mean_l2`
- tertiary: TAP-Vid endpoint l2
- quaternary: TAPVid-3D limited endpoint l2

`total_loss` is reference-only and cannot decide best checkpoint.

## 7. Required Output Artifacts

- `reports/stage1_v2_220m_longtrain_progress_20260408.json`
- `reports/stage1_v2_220m_longtrain_final_20260408.json`
- `reports/stage1_v2_220m_longtrain_10000_confirmation_20260408.json`
- `docs/STAGE1_V2_220M_LONGTRAIN_RESULTS_20260408.md`

Checkpoint directory must include at least:
- `best.pt`
- `latest.pt`
- `step_0001000.pt`
- `step_0002000.pt`
- `step_0003000.pt`
- `step_0004000.pt`
- `step_0005000.pt`
- `step_0006000.pt`
- `step_0007000.pt`
- `step_0008000.pt`
- `step_0009000.pt`
- `step_0010000.pt`

## 8. Fixed Runtime Envelope

- tmux session: `tracewm_stage1_v2_220m_longtrain_continue_20260408`
- log: `/home/chen034/workspace/stwm/logs/tracewm_stage1_v2_220m_longtrain_continue_20260408.log`
