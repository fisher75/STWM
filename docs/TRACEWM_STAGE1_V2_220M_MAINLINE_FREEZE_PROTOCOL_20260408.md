# TRACEWM Stage1-v2 220M Mainline Freeze Protocol (2026-04-08)

## 1. Current Frozen Facts

1. Scientific revalidation, 220M gap closure, and mainline confirmation are already completed.
2. Current confirmation outcome is frozen as:
   - `does_220m_now_surpass_small = true`
   - `final_mainline_decision = promote_220m_as_mainline`
3. The problem is no longer whether 220M can compete; this round only checks whether 220M can remain stable as Stage1 backbone under a longer but controlled budget.
4. This round does not modify:
   - data contract
   - perf configuration
   - state definition
   - loss family
   - Stage2
   - WAN / MotionCrafter
   - new data

## 2. Scope Boundary

- Only Stage1-v2 freeze confirmation is allowed.
- No optimizer/loss/backbone/state search.
- No joint training or Stage2.
- No extra sweep beyond the fixed two-run matrix.

## 3. Fixed Run Matrix (Exactly 2)

1. `stage1_v2_freeze_220m_mainline`
   - source recipe: mainline confirmation winner (`prototype_220m bestrecipe`)
   - role: only mainline freeze run

2. `stage1_v2_freeze_debugsmall_ref`
   - source recipe: current best small recipe from mainline confirmation
   - role: final reference only, no search

## 4. Training Budget Policy

- Both runs must use the exact same budget level.
- Budget must be clearly above confirmation budget (`train_steps=96`) while still controlled.
- Freeze-round budget is fixed to:
  - `epochs=1`
  - `train_steps=192`
  - `eval_steps=16`

Both runs must output real values for `optimizer_steps`, `effective_batch`, `epochs`, and `eval_steps`.

## 5. Fixed Evaluation and Ranking

Each run must output:
- `teacher_forced_coord_loss`
- `free_rollout_coord_mean_l2`
- `free_rollout_endpoint_l2`
- TAP-Vid eval
- TAPVid-3D limited eval
- `parameter_count`

Winner ranking is strictly lexicographic:
- primary: `free_rollout_endpoint_l2`
- secondary: `free_rollout_coord_mean_l2`
- tertiary: TAP-Vid endpoint L2
- quaternary: TAPVid-3D limited endpoint L2

`total_loss` is reference-only and cannot be used to select winner.

## 6. Required Output Artifacts

- `reports/stage1_v2_220m_mainline_freeze_runs_20260408.json`
- `reports/stage1_v2_220m_mainline_freeze_comparison_20260408.json`
- `docs/STAGE1_V2_220M_MAINLINE_FREEZE_RESULTS_20260408.md`

Comparison must explicitly answer:
1. whether 220M mainline is still better than debugsmall under longer budget;
2. `final_stage1_backbone_decision` in:
   - `freeze_220m_as_stage1_backbone`
   - `keep_220m_as_candidate_but_not_frozen`
   - `revert_to_debugsmall`
3. exact blocking reason if not frozen;
4. `next_step_choice` in:
   - `freeze_stage1_and_prepare_stage2`
   - `do_one_last_stage1_followup`
   - `revert_to_small_backbone`

## 7. Execution Contract

- single-GPU recommended runtime policy remains unchanged.
- fixed tmux session:
  - `tracewm_stage1_v2_220m_mainline_freeze_20260408`
- fixed runner:
  - `scripts/run_tracewm_stage1_v2_220m_mainline_freeze_20260408.sh`
- fixed tmux launcher:
  - `scripts/start_tracewm_stage1_v2_220m_mainline_freeze_tmux_20260408.sh`
- fixed log:
  - `/home/chen034/workspace/stwm/logs/tracewm_stage1_v2_220m_mainline_freeze_20260408.log`
