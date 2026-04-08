# TRACEWM Stage1-v2 Mainline Confirmation Protocol (2026-04-08)

## 1. Objective

Run one tightly-scoped Stage1-v2 mainline confirmation round to determine whether the current `prototype_220m` path can now surpass `debug_small` on the frozen Stage1-v2 endpoint-driven ranking under a higher but controlled unified budget.

This round is confirmation-only and must not expand scope into new search spaces.

## 2. Frozen Facts and Constraints

- Keep Stage1-v2 data contract and Stage1 minisplit fixed.
- Keep model family and architecture fixed: only `debug_small` and `prototype_220m`.
- Keep objective family fixed: no new loss family introduction.
- Keep execution mode fixed to single-GPU recommended runtime policy.
- No Stage2, no WAN/MotionCrafter VAE work, no data expansion, no protocol/search explosion.
- Use exactly three runs in this round.

## 3. Allowed Adjustments

- Budget is allowed to be significantly higher than short-budget: set `train_steps > 40`.
- For this round, set unified budget to:
  - `epochs=1`
  - `train_steps=96`
  - `eval_steps=12`
- Keep remaining data-loader and token controls consistent with prior controlled rounds.

## 4. Fixed Run Matrix (Exactly 3)

1. `stage1_v2_confirm_debugsmall_mainline`
   - `model_mode=debug_small`
   - purpose: frozen mainline anchor under confirmation budget.
2. `stage1_v2_confirm_220m_ref`
   - `model_mode=prototype_220m`
   - purpose: baseline 220M reference under confirmation budget.
3. `stage1_v2_confirm_220m_bestrecipe`
   - `model_mode=prototype_220m`
   - recipe source: best validated recipe from 220M gap-closure outputs.

No additional runs are permitted in this protocol.

## 5. Fixed Evaluation and Ranking Policy

Primary endpoint (must win first):
- `val_metrics.rollout_l2_endpoint`

Secondary (tie-break 1):
- `val_metrics.rollout_l2_mean`

Tertiary (tie-break 2):
- `tapvid_metrics.tapvid_l2_endpoint`

Quaternary (tie-break 3):
- `tapvid3d_metrics.l2_endpoint_limited`

Reference-only:
- `best_val_loss_total`

Selection is lexicographic by the above order and based on lower-is-better endpoint errors.

## 6. Required Output Artifacts

- `reports/stage1_v2_mainline_confirmation_runs_20260408.json`
- `reports/stage1_v2_mainline_confirmation_comparison_20260408.json`
- `docs/STAGE1_V2_MAINLINE_CONFIRMATION_RESULTS_20260408.md`
- `logs/tracewm_stage1_v2_mainline_confirmation_20260408.log`

## 7. Decision Semantics

The round must explicitly output:

- `best_small_run`
- `best_220m_run`
- `does_220m_now_surpass_small`
- `final_mainline_decision`
- `next_step_choice`

Decision rule:
- If best 220M is lexicographically better than best small under fixed ranking, set `does_220m_now_surpass_small=true` and `final_mainline_decision=promote_220m_mainline`.
- Otherwise keep `final_mainline_decision=retain_debugsmall_mainline` and set next step to controlled stop/refine without opening new scope in this round.

## 8. Execution Contract

- tmux session name is fixed:
  - `tracewm_stage1_v2_mainline_confirmation_20260408`
- launcher script is fixed:
  - `scripts/start_tracewm_stage1_v2_mainline_confirmation_tmux_20260408.sh`
- run script is fixed:
  - `scripts/run_tracewm_stage1_v2_mainline_confirmation_20260408.sh`
- log file is fixed:
  - `/home/chen034/workspace/stwm/logs/tracewm_stage1_v2_mainline_confirmation_20260408.log`
