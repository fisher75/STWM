# TRACEWM Stage1-v2 220M Long-Train Protocol (2026-04-08)

## 1. Frozen Facts

1. The Stage1-v2 mainline freeze round has already selected the backbone decision:
   - `final_stage1_backbone_decision = freeze_220m_as_stage1_backbone`
2. The current question is no longer backbone selection. The remaining gap is infrastructure and formalization:
   - missing formal long-train infrastructure
   - missing `save_every_1000` / `best` / `latest` / `resume`
   - the 192-step freeze result is not the final long-train paper-level main result

## 2. Scope Boundary (No Expansion)

This round does not modify:
- state definition
- backbone family
- loss family
- perf tooling
- data contract
- Stage2
- WAN / MotionCrafter
- joint training / new search

## 3. Mandatory Long-Train Infrastructure

The trainer `code/stwm/tracewm_v2/trainers/train_tracewm_stage1_v2.py` must support and this round must use:

1. `checkpoint_dir` as the unified checkpoint root
2. `save_every_n_steps` with round default set to 1000
3. `best.pt` update by fixed ranked metrics
4. `latest.pt` update on every save event
5. periodic step checkpoints in format `step_0001000.pt`, `step_0002000.pt`, ...
6. `resume_from` recovery from `latest.pt` or explicit `step_xxxxxxx.pt`
7. checkpoint payload containing at least:
   - model state dict
   - optimizer state dict
   - scheduler state dict (if any)
   - global_step
   - epoch
   - best_metric_so_far
   - config / run metadata
8. resume is full-state resume (optimizer state included), not model-only weight load

## 4. Fixed Run Matrix (Exactly 1)

Only one run is allowed:
- `stage1_v2_longtrain_220m_mainline`

No debugsmall extra run, no additional sweep, no new optimizer/loss/backbone/state search.

## 5. Training Budget and Cadence

This round runs:
- `optimizer_steps = 5000`

With fixed checkpointing/evaluation contract:
- `save_every_n_steps = 1000`
- evaluate at fixed cadence `eval_interval = 1000`
- required step checkpoints include at least:
  - `step_0001000.pt`
  - `step_0002000.pt`
  - `step_0003000.pt`
  - `step_0004000.pt`
  - `step_0005000.pt`
- always maintain `best.pt` and `latest.pt`

The report must explicitly include:
- optimizer_steps
- effective_batch
- epochs
- eval_interval
- save_every_n_steps

## 6. Fixed Evaluation and Best Selection Policy

Each evaluation must record:
1. `teacher_forced_coord_loss`
2. `free_rollout_coord_mean_l2`
3. `free_rollout_endpoint_l2`
4. TAP-Vid eval
5. TAPVid-3D limited eval

Fixed best-ranking chain:
- primary: `free_rollout_endpoint_l2`
- secondary: `free_rollout_coord_mean_l2`
- tertiary: TAP-Vid endpoint L2
- quaternary: TAPVid-3D limited endpoint L2

`best.pt` is updated by this fixed lexicographic ranking only. `total_loss` is reference-only.

## 7. Runtime and Shared Cluster Policy

- Keep current single-GPU recommended runtime policy.
- Keep recommended GPU-first selection.
- If selected GPU memory is insufficient at launch time, allow fallback to another qualifying single GPU.
- Record GPU selection/fallback metadata into run metadata.

## 8. Required Outputs

- `reports/stage1_v2_220m_longtrain_progress_20260408.json`
- `reports/stage1_v2_220m_longtrain_final_20260408.json`
- `docs/STAGE1_V2_220M_LONGTRAIN_RESULTS_20260408.md`

Checkpoint directory:
- `outputs/checkpoints/stage1_v2_longtrain_220m_mainline_20260408/`

Expected files include:
- `best.pt`
- `latest.pt`
- `step_0001000.pt`
- `step_0002000.pt`
- `step_0003000.pt`
- `step_0004000.pt`
- `step_0005000.pt`

## 9. Fixed Execution Contract

- tmux session:
  - `tracewm_stage1_v2_220m_longtrain_20260408`
- run script:
  - `scripts/run_tracewm_stage1_v2_220m_longtrain_20260408.sh`
- tmux launcher:
  - `scripts/start_tracewm_stage1_v2_220m_longtrain_tmux_20260408.sh`
- log file:
  - `/home/chen034/workspace/stwm/logs/tracewm_stage1_v2_220m_longtrain_20260408.log`
