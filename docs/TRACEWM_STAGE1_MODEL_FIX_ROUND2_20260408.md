# TraceWM Stage 1 Model-Fix Round 2 (2026-04-08)

## Frozen Conclusions From Previous Fix Round

1. `balanced_sampler` is the overall best fix in the previous round.
2. `loss_normalized` is best on TAP-Vid and TAPVid-3D limited.
3. `source_conditioned` is not promoted.
4. No previous fix surpasses best single.

## Round 2 Only Mainline

- The only target in this round is to verify whether `balanced sampler + loss normalization` can make joint truly surpass best single.

## Still Forbidden In Round 2

- Stage 2 semantics
- WAN
- MotionCrafter VAE
- DynamicReplica
- New source-conditioned variants
- New data

## Allowed Experiment Matrix (Exactly 3)

1. `tracewm_stage1_fix2_joint_balanced_lossnorm`
   - Joint training with balanced sampler + loss normalization.
   - No warmup.
2. `tracewm_stage1_fix2_point_warmup_then_joint_balanced_lossnorm`
   - PointOdyssey warmup, then switch to joint balanced+lossnorm.
   - No extra tricks.
3. `tracewm_stage1_fix2_kubric_warmup_then_joint_balanced_lossnorm`
   - Kubric warmup, then switch to joint balanced+lossnorm.
   - No extra tricks.

## Baseline Reuse (No Re-Run)

- `tracewm_stage1_iter1_pointodyssey_only`
- `tracewm_stage1_iter1_kubric_only`

## Eval/Protocol Lock

- Keep the same data contract and split protocol used in Stage 1.
- Keep both teacher-forced and free-rollout support enabled.
- Main evaluation: TAP-Vid.
- Limited evaluation: TAPVid-3D.
