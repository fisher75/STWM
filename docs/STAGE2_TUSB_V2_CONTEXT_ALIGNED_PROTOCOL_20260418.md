# Stage2 TUSB-V2 Context-Aligned Protocol 20260418

- Stage1 remains frozen. No training, no unfreeze, no backbone swap.
- TUSB-v2 is already landed.
- anti-collapse is load-bearing.
- z_sem slower_than_z_dyn = true.
- multi-entity training sample path already exists.
- current flat protocol-v3 result does not automatically mean TUSB-v2 is ineffective.
- strong suspicion 1: protocol eval still uses single-target observed context.
- strong suspicion 2: best.pt checkpoint selection is rollout-aligned and can suppress semantic usefulness gains.
- current instance-aware path is still weak in the main training body: VIPSeg true continuity is only partial, VSPW is mostly pseudo/fallback.
- this round only repairs context alignment, checkpoint alignment, and true instance density. No protocol v4, no persistence, no Stage1 edits.
