# STWM Final Claim Boundary V1

## Allowed Claims
- STWM predicts future semantic trace fields under free rollout.
- Copy-gated residual semantic transition improves changed semantic states while preserving stable states.
- Works on mixed VSPW+VIPSeg protocol.
- Does not degrade trace dynamics.
- Belief association utility supports future identity association.

## Forbidden Claims
- STWM is SAM2/CoTracker plugin.
- STWM beats all trackers overall.
- Full RGB generation.
- Closed-loop planner.
- Universal OOD dominance.
- Hiding VIPSeg smaller effect.
- Claiming candidate scorer as method.

## Must Disclose
- VIPSeg changed-subset gain is positive but smaller than VSPW.
- Dedicated LODO cross-dataset training/eval is not yet completed.
- Horizon evidence is H=8 unless H16 appendix is run.
- Current field is best described as semantic trace-unit field unless K-scaling appendix is executed.
