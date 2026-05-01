# STWM Final Claim Boundary

## Allowed
- STWM predicts future semantic trace fields under free rollout.
- Copy-gated residual semantic transition improves changed semantic states while preserving stable states.
- Works on mixed VSPW+VIPSeg protocol.
- Does not degrade trace dynamics.
- Belief association utility supports future identity association.

## Forbidden
- STWM is SAM2/CoTracker plugin.
- STWM beats all trackers overall.
- Full RGB generation.
- Closed-loop planner.
- Universal OOD dominance.
- Hiding VIPSeg smaller effect.
- Claiming candidate scorer as method.

- audit_name: `stwm_final_claim_boundary`
- candidate_scorer_used: `False`
- future_candidate_leakage: `False`
- old_association_report_used: `False`
