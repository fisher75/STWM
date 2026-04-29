# STWM World Model No-Drift Guardrail V20

## Allowed
- CLIP/DINO/SigLIP features may be used to build semantic prototypes or pseudo-labels.
- Final output is a structured semantic trace field.
- Future GT crops are supervision targets only.
- Rollout input remains observed video/trace only.

## Forbidden
- Treating high-dimensional CLIP regression as final semantic field.
- Candidate scorer as main method.
- SAM2/CoTracker plugin framing.
- Future candidate leakage.
- Paper claim before semantic field signal.
