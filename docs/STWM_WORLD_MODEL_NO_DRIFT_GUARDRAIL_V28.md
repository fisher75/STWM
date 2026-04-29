# STWM World Model No-Drift Guardrail V28

## Allowed
- Semantic memory copy prior.
- Sparse residual semantic transition.
- Structured semantic trace field output.
- Stage1 frozen.
- Dynamic trace path frozen.

## Forbidden
- Candidate scorer.
- SAM2/CoTracker plugin framing.
- Future candidate leakage.
- CLIP vector regression as main output.
- Full Stage1 unfreeze.
- Paper claim before heldout robust free-rollout signal.
