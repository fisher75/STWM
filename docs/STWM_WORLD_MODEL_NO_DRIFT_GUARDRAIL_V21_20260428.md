# STWM World Model No-Drift Guardrail V21

## Allowed

- Controlled Stage2 semantic branch unfreeze after trainability audit.
- Frozen Stage1.
- Structured semantic prototype field output.
- CLIP/DINO/SigLIP features only as pseudo-label/prototype construction.
- Future GT crops as supervision targets only.

## Forbidden

- Treating direct CLIP feature regression as final semantic field.
- Candidate scorer as main method.
- SAM2/CoTracker plugin framing.
- Future candidate leakage into rollout input.
- Full Stage2/Stage1 unfreeze without audit.
- Paper claim before robust free-rollout semantic field signal.

V2 keeps the world-model framing intact, but does not yet produce a positive free-rollout semantic prototype field signal.
