# STWM World Model No-Drift Guardrail V22

## Allowed

- Semantic-only Stage2/TUSB unfreeze after boundary audit.
- Stage1 frozen.
- Dynamic/trace path frozen.
- Structured semantic prototype field output.
- CLIP/DINO/SigLIP only as pseudo-label/prototype construction.

## Forbidden

- Calling head/proj-only training semantic branch unfreeze.
- Full Stage2 unfreeze without boundary audit.
- Future candidate leakage.
- SAM2/CoTracker plugin framing.
- Candidate scorer as main method.
- CLIP vector regression as final semantic field.

V1 confirms the boundary can be made correct, but the semantic prototype field signal is still negative, so paper-level world-model claims remain forbidden.
