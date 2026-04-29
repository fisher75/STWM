# STWM World Model No-Drift Guardrail V26

## Allowed
- observed semantic memory as current world state
- semantic persistence baseline
- residual semantic transition
- Stage1 frozen
- dynamic trace path frozen

## Forbidden
- candidate scorer
- SAM2/CoTracker plugin framing
- future candidate leakage
- silent reuse of low-coverage observed cache
- medium training before observed coverage is repaired
- blaming target/model before cache coverage audit
