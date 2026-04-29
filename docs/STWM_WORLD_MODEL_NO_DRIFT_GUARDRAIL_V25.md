# STWM World Model No-Drift Guardrail V25

Allowed:
- observed semantic memory as part of world state
- residual semantic prototype prediction
- future GT semantic target as supervision
- Stage1 frozen
- trace dynamic path frozen

Forbidden:
- candidate scorer
- SAM2/CoTracker plugin
- future candidate leakage
- predicting semantic field without observed semantic state
- paper claim before semantic memory eval
- continuing direct_logits if copy baseline dominates
