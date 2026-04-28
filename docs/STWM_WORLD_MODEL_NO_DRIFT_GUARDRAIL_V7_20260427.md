# STWM World Model No Drift Guardrail V7

## Allowed
- Reappearance supervision is world-state supervision.
- Event-level reappearance is closer to future semantic trajectory world modeling.
- Association remains utility, not method definition.

## Forbidden
- Training reappearance on all non-risk slots as negatives.
- Comparing head-only to one lucky random init.
- Claiming reappearance learned if AP < positive-rate baseline.
- Joint training before head-only v2 signal.
- Moving to 1B before reappearance signal is positive.
