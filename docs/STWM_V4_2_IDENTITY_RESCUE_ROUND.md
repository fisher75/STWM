# STWM V4.2 Identity Rescue Round (Final, Controlled)

## Scope And Boundaries

This round is a final, strictly controlled identity rescue round.

- frozen architecture
- frozen loss composition
- no evaluator mainline changes
- no 1B scaling
- no module additions

## Experimental Design

Seeds: `42, 123`

Runs:

- `full_v4_2`
- `wo_identity_v4_2`

Variants:

1. `control_resume_base`
2. `resume_eventful_mix`
3. `resume_eventful_hardquery_mix`

Continuation settings:

- continuation steps: `60`
- eval-only steps (each protocol): `60`
- sample limit: `18`

Coverage protocols:

1. base minival
2. eventful protocol
3. hard-query protocol

Main summary source:

- `reports/stwm_v4_2_identity_rescue_round_v1.json`
- `reports/stwm_v4_2_identity_rescue_round_v1.md`

## Variant Comparison Table (Eventful, wo_identity - full)

| Variant | d_trajectory_l1 | d_query_localization_error | d_reconnect_success_rate | full_better_traj_count (2 seeds) | full_better_query_count (2 seeds) | full_better_reconnect_count (2 seeds) |
|---|---:|---:|---:|---:|---:|---:|
| control_resume_base | +0.004101 | +0.007811 | +0.058333 | 2 | 1 | 0 |
| resume_eventful_mix | -0.012588 | +0.014953 | +0.025000 | 0 | 2 | 0 |
| resume_eventful_hardquery_mix | +0.017975 | +0.005396 | +0.066667 | 2 | 1 | 0 |

Interpretation:

- `d_* > 0` means `wo_identity` is worse than `full` for lower-better metrics.
- For reconnect success (higher-better), `d_reconnect_success_rate > 0` means `wo_identity` is better.

Result:

- In all three variants, reconnect success favors `wo_identity_v4_2` (full_better_reconnect_count = 0/2).
- Trajectory/query differences remain unstable across variants and metrics.

## Hard-Query Decoupling Delta (wo_identity - full)

| Variant | delta_corr_abs | delta_close_ratio | delta_decoupling_score |
|---|---:|---:|---:|
| control_resume_base | +0.046812 | +0.000000 | -0.023406 |
| resume_eventful_mix | +0.054487 | -0.025000 | -0.014744 |
| resume_eventful_hardquery_mix | +0.043032 | +0.000000 | -0.021516 |

Interpretation:

- `delta_decoupling_score < 0` means `full` has better decoupling than `wo_identity`.
- Hard-query interventions did not reverse identity gap on reconnect, but full keeps modest decoupling advantage.

## Eventful Bucket Power And Direction

Per-variant eventful bucket reports all indicate:

- sufficient_for_reconnect_claim: `True`
- paired seeds for comparison: `2`
- total event rows are non-zero and matched across runs

Yet reconnect direction is consistent against full:

- control: delta reconnect success (`wo - full`) = `+0.102941`
- eventful mix: `+0.044118`
- eventful+hardquery mix: `+0.117647`

## Required Questions (Direct Answers)

1. Did eventful/hard-query oversampling amplify `full_v4_2` vs `wo_identity_v4_2` difference?
   - **Partially yes in magnitude, but in the wrong direction for identity claim.**
   - `resume_eventful_hardquery_mix` increases reconnect gap magnitude vs control, but that gap favors `wo_identity`.

2. Is the difference mainly reconnect_success or also trajectory/query?
   - **Mainly reconnect_success direction is consistent; trajectory/query remain unstable.**
   - Trajectory/query sign consistency is mixed across variants.

3. If only reconnect_success has small stable advantage, is that enough to keep identity as secondary claim?
   - **No under this round's evidence.**
   - In this round reconnect advantage is not for `full`; therefore identity cannot be retained as a positive secondary claim.

4. If identity is still not hard after this round, should we formally downgrade?
   - **Yes.**
   - Recommended: formally downgrade identity/reconnect to secondary analysis, and keep main paper centered on semantic trajectory world state + query decoupling.

## Final Decision Boundary

- Do not run generic long training to continue identity rescue.
- Do not escalate to 1B.
- Keep architecture frozen.
- Move headline and main tables to semantics + query decoupling.
- Keep identity/reconnect as constrained, negative/neutral secondary analysis.
