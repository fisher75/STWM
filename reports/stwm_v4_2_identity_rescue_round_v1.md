# STWM V4.2 Identity Rescue Round Summary

Out root: `/home/chen034/workspace/stwm/outputs/training/stwm_v4_2_identity_rescue_round`
Seeds: `42, 123`

## Variant Comparison (Eventful Delta, wo_identity - full)

| variant | d_traj | d_query | d_reconnect_success | full_better_reconnect_count | full_better_traj_count | full_better_query_count |
|---|---:|---:|---:|---:|---:|---:|
| control_resume_base | +0.004101 | +0.007811 | +0.058333 | 0 | 2 | 1 |
| resume_eventful_mix | -0.012588 | +0.014953 | +0.025000 | 0 | 0 | 2 |
| resume_eventful_hardquery_mix | +0.017975 | +0.005396 | +0.066667 | 0 | 2 | 1 |

## Hard-Query Decoupling Delta (wo_identity - full)

| variant | d_corr_abs | d_close_ratio | d_decoupling_score |
|---|---:|---:|---:|
| control_resume_base | +0.046812 | +0.000000 | -0.023406 |
| resume_eventful_mix | +0.054487 | -0.025000 | -0.014744 |
| resume_eventful_hardquery_mix | +0.043032 | +0.000000 | -0.021516 |

## Amplification vs Control (Reconnect Gap)

| variant | abs_gap_control | abs_gap_variant | amplified |
|---|---:|---:|---|
| resume_eventful_mix | 0.058333 | 0.025000 | False |
| resume_eventful_hardquery_mix | 0.058333 | 0.066667 | True |
