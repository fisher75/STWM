# STWM V4.2 220M Protocol Diagnostic Matrix V1 (D1)

Date: 2026-04-03
Status: READY AND SUBMITTABLE (background-only)
Scale scope: 220M only

## Frozen Protocol Preconditions

This matrix is defined under frozen policy:

- evaluator freeze: v2_4_detached_frozen via code/stwm/evaluators/eval_mini_val.py
- split freeze: manifests/protocol_v2/train_v2.json and protocol_val_main_v1.json
- official best freeze: best_protocol_main.pt with protocol_best_rule_v2

## Matrix V1 Run Set

Seed fixed at 42 for first batch.

1. full_v4_2_seed42_fixed_nowarm_lambda1
   - full model
   - fixed lambda
   - no semantic warm-up
2. full_v4_2_seed42_fixed_warmup_lambda1
   - full model
   - fixed lambda
   - semantic warm-up enabled
3. wo_semantics_v4_2_seed42
   - semantics branch disabled
4. wo_object_bias_v4_2_seed42
   - object bias neutralized
5. full_v4_2_seed42_lsem_0p1_lambda0
   - full model
   - lambda_sem = 0.1 * lambda0
6. full_v4_2_seed42_lsem_0p25_lambda0
   - full model
   - lambda_sem = 0.25 * lambda0
7. full_v4_2_seed42_lsem_0p5_lambda0
   - full model
   - lambda_sem = 0.5 * lambda0
8. lambda_sem = 1.0 * lambda0
   - represented by run #1 (same configuration), no duplicate launch to minimize cost

## Why This Subset Is Highest-Information / Lowest-Cost

1. One fixed seed first isolates protocol behavior and avoids immediate seed-explosion.
2. full versus wo_semantics versus wo_object_bias gives highest causal contrast at minimal branch count.
3. Lambda sweep is limited to the semantic axis on full model only, avoiding combinatorial blow-up.
4. Warm-up is tested once at lambda_sem target setting to validate schedule effect before advanced methods.

## Exact Warm-up Rule

Semantic warm-up applies only to lambda_sem:

- first 10% steps: lambda_sem = 0
- 10% to 30% steps: linear ramp to target lambda_sem
- after 30%: hold target lambda_sem constant

Other loss weights stay fixed.

## Exact Lambda Values

- lambda0 = 0.5
- sweep set:
  - 0.1 * lambda0 = 0.05
  - 0.25 * lambda0 = 0.125
  - 0.5 * lambda0 = 0.25
  - 1.0 * lambda0 = 0.5

## Gradient Audit Signals

Shared trunk audit records:

- ||g_traj||
- ||g_query||
- ||g_sem||
- cos(g_sem, g_traj)
- cos(g_sem, g_query)

Output artifacts:

- reports/stwm_v4_2_gradient_audit_220m_seed42_*.json

## Checkpoint / Official Best Policy for D1

1. latest.pt
   - resume only
2. best.pt
   - trainer-internal reference only
3. best_protocol_main.pt
   - only official best for protocol-frozen claims
4. every 500 steps
   - run detached protocol eval on protocol_val_main_v1
   - update official best via protocol_best_rule_v2

## Background Submission Contract

Long jobs are submitted offline only:

- submission script: scripts/submit_stwm_v4_2_220m_protocol_diag_matrix_v1.sh
- each run returns PID and dedicated log file path immediately after submission

## Entry Gates for Advanced Methods

Uncertainty weighting / GradNorm are not enabled in V1. They are allowed only after:

1. fixed-lambda matrix produces stable protocol-main direction across baseline and lambda sweep
2. gradient audit shows persistent scale imbalance or destructive gradient alignment patterns

PCGrad is not enabled in V1. It is allowed only after:

1. protocol-main stalls or degrades despite best fixed-lambda candidate
2. gradient audit confirms persistent negative alignment with non-trivial magnitude
