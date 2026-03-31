# STWM V4.2 Qualitative Notes (Seed42)

Outputs:

- root: `outputs/visualizations/stwm_v4_2_minival_seed42`
- manifest: `outputs/visualizations/stwm_v4_2_minival_seed42/figure_manifest.json`

## Generated Groups

1. `semantic_sensitive_cases` (8 cases)
2. `identity_reconnect_cases` (8 cases)
3. `query_grounding_cases` (8 cases)

Each panel uses side-by-side comparison with GT/query markers and run-level trajectory/query errors.

## Selection Logic

1. semantic-sensitive:
   - prioritize clips where `wo_semantics_v4_2` degrades more than `full_v4_2` on trajectory+query error.
2. identity-reconnect:
   - prioritize clips where `wo_identity_v4_2` is worse on reconnect/trajectory/query criteria.
3. query-grounding:
   - prioritize clips where `full_v4_2` has lower query error than the worse ablation.

## Quick Quantitative Context on Selected Cases

- semantic-sensitive set (8):
  - mean delta (`wo_semantics - full`) trajectory_l1: `+0.000706`
  - mean delta (`wo_semantics - full`) query_error: `+0.012585`

- identity-reconnect set (8):
  - mean delta (`wo_identity - full`) trajectory_l1: `+0.015449`
  - mean delta (`wo_identity - full`) query_error: `+0.021820`

- query-grounding set (8):
  - mean delta (worst ablation query_error - full query_error): `+0.022792`

## What The Current Pack Supports

1. `full_v4_2` has visually and numerically better query grounding in selected hard cases.
2. Semantics ablation shows larger degradation than identity ablation on many query-sensitive samples.
3. Identity-specific reconnect evidence is still constrained by low event coverage in this seed42 run.

## What The Current Pack Does Not Prove Yet

1. It does not replace multi-seed statistical evidence.
2. It does not prove robust reconnect gains without eventful buckets.
3. It should support, not replace, quantitative tables.
