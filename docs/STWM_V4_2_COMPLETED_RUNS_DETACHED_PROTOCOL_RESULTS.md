# STWM V4.2 Completed Runs Detached Protocol Results

Date: 2026-04-03  
Scope: four completed real runs, each evaluated at `best` and `latest` checkpoints (8 detached artifacts total)

## Execution Status

Detached protocol re-evaluation is complete:

- Requested: 8 / Produced: 8
- Protocol: `v2_3`
- Evaluator: `code/stwm/evaluators/eval_mini_val.py`
- Output root: `outputs/eval/stwm_v4_2_detached_protocol_eval_20260403/`
- All 8 artifacts report `model_config.family = stwm_v4_2`

## Artifact List

- `outputs/eval/stwm_v4_2_detached_protocol_eval_20260403/1b_seed123_full_v4_2_1b/best/detached_protocol_summary_v23.json`
- `outputs/eval/stwm_v4_2_detached_protocol_eval_20260403/1b_seed123_full_v4_2_1b/latest/detached_protocol_summary_v23.json`
- `outputs/eval/stwm_v4_2_detached_protocol_eval_20260403/1b_seed123_wo_semantics_v4_2_1b/best/detached_protocol_summary_v23.json`
- `outputs/eval/stwm_v4_2_detached_protocol_eval_20260403/1b_seed123_wo_semantics_v4_2_1b/latest/detached_protocol_summary_v23.json`
- `outputs/eval/stwm_v4_2_detached_protocol_eval_20260403/220m_seed42_full_v4_2/best/detached_protocol_summary_v23.json`
- `outputs/eval/stwm_v4_2_detached_protocol_eval_20260403/220m_seed42_full_v4_2/latest/detached_protocol_summary_v23.json`
- `outputs/eval/stwm_v4_2_detached_protocol_eval_20260403/220m_seed42_wo_semantics_v4_2/best/detached_protocol_summary_v23.json`
- `outputs/eval/stwm_v4_2_detached_protocol_eval_20260403/220m_seed42_wo_semantics_v4_2/latest/detached_protocol_summary_v23.json`

## Unified Results Table

| Scale | Seed | Run | Checkpoint | disable_semantics | num_clips | future_trajectory_l1 | query_localization_error | query_top1_acc | query_hit_rate | identity_consistency | identity_switch_rate | occlusion_recovery_acc | future_mask_iou | visibility_accuracy | visibility_f1 |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1b | 123 | full_v4_2_1b | best | 0 | 9 | 0.270876 | 0.257365 | 0.111111 | 0.111111 | 0.111111 | 0.888889 | 0.000000 | 0.004277 | 0.791667 | 0.826211 |
| 1b | 123 | full_v4_2_1b | latest | 0 | 9 | 0.270876 | 0.257365 | 0.111111 | 0.111111 | 0.111111 | 0.888889 | 0.000000 | 0.004277 | 0.791667 | 0.826211 |
| 1b | 123 | wo_semantics_v4_2_1b | best | 1 | 9 | 0.271105 | 0.269591 | 0.111111 | 0.111111 | 0.111111 | 0.888889 | 0.000000 | 0.004276 | 0.791667 | 0.826211 |
| 1b | 123 | wo_semantics_v4_2_1b | latest | 1 | 9 | 0.270165 | 0.258985 | 0.111111 | 0.111111 | 0.111111 | 0.888889 | 0.000000 | 0.004282 | 0.791667 | 0.826211 |
| 220m | 42 | full_v4_2 | best | 0 | 9 | 0.272424 | 0.261929 | 0.111111 | 0.111111 | 0.000000 | 1.000000 | 0.000000 | 0.004267 | 0.791667 | 0.826211 |
| 220m | 42 | full_v4_2 | latest | 0 | 9 | 0.270063 | 0.267234 | 0.000000 | 0.111111 | 0.111111 | 0.888889 | 0.000000 | 0.004281 | 0.791667 | 0.826211 |
| 220m | 42 | wo_semantics_v4_2 | best | 1 | 9 | 0.270269 | 0.257490 | 0.111111 | 0.111111 | 0.000000 | 1.000000 | 0.000000 | 0.004282 | 0.791667 | 0.826211 |
| 220m | 42 | wo_semantics_v4_2 | latest | 1 | 9 | 0.270516 | 0.279277 | 0.111111 | 0.111111 | 0.111111 | 0.888889 | 0.000000 | 0.004279 | 0.791667 | 0.826211 |

## Semantics Delta Table (wo_semantics - full)

Interpretation:

- lower-is-better metrics (`future_trajectory_l1`, `query_localization_error`): positive delta => semantics helps
- higher-is-better metrics (`query_top1_acc`, `identity_consistency`): negative delta => semantics helps
- for `identity_switch_rate` (lower is better): positive delta => semantics helps

| Scale | Seed | Checkpoint | delta future_trajectory_l1 (wo-full) | delta query_localization_error (wo-full) | delta query_top1_acc (wo-full) | delta identity_consistency (wo-full) | delta identity_switch_rate (wo-full) |
|---|---:|---|---:|---:|---:|---:|---:|
| 1b | 123 | best | +0.000229 | +0.012226 | +0.000000 | +0.000000 | +0.000000 |
| 1b | 123 | latest | -0.000711 | +0.001620 | +0.000000 | +0.000000 | +0.000000 |
| 220m | 42 | best | -0.002155 | -0.004438 | +0.000000 | +0.000000 | +0.000000 |
| 220m | 42 | latest | +0.000453 | +0.012043 | +0.111111 | +0.000000 | +0.000000 |

## Key Findings

1. Compatibility objective is met.
   - All 8 detached artifacts are successfully generated from V4.2 checkpoints.
2. Semantic mainline signal is present but not universal.
   - On `query_localization_error`, removing semantics worsens 3/4 pairs.
   - Mean delta over 4 pairs: `+0.005363` (wo-full), which favors semantics.
   - One pair reverses (`220m best`), so claim should be directional, not absolute.
3. Identity/reconnect metrics are currently not strong enough for superiority claims.
   - `identity_consistency` remains low (`0.000000` or `0.111111`).
   - `identity_switch_rate` remains high (`0.888889` or `1.000000`).
   - `occlusion_recovery_acc` is `0.000000` for all 8 artifacts.
4. Best-vs-latest differences are mixed and mostly small; one notable discrete jump appears in `query_top1_acc` at `220m latest`.

## Four Hard Questions (Explicit Answers)

1. Is detached compatibility for STWM V4.2 real checkpoints complete?
   - Yes. A full 8/8 run x checkpoint matrix finished with valid `stwm_v4_2` summaries.

2. What is the strongest defensible metric family now?
   - Tier A: `query_localization_error`, `query_top1_acc`, `query_hit_rate`, `identity_consistency`, `identity_switch_rate`, and `occlusion_recovery_acc` (only when event coverage is meaningful).
   - Tier B support: `future_trajectory_l1`, `future_mask_iou`, `visibility_accuracy`, `visibility_f1`.

3. Does detached evidence support a semantic mainline claim?
   - Partially yes for query grounding direction: semantics tends to improve query localization (3/4 pairwise wins, positive mean wo-full delta on lower-better metric).
   - Not a universal claim: one checkpoint pair reverses, and identity/reconnect metrics do not improve.

4. Should protocol diagnosis remain ahead of training-protocol rewrites?
   - Yes. Current bottleneck is evaluator signal quality/coverage for identity and occlusion axes, not a lack of detached compatibility.
   - Training changes should remain secondary until protocol-level discriminability on Tier A identity/reconnect metrics improves.
