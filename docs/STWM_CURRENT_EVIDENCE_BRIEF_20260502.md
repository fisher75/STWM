# STWM Current Evidence Brief

## Fixed Definition
- STWM-FSTF is Future Semantic Trace Field Prediction.
- Input: frozen video-derived trace state plus observed semantic memory.
- Output: future trace units and future semantic prototype field, with visibility/reappearance/identity auxiliaries only when valid.
- Forbidden: candidate scorer, SAM2/CoTracker as same-output baseline, future candidate leakage, raw-video end-to-end training claim, dense trace-field claim without valid K evidence.

## Current Claim Flags
- prototype_scaling_positive: `True`
- horizon_scaling_positive: `True`
- trace_density_scaling_positive: `weak_or_inconclusive`
- model_size_scaling_positive: `False`
- dense_trace_field_claim_allowed: `False`
- long_horizon_claim_allowed: `True`
- raw_visualization_pack_ready: `True`
- next_step_choice: `revise_claim_boundary_and_start_overleaf`

## Important Evidence
- V8: STWM significantly beats the strongest controlled copy-aware same-output baseline (`copy_residual_mlp`).
- V10: H8 future-hidden trace-rollout representation is load-bearing; old V9 no-trace ablation was not valid.
- V13 H16/H24 hidden audit: H16=`True`, H24=`True`.
- C selection: selected_C=`32`, C128_overfit_or_fail=`True`.
- K wording: `semantic trace-unit field`; dense claim allowed=`False`.

## Allowed Strong Claims
- STWM predicts future semantic trace-unit fields over frozen video-derived trace/semantic states.
- STWM improves changed semantic prototype prediction over copy and strong copy-aware baselines while preserving stable semantic memory.
- Future rollout hidden is load-bearing at H8 and remains load-bearing at H16/H24 under V13 hidden-shuffle/random intervention audits.
- C32 is selected as the best prototype vocabulary tradeoff; C128 fails the stability/granularity tradeoff.

## Forbidden Claims
- Raw-video end-to-end training.
- Full RGB video generation world model.
- Dense semantic trace field, because K16/K32 valid-unit coverage is weak/inconclusive.
- Model-size scaling is positive, because base/large do not beat small under strict grouped rules.
- Future trace coordinate or temporal order is load-bearing.
- Universal OOD dominance or universal cross-dataset generalization.
- STWM beats SAM2/CoTracker overall external SOTA or treats SAM2/CoTracker as same-output FSTF baselines.

## Primary Report Pointers
- `reports/stwm_fstf_v13_cvpr_readiness_gate_20260502.json`
- `reports/stwm_fstf_final_claim_boundary_v13_20260502.json`
- `reports/stwm_fstf_scaling_claim_verification_v13_20260502.json`
- `reports/stwm_fstf_trace_conditioning_horizon_v13_20260502.json`
- `reports/stwm_fstf_trace_density_valid_units_audit_v13_20260502.json`
- `reports/stwm_fstf_strong_copyaware_baseline_suite_v8_20260501.json`
