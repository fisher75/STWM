# TRACEWM Stage1 v2 Acceptance Criteria (2026-04-08)

## Gate P0: Real Trace Cache
Pass only if all conditions hold:
1. Contract manifest exists at `/home/chen034/workspace/data/_manifests/stage1_v2_trace_cache_contract_20260408.json`.
2. Audit report exists at `/home/chen034/workspace/stwm/reports/stage1_v2_trace_cache_audit_20260408.json`.
3. For each enabled dataset (PointOdyssey, Kubric), audit marks `status=pass`.
4. Zero deterministic synthetic source labels in cache metadata.
5. Required fields exist in sampled cache entries:
   - `tracks_2d`, `tracks_3d`, `valid`, `visibility`, `point_ids`.
6. Finite coordinate ratio above 0.99 on audited samples.

## Gate P1: Multi-Token State
Pass only if all conditions hold:
1. Dataloader emits shapes exactly:
   - `obs_state`: `[B, T_obs, K, D]`
   - `fut_state`: `[B, T_fut, K, D]`
2. No mean-pooling collapse before model input.
3. `token_mask` and per-time valid mask are present.
4. Batch smoke test reports at least one non-empty token lane (`K > 1`).

## Gate P2: Causal Transformer Backbone
Pass only if all conditions hold:
1. Model uses causal time mask.
2. Target production preset parameter count in range `[200M, 240M]`.
3. Forward pass on debug preset runs without NaN/Inf.
4. Output heads include at least:
   - coord head
   - visibility head
   - residual head
   - velocity head

## Gate P3: Structured Losses
Pass only if all conditions hold:
1. Loss stack includes:
   - coordinate loss
   - visibility loss
   - residual loss
   - velocity loss
2. Optional endpoint loss can be toggled.
3. Report logs each component and total loss separately.
4. Loss values are finite on smoke run.

## Gate G1-G5: Ablation Completion
Pass only if all conditions hold:
1. G1..G5 all executed or explicitly marked skipped with reason.
2. Consolidated JSON exists at `/home/chen034/workspace/stwm/reports/tracewm_stage1_v2_g1_g5_20260408.json`.
3. Consolidated markdown exists at `/home/chen034/workspace/stwm/docs/TRACEWM_STAGE1_V2_G1_G5_20260408.md`.
4. Mainline recommendation references only measured metrics from the above JSON.

## Non-Regression Scope Rules
Must remain true:
1. No Stage2 semantic branch in Stage1 v2 code path.
2. No WAN/MotionCrafter VAE/DynamicReplica/reconstruction branch in Stage1 v2 path.
3. No reuse of old iter/fix/final_rescue trainer entrypoints as mainline.
