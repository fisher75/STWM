# STWM Docs Cleanup V12 20260502

## Purpose

This cleanup reduces generated documentation noise in `docs/` while preserving the useful STWM-FSTF V12 results.

Machine-readable evidence remains in `reports/`. Human-readable evidence is kept in the consolidated and summary docs listed below.

## Retained Summary Docs

- `docs/STWM_DOCS_CLEANUP_V12_20260502.md`
- `docs/STWM_WORLD_MODEL_NO_DRIFT_GUARDRAIL_V44.md`
- `docs/STWM_FSTF_FULL_SCALING_LAWS_V12_20260502.md`
- `docs/STWM_FSTF_HORIZON_CACHE_H16_V12_20260502.md`
- `docs/STWM_FSTF_HORIZON_CACHE_H24_V12_20260502.md`
- `docs/STWM_FSTF_TRACE_DENSITY_CACHE_K16_V12_20260502.md`
- `docs/STWM_FSTF_TRACE_DENSITY_CACHE_K32_V12_20260502.md`
- `docs/STWM_FSTF_HORIZON_SCALING_V12_20260502.md`
- `docs/STWM_FSTF_TRACE_DENSITY_SCALING_V12_20260502.md`
- `docs/STWM_FSTF_PROTOTYPE_SCALING_V12_20260502.md`
- `docs/STWM_FSTF_MODEL_SIZE_SCALING_V12_20260502.md`
- `docs/STWM_FSTF_V12_CVPR_READINESS_GATE_20260502.md`
- `docs/STWM_FSTF_VISUALIZATION_V12_20260502.md`

## Retained Machine Evidence

- `reports/stwm_fstf_full_scaling_laws_v12_20260502.json`
- `reports/stwm_fstf_horizon_cache_h16_v12_20260502.json`
- `reports/stwm_fstf_horizon_cache_h24_v12_20260502.json`
- `reports/stwm_fstf_trace_density_cache_k16_v12_20260502.json`
- `reports/stwm_fstf_trace_density_cache_k32_v12_20260502.json`
- `reports/stwm_fstf_horizon_scaling_v12_20260502.json`
- `reports/stwm_fstf_trace_density_scaling_v12_20260502.json`
- `reports/stwm_fstf_prototype_scaling_v12_20260502.json`
- `reports/stwm_fstf_model_size_scaling_v12_20260502.json`
- Per-shard JSON reports under `reports/stwm_fstf_*_shard*_v12_20260502.json`

## Preserved Result Facts

- Scaling evals completed: `39 / 39`
- New scaling checkpoint count: `39`
- New scaling eval summary count: `39`
- Prototype scaling positive: `true`
- Horizon scaling positive: `true`
- Trace-density scaling positive: `true`
- Model-size scaling positive: `true`
- Dense trace-field claim allowed by V12 density scaling: `true`
- Long-horizon claim allowed by V12 horizon scaling: `true`
- Raw-frame visualization ready: `true`
- Next step: `build_paper_figures_and_start_overleaf`

## Preserved Cache Facts

- H16 cache: train/val/test = `3019 / 647 / 647`, H=`16`, K=`8`, materialization_success=`true`
- H24 cache: train/val/test = `3019 / 647 / 647`, H=`24`, K=`8`, materialization_success=`true`
- K16 cache: train/val/test = `3019 / 647 / 647`, H=`8`, K=`16`, materialization_success=`true`
- K32 cache: train/val/test = `3019 / 647 / 647`, H=`8`, K=`32`, materialization_success=`true`

## Removed Redundant Docs

The removed files were generated low-level `.md` fragments whose information is already preserved in the retained summary docs and JSON reports:

- V12 horizon batch shard docs: `docs/STWM_FSTF_HORIZON_H*_BATCH_*_SHARD*_V12_20260502.md`
- V12 trace-density batch shard docs: `docs/STWM_FSTF_TRACE_DENSITY_K*_BATCH_*_SHARD*_V12_20260502.md`
- V12 horizon aggregate batch docs: `docs/STWM_FSTF_HORIZON_H*_BATCH_{train,val,test}_V12_20260502.md`
- V12 trace-density aggregate batch docs: `docs/STWM_FSTF_TRACE_DENSITY_K*_BATCH_{train,val,test}_V12_20260502.md`
- V12 horizon feature-target shard docs: `docs/STWM_FSTF_HORIZON_h*_{train,val}_shard*_FEATURE_TARGETS_V12_20260502.md`
- V12 trace-density feature-target shard docs: `docs/STWM_FSTF_TRACE_DENSITY_k*_{train,val}_shard*_FEATURE_TARGETS_V12_20260502.md`
- Legacy LODO/materialization shard docs: `docs/*SHARD*.md` and `docs/*shard*.md` after V12 cleanup

The legacy shard docs removed in the second pass were low-level LODO, fullscale V1, and mixed V2 shard fragments. Their useful information is preserved in the corresponding summary docs and JSON reports, including:

- `docs/STWM_FINAL_LODO_V3_20260428.md`
- `docs/STWM_FULLSCALE_SEMANTIC_TRACE_WORLD_MODEL_V1_TRAIN_SUMMARY_20260428.md`
- `docs/STWM_FULLSCALE_SEMANTIC_TRACE_WORLD_MODEL_V1_TEST_EVAL_20260428.md`
- `docs/STWM_MIXED_FULLSCALE_V2_TRAIN_SUMMARY_20260428.md`
- `docs/STWM_MIXED_FULLSCALE_V2_COMPLETE_DECISION_20260428.md`
- Current guardrail: `docs/STWM_WORLD_MODEL_NO_DRIFT_GUARDRAIL_V44.md`

## Cleanup Rule

Keep final summary docs and JSON evidence. Remove repeated per-shard markdown fragments unless they contain a unique human interpretation not present in reports.

Old no-drift guardrail markdown versions were also removed after retaining V44. V44 is the current effective guardrail and carries the latest forbidden/allowed claim boundary.
