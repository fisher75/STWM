# STWM V4.2 Detached Protocol Evaluator Compatibility Patch

Date: 2026-04-03  
Owner: evaluation pipeline hardening (detached, read-only vs running training)

## Objective

Make `code/stwm/evaluators/eval_mini_val.py` directly executable on STWM V4.2 real checkpoints (220m/1b) without changing any running training lanes.

## Pre-Patch Failure

Detached protocol eval failed on real V4.2 checkpoints due to model-config mismatch:

- Legacy evaluator path expected `STWMConfig`/`STWM1B`.
- V4.2 real checkpoints carry `STWMV42Config` fields (for example `trace_dim`).
- Representative failure:

```text
TypeError: STWMConfig.__init__() got an unexpected keyword argument 'trace_dim'
```

## Patch Scope

Single-file compatibility upgrade in:

- `code/stwm/evaluators/eval_mini_val.py`

No changes were made to queue workers, active lane schedulers, or running training processes.

## What Changed

### 1. Dual-family checkpoint loading

- Added V4.2 imports:
  - `STWMV42`, `STWMV42Config`, `load_model_config_v4_2`
- Added `_torch_load_compat(...)` to load checkpoints safely across torch variants.
- In `main()`, checkpoint `model_config` is inspected:
  - If `trace_dim` exists -> load via V4.2 family (`stwm_v4_2`).
  - Else -> keep legacy `stwm_1b` loading path.

### 2. V4.2 feature construction aligned with real trainer contract

- Added `_read_mask_ratio(...)`.
- Added `_build_v4_2_features_for_sample(...)` to build:
  - `trace_features`
  - `semantic_features`
  - `prior_features`
  - `teacher_objectness`

This mirrors the required input surface for `STWMV42.forward(...)` and preserves ablation switches.

### 3. V4.2 output adaptation in evaluator core

In `evaluate_model(...)`, introduced `model_family` branch:

- For `stwm_v4_2`, read outputs from:
  - `trajectory`
  - `visibility`
  - `semantic_logits`
  - `token_time_attention`
  - `query_token_logits`
- Query-frame selection now prefers V4.2 query-token attention when available, then falls back to semantic-energy rule.

### 4. Family-aware summary metadata

`summary["model_config"]` now records family-specific fields:

- `stwm_v4_2`: trace/semantic/prior dimensions + transformer depth/heads.
- `stwm_1b`: legacy config fields.

This is required for artifact auditability.

## Validation Evidence

### Smoke validation

Detached eval completed on V4.2 real checkpoint and produced summary with:

- `model_config.family = "stwm_v4_2"`

### Full completed-runs re-evaluation

All requested 8 artifacts (4 runs x best/latest) were generated under:

- `outputs/eval/stwm_v4_2_detached_protocol_eval_20260403/`

And every result reports:

- `model_config.family = "stwm_v4_2"`
- `num_clips = 9`

## Operational Constraints Preserved

- No lane interruption.
- No kill/restart of running training jobs.
- Detached eval ran as independent processes with retry-on-OOM handling.

## Remaining Limitations (Post-Patch)

- Compatibility is solved, but claim strength is still bounded by protocol coverage and sample size.
- Current detached v2_3 slice has weak occlusion-event signal (`occlusion_recovery_acc = 0` across all 8 artifacts), so reconnect superiority claims remain unsupported.
