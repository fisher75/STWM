# STWM V4.2 Protocol-Level Evaluation Capability Audit

Date: 2026-04-03  
Scope: completed real-matrix runs (220m/1b, full vs wo_semantics)  
Policy: read-only against active training; no interruption of running lanes

## Executive Verdict

1. Standard MOT metrics (HOTA/MOTA/IDF1) are not currently executable in this repository.
2. A protocol-style evaluator exists (`code/stwm/evaluators/eval_mini_val.py`) and exposes identity/occlusion/query metrics, but it is not checkpoint-compatible with current STWM V4.2 real-run checkpoints.
3. The strongest executable fallback today is the real trainer's eval-path metric family:
   - `trajectory_l1`
   - `query_localization_error`
   - `query_traj_gap`
   - `reconnect_success_rate`
   - `reappearance_event_ratio`
4. This fallback supports comparative claims on motion/query/reconnect proxies, but not full MOT tracking claims.

## Evidence Log

### A. No HOTA/MOTA/IDF1 capability found

- Search command (code/scripts/docs): `grep -R -n -E "HOTA|MOTA|IDF1|TrackEval" ...`
- Result: no hits.

### B. Detached mini-val evaluator exists but is architecture-misaligned

- Evaluator file: `code/stwm/evaluators/eval_mini_val.py`
- It imports legacy model API:
  - `from stwm.models.stwm_1b import STWM1B, STWMConfig, load_model_config`
- Real trainer/checkpoints use V4.2 model API:
  - `code/stwm/trainers/train_stwm_v4_2_real.py` loads `STWMV42`
  - `code/stwm/models/stwm_v4_2.py` defines `STWMV42Config` with fields such as `trace_dim`

Reproduced failure (detached evaluator against real checkpoint):

```
TypeError: STWMConfig.__init__() got an unexpected keyword argument 'trace_dim'
```

This confirms direct detached evaluation is currently blocked for real checkpoints.

### C. Two evaluator surfaces are placeholders only

- `code/stwm/evaluators/eval_query_forecast.py` returns `status: "placeholder"`
- `code/stwm/evaluators/eval_future_mask.py` returns `status: "placeholder"`

So these cannot support publication-grade protocol claims yet.

### D. Real trainer eval-only path has a resumable-step caveat

- In `code/stwm/trainers/train_stwm_v4_2_real.py`, if `start_step >= total_steps`, loop is skipped and summary is written with zeroed aggregates.
- Reproduced behavior in offline test:
  - message: `resume step 5000 already reaches target steps 60; writing summary only`

Operational implication:

- For eval-only over resumed checkpoints, `--steps` must be set above checkpoint step to execute any looped evaluation rows.

## Capability Matrix

| Capability | Status | Evidence |
|---|---|---|
| HOTA/MOTA/IDF1 | Not available | No code/script/doc hits |
| Detached identity-consistency protocol eval on real checkpoints | Blocked | `STWMConfig` vs `trace_dim` mismatch |
| Query forecast evaluator | Placeholder | `status: "placeholder"` |
| Future mask evaluator | Placeholder | `status: "placeholder"` |
| Executable protocol-like proxy metrics on real checkpoints | Available | `train_stwm_v4_2_real.py` summary/log fields |

## Strongest Executable Fallback (Current)

Use the real trainer's evaluation metric family as protocol proxies on shared manifests/rules:

- Motion continuity proxy: `trajectory_l1`
- Query grounding proxy: `query_localization_error`, `query_traj_gap`
- Reconnect proxy: `reconnect_success_rate`, `reappearance_event_ratio`

These are executable and comparable across completed runs under shared manifest and budget settings.

## Four Hard Questions (Explicit Answers)

1. Can this repo currently produce HOTA/MOTA/IDF1 for STWM V4.2 real checkpoints?  
   Answer: No.

2. Can current detached protocol evaluator directly score real checkpoints for identity-consistency style metrics?  
   Answer: Not currently; it fails due model/config incompatibility.

3. What is the strongest immediately executable alternative without interrupting training?  
   Answer: Real-trainer eval metric family (`trajectory_l1`, query localization/gap, reconnect/reappearance rates).

4. Can we make publication-grade identity-tracking claims now?  
   Answer: Not with current tooling; only bounded proxy claims are defensible until evaluator-model compatibility and standard MOT stack are added.
