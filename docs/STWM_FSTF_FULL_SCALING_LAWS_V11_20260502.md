# STWM-FSTF Full Scaling Laws V11

- audit_name: `stwm_fstf_full_scaling_laws_v11`
- generated_at_utc: `2026-05-01T17:55:45.271502+00:00`
- expected_eval_count: `21`
- new_checkpoint_count: `12`
- new_eval_summary_count: `12`
- new_train_summary_count: `12`
- gpu_jobs_launched: `24`
- prototype_scaling_positive: `True`
- horizon_scaling_positive: `False`
- trace_density_scaling_positive: `False`
- model_size_scaling_positive: `False`
- dense_trace_field_claim_allowed: `False`
- long_horizon_claim_allowed: `False`
- future_hidden_load_bearing_retained_under_scaling: `True`
- strongest_failure_case: `H/K scaling cache missing`
- reviewer_risk_after_scaling: `high: V11 has live prototype/model-size artifacts, but full scaling remains incomplete`

## Missing Scaling Points
- H=16: H16 future feature/prototype target cache not materialized in V11 yet
- H=24: H24 future feature/prototype target cache not materialized in V11 yet
- K=16: K16 trace-unit materialization cache not materialized in V11 yet
- K=32: K32 trace-unit materialization cache not materialized in V11 yet
