# STWM Medium Semantic-State V1 Eval 20260427

- evaluation_item_limit: `128`
- current_export_data_source: `Stage2SemanticDataset validation split from checkpoint args; not external_389_item_manifest`
- medium free-rollout valid_output_ratio: `1.0`
- medium free-rollout output_degenerate: `False`
- medium free-rollout future_trace_coord_error: `0.20752167323371395`
- pre-medium free-rollout future_trace_coord_error: `0.2079043552512303`
- future_trace_coord_error_delta: `-0.0003826820175163448`
- uncertainty_error_correlation_delta: `-0.008710963392846244`
- trace_rollout_regression_detected: `False`
- visibility_metric_status: `smoke_only_simplified_target`

Medium checkpoint remains non-degenerate and slightly improves free-rollout coord error, but uncertainty/semantic variance signal does not clearly improve over pre-medium baseline.
