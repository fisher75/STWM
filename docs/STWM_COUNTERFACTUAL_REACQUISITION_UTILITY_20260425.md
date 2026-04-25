# STWM Counterfactual Reacquisition Utility 20260425

| variant | top1 | MRR | false_confuser | false_reacquisition |
|---|---:|---:|---:|---:|
| full_trace_belief | 0.5188 | 0.6849 | 0.4813 | 0.4813 |
| no_trace_prior | 0.5771 | 0.7217 | 0.4229 | 0.4229 |
| shuffled_trace | 0.0882 | 0.0882 | 0.9118 | 0.9118 |

- trace_intervention_degrades_reacquisition = `False`
- shuffled_trace_degrades_reacquisition = `True`
- trace_prior_load_bearing_counterfactual = `True`
