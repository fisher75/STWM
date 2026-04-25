# STWM Counterfactual Association Eval 20260425

| variant | top1 | MRR | false_confuser | decision_flip_rate |
|---|---:|---:|---:|---:|
| full_trace_belief | 0.5006 | 0.6573 | 0.4994 | n/a |
| no_trace_prior | 0.5388 | 0.6824 | 0.4612 | 0.1279 |
| shuffled_trace | 0.0756 | 0.0756 | 0.9244 | 0.9221 |

- removing_trace_increases_false_confuser_rate = `False`
- shuffled_trace_changes_decision = `True`
- semantic_swap_fixed_trace = skipped: Raw semantic evidence/candidate appearance token maps are not present in per-item reports; cannot replace semantic evidence while preserving trace belief without fabricating scorer inputs.
