# STWM Semantic-State Feedback V1 Eval

| method | event AP | event AUROC | per-horizon AP | per-horizon AUROC | coord error |
|---|---:|---:|---:|---:|---:|
| joint_v1_no_feedback | 0.803574 | 0.713056 | 0.160716 | 0.284099 | 0.209219 |
| feedback_v1_free_rollout | 0.809433 | 0.718889 | 0.164709 | 0.304070 | 0.209154 |

Feedback V1 gives a small positive feasibility signal without trace rollout regression. This remains Stage2 validation evidence, not a paper-level world-model claim.
