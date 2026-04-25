# STWM Reacquisition V2 Eval 20260425

| variant | valid rows | top1 | top5 | MRR | false confuser | false reacquisition |
|---|---:|---:|---:|---:|---:|---:|
| full_trace_belief | 1038 | 0.5260 | 0.9403 | 0.6974 | 0.4740 | 0.4740 |
| frozen_external_teacher_only | 1038 | 0.5202 | 0.9306 | 0.6954 | 0.4798 | 0.4798 |
| legacysem | 1038 | 0.1840 | 0.7977 | 0.4422 | 0.8160 | 0.8160 |
| calibration-only | 1038 | 0.1638 | 0.7977 | 0.4313 | 0.8362 | 0.8362 |
| cropenc | 1038 | 0.1696 | 0.7987 | 0.4325 | 0.8304 | 0.8304 |
| belief_without_trace_prior | 1038 | 0.5983 | 0.9480 | 0.7437 | 0.4017 | 0.4017 |
| belief_with_shuffled_trace | 1038 | 0.1098 | 0.1098 | 0.1098 | 0.8902 | 0.8902 |
| belief_with_shuffled_gallery | 1038 | 0.1281 | 0.1281 | 0.1281 | 0.8719 | 0.8719 |

per_item_results_hash = `747d5b03a21502f1b3506c7306334708c7513282c9719ee088700bd604ea1b10`
