# STWM Reacquisition Utility Decision 20260425

| question | answer |
|---|---|
| reacquisition utility established | `True` |
| improved vs frozen external teacher | `True` |
| improved vs legacysem | `True` |
| supports main paper utility claim | `True` |
| claim level | `strong_claim` |
| next step | `add_to_main_paper` |

## Limitations

- No raw candidate score maps are present, so utility uses rank-derived target_rank/MRR/top5 and top1_candidate_id rather than a newly trained scorer.
- No numeric gap length or occlusion severity field is present; breakdowns use occlusion/long-gap subset-tag proxies only.
- No matching per-item reacquisition rows in stwm_trace_belief_eval_20260424.json for this method/scoring_mode.
