# V34.9 trace-preserving semantic measurement bank 中文报告

- 中文结论: `V34.9 measurement bank 已用真实 V30 external-GT trace sidecar 重建 obs_points/obs_vis/obs_conf；teacher feature 仍只作为 observed measurement 和 future supervision。`
- trace_preserving_measurement_bank_built: `True`
- output_root: `outputs/cache/stwm_ostf_v34_9_trace_preserving_semantic_measurement_bank/pointodyssey`
- measurement_coverage_by_split: `{'train': 0.8826828002929688, 'val': 0.8714583333333333, 'test': 0.8464658838757396}`
- trace_coverage_by_split: `{'train': 0.8826828002929688, 'val': 0.8714583333333333, 'test': 0.8464658838757396}`
- obs_points_zero_ratio_by_split: `{'train': 7.62939453125e-06, 'val': 6.510416666666667e-06, 'test': 2.8892381656804733e-06}`
- obs_vis_source: `V30 external-GT trace obs_vis via V33.8 semantic_identity source_npz`
- obs_conf_source: `V30 external-GT trace obs_conf via V33.8 semantic_identity source_npz`
- trace_state_contract_passed: `True`
- teacher_agreement_stats: `{'train': {'mean': 0.885353811797278, 'p10': 0.7113430500030518, 'p90': 0.9905799508094788}, 'val': {'mean': 0.8765717261577056, 'p10': 0.7083536624908447, 'p90': 0.9850512742996216}, 'test': {'mean': 0.8766271385666342, 'p10': 0.7110483348369598, 'p90': 0.9844360649585724}}`
- measurement_confidence_stats: `{'train': {'mean': 1.0, 'p10': 1.0, 'p90': 1.0}, 'val': {'mean': 1.0, 'p10': 1.0, 'p90': 1.0}, 'test': {'mean': 1.0, 'p10': 1.0, 'p90': 1.0}}`
- leakage_safe: `True`
- exact_blockers: `{}`
