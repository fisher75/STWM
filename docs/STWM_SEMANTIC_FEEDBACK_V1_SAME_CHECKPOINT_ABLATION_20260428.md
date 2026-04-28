# STWM Semantic Feedback V1 Same-Checkpoint Ablation

- checkpoint: outputs/checkpoints/stage2_tusb_v3p1_semantic_state_feedback_v1_20260427/latest.pt
- max_items: 256
- mode: full_model_free_rollout / readout_only

| variant | alpha | event AP | event AUROC | per-horizon AP | per-horizon AUROC | coord error | gate mean | delta norm |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| disabled_same_ckpt | 0.05 | 0.809432575100 | 0.718888888889 | 0.164708541145 | 0.304070426218 | 0.209153850912 | 0.000000000000 | 0.000000000000 |
| zero_delta | 0.0 | 0.809432575100 | 0.718888888889 | 0.164708541145 | 0.304070426218 | 0.209153850912 | 0.017987697574 | 0.005309318647 |
| alpha0025 | 0.025 | 0.809432575100 | 0.718888888889 | 0.164708541145 | 0.304070426218 | 0.209153850912 | 0.017987697574 | 0.005309318647 |
| enabled | 0.05 | 0.809432575100 | 0.718888888889 | 0.164708541145 | 0.304070426218 | 0.209153850912 | 0.017987697574 | 0.005309318647 |
| alpha010 | 0.1 | 0.809432575100 | 0.718888888889 | 0.164708541145 | 0.304070426218 | 0.209153850912 | 0.017987697574 | 0.005309318647 |
| alpha020 | 0.2 | 0.809432575100 | 0.718888888889 | 0.164708541145 | 0.304070426218 | 0.209153850912 | 0.017987697574 | 0.005309318647 |

## Deltas
- enabled_minus_disabled_event_AP: 0.0
- enabled_minus_zero_delta_event_AP: 0.0
- enabled_minus_disabled_per_horizon_AP: 0.0
- alpha_sensitivity_observed: false

Conclusion: the adapter residual itself has no measurable effect in this readout_only checkpoint. The previous V1 gain should be attributed to the jointly trained semantic-state head/projection/readout rather than the feedback residual.
