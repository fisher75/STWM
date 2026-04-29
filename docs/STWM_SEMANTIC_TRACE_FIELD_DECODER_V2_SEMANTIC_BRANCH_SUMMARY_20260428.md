# STWM Semantic Trace Field Decoder V2 Semantic Branch Summary

Controlled semantic-branch unfreeze completed for 500 optimizer steps. Stage1 stayed frozen and the trace backbone remained non-trainable. The trainable scope was limited to `future_semantic_state_head`, `semantic_fusion.semantic_proj`, and `readout_head`.

## Trainability Boundary

- Trainable params total: `1057030`
- Stage1 trainable params: `0`
- Trace backbone trainable: `False`
- Boundary ok: `True`

## Training Metrics

- Train steps: `500`
- Loss finite ratio: `1.0`
- Output valid ratio: `1.0`
- Proto loss start/end: `4.968008995056152` / `5.189838409423828`
- Proto train accuracy/top5 mean: `0.0017833044292473118` / `0.03882855121121716`
- Visibility/reappearance/event losses: `0.3709388852715492` / `0.2724241378903389` / `0.21471342819929123`

## Interpretation

The run is stable and respects the no-drift boundary, but semantic prototype learning is not positive yet. Free-rollout proto top5 remains below the frequency baseline, so V2 does not establish a usable semantic trace field signal.
