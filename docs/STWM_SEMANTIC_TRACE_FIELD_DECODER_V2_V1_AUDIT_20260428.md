# STWM Semantic Trace Field Decoder V2 V1 Audit (20260428)

## V1 facts
- Target item_count: 6
- Valid feature count: 261
- Prototype count: 64
- Prototype samples per prototype: mean 4.078125, min 1, max 14
- Target valid ratio: 0.6796875
- Training scope: future_semantic_state_head_only (Stage1 frozen, trace backbone frozen)
- Head-only proto loss: start 4.102555274963379 -> end 4.026352882385254
- Free-rollout proto accuracy/top5/masked CE: 0.0 / 0.0 / 4.320476531982422
- Frequency baseline top1/top5: 0.125 / 0.546875

## Why V1 does not prove world model failure
- Target cache has only 6 items, so prototype statistics and evaluation are too unstable.
- Head-only training froze Stage2 semantic backbone, so semantic representations could not adapt.
- Free-rollout eval used only 2 valid items, so zero top5 is not decisive.
- Mean samples per prototype is 4.08, which is too sparse for a 64-way vocabulary.
- This was a 100-step smoke, not a controlled semantic-branch training test.

## Why the next step is large target cache + controlled semantic-branch unfreeze
- Larger cache stabilizes prototype counts and increases sample-per-prototype coverage.
- Controlled unfreeze lets the semantic branch learn the prototype field without touching Stage1.
- Head-only zero top5 suggests representation mismatch, not necessarily objective failure.
- Bigger cache supports meaningful frequency baselines and free-rollout gap checks.
- A semantic-branch test is required before claiming world-model failure or success.

## Source reports
- reports/stwm_future_semantic_trace_feature_targets_v1_20260428.json
- reports/stwm_semantic_trace_prototypes_v1_20260428.json
- reports/stwm_future_semantic_trace_prototype_targets_v1_20260428.json
- reports/stwm_semantic_trace_field_decoder_v1_headonly_summary_20260428.json
- reports/stwm_semantic_trace_field_decoder_v1_headonly_summary_raw_20260428.json
- reports/stwm_semantic_trace_field_decoder_v1_eval_20260428.json
