# STWM Semantic-Only TUSB Unfreeze V1 Training

Two controlled runs were executed from the same V2 checkpoint. Stage1 stayed frozen, trace/dynamic TUSB params stayed frozen, and future candidates were never used.

| run | steps | trainable params | TUSB semantic params | TUSB dynamic params | proto CE start/end | train top5 | checkpoint |
|---|---:|---:|---:|---:|---:|---:|---|
| C=64 | 500 | 1575751 | 592513 | 0 | 4.1351 / 4.3301 | 0.0286 | `outputs/checkpoints/stage2_tusb_semantic_only_unfreeze_v1_c64_20260428/latest.pt` |
| C=128 | 500 | 1649543 | 592513 | 0 | 4.9517 / 5.1394 | 0.0386 | `outputs/checkpoints/stage2_tusb_semantic_only_unfreeze_v1_c128_20260428/latest.pt` |

Both runs completed stably with finite losses and valid outputs.
