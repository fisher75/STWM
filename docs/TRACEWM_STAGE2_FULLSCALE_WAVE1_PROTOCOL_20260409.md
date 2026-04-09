# TRACEWM Stage2 Fullscale Wave1 Protocol

- Stage1 remains frozen and untouched in this round.
- Stage2 current mainline remains `stage2_core_cropenc` with `crop_visual_encoder` as the mainline semantic source.
- Core train/eval binding remains `VSPW + VIPSeg`.
- BURST is allowed only for the explicit `coreplusburst` control run.
- TAO and VISOR remain excluded from this wave.

## Preserved Mainline Hyperparameters
- obs_len: 8
- fut_len: 8
- max_tokens: 64
- semantic_crop_size: 64
- semantic_hidden_dim: 256
- semantic_embed_dim: 256

## Runtime Preflight
- candidate_batch_sizes: [2, 4, 8]
- selected_batch_size: 8
- whether_full_train_used: True
- whether_full_val_used: True

## Wave1 Run Set
- stage2_fullscale_core_cropenc_seed42_20260409
- stage2_fullscale_core_cropenc_seed123_20260409
- stage2_fullscale_core_cropenc_seed456_20260409
- stage2_fullscale_core_legacysem_seed42_20260409
- stage2_fullscale_coreplusburst_cropenc_seed42_20260409

## Training Budget
- train_steps: 10000
- eval_interval: 1000
- save_every_n_steps: 1000
- eval_max_batches: -1 (full validation)

## Scheduling
- single-card per run
- selector + shared lease path reused
- 5 runs launched in parallel, leaving headroom on the 8xB200 node
