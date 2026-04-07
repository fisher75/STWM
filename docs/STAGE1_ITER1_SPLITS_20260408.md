# Stage 1 Iteration-1 Splits (2026-04-08)

- generated_at_utc: 2026-04-07T17:56:06.838467+00:00
- source_minisplit_path: /home/chen034/workspace/data/_manifests/stage1_minisplits_20260408.json
- out_splits: /home/chen034/workspace/data/_manifests/stage1_iter1_splits_20260408.json
- seed: 20260408

## Policy

- Contract and base minisplit are read-only references.
- Iter1 split is an extension for Stage 1 trace-only iteration.
- No Stage 2 semantics, no video reconstruction, no WAN, no MotionCrafter VAE.

## Counts

- pointodyssey_iter1_train: 24
- pointodyssey_iter1_val: 6
- kubric_iter1_train: 72
- kubric_iter1_val: 8
- joint_iter1_train(pointodyssey branch): 24
- joint_iter1_train(kubric branch): 72
- tapvid eval_mini: 6
- tapvid3d eval_mini: 12
