# Stage 1 Minisplit Protocol (2026-04-08)

- generated_at_utc: 2026-04-07T17:09:59.996454+00:00
- seed: 20260408
- contract_path: /home/chen034/workspace/data/_manifests/stage1_data_contract_20260408.json

## Minisplit Design

- All selections are deterministic under fixed seed.
- PointOdyssey and Kubric are selected for train_mini/val_mini and must each form independent mini-batches.
- TAP-Vid is eval_mini only.
- TAPVid-3D is limited eval_mini only.

## Coverage Heuristics

- Long trajectory preference: keep long-frame clips/sequences in each mini pool.
- Visibility/motion diversity: combine top-length picks and seeded random picks.
- Source balance for TAPVid-3D: include pstudio/adt/drivetrack.

## Sizes

- PointOdyssey train_mini: 12
- PointOdyssey val_mini: 6
- Kubric train_mini: 24
- Kubric val_mini: 8
- TAP-Vid eval_mini: 6
- TAPVid-3D eval_mini: 12
