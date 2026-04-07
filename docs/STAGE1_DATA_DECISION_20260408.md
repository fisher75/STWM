# Stage 1 Data Decision (2026-04-08)

## Scope Lock

- Project root: /home/chen034/workspace/stwm
- Data root: /home/chen034/workspace/data
- This round is Stage 1 Trace-only future trace/state generation only.
- No Stage 2 semantics work.
- No legacy STWM v4.2 object-bias/QSTR/QTSA continuation.
- No new data download or data repair in this round.

## Stage 1 Dataset Roles

1. Main training datasets:
- PointOdyssey
- Kubric

2. Main evaluation dataset:
- TAP-Vid

3. Limited evaluation dataset:
- TAPVid-3D

4. Non-blocking optional enhancement (not first-wave Stage 1):
- DynamicReplica (dynamic_replica_data + dynamic_stereo integration path)

## Startup Policy

- Current startup policy is GO_WITH_LIMITATIONS.
- Allowed to launch Stage 1 with PointOdyssey + Kubric + TAP-Vid.
- TAPVid-3D is limited eval only (not full benchmark gate).
- DynamicReplica is explicitly non-blocking in first-wave Stage 1.

## Operational Rule For This Round

- Only read-only confirmation of data state.
- Focus on interface and protocol implementation:
- data contract
- dataset adapters
- smoke tests
- tiny-train
- tiny-eval
- Do not perform dataset acquisition/repair actions in this round.
