# TraceWM Stage 1 Iteration-1 Protocol (2026-04-08)

## Scope

- Stage 1 single mainline only: trace-only future trace/state generation.
- Data contract is frozen at: /home/chen034/workspace/data/_manifests/stage1_data_contract_20260408.json
- Minisplit baseline is frozen at: /home/chen034/workspace/data/_manifests/stage1_minisplits_20260408.json

## Prohibited In This Round

- Stage 2 semantics.
- Video reconstruction.
- WAN.
- MotionCrafter VAE.
- DynamicReplica.

## Fixed Stage 1 Data Roles

- PointOdyssey = main-train.
- Kubric (movi_e-only) = main-train.
- TAP-Vid = main-eval.
- TAPVid-3D = limited-eval only.

## Iteration-1 Experiment Matrix (Only)

- pointodyssey_only
- kubric_only
- joint_po_kubric

## Uniform Training/Eval Rules

- Same seed for all three runs.
- Same model size for all three runs.
- Same loss family for all three runs.
- Same eval protocol for all three runs.
- Teacher-forced and free rollout are both mandatory.
- No semantics branch and no decoder/render loss.
