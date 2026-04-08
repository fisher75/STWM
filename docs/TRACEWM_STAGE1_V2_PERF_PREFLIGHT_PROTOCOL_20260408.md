# TRACEWM Stage1-v2 Perf Preflight Protocol (2026-04-08)

## Frozen Facts
1. Current Stage1-v2 scientific selection is still debug_small as selected_backbone, not prototype_220m.
2. Current Stage1-v2 trainer is single-GPU logic only and is not DDP or any multi-GPU trainer.

## This Round Objective
This round does not change scientific logic or model structure.
The objective is:
1. Prove prototype_220m preset can run in a real short-window training dry run.
2. Select the best single GPU on a shared 8xB200 cluster.
3. Attribute bottlenecks across CPU dataloader, IO, H2D, and GPU compute.

## Scope Boundaries
This round does not modify:
1. Stage2.
2. WAN or MotionCrafter VAE.
3. Video reconstruction logic.
4. New datasets.
5. Model architecture and loss definition.

## Entry Gate
1. Run Stage1-v2 220M preflight first.
2. If preflight fails, stop formal perf round immediately.
3. If preflight passes, proceed to single-GPU hardening and profiling.
