# STWM V4.2 Split Policy V2 (Phase B)

Date: 2026-04-03
Policy version: protocol_v2
Status: FROZEN

## Builder and Outputs

- Builder script: scripts/build_stwm_v4_2_protocol_v2_splits.py
- Output directory: manifests/protocol_v2
- Output manifests:
  - train_v2.json
  - protocol_val_main_v1.json
  - protocol_val_eventful_v1.json
  - internal_final_test_v1.json
  - protocol_v2_split_audit.json

## Official Role Definition

1. train_v2
   - training only
2. protocol_val_main_v1
   - only official model selection split
3. protocol_val_eventful_v1
   - diagnostics only (not used to pick official best)
4. internal_final_test_v1
   - locked final reporting split (no model selection)

## Source Policy and Internal Final Rule

- Primary source files are official train/val/test lists from VSPW and VIPSeg.
- In this workspace, official test split does not provide usable mask-paired clips for detached metric computation.
- Therefore internal_final_test_v1 is deterministically partitioned from official val clips at video level.
- Deterministic rule: hash-bucket by dataset:clip_id with target ratio about 70% val-main and 30% internal-final.

Audit field confirms:

- final_test_source = internal_from_official_val
- official_test_available_in_manifest = false

## Current Audit Summary

From manifests/protocol_v2/protocol_v2_split_audit.json:

- train_v2:
  - clip_count 3814
  - dataset ratio: vspw 0.7347, vipseg 0.2653
- protocol_val_main_v1:
  - clip_count 393
  - dataset ratio: vspw 0.6336, vipseg 0.3664
- protocol_val_eventful_v1:
  - clip_count 133
  - dataset ratio: vspw 0.6015, vipseg 0.3985
- internal_final_test_v1:
  - clip_count 140
  - dataset ratio: vspw 0.6643, vipseg 0.3357

Disjointness checks:

- train_vs_val_overlap = 0
- train_vs_test_overlap = 0
- val_vs_test_overlap = 0

## Usage Boundary (Mandatory)

- Official best checkpoint updates are only allowed on protocol_val_main_v1.
- internal_final_test_v1 is never used for checkpoint selection.
- protocol_val_eventful_v1 is diagnostics-only and can support qualitative or risk analysis, but not official best selection.
