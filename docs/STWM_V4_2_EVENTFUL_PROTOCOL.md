# STWM V4.2 Eventful Protocol

## Goal

Repair identity/reconnect evidence protocol without changing model, loss, or evaluator mainline.

This protocol only changes clip/query sampling and manifest construction.

## Inputs

- source manifest:
  - `manifests/minisplits/stwm_week2_minival_v2.json`
- clip diagnostics source:
  - `outputs/training/stwm_v4_2_minival_multiseed/seed_{42,123,456}/full_v4_2/train_log.jsonl`

## Mining Rules

Per clip, auto-select a target label and score eventfulness from mask dynamics (VSPW excludes label `255`):

- occlusion: `1 -> 0` visibility transition exists
- reappearance: `0 -> 1` transition after a missing span
- reconnect: reappearance with centroid distance under threshold
- visibility_flip: at least two visibility transitions
- identity_ambiguity: strong competitor label area ratio over threshold

Selection score combines:

- event counts (`reappearance`, `disappear`, `flip`, `reconnect`)
- missing-span length, motion, area variation
- clip difficulty from previous full-run diagnostics

## Outputs

- manifest:
  - `manifests/minisplits/stwm_v4_2_eventful_minival_v1.json`
- clip ids:
  - `manifests/minisplits/stwm_v4_2_eventful_clip_ids_v1.json`
- report:
  - `reports/stwm_v4_2_eventful_protocol_v1.json`

## Coverage Summary

From `stwm_v4_2_eventful_protocol_v1.json`:

- candidate_count: `36`
- selected_count: `18`
- coverage_insufficient: `False`

Event type counts on selected clips:

- occlusion: `18` (ratio `1.0000`)
- reappearance: `9` (ratio `0.5000`)
- reconnect: `8` (ratio `0.4444`)
- visibility_flip: `11` (ratio `0.6111`)
- identity_ambiguity: `16` (ratio `0.8889`)

## Protocol Boundary

- no new model module
- no loss composition change
- no evaluator mainline modification
- no 1B scaling
