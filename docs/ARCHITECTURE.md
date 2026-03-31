# STWM Architecture

## Project goal

Represent a video clip as a compact semantic trajectory state, then forecast future object motion, visibility, regions, and queryable semantics.

## Module layout

`TraceAnything` -> `trace_adapter.py` -> geometric trajectory summaries

`OV2VSS / YOLO-World / SAM2` -> `semantic_adapter.py` -> semantic object proposals and labels

`trace summaries + semantic summaries` -> `tokenizer.py` -> semantic trajectory tokens

`stwm_1b.py` -> causal latent dynamics model over object-centric tokens

`train_stwm.py` -> training loop shell

`eval_future_mask.py` / `eval_query_forecast.py` -> protocol shells

`smoke_test_one_clip.py` -> first end-to-end integration target

## Data flow

1. Read one clip and metadata through `STWMDataset`.
2. Extract or load trace features.
3. Extract or load semantic proposals.
4. Fuse them into object-centric tokens.
5. Run a forward pass through the STWM model.
6. Save a lightweight rollout summary and metadata for inspection.

## Week-1 dataset interface

- Canonical mini split manifests now live in `manifests/minisplits/`.
- The combined week-1 debug set is `manifests/minisplits/stwm_week1_mini.json`.
- Each sample stores:
  - `clip_id`
  - `frame_paths`
  - `text_labels`
  - `metadata` with dataset-specific pointers such as mask paths, archive paths, cache dirs, and split names
- `VISOR` uses sampled sequence extraction into `data/cache/visor_sequences/` instead of expanding every sequence archive.

## Caching policy

- Level 1: raw model outputs for smoke-test subsets only
- Level 2: distilled object-centric tokens and lightweight summaries for training

## First training plan

- Week 1: data interfaces, downloads, baseline smoke tests
- Week 2: small prototype on mini split
- Week 3: 1B training stack and protocol A
- Week 4: open-vocab transfer, occlusion persistence, visualization
