# Storage Budget

## Current policy

- Keep all project artifacts under `/home/chen034/workspace/stwm`.
- Keep raw archives in `data/raw/`.
- Keep extracted data in `data/external/`.
- Avoid full dense feature dumps for all datasets.
- Save full raw model outputs only for smoke-test subsets.
- Prefer distilled object-centric caches in `data/cache/`.

## Initial budget estimate

| Item | Estimated size | Notes |
|---|---:|---|
| Code, repos, scripts, docs | 5-20 GB | Includes cloned third-party repos and local package |
| Checkpoints | 20-120 GB | Depends on backbone variants retained |
| VSPW raw + extracted | 60-120 GB | Raw archive plus extracted split |
| VISOR raw + extracted | 40-80 GB | Includes masks and metadata |
| VIPSeg raw + extracted | 80-200 GB | Size depends on retained resolution and archive layout |
| BURST/TAO raw + extracted | 300-800 GB | Largest storage risk in week 1 |
| Smoke-test outputs | 5-30 GB | Visualizations and mini-split caches only |
| Training checkpoints and logs | 100-1000+ GB | Controlled by retention policy |

## Main storage risks

- Keeping both raw archives and extracted trees for every dataset indefinitely.
- Writing dense TraceAnything outputs for full datasets.
- Saving too many training checkpoints or rollout visualizations.
- Downloading multiple duplicate backbone checkpoints across repos and local copies.

## Mitigations

- Stage full downloads by priority instead of all-at-once feature extraction.
- Keep mini splits for early debugging.
- Consolidate checkpoints in `models/checkpoints/`.
- Review raw-vs-extracted retention after checksum and smoke tests.
- Periodically run `scripts/check_storage.sh`.

## Current observed usage

As of 2026-03-30 12:38:56 +08:

| Path | Current size | Notes |
|---|---:|---|
| `models/checkpoints/` | ~11 GB | Core checkpoints already present |
| `third_party/` | ~246 MB | Primary code repositories only |
| `data/raw/vspw/` | ~43 GB | Raw archive retained |
| `data/raw/visor/` | ~29 GB | Raw archive retained |
| `data/raw/vipseg/` | ~15 GB | Raw archive retained |
| `data/raw/burst/` | ~226 GB | Includes annotations, TAO train/val archives, and active TAO test download |
| `data/external/vspw/` | ~45 GB | Extracted dataset tree |
| `data/external/visor/` | ~29 GB | Extracted dataset tree |
| `data/external/vipseg/` | ~15 GB | Extracted dataset tree |
| `data/external/burst/` | ~118 GB | Extracted annotations plus TAO train/val frames |
| `data/cache/` | ~768 MB | Mini split cache only, mostly sampled `VISOR` frame zips unpacked on demand |
| `outputs/` | ~28 KB | Smoke test JSONs only |

## Short-term note

- `aria2` preallocates large zip files, so `TAO test` can look complete in `ls -lh` even while the download is still running.
- Current working footprint is already roughly `~531 GB` across raw, extracted, cache, checkpoints, repos, and outputs.
- The storage policy is still working as intended because the only non-raw cache materialized so far is the sampled mini split cache, not a full dense feature dump.
