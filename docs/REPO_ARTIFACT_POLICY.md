# Repository Artifact Policy

This repository follows a conservative research tracking policy:

1. Track source code, scripts, docs, manifests, and environment manifests.
2. Do not track large runtime artifacts (datasets, checkpoints, caches, logs, scratch files).
3. For `outputs/`, use default-ignore with an explicit keep-list for lightweight summaries and curated paper assets.
4. For `third_party/`, do not ignore the whole directory. Keep vendored source changes trackable, but ignore internal caches/checkpoints/runs.

## Tracked by design

- `code/`, `scripts/`, `docs/`, `manifests/`
- `env/stwm.env`, `env/stwm.yml`
- `reports/*.json`, `reports/*.md`
- Curated figure set under `outputs/visualizations/stwm_v4_2_final_paper_figures/`
- Lightweight output summaries listed in `.gitignore` keep rules

## Ignored by design

- `data/cache`, `data/external`, `data/raw`, `data/processed`
- `models/checkpoints`, model caches
- `logs/`, `tmp/`
- Most of `outputs/` except explicit keep-list
- `third_party` internal heavy artifacts (checkpoints, weights, runs, caches)

## If you need to keep a new artifact

1. Prefer storing a compact summary in `reports/`.
2. If an output must be versioned, add a narrow keep rule in `.gitignore`.
3. Avoid broad exceptions that re-include large directories.
