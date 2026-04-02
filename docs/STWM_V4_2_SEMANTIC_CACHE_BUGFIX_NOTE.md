# STWM V4.2 Semantic Cache Bugfix Note

## Incident Summary

Observed failure in real 1B seed42 full run:

- failure site: semantic cache load path
- runtime error: `zip archive (did you mean to use torch.jit.load()?)`
- failing cache example:
  - `data/cache/semantic_summaries/1417_hCGuWCExVqM_80134f42fe2c.pt`

This failure is not CUDA OOM and not dataloader-worker termination.

## Root Cause

Semantic cache files are loaded through `torch.load` in `code/stwm/modules/semantic_adapter.py`.
Some cache artifacts can become load-incompatible at runtime (zip/jit/weights-only/pickle class of errors),
which previously propagated directly and terminated training.

## Fix Scope

Updated semantic cache behavior in `code/stwm/modules/semantic_adapter.py`:

1. Cache load now classifies recoverable compatibility/corruption errors.
2. On recoverable cache error:
   - emit warning with reason and file path
   - quarantine bad cache file under cache-root `quarantine/`
   - regenerate semantic summary from source inputs
   - atomically write rebuilt cache back to canonical path
3. Cache writes now use atomic replace (`temp -> os.replace`) to reduce partial-write risk.
4. Payload structure validation added for cache reads.

## Operational Behavior After Fix

- Good cache: direct cache hit.
- Bad cache (recoverable): auto self-heal in-place workflow (quarantine + rebuild + atomic rewrite).
- Non-recoverable errors: still fail fast to avoid masking unrelated code bugs.

## Healthcheck Tool

Added tool:

- `code/stwm/tools/check_stwm_v4_2_semantic_cache_health.py`

Purpose:

- scan semantic cache subset driven by current train manifest
- count checked/bad/rebuilt/unhandled
- optional repair mode for bad cache auto-rebuild

Default report path:

- `reports/stwm_v4_2_semantic_cache_healthcheck.json`

Example:

```bash
cd /home/chen034/workspace/stwm
PYTHONPATH=/home/chen034/workspace/stwm/code \
/home/chen034/miniconda3/bin/conda run --no-capture-output -n stwm \
python code/stwm/tools/check_stwm_v4_2_semantic_cache_health.py --repair
```

## Checkpoint Policy Sync (Future Runs)

Real run defaults are now aligned to safer recovery cadence:

- latest checkpoint interval: every 100 optimizer steps
- best checkpoint: updated per existing policy
- milestone checkpoints: disabled by default (`milestone_interval=0`)

Affected launcher defaults:

- `scripts/run_stwm_v4_2_real_train_seed.sh`

Trainer now logs retention policy and interval values at startup:

- `code/stwm/trainers/train_stwm_v4_2_real.py`
