# STWM Semantic Trace Field Decoder V2 Head-Only Sanity

Head-only V2 used the large semantic prototype target cache and trained only `future_semantic_state_head` for 100 steps. Stage1 and the Stage2 semantic/trace trunks remained frozen.

## Result

- Training completed: `True`
- Loss finite ratio: `1.0`
- Output valid ratio: `1.0`
- Selected prototype count: `128`
- Train proto loss start/end: `4.968560695648193` / `5.261046409606934`
- Train proto top5 mean: `0.04945161389281051`
- Free-rollout proto accuracy/top5: `0.0` / `0.033854166666666664`
- Eval frequency baseline top1/top5: `0.042317708333333336` / `0.1796875`
- Top5 over frequency baseline: `-0.14583333333333334`
- Trace regression detected: `False`

## Interpretation

Head-only remains below the frequency baseline on the eval subset, so this run is not evidence of a learned semantic field. This does not falsify the world-model direction because the Stage2 semantic branch was intentionally frozen; it only confirms that the last head alone cannot recover a structured semantic prototype field from the existing hidden state. Controlled semantic-branch unfreeze is the next gated step.
