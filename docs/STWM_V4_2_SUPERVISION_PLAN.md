# STWM V4.2 Supervision Plan

## Objective

Use the smallest supervision set that can validate the V4.2 structure at 220M scale.

## Required Losses

1. trajectory regression loss
   - target: trace centers over time
   - placement: motion head output
   - form: SmoothL1 on predicted vs target centers

2. semantic alignment/classification loss
   - target: semantic prior distributions or class ids
   - placement: semantic head on state tokens
   - form: cross-entropy or KL-style alignment

3. association/re-id InfoNCE loss
   - target: persistent identity consistency signal
   - placement: identity head embeddings
   - form: contrastive objective across positive (same state track) and negatives

4. hard query grounding loss
   - target: query-relevant frame/token alignment
   - placement: tokenizer query assignment or query head
   - form: negative log-likelihood on hard query target index

## Optional Loss

5. reconnect loss (optional)
   - target: reappearance reconstruction quality
   - placement: memory-fused branch
   - form: low-weight consistency term after visibility drops

## Why This Set Is Minimal But Sufficient

1. trajectory loss tests motion substrate quality
2. semantic loss tests object-level semantic state usefulness
3. InfoNCE tests identity persistence in latent space
4. grounding loss tests query sensitivity without adding a new benchmark

Together they cover motion/semantics/identity/query without over-parameterizing the objective stack.

## Why Not Add More Losses Now

1. too many losses obscure failure attribution
2. 220M stage needs structural clarity, not objective complexity
3. unstable weighting would delay model-level decisions

## Suggested Initial Weights (Smoke)

- `L_total = L_traj + 0.5 * L_sem + 0.25 * L_reid + 0.25 * L_query + 0.1 * L_reconnect(optional)`

These are startup defaults, not final tuned values.

## Placement Map

1. motion head -> `L_traj`
2. semantic head -> `L_sem`
3. identity head -> `L_reid`
4. tokenizer/query head -> `L_query`
5. memory-fused reconnect path -> `L_reconnect`
