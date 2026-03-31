# STWM V4.2 Tokenizer And Memory

## Tokenizer Goal

Build object-biased learned state tokens from dense motion geometry and semantic priors, while using teacher signals only as weak guidance.

## Tokenizer Inputs and Outputs

## Inputs

1. trace features per frame
   - center `(x, y)`
   - velocity `(vx, vy)`
   - visibility
2. semantic priors per frame
   - class score vectors
   - pooled text/semantic embeddings
3. optional teacher priors per frame
   - mask area ratio
   - tracklet confidence / continuity hints
4. optional teacher objectness scalar per frame (weak prior)

## Outputs

1. learned state tokens (object-biased)
2. token-to-time assignment weights
3. frame-level objectness estimates
4. diagnostics
   - objectness entropy
   - token usage entropy
   - top-attended frame coverage

## Teacher/Prior Role Boundary

## During Training

- teacher priors enter as additive bias or regularization target
- tokenizer remains trainable and can deviate from teacher prior
- no hard assignment forcing token identity to equal teacher object id

## During Inference

- model should run without mandatory teacher pipeline
- optional prior channel can be provided, but fallback path must exist

Design principle:

- teacher helps bootstrap objectness
- model owns final token geometry/semantics

## Single Retrieval/Reconnect Memory

## Purpose

The memory has one role bundle:

1. reappearance/reconnect support
2. same-category disambiguation
3. query persistence across short occlusion gaps

## Minimal Mechanism

1. maintain compact key/value slots
2. retrieve with token-query attention
3. fuse retrieved context with current tokens via gated residual
4. update memory with momentum and bounded queue behavior

This is intentionally one module, not multiple specialized memories.

## Grain Control (Anti-Free-Form Drift)

To avoid fully unconstrained tokenization:

1. fixed token budget per clip window
2. objectness-biased attention over time
3. light entropy regularization on token assignments
4. weak coverage constraints to prevent all-token collapse

## Anti-Collapse Checklist

1. token entropy should not go near zero too early
2. token-to-time attention should not saturate to one background frame
3. memory gate should be active on reappearance-like segments
4. disabling semantics should reduce semantic head quality
5. disabling identity memory should degrade persistence-related probes
