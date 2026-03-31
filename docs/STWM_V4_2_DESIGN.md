# STWM V4.2 Design

## Context and Decision

The dense/foreground line reached evaluator V2.3 with fixed hard split and 3-seed checks, but did not produce stable cross-seed superiority for `full > wo_semantics` or `full > wo_identity_memory`. Identity and occlusion probes remain weak.

Decision:

- stop evaluator churn after V2.3
- do not jump to 1B on unstable evidence
- shift to a new model structure: STWM V4.2

## Why Current Dense/Foreground Is Not Enough

1. State representation remains too foreground-centric and insufficiently object-centric.
2. Semantics and identity signals are not consistently separable under multi-seed paired checks.
3. Memory effects are weak because representation and supervision do not force durable object state.

The bottleneck is no longer only metrics. It is state design.

## Why Not Strong Teacher-Guided

Teacher masks/tracklets are useful as priors but should not define final state tokens.

Risks of strong Teacher-Guided design:

1. state granularity is locked by external pipeline
2. weak transfer across domains with different teacher quality
3. reduced novelty for world-model core contribution
4. less control over task-specific state abstraction

## V4.2 Core Thesis

STWM V4.2:

Dense 4D Field + Object-Biased Learned Tokenizer + Factorized Dynamics + Single Retrieval Memory

Interpretation:

1. Keep dense trajectory field as geometric/motion substrate.
2. Learn object-biased state tokens instead of directly inheriting teacher objects.
3. Use a compact factorized dynamics stack:
   - motion head
   - semantic head
   - lightweight identity/memory head
4. Keep only one retrieval/reconnect memory module.

## Comparison Against Nearby Options

### Current Dense/Foreground (legacy)

- Pros: stable engineering pipeline, cheap to run.
- Cons: weak object-centric state and weak identity evidence.

### Strong Teacher-Guided Objects

- Pros: easy early gains, interpretable pseudo-targets.
- Cons: teacher lock-in, weaker generalization, lower methodological novelty.

### Pure Object Abstractor (free-form slots)

- Pros: high flexibility.
- Cons: high collapse risk at 220M stage, hard to stabilize quickly.

### V4.1-Style Intermediate (over-factorized)

- Pros: expressive decomposition.
- Cons: too many interacting heads/losses for current stage.

### V4.2 (chosen)

- Pros: balanced complexity, keeps dense prior while upgrading state abstraction.
- Cons: still requires careful anti-collapse controls and memory diagnostics.

## Text Module Diagram

```
Input video clip
  -> dense trace features (centers/velocity/visibility)
  -> semantic priors (class/text/objectness)
  -> weak teacher priors (optional masks/tracklets)

  -> Object-Biased Learned State Tokenizer (trainable)
       -> state tokens + token-level objectness diagnostics

  -> Single Retrieval/Reconnect Memory
       -> token enrichment for persistence/reappearance

  -> Factorized Dynamics
       -> motion head (trajectory/visibility)
       -> semantic head (state semantics)
       -> identity head (association/re-id cues)

  -> supervision bundle
       -> trajectory regression
       -> semantic alignment/classification
       -> re-id InfoNCE
       -> hard query grounding
       -> optional reconnect loss
```

## MVP Scope For This Iteration

1. deliver minimal runnable V4.2 modules and trainer
2. run short smoke for full / wo_semantics / wo_identity
3. report both success and failure patterns
4. decide whether 220M signal is hard enough to plan 1B pre-final
