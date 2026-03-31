# STWM V4.2 To 1B Roadmap

## Why 220M First

1. current uncertainty is structural, not purely capacity-limited
2. 220M is sufficient to test whether V4.2 state design hardens semantics/query/identity signals
3. premature 1B training would amplify unresolved representation noise

## 220M Evidence Needed Before 1B

The following signals should appear before scaling:

1. multi-seed direction improves for semantics ablation (`full_v4_2 > wo_semantics_v4_2`) more consistently than legacy line
2. identity ablation shows clearer degradation direction (`full_v4_2 > wo_identity_v4_2`) with reduced instability
3. tokenizer diagnostics do not show collapse/background domination
4. memory branch shows measurable contribution, not near-zero usage
5. failure panels reveal meaningful reconnect/query differences, not random noise

## Recommended Staged Path

## Stage 0 (this round)

- deliver minimal V4.2 runnable stack
- smoke runs only
- produce risk assessment and structural verdict

## Stage 1 (220M short)

- extend training horizon moderately
- keep fixed split and same ablation trio
- verify trend robustness under additional seeds if needed

## Stage 2 (220M pre-final)

- tune only high-impact knobs (token budget, memory slots, loss weights)
- freeze architecture choices
- lock paper-facing evaluation recipe

## Stage 3 (1B pre-final, conditional)

Only if Stage 2 is positive.

- scale hidden size/depth while preserving V4.2 topology
- keep tokenizer/memory design fixed to avoid confounds
- run limited but representative ablation and stability checks

## Minimal 1B Configuration Suggestion

1. keep V4.2 module graph unchanged
2. increase backbone width/depth and context length conservatively
3. keep single retrieval memory (no memory proliferation)
4. keep loss set unchanged; only retune weights lightly

## Budget and Risk Notes

1. biggest 1B risk is not compute alone; it is signal dilution if state design is still weak
2. do not spend 1B budget before confirming tokenization and memory are truly used
3. if 220M still shows weak identity/occlusion signal, prioritize story narrowing over scaling
