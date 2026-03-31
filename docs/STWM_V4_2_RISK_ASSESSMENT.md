# STWM V4.2 Risk Assessment

## Scope

This assessment uses the completed smoke trio:

- `outputs/training/stwm_v4_2_smoke/full_v4_2/smoke_summary.json`
- `outputs/training/stwm_v4_2_smoke/wo_semantics_v4_2/smoke_summary.json`
- `outputs/training/stwm_v4_2_smoke/wo_identity_v4_2/smoke_summary.json`

Model scale is approximately 220M class:

- `model_parameters = 207,543,382`

## High-Level Signal Snapshot

1. Trajectory loss direction is promising:
   - full trajectory average: `0.0109`
   - wo_semantics trajectory average: `0.0652`
   - wo_identity trajectory average: `0.0334`
2. Tokenizer did not collapse to degenerate single-token behavior by the current entropy proxy.
3. Identity training signal is still unstable in this short run.

## Required Risk Checks

## 1) Tokenizer Collapse Risk

Observed:

- assignment entropy around `0.99`
- token usage entropy around `0.99`

Interpretation:

- no hard collapse to one token
- but entropy is near-uniform, indicating weak specialization pressure

Risk level: medium.

## 2) Background Dominance Risk

Observed:

- current proxy `bg_fg_attention_ratio` is near `0.0` across runs

Interpretation:

- no immediate evidence of background takeover
- proxy may be optimistic because teacher objectness is still weak and smooth

Risk level: low-to-medium (measurement may be under-sensitive).

## 3) Memory Unused Risk

Observed:

- full `memory_gate_mean ~ 0.864`
- wo_semantics `memory_gate_mean ~ 0.424`
- wo_identity `memory_gate_mean = 0.0` (expected by ablation)

Interpretation:

- memory path is active in non-ablated runs
- not a dead branch in smoke

Risk level: low.

## 4) Semantics/Identity Head Decorative Risk

Observed:

- full and wo_semantics runs trigger `identity_decorative_risk = true`
- semantic loss decreases in full/wo_identity, so semantic head is less likely to be pure decoration
- query loss remains high and tends to rise in late steps for all runs

Interpretation:

- identity head currently under-optimized and may not provide reliable discrimination signal yet
- query grounding objective is still weakly coupled to stable token specialization

Risk level: medium-to-high (identity), medium (query grounding).

## 5) Loss-Without-Objectness Risk

Observed:

- total and semantic losses reduce in several runs
- tokenizer entropy remains near-uniform instead of showing stronger object-centric concentration

Interpretation:

- model can lower losses without clearly sharpening object-centric state structure

Risk level: medium.

## Resolved Runtime Blocker (Smoke Stage)

Command:

- `bash scripts/run_stwm_v4_2_smoke.sh`

Error summary:

- PyTorch autograd version conflict from memory state graph reuse across steps.

Fix applied:

1. memory update path switched to detached no-grad state updates in `retrieval_memory_v4_2.py`
2. trainer now detaches memory state before next step reuse in `train_stwm_v4_2.py`

Impact:

- smoke resumed and all three runs finished successfully.

## Immediate Mitigation Plan

1. Increase token specialization pressure without adding extra modules:
   - add mild entropy floor/ceiling regularization for token-time assignments
   - increase objectness contrast in prior channel
2. Stabilize identity objective:
   - improve positive/negative sampling for InfoNCE beyond short adjacent cache
3. Improve query grounding hardness:
   - harder frame selection target and stronger alignment coupling to token assignments
4. Keep architecture fixed for next iteration:
   - no new memory branches
   - no evaluator changes

## Mainline Decision

V4.2 smoke confirms implementation viability and some directional structure gains, but evidence is not yet hard enough for 1B escalation.

Recommended next step:

- continue 220M V4.2 refinement (structure-preserving), then re-evaluate scaling readiness.
