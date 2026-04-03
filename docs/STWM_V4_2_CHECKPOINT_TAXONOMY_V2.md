# STWM V4.2 Checkpoint Taxonomy V2 (Phase C)

Date: 2026-04-03
Status: FROZEN

## Purpose

Define checkpoint roles after protocol freeze, and separate engineering continuity checkpoints from official claim checkpoints.

## Taxonomy

1. latest.pt
   - purpose: resume continuity only
   - claim status: non-official
2. best.pt
   - purpose: trainer-internal best by trainer criterion
   - claim status: non-official under frozen protocol
3. best_protocol_main.pt
   - purpose: official protocol best selected on protocol_val_main_v1 with rule v2
   - claim status: official
4. best_protocol_main_selection.json
   - purpose: audit trail of selection decision
   - claim status: required sidecar for official best provenance
5. protocol_eval/*.json
   - purpose: detached evidence artifacts for each candidate evaluation
   - claim status: evidence required for official best updates

## Promotion Rule

A checkpoint can be promoted to official only through this chain:

1. candidate checkpoint (typically latest.pt at evaluation step)
2. detached eval summary on protocol_val_main_v1 (frozen evaluator)
3. rule v2 comparison against incumbent official best
4. if improved, write best_protocol_main.pt and sidecar

No direct promotion from trainer losses is allowed.

## Freeze Boundary for Legacy Artifacts

- Pre-freeze queue outputs remain available as exploratory/diagnostic evidence.
- Post-freeze main claims must prioritize detached artifacts produced under:
  - frozen evaluator contract
  - frozen split policy
  - frozen protocol best rule

## Operational Note

- Parked queue is intentionally not auto-resumed in this phase.
- Any future training continuation must adopt this taxonomy from the first candidate-eval cycle.
