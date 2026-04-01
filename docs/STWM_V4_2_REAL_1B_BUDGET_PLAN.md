# STWM V4.2 Real 1B Confirmation Budget Plan

## Scope And Frozen Boundaries

This plan keeps method boundaries fixed.

1. No new model module.
2. No loss composition change.
3. No identity rescue continuation.
4. No 3B training in this round.
5. Lightweight staged 1B is sanity-only, not main evidence.

## 220M Main-Evidence Budget (Reference)

Reference scripts and artifacts:

- Base protocol script: `scripts/run_stwm_v4_2_minival_multiseed.sh`
- Harder protocol script: `scripts/run_stwm_v4_2_state_identifiability.sh`
- Main table sources:
  - `outputs/training/stwm_v4_2_minival_multiseed/comparison_multiseed.json`
  - `outputs/training/stwm_v4_2_state_identifiability/comparison_state_identifiability.json`

Observed reference budget:

- Base protocol:
  - model preset: `prototype_220m_v4_2`
  - runs: `full_v4_2, wo_semantics_v4_2, wo_identity_v4_2`
  - seeds: `42,123,456`
  - steps: `120`
  - sample_limit: `18`
  - save checkpoint: yes
- State-identifiability protocol:
  - runs: `full_v4_2, wo_semantics_v4_2, wo_object_bias_v4_2`
  - seeds: `42,123,456`
  - eval steps: `60`
  - sample_limit: `18`
  - mode: eval on resumed checkpoints

## Real 1B Confirmation Budget (This Round)

Real 1B budget is intentionally stronger than lightweight settings and directly aligned with paper protocols.

- Mandatory seeds: `42,123`
- Optional seed: `456` (only after mandatory seeds are stable and resources allow)
- Runs (must all be covered):
  - `full_v4_2_1b`
  - `wo_semantics_v4_2_1b`
  - `wo_object_bias_v4_2_1b`
  - Implementation run names remain `full_v4_2, wo_semantics_v4_2, wo_object_bias_v4_2`.

### Base Protocol Budget

- model preset: `prototype_1b_v4_2`
- steps: `900`
- sample_limit: full base manifest clip count (`70` on current manifest)
- checkpoints/logs/summaries: required

### State-Identifiability Protocol Budget

- model preset: `prototype_1b_v4_2`
- eval steps: `240`
- sample_limit: `18` (full protocol set size)
- checkpoints/logs/summaries: required (`eval_model.pt` + `train_log.jsonl` + `mini_val_summary.json`)

## Why This Qualifies As Real 1B Confirmation

Compared with 220M reference budget:

- Base steps: `900` vs `120` (`7.5x`).
- Base sample coverage: `70` vs `18` (`3.9x`).
- Base clip-step exposure: `900 * 70 = 63000` vs `120 * 18 = 2160` (`29.2x`).
- Harder protocol eval steps: `240` vs `60` (`4x`).

Compared with invalid lightweight staged round:

- Not smoke-like tiny steps.
- Not full-only.
- Not skip-existing pseudo completion.

## Runtime Estimate (Hours/Days)

Empirical reference from staged queue logs:

- full-only one-seed phase with `base120 + state60` took about `105s`.
- Approx step time: `105 / 180 ~= 0.58 s/step` on observed shared environment slice.

Projected real 1B compute per run-seed:

- `900 + 240 = 1140` steps.
- Estimated runtime per run-seed: about `11-16 min` (with contention range).

Projected mandatory matrix (`2 seeds * 3 runs`):

- Single lane serial: about `1.1-1.8 h`.
- Two single-GPU lanes in parallel: about `0.6-1.2 h`.
- Add queue wait/contention buffer on shared cluster: usually `< 0.5 day`, worst-case can extend near `1 day`.

## Execution Commands

- Submit mandatory real jobs:

```bash
bash scripts/submit_stwm_v4_2_real_1b_queue.sh \
  --seeds 42,123 \
  --queue-root outputs/queue/stwm_1b_real \
  --out-root outputs/training/stwm_v4_2_1b_real_confirmation
```

- Optional third seed when mandatory seeds are stable:

```bash
bash scripts/submit_stwm_v4_2_real_1b_queue.sh \
  --seeds 42,123 \
  --include-seed-456 \
  --queue-root outputs/queue/stwm_1b_real \
  --out-root outputs/training/stwm_v4_2_1b_real_confirmation
```

- Finalize summaries and 220M-vs-1B real comparison:

```bash
bash scripts/finalize_stwm_v4_2_1b_real_confirmation.sh \
  outputs/training/stwm_v4_2_1b_real_confirmation
```
