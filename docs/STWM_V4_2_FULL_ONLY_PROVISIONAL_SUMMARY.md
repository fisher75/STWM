# STWM V4.2 Full-Only Provisional Summary

## Status

This note is explicitly:

- `PROVISIONAL`
- `FULL-ONLY`
- `NON-FINAL`
- `PROCESS-MONITORING ONLY`

It is based only on the already completed real full runs:

1. `1B seed123 full_v4_2_1b`
2. `220M seed42 full_v4_2`

It must **not** be used as a main paper conclusion.

It must **not** be used to claim final `220M vs 1B` superiority.

Formal interpretation must wait for:

1. `wo_semantics_v4_2`
2. `wo_object_bias_v4_2`

Also note that the primary source artifacts here are the run-local:

- `mini_val_summary.json`
- `train_log.jsonl`

These are suitable for training-process sanity reading, but they are not a substitute for the completed ablation matrix.

## Sources

- `outputs/training/stwm_v4_2_real_220m/seed_42/full_v4_2/mini_val_summary.json`
- `outputs/training/stwm_v4_2_real_220m/seed_42/full_v4_2/train_log.jsonl`
- `outputs/training/stwm_v4_2_real_220m/seed_42/full_v4_2/checkpoints/{best,latest}.pt`
- `outputs/training/stwm_v4_2_real_1b/seed_123/full_v4_2_1b/mini_val_summary.json`
- `outputs/training/stwm_v4_2_real_1b/seed_123/full_v4_2_1b/train_log.jsonl`
- `outputs/training/stwm_v4_2_real_1b/seed_123/full_v4_2_1b/checkpoints/{best,latest}.pt`

## A. Run-Level Average Summary

These numbers come from `mini_val_summary.json -> average_losses`.

| Run | trajectory L1 | query localization error | query-traj gap | total loss | semantic loss | reid loss |
|---|---:|---:|---:|---:|---:|---:|
| `220M seed42 full_v4_2` | `0.012766` | `0.012503` | `-0.000264` | `1.876764` | `1.397768` | `2.674463` |
| `1B seed123 full_v4_2_1b` | `0.012208` | `0.011966` | `-0.000243` | `1.969490` | `1.486269` | `2.894352` |

Immediate full-only read:

- On run-average geometry/query indicators, `1B` is slightly better than `220M`.
- On run-average total / semantic / reid losses, `220M` is better.
- Both runs keep `query_traj_gap` near zero and negative-small, so there is no obvious query/trajectory decoupling failure from this full-only view.

## B. Best vs Latest Checkpoint

Checkpoint comparison below is aligned to `train_log.jsonl` rows at the relevant steps.

### 220M seed42 full_v4_2

- `best.pt`: step `1000`
- `latest.pt`: step `5000`
- `best_total_loss`: `1.715379`

| Metric | best @1000 | latest @5000 | Direction |
|---|---:|---:|---|
| total loss | `1.715379` | `1.954285` | worse late |
| semantic loss | `1.086735` | `1.475103` | worse late |
| reid loss | `2.773010` | `2.773010` | flat |
| trajectory L1 | `0.006562` | `0.006441` | slightly better late |
| query localization error | `0.006182` | `0.006540` | slightly worse late |
| query-traj gap | `-0.000380` | `+0.000099` | near-zero both |

Read:

- `220M` found its best total-loss checkpoint early at step `1000`.
- After that, total and semantic loss drifted upward.
- Geometry/query quality did not collapse, but the run did not keep improving on the optimization objective used for checkpoint selection.

### 1B seed123 full_v4_2_1b

- `best.pt`: step `5000`
- `latest.pt`: step `5000`
- `best_total_loss`: `1.807952`

| Metric | best @5000 | latest @5000 | Direction |
|---|---:|---:|---|
| total loss | `1.807952` | `1.807952` | same step |
| semantic loss | `1.493492` | `1.493492` | same step |
| reid loss | `3.000049` | `3.000049` | same step |
| trajectory L1 | `0.010235` | `0.010235` | same step |
| query localization error | `0.009105` | `0.009105` | same step |
| query-traj gap | `-0.001130` | `-0.001130` | same step |

Read:

- `1B` ended at its best checkpoint by the summary/log view.
- There is a small checkpoint metadata inconsistency:
  - `best.pt` says `best_step=5000`
  - `latest.pt` still stores stale internal best metadata (`best_step=4500`, `best_total_loss=1.910890`)
- This looks like a checkpoint bookkeeping inconsistency rather than a training collapse, because:
  - both checkpoint files are step `5000`
  - the run summary and train log agree that step `5000` is the best observed total-loss point for this run

## C. Current Training Curve Brief

### 220M seed42 full_v4_2

500-step averages show:

- `total_loss` drops quickly early, then stays in a relatively flat-to-worse band:
  - `1-500`: `1.801355`
  - `501-1000`: `1.824363`
  - `4501-5000`: `1.871091`
- `semantic_loss` is best early and later settles around `~1.477`.
- `trajectory L1` and `query localization error` keep improving over training:
  - `trajectory L1`: `0.025559 -> 0.008516`
  - `query localization error`: `0.025278 -> 0.008265`
- So the 220M curve looks like:
  - early optimization win on total/semantic loss
  - later plateau / regression on the main loss
  - continued slow improvement on geometry/query metrics

### 1B seed123 full_v4_2_1b

500-step averages show a steadier late improvement pattern:

- `total_loss` gradually improves, especially in the last window:
  - `1-500`: `2.022288`
  - `501-1000`: `2.044946`
  - `4501-5000`: `1.898963`
- `trajectory L1` and `query localization error` improve more consistently:
  - `trajectory L1`: `0.034096 -> 0.006453`
  - `query localization error`: `0.033951 -> 0.006185`
- Late-window decoupling proxy is also slightly cleaner than 220M:
  - `avg abs query-traj gap, last 100`: `0.000487` vs `220M: 0.000682`
- So the 1B curve looks like:
  - rougher start
  - more monotonic late-stage geometry/query improvement
  - no late collapse
  - but still weaker reid behavior than 220M

## D. Last-100-Step Trend Snapshot

This is the most useful full-only sanity view if the goal is “is the run heading in a sane direction?”

| Run | total loss | semantic loss | reid loss | trajectory L1 | query localization error | avg abs query-traj gap |
|---|---:|---:|---:|---:|---:|---:|
| `220M last100 avg` | `1.868843` | `1.478680` | `2.772240` | `0.008030` | `0.007760` | `0.000682` |
| `1B last100 avg` | `1.897279` | `1.473776` | `2.999769` | `0.005107` | `0.004791` | `0.000487` |

Late-window read:

- `1B` looks better on:
  - trajectory L1
  - query localization error
  - decoupling proxy
  - slightly on semantic loss
- `220M` looks better on:
  - total loss
  - reid loss

## E. Sanity / Risk Flags

From `mini_val_summary.json`:

- both runs show:
  - `tokenizer_collapse_risk = false`
  - `background_bias_risk = false`
  - `memory_inactive_risk = false`
  - `semantic_decorative_risk = false`
  - `identity_decorative_risk = true`

Important implication:

- There is no obvious sign here of global training collapse.
- The identity / reid branch is still the weakest area in both runs.
- This is especially visible for `1B`, where reid stays near `3.0` throughout late training.

## F. What Can Be Read Now vs What Must Wait

What this full-only summary can support right now:

1. whether the two full runs look numerically sane
2. whether losses are exploding or collapsing
3. whether geometry/query metrics are broadly improving
4. whether `1B` is directionally promising on late-window trajectory/query behavior

What this summary cannot support yet:

1. whether semantics is actually necessary for the observed behavior
2. whether object-bias inputs are actually necessary for the observed behavior
3. whether `1B` is truly better than `220M` in the claim sense
4. any main-paper causal statement about representation value

Those require the matched ablations to complete:

1. `wo_semantics_v4_2`
2. `wo_object_bias_v4_2`

## Bottom Line

Full-only provisional read:

1. `1B` looks more promising on late-stage trajectory/query trend quality.
2. `220M` still looks cleaner on total-loss and reid-loss profile.
3. There is no obvious training collapse in either run.
4. The identity/reid branch remains unresolved.
5. No formal paper conclusion should be made until `wo_semantics` and `wo_object_bias` finish.
