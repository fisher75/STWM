# STWM V4.2 Completed Runs Shared-Metrics Audit

Audit time: 2026-04-03 02:10 +08  
Mode: read-only (no queue mutation, no process intervention)

## 0) Scope

This audit only covers the 4 completed runs requested:

- 1b seed123 full_v4_2_1b
- 1b seed123 wo_semantics_v4_2_1b
- 220m seed42 full_v4_2
- 220m seed42 wo_semantics_v4_2

Goal: decide whether current "wo_semantics looks stronger" is a real shared-metric signal or a metric-caliber artifact.

## 1) Metric Semantics Source Check

From trainer implementation, summary fields are generated from train-log row averages, and summary mode is train in these runs.

- Source: code/stwm/trainers/train_stwm_v4_2_real.py
  - summary mode: "mode": "train" when not eval_only
  - average_losses: aggregated via _avg over training log rows
  - diagnostics: also aggregated via _avg over training log rows

Implication: current mini_val_summary values are training-time aggregate proxies, not standalone protocol-level evaluation outputs.

## 2) Artifact Gate (Is There Protocol-Level Eval Artifact?)

| scale | seed | run | mode | has_mini_val_summary | has_eval_named_file | has_protocol_eval_artifact_hint |
|---|---:|---|---|---|---|---|
| 1b | 123 | full_v4_2_1b | train | True | False | False |
| 1b | 123 | wo_semantics_v4_2_1b | train | True | False | False |
| 220m | 42 | full_v4_2 | train | True | False | False |
| 220m | 42 | wo_semantics_v4_2 | train | True | False | False |

Conclusion of this gate: no explicit protocol-level eval artifact was found for these 4 runs.

## 3) Shared-Metrics-Only Review (Proxy Layer)

### 3.1 Shared proxy table

| scale | seed | run | trajectory_l1(avg) | query_localization_error(avg) | query_traj_gap(avg) | reconnect_success_rate | reappearance_event_ratio | best_step | latest_step |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1b | 123 | full_v4_2_1b | 0.0122 | 0.0120 | -0.0002 | 0.0000 | 0.0000 | 5000 | 5000 |
| 1b | 123 | wo_semantics_v4_2_1b | 0.0073 | 0.0069 | -0.0004 | 0.0000 | 0.0000 | 3300 | 5000 |
| 220m | 42 | full_v4_2 | 0.0128 | 0.0125 | -0.0003 | 0.0000 | 0.0000 | 1000 | 5000 |
| 220m | 42 | wo_semantics_v4_2 | 0.0144 | 0.0141 | -0.0003 | 0.0000 | 0.0000 | 3000 | 5000 |

### 3.2 Full vs wo_semantics delta on shared proxies

| scale | seed | Δtrajectory_l1(wo-full) | Δquery_localization_error(wo-full) | Δquery_traj_gap(wo-full) |
|---|---:|---:|---:|---:|
| 1b | 123 | -0.0049 | -0.0050 | -0.0001 |
| 220m | 42 | +0.0016 | +0.0016 | -0.0000 |

Interpretation:

- 1b seed123: wo_semantics is better on shared proxies.
- 220m seed42: wo_semantics is worse on trajectory_l1 and query_localization_error.
- Cross-scale signal is inconsistent, so no robust "semantic definitely failed" or "wo_semantics definitely wins" statement is justified yet.

## 4) Best vs Latest Checkpoint Comparison (Recoverable Evidence)

Method: compare train_log row at resume.best_step vs latest row. This is a checkpoint-stage proxy, not standalone eval replay on best/latest checkpoints.

| scale | seed | run | Δtrajectory_l1(latest-best) | Δquery_localization_error(latest-best) | Δquery_traj_gap(latest-best) |
|---|---:|---|---:|---:|---:|
| 1b | 123 | full_v4_2_1b | +0.0000 | +0.0000 | +0.0000 |
| 1b | 123 | wo_semantics_v4_2_1b | -0.0017 | -0.0027 | -0.0011 |
| 220m | 42 | full_v4_2 | -0.0001 | +0.0004 | +0.0005 |
| 220m | 42 | wo_semantics_v4_2 | -0.0003 | +0.0032 | +0.0036 |

Interpretation:

- 1b seed123 wo_semantics kept improving from best_step snapshot to latest on these proxies.
- 220m seed42 wo_semantics shows notable degradation in query_localization_error and query_traj_gap from best_step to latest.

## 5) Three-Class Metric Taxonomy (Required)

### 5.1 Truly comparable eval metrics

Current status: none directly available in these 4 run artifacts.

Reason:

- summary mode is train, not protocol eval output.
- no explicit protocol-eval artifact was found (by file naming/content hint scan).

### 5.2 Partially comparable diagnostics / shared proxies

Use with caution:

- average_losses.trajectory_l1
- average_losses.query_localization_error
- average_losses.query_traj_gap
- diagnostics.reconnect_success_rate (currently non-informative because reappearance_event_ratio is 0.0 in all four)

These are shared proxies computed from training-log aggregation, useful for interim monitoring but not sufficient alone for paper headline claims.

### 5.3 Not comparable losses (for ablation headline)

Do not use as core cross-ablation conclusion:

- average_losses.total (explicitly non-comparable under changed loss composition)
- average_losses.query
- average_losses.reid
- average_losses.semantic (disabled branch in wo_semantics)

Reason:

- objective composition/weights and active branches differ across ablations; these losses are optimization-internal, not protocol-level evaluation endpoints.

## 6) Mandatory Answers

### Q1. Is "wo_semantics looks stronger" supported by shared eval metrics, or mostly by non-comparable losses?

Answer:

- It is not reliably supported as a universal shared-metric conclusion.
- A substantial part of the apparent advantage comes from non-comparable loss terms.
- On shared proxies, evidence is split by scale (1b seed123 favors wo_semantics; 220m seed42 does not).

### Q2. If only shared metrics are considered, what is the semantic-mainline status now?

Answer: very dangerous.

Rationale:

- no protocol-level eval artifact yet,
- shared-proxy signal is inconsistent across scales,
- reconnect-related signal is currently uninformative (event coverage zero).

### Q3. What provisional judgment should we use before remaining training completes?

Recommended provisional judgment:

- Do not publish or lock "wo_semantics stronger" as a main claim yet.
- Keep semantic-mainline as unresolved/high-risk rather than failed-by-proof.
- Use shared-proxy trend only as interim monitoring; final claim must wait for consistent protocol-level eval outputs under a unified metric caliber.

## 7) Read-Only Integrity Note

This audit performed no queue edits, no worker restarts, and no process termination.
