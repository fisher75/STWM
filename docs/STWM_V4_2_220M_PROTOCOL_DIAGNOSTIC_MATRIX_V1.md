# STWM V4.2 220M Protocol Diagnostic Matrix V1 (D1)

Date: 2026-04-03
Status: ACTIVE (background queue submission)
Scale scope: 220M only

## 1) Why First Batch Is Exactly 4 Jobs

First batch uses only the minimum 4-way contrast needed to answer whether semantic supervision and object-bias priors are helping under frozen protocol scoring:

1. `full_v4_2_seed42_fixed_nowarm_lambda1`
2. `full_v4_2_seed42_fixed_warmup_lambda1`
3. `wo_semantics_v4_2_seed42`
4. `wo_object_bias_v4_2_seed42`

Reason:

- keeps causal interpretation simple at fixed seed=42
- validates warm-up effect without introducing full λ sweep confounders
- provides immediate ablations against full model for semantic/object-bias components

## 2) Why Full λ Sweep Is Not Enabled Yet

Full λ sweep is intentionally postponed because:

- frozen protocol closure was just validated by D0 and now needs the minimum D1 confirmation first
- adding broad λ grid now would multiply queue latency and delay the key go/no-go signal
- advanced weighting/surgery should only be considered after fixed-λ + warm-up baseline behavior is stable

Therefore this V1 wave does not enqueue extra λ jobs.

## 3) Exact Warm-up Definition

Warm-up is enabled only for `full_v4_2_seed42_fixed_warmup_lambda1` and applies only to `lambda_sem`:

- first 10% steps: `lambda_sem = 0`
- 10% to 30% steps: linear ramp from `0` to target `lambda_sem`
- after 30% steps: hold target `lambda_sem` constant

No other curriculum shape is used.

## 4) Frozen Protocol Policy (Main vs Eventful)

Frozen assets:

- train manifest: `manifests/protocol_v2/train_v2.json`
- official-main eval manifest: `manifests/protocol_v2/protocol_val_main_v1.json`
- eventful diagnostic manifest: `manifests/protocol_v2/protocol_val_eventful_v1.json`
- evaluator: `code/stwm/evaluators/eval_mini_val.py` with `--protocol-version v2_4_detached_frozen`
- best updater: `code/stwm/tools/update_protocol_best_main.py`

Update rule:

- every 500 steps, detached eval on `protocol_val_main_v1` drives official best update
- official best target is only `best_protocol_main.pt` (+ sidecar)
- detached eval on `protocol_val_eventful_v1` is diagnostics only and never updates official best

## 5) Gradient Audit (Strict Implementation)

Gradient audit is enabled for the two full-model jobs only.

Primary audit (required each audit step):

- anchor: last shared trunk block output feature tensor (`seq_backbone` output feature map)
- computed from task losses `traj/query/sem` on this shared feature tensor
- recorded metrics:
  - `||g_traj||`
  - `||g_query||`
  - `||g_sem||`
  - `cos(g_sem, g_traj)`
  - `cos(g_sem, g_query)`

Secondary audit (low frequency):

- anchor: last shared trunk parameter tensor (last tensor under shared trunk parameter set)
- cadence: every `N=5` audit cycles
- records the same norm/cos set for consistency check against primary anchor conclusions

Audit cadence and cost controls:

- primary interval: `--gradient-audit-interval 100` steps
- secondary interval: every 5 audit cycles (effective every 500 steps)
- lightweight constraint: no full-model high-frequency audit, no micro-step audit

Overhead recording (runtime-measured, not guessed):

- per-audit `audit_time_ms`
- per-audit `audit_memory_delta_mb`
- rolling averages stored in gradient audit JSON

Output path pattern:

- `reports/stwm_v4_2_gradient_audit_220m_seed42_<run_name>.json`

## 6) Gate For Dynamic Weighting / Gradient Surgery

Do not enable uncertainty weighting / GradNorm / PCGrad yet.

They are only allowed after this 4-job wave shows all of the following:

1. protocol-main trend is stable and interpretable across full/warm-up/ablation runs
2. gradient audit shows persistent, reproducible conflict/imbalance pattern
3. fixed-λ best candidate still cannot meet protocol-main improvement targets
