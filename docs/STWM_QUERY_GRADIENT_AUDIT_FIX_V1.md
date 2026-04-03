# STWM Query Gradient Audit Fix V1

Date: 2026-04-03
Status: Implemented and smoke-validated

## 1) Problem Statement

Existing gradient audit reported `||g_query|| = 0` on the primary anchor (`shared_trunk_features` / seq trunk output), creating ambiguity about whether query supervision was inactive or disconnected.

## 2) Root Cause

The old primary audit anchor is `shared_trunk_features` (`seq_hidden`), but `query_loss` is computed from `token_time_attention` gather/max/log path.

- `query_loss` does not use `query_token_logits`.
- `token_time_attention` originates from tokenizer path before seq trunk output is consumed.

So `d(query_loss)/d(shared_trunk_features) = 0` can be structurally true even when query loss is active.

## 3) Fix Scope (without changing existing sem-vs-traj shared-trunk audit)

File updated:

- `code/stwm/trainers/train_stwm_v4_2_real.py`

Kept unchanged:

- Existing primary shared-trunk audit fields (`g_traj_norm`, `g_query_norm`, `g_sem_norm`, `cos_sem_traj`, `cos_sem_query`).

Added new query-path-aware audit fields (anchor: `token_time_attention`):

- `query_path_anchor`
- `query_path_anchor_shape`
- `qpath_g_query_norm`
- `qpath_g_sem_norm`
- `qpath_g_traj_norm`
- `qpath_cos_sem_query`
- `qpath_cos_traj_query`
- `qpath_cos_sem_traj`

Also added `query_path_anchor` in gradient-audit metadata header.

## 4) Smoke Validation

Smoke run report:

- `reports/stwm_query_gradient_audit_fix_smoke_v1.json`

Observed:

1. Legacy shared-anchor behavior remains unchanged:
   - `g_query_norm` remains `0.0` in sampled rows.
2. New query-path-aware metrics are non-zero:
   - `qpath_g_query_norm` around `45.9` (non-zero)
   - `qpath_g_sem_norm` non-zero
3. `qpath_g_traj_norm` is `0.0` in sampled rows, indicating no direct traj coupling on this query path anchor.

## 5) Interpretation

1. Old `||g_query||=0` was an anchor mismatch artifact, not query-loss inactivity.
2. Query-vs-sem interaction is now measurable on the query path.
3. Query-vs-traj direct conflict appears absent on this anchor in current design.
