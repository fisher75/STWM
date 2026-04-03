# STWM Gradient Audit Addendum V1

Date: 2026-04-03
Scope: Warmup gradient audit closure + `||g_query||=0` root-cause analysis

## 1) Artifact Closure

Requested warmup artifact is now present:

- `reports/stwm_v4_2_gradient_audit_220m_seed42_full_v4_2_seed42_fixed_warmup_lambda1.json`

Reference nowarm artifact:

- `reports/stwm_v4_2_gradient_audit_220m_seed42_full_v4_2_seed42_fixed_nowarm_lambda1.json`

## 2) Warmup vs Nowarm (Primary Anchor)

Primary anchor: `seq_backbone.output_features` (`shared_trunk_features`).

Warmup currently has 1 audit row (step 100). Nowarm has 4 rows.

### 2.1 `||g_traj||`

- nowarm: first `4.092969757e-04` / median `3.179914420e-04` / last `1.451468852e-04` / min `1.451468852e-04` / max `6.122791674e-04`
- warmup: first `2.518583206e-04` / median `2.518583206e-04` / last `2.518583206e-04` / min `2.518583206e-04` / max `2.518583206e-04`

### 2.2 `||g_sem||`

- nowarm: first `3.661321216e-06` / median `7.865170687e-06` / last `1.565494131e-05` / min `3.708107101e-07` / max `1.565494131e-05`
- warmup: first `1.868089603e-04` / median `1.868089603e-04` / last `1.868089603e-04` / min `1.868089603e-04` / max `1.868089603e-04`

### 2.3 `cos(g_sem, g_traj)`

- nowarm: first `-0.001176590` / median `-0.003660101` / last `0.001393427` / min `-0.036070822` / max `0.001393427`
- warmup: first `0.004915388` / median `0.004915388` / last `0.004915388` / min `0.004915388` / max `0.004915388`

## 3) Interpretation

1. Directionally, warmup row (step 100) shows `cos(g_sem, g_traj)` turning positive versus nowarm median slightly negative; this indicates weaker direct conflict at that observed point.
2. Statistical confidence is still limited because warmup currently has only one audit point.
3. Current audit computes gradients of unweighted per-loss terms (`traj_loss`, `sem_loss`, `query_loss`) instead of weighted training objective terms (e.g., `lambda_sem_effective * sem_loss`). Therefore, warmup schedule is not directly reflected in `g_sem` magnitude inside this audit.

## 4) Root Cause: Why `||g_query|| = 0`

### 4.1 Hypothesis checks

1. Query loss not activated: **No**.
   - `query_loss` is non-zero across nowarm training logs (median around `3.905`).
2. Hook/accounting bug in gradient audit: **No primary evidence**.
   - Audit uses `torch.autograd.grad(..., allow_unused=True)` on a valid anchor tensor.
3. Query path does not backpropagate to shared trunk anchor: **Yes (root cause)**.

### 4.2 Mechanistic explanation

1. Audit anchor is `shared_trunk_features` (`seq_hidden`) from model output.
2. `query_loss` is defined from `token_time_attention` gather/max/log path.
3. `token_time_attention` is produced by tokenizer before shared sequence trunk output is consumed.
4. `query_token_logits` exists but is not used in `query_loss`.

Therefore, `d(query_loss)/d(shared_trunk_features) = 0` by graph structure, yielding `||g_query|| = 0` in primary-anchor audit.

## 5) Practical implication

- `||g_query||=0` currently indicates an objective-anchor mismatch, not necessarily a dead training signal globally.
- If we need meaningful query-vs-trunk conflict diagnostics, either:
  1. audit query gradients on tokenizer-side anchor, or
  2. redefine query objective to include path through `query_token_logits` / shared trunk.
