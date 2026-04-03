# STWM Warmup Gradient Comparison V1

Date: 2026-04-03 20:41:54
Status: Generated after multi-point frontend-default audits

## Inputs

- nowarm: `/home/chen034/workspace/stwm/reports/frontend_default_v1/stwm_v4_2_gradient_audit_220m_seed42_full_v4_2_seed42_fixed_nowarm_lambda1_frontend_default_v1.json`
- warmup: `/home/chen034/workspace/stwm/reports/frontend_default_v1/stwm_v4_2_gradient_audit_220m_seed42_full_v4_2_seed42_fixed_warmup_lambda1_frontend_default_v1.json`
- nowarm rows: 2
- warmup rows: 5

## Core Metrics (first / median / last / min / max)

### `||g_traj||`
- nowarm: first=0.000343280524, median=0.000268662443, last=0.000194044362, min=0.000194044362, max=0.000343280524
- warmup: first=0.00158592535, median=0.000180250863, last=0.000234348438, min=0.000154108435, max=0.00158592535

### `||g_sem||`
- nowarm: first=9.31142807e-08, median=2.96190613e-07, last=4.99266946e-07, min=9.31142807e-08, max=4.99266946e-07
- warmup: first=0.00017125126, median=3.80667551e-08, last=3.80667551e-08, min=1.74474799e-08, max=0.00017125126

### `cos(g_sem, g_traj)`
- nowarm: first=-0.0119660108, median=-0.00712354477, last=-0.00228107874, min=-0.0119660108, max=-0.00228107874
- warmup: first=-0.0788646873, median=-0.00230953985, last=0.000457802259, min=-0.0788646873, max=0.00578203393

### `qpath ||g_query||` (query-path-aware anchor)
- nowarm: first=44.5083771, median=79.917141, last=115.325905, min=44.5083771, max=115.325905
- warmup: first=45.7127228, median=75.5309219, last=27.6179523, min=27.6179523, max=114.810333

## Readout

1. This comparison is computed on frontend-default run artifacts only.
2. Positive shift in `cos(g_sem, g_traj)` indicates reduced direct conflict tendency.
3. Query-path-aware `qpath ||g_query||` confirms query supervision path remains measurable.
