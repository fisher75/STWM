# STWM Main Paper Evidence Map V23

## Main Track
- Main paper should use `STWM-FSTF + TUSB/trace_belief utility` as the core story.
- Source of truth is the live repo at `/raid/chen034/workspace/stwm`.
- OSTF should not be promoted to a main claim under the current CoTracker-teacher H8/H16 setup.

## FSTF Evidence
- Mixed Fullscale V2 is complete at `10/10` runs with `C32/C64 x 5 seeds`.
- Canonical selection is `C32 seed456`, chosen by mixed validation only.
- Residual beats copy on mixed, VSPW, and VIPSeg.
- Changed-gain CI excludes zero on mixed, VSPW, and VIPSeg.
- Stable copy is preserved, trace regression is not detected, and the free-rollout semantic field signal is positive.
- Strong controlled same-output evidence exists through the V8 copy-aware suite. The strongest controlled baseline is `copy_residual_mlp`, and STWM beats it on changed-top5, overall-top5, and stable-drop with zero-excluded paired bootstrap.

## Mechanism and Scaling Boundary
- Future rollout hidden is load-bearing at `H8`, and the V13 horizon hidden audits support the same claim at `H16/H24`.
- Future trace coordinates and temporal order are not supported as load-bearing semantic mechanisms.
- `C32` is the best vocabulary tradeoff. `C128` is a failed stability/granularity tradeoff.
- Horizon scaling is positive under the frozen-cache protocol.
- Trace-density stress tests are only weak/inconclusive because K16/K32 valid-slot coverage is reduced; wording must stay `semantic trace-unit field`.
- Model-size scaling is not positive under the strict grouped rule; do not sell a scaling-law story for capacity.

## Utility and Boundary
- TUSB light readout and trace-belief association are usable supporting evidence, not dead branches.
- Trace-belief association improves over frozen external teacher only and over legacy semantic baselines.
- External SAM2/CoTracker/Cutie results should be framed as boundary evidence. STWM is not external overall SOTA.
- The honest boundary story is complementary continuity-heavy utility, especially versus CoTracker on long-gap and occlusion-heavy subsets.

## Figures and Assets
- A paper-ready FSTF visualization pack exists at `artifacts/stwm_fstf_visualization_pack_v13_20260502.tar.gz`.
- Live assets exist under `assets/videos/stwm_fstf_rollout_v12` and `assets/figures/stwm_fstf_rollout_v12`.
- OSTF real-teacher visualizations also exist, but they belong in appendix/future-work discussion.

## Live-Repo-Only Caveat
- `assets/` and `artifacts/` are gitignored, so exported snapshots can be incomplete even when the live repo is complete.
- OSTF real-teacher caches and videos are live-repo assets and should not be assumed to exist in zip snapshots.

## Main-Paper Claim Boundary
### Allowed
- STWM predicts future semantic trace-unit fields over frozen video-derived trace/semantic states.
- STWM improves changed semantic prototype prediction over copy and strong controlled copy-aware baselines while preserving stable semantic memory.
- Future rollout hidden is load-bearing at H8 and remains load-bearing at H16/H24 under the frozen-cache FSTF protocol.
- C32 is the best prototype-vocabulary tradeoff under val-only selection.
- H16/H24 retain positive changed-subset gains under the frozen-cache protocol.
- Trace-belief / TUSB utility supports future identity association and continuity-heavy reacquisition.

### Forbidden
- Raw-video end-to-end training claim.
- Full RGB video generation claim.
- Dense semantic trace field claim.
- Positive model-size scaling-law claim.
- Future trace coordinate / temporal-order load-bearing claim.
- Universal OOD dominance claim.
- STWM beats SAM2/CoTracker overall.
- STWM is a plugin or reranker for SAM2/CoTracker.

## Recommendation
- Start Overleaf with `FSTF/TUSB` as the main paper and keep `OSTF` as appendix or future-work redesign.
