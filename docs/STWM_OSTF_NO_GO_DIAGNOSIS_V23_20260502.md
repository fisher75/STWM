# STWM OSTF No-Go Diagnosis V23

## Bottom Line
- `OSTF` should not be a main-paper claim under the current `CoTracker-teacher + H8/H16 + current V17-V22 training/eval stack`.
- This is not a claim that object-dense world modeling is impossible.
- It is a claim that the current setup is not the right setup for a strong paper claim.

## What Succeeded
- V16 is a real asset, not a toy artifact.
- The live repo contains a validated `CoTracker official` teacher cache with:
  - `2132` processed clips
  - `4,267,648` points
  - `valid_point_ratio ~= 0.798`
  - persistent point identities
  - no fake dense / anchor-copied trajectories

## Progression
### V17
- Real dense-cache training became feasible, but the naive multi-trace model was too weak.

### V18
- Physics-prior residual modeling clearly improved over V17.
- This is a real technical gain.
- But it still did not settle the main question on the key `M512` setting against `constant_velocity`.

### V19
- Semantic oracle leakage was fixed.
- After that fix, `M128` and `M512` still did not honestly beat CV.

### V20
- Context-aware deterministic modeling still failed on all-average and CV-hard subsets.
- This is where CV saturation became explicit.

### V21
- Oracle multimodal evaluation became positive.
- Best-of-K / minFDE / MissRate beat CV on all-average and CV-hard subsets.
- This is useful diagnostic evidence, but it remains oracle-side evidence.
- Deterministic no-harm did not hold.

### V22
- Calibration closed the loop on mode selection.
- The result was not a new learned dense-motion mode winning deployment.
- The calibrated top-1 selector returned the `CV mode`.
- For the best `V22` run, top-1 remains worse than CV on point and endpoint metrics.

## Why This Is a No-Go for Main Claims
- The current teacher/benchmark regime is too CV-saturated.
- The model can improve oracle best-of-K metrics, but the deployable top-1 path does not convert that into a claimable win.
- That means the setup is better interpreted as:
  - a good appendix diagnostic
  - a real teacher-cache asset
  - a motivation for redesign

## What We Can Still Claim
- Real object-dense teacher caches are feasible.
- Multimodal oracle diagnostics show nontrivial latent alternatives.
- Current CoTracker-teacher H8/H16 setup is not enough to claim an object-dense semantic trajectory world model.

## What We Should Not Claim
- OSTF beats CV in the current setup.
- Object-dense semantic trace field is solved.
- Dense field should replace the FSTF/TUSB main-paper story.

## Recommendation
- Stop iterating small same-setup OSTF variants.
- Move OSTF to appendix / future-work status.
- If OSTF continues, redesign the teacher/benchmark stack first.
