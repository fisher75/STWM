# Identity Memory Diagnosis (Week2)

## Observed Anomaly

Early one-clip ablation reports (`outputs/training/week2_ablations/*.json`) show:

- `full` loss: `0.270403`
- `wo_semantics` loss: `0.323654`
- `wo_trajectory` loss: `0.292344`
- `wo_identity_memory` loss: `0.104229`

The surprising part is `wo_identity_memory` having much lower early loss than `full`.

## Is This Train-Only or Also Seen on Val?

Train-side:

- In week2 mini-val short training (`train_log.jsonl`), train `total_loss` is mixed over steps and does not cleanly favor `wo_identity_memory`; by step 60, `full` is lower (`0.002891`) than `wo_identity_memory` (`0.004590`).
- At step 1, `wo_identity_memory` starts with higher loss than full, then converges quickly.

Val-side (step 60):

- `wo_identity_memory` is worse than `full` on trajectory/query:
  - trajectory L1: `0.027327` vs `0.017848` (delta `+0.009478`)
  - query error: `0.027327` vs `0.017848` (delta `+0.009478`)
- mask IoU difference is small (`-0.000224`).

Conclusion:

- The anomaly is mostly an optimization/loss-surface effect in early or short training.
- It does not overturn validation preference for the full model under current protocol.

## Why This Can Happen

Most likely contributors:

1. Easier optimization without identity append features.
   - `wo_identity_memory` uses lower input dim in the one-clip report (`37` vs `45`), reducing optimization burden.
2. No explicit identity objective in the configured loss.
   - Current formula: `trajectory + 0.5*visibility + 0.2*semantic + 0.1*temporal_consistency`.
3. Identity probes are currently saturated.
   - `identity_consistency=1.0` and `identity_switch_rate=0.0` for all runs and clips.
4. Occlusion probe is not active enough.
   - `occlusion_recovery_acc=0.0` for all runs, so recovery behavior is not being distinguished.

## Does This Prove Identity Memory Is Useless?

No.

Current evidence only says:

- Existing protocol is insufficient to isolate identity-memory value decisively.
- Removing identity memory can simplify fitting in short settings.
- Full model still has better final trajectory/query metrics than `wo_identity_memory`.

## Practical Decision for Current Stage

Recommended near-term stance:

- Keep identity memory in the model path, but treat it as provisional.
- Do not claim strong identity superiority from current metrics.

## Targeted Next Actions (No Full Rerun)

1. Add stronger identity-sensitive validation probes.
   - Track identity switch counts through occlusion windows.
   - Report association consistency on clips with re-appearance events.
2. Add identity-aware supervision term (lightweight).
   - Example: contrastive or correspondence consistency on object tracks.
3. Keep week2 protocol fixed for comparability.
   - Only add sidecar diagnostics first.
4. Decide promotion gate:
   - If identity diagnostics remain flat after targeted probes, keep module optional for 220M scaling.
   - If identity gain appears under stronger probes, preserve and tune before larger models.

## Bottom Line

- `wo_identity_memory` lower early loss is real but not a final verdict.
- Under current mini-val, `full` still wins on trajectory/query at step 60.
- Main issue is metric/protocol sensitivity, not a confirmed module failure.
