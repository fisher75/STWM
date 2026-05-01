# STWM Final Related Work Positioning V3 20260428

## Trace Anything / Trajectory Fields
- Use trajectory-first video state as a conceptual anchor, but STWM predicts future semantic trace-unit fields rather than only reconstructing trajectories.

## SlotFormer / Object-Centric Dynamics
- Closest object-centric dynamics family for same-output baseline intuition: observed slots plus rollout to future semantic slot/state prediction.

## SAVi++ / Real-World Object-Centric Learning
- Important for positioning real-world semantic persistence and object variation as harder than synthetic slot-learning settings.

## FIERY / Future Instance Prediction
- Relevant structured future-state forecasting baseline family, but STWM predicts future semantic prototype fields over trace units rather than future occupancy/map heads.

## DINO-WM / Latent World Models
- Relevant latent-dynamics comparison, though not same-output because STWM outputs structured semantic prototype fields, not generic latent futures.

## Genie / Scaling of World Models
- Relevant to scaling discussion and long-horizon ambitions, but Genie-style generative world models target different output spaces and training scales.

## MotionCrafter / Video Diffusion
- Related as future video synthesis, but not a same-output baseline because RGB diffusion does not directly predict future trace-unit semantic fields under the same protocol.
