# STWM Final Paper Tables

## Semantic Trace Field Main Result
- mixed: STWM top5 0.8926 vs copy 0.8562; changed gain 0.0816; stable drop 0.0002
- VSPW: STWM top5 0.8272 vs copy 0.7513; changed gain 0.1355; stable drop 0.0004
- VIPSeg: STWM top5 0.9422 vs copy 0.9357; changed gain 0.0182; stable drop 0.0001

## Significance
- mixed: changed delta 0.0567, CI [0.04312341928588929, 0.07121790725768344], zero_excluded=True
- VSPW: changed delta 0.1170, CI [0.08793045383360651, 0.14494100372617444], zero_excluded=True
- VIPSeg: changed delta 0.0215, CI [0.008215346046082385, 0.035472012489260005], zero_excluded=True

## Utility / Belief Association
- trace_belief_assoc: positive=True (improves over calibration/cropenc/legacysem)
- bootstrap: positive=True (trace belief zero-excluded on ID panel)
- false-confuser: positive=True (reduces false-confuser errors)
- reacquisition: positive=True (supports reacquisition utility)
- planning-lite: positive=False (supports planning-lite risk utility)
- counterfactual: positive=True (trace counterfactual changes decisions)

- table_count: `5`
