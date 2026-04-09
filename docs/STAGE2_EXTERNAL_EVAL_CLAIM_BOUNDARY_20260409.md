# Stage2 External Eval Claim Boundary

- Current TAP-style numbers must be called `adapter-based TAP-style probe` or `proxy-style external tracking probe`.
- They must not be called `official TAP-Vid benchmark result` because official_task_faithfully_instantiated remains false.
- Current TAP-style numbers may be reported only as adapter-probe evidence that the frozen Stage2 rollout can be exported into the official evaluator interface.
- Current TAP-style numbers must not be reported as commensurate official benchmark scores, must not be placed into a TAP-Vid main benchmark table, and must not be compared against benchmark-native TAP-Vid papers as if the protocol matched.
- TAP3D remains `not_yet_implemented` because aligned 3D GT for the frozen VSPW+VIPSeg binding is absent, camera geometry / lifting path is absent, and a verified exporter to `tracks_XYZ + visibility` is absent.
- The safest current paper wording for external eval is to say that we performed an adapter-based TAP-style probe by running the official TAP-Vid evaluator on a converted non-native payload from the frozen Stage2 rollout, and that this does not constitute an official TAP-Vid benchmark evaluation.

## Forbidden Terms
- `official benchmark completed`
- `faithfully evaluated on TAP-Vid`
- `official TAP-Vid benchmark result`
- `TAPVid-3D result obtained`

## Safe Replacement Terms
- `adapter-based TAP-style probe`
- `proxy-style external tracking probe`
- `official evaluator invoked on a non-benchmark-faithful adapter payload`

## Current Guardrail Summary
- tap_style_eval_status: partially_bridged
- official_evaluator_invoked: True
- official_task_faithfully_instantiated: False
- paper_official_benchmark: False
- safest_one_sentence_description_for_paper: We report an adapter-based TAP-style probe in which the official TAP-Vid evaluator is run on a converted payload from the frozen VSPW+VIPSeg Stage2 rollout; this is not an official TAP-Vid benchmark result.
