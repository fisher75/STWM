# STWM Free-Rollout Semantic Trace Field V4 Eval

## Scope
- Evaluation calls `_free_rollout_predict`, not `_teacher_forced_predict`.
- Input is observed trace/video semantic memory only; future targets are used only for metrics.
- Candidate scorer and old association reports are not used.

## C32
- heldout_item_count: `71`
- best_seed: `42`
- copy top5 overall/stable/changed: `0.6879391137780992` / `1.0` / `0.5073937282220989`
- residual top5 overall/stable/changed: `0.8059133546692985` / `0.9992012780028791` / `0.694085036967356`
- future_trace_coord_error: `0.7186991175015768`

## C64
- heldout_item_count: `71`
- best_seed: `123`
- copy top5 overall/stable/changed: `0.6589578521946136` / `1.0` / `0.4212618028875524`
- residual top5 overall/stable/changed: `0.7593676882165656` / `0.9771917325976229` / `0.6075509278796765`
- future_trace_coord_error: `0.8113599783844418`
