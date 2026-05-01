# STWM Final Prototype Vocab Justification V3 20260428

## Summary
- selected_C_justified: `True`
- selected_C_on_mixed_main_result: `32`
- missing_requested_C: `[16]`

## Reason
- C32 is selected from fullscale mixed validation only. Earlier sweeps show finer vocabularies increase granularity but worsen stability/long-tail behavior; the completed 10-run mixed matrix picked C32 seed456 as the best changed-gain/stable-drop/trace-error tradeoff.
