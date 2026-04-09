# Stage2 External Eval Result Interpretation Guardrails

- current_metric_scope: adapter-based TAP-style probe: official TAP-Vid evaluator run on an adapter-converted, non-benchmark-faithful payload exported from the frozen Stage2 core-only VSPW+VIPSeg binding
- official_benchmark_equivalent: False
- dataset_family_match: False
- query_protocol_match: False
- visibility_protocol_match: False
- adapter_probe_only: True
- paper_official_benchmark: False

## Allowed Paper Usage
- Can be reported as an adapter-based TAP-style probe or proxy-style external tracking probe.
- Can be used as supporting evidence that the frozen Stage2 rollout can be exported into an official evaluator interface.
- Can be discussed in text or appendix as boundary-checked external evidence, with explicit non-official labeling.

## Forbidden Paper Usage
- Do not call these numbers official TAP-Vid benchmark results.
- Do not place these numbers into a main table labeled TAP-Vid benchmark or official external benchmark.
- Do not compare these numbers directly against papers evaluated on the official TAP-Vid dataset family and protocol as if they were commensurate.
- Do not claim TAPVid-3D results are obtained.

## Safest One-Sentence Description
- We report an adapter-based TAP-style probe in which the official TAP-Vid evaluator is run on a converted payload from the frozen VSPW+VIPSeg Stage2 rollout; this is not an official TAP-Vid benchmark result.
