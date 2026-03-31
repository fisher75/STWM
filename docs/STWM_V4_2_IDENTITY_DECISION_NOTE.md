# STWM V4.2 Identity Decision Note

## Scope

This note answers the **final controlled identity rescue round** questions under strict frozen boundaries (no new module, no loss change, no 1B).

Reference summary:

- `docs/STWM_V4_2_IDENTITY_RESCUE_ROUND.md`
- `reports/stwm_v4_2_identity_rescue_round_v1.json`

## Q1. 现有数据里 identity/reconnect 事件到底有没有足够 coverage?

结论：**有（在 eventful protocol 下）**。

Evidence:

- eventful protocol report: `coverage_insufficient = False`
- selected eventful clips: `18`
- event type counts include non-zero:
  - reappearance: `9`
  - reconnect: `8`
- rescue-round eventful bucket remains non-zero and has pairing power across seeds.

## Q2. eventful protocol 是否有效提升了 identity 信号敏感度?

结论：**是**。

Reason:

- previous generic mini-val had reconnect/reappearance event coverage collapse (`0` rows).
- eventful protocol now yields stable non-zero event rows and reconnect metrics.
- this turns identity/reconnect from untestable to testable.

## Q3. full_v4_2 vs wo_identity_v4_2 在 eventful 条件下是否出现更可信的差异?

结论：**出现了可测差异，但方向不支持“identity 有效”主张。**

Observed:

- three rescue variants all show eventful reconnect delta (`wo_identity - full`) as positive:
  - `control_resume_base`: `+0.102941`
  - `resume_eventful_mix`: `+0.044118`
  - `resume_eventful_hardquery_mix`: `+0.117647`
- this means reconnect_success consistently favors `wo_identity` over `full` in this rescue round.
- trajectory/query signs remain unstable across variants and seeds.

Interpretation:

- identity/reconnect is measurable, but current evidence does not support retaining identity as a positive claim.

## Q4. 如果没有强差异，下一步应该怎么选?

结论：**正式降级 identity 为 secondary analysis，不再继续救援长链路。**

Recommended action:

1. keep architecture/loss frozen
2. stop identity rescue escalation
3. keep identity/reconnect analysis as constrained secondary or appendix-level evidence
4. center headline and main tables on semantic trajectory state and query decoupling

## Bottom Line

- semantics pillar remains the primary stable claim.
- identity/reconnect is now measurable, but final rescue-round direction does not justify keeping it as a positive secondary claim.
