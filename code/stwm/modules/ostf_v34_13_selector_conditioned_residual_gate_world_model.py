from __future__ import annotations

from stwm.modules.ostf_v34_13_selector_conditioned_residual_memory import SelectorConditionedResidualMemoryV3413


class SelectorConditionedResidualGateWorldModelV3413(SelectorConditionedResidualMemoryV3413):
    """V34.13 learned gate 占位模型。

    本轮只有 selector-conditioned oracle residual probe 通过后才允许训练该 gate。
    当前类保留接口，避免把 oracle residual 误包装为 learned semantic field。
    """

    pass
