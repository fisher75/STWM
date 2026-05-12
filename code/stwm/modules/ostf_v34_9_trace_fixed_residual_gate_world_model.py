from __future__ import annotations

from stwm.modules.ostf_v34_8_causal_assignment_residual_gate_world_model import CausalAssignmentResidualGateWorldModelV348


class TraceFixedResidualGateWorldModelV349(CausalAssignmentResidualGateWorldModelV348):
    """V34.9 learned gate wrapper.

    This intentionally inherits the V34.8 gate architecture; the V34.9 change is
    the trace-preserving measurement bank and causal targets, not trajectory
    architecture search.
    """
