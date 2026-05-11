from __future__ import annotations

from pathlib import Path

from stwm.modules.ostf_v34_3_pointwise_unit_residual_world_model import PointwiseUnitResidualWorldModelV343


class SupervisedResidualGateWorldModelV344(PointwiseUnitResidualWorldModelV343):
    """V34.4 residual-gate wrapper: pointwise path stays primary, unit memory is gated residual only."""

    def __init__(
        self,
        v30_checkpoint_path: str | Path,
        *,
        teacher_embedding_dim: int = 768,
        identity_dim: int = 64,
        units: int = 16,
        horizon: int = 32,
    ) -> None:
        super().__init__(
            v30_checkpoint_path,
            teacher_embedding_dim=teacher_embedding_dim,
            identity_dim=identity_dim,
            units=units,
            horizon=horizon,
        )
