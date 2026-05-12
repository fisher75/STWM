from __future__ import annotations

from pathlib import Path

from stwm.modules.ostf_v34_3_pointwise_unit_residual_world_model import PointwiseUnitResidualWorldModelV343


class BestResidualGateWorldModelV346(PointwiseUnitResidualWorldModelV343):
    """V34.6 wrapper: pointwise base remains primary, best residual content receives a learned sparse gate."""

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
        self.v34_6_pointwise_base_primary = True
        self.v34_6_unit_branch_residual_only = True
