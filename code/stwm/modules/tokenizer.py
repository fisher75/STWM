from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from stwm.modules.trace_adapter import TraceSummary
from stwm.modules.semantic_adapter import SemanticSummary


@dataclass
class TokenBatch:
    tokens: torch.Tensor
    token_metadata: dict[str, Any]


class SemanticTrajectoryTokenizer:
    """Fuses geometric and semantic summaries into object-centric tokens.

    TODO:
    - support multi-object token packing
    - add identity memory features
    - add mask latent statistics
    """

    def __init__(self, text_dim: int = 16) -> None:
      self.text_dim = text_dim

    def encode(self, trace: TraceSummary, semantics: SemanticSummary) -> TokenBatch:
      steps = trace.centers.shape[0]
      pooled_text = semantics.text_embeddings.mean(dim=1)
      tokens = torch.cat(
        [
          trace.centers,
          trace.velocities,
          trace.visibility,
          pooled_text,
        ],
        dim=-1,
      )
      return TokenBatch(
        tokens=tokens.unsqueeze(0),
        token_metadata={
          "num_steps": steps,
          "token_dim": tokens.shape[-1],
          "labels": semantics.metadata.get("labels", []),
        },
      )
