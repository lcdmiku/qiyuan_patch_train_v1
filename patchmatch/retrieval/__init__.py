"""Layer 3: retrieval (task A) -- ranking library images for a query patch."""

from .cascade import CascadeConfig, CascadeResult, cascade_screen
from .scorers import (
    DescriptorRetrievalScorer,
    NccRetrievalScorer,
    ScreenHit,
    reciprocal_rank_fusion,
)

__all__ = [
    "CascadeConfig",
    "CascadeResult",
    "cascade_screen",
    "DescriptorRetrievalScorer",
    "NccRetrievalScorer",
    "ScreenHit",
    "reciprocal_rank_fusion",
]
