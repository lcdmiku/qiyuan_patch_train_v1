"""Layer 4: grounding (task B) -- locating a patch inside a known image."""

from .locate import GOOD_SCORE, WEAK_SCORE, GroundingConfig, GroundingResult, locate_patch, refine_in_window

__all__ = [
    "GOOD_SCORE",
    "WEAK_SCORE",
    "GroundingConfig",
    "GroundingResult",
    "locate_patch",
    "refine_in_window",
]
