"""patchmatch -- patch-to-image retrieval and region grounding for aerial imagery.

Layered design, each layer depending only on the ones above it::

    layer 1  core/        bbox geometry, the NCC primitive, image decoding + pyramid cache
    layer 2  models/      learned patch-descriptor network, losses, training dataset
    layer 3  retrieval/   library-wide candidate ranking (task A)
    layer 4  grounding/   localisation inside a known image (task B)
    layer 5  infer/       evaluation I/O contract and the solver that drives layers 3-4

The two scored tasks are solved by one primitive.  A 128x128 query patch is an exact
native-resolution crop of its source image, so normalized cross-correlation is an exact
matcher; the engineering problem is making a library-wide search fit in the runtime budget,
which the coarse-to-fine cascade does.  ``models/`` adds a learned descriptor as a
complementary retrieval signal for the case where the evaluation crops are *not* exact.
"""

from .core import (
    ImageBank,
    clip_bbox,
    iou,
    load_gray_f32,
    ncc_map,
    ncc_peak,
    ncc_peak_multi,
    read_json,
)
from .grounding import GroundingConfig, GroundingResult, locate_patch
from .infer import Solver, SolverConfig, solve_directory
from .retrieval import CascadeConfig, CascadeResult, cascade_screen

__version__ = "1.0.0"

__all__ = [
    "ImageBank",
    "clip_bbox",
    "iou",
    "load_gray_f32",
    "ncc_map",
    "ncc_peak",
    "ncc_peak_multi",
    "read_json",
    "GroundingConfig",
    "GroundingResult",
    "locate_patch",
    "Solver",
    "SolverConfig",
    "solve_directory",
    "CascadeConfig",
    "CascadeResult",
    "cascade_screen",
    "__version__",
]
