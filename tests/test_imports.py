"""Smoke-test that every layer imports and that patchmatch exposes its public API.

Run this after moving modules around; an import error deep in a layer is otherwise only
discovered halfway through a solver run.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import patchmatch  # noqa: E402


def main():
    print(f"patchmatch {patchmatch.__version__}")
    print(f"public names: {len(patchmatch.__all__)}")
    for name in patchmatch.__all__:
        obj = getattr(patchmatch, name)
        print(f"  {name:20s} {type(obj).__name__}")

    # each layer must import on its own, so a broken layer is named explicitly
    from patchmatch.core import (  # noqa: F401
        ImageBank, clip_bbox, extract_patch, iou, load_gray_f32,
        ncc_map, ncc_peak, ncc_peak_multi, read_json, window_stats,
    )
    print("layer 1 core       OK")

    from patchmatch.models import (  # noqa: F401
        BatchHardTripletLoss, CircleLoss, CombinedLoss,
        PatchDescriptorNet, build_descriptor,
    )
    print("layer 2 models     OK")

    from patchmatch.retrieval import (  # noqa: F401
        CascadeConfig, CascadeResult, DescriptorRetrievalScorer,
        NccRetrievalScorer, ScreenHit, cascade_screen, reciprocal_rank_fusion,
    )
    print("layer 3 retrieval  OK")

    from patchmatch.grounding import (  # noqa: F401
        GroundingConfig, GroundingResult, locate_patch, refine_in_window,
    )
    print("layer 4 grounding  OK")

    from patchmatch.infer import (  # noqa: F401
        EvalInput, GroundingRecord, Query, RetrievalRecord, Solver, SolverConfig,
        load_eval_input, solve_directory, write_predictions,
    )
    print("layer 5 infer      OK")

    # the descriptor must be constructible and shape-correct without a GPU
    import torch

    model = build_descriptor(embed_dim=128, width=16)
    with torch.no_grad():
        out = model(torch.rand(2, 3, 128, 128))
    print(f"descriptor forward: {tuple(out.shape)}  unit-norm: "
          f"{bool(torch.allclose(out.norm(dim=-1), torch.ones(2), atol=1e-5))}")
    assert out.shape == (2, 128)
    print("ALL LAYERS IMPORT AND RUN")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
