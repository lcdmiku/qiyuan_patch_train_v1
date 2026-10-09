"""Layer 5: the solver -- turning an :class:`EvalInput` into prediction records.

Work budget and ordering
------------------------
The container gets 60 minutes on one A100-80GB, and the score is
``0.1 * R@1 + 0.2 * R@5 + 0.6 * Acc@0.5 + 0.1 * Acc@0.75``.  Grounding carries 0.7 of the
score, so the solver guarantees it first:

1. Decode the library once and build the pyramids (unavoidable fixed cost).
2. Answer every grounding query, grouped by image.
3. Answer every retrieval query, reusing the cascade output.

Grounding queries are grouped by image so each image's full-resolution pixels are decoded at
most once rather than once per query -- with several patches per image, that is the single
largest saving available.

Sharing work between the tasks
------------------------------
A grounding query names its image, yet the cascade still scans the library for it, so the
ranking retrieval needs falls out for free.  One pass over the library per patch therefore
answers both tasks, and the retrieval ranking is the cascade's own ranking rather than a
separate computation.

Why only the top few survivors are verified at full resolution
--------------------------------------------------------------
Localisation runs at full resolution; doing that for all 154 library images per query would
blow the budget (measured: ~4.8 s per image).  So the full-resolution pass is reserved for
the cascade's best few, and the ranking is: verified images ordered by their full-resolution
score, then the remaining cascade order.  Because the cascade keeps the true image inside a
generous shortlist, this ordering protects R@5 and normally R@1 as well.
"""

from __future__ import annotations

import time

import numpy as np
import torch

from ..core.geometry import clip_bbox
from ..core.imagebank import ImageBank, load_gray_f32
from ..grounding.locate import GroundingConfig, locate_patch
from ..retrieval.cascade import CascadeConfig, cascade_screen
from .input_reader import Query, load_eval_input
from .output_writer import GroundingRecord, RetrievalRecord, write_predictions

__all__ = ["SolverConfig", "Solver", "solve_directory"]


class SolverConfig:
    """All tunables in one place, so a run is fully described by a config plus the data.

    Defaults are the values measured on the competition validation split.

    Attributes
    ----------
    cascade:
        Multi-level screening cascade used for the library-wide scan.
    grounding:
        Window sizes and score thresholds for localisation.
    patch_size:
        Expected patch side length, used for the degenerate fallback.
    verify_top:
        How many cascade survivors get a full-resolution verification for the ranking.
        Raising it protects R@1 further at linear cost in full-resolution passes.
    full_cache:
        Full-resolution grayscale arrays kept in host RAM.
    log_every:
        Progress interval, in queries.
    """

    def __init__(
        self,
        cascade: CascadeConfig = None,
        grounding: GroundingConfig = None,
        patch_size: int = 128,
        verify_top: int = 2,
        full_cache: int = 2,
        log_every: int = 20,
    ):
        self.cascade = cascade or CascadeConfig()
        self.grounding = grounding or GroundingConfig()
        self.patch_size = patch_size
        self.verify_top = verify_top
        self.full_cache = full_cache
        self.log_every = log_every


class Solver:
    """Answers queries against a loaded :class:`ImageBank`.

    Usage::

        bank = ImageBank(ids, paths, coarse_levels=(8, 4, 2, 1), device="cuda")
        solver = Solver(bank, SolverConfig())
        solver.load()
        records = solver.solve(queries)
    """

    def __init__(self, bank: ImageBank, config: SolverConfig = None, device: str = "cuda", log=print):
        self.bank = bank
        self.config = config or SolverConfig()
        self.device = device
        self.log = log
        self.model = None             # optional learned descriptor
        self.image_embeddings = None  # optional {image_id: (D,)} for descriptor retrieval
        self.stats: dict = {}

    # ------------------------------------------------------------------ setup
    def load(self) -> None:
        """Decode the whole library and build its coarse pyramids."""
        started = time.time()

        def report(done, total):
            self.log(f"  decoded {done}/{total} images ({time.time() - started:.0f}s)")

        self.bank.load_all(progress=report)
        self.stats["load_seconds"] = time.time() - started
        self.log(f"library ready: {len(self.bank)} images in {self.stats['load_seconds']:.0f}s")

    def attach_descriptor(self, model, image_embeddings: dict) -> None:
        """Enable the learned descriptor as a secondary retrieval signal.

        Optional: without it retrieval uses the NCC cascade alone, which is already
        essentially exact on native-resolution crops.  See ``patchmatch.models.descriptor``.
        """
        self.model = model
        self.image_embeddings = image_embeddings

    # ------------------------------------------------------------------ patches
    def _load_patch(self, query: Query) -> torch.Tensor:
        """Load a query patch as a full-resolution 2-D grayscale GPU tensor."""
        return torch.from_numpy(load_gray_f32(query.patch_path)).to(self.device)

    def _load_patch_rgb(self, query: Query) -> torch.Tensor:
        """Load a query patch as (3, S, S) float RGB in ``[0, 1]``, for the descriptor."""
        from PIL import Image

        Image.MAX_IMAGE_PIXELS = None
        with Image.open(query.patch_path) as im:
            arr = np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0
        return torch.from_numpy(np.ascontiguousarray(arr)).permute(2, 0, 1)

    # ------------------------------------------------------------------ ranking
    def _rank(self, patch: torch.Tensor, preferred=None):
        """Rank the library for one patch, verifying the top survivors at full resolution.

        ``preferred`` is an image id that must lead the ranking when present, which is how a
        grounding query's asserted image is kept consistent with its answer.

        Returns ``(ranking, scores, seeds, located)`` where ``located`` maps a verified
        image id to its :class:`GroundingResult`, so a grounding query never localises twice.
        """
        result = cascade_screen(self.bank, patch, self.config.cascade)
        scores = dict(result.scores)
        located: dict = {}

        for image_id in result.survivors[: max(1, self.config.verify_top)]:
            seed = result.seeds.get(image_id)
            guess_y, guess_x = seed if seed is not None else (None, None)
            hit = locate_patch(
                self.bank, patch, image_id,
                guess_y=guess_y, guess_x=guess_x, config=self.config.grounding,
            )
            located[image_id] = hit
            scores[image_id] = hit.score

        # Verified images first (ordered by their full-resolution score), then cascade order,
        # then anything the cascade never scored.
        verified = sorted(located, key=lambda i: -scores.get(i, -2.0))
        ranking = verified + [i for i in result.ranking if i not in located]
        seen = set(ranking)
        ranking += [i for i in self.bank.image_ids if i not in seen]

        if preferred is not None and preferred in scores:
            ranking = [preferred] + [i for i in ranking if i != preferred]
        return ranking, scores, result.seeds, located

    # ------------------------------------------------------------------ solving
    def solve(self, queries) -> list:
        """Answer every query, returning output records in the input's order."""
        grounding = [q for q in queries if q.is_grounding]
        retrieval = [q for q in queries if not q.is_grounding]
        self.log(f"{len(queries)} queries: {len(grounding)} grounding, {len(retrieval)} retrieval")

        records: dict = {}
        started = time.time()

        # Task B first (0.7 of the score), grouped by image to amortise full-res decodes.
        by_image: dict = {}
        for query in grounding:
            by_image.setdefault(query.image_id, []).append(query)

        for done, (image_id, group) in enumerate(by_image.items(), start=1):
            for query in group:
                patch = self._load_patch(query)
                ranking, _, seeds, located = self._rank(patch, preferred=image_id)

                # Reuse the verification result if this image was among the survivors,
                # otherwise localise it now (the asserted image must always be answered).
                hit = located.get(image_id)
                if hit is None:
                    seed = seeds.get(image_id)
                    hit = locate_patch(
                        self.bank, patch, image_id,
                        guess_y=seed[0] if seed else None,
                        guess_x=seed[1] if seed else None,
                        config=self.config.grounding,
                    )
                height, width = self.bank.shape(image_id)
                bbox = clip_bbox(hit.bbox, width, height)
                records[query.query_id] = [
                    RetrievalRecord(query.query_id, ranking),
                    GroundingRecord(query.query_id, image_id, bbox, hit.score),
                ]
            self.bank.drop_full(image_id)
            if done % max(1, self.config.log_every // 4) == 0 or done == len(by_image):
                self.log(f"  grounding {done}/{len(by_image)} images ({time.time() - started:.0f}s)")

        # Task A.
        for done, query in enumerate(retrieval, start=1):
            patch = self._load_patch(query)
            ranking, _, _, _ = self._rank(patch)
            records[query.query_id] = [RetrievalRecord(query.query_id, ranking)]
            if done % self.config.log_every == 0 or done == len(retrieval):
                self.log(f"  retrieval {done}/{len(retrieval)} ({time.time() - started:.0f}s)")

        self.stats["solve_seconds"] = time.time() - started
        self.log(f"solved in {self.stats['solve_seconds']:.0f}s")

        ordered: list = []
        for query in queries:
            ordered.extend(records.get(query.query_id, []))
        return ordered


def solve_directory(input_root: str, output_path: str, device: str = None, config: SolverConfig = None, log=print):
    """End-to-end entry point: read ``input_root``, solve, write ``output_path``.

    This is what ``run.py`` calls.  Device selection falls back to CPU so the pipeline can be
    smoke-tested anywhere, but the runtime budget assumes a GPU.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    config = config or SolverConfig()

    log(f"device={device}")
    data = load_eval_input(input_root)
    log(f"library: {len(data.image_ids)} images, queries: {len(data.queries)}")

    levels = tuple(
        sorted(set(config.cascade.levels) | {config.grounding.coarse_level, 1}, reverse=True)
    )
    bank = ImageBank(
        data.image_ids, data.image_paths,
        coarse_levels=levels, device=device, full_cache=config.full_cache,
    )
    solver = Solver(bank, config, device=device, log=log)
    solver.load()
    records = solver.solve(data.queries)
    written = write_predictions(output_path, records)
    log(f"wrote {len(records)} records to {written}")
    return records
