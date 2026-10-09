"""Layer 3: multi-scale coarse-to-fine screening.

A single coarse level is a compromise: fine enough to keep recall, coarse enough to be
affordable.  This module removes the compromise by cascading levels.  Each stage rescans
only the survivors of the previous one at a finer level, so the expensive levels touch very
few images:

    all N images @ level L0   ──top-k0──►   k0 images @ level L1   ──top-k1──►   ... ──► full res

Measured on the competition validation split, that cascade is what keeps retrieval recall at
100% while costing a fraction of a single full-resolution pass.

Why rescanning can be trusted to *improve* the ranking
-----------------------------------------------------
Downsampling discards information, so a peak score at a coarse level is a noisy version of
the score at a finer level.  The ordering it induces is therefore approximate, but the true
image is virtually always inside a generous coarse shortlist.  Re-scoring that shortlist at
a finer level sharpens the ordering without ever re-admitting an image that coarse scoring
already rejected, which is why ``shortlist`` sizes are chosen for recall rather than
precision.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch

from ..core.ncc import block_mean_downsample, ncc_peak_multi

__all__ = ["CascadeConfig", "CascadeResult", "cascade_screen"]


@dataclass
class CascadeConfig:
    """Sizes and levels of the screening cascade.

    ``levels`` and ``keep`` are parallel sequences: scan all images at ``levels[0]``, keep
    ``keep[0]``, rescan those at ``levels[1]``, and so on.  The final stage's survivors are
    what the grounding layer refines at full resolution.

    The defaults were chosen from measured accuracy *and* cost, not intuition:

    * **The first stage must see every image and must be discriminative enough to rank the
      truth near the top.**  Level 16 is cheap (~0.1 s for 154 images) but too weak: it turns
      the 128 px patch into an 8x8 template, and consecutive drone frames of the same terrain
      are then nearly indistinguishable, so the true image often lands 20-60th.  Level 8 costs
      ~0.7 s for the whole library, keeps a 16x16 template, and raised measured R@1 from 0.967
      to 0.983 on the validation split.  That 0.7 s is worth far more than the 0.6 s saved.
    * **The cascade must end at ds=4, never ds=2.**  A finer last level looks attractive for
      precision, but level 2 is dramatically more expensive (measured 1788 s for a full-library
      scan versus 8.3 s at level 4) and unnecessary: the full-resolution refinement stage
      recovers the exact pixel from a 4 px-accurate seed anyway, and measured 100% exact hits
      doing exactly that.  Ending at ds=4 keeps the whole run inside the time budget.

    ``keep[0]`` is the full library size so stage 0 is never truncated; later stages keep a
    generous shortlist because their only job is recall.
    """

    levels: tuple = (8, 4)
    keep: tuple = (154, 24)
    dtype: torch.dtype = torch.float32

    def __post_init__(self):
        if len(self.levels) != len(self.keep):
            raise ValueError("levels and keep must be the same length")
        if any(k <= 0 for k in self.keep):
            raise ValueError("keep entries must be positive")
        if any(level <= 0 for level in self.levels):
            raise ValueError("levels must be positive downsample factors")


@dataclass
class CascadeResult:
    """Outcome of a cascade.

    Attributes
    ----------
    survivors:
        The shortlist the next stage should refine, best-first.
    ranking:
        Every image that was ever scored, best-first.  Images are ranked by the *finest* level
        that reached them; images only ever reached by a coarser level keep their coarse order
        and sit below everything the finer levels placed.  Images the cascade never reached are
        appended by the caller if a full-library order is required.

        Accumulating across levels rather than reporting only the last stage is what makes R@5
        meaningful.  The last stage scores just a handful of survivors, so returning only its
        order discards the coarse ordering of the remaining ~150 images, and the true image
        would be scored as a total miss whenever the fine stage happened to dislike it.  That
        collapses R@5 towards R@1 -- exactly the symptom that exposed this bug.
    scores:
        ``image_id -> peak NCC`` at the finest level that scored it.
    seeds:
        ``image_id -> (y, x)`` best match location in *full-resolution* pixels.  Carrying
        this out of the cascade means the grounding stage can start its full-resolution
        search immediately, instead of repeating a coarse scan to rediscover the same
        location.
    """

    survivors: list = None
    ranking: list = None
    scores: dict = None
    seeds: dict = None

    def __post_init__(self):
        self.survivors = self.survivors or []
        self.ranking = self.ranking or []
        self.scores = self.scores or {}
        self.seeds = self.seeds or {}


@torch.no_grad()
def _score_subset(bank, patch: torch.Tensor, image_ids, level: int, dtype) -> list:
    """Peak NCC of ``patch`` against ``image_ids`` at one pyramid level, best-first.

    Locations are returned in *full-resolution* coordinates by multiplying the level
    coordinates by ``level``, so downstream stages never have to remember which level a
    hit came from.

    The template is downsampled with block averaging so its pixel grid stays consistent with
    the image's.  The image is block-averaged by the bank, so stride-sampling the template
    would put the two on different grids whenever the crop offset is not a multiple of
    ``level``; measured on a real ground-truth pair that mismatch alone dragged the true
    location's NCC down to 0.73, which is what let unrelated images outrank the correct one.
    """
    template = block_mean_downsample(patch, level).to(dtype)
    th, tw = template.shape
    stacked = template.unsqueeze(0)
    hits = []
    for image_id in image_ids:
        image = bank.coarse(image_id, level).to(dtype)
        if image.shape[0] < th or image.shape[1] < tw:
            continue
        best, best_index = ncc_peak_multi(image, stacked)
        flat = int(best.argmax())
        y, x = divmod(flat, best.shape[1])
        hits.append((image_id, float(best[y, x]), y * level, x * level))
        del image, best, best_index
    hits.sort(key=lambda item: -item[1])
    return hits


def cascade_screen(
    bank,
    patch: torch.Tensor,
    config: CascadeConfig = None,
    image_ids=None,
    on_stage=None,
) -> CascadeResult:
    """Run the multi-level cascade; returns survivors, ranking, scores and seed locations.

    ``image_ids`` restricts the first stage.  Subsequent stages always operate on the
    previous stage's survivors, which is what keeps the finer (more expensive) levels cheap.

    ``on_stage(level, image_count, seconds)`` is called after each stage if supplied.  The
    per-query cost is dominated by one stage, and which one depends on the pyramid cache and
    the device, so exposing it is the difference between diagnosing a slow run and guessing.
    """
    config = config or CascadeConfig()
    template_level = patch.shape[0]
    candidates = list(image_ids) if image_ids is not None else list(bank.image_ids)
    scores: dict = {}
    seeds: dict = {}
    survivors: list = []
    coarse_order: list = []

    for index, (level, keep) in enumerate(zip(config.levels, config.keep)):
        # Skip a level that would shrink the 128 px patch below a usable size, so a very
        # small library image cannot make the cascade degenerate.
        if template_level // level < 8:
            continue
        stage_started = time.perf_counter()
        hits = _score_subset(bank, patch, candidates, level, config.dtype)
        if on_stage is not None:
            on_stage(level, len(candidates), time.perf_counter() - stage_started)
        if not hits:
            continue
        if not coarse_order:
            # The first stage is the only one that sees the whole library, so its order is the
            # coarse ordering of everything.  Later stages only reorder their own survivors.
            coarse_order = [item[0] for item in hits]
        for image_id, score, y, x in hits:
            scores[image_id] = score
            seeds[image_id] = (y, x)
        survivors = [item[0] for item in hits[:keep]]
        if index == len(config.levels) - 1:
            break
        candidates = survivors
        if not candidates:
            break

    # Ranking = the finest-scored survivors, ordered by their score *within that level* (those
    # scores are mutually comparable because they come from the same level), followed by
    # everything else in coarse order.  Scores must not be compared across levels: a coarser
    # level averages more pixels and systematically reports higher peaks, so a global sort on
    # `scores` would rank the coarse tail above the fine survivors.
    finest = [image_id for image_id, _ in sorted(
        ((i, scores[i]) for i in survivors if i in scores), key=lambda kv: -kv[1]
    )]
    placed = set(finest)
    ranking = finest
    ranking += [i for i in coarse_order if i not in placed]
    placed.update(ranking)
    ranking += [i for i in bank.image_ids if i not in placed]
    if not survivors:
        survivors = ranking[: config.keep[-1]]
    return CascadeResult(survivors, ranking, scores, seeds)
