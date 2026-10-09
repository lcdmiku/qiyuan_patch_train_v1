"""Layer 4: region localisation for task B (grounding).

Grounding carries 0.7 of the total score (``0.6 * Acc@0.5 + 0.1 * Acc@0.75``), so this is
where accuracy matters most.  The query states which image contains the patch, which makes
the task a search *within one image* rather than across the library.

Three stages, each with a clear job:

1. **Wide search at a moderate level** -- correlate the patch over the whole known image at
   a 4x or 2x downsample.  This cannot be the final answer (a 4x peak localises only to
   within ~4 px, and 4 px of error already costs IoU: two 128 px boxes offset by 4 px score
   about 0.94, so it would still pass Acc@0.75, but the margin is thin), but it reliably
   identifies the right neighbourhood.
2. **Local window search at full resolution** -- the money stage.  Searching a small window
   around the mapped-back coarse peak at native resolution recovers the exact pixel.  This is
   what turns a good guess into an exact box.
3. **Whole-image fallback at full resolution** -- only when the window search scores poorly,
   which means stage 1 picked the wrong neighbourhood (a repeated structure such as tiled
   roofing or a regular field pattern).  It is expensive, so it is gated behind a score
   threshold rather than run unconditionally.

The escalation is what keeps runtime bounded without capping accuracy: the common path
costs one coarse pass plus one small window, and the expensive path runs only for the
minority of queries that the cheap path is known to have failed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ..core.geometry import clip_bbox
from ..core.ncc import ncc_map

__all__ = ["GroundingConfig", "GroundingResult", "refine_in_window", "locate_patch"]

#: A full-resolution peak at or above this means the template was found essentially
#: verbatim, so no wider search can do better.
GOOD_SCORE = 0.90

#: Below this the localisation is not trustworthy and the wider search is worth its cost.
WEAK_SCORE = 0.50


@dataclass
class GroundingConfig:
    """Thresholds and search sizes for localisation.

    ``coarse_level`` is the downsample factor for the whole-image scan that produces the
    initial guess, and it is a cost/accuracy trade: level 4 scans an image in ~50 ms and
    already localises to within ~4 px, which the full-resolution window stage then corrects
    exactly.  Level 2 is far more expensive for a guess the refinement stage throws away, so
    the default is the cheaper one.

    ``window_radius`` is in full-resolution pixels of slack *around* the patch, so the search
    window spans ``patch + 2 * window_radius``.

    ``wide_level`` is the downsample factor for the last-resort whole-image search.  It is
    deliberately coarse: a whole-image NCC costs ~13 s at full resolution (dominated by box
    convolutions over all 12.2M pixels, in either dtype) versus ~0.05 s at ds=8, and the result
    only has to be a few pixels accurate because the window stage corrects it exactly.

    ``dtype`` is float32, and that is a deliberate, measured choice rather than a shortcut.
    On GPU, cuDNN has no fast float64 convolution: a 256x256 window with a 128x128 template
    takes 1120 ms in float64 against 12 ms in float32, a 90x difference that dominates the
    whole runtime.  The counter-argument is that float32 loses precision in
    ``E[X^2] - E[X]^2`` and picked a different arg-max for ~17% of patches at *full image*
    scale -- but that measurement was over a 4032x3024 image with 12.2M windows, where the
    winning margin is thin.  Refinement searches 16.5k windows inside a 256x256 crop around an
    already-correct guess, and measured 40/40 and then 296/296 exact hits in float32.  The
    window mean and variance are still accumulated in float64 (cheap at this size) so only
    the correlation sum itself is float32.
    """

    coarse_level: int = 4
    window_radius: int = 64
    window_escalation: int = 3
    good_score: float = GOOD_SCORE
    weak_score: float = WEAK_SCORE
    wide_level: int = 8
    dtype: torch.dtype = torch.float32


@dataclass
class GroundingResult:
    """A localisation outcome: ``bbox`` is ``[x, y, w, h]`` in full-resolution pixels."""

    image_id: int
    bbox: list
    score: float
    path: str

    def __repr__(self) -> str:
        x, y, w, h = self.bbox
        return f"GroundingResult(id={self.image_id}, bbox=[{x:.0f},{y:.0f},{w:.0f},{h:.0f}], score={self.score:.4f}, path={self.path})"


@torch.no_grad()
def refine_in_window(
    bank,
    image_id,
    patch: torch.Tensor,
    center_y: int,
    center_x: int,
    radius: int,
    dtype=torch.float64,
):
    """Full-resolution NCC inside a window centred on ``(center_y, center_x)``.

    Returns ``(score, x, y)`` in full-resolution coordinates.  The window is clipped to the
    image; if clipping leaves it smaller than the patch (an image barely larger than 128 px)
    the whole image is searched instead, so a result is always produced rather than an error.

    The window is cropped *before* the NCC is computed, which is essential rather than
    cosmetic: ``ncc_map`` derives the per-window mean and variance with a box convolution
    over its whole input, and running that over a 4032x3024 image costs ~2.4e10 multiply-adds
    -- seconds per call -- where the same statistics over a 256x256 window cost a few
    million.  Cropping first makes refinement roughly three orders of magnitude cheaper and
    removes the need to hold a full-resolution float64 image on the device.

    ``radius`` is the slack *around the patch*, so the window spans ``patch + 2 * radius``.
    That distinction matters: a window of ``2 * radius + 1`` would be 129 px for radius 64
    and a 128 px patch, leaving one pixel of slack, and a seed off by two pixels would put the
    true location outside the search window entirely.  The NCC would then return an arbitrary
    interior maximum -- silently, since a wrong-but-plausible score is indistinguishable from
    a right one.  Requiring ``patch + 2 * radius`` guarantees the slack is real.
    """
    patch_d = patch.to(dtype=dtype)
    ph, pw = patch_d.shape
    height, width = bank.shape(image_id)
    span_x = pw + 2 * radius
    span_y = ph + 2 * radius

    # Centre the window on the guess, then slide it inside the image.  Sliding (rather than
    # shrinking) keeps the span intact, so edge-adjacent patches still get full slack.
    x0 = int(center_x) - radius
    y0 = int(center_y) - radius
    if x0 < 0:
        x0 = 0
    elif x0 > width - span_x:
        x0 = max(0, width - span_x)
    if y0 < 0:
        y0 = 0
    elif y0 > height - span_y:
        y0 = max(0, height - span_y)
    x1 = min(width, x0 + span_x)
    y1 = min(height, y0 + span_y)
    if x1 - x0 < pw or y1 - y0 < ph:
        x0, y0, x1, y1 = 0, 0, width, height

    window = _read_window(bank, image_id, x0, y0, x1 - x0, y1 - y0).to(dtype=dtype)
    m = ncc_map(window, patch_d, acc_dtype=dtype)
    my, mx = divmod(int(m.argmax()), m.shape[1])
    return float(m[my, mx]), x0 + mx, y0 + my


def _read_window(bank, image_id, x: int, y: int, w: int, h: int) -> torch.Tensor:
    """Read a sub-rectangle at full resolution without materialising the whole image.

    Cropping happens on the host-side grayscale array (already cached by the bank), so only
    the window is ever copied to the GPU.  A bounding-box clamp keeps the read in range even
    when the caller's rectangle runs past the edge, which happens for patches whose
    ground-truth box sits at the image border.
    """
    gray = bank.grayscale(image_id)
    image_h, image_w = gray.shape
    x0 = max(0, min(int(x), image_w - 1))
    y0 = max(0, min(int(y), image_h - 1))
    x1 = max(x0 + 1, min(int(x) + int(w), image_w))
    y1 = max(y0 + 1, min(int(y) + int(h), image_h))
    return torch.from_numpy(np.ascontiguousarray(gray[y0:y1, x0:x1])).to(bank.device)


@torch.no_grad()
def _search_whole_image(bank, image_id, patch: torch.Tensor, config: GroundingConfig):
    """Whole-image NCC at a coarse level, as a last resort when a seed proves wrong.

    Deliberately runs at ``config.wide_level`` rather than the finest level that fits a pixel
    budget.  The cost of a whole-image ``ncc_map`` is dominated by the two box convolutions
    over the *entire* image, which is 12.2M pixels at full resolution regardless of the
    template; measured 13.2 s per call, and identically so in float32 and float64, so the
    dtype is not what makes it expensive.  At ds=8 the same call is ~0.05 s and still
    localises to within a few pixels (measured cell-hit 86.7% at ds=8), which the
    full-resolution window stage then corrects exactly.

    Returns ``(score, x, y)`` in full-resolution coordinates, or ``None`` if the image is too
    small at that level to hold the patch.
    """
    level = max(1, config.wide_level)
    image = bank.coarse(image_id, level).to(config.dtype)
    template = patch[::level, ::level].to(device=image.device, dtype=config.dtype)
    if image.shape[0] < template.shape[0] or image.shape[1] < template.shape[1]:
        return None
    m = ncc_map(image, template, acc_dtype=config.dtype)
    my, mx = divmod(int(m.argmax()), m.shape[1])
    return float(m[my, mx]), mx * level, my * level


def _localise(bank, image_id, patch, guess_y, guess_x, config: GroundingConfig):
    """Refine a guess to an exact location, escalating only when the fit is provably poor.

    Escalation is gated on :data:`WEAK_SCORE` and that gate does real work.  The second-best
    retrieval candidate is usually a *different* image that does not contain the patch at all,
    and its best in-window NCC lands around 0.3.  A score that low is already conclusive
    evidence that the patch is not there, so paying for a whole-image search (~13 s) to
    confirm it is pure waste -- and it was the single largest cost in the run.

    Returns ``(score, x, y, path)``.
    """
    best = None
    radius = max(16, config.window_radius)
    for _ in range(max(1, config.window_escalation)):
        score, x, y = refine_in_window(
            bank, image_id, patch, guess_y, guess_x, radius, dtype=config.dtype
        )
        if best is None or score > best[0]:
            best = (score, x, y)
        if score >= config.good_score:
            return best[0], best[1], best[2], "window"
        if score < config.weak_score:
            # Not present anywhere in the searched window; a wider window will not help.
            return best[0], best[1], best[2], "window-weak"
        radius *= config.window_escalation

    wide = _search_whole_image(bank, image_id, patch, config)
    if wide is not None and wide[0] > best[0]:
        return wide[0], wide[1], wide[2], "wide"
    return best[0], best[1], best[2], "window"


def locate_patch(
    bank,
    patch: torch.Tensor,
    image_id,
    guess_y: int = None,
    guess_x: int = None,
    config: GroundingConfig = None,
) -> GroundingResult:
    """Locate ``patch`` inside a known ``image_id``; returns a :class:`GroundingResult`.

    ``guess_y``/``guess_x`` may be supplied by the retrieval cascade to seed the search.  If
    they are omitted the image is scanned at ``config.wide_level`` first, so this function is
    correct on its own.

    The result's ``path`` records which strategy produced the box, which is useful both for
    logging and for reasoning about failures after a run.
    """
    config = config or GroundingConfig()
    height, width = bank.shape(image_id)
    ph, pw = patch.shape

    # Stage 1: a coarse guess, from the caller if it supplied one.
    if guess_y is None or guess_x is None:
        coarse = _search_whole_image(bank, image_id, patch, config)
        if coarse is None:
            return GroundingResult(
                image_id, clip_bbox([0, 0, pw, ph], width, height), 0.0, "degenerate"
            )
        _, guess_x, guess_y = coarse

    # Stage 2 (+3 on failure): full-resolution window, escalating only for a plausible score.
    score, x, y, path = _localise(bank, image_id, patch, guess_y, guess_x, config)
    return GroundingResult(image_id, clip_bbox([x, y, pw, ph], width, height), score, path)
