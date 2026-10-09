"""Layer 3: candidate ranking for task A (image retrieval).

Task A asks for the source image of a 128x128 patch, ranked.  Scoring is
``0.1 * R@1 + 0.2 * R@5``, so what matters is putting the true image in the top 5 -- and
ideally first -- not the full ordering.

Two independent scorers are provided and combined:

* :class:`NccRetrievalScorer` -- correlate the patch against every image at a coarse
  pyramid level and take the peak.  Because the patches are exact native-resolution crops,
  this peak is ~1.0 on the true image and clearly lower elsewhere.  It is also the only
  scorer that needs no training and transfers to any new library.
* :class:`DescriptorRetrievalScorer` -- embed the patch and compare against per-image
  embeddings from the learned network.  Weaker when crops are exact, but far more tolerant
  of recompression, resampling or colour shifts between library image and query patch.

They are combined by a *rank-fusion* rather than a raw score average, because NCC peaks and
cosine similarities live on different scales; fusing ranks (reciprocal rank fusion) is
scale-free and needs no calibration.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from ..core.ncc import ncc_peak_multi

__all__ = [
    "ScreenHit",
    "NccRetrievalScorer",
    "DescriptorRetrievalScorer",
    "reciprocal_rank_fusion",
]


@dataclass
class ScreenHit:
    """One image's best match against the query: score plus where it matched."""

    image_id: int
    score: float
    y: int
    x: int
    template_index: int = -1

    def __repr__(self) -> str:
        return f"ScreenHit(id={self.image_id}, score={self.score:.4f}, y={self.y}, x={self.x})"


class NccRetrievalScorer:
    """Peak normalized cross-correlation of the patch against every library image.

    ``level`` is the pyramid downsample factor used for scoring.  Measured behaviour on the
    competition validation split: level 4 keeps full recall of the true image, while level
    16 degrades badly (the 128 px patch becomes an 8x8 template that no longer separates
    similar scenes).  The default is therefore the fine end of the coarse range.

    ``dtype`` selects the accumulation precision.  float32 is roughly 3.5x faster and was
    measured to give the same arg-max as float64 at coarse levels; full-resolution
    localisation is the stage that actually needs float64.
    """

    def __init__(self, level: int = 4, dtype=torch.float32):
        self.level = level
        self.dtype = dtype

    def score(self, bank, patch: torch.Tensor, image_ids=None) -> list[ScreenHit]:
        """Score every candidate image; returns hits sorted best-first.

        ``patch`` is a full-resolution 2-D grayscale tensor.  A single template is passed,
        but :func:`ncc_peak_multi` is still used so several query patches can be batched by
        a caller that wants to amortise the per-image pass.
        """
        ids = list(image_ids) if image_ids is not None else bank.image_ids
        template = patch[:: self.level, :: self.level].to(self.dtype)
        th, tw = template.shape

        hits: list[ScreenHit] = []
        stacked = template.unsqueeze(0)
        for image_id in ids:
            image = bank.coarse(image_id, self.level).to(self.dtype)
            if image.shape[0] < th or image.shape[1] < tw:
                continue
            best, best_index = ncc_peak_multi(image, stacked)
            flat = int(best.argmax())
            y, x = divmod(flat, best.shape[1])
            hits.append(ScreenHit(image_id, float(best[y, x]), y, x, int(best_index[y, x])))
            del image, best, best_index
        hits.sort(key=lambda hit: -hit.score)
        return hits


class DescriptorRetrievalScorer:
    """Cosine similarity between the patch embedding and per-image embeddings.

    ``image_embeddings`` maps ``image_id -> (D,)`` unit-norm tensor, precomputed by
    :func:`patchmatch.infer.embedding_index.build_image_embeddings`.  The image embedding is
    the average of its patches' embeddings renormalised to unit length: averaging pool
    embeddings is the standard cheap approximation to an image-level descriptor, and
    renormalising keeps cosine similarity meaningful.
    """

    def __init__(self, model, image_embeddings: dict, device: str = "cuda", batch_size: int = 64):
        self.model = model
        self.image_embeddings = image_embeddings
        self.device = device
        self.batch_size = batch_size

    @torch.no_grad()
    def embed_patches(self, patches: torch.Tensor) -> torch.Tensor:
        """Embed a batch of (N, 3, S, S) float patches in ``[0, 1]``; returns (N, D)."""
        self.model.eval()
        return self.model(patches.to(self.device))

    def score(self, bank, patch_rgb: torch.Tensor, image_ids=None) -> list[ScreenHit]:
        """Score candidate images by embedding similarity; returns hits sorted best-first.

        ``patch_rgb`` is a (3, S, S) float tensor in ``[0, 1]``.  Location fields are left
        at 0 because a pooled embedding carries no localisation information -- the grounding
        layer never consumes these hits.
        """
        if not self.image_embeddings:
            raise ValueError("no image embeddings supplied")
        ids = [i for i in (image_ids if image_ids is not None else self.image_embeddings) if i in self.image_embeddings]
        query = self.embed_patches(patch_rgb.unsqueeze(0))[0]
        matrix = torch.stack([self.image_embeddings[i] for i in ids]).to(query.device)
        similarities = matrix @ query
        order = torch.argsort(similarities, descending=True).tolist()
        return [ScreenHit(ids[j], float(similarities[j]), 0, 0) for j in order]


def reciprocal_rank_fusion(rankings: list[list], k: float = 60.0, weights=None) -> list:
    """Fuse several ranked id lists into one, by reciprocal rank fusion.

    ``score(d) = sum_r weight_r / (k + rank_r(d))``.  Only the *positions* matter, so
    scorers on incomparable scales (an NCC peak near 1.0, a cosine similarity near 0.7) can
    be combined without calibrating them against each other.  ``k`` damps the influence of
    the very top positions; the conventional value is 60.

    Returns ``(id, fused_score)`` pairs sorted best-first.
    """
    weights = weights or [1.0] * len(rankings)
    totals: dict = {}
    for ranking, weight in zip(rankings, weights):
        for rank, item in enumerate(ranking):
            key = item.image_id if hasattr(item, "image_id") else item
            totals[key] = totals.get(key, 0.0) + weight / (k + rank + 1)
    return sorted(totals.items(), key=lambda pair: -pair[1])
