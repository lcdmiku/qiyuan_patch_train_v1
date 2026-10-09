"""Layer 2: metric-learning objectives for the patch descriptor.

Both losses operate on L2-normalised embeddings, so cosine similarity is a dot product and
the temperature controls how sharply the softmax focuses on the hardest negatives.

Why two losses
--------------
The descriptor serves retrieval, whose metric only cares about the *order* of candidates.
Two complementary pressures produce a good order:

* :class:`CircleLoss` reweights each positive/negative pair by how far it is from its
  optimum, so a well-separated pair stops contributing and effort goes to the pairs that
  are actually confused.  With ~150 visually similar aerial images, most negatives are
  trivially separated; without this reweighting the gradient is dominated by them.
* :class:`BatchHardTripletLoss` keeps the classic "hardest positive vs hardest negative"
  pressure, which is simpler and acts as a useful stabiliser early in training.

``CombinedLoss`` mixes them with a configurable weight.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["CircleLoss", "BatchHardTripletLoss", "CombinedLoss"]


def _pairwise_cosine(embeddings: torch.Tensor) -> torch.Tensor:
    """Cosine similarity matrix; inputs are assumed L2-normalised so this is a matmul."""
    return embeddings @ embeddings.t()


class CircleLoss(nn.Module):
    """Circle loss over in-batch positive/negative pairs.

    ``gamma`` sets the decision boundary: a positive is pushed above ``1 - margin`` and a
    negative below ``margin``.  ``temperature`` sharpens the log-sum-exp over negatives.

    ``labels`` groups samples: two samples with the same label are a positive pair.  For
    this task a "label" is the source image id, so patches cropped from the same aerial
    frame pull together and patches from different frames push apart.  Pairs with no
    positive or no negative in the batch contribute nothing, which keeps the loss finite
    when a batch happens to be degenerate.
    """

    def __init__(self, margin: float = 0.25, gamma: float = 64.0, temperature: float = 0.1):
        super().__init__()
        self.margin = margin
        self.gamma = gamma
        self.temperature = temperature

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """``embeddings``: (N, D) L2-normalised.  ``labels``: (N,) source-image ids."""
        sim = _pairwise_cosine(embeddings)
        n = embeddings.shape[0]
        same = labels.unsqueeze(0) == labels.unsqueeze(1)
        eye = torch.eye(n, dtype=torch.bool, device=embeddings.device)
        positive = same & ~eye
        negative = ~same

        # Circle loss reweights each pair by its distance from the optimum.
        pos_weight = (1.0 - sim + self.margin).clamp(min=0.0)
        neg_weight = (sim + self.margin).clamp(min=0.0)
        logits_pos = -self.gamma * pos_weight * (sim - (1.0 - self.margin))
        logits_neg = self.gamma * neg_weight * (sim - self.margin)

        # Mask with -inf so log-sum-exp ignores absent pairs, then stabilise per row.
        big_neg = torch.finfo(sim.dtype).min
        logits_pos = torch.where(positive, logits_pos, torch.full_like(sim, big_neg))
        logits_neg = torch.where(negative, logits_neg, torch.full_like(sim, big_neg))

        has_pos = positive.any(dim=1)
        has_neg = negative.any(dim=1)
        valid = has_pos & has_neg
        if not bool(valid.any()):
            return embeddings.sum() * 0.0  # keep the graph so DDP stays happy

        lse_pos = torch.logsumexp(logits_pos[valid] / self.temperature, dim=1)
        lse_neg = torch.logsumexp(logits_neg[valid] / self.temperature, dim=1)
        return (F.softplus(lse_pos + lse_neg)).mean()


class BatchHardTripletLoss(nn.Module):
    """Batch-hard triplet loss with cosine distance.

    For every anchor it picks the hardest positive (lowest similarity among same-label
    samples) and the hardest negative (highest similarity among different-label samples),
    and requires the anchor to be closer to the positive by ``margin``.  Anchors lacking
    either partner are skipped.
    """

    def __init__(self, margin: float = 0.3):
        super().__init__()
        self.margin = margin

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        distance = 1.0 - _pairwise_cosine(embeddings)
        n = embeddings.shape[0]
        same = labels.unsqueeze(0) == labels.unsqueeze(1)
        eye = torch.eye(n, dtype=torch.bool, device=embeddings.device)
        positive = same & ~eye
        negative = ~same

        has_pos = positive.any(dim=1)
        has_neg = negative.any(dim=1)
        valid = has_pos & has_neg
        if not bool(valid.any()):
            return embeddings.sum() * 0.0

        big = torch.finfo(distance.dtype).max
        hardest_pos = torch.where(positive, distance, torch.full_like(distance, big)).min(dim=1).values
        hardest_neg = torch.where(negative, distance, torch.full_like(distance, -big)).max(dim=1).values
        losses = F.relu(hardest_pos - hardest_neg + self.margin)[valid]
        return losses.mean()


class CombinedLoss(nn.Module):
    """Weighted sum of :class:`CircleLoss` and :class:`BatchHardTripletLoss`."""

    def __init__(
        self,
        circle_weight: float = 1.0,
        triplet_weight: float = 0.5,
        margin: float = 0.25,
        gamma: float = 64.0,
        temperature: float = 0.1,
    ):
        super().__init__()
        self.circle_weight = circle_weight
        self.triplet_weight = triplet_weight
        self.circle = CircleLoss(margin=margin, gamma=gamma, temperature=temperature)
        self.triplet = BatchHardTripletLoss(margin=margin)

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor):
        """Return ``(total, components)`` so callers can log each term separately."""
        circle = self.circle(embeddings, labels)
        triplet = self.triplet(embeddings, labels)
        total = self.circle_weight * circle + self.triplet_weight * triplet
        return total, {"circle": float(circle.detach()), "triplet": float(triplet.detach())}
