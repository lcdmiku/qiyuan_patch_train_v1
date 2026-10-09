"""Layer 1: bbox geometry and IoU.

Pure-Python, no torch/cv2 dependency, so it can be unit-tested and reused anywhere.

The competition fixes the patch size at 128x128, so every predicted box is
``[x, y, 128, 128]``.  The functions here stay general anyway because the coarse
pyramid levels predict boxes at a reduced scale, and those have to be mapped back to
full resolution.
"""

from __future__ import annotations

from typing import Iterable, Sequence

__all__ = [
    "xywh_to_xyxy",
    "xyxy_to_xywh",
    "iou",
    "clip_bbox",
    "scale_bbox",
    "BBox",
]

BBox = Sequence[float]  # [x, y, w, h]


def xywh_to_xyxy(box: BBox) -> tuple[float, float, float, float]:
    """Convert ``[x, y, w, h]`` to ``(x1, y1, x2, y2)``.

    The competition format is ``[x, y, w, h]`` with the top-left corner, so the second
    pair is obtained by adding width/height rather than by reading stored coordinates.
    """
    x, y, w, h = box
    return float(x), float(y), float(x) + float(w), float(y) + float(h)


def xyxy_to_xywh(box: Iterable[float]) -> list[float]:
    """Convert ``(x1, y1, x2, y2)`` to the competition's ``[x, y, w, h]`` format."""
    x1, y1, x2, y2 = box
    return [float(x1), float(y1), float(x2) - float(x1), float(y2) - float(y1)]


def iou(a: BBox, b: BBox) -> float:
    """Intersection-over-union of two ``[x, y, w, h]`` boxes.

    Returns 0.0 when the boxes merely touch: touching rectangles have zero intersection
    area, which the ``max(0, ...)`` guards and the competition's "刚好接触=0" note
    confirms.  Identical boxes give exactly 1.0.
    """
    ax1, ay1, ax2, ay2 = xywh_to_xyxy(a)
    bx1, by1, bx2, by2 = xywh_to_xyxy(b)
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


def clip_bbox(box: BBox, width: int, height: int) -> list[float]:
    """Keep a box inside ``[0, width] x [0, height]`` without changing its size.

    The top-left corner is moved rather than the size being shrunk, because the grader
    compares against a fixed 128x128 truth box: shrinking would guarantee a bad IoU,
    whereas sliding keeps the chance of overlap.  If the image is smaller than the box,
    the corner is pinned to 0 and the size is preserved.
    """
    x, y, w, h = (float(v) for v in box)
    max_x = max(0.0, width - w)
    max_y = max(0.0, height - h)
    return [min(max(x, 0.0), max_x), min(max(y, 0.0), max_y), w, h]


def scale_bbox(box: BBox, factor: float) -> list[float]:
    """Multiply every field of a box by ``factor`` (pyramid level -> full resolution)."""
    return [float(v) * factor for v in box]
