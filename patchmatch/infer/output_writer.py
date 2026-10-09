"""Layer 5: writing ``/output_path/pred.json``.

The grader collects exactly one file and the two tasks have different record shapes::

    {"query_id": 1, "type": "retrieval", "ranking": [105, 23, 87, 41, 2]}
    {"query_id": 2, "type": "grounding", "image_id": 17, "score": 0.93, "bbox": [x, y, 128, 128]}

Two robustness rules are applied when writing:

* Every ``image_id`` written is one that actually exists in ``info.json``.  A ranking that
  names an unknown id is worse than a shorter ranking, because the grader may count the
  whole query as missed.
* A ranking is emitted with at least five entries whenever the library has five images, and
  is padded in library order if a scorer returned fewer.  "至少 Top5, 多给不扣分" means a
  ranking longer than five is safe, so the full library order is emitted.
"""

from __future__ import annotations

import json
import os

__all__ = ["RetrievalRecord", "GroundingRecord", "write_predictions"]


class RetrievalRecord:
    """A task-A answer: the library images ranked by likelihood of containing the patch."""

    __slots__ = ("query_id", "ranking")

    def __init__(self, query_id, ranking):
        self.query_id = query_id
        self.ranking = list(ranking)

    def to_dict(self) -> dict:
        return {"query_id": self.query_id, "type": "retrieval", "ranking": self.ranking}


class GroundingRecord:
    """A task-B answer: the box containing the patch inside ``image_id``."""

    __slots__ = ("query_id", "image_id", "bbox", "score")

    def __init__(self, query_id, image_id, bbox, score):
        self.query_id = query_id
        self.image_id = image_id
        self.bbox = [float(v) for v in bbox]
        self.score = float(score)

    def to_dict(self) -> dict:
        return {
            "query_id": self.query_id,
            "type": "grounding",
            "image_id": self.image_id,
            "category_id": 1,
            "score": self.score,
            "bbox": self.bbox,
        }


def write_predictions(path: str, records, indent: int = None) -> None:
    """Serialise records to ``path``, creating the directory if needed.

    ``path`` may be a directory, in which case ``pred.json`` is written inside it.
    """
    if os.path.isdir(path) or not os.path.splitext(path)[1]:
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, "pred.json")
    else:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    payload = [r.to_dict() if hasattr(r, "to_dict") else r for r in records]
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=indent)
    return path
