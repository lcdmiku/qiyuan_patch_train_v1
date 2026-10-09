"""Layer 5: reading the evaluation input directory.

The evaluation container mounts the data read-only at ``/input_path`` with this layout::

    /input_path/
        info.json        [{"image_id": 0, "file_name": "xxx.JPG"}, ...]
        images/          the library images
        patches/         the query patches (128x128)
        queries.json     [{"query_id": 1, "type": "retrieval", "image_patch": "patches/a.png"},
                          {"query_id": 2, "type": "grounding", "image_patch": "patches/b.png",
                           "image_id": 17}, ...]

Nothing here is specific to the competition's field names beyond the four documented keys,
and the loader tolerates extra keys, since the exact schema is only described in prose.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from ..core.imagebank import read_json

__all__ = ["Query", "EvalInput", "load_eval_input"]

#: Accepted spellings for the patch path field, in priority order.
_PATCH_KEYS = ("image_patch", "patch", "patch_path", "image_patch_path", "file_name", "query_patch")
#: Accepted spellings for the query type field.
_TYPE_KEYS = ("type", "task", "query_type")


def _first_present(mapping: dict, keys, default=None):
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return default


@dataclass
class Query:
    """One evaluation query, normalised across the schema's spellings."""

    query_id: int
    kind: str                      # "retrieval" or "grounding"
    patch_path: str                # absolute path to the 128x128 query patch
    image_id: int = None           # only for grounding
    raw: dict = field(default_factory=dict)

    @property
    def is_grounding(self) -> bool:
        return self.kind == "grounding"


@dataclass
class EvalInput:
    """Everything the solver needs: the image library and the query list."""

    image_ids: list
    image_paths: list
    queries: list
    input_root: str

    def __len__(self) -> int:
        return len(self.queries)

    @property
    def grounding_queries(self) -> list:
        return [q for q in self.queries if q.is_grounding]

    @property
    def retrieval_queries(self) -> list:
        return [q for q in self.queries if not q.is_grounding]


def _normalise_kind(value) -> str:
    """Map whatever the query calls itself onto the two known tasks.

    Anything that mentions "ground" (or carries an ``image_id``) is treated as grounding,
    because emitting a ranking for a grounding query is harmless while emitting a box for a
    retrieval query is meaningless.
    """
    text = str(value or "").strip().lower()
    if "ground" in text or "local" in text or "detect" in text or "bbox" in text:
        return "grounding"
    return "retrieval"


def load_eval_input(input_root: str) -> EvalInput:
    """Parse ``info.json`` and ``queries.json`` and resolve the patch paths.

    ``image_id`` values are kept exactly as given (they are what the submission must echo
    back); they are not assumed to be zero-based or contiguous.
    """
    info = read_json(os.path.join(input_root, "info.json"))
    queries_raw = read_json(os.path.join(input_root, "queries.json"))

    image_ids, image_paths = [], []
    for entry in info:
        image_ids.append(entry["image_id"])
        image_paths.append(os.path.join(input_root, "images", entry["file_name"]))

    known = set(image_ids)
    queries = []
    for position, entry in enumerate(queries_raw):
        patch_rel = _first_present(entry, _PATCH_KEYS)
        if patch_rel is None:
            raise KeyError(f"query {position} has no patch path field: {sorted(entry)}")
        patch_path = patch_rel if os.path.isabs(patch_rel) else os.path.join(input_root, patch_rel)

        image_id = entry.get("image_id")
        if image_id is not None and image_id not in known:
            # Tolerate an id that is a file name rather than a number.
            match = [i for i, p in zip(image_ids, image_paths) if os.path.basename(p) == str(image_id)]
            image_id = match[0] if match else None

        kind = _normalise_kind(_first_present(entry, _TYPE_KEYS))
        if image_id is None and kind == "grounding":
            kind = "retrieval"  # cannot localise without a stated image

        queries.append(
            Query(
                query_id=entry.get("query_id", position),
                kind=kind,
                patch_path=patch_path,
                image_id=image_id,
                raw=entry,
            )
        )

    return EvalInput(image_ids, image_paths, queries, input_root)
