#!/usr/bin/env python3
"""Competition entry point: ``python /workspace/run.py``.

Reads the evaluation data from ``/input_path`` and writes ``/output_path/pred.json``.  Both
paths are overridable by environment variable or CLI flag so the same script can be run
against a local copy of the data during development.

Robustness rules, in order of importance:

1. **Always produce a valid ``pred.json``.**  The grader collects exactly one file, so a
   crash that writes nothing scores zero for every query.  A top-level guard therefore writes
   best-effort fallback records if the real pipeline fails partway.
2. **Never let one query sink the run.**  Each query is solved inside its own ``try``; a
   failure records a fallback answer (library order for retrieval, a centred box for
   grounding) and the run continues.
3. **Prefer completing the high-weight task.**  Grounding is 0.7 of the score, so the solver
   answers all grounding queries before any retrieval query, and a wall-clock budget is
   enforced: if time runs short, remaining retrieval queries are answered cheaply rather than
   risking an unwritten file.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import traceback

# Make the package importable no matter where the script is invoked from.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DEFAULT_INPUT = os.environ.get("INPUT_PATH", "/input_path")
DEFAULT_OUTPUT = os.environ.get("OUTPUT_PATH", "/output_path")

#: Wall-clock budget in seconds.  The container limit is 60 min; stop well short of it so the
#: output file is always written.  Override with ``TIME_BUDGET`` for local runs.
DEFAULT_BUDGET = float(os.environ.get("TIME_BUDGET", 50 * 60))


def log(message, **fields):
    """Print a timestamped progress line, flushed so it appears in streamed container logs."""
    stamp = time.strftime("%H:%M:%S")
    extra = "".join(f" {k}={v}" for k, v in fields.items())
    print(f"[{stamp}] {message}{extra}", flush=True)


class _OneQuery:
    """Minimal stand-in for an :class:`EvalInput` holding a single query.

    Lets the fallback builder be reused for one query without duplicating its record shapes.
    """

    def __init__(self, image_ids, query):
        self.image_ids = image_ids
        self.queries = [query]


def _fallback_records(data, patch_size=128, image_shape=None):
    """Minimal valid answers for every query, used if anything unrecoverable happens.

    Retrieval gets the library in ``info.json`` order (harmless: it satisfies the "at least
    Top5" requirement and costs nothing).  Grounding gets a centred box when the image size is
    known, since a centred guess has a real chance of overlapping the truth whereas a corner
    box has almost none.
    """
    records = []
    ids = list(data.image_ids)
    for query in data.queries:
        if query.is_grounding and query.image_id is not None:
            x, y = 0.0, 0.0
            if image_shape is not None:
                height, width = image_shape
                x = max(0.0, (width - patch_size) / 2.0)
                y = max(0.0, (height - patch_size) / 2.0)
            records.append({
                "query_id": query.query_id,
                "type": "grounding",
                "image_id": query.image_id,
                "category_id": 1,
                "score": 0.0,
                "bbox": [x, y, float(patch_size), float(patch_size)],
            })
        else:
            records.append({
                "query_id": query.query_id,
                "type": "retrieval",
                "ranking": ids[: max(5, len(ids))],
            })
    return records


def _write(path: str, records) -> str:
    """Serialise records to ``path`` (a file, or a directory in which case ``pred.json``).

    Delegates to the package's writer rather than re-implementing it: records are
    ``RetrievalRecord``/``GroundingRecord`` objects, and ``json.dump`` cannot serialise them
    directly.  Keeping one writer also means the fallback path and the real path cannot drift
    apart in the shape of what they emit.
    """
    from patchmatch.infer.output_writer import write_predictions

    return write_predictions(path, records)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Patch retrieval + grounding submission.")
    parser.add_argument("--input", default=DEFAULT_INPUT, help="evaluation input directory")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="pred.json path or directory")
    parser.add_argument("--budget", type=float, default=DEFAULT_BUDGET,
                        help="wall-clock budget in seconds")
    parser.add_argument("--device", default=None, help="torch device (default: cuda if present)")
    parser.add_argument("--cascade-levels", default=None,
                        help="comma-separated coarse-to-fine levels, e.g. 16,8,4")
    parser.add_argument("--cascade-keep", default=None,
                        help="comma-separated survivor counts, e.g. 154,24,8")
    parser.add_argument("--verify-top", type=int, default=2,
                        help="candidates re-localised at full resolution for the ranking")
    parser.add_argument("--window-radius", type=int, default=64)
    parser.add_argument("--limit", type=int, default=0, help="cap queries (debugging only)")
    args = parser.parse_args(argv)

    started = time.time()
    log("starting", input=args.input, output=args.output, budget_s=int(args.budget))

    # Import here so a failure in torch/cuda setup is reportable rather than silent.
    try:
        import torch

        from patchmatch.core.imagebank import ImageBank, load_gray_f32
        from patchmatch.eval.harness import iou  # noqa: F401  (kept importable for diagnostics)
        from patchmatch.grounding.locate import GroundingConfig, locate_patch
        from patchmatch.infer.input_reader import load_eval_input
        from patchmatch.infer.output_writer import GroundingRecord, RetrievalRecord
        from patchmatch.retrieval.cascade import CascadeConfig, cascade_screen
    except Exception:
        traceback.print_exc()
        log("FATAL: imports failed")
        return 1

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    log("environment", torch=torch.__version__, device=device,
        cuda=torch.version.cuda if torch.cuda.is_available() else "none")

    # ------------------------------------------------------------------ read input
    try:
        data = load_eval_input(args.input)
    except Exception:
        traceback.print_exc()
        log("FATAL: could not read input directory")
        return 1

    if args.limit:
        data.queries = data.queries[: args.limit]
    log("input loaded", images=len(data.image_ids), queries=len(data.queries),
        grounding=len(data.grounding_queries), retrieval=len(data.retrieval_queries))

    # ------------------------------------------------------------------ configure
    cascade = CascadeConfig()
    if args.cascade_levels:
        cascade.levels = tuple(int(x) for x in args.cascade_levels.split(","))
    if args.cascade_keep:
        cascade.keep = tuple(int(x) for x in args.cascade_keep.split(","))
    if len(cascade.levels) != len(cascade.keep):
        raise SystemExit("cascade levels and keep must have equal length")
    grounding = GroundingConfig(window_radius=args.window_radius)

    # ------------------------------------------------------------------ load library
    levels = tuple(sorted(set(cascade.levels) | {grounding.coarse_level, grounding.wide_level, 1},
                          reverse=True))
    bank = ImageBank(data.image_ids, data.image_paths, coarse_levels=levels,
                     device=device, full_cache=2)
    try:
        bank.load_all(progress=lambda done, total: log("decoding", done=done, total=total)
                      if done % 50 == 0 else None)
    except Exception:
        traceback.print_exc()
        log("library decode failed; writing fallback answers")
        path = _write(args.output, _fallback_records(data))
        log("wrote fallback", path=path, records=len(data.queries))
        return 0
    log("library ready", seconds=round(time.time() - started))

    # ------------------------------------------------------------------ solve
    def solve_one(query):
        """Answer one query; returns ``(retrieval_record, grounding_record_or_None)``."""
        patch = torch.from_numpy(load_gray_f32(query.patch_path)).to(device)
        result = cascade_screen(bank, patch, cascade)
        scores = dict(result.scores)
        located = {}
        for candidate in result.survivors[: max(1, args.verify_top)]:
            seed = result.seeds.get(candidate)
            hit = locate_patch(bank, patch, candidate,
                               guess_y=seed[0] if seed else None,
                               guess_x=seed[1] if seed else None, config=grounding)
            located[candidate] = hit
            scores[candidate] = hit.score

        ranking = sorted(located, key=lambda i: -scores[i])
        ranking += [i for i in result.ranking if i not in located]
        seen = set(ranking)
        ranking += [i for i in data.image_ids if i not in seen]

        grounding_record = None
        if query.is_grounding and query.image_id is not None:
            hit = located.get(query.image_id)
            if hit is None:
                seed = result.seeds.get(query.image_id)
                hit = locate_patch(bank, patch, query.image_id,
                                   guess_y=seed[0] if seed else None,
                                   guess_x=seed[1] if seed else None, config=grounding)
            height, width = bank.shape(query.image_id)
            x = min(max(hit.bbox[0], 0.0), max(0.0, width - patch.shape[1]))
            y = min(max(hit.bbox[1], 0.0), max(0.0, height - patch.shape[0]))
            bbox = [x, y, float(patch.shape[1]), float(patch.shape[0])]
            grounding_record = GroundingRecord(query.query_id, query.image_id, bbox, hit.score)
            ranking = [query.image_id] + [i for i in ranking if i != query.image_id]

        return RetrievalRecord(query.query_id, ranking), grounding_record

    records = []
    failed = 0
    grounding_queries = data.grounding_queries
    retrieval_queries = data.retrieval_queries

    def fallback_for(query):
        """Best-effort answer for one query whose real solve raised."""
        shape = None
        if getattr(query, "image_id", None) is not None:
            try:
                shape = bank.shape(query.image_id)
            except Exception:
                shape = None
        return _fallback_records(
            _OneQuery(data.image_ids, query), image_shape=shape
        )

    # Grounding first: it carries 0.7 of the score, so it must complete even if time runs out.
    for index, query in enumerate(grounding_queries, start=1):
        try:
            retrieval, ground = solve_one(query)
            records.append(retrieval)
            if ground is not None:
                records.append(ground)
            bank.drop_full(query.image_id)
        except Exception:
            failed += 1
            traceback.print_exc()
            records.extend(fallback_for(query))
        if index % 20 == 0 or index == len(grounding_queries):
            log("grounding", done=index, total=len(grounding_queries), failed=failed,
                elapsed=round(time.time() - started))

    # Then retrieval, with the remaining budget respected.
    for index, query in enumerate(retrieval_queries, start=1):
        if time.time() - started > args.budget:
            log("budget reached; answering remaining retrieval queries cheaply",
                remaining=len(retrieval_queries) - index + 1)
            for rest in retrieval_queries[index - 1:]:
                records.extend(fallback_for(rest))
            break
        try:
            retrieval, _ = solve_one(query)
            records.append(retrieval)
        except Exception:
            failed += 1
            traceback.print_exc()
            records.extend(fallback_for(query))
        if index % 10 == 0 or index == len(retrieval_queries):
            log("retrieval", done=index, total=len(retrieval_queries), failed=failed,
                elapsed=round(time.time() - started))

    # ------------------------------------------------------------------ write output
    # Records are emitted per query in the input's order, which the grader does not require
    # but does make a failed run far easier to diff against the ground truth.  Fallback
    # records are plain dicts while normal ones are record objects, so the sort key is read
    # through a helper rather than by attribute access.
    def _query_id(record):
        return record["query_id"] if isinstance(record, dict) else record.query_id

    order = {q.query_id: n for n, q in enumerate(data.queries)}
    records.sort(key=lambda r: order.get(_query_id(r), 1 << 30))
    path = _write(args.output, records)
    log("done", path=path, records=len(records), failed=failed,
        total_seconds=round(time.time() - started))
    return 0


if __name__ == "__main__":
    sys.exit(main())
