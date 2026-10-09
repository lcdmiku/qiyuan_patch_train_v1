"""Validation harness: score the pipeline on a labelled split with the competition metric.

For each patch in the split the harness answers both tasks and computes

    total = 0.1 * R@1 + 0.2 * R@5 + 0.6 * Acc@0.5 + 0.1 * Acc@0.75

Each patch is solved once and both tasks are scored from that single solve, mirroring how
the real submission shares work between retrieval and grounding.  The harness also reports
*where* the boxes came from (the ``path`` field of each result) and the mean IoU, since a
score alone does not say whether failures are coarse-search misses or refinement misses.

Usage::

    python -m patchmatch.eval.harness --data qiyuan_patch_train_v1 --split val
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch

from ..core.geometry import iou
from ..core.imagebank import ImageBank, load_gray_f32, read_json
from ..grounding.locate import GroundingConfig, locate_patch
from ..retrieval.cascade import CascadeConfig, cascade_screen

__all__ = ["evaluate", "main"]


def _summarise(values) -> dict:
    array = np.asarray(values, dtype=np.float64)
    if not array.size:
        return {}
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p90": float(np.percentile(array, 90)),
        "worst": float(array.min()),
    }


def evaluate(
    data_root: str,
    split: str = "val",
    limit: int = 0,
    cascade: CascadeConfig = None,
    grounding: GroundingConfig = None,
    device: str = None,
    verify_top: int = 2,
    profile: bool = False,
    log=print,
) -> dict:
    """Run the full pipeline over one split and return the metric breakdown.

    ``limit`` caps the number of patches for a quick iteration.  Patches are processed in
    source-image order so each image's full-resolution pixels are decoded once.
    """
    cascade = cascade or CascadeConfig()
    grounding = grounding or GroundingConfig()
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    root = os.path.join(data_root, split)
    gt = read_json(os.path.join(root, "patch_gt.json"))
    image_dir = os.path.join(root, "images")
    patch_dir = os.path.join(root, "patches")
    image_names = sorted(os.listdir(image_dir))

    keys = sorted(gt)
    if limit:
        keys = keys[:limit]

    levels = tuple(sorted(set(cascade.levels) | {grounding.coarse_level, 1}, reverse=True))
    bank = ImageBank(
        list(range(len(image_names))),
        [os.path.join(image_dir, n) for n in image_names],
        coarse_levels=levels, device=device, full_cache=2,
    )
    id_by_name = {name: i for i, name in enumerate(image_names)}

    started = time.time()
    bank.load_all()
    load_seconds = time.time() - started
    log(f"device={device} split={split} images={len(image_names)} patches={len(keys)} "
        f"levels={levels}")
    log(f"decode+pyramid: {load_seconds:.1f}s ({load_seconds/len(image_names)*1000:.0f} ms/image)")

    by_image: dict = {}
    for key in keys:
        by_image.setdefault(gt[key]["image"], []).append(key)

    hits1 = hits5 = 0
    acc50 = acc75 = 0
    total = 0
    ious: list = []
    paths: dict = {}
    wrong_image = 0
    started = time.time()

    for done_image, (image_name, group) in enumerate(by_image.items(), start=1):
        image_id = id_by_name[image_name]
        for key in group:
            query_started = time.perf_counter()
            truth = gt[key]["bbox"]
            patch = torch.from_numpy(load_gray_f32(os.path.join(patch_dir, key))).to(device)

            cascade_started = time.perf_counter()
            stage_times = []
            result = cascade_screen(
                bank, patch, cascade,
                on_stage=lambda level, count, secs: stage_times.append((level, count, secs)),
            )
            cascade_seconds = time.perf_counter() - cascade_started
            scores = dict(result.scores)
            located: dict = {}
            for survivor in result.survivors[: max(1, verify_top)]:
                seed = result.seeds.get(survivor)
                hit = locate_patch(
                    bank, patch, survivor,
                    guess_y=seed[0] if seed else None,
                    guess_x=seed[1] if seed else None,
                    config=grounding,
                )
                located[survivor] = hit
                scores[survivor] = hit.score

            # ranking: verified first by full-resolution score, then cascade order
            ranking = sorted(located, key=lambda i: -scores[i])
            ranking += [i for i in result.ranking if i not in located]
            ranking += [i for i in range(len(image_names)) if i not in set(ranking)]

            # grounding: reuse the verification when the asserted image was a survivor
            hit = located.get(image_id)
            if hit is None:
                seed = result.seeds.get(image_id)
                hit = locate_patch(
                    bank, patch, image_id,
                    guess_y=seed[0] if seed else None,
                    guess_x=seed[1] if seed else None,
                    config=grounding,
                )
            paths[hit.path] = paths.get(hit.path, 0) + 1
            if hit.image_id != image_id:
                wrong_image += 1

            rank = ranking.index(image_id) + 1
            overlap = iou(hit.bbox, truth)
            ious.append(overlap)
            if rank > 1:
                # Diagnose ranking misses: which images outranked the truth, and by how much.
                ahead = [
                    (i, round(scores.get(i, float("nan")), 4), "verified" if i in located else "coarse")
                    for i in ranking[: max(5, rank)]
                    if i != image_id
                ]
                log(f"    RANK MISS {key:28s} rank={rank:3d} truth_id={image_id} "
                    f"cascade_pos={result.ranking.index(image_id) + 1 if image_id in result.ranking else -1} "
                    f"survivors={result.survivors} ahead={ahead}", flush=True)

            total += 1
            hits1 += rank == 1
            hits5 += rank <= 5
            acc50 += overlap >= 0.5
            acc75 += overlap >= 0.75

            if profile:
                elapsed = time.perf_counter() - query_started
                stages = " ".join(f"ds{lvl}x{n}={s:.2f}s" for lvl, n, s in stage_times)
                log(f"    {key:34s} {elapsed:6.2f}s cascade={cascade_seconds:6.2f}s "
                    f"[{stages}] path={hit.path:12s} rank={rank:3d} iou={overlap:.3f}", flush=True)

            if total % 10 == 0 or total == len(keys):
                log(f"  [{done_image}/{len(by_image)}] {total}/{len(keys)}  "
                    f"R@1={hits1/total:.3f} R@5={hits5/total:.3f} "
                    f"Acc@.5={acc50/total:.3f} Acc@.75={acc75/total:.3f} "
                    f"({time.time()-started:.0f}s)", flush=True)
        bank.drop_full(image_id)

    run_seconds = time.time() - started
    n = max(1, total)
    r1, r5 = hits1 / n, hits5 / n
    a50, a75 = acc50 / n, acc75 / n
    score = 0.1 * r1 + 0.2 * r5 + 0.6 * a50 + 0.1 * a75

    summary = {
        "split": split,
        "patches": total,
        "images": len(image_names),
        "R@1": r1,
        "R@5": r5,
        "Acc@0.5": a50,
        "Acc@0.75": a75,
        "score": score,
        "exact_iou_fraction": float(np.mean(np.asarray(ious) == 1.0)),
        "iou": _summarise(ious),
        "grounding_paths": paths,
        "wrong_image": wrong_image,
        "seconds": {"load": load_seconds, "run": run_seconds, "total": load_seconds + run_seconds},
        "ms_per_patch": run_seconds / n * 1000,
    }

    log("")
    log(f"=== {split}: {total} patches over {len(image_names)} images ===")
    log(f"R@1      = {r1:.4f}")
    log(f"R@5      = {r5:.4f}")
    log(f"Acc@0.5  = {a50:.4f}")
    log(f"Acc@0.75 = {a75:.4f}")
    log(f"SCORE    = {score:.4f}")
    log(f"mean IoU = {summary['iou']['mean']:.4f}   exactly 1.0: "
        f"{summary['exact_iou_fraction']*100:.1f}%   worst: {summary['iou']['worst']:.4f}")
    log(f"localisation paths: {paths}")
    log(f"answered a different image than truth: {wrong_image}/{n}")
    log(f"runtime: load {load_seconds:.1f}s + run {run_seconds:.1f}s = "
        f"{load_seconds + run_seconds:.1f}s ({summary['ms_per_patch']:.0f} ms/patch)")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description="Score the pipeline on a labelled split.")
    parser.add_argument("--data", default="qiyuan_patch_train_v1")
    parser.add_argument("--split", default="val")
    parser.add_argument("--limit", type=int, default=0)
    # Defaults are read from the config objects rather than repeated as literals.  Repeating
    # them is how this harness silently ran a cascade ending at ds=2 (102 s per query) while
    # the config said ds=4 (1.6 s): the literal default always overrode the tuned one.
    _cascade_defaults = CascadeConfig()
    _grounding_defaults = GroundingConfig()
    parser.add_argument("--levels", default=",".join(str(x) for x in _cascade_defaults.levels),
                        help="cascade levels, coarse to fine")
    parser.add_argument("--keep", default=",".join(str(x) for x in _cascade_defaults.keep),
                        help="survivors kept per cascade level")
    parser.add_argument("--coarse-level", type=int, default=_grounding_defaults.coarse_level)
    parser.add_argument("--window-radius", type=int, default=_grounding_defaults.window_radius)
    parser.add_argument("--verify-top", type=int, default=2)
    parser.add_argument("--profile", action="store_true",
                        help="log per-query timing and localisation path")
    parser.add_argument("--device", default=None)
    parser.add_argument("--out", default="")
    args = parser.parse_args(argv)

    summary = evaluate(
        args.data, args.split, limit=args.limit,
        cascade=CascadeConfig(
            levels=tuple(int(x) for x in args.levels.split(",")),
            keep=tuple(int(x) for x in args.keep.split(",")),
        ),
        grounding=GroundingConfig(
            coarse_level=args.coarse_level, window_radius=args.window_radius
        ),
        device=args.device, verify_top=args.verify_top, profile=args.profile,
    )
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
        print(f"summary -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
