"""Profile one query end-to-end: where does the time actually go?

Run after any change to the cascade or the NCC core, since a per-image cost that looks
harmless at one level dominates once it is multiplied by the library size and the query
count.
"""
import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from patchmatch.core.imagebank import ImageBank, load_gray_f32, read_json
from patchmatch.core.ncc import ncc_map, ncc_peak_multi, window_stats
from patchmatch.grounding.locate import GroundingConfig, locate_patch
from patchmatch.retrieval.cascade import CascadeConfig, cascade_screen


def timed(label, fn, repeats=1):
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    out = None
    for _ in range(repeats):
        out = fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / repeats
    print(f"  {label:52s} {elapsed*1000:9.2f} ms", flush=True)
    return out, elapsed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="qiyuan_patch_train_v1")
    parser.add_argument("--split", default="val")
    parser.add_argument("--index", type=int, default=0, help="which patch to profile")
    parser.add_argument("--images", type=int, default=8, help="library images to scan")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    root = os.path.join(args.data, args.split)
    gt = read_json(os.path.join(root, "patch_gt.json"))
    patch_dir = os.path.join(root, "patches")
    image_dir = os.path.join(root, "images")
    names = sorted(os.listdir(image_dir))[: args.images]

    key = sorted(gt)[args.index]
    print(f"device={device}  patch={key}  images={len(names)}  truth={gt[key]['bbox']}")

    levels = (8, 4, 2, 1)
    bank = ImageBank(list(range(len(names))), [os.path.join(image_dir, n) for n in names],
                     coarse_levels=levels, device=device, full_cache=2)
    _, load_s = timed("bank.load_all()", lambda: bank.load_all())

    patch = torch.from_numpy(load_gray_f32(os.path.join(patch_dir, key))).to(device)
    print(f"  patch shape {tuple(patch.shape)}")

    # --- raw NCC at each level, one image
    print("\n-- single-image NCC by level (one image) --")
    for level in (1, 2, 4, 8):
        image = bank.coarse(0, level)
        template = patch[::level, ::level]
        if min(template.shape) < 4:
            continue
        _, seconds = timed(f"ncc_peak_multi level={level} image={tuple(image.shape)}",
                           lambda i=image, t=template: ncc_peak_multi(i, t.unsqueeze(0)), repeats=3)
        print(f"      -> {seconds/len(names)*1000:.1f} ms/image, "
              f"{seconds*len(names):.2f} s for a {len(names)}-image scan")

    print("\n-- component costs at level 8 --")
    image = bank.coarse(0, 8)
    template = patch[::8, ::8]
    timed("window_stats (float64)", lambda: window_stats(image, *template.shape), repeats=3)
    timed("window_stats (float32)",
          lambda: window_stats(image.float(), *template.shape), repeats=3)
    timed("ncc_map single template", lambda: ncc_map(image, template), repeats=3)
    timed("ncc_map single float32",
          lambda: ncc_map(image.float(), template.float()), repeats=3)

    print("\n-- cascade --")
    config = CascadeConfig(levels=(8, 4, 2), keep=(24, 8, 3))
    result, cascade_s = timed(f"cascade_screen ({len(names)} images)", 
                              lambda: cascade_screen(bank, patch, config))
    print(f"      survivors={result.survivors}")
    print(f"      truth rank in cascade: "
          f"{result.ranking.index(gt[key]['image']) + 1 if gt[key]['image'] in result.ranking else 'missing'}"
          if gt[key]["image"] in names else "      truth image not in this subset")

    print("\n-- localisation --")
    image_id = 0
    seed = result.seeds.get(image_id)
    grounded, ground_s = timed("locate_patch (with seed)",
                               lambda: locate_patch(bank, patch, image_id,
                                                    guess_y=seed[0] if seed else None,
                                                    guess_x=seed[1] if seed else None,
                                                    config=GroundingConfig(coarse_level=2)))
    print(f"      bbox={grounded.bbox} score={grounded.score:.4f} path={grounded.path}")

    print("\n-- extrapolation --")
    per_query = cascade_s + ground_s
    for n_queries in (100, 350):
        print(f"  {n_queries} queries x {len(names)} images: "
              f"{(per_query + load_s) * n_queries / 60:.1f} min at this scale "
              f"(linear in images/query, not a real estimate for a different library size)")


if __name__ == "__main__":
    main()
