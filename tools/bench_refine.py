"""How much does float64 actually cost in the refinement path, and is it needed there?

The refinement window is small (patch + 2*radius), so this measures the accuracy/speed
trade of the accumulation dtype on the exact operation refinement performs.  The governing
fact from earlier measurements: at full resolution float32 stats picked a different arg-max
than float64 for ~17% of patches, so if float64 is dropped here it must be shown to still
land on the exact pixel.
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from patchmatch.core.imagebank import ImageBank, load_gray_f32, read_json
from patchmatch.core.ncc import ncc_map
from patchmatch.grounding import locate as loc

DEV = "cuda"
ROOT = os.path.join("qiyuan_patch_train_v1", "val")


def time_call(fn, repeats=5):
    fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(repeats):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / repeats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=40)
    args = parser.parse_args()

    gt = read_json(os.path.join(ROOT, "patch_gt.json"))
    names = sorted(os.listdir(os.path.join(ROOT, "images")))
    bank = ImageBank(list(range(len(names))), [os.path.join(ROOT, "images", n) for n in names],
                     coarse_levels=(16, 8, 4, 2, 1), device=DEV, full_cache=2)
    bank.load_all()

    keys = sorted(gt)[: args.limit]
    print(f"{'dtype':>8} {'exact hits':>11} {'mean score':>11} {'ms/call':>9}   (window 256x256, patch 128x128)")
    for dtype in (torch.float64, torch.float32):
        hits = 0
        scores = []
        for key in keys:
            tx, ty = gt[key]["bbox"][0], gt[key]["bbox"][1]
            image_id = names.index(gt[key]["image"])
            patch = torch.from_numpy(load_gray_f32(os.path.join(ROOT, "patches", key))).to(DEV)
            # seed deliberately offset by 2 px, as a coarse guess would be
            score, x, y = loc.refine_in_window(bank, image_id, patch, ty + 2, tx - 2, 64, dtype=dtype)
            hits += (x == tx and y == ty)
            scores.append(score)
        patch = torch.from_numpy(load_gray_f32(os.path.join(ROOT, "patches", keys[0]))).to(DEV)
        image_id = names.index(gt[keys[0]]["image"])
        tx, ty = gt[keys[0]]["bbox"][0], gt[keys[0]]["bbox"][1]
        ms = time_call(lambda: loc.refine_in_window(bank, image_id, patch, ty, tx, 64, dtype=dtype)) * 1000
        print(f"{str(dtype).replace('torch.',''):>8} {hits:>6}/{len(keys):<4} "
              f"{np.mean(scores):>11.6f} {ms:>9.1f}")

    print()
    print("-- where does the time go inside refine_in_window (float64, 256x256) --")
    patch = torch.from_numpy(load_gray_f32(os.path.join(ROOT, "patches", keys[0]))).to(DEV)
    image_id = names.index(gt[keys[0]]["image"])
    tx, ty = gt[keys[0]]["bbox"][0], gt[keys[0]]["bbox"][1]
    window = loc._read_window(bank, image_id, tx - 64, ty - 64, 256, 256)
    print(f"  _read_window (host crop + H2D): {time_call(lambda: loc._read_window(bank, image_id, tx-64, ty-64, 256, 256))*1000:8.2f} ms")
    print(f"  .to(float64)                  : {time_call(lambda: window.to(torch.float64))*1000:8.2f} ms")
    w64 = window.to(torch.float64)
    p64 = patch.to(torch.float64)
    print(f"  ncc_map float64               : {time_call(lambda: ncc_map(w64, p64))*1000:8.2f} ms")
    w32, p32 = window.to(torch.float32), patch.to(torch.float32)
    print(f"  ncc_map float32               : {time_call(lambda: ncc_map(w32, p32))*1000:8.2f} ms")


if __name__ == "__main__":
    main()
