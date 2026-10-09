"""Verify and benchmark the optimised multi-template NCC.

Correctness first: the ``unfold`` + GEMM path must agree with the convolution fallback and
with an explicit sliding-window reference.  Only then does the timing mean anything.

Timing second: the workload is launch-bound, so the measurement that matters is per-image
cost at coarse pyramid levels, multiplied by the library size.
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from patchmatch.core.ncc import (
    MIN_TILE,
    _available_workspace_bytes,
    _peak_multi_loop,
    _peak_multi_unfold,
    ncc_peak_multi,
    window_stats,
)

DEV = "cuda" if torch.cuda.is_available() else "cpu"
BUDGET = _available_workspace_bytes(DEV)


def reference_peak(image: np.ndarray, templates: np.ndarray) -> tuple:
    """Explicit sliding-window best NCC, float64, top-left aligned with zero padding."""
    height, width = image.shape
    k_count, th, tw = templates.shape
    padded = np.zeros((height + th - 1, width + tw - 1), np.float64)
    padded[:height, :width] = image
    best = np.full((height, width), -2.0)
    which = np.zeros((height, width), np.int32)
    for k in range(k_count):
        centred = templates[k].astype(np.float64)
        centred = centred - centred.mean()
        tnorm = np.linalg.norm(centred)
        for y in range(height):
            for x in range(width):
                window = padded[y : y + th, x : x + tw]
                wd = window - window.mean()
                wn = np.linalg.norm(wd)
                score = 0.0 if wn < 1e-9 else float((wd * centred).sum() / (wn * tnorm))
                if score > best[y, x]:
                    best[y, x] = score
                    which[y, x] = k
    return best, which


def check_correctness():
    rng = np.random.default_rng(2)
    height, width = 44, 37
    image = rng.normal(50, 20, (height, width)).astype(np.float32)
    templates = np.stack([
        image[3:15, 5:17],
        image[20:32, 25:37],
        rng.normal(50, 20, (12, 12)).astype(np.float32),
    ])

    ref_best, _ = reference_peak(image, templates)
    image_t = torch.from_numpy(image).to(DEV)
    templates_t = torch.from_numpy(templates).to(DEV)

    got_best, got_index = ncc_peak_multi(image_t, templates_t)
    got_best, got_index = got_best.cpu().numpy(), got_index.cpu().numpy()

    # Force each internal path explicitly.  This image is deliberately tiny: 44x37 = 1628
    # windows, far below MIN_TILE, so the tiling loop must clamp its final tile to exactly
    # what remains.  Getting that clamp wrong was a real bug (tile wider than the data).
    n = float(templates.shape[1] * templates.shape[2])
    _, sigma = window_stats(image_t, templates.shape[1], templates.shape[2])
    denom = (sigma * (n**0.5)).to(torch.float64)
    print(f"  total windows = {image.shape[0] * image.shape[1]} "
          f"(below MIN_TILE={MIN_TILE}, exercises final-tile clamping)")
    unfolded = _peak_multi_unfold(image_t.to(torch.float64), templates_t, denom,
                                  templates.shape[1], templates.shape[2], BUDGET)
    looped = _peak_multi_loop(image_t.to(torch.float64), templates_t, denom, n)

    unfolded_best = unfolded[0].cpu().numpy()
    looped_best = looped[0].cpu().numpy()
    print("-- correctness --")
    print(f"  auto   vs reference: {np.abs(got_best - ref_best).max():.3e}")
    print(f"  unfold vs reference: {np.abs(unfolded_best - ref_best).max():.3e}")
    print(f"  loop   vs reference: {np.abs(looped_best - ref_best).max():.3e}")
    print(f"  unfold vs loop     : {np.abs(unfolded_best - looped_best).max():.3e}")

    # Index agreement is only meaningful where the two paths are not tied.
    tied = np.abs(unfolded_best - looped_best) < 1e-6
    mismatch = int(((unfolded[1].cpu().numpy() != looped[1].cpu().numpy()) & ~tied).sum())
    print(f"  index mismatches unfold vs loop: {mismatch} (tied positions ignored)")

    err = max(
        np.abs(got_best - ref_best).max(),
        np.abs(unfolded_best - ref_best).max(),
        np.abs(looped_best - ref_best).max(),
    )
    assert err < 1e-4, err
    print("  CORRECTNESS OK")


def time_call(fn, repeats=5):
    for _ in range(1):
        fn()
    if DEV == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(repeats):
        fn()
    if DEV == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - start) / repeats * 1000


def bench(images: int = 154, patch_size: int = 128, k_templates: int = 1):
    """Per-image cost at each pyramid level, extrapolated to a full library scan."""
    print(f"\n-- timing (device={DEV}, {images} images/scan, K={k_templates}) --")
    print(f"{'level':>6} {'img shape':>13} {'tpl':>7} {'unfold ms':>10} {'loop ms':>9} {'scan s':>8} {'max err':>9}")
    rng = np.random.default_rng(0)
    for level in (16, 8, 4, 2, 1):
        height = 3024 // level
        width = 4032 // level
        if height < 32 or width < 32:
            continue
        # a small image keeps the benchmark fast at level 1 while preserving the ratio
        height, width = max(64, height // 8), max(64, width // 8)
        image = torch.from_numpy(rng.normal(128, 30, (height, width)).astype(np.float32)).to(DEV)
        templates = torch.from_numpy(
            rng.normal(128, 30, (k_templates, patch_size // level, patch_size // level)).astype(np.float32)
        ).to(DEV)
        if min(templates.shape[1], templates.shape[2]) < 4:
            continue

        unfold_ms = time_call(lambda: _peak_multi_unfold(
            image.to(torch.float64), templates, torch.ones(height, width, device=DEV, dtype=torch.float64),
            templates.shape[1], templates.shape[2], BUDGET))
        loop_ms = time_call(lambda: _peak_multi_loop(
            image.to(torch.float64), templates, torch.ones(height, width, device=DEV, dtype=torch.float64),
            float(templates.shape[1] * templates.shape[2])))

        # sanity: the two must agree (denominator is a dummy constant here, so use raw form)
        u = _peak_multi_unfold(image.to(torch.float64), templates,
                               torch.ones(height, width, device=DEV, dtype=torch.float64),
                               templates.shape[1], templates.shape[2], BUDGET)[0]
        l = _peak_multi_loop(image.to(torch.float64), templates,
                             torch.ones(height, width, device=DEV, dtype=torch.float64),
                             float(templates.shape[1] * templates.shape[2]))[0]
        err = (u - l).abs().max().item()

        pixels_ratio = (3024 * 4032 / level**2) / (height * width)
        scan_s = unfold_ms / 1000 * images * pixels_ratio
        print(f"{level:>6} {str((height,width)):>13} {str(tuple(templates.shape[1:])):>7} "
              f"{unfold_ms:>10.2f} {loop_ms:>9.2f} {scan_s:>8.1f} {err:>9.2e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=1)
    parser.add_argument("--images", type=int, default=154)
    parser.add_argument("--skip-correctness", action="store_true")
    args = parser.parse_args()
    if not args.skip_correctness:
        check_correctness()
    bench(images=args.images, k_templates=args.k)
