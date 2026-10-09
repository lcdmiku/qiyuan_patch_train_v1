"""Profile the pieces of ncc_map on a refinement-sized window.

1141 ms for a 256x256 window with a 128x128 template is absurd: the arithmetic is ~2.4
GFLOP.  Something in the denominator path is selecting a pathological implementation.
"""
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from patchmatch.core.ncc import (
    _box_sum,
    _box_sum_sq,
    _corr_valid,
    _pad_bottom_right,
    _peak_multi_unfold,
    ncc_map,
    window_stats,
)

DEV = "cuda"
torch.manual_seed(0)
H = W = 256
T = 128
image64 = torch.randn(H, W, device=DEV, dtype=torch.float64)
image32 = image64.float()
template64 = torch.randn(T, T, device=DEV, dtype=torch.float64)
template32 = template64.float()


def timeit(label, fn, repeats=10):
    fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(repeats):
        fn()
    torch.cuda.synchronize()
    ms = (time.perf_counter() - start) / repeats * 1000
    print(f"  {label:56s} {ms:9.3f} ms", flush=True)
    return ms


print("window 256x256, template 128x128")
print("\nfloat64:")
timeit("_box_sum (conv 128x128 ones)", lambda: _box_sum(image64, T, T))
timeit("_box_sum_sq (grouped conv, 2 channels)", lambda: _box_sum_sq(image64, T, T))
timeit("window_stats", lambda: window_stats(image64, T, T))
timeit("_pad_bottom_right", lambda: _pad_bottom_right(image64, T, T))
padded = _pad_bottom_right(image64, T, T)
timeit("_corr_valid(padded, template)", lambda: _corr_valid(padded, template64))
timeit("ncc_map", lambda: ncc_map(image64, template64))

print("\nfloat32:")
timeit("_box_sum (conv 128x128 ones)", lambda: _box_sum(image32, T, T))
timeit("_box_sum_sq (grouped conv, 2 channels)", lambda: _box_sum_sq(image32, T, T))
timeit("window_stats", lambda: window_stats(image32, T, T))
timeit("ncc_map", lambda: ncc_map(image32, template32))

print("\n-- reference points --")
flat64 = torch.randn(H * W, T * T, device=DEV, dtype=torch.float64)
k64 = torch.randn(1, T * T, device=DEV, dtype=torch.float64)
timeit("matmul (4096 x 16384) @ (16384 x 1)", lambda: flat64 @ k64.t())
ones = torch.ones(1, 1, T, T, device=DEV, dtype=torch.float64)
timeit("conv2d directly with same shape/args",
       lambda: F.conv2d(_pad_bottom_right(image64, T, T).reshape(1, 1, H + T - 1, W + T - 1), ones))

print("\n-- same conv, different input sizes (float64) --")
for side in (64, 128, 256, 512):
    img = torch.randn(side, side, device=DEV, dtype=torch.float64)
    timeit(f"_corr_valid on {side}x{side} with 128x128 kernel",
           lambda i=img: _corr_valid(_pad_bottom_right(i, T, T), template64), repeats=5)

print("\n-- same conv, float32 --")
for side in (64, 128, 256, 512):
    img = torch.randn(side, side, device=DEV, dtype=torch.float32)
    timeit(f"_corr_valid on {side}x{side} with 128x128 kernel",
           lambda i=img: _corr_valid(_pad_bottom_right(i, T, T), template32), repeats=5)
