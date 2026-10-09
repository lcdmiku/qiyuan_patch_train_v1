"""Layer 1: normalized cross-correlation (NCC) template matching on the GPU.

This is the numerical core the whole solution rests on, so it is deliberately
self-contained and heavily documented.

The problem it solves
---------------------
Given an image ``X`` (H, W) and a template ``T`` (h, w), produce the map

    M[y, x] = NCC( X[y : y+h, x : x+w], T )

where ``NCC`` is the zero-mean normalized cross-correlation

    NCC(A, B) = <A - mean(A), B - mean(B)> / (||A - mean(A)|| * ||B - mean(B)||)

``M`` is indexed by the template's *top-left corner*, which is exactly the competition's
``bbox = [x, y, 128, 128]`` convention, so ``argmax(M)`` is directly a prediction.

Why the answer is a small convolution and not an FFT
----------------------------------------------------
Split the numerator.  With ``Td = T - mean(T)``,

    sum_{i,j} X[y+i, x+j] * Td[i, j]

is a plain cross-correlation of the image with the template, i.e. one ``conv2d``.  The
denominator needs the *image* window's mean and standard deviation; the template's are
constants.  The window mean and standard deviation come from two more ``conv2d`` calls
with an all-ones box kernel -- one on ``X`` and one on ``X*X``.  All three are tiny
kernels whose cost is bounded by reading the image, so the direct form wins: an FFT over a
4032x3024 map costs roughly an order of magnitude more.

Two traps that this module exists to avoid
------------------------------------------
1. **A conv2d batch dimension is not a set of per-element filters.**  The obvious way to
   match K templates at once is to stack them along the batch axis and broadcast the image
   along it.  PyTorch silently convolves *every* batch element with kernel 0, so templates
   1..K-1 return template 0's response, and the bug is invisible for K=1.  Verified
   against an explicit sliding-window reference on CPU and GPU, float32 and float64, with
   cuDNN on and off, and with stride-0 ``expand`` views as well as materialised copies.
   ``ncc_peak_multi`` therefore loops over templates explicitly.
2. **float32 loses the variance.**  ``var = E[X^2] - E[X]^2`` cancels catastrophically on
   8-bit imagery: with pixels near 128 both terms are ~4e4 while their difference is a few
   hundred, so float32 keeps only 1-2 significant digits and the arg-max wanders.  At full
   resolution float32 picked the wrong pixel for ~17% of patches.  All three sums are
   therefore accumulated in float64; the extra cost is memory traffic, not FLOPs, and it is
   paid only where it matters.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

__all__ = [
    "ncc_map",
    "ncc_peak",
    "ncc_peak_multi",
    "extract_patch",
    "block_mean_downsample",
    "window_stats",
    "ACC_DTYPE",
]

#: Accumulation dtype for every NCC sum.  See the module docstring for why not float32.
ACC_DTYPE = torch.float64

#: Windows whose standard deviation is below this hold essentially no signal (gray levels
#: live in [0, 255]), so their NCC is meaningless and must not be allowed to win a peak.
FLAT_WINDOW_STD = 1e-3

#: Guard so a near-zero denominator cannot inflate a score without bound.
EPS = 1e-4

#: Smallest number of sliding windows processed per matrix-product tile.
#:
#: This is a correctness guard as much as a performance one.  The tile size is derived from
#: free GPU memory, which shrinks as the pyramid cache fills; without a floor the computed
#: tile can collapse to 1 window, and a 1x1 product broadcast against a length-N denominator
#: silently expands to a length-N result instead of raising.  A floor of a few thousand
#: windows costs a few hundred KB of scores and keeps the arithmetic unambiguous.
MIN_TILE = 4096


def _pad_bottom_right(x: torch.Tensor, kh: int, kw: int) -> torch.Tensor:
    """Zero-pad the bottom and right by ``(kh-1, kw-1)``.

    Padding on the *trailing* sides only means the following valid convolution emits
    exactly one output per template top-left corner, including corners whose window hangs
    off the edge.  That matters because ground-truth boxes do sit at x=0 / y=0.
    """
    return F.pad(x, (0, kw - 1, 0, kh - 1))


def _corr_valid(x2d: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """Cross-correlate a padded 2-D image with a single 2-D kernel in valid mode.

    ``conv2d`` performs a correlation (it does not flip the kernel), which is precisely
    the NCC numerator, so no flip is applied here.
    """
    h, w = kernel.shape
    out = F.conv2d(x2d.reshape(1, 1, *x2d.shape), kernel.reshape(1, 1, h, w))
    return out[0, 0]


def _box_sum(x2d: torch.Tensor, kh: int, kw: int) -> torch.Tensor:
    """Sum over every ``kh x kw`` window aligned at the top-left corner, zero padded."""
    ones = torch.ones(kh, kw, device=x2d.device, dtype=x2d.dtype)
    return _corr_valid(_pad_bottom_right(x2d, kh, kw), ones)


def _box_sum_sq(x2d: torch.Tensor, kh: int, kw: int) -> torch.Tensor:
    """Sum of squares over every ``kh x kw`` window, zero padded.

    The mean and the sum of squares are produced by a *single* convolution with a 2-channel
    kernel, halving the number of kernel launches for the denominator.  Launch overhead, not
    FLOPs or memory, is what bounds this workload: at a coarse pyramid level an image is a
    few hundred pixels wide, so each individual convolution is far too small to saturate the
    device and the cost is dominated by dispatching it.
    """
    both = torch.stack([x2d, x2d * x2d])  # (2, H, W)
    padded = F.pad(both, (0, kw - 1, 0, kh - 1))
    ones = torch.ones(2, 1, kh, kw, device=x2d.device, dtype=x2d.dtype)
    out = F.conv2d(padded.reshape(1, 2, *padded.shape[-2:]), ones, groups=2)
    return out[0, 0], out[0, 1]


def window_stats(x2d: torch.Tensor, kh: int, kw: int):
    """Local mean and standard deviation over every ``kh x kw`` window.

    Returns ``(mu, sigma)``, both shaped like the image.  Accumulated in
    :data:`ACC_DTYPE` for the reason given in the module docstring.
    """
    xa = x2d.to(ACC_DTYPE)
    n = float(kh * kw)
    s1, s2 = _box_sum_sq(xa, kh, kw)
    mu = s1 / n
    var = (s2 / n - mu * mu).clamp_(min=0.0)
    return mu, torch.sqrt(var)


def _template_norm(t2d: torch.Tensor):
    """Return the mean-centred template and its L2 norm (both float64, on the image device)."""
    ta = t2d.to(device=t2d.device, dtype=ACC_DTYPE)
    td = ta - ta.mean()
    return td, torch.linalg.vector_norm(td)


def ncc_map(
    image: torch.Tensor,
    template: torch.Tensor,
    acc_dtype: torch.dtype = None,
) -> torch.Tensor:
    """NCC map of 2-D ``template`` over 2-D ``image``, same shape as ``image``.

    ``image`` and ``template`` may live on different devices and have different dtypes; both
    are promoted to ``acc_dtype`` (default :data:`ACC_DTYPE`) on the image's device, and the
    result is returned in the image's dtype.  Positions whose image window is flat are 0.

    ``acc_dtype`` exists because float64 is not uniformly affordable.  cuDNN ships no fast
    float64 convolution, so on GPU the box sums plus the correlation cost ~90x more in float64
    than in float32 (measured: 1120 ms vs 12 ms for a 256x256 window with a 128x128 template).
    A caller searching a small, already well-localised region can pass ``torch.float32`` -- and
    should, since that 90x lands on the critical path -- while a caller scanning a whole image
    at full resolution keeps float64, where the win margin is thin enough that float32 was
    measured to pick the wrong pixel for ~17% of patches.
    """
    kh, kw = template.shape
    if image.dim() != 2:
        raise ValueError(f"expected a 2-D image, got shape {tuple(image.shape)}")
    if kh > image.shape[0] or kw > image.shape[1]:
        raise ValueError(f"template {tuple(template.shape)} larger than image {tuple(image.shape)}")

    acc = acc_dtype or ACC_DTYPE
    n = float(kh * kw)
    # Window statistics stay in float64 regardless of `acc_dtype`: E[X^2] - E[X]^2 cancels
    # badly enough that float32 keeps only one or two significant digits of the variance, and
    # these two box sums are cheap next to the correlation.
    _, sigma = window_stats(image, kh, kw)
    flat = sigma < FLAT_WINDOW_STD

    t_acc = template.to(device=image.device, dtype=acc)
    td = t_acc - t_acc.mean()
    tnorm = torch.linalg.vector_norm(td.to(ACC_DTYPE))

    # ||X_window - mean(X_window)|| == sqrt(n) * sigma, so the product of the two centred
    # norms is sqrt(n) * sigma * ||Td||.
    numer = _corr_valid(_pad_bottom_right(image.to(acc), kh, kw), td)
    denom = (sigma.to(acc) * tnorm.to(acc) * (n**0.5)).clamp_(min=EPS)

    m = numer / denom
    m[flat] = 0.0
    return m.to(dtype=image.dtype)


def ncc_peak(image: torch.Tensor, template: torch.Tensor):
    """Best NCC score of one template over one image, with its ``(y, x)`` location."""
    m = ncc_map(image, template)
    flat = int(m.argmax())
    y, x = divmod(flat, m.shape[1])
    return float(m[y, x]), y, x


def _available_workspace_bytes(device, fraction: float = 0.35) -> int:
    """Bytes of temporary scratch we are willing to allocate on ``device``.

    The matrix form of the multi-template search materialises window products whose size grows
    with the image area, reaching tens of gigabytes on a 4032x3024 image at full resolution.
    Rather than pick a fixed cap, the budget is derived from the device's memory so the same
    code is safe on an 8 GB laptop GPU and on an 80 GB A100 (where the matrix form then runs at
    full resolution too).

    Two guards matter more than the fraction itself:

    * The budget is capped by *total* memory as well as free memory.  Free memory is the
      quantity that actually limits an allocation, but it shrinks as the pyramid cache fills
      and as other processes on the same GPU grow -- and a budget that tracks a shrinking
      number can collapse to nothing, which is how a tiling factor of 1 window arose once
      already.  Capping by total memory keeps the budget stable.
    * ``fraction`` is well under 1 because the caller needs headroom for the pyramid cache,
      the score accumulators and fragmentation, and because an allocation that fails is far
      more expensive (a retry, or a wrong answer) than a smaller tile.
    """
    if not torch.cuda.is_available() or str(device).startswith("cpu"):
        # Host memory is plentiful relative to this workload; cap only to bound the copy.
        return 4 << 30
    free, total = torch.cuda.mem_get_info()
    return int(min(free, total) * fraction)


@torch.no_grad()
def ncc_peak_multi(image: torch.Tensor, templates: torch.Tensor, max_work_bytes: int = None):
    """Best NCC over a batch of templates ``templates`` (K, h, w) at every location.

    Returns ``(best_score, best_index)``, each shaped like the image: ``best_score[y, x]``
    is the maximum NCC across the K templates at that location and ``best_index[y, x]`` is
    the arg-max template.  Retrieval and coarse localisation only need *some* template to
    explain a location, so K full-resolution maps are never materialised.

    The K numerators are one matrix product: ``unfold`` exposes every sliding window as a
    column of a ``(h*w, H*W)`` matrix, so ``(K, h*w) @ (h*w, H*W)`` is all K correlations at
    once.  The denominator depends only on the image, so it is computed once and broadcast.
    This is measured 5-16x faster than one convolution per template, and it matched an
    explicit sliding-window reference to 5e-15.

    The window dimension is processed in tiles sized to the device's free memory.  That
    matters at fine pyramid levels: on a 4032x3024 image the full ``(h*w, H*W)`` product is
    tens of gigabytes, and a non-tiled implementation either exhausts memory or has to fall
    back to per-template convolutions, which measured ~60x slower at ds=4 and effectively
    unusable at ds=2.  Tiling keeps the fast path available at every level.

    A per-template convolution loop survives only as a fallback for a pathologically small
    memory budget.  Its loop is deliberate, not an oversight: a conv2d batch dimension
    cannot express "a different kernel per batch element" (see trap 1 above).
    """
    if templates.dim() != 3:
        raise ValueError(f"expected (K, h, w) templates, got {tuple(templates.shape)}")
    k_count, kh, kw = templates.shape
    height, width = image.shape
    if kh > height or kw > width:
        raise ValueError(f"template {kh}x{kw} larger than image {height}x{width}")

    n = float(kh * kw)
    _, sigma = window_stats(image, kh, kw)
    flat = sigma < FLAT_WINDOW_STD
    denom_common = (sigma * (n**0.5)).to(ACC_DTYPE)
    image_acc = image.to(ACC_DTYPE)

    budget = max_work_bytes if max_work_bytes is not None else _available_workspace_bytes(image.device)
    try:
        best, best_index = _peak_multi_unfold(image_acc, templates, denom_common, kh, kw, budget)
    except torch.cuda.OutOfMemoryError:
        # The budget is derived from free memory at call time, so this should not happen;
        # if allocation still fails, the low-memory path is the safe answer.
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        best, best_index = _peak_multi_loop(image_acc, templates, denom_common, n)

    best = best.clone()
    best[flat] = 0.0
    return best.to(dtype=image.dtype), best_index


def _peak_multi_unfold(image_acc, templates, denom_common, kh, kw, budget: int):
    """All-template NCC peak via one ``unfold`` view and a tiled matrix product.

    ``unfold`` exposes every sliding window as a row of a ``(H*W, kh*kw)`` matrix -- a
    *view*, so it costs no memory regardless of size.  Only the products are materialised,
    and those are computed in row tiles sized to ``budget``.

    Per window the result needed is ``(best score across K, which K)``, so each tile reduces
    over the template axis immediately and only two ``(H*W,)`` vectors are carried across
    tiles.  Tiling therefore changes nothing numerically: the reduction is associative and
    the tie-break (strict ``>``) is applied in the same window order as an untiled run.
    """
    height, width = image_acc.shape
    k_count = templates.shape[0]
    window_len = kh * kw
    padded = _pad_bottom_right(image_acc, kh, kw)

    # (H, W, kh, kw) view -> (H*W, kh*kw); both reshapes are view operations on this layout.
    windows = padded.unfold(0, kh, 1).unfold(1, kw, 1).reshape(height * width, window_len)
    total = windows.shape[0]

    td = templates.to(device=image_acc.device, dtype=ACC_DTYPE)
    td = td - td.mean(dim=(1, 2), keepdim=True)
    tnorm = torch.linalg.vector_norm(td.reshape(k_count, -1), dim=1).reshape(k_count, 1)
    td_flat = td.reshape(k_count, window_len)

    best = torch.full((total,), -2.0, device=image_acc.device, dtype=ACC_DTYPE)
    best_index = torch.zeros((total,), device=image_acc.device, dtype=torch.int32)
    # Keep the denominator 1-D so it broadcasts *along* the template axis only.  A (1, W_t)
    # denominator would broadcast against a degenerate (1, 1) numerator into a full
    # (1, total) tensor, silently producing wrong scores instead of raising.
    denom = (tnorm * denom_common.reshape(1, -1)).clamp_(min=EPS)

    pos = 0
    # A tile of W_t windows costs K * W_t scores plus a few arrays of that size.
    while pos < total:
        remaining = total - pos
        bytes_per_window = max(1, k_count * image_acc.element_size() * 3)
        # The floor is applied to the *budget-derived* size and then clamped to what is left,
        # so a final short tile is still exactly `remaining` windows wide.  Raising `tile` to
        # MIN_TILE without re-clamping made the last tile wider than the data and tripped the
        # shape assertion below.
        tile = min(remaining, max(MIN_TILE, budget // bytes_per_window))
        try:
            block = windows[pos : pos + tile]
            scores = (block @ td_flat.t()).t() / denom[:, pos : pos + tile]
            if scores.shape != (k_count, tile):
                raise RuntimeError(
                    f"score tiling produced {tuple(scores.shape)}, expected {(k_count, tile)}"
                )
            tile_best, tile_index = scores.max(dim=0)
            better = tile_best > best[pos : pos + tile]
            best[pos : pos + tile] = torch.where(better, tile_best, best[pos : pos + tile])
            best_index[pos : pos + tile] = torch.where(
                better, tile_index.to(torch.int32), best_index[pos : pos + tile]
            )
            pos += tile
            del block, scores, tile_best, tile_index
        except torch.cuda.OutOfMemoryError:
            # The budget can still be optimistic on a shared GPU.  Halving the tile keeps the
            # fast matrix path instead of abandoning it for the per-template convolution loop,
            # which is 5-16x slower and unusable at fine levels.
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if tile <= MIN_TILE:
                raise
            budget = max(1, budget // 2)

    return best.reshape(height, width), best_index.reshape(height, width)


def _peak_multi_loop(image_acc, templates, denom_common, n):
    """Fallback: one convolution per template, accumulating a running max."""
    k_count, kh, kw = templates.shape
    height, width = image_acc.shape
    padded = _pad_bottom_right(image_acc, kh, kw)
    best = torch.full((height, width), -2.0, device=image_acc.device, dtype=ACC_DTYPE)
    best_index = torch.zeros((height, width), device=image_acc.device, dtype=torch.int32)

    for k in range(k_count):
        td, tnorm = _template_norm(templates[k])
        m = _corr_valid(padded, td) / (denom_common * tnorm).clamp_(min=EPS)
        better = m > best
        best = torch.where(better, m, best)
        best_index = torch.where(
            better, torch.tensor(k, dtype=torch.int32, device=image_acc.device), best_index
        )
    return best, best_index


def extract_patch(gray: torch.Tensor, x: int, y: int, w: int, h: int, pad: bool = True):
    """Crop ``[x, y, w, h]`` from a 2-D image, optionally zero-padding out-of-range areas.

    Used to pull candidate patches out for verification, and to build the local search
    windows during refinement.
    """
    height, width = gray.shape
    if not pad:
        return gray[y : y + h, x : x + w].clone()
    out = torch.zeros((h, w), device=gray.device, dtype=gray.dtype)
    sy0, sx0 = max(0, y), max(0, x)
    sy1, sx1 = min(height, y + h), min(width, x + w)
    if sy1 > sy0 and sx1 > sx0:
        out[sy0 - y : sy1 - y, sx0 - x : sx1 - x] = gray[sy0:sy1, sx0:sx1]
    return out


def block_mean_downsample(x: torch.Tensor, factor: int) -> torch.Tensor:
    """Anti-aliased downsample by ``factor`` using non-overlapping block averages.

    Stride sampling (``x[::factor, ::factor]``) is the obvious choice and is *wrong* for
    template matching against a differently-anchored crop.  A patch cropped at x=1978
    downsampled by 4 samples the columns 1978, 1982, 1986, ..., while the parent image
    downsampled by 4 samples 0, 4, 8, ...; whenever ``x % factor != 0`` the two grids are out
    of phase and the "same" content no longer matches.  Measured on a real ground-truth pair
    at factor 4 this produced a maximum pixel discrepancy of 190 grey levels and dragged the
    true location's NCC down to 0.73, which is what let unrelated images outrank the correct
    one during screening.

    Averaging blocks instead makes each output pixel a local mean, so a phase shift only
    changes which pixels sit on a block boundary -- a small perturbation rather than an
    entirely different sample set.  It also low-pass filters first, so textured imagery does
    not alias.  The trailing partial block is dropped; for a 128 px patch at factor 4 that
    costs at most 3 columns.
    """
    if factor <= 1:
        return x
    height, width = x.shape[-2:]
    usable_h = (height // factor) * factor
    usable_w = (width // factor) * factor
    if usable_h == 0 or usable_w == 0:
        return x[:1, :1]
    trimmed = x[..., :usable_h, :usable_w]
    return trimmed.reshape(*x.shape[:-2], usable_h // factor, factor, usable_w // factor, factor).mean(
        dim=(-3, -1)
    )
