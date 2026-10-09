"""Layer 1: image decoding, grayscale conversion and pyramid caching.

A run touches the whole library (retrieval scans all images) or a single image
(grounding), so decoding is the fixed cost of inference.  ``ImageBank`` decodes each image
exactly once, converts it to grayscale, and keeps the *coarse* pyramid levels resident on
the GPU -- those are what screening uses and they are tiny (a 10 MP image is 0.66 MP at
16x, 2.6 MP at 4x).

Full-resolution pixels are memory-hungry (a 10 MP image is 40 MB as float32, 80 MB as
float64) and are needed only for the handful of images a query refines on, so they are
materialised on demand through a small LRU cache instead of being held for all 154 images.
"""

from __future__ import annotations

import io
import json
import os
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

__all__ = ["load_gray_f32", "read_json", "ImageBank"]


def load_gray_f32(path: str) -> np.ndarray:
    """Decode an image file to a 2-D float32 grayscale array using Rec.601 luma.

    Alpha is composited onto white first.  Transparent (RGBA/LA/P) files appear in both
    competition splits, and dropping alpha would read transparent regions as black and
    wreck the correlation for any patch overlapping them.
    """
    from PIL import Image

    Image.MAX_IMAGE_PIXELS = None  # the largest images exceed Pillow's decompression guard
    with Image.open(path) as im:
        if im.mode in ("RGBA", "LA", "P"):
            im = im.convert("RGBA")
            white = Image.new("RGBA", im.size, (255, 255, 255, 255))
            im = Image.alpha_composite(white, im).convert("RGB")
        else:
            im = im.convert("RGB")
        arr = np.asarray(im, dtype=np.float32)
    return arr @ np.array([0.299, 0.587, 0.114], dtype=np.float32)


def read_json(path: str):
    """Read JSON as UTF-8 regardless of the platform's default encoding.

    The dataset ships non-ASCII file names, so relying on the locale default raises
    ``UnicodeDecodeError`` on a GBK-defaulted Windows machine.
    """
    with io.open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _stride_downsample(gray: np.ndarray, ds: int) -> np.ndarray:
    """Anti-aliased downsample by block averaging; ``ds == 1`` returns the array unchanged.

    Block averaging rather than stride sampling, because the query patches are cropped at
    arbitrary offsets: a patch at x=1978 sampled with stride 4 sits on a different pixel grid
    than its parent image sampled with stride 4, so the two "downsampled by the same factor"
    images do not correspond.  See ``patchmatch.core.ncc.block_mean_downsample`` for the
    measurement that exposed this.
    """
    if ds == 1:
        return gray
    height, width = gray.shape
    usable_h = (height // ds) * ds
    usable_w = (width // ds) * ds
    if usable_h == 0 or usable_w == 0:
        return gray[:1, :1]
    return np.ascontiguousarray(
        gray[:usable_h, :usable_w].reshape(usable_h // ds, ds, usable_w // ds, ds).mean(axis=(1, 3))
    )


class ImageBank:
    """Library images addressed by ``image_id``, cached per pyramid level.

    Parameters
    ----------
    image_ids, paths:
        Parallel sequences; ``image_ids[i]`` is the competition ``image_id`` and
        ``paths[i]`` its file on disk.
    coarse_levels:
        Downsample factors built eagerly for every image and kept resident on the GPU.
    device:
        Torch device string (``"cuda"`` or ``"cpu"``).
    full_cache:
        How many full-resolution grayscale arrays to keep in host RAM.  Refinement uses
        one image many times in a row, so a small LRU removes repeated decodes without
        holding the library.
    """

    def __init__(self, image_ids, paths, coarse_levels=(4, 2), device="cuda", full_cache=2):
        self.image_ids = list(image_ids)
        self.paths = list(paths)
        if len(self.image_ids) != len(self.paths):
            raise ValueError("image_ids and paths must be parallel sequences")
        self.index = {image_id: n for n, image_id in enumerate(self.image_ids)}
        self.device = device
        self.coarse_levels = tuple(sorted(set(coarse_levels), reverse=True))
        self.full_cache = max(1, full_cache)
        self._coarse: dict = {}   # image_id -> {ds: np.ndarray}
        self._gpu: dict = {}      # (image_id, ds) -> torch.Tensor
        self._full: OrderedDict = OrderedDict()
        self._full_shape: dict = {}  # image_id -> (height, width) at full resolution
        self._workers = max(1, min(8, (os.cpu_count() or 4)))

    # ------------------------------------------------------------------ loading
    def __len__(self) -> int:
        return len(self.image_ids)

    def load_all(self, progress=None, workers=None) -> None:
        """Decode every image and build its coarse pyramid, parallelised across files.

        ``progress(done, total)`` is called periodically if supplied.
        """
        workers = workers or self._workers

        def decode(position):
            gray = load_gray_f32(self.paths[position])
            return position, gray.shape, {ds: _stride_downsample(gray, ds) for ds in self.coarse_levels}

        done = 0
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for position, shape, levels in pool.map(decode, range(len(self.image_ids))):
                image_id = self.image_ids[position]
                self._coarse[image_id] = levels
                self._full_shape[image_id] = shape
                done += 1
                if progress and (done % 25 == 0 or done == len(self.image_ids)):
                    progress(done, len(self.image_ids))

    def _ensure_coarse(self, image_id) -> dict:
        """Return (building if necessary) the coarse pyramid for one image."""
        levels = self._coarse.get(image_id)
        if levels is None:
            gray = load_gray_f32(self.paths[self.index[image_id]])
            levels = {ds: _stride_downsample(gray, ds) for ds in self.coarse_levels}
            self._coarse[image_id] = levels
            self._full_shape[image_id] = gray.shape
            self._remember_full(image_id, gray)
        return levels

    def _remember_full(self, image_id, gray: np.ndarray) -> None:
        """Insert into the full-resolution LRU, evicting the oldest entry."""
        self._full[image_id] = gray
        self._full.move_to_end(image_id)
        while len(self._full) > self.full_cache:
            self._full.popitem(last=False)

    # ------------------------------------------------------------------ access
    def coarse(self, image_id, ds: int) -> torch.Tensor:
        """GPU tensor for ``image_id`` at downsample factor ``ds`` (built once, then cached).

        Levels outside ``coarse_levels`` are derived from the full-resolution array and
        cached the same way.
        """
        key = (image_id, ds)
        cached = self._gpu.get(key)
        if cached is None:
            levels = self._ensure_coarse(image_id)
            if ds not in levels:
                levels[ds] = _stride_downsample(self.grayscale(image_id), ds)
            cached = torch.from_numpy(levels[ds]).to(self.device)
            self._gpu[key] = cached
        return cached

    def grayscale(self, image_id) -> np.ndarray:
        """Full-resolution float32 grayscale array via the LRU cache."""
        if image_id in self._full:
            self._full.move_to_end(image_id)
            return self._full[image_id]
        gray = load_gray_f32(self.paths[self.index[image_id]])
        self._remember_full(image_id, gray)
        return self._full[image_id]

    def full_tensor(self, image_id) -> torch.Tensor:
        """Full-resolution GPU tensor, built on demand and deliberately not device-cached."""
        return torch.from_numpy(self.grayscale(image_id)).to(self.device)

    def shape(self, image_id) -> tuple[int, int]:
        """``(height, width)`` of an image at **full resolution**.

        The full-resolution dimensions are recorded when an image is first decoded.  They
        cannot be inferred from a pyramid level: the levels are stored largest-downsample
        first, so the first entry is the *smallest* array, and reporting that as the image
        size silently mis-scales every coordinate derived from it.
        """
        return self._full_shape[image_id]

    def drop_full(self, image_id) -> None:
        """Evict one image from the full-resolution cache (keeps the GPU levels)."""
        self._full.pop(image_id, None)

    def free_device_cache(self) -> None:
        """Release all cached GPU tensors."""
        self._gpu.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------ helpers
    def peak_vram_gb(self) -> float:
        if not torch.cuda.is_available():
            return 0.0
        return torch.cuda.max_memory_allocated() / 2**30
