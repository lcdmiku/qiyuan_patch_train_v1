"""Layer 2: training-time dataset for the patch descriptor.

The training pack gives 2476 ``(patch, source image, bbox)`` triples across 572 images, and
the label the metric-learning loss needs is simply the source-image id: patches cropped from
the same aerial frame are positives of one another, patches from different frames are
negatives.

Batch sampling matters more than usual here.  With ~570 distinct labels and a batch of 64,
uniform sampling gives very few same-label pairs, so most of the loss is wasted on
trivially separable negatives.  ``ClassBalancedSampler`` instead draws a handful of images
per batch and several patches from each, guaranteeing both positive and negative pairs
while keeping the negatives genuinely hard (same scene type, often the same flight).
"""

from __future__ import annotations

import os
import random
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

__all__ = ["PatchDataset", "ClassBalancedSampler", "build_label_index"]


def build_label_index(gt: dict) -> tuple[list[str], dict[str, int]]:
    """Map each source-image name to a contiguous integer label.

    Returns ``(image_names, name_to_label)``.  Integer labels are what the loss compares,
    and making them contiguous keeps any later embedding-table use well defined.
    """
    names = sorted({entry["image"] for entry in gt.values()})
    return names, {name: index for index, name in enumerate(names)}


class PatchDataset(Dataset):
    """Yields ``(patch_tensor, label)`` for every patch in a ``patch_gt.json`` file.

    Parameters
    ----------
    root:
        Split directory containing ``patches/`` and ``images/``.
    gt:
        Parsed ``patch_gt.json``.
    size:
        Output patch side length.  The stored patches are already 128x128; a different
        value triggers a resize, which is only useful for speed experiments.
    augment:
        Apply random flips and small brightness/contrast jitter.  Flips are safe because
        an aerial patch has no canonical orientation, and the jitter makes the descriptor
        tolerate the mild exposure differences between drone frames.
    """

    def __init__(self, root: str, gt: dict, size: int = 128, augment: bool = True):
        self.root = root
        self.size = size
        self.augment = augment
        self.keys = sorted(gt)
        _, self.name_to_label = build_label_index(gt)
        self.labels = np.array([self.name_to_label[gt[k]["image"]] for k in self.keys], dtype=np.int64)

    def __len__(self) -> int:
        return len(self.keys)

    def label_counts(self) -> dict[int, int]:
        """Number of patches per label, used to balance the sampler."""
        counts = defaultdict(int)
        for label in self.labels:
            counts[int(label)] += 1
        return dict(counts)

    def __getitem__(self, index: int):
        import io

        from PIL import Image

        key = self.keys[index]
        path = os.path.join(self.root, "patches", key)
        Image.MAX_IMAGE_PIXELS = None
        with Image.open(path) as im:
            im = im.convert("RGB")
            if im.size != (self.size, self.size):
                im = im.resize((self.size, self.size), Image.BILINEAR)
            arr = np.asarray(im, dtype=np.float32) / 255.0

        if self.augment:
            if random.random() < 0.5:
                arr = arr[:, ::-1]
            if random.random() < 0.5:
                arr = arr[::-1]
            if random.random() < 0.7:
                # brightness/contrast jitter, applied around the patch mean
                gain = 1.0 + random.uniform(-0.15, 0.15)
                bias = random.uniform(-0.08, 0.08)
                arr = np.clip(arr * gain + bias, 0.0, 1.0)
            arr = np.ascontiguousarray(arr)

        # (H, W, C) -> (C, H, W)
        return torch.from_numpy(arr).permute(2, 0, 1), int(self.labels[index])


class ClassBalancedSampler(Sampler):
    """Draws batches containing ``classes_per_batch`` labels x ``samples_per_class`` patches.

    Guarantees every batch has usable positive and negative pairs, and keeps negatives hard
    by drawing the labels themselves at random over the whole split.  Labels with fewer than
    ``samples_per_class`` patches are sampled with replacement so a batch is always full.
    """

    def __init__(self, labels: np.ndarray, batch_size: int, classes_per_batch: int = 16, seed: int = 0):
        if batch_size % classes_per_batch != 0:
            raise ValueError("batch_size must be divisible by classes_per_batch")
        self.labels = np.asarray(labels)
        self.batch_size = batch_size
        self.classes_per_batch = classes_per_batch
        self.samples_per_class = batch_size // classes_per_batch
        self.seed = seed
        self.epoch = 0

        self.by_label: dict[int, list[int]] = defaultdict(list)
        for index, label in enumerate(self.labels):
            self.by_label[int(label)].append(index)
        self.label_ids = sorted(self.by_label)

    def set_epoch(self, epoch: int) -> None:
        """Reseed per epoch so the ordering differs between epochs but is reproducible."""
        self.epoch = epoch

    def __len__(self) -> int:
        """One epoch is defined as ``len(dataset) / batch_size`` batches."""
        return max(1, len(self.labels) // self.batch_size)

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        for _ in range(len(self)):
            chosen = rng.sample(self.label_ids, min(self.classes_per_batch, len(self.label_ids)))
            batch = []
            for label in chosen:
                pool = self.by_label[label]
                if len(pool) >= self.samples_per_class:
                    batch.extend(rng.sample(pool, self.samples_per_class))
                else:
                    batch.extend(rng.choices(pool, k=self.samples_per_class))
            rng.shuffle(batch)
            yield from batch
