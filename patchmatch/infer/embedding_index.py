"""Layer 5: build per-image descriptor embeddings for retrieval.

The learned descriptor scores a patch against an *image*, but the training pack labels
patches by image.  The standard cheap bridge is to embed every annotated patch of an image
and average the embeddings, then renormalise:

    e_image = normalise( mean_i( normalise(f(patch_i)) ) )

Averaging pool embeddings is a well-established approximation to an image-level descriptor
(equivalent in spirit to how image retrieval systems aggregate regional features), and
renormalising keeps cosine similarity meaningful.  Images with no annotated patches get no
embedding and are simply scored by the NCC path alone, which is the correct fallback -- a
missing embedding must not look like a zero-similarity image and push a real candidate down.

The embeddings are cached to a ``.pt`` file because they never change for a fixed library and
model, and computing them costs one forward pass per patch in the pack.
"""

from __future__ import annotations

import os
import time

import numpy as np
import torch

from ..core.imagebank import read_json

__all__ = ["load_descriptor", "build_image_embeddings", "ImageEmbeddingIndex"]


def load_descriptor(path: str, device: str = "cuda"):
    """Rebuild a :class:`PatchDescriptorNet` from a training checkpoint.

    The architecture is read from the checkpoint's ``arch`` block rather than from the
    caller, so a checkpoint can never be loaded into a mismatched network.
    """
    from ..models.descriptor import build_descriptor

    blob = torch.load(path, map_location=device)
    arch = blob.get("arch") or {}
    model = build_descriptor(
        embed_dim=arch.get("embed_dim", 256),
        width=arch.get("width", 32),
        blocks_per_stage=arch.get("blocks_per_stage", 1),
    )
    model.load_state_dict(blob["model_state"])
    model.to(device).eval()
    return model, blob.get("metrics", {})


def _load_patch_rgb(path: str, size: int = 128) -> np.ndarray:
    """Load a patch as float32 RGB in ``[0, 1]``, resized to ``size`` if necessary."""
    from PIL import Image

    Image.MAX_IMAGE_PIXELS = None
    with Image.open(path) as im:
        im = im.convert("RGB")
        if im.size != (size, size):
            im = im.resize((size, size), Image.BILINEAR)
        return np.asarray(im, dtype=np.float32) / 255.0


@torch.no_grad()
def build_image_embeddings(
    model,
    split_root: str,
    gt: dict,
    device: str = "cuda",
    batch_size: int = 64,
    log=print,
) -> dict:
    """Compute ``{image_name: unit-norm (D,) tensor}`` by pooling each image's patches.

    ``gt`` is the parsed ``patch_gt.json`` for the split; ``split_root`` is the directory
    holding ``patches/``.  Grouping by image before embedding means each patch file is read
    once and the forward passes are batched.
    """
    by_image: dict = {}
    for key, entry in gt.items():
        by_image.setdefault(entry["image"], []).append(key)

    embeddings: dict = {}
    started = time.time()
    for done, (image_name, keys) in enumerate(by_image.items(), start=1):
        batch = np.stack([_load_patch_rgb(os.path.join(split_root, "patches", k)) for k in keys])
        tensor = torch.from_numpy(batch).permute(0, 3, 1, 2).contiguous()
        vectors = []
        for start in range(0, tensor.shape[0], batch_size):
            chunk = tensor[start : start + batch_size].to(device)
            vectors.append(model(chunk).cpu())
        pooled = torch.cat(vectors).mean(dim=0)
        embeddings[image_name] = torch.nn.functional.normalize(pooled, dim=-1)
        if done % 50 == 0:
            log(f"  embedded {done}/{len(by_image)} images ({time.time() - started:.0f}s)")
    log(f"image embeddings: {len(embeddings)} images in {time.time() - started:.0f}s")
    return embeddings


class ImageEmbeddingIndex:
    """Per-image embeddings keyed by whatever id the library uses.

    The training pack keys images by *file name* while the evaluation library keys them by
    numeric ``image_id``, so the index stores file names and resolves ids through the
    library's own path list.  Keeping that translation in one place avoids the classic bug
    where every similarity lookup silently misses and retrieval degrades to library order.
    """

    def __init__(self, by_name: dict, name_by_id: dict = None):
        self.by_name = dict(by_name)
        self.name_by_id = dict(name_by_id or {})

    def as_id_map(self) -> dict:
        """``{image_id: embedding}`` for ids whose file name has an embedding."""
        if not self.name_by_id:
            return dict(self.by_name)
        out = {}
        for image_id, name in self.name_by_id.items():
            vector = self.by_name.get(name)
            if vector is None:
                vector = self.by_name.get(os.path.basename(str(name)))
            if vector is not None:
                out[image_id] = vector
        return out

    @classmethod
    def from_checkpoint(cls, model_path: str, split_root: str, gt: dict, device: str = "cuda", log=print):
        """Build an index directly from a checkpoint plus a labelled split."""
        model, metrics = load_descriptor(model_path, device=device)
        log(f"descriptor loaded ({metrics})")
        return cls(build_image_embeddings(model, split_root, gt, device=device, log=log))

    def save(self, path: str) -> None:
        torch.save({"by_name": self.by_name, "name_by_id": self.name_by_id}, path)

    @classmethod
    def load(cls, path: str):
        blob = torch.load(path, map_location="cpu")
        return cls(blob["by_name"], blob.get("name_by_id"))
