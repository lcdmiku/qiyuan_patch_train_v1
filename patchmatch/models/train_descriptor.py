"""Layer 2: train the patch descriptor.

What is being learned
---------------------
The descriptor maps a 128x128 RGB patch to an embedding such that a patch and other patches
from the *same* aerial frame are close, and patches from different frames are far apart.
That is a pure metric-learning problem: no labels beyond "which image did this come from"
are needed, and the training pack supplies 2476 such patches across 572 images.

Training procedure
------------------
1. Build a :class:`PatchDataset` over ``train/patch_gt.json``.
2. Draw batches with :class:`ClassBalancedSampler` so every batch contains same-image pairs
   (positives) and cross-image pairs (negatives) instead of being dominated by trivial
   negatives -- with ~570 labels in a batch of 64, uniform sampling would rarely produce a
   positive at all.
3. Optimise :class:`CombinedLoss` (circle loss + batch-hard triplet) with AdamW.
4. Validate by retrieval R@1 on the held-out split, which is the metric that actually
   matters, and keep the best checkpoint on that number rather than on the loss.

Usage::

    python -m patchmatch.models.train_descriptor \\
        --data qiyuan_patch_train_v1 --epochs 40 --batch-size 64 --out weights/descriptor.pt
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch

from ...core.imagebank import read_json
from .dataset import ClassBalancedSampler, PatchDataset
from .descriptor import build_descriptor
from .losses import CombinedLoss

__all__ = ["TrainConfig", "train", "main"]


class TrainConfig:
    """Hyper-parameters for descriptor training.

    ``classes_per_batch`` is the single most important knob: it sets how many distinct
    source images contribute to a batch and therefore how hard the negatives are.  Too low
    and the loss sees only easy negatives; too high and each class has too few samples to
    form positives.
    """

    def __init__(
        self,
        epochs: int = 40,
        batch_size: int = 64,
        classes_per_batch: int = 16,
        learning_rate: float = 3e-4,
        weight_decay: float = 1e-4,
        embed_dim: int = 256,
        width: int = 32,
        blocks_per_stage: int = 1,
        warmup_epochs: int = 2,
        num_workers: int = 4,
        circle_weight: float = 1.0,
        triplet_weight: float = 0.5,
        seed: int = 0,
        device: str = None,
    ):
        self.epochs = epochs
        self.batch_size = batch_size
        self.classes_per_batch = classes_per_batch
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.embed_dim = embed_dim
        self.width = width
        self.blocks_per_stage = blocks_per_stage
        self.warmup_epochs = warmup_epochs
        self.num_workers = num_workers
        self.circle_weight = circle_weight
        self.triplet_weight = triplet_weight
        self.seed = seed
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def as_dict(self) -> dict:
        return dict(self.__dict__)


def _build_optimizer(model, config: TrainConfig):
    """AdamW with no weight decay on BatchNorm and bias parameters.

    Decaying BatchNorm's scale or a bias toward zero fights the loss for no benefit; this
    split is the standard convention for convolutional embedding networks.
    """
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim <= 1 or name.endswith(".bias") or "norm" in name:
            no_decay.append(param)
        else:
            decay.append(param)
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": config.weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=config.learning_rate,
    )


def _build_scheduler(optimizer, config: TrainConfig, steps_per_epoch: int):
    """Linear warmup then cosine decay, stepped per iteration.

    Warmup matters because a randomly initialised embedding network produces near-random
    similarities, and the circle loss on those early batches has a very large gradient.
    """
    warmup = max(1, config.warmup_epochs * steps_per_epoch)
    total = max(warmup + 1, config.epochs * steps_per_epoch)

    def factor(step):
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, total - warmup)
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


@torch.no_grad()
def evaluate_retrieval(model, dataset, device: str, batch_size: int = 64, max_queries: int = 400) -> dict:
    """Embedding-retrieval R@1 / R@5 on one split, using the split's own patches as gallery.

    Each patch queries the pooled embedding of every *other* image in the split, and a hit
    means a patch from the same source image is ranked in the top-k.  This is the metric the
    competition scores, so it is the only sensible checkpoint criterion.  Note the gallery is
    weak (a single pooled embedding per image), which understates the descriptor's value in
    the real pipeline, where it only ever reorders NCC-shortlisted candidates.
    """
    from torch.utils.data import DataLoader

    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    embeddings, labels = [], []
    for patches, batch_labels in loader:
        embeddings.append(model(patches.to(device)).cpu())
        labels.append(batch_labels)
    embeddings = torch.cat(embeddings)
    labels = torch.cat(labels).numpy()

    # pooled, renormalised embedding per source image
    unique = sorted(set(labels.tolist()))
    gallery = []
    for label in unique:
        pooled = embeddings[labels == label].mean(dim=0)
        gallery.append(torch.nn.functional.normalize(pooled, dim=-1))
    gallery = torch.stack(gallery)
    gallery_labels = np.array(unique)

    similarity = embeddings @ gallery.t()
    order = torch.argsort(similarity, dim=1, descending=True).numpy()
    ranked_labels = gallery_labels[order]

    limit = min(max_queries, len(labels))
    hits1 = hits5 = 0
    for row in range(limit):
        top = ranked_labels[row]
        # the patch's own image always ranks first when it is the only member; skip those
        if (labels == labels[row]).sum() == 1:
            top = top[1:]
        hits1 += bool(len(top) and top[0] == labels[row])
        hits5 += bool(labels[row] in top[:5])
    return {"R@1": hits1 / limit, "R@5": hits5 / limit, "images": len(unique)}


def train(data_root: str, out_path: str, config: TrainConfig = None, log=print) -> dict:
    """Train the descriptor and save the best checkpoint by validation R@1.

    ``data_root`` is the unpacked training pack (``train/`` and ``val/`` inside).  Returns a
    summary dict with the best metric and the checkpoint path.
    """
    from torch.utils.data import DataLoader

    config = config or TrainConfig()
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    train_gt = read_json(os.path.join(data_root, "train", "patch_gt.json"))
    val_gt = read_json(os.path.join(data_root, "val", "patch_gt.json"))
    train_set = PatchDataset(os.path.join(data_root, "train"), train_gt, augment=True)
    val_set = PatchDataset(os.path.join(data_root, "val"), val_gt, augment=False)
    log(f"train patches: {len(train_set)} over {len(set(train_set.labels.tolist()))} images")
    log(f"val   patches: {len(val_set)} over {len(set(val_set.labels.tolist()))} images")

    sampler = ClassBalancedSampler(
        train_set.labels, batch_size=config.batch_size, classes_per_batch=config.classes_per_batch,
        seed=config.seed,
    )
    loader = DataLoader(
        train_set, batch_size=config.batch_size, sampler=sampler,
        num_workers=config.num_workers, drop_last=True, pin_memory=(config.device == "cuda"),
    )

    model = build_descriptor(
        embed_dim=config.embed_dim, width=config.width, blocks_per_stage=config.blocks_per_stage
    ).to(config.device)
    criterion = CombinedLoss(
        circle_weight=config.circle_weight, triplet_weight=config.triplet_weight
    ).to(config.device)
    optimizer = _build_optimizer(model, config)
    scheduler = _build_scheduler(optimizer, config, len(loader))

    best = {"R@1": -1.0, "epoch": -1}
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    started = time.time()

    for epoch in range(config.epochs):
        sampler.set_epoch(epoch)
        model.train()
        running, components = 0.0, {"circle": 0.0, "triplet": 0.0}
        for step, (patches, labels) in enumerate(loader):
            patches = patches.to(config.device, non_blocking=True)
            labels = labels.to(config.device, non_blocking=True)
            embeddings = model(patches)
            loss, parts = criterion(embeddings, labels)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            # Metric-learning gradients on hard batches can spike; clip for stability.
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            scheduler.step()

            running += float(loss.detach())
            for key in components:
                components[key] += parts[key]
            if (step + 1) % 20 == 0:
                log(f"  epoch {epoch+1} step {step+1}/{len(loader)} "
                    f"loss {running/(step+1):.4f} "
                    f"(circle {components['circle']/(step+1):.4f}, "
                    f"triplet {components['triplet']/(step+1):.4f}) "
                    f"lr {scheduler.get_last_lr()[0]:.2e}")

        metrics = evaluate_retrieval(model, val_set, config.device)
        log(f"epoch {epoch+1}/{config.epochs}: train loss {running/max(1,len(loader)):.4f} "
            f"val R@1 {metrics['R@1']:.4f} R@5 {metrics['R@5']:.4f} ({time.time()-started:.0f}s)")

        if metrics["R@1"] > best["R@1"]:
            best = {"R@1": metrics["R@1"], "R@5": metrics["R@5"], "epoch": epoch + 1}
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": config.as_dict(),
                    "metrics": best,
                    "arch": {
                        "embed_dim": config.embed_dim,
                        "width": config.width,
                        "blocks_per_stage": config.blocks_per_stage,
                    },
                },
                out_path,
            )
            log(f"  saved checkpoint -> {out_path}")

    best["path"] = out_path
    best["seconds"] = time.time() - started
    return best


def main(argv=None):
    parser = argparse.ArgumentParser(description="Train the patch descriptor (layer 2).")
    parser.add_argument("--data", default="qiyuan_patch_train_v1",
                        help="unpacked training pack containing train/ and val/")
    parser.add_argument("--out", default="weights/descriptor.pt")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--classes-per-batch", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--embed-dim", type=int, default=256)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--blocks-per-stage", type=int, default=1)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default=None)
    args = parser.parse_args(argv)

    config = TrainConfig(
        epochs=args.epochs, batch_size=args.batch_size, classes_per_batch=args.classes_per_batch,
        learning_rate=args.lr, embed_dim=args.embed_dim, width=args.width,
        blocks_per_stage=args.blocks_per_stage, num_workers=args.workers, device=args.device,
    )
    summary = train(args.data, args.out, config)
    print("best:", summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
