"""Layer 2: neural building blocks for the learned patch descriptor.

Architecture overview
---------------------
The learned branch is a *patch descriptor* network used as a complementary retrieval
signal.  It maps a 128x128 RGB patch to a unit-norm embedding, trained with a
metric-learning objective so that a patch and the image region it came from score higher
than any other region.

    patch (3,128,128) ──► Backbone ──► feature map (C, H/16, W/16)
                                          │
                                          ├─► GeM pool ──► (C,)
                                          │       │
                                          │       └─► Linear ──► L2-normalise ──► embedding (D,)

The backbone is a small ResNet-style convolutional trunk with configurable depth and
width.  Three design choices matter here and each is justified in its class docstring:

* **GeM pooling instead of average pooling** -- aerial patches are dominated by textureless
  background (water, tarmac, canopy), and average pooling dilutes the few discriminative
  pixels into that background.  A generalized-mean pool with a learnable exponent lets the
  network choose how peaky the aggregation is.
* **No downsampling in the first block** -- a 128x128 patch is already small; an aggressive
  stride-2 stem would discard exactly the high-frequency detail that distinguishes one
  patch of canopy from another.
* **L2-normalised output** -- makes cosine similarity a plain dot product and keeps the
  loss scale stable across batch sizes.

An honest note on expected value
--------------------------------
On this competition's data the patches are *exact pixel crops* of their source image at
native resolution (verified: the ground-truth crop and the supplied patch differ by
exactly 0 for every patch sampled in both splits).  That makes normalized cross-correlation
an exact and provably optimal matcher, and it already ranks the true source image first for
essentially every query.  A learned descriptor cannot beat an exact matcher on the task it
is exact for, so the descriptor is wired in as a *second opinion* whose weight defaults to
a level that cannot hurt: it reorders candidates only when the NCC score is ambiguous.  Its
real value is insurance against an evaluation set that is not exact crops (recompression,
resampling or colour shifts between the library image and the query patch), where NCC
degrades faster than a learned embedding.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["GeM", "ConvBlock", "ResidualBlock", "PatchDescriptorNet", "build_descriptor"]


class GeM(nn.Module):
    """Generalized-mean pooling over the spatial dimensions.

    ``GeM(x) = (mean(x**p))**(1/p)`` interpolates between average pooling (``p = 1``) and
    max pooling (``p -> inf``).  ``p`` is a learnable parameter, so the network decides per
    channel how much to emphasise the strongest response.  This matters for aerial patches
    because most of the area is uninformative background and average pooling would let it
    swamp the few discriminative pixels.

    The exponent is clamped to stay positive and finite; without that, ``p`` can drift to 0
    or below and the ``x**p`` terms blow up or produce NaNs on zero-valued pixels.
    """

    def __init__(self, p: float = 3.0, learnable: bool = True, eps: float = 1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.tensor(float(p)), requires_grad=learnable)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x``: (N, C, H, W) -> (N, C)."""
        p = self.p.clamp(min=1.0, max=16.0)
        # clamp_min keeps the power well-defined for zero activations (ReLU output).
        pooled = F.avg_pool2d(x.clamp(min=self.eps).pow(p), (x.shape[-2], x.shape[-1]))
        return pooled.pow(1.0 / p).flatten(1)


class ConvBlock(nn.Module):
    """Conv -> BatchNorm -> ReLU, the standard trunk unit.

    ``stride`` controls downsampling; the first block of the network uses ``stride=1``
    because a 128x128 input is too small to survive an aggressive stem without losing the
    high-frequency texture that separates one patch from another.
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, kernel: int = 3):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel, stride=stride, padding=kernel // 2, bias=False
        )
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class ResidualBlock(nn.Module):
    """Two 3x3 convolutions with an identity (or 1x1 projected) shortcut.

    Residual connections let the trunk go deep enough to build translation-tolerant
    features without the optimisation problems of a plain deep stack.  The shortcut is
    projected whenever the shape changes so the addition is always well defined.
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, 3, stride=stride, padding=1, bias=False
        )
        self.norm1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.norm2 = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

        needs_projection = stride != 1 or in_channels != out_channels
        self.shortcut = (
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
            if needs_projection
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        out = self.act(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        return self.act(out + identity)


class PatchDescriptorNet(nn.Module):
    """128x128 RGB patch -> L2-normalised embedding of dimension ``embed_dim``.

    Structure, layer by layer:

    ===============  =========================  ==================  ==============
    stage            operation                  output resolution   channels
    ===============  =========================  ==================  ==============
    stem             ConvBlock(stride=1)        128 x 128           ``width``
    layer1           ResidualBlock(stride=1)     64 x 64            ``width``
    layer2           ResidualBlock(stride=2)     32 x 32             2 x ``width``
    layer3           ResidualBlock(stride=2)     16 x 16             4 x ``width``
    layer4           ResidualBlock(stride=2)      8 x 8              8 x ``width``
    pool             GeM                         1 x 1               8 x ``width``
    head             Linear -> L2 norm           1 x 1               ``embed_dim``
    ===============  =========================  ==================  ==============

    Total downsampling is 16x, matching the 8x8 spatial grid of the deepest stage.  The
    input normalisation statistics are registered as buffers so that the exact same
    transform is applied at training and inference time, and so the module can be saved and
    reloaded without the caller having to remember them.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        width: int = 32,
        blocks_per_stage: int = 1,
        in_channels: int = 3,
        input_mean: float = 0.45,
        input_std: float = 0.25,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        # Registered (not plain attributes) so they follow the model across .to()/.cuda()
        # and are restored from a checkpoint.
        self.register_buffer("input_mean", torch.tensor(float(input_mean)))
        self.register_buffer("input_std", torch.tensor(float(input_std)))

        self.stem = ConvBlock(in_channels, width, stride=1)
        self.layer1 = self._make_stage(width, width, blocks_per_stage, stride=1)
        self.layer2 = self._make_stage(width, width * 2, blocks_per_stage, stride=2)
        self.layer3 = self._make_stage(width * 2, width * 4, blocks_per_stage, stride=2)
        self.layer4 = self._make_stage(width * 4, width * 8, blocks_per_stage, stride=2)

        self.pool = GeM(p=3.0, learnable=True)
        self.head = nn.Linear(width * 8, embed_dim, bias=True)

        self._init_weights()

    @staticmethod
    def _make_stage(in_channels: int, out_channels: int, blocks: int, stride: int) -> nn.Sequential:
        """A stage is one strided block followed by ``blocks - 1`` identity-stride blocks."""
        layers = [ResidualBlock(in_channels, out_channels, stride=stride)]
        layers += [ResidualBlock(out_channels, out_channels, stride=1) for _ in range(blocks - 1)]
        return nn.Sequential(*layers)

    def _init_weights(self) -> None:
        """Kaiming init for convolutions; BatchNorm starts at identity-ish scaling."""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.01)
                nn.init.zeros_(module.bias)

    def normalize_input(self, x: torch.Tensor) -> torch.Tensor:
        """Scale ``[0, 1]`` RGB patches to roughly zero-mean unit-variance."""
        return (x - self.input_mean) / self.input_std

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Patch batch (N, 3, 128, 128) -> pooled feature vector (N, 8 * width)."""
        x = self.normalize_input(x)
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return self.pool(x)

    def forward(self, x: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        """Patch batch (N, 3, 128, 128) -> L2-normalised embedding (N, ``embed_dim``).

        With ``normalize=False`` the pre-normalisation embedding is returned, which the
        training loss uses when it wants raw logits.
        """
        features = self.forward_features(x)
        embedding = self.head(features)
        return F.normalize(embedding, dim=-1) if normalize else embedding


def build_descriptor(embed_dim: int = 256, width: int = 32, blocks_per_stage: int = 1, **kwargs):
    """Factory mirroring :class:`PatchDescriptorNet`, kept for config-driven construction."""
    return PatchDescriptorNet(
        embed_dim=embed_dim, width=width, blocks_per_stage=blocks_per_stage, **kwargs
    )
