"""Verify the phase-stable downsampler, the bug that broke coarse screening."""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from patchmatch.core.ncc import block_mean_downsample, ncc_map  # noqa: E402


def test_block_mean_shape_and_aa():
    x = torch.arange(64, dtype=torch.float32).reshape(8, 8)
    out = block_mean_downsample(x, 2)
    print(f"  8x8 -> {tuple(out.shape)}")
    assert out.shape == (4, 4)
    # each output pixel is the mean of its 2x2 block
    assert abs(float(out[0, 0]) - (0 + 1 + 8 + 9) / 4) < 1e-6
    assert abs(float(out[1, 2]) - (20 + 21 + 28 + 29) / 4) < 1e-6
    # trailing partial block is dropped, not padded
    assert block_mean_downsample(x, 3).shape == (2, 2)


def test_phase_mismatch_is_what_breaks_stride():
    """A crop at a non-multiple offset: stride sampling fails, block mean does not.

    This is the regression test for the bug that dropped screening R@1 from 1.0 to 0.975.
    """
    rng = np.random.default_rng(0)
    image = torch.from_numpy(rng.normal(128, 40, (256, 256)).astype(np.float32))
    for offset in (0, 2, 4, 6):
        patch = image[48:48 + 128, 60 + offset : 60 + offset + 128]

        # stride sampling: template grid and image grid disagree when offset % 4 != 0
        stride_tpl = patch[::4, ::4]
        stride_img = image[::4, ::4]
        gy, gx = 48 // 4, (60 + offset) // 4
        stride_win = stride_img[gy : gy + 32, gx : gx + 32]
        stride_diff = float((stride_win - stride_tpl).abs().max()) if stride_win.shape == stride_tpl.shape else float("inf")

        # block mean: both are local means, so a phase shift only perturbs block edges
        block_tpl = block_mean_downsample(patch, 4)
        block_img = block_mean_downsample(image, 4)
        by, bx = 48 // 4, (60 + offset) // 4
        block_win = block_img[by : by + block_tpl.shape[0], bx : bx + block_tpl.shape[1]]
        block_diff = float((block_win - block_tpl).abs().max())

        # and what NCC each yields at the true location
        m_stride = ncc_map(stride_img.float(), stride_tpl.float())
        m_block = ncc_map(block_img.float(), block_tpl.float())
        print(f"  offset={offset}: stride maxdiff={stride_diff:7.1f} ncc={float(m_stride[gy, gx]):.4f} | "
              f"block maxdiff={block_diff:6.1f} ncc={float(m_block[by, bx]):.4f}")
        assert block_diff <= stride_diff + 1e-6, (offset, block_diff, stride_diff)
        assert block_diff <= 40.0, (offset, block_diff)


def test_half_block_offset_is_decisive():
    """A half-block crop offset is catastrophic for stride sampling; block mean survives it.

    This is the decisive regression case: an x offset of 62 with factor 4 puts the patch's
    sample grid and the image's sample grid on completely different pixels, and the true
    location's NCC goes *negative* under stride sampling -- worse than an unrelated image, so
    screening ranks the truth almost last.

    Block averaging does not restore an exact 1.0 here and cannot: each block of the patch
    still straddles two blocks of the image.  What it does is turn a sign-flipped, anti-
    correlated window into a mildly correlated one, which is enough for the true image to win
    the screen; the full-resolution refinement stage then recovers the exact pixel.  The
    end-to-end consequence is measured by the validation harness (R@1 went 0.975 -> 1.000
    over 80 patches), not by this unit test.
    """
    rng = np.random.default_rng(0)
    image = torch.from_numpy(rng.normal(128, 40, (256, 256)).astype(np.float32))
    patch = image[48:176, 62:190]  # x offset 62 -> 62 % 4 == 2

    stride_ncc = float(ncc_map(image[::4, ::4].float(), patch[::4, ::4].float())[12, 15])
    block_ncc = float(
        ncc_map(block_mean_downsample(image, 4).float(),
                block_mean_downsample(patch, 4).float())[12, 15]
    )
    print(f"  half-block offset: stride ncc at truth = {stride_ncc:.4f}, "
          f"block ncc at truth = {block_ncc:.4f}")
    assert stride_ncc < 0.0, stride_ncc          # stride is anti-correlated at the truth
    assert block_ncc > 0.4, block_ncc            # block mean is positively correlated
    assert block_ncc > stride_ncc + 0.3, (block_ncc, stride_ncc)


if __name__ == "__main__":
    test_block_mean_shape_and_aa()
    test_phase_mismatch_is_what_breaks_stride()
    test_half_block_offset_is_decisive()
    print("DOWNSAMPLING TESTS PASSED")
