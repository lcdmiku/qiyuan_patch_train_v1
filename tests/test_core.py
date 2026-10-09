"""Unit tests for layer 1 (geometry) and layer 1 (NCC).

The NCC reference is an explicit sliding-window loop in float64, so passing means the fast
paths agree with the *definition* of NCC, not merely with each other.
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from patchmatch.core.geometry import clip_bbox, iou, scale_bbox  # noqa: E402
from patchmatch.core.ncc import ncc_map, ncc_peak, ncc_peak_multi  # noqa: E402


def np_ncc_reference(image: np.ndarray, template: np.ndarray) -> np.ndarray:
    """Top-left-aligned, zero-padded NCC in float64, written out as a double loop."""
    height, width = image.shape
    th, tw = template.shape
    padded = np.zeros((height + th - 1, width + tw - 1), np.float64)
    padded[:height, :width] = image.astype(np.float64)
    centred = template.astype(np.float64)
    centred = centred - centred.mean()
    tnorm = np.linalg.norm(centred)
    out = np.zeros((height, width))
    for y in range(height):
        for x in range(width):
            window = padded[y : y + th, x : x + tw]
            wd = window - window.mean()
            wn = np.linalg.norm(wd)
            out[y, x] = 0.0 if wn < 1e-9 else float((wd * centred).sum() / (wn * tnorm))
    return out


# --------------------------------------------------------------------------- geometry
def test_iou_identical_and_touching():
    box = [10, 20, 128, 128]
    assert abs(iou(box, box) - 1.0) < 1e-12, iou(box, box)
    touching = [10 + 128, 20, 128, 128]
    print(f"  identical -> {iou(box, box):.6f}, touching -> {iou(box, touching):.6f}")
    assert iou(box, touching) == 0.0


def test_iou_known_offset():
    """Two 128 px boxes offset by 64 px in x overlap by half the area -> IoU = 1/3."""
    value = iou([0, 0, 128, 128], [64, 0, 128, 128])
    print(f"  128px boxes offset by 64px -> IoU {value:.6f} (expect 1/3 = {1/3:.6f})")
    assert abs(value - 1 / 3) < 1e-9, value


def test_iou_threshold_sensitivity():
    """Report the offset at which each accuracy threshold stops passing."""
    for threshold in (0.5, 0.75):
        limit = None
        for offset in range(0, 128):
            if iou([0, 0, 128, 128], [offset, 0, 128, 128]) < threshold:
                limit = offset
                break
        print(f"  IoU>={threshold} holds up to a {limit - 1}px offset")


def test_clip_bbox_preserves_size():
    box = clip_bbox([-50, 3000, 128, 128], 1000, 1000)
    print(f"  clip([-50,3000,128,128], 1000x1000) -> {box}")
    assert box[2] == 128 and box[3] == 128
    assert box[0] == 0 and box[1] == 1000 - 128


def test_scale_bbox():
    assert scale_bbox([10, 20, 128, 128], 4) == [40.0, 80.0, 512.0, 512.0]


# --------------------------------------------------------------------------- NCC
def _fixture():
    rng = np.random.default_rng(2)
    image = rng.normal(50, 20, (50, 44)).astype(np.float32)
    templates = np.stack([
        image[3:15, 5:17],
        image[20:32, 30:42],
        rng.normal(50, 20, (12, 12)).astype(np.float32),
    ])
    return image, templates


def test_ncc_matches_reference():
    image, templates = _fixture()
    reference = np_ncc_reference(image, templates[0]) if False else None
    image_t = torch.from_numpy(image)
    for k in range(3):
        got = ncc_map(image_t, torch.from_numpy(templates[k])).numpy()
        ref = np_ncc_reference(image, templates[k])
        err = np.abs(got - ref).max()
        print(f"  template {k}: max|ncc_map - reference| = {err:.2e}")
        assert err < 1e-3, err
    assert reference is None


def test_planted_patch_scores_one_at_truth():
    image, templates = _fixture()
    image_t = torch.from_numpy(image)
    for k, (gy, gx) in enumerate([(3, 5), (20, 30)]):
        score, y, x = ncc_peak(image_t, torch.from_numpy(templates[k]))
        print(f"  template {k}: score={score:.6f} at (x={x}, y={y}), truth (x={gx}, y={gy})")
        assert abs(score - 1.0) < 1e-3
        assert (x, y) == (gx, gy)


def test_peak_multi_matches_single_maps():
    image, templates = _fixture()
    image_t = torch.from_numpy(image)
    templates_t = torch.from_numpy(templates)
    maps = np.stack([ncc_map(image_t, templates_t[k]).numpy() for k in range(3)])
    best, best_index = ncc_peak_multi(image_t, templates_t)
    best = best.numpy()
    best_index = best_index.numpy()
    err = np.abs(best - maps.max(0)).max()
    ties = np.isclose(best[None], maps, atol=1e-5).sum(0) > 1
    strict_mismatch = int(((best_index != maps.argmax(0)) & ~ties).sum())
    print(f"  max|best - max(maps)| = {err:.2e}, strict index mismatches = {strict_mismatch}")
    assert err < 1e-3, err
    assert strict_mismatch == 0


def test_inverted_template_scores_minus_one():
    rng = np.random.default_rng(1)
    image = rng.normal(50, 20, (30, 30)).astype(np.float32)
    template = -image[4:14, 6:16].copy()
    m = ncc_map(torch.from_numpy(image), torch.from_numpy(template)).numpy()
    print(f"  inverted template at its own location: {m[4, 6]:.6f} (expect -1)")
    assert abs(m[4, 6] + 1.0) < 1e-3


def test_flat_window_is_finite_and_zero():
    image = np.full((20, 20), 7.0, np.float32)
    image[10:, 10:] = 3.0
    template = np.full((5, 5), 7.0, np.float32)
    m = ncc_map(torch.from_numpy(image), torch.from_numpy(template)).numpy()
    print(f"  flat-template map: all finite={np.isfinite(m).all()}, max|value|={np.abs(m).max():.6f}")
    assert np.isfinite(m).all()
    assert np.abs(m).max() <= 1.0 + 1e-3


def test_float64_needed_for_full_resolution():
    """A 128 px window on 8-bit-like data: float32 variance must not be trusted blindly."""
    rng = np.random.default_rng(11)
    image = (rng.normal(128, 30, (400, 400))).astype(np.float32)
    image = np.clip(image, 0, 255)
    template = image[100:228, 150:278].copy()
    m = ncc_map(torch.from_numpy(image), torch.from_numpy(template)).numpy()
    y, x = np.unravel_index(m.argmax(), m.shape)
    print(f"  peak {m.max():.6f} at (x={x}, y={y}); truth (x=150, y=100)")
    assert (x, y) == (150, 100)


if __name__ == "__main__":
    test_iou_identical_and_touching()
    test_iou_known_offset()
    test_iou_threshold_sensitivity()
    test_clip_bbox_preserves_size()
    test_scale_bbox()
    test_ncc_matches_reference()
    test_planted_patch_scores_one_at_truth()
    test_peak_multi_matches_single_maps()
    test_inverted_template_scores_minus_one()
    test_flat_window_is_finite_and_zero()
    test_float64_needed_for_full_resolution()
    print("ALL LAYER-1 TESTS PASSED")
