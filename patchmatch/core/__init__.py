"""Layer 1: geometry, the NCC primitive, and image I/O."""

from .geometry import BBox, clip_bbox, iou, scale_bbox, xywh_to_xyxy, xyxy_to_xywh
from .imagebank import ImageBank, load_gray_f32, read_json
from .ncc import extract_patch, ncc_map, ncc_peak, ncc_peak_multi, window_stats

__all__ = [
    "BBox",
    "clip_bbox",
    "iou",
    "scale_bbox",
    "xywh_to_xyxy",
    "xyxy_to_xywh",
    "ImageBank",
    "load_gray_f32",
    "read_json",
    "extract_patch",
    "ncc_map",
    "ncc_peak",
    "ncc_peak_multi",
    "window_stats",
]
