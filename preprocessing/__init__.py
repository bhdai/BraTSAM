"""BraTSAM shared preprocessing module.

This package provides shared preprocessing functions for both training
and inference pipelines, ensuring parity between data transformations.

Exports:
    normalize_slice: Normalize 2D slice to 0-255 range
    load_image: Load PNG/JPG images as numpy arrays
    find_best_slice: Find slice with largest tumor area
    extract_slice: Extract 2D slice from 3D volume
    get_bounding_box: Calculate bounding box from mask
"""

from preprocessing.normalize import normalize_slice
from preprocessing.formats import load_image
from preprocessing.volume import extract_slice, find_best_slice, get_bounding_box

__all__ = [
    "normalize_slice",
    "load_image",
    "find_best_slice",
    "extract_slice",
    "get_bounding_box",
]
