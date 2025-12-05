"""Volume processing utilities for 3D medical image analysis.

This module provides functions for working with 3D volumetric data,
including slice selection and bounding box extraction.
"""

import numpy as np


def find_best_slice(
    mask_volume: np.ndarray,
    tumor_labels: list[int] | None = None,
) -> tuple[int, int]:
    """Find the slice index with the largest tumor area.

    Scans through all slices along the Z-axis and identifies the slice
    containing the maximum number of tumor-labeled pixels.

    Args:
        mask_volume: 3D numpy array with shape (H, W, Z) containing
            segmentation labels.
        tumor_labels: List of integer labels representing tumor regions.
            Defaults to [1, 2, 4] (BraTS convention) if None.

    Returns:
        Tuple of (best_slice_idx, max_area) where:
        - best_slice_idx: Index of the slice with maximum tumor area
        - max_area: The tumor area in that slice (in pixels)


    Example:
        >>> volume = np.zeros((240, 240, 155), dtype=np.int16)
        >>> volume[100:150, 100:150, 77] = 1  # Add tumor region
        >>> idx, area = find_best_slice(volume)
        >>> print(f"Best slice: {idx}, Area: {area} pixels")
    """
    if tumor_labels is None:
        tumor_labels = [1, 2, 4]

    areas = [
        np.isin(mask_volume[:, :, z], tumor_labels).sum()
        for z in range(mask_volume.shape[2])
    ]

    best_slice_idx = np.argmax(areas)
    max_area = areas[best_slice_idx]
    return int(best_slice_idx), int(max_area)


def extract_slice(volume: np.ndarray, index: int) -> np.ndarray:
    """Extract a 2D slice from a 3D volume.

    Extracts a slice along the Z-axis (third dimension) from the volume.

    Args:
        volume: 3D numpy array with shape (H, W, Z).
        index: Index of the slice to extract along the Z-axis.

    Returns:
        2D numpy array with shape (H, W) containing the extracted slice.

    Example:
        >>> volume = np.random.rand(240, 240, 155)
        >>> slice_2d = extract_slice(volume, 77)
        >>> slice_2d.shape
        (240, 240)
    """
    return volume[:, :, index]


def get_bounding_box(mask: np.ndarray) -> list[int] | None:
    """Calculate the tightest bounding box around foreground pixels.

    Finds the smallest axis-aligned rectangle that contains all
    non-zero pixels in the mask.

    Args:
        mask: 2D numpy array representing a binary mask where
            positive values indicate foreground.

    Returns:
        List of 4 integers [x_min, y_min, x_max, y_max] representing
        the bounding box coordinates, or None if the mask is empty.
        Note: x corresponds to columns, y to rows.

    Example:
        >>> mask = np.zeros((100, 100), dtype=np.uint8)
        >>> mask[20:40, 30:60] = 1
        >>> get_bounding_box(mask)
        [30, 20, 59, 39]
    """
    rows, cols = np.where(mask > 0)
    if rows.size == 0 or cols.size == 0:
        return None

    x_min = int(np.min(cols))
    y_min = int(np.min(rows))
    x_max = int(np.max(cols))
    y_max = int(np.max(rows))

    return [x_min, y_min, x_max, y_max]
