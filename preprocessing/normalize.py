"""Normalization functions for image preprocessing.

This module provides intensity normalization functions used to prepare
2D image slices for the SAM model inference pipeline.
"""

import numpy as np


def normalize_slice(slice_2d: np.ndarray) -> np.ndarray:
    """Normalize 2D slice to 0-255 range.

    Performs min-max normalization on the input slice and scales to
    uint8 range (0-255). Handles constant-value slices by returning
    an array of zeros.

    Args:
        slice_2d: 2D numpy array representing an image slice.

    Returns:
        Normalized and scaled slice as uint8 with values in range [0, 255].
        Returns zeros array for constant-value inputs.

    Example:
        >>> import numpy as np
        >>> slice_data = np.array([[0, 100], [50, 200]], dtype=np.float32)
        >>> normalized = normalize_slice(slice_data)
        >>> normalized.dtype
        dtype('uint8')
    """
    if slice_2d.max() > slice_2d.min():
        normalized = (slice_2d - slice_2d.min()) / (slice_2d.max() - slice_2d.min())
    else:
        normalized = np.zeros_like(slice_2d)  # handle constant slice
    return (normalized * 255).astype(np.uint8)
