"""Image format loading utilities.

This module provides functions for loading images from various formats
(PNG, JPG) into numpy arrays for preprocessing.
"""

from pathlib import Path

import numpy as np
from PIL import Image


def load_image(path: str | Path) -> np.ndarray:
    """Load an image file and return as numpy array.

    Supports PNG and JPG image formats. Returns the image as a numpy
    array preserving the original color mode (grayscale or RGB).

    Args:
        path: Path to the image file (string or pathlib.Path).

    Returns:
        Image data as numpy array. Shape is (H, W) for grayscale
        or (H, W, 3) for RGB images.

    Raises:
        FileNotFoundError: If the image file does not exist.
        PIL.UnidentifiedImageError: If the file is not a valid image.

    Example:
        >>> from pathlib import Path
        >>> img = load_image(Path("brain_slice.png"))
        >>> img.shape
        (240, 240)
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {path}")

    img = Image.open(path)
    return np.array(img)
