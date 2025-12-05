"""Image format loading utilities.

This module provides functions for loading images from various formats
(PNG, JPG, NIfTI) into numpy arrays for preprocessing.
"""

from pathlib import Path

import nibabel as nib
import numpy as np
from PIL import Image


def load_image(path: str | Path) -> np.ndarray:
    """Load an image file and return as numpy array.

    Supports PNG and JPG image formats. Returns the image as a numpy
    array preserving the original color mode (grayscale or RGB).

    Args:
        path: Path to the image file (string or pathlib.Path).

    Returns:
        Image data as numpy array. Shape is always (H, W, 3) (RGB).

    Raises:
        FileNotFoundError: If the image file does not exist.
        PIL.UnidentifiedImageError: If the file is not a valid image.

    Example:
        >>> from pathlib import Path
        >>> img = load_image(Path("brain_slice.png"))
        >>> img.shape
        (240, 240, 3)
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {path}")

    img = Image.open(path).convert("RGB")
    return np.array(img)


def load_nifti(path: str | Path) -> np.ndarray:
    """Load a NIfTI volume file and return as numpy array.

    Supports both .nii and .nii.gz formats. Returns volume in
    canonical (RAS+) orientation for consistent processing.

    Args:
        path: Path to the NIfTI file.

    Returns:
        Volume data as numpy array with shape (H, W, D).

    Raises:
        FileNotFoundError: If the volume file does not exist.
        nibabel.filebasedimages.ImageFileError: If the file is not a valid NIfTI.

    Example:
        >>> from pathlib import Path
        >>> volume = load_nifti(Path("brain_volume.nii.gz"))
        >>> volume.shape
        (240, 240, 155)
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Volume file not found: {path}")

    nii = nib.load(path)
    # Convert to canonical orientation (RAS+) for consistency
    nii_canonical = nib.as_closest_canonical(nii)
    return np.array(nii_canonical.get_fdata())
