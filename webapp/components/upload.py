"""Image upload component for BraTSAM web application.

This module provides the file upload functionality for MRI slice images
and 3D volumes, including validation, preview display, and session state management.
"""

import io
import logging
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import nibabel as nib
import numpy as np
import streamlit as st
from PIL import Image, UnidentifiedImageError

logger = logging.getLogger(__name__)

# File size limits (NFR3)
MAX_FILE_SIZE_MB = 10
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024


@dataclass
class UploadedImage:
    """Container for uploaded image data and metadata.
    
    Attributes:
        filename: Original filename of the uploaded image.
        data: Image as numpy array with shape (H, W, 3) in RGB format.
        file_size_bytes: Size of the uploaded file in bytes.
        upload_time: Timestamp when the image was uploaded.
        file_id: Unique identifier for the uploaded file (used for caching).
    """
    
    filename: str
    data: np.ndarray  # Image as numpy array (H, W, 3)
    file_size_bytes: int
    file_id: str = ""  # Unique ID from Streamlit's file_uploader
    upload_time: datetime = field(default_factory=datetime.now)
    
    @property
    def file_size_mb(self) -> float:
        """Return file size in megabytes."""
        return self.file_size_bytes / (1024 * 1024)
    
    @property
    def dimensions(self) -> tuple[int, int]:
        """Return image dimensions as (height, width)."""
        return (self.data.shape[0], self.data.shape[1])


@dataclass
class UploadedVolume:
    """Container for uploaded 3D volume data and metadata.
    
    Attributes:
        filename: Original filename of the uploaded volume.
        volume_data: Volume as numpy array with shape (H, W, D).
        file_size_bytes: Size of the uploaded file in bytes.
        file_id: Unique identifier for caching.
        upload_time: Timestamp when the volume was uploaded.
    """
    
    filename: str
    volume_data: np.ndarray  # Shape: (H, W, D)
    file_size_bytes: int
    file_id: str = ""
    upload_time: datetime = field(default_factory=datetime.now)
    
    @property
    def file_size_mb(self) -> float:
        """Return file size in megabytes."""
        return self.file_size_bytes / (1024 * 1024)
    
    @property
    def dimensions(self) -> tuple[int, int, int]:
        """Return volume dimensions as (height, width, depth)."""
        return (
            self.volume_data.shape[0],
            self.volume_data.shape[1],
            self.volume_data.shape[2],
        )
    
    @property
    def num_slices(self) -> int:
        """Return number of slices along the Z-axis."""
        return self.volume_data.shape[2]


def is_nifti_file(filename: str) -> bool:
    """Check if filename indicates a NIfTI file.
    
    Args:
        filename: The filename to check.
    
    Returns:
        True if the filename ends with .nii or .nii.gz (case insensitive).
    """
    lower = filename.lower()
    return lower.endswith(".nii") or lower.endswith(".nii.gz")


def clear_upload() -> None:
    """Clear the uploaded data from session state.
    
    This helper function removes the currently uploaded image or volume,
    allowing users to start fresh with a new upload.
    """
    st.session_state["uploaded_data"] = None
    # Also clear legacy key for backward compatibility
    if "uploaded_image" in st.session_state:
        st.session_state["uploaded_image"] = None
    logger.info("Upload cleared from session state")


def render_upload_component() -> UploadedImage | UploadedVolume | None:
    """Render file upload component and return uploaded data if valid.
    
    Displays a file uploader widget that accepts PNG, JPG images and
    NIfTI volumes (.nii, .nii.gz). Validates file size (max 10MB) and
    file format. Stores valid uploads in st.session_state["uploaded_data"].
    
    Uses file_id caching to avoid redundant processing on reruns.
    Clears zombie state when validation fails.
    
    Returns:
        UploadedImage for 2D images, UploadedVolume for 3D volumes, or None.
    """
    # Initialize session state if not present
    if "uploaded_data" not in st.session_state:
        st.session_state["uploaded_data"] = None
    
    uploaded_file = st.file_uploader(
        "Upload MRI slice or volume",
        type=["png", "jpg", "jpeg", "nii", "nii.gz"],
        help="Upload a 2D brain MRI slice (PNG/JPG) or 3D volume (NIfTI), max 10MB",
    )
    
    if uploaded_file is None:
        # Clear zombie state when no file is present
        if st.session_state["uploaded_data"] is not None:
            clear_upload()
        return None
    
    # Check if we already processed this exact file (avoid redundant processing)
    current_file_id = uploaded_file.file_id
    existing_data = st.session_state["uploaded_data"]
    if existing_data is not None and existing_data.file_id == current_file_id:
        # Same file, return cached result
        return existing_data
    
    # File size validation (AC: #4)
    if uploaded_file.size > MAX_FILE_SIZE_BYTES:
        st.error(
            f"⚠️ **File too large** — Your file is {uploaded_file.size / (1024 * 1024):.1f}MB. "
            f"Please upload a file smaller than {MAX_FILE_SIZE_MB}MB."
        )
        logger.warning(
            f"Upload rejected: file too large ({uploaded_file.size} bytes)"
        )
        # Clear zombie state on validation failure
        st.session_state["uploaded_data"] = None
        return None
    
    # Detect file type and branch logic (AC: #1, #6)
    filename = uploaded_file.name
    
    if is_nifti_file(filename):
        # Handle NIfTI 3D volume
        return _load_nifti_volume(uploaded_file, current_file_id)
    else:
        # Handle 2D image (PNG/JPG)
        return _load_2d_image(uploaded_file, current_file_id)


def _load_nifti_volume(
    uploaded_file: st.runtime.uploaded_file_manager.UploadedFile,
    file_id: str,
) -> UploadedVolume | None:
    """Load a NIfTI volume from uploaded file.
    
    Args:
        uploaded_file: Streamlit uploaded file object.
        file_id: Unique file identifier for caching.
    
    Returns:
        UploadedVolume if successful, None on error.
    """
    try:
        # Write to temp file for nibabel to read (nibabel needs file path)
        with tempfile.NamedTemporaryFile(
            suffix=_get_nifti_suffix(uploaded_file.name),
            delete=False,
        ) as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = Path(tmp.name)
        
        # Load the NIfTI volume
        nii = nib.load(tmp_path)
        # Convert to canonical orientation (RAS+) for consistency
        nii_canonical = nib.as_closest_canonical(nii)
        volume_data = np.array(nii_canonical.get_fdata())
        
        # Clean up temp file
        tmp_path.unlink()
        
        # Create UploadedVolume dataclass
        uploaded_volume = UploadedVolume(
            filename=uploaded_file.name,
            volume_data=volume_data,
            file_size_bytes=uploaded_file.size,
            file_id=file_id,
        )
        
        # Store in session state
        st.session_state["uploaded_data"] = uploaded_volume
        logger.info(
            f"Volume uploaded successfully: {uploaded_file.name} "
            f"({uploaded_volume.file_size_mb:.2f}MB, {uploaded_volume.dimensions})"
        )
        
        return uploaded_volume
        
    except Exception as e:
        st.error(
            "⚠️ **Failed to load NIfTI volume** — The file could not be read as a valid NIfTI. "
            "Please upload a valid .nii or .nii.gz file."
        )
        logger.error(f"NIfTI load error: {e}")
        st.session_state["uploaded_data"] = None
        return None


def _load_2d_image(
    uploaded_file: st.runtime.uploaded_file_manager.UploadedFile,
    file_id: str,
) -> UploadedImage | None:
    """Load a 2D image from uploaded file.
    
    Args:
        uploaded_file: Streamlit uploaded file object.
        file_id: Unique file identifier for caching.
    
    Returns:
        UploadedImage if successful, None on error.
    """
    try:
        img = Image.open(uploaded_file).convert("RGB")
        img_array = np.array(img)
    except UnidentifiedImageError:
        st.error(
            "⚠️ **Invalid image file** — The file could not be read as an image. "
            "Please upload a valid PNG or JPG file."
        )
        logger.error(f"Image load error: unidentified image format for {uploaded_file.name}")
        st.session_state["uploaded_data"] = None
        return None
    except Exception as e:
        st.error(
            "⚠️ **Failed to load image** — An unexpected error occurred while reading the file. "
            "Please try again with a different image."
        )
        logger.error(f"Image load error: {e}")
        st.session_state["uploaded_data"] = None
        return None
    
    # Create UploadedImage dataclass
    uploaded_image = UploadedImage(
        filename=uploaded_file.name,
        data=img_array,
        file_size_bytes=uploaded_file.size,
        file_id=file_id,
    )
    
    # Store in session state
    st.session_state["uploaded_data"] = uploaded_image
    logger.info(
        f"Image uploaded successfully: {uploaded_file.name} "
        f"({uploaded_image.file_size_mb:.2f}MB, {uploaded_image.dimensions})"
    )
    
    return uploaded_image


def _get_nifti_suffix(filename: str) -> str:
    """Get the proper suffix for a NIfTI filename.
    
    Args:
        filename: The original filename.
    
    Returns:
        Either '.nii.gz' or '.nii' depending on the filename.
    """
    if filename.lower().endswith(".nii.gz"):
        return ".nii.gz"
    return ".nii"
