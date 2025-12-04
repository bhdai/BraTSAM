"""Image upload component for BraTSAM web application.

This module provides the file upload functionality for MRI slice images,
including validation, preview display, and session state management.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime

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


def clear_upload() -> None:
    """Clear the uploaded image from session state.
    
    This helper function removes the currently uploaded image,
    allowing users to start fresh with a new upload.
    """
    st.session_state["uploaded_image"] = None
    logger.info("Upload cleared from session state")


def render_upload_component() -> UploadedImage | None:
    """Render file upload component and return uploaded image if valid.
    
    Displays a file uploader widget that accepts PNG and JPG images.
    Validates file size (max 10MB) and image format. Stores valid
    uploads in st.session_state.uploaded_image.
    
    Uses file_id caching to avoid redundant image processing on reruns.
    Clears zombie state when validation fails.
    
    Returns:
        UploadedImage if valid file uploaded, None otherwise.
    """
    # Initialize session state if not present
    if "uploaded_image" not in st.session_state:
        st.session_state["uploaded_image"] = None
    
    uploaded_file = st.file_uploader(
        "Upload MRI slice",
        type=["png", "jpg", "jpeg"],
        help="Upload a 2D brain MRI slice (PNG or JPG, max 10MB)",
    )
    
    if uploaded_file is None:
        # Clear zombie state when no file is present
        if st.session_state["uploaded_image"] is not None:
            clear_upload()
        return None
    
    # Check if we already processed this exact file (avoid redundant processing)
    current_file_id = uploaded_file.file_id
    existing_image = st.session_state["uploaded_image"]
    if existing_image is not None and existing_image.file_id == current_file_id:
        # Same file, return cached result
        return existing_image
    
    # File size validation (AC: #4)
    if uploaded_file.size > MAX_FILE_SIZE_BYTES:
        st.error(
            f"⚠️ **File too large** — Your file is {uploaded_file.size / (1024 * 1024):.1f}MB. "
            f"Please upload an image smaller than {MAX_FILE_SIZE_MB}MB."
        )
        logger.warning(
            f"Upload rejected: file too large ({uploaded_file.size} bytes)"
        )
        # Clear zombie state on validation failure
        st.session_state["uploaded_image"] = None
        return None
    
    # Load and convert image (AC: #1, #3)
    try:
        img = Image.open(uploaded_file).convert("RGB")
        img_array = np.array(img)
    except UnidentifiedImageError:
        st.error(
            "⚠️ **Invalid image file** — The file could not be read as an image. "
            "Please upload a valid PNG or JPG file."
        )
        logger.error(f"Image load error: unidentified image format for {uploaded_file.name}")
        # Clear zombie state on validation failure
        st.session_state["uploaded_image"] = None
        return None
    except Exception as e:
        st.error(
            "⚠️ **Failed to load image** — An unexpected error occurred while reading the file. "
            "Please try again with a different image."
        )
        logger.error(f"Image load error: {e}")
        # Clear zombie state on validation failure
        st.session_state["uploaded_image"] = None
        return None
    
    # Create UploadedImage dataclass (AC: #2)
    uploaded_image = UploadedImage(
        filename=uploaded_file.name,
        data=img_array,
        file_size_bytes=uploaded_file.size,
        file_id=current_file_id,
    )
    
    # Store in session state (AC: #2, #6 - replacement)
    st.session_state["uploaded_image"] = uploaded_image
    logger.info(
        f"Image uploaded successfully: {uploaded_file.name} "
        f"({uploaded_image.file_size_mb:.2f}MB, {uploaded_image.dimensions})"
    )
    
    return uploaded_image
