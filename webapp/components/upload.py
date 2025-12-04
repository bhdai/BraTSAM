"""Image upload component for BraTSAM web application.

This module provides the file upload functionality for MRI slice images,
including validation, preview display, and session state management.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import streamlit as st
from PIL import Image

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
    """
    
    filename: str
    data: np.ndarray  # Image as numpy array (H, W, 3)
    file_size_bytes: int
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
        return None
    
    # File size validation (AC: #4)
    if uploaded_file.size > MAX_FILE_SIZE_BYTES:
        st.error(
            f"File too large ({uploaded_file.size / (1024 * 1024):.1f}MB). "
            f"Maximum size is {MAX_FILE_SIZE_MB}MB."
        )
        logger.warning(
            f"Upload rejected: file too large ({uploaded_file.size} bytes)"
        )
        return None
    
    # Load and convert image (AC: #1, #3)
    try:
        img = Image.open(uploaded_file).convert("RGB")
        img_array = np.array(img)
    except Exception as e:
        st.error(f"Failed to load image: {e}")
        logger.error(f"Image load error: {e}")
        return None
    
    # Create UploadedImage dataclass (AC: #2)
    uploaded_image = UploadedImage(
        filename=uploaded_file.name,
        data=img_array,
        file_size_bytes=uploaded_file.size,
    )
    
    # Store in session state (AC: #2, #6 - replacement)
    st.session_state["uploaded_image"] = uploaded_image
    logger.info(
        f"Image uploaded successfully: {uploaded_file.name} "
        f"({uploaded_image.file_size_mb:.2f}MB, {uploaded_image.dimensions})"
    )
    
    return uploaded_image
