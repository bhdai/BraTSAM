"""BraTSAM Streamlit Web Application Entry Point.

This module serves as the main entry point for the BraTSAM web application,
providing a Streamlit-based interface for brain tumor segmentation workflows.
"""

import logging

import numpy as np
import streamlit as st

from preprocessing.normalize import normalize_slice
from preprocessing.volume import extract_slice
from webapp.components.slice_selector import render_slice_selector
from webapp.components.upload import UploadedImage, UploadedVolume, render_upload_component

# Configure logging (backend)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Page configuration - MUST be first Streamlit command
st.set_page_config(
    page_title="BraTSAM",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


def main() -> None:
    """Main entry point for the BraTSAM application."""
    logger.info("BraTSAM application started")
    
    st.title("🧠 BraTSAM")
    st.markdown("**Brain Tumor Segmentation with SAM**")
    
    st.sidebar.header("Navigation")
    st.sidebar.info("Welcome to BraTSAM! Feature components will be added here.")
    
    st.markdown(
        """
        ## Welcome to BraTSAM
        
        This application provides a streamlined workflow for brain tumor 
        segmentation using state-of-the-art deep learning models:
        
        - **YOLO** for automated tumor detection
        - **SAM** (Segment Anything Model) for precise segmentation
        
        ### Getting Started
        
        Upload a brain MRI slice below to begin analysis.
        """
    )
    
    # Upload section (AC: #1-6)
    st.header("📤 Upload Image")
    uploaded = render_upload_component()
    
    if uploaded:
        if isinstance(uploaded, UploadedVolume):
            # 3D Volume handling (AC: #1, #2, #3, #4, #5)
            st.info(
                f"🔬 3D volume detected: "
                f"{uploaded.dimensions[0]}×{uploaded.dimensions[1]}×{uploaded.dimensions[2]}"
            )
            
            # Slice selector (AC: #3, #4) - pass file_id to reset on new volume
            slice_idx = render_slice_selector(uploaded.num_slices, uploaded.file_id)
            
            # Extract and normalize slice (AC: #2, #4)
            slice_2d = extract_slice(uploaded.volume_data, slice_idx)
            normalized = normalize_slice(slice_2d)
            
            # Convert grayscale to RGB for display
            display_img = np.stack([normalized, normalized, normalized], axis=-1)
            
            st.image(
                display_img,
                caption=f"{uploaded.filename} - Slice {slice_idx}",
                use_container_width=True,
            )
            
            # Metadata display (AC: #5)
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Filename", uploaded.filename)
            col2.metric("Size", f"{uploaded.file_size_mb:.2f} MB")
            col3.metric(
                "Dimensions",
                f"{uploaded.dimensions[0]}×{uploaded.dimensions[1]}×{uploaded.dimensions[2]}",
            )
            col4.metric("Current Slice", f"{slice_idx + 1}/{uploaded.num_slices}")
        else:
            # 2D Image handling (existing behavior) (AC: #6)
            st.success("✅ Ready to process")
            st.image(uploaded.data, caption=uploaded.filename, use_container_width=True)
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Filename", uploaded.filename)
            col2.metric("Size", f"{uploaded.file_size_mb:.2f} MB")
            col3.metric("Dimensions", f"{uploaded.dimensions[1]}x{uploaded.dimensions[0]}")
    
    logger.info("BraTSAM main page rendered")


if __name__ == "__main__":
    main()
