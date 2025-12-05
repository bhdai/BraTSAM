"""BraTSAM Streamlit Web Application Entry Point.

This module serves as the main entry point for the BraTSAM web application,
providing a Streamlit-based interface for brain tumor segmentation workflows.
"""

import logging

import numpy as np
import streamlit as st
from PIL import Image

from preprocessing.normalize import normalize_slice
from preprocessing.volume import extract_slice
from webapp.components.confidence_indicator import (
    render_confidence_indicator,
    render_confidence_score,
)
from webapp.components.slice_selector import render_slice_selector
from webapp.components.upload import UploadedImage, UploadedVolume, render_upload_component
from webapp.components.viewer import render_image_viewer
from webapp.utils.inference import PipelineResult, load_models, run_pipeline

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


def display_inference_results(result: PipelineResult) -> None:
    """Display inference results with confidence indicator (AC #3, #5).

    Shows the triage classification badge, confidence score, and
    relevant result details based on pipeline outcome.

    Args:
        result: PipelineResult from the inference pipeline.
    """
    st.subheader("📊 Inference Results")

    # Display confidence indicator badge (AC #3)
    col1, col2 = st.columns([1, 3])

    with col1:
        render_confidence_indicator(result)

    with col2:
        render_confidence_score(result)

    # Display threshold explanation tooltip
    st.markdown(
        """
        <div style="font-size: 0.8em; color: #666; margin-top: 8px;">
        <b>Thresholds:</b> ✅ Auto-Approved ≥90% | ⚠️ Needs Review 50-89% | 🔴 Manual Required &lt;50%
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Display additional result details
    if result.success:
        st.success(f"✅ Pipeline completed at stage: {result.stage}")

        if result.yolo_confidence is not None:
            st.markdown(f"**YOLO Detection Confidence:** {result.yolo_confidence:.2%}")

        if result.sam_iou is not None:
            st.markdown(f"**SAM Segmentation IoU:** {result.sam_iou:.2%}")
    else:
        st.error(f"❌ Pipeline failed at stage: {result.stage}")
        if result.error_message:
            st.warning(f"**Error:** {result.error_message}")


def main() -> None:
    """Main entry point for the BraTSAM application."""
    logger.info("BraTSAM application started")
    
    st.title("🧠 BraTSAM")
    st.markdown("**Brain Tumor Segmentation with SAM**")
    
    st.sidebar.header("Navigation")
    st.sidebar.info("Upload a brain MRI to begin automatic tumor detection and segmentation.")
    
    st.markdown(
        """
        ## Welcome to BraTSAM
        
        This application provides automated brain tumor segmentation:
        
        - **YOLO** for automated tumor detection
        - **SAM** (Segment Anything Model) for precise segmentation
        
        ### Getting Started
        
        Upload a brain MRI slice below to begin analysis.
        """
    )
    
    # Upload section
    st.header("📤 Upload Image")
    uploaded = render_upload_component()
    
    if uploaded:
        # Initialize overlay toggle states in session state
        if "show_bounding_box" not in st.session_state:
            st.session_state.show_bounding_box = True
        if "show_segmentation_mask" not in st.session_state:
            st.session_state.show_segmentation_mask = True

        # Overlay toggle controls
        st.subheader("🎛️ Display Options")
        toggle_col1, toggle_col2 = st.columns(2)
        with toggle_col1:
            st.checkbox("Show Bounding Box", key="show_bounding_box")
        with toggle_col2:
            st.checkbox("Show Mask", key="show_segmentation_mask")

        if isinstance(uploaded, UploadedVolume):
            # 3D Volume handling
            st.info(
                f"🔬 3D volume detected: "
                f"{uploaded.dimensions[0]}×{uploaded.dimensions[1]}×{uploaded.dimensions[2]}"
            )
            
            # Slice selector - pass file_id to reset on new volume
            slice_idx = render_slice_selector(uploaded.num_slices, uploaded.file_id)
            
            # Extract and normalize slice
            slice_2d = extract_slice(uploaded.volume_data, slice_idx)
            normalized = normalize_slice(slice_2d)
            
            # Convert grayscale to RGB for display and inference
            display_img = np.stack([normalized, normalized, normalized], axis=-1)
            
            # Create a unique key for this slice to cache results
            slice_key = f"{uploaded.file_id}_slice_{slice_idx}"
            
            # Run inference automatically if not cached
            if f"result_{slice_key}" not in st.session_state:
                with st.spinner("🔬 Running YOLO detection and SAM segmentation..."):
                    try:
                        yolo, sam_processor, sam_model = load_models()
                        result = run_pipeline(display_img, yolo, sam_processor, sam_model)
                        st.session_state[f"result_{slice_key}"] = result
                    except Exception as e:
                        logger.error(f"Inference failed: {e}")
                        st.error(f"Inference failed: {e}")
                        result = None
            else:
                result = st.session_state[f"result_{slice_key}"]
            
            # Display the image with overlays
            st.subheader("📊 Segmentation Result")
            render_image_viewer(
                display_img,
                result=result,
                caption=f"{uploaded.filename} - Slice {slice_idx}",
                show_box=st.session_state.show_bounding_box,
                show_mask=st.session_state.show_segmentation_mask,
            )
            
            # Display inference results if available
            if result:
                display_inference_results(result)
            
            # Metadata display
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Filename", uploaded.filename)
            col2.metric("Size", f"{uploaded.file_size_mb:.2f} MB")
            col3.metric(
                "Dimensions",
                f"{uploaded.dimensions[0]}×{uploaded.dimensions[1]}×{uploaded.dimensions[2]}",
            )
            col4.metric("Current Slice", f"{slice_idx + 1}/{uploaded.num_slices}")
        else:
            # 2D Image handling
            st.success("✅ 2D image loaded - processing...")
            
            # Ensure image is numpy array for inference
            if isinstance(uploaded.data, np.ndarray):
                img_array = uploaded.data
            else:
                img_array = np.array(uploaded.data)
            
            # Ensure RGB format
            if img_array.ndim == 2:
                img_array = np.stack([img_array, img_array, img_array], axis=-1)
            elif img_array.shape[-1] == 4:  # RGBA
                img_array = img_array[:, :, :3]
            
            # Create a unique key for this image to cache results
            image_key = f"result_{uploaded.file_id}"
            
            # Run inference automatically if not cached
            if image_key not in st.session_state:
                with st.spinner("🔬 Running YOLO detection and SAM segmentation..."):
                    try:
                        yolo, sam_processor, sam_model = load_models()
                        result = run_pipeline(img_array, yolo, sam_processor, sam_model)
                        st.session_state[image_key] = result
                    except Exception as e:
                        logger.error(f"Inference failed: {e}")
                        st.error(f"Inference failed: {e}")
                        result = None
            else:
                result = st.session_state[image_key]
            
            # Display the image with overlays
            st.subheader("📊 Segmentation Result")
            render_image_viewer(
                img_array,
                result=result,
                caption=uploaded.filename,
                show_box=st.session_state.show_bounding_box,
                show_mask=st.session_state.show_segmentation_mask,
            )
            
            # Display inference results if available
            if result:
                display_inference_results(result)
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Filename", uploaded.filename)
            col2.metric("Size", f"{uploaded.file_size_mb:.2f} MB")
            col3.metric("Dimensions", f"{uploaded.dimensions[1]}x{uploaded.dimensions[0]}")
    
    logger.info("BraTSAM main page rendered")


if __name__ == "__main__":
    main()
