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
from webapp.components.canvas import (
    cancel_edit_mode,
    clip_bbox_to_bounds,
    extract_bbox_from_canvas,
    init_edit_mode_state,
    is_edit_mode,
    render_drawing_canvas,
    toggle_edit_mode,
    validate_bbox,
)
from webapp.components.confidence_indicator import (
    render_confidence_indicator,
    render_confidence_score,
)
from webapp.components.slice_selector import render_slice_selector
from webapp.components.upload import UploadedImage, UploadedVolume, render_upload_component
from webapp.components.viewer import render_image_viewer, render_interactive_viewer
from webapp.utils.inference import PipelineResult, run_segmentation_only

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
        # Initialize overlay toggle states in session state
        if "show_bounding_box" not in st.session_state:
            st.session_state.show_bounding_box = True
        if "show_segmentation_mask" not in st.session_state:
            st.session_state.show_segmentation_mask = True
        # Initialize zoom/pan toggle state (Story 3.3)
        if "enable_zoom_pan" not in st.session_state:
            st.session_state.enable_zoom_pan = True

        # Queue Master layout: sidebar for review queue (Story 4.2), main area for viewer
        queue_col, viewer_col = st.columns([1, 3])
        
        with queue_col:
            st.markdown("### 📋 Review Queue")
            st.caption("Queue functionality coming in Epic 4")
            # Placeholder for Story 4.2 - batch review queue
        
        with viewer_col:
            # Initialize edit mode state (Story 3.4)
            init_edit_mode_state()

            # Overlay toggle controls (Story 3.2, AC #3, #4) + Edit Mode (Story 3.4, AC #1, #6)
            toggle_col1, toggle_col2, toggle_col3, toggle_col4 = st.columns(4)
            with toggle_col1:
                st.checkbox("Show Bounding Box", key="show_bounding_box")
            with toggle_col2:
                st.checkbox("Show Mask", key="show_segmentation_mask")
            with toggle_col3:
                st.checkbox("Enable Zoom/Pan", key="enable_zoom_pan")
            with toggle_col4:
                # Edit mode toggle button (AC #1, #6)
                if is_edit_mode():
                    if st.button("❌ Cancel Edit", key="cancel_edit_btn"):
                        cancel_edit_mode()
                        st.rerun()
                else:
                    if st.button("✏️ Edit", key="edit_mode_btn", help="Draw manual bounding box (E)"):
                        toggle_edit_mode()
                        st.rerun()

            # Edit mode indicator (AC #1)
            if is_edit_mode():
                st.warning("📝 **Edit Mode Active** - Draw a bounding box around the tumor")

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

                # Edit mode: show drawing canvas (Story 3.4, AC #2)
                if is_edit_mode():
                    # Convert to PIL for canvas background
                    pil_image = Image.fromarray(display_img)
                    canvas_result = render_drawing_canvas(pil_image)

                    # Process canvas result (AC #2, #3)
                    if canvas_result and canvas_result.json_data:
                        bbox = extract_bbox_from_canvas(canvas_result)
                        if bbox and validate_bbox(bbox, pil_image.width, pil_image.height):
                            # Clip bbox to image bounds
                            bbox = clip_bbox_to_bounds(bbox, pil_image.width, pil_image.height)
                            st.info(f"📦 Manual box: {bbox}")

                            # Run segmentation button (AC #3)
                            if st.button("🔬 Run Segmentation", key="run_seg_volume"):
                                with st.spinner("Running SAM segmentation..."):
                                    result = run_segmentation_only(display_img, bbox)
                                    st.session_state.manual_result = result
                                    cancel_edit_mode()
                                    st.rerun()
                else:
                    # Normal viewer mode (Story 3.1, 3.2, 3.3)
                    result = st.session_state.get("manual_result")
                    render_interactive_viewer(
                        display_img,
                        result=result,
                        caption=f"{uploaded.filename} - Slice {slice_idx}",
                        show_box=st.session_state.show_bounding_box,
                        show_mask=st.session_state.show_segmentation_mask,
                        enable_zoom_pan=st.session_state.enable_zoom_pan,
                    )

                    # Display inference results if available (AC #4)
                    if result and result.success:
                        display_inference_results(result)
                
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

                # Edit mode: show drawing canvas (Story 3.4, AC #2)
                if is_edit_mode():
                    # Convert to PIL for canvas background
                    if isinstance(uploaded.data, np.ndarray):
                        pil_image = Image.fromarray(uploaded.data)
                    else:
                        pil_image = uploaded.data

                    canvas_result = render_drawing_canvas(pil_image)

                    # Process canvas result (AC #2, #3)
                    if canvas_result and canvas_result.json_data:
                        bbox = extract_bbox_from_canvas(canvas_result)
                        if bbox and validate_bbox(bbox, pil_image.width, pil_image.height):
                            # Clip bbox to image bounds
                            bbox = clip_bbox_to_bounds(bbox, pil_image.width, pil_image.height)
                            st.info(f"📦 Manual box: {bbox}")

                            # Run segmentation button (AC #3)
                            if st.button("🔬 Run Segmentation", key="run_seg_2d"):
                                with st.spinner("Running SAM segmentation..."):
                                    # Ensure image is numpy array for inference
                                    if isinstance(uploaded.data, np.ndarray):
                                        img_array = uploaded.data
                                    else:
                                        img_array = np.array(uploaded.data)
                                    result = run_segmentation_only(img_array, bbox)
                                    st.session_state.manual_result = result
                                    cancel_edit_mode()
                                    st.rerun()
                else:
                    # Normal viewer mode (Story 3.1, 3.2, 3.3)
                    result = st.session_state.get("manual_result")
                    render_interactive_viewer(
                        uploaded.data,
                        result=result,
                        caption=uploaded.filename,
                        show_box=st.session_state.show_bounding_box,
                        show_mask=st.session_state.show_segmentation_mask,
                        enable_zoom_pan=st.session_state.enable_zoom_pan,
                    )

                    # Display inference results if available (AC #4)
                    if result and result.success:
                        display_inference_results(result)
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Filename", uploaded.filename)
                col2.metric("Size", f"{uploaded.file_size_mb:.2f} MB")
                col3.metric("Dimensions", f"{uploaded.dimensions[1]}x{uploaded.dimensions[0]}")
    
    logger.info("BraTSAM main page rendered")


if __name__ == "__main__":
    main()
