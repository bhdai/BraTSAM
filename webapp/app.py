"""BraTSAM Streamlit Web Application Entry Point.

This module serves as the main entry point for the BraTSAM web application,
providing a Streamlit-based interface for brain tumor segmentation workflows.
"""

import logging

import streamlit as st

from webapp.components.upload import render_upload_component

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
        st.success("✅ Ready to process")
        st.image(uploaded.data, caption=uploaded.filename, use_container_width=True)
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Filename", uploaded.filename)
        col2.metric("Size", f"{uploaded.file_size_mb:.2f} MB")
        col3.metric("Dimensions", f"{uploaded.dimensions[1]}x{uploaded.dimensions[0]}")
    
    logger.info("BraTSAM main page rendered")


if __name__ == "__main__":
    main()
