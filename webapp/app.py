"""BraTSAM Streamlit Web Application Entry Point.

This module serves as the main entry point for the BraTSAM web application,
providing a Streamlit-based interface for brain tumor segmentation workflows.
"""

import logging

import streamlit as st

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
        
        Use the sidebar navigation to access different features as they 
        become available.
        """
    )
    
    logger.info("BraTSAM main page rendered")


if __name__ == "__main__":
    main()
