"""Slice selector component for 3D volume navigation.

This module provides a slider component for navigating through slices
of a 3D volumetric image (NIfTI) in the BraTSAM web application.
"""

import streamlit as st


def render_slice_selector(num_slices: int) -> int:
    """Render slice selector slider and return selected index.
    
    Displays a slider widget allowing users to navigate through slices
    of a 3D volume. Stores the selected index in session state for
    persistence across reruns.
    
    Args:
        num_slices: Total number of slices in the volume.
    
    Returns:
        Currently selected slice index (0-indexed).
    
    Example:
        >>> # In Streamlit app with 155-slice volume
        >>> slice_idx = render_slice_selector(155)
        >>> print(f"Selected slice: {slice_idx}")
    """
    # Initialize session state if needed
    if "current_slice_idx" not in st.session_state:
        st.session_state["current_slice_idx"] = num_slices // 2
    
    # Ensure current index is within valid range (handles volume changes)
    if st.session_state["current_slice_idx"] >= num_slices:
        st.session_state["current_slice_idx"] = num_slices // 2
    
    selected_idx = st.slider(
        "Select Slice",
        min_value=0,
        max_value=num_slices - 1,
        value=st.session_state["current_slice_idx"],
        key="slice_slider",
        help=f"Navigate through {num_slices} slices (0 to {num_slices - 1})",
    )
    
    st.session_state["current_slice_idx"] = selected_idx
    return selected_idx
