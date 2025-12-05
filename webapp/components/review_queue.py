"""Review queue UI component for Queue Master panel (Story 4.2).

This module provides the Queue Master UI component that displays batch items
in a scrollable list with filtering, selection highlighting, and progress tracking.

The queue panel shows:
- Progress tracking (X/Y reviewed)
- Filter tabs (All | Review | Manual)
- Scrollable list of queue item cards with thumbnails and confidence indicators
"""

from __future__ import annotations

import base64
import io
import logging
from typing import TYPE_CHECKING

import numpy as np
import streamlit as st
from PIL import Image

from webapp.components.confidence_indicator import CONFIDENCE_COLORS
from webapp.utils.batch import BatchItem, get_items_by_classification
from webapp.utils.session import (
    SESSION_KEYS,
    get_batch_queue,
    get_batch_statistics,
    get_current_index,
    set_current_index,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Filter key for session state
FILTER_KEY = "queue_filter"

# Color constants from UX Design Specification
SELECTED_BORDER_COLOR = "#1E40AF"  # Deep Blue
SELECTED_BG_COLOR = "#EFF6FF"  # Light Blue
DEFAULT_BORDER_COLOR = "#E2E8F0"  # Slate
HOVER_BG_COLOR = "#F1F5F9"  # Gray

# Confidence dot colors
DOT_COLORS = {
    "auto-approved": "#22C55E",  # Green
    "needs-review": "#F59E0B",  # Amber
    "manual-required": "#EF4444",  # Red
}

# Confidence dot emojis
DOT_EMOJIS = {
    "auto-approved": "🟢",
    "needs-review": "🟡",
    "manual-required": "🔴",
}


def truncate_filename(filename: str, max_length: int = 20) -> str:
    """Truncate filename to specified length with ellipsis.

    Args:
        filename: Original filename.
        max_length: Maximum length before truncation.

    Returns:
        Truncated filename with '...' if longer than max_length.
    """
    if len(filename) <= max_length:
        return filename
    return filename[:max_length] + "..."


def generate_thumbnail(
    image_array: np.ndarray,
    size: tuple[int, int] = (48, 48),
) -> bytes:
    """Generate a thumbnail from numpy array.

    Args:
        image_array: Source image as numpy array.
        size: Thumbnail dimensions (width, height).

    Returns:
        PNG bytes of thumbnail.
    """
    if image_array.ndim == 2:
        # Grayscale
        img = Image.fromarray(image_array)
    else:
        img = Image.fromarray(image_array)

    img.thumbnail(size, Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue()


def get_confidence_dot(classification: str) -> str:
    """Get the confidence dot emoji for a classification.

    Args:
        classification: One of "auto-approved", "needs-review", "manual-required".

    Returns:
        Emoji string representing the confidence level.
    """
    return DOT_EMOJIS.get(classification, "⚪")


def filter_items(items: list[BatchItem], filter_type: str) -> list[BatchItem]:
    """Filter batch items by classification type.

    Args:
        items: List of BatchItem objects to filter.
        filter_type: One of "all", "needs-review", "manual-required".

    Returns:
        Filtered list of BatchItem objects.
    """
    if filter_type == "all":
        return items
    return get_items_by_classification(items, filter_type)


def handle_item_click(index: int) -> None:
    """Handle click on a queue item.

    Updates the current batch index and triggers a Streamlit rerun
    to update the main viewer.

    Args:
        index: Index of the clicked item.
    """
    set_current_index(index)
    st.rerun()


def render_progress_header() -> None:
    """Render progress tracking in queue header.

    Displays the "✓ X/Y reviewed" format showing how many items
    have been processed vs total items.
    """
    stats = get_batch_statistics()
    reviewed = stats.processed
    total = stats.total

    st.markdown(f"**✓ {reviewed}/{total} reviewed**")


def render_empty_state() -> None:
    """Render empty state message when queue is empty.

    Shows a message indicating no images are in the queue
    and instructions to upload images.
    """
    st.info("📭 No images in queue")
    st.markdown("Upload images to begin processing and review.")


def render_queue_item(
    item: BatchItem,
    index: int,
    is_selected: bool,
) -> None:
    """Render a single queue item card.

    Displays thumbnail, filename, confidence indicator, and approval status
    with appropriate styling based on selection and approval state.

    Args:
        item: BatchItem to render.
        index: Index of the item in the queue.
        is_selected: Whether this item is currently selected.
    """
    classification = item.classification
    is_approved = item.status == "approved"

    # Get confidence dot
    dot = get_confidence_dot(classification)

    # Truncate filename
    display_name = truncate_filename(item.filename)

    # Selection and approval styles
    border_style = f"border-left: 4px solid {SELECTED_BORDER_COLOR};" if is_selected else f"border-left: 4px solid transparent;"
    bg_color = SELECTED_BG_COLOR if is_selected else "#FFFFFF"
    opacity_style = "opacity: 0.7;" if is_approved else ""

    # Checkmark for approved items
    checkmark = "✓" if is_approved else ""

    # Generate thumbnail
    thumbnail_html = ""
    if item.result.sam_mask is not None:
        try:
            thumb_bytes = generate_thumbnail(item.result.sam_mask)
            thumb_b64 = base64.b64encode(thumb_bytes).decode()
            thumbnail_html = f'<img src="data:image/png;base64,{thumb_b64}" width="48" height="48" style="border-radius: 4px; object-fit: cover;" />'
        except Exception as e:
            logger.debug(f"Failed to generate thumbnail: {e}")
            thumbnail_html = '<div style="width: 48px; height: 48px; background: #E2E8F0; border-radius: 4px; display: flex; align-items: center; justify-content: center;">📷</div>'
    else:
        thumbnail_html = '<div style="width: 48px; height: 48px; background: #E2E8F0; border-radius: 4px; display: flex; align-items: center; justify-content: center;">📷</div>'

    # Confidence percentage
    conf_text = ""
    if item.confidence is not None:
        conf_text = f"{item.confidence:.0%}"

    # Build card HTML
    card_html = f"""
    <div style="
        {border_style}
        background-color: {bg_color};
        {opacity_style}
        padding: 8px;
        margin-bottom: 4px;
        border-radius: 4px;
        cursor: pointer;
        display: flex;
        gap: 8px;
        align-items: center;
    ">
        {thumbnail_html}
        <div style="flex: 1; min-width: 0;">
            <div style="font-size: 0.85em; font-weight: 500; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">
                {display_name}
            </div>
            <div style="font-size: 0.75em; color: #666;">
                {conf_text} {dot} {checkmark}
            </div>
        </div>
    </div>
    """

    st.markdown(card_html, unsafe_allow_html=True)

    # Clickable button overlay (using unique key)
    if st.button(
        f"Select {display_name}",
        key=f"queue_item_{index}",
        help=f"Click to view {item.filename}",
        use_container_width=True,
        type="secondary",
    ):
        handle_item_click(index)


def render_filter_tabs() -> str:
    """Render filter tabs and return active filter.

    Displays tabs for "All", "🟡 Review", "🔴 Manual" filters.

    Returns:
        Active filter type: "all", "needs-review", or "manual-required".
    """
    # Initialize filter state
    if FILTER_KEY not in st.session_state:
        st.session_state[FILTER_KEY] = "all"

    tab_all, tab_review, tab_manual = st.tabs(["All", "🟡 Review", "🔴 Manual"])

    with tab_all:
        if st.session_state[FILTER_KEY] != "all":
            st.session_state[FILTER_KEY] = "all"

    with tab_review:
        if st.session_state[FILTER_KEY] != "needs-review":
            st.session_state[FILTER_KEY] = "needs-review"

    with tab_manual:
        if st.session_state[FILTER_KEY] != "manual-required":
            st.session_state[FILTER_KEY] = "manual-required"

    return st.session_state[FILTER_KEY]


def render_review_queue() -> None:
    """Render the complete review queue panel (AC #1).

    This is the main entry point for the Queue Master component.
    Renders the progress header, filter tabs, and scrollable list
    of queue items with selection highlighting.
    """
    queue = get_batch_queue()
    current_index = get_current_index()

    # Render progress header (AC #7)
    render_progress_header()

    # Check for empty queue (AC #8)
    if not queue:
        render_empty_state()
        return

    # Render filter tabs (AC #6)
    # Note: Using a simpler approach due to Streamlit tabs limitation
    # The active filter is tracked in session state
    if FILTER_KEY not in st.session_state:
        st.session_state[FILTER_KEY] = "all"

    tab_all, tab_review, tab_manual = st.tabs(["All", "🟡 Review", "🔴 Manual"])

    def render_queue_list(items: list[BatchItem], original_queue: list[BatchItem]) -> None:
        """Render the filtered queue list."""
        if not items:
            st.caption("No items match this filter.")
            return

        # Create a scrollable container
        with st.container(height=400):
            for item in items:
                # Find original index in full queue
                original_index = original_queue.index(item)
                is_selected = original_index == current_index
                render_queue_item(item, original_index, is_selected)

    with tab_all:
        render_queue_list(queue, queue)

    with tab_review:
        filtered = filter_items(queue, "needs-review")
        render_queue_list(filtered, queue)

    with tab_manual:
        filtered = filter_items(queue, "manual-required")
        render_queue_list(filtered, queue)
