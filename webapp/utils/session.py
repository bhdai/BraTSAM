"""Session state management for batch processing (Story 4.1).

This module manages Streamlit session state for the batch processing queue,
providing functions to initialize, access, and modify the batch queue state.

All batch queue state is stored in st.session_state to persist across
Streamlit reruns within a session.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

import streamlit as st

from webapp.utils.batch import BatchItem
from webapp.utils.inference import PipelineResult

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Session state key constants
SESSION_KEYS = {
    "batch_queue": "batch_queue",
    "current_index": "current_batch_index",
    "batch_initialized": "batch_initialized",
}

# Valid status values for batch items
VALID_STATUSES = {"pending", "approved", "rejected", "edited"}


def initialize_batch_state() -> None:
    """Initialize batch processing state in session.

    Safe to call multiple times - only sets defaults if not present.
    Should be called at app startup before any batch operations.
    """
    if SESSION_KEYS["batch_initialized"] not in st.session_state:
        st.session_state[SESSION_KEYS["batch_queue"]] = []
        st.session_state[SESSION_KEYS["current_index"]] = 0
        st.session_state[SESSION_KEYS["batch_initialized"]] = True
        logger.debug("Batch state initialized")


def get_batch_queue() -> list[BatchItem]:
    """Get the batch queue from session state.

    Initializes state if not already done. Returns the current
    batch queue list (mutable reference).

    Returns:
        List of BatchItem objects in the queue.
    """
    initialize_batch_state()
    return st.session_state[SESSION_KEYS["batch_queue"]]


def add_to_batch(
    filename: str,
    result: PipelineResult | None = None,
    slice_index: int | None = None,
) -> int:
    """Add a new item to the batch queue.

    Creates a new BatchItem with the given filename and optional result,
    and appends it to the queue.

    Args:
        filename: Name or identifier for the image/slice.
        result: Optional PipelineResult from inference.
        slice_index: Optional slice index for 3D volume tracking.

    Returns:
        Index of the newly added item in the queue.
    """
    queue = get_batch_queue()

    # Create a placeholder result if none provided
    if result is None:
        result = PipelineResult(success=False, stage="upload")

    item = BatchItem(
        filename=filename,
        result=result,
        status="pending",
        slice_index=slice_index,
        timestamp=time.time(),
    )
    queue.append(item)
    index = len(queue) - 1
    logger.debug(f"Added item to batch: {filename} at index {index}")
    return index


def clear_batch_queue() -> None:
    """Clear the entire batch queue and reset index.

    Removes all items from the queue and resets the current
    index to 0.
    """
    initialize_batch_state()
    st.session_state[SESSION_KEYS["batch_queue"]] = []
    st.session_state[SESSION_KEYS["current_index"]] = 0
    logger.debug("Batch queue cleared")


def get_batch_size() -> int:
    """Get the number of items in the batch queue.

    Returns:
        Current queue length.
    """
    queue = get_batch_queue()
    return len(queue)


def update_status(index: int, new_status: str) -> bool:
    """Update status of batch item at index.

    Args:
        index: Zero-based index of item to update.
        new_status: New status value.

    Returns:
        True if update successful, False if index out of bounds.

    Raises:
        ValueError: If new_status is not a valid status string.
    """
    if new_status not in VALID_STATUSES:
        raise ValueError(
            f"Invalid status: {new_status}. Must be one of {VALID_STATUSES}"
        )

    queue = get_batch_queue()
    if 0 <= index < len(queue):
        old_status = queue[index].status
        queue[index].status = new_status
        logger.debug(f"Updated item {index} status: {old_status} -> {new_status}")
        return True
    return False


def get_next_pending_index() -> int | None:
    """Find the index of the first pending item.

    Scans the queue from the beginning to find the first item
    with status "pending".

    Returns:
        Index of first pending item, or None if no pending items exist.
    """
    queue = get_batch_queue()
    for i, item in enumerate(queue):
        if item.status == "pending":
            return i
    return None


def get_current_index() -> int:
    """Get the currently selected batch item index.

    Returns:
        Current batch index (0 if not set).
    """
    initialize_batch_state()
    return st.session_state[SESSION_KEYS["current_index"]]


def set_current_index(index: int) -> bool:
    """Set the currently selected batch item index.

    Args:
        index: Index to set as current.

    Returns:
        True if successful, False if index out of bounds.
    """
    queue = get_batch_queue()
    if len(queue) == 0:
        # Allow setting to 0 for empty queue
        if index == 0:
            st.session_state[SESSION_KEYS["current_index"]] = 0
            return True
        return False

    if 0 <= index < len(queue):
        st.session_state[SESSION_KEYS["current_index"]] = index
        logger.debug(f"Set current index to {index}")
        return True
    return False


def navigate_next() -> bool:
    """Move to the next item in the queue.

    Returns:
        True if navigation successful, False if at end of queue.
    """
    current = get_current_index()
    queue_size = get_batch_size()

    if current < queue_size - 1:
        return set_current_index(current + 1)
    return False


def navigate_previous() -> bool:
    """Move to the previous item in the queue.

    Returns:
        True if navigation successful, False if at start of queue.
    """
    current = get_current_index()

    if current > 0:
        return set_current_index(current - 1)
    return False


def get_batch_item(index: int) -> BatchItem | None:
    """Safely get a batch item by index.

    Args:
        index: Index of item to retrieve.

    Returns:
        BatchItem at index, or None if index out of bounds.
    """
    queue = get_batch_queue()
    if 0 <= index < len(queue):
        return queue[index]
    return None


def get_current_item() -> BatchItem | None:
    """Get the currently selected batch item.

    Returns:
        Currently selected BatchItem, or None if queue is empty.
    """
    index = get_current_index()
    return get_batch_item(index)


def update_batch_item_result(index: int, result: PipelineResult) -> bool:
    """Update the result field of a batch item.

    Args:
        index: Index of item to update.
        result: New PipelineResult to store.

    Returns:
        True if update successful, False if index out of bounds.
    """
    queue = get_batch_queue()
    if 0 <= index < len(queue):
        queue[index].result = result
        logger.debug(f"Updated result for item {index}")
        return True
    return False


def get_batch_statistics():
    """Get statistics for the current batch queue.

    Convenience wrapper around compute_batch_statistics()
    that uses the session queue.

    Returns:
        BatchStatistics for the current queue.
    """
    from webapp.utils.batch import compute_batch_statistics

    queue = get_batch_queue()
    return compute_batch_statistics(queue)
