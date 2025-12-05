"""Keyboard shortcuts module for three-tier review workflow (Story 4.3).

This module provides keyboard shortcut handling for efficient batch review:
- SPACE: Approve current item and auto-advance
- E: Enter Edit Mode for manual bounding box
- A/←: Navigate to previous item
- D/→: Navigate to next item  
- ESC: Cancel Edit Mode

The streamlit-shortcuts library binds keyboard shortcuts to Streamlit UI elements
(buttons with specific keys), triggering click events when the shortcut is pressed.

Example:
    >>> from webapp.utils.shortcuts import register_shortcuts, render_shortcut_help
    >>> register_shortcuts()  # Call after creating buttons with matching keys
"""

from __future__ import annotations

import logging

import streamlit as st
from streamlit_shortcuts import add_shortcuts

from webapp.components.canvas import cancel_edit_mode, is_edit_mode, toggle_edit_mode
from webapp.utils.session import (
    get_batch_queue,
    get_current_index,
    navigate_next,
    navigate_previous,
    set_current_index,
    update_status,
)

logger = logging.getLogger(__name__)

# Shortcut key mappings (AC #1-6)
# Maps button keys to keyboard shortcuts
SHORTCUTS: dict[str, str | list[str]] = {
    "btn_approve": " ",  # SPACE = Approve
    "btn_edit": "e",  # E = Edit Mode
    "btn_prev": ["a", "ArrowLeft"],  # A/← = Previous
    "btn_next": ["d", "ArrowRight"],  # D/→ = Next
    "btn_cancel": "Escape",  # ESC = Cancel Edit Mode
}


def register_shortcuts() -> None:
    """Register all keyboard shortcuts for button elements.

    Binds keyboard shortcuts to buttons with matching keys.
    Must be called after buttons are created with the corresponding keys.
    """
    add_shortcuts(**SHORTCUTS)


def navigate_to_next_pending() -> bool:
    """Advance to next item needing review (AC #7).

    Skips already-approved items. Wraps around to beginning if needed.

    Returns:
        True if advanced to a pending item, False if all reviewed.
    """
    queue = get_batch_queue()
    current = get_current_index()
    queue_len = len(queue)

    if queue_len == 0:
        return False

    # Search forward from current position
    for i in range(current + 1, queue_len):
        if queue[i].status == "pending":
            set_current_index(i)
            logger.debug(f"Auto-advanced to pending item at index {i}")
            return True

    # Wrap around: search from beginning
    for i in range(0, current):
        if queue[i].status == "pending":
            set_current_index(i)
            logger.debug(f"Auto-advanced (wrapped) to pending item at index {i}")
            return True

    # All items reviewed
    logger.debug("No pending items remaining in queue")
    return False


def handle_approve_shortcut() -> None:
    """Handle SPACE key approval (AC #1, #5).

    - Updates current item status to "approved"
    - Auto-advances to next pending item
    - Triggers rerun to update UI

    Does nothing if in Edit Mode (AC #8).
    """
    # Guard: Don't approve while in Edit Mode
    if is_edit_mode():
        logger.debug("Approve shortcut blocked - Edit Mode active")
        return

    current_idx = get_current_index()

    # Mark as approved
    update_status(current_idx, "approved")
    logger.info(f"Approved item at index {current_idx}")

    # Auto-advance to next pending item
    navigate_to_next_pending()

    # Trigger UI update
    st.rerun()


def handle_navigation_shortcut(direction: str) -> None:
    """Handle A/D/←/→ navigation shortcuts (AC #3, #4).

    Args:
        direction: "previous" or "next"

    Does nothing if in Edit Mode (AC #8).
    """
    # Guard: Don't navigate while in Edit Mode
    if is_edit_mode():
        logger.debug("Navigation shortcut blocked - Edit Mode active")
        return

    if direction == "previous":
        navigate_previous()
        logger.debug("Navigated to previous item")
    else:
        navigate_next()
        logger.debug("Navigated to next item")

    st.rerun()


def handle_edit_shortcut() -> None:
    """Handle E key for Edit Mode (AC #2, #6).

    Toggles Edit Mode to enable bounding box drawing.
    """
    toggle_edit_mode()
    logger.info("Edit Mode toggled via shortcut")
    st.rerun()


def handle_cancel_shortcut() -> None:
    """Handle ESC key to cancel Edit Mode (AC #8).

    Cancels Edit Mode without saving or triggering segmentation.
    """
    cancel_edit_mode()
    logger.info("Edit Mode cancelled via shortcut")
    st.rerun()


def process_shortcuts(shortcuts: dict[str, bool]) -> None:
    """Process triggered shortcuts and execute handlers.

    Routes triggered shortcuts to appropriate handler functions.
    Only one shortcut action is processed per invocation.

    Args:
        shortcuts: Dict mapping action names to triggered states.
    """
    if shortcuts.get("approve_item"):
        handle_approve_shortcut()
    elif shortcuts.get("enter_edit_mode"):
        handle_edit_shortcut()
    elif shortcuts.get("navigate_previous"):
        handle_navigation_shortcut("previous")
    elif shortcuts.get("navigate_next"):
        handle_navigation_shortcut("next")
    elif shortcuts.get("cancel_edit"):
        handle_cancel_shortcut()


def render_shortcut_help() -> None:
    """Render keyboard shortcut reference panel (AC #9)."""
    with st.expander("⌨️ Keyboard Shortcuts"):
        st.markdown(
            """
| Key | Action |
|-----|--------|
| **SPACE** | Approve current & advance |
| **E** | Enter Edit Mode (draw box) |
| **A** or **←** | Previous item |
| **D** or **→** | Next item |
| **ESC** | Cancel Edit Mode |
"""
        )


__all__ = [
    "SHORTCUTS",
    "handle_approve_shortcut",
    "handle_cancel_shortcut",
    "handle_edit_shortcut",
    "handle_navigation_shortcut",
    "navigate_to_next_pending",
    "process_shortcuts",
    "register_shortcuts",
    "render_shortcut_help",
]
