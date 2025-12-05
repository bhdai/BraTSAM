"""Unit tests for keyboard shortcuts module (Story 4.3).

Tests the keyboard shortcut infrastructure, handlers, and guards.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


class TestShortcutConstants:
    """Test shortcut constants and mappings."""

    def test_shortcut_constants_defined(self):
        """Test that all required shortcut button keys are defined."""
        from webapp.utils.shortcuts import SHORTCUTS
        
        # Verify required button keys are present
        assert "btn_approve" in SHORTCUTS
        assert "btn_edit" in SHORTCUTS
        assert "btn_prev" in SHORTCUTS
        assert "btn_next" in SHORTCUTS
        assert "btn_cancel" in SHORTCUTS

    def test_shortcut_action_mappings(self):
        """Test that shortcuts map to correct keyboard keys."""
        from webapp.utils.shortcuts import SHORTCUTS
        
        # Verify key mappings
        assert SHORTCUTS["btn_approve"] == " "  # SPACE
        assert SHORTCUTS["btn_edit"] == "e"
        assert "a" in SHORTCUTS["btn_prev"] or SHORTCUTS["btn_prev"] == "a"
        assert "d" in SHORTCUTS["btn_next"] or SHORTCUTS["btn_next"] == "d"
        assert SHORTCUTS["btn_cancel"] == "Escape"


class TestRegisterShortcuts:
    """Test shortcut registration function."""

    @patch("webapp.utils.shortcuts.add_shortcuts")
    def test_register_shortcuts_calls_library(self, mock_add_shortcuts):
        """Test that register_shortcuts calls the library function."""
        from webapp.utils.shortcuts import SHORTCUTS, register_shortcuts
        
        register_shortcuts()
        
        mock_add_shortcuts.assert_called_once_with(**SHORTCUTS)

    @patch("webapp.utils.shortcuts.add_shortcuts")
    def test_register_shortcuts_returns_none(self, mock_add_shortcuts):
        """Test that register_shortcuts returns None (void function)."""
        from webapp.utils.shortcuts import register_shortcuts
        
        result = register_shortcuts()
        
        assert result is None


class TestApproveShortcut:
    """Test SPACE key approval handler (AC #1, #5)."""

    @patch("webapp.utils.shortcuts.st")
    @patch("webapp.utils.shortcuts.navigate_to_next_pending")
    @patch("webapp.utils.shortcuts.update_status")
    @patch("webapp.utils.shortcuts.get_current_index")
    @patch("webapp.utils.shortcuts.is_edit_mode")
    def test_approve_shortcut_updates_status(
        self,
        mock_is_edit,
        mock_get_index,
        mock_update_status,
        mock_navigate,
        mock_st,
    ):
        """Test SPACE key marks item as approved."""
        from webapp.utils.shortcuts import handle_approve_shortcut
        
        mock_is_edit.return_value = False
        mock_get_index.return_value = 2
        mock_navigate.return_value = True
        
        handle_approve_shortcut()
        
        mock_update_status.assert_called_once_with(2, "approved")

    @patch("webapp.utils.shortcuts.st")
    @patch("webapp.utils.shortcuts.navigate_to_next_pending")
    @patch("webapp.utils.shortcuts.update_status")
    @patch("webapp.utils.shortcuts.get_current_index")
    @patch("webapp.utils.shortcuts.is_edit_mode")
    def test_approve_shortcut_auto_advances(
        self,
        mock_is_edit,
        mock_get_index,
        mock_update_status,
        mock_navigate,
        mock_st,
    ):
        """Test approval advances to next pending item."""
        from webapp.utils.shortcuts import handle_approve_shortcut
        
        mock_is_edit.return_value = False
        mock_get_index.return_value = 0
        mock_navigate.return_value = True
        
        handle_approve_shortcut()
        
        mock_navigate.assert_called_once()
        mock_st.rerun.assert_called_once()

    @patch("webapp.utils.shortcuts.st")
    @patch("webapp.utils.shortcuts.update_status")
    @patch("webapp.utils.shortcuts.is_edit_mode")
    def test_approve_shortcut_blocked_in_edit_mode(
        self,
        mock_is_edit,
        mock_update_status,
        mock_st,
    ):
        """Test approval is blocked when in Edit Mode (AC #8)."""
        from webapp.utils.shortcuts import handle_approve_shortcut
        
        mock_is_edit.return_value = True
        
        handle_approve_shortcut()
        
        mock_update_status.assert_not_called()
        mock_st.rerun.assert_not_called()


class TestNavigationShortcuts:
    """Test A/D/←/→ navigation handlers (AC #3, #4)."""

    @patch("webapp.utils.shortcuts.st")
    @patch("webapp.utils.shortcuts.navigate_previous")
    @patch("webapp.utils.shortcuts.is_edit_mode")
    def test_navigation_previous_decrements_index(
        self,
        mock_is_edit,
        mock_navigate_prev,
        mock_st,
    ):
        """Test A/← moves to previous item."""
        from webapp.utils.shortcuts import handle_navigation_shortcut
        
        mock_is_edit.return_value = False
        mock_navigate_prev.return_value = True
        
        handle_navigation_shortcut("previous")
        
        mock_navigate_prev.assert_called_once()
        mock_st.rerun.assert_called_once()

    @patch("webapp.utils.shortcuts.st")
    @patch("webapp.utils.shortcuts.navigate_next")
    @patch("webapp.utils.shortcuts.is_edit_mode")
    def test_navigation_next_increments_index(
        self,
        mock_is_edit,
        mock_navigate_next,
        mock_st,
    ):
        """Test D/→ moves to next item."""
        from webapp.utils.shortcuts import handle_navigation_shortcut
        
        mock_is_edit.return_value = False
        mock_navigate_next.return_value = True
        
        handle_navigation_shortcut("next")
        
        mock_navigate_next.assert_called_once()
        mock_st.rerun.assert_called_once()

    @patch("webapp.utils.shortcuts.st")
    @patch("webapp.utils.shortcuts.navigate_next")
    @patch("webapp.utils.shortcuts.navigate_previous")
    @patch("webapp.utils.shortcuts.is_edit_mode")
    def test_edit_mode_blocks_navigation(
        self,
        mock_is_edit,
        mock_navigate_prev,
        mock_navigate_next,
        mock_st,
    ):
        """Test shortcuts disabled during Edit Mode (AC #8)."""
        from webapp.utils.shortcuts import handle_navigation_shortcut
        
        mock_is_edit.return_value = True
        
        handle_navigation_shortcut("next")
        handle_navigation_shortcut("previous")
        
        mock_navigate_next.assert_not_called()
        mock_navigate_prev.assert_not_called()
        mock_st.rerun.assert_not_called()


class TestEditShortcut:
    """Test E key edit mode handler (AC #2, #6)."""

    @patch("webapp.utils.shortcuts.st")
    @patch("webapp.utils.shortcuts.toggle_edit_mode")
    def test_edit_shortcut_toggles_mode(self, mock_toggle, mock_st):
        """Test E key enters Edit Mode."""
        from webapp.utils.shortcuts import handle_edit_shortcut
        
        handle_edit_shortcut()
        
        mock_toggle.assert_called_once()
        mock_st.rerun.assert_called_once()


class TestCancelShortcut:
    """Test ESC key cancel handler (AC #8)."""

    @patch("webapp.utils.shortcuts.st")
    @patch("webapp.utils.shortcuts.cancel_edit_mode")
    def test_cancel_shortcut_exits_edit_mode(self, mock_cancel, mock_st):
        """Test ESC cancels Edit Mode."""
        from webapp.utils.shortcuts import handle_cancel_shortcut
        
        handle_cancel_shortcut()
        
        mock_cancel.assert_called_once()
        mock_st.rerun.assert_called_once()


class TestAutoAdvance:
    """Test auto-advance to next pending item (AC #7)."""

    def test_auto_advance_skips_approved_items(self):
        """Test auto-advance skips already approved items."""
        from webapp.utils.shortcuts import navigate_to_next_pending
        
        with patch("webapp.utils.shortcuts.get_batch_queue") as mock_queue, \
             patch("webapp.utils.shortcuts.get_current_index") as mock_index, \
             patch("webapp.utils.shortcuts.set_current_index") as mock_set_index:
            
            # Create mock batch items
            item1 = MagicMock(status="approved")
            item2 = MagicMock(status="approved")
            item3 = MagicMock(status="pending")
            mock_queue.return_value = [item1, item2, item3]
            mock_index.return_value = 0
            
            result = navigate_to_next_pending()
            
            # Should skip items 0 and 1 (approved), go to item 2 (pending)
            mock_set_index.assert_called_once_with(2)
            assert result is True

    def test_auto_advance_returns_false_when_all_reviewed(self):
        """Test returns False when no pending items."""
        from webapp.utils.shortcuts import navigate_to_next_pending
        
        with patch("webapp.utils.shortcuts.get_batch_queue") as mock_queue, \
             patch("webapp.utils.shortcuts.get_current_index") as mock_index, \
             patch("webapp.utils.shortcuts.set_current_index") as mock_set_index:
            
            # All items approved
            item1 = MagicMock(status="approved")
            item2 = MagicMock(status="approved")
            mock_queue.return_value = [item1, item2]
            mock_index.return_value = 0
            
            result = navigate_to_next_pending()
            
            mock_set_index.assert_not_called()
            assert result is False

    def test_auto_advance_wraps_around(self):
        """Test auto-advance wraps from end to beginning."""
        from webapp.utils.shortcuts import navigate_to_next_pending
        
        with patch("webapp.utils.shortcuts.get_batch_queue") as mock_queue, \
             patch("webapp.utils.shortcuts.get_current_index") as mock_index, \
             patch("webapp.utils.shortcuts.set_current_index") as mock_set_index:
            
            # Pending item at beginning, current at end
            item1 = MagicMock(status="pending")
            item2 = MagicMock(status="approved")
            item3 = MagicMock(status="approved")
            mock_queue.return_value = [item1, item2, item3]
            mock_index.return_value = 2
            
            result = navigate_to_next_pending()
            
            # Should wrap around to item 0
            mock_set_index.assert_called_once_with(0)
            assert result is True

    def test_auto_advance_empty_queue(self):
        """Test auto-advance returns False for empty queue."""
        from webapp.utils.shortcuts import navigate_to_next_pending
        
        with patch("webapp.utils.shortcuts.get_batch_queue") as mock_queue, \
             patch("webapp.utils.shortcuts.get_current_index") as mock_index, \
             patch("webapp.utils.shortcuts.set_current_index") as mock_set_index:
            
            mock_queue.return_value = []
            mock_index.return_value = 0
            
            result = navigate_to_next_pending()
            
            mock_set_index.assert_not_called()
            assert result is False


class TestShortcutHelpReference:
    """Test shortcut help component (AC #9)."""

    @patch("webapp.utils.shortcuts.st")
    def test_render_shortcut_help_displays_panel(self, mock_st):
        """Test help panel renders with shortcuts."""
        from webapp.utils.shortcuts import render_shortcut_help
        
        # Mock the expander context manager
        mock_expander = MagicMock()
        mock_st.expander.return_value.__enter__ = MagicMock(return_value=mock_expander)
        mock_st.expander.return_value.__exit__ = MagicMock(return_value=False)
        
        render_shortcut_help()
        
        mock_st.expander.assert_called_once()
        mock_st.markdown.assert_called()


class TestProcessShortcuts:
    """Test the process_shortcuts function that handles triggered shortcuts."""

    @patch("webapp.utils.shortcuts.handle_approve_shortcut")
    @patch("webapp.utils.shortcuts.handle_edit_shortcut")
    @patch("webapp.utils.shortcuts.handle_navigation_shortcut")
    @patch("webapp.utils.shortcuts.handle_cancel_shortcut")
    def test_process_approve_shortcut(
        self,
        mock_cancel,
        mock_nav,
        mock_edit,
        mock_approve,
    ):
        """Test process_shortcuts routes approve action."""
        from webapp.utils.shortcuts import process_shortcuts
        
        shortcuts = {"approve_item": True, "navigate_next": False}
        
        process_shortcuts(shortcuts)
        
        mock_approve.assert_called_once()
        mock_edit.assert_not_called()
        mock_nav.assert_not_called()

    @patch("webapp.utils.shortcuts.handle_approve_shortcut")
    @patch("webapp.utils.shortcuts.handle_edit_shortcut")
    @patch("webapp.utils.shortcuts.handle_navigation_shortcut")
    @patch("webapp.utils.shortcuts.handle_cancel_shortcut")
    def test_process_navigation_shortcut(
        self,
        mock_cancel,
        mock_nav,
        mock_edit,
        mock_approve,
    ):
        """Test process_shortcuts routes navigation actions."""
        from webapp.utils.shortcuts import process_shortcuts
        
        shortcuts = {"navigate_previous": True}
        
        process_shortcuts(shortcuts)
        
        mock_nav.assert_called_once_with("previous")
        mock_approve.assert_not_called()

    @patch("webapp.utils.shortcuts.handle_approve_shortcut")
    @patch("webapp.utils.shortcuts.handle_edit_shortcut")
    @patch("webapp.utils.shortcuts.handle_navigation_shortcut")
    @patch("webapp.utils.shortcuts.handle_cancel_shortcut")
    def test_process_edit_shortcut(
        self,
        mock_cancel,
        mock_nav,
        mock_edit,
        mock_approve,
    ):
        """Test process_shortcuts routes edit action."""
        from webapp.utils.shortcuts import process_shortcuts
        
        shortcuts = {"enter_edit_mode": True}
        
        process_shortcuts(shortcuts)
        
        mock_edit.assert_called_once()
        mock_approve.assert_not_called()

    @patch("webapp.utils.shortcuts.handle_approve_shortcut")
    @patch("webapp.utils.shortcuts.handle_edit_shortcut")
    @patch("webapp.utils.shortcuts.handle_navigation_shortcut")
    @patch("webapp.utils.shortcuts.handle_cancel_shortcut")
    def test_process_cancel_shortcut(
        self,
        mock_cancel,
        mock_nav,
        mock_edit,
        mock_approve,
    ):
        """Test process_shortcuts routes cancel action."""
        from webapp.utils.shortcuts import process_shortcuts
        
        shortcuts = {"cancel_edit": True}
        
        process_shortcuts(shortcuts)
        
        mock_cancel.assert_called_once()
        mock_approve.assert_not_called()
