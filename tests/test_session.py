"""Tests for session state management (Story 4.1).

Tests the batch processing queue state management functions
that handle Streamlit session state for batch operations.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


class TestSessionKeys:
    """Tests for SESSION_KEYS constants."""

    def test_session_keys_exist(self):
        """SESSION_KEYS should define required keys."""
        from webapp.utils.session import SESSION_KEYS

        assert "batch_queue" in SESSION_KEYS
        assert "current_index" in SESSION_KEYS
        assert "batch_initialized" in SESSION_KEYS


class TestInitializeBatchState:
    """Tests for initialize_batch_state function."""

    def test_initialize_creates_empty_queue(self):
        """initialize_batch_state should create empty batch queue."""
        from webapp.utils.session import SESSION_KEYS, initialize_batch_state

        mock_state = {}
        with patch("webapp.utils.session.st") as mock_st:
            mock_st.session_state = mock_state
            initialize_batch_state()

            assert SESSION_KEYS["batch_queue"] in mock_state
            assert isinstance(mock_state[SESSION_KEYS["batch_queue"]], list)
            assert len(mock_state[SESSION_KEYS["batch_queue"]]) == 0

    def test_initialize_sets_current_index_zero(self):
        """initialize_batch_state should set current_index to 0."""
        from webapp.utils.session import SESSION_KEYS, initialize_batch_state

        mock_state = {}
        with patch("webapp.utils.session.st") as mock_st:
            mock_st.session_state = mock_state
            initialize_batch_state()

            assert mock_state[SESSION_KEYS["current_index"]] == 0

    def test_initialize_marks_initialized(self):
        """initialize_batch_state should set initialized flag."""
        from webapp.utils.session import SESSION_KEYS, initialize_batch_state

        mock_state = {}
        with patch("webapp.utils.session.st") as mock_st:
            mock_st.session_state = mock_state
            initialize_batch_state()

            assert mock_state[SESSION_KEYS["batch_initialized"]] is True

    def test_initialize_idempotent(self):
        """initialize_batch_state should not overwrite existing state."""
        from webapp.utils.session import SESSION_KEYS, initialize_batch_state
        from webapp.utils.batch import BatchItem
        from webapp.utils.inference import PipelineResult

        result = PipelineResult(success=True, stage="segmentation")
        existing_item = BatchItem(filename="existing.png", result=result)

        mock_state = {
            SESSION_KEYS["batch_queue"]: [existing_item],
            SESSION_KEYS["current_index"]: 5,
            SESSION_KEYS["batch_initialized"]: True,
        }

        with patch("webapp.utils.session.st") as mock_st:
            mock_st.session_state = mock_state
            initialize_batch_state()

            # Should not overwrite
            assert len(mock_state[SESSION_KEYS["batch_queue"]]) == 1
            assert mock_state[SESSION_KEYS["current_index"]] == 5


class TestGetBatchQueue:
    """Tests for get_batch_queue function."""

    def test_get_batch_queue_returns_list(self):
        """get_batch_queue should return the batch queue list."""
        from webapp.utils.session import SESSION_KEYS, get_batch_queue

        mock_state = {
            SESSION_KEYS["batch_queue"]: [],
            SESSION_KEYS["batch_initialized"]: True,
        }

        with patch("webapp.utils.session.st") as mock_st:
            mock_st.session_state = mock_state
            queue = get_batch_queue()
            assert isinstance(queue, list)

    def test_get_batch_queue_initializes_if_needed(self):
        """get_batch_queue should initialize state if not done."""
        from webapp.utils.session import SESSION_KEYS, get_batch_queue

        mock_state = {}
        with patch("webapp.utils.session.st") as mock_st:
            mock_st.session_state = mock_state
            queue = get_batch_queue()

            assert SESSION_KEYS["batch_initialized"] in mock_state
            assert isinstance(queue, list)


class TestAddToBatch:
    """Tests for add_to_batch function."""

    def test_add_to_batch_creates_item(self):
        """add_to_batch should create a new BatchItem."""
        from webapp.utils.session import SESSION_KEYS, add_to_batch

        mock_state = {
            SESSION_KEYS["batch_queue"]: [],
            SESSION_KEYS["current_index"]: 0,
            SESSION_KEYS["batch_initialized"]: True,
        }

        with patch("webapp.utils.session.st") as mock_st:
            mock_st.session_state = mock_state
            index = add_to_batch("test.png")

            assert index == 0
            assert len(mock_state[SESSION_KEYS["batch_queue"]]) == 1

    def test_add_to_batch_sets_pending_status(self):
        """add_to_batch should set status to pending."""
        from webapp.utils.session import SESSION_KEYS, add_to_batch

        mock_state = {
            SESSION_KEYS["batch_queue"]: [],
            SESSION_KEYS["current_index"]: 0,
            SESSION_KEYS["batch_initialized"]: True,
        }

        with patch("webapp.utils.session.st") as mock_st:
            mock_st.session_state = mock_state
            add_to_batch("test.png")

            item = mock_state[SESSION_KEYS["batch_queue"]][0]
            assert item.status == "pending"

    def test_add_to_batch_stores_filename(self):
        """add_to_batch should store the filename correctly."""
        from webapp.utils.session import SESSION_KEYS, add_to_batch

        mock_state = {
            SESSION_KEYS["batch_queue"]: [],
            SESSION_KEYS["current_index"]: 0,
            SESSION_KEYS["batch_initialized"]: True,
        }

        with patch("webapp.utils.session.st") as mock_st:
            mock_st.session_state = mock_state
            add_to_batch("my_scan_001.png")

            item = mock_state[SESSION_KEYS["batch_queue"]][0]
            assert item.filename == "my_scan_001.png"

    def test_add_to_batch_with_result(self):
        """add_to_batch should accept optional PipelineResult."""
        from webapp.utils.session import SESSION_KEYS, add_to_batch
        from webapp.utils.inference import PipelineResult

        result = PipelineResult(
            success=True,
            stage="segmentation",
            yolo_confidence=0.9,
            sam_iou=0.85,
        )

        mock_state = {
            SESSION_KEYS["batch_queue"]: [],
            SESSION_KEYS["current_index"]: 0,
            SESSION_KEYS["batch_initialized"]: True,
        }

        with patch("webapp.utils.session.st") as mock_st:
            mock_st.session_state = mock_state
            add_to_batch("test.png", result=result)

            item = mock_state[SESSION_KEYS["batch_queue"]][0]
            assert item.result is result

    def test_add_to_batch_returns_correct_index(self):
        """add_to_batch should return the index of the new item."""
        from webapp.utils.session import SESSION_KEYS, add_to_batch
        from webapp.utils.batch import BatchItem
        from webapp.utils.inference import PipelineResult

        result = PipelineResult(success=True, stage="segmentation")
        existing = BatchItem(filename="first.png", result=result)

        mock_state = {
            SESSION_KEYS["batch_queue"]: [existing],
            SESSION_KEYS["current_index"]: 0,
            SESSION_KEYS["batch_initialized"]: True,
        }

        with patch("webapp.utils.session.st") as mock_st:
            mock_st.session_state = mock_state
            index = add_to_batch("second.png")

            assert index == 1


class TestClearBatchQueue:
    """Tests for clear_batch_queue function."""

    def test_clear_batch_queue_empties_list(self):
        """clear_batch_queue should empty the batch queue."""
        from webapp.utils.session import SESSION_KEYS, clear_batch_queue
        from webapp.utils.batch import BatchItem
        from webapp.utils.inference import PipelineResult

        result = PipelineResult(success=True, stage="segmentation")
        item = BatchItem(filename="test.png", result=result)

        mock_state = {
            SESSION_KEYS["batch_queue"]: [item, item],
            SESSION_KEYS["current_index"]: 1,
            SESSION_KEYS["batch_initialized"]: True,
        }

        with patch("webapp.utils.session.st") as mock_st:
            mock_st.session_state = mock_state
            clear_batch_queue()

            assert len(mock_state[SESSION_KEYS["batch_queue"]]) == 0

    def test_clear_batch_queue_resets_index(self):
        """clear_batch_queue should reset current index to 0."""
        from webapp.utils.session import SESSION_KEYS, clear_batch_queue
        from webapp.utils.batch import BatchItem
        from webapp.utils.inference import PipelineResult

        result = PipelineResult(success=True, stage="segmentation")
        item = BatchItem(filename="test.png", result=result)

        mock_state = {
            SESSION_KEYS["batch_queue"]: [item],
            SESSION_KEYS["current_index"]: 5,
            SESSION_KEYS["batch_initialized"]: True,
        }

        with patch("webapp.utils.session.st") as mock_st:
            mock_st.session_state = mock_state
            clear_batch_queue()

            assert mock_state[SESSION_KEYS["current_index"]] == 0


class TestGetBatchSize:
    """Tests for get_batch_size function."""

    def test_get_batch_size_returns_count(self):
        """get_batch_size should return number of items in queue."""
        from webapp.utils.session import SESSION_KEYS, get_batch_size
        from webapp.utils.batch import BatchItem
        from webapp.utils.inference import PipelineResult

        result = PipelineResult(success=True, stage="segmentation")
        items = [
            BatchItem(filename="a.png", result=result),
            BatchItem(filename="b.png", result=result),
            BatchItem(filename="c.png", result=result),
        ]

        mock_state = {
            SESSION_KEYS["batch_queue"]: items,
            SESSION_KEYS["batch_initialized"]: True,
        }

        with patch("webapp.utils.session.st") as mock_st:
            mock_st.session_state = mock_state
            assert get_batch_size() == 3

    def test_get_batch_size_empty_queue(self):
        """get_batch_size should return 0 for empty queue."""
        from webapp.utils.session import SESSION_KEYS, get_batch_size

        mock_state = {
            SESSION_KEYS["batch_queue"]: [],
            SESSION_KEYS["batch_initialized"]: True,
        }

        with patch("webapp.utils.session.st") as mock_st:
            mock_st.session_state = mock_state
            assert get_batch_size() == 0


class TestUpdateStatus:
    """Tests for update_status function."""

    def test_update_status_valid(self):
        """update_status should update item status."""
        from webapp.utils.session import SESSION_KEYS, update_status
        from webapp.utils.batch import BatchItem
        from webapp.utils.inference import PipelineResult

        result = PipelineResult(success=True, stage="segmentation")
        item = BatchItem(filename="test.png", result=result, status="pending")

        mock_state = {
            SESSION_KEYS["batch_queue"]: [item],
            SESSION_KEYS["batch_initialized"]: True,
        }

        with patch("webapp.utils.session.st") as mock_st:
            mock_st.session_state = mock_state
            success = update_status(0, "approved")

            assert success is True
            assert item.status == "approved"

    def test_update_status_invalid_status_raises(self):
        """update_status should raise ValueError for invalid status."""
        from webapp.utils.session import SESSION_KEYS, update_status
        from webapp.utils.batch import BatchItem
        from webapp.utils.inference import PipelineResult

        result = PipelineResult(success=True, stage="segmentation")
        item = BatchItem(filename="test.png", result=result)

        mock_state = {
            SESSION_KEYS["batch_queue"]: [item],
            SESSION_KEYS["batch_initialized"]: True,
        }

        with patch("webapp.utils.session.st") as mock_st:
            mock_st.session_state = mock_state
            with pytest.raises(ValueError):
                update_status(0, "invalid_status")

    def test_update_status_out_of_bounds(self):
        """update_status should return False for out-of-bounds index."""
        from webapp.utils.session import SESSION_KEYS, update_status

        mock_state = {
            SESSION_KEYS["batch_queue"]: [],
            SESSION_KEYS["batch_initialized"]: True,
        }

        with patch("webapp.utils.session.st") as mock_st:
            mock_st.session_state = mock_state
            success = update_status(5, "approved")
            assert success is False

    def test_update_status_all_valid_statuses(self):
        """update_status should accept all valid status values."""
        from webapp.utils.session import SESSION_KEYS, update_status, VALID_STATUSES
        from webapp.utils.batch import BatchItem
        from webapp.utils.inference import PipelineResult

        result = PipelineResult(success=True, stage="segmentation")
        item = BatchItem(filename="test.png", result=result)

        mock_state = {
            SESSION_KEYS["batch_queue"]: [item],
            SESSION_KEYS["batch_initialized"]: True,
        }

        with patch("webapp.utils.session.st") as mock_st:
            mock_st.session_state = mock_state
            for status in VALID_STATUSES:
                assert update_status(0, status) is True


class TestNavigationHelpers:
    """Tests for navigation functions."""

    def test_get_next_pending_index_found(self):
        """get_next_pending_index should return first pending index."""
        from webapp.utils.session import SESSION_KEYS, get_next_pending_index
        from webapp.utils.batch import BatchItem
        from webapp.utils.inference import PipelineResult

        result = PipelineResult(success=True, stage="segmentation")
        items = [
            BatchItem(filename="a.png", result=result, status="approved"),
            BatchItem(filename="b.png", result=result, status="approved"),
            BatchItem(filename="c.png", result=result, status="pending"),
            BatchItem(filename="d.png", result=result, status="pending"),
        ]

        mock_state = {
            SESSION_KEYS["batch_queue"]: items,
            SESSION_KEYS["batch_initialized"]: True,
        }

        with patch("webapp.utils.session.st") as mock_st:
            mock_st.session_state = mock_state
            index = get_next_pending_index()
            assert index == 2

    def test_get_next_pending_index_none(self):
        """get_next_pending_index should return None if no pending."""
        from webapp.utils.session import SESSION_KEYS, get_next_pending_index
        from webapp.utils.batch import BatchItem
        from webapp.utils.inference import PipelineResult

        result = PipelineResult(success=True, stage="segmentation")
        items = [
            BatchItem(filename="a.png", result=result, status="approved"),
            BatchItem(filename="b.png", result=result, status="rejected"),
        ]

        mock_state = {
            SESSION_KEYS["batch_queue"]: items,
            SESSION_KEYS["batch_initialized"]: True,
        }

        with patch("webapp.utils.session.st") as mock_st:
            mock_st.session_state = mock_state
            index = get_next_pending_index()
            assert index is None

    def test_get_current_index(self):
        """get_current_index should return current index."""
        from webapp.utils.session import SESSION_KEYS, get_current_index

        mock_state = {
            SESSION_KEYS["current_index"]: 3,
            SESSION_KEYS["batch_initialized"]: True,
        }

        with patch("webapp.utils.session.st") as mock_st:
            mock_st.session_state = mock_state
            assert get_current_index() == 3

    def test_set_current_index_valid(self):
        """set_current_index should update index when valid."""
        from webapp.utils.session import SESSION_KEYS, set_current_index
        from webapp.utils.batch import BatchItem
        from webapp.utils.inference import PipelineResult

        result = PipelineResult(success=True, stage="segmentation")
        items = [
            BatchItem(filename="a.png", result=result),
            BatchItem(filename="b.png", result=result),
            BatchItem(filename="c.png", result=result),
        ]

        mock_state = {
            SESSION_KEYS["batch_queue"]: items,
            SESSION_KEYS["current_index"]: 0,
            SESSION_KEYS["batch_initialized"]: True,
        }

        with patch("webapp.utils.session.st") as mock_st:
            mock_st.session_state = mock_state
            success = set_current_index(2)
            assert success is True
            assert mock_state[SESSION_KEYS["current_index"]] == 2

    def test_set_current_index_out_of_bounds(self):
        """set_current_index should return False for invalid index."""
        from webapp.utils.session import SESSION_KEYS, set_current_index
        from webapp.utils.batch import BatchItem
        from webapp.utils.inference import PipelineResult

        result = PipelineResult(success=True, stage="segmentation")
        items = [BatchItem(filename="a.png", result=result)]

        mock_state = {
            SESSION_KEYS["batch_queue"]: items,
            SESSION_KEYS["current_index"]: 0,
            SESSION_KEYS["batch_initialized"]: True,
        }

        with patch("webapp.utils.session.st") as mock_st:
            mock_st.session_state = mock_state
            assert set_current_index(5) is False
            assert set_current_index(-1) is False

    def test_navigate_next_success(self):
        """navigate_next should move to next item."""
        from webapp.utils.session import SESSION_KEYS, navigate_next
        from webapp.utils.batch import BatchItem
        from webapp.utils.inference import PipelineResult

        result = PipelineResult(success=True, stage="segmentation")
        items = [
            BatchItem(filename="a.png", result=result),
            BatchItem(filename="b.png", result=result),
        ]

        mock_state = {
            SESSION_KEYS["batch_queue"]: items,
            SESSION_KEYS["current_index"]: 0,
            SESSION_KEYS["batch_initialized"]: True,
        }

        with patch("webapp.utils.session.st") as mock_st:
            mock_st.session_state = mock_state
            success = navigate_next()
            assert success is True
            assert mock_state[SESSION_KEYS["current_index"]] == 1

    def test_navigate_next_at_end(self):
        """navigate_next should return False at end of queue."""
        from webapp.utils.session import SESSION_KEYS, navigate_next
        from webapp.utils.batch import BatchItem
        from webapp.utils.inference import PipelineResult

        result = PipelineResult(success=True, stage="segmentation")
        items = [BatchItem(filename="a.png", result=result)]

        mock_state = {
            SESSION_KEYS["batch_queue"]: items,
            SESSION_KEYS["current_index"]: 0,
            SESSION_KEYS["batch_initialized"]: True,
        }

        with patch("webapp.utils.session.st") as mock_st:
            mock_st.session_state = mock_state
            success = navigate_next()
            assert success is False

    def test_navigate_previous_success(self):
        """navigate_previous should move to previous item."""
        from webapp.utils.session import SESSION_KEYS, navigate_previous
        from webapp.utils.batch import BatchItem
        from webapp.utils.inference import PipelineResult

        result = PipelineResult(success=True, stage="segmentation")
        items = [
            BatchItem(filename="a.png", result=result),
            BatchItem(filename="b.png", result=result),
        ]

        mock_state = {
            SESSION_KEYS["batch_queue"]: items,
            SESSION_KEYS["current_index"]: 1,
            SESSION_KEYS["batch_initialized"]: True,
        }

        with patch("webapp.utils.session.st") as mock_st:
            mock_st.session_state = mock_state
            success = navigate_previous()
            assert success is True
            assert mock_state[SESSION_KEYS["current_index"]] == 0

    def test_navigate_previous_at_start(self):
        """navigate_previous should return False at start of queue."""
        from webapp.utils.session import SESSION_KEYS, navigate_previous
        from webapp.utils.batch import BatchItem
        from webapp.utils.inference import PipelineResult

        result = PipelineResult(success=True, stage="segmentation")
        items = [BatchItem(filename="a.png", result=result)]

        mock_state = {
            SESSION_KEYS["batch_queue"]: items,
            SESSION_KEYS["current_index"]: 0,
            SESSION_KEYS["batch_initialized"]: True,
        }

        with patch("webapp.utils.session.st") as mock_st:
            mock_st.session_state = mock_state
            success = navigate_previous()
            assert success is False


class TestBatchItemRetrieval:
    """Tests for batch item retrieval functions."""

    def test_get_batch_item_valid(self):
        """get_batch_item should return item at index."""
        from webapp.utils.session import SESSION_KEYS, get_batch_item
        from webapp.utils.batch import BatchItem
        from webapp.utils.inference import PipelineResult

        result = PipelineResult(success=True, stage="segmentation")
        item = BatchItem(filename="test.png", result=result)

        mock_state = {
            SESSION_KEYS["batch_queue"]: [item],
            SESSION_KEYS["batch_initialized"]: True,
        }

        with patch("webapp.utils.session.st") as mock_st:
            mock_st.session_state = mock_state
            retrieved = get_batch_item(0)
            assert retrieved is item

    def test_get_batch_item_out_of_bounds(self):
        """get_batch_item should return None for invalid index."""
        from webapp.utils.session import SESSION_KEYS, get_batch_item

        mock_state = {
            SESSION_KEYS["batch_queue"]: [],
            SESSION_KEYS["batch_initialized"]: True,
        }

        with patch("webapp.utils.session.st") as mock_st:
            mock_st.session_state = mock_state
            assert get_batch_item(0) is None
            assert get_batch_item(-1) is None

    def test_get_current_item(self):
        """get_current_item should return currently selected item."""
        from webapp.utils.session import SESSION_KEYS, get_current_item
        from webapp.utils.batch import BatchItem
        from webapp.utils.inference import PipelineResult

        result = PipelineResult(success=True, stage="segmentation")
        items = [
            BatchItem(filename="a.png", result=result),
            BatchItem(filename="b.png", result=result),
        ]

        mock_state = {
            SESSION_KEYS["batch_queue"]: items,
            SESSION_KEYS["current_index"]: 1,
            SESSION_KEYS["batch_initialized"]: True,
        }

        with patch("webapp.utils.session.st") as mock_st:
            mock_st.session_state = mock_state
            item = get_current_item()
            assert item.filename == "b.png"

    def test_get_current_item_empty_queue(self):
        """get_current_item should return None for empty queue."""
        from webapp.utils.session import SESSION_KEYS, get_current_item

        mock_state = {
            SESSION_KEYS["batch_queue"]: [],
            SESSION_KEYS["current_index"]: 0,
            SESSION_KEYS["batch_initialized"]: True,
        }

        with patch("webapp.utils.session.st") as mock_st:
            mock_st.session_state = mock_state
            assert get_current_item() is None

    def test_update_batch_item_result(self):
        """update_batch_item_result should update item's result."""
        from webapp.utils.session import SESSION_KEYS, update_batch_item_result
        from webapp.utils.batch import BatchItem
        from webapp.utils.inference import PipelineResult

        old_result = PipelineResult(success=False, stage="upload")
        new_result = PipelineResult(
            success=True,
            stage="segmentation",
            yolo_confidence=0.95,
            sam_iou=0.9,
        )
        item = BatchItem(filename="test.png", result=old_result)

        mock_state = {
            SESSION_KEYS["batch_queue"]: [item],
            SESSION_KEYS["batch_initialized"]: True,
        }

        with patch("webapp.utils.session.st") as mock_st:
            mock_st.session_state = mock_state
            success = update_batch_item_result(0, new_result)

            assert success is True
            assert item.result is new_result
            assert item.result.success is True

    def test_update_batch_item_result_out_of_bounds(self):
        """update_batch_item_result should return False for invalid index."""
        from webapp.utils.session import SESSION_KEYS, update_batch_item_result
        from webapp.utils.inference import PipelineResult

        result = PipelineResult(success=True, stage="segmentation")

        mock_state = {
            SESSION_KEYS["batch_queue"]: [],
            SESSION_KEYS["batch_initialized"]: True,
        }

        with patch("webapp.utils.session.st") as mock_st:
            mock_st.session_state = mock_state
            assert update_batch_item_result(0, result) is False


class TestGetBatchStatistics:
    """Tests for get_batch_statistics wrapper function."""

    def test_get_batch_statistics_returns_stats(self):
        """get_batch_statistics should compute stats for session queue."""
        from webapp.utils.session import SESSION_KEYS, get_batch_statistics
        from webapp.utils.batch import BatchItem, BatchStatistics
        from webapp.utils.inference import PipelineResult

        # Create items with different classifications
        high_conf = PipelineResult(
            success=True, stage="segmentation", yolo_confidence=0.95, sam_iou=0.92
        )
        med_conf = PipelineResult(
            success=True, stage="segmentation", yolo_confidence=0.75, sam_iou=0.7
        )

        items = [
            BatchItem(filename="a.png", result=high_conf, status="approved"),
            BatchItem(filename="b.png", result=med_conf, status="pending"),
        ]

        mock_state = {
            SESSION_KEYS["batch_queue"]: items,
            SESSION_KEYS["batch_initialized"]: True,
        }

        with patch("webapp.utils.session.st") as mock_st:
            mock_st.session_state = mock_state
            stats = get_batch_statistics()

            assert isinstance(stats, BatchStatistics)
            assert stats.total == 2
            assert stats.auto_approved == 1
            assert stats.needs_review == 1
            assert stats.processed == 1

    def test_get_batch_statistics_empty_queue(self):
        """get_batch_statistics should handle empty queue."""
        from webapp.utils.session import SESSION_KEYS, get_batch_statistics

        mock_state = {
            SESSION_KEYS["batch_queue"]: [],
            SESSION_KEYS["batch_initialized"]: True,
        }

        with patch("webapp.utils.session.st") as mock_st:
            mock_st.session_state = mock_state
            stats = get_batch_statistics()

            assert stats.total == 0
            assert stats.pending == 0


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_queue_navigation(self):
        """Navigation should handle empty queue gracefully."""
        from webapp.utils.session import (
            SESSION_KEYS,
            navigate_next,
            navigate_previous,
            get_current_item,
        )

        mock_state = {
            SESSION_KEYS["batch_queue"]: [],
            SESSION_KEYS["current_index"]: 0,
            SESSION_KEYS["batch_initialized"]: True,
        }

        with patch("webapp.utils.session.st") as mock_st:
            mock_st.session_state = mock_state
            assert navigate_next() is False
            assert navigate_previous() is False
            assert get_current_item() is None

    def test_negative_index_handling(self):
        """Functions should handle negative indices correctly."""
        from webapp.utils.session import (
            SESSION_KEYS,
            get_batch_item,
            update_status,
            update_batch_item_result,
        )
        from webapp.utils.batch import BatchItem
        from webapp.utils.inference import PipelineResult

        result = PipelineResult(success=True, stage="segmentation")
        item = BatchItem(filename="test.png", result=result)

        mock_state = {
            SESSION_KEYS["batch_queue"]: [item],
            SESSION_KEYS["batch_initialized"]: True,
        }

        with patch("webapp.utils.session.st") as mock_st:
            mock_st.session_state = mock_state
            assert get_batch_item(-1) is None
            assert update_status(-1, "approved") is False
            assert update_batch_item_result(-1, result) is False

    def test_persistence_across_reruns(self):
        """Session state should persist across simulated reruns."""
        from webapp.utils.session import (
            SESSION_KEYS,
            initialize_batch_state,
            add_to_batch,
            get_batch_queue,
        )
        from webapp.utils.inference import PipelineResult

        result = PipelineResult(success=True, stage="segmentation")

        # Simulate shared session state across reruns
        shared_state = {}

        with patch("webapp.utils.session.st") as mock_st:
            mock_st.session_state = shared_state

            # First "run"
            initialize_batch_state()
            add_to_batch("file1.png", result)

            # Verify state exists
            assert len(shared_state[SESSION_KEYS["batch_queue"]]) == 1

        with patch("webapp.utils.session.st") as mock_st:
            mock_st.session_state = shared_state

            # Second "run" - state should persist
            initialize_batch_state()  # Should be idempotent
            queue = get_batch_queue()

            assert len(queue) == 1
            assert queue[0].filename == "file1.png"


class TestBatchItemExtensions:
    """Tests for BatchItem extended fields (slice_index, timestamp)."""

    def test_add_to_batch_with_slice_index(self):
        """add_to_batch should store slice_index."""
        from webapp.utils.session import SESSION_KEYS, add_to_batch

        mock_state = {
            SESSION_KEYS["batch_queue"]: [],
            SESSION_KEYS["current_index"]: 0,
            SESSION_KEYS["batch_initialized"]: True,
        }

        with patch("webapp.utils.session.st") as mock_st:
            mock_st.session_state = mock_state
            add_to_batch("volume_slice.png", slice_index=42)

            item = mock_state[SESSION_KEYS["batch_queue"]][0]
            assert item.slice_index == 42

    def test_add_to_batch_sets_timestamp(self):
        """add_to_batch should set creation timestamp."""
        from webapp.utils.session import SESSION_KEYS, add_to_batch
        import time

        mock_state = {
            SESSION_KEYS["batch_queue"]: [],
            SESSION_KEYS["current_index"]: 0,
            SESSION_KEYS["batch_initialized"]: True,
        }

        before = time.time()
        with patch("webapp.utils.session.st") as mock_st:
            mock_st.session_state = mock_state
            add_to_batch("test.png")
        after = time.time()

        item = mock_state[SESSION_KEYS["batch_queue"]][0]
        assert item.timestamp is not None
        assert before <= item.timestamp <= after
