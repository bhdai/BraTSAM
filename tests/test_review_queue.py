"""Tests for review queue component (Story 4.2).

Tests the Queue Master UI component that displays batch items,
handles filtering, selection, and progress tracking.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


class TestRenderReviewQueue:
    """Tests for render_review_queue function (AC #1)."""

    def test_render_review_queue_exists(self):
        """render_review_queue function should exist."""
        from webapp.components.review_queue import render_review_queue

        assert callable(render_review_queue)

    def test_render_review_queue_returns_none(self):
        """render_review_queue should render and return None."""
        from webapp.components.review_queue import render_review_queue
        from webapp.utils.session import SESSION_KEYS

        mock_state = {
            SESSION_KEYS["batch_queue"]: [],
            SESSION_KEYS["current_index"]: 0,
            SESSION_KEYS["batch_initialized"]: True,
            "queue_filter": "all",
        }

        with patch("webapp.components.review_queue.st") as mock_st:
            mock_st.session_state = mock_state
            mock_st.tabs = MagicMock(return_value=[MagicMock(), MagicMock(), MagicMock()])
            mock_st.container = MagicMock(return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock()))
            result = render_review_queue()
            assert result is None


class TestTruncateFilename:
    """Tests for filename truncation (AC #2)."""

    def test_truncate_short_filename(self):
        """Short filenames should not be truncated."""
        from webapp.components.review_queue import truncate_filename

        assert truncate_filename("short.png") == "short.png"
        assert truncate_filename("exactly20chars.png") == "exactly20chars.png"

    def test_truncate_long_filename(self):
        """Filenames longer than 20 chars should be truncated."""
        from webapp.components.review_queue import truncate_filename

        result = truncate_filename("very_long_filename_exceeds_limit.png")
        assert len(result) == 23  # 20 + "..."
        assert result.endswith("...")
        assert result == "very_long_filename_e..."

    def test_truncate_exact_20_chars(self):
        """Filenames with exactly 20 chars should not be truncated."""
        from webapp.components.review_queue import truncate_filename

        filename = "12345678901234567890"  # exactly 20 chars
        assert truncate_filename(filename) == filename


class TestGenerateThumbnail:
    """Tests for thumbnail generation (AC #2)."""

    def test_generate_thumbnail_exists(self):
        """generate_thumbnail function should exist."""
        from webapp.components.review_queue import generate_thumbnail

        assert callable(generate_thumbnail)

    def test_generate_thumbnail_grayscale(self):
        """Thumbnail should be generated from grayscale array."""
        import numpy as np

        from webapp.components.review_queue import generate_thumbnail

        # Create a grayscale test image
        image = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        thumbnail_bytes = generate_thumbnail(image)

        assert isinstance(thumbnail_bytes, bytes)
        assert len(thumbnail_bytes) > 0

    def test_generate_thumbnail_rgb(self):
        """Thumbnail should be generated from RGB array."""
        import numpy as np

        from webapp.components.review_queue import generate_thumbnail

        # Create an RGB test image
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        thumbnail_bytes = generate_thumbnail(image)

        assert isinstance(thumbnail_bytes, bytes)
        assert len(thumbnail_bytes) > 0

    def test_generate_thumbnail_custom_size(self):
        """Thumbnail size should be customizable."""
        import numpy as np
        from PIL import Image
        import io

        from webapp.components.review_queue import generate_thumbnail

        image = np.random.randint(0, 255, (200, 200), dtype=np.uint8)
        thumbnail_bytes = generate_thumbnail(image, size=(32, 32))

        # Verify the thumbnail dimensions
        thumb = Image.open(io.BytesIO(thumbnail_bytes))
        assert thumb.width <= 32
        assert thumb.height <= 32


class TestRenderQueueItem:
    """Tests for queue item rendering (AC #2, #3, #4)."""

    def test_render_queue_item_exists(self):
        """render_queue_item function should exist."""
        from webapp.components.review_queue import render_queue_item

        assert callable(render_queue_item)

    def test_render_queue_item_shows_confidence_indicator(self):
        """Queue item should display confidence indicator based on classification."""
        import numpy as np

        from webapp.components.review_queue import render_queue_item
        from webapp.utils.batch import BatchItem
        from webapp.utils.inference import PipelineResult
        from webapp.utils.session import SESSION_KEYS

        result = PipelineResult(
            success=True,
            stage="segmentation",
            yolo_confidence=0.95,
            sam_iou=0.92,
            sam_mask=np.zeros((100, 100), dtype=np.uint8),
        )
        item = BatchItem(filename="test.png", result=result)

        mock_state = {
            SESSION_KEYS["batch_queue"]: [item],
            SESSION_KEYS["current_index"]: 0,
            SESSION_KEYS["batch_initialized"]: True,
        }

        with patch("webapp.components.review_queue.st") as mock_st:
            mock_st.session_state = mock_state
            mock_st.button = MagicMock(return_value=False)
            mock_st.markdown = MagicMock()
            render_queue_item(item, 0, is_selected=False)

            # Verify markdown was called (for HTML rendering)
            assert mock_st.markdown.called

    def test_render_queue_item_selected_styles(self):
        """Selected item should have blue border and light blue background."""
        import numpy as np

        from webapp.components.review_queue import render_queue_item
        from webapp.utils.batch import BatchItem
        from webapp.utils.inference import PipelineResult
        from webapp.utils.session import SESSION_KEYS

        result = PipelineResult(
            success=True,
            stage="segmentation",
            sam_mask=np.zeros((100, 100), dtype=np.uint8),
        )
        item = BatchItem(filename="test.png", result=result)

        mock_state = {
            SESSION_KEYS["batch_queue"]: [item],
            SESSION_KEYS["current_index"]: 0,
            SESSION_KEYS["batch_initialized"]: True,
        }

        with patch("webapp.components.review_queue.st") as mock_st:
            mock_st.session_state = mock_state
            mock_st.button = MagicMock(return_value=False)
            mock_st.markdown = MagicMock()
            render_queue_item(item, 0, is_selected=True)

            # Check that markdown was called with selection styling
            calls = mock_st.markdown.call_args_list
            html_content = "".join(str(call) for call in calls)
            assert "#1E40AF" in html_content or "#EFF6FF" in html_content

    def test_render_queue_item_approved_opacity(self):
        """Approved items should have reduced opacity (0.7)."""
        import numpy as np

        from webapp.components.review_queue import render_queue_item
        from webapp.utils.batch import BatchItem
        from webapp.utils.inference import PipelineResult
        from webapp.utils.session import SESSION_KEYS

        result = PipelineResult(
            success=True,
            stage="segmentation",
            sam_mask=np.zeros((100, 100), dtype=np.uint8),
        )
        item = BatchItem(filename="test.png", result=result, status="approved")

        mock_state = {
            SESSION_KEYS["batch_queue"]: [item],
            SESSION_KEYS["current_index"]: 0,
            SESSION_KEYS["batch_initialized"]: True,
        }

        with patch("webapp.components.review_queue.st") as mock_st:
            mock_st.session_state = mock_state
            mock_st.button = MagicMock(return_value=False)
            mock_st.markdown = MagicMock()
            render_queue_item(item, 0, is_selected=False)

            # Check that opacity styling is present
            calls = mock_st.markdown.call_args_list
            html_content = "".join(str(call) for call in calls)
            assert "opacity" in html_content.lower() or "0.7" in html_content


class TestRenderProgressHeader:
    """Tests for progress header (AC #7)."""

    def test_render_progress_header_exists(self):
        """render_progress_header function should exist."""
        from webapp.components.review_queue import render_progress_header

        assert callable(render_progress_header)

    def test_render_progress_header_shows_counts(self):
        """Progress header should show X/Y reviewed format."""
        from webapp.components.review_queue import render_progress_header
        from webapp.utils.batch import BatchItem, BatchStatistics
        from webapp.utils.inference import PipelineResult
        from webapp.utils.session import SESSION_KEYS

        # Create batch with mix of statuses
        items = [
            BatchItem(
                filename=f"test{i}.png",
                result=PipelineResult(success=True, stage="segmentation"),
                status="approved" if i < 3 else "pending",
            )
            for i in range(5)
        ]

        mock_state = {
            SESSION_KEYS["batch_queue"]: items,
            SESSION_KEYS["current_index"]: 0,
            SESSION_KEYS["batch_initialized"]: True,
        }

        # Mock both st module and get_batch_statistics
        with patch("webapp.components.review_queue.st") as mock_st, \
             patch("webapp.components.review_queue.get_batch_statistics") as mock_stats:
            mock_st.session_state = mock_state
            mock_st.markdown = MagicMock()
            # Return stats showing 3/5 reviewed
            mock_stats.return_value = BatchStatistics(total=5, processed=3)
            render_progress_header()

            # Check that progress format was rendered
            calls = mock_st.markdown.call_args_list
            html_content = "".join(str(call) for call in calls)
            assert "3" in html_content and "5" in html_content


class TestRenderEmptyState:
    """Tests for empty state handling (AC #8)."""

    def test_render_empty_state_exists(self):
        """render_empty_state function should exist."""
        from webapp.components.review_queue import render_empty_state

        assert callable(render_empty_state)

    def test_render_empty_state_shows_message(self):
        """Empty state should show 'No images in queue' message."""
        from webapp.components.review_queue import render_empty_state

        with patch("webapp.components.review_queue.st") as mock_st:
            mock_st.info = MagicMock()
            mock_st.markdown = MagicMock()
            render_empty_state()

            # Check for empty state message
            all_calls = str(mock_st.info.call_args_list) + str(mock_st.markdown.call_args_list)
            assert "No images in queue" in all_calls or "upload" in all_calls.lower()


class TestFilterItems:
    """Tests for filter functionality (AC #6)."""

    def test_filter_items_all(self):
        """'all' filter should return all items."""
        from webapp.components.review_queue import filter_items
        from webapp.utils.batch import BatchItem
        from webapp.utils.inference import PipelineResult

        items = [
            BatchItem(
                filename="auto.png",
                result=PipelineResult(
                    success=True, stage="segmentation", yolo_confidence=0.95, sam_iou=0.95
                ),
            ),
            BatchItem(
                filename="review.png",
                result=PipelineResult(
                    success=True, stage="segmentation", yolo_confidence=0.7, sam_iou=0.7
                ),
            ),
            BatchItem(
                filename="manual.png",
                result=PipelineResult(success=False, stage="detection"),
            ),
        ]

        filtered = filter_items(items, "all")
        assert len(filtered) == 3

    def test_filter_items_needs_review(self):
        """'needs-review' filter should return only needs-review items."""
        from webapp.components.review_queue import filter_items
        from webapp.utils.batch import BatchItem
        from webapp.utils.inference import PipelineResult

        items = [
            BatchItem(
                filename="auto.png",
                result=PipelineResult(
                    success=True, stage="segmentation", yolo_confidence=0.95, sam_iou=0.95
                ),
            ),
            BatchItem(
                filename="review.png",
                result=PipelineResult(
                    success=True, stage="segmentation", yolo_confidence=0.7, sam_iou=0.7
                ),
            ),
            BatchItem(
                filename="manual.png",
                result=PipelineResult(success=False, stage="detection"),
            ),
        ]

        filtered = filter_items(items, "needs-review")
        assert len(filtered) == 1
        assert filtered[0].filename == "review.png"

    def test_filter_items_manual_required(self):
        """'manual-required' filter should return only manual-required items."""
        from webapp.components.review_queue import filter_items
        from webapp.utils.batch import BatchItem
        from webapp.utils.inference import PipelineResult

        items = [
            BatchItem(
                filename="auto.png",
                result=PipelineResult(
                    success=True, stage="segmentation", yolo_confidence=0.95, sam_iou=0.95
                ),
            ),
            BatchItem(
                filename="review.png",
                result=PipelineResult(
                    success=True, stage="segmentation", yolo_confidence=0.7, sam_iou=0.7
                ),
            ),
            BatchItem(
                filename="manual.png",
                result=PipelineResult(success=False, stage="detection"),
            ),
        ]

        filtered = filter_items(items, "manual-required")
        assert len(filtered) == 1
        assert filtered[0].filename == "manual.png"


class TestQueueItemClick:
    """Tests for queue item click handling (AC #5)."""

    def test_handle_item_click_updates_index(self):
        """Clicking a queue item should update current_batch_index."""
        from webapp.components.review_queue import handle_item_click
        from webapp.utils.session import SESSION_KEYS

        mock_state = {
            SESSION_KEYS["batch_queue"]: [None, None, None],  # 3 items
            SESSION_KEYS["current_index"]: 0,
            SESSION_KEYS["batch_initialized"]: True,
        }

        with patch("webapp.components.review_queue.st") as mock_st, \
             patch("webapp.components.review_queue.set_current_index") as mock_set_idx:
            mock_st.session_state = mock_state
            mock_st.rerun = MagicMock()

            handle_item_click(2)

            # Verify set_current_index was called with correct index
            mock_set_idx.assert_called_once_with(2)
            # Verify rerun was triggered
            mock_st.rerun.assert_called_once()


class TestGetConfidenceDot:
    """Tests for confidence dot rendering (AC #2)."""

    def test_get_confidence_dot_green(self):
        """Auto-approved classification should show green dot."""
        from webapp.components.review_queue import get_confidence_dot

        dot = get_confidence_dot("auto-approved")
        assert "🟢" in dot or "#22C55E" in dot

    def test_get_confidence_dot_amber(self):
        """Needs-review classification should show amber dot."""
        from webapp.components.review_queue import get_confidence_dot

        dot = get_confidence_dot("needs-review")
        assert "🟡" in dot or "#F59E0B" in dot

    def test_get_confidence_dot_red(self):
        """Manual-required classification should show red dot."""
        from webapp.components.review_queue import get_confidence_dot

        dot = get_confidence_dot("manual-required")
        assert "🔴" in dot or "#EF4444" in dot
