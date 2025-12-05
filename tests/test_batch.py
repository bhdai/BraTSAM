"""Tests for batch processing queue and statistics (Story 2.4, AC #6)."""

import pytest

from webapp.utils.batch import (
    BatchItem,
    BatchStatistics,
    compute_batch_statistics,
    get_items_by_classification,
)
from webapp.utils.inference import PipelineResult


class TestBatchItem:
    """Tests for BatchItem dataclass."""

    def test_batch_item_creation(self):
        """BatchItem should store all fields correctly."""
        result = PipelineResult(
            success=True,
            stage="segmentation",
            yolo_confidence=0.95,
            sam_iou=0.92,
        )
        item = BatchItem(
            filename="test_image.png",
            result=result,
            status="pending",
        )
        assert item.filename == "test_image.png"
        assert item.result is result
        assert item.status == "pending"
        assert item.user_box is None

    def test_batch_item_default_status(self):
        """BatchItem should default to pending status."""
        result = PipelineResult(success=True, stage="segmentation")
        item = BatchItem(filename="test.png", result=result)
        assert item.status == "pending"

    def test_batch_item_delegates_classification(self):
        """BatchItem.classification should delegate to PipelineResult."""
        result = PipelineResult(
            success=True,
            stage="segmentation",
            yolo_confidence=0.95,
            sam_iou=0.92,
        )
        item = BatchItem(filename="test.png", result=result)
        assert item.classification == "auto-approved"
        assert item.classification == result.classification

    def test_batch_item_delegates_confidence(self):
        """BatchItem.confidence should delegate to PipelineResult."""
        result = PipelineResult(
            success=True,
            stage="segmentation",
            yolo_confidence=0.87,
            sam_iou=0.85,
        )
        item = BatchItem(filename="test.png", result=result)
        assert item.confidence == 0.85
        assert item.confidence == result.confidence

    def test_batch_item_with_user_override(self):
        """BatchItem should store manual bounding box override."""
        result = PipelineResult(success=True, stage="segmentation")
        item = BatchItem(
            filename="test.png",
            result=result,
            status="edited",
            user_box=[100, 150, 300, 400],
        )
        assert item.user_box == [100, 150, 300, 400]
        assert item.status == "edited"


class TestBatchStatistics:
    """Tests for BatchStatistics dataclass."""

    def test_batch_statistics_defaults(self):
        """BatchStatistics should initialize with zero counts."""
        stats = BatchStatistics()
        assert stats.total == 0
        assert stats.auto_approved == 0
        assert stats.needs_review == 0
        assert stats.manual_required == 0
        assert stats.processed == 0

    def test_batch_statistics_pending_property(self):
        """BatchStatistics.pending should compute remaining items."""
        stats = BatchStatistics(total=10, processed=3)
        assert stats.pending == 7


class TestComputeBatchStatistics:
    """Tests for compute_batch_statistics function (AC #6)."""

    def test_compute_batch_statistics_empty_list(self):
        """compute_batch_statistics should handle empty list."""
        stats = compute_batch_statistics([])
        assert stats.total == 0
        assert stats.auto_approved == 0
        assert stats.needs_review == 0
        assert stats.manual_required == 0

    def test_compute_batch_statistics_counts_classifications(self):
        """compute_batch_statistics should count items by classification tier."""
        items = [
            BatchItem(
                filename="auto1.png",
                result=PipelineResult(
                    success=True, stage="segmentation", yolo_confidence=0.95, sam_iou=0.92
                ),
            ),
            BatchItem(
                filename="auto2.png",
                result=PipelineResult(
                    success=True, stage="segmentation", yolo_confidence=0.91, sam_iou=0.90
                ),
            ),
            BatchItem(
                filename="review1.png",
                result=PipelineResult(
                    success=True, stage="segmentation", yolo_confidence=0.75, sam_iou=0.70
                ),
            ),
            BatchItem(
                filename="manual1.png",
                result=PipelineResult(
                    success=False, stage="detection", error_message="No tumor"
                ),
            ),
        ]

        stats = compute_batch_statistics(items)

        assert stats.total == 4
        assert stats.auto_approved == 2
        assert stats.needs_review == 1
        assert stats.manual_required == 1

    def test_compute_batch_statistics_counts_processed(self):
        """compute_batch_statistics should count processed items."""
        result = PipelineResult(
            success=True, stage="segmentation", yolo_confidence=0.95, sam_iou=0.92
        )
        items = [
            BatchItem(filename="pending.png", result=result, status="pending"),
            BatchItem(filename="approved.png", result=result, status="approved"),
            BatchItem(filename="rejected.png", result=result, status="rejected"),
            BatchItem(filename="edited.png", result=result, status="edited"),
        ]

        stats = compute_batch_statistics(items)

        assert stats.total == 4
        assert stats.processed == 3  # approved, rejected, edited
        assert stats.pending == 1  # Only pending


class TestGetItemsByClassification:
    """Tests for get_items_by_classification function (AC #6)."""

    @pytest.fixture
    def mixed_batch(self):
        """Create a batch with items in all classification tiers."""
        return [
            BatchItem(
                filename="auto1.png",
                result=PipelineResult(
                    success=True, stage="segmentation", yolo_confidence=0.95, sam_iou=0.92
                ),
            ),
            BatchItem(
                filename="auto2.png",
                result=PipelineResult(
                    success=True, stage="segmentation", yolo_confidence=0.90, sam_iou=0.91
                ),
            ),
            BatchItem(
                filename="review1.png",
                result=PipelineResult(
                    success=True, stage="segmentation", yolo_confidence=0.75, sam_iou=0.70
                ),
            ),
            BatchItem(
                filename="manual1.png",
                result=PipelineResult(
                    success=False, stage="detection"
                ),
            ),
        ]

    def test_filter_auto_approved(self, mixed_batch):
        """get_items_by_classification should filter auto-approved items."""
        auto = get_items_by_classification(mixed_batch, "auto-approved")
        assert len(auto) == 2
        assert all(item.classification == "auto-approved" for item in auto)

    def test_filter_needs_review(self, mixed_batch):
        """get_items_by_classification should filter needs-review items."""
        review = get_items_by_classification(mixed_batch, "needs-review")
        assert len(review) == 1
        assert review[0].filename == "review1.png"

    def test_filter_manual_required(self, mixed_batch):
        """get_items_by_classification should filter manual-required items."""
        manual = get_items_by_classification(mixed_batch, "manual-required")
        assert len(manual) == 1
        assert manual[0].filename == "manual1.png"

    def test_filter_returns_empty_if_none_match(self, mixed_batch):
        """get_items_by_classification should return empty list if no matches."""
        # Remove all auto-approved items from batch
        batch_no_auto = [i for i in mixed_batch if i.classification != "auto-approved"]
        auto = get_items_by_classification(batch_no_auto, "auto-approved")
        assert len(auto) == 0


class TestBatchItemClassificationMapping:
    """Tests verifying BatchItem classification maps to correct colors (AC #6)."""

    @pytest.mark.parametrize(
        "yolo_conf,sam_iou,expected_class",
        [
            (0.95, 0.92, "auto-approved"),
            (0.75, 0.70, "needs-review"),
            (0.45, 0.40, "manual-required"),
        ],
    )
    def test_batch_item_classification_matches_result(
        self, yolo_conf, sam_iou, expected_class
    ):
        """AC #6: Each BatchItem should have independent classification."""
        result = PipelineResult(
            success=True,
            stage="segmentation",
            yolo_confidence=yolo_conf,
            sam_iou=sam_iou,
        )
        item = BatchItem(filename="test.png", result=result)
        assert item.classification == expected_class
