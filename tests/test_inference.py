"""Unit tests for webapp.utils.inference module.

Tests for PipelineResult dataclass, run_pipeline, load_models,
and confidence-based classification system.
"""

from unittest.mock import MagicMock

import numpy as np
import pytest

# These imports will fail initially (RED phase)
from webapp.utils.inference import (
    CONFIDENCE_AUTO_APPROVED,
    CONFIDENCE_NEEDS_REVIEW,
    PipelineResult,
    load_models,
    run_pipeline,
)


class TestPipelineResult:
    """Tests for PipelineResult dataclass."""

    def test_pipeline_result_success_fields(self):
        """PipelineResult should store all fields correctly."""
        result = PipelineResult(
            success=True,
            stage="segmentation",
            yolo_box=[100, 100, 200, 200],
            yolo_confidence=0.92,
            sam_mask=np.zeros((256, 256), dtype=np.uint8),
            sam_iou=0.88,
        )
        assert result.success is True
        assert result.stage == "segmentation"
        assert result.yolo_box == [100, 100, 200, 200]
        assert result.yolo_confidence == 0.92
        assert result.sam_iou == 0.88
        assert result.sam_mask is not None
        assert result.sam_mask.shape == (256, 256)

    def test_pipeline_result_failure_state(self):
        """PipelineResult should correctly represent failure."""
        result = PipelineResult(
            success=False,
            stage="detection",
            error_message="No tumor detected",
        )
        assert result.success is False
        assert result.stage == "detection"
        assert result.error_message == "No tumor detected"
        assert result.yolo_box is None
        assert result.sam_mask is None

    def test_pipeline_result_default_values(self):
        """PipelineResult should have correct default values."""
        result = PipelineResult(success=True, stage="detection")
        assert result.error_message is None
        assert result.yolo_box is None
        assert result.yolo_confidence is None
        assert result.sam_mask is None
        assert result.sam_iou is None


class TestPipelineResultConfidence:
    """Tests for PipelineResult.confidence property."""

    def test_confidence_both_scores_available(self):
        """Confidence should be minimum of YOLO and SAM scores."""
        result = PipelineResult(
            success=True,
            stage="segmentation",
            yolo_confidence=0.95,
            sam_iou=0.88,
        )
        assert result.confidence == 0.88

    def test_confidence_yolo_lower(self):
        """Confidence should be YOLO when it's lower than SAM."""
        result = PipelineResult(
            success=True,
            stage="segmentation",
            yolo_confidence=0.75,
            sam_iou=0.92,
        )
        assert result.confidence == 0.75

    def test_confidence_only_yolo_available(self):
        """Confidence should fall back to YOLO if no SAM score."""
        result = PipelineResult(
            success=True,
            stage="detection",
            yolo_confidence=0.85,
            sam_iou=None,
        )
        assert result.confidence == 0.85

    def test_confidence_no_scores_available(self):
        """Confidence should be None if no scores available."""
        result = PipelineResult(
            success=False,
            stage="detection",
            error_message="No detection",
        )
        assert result.confidence is None


class TestPipelineResultClassification:
    """Tests for PipelineResult.classification property."""

    def test_classification_auto_approved_high_confidence(self):
        """Classification should be auto-approved for confidence >= 0.90."""
        result = PipelineResult(
            success=True,
            stage="segmentation",
            yolo_confidence=0.95,
            sam_iou=0.92,
        )
        assert result.classification == "auto-approved"

    def test_classification_auto_approved_exactly_threshold(self):
        """Classification should be auto-approved for confidence == 0.90."""
        result = PipelineResult(
            success=True,
            stage="segmentation",
            yolo_confidence=0.90,
            sam_iou=0.95,
        )
        assert result.classification == "auto-approved"

    def test_classification_needs_review_medium_confidence(self):
        """Classification should be needs-review for 0.50 <= confidence < 0.90."""
        result = PipelineResult(
            success=True,
            stage="segmentation",
            yolo_confidence=0.75,
            sam_iou=0.70,
        )
        assert result.classification == "needs-review"

    def test_classification_needs_review_exactly_threshold(self):
        """Classification should be needs-review for confidence == 0.50."""
        result = PipelineResult(
            success=True,
            stage="segmentation",
            yolo_confidence=0.50,
            sam_iou=0.60,
        )
        assert result.classification == "needs-review"

    def test_classification_manual_required_low_confidence(self):
        """Classification should be manual-required for confidence < 0.50."""
        result = PipelineResult(
            success=True,
            stage="segmentation",
            yolo_confidence=0.45,
            sam_iou=0.40,
        )
        assert result.classification == "manual-required"

    def test_classification_manual_required_no_confidence(self):
        """Classification should be manual-required if no confidence score."""
        result = PipelineResult(
            success=False,
            stage="detection",
            error_message="No tumor detected",
        )
        assert result.classification == "manual-required"


class TestConfidenceConstants:
    """Tests for confidence threshold constants."""

    def test_auto_approved_threshold(self):
        """Auto-approved threshold should be 0.90."""
        assert CONFIDENCE_AUTO_APPROVED == 0.90

    def test_needs_review_threshold(self):
        """Needs-review threshold should be 0.50."""
        assert CONFIDENCE_NEEDS_REVIEW == 0.50


class TestRunPipeline:
    """Tests for run_pipeline function."""

    def test_run_pipeline_returns_pipeline_result(self, mock_models, sample_image):
        """run_pipeline should return a PipelineResult object."""
        yolo, sam_processor, sam_model = mock_models
        result = run_pipeline(sample_image, yolo, sam_processor, sam_model)
        assert isinstance(result, PipelineResult)

    def test_run_pipeline_success_on_detection(self, mock_models, sample_image):
        """run_pipeline should return success=True when detection succeeds."""
        yolo, sam_processor, sam_model = mock_models
        result = run_pipeline(sample_image, yolo, sam_processor, sam_model)
        assert result.success is True
        assert result.stage == "segmentation"

    def test_run_pipeline_failure_no_detection(
        self, mock_models_no_detection, sample_image
    ):
        """run_pipeline should return success=False when no detection."""
        yolo, sam_processor, sam_model = mock_models_no_detection
        result = run_pipeline(sample_image, yolo, sam_processor, sam_model)
        assert result.success is False
        assert result.stage == "detection"
        assert result.error_message is not None

    def test_run_pipeline_exception_handling(
        self, mock_models_exception, sample_image
    ):
        """run_pipeline should catch exceptions and return failure result."""
        yolo, sam_processor, sam_model = mock_models_exception
        result = run_pipeline(sample_image, yolo, sam_processor, sam_model)
        assert result.success is False
        assert result.error_message is not None

    def test_run_pipeline_has_yolo_box_on_success(self, mock_models, sample_image):
        """run_pipeline should populate yolo_box on successful detection."""
        yolo, sam_processor, sam_model = mock_models
        result = run_pipeline(sample_image, yolo, sam_processor, sam_model)
        assert result.yolo_box is not None
        assert len(result.yolo_box) == 4

    def test_run_pipeline_has_sam_mask_on_success(self, mock_models, sample_image):
        """run_pipeline should populate sam_mask on successful segmentation."""
        yolo, sam_processor, sam_model = mock_models
        result = run_pipeline(sample_image, yolo, sam_processor, sam_model)
        assert result.sam_mask is not None
        assert isinstance(result.sam_mask, np.ndarray)


# Fixtures for mocking models


@pytest.fixture
def sample_image():
    """Create a sample test image."""
    return np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)


@pytest.fixture
def mock_models():
    """Create mock YOLO and SAM models that return valid results."""
    # Mock YOLO model
    mock_yolo = MagicMock()
    mock_box = MagicMock()
    mock_box.xyxy.__getitem__ = MagicMock(
        return_value=MagicMock(
            cpu=MagicMock(
                return_value=MagicMock(
                    numpy=MagicMock(return_value=np.array([100, 100, 200, 200]))
                )
            )
        )
    )
    mock_box.conf.__getitem__ = MagicMock(
        return_value=MagicMock(
            cpu=MagicMock(
                return_value=MagicMock(numpy=MagicMock(return_value=np.array(0.92)))
            )
        )
    )

    mock_boxes = MagicMock()
    mock_boxes.__len__ = MagicMock(return_value=1)
    mock_boxes.__getitem__ = MagicMock(return_value=mock_box)
    mock_boxes.conf = MagicMock()
    mock_boxes.conf.argmax = MagicMock(return_value=0)

    mock_result = MagicMock()
    mock_result.boxes = mock_boxes

    mock_yolo.return_value = [mock_result]

    # Mock SAM processor
    mock_processor = MagicMock()
    mock_inputs = {
        "pixel_values": MagicMock(),
        "input_boxes": MagicMock(),
    }
    mock_processor.return_value = mock_inputs

    # Mock SAM model
    mock_sam = MagicMock()
    mock_param = MagicMock()
    mock_param.device = "cpu"
    mock_sam.parameters.return_value = iter([mock_param])
    mock_outputs = MagicMock()
    mock_outputs.pred_masks = MagicMock()
    mock_outputs.pred_masks.cpu.return_value.numpy.return_value = np.ones(
        (1, 1, 1, 256, 256)
    )
    mock_sam.return_value = mock_outputs

    return mock_yolo, mock_processor, mock_sam


@pytest.fixture
def mock_models_no_detection():
    """Create mock models where YOLO returns no detection."""
    mock_yolo = MagicMock()
    mock_boxes = MagicMock()
    mock_boxes.__len__ = MagicMock(return_value=0)

    mock_result = MagicMock()
    mock_result.boxes = mock_boxes

    mock_yolo.return_value = [mock_result]

    mock_processor = MagicMock()
    mock_sam = MagicMock()

    return mock_yolo, mock_processor, mock_sam


@pytest.fixture
def mock_models_exception():
    """Create mock models that raise an exception."""
    mock_yolo = MagicMock()
    mock_yolo.side_effect = RuntimeError("Model inference failed")

    mock_processor = MagicMock()
    mock_sam = MagicMock()

    return mock_yolo, mock_processor, mock_sam
