"""Unit tests for webapp.utils.inference module.

Tests for PipelineResult dataclass, run_pipeline, load_models,
confidence-based classification system, YOLO detection, and SAM segmentation.
"""

import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

# These imports will fail initially (RED phase)
from webapp.utils.inference import (
    CONFIDENCE_AUTO_APPROVED,
    CONFIDENCE_NEEDS_REVIEW,
    MASK_BINARY_THRESHOLD,
    YOLO_MIN_CONFIDENCE,
    PipelineResult,
    _detect_tumor,
    _segment_tumor,
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

    # Create confidence tensor supporting filtering operations
    conf_tensor = MagicMock()
    conf_values = np.array([0.92])

    # Handle >= comparison for filtering
    mask_tensor = MagicMock()
    mask_tensor.any = MagicMock(return_value=True)
    conf_tensor.__ge__ = MagicMock(return_value=mask_tensor)

    # Filtered conf tensor
    filtered_conf = MagicMock()
    filtered_conf.argmax = MagicMock(return_value=0)
    filtered_conf.__getitem__ = MagicMock(
        return_value=MagicMock(
            cpu=MagicMock(
                return_value=MagicMock(numpy=MagicMock(return_value=np.array(0.92)))
            )
        )
    )
    conf_tensor.__getitem__ = MagicMock(return_value=filtered_conf)

    # Create xyxy tensor supporting filtering operations
    xyxy_tensor = MagicMock()
    filtered_xyxy = MagicMock()
    filtered_xyxy.__getitem__ = MagicMock(
        return_value=MagicMock(
            cpu=MagicMock(
                return_value=MagicMock(
                    numpy=MagicMock(return_value=np.array([100, 100, 200, 200]))
                )
            )
        )
    )
    xyxy_tensor.__getitem__ = MagicMock(return_value=filtered_xyxy)

    mock_boxes = MagicMock()
    mock_boxes.__len__ = MagicMock(return_value=1)
    mock_boxes.conf = conf_tensor
    mock_boxes.xyxy = xyxy_tensor

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

    # Mock SAM model (SamFineTuner wrapper)
    mock_sam = MagicMock()
    mock_param = MagicMock()
    mock_param.device = torch.device("cpu")
    mock_sam.parameters.return_value = iter([mock_param])

    # Mock sam_model() call with full_outputs=True - returns dict
    mock_sam.return_value = {
        'pred_masks': torch.ones(1, 1, 3, 256, 256),  # [B, N, M, H, W]
        'iou_scores': torch.tensor([[[0.85, 0.80, 0.75]]]),  # [B, N, M]
    }

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


# YOLO Detection Tests (Story 2.2)


class TestYoloMinConfidenceConstant:
    """Tests for YOLO minimum confidence threshold constant."""

    def test_yolo_min_confidence_exists(self):
        """YOLO_MIN_CONFIDENCE constant should exist."""
        assert YOLO_MIN_CONFIDENCE is not None

    def test_yolo_min_confidence_value(self):
        """YOLO_MIN_CONFIDENCE should be 0.25."""
        assert YOLO_MIN_CONFIDENCE == 0.25

    def test_yolo_min_confidence_reasonable_range(self):
        """YOLO_MIN_CONFIDENCE should be between 0 and 1."""
        assert 0.0 < YOLO_MIN_CONFIDENCE < 1.0


class TestDetectTumor:
    """Tests for _detect_tumor function."""

    def test_detect_tumor_returns_tuple(self, mock_yolo_single_detection, sample_image):
        """_detect_tumor should return a tuple of (coords, confidence)."""
        result = _detect_tumor(sample_image, mock_yolo_single_detection)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_detect_tumor_successful_detection_returns_box(
        self, mock_yolo_single_detection, sample_image
    ):
        """_detect_tumor should return box coordinates on successful detection."""
        coords, confidence = _detect_tumor(sample_image, mock_yolo_single_detection)
        assert coords is not None
        assert len(coords) == 4

    def test_detect_tumor_box_format_xyxy(
        self, mock_yolo_single_detection, sample_image
    ):
        """_detect_tumor should return box in [x1, y1, x2, y2] format."""
        coords, _ = _detect_tumor(sample_image, mock_yolo_single_detection)
        assert coords is not None
        x1, y1, x2, y2 = coords
        # Box should have x2 > x1 and y2 > y1
        assert x2 > x1
        assert y2 > y1

    def test_detect_tumor_box_coordinates_are_integers(
        self, mock_yolo_single_detection, sample_image
    ):
        """_detect_tumor should return integer pixel coordinates."""
        coords, _ = _detect_tumor(sample_image, mock_yolo_single_detection)
        assert coords is not None
        assert all(isinstance(c, int) for c in coords)

    def test_detect_tumor_returns_confidence_score(
        self, mock_yolo_single_detection, sample_image
    ):
        """_detect_tumor should return a confidence score between 0 and 1."""
        _, confidence = _detect_tumor(sample_image, mock_yolo_single_detection)
        assert confidence is not None
        assert 0.0 <= confidence <= 1.0

    def test_detect_tumor_no_detection_returns_none(
        self, mock_yolo_no_detection, sample_image
    ):
        """_detect_tumor should return (None, None) when no detection."""
        coords, confidence = _detect_tumor(sample_image, mock_yolo_no_detection)
        assert coords is None
        assert confidence is None

    def test_detect_tumor_multiple_detections_selects_highest_confidence(
        self, mock_yolo_multiple_detections, sample_image
    ):
        """_detect_tumor should select the highest confidence detection."""
        coords, confidence = _detect_tumor(
            sample_image, mock_yolo_multiple_detections
        )
        assert coords is not None
        # Should select box with confidence 0.92 (highest)
        assert confidence == 0.92
        # The high confidence box coordinates
        assert coords == [100, 100, 200, 200]

    def test_detect_tumor_filters_low_confidence_detections(
        self, mock_yolo_low_confidence, sample_image
    ):
        """_detect_tumor should filter out detections below YOLO_MIN_CONFIDENCE."""
        coords, confidence = _detect_tumor(sample_image, mock_yolo_low_confidence)
        # All detections below threshold should be filtered out
        assert coords is None
        assert confidence is None

    def test_detect_tumor_logs_timing(
        self, mock_yolo_single_detection, sample_image, caplog
    ):
        """_detect_tumor should log detection timing."""
        import logging

        with caplog.at_level(logging.INFO):
            _detect_tumor(sample_image, mock_yolo_single_detection)
        # Should log timing information
        assert any("time=" in record.message for record in caplog.records)


# Additional YOLO fixtures


@pytest.fixture
def mock_yolo_single_detection():
    """Create mock YOLO that returns a single valid detection."""
    mock_yolo = MagicMock()

    # Create confidence tensor supporting filtering operations
    conf_tensor = MagicMock()

    # Handle >= comparison for filtering
    mask_tensor = MagicMock()
    mask_tensor.any = MagicMock(return_value=True)
    conf_tensor.__ge__ = MagicMock(return_value=mask_tensor)

    # Filtered conf tensor
    filtered_conf = MagicMock()
    filtered_conf.argmax = MagicMock(return_value=0)
    filtered_conf.__getitem__ = MagicMock(
        return_value=MagicMock(
            cpu=MagicMock(
                return_value=MagicMock(numpy=MagicMock(return_value=np.array(0.85)))
            )
        )
    )
    conf_tensor.__getitem__ = MagicMock(return_value=filtered_conf)

    # Create xyxy tensor supporting filtering operations
    xyxy_tensor = MagicMock()
    filtered_xyxy = MagicMock()
    filtered_xyxy.__getitem__ = MagicMock(
        return_value=MagicMock(
            cpu=MagicMock(
                return_value=MagicMock(
                    numpy=MagicMock(return_value=np.array([100, 100, 200, 200]))
                )
            )
        )
    )
    xyxy_tensor.__getitem__ = MagicMock(return_value=filtered_xyxy)

    # Create boxes container
    mock_boxes = MagicMock()
    mock_boxes.__len__ = MagicMock(return_value=1)
    mock_boxes.conf = conf_tensor
    mock_boxes.xyxy = xyxy_tensor

    mock_result = MagicMock()
    mock_result.boxes = mock_boxes

    mock_yolo.return_value = [mock_result]
    return mock_yolo


@pytest.fixture
def mock_yolo_no_detection():
    """Create mock YOLO that returns no detections."""
    mock_yolo = MagicMock()
    mock_boxes = MagicMock()
    mock_boxes.__len__ = MagicMock(return_value=0)

    mock_result = MagicMock()
    mock_result.boxes = mock_boxes

    mock_yolo.return_value = [mock_result]
    return mock_yolo


@pytest.fixture
def mock_yolo_multiple_detections():
    """Create mock YOLO that returns multiple detections."""
    mock_yolo = MagicMock()

    # Create confidence tensor [0.75, 0.92, 0.60]
    conf_values = np.array([0.75, 0.92, 0.60])
    xyxy_values = np.array([
        [50, 50, 150, 150],    # conf 0.75
        [100, 100, 200, 200],  # conf 0.92 (highest)
        [200, 200, 300, 300],  # conf 0.60
    ])

    mock_boxes = MagicMock()
    mock_boxes.__len__ = MagicMock(return_value=3)

    # Setup conf tensor with filtering support
    conf_tensor = MagicMock()
    conf_tensor.__ge__ = MagicMock(
        return_value=MagicMock(any=MagicMock(return_value=True))
    )

    # After filtering, return filtered values (all above 0.25 threshold)
    filtered_conf = MagicMock()
    filtered_conf.argmax = MagicMock(return_value=1)  # Index 1 has 0.92
    filtered_conf.__getitem__ = MagicMock(
        return_value=MagicMock(
            cpu=MagicMock(
                return_value=MagicMock(numpy=MagicMock(return_value=np.array(0.92)))
            )
        )
    )
    conf_tensor.__getitem__ = MagicMock(return_value=filtered_conf)

    # Setup xyxy tensor
    xyxy_tensor = MagicMock()
    filtered_xyxy = MagicMock()
    filtered_xyxy.__getitem__ = MagicMock(
        return_value=MagicMock(
            cpu=MagicMock(
                return_value=MagicMock(
                    numpy=MagicMock(return_value=np.array([100, 100, 200, 200]))
                )
            )
        )
    )
    xyxy_tensor.__getitem__ = MagicMock(return_value=filtered_xyxy)

    mock_boxes.conf = conf_tensor
    mock_boxes.xyxy = xyxy_tensor

    mock_result = MagicMock()
    mock_result.boxes = mock_boxes

    mock_yolo.return_value = [mock_result]
    return mock_yolo


@pytest.fixture
def mock_yolo_low_confidence():
    """Create mock YOLO with all detections below threshold."""
    mock_yolo = MagicMock()

    # All detections below YOLO_MIN_CONFIDENCE (0.25)
    mock_boxes = MagicMock()
    mock_boxes.__len__ = MagicMock(return_value=2)

    conf_tensor = MagicMock()
    # Filtering returns no True values
    conf_tensor.__ge__ = MagicMock(
        return_value=MagicMock(any=MagicMock(return_value=False))
    )
    mock_boxes.conf = conf_tensor

    mock_result = MagicMock()
    mock_result.boxes = mock_boxes

    mock_yolo.return_value = [mock_result]
    return mock_yolo


# Integration tests with real YOLO model


@pytest.mark.integration
@pytest.mark.slow
class TestYoloIntegration:
    """Integration tests using the real YOLO model.

    These tests require GPU and the actual yolo_model.pt file.
    Mark with @pytest.mark.integration and @pytest.mark.slow.
    """

    @pytest.fixture
    def yolo_model_path(self):
        """Path to the real YOLO model."""
        from pathlib import Path

        model_path = Path(__file__).parent.parent / "models" / "yolo_model.pt"
        if not model_path.exists():
            pytest.skip(f"YOLO model not found at {model_path}")
        return model_path

    @pytest.fixture
    def real_yolo_model(self, yolo_model_path):
        """Load the real YOLO model."""
        from ultralytics import YOLO

        return YOLO(str(yolo_model_path))

    @pytest.fixture
    def sample_brain_image(self):
        """Create a sample brain-like image for testing.

        This is a synthetic image; real integration should use actual brain MRI.
        """
        # Create a 256x256 grayscale image with a bright region (simulated tumor)
        image = np.zeros((256, 256), dtype=np.uint8)
        # Add some background texture
        image += np.random.randint(20, 50, (256, 256), dtype=np.uint8)
        # Add a bright elliptical region (simulated tumor)
        y, x = np.ogrid[:256, :256]
        center_x, center_y = 128, 128
        radius_x, radius_y = 40, 30
        mask = ((x - center_x) / radius_x) ** 2 + ((y - center_y) / radius_y) ** 2 <= 1
        image[mask] = np.random.randint(180, 220, mask.sum(), dtype=np.uint8)
        # Convert to RGB
        rgb_image = np.stack([image, image, image], axis=-1)
        return rgb_image

    def test_real_yolo_model_loads(self, real_yolo_model):
        """Real YOLO model should load successfully."""
        assert real_yolo_model is not None

    def test_real_yolo_detection_returns_results(
        self, real_yolo_model, sample_brain_image
    ):
        """Real YOLO model should return detection results."""
        results = real_yolo_model(sample_brain_image, verbose=False)
        assert results is not None
        assert len(results) > 0

    def test_detect_tumor_with_real_model_timing(
        self, real_yolo_model, sample_brain_image
    ):
        """Detection with real model should complete within timing requirements.

        NFR: <1 second per slice on GPU (allowing more time for CPU/cold start).
        """
        start_time = time.perf_counter()
        coords, confidence = _detect_tumor(sample_brain_image, real_yolo_model)
        elapsed = time.perf_counter() - start_time

        # Allow 5 seconds for first inference (cold start) or CPU
        # Production requirement is <1s on GPU after warm-up
        assert elapsed < 5.0, f"Detection took {elapsed:.2f}s, expected <5s"

    def test_detect_tumor_with_real_model_returns_valid_format(
        self, real_yolo_model, sample_brain_image
    ):
        """Detection with real model should return valid box format if detected."""
        coords, confidence = _detect_tumor(sample_brain_image, real_yolo_model)

        # Detection may or may not find tumor in synthetic image
        if coords is not None:
            # Verify box format
            assert len(coords) == 4
            assert all(isinstance(c, int) for c in coords)
            x1, y1, x2, y2 = coords
            assert x2 > x1
            assert y2 > y1
            # Verify confidence
            assert confidence is not None
            assert 0.0 <= confidence <= 1.0
            assert confidence >= YOLO_MIN_CONFIDENCE


# SAM Segmentation Tests (Story 2.3)


class TestMaskBinaryThresholdConstant:
    """Tests for MASK_BINARY_THRESHOLD constant."""

    def test_mask_binary_threshold_exists(self):
        """MASK_BINARY_THRESHOLD constant should exist."""
        assert MASK_BINARY_THRESHOLD is not None

    def test_mask_binary_threshold_value(self):
        """MASK_BINARY_THRESHOLD should be 127."""
        assert MASK_BINARY_THRESHOLD == 127

    def test_mask_binary_threshold_reasonable_range(self):
        """MASK_BINARY_THRESHOLD should be between 0 and 255."""
        assert 0 < MASK_BINARY_THRESHOLD < 255


class TestSegmentTumor:
    """Tests for _segment_tumor function."""

    def test_segment_tumor_returns_tuple(
        self, mock_sam_processor, mock_sam_model, sample_image
    ):
        """_segment_tumor should return a tuple of (mask, iou_score)."""
        box = [100, 100, 200, 200]
        result = _segment_tumor(sample_image, box, mock_sam_processor, mock_sam_model)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_segment_tumor_returns_binary_mask(
        self, mock_sam_processor, mock_sam_model, sample_image
    ):
        """_segment_tumor should return a binary numpy mask."""
        box = [100, 100, 200, 200]
        mask, _ = _segment_tumor(sample_image, box, mock_sam_processor, mock_sam_model)
        assert isinstance(mask, np.ndarray)
        assert mask.dtype == np.uint8
        # Binary mask should only contain 0 and 1
        unique_values = np.unique(mask)
        assert all(v in [0, 1] for v in unique_values)

    def test_segment_tumor_mask_dimensions_match_input(
        self, mock_sam_processor, mock_sam_model_256_output, sample_image
    ):
        """_segment_tumor mask should have same dimensions as input image."""
        box = [100, 100, 200, 200]
        mask, _ = _segment_tumor(
            sample_image, box, mock_sam_processor, mock_sam_model_256_output
        )
        # Input is (256, 256, 3), mask should be (256, 256)
        assert mask.shape == sample_image.shape[:2]

    def test_segment_tumor_mask_resized_from_256(
        self, mock_sam_processor, mock_sam_model_256_output
    ):
        """_segment_tumor should resize SAM's 256x256 output to match input size."""
        # Create non-256x256 input image
        input_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        box = [100, 100, 200, 200]
        mask, _ = _segment_tumor(
            input_image, box, mock_sam_processor, mock_sam_model_256_output
        )
        # Should resize to 512x512
        assert mask.shape == (512, 512)

    def test_segment_tumor_mask_resized_nonsquare(
        self, mock_sam_processor, mock_sam_model_256_output
    ):
        """_segment_tumor should handle non-square input images."""
        # Create non-square input image
        input_image = np.random.randint(0, 255, (224, 336, 3), dtype=np.uint8)
        box = [50, 50, 150, 150]
        mask, _ = _segment_tumor(
            input_image, box, mock_sam_processor, mock_sam_model_256_output
        )
        # Should resize to match input dimensions
        assert mask.shape == (224, 336)

    def test_segment_tumor_returns_iou_score(
        self, mock_sam_processor, mock_sam_model_with_iou, sample_image
    ):
        """_segment_tumor should return IoU score between 0 and 1."""
        box = [100, 100, 200, 200]
        _, iou_score = _segment_tumor(
            sample_image, box, mock_sam_processor, mock_sam_model_with_iou
        )
        assert iou_score is not None
        assert isinstance(iou_score, float)
        assert 0.0 <= iou_score <= 1.0

    def test_segment_tumor_extracts_iou_from_output(
        self, mock_sam_processor, mock_sam_model_with_iou, sample_image
    ):
        """_segment_tumor should extract IoU from outputs.iou_scores."""
        box = [100, 100, 200, 200]
        _, iou_score = _segment_tumor(
            sample_image, box, mock_sam_processor, mock_sam_model_with_iou
        )
        # Mock returns 0.85
        assert iou_score == pytest.approx(0.85, rel=1e-5)

    def test_segment_tumor_iou_fallback_when_unavailable(
        self, mock_sam_processor, mock_sam_model_no_iou, sample_image
    ):
        """_segment_tumor should use fallback heuristic when IoU unavailable."""
        box = [100, 100, 200, 200]
        _, iou_score = _segment_tumor(
            sample_image, box, mock_sam_processor, mock_sam_model_no_iou
        )
        # Fallback should return a score between 0.5 and 0.95
        assert iou_score is not None
        assert 0.5 <= iou_score <= 0.95

    def test_segment_tumor_mask_preserves_binary_after_resize(
        self, mock_sam_processor, mock_sam_model_256_output
    ):
        """_segment_tumor resized mask should still be binary (0 or 1)."""
        input_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        box = [100, 100, 200, 200]
        mask, _ = _segment_tumor(
            input_image, box, mock_sam_processor, mock_sam_model_256_output
        )
        # After resize, should still be binary
        unique_values = np.unique(mask)
        assert all(v in [0, 1] for v in unique_values)

    def test_segment_tumor_logs_timing(
        self, mock_sam_processor, mock_sam_model, sample_image, caplog
    ):
        """_segment_tumor should log timing information."""
        import logging

        box = [100, 100, 200, 200]
        with caplog.at_level(logging.INFO):
            _segment_tumor(sample_image, box, mock_sam_processor, mock_sam_model)
        # Should log timing information
        assert any("time=" in record.message for record in caplog.records)

    def test_segment_tumor_logs_mask_pixels(
        self, mock_sam_processor, mock_sam_model, sample_image, caplog
    ):
        """_segment_tumor should log mask pixel count."""
        import logging

        box = [100, 100, 200, 200]
        with caplog.at_level(logging.INFO):
            _segment_tumor(sample_image, box, mock_sam_processor, mock_sam_model)
        # Should log mask_pixels
        assert any("mask_pixels=" in record.message for record in caplog.records)

    def test_segment_tumor_logs_iou_score(
        self, mock_sam_processor, mock_sam_model_with_iou, sample_image, caplog
    ):
        """_segment_tumor should log IoU score."""
        import logging

        box = [100, 100, 200, 200]
        with caplog.at_level(logging.INFO):
            _segment_tumor(
                sample_image, box, mock_sam_processor, mock_sam_model_with_iou
            )
        # Should log iou=
        assert any("iou=" in record.message for record in caplog.records)

    def test_segment_tumor_selects_best_mask_by_iou(
        self, mock_sam_processor, sample_image
    ):
        """_segment_tumor should select the mask with highest IoU score."""
        # Create a mock where the second mask (index 1) has highest IoU
        model = MagicMock()
        mock_param = MagicMock()
        mock_param.device = torch.device("cpu")
        model.parameters.return_value = iter([mock_param])

        # Create distinct masks for each proposal
        mask_data = torch.zeros(1, 1, 3, 256, 256)
        mask_data[0, 0, 0, 50:100, 50:100] = 1.0    # Mask 0: IoU 0.70
        mask_data[0, 0, 1, 100:200, 100:200] = 1.0  # Mask 1: IoU 0.95 (BEST)
        mask_data[0, 0, 2, 150:180, 150:180] = 1.0  # Mask 2: IoU 0.60

        # Second mask (index 1) has highest IoU
        model.return_value = {
            'pred_masks': mask_data,
            'iou_scores': torch.tensor([[[0.70, 0.95, 0.60]]]),  # [B, N, M]
        }

        box = [100, 100, 200, 200]
        mask, iou_score = _segment_tumor(sample_image, box, mock_sam_processor, model)

        # Should return IoU for best mask (0.95)
        assert iou_score == pytest.approx(0.95, rel=1e-5)
        # Mask should be the second proposal (100:200, 100:200 region)
        # which covers more area than mask 0 or 2
        assert mask.sum() > 0


# SAM Fixtures


@pytest.fixture
def mock_sam_processor():
    """Create mock SAM processor."""
    processor = MagicMock()
    processor.return_value = {
        "pixel_values": torch.zeros(1, 3, 1024, 1024),
        "input_boxes": torch.tensor([[[[100, 100, 200, 200]]]]),
    }
    return processor


@pytest.fixture
def mock_sam_model():
    """Create mock SAM model with matching output dimensions."""
    model = MagicMock()
    # Mock parameters for device detection
    mock_param = MagicMock()
    mock_param.device = torch.device("cpu")
    model.parameters.return_value = iter([mock_param])

    # Mock sam_model() call with full_outputs=True - returns dict
    # [B, N, M, H, W] format with M=3 mask proposals
    model.return_value = {
        'pred_masks': torch.ones(1, 1, 3, 256, 256),
        'iou_scores': torch.tensor([[[0.85, 0.80, 0.75]]]),  # [B, N, M]
    }
    return model


@pytest.fixture
def mock_sam_model_256_output():
    """Create mock SAM model that outputs 256x256 masks."""
    model = MagicMock()
    mock_param = MagicMock()
    mock_param.device = torch.device("cpu")
    model.parameters.return_value = iter([mock_param])

    # SAM default output is 256x256 with M=3 mask proposals
    mask_data = torch.zeros(1, 1, 3, 256, 256)
    # Add some variation (partial mask) for each proposal
    mask_data[0, 0, 0, 100:150, 100:150] = 1.0  # Best mask (IoU 0.88)
    mask_data[0, 0, 1, 90:160, 90:160] = 1.0    # Second mask
    mask_data[0, 0, 2, 80:170, 80:170] = 1.0    # Third mask

    # Mock sam_model() call with full_outputs=True - returns dict
    model.return_value = {
        'pred_masks': mask_data,
        'iou_scores': torch.tensor([[[0.88, 0.82, 0.78]]]),  # [B, N, M]
    }
    return model


@pytest.fixture
def mock_sam_model_with_iou():
    """Create mock SAM model with IoU scores available."""
    model = MagicMock()
    mock_param = MagicMock()
    mock_param.device = torch.device("cpu")
    model.parameters.return_value = iter([mock_param])

    # Mock sam_model() call with full_outputs=True - returns dict
    # First mask (idx 0) has highest IoU (0.85)
    model.return_value = {
        'pred_masks': torch.ones(1, 1, 3, 256, 256),  # [B, N, M, H, W]
        'iou_scores': torch.tensor([[[0.85, 0.80, 0.75]]]),  # [B, N, M]
    }
    return model


@pytest.fixture
def mock_sam_model_no_iou():
    """Create mock SAM model without IoU scores (tests fallback)."""
    model = MagicMock()
    mock_param = MagicMock()
    mock_param.device = torch.device("cpu")
    model.parameters.return_value = iter([mock_param])

    # Create mask with ~10% coverage for fallback heuristic
    mask_data = torch.zeros(1, 1, 3, 256, 256)
    # Fill ~10% of the mask (for first proposal)
    mask_data[0, 0, 0, 100:130, 100:180] = 1.0

    # Mock sam_model() call with full_outputs=True - returns dict
    # iou_scores is None to trigger fallback
    model.return_value = {
        'pred_masks': mask_data,
        'iou_scores': None,
    }
    return model


# SAM Integration Tests (Story 2.3)


@pytest.mark.integration
@pytest.mark.slow
class TestSamIntegration:
    """Integration tests using the real SAM model.

    These tests require GPU and the actual sam_model.pth file.
    Mark with @pytest.mark.integration and @pytest.mark.slow.
    """

    @pytest.fixture
    def sam_model_path(self):
        """Path to the real SAM model."""
        from pathlib import Path

        model_path = Path(__file__).parent.parent / "models" / "sam_model.pth"
        if not model_path.exists():
            pytest.skip(f"SAM model not found at {model_path}")
        return model_path

    @pytest.fixture
    def real_sam_models(self, sam_model_path):
        """Load the real SAM processor and model with LoRA."""
        import sys
        from pathlib import Path

        import torch
        from transformers import SamProcessor

        # Add project root to path for model import
        project_root = Path(__file__).parent.parent
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))

        from model import SamFineTuner

        processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
        model = SamFineTuner(use_lora=True)
        model.load_state_dict(torch.load(str(sam_model_path), map_location="cpu"))
        model.eval()
        return processor, model

    @pytest.fixture
    def sample_tumor_image(self):
        """Create a sample image with a tumor-like region for testing."""
        # Create a 256x256 grayscale image with a bright region
        image = np.zeros((256, 256), dtype=np.uint8)
        image += np.random.randint(20, 50, (256, 256), dtype=np.uint8)
        # Add a bright elliptical region (simulated tumor)
        y, x = np.ogrid[:256, :256]
        center_x, center_y = 150, 150
        radius_x, radius_y = 40, 30
        mask = ((x - center_x) / radius_x) ** 2 + ((y - center_y) / radius_y) ** 2 <= 1
        image[mask] = np.random.randint(180, 220, mask.sum(), dtype=np.uint8)
        # Convert to RGB
        rgb_image = np.stack([image, image, image], axis=-1)
        return rgb_image

    def test_real_sam_model_loads(self, real_sam_models):
        """Real SAM model should load successfully."""
        processor, model = real_sam_models
        assert processor is not None
        assert model is not None

    def test_segment_tumor_with_real_model_returns_valid_mask(
        self, real_sam_models, sample_tumor_image
    ):
        """Segmentation with real model should return valid binary mask."""
        processor, model = real_sam_models
        # Bounding box around the tumor region
        box = [110, 120, 190, 180]
        mask, iou_score = _segment_tumor(sample_tumor_image, box, processor, model)

        # Verify mask properties
        assert isinstance(mask, np.ndarray)
        assert mask.dtype == np.uint8
        assert mask.shape == sample_tumor_image.shape[:2]
        # Should be binary
        unique_values = np.unique(mask)
        assert all(v in [0, 1] for v in unique_values)

    def test_segment_tumor_with_real_model_returns_iou(
        self, real_sam_models, sample_tumor_image
    ):
        """Segmentation with real model should return IoU score."""
        processor, model = real_sam_models
        box = [110, 120, 190, 180]
        _, iou_score = _segment_tumor(sample_tumor_image, box, processor, model)

        assert iou_score is not None
        assert isinstance(iou_score, float)
        # SAM IoU predictions can occasionally be slightly > 1.0
        assert 0.0 <= iou_score <= 1.1

    def test_segment_tumor_timing_requirements(
        self, real_sam_models, sample_tumor_image
    ):
        """Segmentation should complete within timing requirements.

        NFR: <4 seconds per slice on GPU (allowing more for cold start/CPU).
        """
        processor, model = real_sam_models
        box = [110, 120, 190, 180]

        start_time = time.perf_counter()
        _segment_tumor(sample_tumor_image, box, processor, model)
        elapsed = time.perf_counter() - start_time

        # Allow 15 seconds for first inference (cold start) or CPU
        # Production requirement is <4s on GPU after warm-up
        assert elapsed < 15.0, f"Segmentation took {elapsed:.2f}s, expected <15s"

    def test_segment_tumor_with_real_model_different_sizes(self, real_sam_models):
        """Segmentation should work with different image sizes."""
        processor, model = real_sam_models

        # Test 512x512
        image_512 = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        box_512 = [200, 200, 300, 300]
        mask_512, _ = _segment_tumor(image_512, box_512, processor, model)
        assert mask_512.shape == (512, 512)

        # Test 224x224
        image_224 = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        box_224 = [50, 50, 150, 150]
        mask_224, _ = _segment_tumor(image_224, box_224, processor, model)
        assert mask_224.shape == (224, 224)
