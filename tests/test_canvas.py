"""Tests for manual bounding box canvas component (Story 3.4).

Tests cover:
- Edit mode state management
- Canvas coordinate extraction and validation
- Segmentation-only pipeline
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


class MockSessionState(dict):
    """Mock Streamlit session_state that supports both dict and attribute access."""

    def __getattr__(self, key: str):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"No attribute '{key}'")

    def __setattr__(self, key: str, value) -> None:
        self[key] = value

    def __delattr__(self, key: str) -> None:
        try:
            del self[key]
        except KeyError:
            raise AttributeError(f"No attribute '{key}'")


class TestEditModeState:
    """Tests for edit mode state management (AC #1, #5)."""

    def test_toggle_edit_mode_enables(self) -> None:
        """Toggle should enable edit mode when disabled."""
        from webapp.components.canvas import toggle_edit_mode

        # Simulate session state with attribute access support
        mock_session_state = MockSessionState(edit_mode=False, manual_box=None)

        with patch("webapp.components.canvas.st.session_state", mock_session_state):
            toggle_edit_mode()
            assert mock_session_state["edit_mode"] is True

    def test_toggle_edit_mode_disables(self) -> None:
        """Toggle should disable edit mode when enabled."""
        from webapp.components.canvas import toggle_edit_mode

        mock_session_state = MockSessionState(edit_mode=True, manual_box=[10, 20, 30, 40])

        with patch("webapp.components.canvas.st.session_state", mock_session_state):
            toggle_edit_mode()
            assert mock_session_state["edit_mode"] is False
            # Manual box should be cleared when exiting edit mode
            assert mock_session_state["manual_box"] is None

    def test_cancel_edit_mode_clears_state(self) -> None:
        """Cancel should clear edit mode and manual box."""
        from webapp.components.canvas import cancel_edit_mode

        mock_session_state = MockSessionState(
            edit_mode=True,
            manual_box=[10, 20, 30, 40],
            manual_result=MagicMock(),
        )

        with patch("webapp.components.canvas.st.session_state", mock_session_state):
            cancel_edit_mode()
            assert mock_session_state["edit_mode"] is False
            assert mock_session_state["manual_box"] is None

    def test_init_edit_mode_state_defaults(self) -> None:
        """Init should set default edit mode state values."""
        from webapp.components.canvas import init_edit_mode_state

        mock_session_state = MockSessionState()

        with patch("webapp.components.canvas.st.session_state", mock_session_state):
            init_edit_mode_state()
            assert mock_session_state["edit_mode"] is False
            assert mock_session_state["manual_box"] is None
            assert mock_session_state["manual_result"] is None


class TestCoordinateExtraction:
    """Tests for canvas coordinate extraction (AC #2)."""

    def test_extract_bbox_returns_none_for_empty_canvas(self) -> None:
        """Empty canvas should return None."""
        from webapp.components.canvas import extract_bbox_from_canvas

        # Mock empty canvas result
        canvas_result = MagicMock()
        canvas_result.json_data = None

        result = extract_bbox_from_canvas(canvas_result)
        assert result is None

    def test_extract_bbox_returns_none_for_no_objects(self) -> None:
        """Canvas with no objects should return None."""
        from webapp.components.canvas import extract_bbox_from_canvas

        canvas_result = MagicMock()
        canvas_result.json_data = {"objects": []}

        result = extract_bbox_from_canvas(canvas_result)
        assert result is None

    def test_extract_bbox_formats_correctly(self) -> None:
        """Extracted bbox should be [x_min, y_min, x_max, y_max]."""
        from webapp.components.canvas import extract_bbox_from_canvas

        canvas_result = MagicMock()
        canvas_result.json_data = {
            "objects": [
                {
                    "type": "rect",
                    "left": 100,
                    "top": 150,
                    "width": 200,
                    "height": 250,
                }
            ]
        }

        result = extract_bbox_from_canvas(canvas_result)
        assert result == [100, 150, 300, 400]  # x_max = 100+200, y_max = 150+250

    def test_extract_bbox_handles_multiple_rects(self) -> None:
        """Should use most recent (last) rectangle."""
        from webapp.components.canvas import extract_bbox_from_canvas

        canvas_result = MagicMock()
        canvas_result.json_data = {
            "objects": [
                {
                    "type": "rect",
                    "left": 10,
                    "top": 20,
                    "width": 30,
                    "height": 40,
                },
                {
                    "type": "rect",
                    "left": 100,
                    "top": 150,
                    "width": 200,
                    "height": 250,
                },
            ]
        }

        result = extract_bbox_from_canvas(canvas_result)
        # Should use the last rectangle
        assert result == [100, 150, 300, 400]

    def test_extract_bbox_ignores_non_rect_objects(self) -> None:
        """Should return None if last object is not a rectangle."""
        from webapp.components.canvas import extract_bbox_from_canvas

        canvas_result = MagicMock()
        canvas_result.json_data = {
            "objects": [
                {
                    "type": "circle",
                    "left": 100,
                    "top": 150,
                    "radius": 50,
                }
            ]
        }

        result = extract_bbox_from_canvas(canvas_result)
        assert result is None


class TestBboxValidation:
    """Tests for bounding box validation (AC #2)."""

    def test_validate_bbox_within_bounds(self) -> None:
        """Valid bbox within image bounds should pass."""
        from webapp.components.canvas import validate_bbox

        bbox = [100, 150, 300, 400]
        image_width, image_height = 512, 512

        result = validate_bbox(bbox, image_width, image_height)
        assert result is True

    def test_validate_bbox_clips_to_bounds(self) -> None:
        """Bbox exceeding image bounds should be clipped."""
        from webapp.components.canvas import clip_bbox_to_bounds

        bbox = [-10, -20, 600, 700]
        image_width, image_height = 512, 512

        clipped = clip_bbox_to_bounds(bbox, image_width, image_height)
        assert clipped == [0, 0, 512, 512]

    def test_validate_bbox_invalid_dimensions(self) -> None:
        """Bbox with zero or negative dimensions should be invalid."""
        from webapp.components.canvas import validate_bbox

        # Zero width
        bbox1 = [100, 100, 100, 200]  # x_min == x_max
        # Negative dimensions (inverted)
        bbox2 = [200, 200, 100, 100]  # max < min

        assert validate_bbox(bbox1, 512, 512) is False
        assert validate_bbox(bbox2, 512, 512) is False


class TestCanvasCoordinateConversion:
    """Tests for canvas to image coordinate conversion."""

    def test_convert_canvas_to_image_coords_no_scaling(self) -> None:
        """When canvas and image are same size, coords should be unchanged."""
        from webapp.components.canvas import convert_canvas_to_image_coords

        canvas_bbox = [100, 150, 300, 400]
        canvas_size = (512, 512)
        image_size = (512, 512)

        result = convert_canvas_to_image_coords(canvas_bbox, canvas_size, image_size)
        assert result == [100, 150, 300, 400]

    def test_convert_canvas_to_image_coords_with_scaling(self) -> None:
        """Canvas coords should be scaled to image size."""
        from webapp.components.canvas import convert_canvas_to_image_coords

        # Canvas is 256x256, image is 512x512 (2x scaling)
        canvas_bbox = [50, 75, 150, 200]
        canvas_size = (256, 256)
        image_size = (512, 512)

        result = convert_canvas_to_image_coords(canvas_bbox, canvas_size, image_size)
        # Each coord should be doubled
        assert result == [100, 150, 300, 400]


class TestManualBoxColor:
    """Tests for manual box color constant (AC #2)."""

    def test_manual_box_color_is_yellow(self) -> None:
        """Manual box should use yellow color."""
        from webapp.components.canvas import MANUAL_BOX_COLOR

        assert MANUAL_BOX_COLOR == "#EAB308"


class TestEditModeUIHelpers:
    """Tests for edit mode UI helper functions (AC #1, #6)."""

    def test_is_edit_mode_returns_false_by_default(self) -> None:
        """is_edit_mode should return False when not initialized."""
        from webapp.components.canvas import is_edit_mode

        mock_session_state = MockSessionState()

        with patch("webapp.components.canvas.st.session_state", mock_session_state):
            assert is_edit_mode() is False

    def test_is_edit_mode_returns_current_state(self) -> None:
        """is_edit_mode should return current edit mode state."""
        from webapp.components.canvas import is_edit_mode

        mock_session_state = MockSessionState(edit_mode=True)

        with patch("webapp.components.canvas.st.session_state", mock_session_state):
            assert is_edit_mode() is True

    def test_get_manual_box_returns_none_by_default(self) -> None:
        """get_manual_box should return None when not set."""
        from webapp.components.canvas import get_manual_box

        mock_session_state = MockSessionState()

        with patch("webapp.components.canvas.st.session_state", mock_session_state):
            assert get_manual_box() is None

    def test_get_manual_box_returns_stored_box(self) -> None:
        """get_manual_box should return stored bounding box."""
        from webapp.components.canvas import get_manual_box

        bbox = [100, 150, 300, 400]
        mock_session_state = MockSessionState(manual_box=bbox)

        with patch("webapp.components.canvas.st.session_state", mock_session_state):
            assert get_manual_box() == bbox

    def test_set_manual_box_stores_box(self) -> None:
        """set_manual_box should store bounding box in session state."""
        from webapp.components.canvas import set_manual_box

        bbox = [100, 150, 300, 400]
        mock_session_state = MockSessionState()

        with patch("webapp.components.canvas.st.session_state", mock_session_state):
            set_manual_box(bbox)
            assert mock_session_state["manual_box"] == bbox


class TestSegmentationOnlyPipeline:
    """Tests for manual segmentation pipeline (AC #3, #4)."""

    def test_run_segmentation_only_returns_result(self) -> None:
        """Manual box should produce valid PipelineResult."""
        from webapp.utils.inference import PipelineResult, run_segmentation_only

        # Create a test image
        image = np.random.rand(256, 256, 3).astype(np.float32) * 255
        image = image.astype(np.uint8)
        box = [50, 50, 150, 150]

        # Mock the SAM segmentation
        mock_mask = np.zeros((256, 256), dtype=np.uint8)
        mock_mask[50:150, 50:150] = 1
        mock_iou = 0.85

        with patch(
            "webapp.utils.inference._segment_tumor",
            return_value=(mock_mask, mock_iou),
        ):
            with patch("webapp.utils.inference.load_models") as mock_load:
                # Mock the model returns
                mock_load.return_value = (MagicMock(), MagicMock(), MagicMock())
                result = run_segmentation_only(image, box)

        assert isinstance(result, PipelineResult)
        assert result.success is True
        assert result.stage == "segmentation"
        assert result.yolo_box == box

    def test_segmentation_only_no_yolo_confidence(self) -> None:
        """Result should have None for yolo_confidence."""
        from webapp.utils.inference import PipelineResult, run_segmentation_only

        image = np.zeros((256, 256, 3), dtype=np.uint8)
        box = [50, 50, 150, 150]

        mock_mask = np.zeros((256, 256), dtype=np.uint8)
        mock_iou = 0.85

        with patch(
            "webapp.utils.inference._segment_tumor",
            return_value=(mock_mask, mock_iou),
        ):
            with patch("webapp.utils.inference.load_models") as mock_load:
                mock_load.return_value = (MagicMock(), MagicMock(), MagicMock())
                result = run_segmentation_only(image, box)

        # Manual box should have None YOLO confidence
        assert result.yolo_confidence is None
        # SAM IoU should be present
        assert result.sam_iou == mock_iou

    def test_segmentation_only_handles_error(self) -> None:
        """Should return error result on failure."""
        from webapp.utils.inference import run_segmentation_only

        image = np.zeros((256, 256, 3), dtype=np.uint8)
        box = [50, 50, 150, 150]

        with patch(
            "webapp.utils.inference._segment_tumor",
            side_effect=Exception("SAM model error"),
        ):
            with patch("webapp.utils.inference.load_models") as mock_load:
                mock_load.return_value = (MagicMock(), MagicMock(), MagicMock())
                result = run_segmentation_only(image, box)

        assert result.success is False
        assert result.stage == "segmentation"
        assert "SAM model error" in result.error_message

    def test_segmentation_only_confidence_is_sam_iou_only(self) -> None:
        """Combined confidence should be SAM IoU only (no YOLO)."""
        from webapp.utils.inference import run_segmentation_only

        image = np.zeros((256, 256, 3), dtype=np.uint8)
        box = [50, 50, 150, 150]

        mock_mask = np.zeros((256, 256), dtype=np.uint8)
        mock_iou = 0.75

        with patch(
            "webapp.utils.inference._segment_tumor",
            return_value=(mock_mask, mock_iou),
        ):
            with patch("webapp.utils.inference.load_models") as mock_load:
                mock_load.return_value = (MagicMock(), MagicMock(), MagicMock())
                result = run_segmentation_only(image, box)

        # For manual boxes, confidence should be SAM IoU only
        # since yolo_confidence is None
        assert result.confidence == mock_iou


class TestAppEditModeImports:
    """Tests for app.py edit mode imports (Story 3.4)."""

    def test_app_imports_canvas_functions(self) -> None:
        """app.py should import canvas component functions."""
        from webapp import app
        
        # Verify the imports exist
        assert hasattr(app, "cancel_edit_mode")
        assert hasattr(app, "clip_bbox_to_bounds")
        assert hasattr(app, "extract_bbox_from_canvas")
        assert hasattr(app, "init_edit_mode_state")
        assert hasattr(app, "is_edit_mode")
        assert hasattr(app, "render_drawing_canvas")
        assert hasattr(app, "toggle_edit_mode")
        assert hasattr(app, "validate_bbox")

    def test_app_imports_run_segmentation_only(self) -> None:
        """app.py should import run_segmentation_only."""
        from webapp import app
        
        assert hasattr(app, "run_segmentation_only")

    def test_app_imports_pil_image(self) -> None:
        """app.py should import PIL Image for canvas."""
        from webapp import app
        
        assert hasattr(app, "Image")
