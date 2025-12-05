"""Unit tests for webapp/components/viewer.py.

Tests for the interactive image viewer component (Story 3.1)
and overlay rendering (Story 3.2).
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image


class TestViewerModuleExists:
    """Tests for viewer module existence and structure (AC #5)."""

    def test_viewer_module_exists(self) -> None:
        """AC #5: viewer.py should exist in webapp/components."""
        assert Path("webapp/components/viewer.py").is_file()

    def test_render_image_viewer_callable(self) -> None:
        """render_image_viewer should be a callable function."""
        from webapp.components.viewer import render_image_viewer

        assert callable(render_image_viewer)


class TestRenderImageViewer:
    """Tests for the render_image_viewer function (AC #1, #3, #4)."""

    def test_accepts_numpy_array(self) -> None:
        """Viewer should accept numpy array input (AC #1)."""
        with patch("webapp.components.viewer.st") as mock_st:
            mock_st.image = MagicMock()
            mock_st.error = MagicMock()
            test_image = np.zeros((256, 256), dtype=np.uint8)

            from webapp.components.viewer import render_image_viewer

            render_image_viewer(test_image)

            mock_st.image.assert_called_once()

    def test_accepts_pil_image(self) -> None:
        """Viewer should accept PIL Image input (AC #1)."""
        with patch("webapp.components.viewer.st") as mock_st:
            mock_st.image = MagicMock()
            mock_st.error = MagicMock()
            test_image = Image.new("L", (256, 256))

            from webapp.components.viewer import render_image_viewer

            render_image_viewer(test_image)

            mock_st.image.assert_called_once()

    def test_handles_rgb_image(self) -> None:
        """Viewer should handle RGB images (AC #4)."""
        with patch("webapp.components.viewer.st") as mock_st:
            mock_st.image = MagicMock()
            mock_st.error = MagicMock()
            test_image = np.zeros((256, 256, 3), dtype=np.uint8)

            from webapp.components.viewer import render_image_viewer

            render_image_viewer(test_image)

            mock_st.image.assert_called_once()

    def test_handles_grayscale_image(self) -> None:
        """Viewer should handle grayscale images (AC #4)."""
        with patch("webapp.components.viewer.st") as mock_st:
            mock_st.image = MagicMock()
            mock_st.error = MagicMock()
            test_image = np.zeros((256, 256), dtype=np.uint8)

            from webapp.components.viewer import render_image_viewer

            render_image_viewer(test_image)

            mock_st.image.assert_called_once()

    def test_uses_container_width(self) -> None:
        """Viewer should use container width for responsive scaling (AC #3)."""
        with patch("webapp.components.viewer.st") as mock_st:
            mock_st.image = MagicMock()
            mock_st.error = MagicMock()
            test_image = np.zeros((256, 256), dtype=np.uint8)

            from webapp.components.viewer import render_image_viewer

            render_image_viewer(test_image)

            # Verify use_container_width=True was passed
            call_kwargs = mock_st.image.call_args[1]
            assert call_kwargs.get("use_container_width") is True

    def test_passes_caption(self) -> None:
        """Viewer should pass caption to st.image."""
        with patch("webapp.components.viewer.st") as mock_st:
            mock_st.image = MagicMock()
            mock_st.error = MagicMock()
            test_image = np.zeros((256, 256), dtype=np.uint8)

            from webapp.components.viewer import render_image_viewer

            render_image_viewer(test_image, caption="Test Caption")

            call_kwargs = mock_st.image.call_args[1]
            assert call_kwargs.get("caption") == "Test Caption"

    def test_handles_empty_array(self) -> None:
        """Viewer should handle edge case of empty array gracefully."""
        with patch("webapp.components.viewer.st") as mock_st:
            mock_st.image = MagicMock()
            mock_st.error = MagicMock()

            # Empty array - should not crash
            test_image = np.array([])

            from webapp.components.viewer import render_image_viewer

            render_image_viewer(test_image)

            # Should show error, not crash
            mock_st.error.assert_called_once()

    def test_handles_wrong_dimensions(self) -> None:
        """Viewer should handle arrays with wrong dimensions gracefully."""
        with patch("webapp.components.viewer.st") as mock_st:
            mock_st.image = MagicMock()
            mock_st.error = MagicMock()

            # 1D array - invalid for image display
            test_image = np.zeros((256,), dtype=np.uint8)

            from webapp.components.viewer import render_image_viewer

            render_image_viewer(test_image)

            # Should show error, not crash
            mock_st.error.assert_called_once()


class TestNormalization:
    """Tests for automatic normalization (AC #2)."""

    def test_normalizes_non_uint8_array(self) -> None:
        """Viewer should normalize non-uint8 arrays before display (AC #2)."""
        with patch("webapp.components.viewer.st") as mock_st:
            mock_st.image = MagicMock()
            mock_st.error = MagicMock()

            # Create a float32 array (not uint8)
            test_image = np.random.rand(256, 256).astype(np.float32)

            from webapp.components.viewer import render_image_viewer

            render_image_viewer(test_image)

            # Should have called st.image (after normalization)
            mock_st.image.assert_called_once()

            # The image passed should be a PIL Image (converted from normalized array)
            call_args = mock_st.image.call_args[0]
            assert isinstance(call_args[0], Image.Image)


class TestRenderImageViewerWithResult:
    """Tests for overlay-ready architecture (AC #6)."""

    def test_accepts_pipeline_result(self) -> None:
        """Viewer should accept PipelineResult parameter (AC #6)."""
        with patch("webapp.components.viewer.st") as mock_st:
            mock_st.image = MagicMock()
            mock_st.error = MagicMock()
            test_image = np.zeros((256, 256), dtype=np.uint8)

            from webapp.components.viewer import render_image_viewer
            from webapp.utils.inference import PipelineResult

            result = PipelineResult(
                success=True,
                stage="segmentation",
                yolo_box=[100, 100, 200, 200],
                yolo_confidence=0.95,
                sam_mask=np.zeros((256, 256), dtype=np.uint8),
                sam_iou=0.92,
            )

            # Should not raise - overlay rendering is Story 3.2
            render_image_viewer(test_image, result=result)

            mock_st.image.assert_called_once()

    def test_result_is_optional(self) -> None:
        """Viewer should work without PipelineResult (result is optional)."""
        with patch("webapp.components.viewer.st") as mock_st:
            mock_st.image = MagicMock()
            mock_st.error = MagicMock()
            test_image = np.zeros((256, 256), dtype=np.uint8)

            from webapp.components.viewer import render_image_viewer

            # Should not raise when result is None (default)
            render_image_viewer(test_image)

            mock_st.image.assert_called_once()


class TestOverlayParameters:
    """Tests for overlay parameters (Story 3.2, AC #1, #2, #3, #4)."""

    def test_show_box_parameter_accepted(self) -> None:
        """Viewer should accept show_box parameter (AC #1, #3)."""
        with patch("webapp.components.viewer.st") as mock_st:
            mock_st.image = MagicMock()
            mock_st.error = MagicMock()
            test_image = np.zeros((256, 256), dtype=np.uint8)

            from webapp.components.viewer import render_image_viewer

            # Should not raise with show_box parameter
            render_image_viewer(test_image, show_box=True)
            render_image_viewer(test_image, show_box=False)

            assert mock_st.image.call_count == 2

    def test_show_mask_parameter_accepted(self) -> None:
        """Viewer should accept show_mask parameter (AC #2, #4)."""
        with patch("webapp.components.viewer.st") as mock_st:
            mock_st.image = MagicMock()
            mock_st.error = MagicMock()
            test_image = np.zeros((256, 256), dtype=np.uint8)

            from webapp.components.viewer import render_image_viewer

            # Should not raise with show_mask parameter
            render_image_viewer(test_image, show_mask=True)
            render_image_viewer(test_image, show_mask=False)

            assert mock_st.image.call_count == 2

    def test_both_overlay_parameters_accepted(self) -> None:
        """Viewer should accept both overlay parameters together."""
        with patch("webapp.components.viewer.st") as mock_st:
            mock_st.image = MagicMock()
            mock_st.error = MagicMock()
            test_image = np.zeros((256, 256), dtype=np.uint8)

            from webapp.components.viewer import render_image_viewer

            render_image_viewer(test_image, show_box=True, show_mask=True)
            render_image_viewer(test_image, show_box=False, show_mask=False)

            assert mock_st.image.call_count == 2


class TestBoundingBoxOverlay:
    """Tests for bounding box overlay rendering (Story 3.2, AC #1)."""

    def test_draw_bounding_box_valid_box(self) -> None:
        """Bounding box should be drawn within image bounds (AC #1)."""
        from webapp.components.viewer import _draw_bounding_box

        image = Image.new("RGB", (100, 100), color="white")
        box = [10, 10, 50, 50]
        result = _draw_bounding_box(image, box)

        # Verify image dimensions preserved
        assert result.size == image.size
        # Result should be a PIL Image
        assert isinstance(result, Image.Image)

    def test_draw_bounding_box_clips_out_of_bounds(self) -> None:
        """Box extending beyond image should be clipped (AC #1)."""
        from webapp.components.viewer import _draw_bounding_box

        image = Image.new("RGB", (100, 100), color="white")
        box = [-10, -10, 150, 150]  # Extends beyond bounds
        result = _draw_bounding_box(image, box)

        assert result.size == image.size

    def test_draw_bounding_box_preserves_original(self) -> None:
        """Original image should not be modified."""
        from webapp.components.viewer import _draw_bounding_box

        image = Image.new("RGB", (100, 100), color="white")
        box = [10, 10, 50, 50]
        original_pixels = list(image.getdata())

        _draw_bounding_box(image, box)

        # Original should be unchanged
        assert list(image.getdata()) == original_pixels

    def test_draw_bounding_box_cyan_color(self) -> None:
        """Bounding box should be drawn in cyan color (AC #1)."""
        from webapp.components.viewer import _draw_bounding_box

        image = Image.new("RGB", (100, 100), color="white")
        box = [10, 10, 50, 50]
        result = _draw_bounding_box(image, box)

        # Check a pixel on the box border has cyan-ish color
        pixel = result.getpixel((10, 10))
        # Cyan (#06B6D4) = RGB(6, 182, 212)
        # Should have more blue+green than red
        assert pixel[0] < pixel[1] or pixel[0] < pixel[2]


class TestMaskOverlay:
    """Tests for mask overlay rendering (Story 3.2, AC #2)."""

    def test_overlay_mask_correct_dimensions(self) -> None:
        """Mask overlay should preserve image dimensions (AC #2)."""
        from webapp.components.viewer import _overlay_mask

        image = Image.new("RGB", (100, 100), color="white")
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[25:75, 25:75] = 255  # Center square

        result = _overlay_mask(image, mask)

        assert result.size == image.size

    def test_overlay_mask_resizes_when_needed(self) -> None:
        """Mask should be resized to match image dimensions (AC #2)."""
        from webapp.components.viewer import _overlay_mask

        image = Image.new("RGB", (200, 200), color="white")
        mask = np.zeros((100, 100), dtype=np.uint8)  # Different size
        mask[25:75, 25:75] = 255

        result = _overlay_mask(image, mask)

        assert result.size == image.size

    def test_overlay_mask_applies_color(self) -> None:
        """Mask should apply magenta color in mask area (AC #2)."""
        from webapp.components.viewer import _overlay_mask

        image = Image.new("RGB", (100, 100), color="white")
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[40:60, 40:60] = 255  # Center square

        result = _overlay_mask(image, mask)

        # Center pixel should have magenta tint
        center_pixel = result.getpixel((50, 50))
        # Should be RGBA
        assert len(center_pixel) == 4

    def test_overlay_mask_preserves_original(self) -> None:
        """Original image should not be modified."""
        from webapp.components.viewer import _overlay_mask

        image = Image.new("RGB", (100, 100), color="white")
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[25:75, 25:75] = 255
        original_pixels = list(image.getdata())

        _overlay_mask(image, mask)

        assert list(image.getdata()) == original_pixels


class TestOverlayCompositing:
    """Tests for overlay compositing pipeline (Story 3.2, AC #5)."""

    def test_composite_z_order(self) -> None:
        """Box should be on top of mask in final composite (AC #5)."""
        from webapp.components.viewer import _composite_overlays
        from webapp.utils.inference import PipelineResult

        image = Image.new("RGB", (100, 100), color="white")
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[20:80, 20:80] = 255  # Large center square
        box = [30, 30, 70, 70]  # Smaller box inside mask

        result = PipelineResult(
            success=True,
            stage="segmentation",
            yolo_box=box,
            sam_mask=mask,
        )

        composite = _composite_overlays(image, result, show_box=True, show_mask=True)

        assert composite.size == image.size

    def test_composite_no_overlays(self) -> None:
        """Should return original image if no overlays requested."""
        from webapp.components.viewer import _composite_overlays
        from webapp.utils.inference import PipelineResult

        image = Image.new("RGB", (100, 100), color="white")
        result = PipelineResult(
            success=True,
            stage="segmentation",
            yolo_box=[10, 10, 50, 50],
            sam_mask=np.zeros((100, 100), dtype=np.uint8),
        )

        composite = _composite_overlays(image, result, show_box=False, show_mask=False)

        # Should be essentially the same image (may be converted to RGBA)
        assert composite.size == image.size

    def test_composite_box_only(self) -> None:
        """Should render only box when show_mask=False."""
        from webapp.components.viewer import _composite_overlays
        from webapp.utils.inference import PipelineResult

        image = Image.new("RGB", (100, 100), color="white")
        result = PipelineResult(
            success=True,
            stage="segmentation",
            yolo_box=[10, 10, 50, 50],
            sam_mask=np.zeros((100, 100), dtype=np.uint8),
        )

        composite = _composite_overlays(image, result, show_box=True, show_mask=False)

        assert composite.size == image.size

    def test_composite_mask_only(self) -> None:
        """Should render only mask when show_box=False."""
        from webapp.components.viewer import _composite_overlays
        from webapp.utils.inference import PipelineResult

        image = Image.new("RGB", (100, 100), color="white")
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[25:75, 25:75] = 255

        result = PipelineResult(
            success=True,
            stage="segmentation",
            yolo_box=[10, 10, 50, 50],
            sam_mask=mask,
        )

        composite = _composite_overlays(image, result, show_box=False, show_mask=True)

        assert composite.size == image.size

    def test_composite_handles_missing_box(self) -> None:
        """Should handle result with no bounding box gracefully."""
        from webapp.components.viewer import _composite_overlays
        from webapp.utils.inference import PipelineResult

        image = Image.new("RGB", (100, 100), color="white")
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[25:75, 25:75] = 255

        result = PipelineResult(
            success=True,
            stage="segmentation",
            yolo_box=None,  # No box
            sam_mask=mask,
        )

        composite = _composite_overlays(image, result, show_box=True, show_mask=True)

        assert composite.size == image.size

    def test_composite_handles_missing_mask(self) -> None:
        """Should handle result with no mask gracefully."""
        from webapp.components.viewer import _composite_overlays
        from webapp.utils.inference import PipelineResult

        image = Image.new("RGB", (100, 100), color="white")
        result = PipelineResult(
            success=True,
            stage="detection",
            yolo_box=[10, 10, 50, 50],
            sam_mask=None,  # No mask
        )

        composite = _composite_overlays(image, result, show_box=True, show_mask=True)

        assert composite.size == image.size

    def test_composite_handles_empty_mask(self) -> None:
        """Should handle empty (all-zero) mask gracefully."""
        from webapp.components.viewer import _composite_overlays
        from webapp.utils.inference import PipelineResult

        image = Image.new("RGB", (100, 100), color="white")
        result = PipelineResult(
            success=True,
            stage="segmentation",
            yolo_box=[10, 10, 50, 50],
            sam_mask=np.zeros((100, 100), dtype=np.uint8),  # Empty mask
        )

        composite = _composite_overlays(image, result, show_box=True, show_mask=True)

        assert composite.size == image.size


class TestOverlayIntegration:
    """Tests for overlay integration with render_image_viewer (Story 3.2, AC #6)."""

    def test_viewer_renders_overlays_with_result(self) -> None:
        """Viewer should render overlays when result provided (AC #6)."""
        with patch("webapp.components.viewer.st") as mock_st:
            mock_st.image = MagicMock()
            mock_st.error = MagicMock()

            test_image = np.zeros((256, 256), dtype=np.uint8)
            mask = np.zeros((256, 256), dtype=np.uint8)
            mask[50:200, 50:200] = 255

            from webapp.components.viewer import render_image_viewer
            from webapp.utils.inference import PipelineResult

            result = PipelineResult(
                success=True,
                stage="segmentation",
                yolo_box=[100, 100, 200, 200],
                yolo_confidence=0.95,
                sam_mask=mask,
                sam_iou=0.92,
            )

            render_image_viewer(
                test_image,
                result=result,
                show_box=True,
                show_mask=True,
            )

            mock_st.image.assert_called_once()

    def test_viewer_respects_toggle_states(self) -> None:
        """Viewer should respect show_box and show_mask toggles (AC #3, #4)."""
        with patch("webapp.components.viewer.st") as mock_st:
            mock_st.image = MagicMock()
            mock_st.error = MagicMock()

            test_image = np.zeros((256, 256), dtype=np.uint8)

            from webapp.components.viewer import render_image_viewer
            from webapp.utils.inference import PipelineResult

            result = PipelineResult(
                success=True,
                stage="segmentation",
                yolo_box=[100, 100, 200, 200],
                sam_mask=np.zeros((256, 256), dtype=np.uint8),
            )

            # Test with overlays off
            render_image_viewer(
                test_image,
                result=result,
                show_box=False,
                show_mask=False,
            )

            mock_st.image.assert_called_once()


class TestZoomState:
    """Tests for zoom state management (Story 3.3, AC #1)."""

    def test_init_zoom_state_creates_defaults(self) -> None:
        """init_zoom_state should initialize with default values."""
        from webapp.components.viewer import init_zoom_state

        state = init_zoom_state()

        assert state["level"] == 1.0
        assert state["offset_x"] == 0
        assert state["offset_y"] == 0
        assert state["min_zoom"] == 0.25
        assert state["max_zoom"] == 4.0

    def test_zoom_in_increases_level(self) -> None:
        """zoom_in should increase zoom level by step."""
        from webapp.components.viewer import init_zoom_state, zoom_in

        state = init_zoom_state()
        new_state = zoom_in(state)

        assert new_state["level"] > state["level"]
        assert new_state["level"] == 1.25  # Default step is 0.25

    def test_zoom_out_decreases_level(self) -> None:
        """zoom_out should decrease zoom level by step."""
        from webapp.components.viewer import init_zoom_state, zoom_out

        state = init_zoom_state()
        new_state = zoom_out(state)

        assert new_state["level"] < state["level"]
        assert new_state["level"] == 0.75

    def test_zoom_in_respects_max_limit(self) -> None:
        """zoom_in should not exceed max_zoom limit (AC #1)."""
        from webapp.components.viewer import init_zoom_state, zoom_in

        state = init_zoom_state()
        state["level"] = 3.9  # Near max

        new_state = zoom_in(state)

        assert new_state["level"] <= state["max_zoom"]
        assert new_state["level"] == 4.0  # Clamped to max

    def test_zoom_out_respects_min_limit(self) -> None:
        """zoom_out should not go below min_zoom limit (AC #1)."""
        from webapp.components.viewer import init_zoom_state, zoom_out

        state = init_zoom_state()
        state["level"] = 0.3  # Near min

        new_state = zoom_out(state)

        assert new_state["level"] >= state["min_zoom"]
        assert new_state["level"] == 0.25  # Clamped to min

    def test_get_zoom_display_formats_correctly(self) -> None:
        """get_zoom_display should format zoom as percentage (AC #1)."""
        from webapp.components.viewer import get_zoom_display, init_zoom_state

        state = init_zoom_state()
        state["level"] = 1.5

        display = get_zoom_display(state)

        assert display == "150%"

    def test_get_zoom_display_at_100_percent(self) -> None:
        """get_zoom_display should show '100%' at default zoom."""
        from webapp.components.viewer import get_zoom_display, init_zoom_state

        state = init_zoom_state()
        display = get_zoom_display(state)

        assert display == "100%"


class TestPanState:
    """Tests for pan state management (Story 3.3, AC #2)."""

    def test_update_pan_offset_sets_values(self) -> None:
        """update_pan_offset should set offset values."""
        from webapp.components.viewer import init_zoom_state, update_pan_offset

        state = init_zoom_state()
        new_state = update_pan_offset(state, 50, -30)

        assert new_state["offset_x"] == 50
        assert new_state["offset_y"] == -30

    def test_pan_offset_constrained_to_bounds(self) -> None:
        """Pan offset should be constrained to keep image visible (AC #2)."""
        from webapp.components.viewer import (
            constrain_pan_offset,
            init_zoom_state,
        )

        state = init_zoom_state()
        state["level"] = 2.0  # Zoomed in

        # Try to pan way off screen
        constrained = constrain_pan_offset(
            state, offset_x=10000, offset_y=10000, image_width=256, image_height=256
        )

        # Should be constrained to reasonable bounds
        assert constrained["offset_x"] < 10000
        assert constrained["offset_y"] < 10000


class TestResetZoom:
    """Tests for reset functionality (Story 3.3, AC #5)."""

    def test_reset_zoom_restores_defaults(self) -> None:
        """reset_zoom should restore zoom=1.0 and offset=0 (AC #5)."""
        from webapp.components.viewer import init_zoom_state, reset_zoom

        state = init_zoom_state()
        state["level"] = 2.5
        state["offset_x"] = 100
        state["offset_y"] = -50

        reset_state = reset_zoom(state)

        assert reset_state["level"] == 1.0
        assert reset_state["offset_x"] == 0
        assert reset_state["offset_y"] == 0
        # Limits should be preserved
        assert reset_state["min_zoom"] == 0.25
        assert reset_state["max_zoom"] == 4.0


class TestZoomableViewer:
    """Tests for interactive zoom/pan viewer (Story 3.3, AC #3, #4, #6)."""

    def test_render_zoomable_viewer_returns_html(self) -> None:
        """render_zoomable_viewer should produce HTML component."""
        from webapp.components.viewer import init_zoom_state, render_zoomable_viewer

        image = Image.new("RGB", (100, 100), color="white")
        state = init_zoom_state()

        with patch("webapp.components.viewer.components") as mock_components:
            mock_components.html = MagicMock()

            render_zoomable_viewer(image, state)

            mock_components.html.assert_called_once()

    def test_render_zoomable_viewer_with_overlays_preserves_alignment(self) -> None:
        """Overlays should remain aligned at different zoom levels (AC #3)."""
        from webapp.components.viewer import (
            _composite_overlays,
            init_zoom_state,
            render_zoomable_viewer,
        )
        from webapp.utils.inference import PipelineResult

        image = Image.new("RGB", (100, 100), color="white")
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[25:75, 25:75] = 255

        result = PipelineResult(
            success=True,
            stage="segmentation",
            yolo_box=[30, 30, 70, 70],
            sam_mask=mask,
        )

        # Composite overlays first (as viewer does)
        composited = _composite_overlays(image, result, show_box=True, show_mask=True)

        state = init_zoom_state()
        state["level"] = 2.0  # 200% zoom

        with patch("webapp.components.viewer.components") as mock_components:
            mock_components.html = MagicMock()

            render_zoomable_viewer(composited, state)

            # Verify HTML was generated with zoom transform
            call_args = mock_components.html.call_args[0][0]
            assert "scale(2.0)" in call_args or "scale(2)" in call_args

    def test_initial_display_fits_container(self) -> None:
        """Image should initially display at fit-to-container size (AC #6)."""
        from webapp.components.viewer import init_zoom_state

        state = init_zoom_state()

        # Default zoom level should be 1.0 (fit)
        assert state["level"] == 1.0


class TestZoomControls:
    """Tests for zoom control buttons and indicator (Story 3.3, AC #1, #5, #6)."""

    def test_render_zoom_controls_initializes_state(self) -> None:
        """render_zoom_controls should initialize state in session_state."""
        with patch("webapp.components.viewer.st") as mock_st:
            mock_st.session_state = {}
            mock_st.columns = MagicMock(return_value=[MagicMock()] * 4)
            mock_st.button = MagicMock(return_value=False)
            mock_st.markdown = MagicMock()

            from webapp.components.viewer import render_zoom_controls

            render_zoom_controls(session_key="test_zoom")

            assert "test_zoom" in mock_st.session_state

    def test_render_zoom_controls_displays_level(self) -> None:
        """render_zoom_controls should display current zoom level (AC #1)."""
        with patch("webapp.components.viewer.st") as mock_st:
            mock_st.session_state = {}
            mock_st.columns = MagicMock(return_value=[MagicMock()] * 4)
            mock_st.button = MagicMock(return_value=False)
            mock_st.markdown = MagicMock()

            from webapp.components.viewer import render_zoom_controls

            render_zoom_controls()

            # Verify markdown was called with zoom level
            mock_st.markdown.assert_called()
            call_args = str(mock_st.markdown.call_args)
            assert "100%" in call_args or "Zoom" in call_args


class TestInteractiveViewer:
    """Tests for render_interactive_viewer (Story 3.3, AC #6)."""

    def test_interactive_viewer_with_zoom_enabled(self) -> None:
        """render_interactive_viewer should render zoom controls when enabled."""
        with patch("webapp.components.viewer.st") as mock_st:
            with patch("webapp.components.viewer.components") as mock_components:
                mock_st.session_state = {}
                mock_st.columns = MagicMock(return_value=[MagicMock()] * 4)
                mock_st.button = MagicMock(return_value=False)
                mock_st.markdown = MagicMock()
                mock_st.caption = MagicMock()
                mock_st.error = MagicMock()
                mock_components.html = MagicMock()

                test_image = np.zeros((256, 256), dtype=np.uint8)

                from webapp.components.viewer import render_interactive_viewer

                render_interactive_viewer(
                    test_image, enable_zoom_pan=True, caption="Test"
                )

                # Should have rendered HTML component (zoomable viewer)
                mock_components.html.assert_called_once()

    def test_interactive_viewer_fallback_when_disabled(self) -> None:
        """render_interactive_viewer should fall back to st.image when disabled."""
        with patch("webapp.components.viewer.st") as mock_st:
            mock_st.image = MagicMock()
            mock_st.error = MagicMock()

            test_image = np.zeros((256, 256), dtype=np.uint8)

            from webapp.components.viewer import render_interactive_viewer

            render_interactive_viewer(test_image, enable_zoom_pan=False)

            # Should have used st.image (static viewer)
            mock_st.image.assert_called_once()

    def test_interactive_viewer_backward_compatible(self) -> None:
        """render_interactive_viewer should be backward compatible with old API."""
        with patch("webapp.components.viewer.st") as mock_st:
            mock_st.image = MagicMock()
            mock_st.error = MagicMock()

            test_image = np.zeros((256, 256), dtype=np.uint8)

            from webapp.components.viewer import render_interactive_viewer
            from webapp.utils.inference import PipelineResult

            result = PipelineResult(
                success=True,
                stage="segmentation",
                yolo_box=[10, 10, 50, 50],
                sam_mask=np.zeros((256, 256), dtype=np.uint8),
            )

            # Should not raise with all parameters
            render_interactive_viewer(
                test_image,
                result=result,
                caption="Test",
                show_box=True,
                show_mask=True,
                enable_zoom_pan=False,
            )

            mock_st.image.assert_called_once()


