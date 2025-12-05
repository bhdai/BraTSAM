"""Tests for confidence indicator UI component (Story 2.4, AC #3)."""

from pathlib import Path

import pytest

# Import will fail until component is created
from webapp.components.confidence_indicator import (
    CONFIDENCE_COLORS,
    CONFIDENCE_LABELS,
    render_confidence_indicator,
)
from webapp.utils.inference import PipelineResult


class TestConfidenceIndicatorModule:
    """Tests for confidence_indicator module existence and structure."""

    def test_confidence_indicator_module_exists(self):
        """AC #3: confidence_indicator.py should exist in webapp/components."""
        assert Path("webapp/components/confidence_indicator.py").is_file()

    def test_confidence_colors_defined(self):
        """AC #3: CONFIDENCE_COLORS dict should be defined with correct colors."""
        assert isinstance(CONFIDENCE_COLORS, dict)
        assert "auto-approved" in CONFIDENCE_COLORS
        assert "needs-review" in CONFIDENCE_COLORS
        assert "manual-required" in CONFIDENCE_COLORS
        # Verify exact color codes from UX spec
        assert CONFIDENCE_COLORS["auto-approved"] == "#22C55E"  # Green
        assert CONFIDENCE_COLORS["needs-review"] == "#F59E0B"  # Amber
        assert CONFIDENCE_COLORS["manual-required"] == "#EF4444"  # Red

    def test_confidence_labels_defined(self):
        """AC #3: CONFIDENCE_LABELS dict should be defined."""
        assert isinstance(CONFIDENCE_LABELS, dict)
        assert "auto-approved" in CONFIDENCE_LABELS
        assert "needs-review" in CONFIDENCE_LABELS
        assert "manual-required" in CONFIDENCE_LABELS


class TestRenderConfidenceIndicator:
    """Tests for render_confidence_indicator function."""

    def test_render_confidence_indicator_callable(self):
        """render_confidence_indicator should be callable."""
        assert callable(render_confidence_indicator)

    def test_render_confidence_indicator_accepts_pipeline_result(self):
        """render_confidence_indicator should accept PipelineResult."""
        result = PipelineResult(
            success=True,
            stage="segmentation",
            yolo_confidence=0.95,
            sam_iou=0.92,
        )
        # Should not raise an exception
        render_confidence_indicator(result)

    def test_render_confidence_indicator_handles_none_confidence(self):
        """render_confidence_indicator should handle None confidence."""
        result = PipelineResult(
            success=False,
            stage="detection",
            error_message="No tumor detected",
        )
        # Should not raise an exception
        render_confidence_indicator(result)


class TestConfidenceIndicatorColors:
    """Tests for correct color assignment (AC #3)."""

    @pytest.mark.parametrize(
        "yolo_conf,sam_iou,expected_classification",
        [
            (0.95, 0.92, "auto-approved"),  # High confidence
            (0.90, 0.95, "auto-approved"),  # Exactly at threshold
            (0.75, 0.70, "needs-review"),  # Medium confidence
            (0.50, 0.60, "needs-review"),  # Lower bound
            (0.45, 0.40, "manual-required"),  # Low confidence
        ],
    )
    def test_classification_color_mapping(
        self, yolo_conf, sam_iou, expected_classification
    ):
        """Verify correct classification determines color."""
        result = PipelineResult(
            success=True,
            stage="segmentation",
            yolo_confidence=yolo_conf,
            sam_iou=sam_iou,
        )
        assert result.classification == expected_classification
        assert expected_classification in CONFIDENCE_COLORS


class TestConfidenceIndicatorOutput:
    """Tests for HTML/badge output format."""

    def test_indicator_returns_html_string(self):
        """render_confidence_indicator should return HTML string."""
        result = PipelineResult(
            success=True,
            stage="segmentation",
            yolo_confidence=0.95,
            sam_iou=0.92,
        )
        html = render_confidence_indicator(result, return_html=True)
        assert isinstance(html, str)
        assert "background-color" in html

    def test_indicator_includes_confidence_percentage(self):
        """Indicator should show confidence as percentage when available."""
        result = PipelineResult(
            success=True,
            stage="segmentation",
            yolo_confidence=0.87,
            sam_iou=0.85,
        )
        html = render_confidence_indicator(result, return_html=True)
        # Should show 85% (minimum of both)
        assert "85%" in html

    def test_indicator_handles_no_percentage_when_none(self):
        """Indicator should not crash when confidence is None."""
        result = PipelineResult(
            success=False,
            stage="detection",
            error_message="No detection",
        )
        html = render_confidence_indicator(result, return_html=True)
        assert isinstance(html, str)
        # Should not have a percentage or handle gracefully
        assert "Manual Required" in html or "manual-required" in html.lower()
