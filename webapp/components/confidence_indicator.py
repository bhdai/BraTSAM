"""Confidence indicator UI component for triage visualization (Story 2.4, AC #3).

This module provides visual indicators for confidence-based triage classification.
The indicators use color-coded badges to communicate result quality at a glance:

- Green: Auto-approved (high confidence >= 90%)
- Amber: Needs review (medium confidence 50-89%)
- Red: Manual required (low/no confidence < 50%)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from webapp.utils.inference import PipelineResult

# Color scheme from UX Design Specification (Tailwind colors)
CONFIDENCE_COLORS: dict[str, str] = {
    "auto-approved": "#22C55E",  # Tailwind green-500
    "needs-review": "#F59E0B",  # Tailwind amber-500
    "manual-required": "#EF4444",  # Tailwind red-500
}

# Labels for each classification tier
CONFIDENCE_LABELS: dict[str, str] = {
    "auto-approved": "✅ Auto-Approved",
    "needs-review": "⚠️ Needs Review",
    "manual-required": "🔴 Manual Required",
}

# Tooltip text explaining each classification
CONFIDENCE_TOOLTIPS: dict[str, str] = {
    "auto-approved": "High confidence (≥90%). Minimal review needed.",
    "needs-review": "Medium confidence (50-89%). Verification recommended.",
    "manual-required": "Low/no confidence (<50%). Manual intervention required.",
}


def render_confidence_indicator(
    result: "PipelineResult",
    *,
    return_html: bool = False,
    show_tooltip: bool = True,
) -> str | None:
    """Render a colored confidence badge for a pipeline result.

    Creates a styled HTML badge showing the triage classification with
    optional confidence percentage and tooltip explanation.

    Args:
        result: PipelineResult with confidence/classification data.
        return_html: If True, return HTML string instead of rendering.
            Useful for testing or custom rendering.
        show_tooltip: If True, include tooltip with threshold explanation.

    Returns:
        HTML string if return_html=True, otherwise None (renders via st.markdown).

    Example:
        >>> result = PipelineResult(success=True, stage="segmentation",
        ...                         yolo_confidence=0.95, sam_iou=0.92)
        >>> render_confidence_indicator(result)  # Renders green badge
    """
    classification = result.classification
    color = CONFIDENCE_COLORS[classification]
    label = CONFIDENCE_LABELS[classification]
    tooltip = CONFIDENCE_TOOLTIPS[classification] if show_tooltip else ""
    conf = result.confidence

    # Build the badge content
    if conf is not None:
        badge_text = f"{label} ({conf:.0%})"
    else:
        badge_text = label

    # Build the HTML with inline styles
    tooltip_attr = f'title="{tooltip}"' if show_tooltip and tooltip else ""
    html = (
        f'<span {tooltip_attr} style="'
        f"background-color:{color}; "
        f"color:white; "
        f"padding:4px 8px; "
        f"border-radius:4px; "
        f"font-weight:bold; "
        f"display:inline-block; "
        f"cursor:help;"
        f'">{badge_text}</span>'
    )

    if return_html:
        return html

    # Render in Streamlit context
    try:
        import streamlit as st

        st.markdown(html, unsafe_allow_html=True)
    except ImportError:
        # Non-Streamlit context - just return the HTML
        return html

    return None


def render_confidence_score(
    result: "PipelineResult",
    *,
    return_html: bool = False,
) -> str | None:
    """Render just the confidence score value (e.g., "0.87").

    Useful for displaying the numeric score alongside the badge indicator.

    Args:
        result: PipelineResult with confidence data.
        return_html: If True, return HTML string instead of rendering.

    Returns:
        HTML string if return_html=True, otherwise None (renders via st.markdown).
    """
    conf = result.confidence

    if conf is not None:
        score_text = f"Confidence: {conf:.2f}"
        html = f'<span style="font-size: 0.9em; color: #666;">{score_text}</span>'
    else:
        html = '<span style="font-size: 0.9em; color: #999;">No confidence score</span>'

    if return_html:
        return html

    try:
        import streamlit as st

        st.markdown(html, unsafe_allow_html=True)
    except ImportError:
        return html

    return None


def get_indicator_data(result: "PipelineResult") -> dict[str, str | float | None]:
    """Get confidence indicator data as a dictionary.

    Useful for programmatic access to indicator properties without
    rendering HTML.

    Args:
        result: PipelineResult with confidence data.

    Returns:
        Dictionary with classification, color, label, confidence fields.
    """
    classification = result.classification
    return {
        "classification": classification,
        "color": CONFIDENCE_COLORS[classification],
        "label": CONFIDENCE_LABELS[classification],
        "confidence": result.confidence,
        "tooltip": CONFIDENCE_TOOLTIPS[classification],
    }
