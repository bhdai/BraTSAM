"""Canvas drawing component for manual bounding box fallback (Story 3.4).

This module provides the drawing canvas for users to manually draw
bounding boxes when YOLO detection fails or needs correction.

Features:
- Rectangle drawing mode with streamlit-drawable-canvas
- Yellow (#EAB308) stroke color for manual boxes
- Coordinate extraction in [x_min, y_min, x_max, y_max] format
- Edit mode state management
- Bbox validation and coordinate conversion

Example:
    >>> from PIL import Image
    >>> from webapp.components.canvas import render_drawing_canvas
    >>> image = Image.new("RGB", (512, 512))
    >>> canvas_result = render_drawing_canvas(image)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import streamlit as st

if TYPE_CHECKING:
    from PIL import Image

# Set up logger
logger = logging.getLogger(__name__)

# Manual box color constant (distinct from YOLO cyan)
MANUAL_BOX_COLOR = "#EAB308"  # Yellow
MANUAL_BOX_STROKE_WIDTH = 2
MANUAL_BOX_FILL_OPACITY = 0.1  # 10% fill


def init_edit_mode_state() -> None:
    """Initialize edit mode state in session state.

    Sets default values for edit mode state if not present:
    - edit_mode: False (not in edit mode)
    - manual_box: None (no manual box drawn)
    - manual_result: None (no manual segmentation result)
    """
    if "edit_mode" not in st.session_state:
        st.session_state.edit_mode = False
    if "manual_box" not in st.session_state:
        st.session_state.manual_box = None
    if "manual_result" not in st.session_state:
        st.session_state.manual_result = None


def toggle_edit_mode() -> None:
    """Toggle edit mode on/off (AC #1, #6).

    When exiting edit mode, clears the manual box state.
    """
    st.session_state.edit_mode = not st.session_state.edit_mode
    if not st.session_state.edit_mode:
        # Clear manual box when exiting edit mode
        st.session_state.manual_box = None
    logger.info(f"Edit mode toggled: {st.session_state.edit_mode}")


def cancel_edit_mode() -> None:
    """Cancel edit mode without triggering segmentation (AC #5).

    Clears edit mode state and restores previous view state.
    """
    st.session_state.edit_mode = False
    st.session_state.manual_box = None
    logger.info("Edit mode cancelled")


def is_edit_mode() -> bool:
    """Check if edit mode is currently active.

    Returns:
        True if edit mode is active, False otherwise.
    """
    return getattr(st.session_state, "edit_mode", False)


def get_manual_box() -> list[int] | None:
    """Get the current manual bounding box.

    Returns:
        Manual box [x_min, y_min, x_max, y_max] or None.
    """
    return getattr(st.session_state, "manual_box", None)


def set_manual_box(bbox: list[int]) -> None:
    """Set the manual bounding box in session state.

    Args:
        bbox: Bounding box [x_min, y_min, x_max, y_max].
    """
    st.session_state.manual_box = bbox
    logger.debug(f"Manual box set: {bbox}")


def extract_bbox_from_canvas(canvas_result) -> list[int] | None:
    """Extract bounding box from canvas result (AC #2).

    Canvas stores rectangles as:
    {
        "type": "rect",
        "left": x_min,
        "top": y_min,
        "width": w,
        "height": h
    }

    Args:
        canvas_result: Result object from streamlit-drawable-canvas.

    Returns:
        Bounding box in [x_min, y_min, x_max, y_max] format,
        or None if no valid rectangle found.
    """
    if canvas_result is None:
        return None

    if canvas_result.json_data is None:
        return None

    objects = canvas_result.json_data.get("objects", [])
    if not objects:
        return None

    # Get most recent rectangle (last in list)
    rect = objects[-1]
    if rect.get("type") != "rect":
        logger.debug(f"Last object is not a rectangle: {rect.get('type')}")
        return None

    x_min = int(rect["left"])
    y_min = int(rect["top"])
    x_max = int(rect["left"] + rect["width"])
    y_max = int(rect["top"] + rect["height"])

    bbox = [x_min, y_min, x_max, y_max]
    logger.debug(f"Extracted bbox from canvas: {bbox}")
    return bbox


def validate_bbox(
    bbox: list[int],
    image_width: int,
    image_height: int,
) -> bool:
    """Validate bounding box dimensions (AC #2).

    Args:
        bbox: Bounding box [x_min, y_min, x_max, y_max].
        image_width: Width of the image in pixels.
        image_height: Height of the image in pixels.

    Returns:
        True if bbox has valid positive dimensions, False otherwise.
    """
    x_min, y_min, x_max, y_max = bbox

    # Check for positive dimensions
    if x_max <= x_min or y_max <= y_min:
        logger.warning(f"Invalid bbox dimensions: {bbox}")
        return False

    return True


def clip_bbox_to_bounds(
    bbox: list[int],
    image_width: int,
    image_height: int,
) -> list[int]:
    """Clip bounding box to image bounds (AC #2).

    Args:
        bbox: Bounding box [x_min, y_min, x_max, y_max].
        image_width: Width of the image in pixels.
        image_height: Height of the image in pixels.

    Returns:
        Clipped bounding box within image bounds.
    """
    x_min, y_min, x_max, y_max = bbox

    clipped = [
        max(0, x_min),
        max(0, y_min),
        min(image_width, x_max),
        min(image_height, y_max),
    ]
    logger.debug(f"Clipped bbox {bbox} to {clipped}")
    return clipped


def convert_canvas_to_image_coords(
    canvas_bbox: list[int],
    canvas_size: tuple[int, int],
    image_size: tuple[int, int],
) -> list[int]:
    """Convert canvas coordinates to image coordinates.

    Handles scaling when canvas display size differs from image size.

    Args:
        canvas_bbox: Bounding box in canvas coordinates.
        canvas_size: Canvas (width, height) in pixels.
        image_size: Image (width, height) in pixels.

    Returns:
        Bounding box in image coordinates.
    """
    canvas_w, canvas_h = canvas_size
    image_w, image_h = image_size

    if canvas_w == image_w and canvas_h == image_h:
        return canvas_bbox

    # Calculate scale factors
    scale_x = image_w / canvas_w
    scale_y = image_h / canvas_h

    x_min, y_min, x_max, y_max = canvas_bbox
    image_bbox = [
        int(x_min * scale_x),
        int(y_min * scale_y),
        int(x_max * scale_x),
        int(y_max * scale_y),
    ]

    logger.debug(
        f"Converted canvas coords {canvas_bbox} to image coords {image_bbox} "
        f"(scale: {scale_x:.2f}x{scale_y:.2f})"
    )
    return image_bbox


def render_drawing_canvas(
    background_image: "Image.Image",
    stroke_color: str = MANUAL_BOX_COLOR,
    stroke_width: int = MANUAL_BOX_STROKE_WIDTH,
    key: str = "manual_bbox_canvas",
):
    """Render canvas for manual bounding box drawing (AC #1, #2).

    Args:
        background_image: PIL Image to display as background.
        stroke_color: Color of drawn rectangle (yellow for manual).
        stroke_width: Width of rectangle stroke.
        key: Streamlit component key for canvas state.

    Returns:
        Canvas result dict containing drawn objects, or None if no drawing.
    """
    from streamlit_drawable_canvas import st_canvas

    # Calculate fill color with opacity
    # Parse hex color and add alpha
    r = int(stroke_color[1:3], 16)
    g = int(stroke_color[3:5], 16)
    b = int(stroke_color[5:7], 16)
    fill_color = f"rgba({r}, {g}, {b}, {MANUAL_BOX_FILL_OPACITY})"

    canvas_result = st_canvas(
        fill_color=fill_color,
        stroke_width=stroke_width,
        stroke_color=stroke_color,
        background_image=background_image,
        update_streamlit=True,
        height=background_image.height,
        width=background_image.width,
        drawing_mode="rect",
        key=key,
    )

    logger.debug(f"Canvas rendered: height={background_image.height}, width={background_image.width}")
    return canvas_result


__all__ = [
    "MANUAL_BOX_COLOR",
    "MANUAL_BOX_STROKE_WIDTH",
    "cancel_edit_mode",
    "clip_bbox_to_bounds",
    "convert_canvas_to_image_coords",
    "extract_bbox_from_canvas",
    "get_manual_box",
    "init_edit_mode_state",
    "is_edit_mode",
    "render_drawing_canvas",
    "set_manual_box",
    "toggle_edit_mode",
    "validate_bbox",
]
