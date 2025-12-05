"""Interactive image viewer component for BraTSAM (Story 3.1, 3.2, 3.3).

This module provides the main image viewer for displaying MRI slices
with support for responsive scaling, overlay rendering, and zoom/pan interaction.

The viewer supports:
- 2D numpy arrays (grayscale and RGB)
- PIL Images
- Automatic normalization for non-uint8 arrays
- Responsive scaling via use_container_width
- Bounding box overlay (cyan, 2px stroke)
- Segmentation mask overlay (magenta, 40% opacity)
- Toggle controls for overlay visibility
- Zoom in/out with level indicator (AC #1)
- Pan with click-and-drag (AC #2)
- Overlay alignment during zoom/pan (AC #3)
- Sharp resolution at high zoom levels (AC #4)
- Reset to fit view (AC #5)
- Keyboard accessible controls (AC #6)

Example:
    >>> import numpy as np
    >>> from webapp.components.viewer import render_image_viewer
    >>> image = np.zeros((256, 256), dtype=np.uint8)
    >>> render_image_viewer(image, caption="MRI Slice")
"""

from __future__ import annotations

import base64
import logging
from io import BytesIO
from typing import TYPE_CHECKING, TypedDict

import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image, ImageDraw

from preprocessing.normalize import normalize_slice

if TYPE_CHECKING:
    from webapp.utils.inference import PipelineResult

# Set up logger
logger = logging.getLogger(__name__)

# Overlay color constants (from docs/architecture.md)
BOUNDING_BOX_COLOR = "#06B6D4"  # Cyan
MASK_COLOR = "#D946EF"  # Magenta
MASK_OPACITY = 0.4  # 40% opacity (alpha = 102)
BOX_STROKE_WIDTH = 2  # pixels

# Zoom/pan constants (Story 3.3)
DEFAULT_ZOOM_LEVEL = 1.0
MIN_ZOOM_LEVEL = 0.25  # 25%
MAX_ZOOM_LEVEL = 4.0  # 400%
ZOOM_STEP = 0.25  # 25% per step


class ZoomState(TypedDict):
    """Type definition for zoom/pan state dictionary."""

    level: float
    offset_x: int
    offset_y: int
    min_zoom: float
    max_zoom: float


def init_zoom_state() -> ZoomState:
    """Initialize zoom state with default values.

    Returns:
        ZoomState dictionary with default zoom settings:
        - level: 1.0 (100% / fit to container)
        - offset_x: 0 (no horizontal pan)
        - offset_y: 0 (no vertical pan)
        - min_zoom: 0.25 (25% minimum)
        - max_zoom: 4.0 (400% maximum)
    """
    return ZoomState(
        level=DEFAULT_ZOOM_LEVEL,
        offset_x=0,
        offset_y=0,
        min_zoom=MIN_ZOOM_LEVEL,
        max_zoom=MAX_ZOOM_LEVEL,
    )


def zoom_in(state: ZoomState, step: float = ZOOM_STEP) -> ZoomState:
    """Increase zoom level by step, respecting max limit.

    Args:
        state: Current zoom state.
        step: Zoom increment (default 0.25 = 25%).

    Returns:
        New ZoomState with increased zoom level clamped to max_zoom.
    """
    new_level = min(state["level"] + step, state["max_zoom"])
    return ZoomState(
        level=new_level,
        offset_x=state["offset_x"],
        offset_y=state["offset_y"],
        min_zoom=state["min_zoom"],
        max_zoom=state["max_zoom"],
    )


def zoom_out(state: ZoomState, step: float = ZOOM_STEP) -> ZoomState:
    """Decrease zoom level by step, respecting min limit.

    Args:
        state: Current zoom state.
        step: Zoom decrement (default 0.25 = 25%).

    Returns:
        New ZoomState with decreased zoom level clamped to min_zoom.
    """
    new_level = max(state["level"] - step, state["min_zoom"])
    return ZoomState(
        level=new_level,
        offset_x=state["offset_x"],
        offset_y=state["offset_y"],
        min_zoom=state["min_zoom"],
        max_zoom=state["max_zoom"],
    )


def reset_zoom(state: ZoomState) -> ZoomState:
    """Reset zoom and pan to default values (AC #5).

    Preserves min/max zoom limits while resetting level and offset.

    Args:
        state: Current zoom state.

    Returns:
        New ZoomState with level=1.0 and offset=0.
    """
    return ZoomState(
        level=DEFAULT_ZOOM_LEVEL,
        offset_x=0,
        offset_y=0,
        min_zoom=state["min_zoom"],
        max_zoom=state["max_zoom"],
    )


def get_zoom_display(state: ZoomState) -> str:
    """Format zoom level as percentage string for display (AC #1).

    Args:
        state: Current zoom state.

    Returns:
        Zoom level formatted as percentage (e.g., "150%").
    """
    return f"{int(state['level'] * 100)}%"


def update_pan_offset(state: ZoomState, offset_x: int, offset_y: int) -> ZoomState:
    """Update pan offset values.

    Args:
        state: Current zoom state.
        offset_x: New horizontal pan offset in pixels.
        offset_y: New vertical pan offset in pixels.

    Returns:
        New ZoomState with updated offsets.
    """
    return ZoomState(
        level=state["level"],
        offset_x=offset_x,
        offset_y=offset_y,
        min_zoom=state["min_zoom"],
        max_zoom=state["max_zoom"],
    )


def constrain_pan_offset(
    state: ZoomState,
    offset_x: int,
    offset_y: int,
    image_width: int,
    image_height: int,
) -> ZoomState:
    """Constrain pan offset to keep image visible in viewport (AC #2).

    At zoom levels > 1.0, allows panning within the extended image area.
    At zoom level 1.0 or below, offset is constrained to 0.

    Args:
        state: Current zoom state.
        offset_x: Requested horizontal pan offset.
        offset_y: Requested vertical pan offset.
        image_width: Width of the image in pixels.
        image_height: Height of the image in pixels.

    Returns:
        New ZoomState with constrained offsets.
    """
    zoom = state["level"]

    if zoom <= 1.0:
        # No panning at 100% or below
        return update_pan_offset(state, 0, 0)

    # Calculate maximum pan range based on zoom level
    # At 2x zoom, can pan up to half the image dimension
    max_offset_x = int((image_width * (zoom - 1)) / 2)
    max_offset_y = int((image_height * (zoom - 1)) / 2)

    constrained_x = max(-max_offset_x, min(offset_x, max_offset_x))
    constrained_y = max(-max_offset_y, min(offset_y, max_offset_y))

    return update_pan_offset(state, constrained_x, constrained_y)


def render_zoomable_viewer(
    image: Image.Image,
    state: ZoomState,
    container_height: int = 500,
) -> None:
    """Render image with CSS-based zoom/pan support (AC #3, #4).

    Uses CSS transforms for smooth, sharp zoom/pan without image
    interpolation artifacts. The image maintains full resolution
    at all zoom levels.

    Args:
        image: PIL Image to display (already composited with overlays).
        state: Current zoom/pan state.
        container_height: Height of the viewer container in pixels.
    """
    # Convert image to base64 for embedding in HTML
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode()

    zoom = state["level"]
    offset_x = state["offset_x"]
    offset_y = state["offset_y"]

    # Generate unique key for this viewer instance
    viewer_key = f"zoomable_viewer_{id(image)}"

    # CSS transform for zoom and pan with crisp rendering
    html = f"""
    <style>
        .zoom-container {{
            overflow: hidden;
            width: 100%;
            height: {container_height}px;
            position: relative;
            background: #1a1a1a;
            border-radius: 4px;
            display: flex;
            align-items: center;
            justify-content: center;
        }}
        .zoom-image {{
            max-width: 100%;
            max-height: 100%;
            transform: scale({zoom}) translate({offset_x}px, {offset_y}px);
            transform-origin: center center;
            cursor: {('grab' if zoom > 1 else 'default')};
            image-rendering: pixelated;
            image-rendering: crisp-edges;
            -webkit-image-rendering: pixelated;
        }}
        .zoom-image:active {{
            cursor: {('grabbing' if zoom > 1 else 'default')};
        }}
    </style>
    <div class="zoom-container" id="{viewer_key}">
        <img src="data:image/png;base64,{img_base64}" 
             class="zoom-image"
             alt="MRI Viewer" />
    </div>
    """

    components.html(html, height=container_height + 20)

    logger.debug(f"Rendered zoomable viewer at {get_zoom_display(state)}")


def _draw_bounding_box(
    image: Image.Image,
    box: list[int],
    color: str = BOUNDING_BOX_COLOR,
    width: int = BOX_STROKE_WIDTH,
) -> Image.Image:
    """Draw bounding box on image.

    Args:
        image: PIL Image to draw on (will be copied, not modified).
        box: Bounding box as [x_min, y_min, x_max, y_max].
        color: Hex color string for box stroke.
        width: Stroke width in pixels.

    Returns:
        New PIL Image with bounding box drawn.
    """
    # Create copy to avoid modifying original
    img_with_box = image.copy()

    # Ensure RGB mode for drawing
    if img_with_box.mode not in ("RGB", "RGBA"):
        img_with_box = img_with_box.convert("RGB")

    draw = ImageDraw.Draw(img_with_box)

    # Clip box to image bounds
    x_min = max(0, box[0])
    y_min = max(0, box[1])
    x_max = min(image.width, box[2])
    y_max = min(image.height, box[3])

    # Draw rectangle (outline only)
    draw.rectangle(
        [(x_min, y_min), (x_max, y_max)],
        outline=color,
        width=width,
    )

    logger.debug(f"Drew bounding box [{x_min}, {y_min}, {x_max}, {y_max}]")
    return img_with_box


def _overlay_mask(
    image: Image.Image,
    mask: np.ndarray,
    color: str = MASK_COLOR,
    opacity: float = MASK_OPACITY,
) -> Image.Image:
    """Overlay binary mask on image with color and transparency.

    Args:
        image: Base PIL Image.
        mask: Binary mask as numpy array (H, W), values 0 or non-zero.
        color: Hex color string for mask fill.
        opacity: Opacity of mask overlay (0.0 to 1.0).

    Returns:
        New PIL Image (RGBA) with mask overlaid.
    """
    # Ensure image is RGBA for compositing
    if image.mode != "RGBA":
        base = image.convert("RGBA")
    else:
        base = image.copy()

    # Parse hex color to RGB
    r = int(color[1:3], 16)
    g = int(color[3:5], 16)
    b = int(color[5:7], 16)
    alpha = int(opacity * 255)

    # Resize mask if needed to match image dimensions
    if mask.shape[0] != base.height or mask.shape[1] != base.width:
        mask_resized = np.array(
            Image.fromarray(mask.astype(np.uint8)).resize(
                (base.width, base.height),
                Image.NEAREST,  # Preserve binary values
            )
        )
        logger.debug(f"Resized mask from {mask.shape} to {base.size}")
    else:
        mask_resized = mask

    # Create colored overlay with alpha channel
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    overlay_array = np.array(overlay)

    # Apply color where mask is non-zero
    mask_bool = mask_resized > 0
    overlay_array[mask_bool, 0] = r
    overlay_array[mask_bool, 1] = g
    overlay_array[mask_bool, 2] = b
    overlay_array[mask_bool, 3] = alpha

    overlay = Image.fromarray(overlay_array)

    # Composite overlay onto base
    result = Image.alpha_composite(base, overlay)
    logger.debug(f"Applied mask overlay with opacity {opacity}")
    return result


def _composite_overlays(
    image: Image.Image,
    result: "PipelineResult",
    show_box: bool,
    show_mask: bool,
) -> Image.Image:
    """Composite overlays onto image in correct z-order.

    Z-order: image → mask → box (box on top).

    Args:
        image: Base PIL Image.
        result: PipelineResult containing yolo_box and sam_mask.
        show_box: Whether to show bounding box overlay.
        show_mask: Whether to show segmentation mask overlay.

    Returns:
        PIL Image with overlays composited.
    """
    # Start with a copy of the image
    composited = image.copy()

    # Convert to RGBA for consistent compositing
    if composited.mode != "RGBA":
        composited = composited.convert("RGBA")

    # Apply mask first (underneath box)
    if show_mask and result.sam_mask is not None:
        if result.sam_mask.size > 0 and np.any(result.sam_mask):
            composited = _overlay_mask(composited, result.sam_mask)
        else:
            logger.debug("Skipping empty mask overlay")

    # Apply box on top
    if show_box and result.yolo_box is not None:
        composited = _draw_bounding_box(composited, result.yolo_box)

    logger.debug(f"Composited overlays: show_box={show_box}, show_mask={show_mask}")
    return composited


def render_image_viewer(
    image: np.ndarray | Image.Image,
    result: PipelineResult | None = None,
    caption: str | None = None,
    show_box: bool = True,
    show_mask: bool = True,
) -> None:
    """Render image in the main viewer area with optional overlays.

    Displays the input image with responsive scaling while maintaining
    aspect ratio. Accepts numpy arrays (grayscale or RGB) or PIL Images.

    Non-uint8 numpy arrays are automatically normalized to 0-255 range
    using the shared preprocessing module to ensure consistency with
    model training.

    When a PipelineResult is provided, overlays are rendered based on
    available data and toggle states:
    - Bounding box: Cyan (#06B6D4), 2px stroke
    - Segmentation mask: Magenta (#D946EF), 40% opacity
    - Z-order: image → mask → box (box on top)

    Args:
        image: 2D image array (H, W) or (H, W, C), or PIL Image.
            Grayscale arrays (H, W) are automatically converted to RGB
            for consistent display.
        result: Optional PipelineResult containing box and mask data.
        caption: Optional caption to display below the image.
        show_box: Whether to display bounding box overlay (default: True).
        show_mask: Whether to display segmentation mask overlay (default: True).

    Example:
        >>> # Display a grayscale MRI slice
        >>> slice_data = np.random.rand(256, 256).astype(np.float32)
        >>> render_image_viewer(slice_data, caption="Slice 42")

        >>> # Display with pipeline result and overlays
        >>> from webapp.utils.inference import PipelineResult
        >>> result = PipelineResult(success=True, stage="segmentation")
        >>> render_image_viewer(slice_data, result=result, show_box=True, show_mask=True)
    """
    # Handle PIL Image input
    if isinstance(image, Image.Image):
        logger.debug("Rendering PIL Image")
        display_image = image
        # Apply overlays if PipelineResult provided
        if result is not None:
            display_image = _composite_overlays(
                display_image, result, show_box=show_box, show_mask=show_mask
            )
        st.image(display_image, caption=caption, use_container_width=True)
        return

    # Handle numpy array input
    if not isinstance(image, np.ndarray):
        logger.error(f"Invalid image type: {type(image)}")
        st.error("Invalid image type. Expected numpy array or PIL Image.")
        return

    # Validate array dimensions
    if image.size == 0:
        logger.error("Empty image array provided")
        st.error("Cannot display empty image.")
        return

    if image.ndim == 1:
        logger.error(f"Invalid image dimensions: {image.shape}")
        st.error("Invalid image dimensions. Expected 2D (grayscale) or 3D (RGB) array.")
        return

    if image.ndim not in (2, 3):
        logger.error(f"Invalid image dimensions: {image.ndim}D array")
        st.error("Invalid image dimensions. Expected 2D (grayscale) or 3D (RGB) array.")
        return

    # Normalize non-uint8 arrays to ensure consistent display
    if image.dtype != np.uint8:
        logger.debug(f"Normalizing {image.dtype} array to uint8")
        if image.ndim == 2:
            image = normalize_slice(image)
        else:
            # For RGB images, normalize each channel separately
            normalized_channels = []
            for i in range(image.shape[2]):
                normalized_channels.append(normalize_slice(image[:, :, i]))
            image = np.stack(normalized_channels, axis=-1)

    # Convert numpy array to PIL Image for display
    display_image = Image.fromarray(image)

    # Apply overlays if PipelineResult provided
    if result is not None:
        logger.debug(
            f"PipelineResult provided (stage={result.stage}, "
            f"success={result.success}) - applying overlays"
        )
        display_image = _composite_overlays(
            display_image, result, show_box=show_box, show_mask=show_mask
        )

    # Display with responsive width (maintains aspect ratio automatically)
    st.image(display_image, caption=caption, use_container_width=True)

    logger.debug(f"Rendered image with shape {image.shape}")


def render_zoom_controls(
    session_key: str = "viewer_zoom",
) -> ZoomState:
    """Render zoom control buttons and indicator (AC #1, #5, #6).

    Renders zoom in/out buttons, reset button, and zoom level indicator.
    Manages zoom state in st.session_state using the provided key.

    Controls are keyboard accessible (AC #6):
    - Zoom In button: "+", increase zoom by 25%
    - Zoom Out button: "-", decrease zoom by 25%
    - Reset button: "⟲", reset to 100% fit view

    Args:
        session_key: Key for storing zoom state in session_state.

    Returns:
        Current ZoomState after any button interactions.
    """
    # Initialize zoom state in session state if not present
    if session_key not in st.session_state:
        st.session_state[session_key] = init_zoom_state()

    state: ZoomState = st.session_state[session_key]

    # Create button layout
    col1, col2, col3, col4 = st.columns([1, 1, 1, 2])

    with col1:
        if st.button("➖", key=f"{session_key}_out", help="Zoom Out (-)"):
            state = zoom_out(state)
            st.session_state[session_key] = state

    with col2:
        if st.button("➕", key=f"{session_key}_in", help="Zoom In (+)"):
            state = zoom_in(state)
            st.session_state[session_key] = state

    with col3:
        if st.button("⟲", key=f"{session_key}_reset", help="Reset View (0/Home)"):
            state = reset_zoom(state)
            st.session_state[session_key] = state

    with col4:
        # Display zoom level indicator
        zoom_display = get_zoom_display(state)
        st.markdown(f"**Zoom:** {zoom_display}")

    logger.debug(f"Zoom controls rendered, current level: {state['level']}")
    return state


def render_interactive_viewer(
    image: np.ndarray | Image.Image,
    result: "PipelineResult | None" = None,
    caption: str | None = None,
    show_box: bool = True,
    show_mask: bool = True,
    enable_zoom_pan: bool = True,
    session_key: str = "viewer_zoom",
) -> None:
    """Render image with interactive zoom/pan controls (Story 3.3).

    Enhanced viewer that extends render_image_viewer with zoom/pan
    functionality. When enable_zoom_pan=True, displays:
    - Zoom controls (in/out/reset buttons)
    - Zoom level indicator
    - Zoomable image with CSS transforms

    When enable_zoom_pan=False, falls back to standard render_image_viewer
    for backward compatibility.

    Args:
        image: 2D image array (H, W) or (H, W, C), or PIL Image.
        result: Optional PipelineResult containing box and mask data.
        caption: Optional caption to display below the image.
        show_box: Whether to display bounding box overlay (default: True).
        show_mask: Whether to display segmentation mask overlay (default: True).
        enable_zoom_pan: Enable zoom/pan controls (default: True).
        session_key: Key for storing zoom state in session_state.

    Example:
        >>> # Interactive viewer with zoom/pan
        >>> render_interactive_viewer(image, result=result, enable_zoom_pan=True)

        >>> # Static viewer (backward compatible)
        >>> render_interactive_viewer(image, enable_zoom_pan=False)
    """
    if not enable_zoom_pan:
        # Fall back to static viewer for backward compatibility
        render_image_viewer(
            image,
            result=result,
            caption=caption,
            show_box=show_box,
            show_mask=show_mask,
        )
        return

    # Prepare display image (same logic as render_image_viewer)
    if isinstance(image, Image.Image):
        display_image = image
    elif isinstance(image, np.ndarray):
        if image.size == 0:
            st.error("Cannot display empty image.")
            return
        if image.ndim not in (2, 3):
            st.error(
                "Invalid image dimensions. Expected 2D (grayscale) or 3D (RGB) array."
            )
            return

        # Normalize non-uint8 arrays
        if image.dtype != np.uint8:
            if image.ndim == 2:
                image = normalize_slice(image)
            else:
                normalized_channels = []
                for i in range(image.shape[2]):
                    normalized_channels.append(normalize_slice(image[:, :, i]))
                image = np.stack(normalized_channels, axis=-1)

        display_image = Image.fromarray(image)
    else:
        st.error("Invalid image type. Expected numpy array or PIL Image.")
        return

    # Apply overlays if PipelineResult provided
    if result is not None:
        display_image = _composite_overlays(
            display_image, result, show_box=show_box, show_mask=show_mask
        )

    # Render zoom controls and get current state
    zoom_state = render_zoom_controls(session_key)

    # Render zoomable viewer with current state
    render_zoomable_viewer(display_image, zoom_state)

    # Display caption if provided
    if caption:
        st.caption(caption)

    logger.debug(
        f"Rendered interactive viewer: zoom={get_zoom_display(zoom_state)}, "
        f"show_box={show_box}, show_mask={show_mask}"
    )
