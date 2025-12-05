"""Interactive image viewer component for BraTSAM (Story 3.1, 3.2).

This module provides the main image viewer for displaying MRI slices
with support for responsive scaling and overlay rendering.

The viewer supports:
- 2D numpy arrays (grayscale and RGB)
- PIL Images
- Automatic normalization for non-uint8 arrays
- Responsive scaling via use_container_width
- Bounding box overlay (cyan, 2px stroke)
- Segmentation mask overlay (magenta, 40% opacity)
- Toggle controls for overlay visibility

Example:
    >>> import numpy as np
    >>> from webapp.components.viewer import render_image_viewer
    >>> image = np.zeros((256, 256), dtype=np.uint8)
    >>> render_image_viewer(image, caption="MRI Slice")
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import streamlit as st
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
