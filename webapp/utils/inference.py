"""Inference pipeline orchestration for BraTSAM web application.

This module provides the main inference pipeline that orchestrates
YOLO tumor detection followed by SAM segmentation. It handles model
loading with caching, error handling, and result packaging.

The pipeline is designed for stateless execution compatible with
ThreadPoolExecutor for background processing.
"""

import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from peft import PeftModel
    from transformers import SamModel, SamProcessor
    from ultralytics import YOLO

# Add project root to sys.path for model imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Set up logger
logger = logging.getLogger(__name__)

# Confidence threshold constants
CONFIDENCE_AUTO_APPROVED = 0.90
CONFIDENCE_NEEDS_REVIEW = 0.50


@dataclass
class PipelineResult:
    """Result from YOLO → SAM inference pipeline.

    Attributes:
        success: Whether the pipeline completed successfully.
        stage: Last completed stage ("detection" or "segmentation").
        error_message: Description of failure if success=False.
        yolo_box: Detected bounding box [x_min, y_min, x_max, y_max].
        yolo_confidence: YOLO detection confidence score.
        sam_mask: Binary segmentation mask as numpy array.
        sam_iou: SAM IoU score (placeholder until ground truth available).
    """

    success: bool
    stage: str  # "upload" | "detection" | "segmentation"
    error_message: str | None = None
    yolo_box: list[int] | None = None
    yolo_confidence: float | None = None
    sam_mask: np.ndarray | None = None
    sam_iou: float | None = None

    @property
    def confidence(self) -> float | None:
        """Combined confidence (minimum of both scores).

        Returns:
            Minimum of yolo_confidence and sam_iou if both exist,
            otherwise yolo_confidence as fallback, or None.
        """
        if self.yolo_confidence is not None and self.sam_iou is not None:
            return min(self.yolo_confidence, self.sam_iou)
        return self.yolo_confidence  # Fall back to YOLO if no SAM score

    @property
    def classification(self) -> str:
        """Three-tier triage based on combined confidence.

        Returns:
            "auto-approved" if confidence >= 0.90
            "needs-review" if confidence >= 0.50
            "manual-required" if confidence < 0.50 or None
        """
        conf = self.confidence
        if conf is None:
            return "manual-required"
        if conf >= CONFIDENCE_AUTO_APPROVED:
            return "auto-approved"
        elif conf >= CONFIDENCE_NEEDS_REVIEW:
            return "needs-review"
        return "manual-required"


def load_models() -> tuple["YOLO", "SamProcessor", "PeftModel"]:
    """Load models once, cache across all sessions.

    This function should be decorated with @st.cache_resource when
    called from Streamlit context. For testing or non-Streamlit use,
    call directly.

    Returns:
        Tuple of (YOLO model, SAM processor, SAM model with LoRA).

    Raises:
        FileNotFoundError: If model files are not found.
        RuntimeError: If model loading fails.
    """
    try:
        import streamlit as st

        # Use cached version in Streamlit context
        return _load_models_cached()
    except ImportError:
        # Non-Streamlit context (e.g., testing)
        return _load_models_impl()


def _load_models_impl() -> tuple["YOLO", "SamProcessor", "PeftModel"]:
    """Internal implementation of model loading.

    Returns:
        Tuple of (YOLO model, SAM processor, SAM model with LoRA).
    """
    from peft import PeftModel
    from transformers import SamModel, SamProcessor
    from ultralytics import YOLO

    logger.info("Loading YOLO model...")
    yolo_path = PROJECT_ROOT / "models" / "yolo_model.pt"
    yolo = YOLO(str(yolo_path))

    logger.info("Loading SAM processor...")
    sam_processor = SamProcessor.from_pretrained("facebook/sam-vit-base")

    logger.info("Loading SAM model with LoRA weights...")
    sam_model = SamModel.from_pretrained("facebook/sam-vit-base")
    sam_lora_path = PROJECT_ROOT / "models" / "sam_model.pth"
    sam_model = PeftModel.from_pretrained(sam_model, str(sam_lora_path))

    logger.info("All models loaded successfully")
    return yolo, sam_processor, sam_model


# Lazy-loaded cached version for Streamlit
_cached_load_models = None


def _load_models_cached() -> tuple["YOLO", "SamProcessor", "PeftModel"]:
    """Streamlit-cached version of model loading."""
    import streamlit as st

    @st.cache_resource
    def _cached():
        return _load_models_impl()

    return _cached()


def _detect_tumor(
    image: np.ndarray,
    yolo: "YOLO",
) -> tuple[list[int] | None, float | None]:
    """Run YOLO detection on image.

    Args:
        image: RGB image as numpy array (H, W, 3).
        yolo: Loaded YOLO model.

    Returns:
        Tuple of (bounding_box, confidence) or (None, None) if no detection.
        Box format: [x_min, y_min, x_max, y_max] (pixel coordinates).
    """
    results = yolo(image, verbose=False)

    if len(results[0].boxes) == 0:
        logger.info("No tumor detected by YOLO")
        return None, None

    # Select highest confidence box
    boxes = results[0].boxes
    best_idx = boxes.conf.argmax()
    best_box = boxes[best_idx]

    coords = best_box.xyxy[0].cpu().numpy().astype(int).tolist()
    confidence = float(best_box.conf[0].cpu().numpy())

    logger.info(f"Tumor detected: box={coords}, conf={confidence:.3f}")
    return coords, confidence


def _segment_tumor(
    image: np.ndarray,
    box: list[int],
    sam_processor: "SamProcessor",
    sam_model: "SamModel",
) -> tuple[np.ndarray, float]:
    """Run SAM segmentation with bounding box prompt.

    Args:
        image: RGB image as numpy array (H, W, 3).
        box: Bounding box [x_min, y_min, x_max, y_max].
        sam_processor: SAM processor for image preprocessing.
        sam_model: SAM model with LoRA weights.

    Returns:
        Tuple of (binary_mask, iou_score).
        Mask is numpy array with same H, W as input.
        IoU score is placeholder (0.85) until ground truth comparison added.
    """
    # Preprocess image
    inputs = sam_processor(
        images=image,
        input_boxes=[[box]],  # Nested for batch dimension
        return_tensors="pt",
    )

    # Move to same device as model
    device = next(sam_model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Run inference
    with torch.no_grad():
        outputs = sam_model(**inputs)

    # Extract mask
    masks = outputs.pred_masks.cpu().numpy()
    mask = masks[0, 0, 0]  # [B, N, M, H, W] → select first

    # Resize mask to original image size if needed
    # SAM outputs 256x256 by default
    if mask.shape != image.shape[:2]:
        from PIL import Image as PILImage

        mask_pil = PILImage.fromarray((mask > 0).astype(np.uint8) * 255)
        mask_pil = mask_pil.resize(
            (image.shape[1], image.shape[0]), PILImage.NEAREST
        )
        mask = np.array(mask_pil) > 127

    # Binary threshold
    binary_mask = (mask > 0).astype(np.uint8)

    # Placeholder IoU score (no ground truth during inference)
    # Could compute IoU against previous predictions or other heuristics
    sam_iou = 0.85  # Placeholder - will be replaced with actual computation

    logger.info(f"Segmentation complete: mask_pixels={binary_mask.sum()}")
    return binary_mask, sam_iou


def run_pipeline(
    image: np.ndarray,
    yolo: "YOLO",
    sam_processor: "SamProcessor",
    sam_model: "SamModel",
) -> PipelineResult:
    """Execute YOLO → SAM inference pipeline.

    This function is stateless and compatible with ThreadPoolExecutor
    for background processing.

    Args:
        image: Input image as numpy array (H, W, 3) RGB.
        yolo: Loaded YOLO model instance.
        sam_processor: Loaded SAM processor instance.
        sam_model: Loaded SAM model instance.

    Returns:
        PipelineResult with detection/segmentation results.
    """
    try:
        logger.info("Starting pipeline execution...")

        # Stage 1: Detection
        box, yolo_conf = _detect_tumor(image, yolo)
        if box is None:
            return PipelineResult(
                success=False,
                stage="detection",
                error_message="No tumor detected. Draw bounding box manually?",
            )

        # Stage 2: Segmentation
        mask, sam_iou = _segment_tumor(image, box, sam_processor, sam_model)

        return PipelineResult(
            success=True,
            stage="segmentation",
            yolo_box=box,
            yolo_confidence=yolo_conf,
            sam_mask=mask,
            sam_iou=sam_iou,
        )
    except Exception as e:
        logger.error(f"Pipeline error: {e}")
        return PipelineResult(
            success=False,
            stage="unknown",
            error_message=str(e),
        )


__all__ = [
    "CONFIDENCE_AUTO_APPROVED",
    "CONFIDENCE_NEEDS_REVIEW",
    "PipelineResult",
    "load_models",
    "run_pipeline",
]
