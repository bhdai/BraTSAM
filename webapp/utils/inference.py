"""Inference pipeline orchestration for BraTSAM web application.

This module provides the main inference pipeline that orchestrates
YOLO tumor detection followed by SAM segmentation. It handles model
loading with caching, error handling, and result packaging.

The pipeline is designed for stateless execution compatible with
ThreadPoolExecutor for background processing.
"""

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from PIL import Image as PILImage

# Project root for model paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

if TYPE_CHECKING:
    from transformers import SamProcessor
    from ultralytics import YOLO

    # Import SamFineTuner for type hints only
    import sys
    sys.path.insert(0, str(PROJECT_ROOT))
    from model import SamFineTuner

# Set up logger
logger = logging.getLogger(__name__)

# Confidence threshold constants
CONFIDENCE_AUTO_APPROVED = 0.90
CONFIDENCE_NEEDS_REVIEW = 0.50

# YOLO detection threshold
YOLO_MIN_CONFIDENCE = 0.25

# Mask processing constants
MASK_BINARY_THRESHOLD = 127

# Model path constants
YOLO_MODEL_PATH = PROJECT_ROOT / "models" / "yolo_model.pt"
SAM_MODEL_PATH = PROJECT_ROOT / "models" / "sam_model.pth"
SAM_BASE_MODEL_ID = "facebook/sam-vit-base"


def compute_confidence(yolo_conf: float | None, sam_iou: float | None) -> float | None:
    """Compute combined confidence from YOLO and SAM scores.

    The combined confidence is the minimum of both scores, following
    the conservative scoring strategy where the weakest link determines
    overall confidence.

    Args:
        yolo_conf: YOLO detection confidence score (0-1), or None.
        sam_iou: SAM segmentation IoU score (0-1), or None.

    Returns:
        Combined confidence score (minimum of both) if both are available,
        yolo_conf as fallback if only YOLO score exists, or None.
    """
    if yolo_conf is not None and sam_iou is not None:
        return min(yolo_conf, sam_iou)
    return yolo_conf  # Fall back to YOLO if no SAM score


def classify_confidence(confidence: float | None) -> str:
    """Classify confidence score into triage tier.

    Three-tier classification:
    - "auto-approved": High confidence (>= 0.90), minimal review needed
    - "needs-review": Medium confidence (0.50-0.89), verification recommended
    - "manual-required": Low/no confidence (< 0.50 or None), intervention needed

    Args:
        confidence: Combined confidence score (0-1), or None.

    Returns:
        Classification string: "auto-approved", "needs-review", or "manual-required".
    """
    if confidence is None:
        return "manual-required"
    if confidence >= CONFIDENCE_AUTO_APPROVED:
        return "auto-approved"
    elif confidence >= CONFIDENCE_NEEDS_REVIEW:
        return "needs-review"
    return "manual-required"


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


def load_models() -> tuple["YOLO", "SamProcessor", "SamFineTuner"]:
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


def _load_models_impl() -> tuple["YOLO", "SamProcessor", "SamFineTuner"]:
    """Internal implementation of model loading.

    Returns:
        Tuple of (YOLO model, SAM processor, SAM model with LoRA).
    """
    import sys

    import torch
    from transformers import SamProcessor
    from ultralytics import YOLO

    # Add project root for model import (only at runtime)
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from model import SamFineTuner

    logger.info("Loading YOLO model...")
    yolo = YOLO(str(YOLO_MODEL_PATH))

    logger.info("Loading SAM processor...")
    sam_processor = SamProcessor.from_pretrained(SAM_BASE_MODEL_ID)

    logger.info("Loading SAM model with LoRA weights...")
    sam_model = SamFineTuner(model_id=SAM_BASE_MODEL_ID, use_lora=True)
    sam_model.load_state_dict(torch.load(str(SAM_MODEL_PATH), map_location="cpu"))
    sam_model.eval()

    # Move to GPU if available
    if torch.cuda.is_available():
        sam_model = sam_model.cuda()
        logger.info("SAM model moved to GPU")

    logger.info("All models loaded successfully")
    return yolo, sam_processor, sam_model


# Lazy-loaded cached version for Streamlit
_cached_load_models = None


def _load_models_cached() -> tuple["YOLO", "SamProcessor", "SamFineTuner"]:
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

    Note:
        Detections below YOLO_MIN_CONFIDENCE (0.25) are filtered out.
        If multiple detections remain, the highest confidence one is selected.
    """
    start_time = time.perf_counter()

    results = yolo(image, verbose=False)

    if len(results[0].boxes) == 0:
        elapsed = time.perf_counter() - start_time
        logger.info(f"No tumor detected by YOLO, time={elapsed:.3f}s")
        return None, None

    # Filter by minimum confidence threshold
    boxes = results[0].boxes
    mask = boxes.conf >= YOLO_MIN_CONFIDENCE

    if not mask.any():
        elapsed = time.perf_counter() - start_time
        logger.info(
            f"All detections below threshold ({YOLO_MIN_CONFIDENCE}), time={elapsed:.3f}s"
        )
        return None, None

    # Select highest confidence from filtered boxes
    filtered_conf = boxes.conf[mask]
    filtered_xyxy = boxes.xyxy[mask]
    best_idx = filtered_conf.argmax()

    coords = filtered_xyxy[best_idx].cpu().numpy().astype(int).tolist()
    confidence = float(filtered_conf[best_idx].cpu().numpy())

    elapsed = time.perf_counter() - start_time
    logger.info(f"Tumor detected: box={coords}, conf={confidence:.3f}, time={elapsed:.3f}s")
    return coords, confidence


def _segment_tumor(
    image: np.ndarray,
    box: list[int],
    sam_processor: "SamProcessor",
    sam_model: "SamFineTuner",
) -> tuple[np.ndarray, float]:
    """Run SAM segmentation with bounding box prompt.

    Args:
        image: RGB image as numpy array (H, W, 3).
        box: Bounding box [x_min, y_min, x_max, y_max].
        sam_processor: SAM processor for image preprocessing.
        sam_model: SAM model with LoRA weights (SamFineTuner).

    Returns:
        Tuple of (binary_mask, iou_score).
        Mask is numpy array with same H, W as input image.
        IoU score is SAM's predicted IoU for the best mask proposal.
    """
    start_time = time.perf_counter()

    # Preprocess image
    inputs = sam_processor(
        images=image,
        input_boxes=[[box]],  # Nested for batch dimension
        return_tensors="pt",
    )

    # Move to same device as model
    device = next(sam_model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Run inference using SamFineTuner.forward() with full_outputs
    # This returns all mask proposals and IoU scores for smart selection
    with torch.no_grad():
        outputs = sam_model(
            pixel_values=inputs["pixel_values"],
            input_boxes=inputs["input_boxes"],
            full_outputs=True,
        )

    # Extract masks and IoU scores
    # pred_masks shape: [B, N, M, H, W] where M=3 mask proposals
    # iou_scores shape: [B, N, M]
    pred_masks = outputs["pred_masks"]
    iou_scores = outputs["iou_scores"]

    # Select best mask by IoU score (instead of hardcoded index 0)
    if iou_scores is not None:
        # Get IoU scores for first batch, first prompt: [M]
        iou_per_mask = iou_scores[0, 0]  # shape: [M=3]
        best_mask_idx = int(iou_per_mask.argmax().item())
        sam_iou = float(iou_per_mask[best_mask_idx].cpu().item())
        mask = pred_masks[0, 0, best_mask_idx].cpu().numpy()  # Select best mask
        logger.debug(
            f"Mask selection: IoU scores={iou_per_mask.cpu().tolist()}, "
            f"selected idx={best_mask_idx} with IoU={sam_iou:.3f}"
        )
    else:
        # Fallback: select first mask with heuristic IoU
        mask = pred_masks[0, 0, 0].cpu().numpy()
        mask_area_ratio = (mask > 0).sum() / mask.size
        # Reasonable masks typically cover 1-30% of the image
        sam_iou = min(0.95, max(0.5, 1.0 - abs(mask_area_ratio - 0.10) * 2))
        logger.debug(f"No IoU scores, estimated from mask area: {sam_iou:.3f}")

    # Resize mask to original image size if needed
    # SAM outputs 256x256 by default
    if mask.shape != image.shape[:2]:
        mask_pil = PILImage.fromarray((mask > 0).astype(np.uint8) * 255)
        mask_pil = mask_pil.resize(
            (image.shape[1], image.shape[0]), PILImage.NEAREST
        )
        mask = np.array(mask_pil) > MASK_BINARY_THRESHOLD

    # Binary threshold
    binary_mask = (mask > 0).astype(np.uint8)

    elapsed = time.perf_counter() - start_time
    logger.info(
        f"Segmentation complete: mask_pixels={binary_mask.sum()}, "
        f"iou={sam_iou:.3f}, time={elapsed:.3f}s"
    )
    return binary_mask, sam_iou


def run_pipeline(
    image: np.ndarray,
    yolo: "YOLO",
    sam_processor: "SamProcessor",
    sam_model: "SamFineTuner",
) -> PipelineResult:
    """Execute YOLO → SAM inference pipeline.

    This function is stateless and compatible with ThreadPoolExecutor
    for background processing.

    Args:
        image: Input image as numpy array (H, W, 3) RGB.
        yolo: Loaded YOLO model instance.
        sam_processor: Loaded SAM processor instance.
        sam_model: Loaded SAM model instance (SamFineTuner).

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
    "MASK_BINARY_THRESHOLD",
    "SAM_BASE_MODEL_ID",
    "SAM_MODEL_PATH",
    "YOLO_MIN_CONFIDENCE",
    "YOLO_MODEL_PATH",
    "PipelineResult",
    "_detect_tumor",
    "_segment_tumor",
    "load_models",
    "run_pipeline",
]
