"""Batch processing queue and state management (Story 2.4, AC #6).

This module provides data structures and utilities for batch processing,
including the BatchItem dataclass and functions for aggregating classification
statistics across multiple inference results.

Note: Full batch queue implementation is planned for Story 4.1. This module
provides the foundational data structures needed for confidence-based triage.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from webapp.utils.inference import PipelineResult


@dataclass
class BatchItem:
    """Single item in the review queue (Story 4.1, AC #6).

    Represents a single image/slice in the batch processing queue,
    containing the inference result and review status.

    Attributes:
        filename: Original filename or identifier for the item.
        result: PipelineResult from inference (contains confidence/classification).
        status: Review status - one of "pending", "approved", "rejected", "edited".
        user_box: Manual bounding box override [x_min, y_min, x_max, y_max], if edited.

    Example:
        >>> item = BatchItem(
        ...     filename="scan_001_slice_45.png",
        ...     result=result,
        ...     status="pending",
        ... )
        >>> item.classification  # Delegates to result
        "auto-approved"
    """

    filename: str
    result: "PipelineResult"
    status: str = "pending"  # "pending" | "approved" | "rejected" | "edited"
    user_box: list[int] | None = None  # Manual override box

    @property
    def classification(self) -> str:
        """Get the confidence classification from the pipeline result.

        Delegates to PipelineResult.classification for consistent behavior.

        Returns:
            Classification string: "auto-approved", "needs-review", or "manual-required".
        """
        return self.result.classification

    @property
    def confidence(self) -> float | None:
        """Get the combined confidence score from the pipeline result.

        Delegates to PipelineResult.confidence for consistent behavior.

        Returns:
            Combined confidence score (0-1) or None.
        """
        return self.result.confidence


@dataclass
class BatchStatistics:
    """Aggregate statistics for a batch of items (AC #6).

    Provides counts by classification tier for displaying
    batch progress and triage summary.

    Attributes:
        total: Total number of items in batch.
        auto_approved: Count of "auto-approved" items.
        needs_review: Count of "needs-review" items.
        manual_required: Count of "manual-required" items.
        processed: Count of items no longer "pending".
    """

    total: int = 0
    auto_approved: int = 0
    needs_review: int = 0
    manual_required: int = 0
    processed: int = 0

    @property
    def pending(self) -> int:
        """Count of items still pending review."""
        return self.total - self.processed


def compute_batch_statistics(items: list[BatchItem]) -> BatchStatistics:
    """Compute aggregate statistics for a batch of items (AC #6).

    Counts items by classification tier and review status for
    displaying batch progress in the UI.

    Args:
        items: List of BatchItem objects to aggregate.

    Returns:
        BatchStatistics with counts by classification and status.

    Example:
        >>> stats = compute_batch_statistics(batch_items)
        >>> print(f"{stats.auto_approved} auto-approved, {stats.needs_review} need review")
    """
    stats = BatchStatistics(total=len(items))

    for item in items:
        classification = item.classification

        if classification == "auto-approved":
            stats.auto_approved += 1
        elif classification == "needs-review":
            stats.needs_review += 1
        else:
            stats.manual_required += 1

        if item.status != "pending":
            stats.processed += 1

    return stats


def get_items_by_classification(
    items: list[BatchItem],
    classification: str,
) -> list[BatchItem]:
    """Filter batch items by classification tier (AC #6).

    Useful for displaying items grouped by triage status
    or prioritizing manual-required items for review.

    Args:
        items: List of BatchItem objects to filter.
        classification: Target classification ("auto-approved",
            "needs-review", or "manual-required").

    Returns:
        List of BatchItem objects matching the classification.
    """
    return [item for item in items if item.classification == classification]
