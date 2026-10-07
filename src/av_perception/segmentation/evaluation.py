"""Segmentation robustness metrics."""

from __future__ import annotations

import numpy as np


def mask_iou(clean_mask: np.ndarray, comparison_mask: np.ndarray) -> tuple[float, int, int]:
    """Return IoU, intersection pixels, and union pixels for two boolean masks."""
    if clean_mask.shape != comparison_mask.shape:
        raise ValueError(f"Mask shapes differ: {clean_mask.shape} vs {comparison_mask.shape}")
    clean = clean_mask.astype(bool, copy=False)
    comparison = comparison_mask.astype(bool, copy=False)
    intersection = int(np.logical_and(clean, comparison).sum())
    union = int(np.logical_or(clean, comparison).sum())
    return (intersection / union if union else 1.0), intersection, union


def summarize_consistency(values: list[float]) -> dict[str, float | int]:
    """Summarize per-frame clean-to-corrupted mask IoU values."""
    if not values:
        return {"frames": 0, "mean_iou": 0.0, "median_iou": 0.0, "p10_iou": 0.0}
    array = np.asarray(values, dtype=np.float64)
    return {
        "frames": len(values),
        "mean_iou": float(array.mean()),
        "median_iou": float(np.median(array)),
        "p10_iou": float(np.percentile(array, 10)),
    }
