"""One-to-one IoU matching and detection metrics."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable

import numpy as np

from av_perception.detection.types import BoundingBox, Detection, DetectionMatch, GroundTruth


def intersection_over_union(first: BoundingBox, second: BoundingBox) -> float:
    x1 = max(first.x1, second.x1)
    y1 = max(first.y1, second.y1)
    x2 = min(first.x2, second.x2)
    y2 = min(first.y2, second.y2)
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = first.area + second.area - intersection
    return intersection / union if union > 0 else 0.0


def match_frame(
    detections: list[Detection],
    ground_truth: list[GroundTruth],
    iou_threshold: float = 0.5,
) -> list[DetectionMatch]:
    """Greedily match detections by descending confidence within each class."""
    matches: list[DetectionMatch | None] = [None] * len(detections)
    used_ground_truth: set[int] = set()
    for detection_index in sorted(
        range(len(detections)), key=lambda index: detections[index].confidence, reverse=True
    ):
        detection = detections[detection_index]
        candidates = [
            (gt_index, intersection_over_union(detection.bbox, truth.bbox))
            for gt_index, truth in enumerate(ground_truth)
            if truth.class_name == detection.class_name and gt_index not in used_ground_truth
        ]
        gt_index, best_iou = max(candidates, key=lambda item: item[1], default=(None, 0.0))
        is_match = gt_index is not None and best_iou >= iou_threshold
        if is_match:
            used_ground_truth.add(gt_index)
        matches[detection_index] = DetectionMatch(
            detection_index=detection_index,
            ground_truth_index=gt_index if is_match else None,
            iou=best_iou,
            matched=is_match,
        )
    return [match for match in matches if match is not None]


def average_precision(
    confidences: Iterable[float], matches: Iterable[bool], ground_truth_count: int
) -> float:
    """Compute all-points interpolated AP from confidence-ranked predictions."""
    pairs = sorted(zip(confidences, matches, strict=True), reverse=True)
    if ground_truth_count == 0 or not pairs:
        return 0.0
    true_positives = np.cumsum([int(matched) for _, matched in pairs])
    false_positives = np.cumsum([int(not matched) for _, matched in pairs])
    recall = true_positives / ground_truth_count
    precision = true_positives / (true_positives + false_positives)
    recall = np.concatenate(([0.0], recall, [1.0]))
    precision = np.concatenate(([0.0], precision, [0.0]))
    precision = np.maximum.accumulate(precision[::-1])[::-1]
    changes = np.where(recall[1:] != recall[:-1])[0]
    return float(np.sum((recall[changes + 1] - recall[changes]) * precision[changes + 1]))


def summarize_metrics(rows: list[dict[str, object]], iou_threshold: float) -> dict[str, object]:
    """Summarize detection rows plus GT rows into per-class and overall metrics."""
    classes = sorted({str(row["class"]) for row in rows})
    per_class: dict[str, dict[str, float | int]] = {}
    totals = defaultdict(int)
    for class_name in classes:
        detections = [
            row for row in rows if row["record_type"] == "detection" and row["class"] == class_name
        ]
        ground_truth_count = sum(
            row["record_type"] == "ground_truth" and row["class"] == class_name for row in rows
        )
        true_positives = sum(bool(row["matched"]) for row in detections)
        false_positives = len(detections) - true_positives
        false_negatives = ground_truth_count - true_positives
        precision = true_positives / len(detections) if detections else 0.0
        recall = true_positives / ground_truth_count if ground_truth_count else 0.0
        ap = average_precision(
            (float(row["confidence"]) for row in detections),
            (bool(row["matched"]) for row in detections),
            ground_truth_count,
        )
        per_class[class_name] = {
            "ground_truth": ground_truth_count,
            "predictions": len(detections),
            "true_positives": true_positives,
            "false_positives": false_positives,
            "false_negatives": false_negatives,
            "precision": precision,
            "recall": recall,
            "ap": ap,
        }
        totals["ground_truth"] += ground_truth_count
        totals["predictions"] += len(detections)
        totals["true_positives"] += true_positives
    overall_precision = (
        totals["true_positives"] / totals["predictions"] if totals["predictions"] else 0.0
    )
    overall_recall = (
        totals["true_positives"] / totals["ground_truth"] if totals["ground_truth"] else 0.0
    )
    return {
        "iou_threshold": iou_threshold,
        "per_class": per_class,
        "overall": {
            **totals,
            "precision": overall_precision,
            "recall": overall_recall,
            "map": float(np.mean([metrics["ap"] for metrics in per_class.values()]))
            if per_class
            else 0.0,
        },
    }
