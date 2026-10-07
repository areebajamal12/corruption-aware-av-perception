"""Observable runtime features and offline supervision assembly."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from av_perception.corruption.engine import apply_corruption

FEATURE_NAMES = [
    "brightness_mean",
    "brightness_std",
    "dark_fraction",
    "bright_fraction",
    "sharpness",
    "edge_density",
    "saturation_mean",
    "detection_count",
    "detection_confidence_mean",
    "detection_confidence_max",
    "vehicle_count",
    "pedestrian_count",
    "predicted_road_fraction",
    "track_count",
    "track_confidence_mean",
    "track_continuity",
]


def _key(row: dict[str, str]) -> tuple[str, str, str, int]:
    return row["scene"], row["sample_token"], row["corruption"], int(row["severity"])


def image_quality_features(image: np.ndarray) -> dict[str, float]:
    """Compute inexpensive features available from the current camera frame."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    laplacian = cv2.Laplacian(gray, cv2.CV_32F)
    edges = cv2.Canny(gray, 80, 160)
    return {
        "brightness_mean": float(gray.mean() / 255.0),
        "brightness_std": float(gray.std() / 255.0),
        "dark_fraction": float(np.mean(gray < 32)),
        "bright_fraction": float(np.mean(gray > 224)),
        "sharpness": float(np.log1p(laplacian.var()) / 12.0),
        "edge_density": float(np.mean(edges > 0)),
        "saturation_mean": float(hsv[..., 1].mean() / 255.0),
    }


def assemble_examples(
    detection_csv: Path,
    segmentation_csv: Path,
    tracking_csv: Path,
    quality_threshold: float = 0.6,
) -> list[dict[str, Any]]:
    """Join benchmark outputs into runtime features and offline quality labels."""
    examples: dict[tuple[str, str, str, int], dict[str, Any]] = {}
    with segmentation_csv.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = _key(row)
            road_pixels = float(row["corrupted_road_pixels"])
            examples[key] = {
                "scene": row["scene"],
                "frame": int(row["frame"]),
                "sample_token": row["sample_token"],
                "image_path": row["image_path"],
                "corruption": row["corruption"],
                "severity": int(row["severity"]),
                "seed": int(row["seed"]),
                "predicted_road_fraction": road_pixels / (1600 * 900),
                "segmentation_consistency": float(row["mask_iou"]),
            }

    detection_groups: dict[tuple[str, str, str, int], list[dict[str, str]]] = defaultdict(list)
    with detection_csv.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            detection_groups[_key(row)].append(row)
    for key, rows in detection_groups.items():
        detections = [row for row in rows if row["record_type"] == "detection"]
        truth = [row for row in rows if row["record_type"] == "ground_truth"]
        confidences = [float(row["confidence"]) for row in detections]
        true_positives = sum(row["matched"] == "True" for row in detections)
        precision = true_positives / len(detections) if detections else 0.0
        recall = true_positives / len(truth) if truth else 1.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        example = examples[key]
        example.update(
            {
                "detection_count": float(len(detections)),
                "detection_confidence_mean": float(np.mean(confidences)) if confidences else 0.0,
                "detection_confidence_max": max(confidences, default=0.0),
                "vehicle_count": float(sum(row["class"] == "vehicle" for row in detections)),
                "pedestrian_count": float(sum(row["class"] == "pedestrian" for row in detections)),
                "detection_f1": f1,
            }
        )

    tracking_groups: dict[tuple[str, str, str, int], list[dict[str, str]]] = defaultdict(list)
    with tracking_csv.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            tracking_groups[_key(row)].append(row)
    active_ids: dict[tuple[str, str, int], set[str]] = {}
    for key in sorted(
        tracking_groups, key=lambda value: (value[2], value[3], value[0], examples[value]["frame"])
    ):
        rows = tracking_groups[key]
        tracks = [row for row in rows if row["record_type"] == "track"]
        truth = [row for row in rows if row["record_type"] == "ground_truth"]
        confidences = [float(row["confidence"]) for row in tracks]
        ids = {row["track_id"] for row in tracks}
        history_key = (key[0], key[2], key[3])
        previous = active_ids.get(history_key, set())
        continuity = len(ids & previous) / len(ids | previous) if ids or previous else 1.0
        active_ids[history_key] = ids
        example = examples[key]
        example.update(
            {
                "track_count": float(len(tracks)),
                "track_confidence_mean": float(np.mean(confidences)) if confidences else 0.0,
                "track_continuity": continuity,
                "tracking_recall": sum(row["matched"] == "True" for row in truth) / len(truth)
                if truth
                else 1.0,
            }
        )

    output: list[dict[str, Any]] = []
    for example in examples.values():
        example.setdefault("detection_count", 0.0)
        example.setdefault("detection_confidence_mean", 0.0)
        example.setdefault("detection_confidence_max", 0.0)
        example.setdefault("vehicle_count", 0.0)
        example.setdefault("pedestrian_count", 0.0)
        example.setdefault("detection_f1", 1.0)
        example.setdefault("track_count", 0.0)
        example.setdefault("track_confidence_mean", 0.0)
        example.setdefault("track_continuity", 1.0)
        example.setdefault("tracking_recall", 1.0)
        image = cv2.imread(example["image_path"])
        if image is None:
            raise FileNotFoundError(example["image_path"])
        if example["corruption"] != "clean":
            image = apply_corruption(
                image, example["corruption"], example["severity"], example["seed"]
            )
        example.update(image_quality_features(image))
        quality = (
            0.45 * example["detection_f1"]
            + 0.25 * example["segmentation_consistency"]
            + 0.30 * example["tracking_recall"]
        )
        example["quality"] = quality
        example["acceptable"] = int(quality >= quality_threshold)
        output.append(example)
    return output


def feature_matrix(examples: list[dict[str, Any]]) -> np.ndarray:
    return np.asarray([[float(row[name]) for name in FEATURE_NAMES] for row in examples])
