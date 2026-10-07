"""nuScenes-to-camera ground-truth projection and filtering."""

from __future__ import annotations

from typing import Any

import numpy as np
from nuscenes.utils.geometry_utils import view_points

from av_perception.detection.types import BoundingBox, GroundTruth


def map_nuscenes_category(category: str) -> str | None:
    """Map only categories comparable to the requested COCO detector classes."""
    if category in {"vehicle.car", "vehicle.truck"} or category.startswith("vehicle.bus."):
        return "vehicle"
    if category.startswith("human.pedestrian."):
        return "pedestrian"
    return None


def project_ground_truth(
    nusc: Any,
    sample_data_token: str,
    image_width: int,
    image_height: int,
    min_visibility: int = 2,
    min_box_area: float = 400.0,
) -> list[GroundTruth]:
    """Project relevant 3D annotations and apply visibility/area filters."""
    _, boxes, camera_intrinsic = nusc.get_sample_data(sample_data_token)
    projected: list[GroundTruth] = []
    for box in boxes:
        class_name = map_nuscenes_category(box.name)
        if class_name is None:
            continue
        annotation = nusc.get("sample_annotation", box.token)
        visibility = int(annotation["visibility_token"])
        if visibility < min_visibility:
            continue

        corners = view_points(box.corners(), np.asarray(camera_intrinsic), normalize=True)[:2]
        bbox = BoundingBox(
            x1=float(np.min(corners[0])),
            y1=float(np.min(corners[1])),
            x2=float(np.max(corners[0])),
            y2=float(np.max(corners[1])),
        ).clipped(image_width, image_height)
        if bbox.area < min_box_area or bbox.width <= 0 or bbox.height <= 0:
            continue
        projected.append(
            GroundTruth(
                class_name=class_name,
                bbox=bbox,
                annotation_token=box.token,
                source_category=box.name,
                visibility=visibility,
                instance_token=str(annotation["instance_token"]),
            )
        )
    return projected
